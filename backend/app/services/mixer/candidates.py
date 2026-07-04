"""Seam-candidate generation for planner v2.

The single biggest failure mode of the free-form (v1) LLM planner was
asking the model to *derive* seam timestamps from a sections array —
LLMs are unreliable at that kind of arithmetic, so plans routinely
violated headroom/vocal-safety rules and were silently replaced by the
deterministic fallback.

Planner v2 inverts the responsibility: this module pre-computes a small
menu of *valid-by-construction* seam candidates (downbeat-snapped,
inside the headroom budget, annotated with section role / energy /
vocal-safety), and the LLM only has to pick from the menu — e.g.
"A2 → B1". A wrong number becomes impossible; a wrong *choice* is at
worst a matter of taste.

Pure module: no DB, no storage, no settings. Everything arrives via
arguments so it can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.mixer.types import AnalysisBundle

# Maximum crossfade length the planner may use; headroom is reserved for
# it when computing max_seam_time. Mirrors the deterministic planner.
MAX_CROSSFADE_BARS = 16
# Safety buffer (seconds) past the crossfade end. Absorbs stem-WAV drift
# vs `Song.duration_seconds` metadata (Demucs trims/pads, yt-dlp rounds).
SEAM_SAFETY_SECONDS = 5.0
# Cap the menu so the prompt stays small and the choice stays easy.
MAX_CANDIDATES = 6
# Two candidates closer than this are duplicates for mixing purposes.
DEDUP_SECONDS = 2.0
# B entry points past this fraction of the song are pointless — the
# listener would barely hear B before the *next* transition starts.
B_ENTRY_MAX_FRACTION = 0.5
# A vocal-safe entry must stay clear of vocals for this many bars after
# the candidate point for a hard cut to land cleanly.
VOCAL_SAFE_LOOKAHEAD_BARS = 2.0
# Sections with normalized energy at/above this read as a drop/chorus.
HIGH_ENERGY_THRESHOLD = 0.8
# ...and at/below this read as a breakdown/outro (breakdown_blend fuel).
LOW_ENERGY_THRESHOLD = 0.4
# Phrase lengths to try when building the phrase grid, in bars. Human DJs
# mix on 16/32-bar phrases; 16 fits typical 3+ minute dance tracks, 8 is
# the fallback for shorter songs where a 16-bar grid leaves too few
# boundaries to choose from.
PHRASE_BARS_CHOICES = (16, 8)
# A phrase-snapped time may overshoot the raw target by at most one
# phrase; anything past the headroom ceiling falls back to downbeat snap.
MIN_GRID_BOUNDARIES = 3


def max_seam_time(duration: float, bpm: float, time_signature: int) -> float:
    """Latest seam time that leaves room for a full-length crossfade.

    Returned in original-song seconds. Clamped to 0 if the song is
    shorter than the reserved tail.
    """
    if not bpm or not duration:
        return 0.0
    sec_per_bar = (60.0 / bpm) * time_signature
    return max(0.0, duration - MAX_CROSSFADE_BARS * sec_per_bar - SEAM_SAFETY_SECONDS)


def enrich_sections(sections: list[dict], energy_curve: list[float]) -> list[dict]:
    """Annotate each section with mean energy, normalized 0..1 over the song.

    The librosa analyzer's section `label`s are opaque cluster IDs
    (`section_1`…) that tell a planner nothing; a structure-aware
    detector (e.g. allin1) emits real roles (`chorus`, `verse`). We keep
    meaningful labels and drop the opaque ones. The energy curve is
    sampled at 1 Hz (index ≈ second); we average it per section and
    normalize to the song's hottest section so ~1.0 reads as a
    drop/chorus and low values as intro/breakdown/outro.
    """
    if not sections:
        return []
    n = len(energy_curve)
    raw: list[float] = []
    for s in sections:
        lo = int(s["start"])
        hi = max(lo + 1, int(round(s["end"])))
        window = energy_curve[lo:hi] if n else []
        raw.append(sum(window) / len(window) if window else 0.0)
    peak = max(raw) or 1.0
    out = []
    for s, e in zip(sections, raw):
        item = {
            "start": round(s["start"], 1),
            "end": round(s["end"], 1),
            "energy": round(e / peak, 2),
        }
        label = s.get("label")
        if label and not str(label).startswith("section_"):
            item["label"] = label
        out.append(item)
    return out


@dataclass(frozen=True)
class SeamCandidate:
    """One pre-validated seam point the LLM may choose."""

    id: str                # "A1", "B3", ...
    time: float            # seconds, original-song time, downbeat-snapped
    description: str       # human/LLM-readable role, e.g. "start of final section"
    energy: float          # normalized 0..1 section energy at this point
    vocal_safe: bool       # True if a hard cut here avoids chopping a word
    # Densities are 0.0-1.0 over the lookahead window, or None when the
    # underlying data (safety regions / stem envelopes) is missing —
    # None is omitted from the LLM dict rather than passed off as 0.0.
    vocal_density: float | None = None
    drum_density: float | None = None
    bass_density: float | None = None
    lyrics_preview: str | None = None

    def to_llm_dict(self) -> dict:
        d = {
            "id": self.id,
            "time": round(self.time, 2),
            "description": self.description,
            "energy": self.energy,
            "vocal_safe": self.vocal_safe,
        }
        for name in ("vocal_density", "drum_density", "bass_density"):
            value = getattr(self, name)
            if value is not None:
                d[name] = round(value, 2)
        if self.lyrics_preview:
            d["lyrics_preview"] = self.lyrics_preview
        return d


@dataclass(frozen=True)
class PairCandidates:
    out_candidates: list[SeamCandidate] = field(default_factory=list)
    in_candidates: list[SeamCandidate] = field(default_factory=list)

    def find(self, candidate_id: str) -> SeamCandidate | None:
        for c in (*self.out_candidates, *self.in_candidates):
            if c.id == candidate_id:
                return c
        return None


def _snap_to_downbeat(t: float, downbeats: list[float]) -> float:
    """First downbeat ≥ t, or the last downbeat if none qualifies."""
    if not downbeats:
        return t
    for d in downbeats:
        if d >= t:
            return d
    return downbeats[-1]


def _snap_to_downbeat_at_or_before(t: float, downbeats: list[float]) -> float:
    """Latest downbeat ≤ t, or the first downbeat if none qualifies."""
    if not downbeats:
        return t
    best = None
    for d in downbeats:
        if d <= t:
            best = d
        else:
            break
    return best if best is not None else downbeats[0]


def phrase_grid(
    downbeats: list[float],
    sections: list[dict],
    bars_per_phrase: int | None = None,
) -> list[float]:
    """Times of musical phrase starts, derived from the downbeat grid.

    Human DJs mix on 8/16-bar phrase boundaries, not arbitrary downbeats
    — a seam on bar 3 of a phrase is on-grid but musically "off". Each
    section re-anchors the phrase counter at its first downbeat (song
    structure resets phrasing); within a section every Nth downbeat is a
    phrase start. With no section data the grid anchors at downbeat 0.

    When `bars_per_phrase` is None, tries PHRASE_BARS_CHOICES in order
    and keeps the first grid dense enough to be a real menu
    (>= MIN_GRID_BOUNDARIES boundaries) — long tracks get 16-bar
    phrasing, short ones fall back to 8.
    """
    if not downbeats:
        return []

    def _build(n_bars: int) -> list[float]:
        anchors: list[tuple[float, float]] = []  # (anchor_time, region_end)
        if sections:
            for i, s in enumerate(sections):
                end = sections[i + 1]["start"] if i + 1 < len(sections) else float("inf")
                anchors.append((float(s["start"]), float(end)))
        else:
            anchors.append((downbeats[0], float("inf")))

        grid: list[float] = []
        for start, end in anchors:
            # First downbeat at/after the anchor.
            idx = next((i for i, d in enumerate(downbeats) if d >= start), None)
            if idx is None:
                continue
            i = idx
            while i < len(downbeats) and downbeats[i] < end:
                if not grid or downbeats[i] > grid[-1]:
                    grid.append(downbeats[i])
                i += n_bars
        return grid

    if bars_per_phrase is not None:
        return _build(bars_per_phrase)
    grid: list[float] = []
    for n_bars in PHRASE_BARS_CHOICES:
        grid = _build(n_bars)
        if len(grid) >= MIN_GRID_BOUNDARIES:
            return grid
    return grid


def _snap_to_phrase(
    t: float,
    grid: list[float],
    downbeats: list[float],
    ceiling: float | None = None,
) -> float:
    """Next phrase boundary ≥ t; plain downbeat snap when the grid is
    empty, exhausted, or the boundary would overshoot the ceiling."""
    for g in grid:
        if g >= t:
            if ceiling is not None and g > ceiling:
                break
            return g
    return _snap_to_downbeat(t, downbeats)


def _snap_to_phrase_at_or_before(
    t: float, grid: list[float], downbeats: list[float]
) -> float:
    """Latest phrase boundary ≤ t; downbeat-at-or-before fallback."""
    best = None
    for g in grid:
        if g <= t:
            best = g
        else:
            break
    if best is not None:
        return best
    return _snap_to_downbeat_at_or_before(t, downbeats)


def _is_vocal_safe(
    t: float,
    safe_regions: list[dict],
    lookahead_seconds: float,
) -> bool:
    """True when [t, t + lookahead] sits inside one no-vocal span.

    `safe_regions` rows are `{"start", "end"}` (optionally with a
    `safe`/`reason` field from the vocal-safety service — when a `safe`
    key is present, only rows with safe=True count). With no region data
    at all we conservatively return False: "unknown" must not read as
    "safe" or hard cuts would chop words on un-transcribed songs.
    """
    if not safe_regions:
        return False
    for r in safe_regions:
        if "safe" in r and not r.get("safe"):
            continue
        if r["start"] <= t and (t + lookahead_seconds) <= r["end"]:
            return True
    return False


def _vocal_density(
    t: float,
    safe_regions: list[dict],
    lookahead_seconds: float,
) -> float | None:
    """Fraction of [t, t + lookahead] NOT covered by a safe (silent) region.

    Returns None when there is no safety data at all — "unknown", which
    to_llm_dict omits, matching _is_vocal_safe's conservative False for
    the same case rather than claiming the point is instrumental.
    """
    if not safe_regions or lookahead_seconds <= 0:
        return None

    safe_dur = 0.0
    end_t = t + lookahead_seconds
    for r in safe_regions:
        if "safe" in r and not r.get("safe"):
            continue
        overlap_start = max(t, r["start"])
        overlap_end = min(end_t, r["end"])
        if overlap_start < overlap_end:
            safe_dur += (overlap_end - overlap_start)
            
    return min(1.0, (lookahead_seconds - safe_dur) / lookahead_seconds)


def _stem_density(
    t: float, envelopes: dict | None, stem_name: str, lookahead_seconds: float
) -> float | None:
    """Average stem RMS over the lookahead window, scaled to 0..1.

    None (= "unknown", omitted from the LLM dict) when the envelope
    sidecar is missing or predates the multi-stem schema."""
    if not envelopes or stem_name not in envelopes or lookahead_seconds <= 0:
        return None
    frame_hz = envelopes.get("frame_hz", 10)
    rms_array = envelopes[stem_name].get("rms", [])
    if not rms_array:
        return None
    start_frame = int(t * frame_hz)
    end_frame = int((t + lookahead_seconds) * frame_hz)
    end_frame = max(start_frame + 1, min(end_frame, len(rms_array)))
    if start_frame >= len(rms_array):
        return 0.0
    window = rms_array[start_frame:end_frame]
    avg = sum(window) / len(window)
    # Scale: an RMS of 0.15 is generally "full" density for a single stem
    return min(1.0, avg / 0.15)


def _lyrics_preview(
    t: float, segments: list[dict] | None, lookahead_seconds: float, is_out: bool
) -> str | None:
    if not segments:
        return None
    start_t = t - lookahead_seconds if is_out else t
    end_t = t if is_out else t + lookahead_seconds
    
    words = []
    for seg in segments:
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                if start_t <= w["start"] and w["end"] <= end_t:
                    words.append(w["word"].strip())
        else:
            if start_t <= seg["start"] and seg["end"] <= end_t:
                words.append(seg["text"].strip())
    
    if not words:
        return None
    text = " ".join(words)
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _phrase_tag(t: float, grid: list[float]) -> str:
    """Suffix for candidate descriptions when `t` sits on the phrase grid,
    so the LLM can prefer phrase-aligned seams explicitly."""
    return " (phrase-aligned)" if any(abs(t - g) < 1e-6 for g in grid) else ""


def _dedup_and_cap(cands: list[SeamCandidate]) -> list[SeamCandidate]:
    cands = sorted(cands, key=lambda c: c.time)
    kept: list[SeamCandidate] = []
    for c in cands:
        if kept and abs(c.time - kept[-1].time) < DEDUP_SECONDS:
            continue
        kept.append(c)
    return kept[:MAX_CANDIDATES]


def _reid(cands: list[SeamCandidate], prefix: str) -> list[SeamCandidate]:
    return [
        SeamCandidate(
            id=f"{prefix}{i + 1}",
            time=c.time,
            description=c.description,
            energy=c.energy,
            vocal_safe=c.vocal_safe,
            vocal_density=c.vocal_density,
            drum_density=c.drum_density,
            bass_density=c.bass_density,
            lyrics_preview=c.lyrics_preview,
        )
        for i, c in enumerate(cands)
    ]


def build_out_candidates(
    a: AnalysisBundle,
    energy_curve: list[float],
    safe_regions: list[dict],
) -> list[SeamCandidate]:
    """OUT points for the outgoing song A: late-song section starts that
    leave a full crossfade of headroom, downbeat-snapped."""
    ceiling = max_seam_time(a.duration, a.bpm, a.time_signature)
    if ceiling <= 0:
        return []
    sec_per_bar = (60.0 / a.bpm) * a.time_signature if a.bpm else 0.0
    lookahead = VOCAL_SAFE_LOOKAHEAD_BARS * sec_per_bar
    sections = enrich_sections(a.sections, energy_curve)
    grid = phrase_grid(a.downbeats, a.sections)

    def _out_candidate(t: float, description: str, energy: float) -> SeamCandidate:
        return SeamCandidate(
            id="A?", time=t, description=description, energy=energy,
            vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
            vocal_density=_vocal_density(t, safe_regions, lookahead),
            drum_density=_stem_density(t, a.envelopes, "drums", lookahead),
            bass_density=_stem_density(t, a.envelopes, "bass", lookahead),
            lyrics_preview=_lyrics_preview(
                t, a.transcription_segments, lookahead, is_out=True
            ),
        )

    raw: list[SeamCandidate] = []
    n = len(sections)

    # The last high-energy section start ("the drop") and the final
    # low-energy section start ("the breakdown/outro") — double_drop and
    # breakdown_blend material. Appended FIRST: on a time collision with
    # the generic section candidates below, dedup keeps the first entry,
    # and these role labels tell the LLM strictly more.
    for s in reversed(sections):
        if s["energy"] >= HIGH_ENERGY_THRESHOLD:
            t = _snap_to_phrase(s["start"], grid, a.downbeats, ceiling)
            if t <= ceiling:
                raw.append(_out_candidate(
                    t,
                    "last drop/chorus (high energy)" + _phrase_tag(t, grid),
                    s["energy"],
                ))
            break
    for s in reversed(sections):
        if s["energy"] <= LOW_ENERGY_THRESHOLD:
            t = _snap_to_phrase(s["start"], grid, a.downbeats, ceiling)
            if t <= ceiling:
                raw.append(_out_candidate(
                    t,
                    "final breakdown (low energy)" + _phrase_tag(t, grid),
                    s["energy"],
                ))
            break

    for i, s in enumerate(sections):
        t = _snap_to_phrase(s["start"], grid, a.downbeats, ceiling)
        if t > ceiling:
            # Section starts too late to fit a crossfade — skip; the
            # "late as possible" fallback below covers the tail.
            continue
        # Only late-song sections are musical OUT points.
        if n >= 3 and i < n - 3:
            continue
        role = "final section" if i == n - 1 else (
            "second-to-last section" if i == n - 2 else "late section"
        )
        label = s.get("label")
        desc = (
            f"start of {role}" + (f" ({label})" if label else "")
            + _phrase_tag(t, grid)
        )
        raw.append(_out_candidate(t, desc, s["energy"]))

    # Always offer "as late as the headroom allows" — the v1 default.
    late = _snap_to_phrase_at_or_before(ceiling, grid, a.downbeats)
    if late <= ceiling:
        raw.append(_out_candidate(
            late,
            "latest possible out point" + _phrase_tag(late, grid),
            _energy_at(sections, late),
        ))

    return _reid(_dedup_and_cap(raw), "A")


def build_in_candidates(
    b: AnalysisBundle,
    energy_curve: list[float],
    safe_regions: list[dict],
) -> list[SeamCandidate]:
    """IN points for the incoming song B: early-song moments — post-intro
    downbeat, first energy rise, first couple of section starts."""
    ceiling = min(
        max_seam_time(b.duration, b.bpm, b.time_signature),
        b.duration * B_ENTRY_MAX_FRACTION,
    )
    if ceiling <= 0:
        return []
    sec_per_bar = (60.0 / b.bpm) * b.time_signature if b.bpm else 0.0
    lookahead = VOCAL_SAFE_LOOKAHEAD_BARS * sec_per_bar
    sections = enrich_sections(b.sections, energy_curve)
    grid = phrase_grid(b.downbeats, b.sections)

    def _candidate(t: float, description: str, energy: float) -> SeamCandidate:
        return SeamCandidate(
            id="B?", time=t, description=description, energy=energy,
            vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
            vocal_density=_vocal_density(t, safe_regions, lookahead),
            drum_density=_stem_density(t, b.envelopes, "drums", lookahead),
            bass_density=_stem_density(t, b.envelopes, "bass", lookahead),
            lyrics_preview=_lyrics_preview(
                t, b.transcription_segments, lookahead, is_out=False
            ),
        )

    raw: list[SeamCandidate] = []

    # v1's default: first phrase boundary after the first section (skips
    # silent intros / count-ins).
    if sections:
        t = _snap_to_phrase(sections[0]["end"], grid, b.downbeats, ceiling)
        if t <= ceiling:
            raw.append(
                _candidate(
                    t,
                    "end of intro / first section" + _phrase_tag(t, grid),
                    _energy_at(sections, t),
                )
            )
    else:
        t = _snap_to_downbeat(0.0, b.downbeats)
        if t <= ceiling:
            raw.append(_candidate(t, "start of the song", 0.5))

    # First high-energy section start = "the first drop / chorus".
    for i, s in enumerate(sections):
        if s["energy"] >= HIGH_ENERGY_THRESHOLD:
            t = _snap_to_phrase(s["start"], grid, b.downbeats, ceiling)
            if t <= ceiling:
                label = s.get("label")
                desc = (
                    "first high-energy section (drop/chorus)"
                    + (f" ({label})" if label else "")
                    + _phrase_tag(t, grid)
                )
                raw.append(_candidate(t, desc, s["energy"]))
            break

    # Starts of sections 2 and 3 round out the early-song menu.
    for i, s in enumerate(sections[1:3], start=2):
        t = _snap_to_phrase(s["start"], grid, b.downbeats, ceiling)
        if t <= ceiling:
            label = s.get("label")
            desc = (
                f"start of section {i}" + (f" ({label})" if label else "")
                + _phrase_tag(t, grid)
            )
            raw.append(_candidate(t, desc, s["energy"]))

    if not raw:
        # Degenerate analysis — offer the first usable downbeat.
        t = _snap_to_downbeat(0.0, b.downbeats)
        if t <= ceiling:
            raw.append(_candidate(t, "start of the song", 0.5))

    return _reid(_dedup_and_cap(raw), "B")


def _energy_at(enriched_sections: list[dict], t: float) -> float:
    for s in enriched_sections:
        if s["start"] <= t < s["end"]:
            return s["energy"]
    return enriched_sections[-1]["energy"] if enriched_sections else 0.5


def build_pair_candidates(
    a: AnalysisBundle,
    b: AnalysisBundle,
    a_energy_curve: list[float],
    b_energy_curve: list[float],
    a_safe_regions: list[dict],
    b_safe_regions: list[dict],
) -> PairCandidates:
    return PairCandidates(
        out_candidates=build_out_candidates(a, a_energy_curve, a_safe_regions),
        in_candidates=build_in_candidates(b, b_energy_curve, b_safe_regions),
    )
