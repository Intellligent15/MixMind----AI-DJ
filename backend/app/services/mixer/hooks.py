"""Hook + pocket finding for the tease move (F9).

A human DJ teases the NEXT track's vocal hook over the current one a few
minutes before the transition, so the drop pays off familiarity. Two
pure finders:

  * find_hook(...)   — B's most recognizable vocal line (the repeated
    chorus line, preferably its occurrence inside the hottest section);
  * find_pocket(...) — a vocal-free, drum-solid stretch late in A where
    that line can ride without fighting A's own vocal.

Both return None freely — a tease is a garnish, and no tease always
beats a bad one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Hook length bounds (seconds). Shorter reads as a glitch; longer stops
# being a tease and starts being a mashup.
HOOK_MIN_SECONDS = 2.0
HOOK_MAX_SECONDS = 8.0
# A line must repeat at least this often to count as "the hook".
HOOK_MIN_REPEATS = 2
HOOK_MIN_WORDS = 3
HOOK_MAX_WORDS = 12
# Pocket requirements: fits the hook plus breathing room, solid groove
# underneath, and located in the tease zone late in A (after the stitch
# junction, before the transition itself).
POCKET_PAD_SECONDS = 2.0
POCKET_MIN_DRUM_DENSITY = 0.5
# The stitch junction into A's render lands near the midpoint between the
# previous crossfade end and A's seam; 0.6 * seam is a conservative floor
# so the tease always lands inside THIS pair's render.
TEASE_ZONE_START_FRACTION = 0.6


@dataclass(frozen=True)
class Hook:
    start: float
    end: float
    text: str


def _norm_line(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def find_hook(
    transcription_segments: list[dict] | None,
    sections: list[dict] | None = None,
    energy_curve: list[float] | None = None,
) -> Hook | None:
    """B's most recognizable vocal line, from Whisper segments.

    Strategy: group segments by normalized text; the most-repeated line
    with a sane word count is the hook; prefer the occurrence inside the
    highest-energy region (the chorus proper, not the first verse echo).
    """
    if not transcription_segments:
        return None

    groups: dict[str, list[dict]] = {}
    for seg in transcription_segments:
        text = (seg.get("text") or "").strip()
        norm = _norm_line(text)
        n_words = len(norm.split())
        if not norm or not (HOOK_MIN_WORDS <= n_words <= HOOK_MAX_WORDS):
            continue
        dur = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        if not (HOOK_MIN_SECONDS * 0.5 <= dur <= HOOK_MAX_SECONDS):
            continue
        groups.setdefault(norm, []).append(seg)

    repeated = {
        norm: segs for norm, segs in groups.items()
        if len(segs) >= HOOK_MIN_REPEATS
    }
    if not repeated:
        return None
    # Most-repeated wins; ties break toward the longer line (more hook-y).
    best_norm = max(repeated, key=lambda k: (len(repeated[k]), len(k)))
    occurrences = repeated[best_norm]

    def _energy_at(t: float) -> float:
        if not energy_curve:
            return 0.0
        idx = min(len(energy_curve) - 1, max(0, int(t)))
        return float(energy_curve[idx])

    chosen = max(occurrences, key=lambda s: _energy_at(float(s["start"])))
    start = float(chosen["start"])
    end = min(float(chosen["end"]), start + HOOK_MAX_SECONDS)
    if end - start < HOOK_MIN_SECONDS:
        return None
    return Hook(start=start, end=end, text=(chosen.get("text") or "").strip())


def find_pocket(
    hook_seconds: float,
    seam_a: float,
    a_safe_regions: list[dict] | None,
    a_envelopes: dict | None,
) -> float | None:
    """Start time in A for the tease: inside a vocal-safe span long
    enough for the hook + pad, with a solid drum groove underneath, and
    within the tease zone (last stretch of A before the seam)."""
    if not a_safe_regions or hook_seconds <= 0:
        return None
    zone_start = seam_a * TEASE_ZONE_START_FRACTION
    need = hook_seconds + POCKET_PAD_SECONDS

    candidates: list[float] = []
    for region in a_safe_regions:
        if "safe" in region and not region.get("safe"):
            continue
        lo = max(float(region["start"]), zone_start)
        hi = min(float(region["end"]), seam_a - POCKET_PAD_SECONDS)
        if hi - lo < need:
            continue
        candidates.append(lo + (hi - lo - need) / 2.0)  # center of the span

    if not candidates:
        return None

    def _drum_density(t: float) -> float:
        if not a_envelopes or "drums" not in a_envelopes:
            return POCKET_MIN_DRUM_DENSITY  # unknown -> assume adequate
        frame_hz = a_envelopes.get("frame_hz", 10)
        rms = a_envelopes["drums"].get("rms") or []
        lo = int(t * frame_hz)
        hi = min(len(rms), int((t + hook_seconds) * frame_hz))
        if hi <= lo or lo >= len(rms):
            return 0.0
        window = rms[lo:hi]
        return min(1.0, (sum(window) / len(window)) / 0.15)

    # Latest qualifying pocket: closest to the transition = most tease.
    for t in sorted(candidates, reverse=True):
        if _drum_density(t) >= POCKET_MIN_DRUM_DENSITY:
            return round(t, 3)
    return None
