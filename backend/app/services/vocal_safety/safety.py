"""Compute vocal-safe regions of a song — intervals where a transition
can safely place a hard cut, stem swap, or drop swap without chopping
a syllable.

See ai-dj-spec.md → "Vocal Safety Model". Cross-references Whisper /
aligned-lyrics word boundaries against the vocal stem's frame-wise
RMS + peak envelope written by ``separate_stems``.

Envelope sidecar shape (written by
``app/services/stems/service.py::_compute_stem_envelopes``):
``{"frame_hz": int, "vocals": {"rms": [...], "peak": [...]}, ...}``
with one sub-dict per stem. Two legacy shapes are also tolerated:
the flat vocal-only ``{"frame_hz": int, "rms": [...], "peak": [...]}``
and the Modal-era ``{"hop_seconds": float, "rms": [...]}``."""

from __future__ import annotations

from typing import Any

# Allow up to this fraction of frames in a candidate quiet gap to
# exceed the quiet thresholds. Accommodates Demucs stem bleed.
QUIET_NOISE_TOLERANCE = 0.15


def _hop_seconds(envelope: dict[str, Any]) -> float:
    if "frame_hz" in envelope and envelope["frame_hz"]:
        return 1.0 / float(envelope["frame_hz"])
    if "hop_seconds" in envelope and envelope["hop_seconds"]:
        return float(envelope["hop_seconds"])
    return 0.1


def _extract_vocal_envelope(
    envelope: dict[str, Any],
) -> tuple[list[float], list[float], float]:
    """Vocal ``(rms, peak, hop_seconds)`` from any known sidecar shape.

    Accepts the current multi-stem schema (vocal data nested under
    ``"vocals"``) and both legacy flat shapes (rms/peak at the top
    level) still present on Stems rows separated before the rename."""
    vocals = envelope.get("vocals")
    source = vocals if isinstance(vocals, dict) else envelope
    rms = list(source.get("rms") or [])
    peak = list(source.get("peak") or [])
    return rms, peak, _hop_seconds(envelope)


def _envelope_value(values: list[float], idx: int) -> float:
    if not values:
        return 0.0
    if idx < 0:
        return values[0]
    if idx >= len(values):
        return values[-1]
    return float(values[idx])


def _max_over_range(
    values: list[float],
    start_idx: int,
    end_idx: int,
) -> float:
    """Maximum value in the half-open frame window. Returns 0 if the
    range is empty or out of bounds on both sides."""
    if not values:
        return 0.0
    lo = max(0, start_idx)
    hi = min(len(values), end_idx)
    if hi <= lo:
        return _envelope_value(values, start_idx)
    return max(values[lo:hi])


def _word_has_audio_support(
    rms_list: list[float],
    peak_list: list[float],
    hop: float,
    word_start: float,
    word_end: float,
    rms_presence: float,
    peak_presence: float,
) -> bool:
    """Confirm the vocal stem actually has energy across the word's
    timestamp. Uses MAX over the word window (not midpoint) so
    inter-syllable consonant gaps don't false-reject real words.
    Requires BOTH RMS AND peak to clear their thresholds — spec's
    usable_vocal_word condition. When the envelope predates the peak
    sidecar (legacy stems), peak is absent and only RMS is required."""
    if word_end <= word_start:
        return False
    start_idx = int(word_start / hop)
    # +1 so a sub-frame word still inspects at least one frame.
    end_idx = max(start_idx + 1, int(word_end / hop) + 1)
    rms_max = _max_over_range(rms_list, start_idx, end_idx)
    if not peak_list:
        return rms_max >= rms_presence
    peak_max = _max_over_range(peak_list, start_idx, end_idx)
    return rms_max >= rms_presence and peak_max >= peak_presence


def _quiet_fraction_ok(
    rms_list: list[float],
    peak_list: list[float],
    start_idx: int,
    end_idx: int,
    rms_quiet: float,
    peak_quiet: float,
) -> bool:
    total = 0
    noisy = 0
    for i in range(start_idx, end_idx):
        if i < 0 or i >= len(rms_list):
            continue
        total += 1
        rms_val = rms_list[i]
        peak_val = _envelope_value(peak_list, i) if peak_list else 0.0
        if rms_val >= rms_quiet or peak_val >= peak_quiet:
            noisy += 1
    if total == 0:
        return True
    return (noisy / total) < QUIET_NOISE_TOLERANCE


def _collect_from_alignment(
    aligned_words: list[dict[str, Any]],
    rms_list: list[float],
    peak_list: list[float],
    hop: float,
    rms_presence: float,
    peak_presence: float,
) -> list[tuple[float, float]]:
    """Aligned-words path: trust ``whisper_match`` and
    ``whisper_substitution`` outright (audio-supported by construction),
    require audio support for ``interpolated`` (their timestamps are
    guessed)."""
    out: list[tuple[float, float]] = []
    for w in aligned_words:
        src = w.get("source")
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            continue
        if src in ("whisper_match", "whisper_substitution"):
            out.append((float(start), float(end)))
        elif src == "interpolated":
            if _word_has_audio_support(
                rms_list, peak_list, hop, start, end, rms_presence, peak_presence
            ):
                out.append((float(start), float(end)))
    return out


def _collect_from_whisper(
    segments: list[dict[str, Any]],
    rms_list: list[float],
    peak_list: list[float],
    hop: float,
    word_prob_min: float,
    segment_logprob_min: float,
    rms_presence: float,
    peak_presence: float,
) -> list[tuple[float, float]]:
    """Raw-Whisper fallback. Applies spec's usable_vocal_word filter."""
    out: list[tuple[float, float]] = []
    for seg in segments or []:
        if seg.get("avg_logprob", 0.0) < segment_logprob_min:
            continue
        for w in seg.get("words") or []:
            if w.get("probability", 0.0) < word_prob_min:
                continue
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                continue
            if not _word_has_audio_support(
                rms_list, peak_list, hop, start, end, rms_presence, peak_presence
            ):
                continue
            out.append((float(start), float(end)))
    return out


def _merge_intervals(
    intervals: list[tuple[float, float]],
    join_gap: float = 0.2,
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + join_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _quiet_subregions(
    rms_list: list[float],
    peak_list: list[float],
    gap_start: float,
    gap_end: float,
    hop: float,
    rms_quiet: float,
    peak_quiet: float,
    min_seconds: float,
    bridge_seconds: float = 0.3,
) -> list[tuple[float, float]]:
    """Find the contiguous *actually-quiet* sub-spans inside an inter-vocal
    gap. A long gap can mix a loud outro/instrumental break with a genuinely
    silent pocket; judging the whole gap at once (as a single noisy-fraction
    test) throws the silent pocket out with the loud parts. Instead we walk
    the frames, growing a quiet run and tolerating noisy blips up to
    ``bridge_seconds`` (longer noisy stretches split the run). Each surviving
    run must still clear the overall noisy-fraction tolerance and the
    minimum-length floor."""
    start_idx = max(0, int(gap_start / hop))
    end_idx = min(len(rms_list), int(gap_end / hop))
    if end_idx <= start_idx:
        return []

    bridge = max(1, int(bridge_seconds / hop))
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    noisy_streak = 0
    for i in range(start_idx, end_idx):
        peak_val = peak_list[i] if (peak_list and i < len(peak_list)) else 0.0
        quiet = rms_list[i] < rms_quiet and peak_val < peak_quiet
        if quiet:
            if run_start is None:
                run_start = i
            noisy_streak = 0
        elif run_start is not None:
            noisy_streak += 1
            if noisy_streak > bridge:
                # Close the run at the first noisy frame of this streak so
                # trailing bleed isn't folded into the safe span.
                runs.append((run_start, i - noisy_streak + 1))
                run_start = None
                noisy_streak = 0
    if run_start is not None:
        runs.append((run_start, end_idx))

    min_frames = min_seconds / hop
    out: list[tuple[float, float]] = []
    for a, b in runs:
        if (b - a) >= min_frames and _quiet_fraction_ok(
            rms_list, peak_list, a, b, rms_quiet, peak_quiet
        ):
            out.append((a * hop, b * hop))
    return out


def vocal_safe_regions(
    transcription_segments: list[dict[str, Any]],
    envelope: dict[str, Any],
    aligned_words: list[dict[str, Any]] | None = None,
    word_prob_min: float = 0.35,
    segment_logprob_min: float = -1.2,
    stem_rms_presence: float = 0.02,
    stem_peak_presence: float = 0.08,
    stem_rms_quiet: float = 0.01,
    stem_peak_quiet: float = 0.04,
    min_safe_region_seconds: float = 1.5,
    duration_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Returns ``[{"start", "end", "safe", "reason"}, ...]``."""
    rms_list, peak_list, hop = _extract_vocal_envelope(envelope)

    if not rms_list:
        return [{
            "start": 0.0,
            "end": float(duration_seconds),
            "safe": True,
            "reason": "no_envelope",
        }]

    duration = float(duration_seconds) or (len(rms_list) * hop)

    if aligned_words is not None:
        intervals = _collect_from_alignment(
            aligned_words, rms_list, peak_list, hop,
            stem_rms_presence, stem_peak_presence,
        )
    else:
        intervals = _collect_from_whisper(
            transcription_segments, rms_list, peak_list, hop,
            word_prob_min, segment_logprob_min,
            stem_rms_presence, stem_peak_presence,
        )

    merged = _merge_intervals(intervals)

    safe_regions: list[dict[str, Any]] = []
    cursor = 0.0
    bounds = merged + [(duration, duration)]  # synthetic tail
    for word_start, word_end in bounds:
        gap_start = cursor
        gap_end = word_start
        cursor = word_end
        if gap_end - gap_start < min_safe_region_seconds:
            continue
        # Don't accept/reject the whole gap wholesale — a long inter-vocal
        # gap can hold a loud break and a silent pocket. Emit each quiet
        # sub-span inside it.
        for q_start, q_end in _quiet_subregions(
            rms_list, peak_list, gap_start, gap_end, hop,
            stem_rms_quiet, stem_peak_quiet, min_safe_region_seconds,
        ):
            safe_regions.append({
                "start": q_start,
                "end": q_end,
                "safe": True,
                "reason": "quiet_gap",
            })

    return safe_regions
