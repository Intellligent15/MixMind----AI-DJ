"""Render QA — the DJ listening back to their own transition.

`score_render` inspects a rendered transition WAV against its plan and
flags the objective failure modes a listener would notice immediately:

  * clipping        — samples pinned at full scale
  * dropout         — unintended silence inside the transition window
  * click           — a sample-to-sample discontinuity far above the
                      song's own transient floor near the seam
  * rms_jump        — a big loudness step across the seam

Verdicts: "pass" | "warn" | "fail". The render worker re-plans once on
"fail" (with the failed style excluded); "warn" ships but is surfaced in
the debug UI. Thresholds are deliberately conservative — a false reroll
costs a render, a false pass costs nothing that the listener wasn't
already getting today.

Pure module: bytes + plan + analysis in, report out. No DB, no storage.
`scripts/eval_transitions.py` imports from here so offline evals and the
in-pipeline gate can never drift apart.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from app.services.mixer.tempo_map import a_output_sample, a_ramp_from_plan
from app.services.mixer.types import AnalysisBundle

logger = logging.getLogger(__name__)

# Loudness step across the seam (2-bar RMS windows), in dB.
RMS_DELTA_WARN_DB = 6.0
RMS_DELTA_FAIL_DB = 8.0
# Fraction of samples at/above full scale. 1e-4 of a 3-minute file is
# ~1600 samples of flat-top — clearly audible crunch.
CLIP_WARN_RATIO = 1e-5
CLIP_FAIL_RATIO = 1e-4
CLIP_SAMPLE_LEVEL = 0.999
# Unintended-silence detector inside the transition window.
DROPOUT_WINDOW_MS = 100.0
DROPOUT_MIN_MS = 400.0
DROPOUT_FLOOR_DBFS = -60.0
# Styles whose whole point is a moment of silence.
DROPOUT_EXEMPT_STYLES = ("vinyl_stop", "backspin")
# Click detector: max |x[n]-x[n-1]| in the 4 bars around the seam,
# relative to the 99.9th percentile of the same measure over the song
# body before the transition.
CLICK_WARN_RATIO = 3.0
CLICK_FAIL_RATIO = 4.5
# Segments quieter than this are ignored by the RMS-step check — a step
# from silence to silence is meaningless.
RMS_CHECK_FLOOR_DBFS = -50.0

_VERDICT_RANK = {"pass": 0, "warn": 1, "fail": 2}


@dataclass
class QAReport:
    verdict: str = "pass"                      # pass | warn | fail
    metrics: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)   # e.g. ["fail:clipping"]

    def worse_than(self, other: "QAReport") -> bool:
        return _VERDICT_RANK[self.verdict] > _VERDICT_RANK[other.verdict]


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return float("-inf")
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(r, 1e-9))


def _snap_downbeat(t: float, downbeats: list[float]) -> float:
    if not downbeats:
        return t
    for d in downbeats:
        if d >= t:
            return d
    return downbeats[-1]


def _window_bounds(plan: list[dict], a: AnalysisBundle) -> tuple[float, float] | None:
    """(seam_seconds, transition_end_seconds) in OUTPUT time.

    Output time equals A's original time up to the seam, except when the
    plan carries an A-side meet-in-the-middle tempo ramp — then the seam
    is mapped through the shared tempo map, and the bars inside the
    window run at the meet tempo (the ramp's end_bpm).
    """
    window = next(
        (c for c in plan if c.get("tool") == "set_transition_window"), None
    )
    if window is None or not a.bpm:
        return None
    seam_orig = _snap_downbeat(
        float(window.get("from_song_time_start", 0.0)), list(a.downbeats or [])
    )
    sr = 44100
    seam = a_output_sample(plan, a.bpm, seam_orig, sr) / sr
    a_ramp = a_ramp_from_plan(plan)
    window_bpm = float(a_ramp.get("end_bpm") or a.bpm) if a_ramp else a.bpm
    spb = (60.0 / window_bpm) * a.time_signature
    stem_calls = [c for c in plan if c.get("tool") == "crossfade_stem"]
    total_bars = max(
        (int(c.get("start_bar", 0)) + int(c.get("duration_bars", 0))
         for c in stem_calls),
        default=int(window.get("duration_bars", 8)),
    )
    return seam, seam + total_bars * spb


def _check_clipping(audio: np.ndarray, metrics: dict) -> str | None:
    ratio = float(np.mean(np.abs(audio) >= CLIP_SAMPLE_LEVEL))
    metrics["clip_ratio"] = round(ratio, 8)
    if ratio > CLIP_FAIL_RATIO:
        return "fail"
    if ratio > CLIP_WARN_RATIO:
        return "warn"
    return None


def _check_rms_step(
    audio: np.ndarray, sr: int, seam_s: float, spb: float, metrics: dict
) -> str | None:
    seam = int(seam_s * sr)
    two_bars = int(2 * spb * sr)
    pre = audio[max(0, seam - two_bars): seam]
    post = audio[seam: seam + two_bars]
    pre_db, post_db = _rms_db(pre), _rms_db(post)
    metrics["rms_pre_db"] = round(pre_db, 2)
    metrics["rms_post_db"] = round(post_db, 2)
    if pre_db < RMS_CHECK_FLOOR_DBFS or post_db < RMS_CHECK_FLOOR_DBFS:
        metrics["rms_delta_db"] = None  # a step to/from silence isn't a step
        return None
    delta = abs(pre_db - post_db)
    metrics["rms_delta_db"] = round(delta, 2)
    if delta > RMS_DELTA_FAIL_DB:
        return "fail"
    if delta > RMS_DELTA_WARN_DB:
        return "warn"
    return None


def _check_dropout(
    audio: np.ndarray, sr: int, seam_s: float, end_s: float,
    style: str | None, metrics: dict,
) -> str | None:
    if style in DROPOUT_EXEMPT_STYLES:
        metrics["dropout_ms"] = None
        return None
    lo = max(0, int(seam_s * sr))
    hi = min(audio.shape[0], int(end_s * sr))
    region = audio[lo:hi]
    hop = max(1, int(DROPOUT_WINDOW_MS / 1000.0 * sr))
    n_hops = region.shape[0] // hop
    if n_hops < 2:
        metrics["dropout_ms"] = 0.0
        return None
    mono = region[: n_hops * hop].mean(axis=1).reshape(n_hops, hop)
    hop_rms_db = 20.0 * np.log10(
        np.maximum(np.sqrt(np.mean(mono.astype(np.float64) ** 2, axis=1)), 1e-9)
    )
    silent = hop_rms_db < DROPOUT_FLOOR_DBFS
    # Longest run of consecutive silent hops.
    longest = run = 0
    for s in silent:
        run = run + 1 if s else 0
        longest = max(longest, run)
    dropout_ms = longest * DROPOUT_WINDOW_MS
    metrics["dropout_ms"] = round(dropout_ms, 1)
    if dropout_ms >= DROPOUT_MIN_MS:
        return "fail"
    return None


def _check_click(
    audio: np.ndarray, sr: int, seam_s: float, spb: float, metrics: dict
) -> str | None:
    mono = audio.mean(axis=1)
    seam = int(seam_s * sr)
    two_bars = int(2 * spb * sr)
    lo = max(1, seam - two_bars)
    hi = min(mono.shape[0], seam + two_bars)
    if hi - lo < sr // 10:
        metrics["click_ratio"] = None
        return None
    region_max = float(np.max(np.abs(np.diff(mono[lo:hi]))))
    # Baseline: the song body before the transition region (fall back to
    # the whole file when the seam is at 0).
    body = mono[: lo - 1] if lo > sr else mono
    body_diff = np.abs(np.diff(body))
    if body_diff.size < sr // 10:
        metrics["click_ratio"] = None
        return None
    baseline = float(np.percentile(body_diff, 99.9))
    if baseline < 1e-6:
        metrics["click_ratio"] = None  # near-silent body — ratio meaningless
        return None
    ratio = region_max / baseline
    metrics["click_ratio"] = round(ratio, 2)
    if ratio > CLICK_FAIL_RATIO:
        return "fail"
    if ratio > CLICK_WARN_RATIO:
        return "warn"
    return None


def score_render(
    wav_bytes: bytes,
    plan: list[dict],
    a: AnalysisBundle,
    style: str | None = None,
) -> QAReport:
    """Score one rendered transition. Never raises — an unscoreable render
    returns a pass with whatever metrics were computable (QA must not be
    the thing that breaks a render)."""
    report = QAReport()
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True, dtype="float32")
        report.metrics["duration_s"] = round(audio.shape[0] / sr, 2)

        bounds = _window_bounds(plan, a)
        if bounds is None:
            report.flags.append("skip:no_window_or_bpm")
            return report
        seam_s, end_s = bounds
        report.metrics["seam_s"] = round(seam_s, 2)
        spb = (60.0 / a.bpm) * a.time_signature

        checks = (
            ("clipping", _check_clipping(audio, report.metrics)),
            ("rms_jump", _check_rms_step(audio, sr, seam_s, spb, report.metrics)),
            ("dropout", _check_dropout(audio, sr, seam_s, end_s, style, report.metrics)),
            ("click", _check_click(audio, sr, seam_s, spb, report.metrics)),
        )
        for name, level in checks:
            if level is not None:
                report.flags.append(f"{level}:{name}")
                if _VERDICT_RANK[level] > _VERDICT_RANK[report.verdict]:
                    report.verdict = level
    except Exception:
        logger.exception("qa: scoring failed; treating as pass")
        report.flags.append("skip:error")
    return report
