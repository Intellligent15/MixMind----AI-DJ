"""Tests for services/mixer/qa.py — the render-QA scorer — and the
render worker's QA-retry loop.

Scorer tests build small synthetic WAVs with one injected defect each and
assert exactly that defect is flagged. The pass-case signal is a steady
sine: flat RMS, no clipping, no silence, uniform sample-to-sample deltas.
"""

from __future__ import annotations

import io
import uuid
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile as sf

from app.services.mixer.qa import QAReport, score_render
from app.services.mixer.types import AnalysisBundle

SR = 44100
DUR = 30.0
N = int(SR * DUR)
SEAM = 10.0  # seconds; a downbeat on the 120bpm grid below
FREQ = 200.0  # integer cycles at every whole second -> zero crossings at seams


def _bundle(bpm: float = 120.0) -> AnalysisBundle:
    sec_per_bar = (60.0 / bpm) * 4
    return AnalysisBundle(
        bpm=bpm, key="C", camelot_key="8B", time_signature=4,
        beat_grid=[i * 0.5 for i in range(int(DUR / 0.5))],
        downbeats=[i * sec_per_bar for i in range(int(DUR / sec_per_bar))],
        sections=[{"start": 0.0, "end": DUR, "label": "body"}],
        duration=DUR,
    )


def _plan(duration_bars: int = 2) -> list[dict]:
    return [
        {"tool": "set_transition_window",
         "from_song_time_start": SEAM, "to_song_time_start": 0.0,
         "duration_bars": duration_bars},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A",
             "to_song": "B", "start_bar": 0, "duration_bars": duration_bars,
             "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]


def _sine(amp: float = 0.3, n: int = N) -> np.ndarray:
    x = (amp * np.sin(2 * np.pi * FREQ * np.arange(n) / SR)).astype(np.float32)
    return np.column_stack((x, x))


def _wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def test_clean_render_passes():
    report = score_render(_wav(_sine()), _plan(), _bundle())
    assert report.verdict == "pass"
    assert report.flags == []
    assert report.metrics["clip_ratio"] == 0.0
    assert report.metrics["dropout_ms"] == 0.0


def test_clipping_fails():
    audio = _sine()
    audio[int(11.0 * SR): int(11.5 * SR)] = 1.0
    report = score_render(_wav(audio), _plan(), _bundle())
    assert "fail:clipping" in report.flags
    assert report.verdict == "fail"


def test_dropout_inside_window_fails():
    audio = _sine()
    lo, hi = int(11.0 * SR), int(12.0 * SR)
    # Fade edges so the gap itself doesn't also read as a click.
    ramp = int(0.005 * SR)
    audio[lo - ramp: lo] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)[:, None]
    audio[hi: hi + ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)[:, None]
    audio[lo:hi] = 0.0
    report = score_render(_wav(audio), _plan(), _bundle())
    assert "fail:dropout" in report.flags
    assert report.metrics["dropout_ms"] >= 400


def test_dropout_exempt_for_stop_styles():
    audio = _sine()
    audio[int(11.0 * SR): int(12.0 * SR)] = 0.0
    report = score_render(_wav(audio), _plan(), _bundle(), style="vinyl_stop")
    assert not any("dropout" in f for f in report.flags)
    assert report.metrics["dropout_ms"] is None


def test_click_at_seam_fails():
    audio = _sine()
    audio[int((SEAM + 0.5) * SR), :] += 0.5
    report = score_render(_wav(audio), _plan(), _bundle())
    assert "fail:click" in report.flags
    assert report.metrics["click_ratio"] > 4.5


def test_rms_jump_across_seam_fails():
    audio = _sine(amp=0.5)
    seam_samp = int(SEAM * SR)
    audio[seam_samp:] *= 0.1  # -20 dB step, phase-continuous at the seam
    report = score_render(_wav(audio), _plan(), _bundle())
    assert "fail:rms_jump" in report.flags
    assert report.metrics["rms_delta_db"] > 8.0


def test_rms_step_to_silence_is_not_a_step():
    audio = _sine()
    audio[int(SEAM * SR):] = 0.0  # legit end-of-audio silence
    report = score_render(_wav(audio), _plan(), _bundle())
    assert not any("rms_jump" in f for f in report.flags)
    assert report.metrics["rms_delta_db"] is None


def test_undecodable_bytes_pass_with_skip_flag():
    report = score_render(b"not a wav", _plan(), _bundle())
    assert report.verdict == "pass"
    assert "skip:error" in report.flags


def test_plan_without_window_skips():
    report = score_render(_wav(_sine()), [{"tool": "volume_fade"}], _bundle())
    assert report.verdict == "pass"
    assert "skip:no_window_or_bpm" in report.flags


def test_report_ranking():
    assert QAReport(verdict="fail").worse_than(QAReport(verdict="warn"))
    assert not QAReport(verdict="pass").worse_than(QAReport(verdict="pass"))
