"""Hook teasing (F9): hook finder, pocket finder, expansion wiring, and
the executor's vocal_tease tool."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import numpy as np
import pytest

from app.services.mixer.archetypes import default_decision, expand
from app.services.mixer.candidates import build_pair_candidates
from app.services.mixer.hooks import find_hook, find_pocket
from app.services.mixer.validation import validate_plan
from app.tests.test_planner_v2 import FULL_SAFE, energy_curve_for, make_bundle


def _segments_with_chorus() -> list[dict]:
    """A verse line (once) + a chorus line repeated 3x, hottest at 120s."""
    segs = [
        {"start": 10.0, "end": 14.0, "text": "just a quiet verse line here"},
    ]
    for t in (40.0, 120.0, 200.0):
        segs.append({
            "start": t, "end": t + 4.0,
            "text": "we own the night tonight",
        })
    return segs


def _energy_peak_at_120(duration: float = 240.0) -> list[float]:
    return [1.0 if 100 <= i <= 150 else 0.3 for i in range(int(duration))]


# ------------------------------------------------------------------ finders

def test_find_hook_picks_repeated_line_in_hottest_section():
    hook = find_hook(_segments_with_chorus(), None, _energy_peak_at_120())
    assert hook is not None
    assert hook.start == 120.0            # the chorus occurrence, not 40.0
    assert "own the night" in hook.text.lower()


def test_find_hook_needs_repetition():
    segs = [
        {"start": 10.0, "end": 14.0, "text": "one line sung only once"},
        {"start": 40.0, "end": 44.0, "text": "another line also sung once"},
    ]
    assert find_hook(segs, None, None) is None
    assert find_hook(None, None, None) is None
    assert find_hook([], None, None) is None


def test_find_pocket_prefers_late_drummy_gaps():
    seam = 200.0
    safe = [
        {"start": 20.0, "end": 60.0, "safe": True},    # too early (zone 120+)
        {"start": 130.0, "end": 150.0, "safe": True},  # in zone
        {"start": 170.0, "end": 190.0, "safe": True},  # in zone, later
    ]
    envelopes = {
        "frame_hz": 10,
        "drums": {"rms": [0.15] * 2000, "peak": [0.2] * 2000},
    }
    t = find_pocket(4.0, seam, safe, envelopes)
    assert t is not None
    assert 170.0 <= t <= 186.0            # latest qualifying span


def test_find_pocket_rejects_drumless_spans():
    seam = 200.0
    safe = [{"start": 170.0, "end": 190.0, "safe": True}]
    envelopes = {
        "frame_hz": 10,
        "drums": {"rms": [0.0] * 2000, "peak": [0.0] * 2000},
    }
    assert find_pocket(4.0, seam, safe, envelopes) is None
    assert find_pocket(4.0, seam, [], envelopes) is None


# ---------------------------------------------------------------- expansion

def _teasable_pair():
    a = make_bundle()
    b = dataclasses.replace(
        make_bundle(bpm=120.0),
        transcription_segments=_segments_with_chorus(),
    )
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), _energy_peak_at_120(),
        FULL_SAFE, FULL_SAFE,
    )
    return a, b, cands


def test_expand_emits_vocal_tease_when_opted_in():
    a, b, cands = _teasable_pair()
    decision = default_decision(cands).model_copy(update={"tease": True})
    plan = expand(
        decision, a, b, cands,
        a_safe_regions=FULL_SAFE, b_energy_curve=_energy_peak_at_120(),
    )
    validate_plan(plan)
    tease = next(c for c in plan if c["tool"] == "vocal_tease")
    assert tease["hook_start"] == 120.0
    assert tease["bpm_from"] == b.bpm and tease["bpm_to"] == a.bpm
    assert tease["semitones"] == 0        # compatible keys
    assert tease["start_time"] < plan[0]["from_song_time_start"]


def test_expand_no_tease_by_default_or_without_hook():
    a, b, cands = _teasable_pair()
    plan = expand(
        default_decision(cands), a, b, cands,
        a_safe_regions=FULL_SAFE, b_energy_curve=_energy_peak_at_120(),
    )
    assert not any(c["tool"] == "vocal_tease" for c in plan)

    # Opted in but B has no transcription -> silently dropped.
    b_mute = make_bundle(bpm=120.0)
    cands2 = build_pair_candidates(
        a, b_mute, energy_curve_for(a.duration), energy_curve_for(240.0),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands2).model_copy(update={"tease": True})
    plan = expand(decision, a, b_mute, cands2, a_safe_regions=FULL_SAFE)
    assert not any(c["tool"] == "vocal_tease" for c in plan)


def test_expand_drops_tease_on_uncappable_key_clash():
    a, b, cands = _teasable_pair()
    a_clash = dataclasses.replace(a, key="C", camelot_key="8B")
    b_clash = dataclasses.replace(b, key="F#", camelot_key="2B")  # 6 semis
    decision = default_decision(cands).model_copy(update={"tease": True})
    plan = expand(
        decision, a_clash, b_clash, cands,
        a_safe_regions=FULL_SAFE, b_energy_curve=_energy_peak_at_120(),
    )
    assert not any(c["tool"] == "vocal_tease" for c in plan)


# ----------------------------------------------------------------- executor

def test_apply_vocal_tease_places_hook_and_ducks_other():
    from app.services.mixer.executor import _apply_vocal_tease

    sr = 44100
    n = sr * 10
    a_stems = {
        "vocals": np.zeros((n, 2), dtype=np.float32),
        "drums": np.full((n, 2), 0.2, dtype=np.float32),
        "bass": np.full((n, 2), 0.2, dtype=np.float32),
        "other": np.full((n, 2), 0.4, dtype=np.float32),
    }
    b_vocal = np.full((n, 2), 0.5, dtype=np.float32)
    call = {
        "tool": "vocal_tease", "song": "A", "start_time": 4.0,
        "hook_start": 1.0, "hook_end": 3.0,
        "bpm_from": 120.0, "bpm_to": 120.0,   # no stretch needed
        "semitones": 0, "gain": 0.9,
    }
    _apply_vocal_tease(a_stems, b_vocal, call)

    mid = int(5.0 * 44100)   # middle of the 2 s tease at 4.0s
    assert a_stems["vocals"][mid, 0] == pytest.approx(0.45, abs=0.01)
    assert a_stems["other"][mid, 0] == pytest.approx(0.4 * 0.65, abs=0.01)
    # Outside the tease: untouched.
    before = int(3.5 * 44100)
    assert a_stems["vocals"][before, 0] == 0.0
    assert a_stems["other"][before, 0] == pytest.approx(0.4)
    # Drums never ducked.
    assert a_stems["drums"][mid, 0] == pytest.approx(0.2)


def test_apply_vocal_tease_stretches_and_shifts_when_needed():
    from app.services.mixer.executor import _apply_vocal_tease

    sr = 44100
    n = sr * 10
    a_stems = {s: np.zeros((n, 2), dtype=np.float32)
               for s in ("vocals", "drums", "bass", "other")}
    b_vocal = np.full((n, 2), 0.5, dtype=np.float32)
    call = {
        "tool": "vocal_tease", "song": "A", "start_time": 4.0,
        "hook_start": 1.0, "hook_end": 3.0,
        "bpm_from": 100.0, "bpm_to": 120.0, "semitones": 2, "gain": 0.9,
    }
    with (
        patch("pyrubberband.pyrb.time_stretch",
              side_effect=lambda y, sr_, rate: y) as stretch,
        patch("pyrubberband.pyrb.pitch_shift",
              side_effect=lambda y, sr_, semis: y) as shift,
    ):
        _apply_vocal_tease(a_stems, b_vocal, call)
    assert stretch.call_args[0][2] == pytest.approx(1.2)  # 120/100
    assert shift.call_args[0][2] == 2
