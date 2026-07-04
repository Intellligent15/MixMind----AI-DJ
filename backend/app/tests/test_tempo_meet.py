"""Tempo meet-in-the-middle (F3) + shared tempo-map math.

120 → 130 BPM is an ~8.3% gap: inside the (4%, 12%] meet window. The
midpoint is 125, so A speeds up over its last 16 bars and B enters
stretched only to 125 instead of 120.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.services.mixer.archetypes import expand, default_decision
from app.services.mixer.candidates import build_pair_candidates
from app.services.mixer.decision import TransitionStyle
from app.services.mixer.tempo_map import (
    a_output_sample,
    map_sample,
    ramp_time_map,
)
from app.services.mixer.validation import (
    enforce_revert_after_crossfade,
    validate_plan,
)
from app.tests.test_planner_v2 import FULL_SAFE, energy_curve_for, make_bundle

SR = 44100


# ------------------------------------------------------------- tempo_map

def test_ramp_time_map_is_monotonic_and_hits_rates():
    total, start, end = 100_000, 20_000, 60_000
    pairs = ramp_time_map(total, start, end, 1.0, 1.25)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    assert xs == sorted(xs) and ys == sorted(ys)
    # Identity before the ramp.
    assert map_sample(pairs, start) == start
    # After the ramp everything advances at rate 1.25 (shorter output).
    tail_in = total - end
    expected_end_target = pairs[-1][1] - tail_in / 1.25
    assert map_sample(pairs, end) == pytest.approx(expected_end_target, abs=2)
    # Ramp output duration ≈ integral of dt/rate < source duration.
    ramp_out = map_sample(pairs, end) - map_sample(pairs, start)
    assert (end - start) / 1.25 < ramp_out < (end - start)


def test_ramp_time_map_matches_executor_b_convention():
    """rate_before=rate_A, rate_after=1.0 reproduces the executor's
    historical B-ramp anchors."""
    rate_A = 120.0 / 126.0
    total, start, end = 500_000, 100_000, 200_000
    pairs = ramp_time_map(total, start, end, rate_A, 1.0)
    assert pairs[0] == (0, 0)
    assert map_sample(pairs, start) == pytest.approx(start / rate_A, abs=2)
    # Post-ramp advances 1:1.
    d_out = map_sample(pairs, end + 50_000) - map_sample(pairs, end)
    assert d_out == pytest.approx(50_000, abs=2)


def test_a_output_sample_identity_without_ramp():
    plan = [{"tool": "set_transition_window", "from_song_time_start": 60.0,
             "to_song_time_start": 0.0, "duration_bars": 8}]
    assert a_output_sample(plan, 120.0, 60.0, SR) == 60 * SR


def test_a_output_sample_maps_through_meet_ramp():
    # A speeds up 120 -> 125 over [28s, 60s]: the seam lands EARLIER in
    # output samples than in source samples.
    plan = [
        {"tool": "set_transition_window", "from_song_time_start": 60.0,
         "to_song_time_start": 0.0, "duration_bars": 8},
        {"tool": "set_tempo_ramp", "song": "A", "start_time": 28.0,
         "end_time": 60.0, "start_bpm": 120.0, "end_bpm": 125.0},
    ]
    out = a_output_sample(plan, 120.0, 60.0, SR)
    assert out < 60 * SR
    # 32 s ramp at average rate ~1.0206 -> ~31.35 s of output.
    expected = (28.0 + 32.0 / ((1.0 + 125.0 / 120.0) / 2.0)) * SR
    assert out == pytest.approx(expected, rel=0.001)


# ------------------------------------------------------------ archetypes

def _pair(a_bpm: float, b_bpm: float):
    a = make_bundle(bpm=a_bpm)
    b = make_bundle(bpm=b_bpm)
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    return a, b, cands


def test_expand_emits_meet_ramp_for_mid_gap():
    a, b, cands = _pair(120.0, 130.0)  # 8.3% gap
    plan = expand(default_decision(cands), a, b, cands)
    validate_plan(plan)
    a_ramp = next(c for c in plan
                  if c["tool"] == "set_tempo_ramp" and c.get("song") == "A")
    assert a_ramp["end_bpm"] == 125.0
    assert a_ramp["end_time"] == plan[0]["from_song_time_start"]
    assert a_ramp["start_time"] < a_ramp["end_time"]
    # B's post-crossfade ramp starts from the meet tempo, not A's.
    b_ramp = next(c for c in plan
                  if c["tool"] == "set_tempo_ramp" and c.get("song") == "B")
    assert b_ramp["start_bpm"] == 125.0
    assert b_ramp["end_bpm"] == 130.0


def test_expand_skips_meet_for_small_and_huge_gaps():
    for a_bpm, b_bpm in ((120.0, 122.0), (85.0, 170.0)):
        a, b, cands = _pair(a_bpm, b_bpm)
        plan = expand(default_decision(cands), a, b, cands)
        assert not any(
            c["tool"] == "set_tempo_ramp" and c.get("song") == "A"
            for c in plan
        ), f"unexpected meet ramp for {a_bpm}/{b_bpm}"


def test_expand_meet_disabled_by_flag():
    a, b, cands = _pair(120.0, 130.0)
    plan = expand(default_decision(cands), a, b, cands, tempo_meet=False)
    assert not any(
        c["tool"] == "set_tempo_ramp" and c.get("song") == "A" for c in plan
    )


def test_enforce_revert_leaves_a_ramp_alone():
    a, b, cands = _pair(120.0, 130.0)
    plan = expand(default_decision(cands), a, b, cands)
    a_ramp_before = next(c for c in plan
                         if c["tool"] == "set_tempo_ramp" and c.get("song") == "A")
    out = enforce_revert_after_crossfade(plan, b)
    a_ramp_after = next(c for c in out
                        if c["tool"] == "set_tempo_ramp" and c.get("song") == "A")
    assert a_ramp_after == a_ramp_before  # pre-seam by design, never deferred


# -------------------------------------------------------------- executor

def test_render_meet_ramp_stretches_a_and_halves_b_rate():
    from app.services.mixer.executor import render
    from app.services.mixer.types import SongRenderInputs
    from app.tests.test_mixer_executor import (
        _bundle, _fake_sf_read, _stems_dict,
    )

    long_dur = 5.0
    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle(bpm=120.0))
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle(bpm=130.0))

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "set_tempo_ramp", "song": "A", "start_time": 0.0,
         "end_time": 2.0, "start_bpm": 120.0, "end_bpm": 125.0},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A",
             "to_song": "B", "start_bar": 0, "duration_bars": 1,
             "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    stretch_calls = []

    def _spy_timemap(y, sr, pairs):
        stretch_calls.append(("timemap", len(y)))
        return y

    def _spy_stretch(y, sr, rate):
        stretch_calls.append(("stretch", rate))
        return y

    with (
        patch("soundfile.read", side_effect=_fake_sf_read()),
        patch("pyrubberband.pyrb.timemap_stretch", side_effect=_spy_timemap),
        patch("pyrubberband.pyrb.time_stretch", side_effect=_spy_stretch),
    ):
        render(plan, a, b)

    # A's four stems + the master all went through the ramp map.
    assert sum(1 for kind, _ in stretch_calls if kind == "timemap") == 5
    # B was stretched to the MEET tempo (125/130), not to A's native 120.
    b_rates = [v for kind, v in stretch_calls if kind == "stretch"]
    assert b_rates and all(r == pytest.approx(125.0 / 130.0) for r in b_rates)


# --------------------------------------------------------------- stitcher

def test_get_mix0_sample_accounts_for_a_meet_ramp():
    from app.workers.stitch_queue import _get_mix0_sample

    base_window = {"tool": "set_transition_window",
                   "from_song_time_start": 60.0,
                   "to_song_time_start": 10.0, "duration_bars": 8}
    meet_plan = [
        dict(base_window),
        {"tool": "set_tempo_ramp", "song": "A", "start_time": 28.0,
         "end_time": 60.0, "start_bpm": 120.0, "end_bpm": 125.0},
    ]
    plain_plan = [dict(base_window)]

    # Same B mapping in both calls (rate fixed here); only A's seam moves.
    rate = 125.0 / 130.0
    s_meet = _get_mix0_sample(meet_plan, rate, 40.0, SR, a_bpm=120.0)
    s_plain = _get_mix0_sample(plain_plan, rate, 40.0, SR, a_bpm=120.0)
    # A sped up 120->125 over 32 s: its seam lands ~0.65 s earlier.
    shift = (s_plain - s_meet) / SR
    assert shift == pytest.approx(0.646, abs=0.02)


def test_qa_seam_maps_through_meet_ramp():
    from app.services.mixer.qa import _window_bounds

    a = make_bundle(bpm=120.0)
    plan = [
        {"tool": "set_transition_window", "from_song_time_start": 60.0,
         "to_song_time_start": 0.0, "duration_bars": 8},
        {"tool": "set_tempo_ramp", "song": "A", "start_time": 28.0,
         "end_time": 60.0, "start_bpm": 120.0, "end_bpm": 125.0},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A",
             "to_song": "B", "start_bar": 0, "duration_bars": 8,
             "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    seam, end = _window_bounds(plan, a)
    assert seam < 60.0                     # mapped through the ramp
    assert end - seam == pytest.approx(8 * (60.0 / 125.0) * 4)  # meet-tempo bars
