"""Unit tests for planner v2: candidates, decision schema, archetype
expansion, repair-not-reject validation, and orchestration. All pure —
no DB, no storage, no network."""

from __future__ import annotations

import asyncio

import pytest

from app.services.mixer.archetypes import (
    ArchetypeError,
    camelot_compatible,
    default_decision,
    expand,
)
from app.services.mixer.candidates import (
    build_in_candidates,
    build_out_candidates,
    build_pair_candidates,
    enrich_sections,
    max_seam_time,
)
from app.services.mixer.decision import (
    STYLE_DURATION_CHOICES,
    TransitionDecision,
    TransitionExtra,
    TransitionStyle,
)
from app.services.mixer.planner_v2 import SongMeta, build_plan_v2
from app.services.mixer.types import AnalysisBundle
from app.services.mixer.validation import (
    enforce_revert_after_crossfade,
    repair_plan,
    validate_plan,
)


def make_bundle(
    bpm: float = 120.0,
    duration: float = 240.0,
    key: str = "C",
    camelot: str = "8B",
    n_sections: int = 5,
) -> AnalysisBundle:
    spb = 60.0 / bpm
    beat_grid = [i * spb for i in range(int(duration / spb))]
    downbeats = beat_grid[::4]
    sec_len = duration / n_sections
    sections = [
        {"start": i * sec_len, "end": (i + 1) * sec_len, "label": f"section_{i}"}
        for i in range(n_sections)
    ]
    return AnalysisBundle(
        bpm=bpm, key=key, camelot_key=camelot, time_signature=4,
        beat_grid=beat_grid, downbeats=downbeats, sections=sections,
        duration=duration,
    )


def energy_curve_for(duration: float, peak_at: float = 0.5) -> list[float]:
    n = int(duration)
    return [
        1.0 - abs((i / max(1, n - 1)) - peak_at) for i in range(n)
    ]


FULL_SAFE = [{"start": 0.0, "end": 10_000.0, "safe": True}]


# ---------------------------------------------------------------- candidates

def test_out_candidates_respect_headroom_and_downbeats():
    a = make_bundle()
    cands = build_out_candidates(a, energy_curve_for(a.duration), FULL_SAFE)
    ceiling = max_seam_time(a.duration, a.bpm, a.time_signature)
    assert 1 <= len(cands) <= 5
    for c in cands:
        assert c.time <= ceiling
        assert any(abs(c.time - d) < 1e-6 for d in a.downbeats)
        assert c.id.startswith("A")
        assert c.vocal_safe is True


def test_in_candidates_stay_in_first_half():
    b = make_bundle()
    cands = build_in_candidates(b, energy_curve_for(b.duration, peak_at=0.3), [])
    assert cands
    for c in cands:
        assert c.time <= b.duration * 0.5 + 1e-6
        assert c.id.startswith("B")
        # No vocal-safety data → conservatively unsafe.
        assert c.vocal_safe is False


def test_candidates_empty_for_too_short_song():
    a = make_bundle(duration=20.0)
    assert build_out_candidates(a, [], []) == []


def test_enrich_sections_normalizes_energy_and_keeps_real_labels():
    sections = [
        {"start": 0.0, "end": 10.0, "label": "section_0"},
        {"start": 10.0, "end": 20.0, "label": "chorus"},
    ]
    curve = [0.1] * 10 + [0.4] * 10
    out = enrich_sections(sections, curve)
    assert out[1]["energy"] == 1.0
    assert out[0]["energy"] == 0.25
    assert "label" not in out[0]      # opaque cluster id dropped
    assert out[1]["label"] == "chorus"  # real label kept


# ------------------------------------------------------------------ decision

def test_decision_caps_extras_and_normalizes_duration():
    d = TransitionDecision.model_validate({
        "out": "A1", "in": "B1", "style": "drop_swap",
        "duration_bars": 9,
        "extras": ["echo_tail", "bass_kill", "reverb_tail"],
    })
    assert len(d.extras) == 2
    assert d.normalized_duration() in STYLE_DURATION_CHOICES[TransitionStyle.drop_swap]


def test_camelot_compatibility_rules():
    assert camelot_compatible("8A", "8B")    # relative major/minor
    assert camelot_compatible("8A", "9A")    # neighbour
    assert camelot_compatible("12B", "1B")   # wheel wraps
    assert not camelot_compatible("8A", "3B")
    assert camelot_compatible(None, "3B")    # unknown → assume fine


# ---------------------------------------------------------------- archetypes

@pytest.mark.parametrize("style", list(TransitionStyle))
def test_every_archetype_expands_to_a_valid_plan(style):
    a = make_bundle(bpm=126.0, key="Am", camelot="8A")
    b = make_bundle(bpm=120.0, key="Fm", camelot="4A")  # key clash
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=style)
    plan = expand(decision, a, b, cands, pitch_mode="temporary")

    validate_plan(plan)  # must not raise
    assert plan[0]["tool"] == "set_transition_window"

    window = plan[0]
    spb_b = (60.0 / b.bpm) * b.time_signature
    stem_calls = [c for c in plan if c["tool"] == "crossfade_stem"]
    crossfade_end_b = window["to_song_time_start"] + spb_b * max(
        c["start_bar"] + c["duration_bars"] for c in stem_calls
    )
    for call in plan:
        if call["tool"] == "set_tempo_ramp":
            assert call["start_time"] >= crossfade_end_b - 1e-6
        if call["tool"] == "temporary_pitch_shift":
            assert call["start_time"] >= crossfade_end_b - 1e-6
            assert abs(call["semitones"]) <= 2


def test_whole_song_and_off_modes_emit_no_pitch_tools():
    """Outside "temporary" mode the plan must NEVER glide pitch — that is
    the whole point of the whole-song feature. Tempo ramps remain."""
    a = make_bundle(bpm=126.0, key="Am", camelot="8A")
    b = make_bundle(bpm=120.0, key="Fm", camelot="4A")  # hard key clash
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.smooth_blend)
    for mode in ("whole_song", "off"):
        plan = expand(decision, a, b, cands, pitch_mode=mode)
        validate_plan(plan)
        tools = {c["tool"] for c in plan}
        assert "temporary_pitch_shift" not in tools
        assert "pitch_shift" not in tools
        assert "set_tempo_ramp" in tools  # beatmatching is unaffected


def test_strip_pitch_tools_removes_both_kinds():
    from app.services.mixer.validation import strip_pitch_tools
    plan = [
        {"tool": "set_transition_window", "from_song_time_start": 10.0,
         "to_song_time_start": 0.0, "duration_bars": 8},
        {"tool": "temporary_pitch_shift", "song": "B", "start_time": 30.0,
         "semitones": -2, "fade_in_bars": 0, "hold_bars": 0,
         "fade_out_bars": 4},
        {"tool": "pitch_shift", "song": "B", "semitones": 1},
        {"tool": "set_tempo_ramp", "song": "B", "start_time": 30.0,
         "end_time": 60.0, "start_bpm": 126.0, "end_bpm": 120.0},
    ]
    out = strip_pitch_tools(plan)
    tools = [c["tool"] for c in out]
    assert "temporary_pitch_shift" not in tools
    assert "pitch_shift" not in tools
    assert tools == ["set_transition_window", "set_tempo_ramp"]


def test_pair_facts_routes_unresolvable_clash_to_short_styles():
    from app.services.mixer.planner_v2 import _pair_facts
    a = make_bundle(key="Am", camelot="8A")
    b = make_bundle(key="C#m", camelot="12A")  # beyond the ±2 cap
    verdict = _pair_facts(a, b, pitch_mode="whole_song")["key_verdict"]
    assert "cannot be matched" in verdict
    assert "drop_swap" in verdict
    # temporary mode keeps the old hold-in-key wording
    verdict_t = _pair_facts(a, b, pitch_mode="temporary")["key_verdict"]
    assert "held" in verdict_t


def test_drum_bridge_offsets_drums():
    a, b = make_bundle(), make_bundle(bpm=124.0)
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.drum_bridge)
    plan = expand(decision, a, b, cands)
    drums = next(c for c in plan if c.get("stem") == "drums")
    # Bass gets the midpoint EQ-swap on long blends; vocals/other share
    # the full window from bar 0.
    others = [c for c in plan if c["tool"] == "crossfade_stem"
              and c["stem"] in ("vocals", "other")]
    bass = next(c for c in plan if c.get("stem") == "bass")
    assert drums["start_bar"] > 0
    assert all(c["start_bar"] == 0 for c in others)
    assert bass["start_bar"] > 0 and bass["duration_bars"] == 2


def test_stutter_skips_loop_when_not_vocal_safe():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        [], [],  # no vocal data → unsafe
    )
    decision = default_decision(cands, style=TransitionStyle.stutter_buildup)
    plan = expand(decision, a, b, cands)
    assert not any(c["tool"] == "loop_section" for c in plan)
    validate_plan(plan)


def test_acapella_out_holds_a_vocals_over_b_backing():
    """The requested feature: B's instrumental takes over at the seam
    while A's vocals ride on top, handing over late at a safe bar."""
    a, b = make_bundle(), make_bundle(bpm=124.0)
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.acapella_out)
    plan = expand(decision, a, b, cands, pitch_mode="whole_song",
                  a_safe_regions=FULL_SAFE, b_safe_regions=FULL_SAFE)
    validate_plan(plan)
    xfades = [c for c in plan if c["tool"] == "crossfade_stem"]
    vocals = next(c for c in xfades if c["stem"] == "vocals")
    drums = next(c for c in xfades if c["stem"] == "drums")
    bass = next(c for c in xfades if c["stem"] == "bass")
    other = next(c for c in xfades if c["stem"] == "other")
    # Staged backing: drums/melodics crossfade from bar 0, bass
    # quick-swaps (~1 bar) a couple of bars in — basses never overlap.
    assert drums["start_bar"] == 0 and other["start_bar"] == 0
    assert bass["start_bar"] == 2 and bass["duration_bars"] == 1
    assert all(c["duration_bars"] <= 4 for c in (drums, bass, other))
    assert vocals["start_bar"] >= 8                          # A vocal rides
    duration = plan[0]["duration_bars"]
    assert vocals["start_bar"] + vocals["duration_bars"] <= duration
    # B's melodics ride DUCKED under the vocal, restored at handover with
    # the exact reciprocal gain (volume_fade end_gain persists, and gains
    # compound — reciprocal restore is what returns net unity).
    ducks = [c for c in plan if c["tool"] == "volume_fade"
             and c.get("stem") == "other" and c["song"] == "B"]
    assert len(ducks) == 2
    assert ducks[0]["end_gain"] < 1.0
    assert abs(ducks[0]["end_gain"] * ducks[1]["end_gain"] - 1.0) < 1e-3
    spb_b = (60.0 / b.bpm) * b.time_signature
    expected_restore = plan[0]["to_song_time_start"] + vocals["start_bar"] * spb_b
    assert abs(ducks[1]["start_time"] - expected_restore) < 1e-3


def test_acapella_out_handover_lands_on_a_safe_bar():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.acapella_out)
    seam_a = cands.find(decision.out).time
    spb_a = (60.0 / a.bpm) * a.time_signature  # 2.0s
    # Only bar 10 after the seam is vocal-safe (needs [t, t+spb] inside).
    safe = [{"start": seam_a + 10 * spb_a - 0.1,
             "end": seam_a + 11 * spb_a + 0.1, "safe": True}]
    plan = expand(decision, a, b, cands, a_safe_regions=safe)
    vocals = next(c for c in plan if c["tool"] == "crossfade_stem" and c["stem"] == "vocals")
    assert vocals["start_bar"] == 10


def test_acapella_out_masks_when_no_safe_bar():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.acapella_out)
    plan = expand(decision, a, b, cands, a_safe_regions=[])  # no vocal data
    vocals = next(c for c in plan if c["tool"] == "crossfade_stem" and c["stem"] == "vocals")
    assert vocals["duration_bars"] == 4  # longer fade masks the chop
    validate_plan(plan)


def test_acapella_in_teases_b_vocals_over_a_backing():
    """The vice-versa: B's vocals arrive at bar 0 over A's still-playing
    backing; A's instrumental swaps to B's later."""
    a, b = make_bundle(), make_bundle(bpm=124.0)
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.acapella_in)
    plan = expand(decision, a, b, cands,
                  a_safe_regions=FULL_SAFE, b_safe_regions=FULL_SAFE)
    validate_plan(plan)
    vocals = next(c for c in plan if c["tool"] == "crossfade_stem" and c["stem"] == "vocals")
    backing = [c for c in plan if c["tool"] == "crossfade_stem"
               and c["stem"] != "vocals"]
    assert vocals["start_bar"] == 0                          # B vocal teases
    assert vocals["duration_bars"] == 2                      # clean exit (safe)
    assert all(c["start_bar"] >= 4 for c in backing)         # A beat rides
    bass = next(c for c in backing if c["stem"] == "bass")
    others = [c for c in backing if c["stem"] != "bass"]
    assert bass["duration_bars"] == 1                        # quick bass swap
    assert bass["start_bar"] > min(c["start_bar"] for c in others)
    # A's melodics duck under the teasing vocal.
    duck = next(c for c in plan if c["tool"] == "volume_fade"
                and c.get("stem") == "other" and c["song"] == "A")
    assert duck["end_gain"] < 1.0
    # Tempo ramp still waits for the LAST stem (the late backing swap).
    window = plan[0]
    spb_b = (60.0 / b.bpm) * b.time_signature
    total = max(c["start_bar"] + c["duration_bars"] for c in plan
                if c["tool"] == "crossfade_stem")
    for c in plan:
        if c["tool"] == "set_tempo_ramp":
            assert c["start_time"] >= window["to_song_time_start"] + total * spb_b - 1e-6


def test_acapella_extras_keep_the_riding_vocal_clean():
    """reverb_tail on acapella_out lands at the vocal HANDOVER, not the
    seam; sweep/bass extras are dropped entirely."""
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    base = default_decision(cands, style=TransitionStyle.acapella_out)
    decision = base.model_copy(update={"extras": [
        TransitionExtra.reverb_tail, TransitionExtra.filter_sweep_out,
    ]})
    plan = expand(decision, a, b, cands, a_safe_regions=FULL_SAFE)
    validate_plan(plan)
    assert not any(c["tool"] == "filter_sweep" for c in plan)  # dropped
    reverbs = [c for c in plan if c["tool"] == "apply_reverb"]
    assert len(reverbs) == 1
    vocals = next(c for c in plan if c["tool"] == "crossfade_stem" and c["stem"] == "vocals")
    seam_a = plan[0]["from_song_time_start"]
    spb_a = (60.0 / a.bpm) * a.time_signature
    handover_time = seam_a + vocals["start_bar"] * spb_a
    assert abs(reverbs[0]["start_time"] - handover_time) < 1e-6
    assert reverbs[0]["wet_level"] <= 0.5  # light wash, not a drowning


def test_acapella_in_echo_fires_at_vocal_exit_bar_zero():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    base = default_decision(cands, style=TransitionStyle.acapella_in)
    decision = base.model_copy(update={"extras": [TransitionExtra.echo_tail]})
    plan = expand(decision, a, b, cands, a_safe_regions=FULL_SAFE)
    validate_plan(plan)
    echo = next(c for c in plan if c["tool"] == "echo_out")
    assert abs(echo["start_time"] - plan[0]["from_song_time_start"]) < 1e-6


def test_loudness_match_boosts_quiet_b_and_recovers():
    """A loud, B quiet: B gets a (capped) boost at the seam and the exact
    reciprocal recovery ramp after the crossfade — net unity, so the next
    stitch junction can't jump."""
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.smooth_blend)
    loud_a = [0.5] * int(a.duration)
    quiet_b = [0.1] * int(b.duration)   # 14 dB apart -> capped to +4 dB
    plan = expand(decision, a, b, cands,
                  a_energy_curve=loud_a, b_energy_curve=quiet_b)
    validate_plan(plan)
    fades = [c for c in plan if c["tool"] == "volume_fade"
             and c["song"] == "B" and "stem" not in c]
    assert len(fades) == 2
    match, recover = fades
    assert abs(match["end_gain"] - 10 ** (4.0 / 20.0)) < 1e-3  # +4 dB cap
    assert abs(match["end_gain"] * recover["end_gain"] - 1.0) < 1e-3
    # Recovery starts strictly after the crossfade finishes.
    spb_b = (60.0 / b.bpm) * b.time_signature
    total = max(c["start_bar"] + c["duration_bars"] for c in plan
                if c["tool"] == "crossfade_stem")
    assert recover["start_time"] >= plan[0]["to_song_time_start"] + total * spb_b - 1e-6


def test_loudness_match_cuts_loud_b():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.smooth_blend)
    plan = expand(decision, a, b, cands,
                  a_energy_curve=[0.1] * 240, b_energy_curve=[0.5] * 240)
    fades = [c for c in plan if c["tool"] == "volume_fade"
             and c["song"] == "B" and "stem" not in c]
    assert fades and fades[0]["end_gain"] < 1.0
    assert fades[0]["end_gain"] >= 10 ** (-6.0 / 20.0) - 1e-3  # -6 dB cap


def test_loudness_match_skips_balanced_and_missing():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.smooth_blend)
    balanced = expand(decision, a, b, cands,
                      a_energy_curve=[0.3] * 240, b_energy_curve=[0.3] * 240)
    missing = expand(decision, a, b, cands)  # no curves at all
    for plan in (balanced, missing):
        assert not any(c["tool"] == "volume_fade" and "stem" not in c
                       for c in plan)


def test_loudness_match_can_be_disabled():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.smooth_blend)
    plan = expand(decision, a, b, cands,
                  a_energy_curve=[0.5] * 240, b_energy_curve=[0.1] * 240,
                  loudness_match=False)
    assert not any(c["tool"] == "volume_fade" and "stem" not in c
                   for c in plan)


def test_non_acapella_extras_unchanged():
    """The generic extras behavior for the original styles is untouched."""
    a, b = make_bundle(), make_bundle(bpm=124.0)
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    base = default_decision(cands, style=TransitionStyle.smooth_blend)
    decision = base.model_copy(update={"extras": [TransitionExtra.filter_sweep_out]})
    plan = expand(decision, a, b, cands)
    assert any(c["tool"] == "filter_sweep" for c in plan)


def test_acapella_in_unsafe_out_lengthens_vocal_fade():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        [], FULL_SAFE,  # A has no vocal data -> out candidates unsafe
    )
    decision = default_decision(cands, style=TransitionStyle.acapella_in)
    plan = expand(decision, a, b, cands)
    vocals = next(c for c in plan if c["tool"] == "crossfade_stem" and c["stem"] == "vocals")
    assert vocals["duration_bars"] == 4  # masked exit


def test_planner_downgrades_unpinned_acapella_on_key_clash():
    a = SongMeta("Track A", "X", make_bundle(key="Am", camelot="8A"),
                 energy_curve_for(240.0), FULL_SAFE)
    b = SongMeta("Track B", "Y", make_bundle(key="C#m", camelot="12A"),
                 energy_curve_for(240.0), FULL_SAFE)  # beyond pitch cap
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "acapella_out", "duration_bars": 12,
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b, pitch_mode="whole_song"))
    assert outcome.style == "smooth_blend"  # downgraded, not honored
    validate_plan(outcome.plan)


def test_planner_allows_acapella_on_camelot_compatible_keys():
    """Keys don't have to be identical for acapella styles — any standard
    Camelot match (relative major/minor, ±1 on the wheel) must survive the
    downgrade guard."""
    for b_key, b_camelot in (("Em", "9A"), ("C", "8B")):  # neighbour, relative
        a = SongMeta("Track A", "X", make_bundle(key="Am", camelot="8A"),
                     energy_curve_for(240.0), FULL_SAFE)
        b = SongMeta("Track B", "Y", make_bundle(key=b_key, camelot=b_camelot),
                     energy_curve_for(240.0), FULL_SAFE)
        provider = StubProvider(response={
            "out": "A1", "in": "B1", "style": "acapella_out", "duration_bars": 12,
        })
        outcome = asyncio.run(build_plan_v2(provider, a, b))
        assert outcome.style == "acapella_out", (
            f"acapella_out downgraded on Camelot-compatible pair 8A/{b_camelot}"
        )
        validate_plan(outcome.plan)


def test_planner_honors_pinned_acapella_despite_clash():
    a = SongMeta("Track A", "X", make_bundle(key="Am", camelot="8A"),
                 energy_curve_for(240.0), FULL_SAFE)
    b = SongMeta("Track B", "Y", make_bundle(key="C#m", camelot="12A"),
                 energy_curve_for(240.0), FULL_SAFE)
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    outcome = asyncio.run(build_plan_v2(
        provider, a, b, style_override="acapella_in", pitch_mode="whole_song",
    ))
    assert outcome.style == "acapella_in"  # the user's call
    validate_plan(outcome.plan)


def test_expand_rejects_unknown_candidate():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    bad = TransitionDecision.model_validate({
        "out": "A99", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    with pytest.raises(ArchetypeError):
        expand(bad, a, b, cands)


# ---------------------------------------------------------------- validation

def _minimal_plan(seam_a=100.0, seam_b=20.0, bars=8):
    plan = [{
        "tool": "set_transition_window",
        "from_song_time_start": seam_a,
        "to_song_time_start": seam_b,
        "duration_bars": bars,
    }]
    for stem in ("vocals", "drums", "bass", "other"):
        plan.append({
            "tool": "crossfade_stem", "stem": stem,
            "from_song": "A", "to_song": "B",
            "start_bar": 0, "duration_bars": bars, "curve": "equal_power",
        })
    return plan


def test_repair_normalizes_song_refs():
    a, b = make_bundle(), make_bundle()
    plan = _minimal_plan()
    for c in plan[1:]:
        c["from_song"], c["to_song"] = "Song A", "song_b"
    repaired = repair_plan(plan, a, b)
    validate_plan(repaired)
    for c in repaired:
        if c["tool"] == "crossfade_stem":
            assert (c["from_song"], c["to_song"]) == ("A", "B")


def test_repair_converts_permanent_pitch_shift():
    a, b = make_bundle(), make_bundle()
    plan = _minimal_plan() + [
        {"tool": "pitch_shift", "song": "B", "semitones": -5}
    ]
    repaired = repair_plan(plan, a, b)
    validate_plan(repaired)
    tps = [c for c in repaired if c["tool"] == "temporary_pitch_shift"]
    assert len(tps) == 1 and tps[0]["semitones"] == -2  # capped
    assert not any(c["tool"] == "pitch_shift" for c in repaired)


def test_repair_clamps_late_seam_and_fills_missing_stems():
    a, b = make_bundle(duration=200.0), make_bundle()
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 199.0,  # way past headroom
         "to_song_time_start": 20.0, "duration_bars": 8},
        {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A",
         "to_song": "B", "start_bar": 0, "duration_bars": 8,
         "curve": "equal_power"},
    ]
    repaired = repair_plan(plan, a, b)
    validate_plan(repaired)
    ceiling = max_seam_time(a.duration, a.bpm, a.time_signature)
    assert repaired[0]["from_song_time_start"] <= ceiling
    stems = {c["stem"] for c in repaired if c["tool"] == "crossfade_stem"}
    assert stems == {"vocals", "drums", "bass", "other"}


def test_repair_drops_unknown_tools():
    a, b = make_bundle(), make_bundle()
    plan = _minimal_plan() + [{"tool": "explode_speakers", "song": "A"}]
    repaired = repair_plan(plan, a, b)
    validate_plan(repaired)


def test_validate_rejects_planless_garbage():
    with pytest.raises(ValueError):
        validate_plan([{"tool": "crossfade_stem", "stem": "vocals",
                        "from_song": "A", "to_song": "B",
                        "start_bar": 0, "duration_bars": 8,
                        "curve": "equal_power"}])


def test_enforce_revert_defers_early_ramp():
    b = make_bundle(bpm=120.0)
    plan = _minimal_plan(seam_b=20.0, bars=8) + [{
        "tool": "set_tempo_ramp", "song": "B",
        "start_time": 21.0, "end_time": 40.0,
        "start_bpm": 126.0, "end_bpm": 120.0,
    }]
    out = enforce_revert_after_crossfade(plan, b)
    ramp = next(c for c in out if c["tool"] == "set_tempo_ramp")
    spb_b = (60.0 / b.bpm) * b.time_signature
    assert ramp["start_time"] >= 20.0 + 8 * spb_b - 1e-6


# ---------------------------------------------------------------- planner v2

class StubProvider:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    async def complete_json(self, *, system, user, nonce=0, cache_namespace="x"):
        self.calls.append({"system": system, "user": user, "nonce": nonce})
        if self.exc:
            raise self.exc
        return self.response


def _metas():
    a = make_bundle(bpm=126.0)
    b = make_bundle(bpm=120.0)
    return (
        SongMeta("Levels", "Avicii", a, energy_curve_for(a.duration), FULL_SAFE),
        SongMeta("One More Time", "Daft Punk", b,
                 energy_curve_for(b.duration), FULL_SAFE),
    )


def test_planner_v2_happy_path_uses_llm_decision():
    a, b = _metas()
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend",
        "duration_bars": 16, "a_fade_out_bars": 8,
        "rationale": "both four-on-the-floor at similar energy",
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.source == "llm_v2"
    assert outcome.style == "smooth_blend"
    validate_plan(outcome.plan)
    # Identity made it into the prompt.
    assert "Avicii" in provider.calls[0]["user"]
    assert "Daft Punk" in provider.calls[0]["user"]


def test_planner_v2_repairs_near_miss_decision():
    a, b = _metas()
    provider = StubProvider(response={
        "style": "drum_bridge", "out": "A1",   # missing "in", bad shape
        "duration_bars": "lots",
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.source == "llm_v2_repaired"
    assert outcome.style == "drum_bridge"
    validate_plan(outcome.plan)


def test_planner_v2_falls_back_when_llm_dies():
    a, b = _metas()
    provider = StubProvider(exc=RuntimeError("api down"))
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.source == "style_default"
    validate_plan(outcome.plan)


def test_planner_v2_pinned_style_wins():
    a, b = _metas()
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    outcome = asyncio.run(
        build_plan_v2(provider, a, b, style_override="vinyl_stop")
    )
    assert outcome.style == "vinyl_stop"
    assert any(c["tool"] == "turntable_stop" for c in outcome.plan)
    validate_plan(outcome.plan)


def test_planner_v2_passes_nonce_and_context():
    a, b = _metas()
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    asyncio.run(build_plan_v2(
        provider, a, b, style_hint="wash_out",
        previous_styles=["drop_swap"], nonce=3,
    ))
    call = provider.calls[0]
    assert call["nonce"] == 3
    assert "wash_out" in call["user"]
    assert "drop_swap" in call["user"]


def test_planner_v2_passes_set_position():
    a, b = _metas()
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    asyncio.run(build_plan_v2(
        provider, a, b, pair_label="transition 2 of 4",
    ))
    assert "transition 2 of 4" in provider.calls[0]["user"]


# ------------------------------------------- candidate densities / LLM input

def _full_envelopes(duration: float, level: float = 0.15) -> dict:
    n = int(duration * 10)
    stem = {"rms": [level] * n, "peak": [level * 1.5] * n}
    return {"frame_hz": 10, "drums": dict(stem), "bass": dict(stem),
            "other": dict(stem), "vocals": dict(stem)}


def _wordy_segments(duration: float) -> list[dict]:
    words = [
        {"word": f"w{i}", "start": float(i), "end": i + 0.5, "probability": 0.9}
        for i in range(int(duration))
    ]
    return [{"start": 0.0, "end": duration, "text": "la", "words": words}]


def test_unknown_densities_omitted_from_llm_dict():
    """No safety data and no envelopes → density fields are absent, not 0.0."""
    b = make_bundle()
    cands = build_in_candidates(b, energy_curve_for(b.duration), [])
    assert cands
    for c in cands:
        d = c.to_llm_dict()
        assert "vocal_density" not in d
        assert "drum_density" not in d
        assert "bass_density" not in d


def test_in_candidates_carry_densities_and_lyrics():
    """B-side candidates get drum/bass densities and lyric previews when
    the bundle carries envelopes + transcription (regression: they used
    to be hardcoded 0.0/None on the in side)."""
    import dataclasses

    b = dataclasses.replace(
        make_bundle(),
        envelopes=_full_envelopes(240.0),
        transcription_segments=_wordy_segments(240.0),
    )
    cands = build_in_candidates(b, energy_curve_for(b.duration), FULL_SAFE)
    assert cands
    for c in cands:
        assert c.drum_density == pytest.approx(1.0)
        assert c.bass_density == pytest.approx(1.0)
        assert c.lyrics_preview  # words exist throughout the song
        d = c.to_llm_dict()
        assert d["drum_density"] == 1.0
        assert d["bass_density"] == 1.0


def test_song_llm_input_includes_tags():
    import dataclasses

    from app.services.mixer.planner_v2 import _song_llm_input

    tags = {"genres": ["house"], "moods": ["energetic"]}
    bundle = dataclasses.replace(make_bundle(), tags=tags)
    meta = SongMeta("T", "A", bundle, energy_curve_for(240.0), FULL_SAFE)
    assert _song_llm_input(meta, [])["tags"] == tags

    untagged = SongMeta("T", "A", make_bundle(), energy_curve_for(240.0), FULL_SAFE)
    assert "tags" not in _song_llm_input(untagged, [])


# ---------------------------------------------------------------- bass swap

def _bass_call(plan: list[dict]) -> dict:
    return next(
        c for c in plan
        if c["tool"] == "crossfade_stem" and c["stem"] == "bass"
    )


def test_bass_swap_on_long_blend():
    """smooth_blend >= 8 bars: the bass stem hands over in a 2-bar window
    at the crossfade midpoint instead of blending the whole way."""
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.smooth_blend)
    duration = decision.normalized_duration()
    plan = expand(decision, a, b, cands)
    bass = _bass_call(plan)
    assert bass["duration_bars"] == 2
    assert bass["start_bar"] == min(duration // 2, decision.normalized_a_fade(duration)) - 1
    assert bass["a_fade_out_bars"] == 1
    # The other full-window stems are untouched.
    vocals = next(c for c in plan
                  if c["tool"] == "crossfade_stem" and c["stem"] == "vocals")
    assert vocals["start_bar"] == 0
    assert vocals["duration_bars"] == duration


def test_bass_swap_skipped_for_short_styles():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.drop_swap)
    plan = expand(decision, a, b, cands)
    bass = _bass_call(plan)
    assert bass["start_bar"] == 0
    assert bass["duration_bars"] == decision.normalized_duration()


def test_bass_swap_disabled_restores_coupled_fade():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.smooth_blend)
    plan = expand(decision, a, b, cands, bass_swap=False)
    bass = _bass_call(plan)
    assert bass["start_bar"] == 0
    assert bass["duration_bars"] == decision.normalized_duration()


def test_bass_swap_never_outlives_a_fade():
    """When the decision cuts A out early, A's bass must not ride alone
    past the rest of A."""
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    base = default_decision(cands, style=TransitionStyle.smooth_blend)
    decision = base.model_copy(update={"duration_bars": 16, "a_fade_out_bars": 4})
    plan = expand(decision, a, b, cands)
    bass = _bass_call(plan)
    # Handover is pulled forward to A's exit (bar 4), not the midpoint (8).
    assert bass["start_bar"] + bass["a_fade_out_bars"] <= 4


# ----------------------------------------------------------- new archetypes

def test_halftime_ratio_detection():
    from app.services.mixer.archetypes import halftime_ratio
    assert halftime_ratio(85.0, 170.0) == 2.0     # B double-time
    assert halftime_ratio(85.0, 172.0) == 2.0     # within 4%
    assert halftime_ratio(170.0, 85.0) == 0.5     # B half-time
    assert halftime_ratio(120.0, 128.0) == 1.0    # ordinary gap
    assert halftime_ratio(85.0, 120.0) == 1.0     # nowhere near 2:1
    assert halftime_ratio(None, 120.0) == 1.0


def test_expand_stamps_tempo_ratio_for_halftime_pair():
    a = make_bundle(bpm=85.0)
    b = make_bundle(bpm=170.0)
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.drum_bridge)
    plan = expand(decision, a, b, cands)
    validate_plan(plan)
    window = plan[0]
    assert window["tempo_ratio"] == 2.0
    # 85 * 2 == 170 exactly -> B needs no post-crossfade tempo ramp at all.
    assert not any(c["tool"] == "set_tempo_ramp" for c in plan)


def test_expand_no_ratio_for_ordinary_gap():
    a, b = make_bundle(bpm=120.0), make_bundle(bpm=126.0)
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    plan = expand(default_decision(cands), a, b, cands)
    assert "tempo_ratio" not in plan[0]


def test_double_drop_expansion_gives_b_the_bass():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.double_drop)
    plan = expand(decision, a, b, cands)
    validate_plan(plan)
    bass = next(c for c in plan
                if c["tool"] == "crossfade_stem" and c["stem"] == "bass")
    assert bass["start_bar"] == 0
    assert bass["duration_bars"] == 1
    assert bass["a_fade_out_bars"] == 0   # A's bass never stacks on B's drop


def test_backspin_expansion_emits_tool_at_seam():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.backspin)
    plan = expand(decision, a, b, cands)
    validate_plan(plan)
    spin = next(c for c in plan if c["tool"] == "backspin")
    assert spin["song"] == "A"
    assert spin["start_time"] == plan[0]["from_song_time_start"]
    assert spin["duration_beats"] == 4.0


def test_breakdown_blend_is_long_and_bass_swapped():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.breakdown_blend)
    plan = expand(decision, a, b, cands)
    validate_plan(plan)
    assert plan[0]["duration_bars"] == 16
    bass = next(c for c in plan
                if c["tool"] == "crossfade_stem" and c["stem"] == "bass")
    assert bass["duration_bars"] == 2   # midpoint EQ swap


def test_out_candidates_offer_drop_and_breakdown():
    a = make_bundle()  # 5 sections of 48 s each
    # Section energies: quiet intro, a real drop in section 2 (before the
    # late-3 window), then a fading tail below the breakdown threshold.
    curve = [0.2] * 48 + [1.0] * 48 + [0.6] * 48 + [0.3] * 48 + [0.15] * 48
    cands = build_out_candidates(a, curve, FULL_SAFE)
    descs = " | ".join(c.description for c in cands)
    assert "last drop/chorus" in descs
    assert "final breakdown" in descs
    drop = next(c for c in cands if "drop/chorus" in c.description)
    assert drop.energy >= 0.8


def test_planner_downgrades_double_drop_on_key_clash():
    a = SongMeta("A", "X", make_bundle(key="Am", camelot="8A"),
                 energy_curve_for(240.0), FULL_SAFE)
    b = SongMeta("B", "Y", make_bundle(key="C#m", camelot="12A"),
                 energy_curve_for(240.0), FULL_SAFE)
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "double_drop", "duration_bars": 8,
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.style == "drop_swap"
    validate_plan(outcome.plan)


def test_planner_allows_double_drop_when_both_seams_are_drops():
    a_bundle, b_bundle = make_bundle(), make_bundle()
    a_curve = energy_curve_for(240.0, peak_at=0.8)   # A peaks late
    b_curve = energy_curve_for(240.0, peak_at=0.2)   # B peaks early
    cands = build_pair_candidates(
        a_bundle, b_bundle, a_curve, b_curve, FULL_SAFE, FULL_SAFE
    )
    out_id = max(cands.out_candidates, key=lambda c: c.energy).id
    in_id = max(cands.in_candidates, key=lambda c: c.energy).id
    a = SongMeta("A", "X", a_bundle, a_curve, FULL_SAFE)
    b = SongMeta("B", "Y", b_bundle, b_curve, FULL_SAFE)
    provider = StubProvider(response={
        "out": out_id, "in": in_id, "style": "double_drop", "duration_bars": 8,
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.style == "double_drop"
    validate_plan(outcome.plan)


def test_planner_downgrades_vinyl_stop_on_mixable_pair():
    """vinyl_stop on a pair that could perfectly well blend (tight tempo,
    compatible keys) is showing off — downgraded to drop_swap."""
    a = SongMeta("A", "X", make_bundle(bpm=124.0, key="C", camelot="8B"),
                 energy_curve_for(240.0), FULL_SAFE)
    b = SongMeta("B", "Y", make_bundle(bpm=126.0, key="G", camelot="9B"),
                 energy_curve_for(240.0), FULL_SAFE)
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "vinyl_stop", "duration_bars": 4,
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.style == "drop_swap"
    validate_plan(outcome.plan)


def test_planner_allows_vinyl_stop_when_justified():
    """A 25% tempo gap with no half-time relationship AND a beyond-cap key
    clash: nothing can blend this — the full stop is the honest move."""
    a = SongMeta("A", "X", make_bundle(bpm=100.0, key="Am", camelot="8A"),
                 energy_curve_for(240.0), FULL_SAFE)
    b = SongMeta("B", "Y", make_bundle(bpm=125.0, key="C#m", camelot="12A"),
                 energy_curve_for(240.0), FULL_SAFE)
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "vinyl_stop", "duration_bars": 4,
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.style == "vinyl_stop"
    validate_plan(outcome.plan)


def test_planner_blocks_second_theatric_per_set():
    """Even a justified vinyl_stop is downgraded when the set already
    used a vinyl_stop or backspin — once per set, mechanically."""
    a = SongMeta("A", "X", make_bundle(bpm=100.0, key="Am", camelot="8A"),
                 energy_curve_for(240.0), FULL_SAFE)
    b = SongMeta("B", "Y", make_bundle(bpm=125.0, key="C#m", camelot="12A"),
                 energy_curve_for(240.0), FULL_SAFE)
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "vinyl_stop", "duration_bars": 4,
    })
    outcome = asyncio.run(build_plan_v2(
        provider, a, b, previous_styles=["smooth_blend", "backspin"],
    ))
    assert outcome.style == "drop_swap"
    validate_plan(outcome.plan)


def test_planner_honors_pinned_vinyl_stop():
    a = SongMeta("A", "X", make_bundle(bpm=124.0), energy_curve_for(240.0),
                 FULL_SAFE)
    b = SongMeta("B", "Y", make_bundle(bpm=126.0), energy_curve_for(240.0),
                 FULL_SAFE)
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    outcome = asyncio.run(build_plan_v2(
        provider, a, b, style_override="vinyl_stop",
    ))
    assert outcome.style == "vinyl_stop"   # the user's call, always
    validate_plan(outcome.plan)
