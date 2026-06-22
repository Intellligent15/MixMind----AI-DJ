"""Archetype expander: TransitionDecision → exact MixPlanJSON.

This is the deterministic half of planner v2. The LLM chooses *what*
(candidates, style, a couple of knobs); this module computes *how* —
every timestamp, every tempo-ramp boundary, every pitch decision — using
the same math as the battle-tested deterministic planner. The output
satisfies the executor's invariants by construction:

  * exactly one set_transition_window, first;
  * exactly four crossfade_stem calls covering vocals/drums/bass/other;
  * seams at downbeats within the headroom budget;
  * tempo ramp / temporary pitch return strictly after the crossfade;
  * never a permanent pitch_shift.

Pure module: no I/O, no DB, no settings.
"""

from __future__ import annotations

import logging

from app.services.mixer.candidates import (
    PairCandidates,
    SeamCandidate,
    max_seam_time,
)
from app.services.mixer.decision import (
    TransitionDecision,
    TransitionExtra,
    TransitionStyle,
)
from app.services.mixer.plan import compute_pitch_shift
from app.services.mixer.types import AnalysisBundle, MixPlanJSON

logger = logging.getLogger(__name__)

STEMS = ("vocals", "drums", "bass", "other")

# Tempo settle window after the crossfade. Doubled from v1's 16: a
# longer ramp halves the per-bar tempo drift (e.g. a 5%% BPM gap becomes
# ~0.16%%/bar — imperceptible). Tempo glides, unlike pitch glides, don't
# change key, so longer is strictly gentler. The ramp end is clamped to
# B's seam-headroom ceiling so it always settles before any plausible
# next-transition seam.
TEMPO_RAMP_BARS = 32
PITCH_RETURN_BARS = 4
# Pitch shifts past ±2 semitones trade a key match for pyrubberband
# artifacts that sound worse than the clash. Hard cap.
PITCH_SHIFT_CAP = 2
# Bars of A's bass killed early for the bass_kill extra.
BASS_KILL_BARS = 4
# stutter_buildup loop parameters: half-beat slices repeated 8 times = 1
# bar of stutter at A's tempo, right before the seam.
STUTTER_BEAT_FRACTION = 0.5
STUTTER_REPEATS = 8
# vinyl_stop brake length.
VINYL_STOP_BARS = 1.5
# Acapella styles: how fast the NON-held stems swap, and the vocal
# handover crossfade length. The held layer (A's vocals in acapella_out,
# A's instrumental in acapella_in) rides until the handover bar.
ACAPELLA_SWAP_BARS = 4
ACAPELLA_HANDOVER_XFADE_BARS = 2
# When no vocal-safe handover bar exists, a longer fade masks the chop.
ACAPELLA_MASK_XFADE_BARS = 4
# Staged backing entry: basses never overlap (mud), so the bass stem
# quick-swaps in ~1 bar a couple of bars into the window while drums and
# melodics crossfade normally.
ACAPELLA_BASS_SWAP_BAR = 2
ACAPELLA_BASS_SWAP_DUR = 1
# While a vocal rides over the OTHER song's backing, that backing's
# melodic stem ("other") is ducked so two melodies never fight the
# vocal; restored once the vocal hands over.
ACAPELLA_DUCK_GAIN = 0.65
ACAPELLA_DUCK_RESTORE_BARS = 2
# Seam loudness matching: align B's perceived level to A's at the seam
# (no energy pothole / spike), then glide back to B's native level after
# the crossfade. Boosts are capped tighter than cuts — the output
# limiter absorbs modest peaks but pumping is ugly.
LOUDNESS_MATCH_MIN_DB = 1.25   # smaller gaps aren't worth touching
LOUDNESS_MATCH_MAX_BOOST_DB = 4.0
LOUDNESS_MATCH_MAX_CUT_DB = 6.0
LOUDNESS_RECOVERY_BARS = 24
LOUDNESS_WINDOW_BARS = 8       # measured over this much audio each side


class ArchetypeError(ValueError):
    """Raised when a decision can't be expanded (bad candidate id, no
    overlap room, …). Callers treat it like an invalid LLM plan."""


def _sec_per_bar(bundle: AnalysisBundle) -> float:
    if not bundle.bpm:
        raise ArchetypeError("song has no bpm")
    return (60.0 / bundle.bpm) * bundle.time_signature


def camelot_compatible(a_camelot: str | None, b_camelot: str | None) -> bool:
    """Equal or adjacent on the Camelot wheel (same number other letter,
    or ±1 number same letter) → harmonically mixable, no pitch shift.

    The Camelot wheel is the DJ shorthand for key compatibility: '8A' is
    A minor, '8B' is C major; neighbours share most of their notes, so
    blending them sounds consonant. Unknown keys → assume compatible
    (a wrong no-op beats a wrong shift).
    """
    if not a_camelot or not b_camelot:
        return True
    try:
        a_num, a_letter = int(a_camelot[:-1]), a_camelot[-1].upper()
        b_num, b_letter = int(b_camelot[:-1]), b_camelot[-1].upper()
    except (ValueError, IndexError):
        return True
    if a_num == b_num:
        return True  # same or relative major/minor
    if a_letter == b_letter and (a_num - b_num) % 12 in (1, 11):
        return True
    return False


def _clamp_duration_bars(
    duration_bars: int,
    a: AnalysisBundle,
    b: AnalysisBundle,
    seam_a: float,
    seam_b: float,
) -> int:
    """Shrink the crossfade if either song lacks the room — same clamp the
    deterministic planner applies, so the executor never has to."""
    spb_a = _sec_per_bar(a)
    stretch = b.bpm / a.bpm if a.bpm and b.bpm else 1.0
    available_a = a.duration - seam_a
    available_b_stretched = (b.duration - seam_b) * stretch
    clamped = min(
        duration_bars,
        int(available_a / spb_a),
        int(available_b_stretched / spb_a),
    )
    if clamped < 2:
        raise ArchetypeError(
            f"no overlap room at chosen seams (a={available_a:.1f}s, "
            f"b_stretched={available_b_stretched:.1f}s)"
        )
    return clamped


def _crossfade_calls(
    duration_bars: int,
    a_fade_out_bars: int,
    *,
    drums_start_bar: int = 0,
    drums_duration_bars: int | None = None,
) -> list[dict]:
    """The four stem crossfades. `drums_*` lets drum_bridge offset the
    drum stem; everything else shares the main window."""
    calls = []
    for stem in STEMS:
        if stem == "drums" and (drums_start_bar or drums_duration_bars):
            calls.append({
                "tool": "crossfade_stem", "stem": stem,
                "from_song": "A", "to_song": "B",
                "start_bar": drums_start_bar,
                "duration_bars": drums_duration_bars or duration_bars,
                "curve": "equal_power",
            })
            continue
        call = {
            "tool": "crossfade_stem", "stem": stem,
            "from_song": "A", "to_song": "B",
            "start_bar": 0, "duration_bars": duration_bars,
            "curve": "equal_power",
        }
        if a_fade_out_bars < duration_bars:
            call["a_fade_out_bars"] = a_fade_out_bars
        calls.append(call)
    return calls


def _tempo_and_pitch_calls(
    a: AnalysisBundle,
    b: AnalysisBundle,
    seam_b: float,
    crossfade_total_bars: int,
    pitch_mode: str = "temporary",
) -> list[dict]:
    """Tempo ramp + temporary pitch return, both anchored strictly AFTER
    the bar where the last stem finishes its fade. Computed, not prompted."""
    calls: list[dict] = []
    spb_b = _sec_per_bar(b)
    crossfade_end_b = seam_b + crossfade_total_bars * spb_b

    needs_ramp = a.bpm and b.bpm and abs(a.bpm - b.bpm) / a.bpm > 0.02
    # Settle before B's own seam-headroom ceiling (a conservative proxy
    # for "before the NEXT transition could start"), shrinking the ramp
    # if the song is short rather than dropping it entirely.
    ceiling_b = max_seam_time(b.duration, b.bpm, b.time_signature) or b.duration
    ramp_bars = TEMPO_RAMP_BARS
    while ramp_bars >= 8 and crossfade_end_b + ramp_bars * spb_b > ceiling_b:
        ramp_bars //= 2
    ramp_end_b = crossfade_end_b + ramp_bars * spb_b
    if needs_ramp and b.duration >= ramp_end_b and ramp_end_b <= ceiling_b:
        calls.append({
            "tool": "set_tempo_ramp", "song": "B",
            "start_time": round(crossfade_end_b, 3),
            "end_time": round(ramp_end_b, 3),
            "start_bpm": a.bpm, "end_bpm": b.bpm,
        })
    elif needs_ramp:
        logger.info(
            "archetypes: B too short for a tempo ramp (%.1fs < %.1fs); "
            "B stays at A's tempo", b.duration, ramp_end_b,
        )

    if pitch_mode != "temporary":
        # "whole_song": stems arrive pre-shifted and a/b carry EFFECTIVE
        # keys, so a remaining clash means the resolver accepted it — the
        # style choice masks it. "off": never shift. Either way, no
        # in-plan pitch tools (no glide, ever).
        return calls
    if not camelot_compatible(a.camelot_key, b.camelot_key):
        try:
            n_steps = compute_pitch_shift(a.key, b.key)
        except ValueError:
            # Unparseable key string — skip pitch handling rather than
            # killing the whole expansion (a wrong no-op beats no plan).
            n_steps = 0
        n_steps = max(-PITCH_SHIFT_CAP, min(PITCH_SHIFT_CAP, n_steps))
        return_end_b = crossfade_end_b + PITCH_RETURN_BARS * spb_b
        if n_steps != 0 and b.duration >= return_end_b:
            calls.append({
                "tool": "temporary_pitch_shift", "song": "B",
                "start_time": round(crossfade_end_b, 3),
                "semitones": n_steps,
                "fade_in_bars": 0, "hold_bars": 0,
                "fade_out_bars": PITCH_RETURN_BARS,
            })
    return calls


def _bar_is_vocal_safe(
    t: float, lookahead: float, safe_regions: list[dict] | None
) -> bool:
    """True when [t, t+lookahead] sits inside one no-vocal span. Unknown
    (no data) -> False, mirroring candidates._is_vocal_safe."""
    if not safe_regions:
        return False
    for r in safe_regions:
        if "safe" in r and not r.get("safe"):
            continue
        if r["start"] <= t and (t + lookahead) <= r["end"]:
            return True
    return False


def _find_safe_handover_bar(
    seam_a: float,
    spb_a: float,
    lo_bar: int,
    hi_bar: int,
    a_safe_regions: list[dict] | None,
) -> int | None:
    """Latest bar in [lo_bar, hi_bar] where A's vocals are silent (so
    fading them out there can't chop a word). Bars are on A's grid since
    B is stretched to A's tempo across the window."""
    for bar in range(hi_bar, lo_bar - 1, -1):
        t = seam_a + bar * spb_a
        if _bar_is_vocal_safe(t, spb_a, a_safe_regions):
            return bar
    return None


def _expand_acapella(
    decision: TransitionDecision,
    a: AnalysisBundle,
    b: AnalysisBundle,
    seam_a: float,
    seam_b: float,
    duration: int,
    out_vocal_safe: bool,
    a_safe_regions: list[dict] | None,
) -> tuple[list[dict], list[dict], int]:
    """Stem calls + ducking automation for the two acapella shapes.
    Returns (stem_calls, duck_calls, crossfade_total_bars).

    The backing track never "switches randomly" — it enters STAGED, the
    way a DJ brings an instrumental in under a vocal:

      * DRUMS crossfade across the swap window (groove morphs smoothly);
      * BASS quick-swaps in ~1 bar a couple of bars in — two basslines
        overlapping is instant mud, so basses hand over, never blend;
      * OTHER (melodics) crossfades with the drums but arrives DUCKED
        under the riding vocal, restoring to full once the vocal hands
        over — melody-vs-vocal fights are what make layering sound
        amateur.

    acapella_out — B's staged backing under A's riding vocal:
        drums/other: [0 .. swap]    bass: [2 .. 3]
        vocals:      [handover .. +xfade]
        duck:        B's "other" at DUCK_GAIN until handover, then restore
    acapella_in — B's vocal teases over A's backing, staged swap later:
        vocals:      [0 .. xfade]
        drums/other: [hold .. +swap]    bass: [hold+2 .. +1]
        duck:        A's "other" at DUCK_GAIN under the teasing vocal
    """
    spb_a = _sec_per_bar(a)
    spb_b = _sec_per_bar(b)
    swap = min(ACAPELLA_SWAP_BARS, max(2, duration // 2))

    def _call(stem: str, start: int, dur: int) -> dict:
        return {
            "tool": "crossfade_stem", "stem": stem,
            "from_song": "A", "to_song": "B",
            "start_bar": start, "duration_bars": dur,
            "curve": "equal_power",
        }

    if decision.style == TransitionStyle.acapella_out:
        xfade = ACAPELLA_HANDOVER_XFADE_BARS
        target_hi = duration - xfade
        target_lo = max(swap, duration // 2)
        safe_bar = _find_safe_handover_bar(
            seam_a, spb_a, target_lo, target_hi, a_safe_regions
        )
        if safe_bar is None:
            xfade = min(ACAPELLA_MASK_XFADE_BARS, duration - target_lo)
            handover = duration - xfade
            logger.info(
                "archetypes: no vocal-safe handover bar in [%d, %d]; "
                "masking with a %d-bar vocal fade", target_lo, target_hi, xfade,
            )
        else:
            handover = safe_bar

        bass_start = min(ACAPELLA_BASS_SWAP_BAR, max(0, swap - 1))
        stem_calls = [
            _call("vocals", handover, xfade),
            _call("drums", 0, swap),
            _call("bass", bass_start, ACAPELLA_BASS_SWAP_DUR),
            _call("other", 0, swap),
        ]
        # B's melodics ride DUCKED under A's vocal, restored at handover.
        duck_calls = [
            {
                "tool": "volume_fade", "song": "B", "stem": "other",
                "start_time": round(seam_b, 3), "duration_bars": 1.0,
                "start_gain": 1.0, "end_gain": ACAPELLA_DUCK_GAIN,
                "bpm": b.bpm,
            },
            {
                "tool": "volume_fade", "song": "B", "stem": "other",
                "start_time": round(seam_b + handover * spb_b, 3),
                "duration_bars": float(ACAPELLA_DUCK_RESTORE_BARS),
                "start_gain": 1.0,
                "end_gain": round(1.0 / ACAPELLA_DUCK_GAIN, 4),
                "bpm": b.bpm,
            },
        ]
        total = max(handover + xfade, swap,
                    bass_start + ACAPELLA_BASS_SWAP_DUR)
        return stem_calls, duck_calls, total

    # acapella_in
    vocal_xfade = (
        ACAPELLA_HANDOVER_XFADE_BARS if out_vocal_safe
        else ACAPELLA_MASK_XFADE_BARS
    )
    hold = max(vocal_xfade, duration - swap)
    bass_start = min(hold + ACAPELLA_BASS_SWAP_BAR, duration - 1)
    stem_calls = [
        _call("vocals", 0, vocal_xfade),
        _call("drums", hold, swap),
        _call("bass", bass_start, ACAPELLA_BASS_SWAP_DUR),
        _call("other", hold, swap),
    ]
    # A's melodics duck under B's teasing vocal; no restore needed — A's
    # "other" crossfades OUT at the hold bar anyway, and the persisted
    # gain only colors a stem that's already leaving.
    duck_calls = [
        {
            "tool": "volume_fade", "song": "A", "stem": "other",
            "start_time": round(seam_a, 3), "duration_bars": 1.0,
            "start_gain": 1.0, "end_gain": ACAPELLA_DUCK_GAIN,
            "bpm": a.bpm,
        },
    ]
    total = max(hold + swap, bass_start + ACAPELLA_BASS_SWAP_DUR,
                vocal_xfade)
    return stem_calls, duck_calls, total


def _mean_level(curve: list[float] | None, lo_s: float, hi_s: float) -> float | None:
    """Mean of the 1 Hz analysis energy curve over [lo_s, hi_s]."""
    if not curve:
        return None
    lo, hi = max(0, int(lo_s)), min(len(curve), max(int(lo_s) + 1, int(hi_s)))
    window = curve[lo:hi]
    if not window:
        return None
    level = sum(window) / len(window)
    return level if level > 1e-6 else None


def _loudness_match_calls(
    a: AnalysisBundle,
    b: AnalysisBundle,
    seam_a: float,
    seam_b: float,
    crossfade_total_bars: int,
    a_energy_curve: list[float] | None,
    b_energy_curve: list[float] | None,
) -> list[dict]:
    """Song-level gain staging: bring B to A's perceived level at the
    seam, then glide back to B's native level after the crossfade.

    Two volume_fade calls on B (gains COMPOUND, and end_gain persists to
    the song's end, so the second call's reciprocal ramp is what
    restores unity — without it the next stitch junction would jump):

        1) 1.0 -> g over the first bar of the window  (match at seam)
        2) 1.0 -> 1/g over LOUDNESS_RECOVERY_BARS after the crossfade
           (net gain returns to exactly 1.0 and holds)

    Skipped when the gap is inaudible, the energy curves are missing, or
    B is too short to fit the recovery ramp before its headroom ceiling
    (a permanent re-gain would mismatch the junction — never acceptable).
    """
    import math

    spb_a = _sec_per_bar(a)
    spb_b = _sec_per_bar(b)
    a_level = _mean_level(
        a_energy_curve, seam_a - LOUDNESS_WINDOW_BARS * spb_a, seam_a
    )
    b_level = _mean_level(
        b_energy_curve, seam_b, seam_b + LOUDNESS_WINDOW_BARS * spb_b
    )
    if a_level is None or b_level is None:
        return []

    diff_db = 20.0 * math.log10(a_level / b_level)
    if abs(diff_db) < LOUDNESS_MATCH_MIN_DB:
        return []
    diff_db = max(-LOUDNESS_MATCH_MAX_CUT_DB,
                  min(LOUDNESS_MATCH_MAX_BOOST_DB, diff_db))
    gain = round(10.0 ** (diff_db / 20.0), 4)

    crossfade_end_b = seam_b + crossfade_total_bars * spb_b
    ceiling_b = max_seam_time(b.duration, b.bpm, b.time_signature) or b.duration
    recovery_bars = LOUDNESS_RECOVERY_BARS
    while recovery_bars >= 8 and crossfade_end_b + recovery_bars * spb_b > ceiling_b:
        recovery_bars //= 2
    if crossfade_end_b + recovery_bars * spb_b > ceiling_b:
        logger.info(
            "archetypes: B too short for loudness recovery; skipping match"
        )
        return []

    logger.info(
        "archetypes: loudness match — B %+.1f dB at the seam, recovering "
        "over %d bars", diff_db, recovery_bars,
    )
    return [
        {
            "tool": "volume_fade", "song": "B",
            "start_time": round(seam_b, 3),
            "duration_bars": 1.0,
            "start_gain": 1.0, "end_gain": gain, "bpm": b.bpm,
        },
        {
            "tool": "volume_fade", "song": "B",
            "start_time": round(crossfade_end_b, 3),
            "duration_bars": float(recovery_bars),
            "start_gain": 1.0, "end_gain": round(1.0 / gain, 4),
            "bpm": b.bpm,
        },
    ]


ACAPELLA_STYLES = (TransitionStyle.acapella_out, TransitionStyle.acapella_in)


def expand(
    decision: TransitionDecision,
    a: AnalysisBundle,
    b: AnalysisBundle,
    candidates: PairCandidates,
    pitch_mode: str = "temporary",
    a_safe_regions: list[dict] | None = None,
    b_safe_regions: list[dict] | None = None,
    a_energy_curve: list[float] | None = None,
    b_energy_curve: list[float] | None = None,
    loudness_match: bool = True,
) -> MixPlanJSON:
    """Expand a validated decision into the final tool-call list."""
    out_c = candidates.find(decision.out)
    in_c = candidates.find(decision.in_)
    if out_c is None or not decision.out.startswith("A"):
        raise ArchetypeError(f"unknown OUT candidate {decision.out!r}")
    if in_c is None or not decision.in_.startswith("B"):
        raise ArchetypeError(f"unknown IN candidate {decision.in_!r}")

    seam_a, seam_b = out_c.time, in_c.time
    duration = _clamp_duration_bars(
        decision.normalized_duration(), a, b, seam_a, seam_b
    )
    a_fade = decision.normalized_a_fade(duration)
    spb_a = _sec_per_bar(a)

    style = decision.style
    drums_start, drums_dur = 0, None
    pre_window_calls: list[dict] = []   # effects placed before/around the seam

    if style == TransitionStyle.drop_swap:
        # A snaps out fast; coupled short fade reads as an instant swap.
        a_fade = min(a_fade, duration)
    elif style == TransitionStyle.drum_bridge:
        # Drums hold longest: they start late on the grid but run past the
        # other stems, bridging the grooves (mirrors the classic shape).
        bridge = max(4, duration // 2)
        drums_start = min(bridge, duration - 2)
        drums_dur = duration + bridge - drums_start
        # Keep total within the clamp budget.
        total = drums_start + drums_dur
        room = _clamp_duration_bars(total, a, b, seam_a, seam_b)
        if total > room:
            drums_dur = max(2, room - drums_start)
    elif style == TransitionStyle.wash_out:
        pre_window_calls.append({
            "tool": "apply_reverb", "song": "A",
            "start_time": round(seam_a, 3),
            "tail_duration_bars": float(max(2, a_fade // 2)),
            "wet_level": 0.8, "bpm": a.bpm,
        })
        if TransitionExtra.filter_sweep_out not in decision.extras:
            decision.extras.append(TransitionExtra.filter_sweep_out)
    elif style == TransitionStyle.stutter_buildup:
        if out_c.vocal_safe:
            stutter_start = max(0.0, seam_a - spb_a)  # last bar before the seam
            pre_window_calls.append({
                "tool": "loop_section", "song": "A",
                "start_time": round(stutter_start, 3),
                "beats": STUTTER_BEAT_FRACTION,
                "repeats": STUTTER_REPEATS, "bpm": a.bpm,
            })
        else:
            logger.info(
                "archetypes: OUT point not vocal-safe; dropping stutter loop"
            )
    elif style == TransitionStyle.vinyl_stop:
        pre_window_calls.append({
            "tool": "turntable_stop", "song": "A",
            "start_time": round(seam_a, 3),
            "duration_bars": VINYL_STOP_BARS, "bpm": a.bpm,
        })
        a_fade = min(a_fade, 2)

    # Acapella styles build their stem calls early so the extras below
    # can target the VOCAL HANDOVER moment instead of the seam — under
    # these styles song A's vocals are the held star layer, and seam-wide
    # effects (a lowpass sweep, heavy reverb from bar 0) would smother
    # the very thing the style exists to showcase.
    acapella_stem_calls: list[dict] | None = None
    acapella_total_bars = 0
    vocal_exit_time = seam_a  # when A's vocal leaves (acapella_in: bar 0)
    extras_to_apply = list(decision.extras)
    if style in ACAPELLA_STYLES:
        duration = _clamp_duration_bars(duration, a, b, seam_a, seam_b)
        acapella_stem_calls, acapella_duck_calls, acapella_total_bars = (
            _expand_acapella(
                decision, a, b, seam_a, seam_b, duration,
                out_c.vocal_safe, a_safe_regions,
            )
        )
        pre_window_calls.extend(acapella_duck_calls)
        if style == TransitionStyle.acapella_out:
            vocals_call = next(
                c for c in acapella_stem_calls if c["stem"] == "vocals"
            )
            vocal_exit_time = seam_a + vocals_call["start_bar"] * spb_a
        kept: list[TransitionExtra] = []
        for extra in extras_to_apply:
            if extra in (TransitionExtra.bass_kill,
                         TransitionExtra.filter_sweep_out):
                # bass already swapped at bar 0; a sweep would muffle the
                # riding vocal. Neither has a place here.
                logger.info(
                    "archetypes: dropping extra %s (not acapella-compatible)",
                    extra.value,
                )
            elif extra == TransitionExtra.reverb_tail:
                # The classic move, placed where it belongs: a light wash
                # on A's vocal AS IT HANDS OVER, not across the ride.
                extra_calls_pre = {
                    "tool": "apply_reverb", "song": "A",
                    "start_time": round(vocal_exit_time, 3),
                    "tail_duration_bars": 2.0,
                    "wet_level": 0.45, "bpm": a.bpm,
                }
                pre_window_calls.append(extra_calls_pre)
            elif extra == TransitionExtra.echo_tail:
                pre_window_calls.append({
                    "tool": "echo_out", "song": "A",
                    "start_time": round(vocal_exit_time, 3),
                    "beats": 4, "feedback": 0.45, "bpm": a.bpm,
                })
            else:
                kept.append(extra)
        extras_to_apply = kept

    extra_calls: list[dict] = []
    for extra in extras_to_apply:
        if extra == TransitionExtra.bass_kill:
            kill_start = max(0.0, seam_a - BASS_KILL_BARS * spb_a)
            extra_calls.append({
                "tool": "volume_fade", "song": "A", "stem": "bass",
                "start_time": round(kill_start, 3),
                "duration_bars": float(BASS_KILL_BARS),
                "start_gain": 1.0, "end_gain": 0.0, "bpm": a.bpm,
            })
        elif extra == TransitionExtra.filter_sweep_out:
            sweep_end = seam_a + a_fade * spb_a
            extra_calls.append({
                "tool": "filter_sweep", "song": "A", "type": "lowpass",
                "start_time": round(seam_a, 3),
                "end_time": round(sweep_end, 3),
                "start_cutoff_hz": 20000.0, "end_cutoff_hz": 120.0,
            })
        elif extra == TransitionExtra.echo_tail:
            echo_start = seam_a + a_fade * spb_a
            extra_calls.append({
                "tool": "echo_out", "song": "A",
                "start_time": round(echo_start, 3),
                "beats": 4, "feedback": 0.5, "bpm": a.bpm,
            })
        elif extra == TransitionExtra.reverb_tail:
            if style != TransitionStyle.wash_out:  # wash_out already has one
                extra_calls.append({
                    "tool": "apply_reverb", "song": "A",
                    "start_time": round(seam_a, 3),
                    "tail_duration_bars": 4.0,
                    "wet_level": 0.6, "bpm": a.bpm,
                })

    if acapella_stem_calls is not None:
        stem_calls = acapella_stem_calls
        crossfade_total_bars = acapella_total_bars
    else:
        stem_calls = _crossfade_calls(
            duration, a_fade,
            drums_start_bar=drums_start, drums_duration_bars=drums_dur,
        )
        crossfade_total_bars = max(
            int(c["start_bar"]) + int(c["duration_bars"]) for c in stem_calls
        )

    loudness_calls: list[dict] = []
    if loudness_match:
        loudness_calls = _loudness_match_calls(
            a, b, seam_a, seam_b, crossfade_total_bars,
            a_energy_curve, b_energy_curve,
        )

    plan: MixPlanJSON = [
        {
            "tool": "set_transition_window",
            "from_song_time_start": round(seam_a, 3),
            "to_song_time_start": round(seam_b, 3),
            "duration_bars": duration,
        },
        *pre_window_calls,
        *extra_calls,
        *loudness_calls,
        *_tempo_and_pitch_calls(a, b, seam_b, crossfade_total_bars, pitch_mode),
        *stem_calls,
    ]
    return plan


def default_decision(candidates: PairCandidates, style: TransitionStyle | None = None) -> TransitionDecision:
    """A sensible decision when the LLM is unavailable but a style was
    pinned (e.g. user override): latest OUT, earliest IN, default knobs."""
    if not candidates.out_candidates or not candidates.in_candidates:
        raise ArchetypeError("no seam candidates available")
    chosen = style or TransitionStyle.smooth_blend
    # Prefer a high-energy IN for energetic styles; first candidate else.
    in_c: SeamCandidate = candidates.in_candidates[0]
    if chosen in (TransitionStyle.drop_swap, TransitionStyle.stutter_buildup,
                  TransitionStyle.acapella_in):
        for c in candidates.in_candidates:
            if c.energy >= 0.8:
                in_c = c
                break
    out_c = candidates.out_candidates[-1]
    if chosen == TransitionStyle.acapella_in:
        # A's vocal exits at the seam — prefer an exit between phrases.
        for c in reversed(candidates.out_candidates):
            if c.vocal_safe:
                out_c = c
                break
    from app.services.mixer.decision import STYLE_DURATION_CHOICES

    duration = STYLE_DURATION_CHOICES[chosen][-1]
    return TransitionDecision(
        **{
            "out": out_c.id,
            "in": in_c.id,
            "style": chosen,
            "duration_bars": duration,
            "a_fade_out_bars": max(1, duration // 2),
            "rationale": "default expansion (no LLM decision available)",
        }
    )
