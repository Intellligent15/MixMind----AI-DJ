"""Plan validation and repair — pure, unit-testable, no DB.

Philosophy change vs v1: every rejection used to throw the whole LLM
plan away and silently substitute the deterministic fallback — so a
musically interesting plan died over a fixable numeric slip, and most
"LLM transitions" were not LLM transitions at all. Now we REPAIR what
can be mechanically repaired (normalize song refs, clamp seams to the
headroom budget, convert a forbidden permanent pitch_shift into the
temporary equivalent, drop unknown tools) and only reject what is
genuinely unsalvageable (no transition window, missing stems with no
basis to synthesize them).
"""

from __future__ import annotations

import logging

from app.services.mixer.candidates import max_seam_time
from app.services.mixer.types import AnalysisBundle

logger = logging.getLogger(__name__)

LEGAL_TOOLS = {
    "set_transition_window",
    "set_tempo_ramp",
    "temporary_pitch_shift",
    "pitch_shift",
    "crossfade_stem",
    "filter_sweep",
    "echo_out",
    "loop_section",
    "swap_stem",
    "apply_reverb",
    "turntable_stop",
    "volume_fade",
}
CANONICAL_STEMS = {"vocals", "drums", "bass", "other"}
LEGAL_SONG_REFS = {"A", "B"}
SONG_FIELDS_BY_TOOL = {
    "crossfade_stem": ("from_song", "to_song"),
    "pitch_shift": ("song",),
    "temporary_pitch_shift": ("song",),
    "set_tempo_ramp": ("song",),
    "filter_sweep": ("song",),
    "echo_out": ("song",),
    "loop_section": ("song",),
    "swap_stem": ("from_song", "to_song"),
    "apply_reverb": ("song",),
    "turntable_stop": ("song",),
    "volume_fade": ("song",),
}
PERMANENT_PITCH_SHIFT_CAP = 2
PITCH_RETURN_BARS = 4


def _normalize_song_ref(value) -> str | None:
    """'Song A' / 'song_a' / 'a' / 1 → 'A'. None when unrecognizable."""
    if value in LEGAL_SONG_REFS:
        return value
    s = str(value).strip().upper()
    s = s.replace("SONG", "").replace("_", "").replace(" ", "")
    if s in LEGAL_SONG_REFS:
        return s
    if s in ("1", "FROM", "OUT", "OUTGOING"):
        return "A"
    if s in ("2", "TO", "IN", "INCOMING"):
        return "B"
    return None


def repair_plan(
    plan: list[dict],
    a: AnalysisBundle,
    b: AnalysisBundle,
) -> list[dict]:
    """Mechanically repair a free-form LLM plan in place of rejecting it.

    Returns a new plan list. Raises ValueError only when the plan is
    beyond repair (validate_plan does the final check).
    """
    if not isinstance(plan, list) or not plan:
        raise ValueError("LLM plan is not a non-empty list")

    repaired: list[dict] = []
    for call in plan:
        if not isinstance(call, dict) or "tool" not in call:
            logger.warning("repair_plan: dropping malformed call %r", call)
            continue
        tool = call["tool"]
        if tool not in LEGAL_TOOLS:
            logger.warning("repair_plan: dropping unknown tool %r", tool)
            continue
        call = dict(call)

        # Normalize song references instead of rejecting on drift.
        for field in SONG_FIELDS_BY_TOOL.get(tool, ()):
            fixed = _normalize_song_ref(call.get(field))
            if fixed is None:
                logger.warning(
                    "repair_plan: dropping %s with unrecognizable %s=%r",
                    tool, field, call.get(field),
                )
                call = None
                break
            call[field] = fixed
        if call is None:
            continue

        # Permanent pitch_shift breaks the next stitch junction (B snaps
        # back to true pitch when it's later mixed out). Convert to the
        # temporary equivalent instead of discarding the whole plan.
        if tool == "pitch_shift":
            semis = float(call.get("semitones", 0) or 0)
            semis = max(-PERMANENT_PITCH_SHIFT_CAP,
                        min(PERMANENT_PITCH_SHIFT_CAP, semis))
            if semis == 0 or call.get("song") != "B":
                continue
            logger.info(
                "repair_plan: converting permanent pitch_shift(%.1f) to "
                "temporary_pitch_shift", semis,
            )
            call = {
                "tool": "temporary_pitch_shift",
                "song": "B",
                # start_time gets pushed past the crossfade end by
                # _enforce_revert_after_crossfade downstream.
                "start_time": 0.0,
                "semitones": semis,
                "fade_in_bars": 0,
                "hold_bars": 0,
                "fade_out_bars": PITCH_RETURN_BARS,
            }
        elif tool == "temporary_pitch_shift":
            semis = float(call.get("semitones", 0) or 0)
            capped = max(-PERMANENT_PITCH_SHIFT_CAP,
                         min(PERMANENT_PITCH_SHIFT_CAP, semis))
            if capped != semis:
                logger.info(
                    "repair_plan: clamping temporary pitch shift %.1f -> %.1f",
                    semis, capped,
                )
                call["semitones"] = capped
            if capped == 0:
                continue

        repaired.append(call)

    # Clamp the seam window into the headroom budget rather than rejecting.
    window = next(
        (c for c in repaired if c.get("tool") == "set_transition_window"), None
    )
    if window is not None:
        a_ceiling = max_seam_time(a.duration, a.bpm, a.time_signature)
        b_ceiling = max_seam_time(b.duration, b.bpm, b.time_signature)
        from_t = float(window.get("from_song_time_start", 0.0))
        to_t = float(window.get("to_song_time_start", 0.0))
        if from_t > a_ceiling:
            snapped = _latest_downbeat_at_or_before(a_ceiling, a.downbeats)
            logger.info(
                "repair_plan: clamping A seam %.1f -> %.1f (headroom)",
                from_t, snapped,
            )
            window["from_song_time_start"] = snapped
        if to_t > b_ceiling:
            snapped = _latest_downbeat_at_or_before(b_ceiling, b.downbeats)
            logger.info(
                "repair_plan: clamping B seam %.1f -> %.1f (headroom)",
                to_t, snapped,
            )
            window["to_song_time_start"] = snapped

    # Synthesize any missing stem crossfades from the ones present (or
    # window defaults), and drop duplicates beyond the first per stem.
    repaired = _normalize_stem_calls(repaired, window)

    # Exactly one window, and it leads the list.
    windows = [c for c in repaired if c.get("tool") == "set_transition_window"]
    if len(windows) >= 1:
        first = windows[0]
        repaired = [first] + [
            c for c in repaired if c.get("tool") != "set_transition_window"
        ]

    return repaired


def _latest_downbeat_at_or_before(t: float, downbeats: list[float]) -> float:
    if not downbeats:
        return max(0.0, t)
    best = None
    for d in downbeats:
        if d <= t:
            best = d
        else:
            break
    return best if best is not None else downbeats[0]


def _normalize_stem_calls(plan: list[dict], window: dict | None) -> list[dict]:
    """Guarantee exactly 4 crossfade_stem calls covering the canonical
    stems. Extra calls per stem are dropped (first wins); missing stems
    are cloned from the most common existing shape, or a default."""
    stem_calls = {}
    rest = []
    for c in plan:
        if c.get("tool") == "crossfade_stem":
            stem = c.get("stem")
            if stem in CANONICAL_STEMS and stem not in stem_calls:
                stem_calls[stem] = c
            else:
                logger.warning(
                    "repair_plan: dropping duplicate/unknown crossfade_stem %r",
                    stem,
                )
        else:
            rest.append(c)

    if not stem_calls and window is None:
        # Nothing to clone from and no window to size a default — beyond
        # repair; validate_plan will reject.
        return plan

    template = next(iter(stem_calls.values()), None)
    default_bars = (
        int(template["duration_bars"]) if template else
        int(window.get("duration_bars", 8)) if window else 8
    )
    for stem in CANONICAL_STEMS - set(stem_calls):
        logger.info("repair_plan: synthesizing missing crossfade_stem %r", stem)
        stem_calls[stem] = {
            "tool": "crossfade_stem", "stem": stem,
            "from_song": "A", "to_song": "B",
            "start_bar": int(template["start_bar"]) if template else 0,
            "duration_bars": default_bars,
            "curve": template.get("curve", "equal_power") if template else "equal_power",
        }

    ordered_stems = [stem_calls[s] for s in ("vocals", "drums", "bass", "other")]
    return rest + ordered_stems


def validate_plan(plan: list[dict]) -> None:
    """Final gate after repair: raise ValueError only for the genuinely
    unsalvageable. Mirrors the executor's own preconditions."""
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan is not a non-empty list")
    windows = [c for c in plan if c.get("tool") == "set_transition_window"]
    if len(windows) != 1:
        raise ValueError(f"expected 1 set_transition_window, got {len(windows)}")
    stem_calls = [c for c in plan if c.get("tool") == "crossfade_stem"]
    if len(stem_calls) != 4:
        raise ValueError(f"expected 4 crossfade_stem calls, got {len(stem_calls)}")
    if {c.get("stem") for c in stem_calls} != CANONICAL_STEMS:
        raise ValueError("crossfade_stem calls must cover vocals/drums/bass/other")
    illegal = {c.get("tool") for c in plan} - LEGAL_TOOLS
    if illegal:
        raise ValueError(f"illegal tools in plan: {sorted(illegal)}")
    if any(c.get("tool") == "pitch_shift" for c in plan):
        raise ValueError("permanent pitch_shift survived repair")
    for call in plan:
        for field in SONG_FIELDS_BY_TOOL.get(call.get("tool"), ()):
            if call.get(field) not in LEGAL_SONG_REFS:
                raise ValueError(
                    f"{call['tool']}.{field}={call.get(field)!r} must be 'A' or 'B'"
                )


def enforce_revert_after_crossfade(
    plan: list[dict], b: AnalysisBundle
) -> list[dict]:
    """Defer B's tempo ramp / temporary-pitch return until AFTER the A→B
    crossfade has fully finished (the bar where the LAST stem ends).

    B is auto-stretched to A's tempo (and held in A's key) at the seam;
    reverting mid-crossfade makes the two beat grids drift while both are
    audible. Archetype-expanded plans already satisfy this; the guard
    keeps free-form / cached plans honest too.
    """
    window = next(
        (c for c in plan if c.get("tool") == "set_transition_window"), None
    )
    if window is None:
        return plan
    sec_per_bar_b = (60.0 / b.bpm) * b.time_signature if b.bpm else 0.0
    if sec_per_bar_b <= 0:
        return plan

    stem_calls = [c for c in plan if c.get("tool") == "crossfade_stem"]
    if not stem_calls:
        return plan
    n_bars = max(
        int(c.get("start_bar", 0)) + int(c.get("duration_bars", 0))
        for c in stem_calls
    )
    to_start = float(window.get("to_song_time_start", 0.0))
    crossfade_end_b = to_start + n_bars * sec_per_bar_b

    out: list[dict] = []
    for call in plan:
        tool = call.get("tool")
        if tool == "set_tempo_ramp":
            start = float(call.get("start_time", 0.0))
            if start < crossfade_end_b:
                delta = crossfade_end_b - start
                end = float(call.get("end_time", start)) + delta
                end = min(end, b.duration)
                if end <= crossfade_end_b:
                    end = b.duration
                logger.info(
                    "validation: deferred tempo ramp start %.2f -> %.2f",
                    start, crossfade_end_b,
                )
                call = {**call, "start_time": crossfade_end_b, "end_time": end}
        elif tool == "temporary_pitch_shift":
            start = float(call.get("start_time", 0.0))
            if start < crossfade_end_b:
                logger.info(
                    "validation: deferred pitch return start %.2f -> %.2f",
                    start, crossfade_end_b,
                )
                call = {**call, "start_time": crossfade_end_b}
        out.append(call)
    return out


def strip_pitch_tools(plan: list[dict]) -> list[dict]:
    """Remove every in-plan pitch tool (permanent AND temporary).

    Applied by the worker whenever settings.pitch_mode != "temporary":
    under whole-song mode the audio is pre-shifted before rendering and
    under "off" mode clashes are accepted, so any pitch call left in a
    plan (legacy prompt output, deterministic fallback on a clash, or a
    cached pre-migration plan) would double-shift or glide. Tempo ramps
    are untouched — they don't change key.
    """
    return [
        c for c in plan
        if c.get("tool") not in ("pitch_shift", "temporary_pitch_shift")
    ]
