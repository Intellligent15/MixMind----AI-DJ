"""Planner v2 orchestration.

build_plan_v2() is the single entry point the render worker calls:

  1. build the seam-candidate menus (pure math, always valid);
  2. compute the pair facts the model should NOT have to derive
     (tempo gap, key-compatibility verdict);
  3. ask the LLM for a TransitionDecision (style hint / user override /
     previously-used styles ride along as context; a reroll nonce busts
     the response cache);
  4. expand the decision deterministically into tool calls.

Every step that can fail degrades gracefully: a bad decision triggers
one repair attempt (default knobs, keep the model's style if legal),
and total LLM failure falls back to either a pinned-style default
expansion or the v1 deterministic planner. The caller learns which path
produced the plan via PlanOutcome.source — no more silent fallbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import ValidationError

from app.services.mixer.archetypes import (
    ACAPELLA_STYLES,
    ArchetypeError,
    camelot_compatible,
    default_decision,
    expand,
    halftime_ratio,
)
from app.services.mixer.candidates import (
    PairCandidates,
    build_pair_candidates,
    enrich_sections,
)
from app.services.mixer.decision import TransitionDecision, TransitionStyle
from app.services.mixer.pitch_resolver import effective_bundle
from app.services.mixer.plan import build_pair_plan, compute_pitch_shift
from app.services.mixer.types import AnalysisBundle, MixPlanJSON

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SongMeta:
    """Identity + planning inputs the worker snapshots from the DB."""

    title: str
    artist: str | None
    bundle: AnalysisBundle
    energy_curve: list[float]
    safe_regions: list[dict]
    # Whole-song semitone offset assigned by the set-level pitch
    # resolver (0 unless settings.pitch_mode == "whole_song"). The
    # planner reasons about EFFECTIVE keys (native + offset); the worker
    # pre-shifts the audio to match.
    pitch_offset: int = 0


@dataclass(frozen=True)
class PlanOutcome:
    plan: MixPlanJSON
    source: str            # "llm_v2" | "llm_v2_repaired" | "style_default" | "deterministic_fallback"
    style: str | None
    rationale: str | None


def _song_llm_input(meta: SongMeta, candidates: list) -> dict:
    b = effective_bundle(meta.bundle, meta.pitch_offset)
    out = {
        "title": meta.title,
        "artist": meta.artist,
        "bpm": b.bpm,
        "key": b.key,
        "camelot_key": b.camelot_key,
        "duration": round(b.duration, 1),
        "sections": enrich_sections(b.sections, meta.energy_curve)[:12],
        "candidates": [c.to_llm_dict() for c in candidates],
    }
    if b.tags:
        out["tags"] = b.tags
    return out


def _pair_facts(
    a: AnalysisBundle, b: AnalysisBundle, pitch_mode: str = "temporary"
) -> dict:
    tempo_gap_pct = (
        round(abs(a.bpm - b.bpm) / a.bpm * 100.0, 1) if a.bpm and b.bpm else None
    )
    compatible = camelot_compatible(a.camelot_key, b.camelot_key)
    if compatible:
        key_verdict = (
            "compatible under the standard Camelot rule (same code, relative "
            "major/minor, or adjacent on the wheel) — no pitch handling "
            "needed; acapella styles are fair game even if the keys differ"
        )
    elif pitch_mode == "temporary":
        try:
            delta = compute_pitch_shift(a.key, b.key)
        except ValueError:
            delta = 0
        key_verdict = (
            f"clash — B will be held {delta:+d} semitones in A's key during "
            f"the blend, automatically"
        )
    else:
        # whole_song mode with a remaining clash (resolver gave up: the
        # needed shift exceeded the artifact cap), or pitch_mode "off".
        # Route the model toward styles where the keys barely overlap.
        key_verdict = (
            "clash — keys cannot be matched; strongly prefer a short or "
            "washy transition (drop_swap, wash_out, vinyl_stop, or a short "
            "stutter_buildup) and avoid long melodic blends"
        )
    facts = {
        "tempo_gap_percent": tempo_gap_pct,
        "tempo_note": "B is beatmatched to A automatically; ignore tempo math",
        "key_verdict": key_verdict,
    }
    if (
        a.bpm and b.bpm
        and tempo_gap_pct is not None and tempo_gap_pct > 12.0
        and halftime_ratio(a.bpm, b.bpm) != 1.0
    ):
        facts["halftime_note"] = (
            "the BPMs sit at a 2:1 half/double-time ratio — the beat grids "
            "interlock natively, so treat the tempo gap as SMALL. Drum-led "
            "styles (drum_bridge, drop_swap, double_drop) work great here."
        )
    return facts


def _parse_decision(obj) -> TransitionDecision:
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError(f"decision is not a JSON object: {type(obj).__name__}")
    # Some models wrap: {"decision": {...}} / {"plan": {...}}.
    for key in ("decision", "plan", "transition"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return TransitionDecision.model_validate(obj)


def _repair_decision(
    obj, candidates: PairCandidates
) -> TransitionDecision:
    """Second chance for a near-miss decision: keep whatever fields are
    legal, default the rest."""
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError("irreparable decision payload")
    style = None
    raw_style = obj.get("style")
    if isinstance(raw_style, str):
        try:
            style = TransitionStyle(raw_style.strip().lower())
        except ValueError:
            style = None
    base = default_decision(candidates, style=style)
    out = obj.get("out")
    in_ = obj.get("in")
    data = base.model_dump(by_alias=True)
    if isinstance(out, str) and candidates.find(out) and out.startswith("A"):
        data["out"] = out
    if isinstance(in_, str) and candidates.find(in_) and in_.startswith("B"):
        data["in"] = in_
    if isinstance(obj.get("rationale"), str):
        data["rationale"] = obj["rationale"][:600]
    return TransitionDecision.model_validate(data)


async def build_plan_v2(
    provider,
    a: SongMeta,
    b: SongMeta,
    *,
    style_hint: str | None = None,
    style_override: str | None = None,
    previous_styles: list[str] | None = None,
    pair_label: str | None = None,
    nonce: int = 0,
    pitch_mode: str = "temporary",
    loudness_match: bool = True,
    avoid_styles: list[str] | None = None,
    bass_swap: bool = True,
    tempo_meet: bool = True,
    extra_context: dict | None = None,
    tease_enabled: bool = False,
) -> PlanOutcome:
    # Lazy import keeps the mixer package importable without the llm
    # package's heavy provider dependencies (pure unit tests, tooling).
    from app.services.llm.prompts import (
        DECISION_SYSTEM_PROMPT,
        decision_user_prompt,
    )

    # EFFECTIVE bundles fold each song's whole-song offset into its
    # key/camelot — all key reasoning below (pair facts, archetype pitch
    # logic, deterministic fallback) sees the keys the listener will hear.
    a_eff = effective_bundle(a.bundle, a.pitch_offset)
    b_eff = effective_bundle(b.bundle, b.pitch_offset)

    candidates = build_pair_candidates(
        a.bundle, b.bundle,
        a.energy_curve, b.energy_curve,
        a.safe_regions, b.safe_regions,
    )

    pinned_style: TransitionStyle | None = None
    if style_override:
        try:
            pinned_style = TransitionStyle(style_override)
        except ValueError:
            logger.warning("planner_v2: unknown style_override %r", style_override)

    if not candidates.out_candidates or not candidates.in_candidates:
        logger.warning(
            "planner_v2: no usable seam candidates; deterministic fallback"
        )
        return PlanOutcome(
            plan=build_pair_plan(a_eff, b_eff),
            source="deterministic_fallback",
            style=None,
            rationale="songs too short for candidate generation",
        )

    context: dict = {}
    if extra_context:
        # Occasion / vibe / listener-feedback / energy-dial lines from the
        # worker. Merged first so the planner-specific keys below win on
        # collision.
        context.update(extra_context)
    if tease_enabled:
        context["hook_teasing"] = (
            "Hook teasing is ENABLED for this set: you may add "
            '"tease": true to your decision to sneak B\'s vocal hook over '
            "an instrumental pocket late in A, minutes before the seam. "
            "Use it SPARINGLY (at most ~2 per set) and only when B's hook "
            "is genuinely iconic; placement is computed automatically."
        )
    if pinned_style is not None:
        context["forced_style"] = (
            f"The user pinned this transition's style to '{pinned_style.value}'. "
            f"You MUST use it; choose only the seams and knobs."
        )
    elif style_hint:
        context["suggested_style"] = style_hint
    if previous_styles:
        context["styles_used_so_far"] = previous_styles
    if avoid_styles:
        context["avoid_styles"] = (
            f"A previous render of this pair failed automated audio QA using "
            f"style(s) {', '.join(avoid_styles)} — choose a DIFFERENT style."
        )
    if pair_label:
        # e.g. "transition 3 of 5" — lets the model shape the set's arc
        # (open gently, peak in the middle, land the closer).
        context["position_in_set"] = pair_label

    user = decision_user_prompt(
        _song_llm_input(a, candidates.out_candidates),
        _song_llm_input(b, candidates.in_candidates),
        _pair_facts(a_eff, b_eff, pitch_mode),
        context or None,
    )

    decision: TransitionDecision | None = None
    source = "llm_v2"
    try:
        obj = await provider.complete_json(
            system=DECISION_SYSTEM_PROMPT, user=user, nonce=nonce
        )
        try:
            decision = _parse_decision(obj)
        except (ValidationError, ValueError) as exc:
            logger.warning("planner_v2: repairing decision (%s)", exc)
            decision = _repair_decision(obj, candidates)
            source = "llm_v2_repaired"
    except Exception as exc:
        logger.error("planner_v2: LLM decision failed: %s", exc)

    if decision is not None and decision.tease and not tease_enabled:
        # The model opted into a feature the queue didn't enable.
        decision = decision.model_copy(update={"tease": False})

    if decision is not None and pinned_style is not None:
        if decision.style != pinned_style:
            decision = decision.model_copy(update={"style": pinned_style})
            decision = decision.model_copy(
                update={"duration_bars": decision.normalized_duration()}
            )

    # Acapella styles layer one song's VOCALS over the other's
    # instrumental — the single most key-exposed move in the toolkit. If
    # the model picked one on a pair whose effective keys still clash
    # (resolver hit the artifact cap), downgrade to a smooth blend. A
    # user pin is honored as-is: their call.
    if (
        decision is not None
        and pinned_style is None
        and decision.style in ACAPELLA_STYLES
        and not camelot_compatible(a_eff.camelot_key, b_eff.camelot_key)
    ):
        logger.info(
            "planner_v2: %s chosen on a key clash; downgrading to smooth_blend",
            decision.style.value,
        )
        decision = decision.model_copy(update={"style": TransitionStyle.smooth_blend})
        decision = decision.model_copy(
            update={"duration_bars": decision.normalized_duration()}
        )

    # double_drop stacks both songs at full energy — it only works when
    # the keys are Camelot-compatible, the (half-time-aware) tempo gap is
    # small, and BOTH chosen seams really are drops. Anything less
    # downgrades to a plain drop_swap. A user pin is honored as-is.
    if (
        decision is not None
        and pinned_style is None
        and decision.style == TransitionStyle.double_drop
    ):
        out_c = candidates.find(decision.out)
        in_c = candidates.find(decision.in_)
        gap_ok = False
        if a_eff.bpm and b_eff.bpm:
            ratio = halftime_ratio(a_eff.bpm, b_eff.bpm)
            target = a_eff.bpm * ratio
            gap_ok = abs(target - b_eff.bpm) / target <= 0.06
        ok = (
            gap_ok
            and camelot_compatible(a_eff.camelot_key, b_eff.camelot_key)
            and out_c is not None and out_c.energy >= 0.8
            and in_c is not None and in_c.energy >= 0.8
        )
        if not ok:
            logger.info(
                "planner_v2: double_drop conditions not met; downgrading "
                "to drop_swap"
            )
            decision = decision.model_copy(
                update={"style": TransitionStyle.drop_swap}
            )
            decision = decision.model_copy(
                update={"duration_bars": decision.normalized_duration()}
            )

    # Theatrics gate: vinyl_stop / backspin are full-stop moves — the
    # escape hatch for pairs that genuinely can't blend (unbridgeable
    # tempo gap with no half-time relationship, or an unmatchable key
    # clash). On an ordinary pair they read as showing off, and twice in
    # one set they read as a broken record. Downgrade to drop_swap (keeps
    # the snap) unless the user pinned the style.
    THEATRICS = (TransitionStyle.vinyl_stop, TransitionStyle.backspin)
    if (
        decision is not None
        and pinned_style is None
        and decision.style in THEATRICS
    ):
        already_used = any(s in {t.value for t in THEATRICS}
                           for s in (previous_styles or []))
        gap_bridgeable = True
        if a_eff.bpm and b_eff.bpm:
            ratio = halftime_ratio(a_eff.bpm, b_eff.bpm)
            target = a_eff.bpm * ratio
            gap_bridgeable = abs(target - b_eff.bpm) / target <= 0.12
        keys_workable = camelot_compatible(a_eff.camelot_key, b_eff.camelot_key)
        if pitch_mode != "off" and not keys_workable:
            # A modest clash is fixable by the pitch machinery; only a
            # beyond-cap clash justifies the full stop.
            try:
                keys_workable = abs(
                    compute_pitch_shift(a_eff.key, b_eff.key)
                ) <= 2
            except ValueError:
                keys_workable = False
        justified = not (gap_bridgeable and keys_workable)
        if already_used or not justified:
            logger.info(
                "planner_v2: downgrading %s (used_before=%s, justified=%s)",
                decision.style.value, already_used, justified,
            )
            decision = decision.model_copy(
                update={"style": TransitionStyle.drop_swap}
            )
            decision = decision.model_copy(
                update={"duration_bars": decision.normalized_duration()}
            )

    if decision is not None:
        try:
            plan = expand(
                decision, a_eff, b_eff, candidates, pitch_mode,
                a_safe_regions=a.safe_regions, b_safe_regions=b.safe_regions,
                a_energy_curve=a.energy_curve, b_energy_curve=b.energy_curve,
                loudness_match=loudness_match, bass_swap=bass_swap,
                tempo_meet=tempo_meet,
            )
            return PlanOutcome(
                plan=plan, source=source,
                style=decision.style.value,
                rationale=decision.rationale or None,
            )
        except ArchetypeError as exc:
            logger.error("planner_v2: expansion failed (%s); using defaults", exc)

    # LLM unavailable or decision unusable: pinned style still expands
    # deterministically; otherwise fall back to the v1 planner.
    try:
        fallback = default_decision(candidates, style=pinned_style)
        plan = expand(
            fallback, a_eff, b_eff, candidates, pitch_mode,
            a_safe_regions=a.safe_regions, b_safe_regions=b.safe_regions,
            a_energy_curve=a.energy_curve, b_energy_curve=b.energy_curve,
            loudness_match=loudness_match, bass_swap=bass_swap,
            tempo_meet=tempo_meet,
        )
        return PlanOutcome(
            plan=plan, source="style_default",
            style=fallback.style.value, rationale=fallback.rationale,
        )
    except ArchetypeError as exc:
        logger.error("planner_v2: default expansion failed (%s)", exc)
        return PlanOutcome(
            plan=build_pair_plan(a_eff, b_eff),
            source="deterministic_fallback",
            style=None, rationale=None,
        )
