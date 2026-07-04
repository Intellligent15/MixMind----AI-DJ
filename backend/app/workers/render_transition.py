"""Render a MixPlan's transition into a WAV via the mixer executor.

Atomic-claim pattern mirrors separate_stems / transcribe_song: a single
UPDATE WHERE status=pending|failed|ready transitions the row to
`rendering`. Losers (status already rendering, or row missing) return
None.

`plan_json` is generated lazily on the first render so we don't burn
work for plans the user never asks to render.

Planner architecture (settings.planner_version):

* "v2" (default) — `planner_v2.build_plan_v2`: pre-computed seam
  candidates + an LLM-chosen transition archetype, deterministically
  expanded. The LLM sees song *identity* (title/artist), section
  structure, and pre-computed pair facts; it never does timestamp math.
* "legacy" — the v1 free-form tool-call prompt, now run through
  `validation.repair_plan` (normalize song refs, clamp seams, convert
  forbidden permanent pitch shifts) before the final `validate_plan`
  gate, so a fixable slip no longer discards a musical plan.

Either way the worker records `plan_source` / `style` / `rationale` on
the row — a fallback is never silent again.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import select, update

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    Song,
    SongStatus,
    Stems,
    Transcription,
)
from app.models.lyrics import Lyrics, LyricsAlignmentStatus
from app.services.llm import get_llm_provider
from app.services.mixer.candidates import enrich_sections, max_seam_time
from app.services.mixer.executor import render
from app.services.feedback.summary import feedback_context
from app.services.mixer.occasions import occasion_context
from app.services.mixer.plan import build_pair_plan
from app.services.mixer.pitch_resolver import effective_bundle
from app.services.mixer.planner_v2 import PlanOutcome, SongMeta, build_plan_v2
from app.services.mixer.preshift import ensure_pitched_inputs
from app.services.mixer.qa import score_render
from app.services.mixer.types import AnalysisBundle, SongRenderInputs
from app.services.mixer.validation import (
    enforce_revert_after_crossfade,
    repair_plan,
    strip_pitch_tools,
    validate_plan,
)
from app.services.storage import get_storage
from app.services.vocal_safety.safety import vocal_safe_regions
from app.workers import celery_app

logger = logging.getLogger(__name__)

CLAIMABLE_STATUSES = (
    MixPlanStatus.pending,
    MixPlanStatus.failed,
    MixPlanStatus.ready,  # allows re-render of an already-rendered pair
)

# Back-compat aliases: tests and older callers import these from here.
_enrich_sections = enrich_sections
_max_seam_time = max_seam_time
_validate_llm_plan = validate_plan
_enforce_revert_after_crossfade = enforce_revert_after_crossfade


def _to_bundle(analysis: Analysis, duration: float) -> AnalysisBundle:
    return AnalysisBundle(
        bpm=analysis.bpm,
        key=analysis.key,
        camelot_key=analysis.camelot_key,
        time_signature=analysis.time_signature,
        beat_grid=list(analysis.beat_grid),
        downbeats=list(analysis.downbeats),
        sections=list(analysis.sections),
        duration=duration,
        envelopes=None,
        transcription_segments=None,
        tags=analysis.tags,
    )


def _stem_paths(stems: Stems) -> dict[str, str]:
    return {
        "vocals": stems.vocals_path,
        "drums": stems.drums_path,
        "bass": stems.bass_path,
        "other": stems.other_path,
    }


def _bundle_to_legacy_llm_dict(
    bundle: AnalysisBundle, energy_curve: list[float],
    title: str, artist: str | None,
) -> dict:
    """v1 free-form prompt input — now carrying song identity too."""
    sec_per_bar = (
        round((60.0 / bundle.bpm) * bundle.time_signature, 3)
        if bundle.bpm else 0.0
    )
    return {
        "title": title,
        "artist": artist,
        "bpm": bundle.bpm,
        "key": bundle.key,
        "camelot_key": bundle.camelot_key,
        "time_signature": bundle.time_signature,
        "seconds_per_bar": sec_per_bar,
        "duration": bundle.duration,
        "sections": enrich_sections(bundle.sections, energy_curve),
        "max_seam_time": max_seam_time(
            bundle.duration, bundle.bpm, bundle.time_signature
        ),
    }


_LEGACY_TOOLS_SCHEMA = json.dumps([
    {"tool": "set_transition_window", "from_song_time_start": "float", "to_song_time_start": "float", "duration_bars": "int"},
    {"tool": "crossfade_stem", "stem": "str", "from_song": "str", "to_song": "str", "start_bar": "int", "duration_bars": "int", "curve": "str", "a_fade_out_bars": "int (optional; bars over which A fades out, <= duration_bars; default duration_bars)"},
    {"tool": "temporary_pitch_shift", "song": "str", "start_time": "float", "semitones": "float", "fade_in_bars": "int", "hold_bars": "int", "fade_out_bars": "int"},
    {"tool": "set_tempo_ramp", "song": "str", "start_time": "float", "end_time": "float", "start_bpm": "float", "end_bpm": "float"},
    {"tool": "filter_sweep", "song": "str", "type": "str (lowpass|highpass)", "start_time": "float", "end_time": "float", "start_cutoff_hz": "float", "end_cutoff_hz": "float"},
    {"tool": "echo_out", "song": "str", "start_time": "float", "beats": "int", "feedback": "float (0..0.9)", "bpm": "float"},
    {"tool": "loop_section", "song": "str", "start_time": "float", "beats": "float (may be fractional, e.g. 0.5/0.25, for rapid stutters)", "repeats": "int", "bpm": "float"},
    {"tool": "swap_stem", "from_song": "str", "to_song": "str", "stem": "str (vocals|drums|bass|other)", "time": "float (output-timeline seconds)"},
    {"tool": "apply_reverb", "song": "str", "start_time": "float", "tail_duration_bars": "float", "wet_level": "float (0..1)", "bpm": "float"},
    {"tool": "turntable_stop", "song": "str", "start_time": "float", "duration_bars": "float", "bpm": "float"},
    {"tool": "volume_fade", "song": "str", "start_time": "float", "duration_bars": "float", "start_gain": "float", "end_gain": "float", "bpm": "float", "stem": "str (optional: vocals|drums|bass|other)"},
], indent=2)


@celery_app.task(name="app.workers.render_transition.render_transition")
def render_transition(mix_plan_id: str) -> str | None:
    plan_uuid = uuid.UUID(mix_plan_id)
    storage = get_storage()

    # Phase 1: load the row, validate, atomically claim.
    with SessionLocal() as db:
        row = db.get(MixPlan, plan_uuid)
        if row is None:
            logger.warning("render_transition: %s not found", mix_plan_id)
            return None

        a = db.get(Song, row.from_song_id)
        b = db.get(Song, row.to_song_id)
        if a is None or b is None:
            logger.error("render_transition: %s missing songs", mix_plan_id)
            return None
        if a.status != SongStatus.ready or b.status != SongStatus.ready:
            logger.warning(
                "render_transition: %s songs not ready (a=%s, b=%s)",
                mix_plan_id, a.status.value, b.status.value,
            )
            return None

        claim = db.execute(
            update(MixPlan)
            .where(MixPlan.id == plan_uuid)
            .where(MixPlan.status.in_(CLAIMABLE_STATUSES))
            .values(status=MixPlanStatus.rendering, error_text=None)
        )
        db.commit()
        if claim.rowcount == 0:
            db.refresh(row)
            logger.info(
                "render_transition: %s already %s, skipping",
                mix_plan_id, row.status.value,
            )
            return None

        # Snapshot what we need outside the session.
        a_analysis = db.scalar(select(Analysis).where(Analysis.song_id == a.id))
        b_analysis = db.scalar(select(Analysis).where(Analysis.song_id == b.id))
        a_stems = db.scalar(select(Stems).where(Stems.song_id == a.id))
        b_stems = db.scalar(select(Stems).where(Stems.song_id == b.id))
        if not (a_analysis and b_analysis and a_stems and b_stems):
            logger.error("render_transition: %s missing analysis/stems", mix_plan_id)
            _mark_failed(plan_uuid, "missing analysis or stems")
            return None

        # Optional inputs for LLM planning.
        a_transcription = db.scalar(select(Transcription).where(Transcription.song_id == a.id))
        b_transcription = db.scalar(select(Transcription).where(Transcription.song_id == b.id))
        a_lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == a.id))
        b_lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == b.id))

        a_bundle = _to_bundle(a_analysis, a.duration_seconds)
        b_bundle = _to_bundle(b_analysis, b.duration_seconds)
        existing_plan_json = row.plan_json
        a_audio_key = a.audio_path
        b_audio_key = b.audio_path

        # Song identity — the model knows real songs; let it use that.
        a_title, a_artist = a.title, a.artist
        b_title, b_artist = b.title, b.artist

        a_envelope_path = a_stems.envelopes_path
        b_envelope_path = b_stems.envelopes_path

        # `energy_curve` is sampled at 1Hz in the analyzer; we average it
        # per section so structure and energy arrive together.
        a_energy_curve = list(a_analysis.energy_curve or [])
        b_energy_curve = list(b_analysis.energy_curve or [])

        # Set-level pass suggestion (soft), user pin (hard), reroll nonce.
        style_hint = row.style_hint
        style_override = row.style_override
        reroll_nonce = row.reroll_nonce or 0
        # Occasion / vibe / energy-dial context for the decision prompt.
        queue_row = db.get(Queue, row.queue_id)
        tease_enabled = bool(queue_row.tease_hooks) if queue_row else False
        extra_context = occasion_context(
            queue_row.occasion if queue_row else None,
            queue_row.vibe_note if queue_row else None,
        )
        if row.energy_bias == "up":
            extra_context["energy_dial"] = (
                "The listener just asked to RAISE the energy: prefer "
                "high-energy IN candidates and energetic styles "
                "(drop_swap, stutter_buildup, double_drop), shorter blends."
            )
        elif row.energy_bias == "down":
            extra_context["energy_dial"] = (
                "The listener just asked to COOL things down: prefer "
                "low-energy IN candidates and gentle styles "
                "(breakdown_blend, wash_out, long smooth_blend)."
            )
        # Reaction history (F5): what this listener liked/disliked/skipped.
        try:
            def _genre(analysis):
                tags = analysis.tags or {}
                genres = tags.get("genres") or []
                return genres[0] if genres else None

            fb_line = feedback_context(db, _genre(a_analysis), _genre(b_analysis))
            if fb_line:
                extra_context["listener_feedback"] = fb_line
        except Exception:  # feedback must never block a render
            logger.warning("render_transition: feedback summary failed",
                           exc_info=True)
        # Style of a cached plan_json, for QA's dropout exemption and the
        # avoid-list when a cached plan fails QA.
        existing_style = row.style

        # Styles of this queue's other already-planned pairs, in QUEUE
        # ORDER, so the per-pair decision can avoid repeating the same
        # trick. The explicit sort matters twice over: the model reads the
        # set's history in playback order, and the list is part of the LLM
        # cache key — an unordered query would hash differently from run
        # to run and cause spurious cache misses.
        queue_items = db.scalars(
            select(QueueItem).where(QueueItem.queue_id == row.queue_id)
        ).all()
        positions = {it.song_id: it.position for it in queue_items}
        # Whole-song pitch offsets resolved by the set-level pass (0 in
        # "temporary"/"off" modes or before the pass has run).
        pitch_offsets = {
            it.song_id: (it.pitch_offset_semitones or 0) for it in queue_items
        }
        a_pitch_offset = pitch_offsets.get(a.id, 0)
        b_pitch_offset = pitch_offsets.get(b.id, 0)
        # Plain-string snapshots for use after the session closes
        # (detached ORM attributes can expire on commit).
        a_song_id = str(a.id)
        b_song_id = str(b.id)
        siblings = sorted(
            db.scalars(
                select(MixPlan).where(
                    MixPlan.queue_id == row.queue_id, MixPlan.id != row.id
                )
            ).all(),
            key=lambda s: positions.get(s.from_song_id, 1_000_000),
        )
        previous_styles = [s.style for s in siblings if s.style]
        # 1-based "transition N of M" label for the set context.
        n_pairs = max(len(positions) - 1, 1)
        this_pos = positions.get(row.from_song_id)
        pair_label = (
            f"transition {this_pos + 1} of {n_pairs}"
            if this_pos is not None else None
        )

    async def _fetch_safe_regions_and_envelope(transcription, lyrics, envelope_path, duration):
        if not transcription or not envelope_path:
            return [], None
        try:
            envelope_data = await storage.read(envelope_path)
            envelope = json.loads(envelope_data.decode("utf-8"))
        except Exception:
            return [], None

        aligned_words = None
        if lyrics and lyrics.alignment_status == LyricsAlignmentStatus.success:
            aligned_words = lyrics.aligned_words

        safe_regions = vocal_safe_regions(
            transcription_segments=transcription.segments,
            envelope=envelope,
            aligned_words=aligned_words,
            duration_seconds=duration or 0.0,
        )
        return safe_regions, envelope

    async def _build_plan(
        avoid_styles: list[str] | None = None, nonce: int | None = None
    ) -> PlanOutcome:
        nonlocal a_bundle, b_bundle

        if not settings.use_llm_planner:
            return PlanOutcome(
                plan=build_pair_plan(
                    effective_bundle(a_bundle, a_pitch_offset),
                    effective_bundle(b_bundle, b_pitch_offset),
                ),
                source="deterministic", style=None, rationale=None,
            )

        a_safe_regions, a_env = await _fetch_safe_regions_and_envelope(
            a_transcription, a_lyrics, a_envelope_path, a_bundle.duration
        )
        b_safe_regions, b_env = await _fetch_safe_regions_and_envelope(
            b_transcription, b_lyrics, b_envelope_path, b_bundle.duration
        )

        a_bundle = dataclasses.replace(
            a_bundle,
            envelopes=a_env,
            transcription_segments=a_transcription.segments if a_transcription else None,
        )
        b_bundle = dataclasses.replace(
            b_bundle,
            envelopes=b_env,
            transcription_segments=b_transcription.segments if b_transcription else None,
        )

        a_eff = effective_bundle(a_bundle, a_pitch_offset)
        b_eff = effective_bundle(b_bundle, b_pitch_offset)

        provider = get_llm_provider()

        if settings.planner_version == "v2":
            return await build_plan_v2(
                provider,
                SongMeta(a_title, a_artist, a_bundle, a_energy_curve,
                         a_safe_regions, pitch_offset=a_pitch_offset),
                SongMeta(b_title, b_artist, b_bundle, b_energy_curve,
                         b_safe_regions, pitch_offset=b_pitch_offset),
                style_hint=style_hint,
                style_override=style_override,
                previous_styles=previous_styles,
                pair_label=pair_label,
                nonce=nonce if nonce is not None else reroll_nonce,
                pitch_mode=settings.pitch_mode,
                loudness_match=settings.loudness_match,
                avoid_styles=avoid_styles,
                bass_swap=settings.bass_swap,
                tempo_meet=settings.tempo_meet_in_middle,
                extra_context=extra_context,
                tease_enabled=tease_enabled,
            )

        # ---- legacy free-form path, with repair-not-reject ----
        a_llm_input = {
            "analysis": _bundle_to_legacy_llm_dict(
                a_eff, a_energy_curve, a_title, a_artist
            ),
            "vocal_safe_regions": a_safe_regions,
        }
        b_llm_input = {
            "analysis": _bundle_to_legacy_llm_dict(
                b_eff, b_energy_curve, b_title, b_artist
            ),
            "vocal_safe_regions": b_safe_regions,
        }
        try:
            plan = await provider.plan_transition(
                a_llm_input, b_llm_input, _LEGACY_TOOLS_SCHEMA
            )
            repaired = repair_plan(plan, a_eff, b_eff)
            validate_plan(repaired)
            source = "llm_legacy" if repaired == plan else "llm_legacy_repaired"
            return PlanOutcome(plan=repaired, source=source, style=None, rationale=None)
        except Exception as exc:
            logger.error(
                "render_transition: legacy LLM planner failed, falling back "
                "to deterministic: %s", exc,
            )
            return PlanOutcome(
                plan=build_pair_plan(a_eff, b_eff),
                source="deterministic_fallback", style=None, rationale=None,
            )

    if existing_plan_json:
        outcome = PlanOutcome(
            plan=existing_plan_json, source="cached", style=None, rationale=None
        )
    else:
        outcome = asyncio.run(_build_plan())
        logger.info(
            "render_transition: %s plan source=%s style=%s",
            mix_plan_id, outcome.source, outcome.style,
        )

    def _finalize_plan(plan: list[dict]) -> list[dict]:
        # Outside "temporary" mode no plan may carry pitch tools: whole-song
        # mode pre-shifts the audio itself (a leftover tool would double-
        # shift) and "off" mode accepts clashes (a leftover tool would
        # glide). Covers legacy-prompt output, deterministic fallbacks on a
        # clash, and cached pre-migration plans alike. Then guarantee B's
        # tempo/pitch revert only fires once the crossfade is done.
        if settings.pitch_mode != "temporary":
            plan = strip_pitch_tools(plan)
        return enforce_revert_after_crossfade(plan, b_bundle)

    plan_json = _finalize_plan(outcome.plan)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        async def _download_inputs():
            async def _stems(stems: Stems, prefix: str):
                paths = {}
                for k, key in _stem_paths(stems).items():
                    dest = tmp / f"{prefix}_{k}.wav"
                    await storage.download_file(key, dest)
                    paths[k] = str(dest)
                return paths

            async def _original(key: str | None, prefix: str) -> str | None:
                if not key:
                    return None
                dest = tmp / f"{prefix}_original.wav"
                await storage.download_file(key, dest)
                return str(dest)

            return (
                await _stems(a_stems, "a"),
                await _stems(b_stems, "b"),
                await _original(a_audio_key, "a"),
                await _original(b_audio_key, "b"),
            )

        a_paths, b_paths, a_orig, b_orig = asyncio.run(_download_inputs())

        # Whole-song pitch: mutate the local copies so the executor (and
        # therefore BOTH renders that touch each song) sees the shifted
        # audio. Cached in storage per (song, offset) — the second render
        # sharing a song downloads instead of re-shifting.
        if settings.pitch_mode == "whole_song" and (a_pitch_offset or b_pitch_offset):
            async def _preshift_all():
                await ensure_pitched_inputs(
                    storage, a_song_id, a_pitch_offset, a_paths, a_orig
                )
                await ensure_pitched_inputs(
                    storage, b_song_id, b_pitch_offset, b_paths, b_orig
                )
            asyncio.run(_preshift_all())

        a_inputs = SongRenderInputs(
            stem_paths=a_paths, analysis=a_bundle, original_audio_path=a_orig
        )
        b_inputs = SongRenderInputs(
            stem_paths=b_paths, analysis=b_bundle, original_audio_path=b_orig
        )

        try:
            result = render(plan_json, a_inputs, b_inputs)
        except Exception as exc:
            logger.exception("render_transition: %s render failed", mix_plan_id)
            _mark_failed(plan_uuid, f"{type(exc).__name__}: {exc}")
            return None

        # Render QA: score the output; on a hard fail, re-plan ONCE with
        # the failed style excluded and keep whichever render scores
        # better. A user style pin is honored as-is (their call), and the
        # retry needs the v2 LLM path to produce a different plan at all.
        style_for_qa = outcome.style or existing_style
        qa = score_render(result.wav_bytes, plan_json, a_bundle, style=style_for_qa)
        adopted_retry_nonce: int | None = None
        if (
            qa.verdict == "fail"
            and settings.use_llm_planner
            and settings.planner_version == "v2"
            and not style_override
        ):
            logger.warning(
                "render_transition: %s failed QA (%s); re-planning once",
                mix_plan_id, ", ".join(qa.flags),
            )
            try:
                retry_outcome = asyncio.run(_build_plan(
                    avoid_styles=[s for s in (style_for_qa,) if s],
                    nonce=reroll_nonce + 1,
                ))
                retry_plan = _finalize_plan(retry_outcome.plan)
                retry_result = render(retry_plan, a_inputs, b_inputs)
                retry_qa = score_render(
                    retry_result.wav_bytes, retry_plan, a_bundle,
                    style=retry_outcome.style,
                )
                if not retry_qa.worse_than(qa):
                    result, qa = retry_result, retry_qa
                    plan_json, outcome = retry_plan, retry_outcome
                    # Persisted below so future rerolls' LLM cache keys
                    # move past the plan QA already rejected.
                    adopted_retry_nonce = reroll_nonce + 1
                    logger.info(
                        "render_transition: %s QA retry adopted (style=%s, "
                        "verdict=%s)", mix_plan_id, outcome.style, qa.verdict,
                    )
                else:
                    logger.info(
                        "render_transition: %s QA retry scored worse (%s); "
                        "keeping original", mix_plan_id, retry_qa.verdict,
                    )
            except Exception:
                logger.exception(
                    "render_transition: %s QA retry errored; keeping "
                    "original render", mix_plan_id,
                )
        if qa.verdict != "pass":
            logger.warning(
                "render_transition: %s shipping with QA verdict=%s (%s)",
                mix_plan_id, qa.verdict, ", ".join(qa.flags),
            )

    # Phase 4: persist output via storage, flip to ready.
    key = f"mixes/{mix_plan_id}.wav"
    asyncio.run(storage.write(key, result.wav_bytes))

    with SessionLocal() as db:
        row = db.get(MixPlan, plan_uuid)
        if row is None:
            return None
        row.plan_json = plan_json
        row.rendered_audio_path = key
        row.status = MixPlanStatus.ready
        row.error_text = None
        row.qa_metrics = {"flags": qa.flags, **qa.metrics}
        row.qa_verdict = qa.verdict
        if adopted_retry_nonce is not None:
            row.reroll_nonce = adopted_retry_nonce
        if outcome.source != "cached":
            row.plan_source = outcome.source
            row.style = outcome.style
            row.rationale = outcome.rationale
        db.commit()
    return mix_plan_id


def _mark_failed(plan_uuid: uuid.UUID, message: str) -> None:
    with SessionLocal() as db:
        row = db.get(MixPlan, plan_uuid)
        if row is None:
            return
        row.status = MixPlanStatus.failed
        row.error_text = message[:1000]  # cap for sanity
        db.commit()
