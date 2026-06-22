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
from app.services.mixer.plan import build_pair_plan
from app.services.mixer.pitch_resolver import effective_bundle
from app.services.mixer.planner_v2 import PlanOutcome, SongMeta, build_plan_v2
from app.services.mixer.preshift import ensure_pitched_inputs
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

        a_envelope_path = a_stems.vocal_envelope_path
        b_envelope_path = b_stems.vocal_envelope_path

        # `energy_curve` is sampled at 1Hz in the analyzer; we average it
        # per section so structure and energy arrive together.
        a_energy_curve = list(a_analysis.energy_curve or [])
        b_energy_curve = list(b_analysis.energy_curve or [])

        # Set-level pass suggestion (soft), user pin (hard), reroll nonce.
        style_hint = row.style_hint
        style_override = row.style_override
        reroll_nonce = row.reroll_nonce or 0

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

    async def _fetch_safe_regions(transcription, lyrics, envelope_path, duration):
        if not transcription or not envelope_path:
            return []
        try:
            envelope_data = await storage.read(envelope_path)
            envelope = json.loads(envelope_data.decode("utf-8"))
        except Exception:
            return []

        aligned_words = None
        if lyrics and lyrics.alignment_status == LyricsAlignmentStatus.success:
            aligned_words = lyrics.aligned_words

        return vocal_safe_regions(
            transcription_segments=transcription.segments,
            envelope=envelope,
            aligned_words=aligned_words,
            duration_seconds=duration or 0.0,
        )

    async def _build_plan() -> PlanOutcome:
        a_eff = effective_bundle(a_bundle, a_pitch_offset)
        b_eff = effective_bundle(b_bundle, b_pitch_offset)

        if not settings.use_llm_planner:
            return PlanOutcome(
                plan=build_pair_plan(a_eff, b_eff),
                source="deterministic", style=None, rationale=None,
            )

        a_regions = await _fetch_safe_regions(
            a_transcription, a_lyrics, a_envelope_path, a_bundle.duration
        )
        b_regions = await _fetch_safe_regions(
            b_transcription, b_lyrics, b_envelope_path, b_bundle.duration
        )
        provider = get_llm_provider()

        if settings.planner_version == "v2":
            return await build_plan_v2(
                provider,
                SongMeta(a_title, a_artist, a_bundle, a_energy_curve,
                         a_regions, pitch_offset=a_pitch_offset),
                SongMeta(b_title, b_artist, b_bundle, b_energy_curve,
                         b_regions, pitch_offset=b_pitch_offset),
                style_hint=style_hint,
                style_override=style_override,
                previous_styles=previous_styles,
                pair_label=pair_label,
                nonce=reroll_nonce,
                pitch_mode=settings.pitch_mode,
                loudness_match=settings.loudness_match,
            )

        # ---- legacy free-form path, with repair-not-reject ----
        a_llm_input = {
            "analysis": _bundle_to_legacy_llm_dict(
                a_eff, a_energy_curve, a_title, a_artist
            ),
            "vocal_safe_regions": a_regions,
        }
        b_llm_input = {
            "analysis": _bundle_to_legacy_llm_dict(
                b_eff, b_energy_curve, b_title, b_artist
            ),
            "vocal_safe_regions": b_regions,
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

    # Outside "temporary" mode no plan may carry pitch tools: whole-song
    # mode pre-shifts the audio itself (a leftover tool would double-
    # shift) and "off" mode accepts clashes (a leftover tool would
    # glide). Covers legacy-prompt output, deterministic fallbacks on a
    # clash, and cached pre-migration plans alike.
    plan_for_render = outcome.plan
    if settings.pitch_mode != "temporary":
        plan_for_render = strip_pitch_tools(plan_for_render)

    # Guarantee B's tempo/pitch revert only fires once the crossfade is
    # done, whatever the source of the plan (fresh, cached, or fallback).
    plan_json = enforce_revert_after_crossfade(plan_for_render, b_bundle)

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
