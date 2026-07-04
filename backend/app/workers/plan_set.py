"""Set-level planning pass.

The v1 prompt demanded "variety across the set" while each per-pair call
saw exactly one pair — the model had no idea what the previous
transition was. This task runs ONCE per locked queue, before any
per-pair render: a single LLM call sees the whole ordered queue
(titles, artists, BPMs, keys, energy shape) and assigns each adjacent
pair a suggested transition style plus the set's energy arc. The
suggestion lands in MixPlan.style_hint; the per-pair planner treats it
as a strong default and the user's style_override (if any) still wins.

Failure-tolerant by design: any error logs and returns — per-pair
planning works fine without hints, just with less set-level coherence.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import Analysis, MixPlan, Queue, QueueItem, Song
from app.services.llm import get_llm_provider
from app.services.llm.prompts import SET_PLAN_SYSTEM_PROMPT, set_plan_user_prompt
from app.services.mixer.decision import TransitionStyle
from app.services.mixer.pitch_resolver import SongKey, resolve_pitch_offsets
from app.workers import celery_app

logger = logging.getLogger(__name__)


def _peak_energy_position(energy_curve: list[float]) -> float | None:
    if not energy_curve:
        return None
    peak_idx = max(range(len(energy_curve)), key=lambda i: energy_curve[i])
    return round(peak_idx / max(1, len(energy_curve) - 1), 2)


@celery_app.task(name="app.workers.plan_set.plan_set")
def plan_set(queue_id: str) -> str | None:
    """Set-level pre-render pass: (1) resolve whole-song pitch offsets,
    (2) ask the LLM for per-pair style hints. Each half is independently
    skippable and failure-tolerant — renders proceed regardless."""
    queue_uuid = uuid.UUID(queue_id)

    if settings.pitch_mode == "whole_song":
        try:
            _resolve_queue_pitch_offsets(queue_uuid)
        except Exception as exc:  # never block the render chord on this
            logger.error(
                "plan_set: pitch resolution failed for queue %s: %s",
                queue_id, exc,
            )

    if not settings.use_llm_planner or settings.planner_version != "v2":
        return None
    try:
        return _plan_set_inner(queue_uuid)
    except Exception as exc:  # never block the render chord on this
        logger.error("plan_set: failed for queue %s: %s", queue_id, exc)
        return None


def _resolve_queue_pitch_offsets(queue_uuid: uuid.UUID) -> None:
    """Deterministic greedy walk over the locked queue; persists one
    whole-song offset per QueueItem. Runs before any pair render (this
    task is chained ahead of the render chord), so every render of a
    song sees the same offset and the stitch junctions line up. Re-runs
    (e.g. after a re-roll resets the stitch) recompute the same values —
    keys don't change — so it's naturally idempotent."""
    with SessionLocal() as db:
        queue = db.get(Queue, queue_uuid)
        if queue is None or not queue.locked:
            return
        items = sorted(queue.items, key=lambda it: it.position)
        if len(items) < 2:
            return

        keys: list[SongKey] = []
        for item in items:
            analysis = db.scalar(
                select(Analysis).where(Analysis.song_id == item.song_id)
            )
            if analysis is None:
                logger.info(
                    "plan_set: song %s not analyzed; skipping pitch pass",
                    item.song_id,
                )
                return
            keys.append(SongKey(key=analysis.key, camelot_key=analysis.camelot_key))

        offsets = resolve_pitch_offsets(keys)
        changed = 0
        for item, offset in zip(items, offsets):
            row = db.get(QueueItem, item.id)
            if row is not None and row.pitch_offset_semitones != offset:
                row.pitch_offset_semitones = offset
                changed += 1
        db.commit()
        if any(offsets):
            logger.info(
                "plan_set: queue %s whole-song pitch offsets %s (%d updated)",
                queue_uuid, offsets, changed,
            )


def _plan_set_inner(queue_uuid: uuid.UUID) -> str | None:
    with SessionLocal() as db:
        queue = db.get(Queue, queue_uuid)
        if queue is None or not queue.locked:
            return None
        items = sorted(queue.items, key=lambda it: it.position)
        if len(items) < 2:
            return None
        queue_occasion = queue.occasion
        queue_vibe_note = queue.vibe_note
        queue_arc_template = queue.arc_template

        songs_payload: list[dict] = []
        for idx, item in enumerate(items):
            song = db.get(Song, item.song_id)
            analysis = db.scalar(
                select(Analysis).where(Analysis.song_id == item.song_id)
            )
            if song is None or analysis is None:
                logger.info("plan_set: song %s not analyzed yet; skipping pass",
                            item.song_id)
                return None
            songs_payload.append({
                "index": idx,
                "title": song.title,
                "artist": song.artist,
                "bpm": analysis.bpm,
                "key": analysis.key,
                "camelot_key": analysis.camelot_key,
                "duration": round(song.duration_seconds or 0.0, 1),
                "peak_energy_position": _peak_energy_position(
                    list(analysis.energy_curve or [])
                ),
            })

        plans = db.scalars(
            select(MixPlan).where(MixPlan.queue_id == queue_uuid)
        ).all()
        song_order = {item.song_id: idx for idx, item in enumerate(items)}
        plan_by_pair_index = {
            song_order[p.from_song_id]: p
            for p in plans
            if p.from_song_id in song_order
        }

    # Occasion / arc / vibe context (F7 + F10): plain-language guidance
    # plus per-pair energy targets interpolated from the arc template.
    from app.services.mixer.occasions import (
        OCCASIONS,
        arc_targets_for_pairs,
        occasion_context,
    )

    set_context = occasion_context(queue_occasion, queue_vibe_note)
    try:
        from app.services.feedback.summary import feedback_context

        with SessionLocal() as db:
            fb_line = feedback_context(db)
        if fb_line:
            set_context["listener_feedback"] = fb_line
    except Exception:  # feedback must never block set planning
        logger.warning("plan_set: feedback summary failed", exc_info=True)
    arc_id = queue_arc_template or (
        OCCASIONS[queue_occasion].default_arc
        if queue_occasion in OCCASIONS else None
    )
    if arc_id:
        targets = arc_targets_for_pairs(arc_id, len(songs_payload) - 1)
        if targets:
            set_context["energy_targets"] = (
                f"arc '{arc_id}': per-pair target energies {targets} — "
                "assign styles so each transition lands near its target "
                "(high target = energetic styles, low = gentle ones)"
            )

    provider = get_llm_provider()
    obj = asyncio.run(
        provider.complete_json(
            system=SET_PLAN_SYSTEM_PROMPT,
            user=set_plan_user_prompt(songs_payload, set_context or None),
            cache_namespace="set_plan_logs",
        )
    )
    if not isinstance(obj, dict) or not isinstance(obj.get("pairs"), list):
        logger.warning("plan_set: malformed response, skipping hints")
        return None

    legal = {s.value for s in TransitionStyle}
    hints: dict[int, str] = {}
    for entry in obj["pairs"]:
        if not isinstance(entry, dict):
            continue
        idx, style = entry.get("index"), entry.get("style")
        if isinstance(idx, int) and isinstance(style, str) and style in legal:
            hints[idx] = style

    if not hints:
        return None

    with SessionLocal() as db:
        wrote = 0
        for idx, style in hints.items():
            stale = plan_by_pair_index.get(idx)
            if stale is None:
                continue
            row = db.get(MixPlan, stale.id)
            if row is None:
                continue
            row.style_hint = style
            wrote += 1
        db.commit()
    arc = obj.get("arc")
    logger.info(
        "plan_set: wrote %d style hints for queue %s (arc: %s)",
        wrote, queue_uuid, arc,
    )
    return str(queue_uuid)
