"""LRU cache eviction (Phase 11).

The generated-artifact cache (audio/, stems/, mixes/, queue_mixes/,
mix_plan_logs/) grows unbounded as the user searches, queues, and renders.
This caps it at ``cache_max_size_gb`` by evicting **Songs**, least-recently
accessed first — the spec's model: ``Song.last_accessed_at`` drives LRU, and
a Song owns its audio + stems (+ derived rows + per-pair renders).

Hazards this guards against (an over-eager evictor is the main failure mode):

* **Never evict an in-use song.** A song referenced by ANY ``QueueItem``
  (the queue being built OR a locked/playing queue) is exempt. A song
  mid-pipeline (its files are being written this moment) is exempt.
* **Never evict the LLM plan cache.** ``mix_plan_logs/`` holds the cached
  Gemini/Groq mix plans, keyed by input hash, not owned by any song.
  Deleting one forces a costly LLM re-call on the next render. Those keys are
  counted toward the total (honest accounting) but never deleted.

Enumeration is done through ``StorageBackend.list_objects`` so it works
identically for the local filesystem and the future S3 backend — one listing
yields both the cache total and every blob's size.
"""

from __future__ import annotations

import logging

from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session

from app.models import MixPlan, QueueItem, QueueRender, Song, SongStatus, Stems
from app.services.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Prefixes the evictor will never delete from, no matter how stale.
PROTECTED_PREFIXES: tuple[str, ...] = ("mix_plan_logs/",)

# Prefixes the orphan sweep is allowed to delete from. Keys outside
# these (e.g. a prefix added by future code before this list is
# updated) are left alone rather than treated as orphans.
SWEPT_PREFIXES: tuple[str, ...] = (
    "audio/",
    "stems/",
    "stems_pitched/",
    "mixes/",
    "queue_mixes/",
    "transcriptions/",
)

# Stem names mirrored from app.services.mixer.preshift (kept inline so
# the evictor doesn't import the mixer stack).
PITCHED_STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "other")

# Song states where files are actively being written — exempt so a sweep
# can't yank an audio/stem out from under a running worker.
ACTIVE_STATUSES: tuple[SongStatus, ...] = (
    SongStatus.pending,
    SongStatus.downloading,
    SongStatus.analyzing,
    SongStatus.separating,
    SongStatus.transcribing,
)


def collect_song_storage_keys(song: Song, db: Session) -> list[str]:
    """Every storage blob a Song owns.

    Its downloaded audio, each stem WAV + the vocal-envelope sidecar, the
    Modal transcription sidecar JSON, and any per-pair ``MixPlan`` renders for
    transitions touching this song. Mirrors the explicit enumeration in the
    ``DELETE /api/songs/{id}`` route so eviction and manual delete free the
    same set.
    """
    keys: list[str] = []
    if song.audio_path:
        keys.append(song.audio_path)

    stems = db.scalar(select(Stems).where(Stems.song_id == song.id))
    if stems is not None:
        for k in (
            stems.vocals_path,
            stems.drums_path,
            stems.bass_path,
            stems.other_path,
            stems.envelopes_path,
        ):
            if k:
                keys.append(k)

    # Local mlx-whisper path doesn't write this; the Modal path does. delete
    # is idempotent on a miss, so listing it unconditionally is free.
    keys.append(f"transcriptions/{song.youtube_video_id}.json")

    plans = db.scalars(
        select(MixPlan).where(
            or_(MixPlan.from_song_id == song.id, MixPlan.to_song_id == song.id)
        )
    ).all()
    for plan in plans:
        if plan.rendered_audio_path:
            keys.append(plan.rendered_audio_path)

    return keys


def evictable_songs(db: Session) -> list[Song]:
    """Songs eligible for eviction, least-recently-accessed first.

    Excluded: songs in ANY queue (``QueueItem`` exists) and songs mid-pipeline
    (status in :data:`ACTIVE_STATUSES`). What's left is orphaned library
    content — exactly what LRU should reclaim.
    """
    in_queue = exists().where(QueueItem.song_id == Song.id)
    return list(
        db.scalars(
            select(Song)
            .where(~in_queue)
            .where(Song.status.notin_(ACTIVE_STATUSES))
            .order_by(Song.last_accessed_at.asc())
        ).all()
    )


def _is_protected(key: str) -> bool:
    return any(key.startswith(p) for p in PROTECTED_PREFIXES)


async def enforce_cache_budget(
    db: Session, storage: StorageBackend, budget_bytes: int
) -> dict:
    """Evict least-recently-accessed Songs until the cache is under budget.

    Returns a summary dict (footprint before/after, freed bytes, evicted song
    ids) for logging / the manual endpoint. No-op (and cheap) when already
    under budget. Storage deletes are best-effort; a missing/erroring blob
    never strands the DB delete.
    """
    objects = await storage.list_objects("")
    sizes: dict[str, int] = {k: s for k, s in objects}
    total = sum(sizes.values())

    result: dict = {
        "total_before": total,
        "budget": budget_bytes,
        "evicted": [],
        "orphans_deleted": 0,
        "freed": 0,
        "total_after": total,
    }

    # Step 1: Mark-and-Sweep Garbage Collection
    # Collect all deterministic keys that *could* exist for active DB rows.
    # Anything in storage not matching these keys is an orphan and can be deleted.
    valid_keys: set[str] = set()
    for song_id, video_id in db.execute(select(Song.id, Song.youtube_video_id)):
        valid_keys.add(f"audio/{video_id}.wav")
        valid_keys.add(f"audio/{video_id}.opus")
        valid_keys.add(f"audio/{video_id}.m4a")
        valid_keys.add(f"transcriptions/{video_id}.json")
        valid_keys.add(f"stems/{video_id}/vocals.wav")
        valid_keys.add(f"stems/{video_id}/drums.wav")
        valid_keys.add(f"stems/{video_id}/bass.wav")
        valid_keys.add(f"stems/{video_id}/other.wav")
        valid_keys.add(f"stems/{video_id}/envelopes.json")
    for mix_plan_id in db.scalars(select(MixPlan.id)):
        valid_keys.add(f"mixes/{mix_plan_id}.wav")
        valid_keys.add(f"mixes/{mix_plan_id}.flac")
    for queue_id in db.scalars(select(QueueRender.queue_id)):
        valid_keys.add(f"queue_mixes/{queue_id}.m4a")
        valid_keys.add(f"queue_mixes/{queue_id}.flac")
        valid_keys.add(f"queue_mixes/{queue_id}.wav")
    # Whole-song pitch-shifted stems (preshift.py cache). Keyed by the
    # offset the set-level resolver assigned, so stale offsets from a
    # re-resolved queue age out as orphans while live ones survive.
    for song_id, offset in db.execute(
        select(QueueItem.song_id, QueueItem.pitch_offset_semitones)
    ):
        if offset:
            for name in PITCHED_STEM_NAMES:
                valid_keys.add(f"stems_pitched/{song_id}/{offset:+d}/{name}.wav")

    for key, size in list(sizes.items()):
        if (
            key.startswith(SWEPT_PREFIXES)
            and not _is_protected(key)
            and key not in valid_keys
        ):
            try:
                await storage.delete(key)
                total -= size
                result["freed"] += size
                result["orphans_deleted"] += 1
                del sizes[key]
            except Exception:
                logger.warning(
                    "enforce_cache_budget: failed to delete orphan %r", key, exc_info=True
                )
    if total <= budget_bytes:
        return result

    for song in evictable_songs(db):
        keys = collect_song_storage_keys(song, db)
        freed = sum(sizes.get(k, 0) for k in keys)
        song_id = str(song.id)

        for key in keys:
            if _is_protected(key):
                continue
            try:
                await storage.delete(key)
            except FileNotFoundError:
                pass
            except Exception:
                logger.warning(
                    "enforce_cache_budget: failed to delete %r", key,
                    exc_info=True,
                )

        # Cascading FKs drop analyses/stems/transcriptions/lyrics/mix_plans/
        # queue_items with the song.
        db.delete(song)
        db.commit()

        total -= freed
        result["evicted"].append(song_id)
        result["freed"] += freed
        if total <= budget_bytes:
            break

    result["total_after"] = total
    logger.info(
        "enforce_cache_budget: evicted %d song(s), freed %d bytes "
        "(%d -> %d, budget %d)",
        len(result["evicted"]),
        result["freed"],
        result["total_before"],
        total,
        budget_bytes,
    )
    return result
