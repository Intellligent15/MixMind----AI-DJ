from __future__ import annotations

import logging
import uuid

from sqlalchemy import update

from app.core.db import SessionLocal
from app.models import Song, SongStatus
from app.services.storage import get_storage
from app.services.youtube import YouTubeService
from app.workers import PRI_ANALYZE, celery_app
from app.workers.analyze import analyze_song

logger = logging.getLogger(__name__)

# Statuses we can transition into `downloading`. A song already mid-download
# or already downloaded is left alone — preventing two parallel tasks from
# racing on the same audio file.
CLAIMABLE_STATUSES = (SongStatus.pending, SongStatus.failed)

# YouTube flakes routinely: signed googlevideo URLs expire between extract
# and download (403), and per-IP throttling comes and goes (429). Those
# succeed on a retry seconds later. Genuinely permanent failures (private,
# removed, age-gated) must fail fast instead of burning retries.
TRANSIENT_ERROR_MARKERS = (
    "403", "429", "timed out", "timeout", "connection", "temporarily",
    "temporary", "reset by peer", "incomplete read", "500", "502", "503",
)
PERMANENT_ERROR_MARKERS = (
    "private video", "video unavailable", "removed", "age", "copyright",
    "not available in your country",
)
MAX_DOWNLOAD_RETRIES = 3
RETRY_BASE_COUNTDOWN_S = 15


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(m in msg for m in PERMANENT_ERROR_MARKERS):
        return False
    return any(m in msg for m in TRANSIENT_ERROR_MARKERS)


@celery_app.task(
    name="app.workers.download.download_song",
    bind=True,
    max_retries=MAX_DOWNLOAD_RETRIES,
)
def download_song(self, song_id: str) -> str | None:
    """Download the audio for a Song to local storage.

    Idempotent under concurrent dispatch: the pending/failed -> downloading
    transition is an atomic SQL UPDATE that only one task can win. Losers
    log and return the existing audio_path (which may be None if another
    task is mid-download). The winner proceeds to yt-dlp.

    Transitions: pending/failed -> downloading -> downloaded (or failed).
    Returns the storage key on success, or None if the call was a no-op
    because another dispatch already owned the download.
    """
    song_uuid = uuid.UUID(song_id)
    storage = get_storage()
    yt = YouTubeService()

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        if song is None:
            # Can happen if a song was deleted between dispatch and pickup,
            # or if stale tasks survive in the broker after a DB reset.
            logger.warning("download_song: song %s not found, skipping", song_id)
            return None

        claim = db.execute(
            update(Song)
            .where(Song.id == song_uuid)
            .where(Song.status.in_(CLAIMABLE_STATUSES))
            .values(status=SongStatus.downloading, error_text=None)
        )
        db.commit()

        if claim.rowcount == 0:
            db.refresh(song)
            logger.info(
                "download_song: %s already %s, skipping duplicate dispatch",
                song_id,
                song.status.value,
            )
            return song.audio_path

        video_id = song.youtube_video_id

    import tempfile
    import asyncio
    from pathlib import Path

    key = f"audio/{video_id}.wav"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / f"{video_id}.wav"
        try:
            yt.download(video_id, dest)
            asyncio.run(storage.upload_file(dest, key))
        except Exception as exc:
            retries_left = self.max_retries - self.request.retries
            if _is_transient(exc) and retries_left > 0:
                # Reset to `pending` so the retried task can win the
                # pending/failed -> downloading claim again (it would
                # no-op against a row stuck in `downloading`).
                logger.warning(
                    "download_song: transient failure for %s (%s); "
                    "retry %d/%d",
                    video_id, exc,
                    self.request.retries + 1, self.max_retries,
                )
                with SessionLocal() as db:
                    song = db.get(Song, song_uuid)
                    if song is not None:
                        song.status = SongStatus.pending
                        song.error_text = None
                        db.commit()
                raise self.retry(
                    exc=exc,
                    countdown=RETRY_BASE_COUNTDOWN_S * (self.request.retries + 1),
                )
            logger.exception("download failed for %s", video_id)
            with SessionLocal() as db:
                song = db.get(Song, song_uuid)
                if song is not None:
                    song.status = SongStatus.failed
                    song.error_text = f"download failed: {exc}"[:1000]
                    db.commit()
            raise

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        assert song is not None
        # Store the logical storage key, not the absolute filesystem path.
        # The host worker and the dockerized backend resolve this key
        # against different roots (./cache vs /app/cache).
        song.audio_path = key
        song.status = SongStatus.downloaded
        db.commit()
        wants_pipeline = bool(song.pipeline_requested)

    # Auto-chain into analyze IFF the user signaled they want downstream
    # processing — by locking a queue with this song or by hitting one
    # of the manual /analyze /separate /transcribe endpoints. Library-
    # added songs that nobody queued stop here.
    if wants_pipeline:
        analyze_song.apply_async(args=[song_id], priority=PRI_ANALYZE)

    # A fresh download is the moment the cache grows from user action — kick
    # the LRU evictor (no-op under budget). By NAME so this stays decoupled.
    try:
        celery_app.send_task("app.workers.evict_cache.enforce_cache_budget")
    except Exception:  # pragma: no cover - broker hiccup must not fail the DL
        logger.warning("download_song: failed to dispatch cache eviction", exc_info=True)

    return key
