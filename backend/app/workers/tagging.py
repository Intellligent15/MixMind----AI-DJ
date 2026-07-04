"""Worker for running Essentia ML tagging on Modal.

Tagging is a nice-to-have enrichment: the planner reads
``Analysis.tags`` when present but degrades gracefully without it, so
this task never touches ``Song.status`` — a song whose stems and
transcription landed stays ``ready`` even if tagging gives up.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.workers import celery_app
from app.core.config import settings
from app.core.db import SessionLocal
from app.models.analysis import Analysis
from app.models.song import Song

logger = logging.getLogger(__name__)


@celery_app.task(
    name="tag_audio",
    bind=True,
    max_retries=3,
)
def tag_audio_task(self: Any, song_id: str) -> None:
    """Run Essentia tagging via Modal and store the result on Analysis."""
    logger.info("Starting tag_audio for song_id=%s", song_id)
    song_uuid = uuid.UUID(song_id)

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        if not song:
            logger.error("tag_audio failed: Song %s not found", song_id)
            return

        if not song.audio_path:
            logger.error("tag_audio failed: Song %s has no audio_path", song_id)
            return

        analysis = db.query(Analysis).filter(Analysis.song_id == song_uuid).first()
        if not analysis:
            logger.error("tag_audio failed: Song %s has no analysis", song_id)
            return

        audio_key = song.audio_path

    try:
        import modal
        from app.workers.modal_stubs import APP_NAME

        f = modal.Function.from_name(APP_NAME, "tag_audio")
        result = f.remote(
            audio_key=audio_key,
            s3_endpoint=settings.s3_endpoint_url,
            s3_bucket=settings.s3_bucket_name,
            s3_access=settings.s3_access_key,
            s3_secret=settings.s3_secret_key,
            s3_region=settings.s3_region_name,
        )
    except Exception as e:
        logger.exception("Modal GPU tagging failed for %s", song_id)
        if self.request.retries >= self.max_retries:
            # Give up quietly: tags stay null and the pipeline proceeds.
            logger.error(
                "tag_audio giving up for song_id=%s after %d retries: %s",
                song_id, self.request.retries, e,
            )
            return
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))

    with SessionLocal() as db:
        analysis = db.query(Analysis).filter(Analysis.song_id == song_uuid).first()
        if not analysis:
            return

        analysis.tags = result
        db.commit()

    logger.info("tag_audio complete for song_id=%s: %s", song_id, result)
