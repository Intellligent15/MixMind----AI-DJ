from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy import update

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import Song, SongStatus, Stems, StemsStatus, Transcription
from app.services.storage import get_storage
from app.workers import PRI_TRANSCRIBE, celery_app

logger = logging.getLogger(__name__)

# Statuses we can transition into `separating`. analyzed/ready/failed all
# legitimately want a (re-)separation; downloaded is too early (analysis
# hasn't run yet); pending/downloading/analyzing/separating/transcribing
# are mid-flight and would clobber a winning task.
CLAIMABLE_STATUSES = (SongStatus.analyzed, SongStatus.ready, SongStatus.failed)

# Mirrors STEM_NAMES in app.services.stems. Duplicated here so this module
# can be imported without pulling in torch/demucs (the slim worker image
# on the droplet doesn't ship those — heavy ML goes via Modal).
STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "other")

# Bind StemSeparationService at module top *if* the local stack is installed.
# Same dual-context story as transcribe.py: lets tests `patch(
# "app.workers.separate.StemSeparationService")` on macOS, gracefully
# falls back to None on slim Linux containers without torch+demucs.
try:
    from app.services.stems import StemSeparationService  # noqa: F401
except ImportError:
    StemSeparationService = None  # type: ignore[assignment]


def _stem_key(video_id: str, stem: str) -> str:
    return f"stems/{video_id}/{stem}.wav"


def _envelope_key(video_id: str) -> str:
    return f"stems/{video_id}/envelopes.json"


def _separate_via_modal(audio_key: str, video_id: str) -> dict:
    """Dispatch separation to the deployed Modal app. Pass credentials
    explicitly so the Modal side doesn't need its own `.env` or Secret."""
    import modal

    fn = modal.Function.from_name("ai-dj-gpu-workers", "run_separation")
    return fn.remote(
        audio_key,
        video_id,
        settings.s3_endpoint_url,
        settings.s3_bucket_name,
        settings.s3_access_key,
        settings.s3_secret_key,
        settings.s3_region_name,
    )


@celery_app.task(name="app.workers.separate.separate_stems")
def separate_stems(song_id: str) -> str | None:
    """Run Demucs htdemucs over the song's audio and persist four stems.

    Idempotent under concurrent dispatch via the same atomic-claim pattern
    as download_song / analyze_song: the analyzed/ready/failed -> separating
    transition is a single UPDATE; losers log and return.

    Transitions: analyzed/ready/failed -> separating -> analyzed (or failed).
    Re-running on a song with an existing Stems row deletes the old row and
    its on-disk files before writing the new ones.

    Note we exit at `analyzed` not `separated` — `separated` lives on the
    Stems row, not the Song row. The Song status reflects "what's the most
    advanced pipeline stage the song has fully cleared"; until Phase 6
    transcription lands, analyzed is still the terminal Song status.

    When MODAL_TOKEN_ID is set, the heavy lift goes through Modal's GPUs
    (the deployed `run_separation` function). Otherwise the worker runs
    locally — which only works on a host with MPS/CUDA and torch+demucs.
    """
    song_uuid = uuid.UUID(song_id)
    storage = get_storage()

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        if song is None:
            logger.warning("separate_stems: song %s not found, skipping", song_id)
            return None
        if not song.audio_path:
            logger.warning(
                "separate_stems: song %s has no audio_path yet, skipping", song_id
            )
            return None

        claim = db.execute(
            update(Song)
            .where(Song.id == song_uuid)
            .where(Song.status.in_(CLAIMABLE_STATUSES))
            .values(status=SongStatus.separating, error_text=None)
        )
        db.commit()

        if claim.rowcount == 0:
            db.refresh(song)
            logger.info(
                "separate_stems: %s already %s, skipping duplicate dispatch",
                song_id,
                song.status.value,
            )
            return None

        audio_key = song.audio_path
        video_id = song.youtube_video_id

    import tempfile
    import asyncio
    from pathlib import Path

    use_modal = bool(settings.modal_token_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"

        try:
            if use_modal:
                result = _separate_via_modal(audio_key, video_id)
                keys = {
                    "vocals": result["vocals_path"],
                    "drums": result["drums_path"],
                    "bass": result["bass_path"],
                    "other": result["other_path"],
                }
                envelope_key = result["envelopes_path"]
                vocal_rms = result["vocal_rms"]
                model_name = result["model_name"]
            else:
                if StemSeparationService is None:
                    raise RuntimeError(
                        "separate_stems: local Demucs unavailable "
                        "(torch+demucs not installed) and MODAL_TOKEN_ID unset"
                    )
                service = StemSeparationService()
                asyncio.run(storage.download_file(audio_key, audio_path))
                result = service.separate(audio_path)
                keys = {}
                for stem_name in STEM_NAMES:
                    key = _stem_key(video_id, stem_name)
                    dest = Path(tmpdir) / f"{stem_name}.wav"
                    service.write_stem(result.stems[stem_name], result.sample_rate, dest)
                    asyncio.run(storage.upload_file(dest, key))
                    keys[stem_name] = key

                envelope_key = _envelope_key(video_id)
                envelope_dest = Path(tmpdir) / "envelopes.json"
                envelope_dest.write_text(json.dumps(result.envelopes))
                asyncio.run(storage.upload_file(envelope_dest, envelope_key))
                vocal_rms = result.vocal_rms
                model_name = service.model_name

        except Exception as exc:
            logger.exception("separation failed for song %s", song_id)
            with SessionLocal() as db:
                song = db.get(Song, song_uuid)
                if song is not None:
                    song.status = SongStatus.failed
                    song.error_text = f"separation failed: {exc}"[:1000]
                    db.commit()
            raise

    with SessionLocal() as db:
        existing_stems = db.query(Stems).filter(Stems.song_id == song_uuid).one_or_none()
        if existing_stems is not None:
            db.delete(existing_stems)
        
        existing_trans = db.query(Transcription).filter(Transcription.song_id == song_uuid).one_or_none()
        if existing_trans is not None:
            db.delete(existing_trans)
            
        db.flush()
        stems_row = Stems(
            song_id=song_uuid,
            model_name=model_name,
            status=StemsStatus.separated,
            vocals_path=keys["vocals"],
            drums_path=keys["drums"],
            bass_path=keys["bass"],
            other_path=keys["other"],
            vocal_rms=vocal_rms,
            envelopes_path=envelope_key,
        )
        db.add(stems_row)
        song = db.get(Song, song_uuid)
        assert song is not None
        # Bounce back to `analyzed` — that's still the terminal Song status
        # until Phase 6 lands transcription and Phase 7+ lands `ready`.
        song.status = SongStatus.analyzed
        db.commit()
        wants_pipeline = bool(song.pipeline_requested)

    # Auto-chain into transcription IFF the user wants the full pipeline.
    # Dispatch via send_task by name so this module doesn't need to import
    # the transcribe worker (which pulls in mlx-whisper on environments
    # that don't have it). transcribe_song's atomic claim makes duplicate
    # dispatch a no-op.
    if wants_pipeline:
        celery_app.send_task(
            "app.workers.transcribe.transcribe_song",
            args=[str(song_uuid)],
            priority=PRI_TRANSCRIBE,
        )

    return str(song_uuid)
