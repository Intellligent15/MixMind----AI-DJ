from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import Analysis, Song, SongStatus, Stems, Transcription
from app.schemas import (
    AnalysisRead,
    SongCreate,
    SongRead,
    StemsRead,
    TranscriptionRead,
    LyricsRead,
)
from app.services.storage import get_storage
from app.workers import (
    PRI_ANALYZE,
    PRI_DOWNLOAD,
    PRI_SEPARATE,
    PRI_TRANSCRIBE,
    PRI_TAG,
    celery_app,
)
from app.workers.analyze import analyze_song
from app.workers.download import download_song

# Dispatched via celery_app.send_task so the API container doesn't have to
# import app.workers.separate / app.workers.transcribe — those modules pull
# in torch + demucs + mlx-whisper, which only the native worker needs.
SEPARATE_TASK = "app.workers.separate.separate_stems"
TRANSCRIBE_TASK = "app.workers.transcribe.transcribe_song"
TAG_TASK = "tag_audio"
# Mirrors STEM_NAMES in app.services.stems (kept inline so the API doesn't
# import the stems service module either — same reason as above).
STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "other")

router = APIRouter(prefix="/api/songs", tags=["songs"])


def _touch_song(db: Session, song: Song) -> None:
    """Bump ``last_accessed_at`` to now — the LRU clock the cache evictor
    orders by. Called wherever a song's audio is actually served. Cheap and
    best-effort; a failure here must never break the audio response."""
    from datetime import datetime, timezone

    try:
        song.last_accessed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        db.rollback()


def _song_read_with_flags(
    song: Song, has_stems: bool, has_transcription: bool
) -> SongRead:
    sr = SongRead.model_validate(song)
    sr.has_stems = bool(has_stems)
    sr.has_transcription = bool(has_transcription)
    return sr


@router.get("", response_model=list[SongRead])
def list_songs(db: Session = Depends(get_db)) -> list[SongRead]:
    rows = db.execute(
        select(
            Song,
            exists().where(Stems.song_id == Song.id).label("has_stems"),
            exists()
            .where(Transcription.song_id == Song.id)
            .label("has_transcription"),
        ).order_by(Song.created_at.desc())
    ).all()
    return [_song_read_with_flags(s, hs, ht) for (s, hs, ht) in rows]


@router.post("", response_model=SongRead, status_code=status.HTTP_201_CREATED)
def create_song(payload: SongCreate, db: Session = Depends(get_db)) -> Song:
    existing = db.scalar(
        select(Song).where(Song.youtube_video_id == payload.youtube_video_id)
    )
    if existing is not None:
        return existing

    song = Song(
        youtube_video_id=payload.youtube_video_id,
        title=payload.title,
        artist=payload.artist,
        duration_seconds=payload.duration_seconds,
        thumbnail_url=payload.thumbnail_url,
        status=SongStatus.pending,
    )
    db.add(song)
    db.commit()
    db.refresh(song)
    # Library add: download only. pipeline_requested defaults to False
    # so the song stops at `downloaded` until the user queues it or
    # manually triggers a downstream stage.
    download_song.apply_async(args=[str(song.id)], priority=PRI_DOWNLOAD)
    celery_app.send_task("app.workers.fetch_lyrics.fetch_lyrics", args=[str(song.id)])
    return song


@router.get("/{song_id}", response_model=SongRead)
def get_song(song_id: uuid.UUID, db: Session = Depends(get_db)) -> SongRead:
    row = db.execute(
        select(
            Song,
            exists().where(Stems.song_id == Song.id).label("has_stems"),
            exists()
            .where(Transcription.song_id == Song.id)
            .label("has_transcription"),
        ).where(Song.id == song_id)
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="song not found")
    song, hs, ht = row
    return _song_read_with_flags(song, hs, ht)


@router.delete(
    "/{song_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_song(
    song_id: uuid.UUID, db: Session = Depends(get_db)
) -> Response:
    """Delete a song row + every blob it owns in object storage.

    Cascading FKs handle the DB side (analyses/stems/transcriptions/
    mix_plans/queue_items go with the song). For storage we enumerate
    the blobs explicitly: the song's audio, each stem WAV + vocal
    envelope, any rendered MixPlan WAVs for pairs touching this song,
    and the transcription JSON the Modal path uploads as a sidecar.

    Storage deletes are best-effort: a missing blob (or transient
    network blip) won't block the DB delete. Idempotent — a missing
    song row still 204s.
    """
    logger = logging.getLogger(__name__)
    song = db.get(Song, song_id)
    if song is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Same enumeration the LRU evictor uses, so manual delete + eviction free
    # the identical blob set (audio, stems, vocal envelope, transcription
    # sidecar, and per-pair renders touching this song).
    from app.services.cache.eviction import collect_song_storage_keys

    keys = collect_song_storage_keys(song, db)

    storage = get_storage()
    for key in keys:
        try:
            await storage.delete(key)
        except FileNotFoundError:
            # Already gone — nothing to clean up.
            pass
        except Exception:
            # Any other failure: log and move on so we don't strand
            # the DB row.
            logger.warning(
                "delete_song: failed to delete storage key %r", key,
                exc_info=True,
            )

    db.delete(song)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _parse_range_header(value: str | None, total: int | None = None) -> tuple[int | None, int | None]:
    """Parse a single 'bytes=lo-hi' Range header. Returns (lo, hi).

    We only support a single byte range — that's all browsers ever ask for
    in practice. `total` lets us resolve open-ended ranges (`bytes=0-`)
    when known; otherwise leave hi=None and let the storage backend
    interpret it. Returns (None, None) for an unparsable / missing header
    (caller falls back to full-body)."""
    if not value or not value.startswith("bytes="):
        return None, None
    spec = value[len("bytes="):].split(",", 1)[0].strip()
    if "-" not in spec:
        return None, None
    lo_s, hi_s = spec.split("-", 1)
    lo = int(lo_s) if lo_s else 0
    hi: int | None
    if hi_s:
        hi = int(hi_s)
    else:
        hi = (total - 1) if total is not None else None
    return lo, hi


async def _stream_audio_response(
    key: str,
    media_type: str,
    range_header: str | None,
    download_filename: str | None = None,
) -> StreamingResponse:
    """Shared streaming-with-Range body for the audio + stem endpoints.

    Streams bytes through the backend so the browser sees a same-origin
    response (CORS handled by the FastAPI middleware) — DO Spaces' own
    CORS on presigned URLs is unreliable, so this is the simpler path."""
    lo, hi = _parse_range_header(range_header)
    storage = get_storage()
    try:
        iterator, total_size, content_length = await storage.stream(
            key, start=lo, end=hi,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="audio missing in storage") from exc

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        # Stems and downloaded audio change only on re-separate / re-download.
        # 5 minutes is a sweet spot: long enough that the debug page round-trip
        # (waveform render → play → seek) doesn't re-hit storage, short enough
        # that a re-separate quickly becomes visible on the next visit.
        "Cache-Control": "private, max-age=300",
    }
    if download_filename:
        headers["Content-Disposition"] = f'attachment; filename="{download_filename}"'
        
    if lo is not None:
        # Partial response: include Content-Range and use 206.
        effective_hi = (hi if hi is not None else total_size - 1)
        headers["Content-Range"] = f"bytes {lo}-{effective_hi}/{total_size}"
        return StreamingResponse(
            iterator, status_code=206, media_type=media_type, headers=headers,
        )
    return StreamingResponse(iterator, media_type=media_type, headers=headers)


@router.get("/{song_id}/audio")
async def get_song_audio(
    song_id: uuid.UUID,
    db: Session = Depends(get_db),
    range: str | None = Header(default=None),
):
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status in (SongStatus.pending, SongStatus.downloading) or not song.audio_path:
        raise HTTPException(
            status_code=409,
            detail=f"audio not available (status={song.status.value})",
        )
    # Serving the audio == the song is being played: refresh its LRU clock so
    # the cache evictor reclaims genuinely-stale library content first.
    _touch_song(db, song)
    return await _stream_audio_response(song.audio_path, "audio/wav", range)


@router.post(
    "/{song_id}/analyze",
    response_model=SongRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_analyze_song(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Song:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status not in (SongStatus.downloaded, SongStatus.analyzed, SongStatus.failed):
        raise HTTPException(
            status_code=409,
            detail=f"song not ready for analysis (status={song.status.value})",
        )
    # User opted into downstream processing — flag carries forward so
    # the worker auto-chain continues into separate + transcribe.
    song.pipeline_requested = True
    db.commit()
    analyze_song.apply_async(args=[str(song.id)], priority=PRI_ANALYZE)
    return song


@router.post(
    "/{song_id}/retry",
    response_model=SongRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_song(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Song:
    """Re-dispatch a failed song from the most-advanced stage it hasn't yet
    cleared, so a transient worker error is one click to recover from.

    Picks the stage by which artifacts already exist (not by status), so a
    song that failed at transcription doesn't pay for a fresh download +
    Demucs run. Each worker's claim gate accepts `failed`, and the
    download→analyze→separate→transcribe auto-chain (gated on
    `pipeline_requested`) carries it the rest of the way — which also re-fires
    the eager-stitch hook when the queue's last song lands `ready`.
    """
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status != SongStatus.failed:
        raise HTTPException(
            status_code=409,
            detail=f"song is {song.status.value}, not failed — nothing to retry",
        )

    has_analysis = (
        db.scalar(select(Analysis.id).where(Analysis.song_id == song_id)) is not None
    )
    has_stems = (
        db.scalar(select(Stems.id).where(Stems.song_id == song_id)) is not None
    )
    has_transcription = (
        db.scalar(select(Transcription.id).where(Transcription.song_id == song_id))
        is not None
    )
    analysis = db.scalar(select(Analysis).where(Analysis.song_id == song_id))
    has_tags = analysis is not None and analysis.tags is not None

    song.pipeline_requested = True
    song.error_text = None
    db.commit()

    sid = str(song.id)
    if not song.audio_path:
        download_song.apply_async(args=[sid], priority=PRI_DOWNLOAD)
    elif not has_analysis:
        analyze_song.apply_async(args=[sid], priority=PRI_ANALYZE)
    else:
        if not has_tags:
            celery_app.send_task(TAG_TASK, args=[sid], priority=PRI_TAG)

        if not has_stems:
            celery_app.send_task(SEPARATE_TASK, args=[sid], priority=PRI_SEPARATE)
        elif not has_transcription:
            celery_app.send_task(TRANSCRIBE_TASK, args=[sid], priority=PRI_TRANSCRIBE)
        else:
            # Everything present but the song is failed — re-run the cheapest
            # terminal stage to settle it back to `ready`.
            celery_app.send_task(TRANSCRIBE_TASK, args=[sid], priority=PRI_TRANSCRIBE)
    return song


@router.get("/{song_id}/analysis", response_model=AnalysisRead)
def get_song_analysis(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Analysis:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    analysis = db.scalar(select(Analysis).where(Analysis.song_id == song_id))
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis not available")
    return analysis


# Same gate as the separate_stems worker's CLAIMABLE_STATUSES — keep these
# in sync; the API rejects early so the frontend gets a useful error
# instead of a silently-dropped task.
SEPARATE_GATE = (SongStatus.analyzed, SongStatus.ready, SongStatus.failed)


@router.post(
    "/{song_id}/separate",
    response_model=SongRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_separate_song(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Song:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status not in SEPARATE_GATE:
        raise HTTPException(
            status_code=409,
            detail=f"song not ready for separation (status={song.status.value})",
        )
    song.pipeline_requested = True
    db.commit()
    celery_app.send_task(SEPARATE_TASK, args=[str(song.id)], priority=PRI_SEPARATE)
    return song


@router.get("/{song_id}/stems", response_model=StemsRead)
def get_song_stems(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Stems:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    stems = db.scalar(select(Stems).where(Stems.song_id == song_id))
    if stems is None:
        raise HTTPException(status_code=404, detail="stems not available")
    return stems


@router.get("/{song_id}/stems/{stem_name}")
async def get_song_stem_audio(
    song_id: uuid.UUID,
    stem_name: str,
    db: Session = Depends(get_db),
    range: str | None = Header(default=None),
):
    if stem_name not in STEM_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"unknown stem (expected one of {list(STEM_NAMES)})",
        )
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    stems = db.scalar(select(Stems).where(Stems.song_id == song_id))
    if stems is None:
        raise HTTPException(status_code=404, detail="stems not available")

    key = {
        "vocals": stems.vocals_path,
        "drums": stems.drums_path,
        "bass": stems.bass_path,
        "other": stems.other_path,
    }[stem_name]
    if key is None:
        raise HTTPException(status_code=409, detail=f"{stem_name} stem not written yet")

    _touch_song(db, song)
    return await _stream_audio_response(key, "audio/wav", range)


# Same gate as the transcribe_song worker's CLAIMABLE_STATUSES — keep in
# sync. The API rejects early so the frontend gets a useful error instead
# of a silently-dropped task.
TRANSCRIBE_GATE = (SongStatus.analyzed, SongStatus.ready, SongStatus.failed)


@router.post(
    "/{song_id}/transcribe",
    response_model=SongRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_transcribe_song(
    song_id: uuid.UUID, db: Session = Depends(get_db)
) -> Song:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status not in TRANSCRIBE_GATE:
        raise HTTPException(
            status_code=409,
            detail=f"song not ready for transcription (status={song.status.value})",
        )
    # Stems must exist — the worker will fail loudly otherwise, but
    # rejecting up-front gives the frontend a usable error.
    stems = db.scalar(select(Stems).where(Stems.song_id == song_id))
    if stems is None or stems.vocals_path is None:
        raise HTTPException(
            status_code=409,
            detail="song has no separated vocal stem yet",
        )
    song.pipeline_requested = True
    db.commit()
    celery_app.send_task(TRANSCRIBE_TASK, args=[str(song.id)], priority=PRI_TRANSCRIBE)
    return song


@router.get("/{song_id}/transcription", response_model=TranscriptionRead)
def get_song_transcription(
    song_id: uuid.UUID, db: Session = Depends(get_db)
) -> Transcription:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    transcription = db.scalar(
        select(Transcription).where(Transcription.song_id == song_id)
    )
    if transcription is None:
        raise HTTPException(status_code=404, detail="transcription not available")
    return transcription


@router.get("/{song_id}/lyrics", response_model=LyricsRead)
def get_song_lyrics(
    song_id: uuid.UUID, db: Session = Depends(get_db)
):
    from app.models.lyrics import Lyrics
    
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
    if lyrics is None:
        raise HTTPException(status_code=404, detail="lyrics not available")
    return lyrics


@router.get("/{song_id}/vocal_safe_regions")
async def get_song_vocal_safe_regions(
    song_id: uuid.UUID,
    db: Session = Depends(get_db),
    word_prob_min: float = 0.35,
    segment_logprob_min: float = -1.2,
    stem_rms_presence: float = 0.02,
    stem_rms_quiet: float = 0.01,
    min_safe_region_seconds: float = 6.0,
):
    import json
    from app.models.lyrics import Lyrics
    from app.services.vocal_safety.safety import vocal_safe_regions
    
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
        
    transcription = db.scalar(select(Transcription).where(Transcription.song_id == song_id))
    stems = db.scalar(select(Stems).where(Stems.song_id == song_id))
    lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
    
    if not transcription or not stems or not stems.envelopes_path:
        raise HTTPException(status_code=409, detail="song not fully processed yet")
        
    storage = get_storage()
    try:
        envelope_data = await storage.read(stems.envelopes_path)
        envelope = json.loads(envelope_data)
    except Exception as exc:
        logging.getLogger(__name__).exception(
            "vocal_safe_regions: failed to read envelope at %s",
            stems.envelopes_path,
        )
        raise HTTPException(
            status_code=500,
            detail=f"failed to read vocal envelope: {type(exc).__name__}",
        )
        
    from app.models.lyrics import LyricsAlignmentStatus
    aligned_words = None
    if lyrics and lyrics.alignment_status == LyricsAlignmentStatus.success:
        aligned_words = lyrics.aligned_words
    
    regions = vocal_safe_regions(
        transcription_segments=transcription.segments,
        envelope=envelope,
        aligned_words=aligned_words,
        word_prob_min=word_prob_min,
        segment_logprob_min=segment_logprob_min,
        stem_rms_presence=stem_rms_presence,
        stem_rms_quiet=stem_rms_quiet,
        min_safe_region_seconds=min_safe_region_seconds,
        duration_seconds=song.duration_seconds or 0.0,
    )
    
    return {"regions": regions}
