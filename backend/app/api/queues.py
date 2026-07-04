from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import (
    MixPlan,
    Queue,
    QueueItem,
    Song,
    SongStatus,
    Stems,
    Transcription,
)
from app.schemas import (
    QueueContextUpdate,
    QueueItemAdd,
    QueueRead,
    QueueRenderRead,
    QueueReorder,
)
from app.workers import (
    PRI_ANALYZE,
    PRI_DOWNLOAD,
    PRI_SEPARATE,
    PRI_TRANSCRIBE,
    celery_app,
)
from app.workers.analyze import analyze_song
from app.workers.download import download_song

# Dispatched by task name so the API container doesn't have to import
# app.workers.separate / app.workers.transcribe (which pull torch + demucs
# + mlx-whisper — native-worker only).
SEPARATE_TASK = "app.workers.separate.separate_stems"
TRANSCRIBE_TASK = "app.workers.transcribe.transcribe_song"


class _TaskShim:
    """Adapter exposing the .s/.si/.delay surface tests + chain composition
    expect, without importing the underlying ML-bound modules."""

    def __init__(self, task_name: str) -> None:
        self._task_name = task_name

    def s(self, *args):
        return celery_app.signature(self._task_name, args=args)

    def si(self, *args):
        return celery_app.signature(self._task_name, args=args, immutable=True)

    def delay(self, *args):
        return celery_app.send_task(self._task_name, args=list(args))

    def apply_async(self, args=None, priority=None, **kwargs):
        send_kwargs = {"args": list(args or []), **kwargs}
        if priority is not None:
            send_kwargs["priority"] = priority
        return celery_app.send_task(self._task_name, **send_kwargs)


separate_stems = _TaskShim(SEPARATE_TASK)
transcribe_song = _TaskShim(TRANSCRIBE_TASK)

router = APIRouter(prefix="/api/queues", tags=["queues"])


QUEUE_CAP = 20


def _compact_positions(db: Session, queue_id: uuid.UUID) -> None:
    """Re-pack item positions to be 0..N-1 in their current order."""
    items = list(
        db.scalars(
            select(QueueItem)
            .where(QueueItem.queue_id == queue_id)
            .order_by(QueueItem.position)
        ).all()
    )
    # Two passes to avoid colliding with the (queue_id, position) unique
    # constraint while we shuffle.
    for idx, item in enumerate(items):
        item.position = -1000 - idx
    db.flush()
    for idx, item in enumerate(items):
        item.position = idx
    db.flush()


@router.post("", response_model=QueueRead, status_code=status.HTTP_201_CREATED)
async def create_queue(db: Session = Depends(get_db)) -> Queue:
    from app.models.queue_render import QueueRender
    from app.services.storage import get_storage
    import logging
    
    storage = get_storage()
    logger = logging.getLogger(__name__)
    old_queues = list(db.scalars(select(Queue)).all())
    for old_q in old_queues:
        # Collect every storage blob the queue owns before the cascade
        # delete removes the rows that point to them. Both the stitched
        # queue mix AND each per-pair transition render (mixes/<id>.wav)
        # must go — otherwise the per-pair WAVs orphan in object storage.
        keys: list[str] = []
        render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == old_q.id))
        if render_row and render_row.rendered_audio_path:
            keys.append(render_row.rendered_audio_path)
        plans = db.scalars(select(MixPlan).where(MixPlan.queue_id == old_q.id)).all()
        for plan in plans:
            if plan.rendered_audio_path:
                keys.append(plan.rendered_audio_path)

        for key in keys:
            try:
                await storage.delete(key)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"create_queue: failed to delete old blob {key!r}: {e}")
        db.delete(old_q)
    db.commit()

    queue = Queue()
    db.add(queue)
    db.commit()
    db.refresh(queue)
    return queue


@router.get("/current", response_model=QueueRead)
def get_current_queue(db: Session = Depends(get_db)) -> Queue:
    queue = db.scalar(
        select(Queue).where(Queue.locked.is_(False)).order_by(Queue.created_at.desc())
    )
    if queue is None:
        queue = db.scalar(select(Queue).order_by(Queue.created_at.desc()))
    if queue is None:
        raise HTTPException(status_code=404, detail="no queue exists")
    return queue


@router.get("/{queue_id}", response_model=QueueRead)
def get_queue(queue_id: uuid.UUID, db: Session = Depends(get_db)) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    return queue


@router.post(
    "/{queue_id}/items",
    response_model=QueueRead,
    status_code=status.HTTP_201_CREATED,
)
def add_queue_item(
    queue_id: uuid.UUID, payload: QueueItemAdd, db: Session = Depends(get_db)
) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is locked")

    song = db.get(Song, payload.song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")

    current_count = len(queue.items)
    if current_count >= QUEUE_CAP:
        raise HTTPException(
            status_code=409,
            detail=f"queue is full (cap={QUEUE_CAP})",
        )

    item = QueueItem(
        queue_id=queue_id,
        song_id=song.id,
        position=current_count,
    )
    db.add(item)
    # Queuing a song is an access — refresh its LRU clock so it sorts as
    # recently-used even after this queue is later replaced.
    song.last_accessed_at = datetime.now(timezone.utc)
    db.commit()

    # Queueing signals intent to mix — start the pipeline NOW instead of
    # waiting for lock: analysis unlocks Suggest Order pre-lock, and
    # stems/transcription done early shorten the post-lock wait.
    # dispatch_download=False because POST /api/songs already dispatched
    # the download for fresh songs — a second parallel yt-dlp writing the
    # same .wav would race it (one fails and marks the song failed).
    _enqueue_pipeline_for_song(song, db, dispatch_download=False)

    db.refresh(queue)
    return queue


@router.delete(
    "/{queue_id}/items/{item_id}",
    response_model=QueueRead,
)
def remove_queue_item(
    queue_id: uuid.UUID, item_id: uuid.UUID, db: Session = Depends(get_db)
) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is locked")

    item = db.get(QueueItem, item_id)
    if item is None or item.queue_id != queue_id:
        raise HTTPException(status_code=404, detail="queue item not found")

    db.delete(item)
    db.flush()
    _compact_positions(db, queue_id)
    db.commit()
    db.refresh(queue)
    return queue


@router.patch(
    "/{queue_id}/items",
    response_model=QueueRead,
)
def reorder_queue_items(
    queue_id: uuid.UUID, payload: QueueReorder, db: Session = Depends(get_db)
) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is locked")

    current_ids = {item.id for item in queue.items}
    requested_ids = list(payload.ordered_item_ids)
    if set(requested_ids) != current_ids or len(requested_ids) != len(current_ids):
        raise HTTPException(
            status_code=400,
            detail="ordered_item_ids must be a permutation of the queue's items",
        )

    by_id = {item.id: item for item in queue.items}
    # Two-pass shuffle to avoid the unique (queue_id, position) collision.
    for idx, item_id in enumerate(requested_ids):
        by_id[item_id].position = -1000 - idx
    db.flush()
    for idx, item_id in enumerate(requested_ids):
        by_id[item_id].position = idx
    db.commit()
    db.refresh(queue)
    return queue


# A re-planned transition needs render + stitch time before the playhead
# reaches it; transitions starting sooner than this stay as rendered.
ENERGY_DIAL_LEAD_SECONDS = 90.0


class EnergyDialRequest(BaseModel):
    direction: Literal["up", "hold", "down"]
    position_seconds: float = 0.0


@router.post("/{queue_id}/energy")
def set_energy_dial(
    queue_id: uuid.UUID,
    payload: "EnergyDialRequest",
    db: Session = Depends(get_db),
) -> dict:
    """Live energy dial: bend the not-yet-played remainder of the set.

    Selects transitions starting >= ENERGY_DIAL_LEAD_SECONDS ahead of the
    playhead, stamps them with the bias, re-plans + re-renders just those
    pairs and re-stitches. Content before the first changed pair is
    time-identical, so the player can hot-swap and keep its position.
    """
    from app.models import MixPlanStatus, QueueRender, QueueRenderStatus

    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if not queue.locked:
        raise HTTPException(status_code=409, detail="queue is not locked")

    render = db.scalar(
        select(QueueRender).where(QueueRender.queue_id == queue_id)
    )
    if render is None or not (render.timeline or {}).get("transitions"):
        raise HTTPException(
            status_code=409, detail="no rendered mix timeline yet"
        )
    if render.status == QueueRenderStatus.rendering:
        raise HTTPException(
            status_code=409,
            detail="queue mix is being stitched; try again when it lands",
        )

    bias = None if payload.direction == "hold" else payload.direction
    horizon = payload.position_seconds + ENERGY_DIAL_LEAD_SECONDS
    affected: list[int] = []
    for tr in render.timeline["transitions"]:
        if tr["start"] < horizon:
            continue
        plan = db.scalar(
            select(MixPlan)
            .where(MixPlan.queue_id == queue_id)
            .where(MixPlan.from_song_id == uuid.UUID(tr["from_song_id"]))
            .where(MixPlan.to_song_id == uuid.UUID(tr["to_song_id"]))
        )
        if plan is None or plan.status == MixPlanStatus.rendering:
            continue
        if (plan.energy_bias or None) == bias:
            continue  # already pointing that way — don't burn a render
        plan.energy_bias = bias
        plan.reroll_nonce = (plan.reroll_nonce or 0) + 1
        plan.plan_json = None
        plan.rendered_audio_path = None
        plan.status = MixPlanStatus.pending
        plan.error_text = None
        affected.append(tr["index"])
    db.commit()

    if not affected:
        return {"affected_transitions": [], "direction": payload.direction}

    from app.workers.auto_stitch import reset_and_dispatch_stitch

    reset_and_dispatch_stitch(queue_id, db)
    return {
        "affected_transitions": affected,
        "direction": payload.direction,
        "lead_seconds": ENERGY_DIAL_LEAD_SECONDS,
    }


@router.get("/meta/context_options")
def list_context_options() -> dict:
    """Occasion + arc-template menus for the queue builder UI."""
    from app.services.mixer.occasions import ARC_TEMPLATES, OCCASIONS

    from app.services.host.script import HOST_PERSONAS

    return {
        "occasions": [
            {"id": o.id, "label": o.label, "default_arc": o.default_arc}
            for o in OCCASIONS.values()
        ],
        "arcs": [
            {"id": a.id, "label": a.label, "description": a.description}
            for a in ARC_TEMPLATES.values()
        ],
        "host_frequencies": ["off", "intro_only", "sparse", "chatty"],
        "host_personas": sorted(HOST_PERSONAS),
    }


@router.patch("/{queue_id}", response_model=QueueRead)
def update_queue_context(
    queue_id: uuid.UUID,
    payload: QueueContextUpdate,
    db: Session = Depends(get_db),
) -> Queue:
    """Set the gig context: occasion, vibe note, arc template, hook
    teasing. Allowed until the queue is locked (the planners snapshot it
    at plan time)."""
    from app.services.mixer.occasions import ARC_TEMPLATES, OCCASIONS

    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is locked")

    fields = payload.model_dump(exclude_unset=True)
    if "occasion" in fields and fields["occasion"] is not None \
            and fields["occasion"] not in OCCASIONS:
        raise HTTPException(status_code=422, detail="unknown occasion")
    if "arc_template" in fields and fields["arc_template"] is not None \
            and fields["arc_template"] not in ARC_TEMPLATES:
        raise HTTPException(status_code=422, detail="unknown arc_template")
    if "host_frequency" in fields and fields["host_frequency"] is not None \
            and fields["host_frequency"] not in (
                "off", "intro_only", "sparse", "chatty"):
        raise HTTPException(status_code=422, detail="unknown host_frequency")
    if "host_persona" in fields and fields["host_persona"] is not None:
        from app.services.host.script import HOST_PERSONAS

        if fields["host_persona"] not in HOST_PERSONAS:
            raise HTTPException(status_code=422, detail="unknown host_persona")
    for name, value in fields.items():
        setattr(queue, name, value)
    db.commit()
    db.refresh(queue)
    return queue


@router.post("/{queue_id}/suggest_order")
def suggest_queue_order(
    queue_id: uuid.UUID,
    pin_first: bool = True,
    db: Session = Depends(get_db),
) -> dict:
    """Propose a track order that minimizes key/tempo/energy/genre
    friction across the whole set. Suggestion only — the caller applies
    it via the existing reorder PATCH. Requires every song analyzed."""
    from app.models import Analysis
    from app.services.mixer.ordering import (
        PairInput,
        edge_info,
        order_cost,
        suggest_order,
    )

    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is locked")
    items = sorted(queue.items, key=lambda it: it.position)
    if len(items) < 3:
        raise HTTPException(
            status_code=409, detail="need at least 3 songs to reorder"
        )

    inputs: list[PairInput] = []
    missing: list[str] = []
    for item in items:
        song = db.get(Song, item.song_id)
        analysis = db.scalar(
            select(Analysis).where(Analysis.song_id == item.song_id)
        )
        if song is None or analysis is None or not analysis.bpm:
            missing.append(song.title if song else str(item.song_id))
            continue
        genres = tuple(
            (analysis.tags or {}).get("genres") or []
        ) if analysis.tags else ()
        inputs.append(PairInput(
            song_id=str(item.song_id),
            title=song.title,
            bpm=analysis.bpm,
            key=analysis.key,
            camelot_key=analysis.camelot_key,
            energy_curve=list(analysis.energy_curve or []),
            genres=genres,
        ))
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                "still analyzing (ordering needs BPM/key/energy): "
                + ", ".join(missing)
                + " — try again in a moment"
            ),
        )

    order, edges = suggest_order(inputs, pin_first=pin_first)
    current = list(range(len(inputs)))
    current_edges = [
        edge_info(inputs[i], inputs[i + 1]) for i in range(len(inputs) - 1)
    ]
    suggested_cost = order_cost(inputs, order)
    current_cost = order_cost(inputs, current)
    # Item ids in suggested order — directly usable by the reorder PATCH.
    item_by_song = {str(it.song_id): it.id for it in items}
    return {
        "order": [inputs[i].song_id for i in order],
        "ordered_item_ids": [item_by_song[inputs[i].song_id] for i in order],
        "edges": [e.__dict__ for e in edges],
        "current_edges": [e.__dict__ for e in current_edges],
        "current_cost": round(current_cost, 3),
        "suggested_cost": round(suggested_cost, 3),
        "improved": suggested_cost + 1e-9 < current_cost,
    }


def _enqueue_pipeline_for_song(
    song: Song, db: Session, dispatch_download: bool = True
) -> None:
    """Kick the appropriate next pipeline stage for one song.

    Workers auto-chain on success (download→analyze→separate→transcribe)
    when `song.pipeline_requested` is True, so this only needs to dispatch
    the SINGLE current-needed stage. Songs already in flight
    (`downloading`/`analyzing`/etc.) are no-ops — the in-flight worker
    will auto-dispatch its successor on completion. That auto-chain is
    what fixes the "lock-during-download stalls the pipeline" bug.

    ``dispatch_download=False`` is for callers that run right after song
    creation (add-to-queue): POST /api/songs already dispatched the
    download, and a second yt-dlp writing the same WAV races the first.
    Setting `pipeline_requested` alone is enough there — the in-flight
    download auto-chains into analyze when it lands.

    Lyrics fetch is fire-and-forget, independent of the audio pipeline.
    """
    sid = str(song.id)
    # Signal that the user wants the full pipeline for this song. Workers
    # gate their auto-dispatch on this flag, so flipping it true here is
    # what makes a currently-running download or analyze auto-progress
    # all the way through transcribe.
    song.pipeline_requested = True
    db.commit()

    has_stems = (
        db.scalar(select(Stems.id).where(Stems.song_id == song.id)) is not None
    )
    has_transcription = (
        db.scalar(select(Transcription.id).where(Transcription.song_id == song.id))
        is not None
    )

    from app.models.lyrics import Lyrics, LyricsFetchStatus
    has_lyrics = (
        db.scalar(select(Lyrics.id).where(
            (Lyrics.song_id == song.id) &
            (Lyrics.fetch_status.in_([
                LyricsFetchStatus.success,
                LyricsFetchStatus.not_found,
                LyricsFetchStatus.error,
            ]))
        )) is not None
    )
    if not has_lyrics:
        _TaskShim("app.workers.fetch_lyrics.fetch_lyrics").delay(sid)

    # In-flight states: a running worker will carry the pipeline forward
    # on its own via auto-chain (which now sees pipeline_requested=True).
    if song.status in (
        SongStatus.downloading,
        SongStatus.analyzing,
        SongStatus.separating,
        SongStatus.transcribing,
    ):
        return

    # Idle states: kick the next-needed stage. Auto-chain handles the rest.
    if song.status in (SongStatus.pending, SongStatus.failed) and not song.audio_path:
        # A failed download is never in flight, so retrying is always safe;
        # a `pending` one may have JUST been dispatched by song creation.
        if song.status == SongStatus.failed or dispatch_download:
            download_song.apply_async(args=[sid], priority=PRI_DOWNLOAD)
    elif song.status in (SongStatus.downloaded, SongStatus.failed):
        analyze_song.apply_async(args=[sid], priority=PRI_ANALYZE)
    elif song.status in (SongStatus.analyzed, SongStatus.ready):
        if not has_stems:
            separate_stems.apply_async(args=[sid], priority=PRI_SEPARATE)
        elif not has_transcription:
            transcribe_song.apply_async(args=[sid], priority=PRI_TRANSCRIBE)
        # else: fully processed — nothing to do.


@router.post(
    "/{queue_id}/lock",
    response_model=QueueRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def lock_queue(queue_id: uuid.UUID, db: Session = Depends(get_db)) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is already locked")
    if not queue.items:
        raise HTTPException(status_code=409, detail="queue is empty")

    queue.locked = True
    queue.locked_at = datetime.now(timezone.utc)
    # Snapshot songs while the session is open; we'll enqueue after commit
    # so a transient task-broker failure doesn't roll back the lock.
    songs_to_pipeline = [item.song for item in queue.items]
    # Locking accesses every song in the queue — refresh their LRU clocks.
    now = datetime.now(timezone.utc)
    for item in queue.items:
        item.song.last_accessed_at = now
    db.commit()
    db.refresh(queue)

    for song in songs_to_pipeline:
        _enqueue_pipeline_for_song(song, db)

    # Phase 7: seed MixPlan rows for each adjacent pair. plan_json is
    # generated lazily at render time so the LLM call in Phase 9 doesn't
    # fire for plans the user never asks to render. Local import to dodge
    # the api/queues ↔ api/mix_plans circular at module load.
    from app.api.mix_plans import _seed_mix_plans
    _seed_mix_plans(queue, db)

    # Phase 10 eager stitch: if every song is ALREADY `ready` at lock time
    # (fully-cached queue), no worker will re-run to fire the completion
    # hook — so kick the render+stitch chord here. maybe_dispatch_stitch
    # no-ops when songs are still processing (the normal case), where the
    # transcribe-completion hook fires it instead.
    from app.workers.auto_stitch import maybe_dispatch_stitch
    maybe_dispatch_stitch(queue.id, db)

    return queue


@router.post(
    "/{queue_id}/stitch",
    status_code=status.HTTP_202_ACCEPTED,
)
def stitch_queue_route(queue_id: uuid.UUID, db: Session = Depends(get_db)):
    """Manual (re)render of the continuous mix. The eager auto-stitch (see
    auto_stitch.maybe_dispatch_stitch) already fires this chord during the
    Processing state once every song is ready; this endpoint is the
    user-triggered recovery / re-render path."""
    queue = db.get(Queue, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="queue not found")
    if not queue.locked:
        raise HTTPException(status_code=409, detail="queue must be locked to stitch")

    from app.workers.auto_stitch import reset_and_dispatch_stitch
    reset_and_dispatch_stitch(queue_id, db)

    return {"message": "Stitching started"}


@router.get("/{queue_id}/mix", response_model=QueueRenderRead)
def get_queue_mix(queue_id: uuid.UUID, db: Session = Depends(get_db)):
    from app.models.queue_render import QueueRender
    
    render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_id))
    if not render_row:
        raise HTTPException(status_code=404, detail="no mix found for queue")
    return render_row


@router.get("/{queue_id}/mix/audio")
async def get_queue_mix_audio(
    queue_id: uuid.UUID,
    db: Session = Depends(get_db),
    range: str | None = Header(default=None),
):
    from app.models.queue_render import QueueRender, QueueRenderStatus
    from app.api.songs import _stream_audio_response
    
    render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_id))
    if not render_row or render_row.status != QueueRenderStatus.ready or not render_row.rendered_audio_path:
        raise HTTPException(status_code=404, detail="mix audio not ready")

    # Rows rendered before the AAC switch still point at .flac keys.
    if render_row.rendered_audio_path.endswith(".flac"):
        media_type, filename = "audio/flac", "mix.flac"
    else:
        media_type, filename = "audio/mp4", "mix.m4a"
    return await _stream_audio_response(
        render_row.rendered_audio_path, media_type, range, download_filename=filename
    )
