"""Listener feedback capture — the crowd-reading half of the DJ.

Explicit: thumbs up/down on the currently-playing transition (the player
knows the from/to song pair from the mix timeline). Implicit: skip /
replay events with an output-time position, attributed to a transition
via the QueueRender timeline when one matches.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import (
    Analysis,
    ListenerEvent,
    ListenerEventKind,
    MixPlan,
    QueueRender,
)
from app.services.feedback.summary import feedback_context, style_tallies

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

# A skip that lands within this many seconds after a transition's end is
# still blamed on it — the listener reacted to what they just heard.
SKIP_ATTRIBUTION_TAIL_SECONDS = 30.0


class TransitionFeedback(BaseModel):
    queue_id: uuid.UUID
    from_song_id: uuid.UUID
    to_song_id: uuid.UUID
    kind: ListenerEventKind = Field(
        description="thumbs_up or thumbs_down"
    )


class PlaybackEvent(BaseModel):
    queue_id: uuid.UUID
    kind: ListenerEventKind = Field(description="skip or replay")
    position_seconds: float


def _first_genre(db: Session, song_id: uuid.UUID) -> str | None:
    analysis = db.scalar(select(Analysis).where(Analysis.song_id == song_id))
    if analysis is None or not analysis.tags:
        return None
    genres = analysis.tags.get("genres") or []
    return genres[0] if genres else None


@router.post("", status_code=201)
def submit_transition_feedback(
    payload: TransitionFeedback, db: Session = Depends(get_db)
) -> dict:
    if payload.kind not in (
        ListenerEventKind.thumbs_up, ListenerEventKind.thumbs_down
    ):
        raise HTTPException(422, "kind must be thumbs_up or thumbs_down")
    plan = db.scalar(
        select(MixPlan)
        .where(MixPlan.queue_id == payload.queue_id)
        .where(MixPlan.from_song_id == payload.from_song_id)
        .where(MixPlan.to_song_id == payload.to_song_id)
    )
    event = ListenerEvent(
        queue_id=payload.queue_id,
        mix_plan_id=plan.id if plan else None,
        kind=payload.kind,
        style=plan.style if plan else None,
        from_genre=_first_genre(db, payload.from_song_id),
        to_genre=_first_genre(db, payload.to_song_id),
    )
    db.add(event)
    db.commit()
    return {"status": "noted", "style": event.style}


@router.post("/playback", status_code=201)
def submit_playback_event(
    payload: PlaybackEvent, db: Session = Depends(get_db)
) -> dict:
    if payload.kind not in (ListenerEventKind.skip, ListenerEventKind.replay):
        raise HTTPException(422, "kind must be skip or replay")

    # Attribute to a transition when the position sits inside one or just
    # after it (the listener reacted to what they just heard).
    style = None
    mix_plan_id = None
    from_genre = to_genre = None
    render = db.scalar(
        select(QueueRender).where(QueueRender.queue_id == payload.queue_id)
    )
    timeline = (render.timeline or {}) if render else {}
    for tr in timeline.get("transitions", []):
        if tr["start"] <= payload.position_seconds <= (
            tr["end"] + SKIP_ATTRIBUTION_TAIL_SECONDS
        ):
            style = tr.get("label")
            plan = db.scalar(
                select(MixPlan)
                .where(MixPlan.queue_id == payload.queue_id)
                .where(MixPlan.from_song_id == uuid.UUID(tr["from_song_id"]))
                .where(MixPlan.to_song_id == uuid.UUID(tr["to_song_id"]))
            )
            if plan is not None:
                mix_plan_id = plan.id
                style = plan.style or style
                from_genre = _first_genre(db, plan.from_song_id)
                to_genre = _first_genre(db, plan.to_song_id)
            break

    event = ListenerEvent(
        queue_id=payload.queue_id,
        mix_plan_id=mix_plan_id,
        kind=payload.kind,
        style=style,
        from_genre=from_genre,
        to_genre=to_genre,
        position_seconds=payload.position_seconds,
    )
    db.add(event)
    db.commit()
    return {"status": "noted", "attributed_style": style}


@router.get("/summary")
def get_feedback_summary(db: Session = Depends(get_db)) -> dict:
    tallies = style_tallies(db)
    return {
        "styles": {
            style: {"up": t.up, "down": t.down, "skips": t.skips}
            for style, t in tallies.items()
        },
        "context_line": feedback_context(db),
    }
