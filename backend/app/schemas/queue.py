from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.song import SongRead


class QueueItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    queue_id: uuid.UUID
    position: int
    pitch_offset_semitones: int = 0
    song: SongRead


class QueueRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    locked: bool
    created_at: datetime
    locked_at: datetime | None
    occasion: str | None = None
    vibe_note: str | None = None
    arc_template: str | None = None
    tease_hooks: bool = False
    host_frequency: str | None = None
    host_persona: str | None = None
    items: list[QueueItemRead]


class QueueContextUpdate(BaseModel):
    """PATCH /api/queues/{id} — occasion / vibe / arc / tease settings.
    All optional; only provided fields change."""

    occasion: str | None = None
    vibe_note: str | None = None
    arc_template: str | None = None
    tease_hooks: bool | None = None
    host_frequency: str | None = None
    host_persona: str | None = None


class QueueItemAdd(BaseModel):
    song_id: uuid.UUID


class QueueReorder(BaseModel):
    ordered_item_ids: list[uuid.UUID]
