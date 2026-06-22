from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.song import Song


class Queue(Base):
    __tablename__ = "queues"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    locked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    items: Mapped[list["QueueItem"]] = relationship(
        "QueueItem",
        back_populates="queue",
        cascade="all, delete-orphan",
        order_by="QueueItem.position",
    )


class QueueItem(Base):
    __tablename__ = "queue_items"
    __table_args__ = (
        UniqueConstraint("queue_id", "position", name="uq_queue_items_position"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    queue_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queues.id", ondelete="CASCADE"),
        nullable=False,
    )
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    # Whole-song semitone offset assigned by the set-level pitch resolver
    # (pitch_mode="whole_song"): this song plays shifted by this many
    # semitones for its ENTIRE duration in this queue's mix, eliminating
    # mid-song key glides. 0 = native key. Queue-scoped on purpose — the
    # same song can need different offsets next to different neighbors.
    pitch_offset_semitones: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )

    queue: Mapped["Queue"] = relationship("Queue", back_populates="items")
    song: Mapped["Song"] = relationship("Song", lazy="joined")
