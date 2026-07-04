import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class ListenerEventKind(str, enum.Enum):
    thumbs_up = "thumbs_up"
    thumbs_down = "thumbs_down"
    skip = "skip"
    replay = "replay"


class ListenerEvent(Base):
    """One reaction signal — explicit (thumbs on a transition) or implicit
    (skip/replay during mix playback).

    `style` and the genre pair are denormalized at write time so the
    listening history survives plan re-rolls and queue deletion — the
    feedback loop cares about "wash_out on house→techno went badly",
    not about the specific row that rendered it.
    """

    __tablename__ = "listener_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    queue_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queues.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    mix_plan_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mix_plans.id", ondelete="SET NULL"),
        nullable=True,
    )
    kind: Mapped[ListenerEventKind] = mapped_column(
        Enum(ListenerEventKind, name="listener_event_kind"), nullable=False
    )
    style: Mapped[str | None] = mapped_column(String, nullable=True)
    from_genre: Mapped[str | None] = mapped_column(String, nullable=True)
    to_genre: Mapped[str | None] = mapped_column(String, nullable=True)
    position_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
