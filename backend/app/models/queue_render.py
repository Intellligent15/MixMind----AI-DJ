import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class QueueRenderStatus(str, enum.Enum):
    pending = "pending"
    rendering = "rendering"
    ready = "ready"
    failed = "failed"


class QueueRender(Base):
    __tablename__ = "queue_renders"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    queue_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queues.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # StorageBackend key, e.g. "queue_mixes/<queue_id>.m4a". Null until rendered.
    rendered_audio_path: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[QueueRenderStatus] = mapped_column(
        Enum(QueueRenderStatus, name="queue_render_status"),
        nullable=False,
        default=QueueRenderStatus.pending,
    )

    error_text: Mapped[str | None] = mapped_column(String, nullable=True)

    # Phase 10: output-timeline map for the player's transition indicator.
    # Written by stitch_queue alongside the M4A. Shape:
    #   {"duration": float,
    #    "songs": [{"index", "song_id", "title", "artist", "start", "end"}],
    #    "transitions": [{"index", "from_song_id", "to_song_id", "start",
    #                     "end", "label", "stems": [...], "reasoning"}]}
    # Times are seconds in the stitched-mix timeline. Null until stitched
    # (and on rows rendered before this column existed — re-stitch to fill).
    timeline: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
