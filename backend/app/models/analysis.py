import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    bpm: Mapped[float] = mapped_column(Float, nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False)
    camelot_key: Mapped[str] = mapped_column(String, nullable=False)
    time_signature: Mapped[int] = mapped_column(Integer, nullable=False, default=4)

    beat_grid: Mapped[list[float]] = mapped_column(JSONB, nullable=False)
    downbeats: Mapped[list[float]] = mapped_column(JSONB, nullable=False)
    sections: Mapped[list[dict]] = mapped_column(JSONB, nullable=False)
    energy_curve: Mapped[list[float]] = mapped_column(JSONB, nullable=False)
    # Populated by Whisper in Phase 6; persisted as an empty list until then.
    vocal_segments: Mapped[list[list[float]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    tags: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
