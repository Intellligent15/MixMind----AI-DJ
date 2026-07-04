import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class StemsStatus(str, enum.Enum):
    pending = "pending"
    separating = "separating"
    separated = "separated"
    failed = "failed"


class Stems(Base):
    __tablename__ = "stems"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String, nullable=False, default="htdemucs"
    )
    status: Mapped[StemsStatus] = mapped_column(
        Enum(StemsStatus, name="stems_status"),
        nullable=False,
        default=StemsStatus.pending,
    )

    vocals_path: Mapped[str | None] = mapped_column(String, nullable=True)
    drums_path: Mapped[str | None] = mapped_column(String, nullable=True)
    bass_path: Mapped[str | None] = mapped_column(String, nullable=True)
    other_path: Mapped[str | None] = mapped_column(String, nullable=True)

    # RMS amplitude of the vocal stem (linear, 0..~1). Phase 6's Whisper-skip
    # decision reads this instead of re-loading the vocal stem.
    vocal_rms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Storage key for the frame-wise per-stem RMS+peak envelope sidecar
    # (10 Hz JSON). Null on Stems rows that pre-date this column —
    # re-separate to populate.
    # Schema lives in services/stems/service.py::_compute_stem_envelopes.
    # Consumed by the Vocal Safety service and the seam-candidate builder.
    envelopes_path: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
