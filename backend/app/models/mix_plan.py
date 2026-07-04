import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class MixPlanStatus(str, enum.Enum):
    pending = "pending"
    rendering = "rendering"
    ready = "ready"
    failed = "failed"


class MixPlan(Base):
    __tablename__ = "mix_plans"
    __table_args__ = (
        UniqueConstraint(
            "queue_id", "from_song_id", "to_song_id", name="uq_mix_plan_pair"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    queue_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queues.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        nullable=False,
    )
    to_song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Null at lock time (generated lazily at render time so Phase 9's LLM
    # call doesn't fire for plans the user never asks to render). Filled
    # with the tool-call list before the executor walks it.
    plan_json: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)

    # StorageBackend key, e.g. "mixes/<mix_plan_id>.wav". Null until rendered.
    rendered_audio_path: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[MixPlanStatus] = mapped_column(
        Enum(MixPlanStatus, name="mix_plan_status"),
        nullable=False,
        default=MixPlanStatus.pending,
    )

    # Populated when status=failed so the debug UI can show the user *why*
    # without asking them to scrape worker logs.
    error_text: Mapped[str | None] = mapped_column(String, nullable=True)

    # ------- planner v2 telemetry & controls -------
    # Which path produced plan_json: "llm_v2", "llm_v2_repaired",
    # "llm_legacy", "llm_legacy_repaired", "style_default", or
    # "deterministic_fallback". Surfaced in the UI so a fallback is never
    # silent again.
    plan_source: Mapped[str | None] = mapped_column(String, nullable=True)

    # Archetype the plan used (TransitionStyle value), and the LLM's own
    # one-liner about why — shown next to the transition in the player.
    style: Mapped[str | None] = mapped_column(String, nullable=True)
    rationale: Mapped[str | None] = mapped_column(String, nullable=True)

    # Set-level pass suggestion (soft) vs user pin (hard). The planner
    # treats style_hint as a strong default and style_override as law.
    style_hint: Mapped[str | None] = mapped_column(String, nullable=True)
    style_override: Mapped[str | None] = mapped_column(String, nullable=True)

    # Bumped by POST /mix_plans/{id}/reroll; part of the LLM cache key, so
    # each re-roll gets a genuinely fresh plan instead of a cache hit.
    reroll_nonce: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )

    # Live energy dial (F6): "up" | "down" when the listener asked to bend
    # the remaining set mid-playback; fed to the planner as context on the
    # re-render. Null = no bias.
    energy_bias: Mapped[str | None] = mapped_column(String, nullable=True)

    # Render QA (services/mixer/qa.py): metrics dict + flags from the last
    # render, and its verdict ("pass" | "warn" | "fail"). A "fail" here
    # means the auto-reroll budget was exhausted and the render shipped
    # anyway — surfaced as a badge in the debug UI, never a hard error.
    qa_metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    qa_verdict: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
