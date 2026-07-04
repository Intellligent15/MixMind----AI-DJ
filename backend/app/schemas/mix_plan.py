from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import MixPlanStatus


class MixPlanRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    queue_id: uuid.UUID
    from_song_id: uuid.UUID
    to_song_id: uuid.UUID
    plan_json: list[dict] | None
    rendered_audio_path: str | None
    status: MixPlanStatus
    error_text: str | None
    plan_source: str | None = None
    style: str | None = None
    rationale: str | None = None
    style_hint: str | None = None
    style_override: str | None = None
    reroll_nonce: int = 0
    qa_metrics: dict | None = None
    qa_verdict: str | None = None
    created_at: datetime
    updated_at: datetime
