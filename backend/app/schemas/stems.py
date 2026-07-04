from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import StemsStatus


class StemsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    song_id: uuid.UUID
    model_name: str
    status: StemsStatus
    vocals_path: str | None
    drums_path: str | None
    bass_path: str | None
    other_path: str | None
    vocal_rms: float | None
    envelopes_path: str | None
    created_at: datetime
    updated_at: datetime
