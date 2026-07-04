"""Test fixtures.

DB-touching tests use the dev Postgres (the one docker compose brings up).
Each test runs inside a transaction that is rolled back at teardown so
in-test writes don't persist. We also TRUNCATE the small set of tables
that the queue/songs API reads at fixture setup — without this, rows left
over from interactive dev (or earlier test runs that committed outside
the fixture) leak into list/count assertions. The whole arrangement can
be swapped for testcontainers in a later polish phase.

Storage: production runs `STORAGE_BACKEND=s3` for DO Spaces. Tests force
the local backend rooted at a per-session tmpdir so we don't accidentally
talk to (or rely on network access to) DO Spaces during pytest.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import SessionLocal, engine
from app.models import Song  # noqa: F401 — ensure Song table is registered
from app.services.storage import factory as _storage_factory


@pytest.fixture(autouse=True, scope="session")
def _force_local_storage_for_tests() -> Generator[Path, None, None]:
    """Pin every test to LocalFilesystemStorage under a tmpdir, regardless
    of what .env says. `get_storage` is lru_cached, so we both flip the
    setting and clear the cache. Also force `modal_token_id=""` so the
    worker tests take the local (mockable) code path instead of trying to
    look up a deployed Modal function.
    """
    with tempfile.TemporaryDirectory(prefix="aidj-tests-") as td:
        root = Path(td)
        original_backend = settings.storage_backend
        original_root = settings.local_storage_path
        original_modal_id = settings.modal_token_id
        original_tts = settings.tts_provider
        settings.storage_backend = "local"
        settings.local_storage_path = root
        settings.modal_token_id = ""
        # The TTS host must never run real LLM/TTS inside unit tests —
        # with a live .env it genuinely will (kokoro synthesizes locally).
        # Host tests patch the provider back on explicitly.
        settings.tts_provider = "off"
        _storage_factory.get_storage.cache_clear()
        try:
            yield root
        finally:
            settings.storage_backend = original_backend
            settings.local_storage_path = original_root
            settings.modal_token_id = original_modal_id
            settings.tts_provider = original_tts
            _storage_factory.get_storage.cache_clear()


# Tables to wipe before each db_session test. Order doesn't matter — TRUNCATE
# ... CASCADE handles the FKs. queue_items and queues are listed for clarity
# even though the CASCADE from songs would catch them via queue_items.song_id.
_TRUNCATE_TABLES = (
    "queue_renders",
    "mix_plans",
    "queue_items",
    "queues",
    "lyrics",
    "transcriptions",
    "stems",
    "analyses",
    "songs",
)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    connection = engine.connect()
    # Truncate outside the rollback envelope so the wipe persists even when
    # tests don't commit. RESTART IDENTITY isn't needed (all our PKs are
    # UUIDs) but CASCADE is — queue_items → songs FK would otherwise block.
    connection.execute(
        text(f"TRUNCATE TABLE {', '.join(_TRUNCATE_TABLES)} CASCADE")
    )
    connection.commit()

    trans = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()
