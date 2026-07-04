"""Tests for the Phase 8 queue-mix HTTP surface."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.main import app
from app.models import (
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    QueueRender,
    QueueRenderStatus,
    Song,
    SongStatus,
)


def _client(db_session: Session) -> TestClient:
    def override_db():
        yield db_session
    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def _make_song(db: Session, vid: str) -> Song:
    s = Song(
        youtube_video_id=vid,
        title=f"song-{vid}",
        duration_seconds=180.0,
        audio_path=f"audio/{vid}.wav",
        status=SongStatus.ready,
    )
    db.add(s); db.flush()
    return s


def _seed_locked_queue(db: Session, n: int = 2) -> Queue:
    q = Queue(locked=True); db.add(q); db.flush()
    songs = [_make_song(db, f"qm-{i}-{uuid.uuid4().hex[:8]}") for i in range(n)]
    for i, s in enumerate(songs):
        db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
    for i in range(n - 1):
        db.add(MixPlan(
            queue_id=q.id,
            from_song_id=songs[i].id, to_song_id=songs[i + 1].id,
            status=MixPlanStatus.ready,
            rendered_audio_path=f"mixes/m{i}.wav",
            plan_json=[],
        ))
    db.flush()
    return q


def test_stitch_404_when_queue_missing(db_session: Session):
    client = _client(db_session)
    r = client.post(f"/api/queues/{uuid.uuid4()}/stitch")
    assert r.status_code == 404


def test_stitch_409_when_queue_not_locked(db_session: Session):
    q = Queue(locked=False); db_session.add(q); db_session.flush()
    client = _client(db_session)
    r = client.post(f"/api/queues/{q.id}/stitch")
    assert r.status_code == 409


def test_stitch_dispatches_when_all_mixes_ready(db_session: Session):
    # Phase 10 routes the manual stitch through auto_stitch; all plans are
    # ready in the seed, so the stitch task fires directly (no render chord).
    q = _seed_locked_queue(db_session, n=2)
    client = _client(db_session)
    with patch("app.workers.auto_stitch.celery_app") as celery, \
            patch("app.workers.auto_stitch.chord") as chord_mock:
        r = client.post(f"/api/queues/{q.id}/stitch")
    assert r.status_code == 202
    sig_names = [c.args[0] for c in celery.signature.call_args_list]
    assert "app.workers.stitch_queue.stitch_queue" in sig_names
    # All pairs ready → no render fan-out, so no chord.
    chord_mock.assert_not_called()


def test_stitch_creates_queue_render_row(db_session: Session):
    q = _seed_locked_queue(db_session, n=2)
    client = _client(db_session)
    with patch("app.workers.auto_stitch.celery_app"), \
            patch("app.workers.auto_stitch.chord"):
        client.post(f"/api/queues/{q.id}/stitch")
    row = db_session.scalar(select(QueueRender).where(QueueRender.queue_id == q.id))
    assert row is not None
    assert row.status == QueueRenderStatus.pending


def test_stitch_resets_failed_row_to_pending(db_session: Session):
    q = _seed_locked_queue(db_session, n=2)
    db_session.add(QueueRender(
        queue_id=q.id,
        status=QueueRenderStatus.failed,
        error_text="earlier",
    ))
    db_session.flush()
    client = _client(db_session)
    with patch("app.workers.auto_stitch.celery_app"), \
            patch("app.workers.auto_stitch.chord"):
        client.post(f"/api/queues/{q.id}/stitch")
    db_session.expire_all()
    row = db_session.scalar(select(QueueRender).where(QueueRender.queue_id == q.id))
    assert row.status == QueueRenderStatus.pending
    assert row.error_text is None


def test_get_mix_404_when_missing(db_session: Session):
    q = Queue(locked=True); db_session.add(q); db_session.flush()
    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix")
    assert r.status_code == 404


def test_get_mix_returns_row(db_session: Session):
    q = Queue(locked=True); db_session.add(q); db_session.flush()
    db_session.add(QueueRender(
        queue_id=q.id,
        status=QueueRenderStatus.ready,
        rendered_audio_path="queue_mixes/x.m4a",
    ))
    db_session.flush()
    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["rendered_audio_path"] == "queue_mixes/x.m4a"


def test_get_mix_audio_404_when_not_ready(db_session: Session):
    q = Queue(locked=True); db_session.add(q); db_session.flush()
    db_session.add(QueueRender(queue_id=q.id, status=QueueRenderStatus.rendering))
    db_session.flush()
    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix/audio")
    assert r.status_code == 404


def test_get_mix_audio_streams_m4a(db_session: Session):
    """When ready, the route streams the M4A bytes with
    Content-Disposition: attachment."""
    q = Queue(locked=True); db_session.add(q); db_session.flush()
    key = f"queue_mixes/{q.id}.m4a"
    db_session.add(QueueRender(
        queue_id=q.id,
        status=QueueRenderStatus.ready,
        rendered_audio_path=key,
    ))
    db_session.flush()

    from app.services.storage import get_storage
    asyncio.run(get_storage().write(key, b"m4a-placeholder"))

    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix/audio")
    assert r.status_code == 200
    assert r.content == b"m4a-placeholder"
    cd = r.headers.get("content-disposition", "")
    assert "attachment" in cd and "mix.m4a" in cd
