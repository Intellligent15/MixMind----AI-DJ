"""stitch_queue Celery task tests. Builds a locked queue with rendered
MixPlan WAVs, mocks storage so dummy stereo audio stands in for the
real per-pair transitions, then runs stitch_queue end-to-end and
asserts on QueueRender's terminal state."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    QueueRender,
    QueueRenderStatus,
    Song,
    SongStatus,
)


# --- Helpers ------------------------------------------------------------


def _make_analysis(song_id, bpm: float = 120.0, key: str = "C") -> Analysis:
    return Analysis(
        song_id=song_id,
        bpm=bpm,
        key=key,
        camelot_key="8B",
        time_signature=4,
        beat_grid=[i * 0.5 for i in range(360)],
        downbeats=[i * 2.0 for i in range(90)],
        sections=[
            {"start": 0.0, "end": 30.0, "label": "intro"},
            {"start": 30.0, "end": 150.0, "label": "body"},
            {"start": 150.0, "end": 180.0, "label": "outro"},
        ],
        energy_curve=[0.5] * 180,
        vocal_segments=[],
    )


def _seed_locked_queue(
    n_songs: int = 3,
    mix_plan_status: MixPlanStatus = MixPlanStatus.ready,
    include_tempo_ramp: bool = False,
    include_queue_render: bool = True,
) -> str:
    """Build N-song locked queue with N-1 MixPlans and (optionally) a
    pending QueueRender row. Returns the queue id."""
    with SessionLocal() as db:
        q = Queue(locked=True)
        db.add(q)
        db.flush()

        songs = []
        for i in range(n_songs):
            s = Song(
                youtube_video_id=f"sq-{i}-{uuid.uuid4().hex}",
                title=f"Song {i}",
                duration_seconds=180.0,
                audio_path=f"audio/{i}.wav",
                status=SongStatus.ready,
            )
            db.add(s)
            songs.append(s)
        db.flush()

        for i, s in enumerate(songs):
            db.add(_make_analysis(s.id, bpm=120.0 + i * 5))
            db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
        db.flush()

        for i in range(n_songs - 1):
            plan_json: list[dict] = [{
                "tool": "set_transition_window",
                "from_song_time_start": 150.0,
                "to_song_time_start": 30.0,
                "duration_bars": 16,
            }]
            if include_tempo_ramp:
                plan_json.append({
                    "tool": "set_tempo_ramp",
                    "song": "B",
                    "start_time": 62.0,
                    "end_time": 78.0,
                    "start_bpm": 120.0,
                    "end_bpm": 120.0 + (i + 1) * 5,
                })
            db.add(MixPlan(
                queue_id=q.id,
                from_song_id=songs[i].id,
                to_song_id=songs[i + 1].id,
                status=mix_plan_status,
                rendered_audio_path=(
                    f"mixes/mp{i}.wav"
                    if mix_plan_status == MixPlanStatus.ready
                    else None
                ),
                plan_json=plan_json,
            ))

        if include_queue_render:
            db.add(QueueRender(queue_id=q.id, status=QueueRenderStatus.pending))
        db.commit()
        return str(q.id)


def _storage_writing_stereo_wav(samples_per_pair: int = 44100) -> AsyncMock:
    """Storage mock whose download_file writes a fixed-length stereo
    WAV at the destination path. AsyncMock so call assertions work."""
    storage = AsyncMock()

    async def _download_side(_key, dest):
        sf.write(str(dest), np.zeros((samples_per_pair, 2), dtype=np.float32), 44100)

    storage.download_file = AsyncMock(side_effect=_download_side)
    storage.write = AsyncMock(return_value="/abs/out")
    return storage


# --- Tests --------------------------------------------------------------


@pytest.fixture
def locked_queue_with_mixes():
    yield _seed_locked_queue(n_songs=3, include_tempo_ramp=True)


def test_stitch_queue_happy_path(locked_queue_with_mixes):
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(locked_queue_with_mixes)

    assert res == locked_queue_with_mixes
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(locked_queue_with_mixes)
        ))
        assert qr.status == QueueRenderStatus.ready
        assert qr.rendered_audio_path == f"queue_mixes/{locked_queue_with_mixes}.m4a"
        assert qr.error_text is None
        # Phase 10: the player timeline is persisted alongside the M4A.
        tl = qr.timeline
        assert tl is not None
        assert len(tl["songs"]) == 3
        assert len(tl["transitions"]) == 2
        for t in tl["transitions"]:
            assert t["from_song_id"] and t["to_song_id"]
            assert t["label"]  # human-readable technique
            assert t["end"] >= t["start"]


def test_stitch_queue_atomic_claim_skips_when_already_rendering():
    queue_id = _seed_locked_queue(n_songs=2)
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(queue_id)
        ))
        qr.status = QueueRenderStatus.rendering
        db.commit()

    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)

    assert res is None
    storage.download_file.assert_not_called()
    storage.write.assert_not_called()


def test_stitch_queue_fails_when_only_one_song():
    queue_id = _seed_locked_queue(n_songs=1)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)

    assert res is None
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(queue_id)
        ))
        assert qr.status == QueueRenderStatus.failed
        assert "at least 2 songs" in (qr.error_text or "")


def test_stitch_queue_fails_when_mix_plan_not_ready():
    queue_id = _seed_locked_queue(n_songs=2, mix_plan_status=MixPlanStatus.pending)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)

    assert res is None
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(queue_id)
        ))
        assert qr.status == QueueRenderStatus.failed
        assert "MixPlan" in (qr.error_text or "")


def test_stitch_queue_warns_when_no_queue_render_row():
    """Queue without an explicit QueueRender row → no-op, no crash."""
    queue_id = _seed_locked_queue(n_songs=2, include_queue_render=False)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)
    assert res is None


def test_stitch_queue_persists_m4a_key_and_calls_storage_write():
    queue_id = _seed_locked_queue(n_songs=3)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        stitch_queue(queue_id)

    expected_key = f"queue_mixes/{queue_id}.m4a"
    assert storage.write.call_count == 1
    call = storage.write.call_args
    assert call.args[0] == expected_key
    assert len(call.args[1]) > 0  # non-empty bytes


def test_stitch_queue_handles_tempo_ramp_in_plan_json():
    queue_id = _seed_locked_queue(n_songs=3, include_tempo_ramp=True)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)

    assert res == queue_id
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(queue_id)
        ))
        assert qr.status == QueueRenderStatus.ready
