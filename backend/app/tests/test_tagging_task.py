"""tag_audio Celery task tests.

Same pattern as test_separate_task: real DB, clean up by id at the end,
mock out the Modal call. The load-bearing property: tagging is a
nice-to-have — its failure must never touch Song.status.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.core.db import SessionLocal
from app.models import Analysis, Song, SongStatus


@pytest.fixture
def analyzed_song():
    sid_str = None
    with SessionLocal() as db:
        song = Song(
            youtube_video_id=f"tagtest-{id(object())}",
            title="T",
            artist=None,
            duration_seconds=10.0,
            thumbnail_url=None,
            audio_path="audio/fake.wav",
            status=SongStatus.ready,
        )
        db.add(song)
        db.commit()
        db.add(
            Analysis(
                song_id=song.id,
                bpm=120.0,
                key="C",
                camelot_key="8B",
                beat_grid=[],
                downbeats=[],
                sections=[],
                energy_curve=[],
            )
        )
        db.commit()
        sid_str = str(song.id)
    yield sid_str
    with SessionLocal() as db:
        song = db.get(Song, uuid.UUID(sid_str))
        if song is not None:
            db.delete(song)
        db.commit()


def test_tag_audio_writes_tags(analyzed_song: str):
    tags = {"genres": ["house"], "moods": ["energetic"]}
    remote_fn = MagicMock()
    remote_fn.remote.return_value = tags

    with patch("modal.Function.from_name", return_value=remote_fn):
        from app.workers.tagging import tag_audio_task

        tag_audio_task(analyzed_song)

    with SessionLocal() as db:
        row = db.query(Analysis).filter(
            Analysis.song_id == uuid.UUID(analyzed_song)
        ).one()
        assert row.tags == tags


def test_tag_audio_failure_retries_without_failing_song(analyzed_song: str):
    remote_fn = MagicMock()
    remote_fn.remote.side_effect = RuntimeError("modal boom")

    with patch("modal.Function.from_name", return_value=remote_fn):
        from app.workers.tagging import tag_audio_task

        # Called directly (no worker), self.retry re-raises the original
        # error rather than celery.exceptions.Retry.
        with pytest.raises(RuntimeError, match="modal boom"):
            tag_audio_task(analyzed_song)

    with SessionLocal() as db:
        song = db.get(Song, uuid.UUID(analyzed_song))
        assert song.status == SongStatus.ready
        assert song.error_text is None


def test_tag_audio_gives_up_quietly_after_max_retries(analyzed_song: str):
    remote_fn = MagicMock()
    remote_fn.remote.side_effect = RuntimeError("modal boom")

    from app.workers.tagging import tag_audio_task

    with patch("modal.Function.from_name", return_value=remote_fn):
        # Simulate the final retry's execution context.
        tag_audio_task.push_request(retries=tag_audio_task.max_retries)
        try:
            # Must not raise: the task gives up and leaves tags null.
            tag_audio_task.run(analyzed_song)
        finally:
            tag_audio_task.pop_request()

    with SessionLocal() as db:
        song = db.get(Song, uuid.UUID(analyzed_song))
        assert song.status == SongStatus.ready
        assert song.error_text is None
        row = db.query(Analysis).filter(
            Analysis.song_id == uuid.UUID(analyzed_song)
        ).one()
        assert row.tags is None


def test_tag_audio_missing_song_returns():
    from app.workers.tagging import tag_audio_task

    assert tag_audio_task(str(uuid.uuid4())) is None
