"""Download task tests.

The Celery task opens its own SessionLocal, so these tests use the real DB
(not the rollback-on-teardown fixture) and clean up by id at the end.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.db import SessionLocal
from app.models import Song, SongStatus
from app.services.youtube.service import YouTubeDownloadError


@pytest.fixture
def song_id():
    with SessionLocal() as db:
        song = Song(
            youtube_video_id=f"test-{id(object())}",
            title="T",
            artist=None,
            duration_seconds=10.0,
            thumbnail_url=None,
            status=SongStatus.pending,
        )
        db.add(song)
        db.commit()
        sid = str(song.id)
    yield sid
    with SessionLocal() as db:
        row = db.get(Song, __import__("uuid").UUID(sid))
        if row is not None:
            db.delete(row)
            db.commit()


def test_download_song_happy_path(song_id: str, tmp_path: Path):
    dest = tmp_path / "audio" / "out.wav"

    storage = AsyncMock()
    storage.path.return_value = dest

    yt = MagicMock()

    def fake_download(video_id, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"RIFFwave")

    yt.download.side_effect = fake_download

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        result = download_song(song_id)

    with SessionLocal() as db:
        import uuid

        row = db.get(Song, uuid.UUID(song_id))
        assert row is not None
        assert row.status == SongStatus.downloaded
        assert row.audio_path == f"audio/{row.youtube_video_id}.wav"
        assert result == row.audio_path


def test_download_song_marks_failed_on_error(song_id: str, tmp_path: Path):
    storage = AsyncMock()
    storage.path.return_value = tmp_path / "out.wav"
    yt = MagicMock()
    yt.download.side_effect = YouTubeDownloadError("boom")

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        with pytest.raises(YouTubeDownloadError):
            download_song(song_id)

    with SessionLocal() as db:
        import uuid

        row = db.get(Song, uuid.UUID(song_id))
        assert row is not None
        assert row.status == SongStatus.failed
        assert row.audio_path is None


def test_download_song_missing_row_logs_and_returns():
    """A stale broker message for a deleted song must not crash the worker."""
    import uuid as _uuid

    with (
        patch("app.workers.download.get_storage", return_value=AsyncMock()),
        patch("app.workers.download.YouTubeService", return_value=AsyncMock()),
    ):
        from app.workers.download import download_song

        assert download_song(str(_uuid.uuid4())) is None


def test_download_song_skips_if_already_downloading(song_id: str):
    """Concurrent dispatch: the loser sees status=downloading and bails."""
    import uuid as _uuid

    # Pre-mark the row as downloading to simulate another task winning the
    # atomic claim first.
    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        row.status = SongStatus.downloading
        db.commit()

    yt = MagicMock()
    with (
        patch("app.workers.download.get_storage", return_value=AsyncMock()),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        result = download_song(song_id)

    assert result is None
    yt.download.assert_not_called()
    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        # Status unchanged: the loser did not touch it.
        assert row.status == SongStatus.downloading


def test_download_dispatches_analyze_when_pipeline_requested(
    song_id: str, tmp_path: Path
):
    """Auto-chain: with pipeline_requested=True, a successful download
    enqueues analyze_song at PRI_ANALYZE so it cuts ahead of other
    songs' downloads on the priority queue."""
    import uuid as _uuid

    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        row.pipeline_requested = True
        db.commit()

    storage = AsyncMock()
    storage.path.return_value = tmp_path / "audio" / "out.wav"
    yt = MagicMock()

    def fake_download(video_id, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"RIFFwave")

    yt.download.side_effect = fake_download

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
        patch("app.workers.download.analyze_song") as analyze_mock,
    ):
        from app.workers import PRI_ANALYZE
        from app.workers.download import download_song

        download_song(song_id)

    analyze_mock.apply_async.assert_called_once_with(
        args=[song_id], priority=PRI_ANALYZE
    )


def test_download_does_not_dispatch_analyze_when_pipeline_not_requested(
    song_id: str, tmp_path: Path
):
    """Library-added song (pipeline_requested=False) stops at downloaded."""
    storage = AsyncMock()
    storage.path.return_value = tmp_path / "audio" / "out.wav"
    yt = MagicMock()

    def fake_download(video_id, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"RIFFwave")

    yt.download.side_effect = fake_download

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
        patch("app.workers.download.analyze_song") as analyze_mock,
    ):
        from app.workers.download import download_song

        download_song(song_id)

    analyze_mock.apply_async.assert_not_called()


def test_download_song_skips_if_already_downloaded(song_id: str):
    """Re-dispatch after success is a no-op (no re-download)."""
    import uuid as _uuid

    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        row.status = SongStatus.downloaded
        row.audio_path = f"audio/{row.youtube_video_id}.wav"
        db.commit()

    yt = MagicMock()
    with (
        patch("app.workers.download.get_storage", return_value=AsyncMock()),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        result = download_song(song_id)

    assert result is not None and result.endswith(".wav")
    yt.download.assert_not_called()
    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        assert row.status == SongStatus.downloaded


def test_is_transient_classifier():
    from app.workers.download import _is_transient

    assert _is_transient(YouTubeDownloadError(
        "ERROR: unable to download video data: HTTP Error 403: Forbidden"))
    assert _is_transient(YouTubeDownloadError("HTTP Error 429: Too Many Requests"))
    assert _is_transient(YouTubeDownloadError("The read operation timed out"))
    assert not _is_transient(YouTubeDownloadError("Private video"))
    assert not _is_transient(YouTubeDownloadError("Video unavailable"))
    assert not _is_transient(YouTubeDownloadError("boom"))


def test_download_song_retries_transient_403(song_id: str, tmp_path: Path):
    """A 403 resets the song to `pending` (so the retried task can win the
    claim again) and schedules a Celery retry instead of marking failed."""
    storage = AsyncMock()
    storage.path.return_value = tmp_path / "out.wav"
    yt = MagicMock()
    yt.download.side_effect = YouTubeDownloadError(
        "ERROR: unable to download video data: HTTP Error 403: Forbidden"
    )

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        # Direct (eager) invocation makes self.retry re-raise the original
        # exception rather than celery.exceptions.Retry.
        with pytest.raises(YouTubeDownloadError):
            download_song(song_id)

    with SessionLocal() as db:
        import uuid

        row = db.get(Song, uuid.UUID(song_id))
        assert row.status == SongStatus.pending   # claimable by the retry
        assert row.error_text is None


def test_download_song_403_fails_after_max_retries(song_id: str, tmp_path: Path):
    storage = AsyncMock()
    storage.path.return_value = tmp_path / "out.wav"
    yt = MagicMock()
    yt.download.side_effect = YouTubeDownloadError("HTTP Error 403: Forbidden")

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        # Simulate the final retry's execution context.
        download_song.push_request(retries=download_song.max_retries)
        try:
            with pytest.raises(YouTubeDownloadError):
                download_song.run(song_id)
        finally:
            download_song.pop_request()

    with SessionLocal() as db:
        import uuid

        row = db.get(Song, uuid.UUID(song_id))
        assert row.status == SongStatus.failed
        assert "403" in (row.error_text or "")


def test_download_song_permanent_error_fails_immediately(song_id: str, tmp_path: Path):
    storage = AsyncMock()
    storage.path.return_value = tmp_path / "out.wav"
    yt = MagicMock()
    yt.download.side_effect = YouTubeDownloadError("Video unavailable")

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        with pytest.raises(YouTubeDownloadError):
            download_song(song_id)

    assert yt.download.call_count == 1   # no retries burned on a dead video
    with SessionLocal() as db:
        import uuid

        row = db.get(Song, uuid.UUID(song_id))
        assert row.status == SongStatus.failed
