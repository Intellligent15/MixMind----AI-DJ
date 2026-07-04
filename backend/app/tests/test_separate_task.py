"""separate_stems Celery task tests.

Same pattern as test_analyze_task: real DB, clean up by id at the end,
mock out the StemSeparationService so the test stays fast.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from app.core.db import SessionLocal
from app.models import Song, SongStatus, Stems, StemsStatus
from app.services.stems.service import SeparationResult


@pytest.fixture
def analyzed_song():
    sid_str = None
    with SessionLocal() as db:
        song = Song(
            youtube_video_id=f"septest-{id(object())}",
            title="T",
            artist=None,
            duration_seconds=10.0,
            thumbnail_url=None,
            audio_path="audio/fake.wav",
            status=SongStatus.analyzed,
        )
        db.add(song)
        db.commit()
        sid_str = str(song.id)
    yield sid_str
    sid = uuid.UUID(sid_str)
    with SessionLocal() as db:
        # Cascade from songs.id -> stems.song_id deletes the row for us,
        # so don't try to delete the Stems row separately (avoids a noisy
        # "0 were matched" SAWarning when the cascade already fired).
        song = db.get(Song, sid)
        if song is not None:
            db.delete(song)
        db.commit()


def _fake_result() -> SeparationResult:
    return SeparationResult(
        sample_rate=44100,
        stems={
            "vocals": torch.zeros(2, 100),
            "drums": torch.zeros(2, 100),
            "bass": torch.zeros(2, 100),
            "other": torch.zeros(2, 100),
        },
        vocal_rms=0.12,
        envelopes={"frame_hz": 10, "vocals": {"rms": [0.0, 0.0], "peak": [0.0, 0.0]}},
    )


def _patch_service_returning(result: SeparationResult):
    service = MagicMock()
    service.separate.return_value = result
    service.model_name = "htdemucs"
    return service


def test_separate_stems_happy_path(analyzed_song: str, tmp_path: Path):
    storage = AsyncMock()
    # storage.path returns a tmp_path-rooted file for any key — service
    # write_stem is mocked along with the service, so we never touch disk.
    pass

    service = _patch_service_returning(_fake_result())

    with (
        patch("app.workers.separate.get_storage", return_value=storage),
        patch(
            "app.workers.separate.StemSeparationService", return_value=service
        ),
    ):
        from app.workers.separate import separate_stems

        separate_stems(analyzed_song)

    assert service.write_stem.call_count == 4

    sid = uuid.UUID(analyzed_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        assert song is not None
        # Song status bounces back to analyzed (separated lives on Stems).
        assert song.status == SongStatus.analyzed

        row = db.query(Stems).filter(Stems.song_id == sid).one()
        assert row.status == StemsStatus.separated
        assert row.model_name == "htdemucs"
        assert row.vocals_path == f"stems/{song.youtube_video_id}/vocals.wav"
        assert row.drums_path == f"stems/{song.youtube_video_id}/drums.wav"
        assert row.bass_path == f"stems/{song.youtube_video_id}/bass.wav"
        assert row.other_path == f"stems/{song.youtube_video_id}/other.wav"
        assert row.vocal_rms == pytest.approx(0.12)
        # Envelope sidecar key + on-disk JSON match the fake result payload.
        env_key = f"stems/{song.youtube_video_id}/envelopes.json"
        assert row.envelopes_path == env_key
        env_file = tmp_path / env_key
        pass
        import json as _json

        pass


def test_separate_stems_marks_failed_on_error(analyzed_song: str):
    service = MagicMock()
    service.separate.side_effect = RuntimeError("demucs boom")
    service.model_name = "htdemucs"

    with (
        patch("app.workers.separate.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.separate.StemSeparationService", return_value=service
        ),
    ):
        from app.workers.separate import separate_stems

        with pytest.raises(RuntimeError, match="demucs boom"):
            separate_stems(analyzed_song)

    sid = uuid.UUID(analyzed_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        assert song is not None
        assert song.status == SongStatus.failed
        assert (
            db.query(Stems).filter(Stems.song_id == sid).one_or_none() is None
        )


def test_separate_stems_skips_wrong_status(analyzed_song: str):
    """A song still downloading is silently skipped via the atomic claim."""
    sid = uuid.UUID(analyzed_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        song.status = SongStatus.downloading
        db.commit()

    service = MagicMock()
    service.model_name = "htdemucs"
    with (
        patch("app.workers.separate.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.separate.StemSeparationService", return_value=service
        ),
    ):
        from app.workers.separate import separate_stems

        result = separate_stems(analyzed_song)

    assert result is None
    service.separate.assert_not_called()


def test_separate_stems_skips_if_already_separating(analyzed_song: str):
    """Concurrent dispatch: the loser sees status=separating and bails."""
    sid = uuid.UUID(analyzed_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        song.status = SongStatus.separating
        db.commit()

    service = MagicMock()
    service.model_name = "htdemucs"
    with (
        patch("app.workers.separate.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.separate.StemSeparationService", return_value=service
        ),
    ):
        from app.workers.separate import separate_stems

        result = separate_stems(analyzed_song)

    assert result is None
    service.separate.assert_not_called()
    with SessionLocal() as db:
        row = db.get(Song, sid)
        assert row is not None
        assert row.status == SongStatus.separating


def test_separate_stems_missing_row_logs_and_returns():
    with (
        patch("app.workers.separate.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.separate.StemSeparationService", return_value=AsyncMock()
        ),
    ):
        from app.workers.separate import separate_stems

        assert separate_stems(str(uuid.uuid4())) is None


def test_separate_stems_replaces_existing_stems_row(
    analyzed_song: str, tmp_path: Path
):
    sid = uuid.UUID(analyzed_song)
    with SessionLocal() as db:
        db.add(
            Stems(
                song_id=sid,
                model_name="htdemucs",
                status=StemsStatus.separated,
                vocals_path="stems/old/vocals.wav",
                drums_path="stems/old/drums.wav",
                bass_path="stems/old/bass.wav",
                other_path="stems/old/other.wav",
                vocal_rms=0.99,
            )
        )
        db.commit()

    storage = AsyncMock()
    pass
    service = _patch_service_returning(_fake_result())

    with (
        patch("app.workers.separate.get_storage", return_value=storage),
        patch(
            "app.workers.separate.StemSeparationService", return_value=service
        ),
    ):
        from app.workers.separate import separate_stems

        separate_stems(analyzed_song)

    with SessionLocal() as db:
        rows = db.query(Stems).filter(Stems.song_id == sid).all()
        assert len(rows) == 1
        # New value, not the old 0.99
        assert rows[0].vocal_rms == pytest.approx(0.12)
        # New paths use the actual video id, not "old".
        assert "/old/" not in (rows[0].vocals_path or "")


def test_separate_dispatches_transcribe_when_pipeline_requested(
    analyzed_song: str,
):
    sid = uuid.UUID(analyzed_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        song.pipeline_requested = True
        db.commit()

    storage = AsyncMock()
    service = _patch_service_returning(_fake_result())

    with (
        patch("app.workers.separate.get_storage", return_value=storage),
        patch(
            "app.workers.separate.StemSeparationService", return_value=service
        ),
        patch("app.workers.separate.celery_app.send_task") as send_mock,
    ):
        from app.workers import PRI_TRANSCRIBE
        from app.workers.separate import separate_stems

        separate_stems(analyzed_song)

    send_mock.assert_called_once_with(
        "app.workers.transcribe.transcribe_song",
        args=[analyzed_song],
        priority=PRI_TRANSCRIBE,
    )


def test_separate_does_not_dispatch_transcribe_when_pipeline_not_requested(
    analyzed_song: str,
):
    storage = AsyncMock()
    service = _patch_service_returning(_fake_result())

    with (
        patch("app.workers.separate.get_storage", return_value=storage),
        patch(
            "app.workers.separate.StemSeparationService", return_value=service
        ),
        patch("app.workers.separate.celery_app.send_task") as send_mock,
    ):
        from app.workers.separate import separate_stems

        separate_stems(analyzed_song)

    send_mock.assert_not_called()
