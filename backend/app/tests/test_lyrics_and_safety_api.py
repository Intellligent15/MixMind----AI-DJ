"""Tests for GET /api/songs/{id}/lyrics and
GET /api/songs/{id}/vocal_safe_regions."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.main import app
from app.models import (
    Lyrics,
    LyricsAlignmentStatus,
    LyricsFetchStatus,
    Song,
    SongStatus,
    Stems,
    StemsStatus,
    Transcription,
    TranscriptionStatus,
)
from app.services.storage import get_storage


def _client(db_session: Session) -> TestClient:
    def override_db():
        yield db_session
    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def _make_song(db: Session, vid: str = "lsa") -> Song:
    s = Song(
        youtube_video_id=f"{vid}-{uuid.uuid4().hex[:8]}",
        title="Test", artist="A",
        duration_seconds=10.0,
        audio_path=f"audio/{vid}.wav",
        status=SongStatus.ready,
    )
    db.add(s); db.flush()
    return s


# --- /lyrics ------------------------------------------------------------


def test_get_lyrics_404_song_not_found(db_session: Session):
    r = _client(db_session).get(f"/api/songs/{uuid.uuid4()}/lyrics")
    assert r.status_code == 404


def test_get_lyrics_404_when_no_row(db_session: Session):
    s = _make_song(db_session)
    r = _client(db_session).get(f"/api/songs/{s.id}/lyrics")
    assert r.status_code == 404


def test_get_lyrics_returns_row(db_session: Session):
    s = _make_song(db_session)
    db_session.add(Lyrics(
        song_id=s.id,
        genius_id=12345,
        text="Hello world",
        fetch_status=LyricsFetchStatus.success,
        alignment_status=LyricsAlignmentStatus.success,
        alignment_quality=1.0,
        aligned_words=[{
            "word": "Hello", "start": 0.0, "end": 0.5,
            "confidence": 0.9, "source": "whisper_match",
        }],
    ))
    db_session.flush()
    r = _client(db_session).get(f"/api/songs/{s.id}/lyrics")
    assert r.status_code == 200
    body = r.json()
    assert body["genius_id"] == 12345
    assert body["text"] == "Hello world"
    assert body["alignment_status"] == "success"
    assert len(body["aligned_words"]) == 1


# --- /vocal_safe_regions -----------------------------------------------


def _seed_stems_with_envelope(db: Session, s: Song, env: dict) -> Stems:
    key = f"stems/{s.youtube_video_id}/envelopes.json"
    asyncio.run(get_storage().write(key, json.dumps(env).encode()))
    stems = Stems(
        song_id=s.id,
        model_name="htdemucs",
        status=StemsStatus.separated,
        vocals_path=f"stems/{s.youtube_video_id}/vocals.wav",
        drums_path=f"stems/{s.youtube_video_id}/drums.wav",
        bass_path=f"stems/{s.youtube_video_id}/bass.wav",
        other_path=f"stems/{s.youtube_video_id}/other.wav",
        vocal_rms=0.15,
        envelopes_path=key,
    )
    db.add(stems); db.flush()
    return stems


def _seed_transcription(db: Session, s: Song, segments: list[dict]) -> Transcription:
    t = Transcription(
        song_id=s.id,
        model_name="large-v3",
        status=TranscriptionStatus.success,
        language="en",
        segments=segments,
        duration_seconds=s.duration_seconds,
    )
    db.add(t); db.flush()
    return t


def test_safe_regions_404_song_missing(db_session: Session):
    r = _client(db_session).get(f"/api/songs/{uuid.uuid4()}/vocal_safe_regions")
    assert r.status_code == 404


def test_safe_regions_409_when_not_processed(db_session: Session):
    s = _make_song(db_session)
    r = _client(db_session).get(f"/api/songs/{s.id}/vocal_safe_regions")
    assert r.status_code == 409


def test_safe_regions_returns_full_when_quiet(db_session: Session):
    s = _make_song(db_session)
    _seed_stems_with_envelope(db_session, s, {
        "frame_hz": 10, "rms": [0.001] * 100, "peak": [0.005] * 100,
    })
    _seed_transcription(db_session, s, segments=[])
    r = _client(db_session).get(f"/api/songs/{s.id}/vocal_safe_regions")
    assert r.status_code == 200
    regions = r.json()["regions"]
    assert len(regions) == 1
    assert regions[0]["safe"] is True


def test_safe_regions_uses_aligned_words_when_alignment_success(db_session: Session):
    s = _make_song(db_session)
    _seed_stems_with_envelope(db_session, s, {
        "frame_hz": 10,
        "rms": [0.001] * 50 + [0.05] * 5 + [0.001] * 45,
        "peak": [0.005] * 50 + [0.15] * 5 + [0.005] * 45,
    })
    _seed_transcription(db_session, s, segments=[])
    db_session.add(Lyrics(
        song_id=s.id,
        text="hello",
        fetch_status=LyricsFetchStatus.success,
        alignment_status=LyricsAlignmentStatus.success,
        alignment_quality=1.0,
        aligned_words=[{
            "word": "hello", "start": 5.0, "end": 5.5,
            "confidence": 0.9, "source": "whisper_match",
        }],
    ))
    db_session.flush()
    # Pin a small minimum so this 10s fixture exercises the aligned-words
    # split itself, not the (larger) default min-region threshold.
    r = _client(db_session).get(
        f"/api/songs/{s.id}/vocal_safe_regions?min_safe_region_seconds=1.5"
    )
    assert r.status_code == 200
    # The word at 5.0-5.5 splits the song into two safe regions.
    assert len(r.json()["regions"]) == 2


def test_safe_regions_ignores_aligned_when_low_quality(db_session: Session):
    """alignment_status != success → fall back to raw Whisper."""
    s = _make_song(db_session)
    _seed_stems_with_envelope(db_session, s, {
        "frame_hz": 10, "rms": [0.001] * 100, "peak": [0.005] * 100,
    })
    _seed_transcription(db_session, s, segments=[])
    db_session.add(Lyrics(
        song_id=s.id,
        text="hello",
        fetch_status=LyricsFetchStatus.success,
        alignment_status=LyricsAlignmentStatus.low_quality,
        alignment_quality=0.1,
        aligned_words=[{
            "word": "phantom", "start": 5.0, "end": 5.5,
            "confidence": 0.2, "source": "interpolated",
        }],
    ))
    db_session.flush()
    r = _client(db_session).get(f"/api/songs/{s.id}/vocal_safe_regions")
    assert r.status_code == 200
    # Bad alignment was ignored; raw Whisper has no words → whole song safe.
    assert len(r.json()["regions"]) == 1


def test_safe_regions_accepts_threshold_query_params(db_session: Session):
    s = _make_song(db_session)
    _seed_stems_with_envelope(db_session, s, {
        "frame_hz": 10, "rms": [0.001] * 100, "peak": [0.005] * 100,
    })
    _seed_transcription(db_session, s, segments=[])
    r = _client(db_session).get(
        f"/api/songs/{s.id}/vocal_safe_regions"
        "?min_safe_region_seconds=2.0&word_prob_min=0.5"
    )
    assert r.status_code == 200
