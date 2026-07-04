"""Reaction signals (F5): listener_events capture, position→transition
attribution, summary aggregation, and planner context wiring."""

from __future__ import annotations

import uuid

import pytest

from app.core.db import SessionLocal
from app.models import (
    Analysis,
    ListenerEvent,
    ListenerEventKind,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    QueueRender,
    QueueRenderStatus,
    Song,
    SongStatus,
)
from app.services.feedback.summary import feedback_context, style_tallies


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def mix_fixture():
    """Queue with two tagged songs, a styled MixPlan, and a QueueRender
    timeline placing their transition at [100, 120] output seconds."""
    ids: dict = {}
    with SessionLocal() as db:
        q = Queue(locked=True)
        db.add(q)
        db.flush()
        songs = []
        for i, genre in enumerate(("house", "techno")):
            s = Song(
                youtube_video_id=f"fb-{i}-{id(object())}",
                title=f"S{i}", duration_seconds=240.0,
                audio_path=f"audio/fb{i}.wav", status=SongStatus.ready,
            )
            db.add(s)
            db.flush()
            db.add(Analysis(
                song_id=s.id, bpm=120.0, key="C", camelot_key="8B",
                beat_grid=[], downbeats=[], sections=[], energy_curve=[],
                vocal_segments=[], tags={"genres": [genre], "moods": []},
            ))
            db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
            songs.append(s)
        plan = MixPlan(
            queue_id=q.id, from_song_id=songs[0].id, to_song_id=songs[1].id,
            status=MixPlanStatus.ready, style="wash_out",
        )
        db.add(plan)
        render = QueueRender(
            queue_id=q.id, status=QueueRenderStatus.ready,
            rendered_audio_path="queue_mixes/x.m4a",
            timeline={
                "duration": 300.0,
                "songs": [],
                "transitions": [{
                    "index": 0,
                    "from_song_id": str(songs[0].id),
                    "to_song_id": str(songs[1].id),
                    "start": 100.0, "end": 120.0, "label": "Reverb wash",
                    "stems": [], "effects": [], "reasoning": None,
                }],
            },
        )
        db.add(render)
        db.commit()
        ids = {
            "queue_id": str(q.id),
            "song_ids": [str(s.id) for s in songs],
            "plan_id": str(plan.id),
        }
    yield ids
    with SessionLocal() as db:
        db.query(ListenerEvent).delete()
        for sid in ids["song_ids"]:
            s = db.get(Song, uuid.UUID(sid))
            if s is not None:
                db.delete(s)
        q = db.get(Queue, uuid.UUID(ids["queue_id"]))
        if q is not None:
            db.delete(q)
        db.commit()


def test_thumbs_feedback_denormalizes_style_and_genres(mix_fixture, client):
    resp = client.post("/api/feedback", json={
        "queue_id": mix_fixture["queue_id"],
        "from_song_id": mix_fixture["song_ids"][0],
        "to_song_id": mix_fixture["song_ids"][1],
        "kind": "thumbs_up",
    })
    assert resp.status_code == 201, resp.text
    assert resp.json()["style"] == "wash_out"
    with SessionLocal() as db:
        ev = db.query(ListenerEvent).one()
        assert ev.kind == ListenerEventKind.thumbs_up
        assert ev.style == "wash_out"
        assert ev.from_genre == "house"
        assert ev.to_genre == "techno"
        assert ev.mix_plan_id == uuid.UUID(mix_fixture["plan_id"])


def test_thumbs_rejects_skip_kind(mix_fixture, client):
    resp = client.post("/api/feedback", json={
        "queue_id": mix_fixture["queue_id"],
        "from_song_id": mix_fixture["song_ids"][0],
        "to_song_id": mix_fixture["song_ids"][1],
        "kind": "skip",
    })
    assert resp.status_code == 422


def test_skip_attributed_via_timeline(mix_fixture, client):
    # 130 s = 10 s after the transition end -> still attributed.
    resp = client.post("/api/feedback/playback", json={
        "queue_id": mix_fixture["queue_id"],
        "kind": "skip",
        "position_seconds": 130.0,
    })
    assert resp.status_code == 201
    assert resp.json()["attributed_style"] == "wash_out"

    # 250 s is nowhere near a transition -> stored unattributed.
    resp = client.post("/api/feedback/playback", json={
        "queue_id": mix_fixture["queue_id"],
        "kind": "skip",
        "position_seconds": 250.0,
    })
    assert resp.json()["attributed_style"] is None


def test_summary_aggregation_and_context_line(mix_fixture, client):
    for kind in ("thumbs_up", "thumbs_up", "thumbs_down"):
        client.post("/api/feedback", json={
            "queue_id": mix_fixture["queue_id"],
            "from_song_id": mix_fixture["song_ids"][0],
            "to_song_id": mix_fixture["song_ids"][1],
            "kind": kind,
        })
    client.post("/api/feedback/playback", json={
        "queue_id": mix_fixture["queue_id"],
        "kind": "skip", "position_seconds": 110.0,
    })

    with SessionLocal() as db:
        tallies = style_tallies(db)
        assert tallies["wash_out"].up == 2
        assert tallies["wash_out"].down == 1
        assert tallies["wash_out"].skips == 1
        line = feedback_context(db, "house", "techno")
        assert "wash_out +2/-1" in line
        assert "1 skips" in line
        assert "house→techno" in line

    resp = client.get("/api/feedback/summary")
    body = resp.json()
    assert body["styles"]["wash_out"]["up"] == 2
    assert "wash_out" in body["context_line"]


def test_empty_history_gives_no_context():
    with SessionLocal() as db:
        db.query(ListenerEvent).delete()
        db.commit()
        assert feedback_context(db) is None
