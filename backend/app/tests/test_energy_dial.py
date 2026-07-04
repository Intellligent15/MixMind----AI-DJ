"""Live energy dial (F6): future-pair selection, bias stamping, and the
re-render dispatch."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

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


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def playing_mix():
    """3 songs, 2 ready MixPlans, a rendered timeline with transitions at
    [100, 120] and [400, 420] output seconds."""
    ids: dict = {}
    with SessionLocal() as db:
        q = Queue(locked=True)
        db.add(q)
        db.flush()
        songs = []
        for i in range(3):
            s = Song(
                youtube_video_id=f"dial-{i}-{id(object())}",
                title=f"S{i}", duration_seconds=240.0,
                audio_path=f"audio/d{i}.wav", status=SongStatus.ready,
            )
            db.add(s)
            db.flush()
            db.add(Analysis(
                song_id=s.id, bpm=120.0, key="C", camelot_key="8B",
                beat_grid=[], downbeats=[], sections=[], energy_curve=[],
                vocal_segments=[],
            ))
            db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
            songs.append(s)
        plans = []
        for i in range(2):
            mp = MixPlan(
                queue_id=q.id, from_song_id=songs[i].id,
                to_song_id=songs[i + 1].id, status=MixPlanStatus.ready,
                style="smooth_blend", plan_json=[{"tool": "x"}],
                rendered_audio_path=f"mixes/d{i}.wav",
            )
            db.add(mp)
            plans.append(mp)
        transitions = [
            {"index": i, "from_song_id": str(songs[i].id),
             "to_song_id": str(songs[i + 1].id),
             "start": 100.0 + i * 300.0, "end": 120.0 + i * 300.0,
             "label": "Crossfade", "stems": [], "effects": [],
             "reasoning": None}
            for i in range(2)
        ]
        db.add(QueueRender(
            queue_id=q.id, status=QueueRenderStatus.ready,
            rendered_audio_path="queue_mixes/d.m4a",
            timeline={"duration": 700.0, "songs": [], "transitions": transitions},
        ))
        db.commit()
        ids = {
            "queue_id": str(q.id),
            "plan_ids": [str(p.id) for p in plans],
            "song_ids": [str(s.id) for s in songs],
        }
    yield ids
    with SessionLocal() as db:
        for sid in ids["song_ids"]:
            s = db.get(Song, uuid.UUID(sid))
            if s is not None:
                db.delete(s)
        q = db.get(Queue, uuid.UUID(ids["queue_id"]))
        if q is not None:
            db.delete(q)
        db.commit()


def test_energy_dial_replans_only_future_pairs(playing_mix, client):
    # Playhead at 50 s: transition 0 starts at 100 (< 50+90=140 -> too
    # soon, keep it); transition 1 starts at 400 (>= 140 -> re-plan).
    with patch("app.workers.auto_stitch.reset_and_dispatch_stitch") as dispatch:
        resp = client.post(
            f"/api/queues/{playing_mix['queue_id']}/energy",
            json={"direction": "up", "position_seconds": 50.0},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["affected_transitions"] == [1]
    dispatch.assert_called_once()

    with SessionLocal() as db:
        near = db.get(MixPlan, uuid.UUID(playing_mix["plan_ids"][0]))
        far = db.get(MixPlan, uuid.UUID(playing_mix["plan_ids"][1]))
        # The imminent pair is untouched.
        assert near.status == MixPlanStatus.ready
        assert near.energy_bias is None
        assert near.plan_json is not None
        # The far pair was reset for a biased re-plan.
        assert far.status == MixPlanStatus.pending
        assert far.energy_bias == "up"
        assert far.plan_json is None
        assert far.rendered_audio_path is None
        assert far.reroll_nonce == 1


def test_energy_dial_hold_clears_bias(playing_mix, client):
    with SessionLocal() as db:
        far = db.get(MixPlan, uuid.UUID(playing_mix["plan_ids"][1]))
        far.energy_bias = "up"
        db.commit()
    with patch("app.workers.auto_stitch.reset_and_dispatch_stitch"):
        resp = client.post(
            f"/api/queues/{playing_mix['queue_id']}/energy",
            json={"direction": "hold", "position_seconds": 50.0},
        )
    assert resp.json()["affected_transitions"] == [1]
    with SessionLocal() as db:
        far = db.get(MixPlan, uuid.UUID(playing_mix["plan_ids"][1]))
        assert far.energy_bias is None


def test_energy_dial_noop_when_bias_unchanged(playing_mix, client):
    with patch("app.workers.auto_stitch.reset_and_dispatch_stitch") as dispatch:
        resp = client.post(
            f"/api/queues/{playing_mix['queue_id']}/energy",
            json={"direction": "hold", "position_seconds": 50.0},
        )
    assert resp.json()["affected_transitions"] == []
    dispatch.assert_not_called()


def test_energy_dial_409_while_stitching(playing_mix, client):
    with SessionLocal() as db:
        from sqlalchemy import select
        render = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(playing_mix["queue_id"])
        ))
        render.status = QueueRenderStatus.rendering
        db.commit()
    resp = client.post(
        f"/api/queues/{playing_mix['queue_id']}/energy",
        json={"direction": "up", "position_seconds": 50.0},
    )
    assert resp.status_code == 409
