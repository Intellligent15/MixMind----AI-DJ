"""Auto track ordering (F8): cost function properties, exact search, and
the suggest_order API endpoint."""

from __future__ import annotations

import uuid

import pytest

from app.services.mixer.ordering import (
    Edge,
    PairInput,
    edge_info,
    order_cost,
    pair_cost,
    suggest_order,
)


def _song(
    sid: str,
    bpm: float = 120.0,
    key: str = "C",
    camelot: str = "8B",
    energy: list[float] | None = None,
    genres: tuple[str, ...] = (),
) -> PairInput:
    return PairInput(
        song_id=sid, title=sid, bpm=bpm, key=key, camelot_key=camelot,
        energy_curve=energy if energy is not None else [0.5] * 200,
        genres=genres,
    )


def test_identical_songs_cost_near_zero():
    a = _song("a", genres=("house",))
    b = _song("b", genres=("house",))
    assert pair_cost(a, b) < 0.1


def test_clash_and_gap_cost_dominates():
    smooth = pair_cost(_song("a"), _song("b"))
    rough = pair_cost(
        _song("a", bpm=120.0, key="C", camelot="8B"),
        _song("b", bpm=150.0, key="F#", camelot="2B"),  # clash + 25% gap
    )
    assert rough > smooth + 3.0


def test_halftime_pair_scores_as_compatible():
    cost_half = pair_cost(_song("a", bpm=85.0), _song("b", bpm=170.0))
    cost_plain = pair_cost(_song("a", bpm=85.0), _song("b", bpm=120.0))
    assert cost_half < cost_plain
    edge = edge_info(_song("a", bpm=85.0), _song("b", bpm=170.0))
    assert "half-time" in edge.reason


def test_edge_grades():
    assert edge_info(_song("a", genres=("house",)),
                     _song("b", genres=("house",))).grade == "A"
    assert edge_info(
        _song("a", key="C", camelot="8B"),
        _song("b", key="F#", camelot="2B", bpm=150.0),
    ).grade == "C"


def test_suggest_order_fixes_a_bad_middle():
    """Keys: 8B / 3B / 9B — the user sandwiched the clashing 3B between
    two compatible tracks. Pinning first, the best order pushes 3B last
    (one bad edge instead of two)."""
    songs = [
        _song("s0", key="C", camelot="8B"),
        _song("s1", key="Db", camelot="3B"),
        _song("s2", key="G", camelot="9B"),
    ]
    order, edges = suggest_order(songs, pin_first=True)
    assert order[0] == 0                       # opener pinned
    assert order == [0, 2, 1]
    assert order_cost(songs, order) < order_cost(songs, [0, 1, 2])
    assert len(edges) == 2
    assert isinstance(edges[0], Edge)


def test_suggest_order_beats_greedy_on_crafted_instance():
    """Greedy nearest-neighbour takes the cheap first hop (s0->s1) and
    pays for it later; Held-Karp finds the globally cheaper path."""
    # BPM chain: 120, 124, 100, 104, 128. Greedy from 120 grabs 124, then
    # is stranded relative to 128. Optimal keeps the two BPM clusters
    # contiguous: 120 -> 124 -> 128 stays smooth, then jump once to 100/104.
    songs = [
        _song("s0", bpm=120.0),
        _song("s1", bpm=124.0),
        _song("s2", bpm=100.0),
        _song("s3", bpm=104.0),
        _song("s4", bpm=128.0),
    ]
    order, _ = suggest_order(songs, pin_first=True)
    # One cluster jump only: the 100/104 pair is contiguous and terminal.
    pos = {songs[i].song_id: k for k, i in enumerate(order)}
    assert abs(pos["s2"] - pos["s3"]) == 1
    assert pos["s4"] < min(pos["s2"], pos["s3"]) or \
        pos["s1"] > max(pos["s2"], pos["s3"])
    # And it's at least as cheap as the user's order.
    assert order_cost(songs, order) <= order_cost(songs, list(range(5))) + 1e-9


def test_suggest_order_small_queue_passthrough():
    songs = [_song("s0"), _song("s1")]
    order, edges = suggest_order(songs)
    assert order == [0, 1]
    assert len(edges) == 1


# ------------------------------------------------------------- API endpoint

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def analyzed_queue():
    from app.core.db import SessionLocal
    from app.models import Analysis, Queue, QueueItem, Song, SongStatus

    ids = {}
    with SessionLocal() as db:
        q = Queue(locked=False)
        db.add(q)
        db.flush()
        specs = [
            ("C", "8B", 120.0),
            ("Db", "3B", 121.0),   # the clashing middle
            ("G", "9B", 122.0),
        ]
        song_ids = []
        for i, (key, camelot, bpm) in enumerate(specs):
            s = Song(
                youtube_video_id=f"ord-{i}-{id(object())}",
                title=f"S{i}", duration_seconds=240.0,
                audio_path=f"audio/ord{i}.wav", status=SongStatus.ready,
            )
            db.add(s)
            db.flush()
            db.add(Analysis(
                song_id=s.id, bpm=bpm, key=key, camelot_key=camelot,
                beat_grid=[], downbeats=[], sections=[],
                energy_curve=[0.5] * 240, vocal_segments=[],
            ))
            db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
            song_ids.append(str(s.id))
        db.commit()
        ids = {"queue_id": str(q.id), "song_ids": song_ids}
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


def test_suggest_order_endpoint(analyzed_queue, client):
    resp = client.post(
        f"/api/queues/{analyzed_queue['queue_id']}/suggest_order"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    s0, s1, s2 = analyzed_queue["song_ids"]
    assert body["order"] == [s0, s2, s1]      # clash pushed to the end
    assert body["improved"] is True
    assert body["order"][0] == s0             # opener pinned
    assert len(body["ordered_item_ids"]) == 3
    assert body["edges"][0]["grade"] in ("A", "B")


def test_suggest_order_endpoint_409_when_unanalyzed(analyzed_queue, client):
    from app.core.db import SessionLocal
    import sqlalchemy as sa

    with SessionLocal() as db:
        db.execute(sa.text(
            "DELETE FROM analyses WHERE song_id = :sid"
        ), {"sid": analyzed_queue["song_ids"][1]})
        db.commit()
    resp = client.post(
        f"/api/queues/{analyzed_queue['queue_id']}/suggest_order"
    )
    assert resp.status_code == 409
    # Names the song, not its UUID, and says what to do.
    assert "still analyzing" in resp.json()["detail"]
    assert "S1" in resp.json()["detail"]
