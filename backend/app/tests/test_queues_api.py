from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.main import app
from app.models import (
    Queue,
    QueueItem,
    Song,
    SongStatus,
    Stems,
    StemsStatus,
    Transcription,
    TranscriptionStatus,
)


def _client(db_session: Session) -> TestClient:
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def _make_song(db: Session, vid: str, status: SongStatus = SongStatus.pending) -> Song:
    song = Song(
        youtube_video_id=vid,
        title=f"song-{vid}",
        artist="art",
        duration_seconds=120.0,
        thumbnail_url=None,
        audio_path=f"audio/{vid}.wav" if status != SongStatus.pending else None,
        status=status,
    )
    db.add(song)
    db.flush()
    return song


# --- create / current --------------------------------------------------------


def test_create_queue(db_session: Session):
    client = _client(db_session)
    r = client.post("/api/queues")
    assert r.status_code == 201
    body = r.json()
    assert body["locked"] is False
    assert body["items"] == []
    assert body["locked_at"] is None


def test_create_queue_replaces_prior_queues(db_session: Session):
    """Phase 8 change: creating a new queue is destructive. Every prior
    Queue and its rendered mix get deleted so the player can transition
    cleanly to the new set."""
    client = _client(db_session)
    first = client.post("/api/queues")
    assert first.status_code == 201
    first_id = first.json()["id"]

    second = client.post("/api/queues")
    assert second.status_code == 201
    second_id = second.json()["id"]
    assert second_id != first_id

    # /current returns ONLY the new queue.
    current = client.get("/api/queues/current")
    assert current.status_code == 200
    assert current.json()["id"] == second_id


def test_create_queue_deletes_rendered_mix_file(db_session: Session):
    """When a prior Queue had a rendered M4A, create_queue tells the
    storage backend to delete it."""
    from unittest.mock import AsyncMock, patch

    from app.models import Queue, QueueRender, QueueRenderStatus

    prior = Queue(locked=True)
    db_session.add(prior)
    db_session.flush()
    db_session.add(QueueRender(
        queue_id=prior.id,
        status=QueueRenderStatus.ready,
        rendered_audio_path="queue_mixes/old.m4a",
    ))
    db_session.flush()

    client = _client(db_session)
    storage = AsyncMock()
    storage.delete = AsyncMock()
    with patch("app.services.storage.get_storage", return_value=storage):
        r = client.post("/api/queues")
    assert r.status_code == 201
    storage.delete.assert_awaited_once_with("queue_mixes/old.m4a")


def test_create_queue_deletes_per_pair_mix_renders(db_session: Session):
    """create_queue must also delete each prior MixPlan's rendered WAV
    (mixes/<id>.wav) before the cascade removes the rows — otherwise the
    per-pair transition renders orphan in object storage."""
    from unittest.mock import AsyncMock, patch

    from app.models import MixPlan, MixPlanStatus, Queue, QueueRender, QueueRenderStatus

    prior = Queue(locked=True)
    db_session.add(prior)
    db_session.flush()
    s1 = _make_song(db_session, "mixclean1")
    s2 = _make_song(db_session, "mixclean2")
    db_session.add(QueueRender(
        queue_id=prior.id,
        status=QueueRenderStatus.ready,
        rendered_audio_path="queue_mixes/old.m4a",
    ))
    db_session.add(MixPlan(
        queue_id=prior.id,
        from_song_id=s1.id, to_song_id=s2.id,
        status=MixPlanStatus.ready,
        rendered_audio_path="mixes/pair.wav",
        plan_json=[],
    ))
    db_session.flush()

    client = _client(db_session)
    storage = AsyncMock()
    storage.delete = AsyncMock()
    with patch("app.services.storage.get_storage", return_value=storage):
        r = client.post("/api/queues")
    assert r.status_code == 201
    deleted = {c.args[0] for c in storage.delete.await_args_list}
    assert "queue_mixes/old.m4a" in deleted
    assert "mixes/pair.wav" in deleted


def test_get_current_prefers_unlocked(db_session: Session):
    locked = Queue(locked=True)
    unlocked = Queue(locked=False)
    db_session.add_all([locked, unlocked])
    db_session.flush()
    client = _client(db_session)
    r = client.get("/api/queues/current")
    assert r.status_code == 200
    assert r.json()["id"] == str(unlocked.id)


def test_get_current_falls_back_to_locked(db_session: Session):
    locked = Queue(locked=True)
    db_session.add(locked)
    db_session.flush()
    client = _client(db_session)
    r = client.get("/api/queues/current")
    assert r.status_code == 200
    assert r.json()["id"] == str(locked.id)


def test_get_current_404_when_none(db_session: Session):
    client = _client(db_session)
    r = client.get("/api/queues/current")
    assert r.status_code == 404


# --- add items ---------------------------------------------------------------


def test_add_item_appends_in_position_order(db_session: Session):
    queue = Queue()
    s1 = _make_song(db_session, "v1")
    s2 = _make_song(db_session, "v2")
    db_session.add(queue)
    db_session.flush()
    client = _client(db_session)
    with patch("app.api.queues.download_song.delay"):
        r1 = client.post(
            f"/api/queues/{queue.id}/items", json={"song_id": str(s1.id)}
        )
        r2 = client.post(
            f"/api/queues/{queue.id}/items", json={"song_id": str(s2.id)}
        )
    assert r1.status_code == 201
    assert r2.status_code == 201
    positions = [item["position"] for item in r2.json()["items"]]
    assert positions == [0, 1]
    song_ids = [item["song"]["id"] for item in r2.json()["items"]]
    assert song_ids == [str(s1.id), str(s2.id)]


def test_add_item_allows_duplicate_song(db_session: Session):
    queue = Queue()
    song = _make_song(db_session, "dup")
    db_session.add(queue)
    db_session.flush()
    client = _client(db_session)
    with patch("app.api.queues.download_song.delay"):
        client.post(f"/api/queues/{queue.id}/items", json={"song_id": str(song.id)})
        r = client.post(
            f"/api/queues/{queue.id}/items", json={"song_id": str(song.id)}
        )
    assert r.status_code == 201
    items = r.json()["items"]
    assert len(items) == 2
    assert items[0]["song"]["id"] == items[1]["song"]["id"]
    assert items[0]["id"] != items[1]["id"]


def test_add_item_does_not_dispatch_download(db_session: Session):
    # POST /api/songs already dispatches download_song on song creation, and
    # POST /lock catches anything still pending. The queue add handler must
    # not re-dispatch — that previously raced with the songs-API dispatch.
    queue = Queue()
    pending = _make_song(db_session, "pending-vid")
    downloaded = _make_song(db_session, "downloaded-vid", SongStatus.downloaded)
    db_session.add(queue)
    db_session.flush()
    client = _client(db_session)
    with patch("app.api.queues.download_song") as download_mock:
        client.post(
            f"/api/queues/{queue.id}/items", json={"song_id": str(pending.id)}
        )
        client.post(
            f"/api/queues/{queue.id}/items", json={"song_id": str(downloaded.id)}
        )
    download_mock.delay.assert_not_called()
    download_mock.s.assert_not_called()


def test_add_item_409_when_locked(db_session: Session):
    queue = Queue(locked=True)
    song = _make_song(db_session, "v1")
    db_session.add(queue)
    db_session.flush()
    client = _client(db_session)
    r = client.post(f"/api/queues/{queue.id}/items", json={"song_id": str(song.id)})
    assert r.status_code == 409


def test_add_item_404_unknown_song(db_session: Session):
    queue = Queue()
    db_session.add(queue)
    db_session.flush()
    client = _client(db_session)
    r = client.post(
        f"/api/queues/{queue.id}/items",
        json={"song_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert r.status_code == 404


def test_add_item_409_when_full(db_session: Session):
    queue = Queue()
    db_session.add(queue)
    db_session.flush()
    songs = [_make_song(db_session, f"v{i}") for i in range(21)]
    client = _client(db_session)
    with patch("app.api.queues.download_song.delay"):
        for s in songs[:20]:
            r = client.post(
                f"/api/queues/{queue.id}/items", json={"song_id": str(s.id)}
            )
            assert r.status_code == 201
        r = client.post(
            f"/api/queues/{queue.id}/items", json={"song_id": str(songs[20].id)}
        )
    assert r.status_code == 409


# --- remove ------------------------------------------------------------------


def test_remove_item_compacts_positions(db_session: Session):
    queue = Queue()
    db_session.add(queue)
    songs = [_make_song(db_session, f"r{i}") for i in range(3)]
    db_session.flush()
    items = [
        QueueItem(queue_id=queue.id, song_id=s.id, position=i)
        for i, s in enumerate(songs)
    ]
    db_session.add_all(items)
    db_session.flush()

    client = _client(db_session)
    r = client.delete(f"/api/queues/{queue.id}/items/{items[0].id}")
    assert r.status_code == 200
    positions = [item["position"] for item in r.json()["items"]]
    assert positions == [0, 1]
    remaining_song_ids = [item["song"]["id"] for item in r.json()["items"]]
    assert remaining_song_ids == [str(songs[1].id), str(songs[2].id)]


def test_remove_item_409_when_locked(db_session: Session):
    queue = Queue(locked=True)
    db_session.add(queue)
    song = _make_song(db_session, "vL")
    db_session.flush()
    item = QueueItem(queue_id=queue.id, song_id=song.id, position=0)
    db_session.add(item)
    db_session.flush()
    client = _client(db_session)
    r = client.delete(f"/api/queues/{queue.id}/items/{item.id}")
    assert r.status_code == 409


# --- reorder -----------------------------------------------------------------


def test_reorder_items(db_session: Session):
    queue = Queue()
    db_session.add(queue)
    songs = [_make_song(db_session, f"o{i}") for i in range(3)]
    db_session.flush()
    items = [
        QueueItem(queue_id=queue.id, song_id=s.id, position=i)
        for i, s in enumerate(songs)
    ]
    db_session.add_all(items)
    db_session.flush()

    reversed_ids = [str(items[2].id), str(items[1].id), str(items[0].id)]
    client = _client(db_session)
    r = client.patch(
        f"/api/queues/{queue.id}/items",
        json={"ordered_item_ids": reversed_ids},
    )
    assert r.status_code == 200
    body_ids = [item["id"] for item in r.json()["items"]]
    assert body_ids == reversed_ids
    positions = [item["position"] for item in r.json()["items"]]
    assert positions == [0, 1, 2]


def test_reorder_400_on_mismatched_ids(db_session: Session):
    queue = Queue()
    db_session.add(queue)
    songs = [_make_song(db_session, f"m{i}") for i in range(2)]
    db_session.flush()
    items = [
        QueueItem(queue_id=queue.id, song_id=s.id, position=i)
        for i, s in enumerate(songs)
    ]
    db_session.add_all(items)
    db_session.flush()

    client = _client(db_session)
    # Drop one item id
    r = client.patch(
        f"/api/queues/{queue.id}/items",
        json={"ordered_item_ids": [str(items[0].id)]},
    )
    assert r.status_code == 400


# --- lock --------------------------------------------------------------------


def _add_stems_row(db: Session, song: Song) -> Stems:
    row = Stems(
        song_id=song.id,
        model_name="htdemucs",
        status=StemsStatus.separated,
        vocals_path=f"stems/{song.youtube_video_id}/vocals.wav",
        drums_path=f"stems/{song.youtube_video_id}/drums.wav",
        bass_path=f"stems/{song.youtube_video_id}/bass.wav",
        other_path=f"stems/{song.youtube_video_id}/other.wav",
        vocal_rms=0.1,
    )
    db.add(row)
    db.flush()
    return row


def _add_transcription_row(db: Session, song: Song) -> Transcription:
    row = Transcription(
        song_id=song.id,
        model_name="large-v3",
        status=TranscriptionStatus.success,
        language="en",
        segments=[],
        vocal_rms_threshold=0.005,
        vocal_rms_observed=0.1,
        duration_seconds=1.0,
    )
    db.add(row)
    db.flush()
    return row


def test_lock_fans_out_pipeline_by_song_status(db_session: Session):
    queue = Queue()
    db_session.add(queue)
    pending = _make_song(db_session, "p1")
    downloaded = _make_song(db_session, "d1", SongStatus.downloaded)
    # analyzed + no stems -> separate + transcribe
    analyzed_no_stems = _make_song(db_session, "a1", SongStatus.analyzed)
    # analyzed + stems, no transcription -> transcribe only
    analyzed_with_stems = _make_song(db_session, "a2", SongStatus.analyzed)
    _add_stems_row(db_session, analyzed_with_stems)
    # ready + stems + transcription -> nothing to do
    done = _make_song(db_session, "z1", SongStatus.ready)
    _add_stems_row(db_session, done)
    _add_transcription_row(db_session, done)
    db_session.flush()
    for i, s in enumerate(
        [pending, downloaded, analyzed_no_stems, analyzed_with_stems, done]
    ):
        db_session.add(QueueItem(queue_id=queue.id, song_id=s.id, position=i))
    db_session.flush()

    from app.workers import (
        PRI_ANALYZE, PRI_DOWNLOAD, PRI_SEPARATE, PRI_TRANSCRIBE,
    )
    client = _client(db_session)
    with patch("app.api.queues.analyze_song") as analyze_mock, patch(
        "app.api.queues.download_song"
    ) as download_mock, patch(
        "app.api.queues.separate_stems"
    ) as separate_mock, patch(
        "app.api.queues.transcribe_song"
    ) as transcribe_mock:
        r = client.post(f"/api/queues/{queue.id}/lock")

    assert r.status_code == 202
    assert r.json()["locked"] is True
    assert r.json()["locked_at"] is not None

    # New (post-Option-B) shape: lock kicks ONE stage per song; the worker
    # auto-chain (gated by pipeline_requested) carries the song through
    # the rest. pending → download; downloaded → analyze; analyzed (no
    # stems) → separate; analyzed (stems, no transcription) → transcribe;
    # ready+stems+transcription → nothing.
    download_mock.apply_async.assert_called_once_with(
        args=[str(pending.id)], priority=PRI_DOWNLOAD
    )
    analyze_mock.apply_async.assert_called_once_with(
        args=[str(downloaded.id)], priority=PRI_ANALYZE
    )
    separate_mock.apply_async.assert_called_once_with(
        args=[str(analyzed_no_stems.id)], priority=PRI_SEPARATE
    )
    transcribe_mock.apply_async.assert_called_once_with(
        args=[str(analyzed_with_stems.id)], priority=PRI_TRANSCRIBE
    )

    # All queued songs must have pipeline_requested=True after lock so
    # the worker auto-chain carries them forward.
    for s in [pending, downloaded, analyzed_no_stems, analyzed_with_stems, done]:
        db_session.refresh(s)
        assert s.pipeline_requested is True

    # Phase 7: lock also seeds N-1 MixPlan rows for adjacent pairs in the
    # queue. plan_json is null at seed time (lazy generation at render).
    from sqlalchemy import select

    from app.models import MixPlan

    songs_in_order = [pending, downloaded, analyzed_no_stems, analyzed_with_stems, done]
    rows = list(
        db_session.scalars(
            select(MixPlan).where(MixPlan.queue_id == queue.id)
        )
    )
    assert len(rows) == len(songs_in_order) - 1
    pos_by_song = {s.id: i for i, s in enumerate(songs_in_order)}
    for row in rows:
        assert pos_by_song[row.to_song_id] - pos_by_song[row.from_song_id] == 1
        assert row.plan_json is None
        assert row.rendered_audio_path is None


def test_lock_409_if_empty(db_session: Session):
    queue = Queue()
    db_session.add(queue)
    db_session.flush()
    client = _client(db_session)
    r = client.post(f"/api/queues/{queue.id}/lock")
    assert r.status_code == 409


def test_lock_with_in_flight_song_sets_flag_without_dispatching(
    db_session: Session,
):
    """Bug B regression: previously _enqueue_pipeline_for_song had no
    branch for `downloading`/`analyzing`/etc., so locking a queue while
    a song was mid-download silently dropped the chain. The fix relies
    on workers' auto-chain (gated by pipeline_requested) to carry the
    song forward when the in-flight task completes — so at lock time we
    must NOT redundantly re-dispatch the running stage, but we MUST set
    pipeline_requested=True so the running worker auto-dispatches the
    next stage on success."""
    queue = Queue()
    db_session.add(queue)
    in_flight = _make_song(db_session, "if1", SongStatus.downloading)
    db_session.flush()
    db_session.add(QueueItem(queue_id=queue.id, song_id=in_flight.id, position=0))
    db_session.flush()

    client = _client(db_session)
    with patch("app.api.queues.download_song") as dl, patch(
        "app.api.queues.analyze_song"
    ) as an, patch("app.api.queues.separate_stems") as sp, patch(
        "app.api.queues.transcribe_song"
    ) as tr:
        r = client.post(f"/api/queues/{queue.id}/lock")

    assert r.status_code == 202
    # No redundant dispatches — the running download will auto-chain on
    # success because pipeline_requested is now True.
    dl.apply_async.assert_not_called()
    an.apply_async.assert_not_called()
    sp.apply_async.assert_not_called()
    tr.apply_async.assert_not_called()
    db_session.refresh(in_flight)
    assert in_flight.pipeline_requested is True


def test_lock_409_if_already_locked(db_session: Session):
    queue = Queue(locked=True)
    db_session.add(queue)
    song = _make_song(db_session, "L1")
    db_session.flush()
    db_session.add(QueueItem(queue_id=queue.id, song_id=song.id, position=0))
    db_session.flush()
    client = _client(db_session)
    r = client.post(f"/api/queues/{queue.id}/lock")
    assert r.status_code == 409


def test_lock_404_unknown_queue(db_session: Session):
    client = _client(db_session)
    r = client.post("/api/queues/00000000-0000-0000-0000-000000000000/lock")
    assert r.status_code == 404


def test_add_item_kicks_pipeline_eagerly(db_session: Session):
    """Queueing a song signals intent to mix: pipeline_requested flips on
    immediately (so the in-flight download auto-chains into analysis) and
    an already-downloaded song gets its analyze dispatched right away —
    that's what unlocks Suggest Order before the queue is locked."""
    queue = Queue()
    pending = _make_song(db_session, "eager-pending-vid")
    downloaded = _make_song(db_session, "eager-dl-vid", SongStatus.downloaded)
    db_session.add(queue)
    db_session.flush()
    client = _client(db_session)
    with (
        patch("app.api.queues.download_song") as download_mock,
        patch("app.api.queues.analyze_song") as analyze_mock,
    ):
        client.post(
            f"/api/queues/{queue.id}/items", json={"song_id": str(pending.id)}
        )
        client.post(
            f"/api/queues/{queue.id}/items", json={"song_id": str(downloaded.id)}
        )
    # Fresh song: flag only — its already-dispatched download will chain.
    download_mock.apply_async.assert_not_called()
    assert pending.pipeline_requested is True
    # Downloaded song: analysis starts now, not at lock time.
    analyze_mock.apply_async.assert_called_once()
    assert analyze_mock.apply_async.call_args.kwargs.get("args") \
        or analyze_mock.apply_async.call_args[1].get("args") \
        or analyze_mock.apply_async.call_args[0]
    assert downloaded.pipeline_requested is True
