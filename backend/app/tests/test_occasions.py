"""Occasion presets + arc templates (F7/F10)."""

from __future__ import annotations

import pytest

from app.services.mixer.occasions import (
    ARC_TEMPLATES,
    OCCASIONS,
    arc_targets_for_pairs,
    occasion_context,
)
from app.services.mixer.decision import TransitionStyle


def test_presets_reference_only_legal_styles():
    legal = {s.value for s in TransitionStyle}
    for occ in OCCASIONS.values():
        for token in occ.style_bias.replace(",", " ").replace(";", " ").split():
            if "_" in token and token in legal | {"style"}:
                assert token in legal
        assert 0.0 <= occ.energy_floor <= occ.energy_ceiling <= 1.0
        assert occ.default_arc is None or occ.default_arc in ARC_TEMPLATES


def test_arc_targets_interpolation():
    targets = arc_targets_for_pairs("slow_burn", 4)
    assert len(targets) == 4
    assert targets == sorted(targets)          # monotonic climb
    assert targets[0] == 0.25 and targets[-1] == 1.0
    assert arc_targets_for_pairs("slow_burn", 1) is not None
    assert arc_targets_for_pairs("nope", 4) is None


def test_occasion_context_lines():
    ctx = occasion_context("gym", "leg day, go hard")
    assert "sustained high energy" in ctx["occasion"]
    assert ctx["vibe_note"] == "leg day, go hard"
    assert occasion_context(None, None) == {}
    assert occasion_context("unknown_id", None) == {}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_queue_context_patch_roundtrip(client):
    import uuid as _uuid

    from app.core.db import SessionLocal
    from app.models import Queue

    with SessionLocal() as db:
        q = Queue(locked=False)
        db.add(q)
        db.commit()
        qid = str(q.id)
    try:
        resp = client.patch(f"/api/queues/{qid}", json={
            "occasion": "house_party", "vibe_note": "birthday",
            "arc_template": "peak_late", "tease_hooks": True,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["occasion"] == "house_party"
        assert body["vibe_note"] == "birthday"
        assert body["arc_template"] == "peak_late"
        assert body["tease_hooks"] is True

        # Unknown ids are rejected.
        assert client.patch(
            f"/api/queues/{qid}", json={"occasion": "rave_on_mars"}
        ).status_code == 422
        assert client.patch(
            f"/api/queues/{qid}", json={"arc_template": "spiral"}
        ).status_code == 422

        # Clearing works; locked queues 409.
        resp = client.patch(f"/api/queues/{qid}", json={"occasion": None})
        assert resp.json()["occasion"] is None
        with SessionLocal() as db:
            row = db.get(Queue, _uuid.UUID(qid))
            row.locked = True
            db.commit()
        assert client.patch(
            f"/api/queues/{qid}", json={"occasion": "gym"}
        ).status_code == 409
    finally:
        with SessionLocal() as db:
            row = db.get(Queue, _uuid.UUID(qid))
            if row is not None:
                db.delete(row)
            db.commit()


def test_context_options_endpoint(client):
    resp = client.get("/api/queues/meta/context_options")
    assert resp.status_code == 200
    body = resp.json()
    assert any(o["id"] == "gym" for o in body["occasions"])
    assert any(a["id"] == "wave" for a in body["arcs"])
