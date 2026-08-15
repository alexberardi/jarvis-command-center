"""Endpoint tests for POST /api/v0/signals — the dual-auth open/directed ingress.

SQLite + FastAPI dependency overrides (house style from tests/test_memories_authz.py).
Locks the Phase 1 ingress contract: node/app auth + anti-spoof, open vs directed
wiring, node-bounded directed refusal, the per-household enable gate, the
in-memory rate limit, and the cacheable write-boundary rejection.

The directed *card emission* (`_emit_directed_proposal`) is a patched seam here —
the real proposable-card path is wired with the matcher in Phase 3.
"""
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.models import Base, Signal
from app.deps import get_db
from app.api import signals as signals_api
from app.api.signals import _verify_signals_auth, SignalsAuthContext


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db):
    """Client with get_db + auth overridden to a node in household hh-1."""
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[_verify_signals_auth] = lambda: SignalsAuthContext(
        auth_type="node", household_id="hh-1"
    )
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def raw_client(db):
    """Client with only get_db overridden — exercises the real _verify_signals_auth."""
    app.dependency_overrides[get_db] = lambda: db
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _open_body(**over):
    body = {
        "signal": {
            "kind": "presence.seen", "source_key": "presence:alex",
            "summary": "Alex is home", "ttl_seconds": 900,
        },
        "data": {"user": "alex"},
    }
    body.update(over)
    return body


class TestPersist:
    def test_open_persists_with_derived_household(self, client, db):
        r = client.post("/api/v0/signals", json=_open_body())
        assert r.status_code == 200
        assert r.json()["mode"] == "open"
        assert db.query(Signal).count() == 1
        row = db.query(Signal).first()
        assert row.household_id == "hh-1"       # derived from node auth, not the body
        assert row.source_key == "presence:alex"

    def test_open_defers_to_matcher_no_proposal(self, client):
        with patch.object(signals_api, "_emit_directed_proposal") as emit:
            r = client.post("/api/v0/signals", json=_open_body())
        assert r.status_code == 200
        assert r.json()["proposed"] is False
        emit.assert_not_called()                # open mode never emits directly


class TestDirectedWiring:
    def test_directed_calls_emit_seam(self, client):
        with patch.object(signals_api, "_emit_directed_proposal", return_value=True) as emit:
            r = client.post("/api/v0/signals", json=_open_body(command="add_event"))
        assert r.status_code == 200
        assert r.json()["mode"] == "directed"
        assert r.json()["proposed"] is True
        emit.assert_called_once()

    def test_directed_non_advertised_refused_but_persisted(self, client, db):
        # seam returns False (command not advertised by the node) → no card,
        # but the Signal itself is still recorded.
        with patch.object(signals_api, "_emit_directed_proposal", return_value=False):
            r = client.post("/api/v0/signals", json=_open_body(command="unlock_door"))
        assert r.status_code == 200
        assert r.json()["proposed"] is False
        assert db.query(Signal).count() == 1


class TestAuth:
    def test_no_credentials_401(self, raw_client):
        r = raw_client.post("/api/v0/signals", json=_open_body())
        assert r.status_code == 401

    def test_node_household_mismatch_403(self, raw_client):
        # node authenticates as hh-1 but the body claims hh-2 → anti-spoof 403
        fake_ctx = MagicMock(household_id="hh-1")
        with patch("app.api.signals.verify_api_key", return_value=fake_ctx):
            r = raw_client.post(
                "/api/v0/signals",
                headers={"X-Api-Key": "node-1:key"},
                json=_open_body(household_id="hh-2"),
            )
        assert r.status_code == 403


class TestGates:
    def test_signals_disabled_409(self, client):
        with patch.object(signals_api, "_signals_enabled", return_value=False):
            r = client.post("/api/v0/signals", json=_open_body())
        assert r.status_code == 409

    def test_rate_limited_429(self, client):
        with patch.object(signals_api, "_rate_ok", return_value=False):
            r = client.post("/api/v0/signals", json=_open_body())
        assert r.status_code == 429

    def test_cacheable_invalid_422(self, client):
        body = _open_body()
        body["signal"]["cacheable"] = True
        body["signal"]["summary"] = "It's 72.5 degrees right now"
        r = client.post("/api/v0/signals", json=body)
        assert r.status_code == 422
