"""Mobile schedules screen endpoints — list + cancel a household's schedules.

User-JWT auth (overridden), household role checked inside the handler (patched), and a
real in-memory SQLite schedules table so the actual list/cancel SQL runs.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.deps import AuthenticatedUser, verify_user_jwt
from app.main import app
from app.models import Base, Schedule


def _sessionmaker():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[Schedule.__table__])
    return sessionmaker(bind=engine)


def _seed(Session, **kw) -> str:
    db = Session()
    row = Schedule(**{
        "household_id": "hh-1", "node_id": "n", "user_id": 7, "intent": "check the weather",
        "timezone": "UTC", "next_fire_at": datetime.utcnow() + timedelta(days=1),
        "state": "active", **kw,
    })
    db.add(row)
    db.commit()
    sid = row.id
    db.close()
    return sid


@pytest.fixture
def client():
    app.dependency_overrides[verify_user_jwt] = lambda: AuthenticatedUser(
        user_id=7, email="a@b.c", is_superuser=False)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_list_returns_enriched_schedules(client):
    Session = _sessionmaker()
    _seed(Session, recurrence=json.dumps({"type": "cron", "cron": "0 9 * * *"}))
    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.api.mobile_schedules.verify_household_role"):
        r = client.get("/api/v0/mobile/household/hh-1/schedules")
    assert r.status_code == 200
    body = r.json()
    assert body["household_id"] == "hh-1"
    s = body["schedules"][0]
    assert s["intent"] == "check the weather"
    assert s["is_recurring"] is True
    assert s["cadence"] == "every day at 9:00 AM"   # CC-owned display string
    assert s["next_local"]                          # a rendered local time
    assert "every day at 9:00 AM" in s["description"]


def test_list_only_live_schedules(client):
    Session = _sessionmaker()
    _seed(Session, state="done")
    _seed(Session, state="cancelled")
    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.api.mobile_schedules.verify_household_role"):
        r = client.get("/api/v0/mobile/household/hh-1/schedules")
    assert r.json()["schedules"] == []  # terminal ones are hidden


def test_cancel_stops_and_returns_fresh_list(client):
    Session = _sessionmaker()
    sid = _seed(Session)
    _seed(Session, intent="dentist")
    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.api.mobile_schedules.verify_household_role"):
        r = client.post(f"/api/v0/mobile/household/hh-1/schedules/{sid}/cancel")
    body = r.json()
    assert body["cancelled"] is True
    intents = [s["intent"] for s in body["schedules"]]
    assert "check the weather" not in intents and "dentist" in intents  # in-place refresh


def test_cancel_across_households_is_a_noop(client):
    Session = _sessionmaker()
    sid = _seed(Session, household_id="hh-1")
    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.api.mobile_schedules.verify_household_role"):
        r = client.post(f"/api/v0/mobile/household/hh-2/schedules/{sid}/cancel")
    assert r.json()["cancelled"] is False  # can't cancel another household's schedule


def test_requires_authentication():
    # No verify_user_jwt override (this test doesn't use the fixture) → the real dep
    # runs and a missing Bearer token is a 401.
    r = TestClient(app).get("/api/v0/mobile/household/hh-1/schedules")
    assert r.status_code == 401
