"""POST /api/v0/mobile/presence — the phone-as-presence-producer bridge.

Pins the auth boundary (JWT membership check, presence self-scoped to the token's
user — never client-supplied), the state→kind mapping, and that it drives the same
fan-out as /signals. Hermetic — auth/DB/writer/fan-out are stubbed.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.deps import AuthenticatedUser, get_db, verify_user_jwt
from app.main import app

URL = "/api/v0/mobile/presence"
HH = "hh-123"
TOKEN_UID = 42
ROLE = "app.api.mobile_presence.verify_household_role"
REC = "app.api.mobile_presence.record_presence"
FANOUT = "app.api.mobile_presence.dispatch_signal_edges"
ENABLED = "app.api.mobile_presence._signals_enabled"
RATE = "app.api.mobile_presence._rate_ok"


@pytest.fixture(autouse=True)
def _gates_open():
    # Default the two Signal-Bus gates to open so the existing behavioural tests
    # exercise the happy path; the gate tests patch them explicitly.
    with patch(ENABLED, return_value=True), patch(RATE, return_value=True):
        yield


@pytest.fixture
def client():
    app.dependency_overrides[verify_user_jwt] = lambda: AuthenticatedUser(user_id=TOKEN_UID)
    app.dependency_overrides[get_db] = lambda: MagicMock()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _sig(kind="presence.seen"):
    return MagicMock(id=7, kind=kind)


def test_home_records_presence_scoped_to_token_user(client):
    rec = MagicMock(return_value=_sig("presence.seen"))
    with patch(ROLE) as role, patch(REC, rec), patch(FANOUT) as fan:
        r = client.post(URL, json={"household_id": HH, "state": "home", "room": "kitchen"})
    assert r.status_code == 200
    assert r.json()["kind"] == "presence.seen"
    # membership verified for the TOKEN user (not a client value) + the named household
    assert role.call_args.args[0] == TOKEN_UID
    assert role.call_args.args[1] == HH
    kw = rec.call_args.kwargs
    assert kw["user_id"] == TOKEN_UID and kw["household_id"] == HH
    assert kw["state"] == "home" and kw["source_agent"] == "mobile" and kw["room"] == "kitchen"
    fan.assert_called_once()  # drives the shared bus fan-out


def test_away_maps_to_presence_left(client):
    rec = MagicMock(return_value=_sig("presence.left"))
    with patch(ROLE), patch(REC, rec), patch(FANOUT):
        r = client.post(URL, json={"household_id": HH, "state": "away"})
    assert r.status_code == 200
    assert r.json()["kind"] == "presence.left"
    assert rec.call_args.kwargs["state"] == "away"


def test_invalid_state_is_422_and_writes_nothing(client):
    with patch(ROLE), patch(REC) as rec, patch(FANOUT):
        r = client.post(URL, json={"household_id": HH, "state": "banana"})
    assert r.status_code == 422
    rec.assert_not_called()


def test_non_member_is_403_and_writes_nothing(client):
    with patch(ROLE, side_effect=HTTPException(status_code=403, detail="nope")), \
         patch(REC) as rec, patch(FANOUT):
        r = client.post(URL, json={"household_id": HH, "state": "home"})
    assert r.status_code == 403
    rec.assert_not_called()


def test_user_id_cannot_be_spoofed_via_body(client):
    # A client-sent user_id is ignored; presence is always the JWT's user.
    rec = MagicMock(return_value=_sig())
    with patch(ROLE), patch(REC, rec), patch(FANOUT):
        r = client.post(URL, json={"household_id": HH, "state": "home", "user_id": 999})
    assert r.status_code == 200
    assert rec.call_args.kwargs["user_id"] == TOKEN_UID


def test_display_name_is_forwarded_for_summary(client):
    # The optional display name flows to the writer ("Alex is home"); it's
    # display-only and does not change the token-bound identity.
    rec = MagicMock(return_value=_sig())
    with patch(ROLE), patch(REC, rec), patch(FANOUT):
        r = client.post(URL, json={"household_id": HH, "state": "home", "name": "  Alex  "})
    assert r.status_code == 200
    assert rec.call_args.kwargs["name"] == "Alex"  # trimmed


def test_missing_or_blank_name_forwards_none(client):
    rec = MagicMock(return_value=_sig())
    with patch(ROLE), patch(REC, rec), patch(FANOUT):
        r = client.post(URL, json={"household_id": HH, "state": "home", "name": "   "})
    assert r.status_code == 200
    assert rec.call_args.kwargs["name"] is None


def test_disabled_bus_is_409_and_writes_nothing(client):
    # A household that turned the Signal Bus off must not receive phone presence.
    with patch(ROLE), patch(ENABLED, return_value=False), \
         patch(REC) as rec, patch(FANOUT) as fan:
        r = client.post(URL, json={"household_id": HH, "state": "home"})
    assert r.status_code == 409
    rec.assert_not_called()
    fan.assert_not_called()


def test_rate_limited_is_429_and_writes_nothing(client):
    with patch(ROLE), patch(RATE, return_value=False), \
         patch(REC) as rec, patch(FANOUT) as fan:
        r = client.post(URL, json={"household_id": HH, "state": "home"})
    assert r.status_code == 429
    rec.assert_not_called()
    fan.assert_not_called()
