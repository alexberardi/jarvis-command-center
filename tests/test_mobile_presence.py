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
