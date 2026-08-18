"""GET/PUT /api/v0/mobile/household/{id}/signal-automations.

Pins: the catalog is surfaced with observed + current-instruction annotations, the
auth boundary (member reads / admin writes), rejecting non-authorable kinds, the
read-modify-write on the JSON setting, and that a blank instruction clears a rule.
Hermetic — auth, the settings service, and the observed-kinds query are stubbed.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.deps import AuthenticatedUser, verify_user_jwt
from app.main import app

BASE = "/api/v0/mobile/household/hh-1/signal-automations"
TOKEN_UID = 42
MOD = "app.api.mobile_signal_automations"
ROLE = f"{MOD}.verify_household_role"
# Reads/writes go through the shared store, which imports get_settings_service
# function-locally — so patch it at the source module.
SETTINGS = "app.services.settings_service.get_settings_service"
OBSERVED = f"{MOD}._observed_kinds"


@pytest.fixture
def client():
    app.dependency_overrides[verify_user_jwt] = lambda: AuthenticatedUser(user_id=TOKEN_UID)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _settings(get_return="{}"):
    svc = MagicMock()
    svc.get.return_value = get_return
    svc.set.return_value = True
    return svc


# ── GET ──────────────────────────────────────────────────────────────────────
def test_get_lists_catalog_with_observed_and_current_rule(client):
    stored = json.dumps({"presence.left": {"instruction": "Lock the door", "enabled": True}})
    with patch(ROLE) as role, patch(SETTINGS, return_value=_settings(stored)), patch(
        OBSERVED, return_value={"presence.left"}
    ):
        r = client.get(BASE)
    assert r.status_code == 200
    role.assert_called_once()  # membership verified
    items = {a["kind"]: a for a in r.json()["automations"]}
    assert set(items) == {"presence.left", "presence.seen", "appt.upcoming"}
    left = items["presence.left"]
    assert left["instruction"] == "Lock the door" and left["enabled"] is True
    assert left["observed"] is True
    # A kind with no rule + never observed reads as blank + not observed.
    assert items["appt.upcoming"]["instruction"] == ""
    assert items["appt.upcoming"]["observed"] is False


def test_get_survives_unparseable_setting(client):
    with patch(ROLE), patch(SETTINGS, return_value=_settings("not json{")), patch(
        OBSERVED, return_value=set()
    ):
        r = client.get(BASE)
    assert r.status_code == 200
    assert all(a["instruction"] == "" for a in r.json()["automations"])


# ── PUT ──────────────────────────────────────────────────────────────────────
def test_put_sets_a_rule_via_read_modify_write(client):
    svc = _settings("{}")
    with patch(ROLE), patch(SETTINGS, return_value=svc):
        r = client.put(f"{BASE}/presence.left", json={"instruction": "Lock the door", "enabled": True})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["cleared"] is False
    saved = json.loads(svc.set.call_args.args[1])
    assert saved["presence.left"] == {"instruction": "Lock the door", "enabled": True}


def test_put_preserves_other_kinds_rules(client):
    existing = json.dumps({"presence.seen": {"instruction": "Lights on", "enabled": True}})
    svc = _settings(existing)
    with patch(ROLE), patch(SETTINGS, return_value=svc):
        r = client.put(f"{BASE}/presence.left", json={"instruction": "Lock up", "enabled": False})
    assert r.status_code == 200
    saved = json.loads(svc.set.call_args.args[1])
    assert saved["presence.seen"] == {"instruction": "Lights on", "enabled": True}
    assert saved["presence.left"] == {"instruction": "Lock up", "enabled": False}


def test_put_blank_instruction_clears_the_rule(client):
    existing = json.dumps({"presence.left": {"instruction": "Lock", "enabled": True}})
    svc = _settings(existing)
    with patch(ROLE), patch(SETTINGS, return_value=svc):
        r = client.put(f"{BASE}/presence.left", json={"instruction": "   ", "enabled": True})
    assert r.status_code == 200
    assert r.json()["cleared"] is True
    saved = json.loads(svc.set.call_args.args[1])
    assert "presence.left" not in saved


def test_put_rejects_non_authorable_kind_before_touching_settings(client):
    svc = _settings("{}")
    with patch(ROLE) as role, patch(SETTINGS, return_value=svc):
        r = client.put(f"{BASE}/leave_by.suggested", json={"instruction": "x"})
    assert r.status_code == 404
    svc.set.assert_not_called()
    role.assert_not_called()  # kind is validated before the (admin) auth check


def test_put_500_when_settings_write_fails(client):
    svc = _settings("{}")
    svc.set.return_value = False
    with patch(ROLE), patch(SETTINGS, return_value=svc):
        r = client.put(f"{BASE}/presence.left", json={"instruction": "Lock", "enabled": True})
    assert r.status_code == 500


def test_put_enforces_instruction_length(client):
    svc = _settings("{}")
    with patch(ROLE), patch(SETTINGS, return_value=svc):
        r = client.put(f"{BASE}/presence.left", json={"instruction": "x" * 501})
    # The app maps request-validation errors to 400 (custom handler), not 422.
    assert r.status_code == 400  # pydantic max_length rejected before any write
    svc.set.assert_not_called()
