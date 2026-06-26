"""Tests for the mobile household-settings endpoints (the web-search toggle).

Validates the auth boundary that makes this endpoint safe for a household admin
(rather than a global superuser): allowlist-only keys, household-role checks, and
correct household-scoped writes. Hermetic — auth + settings are stubbed.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.deps import AuthenticatedUser, verify_user_jwt

HH = "hh-123"
BASE = f"/api/v0/mobile/household/{HH}/settings"
ROLE = "app.api.mobile_household_settings.verify_household_role"
SETTINGS = "app.api.mobile_household_settings.get_settings_service"


@pytest.fixture
def client():
    app.dependency_overrides[verify_user_jwt] = lambda: AuthenticatedUser(user_id=1)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _settings_stub(value):
    svc = MagicMock()
    svc.get.return_value = value
    svc.set.return_value = True
    return MagicMock(return_value=svc)


def test_get_returns_default_off(client):
    # No override row → code default (False). No seed data needed.
    with patch(ROLE), patch(SETTINGS, _settings_stub(False)):
        r = client.get(BASE)
    assert r.status_code == 200
    assert r.json()["settings"]["web_search.enabled"] is False


def test_put_enables_with_household_scope(client):
    stub = _settings_stub(False)
    with patch(ROLE) as role, patch(SETTINGS, stub):
        r = client.put(f"{BASE}/web_search.enabled", json={"value": True})
    assert r.status_code == 200
    assert r.json() == {"success": True, "key": "web_search.enabled", "value": True}
    stub.return_value.set.assert_called_once_with(
        "web_search.enabled", True, household_id=HH
    )
    # Write must require the household admin role.
    assert role.call_args.kwargs.get("required_role") == "admin"


def test_put_rejects_non_allowlisted_key_before_role_check(client):
    with patch(ROLE) as role, patch(SETTINGS, _settings_stub(False)):
        r = client.put(f"{BASE}/memory.enabled", json={"value": True})
    assert r.status_code == 404
    role.assert_not_called()  # allowlist is the security boundary, checked first


def test_put_forbidden_for_non_admin(client):
    def deny(*a, **k):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    with patch(ROLE, side_effect=deny), patch(SETTINGS, _settings_stub(False)) as s:
        r = client.put(f"{BASE}/web_search.enabled", json={"value": True})
    assert r.status_code == 403
    s.return_value.set.assert_not_called()  # no write when role check fails


def test_get_requires_membership(client):
    def deny(*a, **k):
        raise HTTPException(status_code=403, detail="Not a member")

    with patch(ROLE, side_effect=deny), patch(SETTINGS, _settings_stub(False)):
        r = client.get(BASE)
    assert r.status_code == 403
