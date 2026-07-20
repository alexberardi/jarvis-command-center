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


# =============================================================================
# int coercion (prereq for the numeric phone_calls.* settings)
# =============================================================================


class TestIntCoercion:
    """_coerce("int") semantics + endpoint behavior with an int-typed key."""

    INT_ALLOWLIST = {
        "web_search.enabled": "bool",
        "phone_calls.plan_ttl_minutes": "int",
    }

    def test_coerce_int_accepts_int_str_and_whole_float(self):
        from app.api.mobile_household_settings import _coerce

        assert _coerce(20, "int") == 20
        assert _coerce("20", "int") == 20
        assert _coerce(" 45 ", "int") == 45
        assert _coerce(7.0, "int") == 7

    def test_coerce_int_rejects_garbage(self):
        from app.api.mobile_household_settings import _coerce

        for bad in ("twenty", "", 7.5, True, False, None, [1]):
            with pytest.raises((ValueError, TypeError)):
                _coerce(bad, "int")

    def test_put_int_setting_writes_coerced_value(self, client):
        stub = _settings_stub("20")
        with patch(ROLE), patch(SETTINGS, stub), patch(
            "app.api.mobile_household_settings.HOUSEHOLD_CONTROLLABLE_SETTINGS",
            self.INT_ALLOWLIST,
        ):
            r = client.put(f"{BASE}/phone_calls.plan_ttl_minutes", json={"value": "45"})
        assert r.status_code == 200
        assert r.json()["value"] == 45
        stub.return_value.set.assert_called_once_with(
            "phone_calls.plan_ttl_minutes", 45, household_id=HH
        )

    def test_put_int_setting_rejects_garbage_with_400(self, client):
        stub = _settings_stub("20")
        with patch(ROLE), patch(SETTINGS, stub), patch(
            "app.api.mobile_household_settings.HOUSEHOLD_CONTROLLABLE_SETTINGS",
            self.INT_ALLOWLIST,
        ):
            r = client.put(
                f"{BASE}/phone_calls.plan_ttl_minutes", json={"value": "twenty"}
            )
        assert r.status_code == 400
        stub.return_value.set.assert_not_called()

    def test_put_int_setting_rejects_bool_with_400(self, client):
        stub = _settings_stub("20")
        with patch(ROLE), patch(SETTINGS, stub), patch(
            "app.api.mobile_household_settings.HOUSEHOLD_CONTROLLABLE_SETTINGS",
            self.INT_ALLOWLIST,
        ):
            r = client.put(
                f"{BASE}/phone_calls.plan_ttl_minutes", json={"value": True}
            )
        assert r.status_code == 400
        stub.return_value.set.assert_not_called()

    def test_get_returns_none_for_uncoercible_stored_value(self, client):
        # One corrupt row must not 500 the whole settings screen.
        stub = _settings_stub("garbage")
        with patch(ROLE), patch(SETTINGS, stub), patch(
            "app.api.mobile_household_settings.HOUSEHOLD_CONTROLLABLE_SETTINGS",
            {"phone_calls.plan_ttl_minutes": "int"},
        ):
            r = client.get(BASE)
        assert r.status_code == 200
        assert r.json()["settings"]["phone_calls.plan_ttl_minutes"] is None


class TestStringSettings:
    """`household.location` is the first string-typed household setting —
    getting it wrong calls the wrong business, so the household owns it."""

    def test_location_is_allowlisted_as_string(self):
        from app.api.mobile_household_settings import HOUSEHOLD_CONTROLLABLE_SETTINGS

        assert HOUSEHOLD_CONTROLLABLE_SETTINGS["household.location"] == "string"

    def test_coerce_passes_strings_through_untouched(self):
        from app.api.mobile_household_settings import _coerce

        assert _coerce("Brick, NJ 08724", "string") == "Brick, NJ 08724"
        assert _coerce("", "string") == ""
        # No truthiness games, no int parsing: a bare ZIP stays a string.
        assert _coerce("08724", "string") == "08724"

    def test_put_string_setting_writes_verbatim(self, client):
        stub = _settings_stub("")
        with patch(ROLE), patch(SETTINGS, stub):
            r = client.put(f"{BASE}/household.location", json={"value": "Brick, NJ"})
        assert r.status_code == 200
        assert r.json()["value"] == "Brick, NJ"
        stub.return_value.set.assert_called_once_with(
            "household.location", "Brick, NJ", household_id=HH
        )

    def test_location_definition_exists_with_empty_default(self):
        from app.services.settings_definitions import SETTINGS_DEFINITIONS

        d = next(x for x in SETTINGS_DEFINITIONS if x.key == "household.location")
        assert d.value_type == "string"
        assert d.default == ""
