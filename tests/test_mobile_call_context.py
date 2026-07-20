"""Mobile call-context editor API — the CC half the grid talks to.

USER-scoped (not household): the JWT's user is the entire scope, so these
tests never touch a household or a role. Storage is faked in memory; what's
real is the router, the JSON round-trip, and the coercion the write shares
with a read.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("JARVIS_AUTH_BASE_URL", "http://localhost:7701")

from app.deps import AuthenticatedUser, verify_user_jwt
from app.main import app
from app.services.call_context import SETTING_KEY, parse_call_context

USER = 7
BASE = "/api/v0/mobile/call-context"


class FakeSettings:
    """In-memory (key, user_id) store, matching the client's set/get shape."""

    def __init__(self):
        self.store: dict[tuple[str, int | None], str] = {}
        self.set_ok = True

    def get(self, key, default=None, household_id=None, node_id=None, user_id=None):
        return self.store.get((key, user_id), default)

    def set(self, key, value, household_id=None, node_id=None, user_id=None):
        if not self.set_ok:
            return False
        self.store[(key, user_id)] = value
        return True


@pytest.fixture
def settings(monkeypatch):
    fake = FakeSettings()

    # Patched only in the router's namespace, not globally — the app's startup
    # reads the real settings service (settings_service.definitions), so a
    # global swap would break app boot. The read still runs through the real
    # parse_call_context, and mirrors load_call_context's degrade-to-empty
    # contract so the outage path is genuinely exercised.
    def fake_load(user_id):
        try:
            return parse_call_context(fake.get(SETTING_KEY, user_id=user_id))
        except Exception:
            return []

    monkeypatch.setattr("app.api.mobile_call_context.load_call_context", fake_load)
    monkeypatch.setattr(
        "app.api.mobile_call_context.get_settings_service", lambda: fake
    )
    return fake


@pytest.fixture
def client(settings):
    app.dependency_overrides[verify_user_jwt] = lambda: AuthenticatedUser(
        user_id=USER, email="a@x", is_superuser=False
    )
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


class TestGet:
    def test_empty_store_returns_no_fields_but_a_full_catalog(self, client):
        body = client.get(BASE).json()
        assert body["fields"] == []
        # The grid needs the vocabulary even when the user has stored nothing.
        assert {c["value"] for c in body["catalog"]["categories"]}
        assert any(
            f["key"] == "insurance_member_id" for f in body["catalog"]["well_known"]
        )

    def test_returns_the_users_stored_fields(self, client, settings):
        settings.store[(SETTING_KEY, USER)] = json.dumps(
            {"fields": [{"key": "full_name", "value": "Alex B"}]}
        )
        body = client.get(BASE).json()
        assert [f["value"] for f in body["fields"]] == ["Alex B"]

    def test_a_settings_outage_yields_an_empty_grid_not_a_500(
        self, client, settings, monkeypatch
    ):
        """The screen must still open. load_call_context degrades to [] on any
        failure, so the grid is empty-but-editable rather than blocked."""
        def boom(*a, **k):
            raise RuntimeError("settings down")

        monkeypatch.setattr(settings, "get", boom)
        r = client.get(BASE)
        assert r.status_code == 200
        assert r.json()["fields"] == []


class TestPut:
    def test_saves_and_echoes_the_canonical_result(self, client, settings):
        r = client.put(
            BASE,
            json={"fields": [{"key": "full_name", "value": "Alex B"}]},
        )
        assert r.status_code == 200
        assert [f["value"] for f in r.json()["fields"]] == ["Alex B"]
        # It actually landed, as a JSON string a later read can parse.
        stored = settings.store[(SETTING_KEY, USER)]
        assert json.loads(stored)["fields"][0]["value"] == "Alex B"

    def test_a_custom_field_is_stored_with_a_derived_key(self, client, settings):
        r = client.put(
            BASE, json={"fields": [{"label": "Gate code", "value": "4417"}]}
        )
        (field,) = r.json()["fields"]
        assert field["key"] == "gate_code"
        assert field["tier"] == "if_asked"  # custom -> private by default

    def test_the_response_is_canonical_not_a_mirror(self, client, settings):
        """Blank rows and duplicates come back cleaned, so the grid re-renders
        exactly what a later call will see."""
        r = client.put(
            BASE,
            json={
                "fields": [
                    {"key": "full_name", "value": "Alex B"},
                    {"key": "full_name", "value": "Someone Else"},  # dup
                    {"label": "", "value": ""},                     # blank
                ]
            },
        )
        fields = r.json()["fields"]
        assert [f["value"] for f in fields] == ["Alex B"]

    def test_replaces_rather_than_merges(self, client, settings):
        client.put(BASE, json={"fields": [{"key": "full_name", "value": "Alex B"}]})
        client.put(
            BASE, json={"fields": [{"key": "callback_number", "value": "+15555550123"}]}
        )
        keys = [f["key"] for f in client.get(BASE).json()["fields"]]
        assert keys == ["callback_number"]

    def test_empty_list_clears_the_store(self, client, settings):
        client.put(BASE, json={"fields": [{"key": "full_name", "value": "Alex B"}]})
        r = client.put(BASE, json={"fields": []})
        assert r.json()["fields"] == []

    def test_a_failed_write_surfaces_as_500(self, client, settings):
        """A silent drop of PII the user believes is saved is the worse
        outcome — fail loudly."""
        settings.set_ok = False
        r = client.put(BASE, json={"fields": [{"key": "x", "value": "y"}]})
        assert r.status_code == 500
