"""Phonebook: auto-save after a call, CRUD from mobile, and DNC enforcement.

Before this, `phone_contacts` had no writer at all — `resolve_contact` could
only ever return None, so every confirm card asked for a number the user had
already supplied on the last call. These are the paths that fill it.
"""

import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("JARVIS_AUTH_BASE_URL", "http://localhost:7701")

from app.deps import AuthenticatedUser, get_db, verify_user_jwt
from app.main import app
from app.models import PhoneCallSession, PhoneContact
from app.services.phone_call_service import (
    _normalize_name,
    upsert_contact_from_call,
)

HH = "hh-phonebook"
OTHER_HH = "hh-other"
USER = 42
BASE = f"/api/v0/mobile/household/{HH}/phone-contacts"
ROLE = "app.api.mobile_phone_contacts.verify_household_role"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_session(db, *, state="done", **overrides) -> PhoneCallSession:
    now = datetime.utcnow()
    fields = dict(
        id=str(uuid.uuid4()),
        household_id=HH,
        user_id=USER,
        contact_name="Tony's Pizzeria",
        goal="Order a pizza",
        details="Large pepperoni.",
        resolved_number="+19085551234",
        dialed_number="+19085551234",
        line_type="landline",
        state=state,
        created_at=now,
        expires_at=now + timedelta(minutes=20),
    )
    fields.update(overrides)
    s = PhoneCallSession(**fields)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _mk_contact(db, *, household_id=HH, name="Tony's Pizzeria", **overrides):
    now = datetime.utcnow()
    fields = dict(
        id=str(uuid.uuid4()),
        household_id=household_id,
        name=name,
        normalized_name=_normalize_name(name),
        number="+19085551234",
        source="manual",
        do_not_call=False,
        created_at=now,
    )
    fields.update(overrides)
    c = PhoneContact(**fields)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@pytest.fixture
def client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_jwt():
        return AuthenticatedUser(user_id=USER, email="a@x", is_superuser=False)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_user_jwt] = override_jwt
    with patch(ROLE, return_value=None):
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Auto-save after a call
# ---------------------------------------------------------------------------


class TestAutoSave:
    def test_completed_call_creates_contact(self, test_db):
        s = _mk_session(test_db, contact_address="742 Evergreen Ave, Springfield, IL")
        contact = upsert_contact_from_call(test_db, s)

        assert contact is not None
        assert contact.name == "Tony's Pizzeria"
        assert contact.number == "+19085551234"
        assert contact.source == "call"
        assert contact.line_type == "landline"
        assert contact.address == "742 Evergreen Ave, Springfield, IL"
        assert contact.verified_at is not None
        assert contact.do_not_call is False

    @pytest.mark.parametrize("state", ["failed", "declined", "expired", "in_call"])
    def test_unsuccessful_calls_save_nothing(self, test_db, state):
        """A call that didn't complete proves nothing about the number."""
        s = _mk_session(test_db, state=state)
        assert upsert_contact_from_call(test_db, s) is None
        assert test_db.query(PhoneContact).count() == 0

    def test_user_corrected_number_wins(self, test_db):
        """The number the user fixed on the card is the best signal we have."""
        _mk_contact(test_db, number="+19085559999")
        s = _mk_session(
            test_db, resolved_number="+19085559999", dialed_number="+17325924183"
        )
        contact = upsert_contact_from_call(test_db, s)

        assert contact.number == "+17325924183"
        assert test_db.query(PhoneContact).count() == 1  # updated, not duplicated

    def test_do_not_call_is_never_cleared(self, test_db):
        """Calling a blocked business once must not un-block it."""
        _mk_contact(test_db, do_not_call=True)
        s = _mk_session(test_db)
        contact = upsert_contact_from_call(test_db, s)

        assert contact.do_not_call is True

    def test_repeat_calls_are_idempotent(self, test_db):
        s1 = _mk_session(test_db)
        s2 = _mk_session(test_db)
        upsert_contact_from_call(test_db, s1)
        upsert_contact_from_call(test_db, s2)

        assert test_db.query(PhoneContact).count() == 1

    def test_web_sourced_entry_promoted_to_call(self, test_db):
        """A completed call outranks a search guess as provenance."""
        _mk_contact(test_db, source="web")
        contact = upsert_contact_from_call(test_db, _mk_session(test_db))
        assert contact.source == "call"

    def test_manual_entry_keeps_its_address(self, test_db):
        _mk_contact(test_db, address="Curated address")
        s = _mk_session(test_db, contact_address="Scraped address")
        contact = upsert_contact_from_call(test_db, s)
        assert contact.address == "Curated address"

    def test_missing_number_saves_nothing(self, test_db):
        s = _mk_session(test_db, resolved_number=None, dialed_number=None)
        assert upsert_contact_from_call(test_db, s) is None

    def test_invalid_number_saves_nothing(self, test_db):
        s = _mk_session(test_db, dialed_number="911", resolved_number=None)
        assert upsert_contact_from_call(test_db, s) is None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestList:
    def test_lists_household_contacts_sorted_by_name(self, client, test_db):
        _mk_contact(test_db, name="Zeno's")
        _mk_contact(test_db, name="Acme Deli")
        r = client.get(BASE)

        assert r.status_code == 200
        names = [c["name"] for c in r.json()["contacts"]]
        assert names == ["Acme Deli", "Zeno's"]

    def test_other_households_not_visible(self, client, test_db):
        _mk_contact(test_db, household_id=OTHER_HH, name="Not Mine")
        r = client.get(BASE)
        assert r.json()["contacts"] == []

    def test_response_shape(self, client, test_db):
        _mk_contact(test_db, notes="Ask for Sal", line_type="landline")
        body = client.get(BASE).json()["contacts"][0]
        for key in (
            "id", "name", "number", "address", "source", "line_type",
            "do_not_call", "notes", "verified_at", "created_at",
        ):
            assert key in body


class TestCreate:
    def test_creates_and_normalizes_number(self, client, test_db):
        r = client.post(BASE, json={"name": "Tony's", "number": "(732) 592-4183"})

        assert r.status_code == 201, r.text
        body = r.json()
        assert body["number"] == "+17325924183"
        assert body["source"] == "manual"
        assert body["do_not_call"] is False

    def test_optional_fields_stored(self, client, test_db):
        r = client.post(
            BASE,
            json={
                "name": "Tony's",
                "number": "7325924183",
                "address": "742 Evergreen Ave",
                "notes": "Ask for Sal",
            },
        )
        assert r.json()["address"] == "742 Evergreen Ave"
        assert r.json()["notes"] == "Ask for Sal"

    @pytest.mark.parametrize(
        "number", ["911", "12345", "900-555-1212", "not a number", "+44 20 7946 0958"]
    )
    def test_invalid_numbers_rejected(self, client, test_db, number):
        r = client.post(BASE, json={"name": "Bad", "number": number})
        assert r.status_code == 400
        assert test_db.query(PhoneContact).count() == 0

    def test_duplicate_name_rejected(self, client, test_db):
        _mk_contact(test_db, name="Tony's Pizzeria")
        r = client.post(BASE, json={"name": "tonys pizzeria", "number": "7325924183"})
        assert r.status_code == 409

    def test_name_without_letters_rejected(self, client, test_db):
        r = client.post(BASE, json={"name": "!!!", "number": "7325924183"})
        assert r.status_code == 400


class TestUpdate:
    def test_updates_fields(self, client, test_db):
        c = _mk_contact(test_db)
        r = client.patch(
            f"{BASE}/{c.id}", json={"name": "Tony's Pizza Co", "notes": "New notes"}
        )
        assert r.status_code == 200
        assert r.json()["name"] == "Tony's Pizza Co"
        assert r.json()["notes"] == "New notes"

    def test_number_revalidated_on_update(self, client, test_db):
        c = _mk_contact(test_db)
        assert client.patch(f"{BASE}/{c.id}", json={"number": "911"}).status_code == 400
        # unchanged
        test_db.expire_all()
        assert test_db.query(PhoneContact).filter_by(id=c.id).one().number == "+19085551234"

    def test_number_update_marks_manual_and_verified(self, client, test_db):
        c = _mk_contact(test_db, source="web", verified_at=None)
        r = client.patch(f"{BASE}/{c.id}", json={"number": "732-592-4183"})
        assert r.json()["number"] == "+17325924183"
        assert r.json()["source"] == "manual"
        assert r.json()["verified_at"] is not None

    def test_do_not_call_toggled(self, client, test_db):
        c = _mk_contact(test_db)
        r = client.patch(f"{BASE}/{c.id}", json={"do_not_call": True})
        assert r.json()["do_not_call"] is True

    def test_cross_household_id_is_404(self, client, test_db):
        """Existence must not leak to a caller in another household."""
        other = _mk_contact(test_db, household_id=OTHER_HH, name="Not Mine")
        r = client.patch(f"{BASE}/{other.id}", json={"notes": "hax"})
        assert r.status_code == 404

    def test_rename_collision_rejected(self, client, test_db):
        _mk_contact(test_db, name="Acme Deli")
        c = _mk_contact(test_db, name="Tony's")
        r = client.patch(f"{BASE}/{c.id}", json={"name": "acme deli"})
        assert r.status_code == 409


class TestDelete:
    def test_deletes(self, client, test_db):
        c = _mk_contact(test_db)
        assert client.delete(f"{BASE}/{c.id}").status_code == 204
        assert test_db.query(PhoneContact).count() == 0

    def test_cross_household_delete_is_404(self, client, test_db):
        other = _mk_contact(test_db, household_id=OTHER_HH, name="Not Mine")
        assert client.delete(f"{BASE}/{other.id}").status_code == 404
        assert test_db.query(PhoneContact).count() == 1

    def test_unknown_id_is_404(self, client, test_db):
        assert client.delete(f"{BASE}/{uuid.uuid4()}").status_code == 404


class TestAuth:
    def test_membership_required(self, test_db):
        from fastapi import HTTPException

        def override_get_db():
            try:
                yield test_db
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[verify_user_jwt] = lambda: AuthenticatedUser(
            user_id=999, email="i@x", is_superuser=False
        )
        try:
            with patch(ROLE, side_effect=HTTPException(403, "not a member")):
                with TestClient(app) as c:
                    assert c.get(BASE).status_code == 403
                    assert c.post(
                        BASE, json={"name": "X", "number": "7325924183"}
                    ).status_code == 403
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# DNC round-trip: the flag set from mobile blocks a plan
# ---------------------------------------------------------------------------


class TestDoNotCallRoundTrip:
    @pytest.mark.asyncio
    async def test_dnc_set_from_mobile_refuses_the_call(self, client, test_db):
        from app.services import phone_call_service

        c = _mk_contact(test_db, name="Tony's Pizzeria")
        assert client.patch(f"{BASE}/{c.id}", json={"do_not_call": True}).status_code == 200

        posted = {}

        def _capture(**kwargs):
            posted.update(kwargs)

        with patch.object(phone_call_service, "_post_card", _capture), patch.object(
            phone_call_service, "get_session_local", create=True
        ), patch(
            "app.db.get_session_local", return_value=lambda: test_db
        ), patch.object(
            phone_call_service, "check_caps", return_value=None
        ):
            await phone_call_service.create_call_plan(
                business="Tony's Pizzeria",
                goal="Order a pizza",
                household_id=HH,
                user_id=USER,
            )

        assert "refused" in posted.get("title", "").lower()
        assert "do-not-call" in posted.get("summary", "").lower()
        # No session row was created for a refused business.
        assert test_db.query(PhoneCallSession).count() == 0
