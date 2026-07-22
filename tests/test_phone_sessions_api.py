"""Gateway-facing phone-session endpoints — the CC half of the contract.

Request shapes below are copied verbatim from the gateway's
``jarvis-phone-gateway/services/session_client.py`` (the contract's other
half). If a shape changes here, it MUST change there in the same release.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# app.main's startup resolves the auth URL — same standalone-run pattern as
# the other TestClient(app) suites.
os.environ.setdefault("JARVIS_AUTH_BASE_URL", "http://localhost:7701")

from app.deps import get_db, require_app_auth
from app.main import app
from app.models import PhoneCallSession
from app.services.call_context import parse_call_context

HH = "hh-phone-api"


def _mk_session(db, *, state="confirmed", **overrides) -> PhoneCallSession:
    now = datetime.utcnow()
    fields = dict(
        id=str(uuid.uuid4()),
        household_id=HH,
        user_id=9,
        contact_name="Tony's Pizzeria",
        goal="Book a table",
        details="Table for 4.",
        resolved_number="+19085551234",
        dialed_number="+19085551234",
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


@pytest.fixture
def client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    async def no_op_app_auth():
        return None

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[require_app_auth] = no_op_app_auth
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _events_url(session_id: str) -> str:
    # session_client.py: f"{base}/internal/phone/sessions/{session_id}/events"
    return f"/internal/phone/sessions/{session_id}/events"


class TestGetSession:
    def test_snapshot_shape(self, client, test_db):
        s = _mk_session(test_db)
        r = client.get(f"/internal/phone/sessions/{s.id}")
        assert r.status_code == 200
        body = r.json()
        for key in (
            "id", "state", "household_id", "contact_name", "dialed_number",
            "goal", "details", "constraints", "line_type", "max_call_seconds",
            "restricted_details",
        ):
            assert key in body, f"snapshot missing {key}"
        assert body["state"] == "confirmed"
        assert isinstance(body["max_call_seconds"], int)

    def test_unknown_session_404(self, client):
        assert client.get(f"/internal/phone/sessions/{uuid.uuid4()}").status_code == 404

    def test_check_time_uses_the_sessions_envelope(self, client, test_db):
        """The validator checks against the same bounds the snapshot exposes —
        constraints, else derived from the confirmed brief."""
        s = _mk_session(
            test_db,
            details="make an appointment\nAcceptable times: Thu 9am-8pm; Wed 5-8pm",
        )
        base = f"/internal/phone/sessions/{s.id}/check-time"

        # Noon is inside Thu 9am-8pm — the case the model false-declined live.
        avail = client.post(base, json={"utterance": "Can you do Thursday at noon?"}).json()
        assert avail["time_detected"] is True
        assert avail["available"] is True
        assert "Thursday" in avail["proposed_label"]

        # Noon is before Wed's 5pm open — the case the model false-accepted live.
        no = client.post(base, json={"utterance": "How about Wednesday at 12?"}).json()
        assert no["available"] is False

        # A turn with no time leaves the verdict to the model.
        none = client.post(base, json={"utterance": "And the patient's name?"}).json()
        assert none["time_detected"] is False
        assert none["available"] is None

    def test_check_time_unknown_session_404(self, client):
        r = client.post(
            f"/internal/phone/sessions/{uuid.uuid4()}/check-time",
            json={"utterance": "Thursday at noon"},
        )
        assert r.status_code == 404

    def test_restricted_details_carry_key_label_and_value(self, client, test_db):
        """The gateway's guard needs all three: value to detect the leak,
        label to ask whether it was requested, key to match the verdict back.
        Drop any one and the guard cannot be built on this payload."""
        s = _mk_session(test_db)
        context = [
            {"key": "full_name", "value": "Alex B"},
            {"key": "callback_number", "value": "+15555550123"},
        ]
        with patch(
            "app.services.call_context.load_call_context",
            return_value=parse_call_context({"fields": context}),
        ):
            body = client.get(f"/internal/phone/sessions/{s.id}").json()

        # full_name is tier STATE — it may be said freely, so it is NOT here.
        assert body["restricted_details"] == [
            {
                "key": "callback_number",
                "label": "Callback number",
                "value": "+15555550123",
            }
        ]

    def test_restricted_details_degrade_to_empty_not_500(self, client, test_db):
        """Context lookup is an enhancement. A settings outage must not stop
        the worker fetching its snapshot — that would turn a degraded guard
        into a failed call."""
        s = _mk_session(test_db)
        with patch(
            "app.services.call_context.load_call_context",
            side_effect=RuntimeError("settings down"),
        ):
            r = client.get(f"/internal/phone/sessions/{s.id}")

        assert r.status_code == 200
        assert r.json()["restricted_details"] == []


class TestClaimDial:
    def test_claim_wins_once(self, client, test_db):
        s = _mk_session(test_db)
        payload = {"type": "claim_dial", "worker_url": "http://worker-a:7713"}
        r1 = client.post(_events_url(s.id), json=payload)
        assert r1.status_code == 200
        # Second claim (another worker / duplicate job) must 409 → drop.
        r2 = client.post(
            _events_url(s.id),
            json={"type": "claim_dial", "worker_url": "http://worker-b:7713"},
        )
        assert r2.status_code == 409

        test_db.expire_all()
        row = test_db.query(PhoneCallSession).filter_by(id=s.id).one()
        assert row.state == "dialing"
        assert row.worker_url == "http://worker-a:7713"
        assert row.heartbeat_at is not None

    def test_claim_of_draft_409s(self, client, test_db):
        s = _mk_session(test_db, state="draft")
        r = client.post(
            _events_url(s.id),
            json={"type": "claim_dial", "worker_url": "http://w:7713"},
        )
        assert r.status_code == 409
        test_db.expire_all()
        assert test_db.query(PhoneCallSession).filter_by(id=s.id).one().state == "draft"

    def test_claim_unknown_session_404s(self, client):
        r = client.post(
            _events_url(str(uuid.uuid4())),
            json={"type": "claim_dial", "worker_url": "http://w:7713"},
        )
        assert r.status_code == 404


class TestStateEvents:
    def test_lifecycle_transitions(self, client, test_db):
        s = _mk_session(test_db, state="dialing", worker_url="http://w")
        for state in ("in_call", "wrapup"):
            r = client.post(_events_url(s.id), json={"type": "state", "state": state})
            assert r.status_code == 200, r.text
        test_db.expire_all()
        assert test_db.query(PhoneCallSession).filter_by(id=s.id).one().state == "wrapup"

    def test_illegal_transition_409s(self, client, test_db):
        s = _mk_session(test_db, state="dialing")
        r = client.post(_events_url(s.id), json={"type": "state", "state": "done"})
        assert r.status_code == 409

    def test_failed_state_posts_honest_card(self, client, test_db):
        s = _mk_session(test_db, state="dialing")
        with patch("app.services.phone_call_service._post_card") as card:
            r = client.post(
                _events_url(s.id),
                json={"type": "state", "state": "failed", "error": "busy signal"},
            )
        assert r.status_code == 200
        card.assert_called_once()
        assert "busy signal" in card.call_args.kwargs["summary"]


class TestTurnAndHeartbeat:
    def test_turns_append_and_heartbeat(self, client, test_db):
        s = _mk_session(test_db, state="in_call")
        turn = {
            "role": "callee", "text": "Tony's, how can I help?",
            "timings": {"stt_ms": 62, "llm_ttft_ms": 390, "tts_ttfb_ms": 810},
        }
        r1 = client.post(_events_url(s.id), json={"type": "turn", "turn": turn})
        r2 = client.post(
            _events_url(s.id),
            json={"type": "turn", "turn": {"role": "jarvis", "text": "Hi!"}},
        )
        assert (r1.status_code, r2.status_code) == (200, 200)
        assert r2.json()["turns"] == 2
        test_db.expire_all()
        row = test_db.query(PhoneCallSession).filter_by(id=s.id).one()
        turns = json.loads(row.transcript_json)
        assert turns[0]["timings"]["stt_ms"] == 62
        assert row.heartbeat_at is not None

    def test_turn_on_terminal_session_409s(self, client, test_db):
        s = _mk_session(test_db, state="done")
        r = client.post(
            _events_url(s.id), json={"type": "turn", "turn": {"text": "x"}}
        )
        assert r.status_code == 409

    def test_bare_heartbeat(self, client, test_db):
        s = _mk_session(test_db, state="in_call")
        assert client.post(_events_url(s.id), json={"type": "heartbeat"}).status_code == 200

    def test_heartbeat_on_terminal_409s(self, client, test_db):
        s = _mk_session(test_db, state="failed")
        assert client.post(_events_url(s.id), json={"type": "heartbeat"}).status_code == 409


class TestOutcome:
    def test_outcome_lands_done_with_card(self, client, test_db):
        s = _mk_session(test_db, state="wrapup")
        outcome = {
            "goal_achieved": True,
            "summary": "Booked Friday 7pm, party of 4, under Alex.",
            "facts": {"confirmation": "under Alex", "time": "Friday 7pm"},
        }
        with patch("app.services.phone_call_service._post_card") as card:
            r = client.post(
                _events_url(s.id),
                json={
                    "type": "outcome", "outcome": outcome,
                    "audio_key": f"phone-calls/{HH}/{s.id}.wav",
                    "duration_seconds": 95,
                },
            )
        assert r.status_code == 200
        assert r.json()["state"] == "done"
        card.assert_called_once()
        # Security requirement 3: callee content attributed, not Jarvis's voice.
        assert "business" in card.call_args.kwargs["body"].lower()

        test_db.expire_all()
        row = test_db.query(PhoneCallSession).filter_by(id=s.id).one()
        assert row.audio_object_key.endswith(f"{s.id}.wav")
        assert row.duration_seconds == 95
        assert json.loads(row.outcome_json)["goal_achieved"] is True

    def test_outcome_from_in_call_still_lands_done(self, client, test_db):
        s = _mk_session(test_db, state="in_call")
        with patch("app.services.phone_call_service._post_card"):
            r = client.post(
                _events_url(s.id),
                json={"type": "outcome", "outcome": {"summary": "ended"}},
            )
        assert r.status_code == 200
        assert r.json()["state"] == "done"


class TestEscalation:
    def test_escalation_posts_answer_card(self, client, test_db):
        s = _mk_session(test_db, state="in_call")
        with patch("app.services.phone_call_service._post_card") as card:
            r = client.post(
                _events_url(s.id),
                json={"type": "escalation", "question": "Only 6:30 is available — OK?"},
            )
        assert r.status_code == 200
        card.assert_called_once()
        kwargs = card.call_args.kwargs
        # Security requirement 3: the callee's question is rendered attributed
        # ("asked:"), never in Jarvis's voice, and never as tappable actions.
        assert "asked" in kwargs["summary"].lower()
        assert "Only 6:30 is available" in kwargs["body"]
        meta = kwargs["metadata"]
        element_callbacks = {e["callback"] for e in meta["interactive_elements"]}
        assert element_callbacks == {"escalation_answer", "cancel_call"}
        for el in meta["interactive_elements"]:
            assert el["target"] == "server"
            assert el["data"]["session_id"] == s.id
        # The answer flows through the multiline editor into data["answer"].
        assert meta["editable_fields"][0]["data_key"] == "answer"
        assert meta["editor_schema"] == 2

    def test_escalation_requires_question(self, client, test_db):
        s = _mk_session(test_db, state="in_call")
        with patch("app.services.phone_call_service._post_card") as card:
            r = client.post(_events_url(s.id), json={"type": "escalation"})
        assert r.status_code == 400
        card.assert_not_called()

    def test_escalation_rejected_outside_in_call(self, client, test_db):
        s = _mk_session(test_db, state="wrapup")
        with patch("app.services.phone_call_service._post_card") as card:
            r = client.post(
                _events_url(s.id),
                json={"type": "escalation", "question": "anything"},
            )
        assert r.status_code == 409
        card.assert_not_called()


class TestAuthAndValidation:
    def test_unknown_event_type_400s(self, client, test_db):
        s = _mk_session(test_db)
        assert client.post(_events_url(s.id), json={"type": "bogus"}).status_code == 400

    def test_app_auth_required(self, test_db):
        """Without the override, missing app creds must 401 (not execute)."""
        def override_get_db():
            try:
                yield test_db
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        try:
            with TestClient(app) as c:
                s = _mk_session(test_db)
                r = c.post(_events_url(s.id), json={"type": "heartbeat"})
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 401
