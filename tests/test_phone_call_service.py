"""Phone-call service: validation, state machine, resolve, caps, callbacks.

DB-backed tests use the shared test_db fixture; handler tests bind the
service's own sessions to it by patching app.db.get_session_local.
"""

import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.models import PhoneCallSession, PhoneContact
from app.services import phone_call_service as svc
from app.services.server_callback_registry import ServerCallbackContext

HH = "hh-phone-test"


class _NonClosingSession:
    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        pass


@pytest.fixture
def bound_db(test_db):
    """Patch the service's session factory to the test session."""
    with patch(
        "app.db.get_session_local",
        return_value=lambda: _NonClosingSession(test_db),
    ):
        yield test_db


def _mk_session(db, *, state="draft", expires_in_min=20, **overrides) -> PhoneCallSession:
    now = datetime.utcnow()
    fields = dict(
        id=str(uuid.uuid4()),
        household_id=HH,
        user_id=7,
        contact_name="Tony's Pizzeria",
        goal="Book a table for 4 on Friday at 7pm",
        details="Table for 4 on Friday at 7pm, under Alex.",
        resolved_number="+19085551234",
        state=state,
        created_at=now,
        expires_at=now + timedelta(minutes=expires_in_min),
    )
    fields.update(overrides)
    s = PhoneCallSession(**fields)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# =============================================================================
# Number validation (decision 10)
# =============================================================================


class TestNormalizeUsNumber:
    def test_valid_formats_normalize(self):
        for raw in ("+1 555-555-0123", "5555550123", "1-555-555-0123", "555-555-0123"):
            assert svc.normalize_us_number(raw) == "+15555550123"

    def test_emergency_numbers_refused(self):
        for raw in ("911", "+1911", "112", "988"):
            with pytest.raises(svc.NumberValidationError):
                svc.normalize_us_number(raw)

    def test_short_codes_refused(self):
        with pytest.raises(svc.NumberValidationError):
            svc.normalize_us_number("55444")

    def test_premium_rate_refused(self):
        with pytest.raises(svc.NumberValidationError):
            svc.normalize_us_number("1-900-555-1234")

    def test_non_us_refused(self):
        with pytest.raises(svc.NumberValidationError):
            svc.normalize_us_number("+44 20 7946 0958")

    def test_garbage_refused(self):
        for raw in ("", "   ", "call me maybe"):
            with pytest.raises(svc.NumberValidationError):
                svc.normalize_us_number(raw)

    def test_invalid_area_code_refused(self):
        with pytest.raises(svc.NumberValidationError):
            svc.normalize_us_number("1234567890")  # area code starting with 1


# =============================================================================
# State machine
# =============================================================================


class TestTransition:
    def test_legal_path(self, test_db):
        s = _mk_session(test_db)
        for to in ("confirmed", "dialing", "in_call", "wrapup", "done"):
            assert svc.transition(s, to), f"draft->...->{to} should be legal"
        assert s.ended_at is not None

    def test_illegal_transitions_refused(self, test_db):
        s = _mk_session(test_db, state="done")
        for to in ("draft", "confirmed", "dialing", "in_call", "failed"):
            assert not svc.transition(s, to)
        assert s.state == "done"

    def test_draft_cannot_skip_to_dialing(self, test_db):
        s = _mk_session(test_db)
        assert not svc.transition(s, "dialing")
        assert s.state == "draft"


# =============================================================================
# Contact resolution + DNC
# =============================================================================


class TestResolveContact:
    def _contact(self, db, name="Tony's Pizzeria", **overrides):
        c = PhoneContact(
            id=str(uuid.uuid4()),
            household_id=HH,
            name=name,
            normalized_name=svc._normalize_name(name),
            number="+19085551234",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            **overrides,
        )
        db.add(c)
        db.commit()
        return c

    def test_fuzzy_match_within_household(self, test_db):
        c = self._contact(test_db)
        found = svc.resolve_contact(test_db, HH, "tonys pizza")
        assert found is not None and found.id == c.id

    def test_no_cross_household_leak(self, test_db):
        self._contact(test_db)
        assert svc.resolve_contact(test_db, "other-hh", "Tony's Pizzeria") is None

    def test_no_match_returns_none(self, test_db):
        self._contact(test_db)
        assert svc.resolve_contact(test_db, HH, "completely unrelated dentist") is None


# =============================================================================
# Caps (fail-closed)
# =============================================================================


class TestCaps:
    def test_under_caps_allows(self, test_db):
        assert svc.check_caps(test_db, HH) is None

    def test_daily_cap_blocks(self, test_db):
        with patch.object(svc, "_int_setting", side_effect=lambda k, h, d: 2 if "per_day" in k else d):
            _mk_session(test_db, state="done")
            _mk_session(test_db, state="declined")
            refusal = svc.check_caps(test_db, HH)
        assert refusal is not None and "daily" in refusal.lower()

    def test_concurrent_cap_blocks(self, test_db):
        _mk_session(test_db, state="in_call")
        refusal = svc.check_caps(test_db, HH)
        assert refusal is not None and "in progress" in refusal.lower()

    def test_monthly_minutes_cap_blocks(self, test_db):
        _mk_session(test_db, state="done", duration_seconds=3600)
        with patch.object(svc, "_int_setting", side_effect=lambda k, h, d: 30 if "monthly" in k else d):
            refusal = svc.check_caps(test_db, HH)
        assert refusal is not None and "monthly" in refusal.lower()

    def test_errors_fail_closed(self, test_db):
        with patch.object(svc, "_int_setting", side_effect=RuntimeError("boom")):
            assert svc.check_caps(test_db, HH) is not None


# =============================================================================
# Confirm tap (the authorization moment)
# =============================================================================


def _confirm_ctx(session_id, *, number="+1 555-555-0123", details="Book it.", user_id=42):
    return ServerCallbackContext(
        job_id=str(uuid.uuid4()),
        household_id=HH,
        user_id=user_id,
        data={"session_id": session_id, "dialed_number": number, "details": details},
    )


class TestConfirmCall:
    def test_happy_path_confirms_audits_and_enqueues(self, bound_db):
        s = _mk_session(bound_db)
        with (
            patch.object(svc, "phone_calls_enabled", return_value=True),
            patch.object(svc, "enqueue_dial") as enq,
        ):
            result = svc._handle_confirm_call(_confirm_ctx(s.id))

        assert result.success, result.error
        bound_db.refresh(s)
        assert s.state == "confirmed"
        assert s.dialed_number == "+15555550123"
        assert s.number_edited is True  # differs from resolved +19085551234
        assert s.confirmed_by == 42
        enq.assert_called_once_with(s.id, HH)

    def test_unedited_number_not_flagged(self, bound_db):
        s = _mk_session(bound_db)
        with (
            patch.object(svc, "phone_calls_enabled", return_value=True),
            patch.object(svc, "enqueue_dial"),
        ):
            svc._handle_confirm_call(_confirm_ctx(s.id, number="+19085551234"))
        bound_db.refresh(s)
        assert s.number_edited is False

    def test_second_tap_is_noop(self, bound_db):
        s = _mk_session(bound_db, state="confirmed")
        with patch.object(svc, "phone_calls_enabled", return_value=True):
            result = svc._handle_confirm_call(_confirm_ctx(s.id))
        assert result.success
        assert "already" in result.context_data["inbox"]["summary"].lower()
        bound_db.refresh(s)
        assert s.state == "confirmed"

    def test_expired_plan_refused(self, bound_db):
        s = _mk_session(bound_db, expires_in_min=-1)
        with patch.object(svc, "phone_calls_enabled", return_value=True):
            result = svc._handle_confirm_call(_confirm_ctx(s.id))
        assert not result.success
        assert "expired" in result.error.lower()
        bound_db.refresh(s)
        assert s.state == "expired"

    def test_gate_off_declines_plan(self, bound_db):
        s = _mk_session(bound_db)
        with patch.object(svc, "phone_calls_enabled", return_value=False):
            result = svc._handle_confirm_call(_confirm_ctx(s.id))
        assert not result.success
        bound_db.refresh(s)
        assert s.state == "declined"

    def test_edited_number_revalidated(self, bound_db):
        s = _mk_session(bound_db)
        with patch.object(svc, "phone_calls_enabled", return_value=True):
            result = svc._handle_confirm_call(_confirm_ctx(s.id, number="911"))
        assert not result.success
        bound_db.refresh(s)
        assert s.state == "draft"  # still confirmable with a fixed number

    def test_enqueue_failure_fails_session_honestly(self, bound_db):
        s = _mk_session(bound_db)
        with (
            patch.object(svc, "phone_calls_enabled", return_value=True),
            patch.object(svc, "enqueue_dial", side_effect=RuntimeError("redis down")),
        ):
            result = svc._handle_confirm_call(_confirm_ctx(s.id))
        assert not result.success
        bound_db.refresh(s)
        assert s.state == "failed"

    def test_wrong_household_cannot_confirm(self, bound_db):
        s = _mk_session(bound_db)
        ctx = _confirm_ctx(s.id)
        ctx.household_id = "attacker-hh"
        with patch.object(svc, "phone_calls_enabled", return_value=True):
            result = svc._handle_confirm_call(ctx)
        assert not result.success
        bound_db.refresh(s)
        assert s.state == "draft"


class TestCancelCall:
    def test_cancel_declines_draft(self, bound_db):
        s = _mk_session(bound_db)
        result = svc._handle_cancel_call(
            ServerCallbackContext(
                job_id="j", household_id=HH, user_id=7, data={"session_id": s.id}
            )
        )
        assert result.success
        bound_db.refresh(s)
        assert s.state == "declined"


# =============================================================================
# Escalation forward
# =============================================================================


class TestEscalationAnswer:
    @pytest.mark.asyncio
    async def test_forwards_to_worker(self, bound_db):
        s = _mk_session(bound_db, state="in_call", worker_url="http://worker:7713")
        captured = {}

        class _Resp:
            status_code = 200

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["json"] = json
                return _Resp()

        with patch.object(svc.httpx, "AsyncClient", _Client):
            result = await svc._handle_escalation_answer(
                ServerCallbackContext(
                    job_id="j",
                    household_id=HH,
                    user_id=42,
                    data={"session_id": s.id, "answer": "6:30 works"},
                )
            )
        assert result.success
        assert captured["url"].endswith(f"/internal/call/{s.id}/escalation-answer")
        assert captured["json"]["answer"] == "6:30 works"
        assert captured["json"]["answered_by"] == 42

    @pytest.mark.asyncio
    async def test_ended_call_refused(self, bound_db):
        s = _mk_session(bound_db, state="done", worker_url="http://worker:7713")
        result = await svc._handle_escalation_answer(
            ServerCallbackContext(
                job_id="j", household_id=HH, user_id=42,
                data={"session_id": s.id, "answer": "yes"},
            )
        )
        assert not result.success


# =============================================================================
# Gate + reaper
# =============================================================================


class TestGate:
    def test_gate_fails_closed_on_error(self):
        with patch(
            "app.services.settings_service.get_settings_service",
            side_effect=RuntimeError("settings down"),
        ):
            assert svc.phone_calls_enabled(HH) is False

    def test_gate_defaults_false_without_household(self):
        assert svc.phone_calls_enabled(None) is False


class TestReaper:
    @pytest.mark.asyncio
    async def test_stale_heartbeat_reaped(self, bound_db):
        s = _mk_session(
            bound_db,
            state="in_call",
            worker_url="http://worker:7713",
            heartbeat_at=datetime.utcnow() - timedelta(seconds=120),
            confirmed_at=datetime.utcnow() - timedelta(seconds=180),
        )
        with (
            patch.object(svc, "_post_card"),
            patch.object(svc.httpx, "AsyncClient") as client_cls,
        ):
            client_cls.side_effect = RuntimeError("no network in tests")
            reaped = await svc.reap_phone_sessions()
        assert reaped == 1
        bound_db.refresh(s)
        assert s.state == "failed"

    @pytest.mark.asyncio
    async def test_fresh_call_untouched(self, bound_db):
        s = _mk_session(
            bound_db, state="in_call", heartbeat_at=datetime.utcnow(),
            confirmed_at=datetime.utcnow(),
        )
        reaped = await svc.reap_phone_sessions()
        assert reaped == 0
        bound_db.refresh(s)
        assert s.state == "in_call"

    @pytest.mark.asyncio
    async def test_expired_draft_marked(self, bound_db):
        s = _mk_session(bound_db, expires_in_min=-5)
        reaped = await svc.reap_phone_sessions()
        assert reaped == 1
        bound_db.refresh(s)
        assert s.state == "expired"
