"""Unit tests for the proposable-action dispatcher (the server-plane chokepoint).

Boundaries (settings gate, capability lookup, node execution, idempotency, node
resolution) are patched, so these run without a DB or a live node — mirroring the
direct-handler-call pattern used for the errand server callbacks.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from app.services.server_callback_registry import ServerCallbackContext
import app.services.proposable_action_service as svc


def _ctx(data, household_id="hh-1", user_id=7):
    return ServerCallbackContext(
        job_id="job-1", household_id=household_id, user_id=user_id, data=data
    )


def _spec(params=None):
    return {
        "callback": "create_event",
        "params": params
        or [
            {"name": "title", "required": True},
            {"name": "start", "required": True},
            {"name": "idempotency_key", "required": True},
        ],
        "blast_tier": "reversible",
        "idempotency_param": "idempotency_key",
    }


def _card_data(**overrides):
    data = {
        "_action": {
            "target_command": "add_event",
            "target_callback": "create_event",
            "node_id": "node-7",
            "idempotency_key": "appt:1",
        },
        "title": "Dentist",
        "start": "2026-08-10T09:00:00",
        "idempotency_key": "appt:1",
    }
    data.update(overrides)
    return data


# ── validate_against_params ──────────────────────────────────────────────────


class TestValidateParams:
    def test_drops_unknown_keys(self):
        ok, args, err = svc.validate_against_params(
            {"title": "x", "sneaky": "evil"}, [{"name": "title", "required": True}]
        )
        assert ok and args == {"title": "x"} and err is None

    def test_missing_required_rejected(self):
        ok, args, err = svc.validate_against_params({}, [{"name": "title", "required": True}])
        assert not ok and "title" in err

    def test_optional_missing_ok(self):
        ok, args, err = svc.validate_against_params(
            {}, [{"name": "loc", "required": False}]
        )
        assert ok and args == {}

    def test_enum_enforced(self):
        ok, args, err = svc.validate_against_params(
            {"cal": "work"}, [{"name": "cal", "required": True, "enum_values": ["home"]}]
        )
        assert not ok and "one of" in err


# ── _handle_execute pipeline ─────────────────────────────────────────────────


class TestHandleExecute:
    def test_happy_path_runs_callback_on_node(self):
        exec_mock = AsyncMock(return_value=(True, {"message": "Added to your calendar."}, None))
        with patch.object(svc, "_proposals_enabled", return_value=True), patch(
            "app.services.capability_registry.resolve_proposable_action",
            new=AsyncMock(return_value=_spec()),
        ), patch.object(svc, "_already_completed", return_value=False), patch.object(
            svc, "_execute_target_callback_on_node", new=exec_mock
        ):
            res = asyncio.run(svc._handle_execute(_ctx(_card_data())))

        assert res.success is True
        assert res.context_data["inbox"]["title"] == "Added to your calendar."
        # Validated args exclude `_action`, include exactly the declared params.
        kwargs = exec_mock.call_args.kwargs
        assert kwargs["command_name"] == "add_event"
        assert kwargs["callback_name"] == "create_event"
        assert kwargs["node_id"] == "node-7"
        assert kwargs["args"] == {
            "title": "Dentist",
            "start": "2026-08-10T09:00:00",
            "idempotency_key": "appt:1",
        }

    def test_disabled_household_refuses(self):
        exec_mock = AsyncMock()
        with patch.object(svc, "_proposals_enabled", return_value=False), patch.object(
            svc, "_execute_target_callback_on_node", new=exec_mock
        ):
            res = asyncio.run(svc._handle_execute(_ctx(_card_data())))
        assert res.success is False
        assert "disabled" in (res.error or "")
        exec_mock.assert_not_called()

    def test_not_proposable_refused_optin(self):
        exec_mock = AsyncMock()
        with patch.object(svc, "_proposals_enabled", return_value=True), patch(
            "app.services.capability_registry.resolve_proposable_action",
            new=AsyncMock(return_value=None),
        ), patch.object(svc, "_execute_target_callback_on_node", new=exec_mock):
            res = asyncio.run(svc._handle_execute(_ctx(_card_data())))
        assert res.success is False
        assert "not proposable" in (res.error or "")
        exec_mock.assert_not_called()

    def test_invalid_params_refused(self):
        exec_mock = AsyncMock()
        # Card omits the required `start`.
        data = _card_data()
        del data["start"]
        with patch.object(svc, "_proposals_enabled", return_value=True), patch(
            "app.services.capability_registry.resolve_proposable_action",
            new=AsyncMock(return_value=_spec()),
        ), patch.object(svc, "_already_completed", return_value=False), patch.object(
            svc, "_execute_target_callback_on_node", new=exec_mock
        ):
            res = asyncio.run(svc._handle_execute(_ctx(data)))
        assert res.success is False
        assert "invalid params" in (res.error or "")
        exec_mock.assert_not_called()

    def test_idempotent_short_circuit(self):
        exec_mock = AsyncMock()
        with patch.object(svc, "_proposals_enabled", return_value=True), patch(
            "app.services.capability_registry.resolve_proposable_action",
            new=AsyncMock(return_value=_spec()),
        ), patch.object(svc, "_already_completed", return_value=True), patch.object(
            svc, "_execute_target_callback_on_node", new=exec_mock
        ):
            res = asyncio.run(svc._handle_execute(_ctx(_card_data())))
        assert res.success is True
        exec_mock.assert_not_called()  # never re-runs a completed action

    def test_node_failure_surfaces_visible_card(self):
        exec_mock = AsyncMock(return_value=(False, None, "Unknown command"))
        with patch.object(svc, "_proposals_enabled", return_value=True), patch(
            "app.services.capability_registry.resolve_proposable_action",
            new=AsyncMock(return_value=_spec()),
        ), patch.object(svc, "_already_completed", return_value=False), patch.object(
            svc, "_execute_target_callback_on_node", new=exec_mock
        ):
            res = asyncio.run(svc._handle_execute(_ctx(_card_data())))
        assert res.success is False
        assert res.context_data["inbox"]["title"]  # a visible failure card, not silent

    def test_missing_node_refused(self):
        data = _card_data()
        data["_action"]["node_id"] = None
        with patch.object(svc, "_proposals_enabled", return_value=True), patch.object(
            svc, "resolve_household_node_for_command", new=AsyncMock(return_value=None)
        ):
            res = asyncio.run(svc._handle_execute(_ctx(data)))
        assert res.success is False
        assert "no node" in (res.error or "")

    def test_resolves_household_node_when_card_has_no_node_id(self):
        # No explicit node on the card → the command-aware resolver picks a node that
        # advertises the command, and the callback executes on THAT node end-to-end.
        data = _card_data()
        data["_action"]["node_id"] = None
        exec_mock = AsyncMock(return_value=(True, {"message": "Added to your calendar."}, None))
        with patch.object(svc, "_proposals_enabled", return_value=True), patch.object(
            svc, "resolve_household_node_for_command", new=AsyncMock(return_value="calendar-node")
        ), patch(
            "app.services.capability_registry.resolve_proposable_action",
            new=AsyncMock(return_value=_spec()),
        ), patch.object(svc, "_already_completed", return_value=False), patch.object(
            svc, "_execute_target_callback_on_node", new=exec_mock
        ):
            res = asyncio.run(svc._handle_execute(_ctx(data)))
        assert res.success is True
        assert exec_mock.call_args.kwargs["node_id"] == "calendar-node"  # the resolved node


class TestHandleDismiss:
    def test_dismiss_is_a_no_op_success(self):
        res = asyncio.run(svc._handle_dismiss(_ctx({"_action": {"idempotency_key": "appt:1"}})))
        assert res.success is True
        assert res.context_data is None


class TestHandleSuppress:
    def test_records_suppression_and_confirms(self):
        rec = None

        def _fake_record(**kw):
            nonlocal rec
            rec = kw
            return "sup_1"

        data = {
            "_action": {"target_command": "add_event", "idempotency_key": "appt:1"},
            "source": "notifications@github.com",
            "descriptor": "CI/build-failure notification",
        }
        with patch("app.services.proposal_suppressions.record_suppression", new=_fake_record):
            res = asyncio.run(svc._handle_suppress(_ctx(data)))
        assert res.success is True
        assert res.context_data["inbox"]["title"] == "Won't suggest that again"
        assert rec["household_id"] == "hh-1" and rec["user_id"] == 7
        assert rec["command"] == "add_event"
        assert rec["source_key"] == "notifications@github.com"
        assert rec["descriptor"] == "CI/build-failure notification"

    def test_nothing_to_suppress_is_rejected(self):
        data = {"_action": {"target_command": "add_event"}}  # no source, no descriptor
        res = asyncio.run(svc._handle_suppress(_ctx(data)))
        assert res.success is False
        assert "nothing to suppress" in (res.error or "")


class TestRegistration:
    def test_registers_execute_dismiss_and_suppress(self):
        from app.services.server_callback_registry import get_server_callback

        svc.register_proposable_action_callbacks()
        assert get_server_callback("jarvis.proposable_action", "execute") is not None
        assert get_server_callback("jarvis.proposable_action", "dismiss") is not None
        assert get_server_callback("jarvis.proposable_action", "suppress") is not None
