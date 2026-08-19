"""Signal-automation execution — interpret a free-text instruction at fire time.

Pins the DECISION logic via dependency injection: the rule's DELIVERY choice
(automatic runs / notification confirms — the user's choice, not a classifier), the
no-rule / dedup / no-tools / no-action paths, the confirmed-action server callback,
and registration. The LLM inference, node dispatch, and inbox post are injected.
"""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

import app.services.signal_automation_executor as ex
from app.services.server_callback_registry import ServerCallbackContext
from app.services.signal_reaction_registry import ReactionContext


@pytest.fixture(autouse=True)
def _clean_state():
    ex.clear_state()
    yield
    ex.clear_state()


def _ctx(kind="presence.left", household_id="hh-1", user_id=7, facts=None):
    return ReactionContext(
        household_id=household_id,
        node_id=None,
        user_id=user_id,
        kind=kind,
        facts=facts if facts is not None else {"state": "away"},
    )


def _run(
    ctx,
    *,
    instruction="Lock the door",
    delivery="automatic",
    resolve=None,
    pick=None,
    run_automatic=None,
    emit_confirm=None,
):
    # instruction=None models "no enabled rule" (get_rule → None).
    rule = None if instruction is None else {"instruction": instruction, "delivery": delivery}
    return asyncio.run(
        ex.react_to_signal_automation(
            ctx,
            get_rule=lambda hh, kind: rule,
            resolve=resolve or AsyncMock(return_value=("node-1", [{"type": "function"}])),
            pick=pick
            or AsyncMock(return_value=("control_device", {"action": "lock", "entity_id": "lock.front"})),
            run_automatic=run_automatic or AsyncMock(return_value=True),
            emit_confirm=emit_confirm or Mock(return_value=True),
        )
    )


# ── the reaction ─────────────────────────────────────────────────────────────
def test_no_enabled_rule_is_a_noop():
    resolve = AsyncMock()
    assert _run(_ctx(), instruction=None, resolve=resolve) == "no_rule"
    resolve.assert_not_awaited()


def test_automatic_delivery_runs_immediately():
    run_auto = AsyncMock(return_value=True)
    emit = Mock(return_value=True)
    result = _run(
        _ctx(),
        delivery="automatic",
        pick=AsyncMock(return_value=("control_device", {"action": "unlock"})),
        run_automatic=run_auto,
        emit_confirm=emit,
    )
    assert result == "ran:control_device"
    run_auto.assert_awaited_once()
    emit.assert_not_called()  # automatic never posts a card — even for unlock


def test_automatic_dispatch_failure_is_reported():
    result = _run(
        _ctx(),
        delivery="automatic",
        run_automatic=AsyncMock(return_value=False),
    )
    assert result == "failed:control_device"


def test_notification_delivery_posts_a_card_and_does_not_run():
    run_auto = AsyncMock(return_value=True)
    emit = Mock(return_value=True)
    result = _run(
        _ctx(),
        delivery="notification",
        pick=AsyncMock(return_value=("control_device", {"action": "lock"})),
        run_automatic=run_auto,
        emit_confirm=emit,
    )
    assert result == "confirm:control_device"
    emit.assert_called_once()
    run_auto.assert_not_awaited()  # notification NEVER auto-runs — even a lock


def test_notification_delivery_reports_a_failed_card_post():
    result = _run(_ctx(), delivery="notification", emit_confirm=Mock(return_value=False))
    assert result == "confirm_failed"


def test_no_tool_chosen_is_noop():
    assert _run(_ctx(), pick=AsyncMock(return_value=None)) == "no_action"


def test_no_reachable_node_or_tools():
    assert _run(_ctx(), resolve=AsyncMock(return_value=(None, []))) == "no_tools"


# ── transition dedup (heartbeat re-asserts must not re-act) ──────────────────
def test_repeated_same_signal_is_deduped():
    ctx = _ctx(facts={"state": "away"})
    run_auto = AsyncMock(return_value=True)
    pick = AsyncMock(return_value=("control_device", {"action": "lock"}))
    assert _run(ctx, run_automatic=run_auto, pick=pick) == "ran:control_device"
    assert _run(ctx, run_automatic=run_auto, pick=pick) == "unchanged"
    assert run_auto.await_count == 1


def test_no_action_still_latches_so_reasserts_do_not_rerun():
    ctx = _ctx(facts={"state": "away"})
    pick = AsyncMock(return_value=None)
    assert _run(ctx, pick=pick) == "no_action"
    assert _run(ctx, pick=pick) == "unchanged"
    assert pick.await_count == 1


def test_leave_arrive_leave_all_act():
    # presence.left ALWAYS carries state="away"; a shared presence key keeps
    # leave→arrive→leave from latching "away" and skipping the second departure.
    run_auto = AsyncMock(return_value=True)
    leave_pick = AsyncMock(return_value=("control_device", {"action": "lock"}))
    arrive_pick = AsyncMock(return_value=("control_device", {"action": "turn_on"}))
    assert (
        _run(_ctx("presence.left", facts={"state": "away"}), run_automatic=run_auto, pick=leave_pick)
        == "ran:control_device"
    )
    assert (
        _run(
            _ctx("presence.seen", facts={"state": "home"}),
            instruction="lights on",
            run_automatic=run_auto,
            pick=arrive_pick,
        )
        == "ran:control_device"
    )
    assert (
        _run(_ctx("presence.left", facts={"state": "away"}), run_automatic=run_auto, pick=leave_pick)
        == "ran:control_device"
    )
    assert run_auto.await_count == 3


def test_event_kinds_dedup_per_event_id():
    run_auto = AsyncMock(return_value=True)
    pick = AsyncMock(return_value=("reminder", {"text": "leave"}))

    def _appt(eid):
        return ReactionContext(
            household_id="hh-1", node_id="n", user_id=7, kind="appt.upcoming",
            facts={"event_id": eid},
        )

    assert _run(_appt("e1"), instruction="remind me", run_automatic=run_auto, pick=pick) == "ran:reminder"
    assert _run(_appt("e1"), instruction="remind me", run_automatic=run_auto, pick=pick) == "unchanged"
    assert _run(_appt("e2"), instruction="remind me", run_automatic=run_auto, pick=pick) == "ran:reminder"
    assert run_auto.await_count == 2


def test_an_error_never_raises():
    assert _run(_ctx(), resolve=AsyncMock(side_effect=RuntimeError("boom"))) == "error"


# ── the confirm-card server callback ─────────────────────────────────────────
def test_confirmed_action_dispatches_to_the_node(monkeypatch):
    import app.services.node_command_service as ncs

    disp = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(ncs, "dispatch_node_command", disp)
    sctx = ServerCallbackContext(
        job_id="job-1",
        household_id="hh-1",
        user_id=7,
        data={
            "_action": {
                "node_id": "node-1",
                "command_name": "control_device",
                "arguments": {"action": "unlock", "entity_id": "lock.front"},
            }
        },
    )
    res = asyncio.run(ex._execute_confirmed_action(sctx))
    assert res.success is True
    disp.assert_awaited_once()
    assert disp.await_args.args[1] == "control_device"


def test_confirmed_action_rejects_a_malformed_payload():
    sctx = ServerCallbackContext(job_id="j", household_id="hh-1", user_id=7, data={"_action": {}})
    res = asyncio.run(ex._execute_confirmed_action(sctx))
    assert res.success is False


def test_dismiss_is_a_clean_noop():
    sctx = ServerCallbackContext(job_id="j", household_id="hh-1", user_id=7, data={})
    res = asyncio.run(ex._dismiss_confirmed_action(sctx))
    assert res.success is True


# ── registration ─────────────────────────────────────────────────────────────
def test_registers_a_reaction_for_every_catalog_kind_and_the_callbacks():
    from app.services import server_callback_registry as scb
    from app.services import signal_reaction_registry as reg

    reg.clear_reactions()
    try:
        ex.register_signal_automation_executor()
        for kind in ("presence.left", "presence.seen", "appt.upcoming"):
            assert "automation" in [n for n, _ in reg.reactions_for(kind)]
        cbs = scb.registered_server_callbacks()
        assert ("jarvis.signal_automation", "execute") in cbs
        assert ("jarvis.signal_automation", "dismiss") in cbs
    finally:
        reg.clear_reactions()
        scb.unregister_server_callback("jarvis.signal_automation", "execute")
        scb.unregister_server_callback("jarvis.signal_automation", "dismiss")
