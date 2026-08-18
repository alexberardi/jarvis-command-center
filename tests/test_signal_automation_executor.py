"""Signal-automation execution — interpret a free-text instruction at fire time.

Pins the DECISION logic via dependency injection: the reversible-vs-sensitive
classifier, the no-rule / dedup / no-tools / no-action paths, the reversible
auto-run vs sensitive confirm-card branch, the confirmed-action server callback,
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
    resolve=None,
    pick=None,
    run_reversible=None,
    emit_confirm=None,
):
    return asyncio.run(
        ex.react_to_signal_automation(
            ctx,
            get_instruction=lambda hh, kind: instruction,
            resolve=resolve or AsyncMock(return_value=("node-1", [{"type": "function"}])),
            pick=pick
            or AsyncMock(return_value=("control_device", {"action": "lock", "entity_id": "lock.front"})),
            run_reversible=run_reversible or AsyncMock(return_value=True),
            emit_confirm=emit_confirm or Mock(return_value=True),
        )
    )


# ── classification (the guardrail boundary) ──────────────────────────────────
@pytest.mark.parametrize(
    "cmd,args,expected",
    [
        ("control_device", {"action": "lock"}, "reversible"),
        ("control_device", {"action": "turn_on"}, "reversible"),
        ("control_device", {"action": "turn_off"}, "reversible"),
        ("control_device", {"action": "unlock"}, "sensitive"),  # opens a door
        ("control_device", {"action": "frobnicate"}, "sensitive"),  # unknown action
        ("control_device", {}, "sensitive"),  # no action → fail-safe
        ("reminder", {}, "reversible"),
        ("get_drive_time", {}, "reversible"),
        ("send_message", {}, "sensitive"),  # unknown command → fail-safe
        ("place_call", {}, "sensitive"),
    ],
)
def test_classify(cmd, args, expected):
    assert ex._classify(cmd, args) == expected


# ── the reaction ─────────────────────────────────────────────────────────────
def test_no_enabled_rule_is_a_noop():
    resolve = AsyncMock()
    assert _run(_ctx(), instruction=None, resolve=resolve) == "no_rule"
    resolve.assert_not_awaited()


def test_reversible_action_auto_runs():
    run_rev = AsyncMock(return_value=True)
    emit = Mock(return_value=True)
    result = _run(
        _ctx(),
        pick=AsyncMock(return_value=("control_device", {"action": "lock"})),
        run_reversible=run_rev,
        emit_confirm=emit,
    )
    assert result == "ran:control_device"
    run_rev.assert_awaited_once()
    emit.assert_not_called()  # reversible never posts a card


def test_reversible_dispatch_failure_reported():
    result = _run(
        _ctx(),
        pick=AsyncMock(return_value=("control_device", {"action": "lock"})),
        run_reversible=AsyncMock(return_value=False),
    )
    assert result == "failed:control_device"


def test_sensitive_action_posts_a_confirm_card_and_does_not_run():
    run_rev = AsyncMock(return_value=True)
    emit = Mock(return_value=True)
    result = _run(
        _ctx(),
        pick=AsyncMock(return_value=("control_device", {"action": "unlock"})),
        run_reversible=run_rev,
        emit_confirm=emit,
    )
    assert result == "confirm:control_device"
    emit.assert_called_once()
    run_rev.assert_not_awaited()  # sensitive is NEVER auto-run


def test_unknown_command_is_treated_as_sensitive():
    emit = Mock(return_value=True)
    result = _run(
        _ctx(),
        pick=AsyncMock(return_value=("send_message", {"to": "mom", "text": "hi"})),
        run_reversible=AsyncMock(return_value=True),
        emit_confirm=emit,
    )
    assert result == "confirm:send_message"
    emit.assert_called_once()


def test_no_tool_chosen_is_noop():
    assert _run(_ctx(), pick=AsyncMock(return_value=None)) == "no_action"


def test_no_reachable_node_or_tools():
    assert _run(_ctx(), resolve=AsyncMock(return_value=(None, []))) == "no_tools"


def test_repeated_same_signal_is_deduped():
    ctx = _ctx(facts={"state": "away"})
    run_rev = AsyncMock(return_value=True)
    pick = AsyncMock(return_value=("control_device", {"action": "lock"}))
    assert _run(ctx, run_reversible=run_rev, pick=pick) == "ran:control_device"
    assert _run(ctx, run_reversible=run_rev, pick=pick) == "unchanged"
    assert run_rev.await_count == 1  # the re-assert did NOT re-run the LLM/action


def test_no_action_still_latches_so_reasserts_do_not_rerun():
    ctx = _ctx(facts={"state": "away"})
    pick = AsyncMock(return_value=None)
    assert _run(ctx, pick=pick) == "no_action"
    assert _run(ctx, pick=pick) == "unchanged"
    assert pick.await_count == 1


def test_leave_arrive_leave_all_act():
    # The bug this pins: presence.left ALWAYS carries state="away", so a per-kind
    # dedup would latch "away" and skip every departure after the first. Sharing the
    # presence key with the arrival ("home") resets it, so leave→arrive→leave all act.
    run_rev = AsyncMock(return_value=True)
    leave_pick = AsyncMock(return_value=("control_device", {"action": "lock"}))
    arrive_pick = AsyncMock(return_value=("control_device", {"action": "turn_on"}))
    assert (
        _run(_ctx("presence.left", facts={"state": "away"}), run_reversible=run_rev, pick=leave_pick)
        == "ran:control_device"
    )
    assert (
        _run(
            _ctx("presence.seen", facts={"state": "home"}),
            instruction="lights on",
            run_reversible=run_rev,
            pick=arrive_pick,
        )
        == "ran:control_device"
    )
    # The SECOND departure must act (it did NOT before the shared-key fix).
    assert (
        _run(_ctx("presence.left", facts={"state": "away"}), run_reversible=run_rev, pick=leave_pick)
        == "ran:control_device"
    )
    assert run_rev.await_count == 3


def test_event_kinds_dedup_per_event_id():
    run_rev = AsyncMock(return_value=True)
    pick = AsyncMock(return_value=("reminder", {"text": "leave"}))

    def _appt(eid):
        return ReactionContext(
            household_id="hh-1", node_id="n", user_id=7, kind="appt.upcoming",
            facts={"event_id": eid},
        )

    assert _run(_appt("e1"), instruction="remind me", run_reversible=run_rev, pick=pick) == "ran:reminder"
    # Same event re-emitted next calendar cycle → one occurrence.
    assert _run(_appt("e1"), instruction="remind me", run_reversible=run_rev, pick=pick) == "unchanged"
    # A different event → acts.
    assert _run(_appt("e2"), instruction="remind me", run_reversible=run_rev, pick=pick) == "ran:reminder"
    assert run_rev.await_count == 2


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
