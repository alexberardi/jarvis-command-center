"""SignalReactionBridge — appt.upcoming → autorun the leave-by plan.

The bridge's execution (Workflow spawn + run_workflow) is injected so these run
without a DB or node; they pin the DECISION logic: the two gates, dedup, the
capability check, and the shape of the plan handed to the runner.
"""
import asyncio
from unittest.mock import AsyncMock

import app.services.signal_reaction_bridge as bridge


def _facts(**over):
    f = {"title": "Dentist", "start_iso": "2026-08-13T15:00:00+00:00",
         "location": "123 Main St", "event_id": "evt-1"}
    f.update(over)
    return f


def _run(facts=None, node_id="node-7", user_id=7, enabled=True, names=None, runner=None):
    bridge._fired.clear()
    return asyncio.run(bridge.react_to_appt_upcoming(
        "hh-1", facts if facts is not None else _facts(), node_id, user_id,
        enabled_check=lambda h: enabled,
        menu_fetch=AsyncMock(return_value=names if names is not None else {"get_drive_time", "reminder"}),
        runner=runner or AsyncMock(return_value="wf_1"),
    ))


def test_disabled_household_is_a_noop():
    runner = AsyncMock()
    assert _run(enabled=False, runner=runner) == "disabled"
    runner.assert_not_awaited()


def test_missing_location_skipped():
    assert _run(facts=_facts(location=None)) == "no_location"


def test_missing_user_skipped():
    assert _run(user_id=None) == "no_user"


def test_node_without_drive_time_skipped():
    runner = AsyncMock()
    assert _run(names={"reminder"}, runner=runner) == "no_drive_time"
    runner.assert_not_awaited()


def test_eligible_autoruns_and_builds_the_leave_by_plan():
    runner = AsyncMock(return_value="wf_1")
    out = _run(runner=runner)
    assert out == "autorun_started"
    kwargs = runner.await_args.kwargs
    steps = kwargs["steps"]
    assert [s["command"] for s in steps] == ["get_drive_time", "reminder"]
    assert steps[0]["args"]["destination"] == "123 Main St"
    # the reminder's relative_minutes is a $leave_by directive the executor resolves
    directive = steps[1]["args"]["relative_minutes"]
    assert "$leave_by" in directive
    assert directive["$leave_by"]["event_start"] == "2026-08-13T15:00:00+00:00"
    assert steps[1]["args"]["text"] == "Leave for Dentist"
    assert kwargs["user_id"] == 7


def test_dedup_second_reaction_for_same_event_is_a_noop():
    bridge._fired.clear()
    runner = AsyncMock(return_value="wf_1")

    async def go():
        common = dict(enabled_check=lambda h: True,
                      menu_fetch=AsyncMock(return_value={"get_drive_time", "reminder"}),
                      runner=runner)
        first = await bridge.react_to_appt_upcoming("hh-1", _facts(), "node-7", 7, **common)
        second = await bridge.react_to_appt_upcoming("hh-1", _facts(), "node-7", 7, **common)
        return first, second

    first, second = asyncio.run(go())
    assert first == "autorun_started"
    assert second == "duplicate"
    assert runner.await_count == 1


def test_schedule_only_fires_for_appt_upcoming():
    # wrong kind → no scheduling attempt (no loop needed, must not raise)
    bridge.set_main_loop(None)
    bridge.schedule_appt_upcoming_reaction("hh-1", "node-7", 7, "presence.seen", _facts())
    # (no assertion beyond "did not raise / did not schedule" — appt.upcoming path is
    #  exercised live; here we just prove the kind guard short-circuits safely)
