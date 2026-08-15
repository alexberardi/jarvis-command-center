"""Plan-start blast gate — decides whether a FRESH plan may autorun (no human tap).

This is the security boundary for signal-triggered autonomy: only a plan that is
entirely low-blast — allowlisted commands, no risky step, no outbound counterparty,
few steps — may run without confirmation. Anything else falls back to a tap card.
(`widens_envelope` only guards amendments MID-run; this guards plan START.)
"""
from app.services.autorun_gate import is_autorun_eligible


def _step(command, is_risky=False, **args):
    return {"command": command, "args": args, "label": command, "is_risky": is_risky}


def test_drive_time_then_reminder_is_eligible():
    steps = [_step("get_drive_time", destination="Cafe"),
             _step("reminder", action="set", text="Leave", relative_minutes=20)]
    ok, reason = is_autorun_eligible(steps)
    assert ok is True


def test_non_allowlisted_command_blocked():
    steps = [_step("get_drive_time"), _step("unlock_door")]
    ok, reason = is_autorun_eligible(steps)
    assert ok is False and "unlock_door" in reason


def test_risky_step_blocked():
    steps = [_step("reminder", is_risky=True, action="set", text="x")]
    ok, reason = is_autorun_eligible(steps)
    assert ok is False and "risky" in reason


def test_outbound_counterparty_blocked():
    # a 'business' arg = an outbound call/contact — never autorun, even if allowlisted
    steps = [_step("reminder", action="set", text="x", business="Pizza Place")]
    ok, reason = is_autorun_eligible(steps)
    assert ok is False


def test_too_many_steps_blocked():
    steps = [_step("get_drive_time"), _step("reminder"), _step("get_drive_time"), _step("reminder")]
    ok, reason = is_autorun_eligible(steps)
    assert ok is False and "step" in reason.lower()


def test_empty_plan_blocked():
    ok, reason = is_autorun_eligible([])
    assert ok is False
