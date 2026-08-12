"""Inter-step value resolver — the primitive that lets a workflow step's args
depend on an EARLIER step's result (the engine only did date-key resolution before).

The load-bearing case: a `reminder` step's `relative_minutes` is computed from a
prior `get_drive_time` step's `duration_minutes` + the appointment start — the LLM
composes the plan SHAPE, but the arithmetic is done deterministically in code here.
"""
from datetime import datetime, timezone

from app.services.step_value_resolver import resolve_step_args


def _now():
    return datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


def _drive_result(minutes, success=True):
    return {"command": "get_drive_time", "success": success, "label": "drive",
            "data": {"duration_minutes": minutes} if success else None}


def test_leave_by_computes_relative_minutes():
    # event 13:00Z, drive 18 min, buffer 5 → leave 12:37 → 37 min from now (12:00Z)
    args = {"action": "set", "text": "Leave for Dentist",
            "relative_minutes": {"$leave_by": {
                "event_start": "2026-08-13T13:00:00+00:00", "drive_from_step": 0, "buffer_minutes": 5}}}
    resolved, skip = resolve_step_args(args, [_drive_result(18)], _now())
    assert skip is None
    assert resolved["relative_minutes"] == 37
    assert resolved["text"] == "Leave for Dentist"       # plain args untouched


def test_leave_by_skips_when_departure_already_passed():
    # event 12:10Z, drive 30 min → leave 11:35 → negative → skip (reminder is moot)
    args = {"relative_minutes": {"$leave_by": {
        "event_start": "2026-08-13T12:10:00+00:00", "drive_from_step": 0, "buffer_minutes": 5}}}
    resolved, skip = resolve_step_args(args, [_drive_result(30)], _now())
    assert skip is not None


def test_from_step_substitutes_prior_result_field():
    args = {"minutes": {"$from_step": {"step": 0, "field": "duration_minutes"}}}
    resolved, skip = resolve_step_args(args, [_drive_result(22)], _now())
    assert skip is None and resolved["minutes"] == 22


def test_missing_or_failed_drive_result_skips():
    args = {"relative_minutes": {"$leave_by": {"event_start": "2026-08-13T13:00:00+00:00", "drive_from_step": 0}}}
    _, skip = resolve_step_args(args, [_drive_result(0, success=False)], _now())
    assert skip is not None


def test_plain_args_pass_through_unchanged():
    args = {"action": "set", "text": "hi", "relative_minutes": 20}
    resolved, skip = resolve_step_args(args, [], _now())
    assert skip is None and resolved == args


def test_naive_event_start_treated_as_utc():
    args = {"relative_minutes": {"$leave_by": {"event_start": "2026-08-13T13:00:00", "drive_from_step": 0, "buffer_minutes": 0}}}
    resolved, skip = resolve_step_args(args, [_drive_result(20)], _now())
    assert skip is None and resolved["relative_minutes"] == 40   # 60 min to event - 20 drive
