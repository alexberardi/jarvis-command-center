"""Errand v0 — routine completion notification.

Covers the new completion-delivery wiring for *detached* routine runs
(scheduler / background), which is the one genuinely-new piece of the v0
"background routine" slice — the headless execute + report round-trip already
ships. See prds/errand-runner.md §8.

- ``_notify_routine_complete`` maps every terminal status to an honest card,
  targets the triggering user when known else the household, and is non-fatal.
- ``execute_routine_on_node`` fires it only when ``notify_on_complete`` is set
  (the default-False is the double-notify guard for the mobile run-now path).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.api import routines
from app.api.routines import _notify_routine_complete

_NOTIFY_TARGET = "app.services.inbox_notification_service.post_inbox_item_sync"


def _routine(name="Morning Briefing", slug="morning", id="rt-1"):
    return SimpleNamespace(name=name, slug=slug, id=id)


# ── _notify_routine_complete: status → card ──────────────────────────────────


def test_success_card_uses_message_and_household_target():
    with patch(_NOTIFY_TARGET) as post:
        _notify_routine_complete(
            _routine(), "hh-1", {"status": "success", "message": "Lights on, weather read."}
        )
    kw = post.call_args.kwargs
    assert kw["title"] == "✅ 'Morning Briefing' finished"
    assert kw["summary"] == "Lights on, weather read."
    assert kw["category"] == "routine"
    assert kw["target_type"] == "household"
    assert kw["user_id"] is None
    assert kw["metadata"]["status"] == "success"
    assert kw["metadata"]["routine_id"] == "rt-1"


def test_partial_card_flags_issues():
    with patch(_NOTIFY_TARGET) as post:
        _notify_routine_complete(
            _routine(), "hh-1", {"status": "partial", "message": "2 done, 1 failed."}
        )
    assert post.call_args.kwargs["title"] == "⚠️ 'Morning Briefing' finished with issues"


def test_timeout_card_has_honest_summary():
    with patch(_NOTIFY_TARGET) as post:
        _notify_routine_complete(_routine(), "hh-1", {"status": "timeout", "message": None})
    kw = post.call_args.kwargs
    assert kw["title"] == "⏱️ 'Morning Briefing' didn't finish"
    assert kw["summary"] == "The node didn't respond in time."


def test_failed_card_falls_back_to_error():
    with patch(_NOTIFY_TARGET) as post:
        _notify_routine_complete(
            _routine(), "hh-1", {"status": "failed", "message": None, "error": "the target node was offline"}
        )
    kw = post.call_args.kwargs
    assert kw["title"] == "⚠️ 'Morning Briefing' couldn't run"
    assert kw["summary"] == "the target node was offline"


def test_unknown_status_treated_as_failure():
    with patch(_NOTIFY_TARGET) as post:
        _notify_routine_complete(_routine(), "hh-1", {"status": "weird"})
    assert post.call_args.kwargs["title"] == "⚠️ 'Morning Briefing' couldn't run"


def test_user_target_when_triggerer_known():
    with patch(_NOTIFY_TARGET) as post:
        _notify_routine_complete(
            _routine(), "hh-1", {"status": "success", "message": "done"}, user_id=42
        )
    kw = post.call_args.kwargs
    assert kw["target_type"] == "user"
    assert kw["user_id"] == 42


def test_notify_is_non_fatal_on_post_failure():
    # A notification blowing up must never propagate into the run.
    with patch(_NOTIFY_TARGET, side_effect=RuntimeError("notifications down")):
        _notify_routine_complete(_routine(), "hh-1", {"status": "success", "message": "done"})


# ── execute_routine_on_node: the notify flag gates delivery ───────────────────


def _run_execute(notify_on_complete):
    routine = _routine()
    db = MagicMock()
    fake = {"output": {"success": True, "passed": 2, "failed": 0, "message": "Done."}}
    with patch("app.services.node_command_service.get_node_command_service"), patch(
        "app.api.routines._wait_for_result_file", new=AsyncMock(return_value=fake)
    ), patch("app.api.routines._notify_routine_complete") as notify:
        result = asyncio.run(
            routines.execute_routine_on_node(
                db, "hh-1", routine, "node-1", "scheduled", notify_on_complete=notify_on_complete
            )
        )
    return result, notify, routine


def test_execute_notifies_when_flag_set():
    result, notify, routine = _run_execute(True)
    assert result["status"] == "success"
    notify.assert_called_once()
    args = notify.call_args.args
    assert args[0] is routine and args[1] == "hh-1"
    assert args[2]["status"] == "success"


def test_execute_silent_when_flag_unset():
    result, notify, _ = _run_execute(False)
    assert result["status"] == "success"
    notify.assert_not_called()
