"""Scheduling sweep: a due schedule fires the plan->card loop exactly once.

Real query logic against an in-memory SQLite (shared connection) with the plan-draft
mocked — proves due/not-due selection, the atomic one-shot claim (no double-fire), and
that the intent is what gets re-planned.
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, Schedule
from app.services import schedule_service

_DRAFT = "app.services.errand_service.draft_errand_plan_detached"


@pytest.fixture
def sched_db():
    """In-memory DB shared across sessions (StaticPool) with just the schedules table,
    patched in as get_session_local so the service's own sessions hit it."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine, tables=[Schedule.__table__])
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with patch("app.db.get_session_local", return_value=SessionLocal):
        yield SessionLocal
    engine.dispose()


def _add(SessionLocal, **kw) -> str:
    db = SessionLocal()
    row = Schedule(**{
        "household_id": "hh-1", "node_id": "node-1", "user_id": 7,
        "intent": "check my refill", "timezone": "UTC",
        "next_fire_at": datetime.utcnow() - timedelta(minutes=1), "state": "active",
        **kw,
    })
    db.add(row)
    db.commit()
    sid = row.id
    db.close()
    return sid


def _get(SessionLocal, sid) -> Schedule:
    db = SessionLocal()
    try:
        return db.query(Schedule).filter_by(id=sid).first()
    finally:
        db.close()


def test_due_schedule_fires_and_is_marked_done(sched_db):
    sid = _add(sched_db)
    with patch(_DRAFT, new=AsyncMock()) as draft:
        fired = asyncio.run(schedule_service.fire_due_schedules())
    assert fired == 1
    draft.assert_awaited_once()
    household, node_id, intent = draft.await_args.args
    assert (household, intent) == ("hh-1", "check my refill")
    assert _get(sched_db, sid).state == "done"  # won't fire again


def test_not_yet_due_is_skipped(sched_db):
    _add(sched_db, next_fire_at=datetime.utcnow() + timedelta(hours=1))
    with patch(_DRAFT, new=AsyncMock()) as draft:
        fired = asyncio.run(schedule_service.fire_due_schedules())
    assert fired == 0 and draft.await_count == 0


def test_one_shot_claim_is_idempotent(sched_db):
    sid = _add(sched_db)
    now = datetime.utcnow()
    assert schedule_service._claim_one_shot(sid, now) is True   # first caller wins
    assert schedule_service._claim_one_shot(sid, now) is False  # already done → no double-fire


def test_a_failed_draft_does_not_leave_the_schedule_active(sched_db):
    # The plan-draft posts its own failure card; the schedule is still consumed (done),
    # so a broken planner can't make the sweep retry forever.
    sid = _add(sched_db)
    with patch(_DRAFT, new=AsyncMock(side_effect=RuntimeError("planner down"))):
        fired = asyncio.run(schedule_service.fire_due_schedules())
    assert fired == 0                       # nothing counted as fired
    assert _get(sched_db, sid).state == "done"  # but it was claimed, not left to loop


def test_create_schedule_persists_an_active_row(sched_db):
    fire = datetime.utcnow() + timedelta(days=1)
    sid = asyncio.run(schedule_service.create_schedule(
        household_id="hh-2", node_id="n1", user_id=3,
        intent="call the dentist tomorrow morning", fire_at=fire))
    row = _get(sched_db, sid)
    assert row.state == "active"
    assert row.intent == "call the dentist tomorrow morning"
    assert row.household_id == "hh-2" and row.recurrence is None


# ── Slice 2: recurrence ───────────────────────────────────────────────────────


def test_compute_next_fire_interval():
    after = datetime(2026, 8, 5, 12, 0, 0)
    nxt = schedule_service.compute_next_fire({"type": "interval", "interval_seconds": 3600}, after)
    assert nxt == datetime(2026, 8, 5, 13, 0, 0)


def test_compute_next_fire_interval_always_advances_past_after():
    # A short interval + a late sweep must still land in the future.
    after = datetime(2026, 8, 5, 12, 0, 30)
    nxt = schedule_service.compute_next_fire({"type": "interval", "interval_seconds": 60}, after)
    assert nxt is not None and nxt > after


def test_compute_next_fire_cron_daily_utc():
    # 9am daily; from noon on the 5th → next is 9am on the 6th (UTC).
    after = datetime(2026, 8, 5, 12, 0, 0)
    nxt = schedule_service.compute_next_fire({"type": "cron", "cron": "0 9 * * *"}, after, "UTC")
    assert nxt == datetime(2026, 8, 6, 9, 0, 0)


def test_compute_next_fire_none_for_oneshot_and_junk():
    after = datetime(2026, 8, 5, 12, 0, 0)
    assert schedule_service.compute_next_fire(None, after) is None
    assert schedule_service.compute_next_fire("", after) is None
    assert schedule_service.compute_next_fire("{bad json", after) is None
    assert schedule_service.compute_next_fire({"type": "interval", "interval_seconds": 0}, after) is None
    assert schedule_service.compute_next_fire({"type": "cron"}, after) is None  # no cron string


def test_recurring_schedule_fires_and_rearms(sched_db):
    # A recurring schedule fires, STAYS active, and moves next_fire_at into the future.
    sid = _add(sched_db, recurrence=json.dumps({"type": "interval", "interval_seconds": 86400}))
    with patch(_DRAFT, new=AsyncMock()) as draft:
        fired = asyncio.run(schedule_service.fire_due_schedules())
    assert fired == 1
    draft.assert_awaited_once()
    row = _get(sched_db, sid)
    assert row.state == "active"                     # recurring → NOT 'done'
    assert row.next_fire_at > datetime.utcnow()      # re-armed forward
    assert row.last_fired_at is not None


def test_recurring_rearm_claim_is_idempotent(sched_db):
    sid = _add(sched_db, recurrence=json.dumps({"type": "interval", "interval_seconds": 3600}))
    now = datetime.utcnow()
    nxt = now + timedelta(hours=1)
    assert schedule_service._claim_and_rearm(sid, now, nxt) is True   # first sweep claims
    assert schedule_service._claim_and_rearm(sid, now, nxt) is False  # moved forward → no double-fire


# ── Slice 2: list / cancel management card ────────────────────────────────────


def test_list_schedules_returns_only_live_ordered_by_next_fire(sched_db):
    now = datetime.utcnow()
    _add(sched_db, intent="weather", next_fire_at=now + timedelta(hours=2))
    _add(sched_db, intent="dentist", next_fire_at=now + timedelta(hours=1))
    _add(sched_db, intent="finished", state="done")
    _add(sched_db, intent="killed", state="cancelled")
    rows = schedule_service.list_schedules("hh-1")
    assert [r["intent"] for r in rows] == ["dentist", "weather"]  # next-fire asc, live only
    assert {"id", "intent", "state", "next_fire_at", "recurrence", "timezone",
            "last_fired_at"} <= set(rows[0])


def test_list_schedules_is_household_scoped(sched_db):
    _add(sched_db, household_id="hh-1", intent="mine")
    _add(sched_db, household_id="hh-2", intent="theirs")
    assert [r["intent"] for r in schedule_service.list_schedules("hh-1")] == ["mine"]


def test_cancel_schedule_stops_it_and_is_idempotent(sched_db):
    sid = _add(sched_db, intent="weather")
    assert schedule_service.cancel_schedule(sid, "hh-1") is True
    assert _get(sched_db, sid).state == "cancelled"       # stops firing
    assert schedule_service.cancel_schedule(sid, "hh-1") is False  # already terminal → no-op


def test_cancel_schedule_wont_cross_households(sched_db):
    sid = _add(sched_db, household_id="hh-1")
    assert schedule_service.cancel_schedule(sid, "hh-2") is False   # not theirs
    assert _get(sched_db, sid).state == "active"                    # untouched


def test_describe_recurrence_phrases():
    d = schedule_service._describe_recurrence
    assert d(None) == "once"
    assert d(json.dumps({"type": "cron", "cron": "0 9 * * *"})) == "every day at 9:00 AM"
    assert d(json.dumps({"type": "cron", "cron": "30 8 * * 1-5"})) == "every weekday at 8:30 AM"
    assert d(json.dumps({"type": "cron", "cron": "0 14 * * 1"})) == "every Monday at 2:00 PM"
    assert d(json.dumps({"type": "cron", "cron": "0 9 5 * *"})) == "monthly on day 5 at 9:00 AM"
    assert d(json.dumps({"type": "interval", "interval_seconds": 3600})) == "every hour"
    assert d(json.dumps({"type": "interval", "interval_seconds": 7200})) == "every 2 hours"
    assert d(json.dumps({"type": "interval", "interval_seconds": 1800})) == "every 30 minutes"
    assert d("{bad json") == "repeating"


def test_list_card_metadata_one_cancel_button_per_schedule():
    scheds = [
        {"id": "s1", "intent": "weather", "recurrence": None, "next_fire_at": None, "timezone": "UTC"},
        {"id": "s2", "intent": "dentist", "recurrence": None, "next_fire_at": None, "timezone": "UTC"},
    ]
    md = schedule_service.build_schedule_list_card_metadata(scheds, "hh-1")
    assert md["household_id"] == "hh-1"
    btns = md["interactive_elements"]
    assert len(btns) == 2
    assert all(b["target"] == "server"
               and b["callback"] == schedule_service.SCHEDULE_CANCEL_CALLBACK for b in btns)
    assert {b["data"]["schedule_id"] for b in btns} == {"s1", "s2"}


def test_list_card_caps_the_buttons():
    scheds = [{"id": f"s{i}", "intent": f"e{i}", "recurrence": None,
               "next_fire_at": None, "timezone": "UTC"} for i in range(12)]
    md = schedule_service.build_schedule_list_card_metadata(scheds, "hh-1")
    assert len(md["interactive_elements"]) == schedule_service._LIST_CARD_MAX


def test_post_list_card_no_card_when_empty(sched_db):
    with patch("app.services.inbox_notification_service.post_inbox_item_sync") as post:
        count, item = schedule_service.post_schedule_list_card("hh-1", 7)
    assert (count, item) == (0, None)
    post.assert_not_called()  # nothing scheduled → no card


def test_post_list_card_posts_when_present(sched_db):
    _add(sched_db, intent="weather")
    with patch("app.services.inbox_notification_service.post_inbox_item_sync",
               return_value="item-1") as post:
        count, item = schedule_service.post_schedule_list_card("hh-1", 7)
    assert count == 1 and item == "item-1"
    kw = post.call_args.kwargs
    assert kw["household_id"] == "hh-1"
    assert kw["category"] == schedule_service.SCHEDULE_LIST_CATEGORY
    assert kw["metadata"]["interactive_elements"][0]["data"]["schedule_id"]


def test_handle_cancel_schedule_cancels_and_reposts_updated_list(sched_db):
    from app.services.server_callback_registry import ServerCallbackContext

    sid = _add(sched_db, intent="weather")
    _add(sched_db, intent="dentist")  # one remains after the cancel
    ctx = ServerCallbackContext(job_id="j1", household_id="hh-1", user_id=7, data={"schedule_id": sid})
    with patch("app.services.inbox_notification_service.post_inbox_item_sync",
               return_value="item-1") as post:
        res = schedule_service._handle_cancel_schedule(ctx)
    assert res.success is True
    assert _get(sched_db, sid).state == "cancelled"
    post.assert_called_once()                       # re-posted the updated list
    assert "1 errand" in res.context_data["inbox"]["summary"]


def test_handle_cancel_schedule_missing_id_is_an_error():
    from app.services.server_callback_registry import ServerCallbackContext

    ctx = ServerCallbackContext(job_id="j1", household_id="hh-1", user_id=7, data={})
    res = schedule_service._handle_cancel_schedule(ctx)
    assert res.success is False
