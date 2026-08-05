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
