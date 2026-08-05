"""Scheduling — a durable TIME TRIGGER for the errand plan->card loop.

A ``Schedule`` fires at ``next_fire_at``: the sweep RE-PLANS its ``intent`` against
fresh context and posts a plan card the user must approve (design decision: re-plan +
re-confirm each run — a schedule never acts autonomously). A ONE-SHOT schedule
(``recurrence`` NULL) fires once then goes ``done``; a RECURRING schedule
(``recurrence`` = a cron/interval spec) re-arms ``next_fire_at`` to its next
occurrence after each fire and stays ``active``.

The sweep runs from the app lifespan alongside ``resume_due_timer_workflows`` and reuses
``draft_errand_plan_detached`` — the exact goal->planner->plan-card flow the ``run_errand``
voice tool uses. It is idempotent: an atomic claim (``active``->``done`` for one-shots, or
move-``next_fire_at``-forward for recurring) serializes concurrent sweeps so a schedule
fires exactly once per occurrence, and it survives restarts because the due state lives in
the ``schedules`` row (the durable-engine pattern).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from app.models import Schedule

logger = logging.getLogger("uvicorn")


def _tz(name: str | None):
    """The schedule's IANA timezone (for cron math), falling back to UTC."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name or "UTC")
    except Exception:  # noqa: BLE001 — an unknown zone must not crash the sweep
        return timezone.utc


def compute_next_fire(
    recurrence: str | dict | None, after: datetime, tz_name: str = "UTC"
) -> datetime | None:
    """The next fire instant (naive UTC) STRICTLY after ``after`` for a recurring
    schedule, or None for a one-shot / unparseable spec. ``recurrence`` is JSON (or a
    dict): ``{"type":"interval","interval_seconds":N}`` or ``{"type":"cron","cron":"..."}``.
    Cron is evaluated in the schedule's ``tz_name``; interval is a fixed offset. Mirrors
    the routine scheduler's cron/interval vocabulary so the two stay consistent."""
    if not recurrence:
        return None
    try:
        spec = json.loads(recurrence) if isinstance(recurrence, str) else dict(recurrence)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    rtype = spec.get("type")
    if rtype == "interval":
        secs = int(spec.get("interval_seconds") or 0)
        if secs <= 0:
            return None
        nxt = after + timedelta(seconds=secs)
        while nxt <= after:  # a very short interval + a late sweep — always advance
            nxt += timedelta(seconds=secs)
        return nxt
    if rtype == "cron":
        cron = spec.get("cron")
        if not cron:
            return None
        try:
            from croniter import croniter
        except ImportError:
            logger.warning("croniter not installed — recurring cron schedules can't re-arm")
            return None
        base_local = after.replace(tzinfo=timezone.utc).astimezone(_tz(tz_name))
        try:
            nxt_local = croniter(cron, base_local).get_next(datetime)
        except Exception:  # noqa: BLE001 — a bad cron mustn't crash the sweep
            logger.warning("Invalid cron %r on a schedule", cron)
            return None
        return nxt_local.astimezone(timezone.utc).replace(tzinfo=None)
    return None


async def create_schedule(
    *,
    household_id: str,
    node_id: str | None,
    user_id: int | None,
    intent: str,
    fire_at: datetime,
    timezone: str = "UTC",
    title: str | None = None,
    recurrence: str | None = None,
) -> str:
    """Create an ACTIVE schedule that fires at ``fire_at`` (naive UTC). Returns its id."""
    from app.db import get_session_local

    db = get_session_local()()
    try:
        row = Schedule(
            household_id=household_id, node_id=node_id, user_id=user_id,
            intent=intent, title=title, timezone=timezone or "UTC",
            next_fire_at=fire_at, recurrence=recurrence, state="active",
        )
        db.add(row)
        db.commit()
        logger.info("🗓️ Schedule %s created: fire_at=%s intent=%r", row.id, fire_at, intent[:80])
        return row.id
    finally:
        db.close()


def _claim_one_shot(schedule_id: str, now: datetime) -> bool:
    """Atomically flip a one-shot schedule ``active``->``done`` so concurrent sweeps
    can't double-fire it. Returns True iff THIS caller claimed it."""
    from app.db import get_session_local

    db = get_session_local()()
    try:
        claimed = (
            db.query(Schedule)
            .filter(Schedule.id == schedule_id, Schedule.state == "active")
            .update(
                {Schedule.state: "done", Schedule.last_fired_at: now},
                synchronize_session=False,
            )
        )
        db.commit()
        return bool(claimed)
    finally:
        db.close()


def _claim_and_rearm(schedule_id: str, now: datetime, next_fire: datetime) -> bool:
    """Atomically fire a RECURRING schedule: keep it ``active`` but move
    ``next_fire_at`` forward to the next occurrence (and stamp ``last_fired_at``),
    only while it's still due. The ``next_fire_at <= now`` guard is the
    serialization point — of two concurrent sweeps, only the one that moves the
    still-due row forward claims the fire; the other sees the advanced row and
    no-ops. Returns True iff THIS caller claimed the fire."""
    from app.db import get_session_local

    db = get_session_local()()
    try:
        claimed = (
            db.query(Schedule)
            .filter(
                Schedule.id == schedule_id,
                Schedule.state == "active",
                Schedule.next_fire_at <= now,
            )
            .update(
                {Schedule.next_fire_at: next_fire, Schedule.last_fired_at: now},
                synchronize_session=False,
            )
        )
        db.commit()
        return bool(claimed)
    finally:
        db.close()


async def fire_due_schedules() -> int:
    """Fire schedules due now: re-plan the intent and post a plan card. A one-shot
    (``recurrence`` NULL) is claimed ``active``->``done``; a recurring schedule is
    re-armed to its next occurrence and stays ``active``. Idempotent via the atomic
    claim; a claim loser skips (another sweep took the occurrence)."""
    from app.db import get_session_local
    from app.services.errand_service import draft_errand_plan_detached

    now = datetime.utcnow()
    scan = get_session_local()()
    try:
        due = (
            scan.query(Schedule)
            .filter(Schedule.state == "active", Schedule.next_fire_at <= now)
            .all()
        )
        # Snapshot the fields so we don't hold the scan session while planning.
        pending = [
            (s.id, s.household_id, s.node_id, s.user_id, s.intent, s.recurrence, s.timezone)
            for s in due
        ]
    finally:
        scan.close()

    fired = 0
    for sch_id, household_id, node_id, user_id, intent, recurrence, tz_name in pending:
        # Recurring → re-arm to the next occurrence; one-shot → finish. The claim is
        # the serialization point either way, so exactly one sweep fires the occurrence.
        next_fire = compute_next_fire(recurrence, now, tz_name or "UTC")
        if next_fire is not None:
            if not _claim_and_rearm(sch_id, now, next_fire):
                continue  # another sweep already took this occurrence
        elif not _claim_one_shot(sch_id, now):
            continue
        try:
            # Re-plan against fresh context and post the plan card (re-confirm each run).
            await draft_errand_plan_detached(household_id, node_id, intent, user_id=user_id)
            fired += 1
        except Exception:  # noqa: BLE001 — draft_errand_plan_detached posts its own failure card
            logger.exception("Scheduled fire failed to draft for schedule %s", sch_id)
    if fired:
        logger.info("🗓️ Schedule sweep: fired %d due schedule(s)", fired)
    return fired
