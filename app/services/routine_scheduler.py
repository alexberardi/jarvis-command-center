"""CC-side routine scheduler.

Scheduling lives here (not on nodes) so it survives node reboots, respects each
routine's timezone, and reuses the run-now execution path. A background worker in
main.py calls run_due_routines() every ~30s when routines.scheduler_enabled is on.

Each scheduled routine carries a `schedule` JSON:
  {type:"cron"|"interval", cron?, interval_seconds?, timezone, target_node_id,
   enabled, last_fired_at?}
We fire on the routine's target_node_id (defaulting to the household primary node,
resolved at save time), advance last_fired_at, and write a routine_executions row.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.models import Node, Routine

logger = logging.getLogger("uvicorn")

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _tz(name: str | None):
    if ZoneInfo is not None and name:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    return timezone.utc


def _aware_utc(dt: datetime) -> datetime:
    """Coerce a possibly-naive datetime to aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_last(schedule: dict) -> datetime | None:
    raw = schedule.get("last_fired_at")
    if not raw:
        return None
    try:
        return _aware_utc(datetime.fromisoformat(raw))
    except (ValueError, TypeError):
        return None


def is_due(schedule: dict, now_utc: datetime, last: datetime | None) -> bool:
    """Whether a schedule should fire at now_utc, given its last fire time."""
    if not schedule or not schedule.get("enabled", True):
        return False

    stype = schedule.get("type")
    if stype == "interval":
        secs = int(schedule.get("interval_seconds") or 0)
        if secs <= 0 or last is None:
            return False  # first-seen interval routines get a baseline, not a fire
        return (now_utc - last).total_seconds() >= secs

    if stype == "cron":
        cron = schedule.get("cron")
        if not cron:
            return False
        try:
            from croniter import croniter
        except ImportError:
            logger.warning("croniter not installed — cron-scheduled routines disabled")
            return False
        tz = _tz(schedule.get("timezone"))
        now_local = now_utc.astimezone(tz)
        # Base the search at the last fire, or one minute ago to catch the
        # current minute on first run.
        base = (last.astimezone(tz) if last else now_local - timedelta(minutes=1))
        try:
            next_fire = croniter(cron, base).get_next(datetime)
        except (ValueError, KeyError) as exc:
            logger.warning("Invalid cron '%s': %s", cron, exc)
            return False
        return next_fire <= now_local

    return False


def _mark_fired(db: Session, routine: Routine, schedule: dict, when_utc: datetime) -> None:
    schedule = dict(schedule)
    schedule["last_fired_at"] = when_utc.isoformat()
    routine.schedule = json.dumps(schedule)
    db.commit()


async def run_due_routines(db: Session, now_utc: datetime | None = None) -> int:
    """Fire every scheduled routine that is due. Returns the number fired."""
    from app.api.routines import (  # lazy: avoid import cycle
        _notify_routine_complete,
        execute_routine_on_node,
    )

    now_utc = now_utc or datetime.now(timezone.utc)
    fired = 0

    rows = (
        db.query(Routine)
        .filter(Routine.enabled.is_(True), Routine.schedule.isnot(None))
        .all()
    )

    for routine in rows:
        try:
            schedule = json.loads(routine.schedule) if routine.schedule else None
        except (ValueError, TypeError):
            continue
        if not schedule or not schedule.get("enabled", True):
            continue

        last = _parse_last(schedule)

        # Establish a baseline for interval routines that have never fired, so
        # they fire one interval from now rather than immediately on creation.
        if schedule.get("type") == "interval" and last is None:
            _mark_fired(db, routine, schedule, now_utc)
            continue

        if not is_due(schedule, now_utc, last):
            continue

        node_id = schedule.get("target_node_id")
        node = None
        if node_id:
            node = (
                db.query(Node)
                .filter(
                    Node.node_id == node_id,
                    Node.household_id == routine.household_id,
                    Node.is_active.is_(True),
                )
                .first()
            )
        if node is None or not node.is_online():
            logger.info(
                "Scheduled routine '%s' skipped — target node %s unavailable",
                routine.slug, node_id,
            )
            # Don't vanish: a scheduled run that couldn't happen must say so.
            _notify_routine_complete(
                routine,
                routine.household_id,
                {"status": "failed", "message": None, "error": "the target node was offline"},
            )
            _mark_fired(db, routine, schedule, now_utc)  # respect cadence, don't pile up
            continue

        logger.info("Firing scheduled routine '%s' on %s", routine.slug, node_id)
        await execute_routine_on_node(
            db, routine.household_id, routine, node_id, "scheduled", notify_on_complete=True
        )
        _mark_fired(db, routine, schedule, now_utc)
        fired += 1

    return fired
