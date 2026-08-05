"""Scheduling — a durable TIME TRIGGER for the errand plan->card loop.

A ``Schedule`` fires at ``next_fire_at``: the sweep RE-PLANS its ``intent`` against
fresh context and posts a plan card the user must approve (design decision: re-plan +
re-confirm each run — a schedule never acts autonomously). One-shot for now
(``recurrence`` NULL); recurrence will re-arm ``next_fire_at`` after each fire.

The sweep runs from the app lifespan alongside ``resume_due_timer_workflows`` and reuses
``draft_errand_plan_detached`` — the exact goal->planner->plan-card flow the ``run_errand``
voice tool uses. It is idempotent: an atomic ``active``->``done`` claim serializes
concurrent sweeps so a schedule fires exactly once, and it survives restarts because the
due state lives in the ``schedules`` row (the durable-engine pattern).
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.models import Schedule

logger = logging.getLogger("uvicorn")


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


async def fire_due_schedules() -> int:
    """Fire schedules due now: re-plan the intent and post a plan card. Idempotent via
    the atomic claim; one-shot only (``recurrence`` NULL) for now — recurrence re-arm
    lands in the next slice."""
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
            (s.id, s.household_id, s.node_id, s.user_id, s.intent) for s in due
        ]
    finally:
        scan.close()

    fired = 0
    for sch_id, household_id, node_id, user_id, intent in pending:
        if not _claim_one_shot(sch_id, now):
            continue  # another sweep already took it
        try:
            # Re-plan against fresh context and post the plan card (re-confirm each run).
            await draft_errand_plan_detached(household_id, node_id, intent, user_id=user_id)
            fired += 1
        except Exception:  # noqa: BLE001 — draft_errand_plan_detached posts its own failure card
            logger.exception("Scheduled fire failed to draft for schedule %s", sch_id)
    if fired:
        logger.info("🗓️ Schedule sweep: fired %d due schedule(s)", fired)
    return fired
