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
from app.services.server_callback_registry import (
    ServerCallbackContext,
    ServerCallbackResult,
    register_server_callback,
)

logger = logging.getLogger("uvicorn")

# The command + callback the "your scheduled errands" card's Cancel buttons target.
SCHEDULE_CALLBACK_COMMAND = "schedule"
SCHEDULE_CANCEL_CALLBACK = "cancel_schedule"
SCHEDULE_LIST_CATEGORY = "schedule"
# Never render more than this many Cancel buttons on one card (a huge card is
# unusable); the spoken/body text notes the rest.
_LIST_CARD_MAX = 8


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


def list_schedules(household_id: str, *, include_done: bool = False) -> list[dict]:
    """The household's schedules for a management view, newest-next-fire first. By
    default only the LIVE ones (active/paused) — the actionable set for list/cancel;
    ``include_done`` adds finished/cancelled for history. Returns plain dicts (no ORM
    escapes the session)."""
    from app.db import get_session_local

    live = ("active", "paused")
    db = get_session_local()()
    try:
        q = db.query(Schedule).filter(Schedule.household_id == household_id)
        if not include_done:
            q = q.filter(Schedule.state.in_(live))
        rows = q.order_by(Schedule.next_fire_at.asc()).all()
        return [
            {
                "id": r.id,
                "intent": r.intent,
                "state": r.state,
                "next_fire_at": r.next_fire_at.isoformat() if r.next_fire_at else None,
                "recurrence": r.recurrence,           # JSON spec or None (one-shot)
                "timezone": r.timezone,
                "last_fired_at": r.last_fired_at.isoformat() if r.last_fired_at else None,
            }
            for r in rows
        ]
    finally:
        db.close()


def cancel_schedule(schedule_id: str, household_id: str) -> bool:
    """Cancel a LIVE (active/paused) schedule so it stops firing. Household-scoped so a
    caller can't cancel another household's schedule. Idempotent: cancelling an
    already-terminal schedule returns False (nothing to do). Returns True iff a live
    schedule was cancelled."""
    from app.db import get_session_local

    db = get_session_local()()
    try:
        cancelled = (
            db.query(Schedule)
            .filter(
                Schedule.id == schedule_id,
                Schedule.household_id == household_id,
                Schedule.state.in_(("active", "paused")),
            )
            .update({Schedule.state: "cancelled"}, synchronize_session=False)
        )
        db.commit()
        return bool(cancelled)
    finally:
        db.close()


# ── List / cancel: the "your scheduled errands" management card ───────────────
# A voice ask ("what errands do I have scheduled?") or the mobile schedules screen
# posts a card listing the household's live schedules, each with its own Cancel
# button (server-plane, target:"server"). Cancel is a TAP — no fuzzy "which one did
# you mean" matching that could stop the wrong errand.

_WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _cron_time_phrase(minute: str, hour: str) -> str:
    """" at 9:00 AM" from cron minute/hour fields when both are concrete, else ""."""
    if not (minute.isdigit() and hour.isdigit()):
        return ""
    h, m = int(hour), int(minute)
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f" at {h12}:{m:02d} {suffix}"


def _describe_recurrence(recurrence: str | None) -> str:
    """A human cadence for a recurrence spec: "every day at 9:00 AM", "on weekdays",
    "every 30 minutes", "every Monday", "monthly on day 5" — or "once" for a one-shot
    (``recurrence`` is None)."""
    if not recurrence:
        return "once"
    try:
        spec = json.loads(recurrence) if isinstance(recurrence, str) else dict(recurrence)
    except (json.JSONDecodeError, TypeError, ValueError):
        return "repeating"
    rtype = spec.get("type")
    if rtype == "interval":
        secs = int(spec.get("interval_seconds") or 0)
        if secs and secs % 3600 == 0:
            h = secs // 3600
            return "every hour" if h == 1 else f"every {h} hours"
        if secs and secs % 60 == 0:
            m = secs // 60
            return "every minute" if m == 1 else f"every {m} minutes"
        return "on a repeating interval"
    if rtype == "cron":
        parts = (spec.get("cron") or "").split()
        if len(parts) != 5:
            return "repeating"
        minute, hour, dom, _mon, dow = parts
        tod = _cron_time_phrase(minute, hour)
        if dow == "1-5":
            return f"every weekday{tod}"
        if dow.isdigit():
            return f"every {_WEEKDAY_NAMES[int(dow) % 7]}{tod}"
        if dom.isdigit():
            return f"monthly on day {dom}{tod}"
        return f"every day{tod}"
    return "repeating"


def _local_when(iso_utc: str | None, tz_name: str | None) -> str:
    """A short local phrase for the next fire ("Tue Aug 5, 9:00 AM"), or "" if unknown."""
    if not iso_utc:
        return ""
    try:
        dt = datetime.fromisoformat(iso_utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(_tz(tz_name))
        return local.strftime("%a %b %-d, %-I:%M %p").replace(":00", "")
    except (ValueError, TypeError):
        return ""


def describe_schedule(sch: dict) -> str:
    """One human line for a schedule dict — "check the weather · every day at 9:00 AM
    · next Tue Aug 5, 9 AM". The pieces present depend on the schedule."""
    intent = (sch.get("intent") or "an errand").strip()
    cadence = _describe_recurrence(sch.get("recurrence"))
    nxt = _local_when(sch.get("next_fire_at"), sch.get("timezone"))
    bits = [intent, cadence]
    if nxt:
        bits.append(f"next {nxt}")
    return " · ".join(b for b in bits if b)


def schedule_view(sch: dict) -> dict:
    """A raw schedule dict enriched with the display strings the mobile renders
    as-is — so the phone screen and the voice card agree on how a cadence reads
    (CC owns the vocabulary). Adds ``is_recurring`` / ``cadence`` / ``next_local`` /
    ``description`` alongside the stored fields."""
    return {
        **sch,
        "is_recurring": bool(sch.get("recurrence")),
        "cadence": _describe_recurrence(sch.get("recurrence")),
        "next_local": _local_when(sch.get("next_fire_at"), sch.get("timezone")),
        "description": describe_schedule(sch),
    }


def list_schedule_views(household_id: str, *, include_done: bool = False) -> list[dict]:
    """``list_schedules`` enriched for display (the mobile schedules screen)."""
    return [schedule_view(s) for s in list_schedules(household_id, include_done=include_done)]


def _schedule_list_body(schedules: list[dict]) -> str:
    """Markdown body for the management card."""
    if not schedules:
        return "You don't have any errands scheduled right now."
    lines = ["Your scheduled errands:", ""]
    for i, s in enumerate(schedules, start=1):
        lines.append(f"{i}. {describe_schedule(s)}")
    if len(schedules) > _LIST_CARD_MAX:
        lines += ["", f"_Showing the first {_LIST_CARD_MAX} of {len(schedules)}._"]
    lines += ["", "Tap **Cancel** on any errand to stop it."]
    return "\n".join(lines)


def build_schedule_list_card_metadata(schedules: list[dict], household_id: str) -> dict:
    """The management card's ``metadata`` — a read-only ``schedules`` view plus one
    server-plane Cancel button per schedule (capped at ``_LIST_CARD_MAX``). ``data``
    carries the ``schedule_id`` so the tap cancels exactly that one."""
    shown = schedules[:_LIST_CARD_MAX]
    return {
        "household_id": household_id,
        "schedules": shown,
        "interactive_elements": [
            {
                "id": f"cancel-schedule-{s['id']}",
                "label": f"Cancel: {(s.get('intent') or 'errand')[:32]}",
                "command": SCHEDULE_CALLBACK_COMMAND,
                "callback": SCHEDULE_CANCEL_CALLBACK,
                "target": "server",
                "data": {"schedule_id": s["id"]},
            }
            for s in shown
        ],
    }


def post_schedule_list_card(household_id: str, user_id: int | None) -> tuple[int, str | None]:
    """Post the household's "scheduled errands" management card. Returns
    ``(count, item_id)`` — ``count`` is the number of live schedules (0 → no card is
    posted, the caller just speaks "nothing scheduled"). Best-effort on the post."""
    from app.services.inbox_notification_service import post_inbox_item_sync

    schedules = list_schedules(household_id)
    if not schedules:
        return 0, None
    n = len(schedules)
    title = f"🗓️ {n} scheduled errand{'s' if n != 1 else ''}"
    summary = describe_schedule(schedules[0]) if n == 1 else f"{n} errands scheduled"
    item_id = post_inbox_item_sync(
        household_id=household_id,
        title=title,
        summary=summary,
        body=_schedule_list_body(schedules),
        category=SCHEDULE_LIST_CATEGORY,
        metadata=build_schedule_list_card_metadata(schedules, household_id),
        user_id=user_id,
        push=True,
        target_type="user" if user_id is not None else "household",
    )
    return n, item_id


def _handle_cancel_schedule(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Cancel tap on the management card → cancel that schedule (household-scoped),
    then re-post the list card so it reflects the removal. Idempotent: a
    second tap on an already-cancelled schedule reports it's already stopped."""
    schedule_id = ctx.data.get("schedule_id")
    if not schedule_id:
        return ServerCallbackResult(success=False, error="No schedule to cancel.")
    cancelled = cancel_schedule(schedule_id, ctx.household_id)
    # Refresh the management card in place of the tap's own result card so the user
    # sees the updated list (or the "nothing left" state).
    remaining, _ = post_schedule_list_card(ctx.household_id, ctx.user_id)
    title = "🗓️ Errand cancelled" if cancelled else "Errand already stopped"
    if remaining:
        summary = f"Done — you have {remaining} errand{'s' if remaining != 1 else ''} still scheduled."
    else:
        summary = "Done — you have no errands scheduled now."
    return ServerCallbackResult(
        success=True,
        context_data={"inbox": {"title": title, "summary": summary,
                                "metadata": {"household_id": ctx.household_id}}},
    )


def register_schedule_callbacks() -> None:
    """Wire the management card's Cancel taps to their handler. Call at startup
    (main.py lifespan); an unregistered pair 400s at the tap."""
    register_server_callback(
        SCHEDULE_CALLBACK_COMMAND, SCHEDULE_CANCEL_CALLBACK, _handle_cancel_schedule
    )
    logger.info("🗓️ Schedule server callbacks registered")


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
