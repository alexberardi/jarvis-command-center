"""Schedule Errand Tool — schedule an errand to run at a future time (once).

Example: "Tomorrow at 9am, call the dentist to book a cleaning."

Voice entry for the scheduler: a spoken goal + a time → a ``Schedule`` row. At the
scheduled time the sweep RE-PLANS the goal against fresh context and posts a plan card
to approve (re-plan + re-confirm each run — nothing runs unattended). Supports a
one-time fire (``fire_at``) or a repeating cadence (``recurrence``: daily / weekdays /
weekly / monthly / hourly / "every N minutes"), which re-arms after each fire.

Mirrors run_errand_tool: derive household/node/speaker from node_context, kick the
schedule off, return a spoken ack. ``fire_at`` is a plain string (NOT format:date-time)
because the shared auto-resolver drops the clock time when a whole phrase arrives as one
string — so ``_resolve_fire_at`` parses the day AND the time here, in the NODE'S timezone
(``node_context["timezone"]``). Resolving in the wrong zone silently shifts the fire time.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.date_resolution import parse_time_string
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")

_SCHEDULE_TASKS: set[asyncio.Task] = set()

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
# Coarse named times when the user gives no clock time.
_NAMED_TIMES = {
    "midnight": (0, 0), "morning": (9, 0), "noon": (12, 0), "afternoon": (13, 0),
    "evening": (18, 0), "tonight": (19, 0), "night": (20, 0),
}
_CLOCK_RE = re.compile(r"\b(\d{1,4})(?:[:.](\d{2}))?\s*(am|pm)\b", re.IGNORECASE)


def _zone(tz_name: str):
    from zoneinfo import ZoneInfo
    try:
        return ZoneInfo(tz_name or "UTC")
    except Exception:  # noqa: BLE001 — unknown zone → UTC
        return ZoneInfo("UTC")


def _resolve_fire_at(raw: str, tz_name: str) -> datetime | None:
    """Resolve the model's natural time phrase ("tonight at 7:51pm", "tomorrow at 9am",
    "friday at noon") — or an ISO instant — into a naive-UTC instant. The date and the
    clock time are parsed HERE (the shared auto-resolver drops the time when the whole
    phrase arrives as one string). Returns None if no time can be determined."""
    s = (raw or "").strip()
    if not s:
        return None

    # Already an ISO instant? (auto-resolved, or the model gave one)
    iso = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_zone(tz_name))
        # A bare date at midnight is almost never the intended time — fall through to
        # phrase parsing so "tonight at 7:51pm" isn't silently turned into midnight.
        if not (dt.hour == 0 and dt.minute == 0):
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        pass

    low = s.lower()
    tz = _zone(tz_name)
    today = datetime.now(tz).date()

    # Date. Track whether the user actually NAMED a day — a bare "tomorrow"/weekday
    # with no clock time should still schedule (at a default hour) rather than fail;
    # only a phrase with neither a day nor a time is truly unschedulable.
    date = today
    day_named = "tomorrow" in low or "today" in low
    if "tomorrow" in low:
        date = today + timedelta(days=1)
    else:
        for name, wd in _WEEKDAYS.items():
            if name in low:
                day_named = True
                days = ((wd - today.weekday()) % 7) or 7  # the NEXT such weekday
                date = today + timedelta(days=days)
                break

    # Time — a clock time wins; else a coarse named time.
    hour, minute = None, 0
    m = _CLOCK_RE.search(low)
    if m:
        token = f"{m.group(1)}{'_' + m.group(2) if m.group(2) else ''}{m.group(3)}"
        hour, minute = parse_time_string(token)
    else:
        for name, (h, mm) in _NAMED_TIMES.items():
            if name in low:
                hour, minute = h, mm
                break
    if hour is None:
        if day_named:
            # A named day but no clock time ("tomorrow", "friday") — the model often
            # drops the time even when the user gave one. Default to 9am on that day
            # rather than silently failing (which the model then narrates as success).
            hour, minute = 9, 0
        else:
            return None  # no day AND no time → genuinely can't schedule; ask the user

    local = datetime(date.year, date.month, date.day, hour, minute, tzinfo=tz)
    return local.astimezone(timezone.utc).replace(tzinfo=None)


def _build_recurrence(descriptor: str | None, fire_dt_utc: datetime, tz_name: str) -> str | None:
    """Turn a coarse recurrence phrase ("daily", "every weekday", "weekly", "monthly",
    "hourly", "every 30 minutes") + the first occurrence's LOCAL time into a
    ``schedule_service`` recurrence spec (JSON), or None for a one-shot / unrecognized
    phrase (treated as one-shot — safer than guessing a cadence). Cron cadences fix the
    time-of-day (and day-of-week / day-of-month) from ``fire_dt`` in the node's zone, so
    the schedule re-fires at the same wall-clock time the user gave."""
    d = (descriptor or "").strip().lower()
    if not d or d in ("none", "once", "one-time", "one time", "just once", "no", "never"):
        return None
    # Explicit interval phrasing: "every 30 minutes", "every 2 hours".
    m = re.search(r"every\s+(\d+)\s*(minute|min|hour|hr)s?", d)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        secs = n * 60 if unit.startswith("min") else n * 3600
        return json.dumps({"type": "interval", "interval_seconds": secs}) if secs > 0 else None
    if "hour" in d:  # "hourly", "every hour"
        return json.dumps({"type": "interval", "interval_seconds": 3600})
    local = fire_dt_utc.replace(tzinfo=timezone.utc).astimezone(_zone(tz_name))
    hh, mm = local.hour, local.minute
    if "weekday" in d:  # "weekdays", "every weekday"
        return json.dumps({"type": "cron", "cron": f"{mm} {hh} * * 1-5"})
    if "week" in d:  # "weekly", "every week", "every Monday"
        cron_dow = (local.weekday() + 1) % 7  # Python Mon=0..Sun=6 → cron Sun=0..Sat=6
        return json.dumps({"type": "cron", "cron": f"{mm} {hh} * * {cron_dow}"})
    if "month" in d:  # "monthly", "every month"
        return json.dumps({"type": "cron", "cron": f"{mm} {hh} {local.day} * *"})
    if "daily" in d or "day" in d:  # "daily" ('day' isn't a substring of it), "every day"
        return json.dumps({"type": "cron", "cron": f"{mm} {hh} * * *"})
    return None  # unrecognized → one-shot


def _default_recurring_fire(tz_name: str) -> datetime:
    """First fire for a recurring errand that gave NO clock time ("every day, check
    the weather"): 9:00 AM local today (a sensible 'daily check' hour). For a cron
    cadence this fixes the time-of-day; for an interval cadence the past-9am then
    rolls forward to now+interval. Returns naive UTC."""
    tz = _zone(tz_name)
    today = datetime.now(tz).date()
    local = datetime(today.year, today.month, today.day, 9, 0, tzinfo=tz)
    return local.astimezone(timezone.utc).replace(tzinfo=None)


class ScheduleErrandTool(IServerTool):
    """Schedule a background errand to run at a future time."""

    @property
    def name(self) -> str:
        return "schedule_errand"

    @property
    def description(self) -> str:
        return (
            "Use this WHENEVER the user says 'schedule an errand', or asks for an errand / "
            "background task to happen at a FUTURE time — 'later', 'tonight', 'tomorrow at "
            "9am', 'at 7:30pm', 'this evening', 'in an hour' — OR on a REPEATING schedule — "
            "'every morning at 8', 'each weekday', 'every Monday', 'monthly', 'every hour'. "
            "Put what to do in `goal`, the (first) time in `fire_at`, and any repeat cadence "
            "in `recurrence` (omit it for a one-time errand). Each time it fires, Jarvis "
            "re-plans the goal and sends a plan card to approve — nothing runs unattended. "
            "PREFER this over run_errand any time the user names a future or repeating time; "
            "use run_errand only for 'run an errand' to do NOW."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "What the errand should accomplish, in the user's own words with "
                        "every detail they gave (do NOT include the time here)."
                    ),
                },
                "fire_at": {
                    "type": "string",
                    "description": (
                        "When to run it (the FIRST time, if it repeats), as the natural day "
                        "+ clock time the user gave — e.g. 'tonight at 7:51pm', 'tomorrow at "
                        "9am', 'friday at noon', 'at 8am'. Include the time; copy it exactly."
                    ),
                },
                "recurrence": {
                    "type": "string",
                    "description": (
                        "Repeat cadence, if the user wants it to REPEAT — one of: 'daily', "
                        "'weekdays', 'weekly', 'monthly', 'hourly', or 'every N minutes/hours'. "
                        "OMIT entirely for a one-time errand. The time-of-day comes from "
                        "`fire_at` (e.g. fire_at='at 8am' + recurrence='daily' → every day at 8am)."
                    ),
                },
            },
            "required": ["goal", "fire_at"],
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        goal: str = (kwargs.get("goal") or "").strip()
        fire_at_raw: str = (kwargs.get("fire_at") or "").strip()
        conversation_id: Optional[str] = kwargs.get("conversation_id")

        if not goal:
            return {"error": "missing_goal", "message": "I need to know what to schedule."}
        if not fire_at_raw:
            return {"error": "missing_time", "message": "When would you like me to do that?"}
        if not conversation_id:
            return {"error": "no_conversation", "message": "No conversation context available"}

        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return {"error": "no_context", "message": "No node context available"}
        household_id = node_context.get("household_id")
        node_id = node_context.get("node_id")
        if not household_id or not node_id:
            return {"error": "no_context", "message": "I couldn't tell which node to schedule this on."}
        user_id = node_context.get("speaker_user_id")
        # The node's IANA zone; fall back to the top-level cached timezone (older
        # cache entries store it there, not in node_context) before UTC. Resolving
        # a clock time in the wrong zone silently shifts the fire time by hours.
        tz_name = (
            node_context.get("timezone")
            or conversation_cache.get_timezone(conversation_id)
            or "UTC"
        )

        fire_dt = _resolve_fire_at(fire_at_raw, tz_name)
        recurrence_desc = (kwargs.get("recurrence") or "").strip()
        if fire_dt is None and recurrence_desc:
            # A RECURRING errand with no stated clock time ("every day, check the
            # weather" → fire_at="today") defaults to 9am local rather than failing.
            fire_dt = _default_recurring_fire(tz_name)
        if fire_dt is None:
            return {"error": "bad_time", "message": "I couldn't work out when to run that — tell me a day and a time, like 'tomorrow at 9am'."}

        # Repeat cadence (optional) — derived from the first fire's local time-of-day.
        recurrence = _build_recurrence(recurrence_desc, fire_dt, tz_name)

        now = datetime.utcnow()
        if fire_dt <= now:
            if recurrence is not None:
                # A recurring first-fire already in the past (e.g. "every day at 8am"
                # said at 2pm) rolls forward to the next occurrence instead of erroring.
                from app.services.schedule_service import compute_next_fire

                rolled = compute_next_fire(recurrence, now, tz_name)
                if rolled is None:
                    return {"error": "bad_time", "message": "I couldn't work out the schedule — try 'every day at 9am'."}
                fire_dt = rolled
            else:
                return {"error": "past_time", "message": "That time has already passed — give me a future time."}

        try:
            from app.services.schedule_service import create_schedule

            loop = asyncio.get_event_loop()
            task = loop.create_task(
                create_schedule(
                    household_id=household_id, node_id=node_id, user_id=user_id,
                    intent=goal, fire_at=fire_dt, timezone=tz_name, recurrence=recurrence,
                )
            )
            _SCHEDULE_TASKS.add(task)
            task.add_done_callback(_SCHEDULE_TASKS.discard)
            logger.info(
                "🗓️ Schedule started: household=%s node=%s fire_at=%s recurrence=%s goal=%r",
                household_id, node_id, fire_dt, recurrence, goal,
            )
        except RuntimeError as e:
            logger.error("Failed to start schedule task: %s", e)
            return {"error": "task_failed", "message": f"I couldn't schedule that: {e}"}

        when = _friendly_when(fire_at_raw, fire_dt, tz_name)
        if recurrence is not None:
            repeat = _friendly_recurrence(kwargs.get("recurrence"))
            msg = (f"Okay — I'll do that {repeat}, starting {when}. Each time, I'll send a "
                   "plan to your phone to approve first.")
        else:
            msg = (f"Okay — I'll get to that {when}. I'll send a plan to your phone to "
                   "approve when the time comes.")
        return {"status": "accepted", "message": msg}


def _friendly_when(fire_at_raw: str, fire_dt: datetime, tz_name: str) -> str:
    """A short spoken time for the ack, in the user's zone (best-effort)."""
    try:
        from zoneinfo import ZoneInfo
        local = fire_dt.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz_name or "UTC"))
        return local.strftime("%A at %-I:%M %p").replace(":00", "")
    except Exception:  # noqa: BLE001 — fall back to the raw phrase
        return "then"


def _friendly_recurrence(descriptor: str | None) -> str:
    """A short spoken cadence for the recurring ack (best-effort)."""
    d = (descriptor or "").strip().lower()
    known = {
        "daily": "every day", "weekdays": "on weekdays", "weekly": "every week",
        "monthly": "every month", "hourly": "every hour",
    }
    if d in known:
        return known[d]
    if d.startswith("every"):
        return d
    return "on a repeating schedule"
