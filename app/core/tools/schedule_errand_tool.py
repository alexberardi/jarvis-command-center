"""Schedule Errand Tool — schedule an errand to run at a future time (once).

Example: "Tomorrow at 9am, call the dentist to book a cleaning."

Voice entry for the scheduler: a spoken goal + a time → a ``Schedule`` row. At the
scheduled time the sweep RE-PLANS the goal against fresh context and posts a plan card
to approve (re-plan + re-confirm each run — nothing runs unattended). This is the
deferred single-run slice; recurrence ("every Monday") is a follow-up.

Mirrors run_errand_tool: derive household/node/speaker from node_context, kick the
schedule off, return a spoken ack. ``fire_at`` is a plain string (NOT format:date-time)
because the shared auto-resolver drops the clock time when a whole phrase arrives as one
string — so ``_resolve_fire_at`` parses the day AND the time here, in the NODE'S timezone
(``node_context["timezone"]``). Resolving in the wrong zone silently shifts the fire time.
"""

import asyncio
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

    # Date
    date = today
    if "tomorrow" in low:
        date = today + timedelta(days=1)
    else:
        for name, wd in _WEEKDAYS.items():
            if name in low:
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
        return None  # no time → can't schedule precisely

    local = datetime(date.year, date.month, date.day, hour, minute, tzinfo=tz)
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
            "9am', 'at 7:30pm', 'this evening', 'in an hour'. Put what to do in `goal` and "
            "the time in `fire_at`. At that time Jarvis re-plans the goal and sends a plan "
            "card to approve — nothing runs unattended. PREFER this over run_errand any "
            "time the user names a future time; use run_errand only for 'run an errand' to "
            "do NOW, and the normal command for a single immediate action."
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
                        "When to run it, as the natural day + clock time the user gave — "
                        "e.g. 'tonight at 7:51pm', 'tomorrow at 9am', 'friday at noon'. "
                        "Include BOTH the day and the time; copy the user's time exactly."
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
        if fire_dt is None:
            return {"error": "bad_time", "message": "I couldn't work out when to run that — tell me a day and a time, like 'tomorrow at 9am'."}
        if fire_dt <= datetime.utcnow():
            return {"error": "past_time", "message": "That time has already passed — give me a future time."}

        try:
            from app.services.schedule_service import create_schedule

            loop = asyncio.get_event_loop()
            task = loop.create_task(
                create_schedule(
                    household_id=household_id, node_id=node_id, user_id=user_id,
                    intent=goal, fire_at=fire_dt, timezone=tz_name,
                )
            )
            _SCHEDULE_TASKS.add(task)
            task.add_done_callback(_SCHEDULE_TASKS.discard)
            logger.info(
                "🗓️ Schedule started: household=%s node=%s fire_at=%s goal=%r",
                household_id, node_id, fire_dt, goal,
            )
        except RuntimeError as e:
            logger.error("Failed to start schedule task: %s", e)
            return {"error": "task_failed", "message": f"I couldn't schedule that: {e}"}

        when = _friendly_when(fire_at_raw, fire_dt, tz_name)
        return {
            "status": "accepted",
            "message": (
                f"Okay — I'll get to that {when}. I'll send a plan to your phone to "
                "approve when the time comes."
            ),
        }


def _friendly_when(fire_at_raw: str, fire_dt: datetime, tz_name: str) -> str:
    """A short spoken time for the ack, in the user's zone (best-effort)."""
    try:
        from zoneinfo import ZoneInfo
        local = fire_dt.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz_name or "UTC"))
        return local.strftime("%A at %-I:%M %p").replace(":00", "")
    except Exception:  # noqa: BLE001 — fall back to the raw phrase
        return "then"
