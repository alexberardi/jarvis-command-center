"""schedule_errand voice tool — schedule an errand for a future time (Slice 1)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from app.core.conversation_cache import conversation_cache
from app.core.tool_registry import tool_registry
from app.core.tools.schedule_errand_tool import ScheduleErrandTool, _resolve_fire_at

_CTX = {"household_id": "hh-1", "node_id": "node-1", "speaker_user_id": 7, "timezone": "UTC"}


def test_schedule_errand_is_auto_registered():
    assert tool_registry.get_tool("schedule_errand") is not None


def test_schema_requires_goal_and_a_datetime():
    of = ScheduleErrandTool().to_openai_format()
    params = of["function"]["parameters"]
    assert of["function"]["name"] == "schedule_errand"
    assert params["required"] == ["goal", "fire_at"]
    # fire_at is a plain string (NOT format:date-time) so the shared auto-resolver leaves
    # it alone — the tool resolves the day + clock time itself (the auto-resolver dropped
    # the time to midnight).
    assert "format" not in params["properties"]["fire_at"]


def test_missing_goal_or_time_or_context():
    t = ScheduleErrandTool()
    assert t.execute(goal=" ", fire_at="2099-01-01T09:00:00Z", conversation_id="c1")["error"] == "missing_goal"
    assert t.execute(goal="x", fire_at=" ", conversation_id="c1")["error"] == "missing_time"
    assert t.execute(goal="x", fire_at="2099-01-01T09:00:00Z")["error"] == "no_conversation"
    with patch.object(conversation_cache, "get_node_context", return_value=None):
        assert t.execute(goal="x", fire_at="2099-01-01T09:00:00Z", conversation_id="c1")["error"] == "no_context"


def test_past_time_is_rejected():
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX):
        res = ScheduleErrandTool().execute(goal="x", fire_at="2000-01-01T09:00:00Z", conversation_id="c1")
    assert res["error"] == "past_time"


def test_bad_time_is_rejected():
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX):
        res = ScheduleErrandTool().execute(goal="x", fire_at="not a date", conversation_id="c1")
    assert res["error"] == "bad_time"


def test_happy_path_creates_a_schedule_task():
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX), \
         patch("app.core.tools.schedule_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.schedule_service.create_schedule") as create:
        res = ScheduleErrandTool().execute(
            goal="call the dentist", fire_at="2099-06-01T09:00:00Z", conversation_id="c1")
    assert res["status"] == "accepted"
    create.assert_called_once()
    kw = create.call_args.kwargs
    assert kw["household_id"] == "hh-1" and kw["intent"] == "call the dentist"
    assert isinstance(kw["fire_at"], datetime) and kw["fire_at"].year == 2099
    fake_loop.create_task.assert_called_once()  # detached, off the request path


def _capture_fire_at(node_ctx, *, get_tz="UTC"):
    """Run execute() with a stubbed cache + create_schedule; return the fire_at passed."""
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=node_ctx), \
         patch.object(conversation_cache, "get_timezone", return_value=get_tz), \
         patch("app.core.tools.schedule_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.schedule_service.create_schedule") as create:
        res = ScheduleErrandTool().execute(
            goal="check the weather", fire_at="2099-01-01T09:00:00", conversation_id="c1")
    assert res["status"] == "accepted"
    return create.call_args.kwargs["fire_at"], create.call_args.kwargs["timezone"]


def test_execute_resolves_in_the_node_timezone_not_utc():
    # The regression: a 9am local time must resolve in the NODE'S zone, not UTC.
    # 2099-01-01 09:00 America/New_York (EST, +5) → 14:00 UTC. If it wrongly used
    # UTC it would stay 09:00 — the exact "resolved in UTC" bug that shipped a
    # schedule ~5h/1-day off.
    ctx = {**_CTX, "timezone": "America/New_York"}
    fire_at, tz = _capture_fire_at(ctx)
    assert tz == "America/New_York"
    assert fire_at == datetime(2099, 1, 1, 14, 0)


def test_execute_falls_back_to_top_level_cached_timezone():
    # Older cache entries store the zone at the top level, not in node_context.
    ctx = {k: v for k, v in _CTX.items() if k != "timezone"}  # no timezone key
    fire_at, tz = _capture_fire_at(ctx, get_tz="America/New_York")
    assert tz == "America/New_York"
    assert fire_at == datetime(2099, 1, 1, 14, 0)


def test_resolve_fire_at_parses_day_and_clock_time():
    today = datetime.now(timezone.utc).date()
    tmr = today + timedelta(days=1)
    # a natural phrase: the DAY and the CLOCK time are both kept (was the bug — the
    # time dropped to midnight)
    assert _resolve_fire_at("tonight at 7:51pm", "UTC") == datetime(today.year, today.month, today.day, 19, 51)
    assert _resolve_fire_at("tomorrow at 9am", "UTC") == datetime(tmr.year, tmr.month, tmr.day, 9, 0)
    # separator-less / dotted times still work (via the fixed parse_time_string)
    assert _resolve_fire_at("tomorrow at 730pm", "UTC") == datetime(tmr.year, tmr.month, tmr.day, 19, 30)
    # a coarse named time when there's no clock
    assert _resolve_fire_at("tomorrow at noon", "UTC") == datetime(tmr.year, tmr.month, tmr.day, 12, 0)
    # an explicit ISO instant is used directly
    assert _resolve_fire_at("2099-06-01T09:00:00Z", "UTC") == datetime(2099, 6, 1, 9, 0)
    # a naive local time converts to UTC (EST winter = +5h)
    assert _resolve_fire_at("2099-01-01T09:00:00", "America/New_York") == datetime(2099, 1, 1, 14, 0)
    # no time at all → can't schedule precisely
    assert _resolve_fire_at("sometime soon", "UTC") is None
    assert _resolve_fire_at("", "UTC") is None


# ── Slice 2: recurrence entry ─────────────────────────────────────────────────


def test_build_recurrence_cadences_utc():
    import json
    from app.core.tools.schedule_errand_tool import _build_recurrence

    fire = datetime(2099, 1, 5, 9, 30)  # naive UTC, 09:30

    def spec(desc):
        r = _build_recurrence(desc, fire, "UTC")
        return json.loads(r) if r else None

    assert spec("daily") == {"type": "cron", "cron": "30 9 * * *"}
    assert spec("weekdays") == {"type": "cron", "cron": "30 9 * * 1-5"}
    assert spec("monthly") == {"type": "cron", "cron": "30 9 5 * *"}  # day-of-month = 5
    assert spec("hourly") == {"type": "interval", "interval_seconds": 3600}
    assert spec("every 30 minutes") == {"type": "interval", "interval_seconds": 1800}
    assert spec("every 2 hours") == {"type": "interval", "interval_seconds": 7200}
    assert spec("weekly")["cron"].startswith("30 9 * * ")  # weekly cron at 9:30, some DOW
    # one-shot / unrecognized → None (treated as a one-time schedule)
    assert spec(None) is None and spec("once") is None and spec("banana") is None


def test_build_recurrence_uses_node_timezone():
    import json
    from app.core.tools.schedule_errand_tool import _build_recurrence

    # 09:30 UTC == 04:30 America/New_York (EST) → the daily cron fixes 4:30 LOCAL
    fire = datetime(2099, 1, 5, 9, 30)
    assert json.loads(_build_recurrence("daily", fire, "America/New_York"))["cron"] == "30 4 * * *"


def test_execute_recurring_passes_recurrence_and_first_fire():
    import json

    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX), \
         patch("app.core.tools.schedule_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.schedule_service.create_schedule") as create:
        res = ScheduleErrandTool().execute(
            goal="check the weather", fire_at="2099-06-01T08:00:00",
            recurrence="daily", conversation_id="c1")
    assert res["status"] == "accepted"
    kw = create.call_args.kwargs
    assert json.loads(kw["recurrence"]) == {"type": "cron", "cron": "0 8 * * *"}
    assert kw["fire_at"] == datetime(2099, 6, 1, 8, 0)  # first occurrence


def test_recurring_past_first_fire_rolls_forward_not_rejected():
    # "every day at 8am" when today's 8am has passed → the FIRST fire rolls to the next
    # 8am (future), instead of the one-shot "past_time" rejection.
    import json

    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX), \
         patch("app.core.tools.schedule_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.schedule_service.create_schedule") as create:
        res = ScheduleErrandTool().execute(
            goal="check the weather", fire_at="2000-01-01T08:00:00",
            recurrence="daily", conversation_id="c1")
    assert res["status"] == "accepted"  # NOT past_time
    kw = create.call_args.kwargs
    assert kw["fire_at"] > datetime.utcnow()  # rolled forward
    assert json.loads(kw["recurrence"])["cron"] == "0 8 * * *"


def test_recurring_without_a_time_defaults_to_9am():
    # "every day, check the weather" → fire_at="today" (no clock time). A recurring
    # errand defaults to 9am local instead of the bad_time rejection.
    import json

    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX), \
         patch("app.core.tools.schedule_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.schedule_service.create_schedule") as create:
        res = ScheduleErrandTool().execute(
            goal="check the weather", fire_at="today", recurrence="daily", conversation_id="c1")
    assert res["status"] == "accepted"  # NOT bad_time — this was the live gap
    kw = create.call_args.kwargs
    assert json.loads(kw["recurrence"]) == {"type": "cron", "cron": "0 9 * * *"}
    assert kw["fire_at"] > datetime.utcnow()


def test_one_shot_without_a_time_still_rejected():
    # No recurrence + no clock time → still can't schedule precisely (the default only
    # applies to recurring errands).
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX):
        res = ScheduleErrandTool().execute(goal="x", fire_at="today", conversation_id="c1")
    assert res["error"] == "bad_time"
