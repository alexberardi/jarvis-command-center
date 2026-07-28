"""Errand planner POC (prds/errand-runner.md §2-§3): goal -> validated steps."""

import asyncio
import json
from unittest.mock import AsyncMock

from app.services import errand_planner
from app.services.errand_planner import plan_errand


def _resp(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _client(content: str) -> AsyncMock:
    c = AsyncMock()
    c.chat_completion = AsyncMock(return_value=_resp(content))
    return c


def _plan(payload) -> str:
    return json.dumps(payload)


def test_plans_valid_steps_and_summary():
    client = _client(_plan({
        "summary": "Reminder to check the gym charge",
        "steps": [{"command": "set_reminder",
                   "args": {"text": "check gym charge", "when": "next month"},
                   "label": "Set reminder"}],
    }))
    plan = asyncio.run(plan_errand("cancel my gym and remind me next month", client))
    assert plan.summary == "Reminder to check the gym charge"
    assert plan.steps[0].command == "set_reminder"
    assert plan.routine_steps()[0] == {
        "command": "set_reminder",
        "args": {"text": "check gym charge", "when": "next month"},
        "label": "Set reminder",
    }


def test_drops_hallucinated_commands():
    client = _client(_plan({
        "summary": "x",
        "steps": [
            {"command": "hack_the_mainframe", "args": {}, "label": "bad"},
            {"command": "get_weather", "args": {"resolved_datetimes": ["today"]}, "label": "weather"},
        ],
    }))
    plan = asyncio.run(plan_errand("x", client))
    assert [s.command for s in plan.steps] == ["get_weather"]


def test_strips_markdown_fences_and_prose():
    client = _client("Here's your plan:\n```json\n" + _plan({
        "summary": "News",
        "steps": [{"command": "get_news", "args": {"category": "general"}, "label": "Headlines"}],
    }) + "\n```")
    plan = asyncio.run(plan_errand("read me the news", client))
    assert plan.steps[0].command == "get_news"


def test_empty_response_raises():
    try:
        asyncio.run(plan_errand("x", _client("")))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_json_raises():
    try:
        asyncio.run(plan_errand("x", _client("not json at all")))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_no_menu_command_raises():
    client = _client(_plan({"summary": "s", "steps": [{"command": "nope", "args": {}, "label": "x"}]}))
    try:
        asyncio.run(plan_errand("x", client))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_menu_is_all_known_commands():
    # Guardrail: the POC menu should only list safe, real command names.
    names = {c["command"] for c in errand_planner.COMMAND_MENU}
    assert "get_weather" in names and "set_reminder" in names
    assert "make_phone_call" not in names  # phone step is a later, gated slice
