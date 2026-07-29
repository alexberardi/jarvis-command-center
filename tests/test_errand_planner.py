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
    # Guardrail: the fallback menu should only list safe, real command names.
    names = {c["command"] for c in errand_planner.COMMAND_MENU}
    assert "get_weather" in names and "set_reminder" in names
    assert "make_phone_call" not in names  # phone step is a later, gated slice


# ── build_errand_menu: node's real commands → planner menu ───────────────────


def test_build_errand_menu_converts_available_commands():
    available = [
        {
            "command_name": "set_timer",
            "description": "Start a timer.",
            "parameters": [
                {"name": "duration", "description": "how long, e.g. '5 minutes'", "type": "string"},
                {"name": "label", "type": "string"},  # no description → falls back to type
            ],
        },
    ]
    menu = errand_planner.build_errand_menu(available)
    assert menu == [{
        "command": "set_timer",
        "description": "Start a timer.",
        "args": {"duration": "how long, e.g. '5 minutes'", "label": "string"},
    }]


def test_build_errand_menu_filters_deny_and_junk():
    available = [
        {"command_name": "get_weather", "description": "w", "parameters": []},
        {"command_name": "chat", "description": "conversation"},        # denied
        {"command_name": "act_on_items", "description": "follow-ups"},  # denied
        {"command_name": "email", "description": "send an email"},      # denied (no provenance)
        {"command_name": "", "description": "blank"},                    # junk
        {"description": "no name"},                                      # junk
    ]
    names = {c["command"] for c in errand_planner.build_errand_menu(available)}
    assert names == {"get_weather"}


def test_build_errand_menu_empty_on_none():
    assert errand_planner.build_errand_menu(None) == []
    assert errand_planner.build_errand_menu([]) == []


# ── refine_errand_plan: revise an existing plan from an instruction ──────────


def test_refine_errand_plan_revises_from_instruction():
    client = _client(_plan({
        "summary": "Call them instead",
        "steps": [{"command": "get_weather", "args": {}, "label": "W"}],
    }))
    plan = asyncio.run(errand_planner.refine_errand_plan(
        "Email the office", [{"command": "get_news", "args": {}, "label": "News"}],
        "call them instead of emailing", client))
    assert plan.summary == "Call them instead"
    # the refine prompt carries the CURRENT plan + the instruction (not a fresh goal)
    sent = client.chat_completion.call_args.kwargs["messages"][0]["content"]
    assert "Email the office" in sent and "call them instead of emailing" in sent


def test_refine_keeps_existing_steps_even_when_menu_degrades():
    # menu degraded to the 6-command default (no set_timer), but the current plan
    # HAS set_timer — a refine must not drop it just because the node fetch failed.
    client = _client(_plan({
        "summary": "Timer plus weather",
        "steps": [
            {"command": "set_timer", "args": {"duration_seconds": "600"}, "label": "Timer"},
            {"command": "get_weather", "args": {}, "label": "W"},
        ],
    }))
    plan = asyncio.run(errand_planner.refine_errand_plan(
        "Timer", [{"command": "set_timer", "args": {}, "label": "Timer"}],
        "also check the weather", client, menu=errand_planner.COMMAND_MENU))
    cmds = [s.command for s in plan.steps]
    assert "set_timer" in cmds and "get_weather" in cmds  # existing set_timer survives


def test_plan_and_refine_prompts_forbid_fabrication():
    guard = "Never invent contact details"
    assert guard in errand_planner._build_prompt("email my doctor", errand_planner.COMMAND_MENU)
    assert guard in errand_planner._build_refine_prompt(
        "Email them", [{"command": "get_news"}], "call instead", errand_planner.COMMAND_MENU)
