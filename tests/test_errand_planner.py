"""Errand planner POC (prds/errand-runner.md §2-§3): goal -> validated steps."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

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
        "is_risky": False,  # stamped from the menu (set_reminder isn't risky)
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
        "is_risky": False,  # not declared → low-stakes default
    }]


def test_build_errand_menu_carries_is_risky_flag():
    # is_risky rides through from the node CommandDefinition (SDK schema) so the
    # errand envelope check can read it off a persisted step. Absent → False.
    available = [
        {"command_name": "send_text", "description": "text someone", "parameters": [], "is_risky": True},
        {"command_name": "get_weather", "description": "weather", "parameters": []},
    ]
    menu = {m["command"]: m["is_risky"] for m in errand_planner.build_errand_menu(available)}
    assert menu == {"send_text": True, "get_weather": False}


def test_build_errand_menu_filters_deny_and_junk():
    available = [
        {"command_name": "get_weather", "description": "w", "parameters": []},
        {"command_name": "chat", "description": "conversation"},        # denied (headless-unfit)
        {"command_name": "act_on_items", "description": "follow-ups"},  # denied
        {"command_name": "email", "description": "send an email"},      # ALLOWED now — real execution + plan-card review
        {"command_name": "run_errand", "description": "recursion"},     # denied (recursion)
        {"command_name": "", "description": "blank"},                    # junk
        {"description": "no name"},                                      # junk
    ]
    names = {c["command"] for c in errand_planner.build_errand_menu(available)}
    assert names == {"get_weather", "email"}  # email is no longer structurally denied


def test_build_errand_menu_empty_on_none():
    assert errand_planner.build_errand_menu(None) == []
    assert errand_planner.build_errand_menu([]) == []


# ── server-tool menu union: node commands ∪ enabled server tools ─────────────


def _bool_setting(mapping: dict):
    """A fake ``_errand_bool_setting`` driven by a {key: value} map (default arg)."""
    def _impl(key, household_id, default):
        return mapping.get(key, default)
    return _impl


def test_enabled_server_tools_gates_like_the_voice_path():
    # phone on, web_search on, memory on, speaker present → the full offered set.
    with patch.object(errand_planner, "_errand_bool_setting",
                      new=_bool_setting({"web_search.enabled": True, "memory.enabled": True,
                                         "memory.recall_enabled": True})), \
         patch("app.services.phone_call_service.phone_calls_enabled", return_value=True):
        names = errand_planner.enabled_errand_server_tools("hh-1", speaker_user_id=7)
    assert set(names) == {"make_phone_call", "deep_research", "quick_search",
                          "remember", "forget", "recall"}
    assert "run_errand" not in names  # recursion — never offered as an errand step


def test_enabled_server_tools_respect_gates_and_speaker():
    # phone OFF, web_search OFF, memory ON but NO speaker → nothing (memory needs a speaker).
    with patch.object(errand_planner, "_errand_bool_setting",
                      new=_bool_setting({"web_search.enabled": False, "memory.enabled": True})), \
         patch("app.services.phone_call_service.phone_calls_enabled", return_value=False):
        assert errand_planner.enabled_errand_server_tools("hh-1", speaker_user_id=None) == []


def test_build_server_tool_menu_normalizes_and_excludes_run_errand():
    # make_phone_call is a real registered server tool → normalized to {command, description, args}.
    with patch.object(errand_planner, "enabled_errand_server_tools",
                      return_value=["make_phone_call"]):
        menu = errand_planner.build_server_tool_menu("hh-1", 7)
    assert len(menu) == 1
    entry = menu[0]
    assert entry["command"] == "make_phone_call"
    assert entry["description"] and set(entry["args"]) == {"business", "goal"}  # from the JSON schema


def test_build_full_errand_menu_unions_and_dedups_server_precedence():
    node_cmds = [
        {"command_name": "get_weather", "description": "w", "parameters": []},
        {"command_name": "make_phone_call", "description": "a NODE cmd shadowed by the server tool",
         "parameters": []},
    ]
    with patch.object(errand_planner, "build_server_tool_menu",
                      return_value=[{"command": "make_phone_call", "description": "SERVER", "args": {}}]):
        menu = errand_planner.build_full_errand_menu(node_cmds, "hh-1", 7)
    by_cmd = {c["command"]: c for c in menu}
    assert set(by_cmd) == {"get_weather", "make_phone_call"}
    # collision resolves to the SERVER entry (matches has_tool dispatch at run time)
    assert by_cmd["make_phone_call"]["description"] == "SERVER"


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


def test_plan_and_refine_prompts_include_guardrail_and_decomposition():
    plan_prompt = errand_planner._build_prompt("email my doctor", errand_planner.COMMAND_MENU)
    refine_prompt = errand_planner._build_refine_prompt(
        "Email them", [{"command": "get_news"}], "call instead", errand_planner.COMMAND_MENU)
    for prompt in (plan_prompt, refine_prompt):
        # still forbids inventing specific contact data...
        assert errand_planner._FABRICATION_GUARDRAIL in prompt
        assert "invent specific data" in prompt.lower()
        # ...but now every action becomes its own step (fixes dropped-clause plans)
        assert errand_planner._DECOMPOSITION_RULE in prompt
        assert "separate step for each" in prompt.lower()


# ── Chunk 5: pause-and-replan — planner emits a checkpoint + cursor-aware re-plan ──


def test_planner_may_emit_a_replan_checkpoint():
    # An outcome-dependent goal → the planner ends the plan at a request_replan
    # checkpoint instead of guessing the dependent steps. The control command isn't
    # in the menu but survives validation (CONTROL_COMMANDS).
    client = _client(_plan({
        "summary": "Check the weather, then decide",
        "steps": [
            {"command": "get_weather", "args": {"resolved_datetimes": ["tomorrow"]},
             "label": "Check weather"},
            {"command": "request_replan", "args": {"reason": "if rain, remind umbrella"},
             "label": "Decide follow-up"},
        ],
    }))
    plan = asyncio.run(plan_errand(
        "check tomorrow's weather and if it'll rain remind me to bring an umbrella", client))
    assert [s.command for s in plan.steps] == ["get_weather", "request_replan"]
    assert plan.steps[1].args["reason"]
    assert plan.steps[1].is_risky is False  # a control step is never risky


def test_checkpoint_rule_is_in_the_prompt():
    prompt = errand_planner._build_prompt("g", errand_planner.COMMAND_MENU)
    assert errand_planner._REPLAN_CHECKPOINT_RULE in prompt
    assert "request_replan" in prompt


def test_is_risky_is_stamped_onto_steps_from_the_menu():
    menu = [
        {"command": "get_weather", "description": "w", "args": {}, "is_risky": False},
        {"command": "make_phone_call", "description": "call",
         "args": {"business": "who"}, "is_risky": True},
    ]
    client = _client(_plan({
        "summary": "call the shop",
        "steps": [{"command": "make_phone_call", "args": {"business": "Tony's"},
                   "label": "Call Tony's"}],
    }))
    plan = asyncio.run(plan_errand("call Tony's", client, menu=menu))
    assert plan.steps[0].is_risky is True  # the risky menu flag rode onto the step


def test_replan_from_progress_returns_a_cursor_aware_continuation():
    from app.services.errand_planner import replan_from_progress

    menu = [{"command": "set_reminder", "description": "r",
             "args": {"text": "t", "when": "w"}, "is_risky": False}]
    client = _client(_plan({
        "summary": "remind umbrella",
        "steps": [{"command": "set_reminder",
                   "args": {"text": "bring umbrella", "when": "tomorrow 7am"},
                   "label": "Remind umbrella"}],
    }))
    done = [{"command": "get_weather", "args": {}, "label": "Check weather"}]
    results = [{"label": "Check weather", "success": True, "message": "Rain tomorrow, 90%"}]
    plan = asyncio.run(replan_from_progress(
        "check weather then remind if rain", done, results, "if rain remind umbrella",
        client, menu=menu))
    assert [s.command for s in plan.steps] == ["set_reminder"]
    # the earlier result is fed to the planner so it can adapt
    sent = client.chat_completion.call_args.kwargs["messages"][0]["content"]
    assert "Rain tomorrow, 90%" in sent and "Check weather" in sent


def test_summarize_progress_surfaces_step_data_not_just_done():
    # A node command returns its real result in `data` (message is null), so the
    # cursor-aware replan MUST see `data` — else it's told only "done" and can't
    # branch on the outcome (the live umbrella bug: no forecast → no reminder step).
    from app.services.errand_planner import _summarize_progress

    done = [{"command": "get_weather", "label": "Check weather"}]
    results = [{"label": "Check weather", "success": True, "message": None,
                "data": {"forecast_summary": "Friday: 100% chance of rain"}}]
    out = _summarize_progress(done, results)
    assert "100% chance of rain" in out
    assert out.strip() != "- Check weather: done"


def test_replan_from_progress_allows_an_empty_continuation():
    from app.services.errand_planner import replan_from_progress

    # "the results mean nothing more is needed" is valid — plan_errand would raise, but
    # a continuation may legitimately be empty (just proceed past the checkpoint).
    client = _client(_plan({"summary": "nothing more", "steps": []}))
    plan = asyncio.run(replan_from_progress(
        "g", [], [], "reason", client,
        menu=[{"command": "x", "description": "", "args": {}}]))
    assert plan.steps == []
