"""run_errand voice tool — Errand Runner POC chunk 6 (voice-inline launcher)."""

from unittest.mock import MagicMock, patch

from app.core.conversation_cache import conversation_cache
from app.core.tool_registry import tool_registry
from app.core.tools.run_errand_tool import RunErrandTool


def test_run_errand_is_auto_registered():
    assert tool_registry.get_tool("run_errand") is not None


def test_run_errand_openai_schema():
    of = RunErrandTool().to_openai_format()
    assert of["function"]["name"] == "run_errand"
    assert of["function"]["parameters"]["required"] == ["goal"]


def test_run_errand_missing_goal():
    assert RunErrandTool().execute(goal="  ", conversation_id="c1")["error"] == "missing_goal"


def test_run_errand_no_conversation_id():
    assert RunErrandTool().execute(goal="x")["error"] == "no_conversation"


def test_run_errand_no_node_context():
    with patch.object(conversation_cache, "get_node_context", return_value=None):
        assert RunErrandTool().execute(goal="x", conversation_id="c1")["error"] == "no_context"


def test_run_errand_missing_node_id():
    # household but no node_id in context → can't target a node
    with patch.object(conversation_cache, "get_node_context", return_value={"household_id": "hh-1"}):
        assert RunErrandTool().execute(goal="x", conversation_id="c1")["error"] == "no_context"


def test_run_errand_happy_path_fires_detached_draft():
    ctx = {"household_id": "hh-1", "node_id": "node-1", "speaker_user_id": 7}
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=ctx), \
         patch.object(conversation_cache, "get_available_commands", return_value=None), \
         patch("app.services.errand_planner.build_server_tool_menu", return_value=[]), \
         patch("app.core.tools.run_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.errand_service.draft_errand_plan_detached") as draft:
        res = RunErrandTool().execute(goal="check the weather", conversation_id="c1")
    assert res["status"] == "accepted" and "plan" in res["message"].lower()
    draft.assert_called_once_with(
        household_id="hh-1", node_id="node-1", goal="check the weather", user_id=7, menu=None
    )
    fake_loop.create_task.assert_called_once()  # detached, off the request path


def test_run_errand_household_broadcast_when_no_speaker():
    ctx = {"household_id": "hh-1", "node_id": "node-1"}  # no speaker_user_id
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=ctx), \
         patch.object(conversation_cache, "get_available_commands", return_value=None), \
         patch("app.services.errand_planner.build_server_tool_menu", return_value=[]), \
         patch("app.core.tools.run_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.errand_service.draft_errand_plan_detached") as draft:
        res = RunErrandTool().execute(goal="weather", conversation_id="c1")
    assert res["status"] == "accepted"
    assert draft.call_args.kwargs["user_id"] is None  # → household broadcast downstream


def test_run_errand_builds_menu_from_conversation_commands():
    """The voice path plans over the node's real installed commands (from the live
    conversation cache), with denied commands filtered out. (Server tools mocked
    to [] here to isolate the node-command half — see the union test below.)"""
    ctx = {"household_id": "hh-1", "node_id": "node-1", "speaker_user_id": 7}
    available = [
        {"command_name": "get_weather", "description": "w", "parameters": []},
        {"command_name": "set_timer", "description": "timer", "parameters": []},
        {"command_name": "chat", "description": "conversation"},  # denied by build_errand_menu
    ]
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=ctx), \
         patch.object(conversation_cache, "get_available_commands", return_value=available), \
         patch("app.services.errand_planner.build_server_tool_menu", return_value=[]), \
         patch("app.core.tools.run_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.errand_service.draft_errand_plan_detached") as draft:
        RunErrandTool().execute(goal="set a 5 minute timer", conversation_id="c1")
    menu = draft.call_args.kwargs["menu"]
    assert {c["command"] for c in menu} == {"get_weather", "set_timer"}  # chat filtered out


def test_run_errand_unions_server_tools_into_menu():
    """The voice-path menu includes ENABLED server tools (phone calls, research…)
    alongside the node's installed commands — the whole point of the rearchitecture."""
    ctx = {"household_id": "hh-1", "node_id": "node-1", "speaker_user_id": 7}
    available = [{"command_name": "get_weather", "description": "w", "parameters": []}]
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=ctx), \
         patch.object(conversation_cache, "get_available_commands", return_value=available), \
         patch("app.services.errand_planner.build_server_tool_menu",
               return_value=[{"command": "make_phone_call", "description": "call a business",
                              "args": {"business": "", "goal": ""}}]), \
         patch("app.core.tools.run_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.errand_service.draft_errand_plan_detached") as draft:
        RunErrandTool().execute(goal="call the pizzeria and check the weather", conversation_id="c1")
    cmds = {c["command"] for c in draft.call_args.kwargs["menu"]}
    assert "get_weather" in cmds and "make_phone_call" in cmds  # node cmd ∪ server tool


def test_run_errand_menu_none_when_no_commands_or_server_tools():
    ctx = {"household_id": "hh-1", "node_id": "node-1", "speaker_user_id": 7}
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=ctx), \
         patch.object(conversation_cache, "get_available_commands", return_value=None), \
         patch("app.services.errand_planner.build_server_tool_menu", return_value=[]), \
         patch("app.core.tools.run_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.errand_service.draft_errand_plan_detached") as draft:
        RunErrandTool().execute(goal="weather", conversation_id="c1")
    assert draft.call_args.kwargs["menu"] is None  # empty → None → downstream fetch/fallback
