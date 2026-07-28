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
         patch("app.core.tools.run_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.errand_service.draft_errand_plan_detached") as draft:
        res = RunErrandTool().execute(goal="check the weather", conversation_id="c1")
    assert res["status"] == "accepted" and "plan" in res["message"].lower()
    draft.assert_called_once_with(
        household_id="hh-1", node_id="node-1", goal="check the weather", user_id=7
    )
    fake_loop.create_task.assert_called_once()  # detached, off the request path


def test_run_errand_household_broadcast_when_no_speaker():
    ctx = {"household_id": "hh-1", "node_id": "node-1"}  # no speaker_user_id
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=ctx), \
         patch("app.core.tools.run_errand_tool.asyncio.get_event_loop", return_value=fake_loop), \
         patch("app.services.errand_service.draft_errand_plan_detached") as draft:
        res = RunErrandTool().execute(goal="weather", conversation_id="c1")
    assert res["status"] == "accepted"
    assert draft.call_args.kwargs["user_id"] is None  # → household broadcast downstream
