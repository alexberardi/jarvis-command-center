"""list_scheduled_errands voice tool — show + offer to cancel scheduled errands."""

from unittest.mock import MagicMock, patch

from app.core.conversation_cache import conversation_cache
from app.core.tool_registry import tool_registry
from app.core.tools.list_scheduled_errands_tool import ListScheduledErrandsTool

_CTX = {"household_id": "hh-1", "node_id": "node-1", "speaker_user_id": 7, "timezone": "UTC"}


def test_list_scheduled_errands_is_auto_registered():
    assert tool_registry.get_tool("list_scheduled_errands") is not None


def test_missing_context_is_reported():
    t = ListScheduledErrandsTool()
    assert t.execute()["error"] == "no_conversation"
    with patch.object(conversation_cache, "get_node_context", return_value=None):
        assert t.execute(conversation_id="c1")["error"] == "no_context"


def test_empty_speaks_nothing_and_posts_no_card():
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX), \
         patch("app.services.schedule_service.list_schedules", return_value=[]), \
         patch("app.services.schedule_service.post_schedule_list_card") as post:
        res = ListScheduledErrandsTool().execute(conversation_id="c1")
    assert res["status"] == "ok" and "don't have any" in res["message"]
    post.assert_not_called()


def test_lists_count_and_posts_the_management_card():
    scheds = [{"id": "s1", "intent": "weather"}, {"id": "s2", "intent": "dentist"}]
    # Force the no-running-loop fallback so the (mocked) card post happens inline and
    # is directly assertable (no dangling coroutine).
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX), \
         patch("app.services.schedule_service.list_schedules", return_value=scheds), \
         patch("app.core.tools.list_scheduled_errands_tool.asyncio.get_event_loop",
               side_effect=RuntimeError), \
         patch("app.services.schedule_service.post_schedule_list_card") as post:
        res = ListScheduledErrandsTool().execute(conversation_id="c1")
    assert res["status"] == "ok" and "2 errands" in res["message"]
    post.assert_called_once_with("hh-1", 7)


def test_single_errand_uses_singular_phrasing():
    scheds = [{"id": "s1", "intent": "weather"}]
    fake_loop = MagicMock()
    with patch.object(conversation_cache, "get_node_context", return_value=_CTX), \
         patch("app.services.schedule_service.list_schedules", return_value=scheds), \
         patch("app.core.tools.list_scheduled_errands_tool.asyncio.get_event_loop",
               return_value=fake_loop), \
         patch("app.services.schedule_service.post_schedule_list_card"):
        res = ListScheduledErrandsTool().execute(conversation_id="c1")
    assert "one errand scheduled" in res["message"]
    fake_loop.create_task.assert_called_once()  # posted off the request path
