"""make_phone_call tool: gate refusal, identity requirement, ack + spawn."""

from unittest.mock import patch

import pytest

from app.core.tools.make_phone_call_tool import MakePhoneCallTool

CONV = "conv-123"
CTX = {"household_id": "hh-1", "speaker_user_id": 7}


@pytest.fixture
def tool():
    return MakePhoneCallTool()


def _with_ctx(ctx):
    return patch(
        "app.core.tools.make_phone_call_tool.conversation_cache.get_node_context",
        return_value=ctx,
    )


class TestGate:
    def test_disabled_gate_speaks_honest_refusal(self, tool):
        """The 2026-07-19 finding: a call-shaped utterance must produce a
        refusal, never improvisation. The tool stays in the prompt; this
        message is what the model speaks."""
        with _with_ctx(CTX), patch(
            "app.services.phone_call_service.phone_calls_enabled", return_value=False
        ):
            result = tool.execute(
                business="Tony's Pizzeria", goal="book a table", conversation_id=CONV
            )
        assert result["error"] == "phone_calls_disabled"
        assert "aren't set up" in result["message"]

    def test_enabled_gate_accepts_and_spawns(self, tool):
        with _with_ctx(CTX), patch(
            "app.services.phone_call_service.phone_calls_enabled", return_value=True
        ), patch(
            "app.services.phone_call_service.create_call_plan"
        ) as plan, patch(
            "app.core.tools.make_phone_call_tool.asyncio.get_event_loop"
        ) as loop:
            result = tool.execute(
                business="Tony's Pizzeria",
                goal="book a table for 4 Friday 7pm",
                conversation_id=CONV,
            )
        assert result["status"] == "accepted"
        assert "phone" in result["message"].lower()
        loop.return_value.create_task.assert_called_once()
        plan.assert_called_once_with(
            business="Tony's Pizzeria",
            goal="book a table for 4 Friday 7pm",
            household_id="hh-1",
            user_id=7,
        )


class TestIdentityAndParams:
    def test_no_identified_speaker_fails_closed(self, tool):
        with _with_ctx({"household_id": "hh-1", "speaker_user_id": None}), patch(
            "app.services.phone_call_service.phone_calls_enabled", return_value=True
        ):
            result = tool.execute(
                business="Tony's", goal="book", conversation_id=CONV
            )
        assert result["error"] == "no_identified_speaker"

    def test_missing_params_rejected(self, tool):
        assert tool.execute(business="", goal="x", conversation_id=CONV)["error"] == "missing_params"
        assert tool.execute(business="x", goal="", conversation_id=CONV)["error"] == "missing_params"

    def test_no_conversation_context(self, tool):
        assert tool.execute(business="x", goal="y")["error"] == "no_conversation"
        with _with_ctx(None):
            assert (
                tool.execute(business="x", goal="y", conversation_id=CONV)["error"]
                == "no_context"
            )


class TestRegistration:
    def test_tool_discovered_by_registry(self):
        from app.core.tool_registry import ToolRegistry

        registry = ToolRegistry()
        assert registry.has_tool("make_phone_call")

    def test_always_in_text_path_whitelist(self):
        """The whitelist addition is load-bearing (hallucination fix) — pin
        that the source keeps make_phone_call unconditional."""
        import inspect
        from app.core import conversation_handler

        src = inspect.getsource(conversation_handler)
        assert '_safe_tool_names.append("make_phone_call")' in src
