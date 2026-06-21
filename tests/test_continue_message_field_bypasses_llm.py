"""Regression pin: a command/tool result carrying a pre-formatted ``message``
field must BYPASS the formatting LLM and be spoken verbatim.

Documented production bug: when a command returned ``{"message": "..."}`` (a
human-facing string the command already wrote), command-center still ran it
through the formatting LLM, which would re-answer / "answer the literal
question" instead of just speaking the command's message.

The fix lives in the non-streaming text-mode formatter,
``ConversationHandler._format_tool_result_text_mode`` (conversation_handler.py
~line 1756): when EVERY tool result carries a non-empty ``message`` (or
``response``) field, it returns that text directly and never calls
``self.llm_client.chat_completion``.

These tests target that exact function with the LLM boundary mocked, so they
fail loudly if the fast-path is reverted (every result would fall through to
``chat_completion`` and the verbatim message would be lost to rephrasing).
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List


def _handler():
    """Text-based handler (prompt_provider=None) so the continue path routes
    through ``_format_tool_result_text_mode`` rather than native tool calling."""
    from app.core.conversation_handler import ConversationHandler

    handler = ConversationHandler(model=MagicMock(), llm_client=AsyncMock())
    handler.prompt_provider = None
    return handler


class TestMessageFieldBypassesLLM:
    """Lock in: a result with a ``message`` field short-circuits the LLM."""

    @pytest.mark.asyncio
    async def test_message_field_returned_verbatim_without_llm_call(self):
        """A single tool result carrying ``message`` is returned verbatim and
        the formatting LLM (``chat_completion``) is never invoked."""
        handler = _handler()
        # If the LLM were reached it would return this rephrased text. The
        # fast-path must make this unreachable.
        handler.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "REPHRASED BY THE LLM"}}],
        })

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "add eggplant to my list"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        ]
        tool_results = [
            {"tool_call_id": "c1", "output": {"success": True, "message": "Added eggplant to your shopping list."}}
        ]

        result = await handler._format_tool_result_text_mode(
            "conv-1", messages, tool_results
        )

        # The LLM boundary was NOT crossed.
        handler.llm_client.chat_completion.assert_not_called()

        # The command's own message is spoken verbatim — no rephrasing.
        assert result["stop_reason"] == "complete"
        assert result["assistant_message"] == "Added eggplant to your shopping list."

        # And it is committed to history verbatim (role='tool' message stripped).
        assert messages[-1] == {
            "role": "assistant",
            "content": "Added eggplant to your shopping list.",
        }
        assert not any(m.get("role") == "tool" for m in messages)

    @pytest.mark.asyncio
    async def test_response_field_also_bypasses_llm(self):
        """The fast-path also recognizes a ``response`` field (alias for
        ``message``) and skips the LLM."""
        handler = _handler()
        handler.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "REPHRASED BY THE LLM"}}],
        })

        messages = [
            {"role": "user", "content": "set a timer"},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        ]
        tool_results = [
            {"tool_call_id": "c1", "output": {"response": "Timer set for five minutes."}}
        ]

        result = await handler._format_tool_result_text_mode(
            "conv-2", messages, tool_results
        )

        handler.llm_client.chat_completion.assert_not_called()
        assert result["assistant_message"] == "Timer set for five minutes."

    @pytest.mark.asyncio
    async def test_json_string_message_field_bypasses_llm(self):
        """A tool ``output`` delivered as a JSON *string* (not a dict) that
        decodes to ``{"message": ...}`` must still be parsed and fast-pathed —
        nodes serialize outputs to strings on the wire."""
        handler = _handler()
        handler.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "REPHRASED BY THE LLM"}}],
        })

        messages = [
            {"role": "user", "content": "turn off the kitchen lights"},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        ]
        tool_results = [
            {"tool_call_id": "c1", "output": '{"success": true, "message": "Kitchen lights are off."}'}
        ]

        result = await handler._format_tool_result_text_mode(
            "conv-3", messages, tool_results
        )

        handler.llm_client.chat_completion.assert_not_called()
        assert result["assistant_message"] == "Kitchen lights are off."

    @pytest.mark.asyncio
    async def test_no_message_field_falls_through_to_llm(self):
        """Counter-case (proves the test guards the conditional, not an
        incidental path): a result WITHOUT a message/response field must reach
        the formatting LLM. If this also bypassed the LLM the fast-path would
        be over-broad — this asserts the guard is scoped to message-bearing
        results only."""
        handler = _handler()
        handler.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "It's sunny and 72F."}}],
        })

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "what's the weather"},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        ]
        # Raw data, no pre-formatted message → needs LLM rephrasing.
        tool_results = [
            {"tool_call_id": "c1", "output": {"temp_f": 72, "condition": "sunny"}}
        ]

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_node_context.return_value = {}
            mock_cache.get_adapter_settings.return_value = None

            result = await handler._format_tool_result_text_mode(
                "conv-4", messages, tool_results
            )

        handler.llm_client.chat_completion.assert_called_once()
        assert result["assistant_message"] == "It's sunny and 72F."
