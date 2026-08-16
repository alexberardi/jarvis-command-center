"""Wake-clip verdict wiring across the three voice paths.

Before 2026-08-15 only the blocking path resolved the wake-verification
verdict; ``stream_voice_response`` and the native tool-stream path read a
``turn_context["wake_verified"]`` key nothing ever set, so the mode meant
different things depending on which path the router picked. All three paths
now share ``_resolve_wake_verified_into_context``:

* verified / pending / off / clip_unreliable → proceed unchanged (fail open)
* unverified + bias → ``turn_context["wake_verified"] = False`` (mild hint)
* unverified + enforce → blocking path returns silent not_for_me; streaming
  paths defer to the blocking path (return None)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.conversation_handler import ConversationHandler


def _handler(prompt_provider=None) -> ConversationHandler:
    return ConversationHandler(
        model=MagicMock(), llm_client=MagicMock(), prompt_provider=prompt_provider
    )


class TestResolveWakeVerifiedIntoContext:
    """The shared helper: verdict → turn_context, one behavior for all paths."""

    def _run(self, resolved, turn_context):
        handler = _handler()
        node_ctx = {"household_id": "hh-1", "node_id": "node-1"}
        with patch(
            "app.core.conversation_handler.resolve_wake_verification",
            new=AsyncMock(return_value=resolved),
        ) as mock_resolve, patch(
            "app.core.conversation_handler.conversation_cache"
        ) as mock_cache:
            mock_cache.get_node_context.return_value = node_ctx
            import asyncio

            suppress = asyncio.run(
                handler._resolve_wake_verified_into_context(
                    "conv-1", "turn on the lights", turn_context
                )
            )
        return suppress, mock_resolve

    def test_no_verdict_fails_open(self):
        turn_context = {"source": "wake"}
        suppress, _ = self._run(None, turn_context)
        assert suppress is False
        assert "wake_verified" not in turn_context

    def test_verified_wake_fails_open(self):
        turn_context = {"source": "wake"}
        suppress, _ = self._run(
            {"verified": True, "mode": "bias"}, turn_context
        )
        assert suppress is False
        assert "wake_verified" not in turn_context

    def test_unverified_bias_sets_soft_flag(self):
        turn_context = {"source": "wake"}
        suppress, _ = self._run(
            {"verified": False, "mode": "bias", "transcript": "eat it"},
            turn_context,
        )
        assert suppress is False
        assert turn_context["wake_verified"] is False

    def test_unverified_enforce_demands_suppression(self):
        turn_context = {"source": "wake"}
        suppress, _ = self._run(
            {"verified": False, "mode": "enforce", "transcript": "eat it"},
            turn_context,
        )
        assert suppress is True

    def test_household_and_node_scope_the_mode_lookup(self):
        # The mode must be resolved with the conversation's household/node so
        # per-household settings mean the same thing on every path.
        turn_context = {"source": "wake"}
        _, mock_resolve = self._run(None, turn_context)
        kwargs = mock_resolve.await_args.kwargs
        assert kwargs["household_id"] == "hh-1"
        assert kwargs["node_id"] == "node-1"


class TestBlockingPathEnforce:
    @pytest.mark.asyncio
    async def test_enforce_returns_silent_not_for_me(self):
        handler = _handler()
        with patch.object(
            handler,
            "_resolve_wake_verified_into_context",
            new=AsyncMock(return_value=True),
        ):
            result = await handler.process_voice_command_with_tools(
                voice_command="turn on the lights",
                conversation_id="conv-1",
                turn_context={"source": "wake", "wake_confidence": 0.95},
            )
        assert result == {"stop_reason": "not_for_me", "assistant_message": ""}


class TestStreamingPathWiring:
    """stream_voice_response resolves the verdict once it commits to the
    fast path: enforce defers to the blocking path (which re-reads the
    cached verdict instantly and suppresses there)."""

    @pytest.mark.asyncio
    async def test_enforce_defers_to_blocking_path(self):
        handler = _handler()
        resolve = AsyncMock(return_value=True)
        with patch("app.core.conversation_handler.conversation_cache") as mock_cache, \
             patch.object(handler, "_apply_tool_filtering", return_value=[]), \
             patch.object(
                 handler,
                 "_apply_tool_routing_with_cache",
                 return_value={"tool_name": "answer_question", "score": 0.95},
             ), \
             patch.object(handler, "_resolve_wake_verified_into_context", new=resolve):
            mock_cache.get_messages.return_value = [
                {"role": "system", "content": "x"}
            ]
            mock_cache.get_tools.return_value = []
            mock_cache.get_available_commands.return_value = []

            result = await handler.stream_voice_response(
                conversation_id="conv-1",
                voice_command="what time is it",
                tts_client=MagicMock(),
                turn_context={"source": "wake", "wake_confidence": 0.95},
            )

        assert result is None
        resolve.assert_awaited_once()


class TestToolStreamPathWiring:
    @pytest.mark.asyncio
    async def test_enforce_defers_to_blocking_path(self, monkeypatch):
        monkeypatch.setenv("JARVIS_STREAM_TOOL_RESPONSES", "true")
        provider = MagicMock()
        provider.supports_native_tools = True
        handler = _handler(prompt_provider=provider)
        resolve = AsyncMock(return_value=True)
        with patch("app.core.conversation_handler.conversation_cache") as mock_cache, \
             patch.object(
                 handler,
                 "_apply_tool_filtering",
                 return_value=[{"function": {"name": "control_device"}}],
             ), \
             patch.object(
                 handler,
                 "_apply_tool_routing_with_cache",
                 return_value={"tool_name": "control_device", "score": 0.95},
             ), \
             patch(
                 "app.core.tool_registry.tool_registry.has_tool",
                 return_value=True,
             ), \
             patch.object(handler, "_resolve_wake_verified_into_context", new=resolve):
            mock_cache.get_messages.return_value = [
                {"role": "system", "content": "x"}
            ]
            mock_cache.get_tools.return_value = [
                {"function": {"name": "control_device"}}
            ]
            mock_cache.get_available_commands.return_value = []

            result = await handler.stream_voice_response_with_tools(
                conversation_id="conv-1",
                voice_command="turn on the lights",
                tts_client=MagicMock(),
                turn_context={"source": "wake", "wake_confidence": 0.95},
            )

        assert result is None
        resolve.assert_awaited_once()
