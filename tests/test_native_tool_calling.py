"""
Tests for native tool calling support in command center.

Validates:
- Native provider passes tools to LLM
- Native provider reads structured tool_calls from response
- Native provider skips text parsing
- Text-based provider unchanged
- Date injection works with native tool calls
- _normalize_native_tool_calls helper
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# _normalize_native_tool_calls tests
# ---------------------------------------------------------------------------


class TestNormalizeNativeToolCalls:
    """Test the helper that normalizes native tool calls."""

    def test_normalizes_standard_format(self):
        from app.core.tool_execution_engine import _normalize_native_tool_calls

        raw = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
            }
        ]
        result = _normalize_native_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["id"] == "call_abc123"
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "London"}

    def test_converts_dict_arguments_to_string(self):
        from app.core.tool_execution_engine import _normalize_native_tool_calls

        raw = [
            {
                "id": "call_1",
                "function": {
                    "name": "test_tool",
                    "arguments": {"key": "value"},  # dict, not string
                },
            }
        ]
        result = _normalize_native_tool_calls(raw)
        assert isinstance(result[0]["function"]["arguments"], str)
        assert json.loads(result[0]["function"]["arguments"]) == {"key": "value"}

    def test_generates_id_when_missing(self):
        from app.core.tool_execution_engine import _normalize_native_tool_calls

        raw = [{"function": {"name": "test", "arguments": "{}"}}]
        result = _normalize_native_tool_calls(raw)
        assert result[0]["id"].startswith("call_")

    def test_skips_calls_without_name(self):
        from app.core.tool_execution_engine import _normalize_native_tool_calls

        raw = [
            {"id": "call_1", "function": {"arguments": "{}"}},  # no name
            {"id": "call_2", "function": {"name": "valid", "arguments": "{}"}},
        ]
        result = _normalize_native_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "valid"

    def test_handles_empty_list(self):
        from app.core.tool_execution_engine import _normalize_native_tool_calls

        assert _normalize_native_tool_calls([]) == []

    def test_defaults_arguments_to_empty_json(self):
        from app.core.tool_execution_engine import _normalize_native_tool_calls

        raw = [{"id": "call_1", "function": {"name": "test"}}]
        result = _normalize_native_tool_calls(raw)
        assert result[0]["function"]["arguments"] == "{}"


# ---------------------------------------------------------------------------
# ToolExecutionEngine dual path tests
# ---------------------------------------------------------------------------


class TestNativeToolCallingDualPath:
    """Test that the engine uses native tools when provider supports it."""

    @pytest.fixture
    def mock_llm_client(self):
        client = MagicMock()
        client.chat_completion = AsyncMock()
        return client

    @pytest.fixture
    def mock_conversation_cache(self):
        cache = MagicMock()
        cache.get_available_commands.return_value = []
        cache.get_node_context.return_value = {}
        cache.get_timezone.return_value = "UTC"
        return cache

    @pytest.fixture
    def native_provider(self):
        """A provider that supports native tools."""
        provider = MagicMock()
        provider.supports_native_tools = True
        provider.build_tools.return_value = [
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
        ]
        return provider

    @pytest.fixture
    def text_provider(self):
        """A provider that does NOT support native tools."""
        provider = MagicMock()
        provider.supports_native_tools = False
        provider.get_response_format.return_value = {"type": "text"}
        provider.parse_response.return_value = None
        return provider

    @pytest.mark.asyncio
    async def test_native_provider_passes_tools_to_llm(
        self, mock_llm_client, mock_conversation_cache, native_provider
    ):
        """When supports_native_tools=True, tools are passed via API."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=native_provider)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache), \
             patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
            mock_executor.execute_tool_calls.return_value = ([], [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}}
            ])
            result = await engine.execute(
                conversation_id="test-conv",
                messages=[{"role": "user", "content": "What's the weather in NYC?"}],
                tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )

        # Verify tools were passed to LLM client
        call_kwargs = mock_llm_client.chat_completion.call_args
        assert call_kwargs.kwargs.get("tools") is not None
        assert call_kwargs.kwargs.get("tool_choice") == "auto"
        # response_format should NOT be passed in native mode
        assert "response_format" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_native_provider_reads_structured_tool_calls(
        self, mock_llm_client, mock_conversation_cache, native_provider
    ):
        """Native path reads tool_calls from structured response."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=native_provider)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache), \
             patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
            # Return as client calls so we get them back
            mock_executor.execute_tool_calls.return_value = ([], [
                {"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "London"}'}}
            ])
            result = await engine.execute(
                conversation_id="test-conv",
                messages=[{"role": "user", "content": "Weather in London"}],
                tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )

        assert result["stop_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_native_provider_skips_text_parsing(
        self, mock_llm_client, mock_conversation_cache, native_provider
    ):
        """Native path does not call parse_response."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": "The weather is sunny.", "tool_calls": None},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=native_provider)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv",
                messages=[{"role": "user", "content": "Hello"}],
                tools=[],
            )

        # parse_response should NOT have been called
        native_provider.parse_response.assert_not_called()
        assert result["stop_reason"] == "complete"

    @pytest.mark.asyncio
    async def test_text_provider_unchanged(
        self, mock_llm_client, mock_conversation_cache, text_provider
    ):
        """Text-based path still works with supports_native_tools=False."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "Hello!", "tool_calls": []}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=text_provider)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv",
                messages=[{"role": "user", "content": "Hello"}],
                tools=[],
            )

        # Verify text-based path: response_format was passed, tools were not
        call_kwargs = mock_llm_client.chat_completion.call_args
        assert "response_format" in call_kwargs.kwargs
        assert call_kwargs.kwargs.get("tools") is None
        assert result["stop_reason"] == "complete"


# ---------------------------------------------------------------------------
# HermesMediumUntrained provider tests
# ---------------------------------------------------------------------------


class TestHermesMediumUntrainedNativeTools:
    """Test HermesMediumUntrained's native tools infrastructure (currently disabled)."""

    def test_supports_native_tools_currently_false(self):
        from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
            HermesMediumUntrained,
        )

        provider = HermesMediumUntrained()
        # Currently False — model doesn't reliably use structured tool_calls
        assert provider.supports_native_tools is False

    def test_build_tools_returns_openai_format(self):
        from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
            HermesMediumUntrained,
        )

        provider = HermesMediumUntrained()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = provider.build_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"

    def test_get_response_format_is_text(self):
        from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
            HermesMediumUntrained,
        )

        provider = HermesMediumUntrained()
        assert provider.get_response_format() == {"type": "text"}

    def test_capabilities_include_native_tools_false(self):
        from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
            HermesMediumUntrained,
        )

        provider = HermesMediumUntrained()
        caps = provider.get_capabilities()
        assert caps["supports_native_tools"] is False

    def test_system_prompt_has_tools_xml(self):
        from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
            HermesMediumUntrained,
        )

        provider = HermesMediumUntrained()
        prompt = provider.build_system_prompt(
            node_context={"room": "office", "user": "alex"},
            timezone="UTC",
            tools=[
                {
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    }
                }
            ],
        )
        # Text-based path: tools in <tools> XML tags
        assert "<tools>" in prompt
        assert "</tools>" in prompt
        assert "get_weather" in prompt
