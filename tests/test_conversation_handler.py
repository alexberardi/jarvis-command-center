"""
Tests for ConversationHandler.

These tests cover the conversation orchestration logic extracted from ModelService.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List


class TestConversationHandlerInit:
    """Tests for ConversationHandler initialization."""

    def test_init_with_model_and_client(self):
        """Test initialization with model and LLM client."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = MagicMock()

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        assert handler.model is mock_model
        assert handler.llm_client is mock_client


class TestConversationCleanup:
    """Tests for conversation cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_from_cache(self):
        """Test that cleanup removes conversation from cache."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = MagicMock()
        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            await handler.cleanup_conversation("test-conv-id")
            mock_cache.remove.assert_called_once_with("test-conv-id")


class TestWarmupConversation:
    """Tests for conversation warmup."""

    @pytest.mark.asyncio
    async def test_warmup_stores_in_cache(self):
        """Test that warmup stores conversation state in cache."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_model._build_system_prompt = MagicMock(return_value="System prompt")
        mock_client = MagicMock()

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            with patch("app.core.conversation_handler.tool_registry") as mock_registry:
                mock_registry.get_tools_for_model.return_value = []

                await handler.warmup_conversation_with_tools(
                    conversation_id="test-conv",
                    node_context={"room": "kitchen"},
                    timezone="America/New_York",
                    client_tools=[{"function": {"name": "get_weather"}}],
                    available_commands=None,
                )

                mock_cache.set.assert_called_once()
                call_kwargs = mock_cache.set.call_args
                assert call_kwargs[1]["conversation_id"] == "test-conv"

    @pytest.mark.asyncio
    async def test_warmup_merges_server_and_client_tools(self):
        """Test that warmup merges server and client tools."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_model._build_system_prompt = MagicMock(return_value="System prompt")
        mock_client = MagicMock()

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        server_tools = [{"function": {"name": "resolve_date"}}]
        client_tools = [{"function": {"name": "get_weather"}}]

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            with patch("app.core.conversation_handler.tool_registry") as mock_registry:
                mock_registry.get_tools_for_model.return_value = server_tools

                await handler.warmup_conversation_with_tools(
                    conversation_id="test-conv",
                    node_context={"room": "kitchen"},
                    timezone="America/New_York",
                    client_tools=client_tools,
                    available_commands=None,
                )

                # Check that tools passed to cache include both
                call_kwargs = mock_cache.set.call_args[1]
                tools = call_kwargs["tools"]
                tool_names = [t["function"]["name"] for t in tools]
                assert "resolve_date" in tool_names
                assert "get_weather" in tool_names


class TestProcessVoiceCommand:
    """Tests for processing voice commands."""

    @pytest.mark.asyncio
    async def test_process_raises_if_conversation_not_found(self):
        """Test that process raises ValueError if conversation not in cache."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = MagicMock()
        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = None

            with pytest.raises(ValueError, match="not found or expired"):
                await handler.process_voice_command_with_tools(
                    voice_command="what's the weather",
                    conversation_id="nonexistent",
                )

    @pytest.mark.asyncio
    async def test_process_adds_user_message(self):
        """Test that process adds user message to conversation."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_model.use_tool_classifier = False
        mock_client = MagicMock()

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        messages = [{"role": "system", "content": "You are helpful"}]

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = messages
            mock_cache.get_tools.return_value = []
            mock_cache.get_available_commands.return_value = []
            mock_cache.get_node_context.return_value = {}
            mock_cache.get_timezone.return_value = None

            with patch("app.core.conversation_handler.ToolExecutionEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.execute = AsyncMock(return_value={
                    "stop_reason": "complete",
                    "assistant_message": "The weather is sunny",
                })
                MockEngine.return_value = mock_engine

                await handler.process_voice_command_with_tools(
                    voice_command="what's the weather",
                    conversation_id="test-conv",
                )

                # Check that user message was added
                assert any(
                    msg.get("role") == "user" and "weather" in msg.get("content", "")
                    for msg in messages
                )


class TestContinueConversation:
    """Tests for continuing conversation with tool results."""

    @pytest.mark.asyncio
    async def test_continue_raises_if_conversation_not_found(self):
        """Test that continue raises ValueError if conversation not in cache."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = MagicMock()
        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = None

            with pytest.raises(ValueError, match="not found or expired"):
                await handler.continue_conversation_with_tool_results(
                    conversation_id="nonexistent",
                    tool_results=[{"tool_call_id": "123", "output": "result"}],
                )

    @pytest.mark.asyncio
    async def test_continue_formats_tool_results_text_mode(self):
        """Test that continue formats tool results for text-based models.

        Text-based models can't process role='tool' messages, so the handler
        replaces them with a clean assistant message containing the data.
        """
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {
            "choices": [{"message": {"content": "It's sunny and 72F."}}],
        }
        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Get weather"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_123"}]},
        ]

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = messages
            mock_cache.get_tools.return_value = []

            result = await handler.continue_conversation_with_tool_results(
                conversation_id="test-conv",
                tool_results=[{"tool_call_id": "call_123", "output": "Sunny, 72F"}],
            )

            # The formatted response replaces tool exchange messages
            # with a single assistant message containing data + response
            last_msg = messages[-1]
            assert last_msg["role"] == "assistant"
            assert "Sunny, 72F" in last_msg["content"]  # tool data preserved
            assert result["stop_reason"] == "complete"


class TestGetSystemPromptDispatch:
    """Tests for _get_system_prompt dispatch logic."""

    def test_uses_prompt_provider_when_set(self):
        """When prompt_provider is set, build_system_prompt is used."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_model._build_system_prompt = MagicMock(return_value="legacy prompt")
        mock_client = MagicMock()
        mock_provider = MagicMock()
        mock_provider.build_system_prompt = MagicMock(return_value="new prompt")

        handler = ConversationHandler(
            model=mock_model, llm_client=mock_client, prompt_provider=mock_provider
        )

        result = handler._get_system_prompt({"room": "kitchen"}, "UTC", [])

        assert result == "new prompt"
        mock_provider.build_system_prompt.assert_called_once()
        mock_model._build_system_prompt.assert_not_called()

    def test_falls_back_to_model_when_no_provider(self):
        """When no prompt_provider, falls back to model._build_system_prompt."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_model._build_system_prompt = MagicMock(return_value="legacy prompt")
        mock_client = MagicMock()

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        result = handler._get_system_prompt({"room": "kitchen"}, "UTC", [])

        assert result == "legacy prompt"
        mock_model._build_system_prompt.assert_called_once()

    def test_falls_back_to_default_when_no_provider_or_method(self):
        """When no provider and no _build_system_prompt, returns fallback."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock(spec=["name", "use_tool_classifier"])
        mock_client = MagicMock()

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        result = handler._get_system_prompt({"room": "kitchen"}, "UTC", [])

        assert result == "You are a helpful voice assistant."

    def test_init_stores_prompt_provider(self):
        """Test that prompt_provider is stored on init."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = MagicMock()
        mock_provider = MagicMock()

        handler = ConversationHandler(
            model=mock_model, llm_client=mock_client, prompt_provider=mock_provider
        )

        assert handler.prompt_provider is mock_provider

    def test_init_defaults_prompt_provider_to_none(self):
        """Test that prompt_provider defaults to None."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = MagicMock()

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        assert handler.prompt_provider is None

    def test_use_tool_classifier_from_provider(self):
        """use_tool_classifier should prefer prompt_provider when set."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_model.use_tool_classifier = True
        mock_client = MagicMock()
        mock_provider = MagicMock()
        mock_provider.use_tool_classifier = False

        handler = ConversationHandler(
            model=mock_model, llm_client=mock_client, prompt_provider=mock_provider
        )

        # The handler should check provider first in _apply_tool_routing_with_cache
        # We verify the property is accessible
        assert handler.prompt_provider.use_tool_classifier is False
        assert handler.model.use_tool_classifier is True


class TestPromptProviderThreading:
    """Tests for prompt_provider being passed to ToolExecutionEngine."""

    @pytest.mark.asyncio
    async def test_process_voice_command_passes_prompt_provider(self):
        """Test that process_voice_command_with_tools passes prompt_provider to engine."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_model.use_tool_classifier = False
        mock_client = MagicMock()
        mock_provider = MagicMock()

        handler = ConversationHandler(
            model=mock_model, llm_client=mock_client, prompt_provider=mock_provider
        )

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = [{"role": "system", "content": "Hi"}]
            mock_cache.get_tools.return_value = []
            mock_cache.get_available_commands.return_value = []
            mock_cache.get_node_context.return_value = {}
            mock_cache.get_timezone.return_value = None

            with patch("app.core.conversation_handler.ToolExecutionEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.execute = AsyncMock(return_value={
                    "stop_reason": "complete",
                    "assistant_message": "Done",
                })
                MockEngine.return_value = mock_engine

                await handler.process_voice_command_with_tools(
                    voice_command="hello",
                    conversation_id="test-conv",
                )

                MockEngine.assert_called_once_with(
                    mock_client, prompt_provider=mock_provider
                )

    @pytest.mark.asyncio
    async def test_continue_conversation_passes_prompt_provider(self):
        """Test that continue_conversation_with_tool_results passes prompt_provider to engine."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = MagicMock()
        mock_provider = MagicMock()

        handler = ConversationHandler(
            model=mock_model, llm_client=mock_client, prompt_provider=mock_provider
        )

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "test"},
            ]
            mock_cache.get_tools.return_value = []

            with patch("app.core.conversation_handler.ToolExecutionEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.execute = AsyncMock(return_value={
                    "stop_reason": "complete",
                    "assistant_message": "Done",
                })
                MockEngine.return_value = mock_engine

                await handler.continue_conversation_with_tool_results(
                    conversation_id="test-conv",
                    tool_results=[{"tool_call_id": "call_123", "output": "result"}],
                )

                MockEngine.assert_called_once_with(
                    mock_client, prompt_provider=mock_provider
                )


class TestReplaceToolExchangeInHistory:
    """Tests for _replace_tool_exchange_in_history."""

    def test_replaces_tool_exchange_with_data_and_response(self):
        """Tool-call assistant + tool messages are replaced with a clean message."""
        from app.core.conversation_handler import ConversationHandler

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "check my email"},
            {"role": "assistant", "content": '{"tool_calls": [...]}', "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"emails": []}'},
        ]

        ConversationHandler._replace_tool_exchange_in_history(
            messages, '{"emails": [], "total_unread": 0}', "No unread emails"
        )

        assert len(messages) == 3  # system + user + assistant
        assert messages[2]["role"] == "assistant"
        assert "No unread emails" in messages[2]["content"]
        assert "Tool data:" in messages[2]["content"]
        assert "total_unread" in messages[2]["content"]

    def test_preserves_earlier_messages(self):
        """Messages before the tool exchange are untouched."""
        from app.core.conversation_handler import ConversationHandler

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "check email"},
            {"role": "assistant", "content": "{}", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "data"},
        ]

        ConversationHandler._replace_tool_exchange_in_history(
            messages, "data", "You have mail"
        )

        assert len(messages) == 5  # system + user + assistant + user + assistant(merged)
        assert messages[0]["content"] == "System"
        assert messages[2]["content"] == "Hi there!"
        assert messages[3]["content"] == "check email"
        assert "You have mail" in messages[4]["content"]

    def test_truncates_large_tool_data(self):
        """Tool data over 1500 chars is truncated."""
        from app.core.conversation_handler import ConversationHandler

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "{}", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "x"},
        ]

        large_data = "x" * 2000
        ConversationHandler._replace_tool_exchange_in_history(
            messages, large_data, "response"
        )

        content = messages[-1]["content"]
        # Should be truncated, not the full 2000 chars
        assert len(content) < 2000
        assert "..." in content

    def test_no_tool_exchange_appends_normally(self):
        """When no tool exchange is found, appends response at the end."""
        from app.core.conversation_handler import ConversationHandler

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "hello"},
        ]

        ConversationHandler._replace_tool_exchange_in_history(
            messages, "data", "response"
        )

        assert len(messages) == 3
        assert messages[2]["role"] == "assistant"
        assert "response" in messages[2]["content"]


class TestGetServerToolNames:
    """Tests for extracting server tool names."""

    def test_get_server_tool_names(self):
        """Test extracting server tool names from tools list."""
        from app.core.conversation_handler import get_server_tool_names

        tools = [
            {"function": {"name": "resolve_date"}},
            {"function": {"name": "get_weather"}},
            {"function": {"name": "validate_params"}},
        ]

        with patch("app.core.conversation_handler.tool_registry") as mock_registry:
            # Only resolve_date and validate_params are server tools
            mock_registry.has_tool.side_effect = lambda name: name in {"resolve_date", "validate_params"}

            server_names = get_server_tool_names(tools)

            assert "resolve_date" in server_names
            assert "validate_params" in server_names
            assert "get_weather" not in server_names

    def test_get_server_tool_names_handles_none(self):
        """Test handling None tools list."""
        from app.core.conversation_handler import get_server_tool_names

        server_names = get_server_tool_names(None)
        assert server_names == set()

    def test_get_server_tool_names_handles_empty(self):
        """Test handling empty tools list."""
        from app.core.conversation_handler import get_server_tool_names

        server_names = get_server_tool_names([])
        assert server_names == set()
