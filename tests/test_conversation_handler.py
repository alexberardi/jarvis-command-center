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


class TestWarmupAlwaysRuns:
    """The warmup inference is ALWAYS performed — it is never skippable.

    Regression lock for the live incident where a node sending
    ``skip_warmup_inference=True`` disabled the pre-warm, making every voice
    turn pay a ~1.4s cold prefill. The skip flag has been removed from the
    warmup path entirely; these tests assert the warmup LLM call fires
    regardless and that no skip parameter is accepted anymore.
    """

    @pytest.mark.asyncio
    async def test_text_path_always_warms(self):
        """Text-based path (no native tools) issues the max_tokens=1 warmup
        call to populate the prompt-prefix KV cache."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_model._build_system_prompt = MagicMock(return_value="System prompt")
        mock_client = AsyncMock()
        mock_client.allows_warmup_caching.return_value = True
        mock_client.get_date_keys.return_value = []

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        with patch("app.core.conversation_handler.conversation_cache"):
            with patch("app.core.conversation_handler.tool_registry") as mock_registry:
                mock_registry.get_tools_for_model.return_value = []

                await handler.warmup_conversation_with_tools(
                    conversation_id="test-conv",
                    node_context={"room": "kitchen"},
                    timezone="America/New_York",
                    client_tools=[{"function": {"name": "get_weather"}}],
                    available_commands=None,
                )

        mock_client.chat_completion.assert_awaited_once()
        assert mock_client.chat_completion.call_args.kwargs.get("max_tokens") == 1

    @pytest.mark.asyncio
    async def test_native_path_always_warms(self):
        """Native-tools path issues the warmup call with tool definitions."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = AsyncMock()
        mock_client.get_date_keys.return_value = []

        provider = MagicMock()
        provider.supports_native_tools = True
        provider.force_tool_calls = False
        provider.build_system_prompt = MagicMock(return_value="System prompt")
        provider.build_tools = MagicMock(return_value=[{"type": "function"}])

        handler = ConversationHandler(
            model=mock_model, llm_client=mock_client, prompt_provider=provider
        )

        with patch("app.core.conversation_handler.conversation_cache"):
            with patch("app.core.conversation_handler.tool_registry") as mock_registry:
                mock_registry.get_tools_for_model.return_value = []

                await handler.warmup_conversation_with_tools(
                    conversation_id="test-conv",
                    node_context={"room": "kitchen"},
                    timezone="America/New_York",
                    client_tools=[{"function": {"name": "get_weather"}}],
                    available_commands=None,
                )

        mock_client.chat_completion.assert_awaited_once()
        assert "tools" in mock_client.chat_completion.call_args.kwargs

    @pytest.mark.asyncio
    async def test_skip_flag_no_longer_accepted(self):
        """The skip_warmup_inference parameter is gone — passing it must raise
        TypeError so the skip path can never be reintroduced by a caller."""
        from app.core.conversation_handler import ConversationHandler

        handler = ConversationHandler(model=MagicMock(), llm_client=AsyncMock())

        with pytest.raises(TypeError):
            await handler.warmup_conversation_with_tools(
                conversation_id="test-conv",
                node_context={"room": "kitchen"},
                timezone="America/New_York",
                client_tools=None,
                available_commands=None,
                skip_warmup_inference=True,
            )


class TestSttNoisePreFilter:
    """The pre-LLM gate added 2026-06-02: bracketed / empty transcripts
    must return ``not_for_me`` without invoking the LLM at all. Locks in
    that the engine is not even constructed for these inputs."""

    @pytest.mark.asyncio
    async def test_asterisk_sniff_short_circuits_without_llm(self):
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock()
        mock_client = MagicMock()
        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        with patch("app.core.conversation_handler.ToolExecutionEngine") as MockEngine:
            with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
                # Cache should NEVER be consulted — we short-circuit before
                # touching it. Set a return value anyway so a mistaken read
                # would still let the test reach the engine assertion.
                mock_cache.get_messages.return_value = [{"role": "system", "content": "sys"}]
                mock_cache.get_tools.return_value = []
                mock_cache.get_available_commands.return_value = []

                result = await handler.process_voice_command_with_tools(
                    voice_command="*sniff*",
                    conversation_id="test-conv",
                )

        assert result["stop_reason"] == "not_for_me"
        assert result["assistant_message"] == ""
        MockEngine.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_string_short_circuits(self):
        from app.core.conversation_handler import ConversationHandler

        handler = ConversationHandler(model=MagicMock(), llm_client=MagicMock())

        with patch("app.core.conversation_handler.ToolExecutionEngine") as MockEngine:
            result = await handler.process_voice_command_with_tools(
                voice_command="",
                conversation_id="test-conv",
            )

        assert result["stop_reason"] == "not_for_me"
        MockEngine.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_command_does_not_short_circuit(self):
        """Sanity: legitimate commands still reach the engine."""
        from app.core.conversation_handler import ConversationHandler

        handler = ConversationHandler(model=MagicMock(), llm_client=MagicMock())
        messages = [{"role": "system", "content": "sys"}]

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
                    "assistant_message": "Sure",
                })
                MockEngine.return_value = mock_engine

                await handler.process_voice_command_with_tools(
                    voice_command="what's the weather",
                    conversation_id="test-conv",
                )

                # Engine must have been invoked exactly once.
                mock_engine.execute.assert_called_once()


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

            # The text-mode path runs the tool output through a formatting
            # LLM call which returns the rephrased user-facing response.
            # The raw tool output ("Sunny, 72F") is NOT preserved verbatim —
            # the formatter rephrases it into natural language.
            last_msg = messages[-1]
            assert last_msg["role"] == "assistant"
            assert last_msg["content"] == "It's sunny and 72F."
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

        assert result.startswith("new prompt")
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

        assert result.startswith("legacy prompt")
        mock_model._build_system_prompt.assert_called_once()

    def test_falls_back_to_default_when_no_provider_or_method(self):
        """When no provider and no _build_system_prompt, returns fallback."""
        from app.core.conversation_handler import ConversationHandler

        mock_model = MagicMock(spec=["name", "use_tool_classifier"])
        mock_client = MagicMock()

        handler = ConversationHandler(model=mock_model, llm_client=mock_client)

        result = handler._get_system_prompt({"room": "kitchen"}, "UTC", [])

        assert result.startswith("You are a helpful voice assistant.")

    def test_appends_not_for_me_instruction_universally(self):
        """The false-wake gating instruction is appended in the shared
        dispatch — every provider, the legacy model path, and the minimal
        fallback all get it. This is the single point of truth for the
        gating prompt; individual providers must NOT include it themselves.
        """
        from app.core.conversation_handler import ConversationHandler
        from app.core.prompt_providers.shared.core_rules import (
            NOT_FOR_ME_INSTRUCTION,
        )

        # Provider path
        mock_model_a = MagicMock()
        mock_provider = MagicMock()
        mock_provider.build_system_prompt = MagicMock(return_value="provider body")
        handler_a = ConversationHandler(
            model=mock_model_a, llm_client=MagicMock(), prompt_provider=mock_provider,
        )
        assert NOT_FOR_ME_INSTRUCTION in handler_a._get_system_prompt({}, "UTC", [])

        # Legacy model path
        mock_model_b = MagicMock()
        mock_model_b._build_system_prompt = MagicMock(return_value="legacy body")
        handler_b = ConversationHandler(
            model=mock_model_b, llm_client=MagicMock(),
        )
        assert NOT_FOR_ME_INSTRUCTION in handler_b._get_system_prompt({}, "UTC", [])

        # Minimal-fallback path
        mock_model_c = MagicMock(spec=["name", "use_tool_classifier"])
        handler_c = ConversationHandler(
            model=mock_model_c, llm_client=MagicMock(),
        )
        assert NOT_FOR_ME_INSTRUCTION in handler_c._get_system_prompt({}, "UTC", [])

    def test_not_for_me_instruction_is_trailing(self):
        """The instruction must be at the end of the system prompt so it
        sits closest to the user message — that's where recency-bias gives
        it the most weight on weaker models."""
        from app.core.conversation_handler import ConversationHandler
        from app.core.prompt_providers.shared.core_rules import (
            NOT_FOR_ME_INSTRUCTION,
        )

        mock_provider = MagicMock()
        mock_provider.build_system_prompt = MagicMock(return_value="provider body")
        handler = ConversationHandler(
            model=MagicMock(), llm_client=MagicMock(), prompt_provider=mock_provider,
        )

        result = handler._get_system_prompt({}, "UTC", [])
        idx_body = result.index("provider body")
        idx_nfm = result.index(NOT_FOR_ME_INSTRUCTION)
        assert idx_nfm > idx_body

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


class _StreamRecorder:
    """Callable that records whether it was invoked and yields the given
    deltas as an llm_client.chat_completion_stream() async iterator."""

    def __init__(self, deltas):
        self.deltas = deltas
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True

        async def _gen():
            for d in self.deltas:
                yield {"delta": d}
            yield {"done": True}

        return _gen()


class TestStreamingContinueFastPathAndGuard:
    """Regression coverage for the 'Task completed.' generic-filler bug.

    The streaming voice continue path (stream_continue_with_tool_results)
    must (1) speak a command's own message verbatim without an LLM call,
    (2) still hit the LLM for knowledge-delegation results, and (3) never
    speak raw JSON / generic filler — substituting a tool-derived message.
    """

    @staticmethod
    def _handler():
        from app.core.conversation_handler import ConversationHandler
        handler = ConversationHandler(model=MagicMock(), llm_client=MagicMock())
        handler.prompt_provider = None  # text-based path
        return handler

    @staticmethod
    def _tts_client(spoken: List[str]) -> MagicMock:
        client = MagicMock()

        async def fake_speak_stream(text: str, timeout: float = 30.0):
            spoken.append(text)

            async def _iter():
                yield b"\x00" * 10

            return _iter(), {"sample_rate": "22050"}

        client.speak_stream = AsyncMock(side_effect=fake_speak_stream)
        return client

    @staticmethod
    async def _drain(gen):
        async for _ in gen:
            pass

    @pytest.mark.asyncio
    async def test_fast_path_speaks_tool_message_without_llm(self):
        """A tool result carrying a clean message is spoken verbatim and the
        formatting LLM is never called."""
        handler = self._handler()
        handler.llm_client.chat_completion_stream = MagicMock()  # must NOT be called
        spoken: List[str] = []
        tts = self._tts_client(spoken)
        tool_results = [
            {"tool_call_id": "t1", "output": {"message": "Added eggplant to your shopping list."}}
        ]
        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = [{"role": "user", "content": "add eggplant"}]
            gen = await handler.stream_continue_with_tool_results("conv-1", tool_results, tts)
            assert gen is not None
            await self._drain(gen)

        handler.llm_client.chat_completion_stream.assert_not_called()
        assert spoken == ["Added eggplant to your shopping list."]
        mock_cache.update_messages.assert_called_once()
        committed = mock_cache.update_messages.call_args[0][1]
        assert committed[-1] == {
            "role": "assistant",
            "content": "Added eggplant to your shopping list.",
        }

    @pytest.mark.asyncio
    async def test_knowledge_query_falls_through_to_llm(self):
        """answer_question-style results (echo only the query) must still hit
        the formatting LLM rather than being fast-pathed."""
        rec = _StreamRecorder(["Quantum entanglement is ", "a real phenomenon."])
        handler = self._handler()
        handler.llm_client.chat_completion_stream = rec
        spoken: List[str] = []
        tts = self._tts_client(spoken)
        tool_results = [{"tool_call_id": "t1", "output": {"query": "quantum entanglement"}}]
        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = [
                {"role": "user", "content": "explain quantum entanglement"}
            ]
            mock_cache.get_node_context.return_value = {}
            gen = await handler.stream_continue_with_tool_results("conv-2", tool_results, tts)
            assert gen is not None
            await self._drain(gen)

        assert rec.called, "knowledge query must reach the formatting LLM"
        assert any("Quantum entanglement" in s for s in spoken), f"spoken={spoken}"

    @pytest.mark.asyncio
    async def test_guard_substitutes_and_never_speaks_raw_json(self):
        """When the formatting LLM re-emits a bare tool-call (raw JSON) instead
        of prose, the user must never hear the JSON — a tool-derived message is
        substituted, and the cache is not polluted with JSON."""
        rec = _StreamRecorder(['{"name": "get_calendar_events", ', '"arguments": {}}'])
        handler = self._handler()
        handler.llm_client.chat_completion_stream = rec
        spoken: List[str] = []
        tts = self._tts_client(spoken)
        tool_results = [
            {"tool_call_id": "t1", "output": {"success": False, "error": "I couldn't reach your calendar."}}
        ]
        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = [
                {"role": "user", "content": "what's on my calendar"}
            ]
            mock_cache.get_node_context.return_value = {}
            gen = await handler.stream_continue_with_tool_results("conv-3", tool_results, tts)
            assert gen is not None
            await self._drain(gen)

        assert not any(s.strip().startswith("{") for s in spoken), f"spoke raw JSON: {spoken}"
        assert any("couldn't reach your calendar" in s for s in spoken), f"spoken={spoken}"
        if mock_cache.update_messages.called:
            committed = mock_cache.update_messages.call_args[0][1]
            assert not committed[-1]["content"].strip().startswith("{")

    @pytest.mark.asyncio
    async def test_native_provider_fast_path_preserves_tool_message(self):
        """Native-tools providers NOW use the fast-path too (skip the formatting
        LLM) — but MUST commit the native history shape, preserving the
        role='tool' message keyed by tool_call_id so a follow-up turn's history
        stays valid. Previously excluded, which regressed the move to the
        native Qwen3.5-9B: every tool command paid a wasted ~0.8s formatting
        call that produced no usable prose -> robotic fallback."""
        rec = _StreamRecorder(["should not be used"])
        handler = self._handler()
        handler.llm_client.chat_completion_stream = rec  # must NOT be called
        provider = MagicMock()
        provider.supports_native_tools = True
        provider.think_delimiters = ("<think>", "</think>")
        handler.prompt_provider = provider
        spoken: List[str] = []
        tts = self._tts_client(spoken)
        tool_results = [
            {"tool_call_id": "t1", "output": {"message": "Added eggplant to your shopping list."}}
        ]
        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = [{"role": "user", "content": "add eggplant"}]
            mock_cache.get_node_context.return_value = {}
            gen = await handler.stream_continue_with_tool_results("conv-4", tool_results, tts)
            assert gen is not None
            await self._drain(gen)

        assert not rec.called, "native provider should now be fast-pathed (no formatting LLM)"
        assert spoken == ["Added eggplant to your shopping list."]
        mock_cache.update_messages.assert_called_once()
        committed = mock_cache.update_messages.call_args[0][1]
        tool_msgs = [m for m in committed if m.get("role") == "tool"]
        assert len(tool_msgs) == 1 and tool_msgs[0]["tool_call_id"] == "t1", \
            f"native commit must preserve the role='tool' message: {committed}"
        assert committed[-1] == {
            "role": "assistant",
            "content": "Added eggplant to your shopping list.",
        }

    def test_transient_system_block_identification_and_dedup(self):
        """Per-turn transient system blocks (speaker, router hint, stream
        override) must be de-duplicated: exactly one of each per turn, never
        accumulating across turns."""
        from app.core.conversation_handler import _is_transient_system_block as T

        name_only = {"role": "system", "content": "You are speaking with alex."}
        combo = {"role": "system", "content": (
            "You are speaking with alex.\n\nUser Profile - If user asks a "
            "question about one of these items you must respond with an "
            "answer:\n- coffee black"
        )}
        mem_only = {"role": "system", "content": (
            "User Profile - If user asks a question about one of these items "
            "you must respond with an answer:\n- vegetarian"
        )}
        router = {"role": "system", "content": "Router hint: likely tool is 'get_weather'."}
        override = {"role": "system", "content": "Respond naturally in plain text. Do not use JSON format or call any tools."}
        static_rule = {"role": "system", "content": "User Profile are different: those are fair game"}
        prefix = {"role": "system", "content": "You are Jarvis, a function calling voice assistant."}
        user = {"role": "user", "content": "what's the weather"}

        # transient blocks (stripped + re-derived each turn)
        assert T(name_only) and T(combo) and T(mem_only)
        assert T(router) and T(override)
        # NOT transient: the cached prefix, the static rule, user turns
        assert not T(static_rule)
        assert not T(prefix)
        assert not T(user)

        # Per-turn strip+append: a history carrying prior speaker + router +
        # override blocks collapses, then exactly one fresh set is re-added.
        msgs = [prefix, name_only, router, override, user, combo]
        msgs[:] = [m for m in msgs if not T(m)]
        assert not any(T(m) for m in msgs)
        msgs.append({"role": "system", "content": "You are speaking with bob."})
        msgs.append({"role": "system", "content": "Router hint: likely tool is 'get_calendar_events'."})
        assert sum(1 for m in msgs if T(m)) == 2  # one speaker + one router, no dupes
        # the cached prefix and real history are untouched
        assert prefix in msgs and user in msgs

    @pytest.mark.asyncio
    async def test_empty_after_clean_falls_through_to_llm(self):
        """A message that clean_for_tts reduces to empty (emoji/markdown-only)
        must NOT fast-path into a 0-byte audio stream — it falls through to the
        LLM path instead of producing a silent 'successful' turn."""
        rec = _StreamRecorder(["Done and done."])
        handler = self._handler()
        handler.llm_client.chat_completion_stream = rec
        spoken: List[str] = []
        tts = self._tts_client(spoken)
        # Emoji-only message → clean_for_tts() → "" → fast-path must not fire.
        tool_results = [{"tool_call_id": "t1", "output": {"message": "👍"}}]
        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = [{"role": "user", "content": "do the thing"}]
            mock_cache.get_node_context.return_value = {}
            gen = await handler.stream_continue_with_tool_results("conv-5", tool_results, tts)
            assert gen is not None
            await self._drain(gen)

        assert rec.called, "empty-after-clean message must fall through to the LLM path"


class TestTerminalFillerGuard:
    """The terminal-filler guard rewrites generic 'Task completed.'-style prose
    on a completed tool turn so it never reaches TTS, while leaving real
    responses and non-prose outcomes untouched."""

    @staticmethod
    def _guard(result):
        from app.core.conversation_handler import ConversationHandler
        return ConversationHandler._rewrite_terminal_filler(result)

    def test_replaces_task_completed(self):
        out = self._guard({"stop_reason": "complete", "assistant_message": "Task completed."})
        assert out["assistant_message"] != "Task completed."
        assert "say it again" in out["assistant_message"].lower()

    def test_replaces_filler_variants_case_insensitive(self):
        for filler in ["task complete", "Task Completed", "COMPLETED.", "Task finished!", "task is complete"]:
            out = self._guard({"stop_reason": "complete", "assistant_message": filler})
            assert out["assistant_message"] not in (filler,), f"did not rewrite {filler!r}"

    def test_keeps_real_response(self):
        msg = "You have 1 event tonight at 7: Boys Night."
        out = self._guard({"stop_reason": "complete", "assistant_message": msg})
        assert out["assistant_message"] == msg

    def test_keeps_sanctioned_terse_done(self):
        # "Done." is a sanctioned fallback (conversation_handler.py:~1607) — not filler.
        out = self._guard({"stop_reason": "complete", "assistant_message": "Done."})
        assert out["assistant_message"] == "Done."

    def test_ignores_non_prose_outcomes(self):
        # tool_calls / not_for_me / validation must never be rewritten even if
        # assistant_message happens to match.
        for sr in ["tool_calls", "not_for_me", "validation_required", "error", "server_tool_complete"]:
            out = self._guard({"stop_reason": sr, "assistant_message": "Task completed."})
            assert out["assistant_message"] == "Task completed.", f"rewrote on stop_reason={sr}"

    def test_handles_missing_assistant_message(self):
        out = self._guard({"stop_reason": "complete"})
        assert out.get("assistant_message") in (None,)  # untouched (no filler match)


class TestConversationStartRequestSkipFlagDeprecated:
    """The deprecated skip_warmup_inference field is kept on the request model
    only so older nodes that still send it don't 422 during rollout. It is
    ignored — the warmup always runs."""

    def test_accepts_skip_true_without_error(self):
        """An old-node payload carrying skip_warmup_inference=True still parses."""
        from app.request_models.conversation_start_request import (
            ConversationStartRequest,
        )

        req = ConversationStartRequest(
            conversation_id="conv-1", skip_warmup_inference=True
        )
        # Field is retained for backward compat but has no effect on warmup.
        assert req.skip_warmup_inference is True

    def test_field_absent_defaults_and_parses(self):
        """A payload omitting the field parses fine (new nodes)."""
        from app.request_models.conversation_start_request import (
            ConversationStartRequest,
        )

        req = ConversationStartRequest(conversation_id="conv-2")
        assert req.skip_warmup_inference is False
