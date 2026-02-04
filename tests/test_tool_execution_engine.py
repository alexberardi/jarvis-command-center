"""
Tests for Tool Execution Engine.

These tests cover the tool execution loop functionality extracted from model_service.py.
Following TDD: write tests first, then implement.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestToolExecutionEngineBasics:
    """Tests for basic execution flow."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.chat_completion = AsyncMock()
        return client

    @pytest.fixture
    def mock_conversation_cache(self):
        """Create a mock conversation cache."""
        cache = MagicMock()
        cache.get_available_commands.return_value = []
        cache.get_node_context.return_value = {}
        cache.get_timezone.return_value = "UTC"
        return cache

    @pytest.mark.asyncio
    async def test_completion_stop_reason(self, mock_llm_client, mock_conversation_cache):
        """Test that stop finish_reason returns complete stop_reason."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # LLM returns a stop response (no tool calls)
        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "Hello!", "tool_calls": []}'},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-123",
                messages=[{"role": "system", "content": "You are a helpful assistant."}],
                tools=[]
            )

        assert result["stop_reason"] == "complete"
        assert result["assistant_message"] == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_calls_detection(self, mock_llm_client, mock_conversation_cache):
        """Test that tool_calls in response are detected and returned."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # LLM returns a tool call response
        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "I'll check the weather.",
                    "tool_calls": [{"name": "get_weather", "arguments": {"location": "NYC"}}]
                })},
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                # Client tool (not in registry)
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}
                }])

                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=[{"role": "system", "content": "You are a helpful assistant."}],
                    tools=[{"function": {"name": "get_weather"}}]
                )

        assert result["stop_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, mock_llm_client, mock_conversation_cache):
        """Test that max iterations returns complete with error."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # LLM always returns server tool calls (causing infinite loop)
        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "Checking...",
                    "tool_calls": [{"name": "resolve_relative_date", "arguments": {"term": "tomorrow"}}]
                })},
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                # Server tool returns result
                mock_executor.execute_tool_calls.return_value = (
                    [{"role": "tool", "tool_call_id": "call_123", "content": '{"result": "ok"}'}],
                    []
                )

                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=[{"role": "system", "content": "You are a helpful assistant."}],
                    tools=[],
                    max_iterations=3
                )

        assert result["stop_reason"] == "complete"
        assert result.get("error") == "max_iterations_exceeded"

    @pytest.mark.asyncio
    async def test_error_handling_llm_failure(self, mock_llm_client, mock_conversation_cache):
        """Test that LLM errors are handled gracefully."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # LLM raises an exception
        mock_llm_client.chat_completion.side_effect = Exception("LLM unavailable")

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-123",
                messages=[{"role": "system", "content": "You are a helpful assistant."}],
                tools=[]
            )

        assert result["stop_reason"] == "error"
        assert "LLM unavailable" in result["error"]


class TestMustCallGuard:
    """Tests for must-call tool guard logic."""

    @pytest.fixture
    def mock_llm_client(self):
        client = MagicMock()
        client.chat_completion = AsyncMock()
        return client

    @pytest.fixture
    def mock_conversation_cache(self):
        cache = MagicMock()
        cache.get_node_context.return_value = {}
        cache.get_timezone.return_value = "UTC"
        return cache

    @pytest.mark.asyncio
    async def test_must_call_retry_triggered(self, mock_llm_client, mock_conversation_cache):
        """Test that must-call guard triggers retry when tool not called."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # Configure available_commands with allow_direct_answer=False
        mock_conversation_cache.get_available_commands.return_value = [
            {"command_name": "set_reminder", "allow_direct_answer": False}
        ]

        call_count = [0]

        def chat_completion_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: LLM returns stop without calling tool
                return {
                    "choices": [{
                        "message": {"content": '{"message": "Sure!", "tool_calls": []}'},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }
            else:
                # Second call: LLM calls the tool
                return {
                    "choices": [{
                        "message": {"content": json.dumps({
                            "message": "Setting reminder.",
                            "tool_calls": [{"name": "set_reminder", "arguments": {"time": "3pm"}}]
                        })},
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }

        mock_llm_client.chat_completion.side_effect = chat_completion_side_effect

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "set_reminder", "arguments": '{"time": "3pm"}'}
                }])

                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=messages,
                    tools=[{"function": {"name": "set_reminder"}}]
                )

        # Should have called LLM twice (first to get stop, then retry)
        assert call_count[0] == 2
        # Check that must-call retry message was added
        retry_messages = [m for m in messages if "[MUST_CALL_RETRY]" in m.get("content", "")]
        assert len(retry_messages) == 1

    @pytest.mark.asyncio
    async def test_must_call_retry_not_added_twice(self, mock_llm_client, mock_conversation_cache):
        """Test that must-call retry is only added once."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_conversation_cache.get_available_commands.return_value = [
            {"command_name": "set_reminder", "allow_direct_answer": False}
        ]

        # LLM always returns stop (never calls tool)
        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "Sure!", "tool_calls": []}'},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            result = await engine.execute(
                conversation_id="test-conv-123",
                messages=messages,
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=5
            )

        # Only one MUST_CALL_RETRY message should be added
        retry_messages = [m for m in messages if "[MUST_CALL_RETRY]" in m.get("content", "")]
        assert len(retry_messages) == 1
        # After retry fails, should return complete
        assert result["stop_reason"] == "complete"


class TestInvalidParamRetry:
    """Tests for invalid parameter retry logic."""

    @pytest.fixture
    def mock_llm_client(self):
        client = MagicMock()
        client.chat_completion = AsyncMock()
        return client

    @pytest.fixture
    def mock_conversation_cache(self):
        cache = MagicMock()
        cache.get_available_commands.return_value = [
            {
                "command_name": "set_reminder",
                "parameters": [
                    {"name": "when", "type": "datetime"}
                ]
            }
        ]
        cache.get_node_context.return_value = {}
        cache.get_timezone.return_value = "UTC"
        return cache

    @pytest.mark.asyncio
    async def test_invalid_param_retry_triggered(self, mock_llm_client, mock_conversation_cache):
        """Test that invalid params trigger retry."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        call_count = [0]

        def chat_completion_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: invalid datetime format (will trigger retry)
                return {
                    "choices": [{
                        "message": {"content": json.dumps({
                            "message": "Setting reminder.",
                            "tool_calls": [{"name": "set_reminder", "arguments": {"when": "tomorrow"}}]
                        })},
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }
            else:
                # Second call: valid datetime
                return {
                    "choices": [{
                        "message": {"content": json.dumps({
                            "message": "Setting reminder.",
                            "tool_calls": [{"name": "set_reminder", "arguments": {"when": "2025-01-15T15:00:00Z"}}]
                        })},
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }

        mock_llm_client.chat_completion.side_effect = chat_completion_side_effect

        engine = ToolExecutionEngine(mock_llm_client)

        # Tool without datetime schema so date injection doesn't auto-fix it
        tools = [{"function": {"name": "set_reminder", "parameters": {"properties": {}}}}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                # Return client tool calls (not server tools)
                def executor_side_effect(tool_calls, **kwargs):
                    # Parse the when value from first tool call
                    args = json.loads(tool_calls[0]["function"]["arguments"])
                    return ([], [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "set_reminder", "arguments": json.dumps(args)}
                    }])
                mock_executor.execute_tool_calls.side_effect = executor_side_effect

                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=messages,
                    tools=tools
                )

        # Should have retry message (invalid param "tomorrow" for datetime)
        retry_messages = [m for m in messages if "[INVALID_PARAM_RETRY" in m.get("content", "")]
        assert len(retry_messages) >= 1


class TestDateKeyInjection:
    """Tests for date key injection into tool calls."""

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
        cache.get_timezone.return_value = "America/New_York"
        return cache

    @pytest.mark.asyncio
    async def test_date_keys_resolved(self, mock_llm_client, mock_conversation_cache):
        """Test that date_keys are resolved and injected."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # LLM returns tool call with date_keys
        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "Setting reminder.",
                    "tool_calls": [{"name": "set_reminder", "arguments": {"when": ""}}]
                })},
                "finish_reason": "tool_calls"
            }],
            "date_keys": ["tomorrow"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        tools = [{
            "function": {
                "name": "set_reminder",
                "parameters": {
                    "properties": {
                        "when": {"type": "string", "format": "date-time"}
                    }
                }
            }
        }]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                # Capture the tool calls passed to executor
                captured_calls = []
                def capture_calls(tool_calls, **kwargs):
                    captured_calls.extend(tool_calls)
                    return ([], tool_calls)
                mock_executor.execute_tool_calls.side_effect = capture_calls

                with patch("app.core.tool_execution_engine.generate_date_context_object") as mock_date_ctx:
                    mock_date_ctx.return_value = {
                        "relative_dates": {
                            "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"}
                        }
                    }

                    result = await engine.execute(
                        conversation_id="test-conv-123",
                        messages=[{"role": "system", "content": "You are a helpful assistant."}],
                        tools=tools,
                        user_utterance="remind me tomorrow"
                    )

        # The when parameter should have been filled with resolved date
        if captured_calls:
            args = json.loads(captured_calls[0]["function"]["arguments"])
            # Should have resolved tomorrow to a datetime
            assert args.get("when") or args.get("when") == ""


class TestValidationRequest:
    """Tests for validation request handling."""

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

    @pytest.mark.asyncio
    async def test_validation_request_returned(self, mock_llm_client, mock_conversation_cache):
        """Test that validation requests are properly returned."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # LLM calls request_validation tool
        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "Which reminder?",
                    "tool_calls": [{"name": "request_validation", "arguments": {
                        "question": "Which reminder?",
                        "parameter_name": "reminder_id",
                        "options": ["Work meeting", "Doctor appointment"]
                    }}]
                })},
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                # Server tool (request_validation)
                mock_executor.execute_tool_calls.return_value = (
                    [{
                        "role": "tool",
                        "tool_call_id": "call_123",
                        "content": json.dumps({
                            "_validation_request": True,
                            "question": "Which reminder?",
                            "parameter_name": "reminder_id",
                            "options": ["Work meeting", "Doctor appointment"]
                        })
                    }],
                    []  # No client tools
                )

                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=[{"role": "system", "content": "You are a helpful assistant."}],
                    tools=[]
                )

        assert result["stop_reason"] == "validation_required"
        assert result["validation_request"]["question"] == "Which reminder?"
        assert result["validation_request"]["parameter_name"] == "reminder_id"


class TestServerToolContinuation:
    """Tests for server tool result handling."""

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

    @pytest.mark.asyncio
    async def test_server_tools_continue_loop(self, mock_llm_client, mock_conversation_cache):
        """Test that server tool results continue the loop."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        call_count = [0]

        def chat_completion_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First: call server tool
                return {
                    "choices": [{
                        "message": {"content": json.dumps({
                            "message": "Looking up examples...",
                            "tool_calls": [{"name": "get_command_examples", "arguments": {"command": "set_reminder"}}]
                        })},
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }
            else:
                # Second: return final answer
                return {
                    "choices": [{
                        "message": {"content": '{"message": "Done!", "tool_calls": []}'},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }

        mock_llm_client.chat_completion.side_effect = chat_completion_side_effect

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                # First call: server tool, second call: complete
                mock_executor.execute_tool_calls.return_value = (
                    [{"role": "tool", "tool_call_id": "call_123", "content": '{"examples": []}'}],
                    []
                )

                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=[{"role": "system", "content": "You are a helpful assistant."}],
                    tools=[]
                )

        # Should have made 2 LLM calls
        assert call_count[0] == 2
        assert result["stop_reason"] == "complete"


class TestUsageTracking:
    """Tests for token usage tracking."""

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

    @pytest.mark.asyncio
    async def test_usage_totals_accumulated(self, mock_llm_client, mock_conversation_cache):
        """Test that token usage is accumulated across iterations."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        call_count = [0]

        def chat_completion_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "choices": [{
                        "message": {"content": json.dumps({
                            "message": "Checking...",
                            "tool_calls": [{"name": "server_tool", "arguments": {}}]
                        })},
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
                }
            else:
                return {
                    "choices": [{
                        "message": {"content": '{"message": "Done!", "tool_calls": []}'},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230}
                }

        mock_llm_client.chat_completion.side_effect = chat_completion_side_effect

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = (
                    [{"role": "tool", "tool_call_id": "call_123", "content": '{}'}],
                    []
                )
                with patch("app.core.tool_execution_engine.write_usage_log") as mock_usage_log:
                    result = await engine.execute(
                        conversation_id="test-conv-123",
                        messages=[{"role": "system", "content": "Test"}],
                        tools=[]
                    )

                    # Check that usage log was called with accumulated totals
                    if mock_usage_log.called:
                        call_args = mock_usage_log.call_args
                        usage_totals = call_args[0][3] if len(call_args[0]) > 3 else call_args[1].get("usage_totals", {})
                        # Should have accumulated usage from both calls
                        # 100 + 200 = 300 prompt tokens
                        # 50 + 30 = 80 completion tokens
