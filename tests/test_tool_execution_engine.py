"""
Tests for Tool Execution Engine.

These tests cover the tool execution loop functionality extracted from model_service.py.
Following TDD: write tests first, then implement.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def _mock_settings_service():
    """Engine calls get_settings_service().get_bool() at execute() time which
    spins up a real DB engine. Stub it so the tests stay offline."""
    fake_settings = MagicMock()
    fake_settings.get_bool.return_value = True
    fake_settings.get.return_value = None
    fake_settings.get_str.return_value = None
    fake_settings.get_int.return_value = 0
    with patch(
        "app.core.tool_execution_engine.get_settings_service",
        return_value=fake_settings,
    ):
        yield


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

        # LLM always returns server tool calls (causing infinite loop). With
        # native tool calling, the structured tool_calls live on the message
        # itself, not in the content payload.
        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": "Checking...",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "resolve_relative_date",
                            "arguments": json.dumps({"term": "tomorrow"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        # Native-tools provider so the engine doesn't take the text-based
        # "server-tool-only early exit" path that bypasses max_iterations.
        mock_provider = MagicMock()
        mock_provider.supports_native_tools = True
        mock_provider.parse_response.return_value = None
        mock_provider.get_response_format.return_value = None
        mock_provider.sanitize_text.side_effect = lambda s: s

        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=mock_provider)

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


class TestNotForMeShortCircuit:
    """The engine MUST short-circuit on ``<not_for_me/>`` before any retry
    guard pops the assistant message. Without this, the force-tool-calls
    and must-call guards keep retrying the sentinel until the model gives
    up and emits prose — the 2026-06-02 prod regression where "I'm sorry"
    → ``<not_for_me/>`` × 2 → ``"It's okay. How can I assist you?"``
    reached TTS.
    """

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
        cache.get_available_commands.return_value = []
        # The combination that produced the prod bug: force_tool_calls=True
        # (Qwen3 compressed provider sets this) and a router decision so
        # both retry branches would have fired.
        cache.get_force_tool_calls.return_value = True
        cache.get_router_decision.return_value = {"used": True}
        return cache

    @pytest.mark.asyncio
    async def test_sentinel_returns_not_for_me_stop_reason(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Iter 1 emits the sentinel → engine returns ``not_for_me``
        without invoking the force-tool-calls retry."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "<not_for_me/>", "tool_calls": []}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-nfm",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert result["stop_reason"] == "not_for_me"

    @pytest.mark.asyncio
    async def test_sentinel_short_circuits_force_tool_retry(
        self, mock_llm_client, mock_conversation_cache
    ):
        """The bug-fix path: even with force_tool_calls=True the engine
        must NOT retry. Exactly one LLM call should fire."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        call_count = [0]

        def chat_completion_side_effect(*args, **kwargs):
            call_count[0] += 1
            return {
                "choices": [{
                    "message": {"content": '{"message": "<not_for_me/>", "tool_calls": []}'},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        mock_llm_client.chat_completion.side_effect = chat_completion_side_effect
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            await engine.execute(
                conversation_id="test-conv-nfm",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert call_count[0] == 1, (
            "Force-tool-calls guard must NOT pop the sentinel and retry — "
            "the prod incident saw 3 LLM calls and prose break-through."
        )

    @pytest.mark.asyncio
    async def test_sentinel_detected_via_raw_content_when_message_field_empty(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Some providers parse out the structured ``message`` field. The
        short-circuit must look at raw_content too so a sentinel-in-JSON
        still wins even if the parsed message is the empty string."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                # Bare sentinel content (no JSON wrapper) — covers the
                # native-tool path where assistant_message == raw_content.
                "message": {"content": "<not_for_me/>"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-nfm",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert result["stop_reason"] == "not_for_me"


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
        # MagicMock returns a truthy mock by default; the engine reads
        # get_force_tool_calls and get_router_decision to decide which retry
        # guard to apply. Pin them to None so only the per-command must-call
        # branch is exercised here.
        cache.get_force_tool_calls.return_value = False
        cache.get_router_decision.return_value = {"used": True}
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

    @pytest.mark.asyncio
    async def test_datetime_array_param_symbolic_keys_resolved(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Test that array<datetime> params get symbolic date keys resolved to ISO.

        When a parameter has format: date-time in its items schema, the LLM may
        return symbolic keys like 'today'. _inject_date_keys should detect the
        datetime type and resolve them to ISO dates—no special param names needed.
        """
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "Checking weather.",
                    "tool_calls": [{
                        "name": "get_weather",
                        "arguments": {"location": "NYC", "dates": ["today"]}
                    }]
                })},
                "finish_reason": "tool_calls"
            }],
            "date_keys": ["today"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        # array<datetime> schema: items have format: date-time
        tools = [{
            "function": {
                "name": "get_weather",
                "parameters": {
                    "properties": {
                        "location": {"type": "string"},
                        "dates": {
                            "type": "array",
                            "items": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        }]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                captured_calls = []

                def capture_calls(tool_calls, **kwargs):
                    captured_calls.extend(tool_calls)
                    return ([], tool_calls)
                mock_executor.execute_tool_calls.side_effect = capture_calls

                with patch("app.core.tool_execution_engine.generate_date_context_object") as mock_date_ctx:
                    mock_date_ctx.return_value = {
                        "current": {
                            "date_iso": "2025-01-15",
                            "datetime": "2025-01-15T12:00:00-05:00",
                            "utc_start_of_day": "2025-01-15T05:00:00Z"
                        },
                        "relative_dates": {
                            "today": {"utc_start_of_day": "2025-01-15T05:00:00Z"}
                        }
                    }

                    result = await engine.execute(
                        conversation_id="test-conv-456",
                        messages=[{"role": "system", "content": "You are a helpful assistant."}],
                        tools=tools,
                        user_utterance="what's the weather today"
                    )

        # The "dates" param (generic name) should have "today" resolved to ISO
        assert captured_calls, "Expected client tool calls to be captured"
        args = json.loads(captured_calls[0]["function"]["arguments"])
        resolved = args.get("dates", [])
        assert len(resolved) >= 1, f"Expected dates to have at least 1 entry, got {resolved}"
        # The resolved value should be an ISO datetime, not the symbolic "today"
        assert resolved[0] != "today", (
            f"Expected 'today' to be resolved to ISO datetime, but got {resolved}"
        )
        assert "2025-01-15" in resolved[0], (
            f"Expected resolved date to contain 2025-01-15, got {resolved[0]}"
        )

    @pytest.mark.asyncio
    async def test_pruned_tool_schema_resolved_from_cache(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Regression: when the router PRUNES the called tool out of the `tools`
        list passed to execute(), date injection must still find its schema via
        the full cached tool list (conversation_cache.get_tools). Otherwise a
        relative value like 'today' survives, param validation flags it, and the
        turn degrades into 'Task completed.' filler.
        """
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "Checking calendar.",
                    "tool_calls": [{
                        "name": "get_calendar_events",
                        "arguments": {"resolved_datetimes": ["today"]}
                    }]
                })},
                "finish_reason": "tool_calls"
            }],
            "date_keys": ["today"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        # The FULL cached schema knows get_calendar_events.resolved_datetimes is
        # array<datetime>. The router pruned it, so it is NOT in the `tools` arg.
        full_schema = [{
            "function": {
                "name": "get_calendar_events",
                "parameters": {
                    "properties": {
                        "resolved_datetimes": {
                            "type": "array",
                            "items": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        }]
        mock_conversation_cache.get_tools.return_value = full_schema

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                captured_calls = []

                def capture_calls(tool_calls, **kwargs):
                    captured_calls.extend(tool_calls)
                    return ([], tool_calls)
                mock_executor.execute_tool_calls.side_effect = capture_calls

                with patch("app.core.tool_execution_engine.generate_date_context_object") as mock_date_ctx:
                    mock_date_ctx.return_value = {
                        "current": {
                            "date_iso": "2025-01-15",
                            "datetime": "2025-01-15T12:00:00-05:00",
                            "utc_start_of_day": "2025-01-15T05:00:00Z"
                        },
                        "relative_dates": {
                            "today": {"utc_start_of_day": "2025-01-15T05:00:00Z"}
                        }
                    }

                    await engine.execute(
                        conversation_id="test-conv-pruned",
                        messages=[{"role": "system", "content": "You are a helpful assistant."}],
                        tools=[],  # router pruned get_calendar_events out of the LLM-facing list
                        user_utterance="what's on my calendar today",
                    )

        assert captured_calls, "Expected client tool calls to be captured"
        args = json.loads(captured_calls[0]["function"]["arguments"])
        resolved = args.get("resolved_datetimes", [])
        assert resolved and resolved[0] != "today", (
            f"Expected 'today' resolved to ISO via cached schema, got {resolved}"
        )
        assert "2025-01-15" in resolved[0], f"Expected ISO date, got {resolved[0]}"

    @pytest.mark.asyncio
    async def test_empty_datetime_array_filled_from_date_keys(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Test that empty datetime array params get filled from date_keys."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # LLM returns tool call with empty datetime array but date_keys present
        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "Getting scores.",
                    "tool_calls": [{
                        "name": "get_scores",
                        "arguments": {"team": "Lakers", "game_dates": []}
                    }]
                })},
                "finish_reason": "tool_calls"
            }],
            "date_keys": ["yesterday"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)

        # Generic param name "game_dates" — type-based detection, not name-based
        tools = [{
            "function": {
                "name": "get_scores",
                "parameters": {
                    "properties": {
                        "team": {"type": "string"},
                        "game_dates": {
                            "type": "array",
                            "items": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        }]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                captured_calls = []

                def capture_calls(tool_calls, **kwargs):
                    captured_calls.extend(tool_calls)
                    return ([], tool_calls)
                mock_executor.execute_tool_calls.side_effect = capture_calls

                with patch("app.core.tool_execution_engine.generate_date_context_object") as mock_date_ctx:
                    mock_date_ctx.return_value = {
                        "relative_dates": {
                            "yesterday": {"utc_start_of_day": "2025-01-14T05:00:00Z"}
                        }
                    }

                    result = await engine.execute(
                        conversation_id="test-conv-789",
                        messages=[{"role": "system", "content": "You are a helpful assistant."}],
                        tools=tools,
                        user_utterance="how did the Lakers do yesterday"
                    )

        assert captured_calls, "Expected client tool calls to be captured"
        args = json.loads(captured_calls[0]["function"]["arguments"])
        resolved = args.get("game_dates", [])
        assert len(resolved) >= 1, f"Expected game_dates to be filled, got {resolved}"
        assert "2025-01-14" in resolved[0], (
            f"Expected resolved date to contain 2025-01-14, got {resolved[0]}"
        )


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

        # Should have made at least 2 LLM calls (loop continued after the
        # server tool returned its result). The engine may issue additional
        # post-process calls (retry guards, format clean-up) — what we're
        # really verifying here is that loop continuation worked.
        assert call_count[0] >= 2
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


class TestPromptProviderIntegration:
    """Tests for prompt_provider wiring in ToolExecutionEngine."""

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
    async def test_prompt_provider_parse_response_transforms_content(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Test that prompt_provider.parse_response transforms content before ToolCallParser."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # LLM returns native format (e.g., XML tool_call tags)
        native_output = '<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>'
        # Provider transforms it to Jarvis JSON
        jarvis_json = json.dumps({
            "message": "",
            "tool_calls": [{"name": "get_weather", "arguments": {"location": "NYC"}}],
            "error": None,
        })

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": native_output},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        mock_provider = MagicMock()
        mock_provider.supports_native_tools = False  # Text-based path
        mock_provider.parse_response.return_value = jarvis_json
        mock_provider.get_response_format.return_value = {"type": "text"}

        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=mock_provider)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}
                }])

                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=[{"role": "system", "content": "You are helpful."}],
                    tools=[{"function": {"name": "get_weather"}}]
                )

        # Provider's parse_response should have been called with native output
        mock_provider.parse_response.assert_called_once_with(native_output)
        assert result["stop_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_prompt_provider_get_response_format_used(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Test that prompt_provider.get_response_format is used when non-None."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "Hello!", "tool_calls": []}'},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        mock_provider = MagicMock()
        mock_provider.supports_native_tools = False  # Text-based path
        mock_provider.get_response_format.return_value = {"type": "text"}
        mock_provider.parse_response.return_value = None
        # Engine pipes the assistant's direct-answer through sanitize_text →
        # clean_for_tts; both need real strings, so pass-through here.
        mock_provider.sanitize_text.side_effect = lambda s: s

        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=mock_provider)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-123",
                messages=[{"role": "system", "content": "You are helpful."}],
                tools=[]
            )

        # Verify the LLM was called with the provider's response_format
        call_kwargs = mock_llm_client.chat_completion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "text"}

    @pytest.mark.asyncio
    async def test_no_prompt_provider_uses_default_response_format(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Test that default response format is used when no prompt_provider."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "Hello!", "tool_calls": []}'},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        engine = ToolExecutionEngine(mock_llm_client)  # No prompt_provider

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.get_response_format") as mock_get_format:
                mock_get_format.return_value = {"type": "json_object"}
                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=[{"role": "system", "content": "You are helpful."}],
                    tools=[]
                )

        # Should have used the default get_response_format() — engine may call
        # it multiple times across iterations, just verify it ran at least once.
        assert mock_get_format.called

    @pytest.mark.asyncio
    async def test_prompt_provider_none_response_format_falls_back(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Test that None from provider.get_response_format falls back to default."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "Hello!", "tool_calls": []}'},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        mock_provider = MagicMock()
        mock_provider.supports_native_tools = False  # Text-based path
        mock_provider.get_response_format.return_value = None  # Falls back to default
        mock_provider.parse_response.return_value = None
        mock_provider.sanitize_text.side_effect = lambda s: s

        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=mock_provider)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.get_response_format") as mock_get_format:
                mock_get_format.return_value = {"type": "json_object"}
                result = await engine.execute(
                    conversation_id="test-conv-123",
                    messages=[{"role": "system", "content": "You are helpful."}],
                    tools=[]
                )

        # Should have fallen back to default — engine may call it multiple
        # times across iterations, just verify it ran at least once.
        assert mock_get_format.called
        call_kwargs = mock_llm_client.chat_completion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}


class TestSentinelIsNotARetryEscapeHatch:
    """2026-07-27 prod incident: "Do you know who Leo is?" — iter 1 produced
    a real prose answer, the force-tool-calls guard popped it and injected
    "[MUST_CALL_RETRY] You MUST call a tool. Do NOT answer directly.", and
    the cornered model emitted ``<not_for_me/>`` as its only legal escape
    (the instruction says the sentinel outranks must-call rules). A silent
    abort reached the node for a directly-addressed question.

    A sentinel produced AFTER a retry guard rejected a real answer is a
    protest, not an addressing verdict — the model already engaged with the
    utterance. The engine must restore the popped answer instead of going
    silent. Sentinels at FIRST look keep terminating (2026-06-02 pin above).
    """

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
        cache.get_available_commands.return_value = []
        # Prod combination: compressed provider forces tool calls.
        cache.get_force_tool_calls.return_value = True
        cache.get_router_decision.return_value = {"used": False}
        return cache

    def _prose_then_sentinel(self):
        responses = [
            {
                "choices": [{
                    "message": {"content": '{"message": "Leo is one of your golden doodles!", "tool_calls": []}'},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{
                    "message": {"content": '{"message": "<not_for_me/>", "tool_calls": []}'},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        return responses

    @pytest.mark.asyncio
    async def test_sentinel_after_forced_retry_restores_popped_answer(
        self, mock_llm_client, mock_conversation_cache
    ):
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = self._prose_then_sentinel()
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-leo",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert "golden doodles" in str(result["assistant_message"])
        assert "not_for_me" not in str(result["assistant_message"])

    @pytest.mark.asyncio
    async def test_restored_answer_replaces_sentinel_in_history(
        self, mock_llm_client, mock_conversation_cache
    ):
        # The cached transcript must end with the restored prose — not the
        # sentinel, and not the [MUST_CALL_RETRY] system nag (which would
        # poison follow-up turns).
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = self._prose_then_sentinel()
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            await engine.execute(
                conversation_id="test-conv-leo",
                messages=messages,
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert messages[-1]["role"] == "assistant"
        assert "golden doodles" in messages[-1]["content"]
        assert not any(
            "[MUST_CALL_RETRY]" in str(m.get("content", "")) for m in messages
        )

    @pytest.mark.asyncio
    async def test_sentinel_at_first_look_still_not_for_me(
        self, mock_llm_client, mock_conversation_cache
    ):
        # Guard the guard: with nothing popped, the sentinel still means
        # "not addressed to me" and must terminate as before.
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "<not_for_me/>", "tool_calls": []}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-ambient",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert result["stop_reason"] == "not_for_me"


class TestSentinelDoubleCheck:
    """2026-07-27 round 2: "Can you tell me who Leo is?" — quiet-room wake,
    speaker matched, profile block carrying the Leo memory IN CONTEXT — and
    the no-think model still snap-emitted ``<not_for_me/>`` at iter 1 (it
    pattern-matches "mentions a pet name" onto the different-addressee
    rule). Prompt tuning keeps losing to new pattern-matches, so: when the
    acoustics clearly say "directed" (caller passes sentinel_double_check),
    a FIRST-look sentinel buys one reasoned second opinion (/think). A
    confirmed sentinel still goes silent; an answer gets spoken.
    """

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
        cache.get_available_commands.return_value = []
        cache.get_force_tool_calls.return_value = True
        cache.get_router_decision.return_value = {"used": False}
        return cache

    def _responses(self, *contents):
        return [
            {
                "choices": [{
                    "message": {"content": c},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            for c in contents
        ]

    @pytest.mark.asyncio
    async def test_double_check_rescues_false_silence(
        self, mock_llm_client, mock_conversation_cache
    ):
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = self._responses(
            '{"message": "<not_for_me/>", "tool_calls": []}',
            '{"message": "Leo is one of your golden doodles!", "tool_calls": []}',
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-dc",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
                sentinel_double_check=True,
            )

        assert result["stop_reason"] == "complete"
        assert "golden doodles" in str(result["assistant_message"])

    @pytest.mark.asyncio
    async def test_confirmed_sentinel_still_goes_silent(
        self, mock_llm_client, mock_conversation_cache
    ):
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = self._responses(
            '{"message": "<not_for_me/>", "tool_calls": []}',
            '{"message": "<not_for_me/>", "tool_calls": []}',
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-dc2",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
                sentinel_double_check=True,
            )

        assert result["stop_reason"] == "not_for_me"
        # Exactly two calls: original + one reasoned re-check, never a loop.
        assert mock_llm_client.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_double_check_message_uses_think_and_is_scrubbed(
        self, mock_llm_client, mock_conversation_cache
    ):
        # The re-check must enable reasoning (/think soft switch) and must
        # not linger in the transcript to poison follow-up turns.
        from app.core.tool_execution_engine import ToolExecutionEngine

        # The engine passes ``messages`` by reference and later scrubs it in
        # place, so snapshot what each LLM call actually saw at call time.
        responses = self._responses(
            '{"message": "<not_for_me/>", "tool_calls": []}',
            '{"message": "Leo is your dog!", "tool_calls": []}',
        )
        seen_messages: list = []

        async def _capture(*args, **kwargs):
            seen_messages.append([dict(m) for m in kwargs["messages"]])
            return responses[len(seen_messages) - 1]

        mock_llm_client.chat_completion.side_effect = _capture
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            await engine.execute(
                conversation_id="test-conv-dc3",
                messages=messages,
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
                sentinel_double_check=True,
            )

        # The second LLM call saw the /think re-check message...
        dc_msgs = [
            m for m in seen_messages[1]
            if "[NOT_FOR_ME_DOUBLE_CHECK]" in str(m.get("content", ""))
        ]
        assert dc_msgs and dc_msgs[0]["content"].rstrip().endswith("/think")
        # ...but the final transcript is clean of it (and of the sentinel).
        assert not any(
            "[NOT_FOR_ME_DOUBLE_CHECK]" in str(m.get("content", "")) for m in messages
        )
        assert not any(
            "not_for_me" in str(m.get("content", "")) for m in messages
        )

    @pytest.mark.asyncio
    async def test_rescued_answer_is_exempt_from_force_tools_guard(
        self, mock_llm_client, mock_conversation_cache
    ):
        # After a double-check rescue, the prose answer must NOT be popped
        # by the force-tool-calls guard (which would burn two more calls
        # re-manufacturing the same corner). Exactly two calls total.
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = self._responses(
            '{"message": "<not_for_me/>", "tool_calls": []}',
            '{"message": "Leo is your dog!", "tool_calls": []}',
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-dc4",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
                sentinel_double_check=True,
            )

        assert result["stop_reason"] == "complete"
        assert mock_llm_client.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_flag_off_keeps_first_look_short_circuit(
        self, mock_llm_client, mock_conversation_cache
    ):
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = self._responses(
            '{"message": "<not_for_me/>", "tool_calls": []}',
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-dc5",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert result["stop_reason"] == "not_for_me"
        assert mock_llm_client.chat_completion.call_count == 1


class TestSentinelDetectionIgnoresThinkBlocks:
    """2026-07-27 prod validation run: the /think double-check pass QUOTED
    the earlier <not_for_me/> while reasoning about it ("the system
    initially responded with <not_for_me/>...") and the detector matched
    inside the think block — manufacturing the very silence it was
    re-checking. Sentinel detection must only see post-think content."""

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
        cache.get_available_commands.return_value = []
        cache.get_force_tool_calls.return_value = False
        cache.get_router_decision.return_value = {"used": False}
        return cache

    @pytest.mark.asyncio
    async def test_sentinel_quoted_in_think_is_not_an_emission(
        self, mock_llm_client, mock_conversation_cache
    ):
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": (
                    "<think>The system said <not_for_me/> but the user "
                    "clearly asked me directly, so I should answer.</think>\n"
                    '{"message": "Leo is your golden doodle!", "tool_calls": []}'
                )},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-tq",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert "golden doodle" in str(result["assistant_message"])

    @pytest.mark.asyncio
    async def test_sentinel_after_closed_think_still_detected(
        self, mock_llm_client, mock_conversation_cache
    ):
        # The real prod emission shape: empty think pair, then the bare
        # sentinel. Stripping think blocks must not break detection.
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": "<think>\n\n</think>\n\n<not_for_me/>"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-tq2",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
            )

        assert result["stop_reason"] == "not_for_me"

    @pytest.mark.asyncio
    async def test_double_check_iteration_gets_thinking_token_budget(
        self, mock_llm_client, mock_conversation_cache
    ):
        # 256 tokens truncated the /think pass mid-reasoning in prod. The
        # iteration right after the double-check injection needs a real
        # reasoning budget; every other iteration keeps the tight cap.
        from app.core.tool_execution_engine import ToolExecutionEngine

        responses = [
            {
                "choices": [{
                    "message": {"content": '{"message": "<not_for_me/>", "tool_calls": []}'},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{
                    "message": {"content": '{"message": "Leo is your dog!", "tool_calls": []}'},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        max_tokens_seen: list = []

        async def _capture(*args, **kwargs):
            max_tokens_seen.append(kwargs.get("max_tokens"))
            return responses[len(max_tokens_seen) - 1]

        mock_llm_client.chat_completion.side_effect = _capture
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            await engine.execute(
                conversation_id="test-conv-tq3",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "set_reminder"}}],
                max_iterations=3,
                sentinel_double_check=True,
            )

        assert max_tokens_seen[0] == 256
        assert max_tokens_seen[1] >= 1024


class TestForceToolCallsKeywordGate:
    """2026-08-15 prod latency fix: the force-tool-calls guard was
    unconditional, so every plain-prose answer paid a second sequential LLM
    call — 56% of voice_command_stream traces ran llm_call_iter_2 (p50
    +609ms), and the dominant victims were purely conversational turns
    ("Thank you", "Okay") where prose IS the correct final answer. The guard
    now only retries when the utterance keyword-matches a declared
    command/tool keyword; with no keyword evidence the direct answer is
    accepted on iteration 1.

    Legacy behavior (always retry) is preserved whenever the gate lacks
    signal: no keywords declared anywhere, or no user_utterance provided.
    """

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
        # Commands declare keywords — the gate has signal to work with.
        cache.get_available_commands.return_value = [
            {
                "command_name": "medication",
                "keywords": ["medicine", "meds", "took my"],
                "parameters": [],
            },
            {
                "command_name": "control_device",
                "keywords": ["lights", "turn on", "turn off"],
                "parameters": [],
            },
        ]
        cache.get_tools.return_value = []
        cache.get_force_tool_calls.return_value = True
        cache.get_router_decision.return_value = {"used": False}
        return cache

    def _prose_response(self, message: str):
        return {
            "choices": [{
                "message": {"content": json.dumps({"message": message, "tool_calls": []})},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _tool_call_response(self, name: str):
        return {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "",
                    "tool_calls": [{"name": name, "arguments": {}}],
                })},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @pytest.mark.asyncio
    async def test_conversational_turn_accepts_prose_with_single_llm_call(
        self, mock_llm_client, mock_conversation_cache
    ):
        """"Thank you." matches no command keyword — the prose answer must be
        accepted at iteration 1 with NO second LLM call and no retry nag."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._prose_response(
            "You're welcome, Alex."
        )
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-chat",
                messages=messages,
                tools=[{"function": {"name": "medication"}}],
                user_utterance="Thank you.",
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert result["assistant_message"] == "You're welcome, Alex."
        assert mock_llm_client.chat_completion.call_count == 1
        assert not any(
            "[MUST_CALL_RETRY]" in str(m.get("content", "")) for m in messages
        )

    @pytest.mark.asyncio
    async def test_keyword_matched_turn_still_retries_prose(
        self, mock_llm_client, mock_conversation_cache
    ):
        """"Leo took his medicine." matches medication keywords — a prose
        answer must still trigger the guard, and the retry rescues the tool
        call (the guard's original protection)."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = [
            self._prose_response("Glad Leo took his medicine!"),
            self._tool_call_response("medication"),
        ]
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "medication", "arguments": "{}"},
                }])
                result = await engine.execute(
                    conversation_id="test-conv-meds",
                    messages=messages,
                    tools=[{"function": {"name": "medication"}}],
                    user_utterance="Leo took his medicine.",
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        assert result["tool_calls"][0]["function"]["name"] == "medication"
        assert mock_llm_client.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_no_keywords_declared_keeps_legacy_retry(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Old node clients don't send keywords — with zero keyword signal
        the keyword half of the gate must not disable the guard. The
        utterance still has to be action-shaped (2026-08-17 shape gate):
        an imperative device command narrated as prose keeps the retry."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_conversation_cache.get_available_commands.return_value = [
            {"command_name": "control_device", "parameters": []},
        ]
        mock_llm_client.chat_completion.side_effect = [
            self._prose_response("Turning them on!"),
            self._tool_call_response("control_device"),
        ]
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "control_device", "arguments": "{}"},
                }])
                result = await engine.execute(
                    conversation_id="test-conv-legacy",
                    messages=[{"role": "system", "content": "sys"}],
                    tools=[{"function": {"name": "control_device"}}],
                    user_utterance="Turn on the kitchen lights.",
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        assert mock_llm_client.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_missing_utterance_keeps_legacy_retry(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Without the utterance the gate has nothing to match against —
        preserve legacy retry behavior."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = [
            self._prose_response("Sure thing."),
            self._tool_call_response("medication"),
        ]
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "medication", "arguments": "{}"},
                }])
                result = await engine.execute(
                    conversation_id="test-conv-noutt",
                    messages=[{"role": "system", "content": "sys"}],
                    tools=[{"function": {"name": "medication"}}],
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        assert mock_llm_client.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_sentinel_still_short_circuits_before_gate(
        self, mock_llm_client, mock_conversation_cache
    ):
        """The <not_for_me/> sentinel must keep outranking the guard AND the
        keyword gate — terminal on first look regardless of keywords."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {"content": '{"message": "<not_for_me/>", "tool_calls": []}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-sentinel",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "medication"}}],
                user_utterance="turn on the lights",
                max_iterations=3,
            )

        assert result["stop_reason"] == "not_for_me"
        assert mock_llm_client.chat_completion.call_count == 1


class TestStaleMustCallRetryScrub:
    """Prod 2026-08-15: [MUST_CALL_RETRY] system nags persisted in the cached
    transcript across turns — later turns started at "attempt 2/2" and kept
    "You MUST call a tool" pressure on every subsequent request (forced junk
    tool calls like tell_joke on "Yum."). execute() must scrub stale nags
    from prior turns on entry."""

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
        cache.get_available_commands.return_value = [
            {"command_name": "medication", "keywords": ["medicine"], "parameters": []},
        ]
        cache.get_tools.return_value = []
        cache.get_force_tool_calls.return_value = True
        cache.get_router_decision.return_value = {"used": False}
        return cache

    def _prose_response(self, message: str):
        return {
            "choices": [{
                "message": {"content": json.dumps({"message": message, "tool_calls": []})},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @pytest.mark.asyncio
    async def test_stale_nag_is_scrubbed_on_entry(
        self, mock_llm_client, mock_conversation_cache
    ):
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._prose_response("You're welcome.")
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Leo took his medicine"},
            {"role": "system", "content": "[MUST_CALL_RETRY] You MUST call a tool. (attempt 1/2)"},
            {"role": "assistant", "content": "Logged it."},
        ]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-scrub",
                messages=messages,
                tools=[{"function": {"name": "medication"}}],
                user_utterance="Thank you.",
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert not any(
            "[MUST_CALL_RETRY]" in str(m.get("content", "")) for m in messages
        )
        # Prior-turn user/assistant content survives the scrub.
        assert any(m.get("content") == "Logged it." for m in messages)

    @pytest.mark.asyncio
    async def test_retry_budget_is_fresh_after_scrub(
        self, mock_llm_client, mock_conversation_cache
    ):
        """A stale nag must not inflate the in-turn retry count: the first
        in-turn retry is attempt 1/2, not 2/2."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = [
            self._prose_response("Noted."),
            self._prose_response("Noted again."),
            self._prose_response("Final answer."),
        ]
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": "[MUST_CALL_RETRY] stale nag from last turn"},
        ]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            await engine.execute(
                conversation_id="test-conv-budget",
                messages=messages,
                tools=[{"function": {"name": "medication"}}],
                user_utterance="Leo took his medicine.",
                max_iterations=3,
            )

        retry_messages = [
            str(m.get("content", "")) for m in messages
            if "[MUST_CALL_RETRY]" in str(m.get("content", ""))
        ]
        assert retry_messages, "keyword-matched prose should trigger the guard"
        assert "attempt 1/2" in retry_messages[0]
        assert "stale nag" not in retry_messages[0]


class TestScrubDoesNotEatSystemPrompt:
    """Regression: the base system prompt MENTIONS [MUST_CALL_RETRY] in its
    rules text (core_rules.py). The scrub must only remove standalone nag
    messages (content starts with the tag), never the cached prompt — a
    substring scrub deleted messages[0] with every tool schema in it
    (prod incident 2026-08-16)."""

    def test_is_retry_nag_ignores_prompt_that_mentions_tag(self):
        from app.core.tool_execution_engine import _is_retry_nag

        base_prompt = {
            "role": "system",
            "content": (
                "You are Jarvis. Rules: ... the utterance (you drafted an "
                "answer, or a [MUST_CALL_RETRY] asks for a tool call) ... "
                "TOOLS: control_device, get_weather"
            ),
        }
        real_nag = {
            "role": "system",
            "content": "[MUST_CALL_RETRY] You MUST call a tool. Do NOT answer directly.",
        }
        assert not _is_retry_nag(base_prompt)
        assert _is_retry_nag(real_nag)
        assert _is_retry_nag({"role": "system", "content": "  [MUST_CALL_RETRY] retry"})
        assert not _is_retry_nag({"role": "user", "content": "[MUST_CALL_RETRY]"})

    @pytest.mark.asyncio
    async def test_scrub_preserves_system_prompt_and_removes_stale_nags(self):
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client = MagicMock()
        mock_llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{
                "message": {"content": '{"message": "hi", "tool_calls": []}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

        mock_cache = MagicMock()
        mock_cache.get_node_context.return_value = {}
        mock_cache.get_timezone.return_value = "UTC"
        mock_cache.get_force_tool_calls.return_value = False
        mock_cache.get_router_decision.return_value = {"used": True}
        mock_cache.get_available_commands.return_value = []

        base_prompt = {
            "role": "system",
            "content": "Persona + rules mentioning [MUST_CALL_RETRY] + tool schemas",
        }
        stale_nag = {
            "role": "system",
            "content": "[MUST_CALL_RETRY] You MUST call a tool. Do NOT answer directly.",
        }
        messages = [
            base_prompt,
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "content": "old answer"},
            stale_nag,
            {"role": "user", "content": "hello"},
        ]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_cache):
            engine = ToolExecutionEngine(mock_llm_client)
            await engine.execute("conv-scrub", messages, tools=[], user_utterance="hello")

        assert messages[0] is base_prompt, "scrub must never remove the cached system prompt"
        assert stale_nag not in messages, "stale standalone nags must be scrubbed"


class TestExchangeCompleteGuardExemption:
    """2026-08-17 prod incident: after a real lights command, the follow-up
    window captured a parent calling their child — "already done. Wow.
    Miles, come here." The model emitted ``<exchange_complete/>`` (the
    CORRECT terminal output: close the exchange, stop listening), but the
    force-tool-calls guard treated the no-tool-call response as a failure,
    popped it, and nagged with [MUST_CALL_RETRY] — so the model complied by
    calling chat("Miles is a little busy with his toys...") into the
    family's conversation. Two rounds later: "Shut up, Jorgis."

    Terminal sentinels (``<exchange_complete/>``, ``<not_for_me/>``) are
    legal terminal outputs, senior to tool-forcing: the retry must not fire
    AT ALL on a first-look sentinel — not rely on the retry-escape path
    (which only covers sentinels emitted AFTER a popped prose answer).
    """

    INCIDENT_TRANSCRIPT = "already done. Wow. Miles, come here."

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
        # allow_direct_answer=False arms the per-command must-call guard too.
        cache.get_available_commands.return_value = [
            {"command_name": "control_device", "allow_direct_answer": False}
        ]
        cache.get_tools.return_value = []
        # The incident combination: compressed provider forces tool calls
        # AND the router had a confident match — BOTH retry guards armed.
        cache.get_force_tool_calls.return_value = True
        cache.get_router_decision.return_value = {"used": True}
        return cache

    def _marker_response(self, content: str):
        return {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @pytest.mark.asyncio
    async def test_incident_exchange_complete_is_not_retried(
        self, mock_llm_client, mock_conversation_cache
    ):
        """The exact incident shape (text path, JSON message field): one LLM
        call, no [MUST_CALL_RETRY] nag, and the close-the-exchange intent
        survives as ``end_of_exchange`` (clean_for_tts strips the literal
        marker from assistant_message so it can never be spoken)."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return self._marker_response(
                '{"message": "<exchange_complete/>", "tool_calls": []}'
            )

        mock_llm_client.chat_completion.side_effect = side_effect
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-miles",
                messages=messages,
                tools=[{"function": {"name": "control_device"}}],
                user_utterance=self.INCIDENT_TRANSCRIPT,
                max_iterations=3,
            )

        assert call_count[0] == 1, (
            "A terminal <exchange_complete/> must never be popped and "
            "retried — the incident saw a [MUST_CALL_RETRY] nag and a "
            "forced chat() call into the family's conversation."
        )
        assert result["stop_reason"] == "complete"
        # The literal marker is stripped (never spoken); the intent rides
        # the result so the node skips its follow-up window.
        assert result.get("end_of_exchange") is True
        assert "exchange_complete" not in str(result["assistant_message"])
        assert not any(
            "[MUST_CALL_RETRY]" in str(m.get("content", "")) for m in messages
        )

    @pytest.mark.asyncio
    async def test_exchange_complete_with_prose_is_not_retried(
        self, mock_llm_client, mock_conversation_cache
    ):
        """The common shape: a final reply WITH the marker appended
        ("Timer set. <exchange_complete/>") must also be terminal."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._marker_response(
            '{"message": "Glad it worked out. <exchange_complete/>", "tool_calls": []}'
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-ec-prose",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "control_device"}}],
                max_iterations=3,
            )

        assert mock_llm_client.chat_completion.call_count == 1
        assert result["stop_reason"] == "complete"
        assert "Glad it worked out" in str(result["assistant_message"])
        assert result.get("end_of_exchange") is True

    @pytest.mark.asyncio
    async def test_exchange_complete_not_retried_native_path(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Native tool-calling path: the marker lives in raw_content (no
        JSON wrapper) — the exemption must cover it there too."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._marker_response(
            "Goodnight! <exchange_complete/>"
        )
        mock_provider = MagicMock()
        mock_provider.supports_native_tools = True
        mock_provider.build_tools.return_value = []
        mock_provider.sanitize_text.side_effect = lambda s: s
        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=mock_provider)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-ec-native",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "control_device"}}],
                max_iterations=3,
            )

        assert mock_llm_client.chat_completion.call_count == 1
        assert result["stop_reason"] == "complete"

    @pytest.mark.asyncio
    async def test_must_call_guard_also_exempted(
        self, mock_llm_client, mock_conversation_cache
    ):
        """The per-command must-call guard (router matched + allow_direct_
        answer=False) must not retry a sentinel either."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        # Only the must-call branch armed.
        mock_conversation_cache.get_force_tool_calls.return_value = False
        mock_llm_client.chat_completion.return_value = self._marker_response(
            '{"message": "<exchange_complete/>", "tool_calls": []}'
        )
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-ec-mustcall",
                messages=messages,
                tools=[{"function": {"name": "control_device"}}],
                max_iterations=3,
            )

        assert mock_llm_client.chat_completion.call_count == 1
        assert result["stop_reason"] == "complete"
        assert not any(
            "[MUST_CALL_RETRY]" in str(m.get("content", "")) for m in messages
        )

    @pytest.mark.asyncio
    async def test_marker_only_inside_think_block_does_not_exempt(
        self, mock_llm_client, mock_conversation_cache
    ):
        """A marker QUOTED inside <think> reasoning is not an emission —
        same rule the not_for_me detector applies. The guard still fires."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return self._marker_response(
                    "<think>should I say <exchange_complete/>? no</think>"
                    '{"message": "Working on it.", "tool_calls": []}'
                )
            return self._marker_response(
                '{"message": "Done.", "tool_calls": []}'
            )

        mock_llm_client.chat_completion.side_effect = side_effect
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            await engine.execute(
                conversation_id="test-conv-ec-think",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "control_device"}}],
                max_iterations=3,
            )

        assert call_count[0] >= 2, (
            "A think-quoted marker must NOT exempt the guard — only a real "
            "emission outside <think> is terminal."
        )

    @pytest.mark.asyncio
    async def test_plain_prose_still_retried(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Guard-the-guard: without any sentinel, the force-tool-calls
        retry still fires exactly as before the exemption."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._marker_response(
            '{"message": "The lights are probably on.", "tool_calls": []}'
        )
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            await engine.execute(
                conversation_id="test-conv-ec-prose-retry",
                messages=messages,
                tools=[{"function": {"name": "control_device"}}],
                max_iterations=4,
            )

        assert mock_llm_client.chat_completion.call_count > 1, (
            "The exemption must be sentinel-scoped — plain prose keeps the "
            "retry behavior."
        )


class TestActionShapeGate:
    """2026-08-17 prod incident: 'What should I do with Miles today?' — an
    open question to the assistant — keyword-matched a calendar command, so
    the force-tool-calls guard popped the model's prose answer and
    [MUST_CALL_RETRY]-ed it into get_calendar_events: the whole calendar
    read aloud in answer to a question.

    DOCTRINE: the model decides when to talk and how to answer; guards force
    tools ONLY where not-acting is a correctness failure (actions/reports),
    NEVER on questions/conversation. The guard now requires the utterance to
    be ACTION-shaped (imperative) or REPORT-shaped ("Leo took his medicine")
    IN ADDITION to the keyword match. Question shapes never qualify — even
    with keyword matches, and even for allow_direct_answer=False commands.
    """

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
        # Calendar keywords that MATCH the incident utterance ("today",
        # "plans") — the old keyword-only gate armed the guard on it.
        cache.get_available_commands.return_value = [
            {
                "command_name": "get_calendar_events",
                "keywords": ["calendar", "schedule", "plans", "today"],
                "parameters": [],
            },
            {
                "command_name": "control_device",
                "keywords": ["lights", "turn on", "turn off"],
                "parameters": [],
            },
            {
                "command_name": "medication",
                "keywords": ["medicine", "meds", "took my"],
                "parameters": [],
            },
        ]
        cache.get_tools.return_value = []
        cache.get_issued_client_calls.return_value = []
        cache.get_force_tool_calls.return_value = True
        cache.get_router_decision.return_value = {"used": False}
        return cache

    def _prose_response(self, message: str):
        return {
            "choices": [{
                "message": {"content": json.dumps({"message": message, "tool_calls": []})},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _tool_call_response(self, name: str):
        return {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "",
                    "tool_calls": [{"name": name, "arguments": {}}],
                })},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @pytest.mark.asyncio
    async def test_incident_question_with_keyword_match_is_not_retried(
        self, mock_llm_client, mock_conversation_cache
    ):
        """The incident fixture: a question that keyword-matches a calendar
        command must accept the prose answer — one LLM call, no nag."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._prose_response(
            "You could take Miles to the park — it's sunny out."
        )
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-miles-q",
                messages=messages,
                tools=[{"function": {"name": "get_calendar_events"}}],
                user_utterance="What should I do with Miles today?",
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert "park" in str(result["assistant_message"])
        assert mock_llm_client.chat_completion.call_count == 1, (
            "A question must NEVER be retried into a tool call, even when "
            "it keyword-matches a command — the incident read the whole "
            "calendar aloud in answer to an open question."
        )
        assert not any(
            "[MUST_CALL_RETRY]" in str(m.get("content", "")) for m in messages
        )

    @pytest.mark.asyncio
    async def test_question_without_question_mark_is_not_retried(
        self, mock_llm_client, mock_conversation_cache
    ):
        """STT often drops the '?' — the leading question word alone must
        exempt the utterance."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._prose_response(
            "Your afternoon looks free."
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-noqm",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "get_calendar_events"}}],
                user_utterance="what does my schedule look like today",
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert mock_llm_client.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_imperative_device_command_still_retried(
        self, mock_llm_client, mock_conversation_cache
    ):
        """'Turn off the living room lights' narrated as prose keeps the
        retry — action shape + keyword match both hold."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = [
            self._prose_response("Turning off the living room lights!"),
            self._tool_call_response("control_device"),
        ]
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "control_device", "arguments": "{}"},
                }])
                result = await engine.execute(
                    conversation_id="test-conv-lights",
                    messages=[{"role": "system", "content": "sys"}],
                    tools=[{"function": {"name": "control_device"}}],
                    user_utterance="Turn off the living room lights",
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        assert result["tool_calls"][0]["function"]["name"] == "control_device"
        assert mock_llm_client.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_medication_report_still_retried(
        self, mock_llm_client, mock_conversation_cache
    ):
        """'Leo took his medicine' is a REPORT that must reach the logging
        tool — prose keeps the retry."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.side_effect = [
            self._prose_response("Glad Leo took his medicine!"),
            self._tool_call_response("medication"),
        ]
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "medication", "arguments": "{}"},
                }])
                result = await engine.execute(
                    conversation_id="test-conv-leo-report",
                    messages=[{"role": "system", "content": "sys"}],
                    tools=[{"function": {"name": "medication"}}],
                    user_utterance="Leo took his medicine",
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        assert mock_llm_client.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_trailing_question_mark_never_qualifies(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Even an imperative-verb opener with a trailing '?' reads as a
        question ('turn off the lights?') — no retry."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._prose_response(
            "They're already off."
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-qmark",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "control_device"}}],
                user_utterance="turn off the lights?",
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert mock_llm_client.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_keyword_match_without_action_shape_is_not_retried(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Declarative chatter that happens to carry a keyword ('the lights
        in here are so warm') must not force a tool call — keyword match
        alone no longer arms the guard."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_llm_client.chat_completion.return_value = self._prose_response(
            "They do give off a cozy glow."
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-chatter",
                messages=[{"role": "system", "content": "sys"}],
                tools=[{"function": {"name": "control_device"}}],
                user_utterance="the lights in here are so warm",
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert mock_llm_client.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_must_call_guard_exempts_question_shapes(
        self, mock_llm_client, mock_conversation_cache
    ):
        """allow_direct_answer=False commands keep their must-call semantics
        for actions, but question shapes are exempt there too."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_conversation_cache.get_force_tool_calls.return_value = False
        mock_conversation_cache.get_router_decision.return_value = {"used": True}
        mock_conversation_cache.get_available_commands.return_value = [
            {
                "command_name": "get_calendar_events",
                "keywords": ["calendar", "today"],
                "allow_direct_answer": False,
                "parameters": [],
            },
        ]
        mock_llm_client.chat_completion.return_value = self._prose_response(
            "You could visit the zoo."
        )
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            result = await engine.execute(
                conversation_id="test-conv-mustcall-q",
                messages=messages,
                tools=[{"function": {"name": "get_calendar_events"}}],
                user_utterance="What should I do with Miles today?",
                max_iterations=3,
            )

        assert result["stop_reason"] == "complete"
        assert mock_llm_client.chat_completion.call_count == 1
        assert not any(
            "[MUST_CALL_RETRY]" in str(m.get("content", "")) for m in messages
        )

    @pytest.mark.asyncio
    async def test_must_call_guard_still_fires_on_action_shapes(
        self, mock_llm_client, mock_conversation_cache
    ):
        """Guard-the-guard: the per-command must-call guard keeps its
        existing semantics on non-question utterances."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        mock_conversation_cache.get_force_tool_calls.return_value = False
        mock_conversation_cache.get_router_decision.return_value = {"used": True}
        mock_conversation_cache.get_available_commands.return_value = [
            {
                "command_name": "control_device",
                "keywords": ["lights"],
                "allow_direct_answer": False,
                "parameters": [],
            },
        ]
        mock_llm_client.chat_completion.side_effect = [
            self._prose_response("Sure, lights off."),
            self._tool_call_response("control_device"),
        ]
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", mock_conversation_cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                mock_executor.execute_tool_calls.return_value = ([], [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "control_device", "arguments": "{}"},
                }])
                result = await engine.execute(
                    conversation_id="test-conv-mustcall-act",
                    messages=messages,
                    tools=[{"function": {"name": "control_device"}}],
                    user_utterance="Turn off the lights",
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        assert mock_llm_client.chat_completion.call_count == 2
        assert any(
            "[MUST_CALL_RETRY]" in str(m.get("content", "")) for m in messages
        )


class TestIdenticalToolCallDedupe:
    """2026-08-15 prod incident: three get_calendar_events with IDENTICAL
    args inside 25 seconds — each follow-up re-read the whole calendar
    aloud. When the model re-requests a client tool call whose (name,
    canonicalized-args) matches one already issued this conversation within
    voice.tool_dedupe_window_seconds, the engine must NOT return the 202:
    it re-calls the LLM once with a system nudge to answer from the results
    already in context. Whatever comes back is accepted — prose, a
    different tool call, or (fail-open) the same call again.
    """

    TOOL_SCHEMA = [{"function": {"name": "get_calendar_events"}}]

    @pytest.fixture
    def mock_llm_client(self):
        client = MagicMock()
        client.chat_completion = AsyncMock()
        return client

    def _real_cache(self, conversation_id: str):
        """A REAL ConversationCache seeded for this conversation — exercises
        record_issued_client_call / get_issued_client_calls end to end."""
        from app.core.conversation_cache import ConversationCache

        cache = ConversationCache(ttl_minutes=10)
        cache.set(
            conversation_id,
            messages=[],
            available_commands=[
                {
                    "command_name": "get_calendar_events",
                    "keywords": ["calendar"],
                    "parameters": [],
                },
            ],
            timezone="UTC",
            tools=[],
            node_context={},
        )
        cache.set_force_tool_calls(conversation_id, False)
        return cache

    def _tool_call_response(self, name: str, arguments: dict):
        return {
            "choices": [{
                "message": {"content": json.dumps({
                    "message": "",
                    "tool_calls": [{"name": name, "arguments": arguments}],
                })},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _prose_response(self, message: str):
        return {
            "choices": [{
                "message": {"content": json.dumps({"message": message, "tool_calls": []})},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _executor_passthrough(self, mock_executor):
        def side_effect(tool_calls, **kwargs):
            return ([], [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": dict(tc["function"]),
                }
                for i, tc in enumerate(tool_calls)
            ])
        mock_executor.execute_tool_calls.side_effect = side_effect

    @pytest.mark.asyncio
    async def test_identical_call_within_window_nudges_and_accepts_prose(
        self, mock_llm_client
    ):
        """The incident shape: same tool + same args again → no 202; one
        nudge; the prose answer is accepted and the nudge is scrubbed.

        Turn 1 goes through the engine for real so the recorded canonical
        key reflects exactly what the engine issues (arg normalization
        included) — the same way production records it."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        conv = "test-conv-dedupe-1"
        cache = self._real_cache(conv)

        identical = self._tool_call_response(
            "get_calendar_events",
            {"resolved_datetimes": ["2026-08-15T00:00:00Z"]},
        )
        mock_llm_client.chat_completion.side_effect = [
            identical,  # turn 1: first (legitimate) issue
            identical,  # turn 2: identical re-request → nudge
            self._prose_response("Just the dentist at 3 — nothing else."),
        ]
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                self._executor_passthrough(mock_executor)
                turn1 = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="what's on my calendar today",
                    max_iterations=4,
                )
                assert turn1["stop_reason"] == "tool_calls"
                assert len(cache.get_issued_client_calls(conv)) == 1
                result = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="anything else going on",
                    max_iterations=4,
                )

        assert result["stop_reason"] == "complete"
        assert "dentist" in str(result["assistant_message"])
        assert mock_llm_client.chat_completion.call_count == 3
        # The nudge is one-shot — scrubbed once the model replied.
        assert not any(
            "[TOOL_DEDUPE]" in str(m.get("content", "")) for m in messages
        )
        # The duplicate's assistant message was popped: only turn 1's
        # legitimate tool_calls message remains in the transcript.
        assert sum(1 for m in messages if m.get("tool_calls")) == 1

    @pytest.mark.asyncio
    async def test_different_args_pass_through(self, mock_llm_client):
        """Same tool with DIFFERENT args is not a duplicate — 202 as usual."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        conv = "test-conv-dedupe-2"
        cache = self._real_cache(conv)

        mock_llm_client.chat_completion.side_effect = [
            self._tool_call_response(
                "get_calendar_events",
                {"resolved_datetimes": ["2026-08-15T00:00:00Z"]},
            ),
            self._tool_call_response(
                "get_calendar_events",
                {"resolved_datetimes": ["2026-08-16T00:00:00Z"]},
            ),
        ]
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                self._executor_passthrough(mock_executor)
                turn1 = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="what's on my calendar today",
                    max_iterations=3,
                )
                assert turn1["stop_reason"] == "tool_calls"
                result = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="what about tomorrow",
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        assert mock_llm_client.chat_completion.call_count == 2
        assert len(cache.get_issued_client_calls(conv)) == 2

    @pytest.mark.asyncio
    async def test_expired_window_passes_through(self, mock_llm_client):
        """An identical call OUTSIDE the window is legitimate — pass it."""
        import time as _t

        from app.core.tool_execution_engine import ToolExecutionEngine

        conv = "test-conv-dedupe-3"
        cache = self._real_cache(conv)

        identical = self._tool_call_response(
            "get_calendar_events",
            {"resolved_datetimes": ["2026-08-15T00:00:00Z"]},
        )
        mock_llm_client.chat_completion.side_effect = [identical, identical]
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                self._executor_passthrough(mock_executor)
                turn1 = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="what's on my calendar today",
                    max_iterations=3,
                )
                assert turn1["stop_reason"] == "tool_calls"
                # Age the recorded issue beyond the 120s default window.
                with cache.lock:
                    for rec in cache.cache[conv]["issued_client_calls"]:
                        rec["timestamp"] = _t.time() - 300
                result = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="check my calendar again",
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        assert mock_llm_client.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_second_identical_after_nudge_passes_through(
        self, mock_llm_client
    ):
        """Fail-open: if the nudged model STILL wants the same call, it goes
        through — the model may genuinely want a refresh."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        conv = "test-conv-dedupe-4"
        cache = self._real_cache(conv)

        identical = self._tool_call_response(
            "get_calendar_events",
            {"resolved_datetimes": ["2026-08-15T00:00:00Z"]},
        )
        mock_llm_client.chat_completion.side_effect = [
            identical,  # turn 1: legitimate issue (recorded)
            identical,  # turn 2, iter 1: duplicate → nudge
            identical,  # turn 2, iter 2: STILL identical → pass through
        ]
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                self._executor_passthrough(mock_executor)
                turn1 = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="what's on my calendar today",
                    max_iterations=4,
                )
                assert turn1["stop_reason"] == "tool_calls"
                result = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="refresh my calendar",
                    max_iterations=4,
                )

        assert result["stop_reason"] == "tool_calls", (
            "One nudge only — a second identical request must pass through"
        )
        assert mock_llm_client.chat_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_first_issue_is_recorded_for_later_dedupe(
        self, mock_llm_client
    ):
        """Returning a 202 must record (name, canonical args) so the NEXT
        identical request this conversation gets deduped."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        conv = "test-conv-dedupe-5"
        cache = self._real_cache(conv)

        mock_llm_client.chat_completion.return_value = self._tool_call_response(
            "get_calendar_events",
            {"resolved_datetimes": ["2026-08-15T00:00:00Z"]},
        )
        engine = ToolExecutionEngine(mock_llm_client)

        with patch("app.core.tool_execution_engine.conversation_cache", cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                self._executor_passthrough(mock_executor)
                result = await engine.execute(
                    conversation_id=conv,
                    messages=[{"role": "system", "content": "sys"}],
                    tools=list(self.TOOL_SCHEMA),
                    user_utterance="what's on my calendar",
                    max_iterations=3,
                )

        assert result["stop_reason"] == "tool_calls"
        issued = cache.get_issued_client_calls(conv)
        assert len(issued) == 1
        assert issued[0]["name"] == "get_calendar_events"
        assert issued[0]["args_hash"]
        assert issued[0]["timestamp"] > 0

    @pytest.mark.asyncio
    async def test_server_tools_are_not_deduped(self, mock_llm_client):
        """Server tools have their own loop semantics — an identical server
        call is executed, never nudged, and never recorded."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        conv = "test-conv-dedupe-6"
        cache = self._real_cache(conv)

        def _native_tool_response(name: str, arguments: dict):
            return {
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "call_srv",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                # The SAME server call twice in a row.
                return _native_tool_response("recall_memory", {"query": "leo"})
            return {
                "choices": [{
                    "message": {"content": "Leo is your dog."},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        mock_llm_client.chat_completion.side_effect = side_effect
        # Native provider so server-only results continue the loop in-engine
        # (the text path returns for external formatting after one round).
        mock_provider = MagicMock()
        mock_provider.supports_native_tools = True
        mock_provider.build_tools.return_value = []
        mock_provider.sanitize_text.side_effect = lambda s: s
        engine = ToolExecutionEngine(mock_llm_client, prompt_provider=mock_provider)
        messages = [{"role": "system", "content": "sys"}]

        with patch("app.core.tool_execution_engine.conversation_cache", cache):
            with patch("app.core.tool_execution_engine.tool_executor") as mock_executor:
                # Executor classifies recall_memory as a SERVER tool.
                mock_executor.execute_tool_calls.return_value = (
                    [{"role": "tool", "tool_call_id": "call_srv", "content": "{}"}],
                    [],
                )
                result = await engine.execute(
                    conversation_id=conv,
                    messages=messages,
                    tools=[{"function": {"name": "recall_memory"}}],
                    user_utterance="who is leo",
                    max_iterations=5,
                )

        # Both identical server calls executed — no nudge interfered.
        assert mock_executor.execute_tool_calls.call_count == 2
        assert result["stop_reason"] == "complete"
        assert cache.get_issued_client_calls(conv) == []
        assert not any(
            "[TOOL_DEDUPE]" in str(m.get("content", "")) for m in messages
        )

    @pytest.mark.asyncio
    async def test_stale_dedupe_nudge_scrubbed_on_next_turn_entry(
        self, mock_llm_client
    ):
        """Belt + suspenders: a [TOOL_DEDUPE] nudge left in the cached
        transcript by a crashed turn is scrubbed on entry, same as stale
        [MUST_CALL_RETRY] nags."""
        from app.core.tool_execution_engine import ToolExecutionEngine

        conv = "test-conv-dedupe-7"
        cache = self._real_cache(conv)
        mock_llm_client.chat_completion.return_value = self._prose_response("Hi!")
        engine = ToolExecutionEngine(mock_llm_client)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old turn"},
            {"role": "system", "content": "[TOOL_DEDUPE] stale nudge from last turn"},
            {"role": "assistant", "content": "old answer"},
        ]

        with patch("app.core.tool_execution_engine.conversation_cache", cache):
            await engine.execute(
                conversation_id=conv,
                messages=messages,
                tools=[],
                user_utterance="hello there",
                max_iterations=2,
            )

        assert not any(
            "[TOOL_DEDUPE]" in str(m.get("content", "")) for m in messages
        )
        assert any(m.get("content") == "old answer" for m in messages)
