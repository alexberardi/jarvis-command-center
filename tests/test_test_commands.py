"""Tests for the test command endpoint (app-to-app authenticated).

TDD: These tests define the expected behavior for POST /api/v0/test/command.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock

from app.api.test_commands import CommandTestRequest, CommandTestResult, _run_test_command


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@pytest.fixture
def mock_model_service():
    """Create a mock ModelService."""
    service = AsyncMock()
    service.warmup_conversation_with_tools = AsyncMock()
    service.process_voice_command_with_tools = AsyncMock()
    service.continue_conversation_with_tool_results = AsyncMock()
    service.cleanup_conversation = AsyncMock()
    return service


@pytest.fixture
def sample_request():
    """Sample test command request."""
    return CommandTestRequest(
        voice_command="What's the weather in Miami?",
        available_commands=[
            {
                "command_name": "get_weather",
                "description": "Get weather",
                "parameters": [{"name": "city", "type": "string", "required": False}],
            }
        ],
        client_tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )


class TestToolCallExtraction:
    """Tests for successful tool call extraction."""

    def test_returns_tool_call_result(self, mock_model_service, sample_request):
        """Returns extracted command_name and parameters from tool calls."""
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "tool_calls",
            "assistant_message": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Miami"}',
                    },
                }
            ],
            "validation_request": None,
        }

        result = _run(_run_test_command(sample_request, mock_model_service))

        assert result.stop_reason == "tool_calls"
        assert result.command_name == "get_weather"
        assert result.parameters == {"city": "Miami"}
        assert result.voice_command == "What's the weather in Miami?"

    def test_returns_tool_call_with_dict_arguments(
        self, mock_model_service, sample_request
    ):
        """Handles arguments as dict (not JSON string)."""
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "tool_calls",
            "assistant_message": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Miami"},
                    },
                }
            ],
            "validation_request": None,
        }

        result = _run(_run_test_command(sample_request, mock_model_service))

        assert result.command_name == "get_weather"
        assert result.parameters == {"city": "Miami"}


class TestDirectCompletion:
    """Tests for direct completion (no tool calls)."""

    def test_direct_completion(self, mock_model_service, sample_request):
        """Returns stop_reason='complete' when LLM answers directly."""
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "complete",
            "assistant_message": "The capital of France is Paris.",
            "tool_calls": None,
            "validation_request": None,
        }

        result = _run(_run_test_command(sample_request, mock_model_service))

        assert result.stop_reason == "complete"
        assert result.command_name is None
        assert result.parameters is None
        assert result.assistant_message == "The capital of France is Paris."


class TestValidationLoops:
    """Tests for validation loop handling."""

    def test_handles_validation_loop(self, mock_model_service, sample_request):
        """Auto-selects first option during validation and retries."""
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "validation_required",
            "assistant_message": "Which city?",
            "tool_calls": None,
            "validation_request": {
                "options": ["New York", "Miami"],
                "question": "Which city?",
            },
        }
        mock_model_service.continue_conversation_with_tool_results.return_value = {
            "stop_reason": "tool_calls",
            "assistant_message": None,
            "tool_calls": [
                {
                    "id": "call_456",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "New York"}',
                    },
                }
            ],
            "validation_request": None,
        }

        result = _run(_run_test_command(sample_request, mock_model_service))

        assert result.validation_occurred is True
        assert result.validation_count == 1
        assert result.command_name == "get_weather"

    def test_max_validation_loops(self, mock_model_service, sample_request):
        """Stops after max validation loops."""
        validation_response = {
            "stop_reason": "validation_required",
            "assistant_message": "Which option?",
            "tool_calls": None,
            "validation_request": {
                "options": ["A", "B"],
                "question": "Pick one",
            },
        }
        mock_model_service.process_voice_command_with_tools.return_value = (
            validation_response
        )
        mock_model_service.continue_conversation_with_tool_results.return_value = (
            validation_response
        )

        result = _run(_run_test_command(sample_request, mock_model_service))

        assert result.validation_count <= 3
        assert result.stop_reason == "validation_required"


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_inference_error(self, mock_model_service, sample_request):
        """Returns error in response without raising exception."""
        mock_model_service.process_voice_command_with_tools.side_effect = RuntimeError(
            "Model failed"
        )

        result = _run(_run_test_command(sample_request, mock_model_service))

        assert result.stop_reason == "error"
        assert "Model failed" in result.error

    def test_cleanup_called_on_error(self, mock_model_service, sample_request):
        """Conversation is cleaned up even on error."""
        mock_model_service.process_voice_command_with_tools.side_effect = RuntimeError(
            "boom"
        )

        _run(_run_test_command(sample_request, mock_model_service))

        mock_model_service.cleanup_conversation.assert_called_once()

    def test_cleanup_called_on_success(self, mock_model_service, sample_request):
        """Conversation is cleaned up on success."""
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "complete",
            "assistant_message": "Done",
            "tool_calls": None,
            "validation_request": None,
        }

        _run(_run_test_command(sample_request, mock_model_service))

        mock_model_service.cleanup_conversation.assert_called_once()


class TestWarmupBehavior:
    """Tests for warmup behavior."""

    def test_default_skips_warmup(self, mock_model_service, sample_request):
        """By default, skip_warmup_inference=True."""
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "complete",
            "assistant_message": "ok",
            "tool_calls": None,
            "validation_request": None,
        }

        _run(_run_test_command(sample_request, mock_model_service))

        warmup_call = mock_model_service.warmup_conversation_with_tools.call_args
        assert warmup_call.kwargs["skip_warmup_inference"] is True

    def test_respects_skip_warmup_false(self, mock_model_service):
        """When skip_warmup_inference=False, warmup runs inference."""
        request = CommandTestRequest(
            voice_command="test",
            available_commands=[],
            client_tools=[],
            skip_warmup_inference=False,
        )
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "complete",
            "assistant_message": "ok",
            "tool_calls": None,
            "validation_request": None,
        }

        _run(_run_test_command(request, mock_model_service))

        warmup_call = mock_model_service.warmup_conversation_with_tools.call_args
        assert warmup_call.kwargs["skip_warmup_inference"] is False

    def test_passes_commands_and_tools(self, mock_model_service, sample_request):
        """Warmup receives available_commands and client_tools."""
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "complete",
            "assistant_message": "ok",
            "tool_calls": None,
            "validation_request": None,
        }

        _run(_run_test_command(sample_request, mock_model_service))

        warmup_call = mock_model_service.warmup_conversation_with_tools.call_args
        assert warmup_call.kwargs["client_tools"] == sample_request.client_tools
        assert (
            warmup_call.kwargs["available_commands"] == sample_request.available_commands
        )

    def test_conversation_id_generated(self, mock_model_service, sample_request):
        """A unique conversation ID is generated."""
        mock_model_service.process_voice_command_with_tools.return_value = {
            "stop_reason": "complete",
            "assistant_message": "ok",
            "tool_calls": None,
            "validation_request": None,
        }

        result = _run(_run_test_command(sample_request, mock_model_service))

        assert result.conversation_id is not None
        assert result.conversation_id.startswith("test-")
