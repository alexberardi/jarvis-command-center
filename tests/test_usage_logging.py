"""
Tests for usage logging helpers.

These tests cover the usage and trace logging functionality extracted from model_service.py.
"""
import pytest
import os
import json
import tempfile
from unittest.mock import patch


class TestWriteUsageLog:
    """Tests for writing usage log entries."""

    def test_writes_to_file(self):
        from app.core.usage_logging import write_usage_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            with patch.dict(os.environ, {"JARVIS_LLM_USAGE_LOG_PATH": log_path}):
                usage_totals = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
                write_usage_log(
                    conversation_id="test-conv-123",
                    turns_used=3,
                    status="complete",
                    usage_totals=usage_totals
                )

            with open(log_path, 'r') as f:
                content = f.read()

            assert "test-conv-123" in content
            assert "turns=3" in content
            assert "status=complete" in content
            assert "prompt_tokens=100" in content
            assert "completion_tokens=50" in content
            assert "total_tokens=150" in content
        finally:
            os.unlink(log_path)

    def test_creates_directory_if_missing(self):
        from app.core.usage_logging import write_usage_log

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "subdir", "usage.log")

            with patch.dict(os.environ, {"JARVIS_LLM_USAGE_LOG_PATH": log_path}):
                usage_totals = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                write_usage_log(
                    conversation_id="test-123",
                    turns_used=1,
                    status="complete",
                    usage_totals=usage_totals
                )

            assert os.path.exists(log_path)

    def test_appends_to_existing_file(self):
        from app.core.usage_logging import write_usage_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("existing content\n")
            log_path = f.name

        try:
            with patch.dict(os.environ, {"JARVIS_LLM_USAGE_LOG_PATH": log_path}):
                usage_totals = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
                write_usage_log(
                    conversation_id="new-entry",
                    turns_used=1,
                    status="complete",
                    usage_totals=usage_totals
                )

            with open(log_path, 'r') as f:
                content = f.read()

            assert "existing content" in content
            assert "new-entry" in content
        finally:
            os.unlink(log_path)

    def test_handles_write_error_gracefully(self):
        from app.core.usage_logging import write_usage_log

        # Use an invalid path
        with patch.dict(os.environ, {"JARVIS_LLM_USAGE_LOG_PATH": "/nonexistent/path/file.log"}):
            # Should not raise
            usage_totals = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            write_usage_log(
                conversation_id="test-123",
                turns_used=1,
                status="error",
                usage_totals=usage_totals
            )

    def test_includes_timestamp(self):
        from app.core.usage_logging import write_usage_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            with patch.dict(os.environ, {"JARVIS_LLM_USAGE_LOG_PATH": log_path}):
                usage_totals = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
                write_usage_log(
                    conversation_id="test-123",
                    turns_used=1,
                    status="complete",
                    usage_totals=usage_totals
                )

            with open(log_path, 'r') as f:
                content = f.read()

            # Should have ISO timestamp
            assert "T" in content  # ISO format contains T
            assert "Z" in content  # UTC indicator
        finally:
            os.unlink(log_path)


class TestWritePromptResponseLog:
    """Tests for writing prompt/response trace logs."""

    def test_writes_json_entry(self):
        from app.core.usage_logging import write_prompt_response_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            with patch.dict(os.environ, {"JARVIS_LLM_TRACE_LOG_PATH": log_path}):
                write_prompt_response_log(
                    conversation_id="test-conv-456",
                    prompt_messages=[{"role": "user", "content": "hello"}],
                    response_obj={"choices": [{"message": {"content": "hi"}}]},
                    raw_content="hi",
                    error=None,
                    iteration=1
                )

            with open(log_path, 'r') as f:
                content = f.read().strip()

            entry = json.loads(content)
            assert entry["conversation_id"] == "test-conv-456"
            assert entry["iteration"] == 1
            assert entry["prompt_messages"] == [{"role": "user", "content": "hello"}]
            assert entry["raw_content"] == "hi"
            assert entry["error"] is None
        finally:
            os.unlink(log_path)

    def test_logs_error_message(self):
        from app.core.usage_logging import write_prompt_response_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            with patch.dict(os.environ, {"JARVIS_LLM_TRACE_LOG_PATH": log_path}):
                write_prompt_response_log(
                    conversation_id="test-error",
                    prompt_messages=[],
                    response_obj=None,
                    raw_content=None,
                    error="Connection timeout",
                    iteration=2
                )

            with open(log_path, 'r') as f:
                content = f.read().strip()

            entry = json.loads(content)
            assert entry["error"] == "Connection timeout"
            assert entry["iteration"] == 2
        finally:
            os.unlink(log_path)

    def test_creates_directory_if_missing(self):
        from app.core.usage_logging import write_prompt_response_log

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "trace", "trace.log")

            with patch.dict(os.environ, {"JARVIS_LLM_TRACE_LOG_PATH": log_path}):
                write_prompt_response_log(
                    conversation_id="test-123",
                    prompt_messages=[],
                    response_obj=None,
                    raw_content="test",
                    error=None,
                    iteration=1
                )

            assert os.path.exists(log_path)

    def test_appends_multiple_entries(self):
        from app.core.usage_logging import write_prompt_response_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            with patch.dict(os.environ, {"JARVIS_LLM_TRACE_LOG_PATH": log_path}):
                write_prompt_response_log(
                    conversation_id="test-123",
                    prompt_messages=[],
                    response_obj=None,
                    raw_content="first",
                    error=None,
                    iteration=1
                )
                write_prompt_response_log(
                    conversation_id="test-123",
                    prompt_messages=[],
                    response_obj=None,
                    raw_content="second",
                    error=None,
                    iteration=2
                )

            with open(log_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 2
            first = json.loads(lines[0])
            second = json.loads(lines[1])
            assert first["raw_content"] == "first"
            assert second["raw_content"] == "second"
        finally:
            os.unlink(log_path)

    def test_handles_write_error_gracefully(self):
        from app.core.usage_logging import write_prompt_response_log

        with patch.dict(os.environ, {"JARVIS_LLM_TRACE_LOG_PATH": "/nonexistent/dir/trace.log"}):
            # Should not raise
            write_prompt_response_log(
                conversation_id="test-123",
                prompt_messages=[],
                response_obj=None,
                raw_content="test",
                error=None,
                iteration=1
            )

    def test_includes_timestamp(self):
        from app.core.usage_logging import write_prompt_response_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            with patch.dict(os.environ, {"JARVIS_LLM_TRACE_LOG_PATH": log_path}):
                write_prompt_response_log(
                    conversation_id="test-123",
                    prompt_messages=[],
                    response_obj=None,
                    raw_content="test",
                    error=None,
                    iteration=1
                )

            with open(log_path, 'r') as f:
                content = f.read().strip()

            entry = json.loads(content)
            assert "timestamp" in entry
            assert "T" in entry["timestamp"]
            assert "Z" in entry["timestamp"]
        finally:
            os.unlink(log_path)

    def test_serializes_complex_response_obj(self):
        from app.core.usage_logging import write_prompt_response_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            complex_response = {
                "choices": [
                    {
                        "message": {
                            "content": '{"tool_calls": []}',
                            "role": "assistant"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}
            }

            with patch.dict(os.environ, {"JARVIS_LLM_TRACE_LOG_PATH": log_path}):
                write_prompt_response_log(
                    conversation_id="test-123",
                    prompt_messages=[{"role": "user", "content": "test"}],
                    response_obj=complex_response,
                    raw_content='{"tool_calls": []}',
                    error=None,
                    iteration=1
                )

            with open(log_path, 'r') as f:
                content = f.read().strip()

            entry = json.loads(content)
            assert entry["response"]["usage"]["prompt_tokens"] == 100
        finally:
            os.unlink(log_path)
