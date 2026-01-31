"""
Integration tests for ToolCallParser.

Tests the parser's ability to extract tool calls from various LLM response formats.
The parser must handle:
- Standard JSON format
- OpenAI function wrapper format
- Arguments as JSON strings
- JSON with comments
- JSON embedded in markdown
"""

import pytest
import json

from app.core.tool_call_parser import ToolCallParser


class TestParseSingleTool:
    """Test parsing of single tool calls."""

    def test_parse_basic_tool_call(self):
        """
        Test: standard format with name and arguments dict.
        """
        llm_output = json.dumps({
            "message": "I'll calculate that for you.",
            "tool_calls": [
                {"name": "calculate", "arguments": {"expression": "2+2"}}
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "calculate"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"expression": "2+2"}
        assert message == "I'll calculate that for you."

    def test_parse_tool_call_no_message(self):
        """
        Test: tool call with empty message.
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {"name": "get_weather", "arguments": {"location": "NYC"}}
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert message == ""

    def test_parse_tool_with_no_arguments(self):
        """
        Test: tool call with empty arguments.
        """
        llm_output = json.dumps({
            "message": "Checking status...",
            "tool_calls": [
                {"name": "check_status", "arguments": {}}
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {}


class TestParseMultipleTools:
    """Test parsing of multiple tool calls."""

    def test_parse_two_tool_calls(self):
        """
        Test: two tool calls in a single response.
        """
        llm_output = json.dumps({
            "message": "I'll get both for you.",
            "tool_calls": [
                {"name": "calculate", "arguments": {"expression": "5+5"}},
                {"name": "get_weather", "arguments": {"location": "Boston"}}
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 2

        # Verify both tools are present
        tool_names = [tc["function"]["name"] for tc in tool_calls]
        assert "calculate" in tool_names
        assert "get_weather" in tool_names

    def test_parse_three_tool_calls(self):
        """
        Test: three tool calls in a single response.
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {"name": "tool_a", "arguments": {"param": "a"}},
                {"name": "tool_b", "arguments": {"param": "b"}},
                {"name": "tool_c", "arguments": {"param": "c"}}
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 3


class TestParseFunctionWrapper:
    """Test parsing of OpenAI function wrapper format."""

    def test_parse_function_wrapper_format(self):
        """
        Test: tool call wrapped in "function" object.

        Format: {"function": {"name": "...", "arguments": {...}}}
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "10*10"}
                    }
                }
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "calculate"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"expression": "10*10"}

    def test_parse_mixed_formats(self):
        """
        Test: mix of direct format and function wrapper format.
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {"name": "direct_tool", "arguments": {"key": "value1"}},
                {
                    "function": {
                        "name": "wrapped_tool",
                        "arguments": {"key": "value2"}
                    }
                }
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 2

        tool_names = [tc["function"]["name"] for tc in tool_calls]
        assert "direct_tool" in tool_names
        assert "wrapped_tool" in tool_names


class TestParseArgumentsAsString:
    """Test parsing when arguments are JSON strings instead of dicts."""

    def test_parse_string_arguments(self):
        """
        Test: arguments provided as JSON string.
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {
                    "name": "calculate",
                    "arguments": '{"expression": "3+3"}'  # String, not dict
                }
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1

        # Arguments should be normalized to JSON string
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args == {"expression": "3+3"}

    def test_parse_complex_string_arguments(self):
        """
        Test: complex arguments as JSON string.
        """
        complex_args = {"items": ["a", "b", "c"], "count": 3, "nested": {"key": "value"}}
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {
                    "name": "process_data",
                    "arguments": json.dumps(complex_args)
                }
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args == complex_args


class TestParseJsonWithComments:
    """Test parsing JSON with # comments (which some LLMs add)."""

    def test_parse_json_with_inline_comments(self):
        """
        Test: JSON with # comments after values.
        """
        llm_output = """{
            "message": "Calculating...",  # This is the message
            "tool_calls": [
                {
                    "name": "calculate",  # The tool name
                    "arguments": {
                        "expression": "2+2"  # The math expression
                    }
                }
            ]
        }"""

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "calculate"

    def test_parse_json_with_comment_lines(self):
        """
        Test: JSON with full comment lines.
        """
        llm_output = """{
            # This is the response
            "message": "",
            # These are the tool calls
            "tool_calls": [
                {"name": "test_tool", "arguments": {}}
            ]
        }"""

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1


class TestParseJsonFromMarkdown:
    """Test extracting JSON from markdown code blocks."""

    def test_extract_json_from_code_block(self):
        """
        Test: JSON wrapped in ```json ... ``` block.
        """
        llm_output = """Here is my response:

```json
{
    "message": "Done!",
    "tool_calls": [
        {"name": "complete_task", "arguments": {"task_id": "123"}}
    ]
}
```

Let me know if you need anything else."""

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "complete_task"

    def test_extract_json_from_generic_code_block(self):
        """
        Test: JSON in generic ``` block (no language specified).
        """
        llm_output = """Response:

```
{"message": "Hello", "tool_calls": []}
```"""

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "stop"
        assert message == "Hello"


class TestParseCompleteResponse:
    """Test parsing responses with no tool calls (complete)."""

    def test_parse_empty_tool_calls(self):
        """
        Test: explicit empty tool_calls array.
        """
        llm_output = json.dumps({
            "message": "The answer is 42.",
            "tool_calls": []
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "stop"
        assert len(tool_calls) == 0
        assert message == "The answer is 42."

    def test_parse_missing_tool_calls(self):
        """
        Test: no tool_calls key at all.
        """
        llm_output = json.dumps({
            "message": "Just a simple response."
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "stop"
        assert len(tool_calls) == 0


class TestParseInvalidJson:
    """Test handling of invalid or non-JSON responses."""

    def test_parse_plain_text_response(self):
        """
        Test: plain text that isn't JSON.
        """
        llm_output = "I'm sorry, I don't understand that request."

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        # Should treat as plain text response
        assert finish_reason == "stop"
        assert len(tool_calls) == 0
        assert "don't understand" in message

    def test_parse_malformed_json(self):
        """
        Test: invalid JSON syntax.
        """
        llm_output = '{"message": "incomplete...'

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        # Should handle gracefully
        assert finish_reason == "stop"

    def test_parse_json_with_extra_text(self):
        """
        Test: valid JSON surrounded by extra text.
        """
        llm_output = """Let me help you with that.

{"message": "Processing...", "tool_calls": [{"name": "process", "arguments": {}}]}

Hope that helps!"""

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "process"


class TestSingularToolCallFallback:
    """Test parsing of singular tool_call (not array) format."""

    def test_parse_singular_tool_call(self):
        """
        Test: tool_call as single object instead of array.
        """
        llm_output = json.dumps({
            "message": "",
            "tool_call": {"name": "singular_tool", "arguments": {"key": "value"}}
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "singular_tool"


class TestResolvedDatetimesNormalization:
    """Test normalization of resolved_datetimes parameter."""

    def test_normalize_datetime_keys_with_spaces(self):
        """
        Test: datetime keys with spaces are converted to underscores.
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {
                    "name": "set_reminder",
                    "arguments": {
                        "resolved_datetimes": ["tomorrow at 3pm", "next monday"]
                    }
                }
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        args = json.loads(tool_calls[0]["function"]["arguments"])

        # Spaces should be converted to underscores
        assert "tomorrow_at_3pm" in args["resolved_datetimes"]
        assert "next_monday" in args["resolved_datetimes"]

    def test_unwrap_nested_resolved_datetimes(self):
        """
        Test: nested resolved_datetimes objects are unwrapped.
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {
                    "name": "schedule_meeting",
                    "arguments": {
                        "resolved_datetimes": [
                            {"resolved_datetimes": ["tomorrow"]}
                        ]
                    }
                }
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert finish_reason == "tool_calls"
        args = json.loads(tool_calls[0]["function"]["arguments"])

        # Should be unwrapped to flat list
        assert args["resolved_datetimes"] == ["tomorrow"]


class TestToolCallIdGeneration:
    """Test that tool call IDs are properly generated."""

    def test_tool_calls_have_unique_ids(self):
        """
        Test: each tool call gets a unique ID.
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {"name": "tool_a", "arguments": {}},
                {"name": "tool_b", "arguments": {}},
                {"name": "tool_c", "arguments": {}}
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        ids = [tc["id"] for tc in tool_calls]
        assert len(ids) == len(set(ids))  # All unique

    def test_tool_call_id_format(self):
        """
        Test: tool call IDs follow expected format (call_*).
        """
        llm_output = json.dumps({
            "message": "",
            "tool_calls": [
                {"name": "test_tool", "arguments": {}}
            ]
        })

        finish_reason, tool_calls, message = ToolCallParser.parse_response(llm_output)

        assert tool_calls[0]["id"].startswith("call_")
