"""Regression tests for ToolCallParser recovering BARE (un-enveloped) tool calls.

Smaller / quantized models (e.g. Qwen3-8B-Q4) frequently emit a tool call
without the {"tool_calls": [...]} envelope, e.g. just:

    {"name": "quick_search", "arguments": {"query": "..."}, "failure_message": "..."}

Before the fallback, parse_response found no envelope key and reported this as a
plain answer (tool_calls=0), silently dropping the call — which is why
quick_search, get_current_time, and entertainment_knowledge all failed the same
way on the dev model. These lock in that the bare form is recovered, while a
genuine plain-text answer is NOT misread as a tool call.
"""
from app.core.tool_call_parser import ToolCallParser


def test_bare_tool_call_is_recovered():
    out = '{"name": "quick_search", "arguments": {"query": "spider-man release"}}'
    finish_reason, tool_calls, _ = ToolCallParser.parse_response(out)
    assert finish_reason == "tool_calls"
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "quick_search"


def test_bare_tool_call_with_failure_message_is_recovered():
    out = (
        '{"name": "get_current_time", "arguments": {}, '
        '"failure_message": "I could not get the time."}'
    )
    finish_reason, tool_calls, _ = ToolCallParser.parse_response(out)
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "get_current_time"


def test_enveloped_tool_calls_still_parse():
    out = '{"message": "", "tool_calls": [{"name": "quick_search", "arguments": {"query": "x"}}]}'
    finish_reason, tool_calls, _ = ToolCallParser.parse_response(out)
    assert finish_reason == "tool_calls"
    assert len(tool_calls) == 1


def test_plain_answer_is_not_misread_as_tool_call():
    # A normal completion has a "message" and no "name"/"function" — must stay "stop".
    out = '{"message": "It is 3 PM.", "tool_calls": []}'
    finish_reason, tool_calls, message = ToolCallParser.parse_response(out)
    assert finish_reason == "stop"
    assert tool_calls == []
    assert "3 PM" in message


def test_bare_object_without_name_is_not_a_tool_call():
    out = '{"answer": "42", "confidence": 0.9}'
    finish_reason, tool_calls, _ = ToolCallParser.parse_response(out)
    assert finish_reason == "stop"
    assert tool_calls == []
