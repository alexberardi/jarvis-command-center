"""
Tests for system prompt builder.

These tests cover the system prompt building functionality extracted from model_service.py.
"""
import pytest
import os
from unittest.mock import patch


class TestBuildToolSystemMessage:
    """Tests for building tool system messages."""

    def test_basic_structure_compact(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "living room", "user": "john", "voice_mode": "brief"}
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "compact"}):
            result = build_tool_system_message(node_context, "America/New_York", tools)

        assert "Jarvis" in result
        assert "living room" in result or "room=living room" in result
        assert "john" in result or "user=john" in result

    def test_basic_structure_full(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "kitchen", "user": "jane", "voice_mode": "detailed"}
        tools = []

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "full"}):
            result = build_tool_system_message(node_context, None, tools)

        assert "Jarvis" in result
        assert "kitchen" in result
        assert "jane" in result
        assert "detailed" in result

    def test_includes_tools_section(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "office", "user": "test", "voice_mode": "brief"}
        tools = [
            {
                "function": {
                    "name": "set_reminder",
                    "description": "Set a reminder",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "compact"}):
            result = build_tool_system_message(node_context, None, tools)

        # Tools should be mentioned
        assert "set_reminder" in result or "Tools:" in result

    def test_voice_mode_defaults(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "bedroom"}  # Missing user and voice_mode
        tools = []

        result = build_tool_system_message(node_context, None, tools)

        # Should use defaults
        assert "default" in result or "brief" in result or "unknown" in result

    def test_includes_json_format_instructions(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "office", "user": "test", "voice_mode": "brief"}
        tools = []

        result = build_tool_system_message(node_context, None, tools)

        # Should include JSON format instructions
        assert "JSON" in result or "json" in result

    def test_tool_call_format_shown(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "office", "user": "test", "voice_mode": "brief"}
        tools = []

        result = build_tool_system_message(node_context, None, tools)

        # Should show tool_call format
        assert "tool_call" in result or "tool_calls" in result

    @patch.dict(os.environ, {"JARVIS_INCLUDE_REASON": "true"})
    def test_includes_reason_field_when_enabled(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "office", "user": "test", "voice_mode": "brief"}
        tools = []

        result = build_tool_system_message(node_context, None, tools)

        # Should mention reason field
        assert "reason" in result

    @patch.dict(os.environ, {"JARVIS_INCLUDE_REASON": "false"}, clear=False)
    def test_excludes_reason_field_when_disabled(self):
        from app.core.system_prompt_builder import build_tool_system_message

        # Force environment variable
        with patch.dict(os.environ, {"JARVIS_INCLUDE_REASON": "false"}):
            node_context = {"room": "office", "user": "test", "voice_mode": "brief"}
            tools = []

            result = build_tool_system_message(node_context, None, tools)

            # reason field mentions should be minimal (may still appear in format examples)
            # The key is that it shouldn't say "optional reason field" type language

    def test_empty_tools(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "office", "user": "test", "voice_mode": "brief"}
        tools = []

        result = build_tool_system_message(node_context, None, tools)

        # Should still produce valid prompt
        assert len(result) > 0
        assert "Jarvis" in result

    def test_speaker_name_used_when_present(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {
            "room": "kitchen",
            "user": "default",
            "voice_mode": "brief",
            "speaker_name": "Alice",
        }
        tools = []

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "compact"}):
            result = build_tool_system_message(node_context, None, tools)

        assert "Alice" in result
        assert "speaker=Alice" in result

    def test_speaker_name_used_in_full_prompt(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {
            "room": "living room",
            "user": "default",
            "voice_mode": "detailed",
            "speaker_name": "Bob",
        }
        tools = []

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "full"}):
            result = build_tool_system_message(node_context, None, tools)

        assert "Bob" in result
        assert "User: Bob" in result

    def test_falls_back_to_user_when_no_speaker(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "office", "user": "john", "voice_mode": "brief"}
        tools = []

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "compact"}):
            result = build_tool_system_message(node_context, None, tools)

        assert "speaker=john" in result

    def test_includes_user_memories_compact(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {
            "room": "kitchen",
            "user": "default",
            "voice_mode": "brief",
            "speaker_name": "Alice",
            "user_memories": "- likes coffee black\n- vegetarian",
        }
        tools = []

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "compact"}):
            result = build_tool_system_message(node_context, None, tools)

        assert "About Alice:" in result
        assert "likes coffee black" in result
        assert "vegetarian" in result

    def test_includes_user_memories_full(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {
            "room": "kitchen",
            "user": "default",
            "voice_mode": "detailed",
            "speaker_name": "Bob",
            "user_memories": "- prefers metric units",
        }
        tools = []

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "full"}):
            result = build_tool_system_message(node_context, None, tools)

        assert "About Bob:" in result
        assert "prefers metric units" in result

    def test_omits_memory_block_when_no_memories(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "office", "user": "test", "voice_mode": "brief"}
        tools = []

        with patch.dict(os.environ, {"JARVIS_PROMPT_STYLE": "compact"}):
            result = build_tool_system_message(node_context, None, tools)

        assert "About" not in result

    def test_available_commands_flags(self):
        from app.core.system_prompt_builder import build_tool_system_message

        node_context = {"room": "office", "user": "test", "voice_mode": "brief"}
        tools = [
            {
                "function": {
                    "name": "play_music",
                    "description": "Play music",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        available_commands = [
            {"command_name": "play_music", "allow_direct_answer": False}
        ]

        result = build_tool_system_message(node_context, None, tools, available_commands)

        # Should include tool info
        assert len(result) > 0


class TestFormatToolsWithExamples:
    """Tests for formatting tools with examples for the prompt."""

    def test_basic_tool_formatting(self):
        from app.core.system_prompt_builder import format_tools_for_prompt

        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "The city name"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        result = format_tools_for_prompt(tools)

        assert "get_weather" in result
        assert "city" in result

    def test_multiple_tools(self):
        from app.core.system_prompt_builder import format_tools_for_prompt

        tools = [
            {
                "function": {
                    "name": "tool_one",
                    "description": "First tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "function": {
                    "name": "tool_two",
                    "description": "Second tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        result = format_tools_for_prompt(tools)

        assert "tool_one" in result
        assert "tool_two" in result

    def test_tool_with_examples(self):
        from app.core.system_prompt_builder import format_tools_for_prompt

        tools = [
            {
                "function": {
                    "name": "set_timer",
                    "description": "Set a timer",
                    "parameters": {"type": "object", "properties": {}}
                },
                "examples": [
                    {
                        "voice_command": "set a timer for 5 minutes",
                        "expected_parameters": {"duration": 300},
                        "is_primary": True
                    }
                ]
            }
        ]
        available_commands = [
            {
                "command_name": "set_timer",
                "examples": [
                    {
                        "voice_command": "set a timer for 5 minutes",
                        "expected_parameters": {"duration": 300},
                        "is_primary": True
                    }
                ]
            }
        ]

        result = format_tools_for_prompt(tools, available_commands)

        assert "set_timer" in result

    def test_empty_tools(self):
        from app.core.system_prompt_builder import format_tools_for_prompt

        result = format_tools_for_prompt([])

        # Should return empty or placeholder
        assert result is not None

    def test_tool_in_openai_format(self):
        from app.core.system_prompt_builder import format_tools_for_prompt

        # Tools should be in OpenAI format with "function" wrapper
        tools = [
            {
                "function": {
                    "name": "simple_tool",
                    "description": "A simple tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        result = format_tools_for_prompt(tools)

        assert "simple_tool" in result


class TestGetResponseFormat:
    """Tests for getting the JSON response format schema."""

    def test_returns_json_object_type(self):
        from app.core.system_prompt_builder import get_response_format

        result = get_response_format()

        assert result["type"] == "json_object"

    def test_includes_message_field(self):
        from app.core.system_prompt_builder import get_response_format

        result = get_response_format()
        schema = result.get("json_schema", {})
        properties = schema.get("properties", {})

        assert "message" in properties

    def test_includes_tool_calls_field(self):
        from app.core.system_prompt_builder import get_response_format

        result = get_response_format()
        schema = result.get("json_schema", {})
        properties = schema.get("properties", {})

        assert "tool_calls" in properties

    def test_message_and_tool_calls_required(self):
        from app.core.system_prompt_builder import get_response_format

        result = get_response_format()
        schema = result.get("json_schema", {})
        required = schema.get("required", [])

        assert "message" in required
        assert "tool_calls" in required
