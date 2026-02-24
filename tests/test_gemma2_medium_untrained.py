"""
Tests for Gemma2MediumUntrained prompt provider.

Verifies:
- Provider implements IJarvisPromptProvider correctly
- build_system_prompt produces valid output with expected sections
- Tools are pretty-printed JSON schemas (no native tool format wrapper)
- Agent context (HA devices) is included when present
- use_tool_classifier returns True
- get_capabilities returns correct metadata
- get_response_format returns text mode
- parse_response transforms <tool_call> tags into Jarvis JSON
"""

import json

import pytest

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.medium.untrained.gemma2_medium_untrained import (
    Gemma2MediumUntrained,
)


@pytest.fixture
def provider() -> Gemma2MediumUntrained:
    return Gemma2MediumUntrained()


@pytest.fixture
def sample_tools() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": "Set a countdown timer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Timer duration in minutes",
                        }
                    },
                    "required": ["duration_minutes"],
                },
            },
        },
    ]


@pytest.fixture
def sample_node_context() -> dict:
    return {
        "room": "kitchen",
        "user": "alex",
        "voice_mode": "brief",
    }


@pytest.fixture
def sample_commands() -> list:
    return [
        {
            "command_name": "get_weather",
            "allow_direct_answer": False,
        },
        {
            "command_name": "calculate",
            "allow_direct_answer": True,
        },
    ]


@pytest.fixture
def sample_node_context_with_agents() -> dict:
    return {
        "room": "living_room",
        "user": "alex",
        "voice_mode": "brief",
        "agents": {
            "home_assistant": {
                "light_controls": {
                    "Living Room": {
                        "entity_id": "light.living_room",
                        "state": "on",
                    }
                },
                "device_controls": {
                    "switch": [
                        {
                            "entity_id": "switch.fan",
                            "name": "Ceiling Fan",
                            "state": "off",
                        }
                    ],
                    "scene": [
                        {
                            "entity_id": "scene.movie_time",
                            "name": "Movie Time",
                        }
                    ],
                },
            }
        },
    }


class TestGemma2MediumUntrainedInterface:
    """Test that the provider implements IJarvisPromptProvider correctly."""

    def test_is_instance_of_interface(self, provider: Gemma2MediumUntrained):
        assert isinstance(provider, IJarvisPromptProvider)

    def test_name_property(self, provider: Gemma2MediumUntrained):
        assert provider.name == "Gemma2MediumUntrained"

    def test_use_tool_classifier_true(self, provider: Gemma2MediumUntrained):
        assert provider.use_tool_classifier is True

    def test_get_response_format_text_mode(self, provider: Gemma2MediumUntrained):
        fmt = provider.get_response_format()
        assert fmt == {"type": "text"}

    def test_get_capabilities(self, provider: Gemma2MediumUntrained):
        caps = provider.get_capabilities()
        assert caps["provider_name"] == "Gemma2MediumUntrained"
        assert caps["model_family"] == "gemma"
        assert caps["size_tier"] == "medium"
        assert caps["training_tier"] == "untrained"
        assert caps["use_tool_classifier"] is True
        assert caps["supports_native_tools"] is False

    def test_supports_native_tools_false(self, provider: Gemma2MediumUntrained):
        assert provider.supports_native_tools is False

    def test_build_tools_delegates_to_tool_builder(
        self, provider: Gemma2MediumUntrained, sample_tools: list
    ):
        result = provider.build_tools(sample_tools)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "function"


class TestGemma2MediumUntrainedPrompt:
    """Test system prompt generation."""

    def test_prompt_includes_identity(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "Jarvis" in prompt
        assert "function calling" in prompt

    def test_prompt_includes_node_context(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "room=kitchen" in prompt
        assert "user=alex" in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_tool_call_format(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "<tool_call>" in prompt
        assert "</tool_call>" in prompt
        assert "brief spoken reply" in prompt

    def test_prompt_uses_pretty_printed_json(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        """Gemma 2 has no native tool format — uses pretty-printed JSON schemas."""
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        # Should NOT use Qwen's <tools> XML wrapper
        assert "<tools>" not in prompt
        assert "</tools>" not in prompt
        # Should have "Available functions:" with pretty-printed JSON
        assert "Available functions:" in prompt
        # Pretty-printed JSON has indentation
        assert '  "type": "function"' in prompt

    def test_prompt_includes_tool_schemas(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert '"get_weather"' in prompt
        assert '"set_timer"' in prompt

    def test_prompt_includes_tools_section(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "Tools:" in prompt
        assert "get_weather" in prompt
        assert "set_timer" in prompt

    def test_prompt_with_direct_answer_policy(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
        sample_commands: list,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context,
            "America/New_York",
            sample_tools,
            sample_commands,
        )
        assert "Direct Answer Policy" in prompt
        assert "MUST call tools for" in prompt
        assert "get_weather" in prompt
        assert "Direct answers allowed for" in prompt
        assert "calculate" in prompt

    def test_prompt_without_commands(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools, []
        )
        assert "Direct Answer Policy" not in prompt

    def test_prompt_with_empty_context(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
    ):
        prompt = provider.build_system_prompt({}, None, sample_tools)
        assert "room=unknown" in prompt
        assert "user=default" in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_agent_context(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context_with_agents: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context_with_agents, "America/New_York", sample_tools
        )
        assert "Living Room" in prompt
        assert "light.living_room" in prompt
        assert "Ceiling Fan" in prompt
        assert "switch.fan" in prompt
        assert "Movie Time" in prompt
        assert "scene.movie_time" in prompt

    def test_prompt_without_agent_context(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "Available Light Controls" not in prompt
        assert "Available Switches" not in prompt

    def test_prompt_returns_string(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        result = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_includes_anti_hallucination(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "NEVER fabricate" in prompt

    def test_prompt_includes_no_iso_dates_rule(
        self,
        provider: Gemma2MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "NEVER output ISO dates" in prompt


class TestGemma2MediumUntrainedParseResponse:
    """Test parse_response for Gemma 2 output -> Jarvis JSON transformation."""

    def test_single_tool_call(self, provider: Gemma2MediumUntrained):
        raw = '<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == ""
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][0]["arguments"] == {"location": "NYC"}
        assert parsed["error"] is None

    def test_tool_call_with_whitespace(self, provider: Gemma2MediumUntrained):
        raw = '<tool_call>\n  {"name":"set_timer","arguments":{"duration_minutes":5}}\n</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "set_timer"

    def test_multiple_tool_calls(self, provider: Gemma2MediumUntrained):
        raw = (
            '<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>\n'
            '<tool_call>{"name":"set_timer","arguments":{"duration_minutes":5}}</tool_call>'
        )
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 2
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][1]["name"] == "set_timer"

    def test_tool_call_with_surrounding_text(self, provider: Gemma2MediumUntrained):
        """Gemma 2 may emit text before tool call."""
        raw = 'I\'ll check the weather for you.\n<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"

    def test_returns_none_for_clean_jarvis_json(self, provider: Gemma2MediumUntrained):
        raw = '{"message":"hello","tool_calls":[],"error":null}'
        result = provider.parse_response(raw)
        assert result is None

    def test_returns_none_for_jarvis_json_with_whitespace(self, provider: Gemma2MediumUntrained):
        raw = '  {"message":"hello","tool_calls":[],"error":null}  '
        result = provider.parse_response(raw)
        assert result is None

    def test_plain_text_wrapped_as_jarvis_json(self, provider: Gemma2MediumUntrained):
        raw = "The weather in NYC is sunny and 72 degrees."
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == raw
        assert parsed["tool_calls"] == []
        assert parsed["error"] is None

    def test_invalid_tool_call_json_skipped(self, provider: Gemma2MediumUntrained):
        raw = '<tool_call>not valid json</tool_call>'
        result = provider.parse_response(raw)
        # No valid tool calls parsed, falls through to plain text wrap
        parsed = json.loads(result)
        assert parsed["tool_calls"] == []

    def test_resolved_datetimes_string_normalized_to_array(self, provider: Gemma2MediumUntrained):
        raw = '<tool_call>{"name":"get_weather","arguments":{"city":"Miami","resolved_datetimes":"today"}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["tool_calls"][0]["arguments"]["resolved_datetimes"] == ["today"]

    def test_resolved_datetimes_array_unchanged(self, provider: Gemma2MediumUntrained):
        raw = '<tool_call>{"name":"get_weather","arguments":{"city":"Miami","resolved_datetimes":["today","tomorrow"]}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["tool_calls"][0]["arguments"]["resolved_datetimes"] == ["today", "tomorrow"]
