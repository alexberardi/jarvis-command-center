"""
Tests for HermesMediumUntrained prompt provider.

Verifies:
- Provider implements IJarvisPromptProvider correctly
- build_system_prompt produces valid output with expected sections
- Tools are wrapped in <tools> XML tags (Hermes native format)
- Agent context (HA devices) is included when present
- use_tool_classifier returns True
- get_capabilities returns correct metadata
- get_response_format returns text mode
- parse_response transforms <tool_call> and <scratch_pad> tags into Jarvis JSON (fallback)
"""

import json

import pytest

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
    HermesMediumUntrained,
)


@pytest.fixture
def provider() -> HermesMediumUntrained:
    return HermesMediumUntrained()


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


class TestHermesMediumUntrainedInterface:
    """Test that the provider implements IJarvisPromptProvider correctly."""

    def test_is_instance_of_interface(self, provider: HermesMediumUntrained):
        assert isinstance(provider, IJarvisPromptProvider)

    def test_name_property(self, provider: HermesMediumUntrained):
        assert provider.name == "HermesMediumUntrained"

    def test_use_tool_classifier_true(self, provider: HermesMediumUntrained):
        assert provider.use_tool_classifier is True

    def test_get_response_format_text_mode(self, provider: HermesMediumUntrained):
        fmt = provider.get_response_format()
        assert fmt == {"type": "text"}

    def test_get_capabilities(self, provider: HermesMediumUntrained):
        caps = provider.get_capabilities()
        assert caps["provider_name"] == "HermesMediumUntrained"
        assert caps["model_family"] == "hermes"
        assert caps["size_tier"] == "medium"
        assert caps["training_tier"] == "untrained"
        assert caps["use_tool_classifier"] is True
        assert caps["supports_native_tools"] is False

    def test_supports_native_tools_false(self, provider: HermesMediumUntrained):
        assert provider.supports_native_tools is False


class TestHermesMediumUntrainedPrompt:
    """Test system prompt generation."""

    def test_prompt_includes_identity(
        self,
        provider: HermesMediumUntrained,
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
        provider: HermesMediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "room=kitchen" in prompt
        assert "user=alex" in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_tools_xml_tags(
        self,
        provider: HermesMediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "<tools>" in prompt
        assert "</tools>" in prompt

    def test_prompt_includes_tool_schemas_in_xml(
        self,
        provider: HermesMediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        # Tool JSON schemas should be inside the standalone <tools> block
        # Skip the first "<tools></tools>" in the intro text
        standalone_start = prompt.index("\n<tools>\n") + len("\n<tools>\n")
        tools_end = prompt.index("\n</tools>")
        tools_content = prompt[standalone_start:tools_end]
        assert '"get_weather"' in tools_content
        assert '"set_timer"' in tools_content
        assert '"type": "function"' in tools_content

    def test_prompt_includes_native_tool_call_format(
        self,
        provider: HermesMediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "<tool_call>" in prompt
        assert "brief spoken reply" in prompt
        # Should NOT fight the model's native format
        assert "JSON object ONLY" not in prompt
        assert "no XML tags" not in prompt

    def test_prompt_includes_tools_section(
        self,
        provider: HermesMediumUntrained,
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
        provider: HermesMediumUntrained,
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
        provider: HermesMediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools, []
        )
        assert "Direct Answer Policy" not in prompt

    def test_prompt_with_empty_context(
        self,
        provider: HermesMediumUntrained,
        sample_tools: list,
    ):
        prompt = provider.build_system_prompt({}, None, sample_tools)
        assert "room=unknown" in prompt
        assert "user=default" in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_agent_context(
        self,
        provider: HermesMediumUntrained,
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
        provider: HermesMediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        # No HA device sections when agents not present
        assert "Available Light Controls" not in prompt
        assert "Available Switches" not in prompt

    def test_prompt_returns_string(
        self,
        provider: HermesMediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        result = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestHermesMediumUntrainedParseResponse:
    """Test parse_response for Hermes native output → Jarvis JSON transformation."""

    def test_single_tool_call_tag(self, provider: HermesMediumUntrained):
        raw = '<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == ""
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][0]["arguments"] == {"location": "NYC"}
        assert parsed["error"] is None

    def test_tool_call_tag_with_whitespace(self, provider: HermesMediumUntrained):
        raw = '<tool_call>\n  {"name":"set_timer","arguments":{"duration_minutes":5}}\n</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "set_timer"

    def test_multiple_tool_call_tags(self, provider: HermesMediumUntrained):
        raw = (
            '<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>'
            '<tool_call>{"name":"set_timer","arguments":{"duration_minutes":5}}</tool_call>'
        )
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 2
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][1]["name"] == "set_timer"

    def test_scratch_pad_with_jarvis_json(self, provider: HermesMediumUntrained):
        raw = '<scratch_pad>thinking about this...</scratch_pad>{"message":"hello","tool_calls":[],"error":null}'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == "hello"
        assert parsed["tool_calls"] == []

    def test_scratch_pad_and_tool_call(self, provider: HermesMediumUntrained):
        raw = '<scratch_pad>let me think</scratch_pad><tool_call>{"name":"x","arguments":{}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "x"

    def test_returns_none_for_clean_jarvis_json(self, provider: HermesMediumUntrained):
        raw = '{"message":"hello","tool_calls":[],"error":null}'
        result = provider.parse_response(raw)
        assert result is None

    def test_returns_none_for_jarvis_json_with_whitespace(self, provider: HermesMediumUntrained):
        raw = '  {"message":"hello","tool_calls":[],"error":null}  '
        result = provider.parse_response(raw)
        assert result is None

    def test_plain_text_wrapped_as_jarvis_json(self, provider: HermesMediumUntrained):
        raw = "The weather in NYC is sunny and 72 degrees."
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == raw
        assert parsed["tool_calls"] == []
        assert parsed["error"] is None

    def test_scratch_pad_with_plain_text(self, provider: HermesMediumUntrained):
        raw = "<scratch_pad>thinking...</scratch_pad>Hello, how can I help?"
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == "Hello, how can I help?"
        assert parsed["tool_calls"] == []
