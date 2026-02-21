"""
Tests for Llama31MediumUntrained prompt provider.

Verifies:
- Provider implements IJarvisPromptProvider correctly
- build_system_prompt produces valid output with expected sections
- Tools are presented as JSON schemas with <function=name>{args}</function> format
- Agent context (HA devices) is included when present
- use_tool_classifier returns True
- get_capabilities returns correct metadata
- get_response_format returns text mode
- parse_response transforms <function=name>{args}</function> tags into Jarvis JSON
"""

import json

import pytest

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.medium.untrained.llama31_medium_untrained import (
    Llama31MediumUntrained,
)


@pytest.fixture
def provider() -> Llama31MediumUntrained:
    return Llama31MediumUntrained()


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


class TestLlama31MediumUntrainedInterface:
    """Test that the provider implements IJarvisPromptProvider correctly."""

    def test_is_instance_of_interface(self, provider: Llama31MediumUntrained):
        assert isinstance(provider, IJarvisPromptProvider)

    def test_name_property(self, provider: Llama31MediumUntrained):
        assert provider.name == "Llama31MediumUntrained"

    def test_use_tool_classifier_true(self, provider: Llama31MediumUntrained):
        assert provider.use_tool_classifier is True

    def test_get_response_format_text_mode(self, provider: Llama31MediumUntrained):
        fmt = provider.get_response_format()
        assert fmt == {"type": "text"}

    def test_get_capabilities(self, provider: Llama31MediumUntrained):
        caps = provider.get_capabilities()
        assert caps["provider_name"] == "Llama31MediumUntrained"
        assert caps["model_family"] == "llama"
        assert caps["size_tier"] == "medium"
        assert caps["training_tier"] == "untrained"
        assert caps["use_tool_classifier"] is True
        assert caps["supports_native_tools"] is False

    def test_supports_native_tools_false(self, provider: Llama31MediumUntrained):
        assert provider.supports_native_tools is False


class TestLlama31MediumUntrainedPrompt:
    """Test system prompt generation."""

    def test_prompt_includes_identity(
        self,
        provider: Llama31MediumUntrained,
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
        provider: Llama31MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "room=kitchen" in prompt
        assert "user=alex" in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_function_call_format(
        self,
        provider: Llama31MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "<function=function_name>" in prompt
        assert "</function>" in prompt
        assert "brief spoken reply" in prompt
        # Should NOT fight the model's native format
        assert "JSON object ONLY" not in prompt
        assert "<tool_call>" not in prompt

    def test_prompt_includes_tool_schemas(
        self,
        provider: Llama31MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert '"get_weather"' in prompt
        assert '"set_timer"' in prompt
        assert '"type": "function"' in prompt

    def test_prompt_includes_tools_section(
        self,
        provider: Llama31MediumUntrained,
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
        provider: Llama31MediumUntrained,
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
        provider: Llama31MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools, []
        )
        assert "Direct Answer Policy" not in prompt

    def test_prompt_with_empty_context(
        self,
        provider: Llama31MediumUntrained,
        sample_tools: list,
    ):
        prompt = provider.build_system_prompt({}, None, sample_tools)
        assert "room=unknown" in prompt
        assert "user=default" in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_agent_context(
        self,
        provider: Llama31MediumUntrained,
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
        provider: Llama31MediumUntrained,
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
        provider: Llama31MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        result = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestLlama31MediumUntrainedParseResponse:
    """Test parse_response for Llama 3.1 native output -> Jarvis JSON transformation."""

    def test_single_function_call(self, provider: Llama31MediumUntrained):
        raw = '<function=get_weather>{"location": "NYC"}</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == ""
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][0]["arguments"] == {"location": "NYC"}
        assert parsed["error"] is None

    def test_function_call_with_whitespace(self, provider: Llama31MediumUntrained):
        raw = '<function=set_timer>\n  {"duration_minutes": 5}\n</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "set_timer"
        assert parsed["tool_calls"][0]["arguments"] == {"duration_minutes": 5}

    def test_multiple_function_calls(self, provider: Llama31MediumUntrained):
        raw = (
            '<function=get_weather>{"location": "NYC"}</function>'
            '<function=set_timer>{"duration_minutes": 5}</function>'
        )
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 2
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][1]["name"] == "set_timer"

    def test_function_call_with_surrounding_text(self, provider: Llama31MediumUntrained):
        raw = 'Let me check the weather for you.\n<function=get_weather>{"location": "NYC"}</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"

    def test_returns_none_for_clean_jarvis_json(self, provider: Llama31MediumUntrained):
        raw = '{"message":"hello","tool_calls":[],"error":null}'
        result = provider.parse_response(raw)
        assert result is None

    def test_returns_none_for_jarvis_json_with_whitespace(self, provider: Llama31MediumUntrained):
        raw = '  {"message":"hello","tool_calls":[],"error":null}  '
        result = provider.parse_response(raw)
        assert result is None

    def test_plain_text_wrapped_as_jarvis_json(self, provider: Llama31MediumUntrained):
        raw = "The weather in NYC is sunny and 72 degrees."
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == raw
        assert parsed["tool_calls"] == []
        assert parsed["error"] is None

    def test_invalid_function_args_skipped(self, provider: Llama31MediumUntrained):
        raw = '<function=get_weather>not valid json</function>'
        result = provider.parse_response(raw)
        # Invalid JSON args → no parsed calls → falls through to plain text wrapping
        assert result is not None
        parsed = json.loads(result)
        assert parsed["tool_calls"] == []
        assert "<function=" in parsed["message"]

    def test_function_call_without_equals_sign(self, provider: Llama31MediumUntrained):
        """Model sometimes emits <function>name{...}</function> without '='."""
        raw = '<function>get_weather>{"location": "NYC"}</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][0]["arguments"] == {"location": "NYC"}

    def test_param_nesting_unwrapped(self, provider: Llama31MediumUntrained):
        """Model sometimes wraps args in {"param": {...}} — should unwrap."""
        raw = '<function=set_timer>{"param": {"duration_seconds": 150, "label": "test"}}</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["arguments"] == {
            "duration_seconds": 150,
            "label": "test",
        }

    def test_param_scalar_not_unwrapped(self, provider: Llama31MediumUntrained):
        """Don't unwrap {"param": "value"} when value is a scalar (not nested)."""
        raw = '<function=search_web>{"param": "Tesla stock"}</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        # Scalar "param" value is kept as-is (not a nesting case)
        assert parsed["tool_calls"][0]["arguments"] == {"param": "Tesla stock"}

    def test_prompt_includes_anti_hallucination(
        self,
        provider: Llama31MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "NEVER fabricate" in prompt
        assert "MUST call a function" in prompt

    def test_prompt_includes_concrete_example(
        self,
        provider: Llama31MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "<function=get_weather>" in prompt
        assert "Miami" in prompt

    def test_unclosed_function_tag(self, provider: Llama31MediumUntrained):
        """Model sometimes omits the closing </function> tag."""
        raw = '<function=get_sports_scores>{"team_name": "Yankees", "resolved_datetimes": "yesterday"}'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_sports_scores"
        assert parsed["tool_calls"][0]["arguments"]["team_name"] == "Yankees"

    def test_unclosed_function_tag_with_surrounding_text(self, provider: Llama31MediumUntrained):
        """Unclosed function tag with preceding text."""
        raw = 'Let me check that.\n<function=get_weather>{"city": "NYC"}'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"

    def test_resolved_datetimes_string_normalized_to_array(self, provider: Llama31MediumUntrained):
        """resolved_datetimes as string should be wrapped in a list."""
        raw = '<function=get_sports_scores>{"team_name": "Lakers", "resolved_datetimes": "today"}</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["tool_calls"][0]["arguments"]["resolved_datetimes"] == ["today"]

    def test_resolved_datetimes_array_unchanged(self, provider: Llama31MediumUntrained):
        """resolved_datetimes already an array should not be double-wrapped."""
        raw = '<function=get_weather>{"city": "NYC", "resolved_datetimes": ["tomorrow"]}</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["tool_calls"][0]["arguments"]["resolved_datetimes"] == ["tomorrow"]

    def test_trailing_semicolon_stripped(self, provider: Llama31MediumUntrained):
        """Model sometimes appends semicolon after JSON args."""
        raw = '<function=set_timer>{"duration_seconds": 1800, "label": "wake-up"};</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "set_timer"
        assert parsed["tool_calls"][0]["arguments"]["duration_seconds"] == 1800

    def test_trailing_quote_stripped(self, provider: Llama31MediumUntrained):
        """Model sometimes appends extra quote after JSON args."""
        raw = '<function=search_web>{"query": "Next SpaceX launch"}"</function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "search_web"
        assert parsed["tool_calls"][0]["arguments"]["query"] == "Next SpaceX launch"

    def test_malformed_closing_tag(self, provider: Llama31MediumUntrained):
        """Model emits <function> instead of </function> as closing tag."""
        raw = '<function=set_timer>{"duration_seconds": 7200, "label": "2 hour timer"}<function>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "set_timer"
        assert parsed["tool_calls"][0]["arguments"]["duration_seconds"] == 7200
        assert parsed["tool_calls"][0]["arguments"]["label"] == "2 hour timer"
