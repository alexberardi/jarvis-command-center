"""
Tests for voice command processing helpers.

These tests cover the refactored helper functions extracted from
process_voice_command_with_tools for better composition.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional


class TestGetToolName:
    """Tests for the _get_tool_name helper."""

    def test_get_name_from_openai_format(self):
        """Test extracting name from OpenAI function format."""
        from app.core.voice_command_helpers import get_tool_name

        tool = {"function": {"name": "get_weather", "description": "Get weather"}}
        assert get_tool_name(tool) == "get_weather"

    def test_get_name_from_simple_format(self):
        """Test extracting name from simple format."""
        from app.core.voice_command_helpers import get_tool_name

        tool = {"name": "get_weather"}
        assert get_tool_name(tool) == "get_weather"

    def test_get_name_returns_none_for_missing(self):
        """Test returns None when name is missing."""
        from app.core.voice_command_helpers import get_tool_name

        tool = {"description": "Some tool"}
        assert get_tool_name(tool) is None

    def test_get_name_handles_empty_function(self):
        """Test handles empty function dict."""
        from app.core.voice_command_helpers import get_tool_name

        tool = {"function": {}}
        assert get_tool_name(tool) is None


class TestFilterAvailableCommands:
    """Tests for filtering available commands to match tool set."""

    def test_filter_commands_to_match_tools(self):
        """Test filtering available commands to only those in tool set."""
        from app.core.voice_command_helpers import filter_available_commands

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}},
        ]
        available_commands = [
            {"command_name": "get_weather", "allow_direct_answer": True},
            {"command_name": "set_reminder", "allow_direct_answer": False},
            {"command_name": "play_music", "allow_direct_answer": False},
        ]
        # Assume get_weather and set_reminder are client tools (not in registry)
        server_tool_names = set()

        filtered = filter_available_commands(tools, available_commands, server_tool_names)

        assert len(filtered) == 2
        assert any(c["command_name"] == "get_weather" for c in filtered)
        assert any(c["command_name"] == "set_reminder" for c in filtered)
        assert not any(c["command_name"] == "play_music" for c in filtered)

    def test_filter_excludes_server_tools(self):
        """Test that server tools are excluded from available commands."""
        from app.core.voice_command_helpers import filter_available_commands

        tools = [
            {"function": {"name": "resolve_relative_date"}},
            {"function": {"name": "get_weather"}},
        ]
        available_commands = [
            {"command_name": "resolve_relative_date", "allow_direct_answer": False},
            {"command_name": "get_weather", "allow_direct_answer": True},
        ]
        server_tool_names = {"resolve_relative_date"}

        filtered = filter_available_commands(tools, available_commands, server_tool_names)

        assert len(filtered) == 1
        assert filtered[0]["command_name"] == "get_weather"

    def test_filter_handles_empty_commands(self):
        """Test handling empty available commands."""
        from app.core.voice_command_helpers import filter_available_commands

        tools = [{"function": {"name": "get_weather"}}]
        filtered = filter_available_commands(tools, [], set())
        assert filtered == []

    def test_filter_handles_none_commands(self):
        """Test handling None available commands."""
        from app.core.voice_command_helpers import filter_available_commands

        tools = [{"function": {"name": "get_weather"}}]
        filtered = filter_available_commands(tools, None, set())
        assert filtered == []


class TestBuildAvailableCommandFlags:
    """Tests for building command flags for system prompt."""

    def test_build_flags_extracts_relevant_fields(self):
        """Test that only relevant fields are extracted."""
        from app.core.voice_command_helpers import build_available_command_flags

        available_commands = [
            {
                "command_name": "get_weather",
                "allow_direct_answer": True,
                "examples": [{"voice_command": "what's the weather"}],
                "description": "Get weather info",
            }
        ]

        flags = build_available_command_flags(available_commands)

        assert len(flags) == 1
        assert flags[0] == {
            "command_name": "get_weather",
            "allow_direct_answer": True,
        }

    def test_build_flags_handles_missing_allow_direct_answer(self):
        """Test handling commands without allow_direct_answer."""
        from app.core.voice_command_helpers import build_available_command_flags

        available_commands = [{"command_name": "get_weather"}]
        flags = build_available_command_flags(available_commands)

        assert flags[0]["allow_direct_answer"] is None


class TestApplyToolRouting:
    """Tests for applying tool routing decisions."""

    def test_apply_routing_returns_none_when_disabled(self):
        """Test returns None when classifier is disabled."""
        from app.core.voice_command_helpers import apply_tool_routing

        result = apply_tool_routing(
            voice_command="what's the weather",
            tools=[{"function": {"name": "get_weather"}}],
            use_classifier=False,
            route_tool_fn=lambda *args: ("get_weather", 0.9),
        )

        assert result is None

    def test_apply_routing_returns_decision_when_confident(self):
        """Test returns decision when classifier is confident."""
        from app.core.voice_command_helpers import apply_tool_routing

        result = apply_tool_routing(
            voice_command="what's the weather",
            tools=[{"function": {"name": "get_weather"}}],
            use_classifier=True,
            route_tool_fn=lambda *args: ("get_weather", 0.9),
            min_confidence=0.6,
        )

        assert result is not None
        assert result["tool_name"] == "get_weather"
        assert result["score"] == 0.9
        assert result["used"] is True

    def test_apply_routing_not_used_when_below_threshold(self):
        """Test decision not used when below confidence threshold."""
        from app.core.voice_command_helpers import apply_tool_routing

        result = apply_tool_routing(
            voice_command="what's the weather",
            tools=[{"function": {"name": "get_weather"}}],
            use_classifier=True,
            route_tool_fn=lambda *args: ("get_weather", 0.5),
            min_confidence=0.6,
        )

        assert result is not None
        assert result["used"] is False

    def test_apply_routing_returns_none_when_no_prediction(self):
        """Test returns None when classifier has no prediction."""
        from app.core.voice_command_helpers import apply_tool_routing

        result = apply_tool_routing(
            voice_command="what's the weather",
            tools=[{"function": {"name": "get_weather"}}],
            use_classifier=True,
            route_tool_fn=lambda *args: None,
        )

        assert result is None


class TestPruneToolsByRouterDecision:
    """Tests for pruning tools based on high-confidence router decision."""

    def test_prune_keeps_predicted_and_server_tools(self):
        """Test pruning keeps predicted tool and server tools."""
        from app.core.voice_command_helpers import prune_tools_by_router_decision

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}},
            {"function": {"name": "resolve_relative_date"}},
        ]
        router_decision = {"tool_name": "get_weather", "score": 0.9}
        server_tool_names = {"resolve_relative_date"}

        pruned = prune_tools_by_router_decision(
            tools=tools,
            router_decision=router_decision,
            server_tool_names=server_tool_names,
            prune_confidence=0.85,
        )

        assert len(pruned) == 2
        names = [t["function"]["name"] for t in pruned]
        assert "get_weather" in names
        assert "resolve_relative_date" in names
        assert "set_reminder" not in names

    def test_prune_returns_all_when_below_threshold(self):
        """Test returns all tools when confidence below threshold."""
        from app.core.voice_command_helpers import prune_tools_by_router_decision

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}},
        ]
        router_decision = {"tool_name": "get_weather", "score": 0.7}

        pruned = prune_tools_by_router_decision(
            tools=tools,
            router_decision=router_decision,
            server_tool_names=set(),
            prune_confidence=0.85,
        )

        assert len(pruned) == 2

    def test_prune_returns_all_when_no_decision(self):
        """Test returns all tools when no router decision."""
        from app.core.voice_command_helpers import prune_tools_by_router_decision

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}},
        ]

        pruned = prune_tools_by_router_decision(
            tools=tools,
            router_decision=None,
            server_tool_names=set(),
            prune_confidence=0.85,
        )

        assert len(pruned) == 2

    def test_prune_returns_all_when_would_be_empty(self):
        """Test returns all tools when pruning would result in empty."""
        from app.core.voice_command_helpers import prune_tools_by_router_decision

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}},
        ]
        # Predicted tool not in list
        router_decision = {"tool_name": "play_music", "score": 0.95}

        pruned = prune_tools_by_router_decision(
            tools=tools,
            router_decision=router_decision,
            server_tool_names=set(),
            prune_confidence=0.85,
        )

        # Should return all tools since pruning would leave nothing useful
        assert len(pruned) == 2
