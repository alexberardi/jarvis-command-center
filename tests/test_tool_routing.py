"""
Tests for Tool Routing.

These tests cover the tool routing functionality extracted from model_service.py.
Following TDD: write tests first, then implement.
"""
import os
import pytest
from unittest.mock import MagicMock, patch


class TestToolClassifierSingleton:
    """Tests for the classifier singleton behavior."""

    def test_get_shared_classifier_returns_none_when_disabled(self):
        """Test that classifier returns None when env var is disabled."""
        from app.core.tool_routing import get_shared_tool_classifier, reset_classifier

        # Reset to ensure clean state
        reset_classifier()

        with patch.dict(os.environ, {"JARVIS_TOOL_CLASSIFIER_ENABLED": "false"}):
            classifier = get_shared_tool_classifier()
            assert classifier is None

    def test_get_shared_classifier_singleton_behavior(self):
        """Test that classifier is reused (singleton)."""
        from app.core.tool_routing import get_shared_tool_classifier, reset_classifier

        reset_classifier()

        # Mock the FastTextToolClassifier at import location
        mock_classifier = MagicMock()

        with patch.dict(os.environ, {"JARVIS_TOOL_CLASSIFIER_ENABLED": "true"}):
            with patch("app.core.tool_router.classifier.FastTextToolClassifier") as MockClass:
                MockClass.from_env.return_value = mock_classifier

                first_call = get_shared_tool_classifier()
                second_call = get_shared_tool_classifier()

                # Should be the same instance
                assert first_call is second_call
                # from_env should only be called once
                assert MockClass.from_env.call_count == 1


class TestRouteToolFunction:
    """Tests for the route_tool function."""

    def test_route_tool_with_no_classifier(self):
        """Test that route_tool returns None when classifier is None."""
        from app.core.tool_routing import route_tool

        result = route_tool("what's the weather", [], None)
        assert result is None

    def test_route_tool_with_empty_utterance(self):
        """Test that route_tool returns None for empty utterance."""
        from app.core.tool_routing import route_tool

        mock_classifier = MagicMock()
        result = route_tool("", [], mock_classifier)
        assert result is None

    def test_route_tool_returns_prediction(self):
        """Test that route_tool returns prediction from classifier."""
        from app.core.tool_routing import route_tool

        mock_classifier = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.tool_name = "get_weather"
        mock_prediction.score = 0.95
        mock_classifier.predict.return_value = mock_prediction

        tools = [{"function": {"name": "get_weather"}}]
        result = route_tool("what's the weather", tools, mock_classifier)

        assert result is not None
        assert result[0] == "get_weather"
        assert result[1] == 0.95

    def test_route_tool_filters_unknown_tools(self):
        """Test that route_tool filters predictions for tools not in list."""
        from app.core.tool_routing import route_tool

        mock_classifier = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.tool_name = "unknown_tool"
        mock_prediction.score = 0.95
        mock_classifier.predict.return_value = mock_prediction

        tools = [{"function": {"name": "get_weather"}}]
        result = route_tool("what's the weather", tools, mock_classifier)

        # Should return None because predicted tool is not in available tools
        assert result is None

    def test_route_tool_handles_simple_tool_format(self):
        """Test that route_tool handles tools with simple name format."""
        from app.core.tool_routing import route_tool

        mock_classifier = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.tool_name = "simple_tool"
        mock_prediction.score = 0.85
        mock_classifier.predict.return_value = mock_prediction

        tools = [{"name": "simple_tool"}]
        result = route_tool("use simple tool", tools, mock_classifier)

        assert result is not None
        assert result[0] == "simple_tool"


class TestFilterToolsForUtterance:
    """Tests for the filter_tools_for_utterance function."""

    def test_filter_returns_all_tools(self):
        """Test that filter currently returns all tools (filtering disabled)."""
        from app.core.tool_routing import filter_tools_for_utterance

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}}
        ]

        result = filter_tools_for_utterance("what's the weather", tools, None)

        # Should return all tools (filtering is disabled)
        assert len(result) == 2


class TestExtractToolNames:
    """Tests for extracting tool names from tool list."""

    def test_extract_from_openai_format(self):
        """Test extracting names from OpenAI function format."""
        from app.core.tool_routing import extract_tool_names

        tools = [
            {"function": {"name": "tool_a"}},
            {"function": {"name": "tool_b"}}
        ]

        names = extract_tool_names(tools)
        assert "tool_a" in names
        assert "tool_b" in names

    def test_extract_from_simple_format(self):
        """Test extracting names from simple format."""
        from app.core.tool_routing import extract_tool_names

        tools = [{"name": "simple_tool"}]

        names = extract_tool_names(tools)
        assert "simple_tool" in names

    def test_extract_handles_empty_list(self):
        """Test extracting from empty list."""
        from app.core.tool_routing import extract_tool_names

        names = extract_tool_names([])
        assert len(names) == 0

    def test_extract_handles_none(self):
        """Test extracting from None."""
        from app.core.tool_routing import extract_tool_names

        names = extract_tool_names(None)
        assert len(names) == 0


class TestPruneToolsByConfidence:
    """Tests for pruning tools by confidence threshold."""

    def test_prune_at_high_confidence(self):
        """Test pruning keeps only predicted tool and server tools at high confidence."""
        from app.core.tool_routing import prune_tools_by_confidence

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}},
            {"function": {"name": "resolve_relative_date"}}  # Server tool
        ]
        server_tool_names = {"resolve_relative_date"}

        pruned = prune_tools_by_confidence(
            tools=tools,
            predicted_tool="get_weather",
            confidence=0.9,
            confidence_threshold=0.85,
            server_tool_names=server_tool_names
        )

        # Should keep get_weather (predicted) and resolve_relative_date (server)
        pruned_names = [t.get("function", {}).get("name") for t in pruned]
        assert "get_weather" in pruned_names
        assert "resolve_relative_date" in pruned_names
        assert "set_reminder" not in pruned_names

    def test_no_pruning_below_threshold(self):
        """Test that no pruning happens below confidence threshold."""
        from app.core.tool_routing import prune_tools_by_confidence

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}}
        ]

        pruned = prune_tools_by_confidence(
            tools=tools,
            predicted_tool="get_weather",
            confidence=0.7,  # Below threshold
            confidence_threshold=0.85,
            server_tool_names=set()
        )

        # Should return all tools (no pruning)
        assert len(pruned) == 2

    def test_prune_returns_none_when_no_tools_left(self):
        """Test that pruning returns None when result would be empty."""
        from app.core.tool_routing import prune_tools_by_confidence

        tools = [
            {"function": {"name": "get_weather"}}
        ]

        pruned = prune_tools_by_confidence(
            tools=tools,
            predicted_tool="unknown_tool",  # Not in list
            confidence=0.9,
            confidence_threshold=0.85,
            server_tool_names=set()
        )

        # Should return None (would result in empty list)
        assert pruned is None


class TestCollectToolKeywords:
    """Tests for collect_tool_keywords (force-tool-calls keyword gate)."""

    def test_collects_from_command_definitions(self):
        from app.core.tool_routing import collect_tool_keywords

        commands = [
            {"command_name": "medication", "keywords": ["medicine", "meds", "took my"]},
            {"command_name": "no_keywords"},
        ]
        assert collect_tool_keywords(commands) == ["medicine", "meds", "took my"]

    def test_collects_from_openai_style_tool_defs(self):
        from app.core.tool_routing import collect_tool_keywords

        tools = [
            {"function": {"name": "control_device", "keywords": ["lights", "turn on"]}},
            {"function": {"name": "recall"}},
        ]
        assert collect_tool_keywords(tools) == ["lights", "turn on"]

    def test_merges_and_deduplicates_multiple_sources(self):
        from app.core.tool_routing import collect_tool_keywords

        commands = [{"command_name": "a", "keywords": ["lights"]}]
        tools = [{"function": {"name": "a", "keywords": ["lights", "dim"]}}]
        assert collect_tool_keywords(commands, tools) == ["lights", "dim"]

    def test_none_sources_and_malformed_entries_are_ignored(self):
        from app.core.tool_routing import collect_tool_keywords

        commands = [
            {"command_name": "bad_type", "keywords": "not-a-list"},
            {"command_name": "mixed", "keywords": ["ok", 42, None, "  "]},
            "not-a-dict",
        ]
        assert collect_tool_keywords(None, commands, []) == ["ok"]


class TestUtteranceMatchesKeywords:
    """Tests for utterance_matches_keywords (force-tool-calls keyword gate)."""

    def test_conversational_ack_matches_nothing(self):
        from app.core.tool_routing import utterance_matches_keywords

        keywords = ["medicine", "meds", "lights", "calendar", "joke"]
        for utterance in ["Thank you.", "Okay.", "Oh.", "Goodnight.", "Mm-hmm.", "Really good."]:
            assert utterance_matches_keywords(utterance, keywords) is False

    def test_single_word_keyword_matches_on_word_boundary(self):
        from app.core.tool_routing import utterance_matches_keywords

        assert utterance_matches_keywords("Leo took his medicine.", ["medicine"]) is True
        # "night" must not match inside "goodnight"
        assert utterance_matches_keywords("goodnight", ["night"]) is False

    def test_multi_word_keyword_matches_as_phrase(self):
        from app.core.tool_routing import utterance_matches_keywords

        assert utterance_matches_keywords("I took my pills already", ["took my"]) is True
        assert utterance_matches_keywords("he took someone's advice", ["took my"]) is False

    def test_short_keywords_are_ignored(self):
        from app.core.tool_routing import utterance_matches_keywords

        # "to" and "hi" are shorter than the minimum meaningful length and
        # would otherwise flag ordinary chatter as tool-shaped.
        assert utterance_matches_keywords("I'm going to go.", ["to", "hi"]) is False

    def test_matching_is_case_and_punctuation_insensitive(self):
        from app.core.tool_routing import utterance_matches_keywords

        assert utterance_matches_keywords("Turn on the LIGHTS!", ["lights"]) is True
        assert utterance_matches_keywords("What's on my calendar?", ["calendar"]) is True

    def test_empty_inputs_do_not_match(self):
        from app.core.tool_routing import utterance_matches_keywords

        assert utterance_matches_keywords("", ["lights"]) is False
        assert utterance_matches_keywords(None, ["lights"]) is False
        assert utterance_matches_keywords("turn on the lights", []) is False
        assert utterance_matches_keywords("!!!", ["lights"]) is False
