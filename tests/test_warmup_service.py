"""
Tests for Warmup Service.

These tests cover the warmup functionality extracted from model_service.py.
Following TDD: write tests first, then implement.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestClientToolsExtraction:
    """Tests for extracting information from client tools."""

    def test_extract_tool_name_openai_format(self):
        """Test extracting tool name from OpenAI function format."""
        from app.core.warmup_service import WarmupService

        tool = {
            "function": {
                "name": "get_weather",
                "description": "Get weather"
            }
        }

        service = WarmupService()
        name = service._get_tool_name(tool)
        assert name == "get_weather"

    def test_extract_tool_name_simple_format(self):
        """Test extracting tool name from simple format."""
        from app.core.warmup_service import WarmupService

        tool = {"name": "get_weather"}

        service = WarmupService()
        name = service._get_tool_name(tool)
        assert name == "get_weather"

    def test_extract_tool_name_missing(self):
        """Test handling missing tool name."""
        from app.core.warmup_service import WarmupService

        tool = {"description": "Some tool"}

        service = WarmupService()
        name = service._get_tool_name(tool)
        assert name is None


class TestExamplesMapBuilding:
    """Tests for building examples map from client tools."""

    def test_build_examples_from_client_tools(self):
        """Test extracting examples from client tools."""
        from app.core.warmup_service import WarmupService

        client_tools = [{
            "function": {"name": "set_reminder"},
            "examples": [
                {"voice_command": "remind me tomorrow", "expected_parameters": {}, "is_primary": True},
                {"voice_command": "set a reminder", "expected_parameters": {}, "is_primary": False}
            ]
        }]

        service = WarmupService()
        examples_map = service.build_examples_map(client_tools)

        assert "set_reminder" in examples_map
        assert len(examples_map["set_reminder"]["examples"]) == 2
        assert examples_map["set_reminder"]["examples"][0]["voice_command"] == "remind me tomorrow"

    def test_build_examples_empty_tools(self):
        """Test with no client tools."""
        from app.core.warmup_service import WarmupService

        service = WarmupService()
        examples_map = service.build_examples_map([])

        assert examples_map == {}

    def test_build_examples_no_examples_array(self):
        """Test client tool without examples."""
        from app.core.warmup_service import WarmupService

        client_tools = [{
            "function": {"name": "get_weather"}
            # No examples array
        }]

        service = WarmupService()
        examples_map = service.build_examples_map(client_tools)

        # Tool is not added to map if it has no examples
        assert "get_weather" not in examples_map

    def test_build_examples_includes_metadata(self):
        """Test that metadata like keywords and allow_direct_answer are included."""
        from app.core.warmup_service import WarmupService

        client_tools = [{
            "function": {"name": "calculate"},
            "keywords": ["math", "calculate"],
            "allow_direct_answer": True,
            "examples": [{"voice_command": "what is 2+2", "expected_parameters": {}, "is_primary": True}]
        }]

        service = WarmupService()
        examples_map = service.build_examples_map(client_tools)

        assert examples_map["calculate"]["keywords"] == ["math", "calculate"]
        assert examples_map["calculate"]["allow_direct_answer"] is True


class TestAntipatternFiltering:
    """Tests for filtering antipatterns."""

    def test_filter_valid_antipatterns(self):
        """Test filtering keeps valid antipatterns."""
        from app.core.warmup_service import WarmupService

        antipatterns = [
            {"command_name": "get_weather", "description": "Don't use for indoor temperature"},
            {"command_name": "get_weather", "description": "Not for historical data"}
        ]
        available_command_names = {"get_weather", "set_reminder"}

        service = WarmupService()
        filtered = service.filter_antipatterns(antipatterns, available_command_names, "test")

        assert len(filtered) == 2

    def test_filter_removes_invalid_command_names(self):
        """Test filtering removes antipatterns for non-existent commands."""
        from app.core.warmup_service import WarmupService

        antipatterns = [
            {"command_name": "unknown_command", "description": "Should be removed"}
        ]
        available_command_names = {"get_weather"}

        service = WarmupService()
        filtered = service.filter_antipatterns(antipatterns, available_command_names, "test")

        assert len(filtered) == 0

    def test_filter_removes_empty_descriptions(self):
        """Test filtering removes antipatterns with empty descriptions."""
        from app.core.warmup_service import WarmupService

        antipatterns = [
            {"command_name": "get_weather", "description": ""},
            {"command_name": "get_weather", "description": "   "}
        ]
        available_command_names = {"get_weather"}

        service = WarmupService()
        filtered = service.filter_antipatterns(antipatterns, available_command_names, "test")

        assert len(filtered) == 0

    def test_filter_handles_none(self):
        """Test filtering handles None input."""
        from app.core.warmup_service import WarmupService

        service = WarmupService()
        filtered = service.filter_antipatterns(None, {"get_weather"}, "test")

        assert filtered == []


class TestToolMerging:
    """Tests for merging server and client tools."""

    def test_merge_server_and_client_tools(self):
        """Test merging server and client tools."""
        from app.core.warmup_service import WarmupService

        server_tools = [{"function": {"name": "resolve_relative_date"}}]
        client_tools = [{"function": {"name": "get_weather"}}]

        service = WarmupService()
        merged = service.merge_tools(server_tools, client_tools)

        assert len(merged) == 2

    def test_merge_with_empty_client_tools(self):
        """Test merging with no client tools."""
        from app.core.warmup_service import WarmupService

        server_tools = [{"function": {"name": "resolve_relative_date"}}]

        service = WarmupService()
        merged = service.merge_tools(server_tools, None)

        assert len(merged) == 1
        assert merged[0]["function"]["name"] == "resolve_relative_date"


class TestAvailableCommandNames:
    """Tests for building available command names set."""

    def test_get_available_command_names_from_tools(self):
        """Test extracting command names from tools."""
        from app.core.warmup_service import WarmupService

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "set_reminder"}},
            {"name": "simple_tool"}
        ]

        service = WarmupService()
        names = service.get_available_command_names(tools, None)

        assert "get_weather" in names
        assert "set_reminder" in names
        assert "simple_tool" in names

    def test_get_available_command_names_with_available_commands(self):
        """Test that available_commands are also included."""
        from app.core.warmup_service import WarmupService

        tools = [{"function": {"name": "get_weather"}}]

        # Create mock CommandDefinition objects
        class MockCommand:
            def __init__(self, name):
                self.command_name = name

        available_commands = [MockCommand("additional_command")]

        service = WarmupService()
        names = service.get_available_command_names(tools, available_commands)

        assert "get_weather" in names
        assert "additional_command" in names


class TestProcessClientTools:
    """Tests for the full client tools processing pipeline."""

    def test_process_client_tools_adds_filtered_antipatterns(self):
        """Test that antipatterns are filtered and attached to tools."""
        from app.core.warmup_service import WarmupService

        client_tools = [{
            "function": {"name": "get_weather"},
            "antipatterns": [
                {"command_name": "get_weather", "description": "Valid antipattern"},
                {"command_name": "unknown", "description": "Should be filtered"}
            ]
        }]
        available_command_names = {"get_weather"}

        service = WarmupService()
        service.process_client_tools(client_tools, available_command_names)

        # The tool should have filtered antipatterns
        assert len(client_tools[0]["antipatterns"]) == 1
        assert client_tools[0]["antipatterns"][0]["command_name"] == "get_weather"


class TestMergeAvailableCommands:
    """Tests for merging examples from available_commands into the map."""

    def test_merge_examples_from_available_commands(self):
        """Test merging examples when command already exists."""
        from app.core.warmup_service import WarmupService

        examples_map = {
            "get_weather": {
                "command_name": "get_weather",
                "examples": [{"voice_command": "what's the weather"}]
            }
        }

        class MockCommand:
            def __init__(self):
                self.command_name = "get_weather"

            def model_dump(self):
                return {
                    "command_name": "get_weather",
                    "examples": [{"voice_command": "weather forecast"}]
                }

        available_commands = [MockCommand()]
        available_command_names = {"get_weather"}

        service = WarmupService()
        result = service.merge_available_commands(examples_map, available_commands, available_command_names)

        # Should have merged examples
        assert len(result) == 1
        assert len([c for c in result if c["command_name"] == "get_weather"][0]["examples"]) == 2

    def test_merge_adds_new_commands(self):
        """Test that new commands from available_commands are added."""
        from app.core.warmup_service import WarmupService

        examples_map = {}

        class MockCommand:
            def __init__(self):
                self.command_name = "new_command"

            def model_dump(self):
                return {
                    "command_name": "new_command",
                    "examples": [{"voice_command": "use new command"}]
                }

        available_commands = [MockCommand()]
        available_command_names = {"new_command"}

        service = WarmupService()
        result = service.merge_available_commands(examples_map, available_commands, available_command_names)

        assert any(c["command_name"] == "new_command" for c in result)
