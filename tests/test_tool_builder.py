"""
Tests for ToolBuilder utility.

Validates:
- Building clean OpenAI-format tool definitions
- Stripping Jarvis extensions
- Description truncation
"""

from app.core.tool_builder import ToolBuilder


class TestToolBuilderBuild:
    """Test ToolBuilder.build()."""

    def test_build_produces_clean_openai_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
                "allow_direct_answer": False,  # Jarvis extension
            }
        ]
        result = ToolBuilder.build(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        func = result[0]["function"]
        assert func["name"] == "get_weather"
        assert func["description"] == "Get weather for a city."
        assert "city" in func["parameters"]["properties"]

    def test_build_skips_tools_without_name(self):
        tools = [
            {"function": {"description": "No name"}},
            {"function": {"name": "valid_tool"}},
        ]
        result = ToolBuilder.build(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "valid_tool"

    def test_build_skips_empty_function(self):
        tools = [{"function": {}}, {"not_a_function": True}]
        result = ToolBuilder.build(tools)
        assert len(result) == 0

    def test_build_without_descriptions(self):
        tools = [
            {
                "function": {
                    "name": "test_tool",
                    "description": "This should be excluded.",
                    "parameters": {"type": "object"},
                },
            }
        ]
        result = ToolBuilder.build(tools, include_descriptions=False)
        assert len(result) == 1
        assert "description" not in result[0]["function"]

    def test_build_truncates_descriptions(self):
        tools = [
            {
                "function": {
                    "name": "test_tool",
                    "description": "A" * 200,
                    "parameters": {"type": "object"},
                },
            }
        ]
        result = ToolBuilder.build(tools, max_description_length=50)
        desc = result[0]["function"]["description"]
        assert len(desc) <= 50
        assert desc.endswith("...")

    def test_build_empty_list(self):
        assert ToolBuilder.build([]) == []

    def test_build_multiple_tools(self):
        tools = [
            {"function": {"name": "tool_a", "parameters": {}}},
            {"function": {"name": "tool_b", "parameters": {}}},
            {"function": {"name": "tool_c", "parameters": {}}},
        ]
        result = ToolBuilder.build(tools)
        assert len(result) == 3
        names = [t["function"]["name"] for t in result]
        assert names == ["tool_a", "tool_b", "tool_c"]


class TestToolBuilderStripExtensions:
    """Test ToolBuilder.strip_jarvis_extensions()."""

    def test_strip_removes_jarvis_keys(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "test"},
                "allow_direct_answer": False,
                "is_server_tool": True,
                "command_name": "test_cmd",
                "keywords": ["test"],
                "examples": [{"voice_command": "test"}],
            }
        ]
        result = ToolBuilder.strip_jarvis_extensions(tools)
        assert len(result) == 1
        assert "allow_direct_answer" not in result[0]
        assert "is_server_tool" not in result[0]
        assert "command_name" not in result[0]
        assert "keywords" not in result[0]
        assert "examples" not in result[0]
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test"

    def test_strip_preserves_openai_keys(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool",
                    "description": "desc",
                    "parameters": {"type": "object"},
                },
            }
        ]
        result = ToolBuilder.strip_jarvis_extensions(tools)
        assert result == tools

    def test_strip_empty_list(self):
        assert ToolBuilder.strip_jarvis_extensions([]) == []
