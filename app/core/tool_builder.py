"""
ToolBuilder - Utility for building OpenAI-format tool definitions.

Standalone utility, not tied to any provider. Converts raw Jarvis tool
definitions into clean OpenAI function-calling format.
"""

from typing import Any, Dict, List, Optional


# Keys that are Jarvis-specific extensions (not part of OpenAI spec)
_JARVIS_EXTENSION_KEYS = {
    "allow_direct_answer",
    "is_server_tool",
    "command_name",
    "keywords",
    "examples",
}


class ToolBuilder:
    """Builds clean OpenAI-format tool definitions from Jarvis tool definitions."""

    @staticmethod
    def build(
        tools: List[Dict[str, Any]],
        include_descriptions: bool = True,
        max_description_length: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build clean OpenAI-format tool definitions.

        Args:
            tools: Raw tool definitions (may include Jarvis extensions).
            include_descriptions: Whether to include tool/param descriptions.
            max_description_length: Truncate descriptions to this length.

        Returns:
            List of OpenAI-format tool definitions:
            [{"type": "function", "function": {"name", "description", "parameters"}}]
        """
        result: List[Dict[str, Any]] = []
        for tool in tools:
            func: Dict[str, Any] = tool.get("function", {})
            if not func or not func.get("name"):
                continue

            clean_func: Dict[str, Any] = {"name": func["name"]}

            # Description
            if include_descriptions and func.get("description"):
                desc: str = func["description"]
                if max_description_length and len(desc) > max_description_length:
                    desc = desc[:max_description_length - 3].rstrip() + "..."
                clean_func["description"] = desc

            # Parameters
            if func.get("parameters"):
                clean_func["parameters"] = func["parameters"]

            result.append({"type": "function", "function": clean_func})

        return result

    @staticmethod
    def strip_jarvis_extensions(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove Jarvis-specific keys, return pure OpenAI format.

        Strips top-level keys like allow_direct_answer, is_server_tool, etc.
        that are Jarvis conventions but not part of the OpenAI tool spec.

        Args:
            tools: Tool definitions potentially containing Jarvis extensions.

        Returns:
            Clean OpenAI-format tool definitions.
        """
        result: List[Dict[str, Any]] = []
        for tool in tools:
            clean: Dict[str, Any] = {}
            for key, value in tool.items():
                if key not in _JARVIS_EXTENSION_KEYS:
                    clean[key] = value
            result.append(clean)
        return result
