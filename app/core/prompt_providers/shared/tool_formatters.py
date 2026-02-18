"""
Tool formatting utilities for prompt providers.

Thin delegation layer to tool_call_parser.format_tools_for_prompt().
Exists so prompt providers import from a consistent shared/ namespace
without directly coupling to tool_call_parser internals.
"""

from typing import Any, Dict, List, Optional

from app.core.tool_call_parser import tool_call_parser


def format_tools_for_prompt(
    tools: List[Dict[str, Any]],
    available_commands: Optional[List[Dict[str, Any]]] = None,
    primary_examples_only: bool = False,
) -> str:
    """
    Format tool definitions for inclusion in a system prompt.

    Args:
        tools: Tool definitions in OpenAI function-calling format.
        available_commands: Optional command flag dicts for examples.
        primary_examples_only: If True, only include is_primary examples
            (used by adapter-trained providers).

    Returns:
        Formatted tools section string.
    """
    return tool_call_parser.format_tools_for_prompt(
        tools, available_commands, primary_examples_only=primary_examples_only
    )
