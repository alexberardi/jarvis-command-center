"""
Voice Command Processing Helpers.

This module provides helper functions for processing voice commands,
extracted from ModelService.process_voice_command_with_tools for
better composition and testability.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("uvicorn")


def get_tool_name(tool: Dict[str, Any]) -> Optional[str]:
    """
    Extract tool name from a tool definition.

    Handles both OpenAI function format and simple format.

    Args:
        tool: Tool definition dict

    Returns:
        Tool name or None if not found
    """
    if isinstance(tool.get("function"), dict):
        return tool.get("function", {}).get("name")
    return tool.get("name")


def filter_available_commands(
    tools: List[Dict[str, Any]],
    available_commands: Optional[List[Dict[str, Any]]],
    server_tool_names: Set[str],
) -> List[Dict[str, Any]]:
    """
    Filter available commands to only those matching the tool set.

    Excludes server tools since they don't have corresponding available_commands.

    Args:
        tools: Current tool set
        available_commands: All available command definitions
        server_tool_names: Set of server tool names to exclude

    Returns:
        Filtered list of available commands
    """
    if not available_commands:
        return []

    # Get client tool names (exclude server tools)
    client_tool_names = {
        get_tool_name(tool)
        for tool in tools
        if get_tool_name(tool) and get_tool_name(tool) not in server_tool_names
    }

    return [
        cmd
        for cmd in available_commands
        if cmd.get("command_name") in client_tool_names
    ]


def build_available_command_flags(
    available_commands: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build command flags for system prompt from available commands.

    Extracts only the fields needed for the system prompt.

    Args:
        available_commands: List of available command definitions

    Returns:
        List of command flag dicts with command_name and allow_direct_answer
    """
    return [
        {
            "command_name": cmd.get("command_name"),
            "allow_direct_answer": cmd.get("allow_direct_answer"),
        }
        for cmd in available_commands
    ]


def apply_tool_routing(
    voice_command: str,
    tools: List[Dict[str, Any]],
    use_classifier: bool,
    route_tool_fn: Callable[[str, List[Dict[str, Any]]], Optional[tuple]],
    min_confidence: float = 0.6,
) -> Optional[Dict[str, Any]]:
    """
    Apply tool routing classifier to predict the best tool.

    Args:
        voice_command: The user's voice command
        tools: Available tools
        use_classifier: Whether to use the classifier
        route_tool_fn: Function to call for routing (returns (tool_name, score) or None)
        min_confidence: Minimum confidence to mark decision as "used"

    Returns:
        Router decision dict or None if classifier disabled/no prediction
    """
    if not use_classifier:
        return None

    routed = route_tool_fn(voice_command, tools)
    if not routed:
        return None

    tool_name, score = routed
    use_hint = score >= min_confidence

    return {
        "tool_name": tool_name,
        "score": score,
        "used": use_hint,
    }


def prune_tools_by_router_decision(
    tools: List[Dict[str, Any]],
    router_decision: Optional[Dict[str, Any]],
    server_tool_names: Set[str],
    prune_confidence: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Prune tools based on high-confidence router decision.

    When the router is highly confident, we can reduce the tool set to
    just the predicted tool and server tools, reducing prompt size.

    Args:
        tools: All available tools
        router_decision: Router decision dict with tool_name and score
        server_tool_names: Set of server tool names to always keep
        prune_confidence: Minimum confidence to enable pruning

    Returns:
        Pruned tool list, or original list if pruning not applicable
    """
    if not router_decision:
        return tools

    score = router_decision.get("score", 0.0)
    if score < prune_confidence:
        return tools

    predicted_tool = router_decision.get("tool_name")

    pruned = []
    for tool in tools:
        name = get_tool_name(tool)
        if not name:
            continue
        if name == predicted_tool or name in server_tool_names:
            pruned.append(tool)

    # If pruning would leave no useful tools, return original
    if not pruned or (len(pruned) == len([t for t in tools if get_tool_name(t) in server_tool_names])):
        return tools

    if len(pruned) < len(tools):
        logger.info(
            "🧹 Pruned tools from %d to %d (predicted=%s, confidence=%.3f)",
            len(tools),
            len(pruned),
            predicted_tool,
            score,
        )

    return pruned
