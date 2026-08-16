"""
Tool Routing for Jarvis Voice Assistant.

This module provides tool routing functionality using a FastText classifier
to predict which tool should handle a given utterance.
"""

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("uvicorn")

# Module-level singleton for tool classifier (expensive to load, reuse across requests)
_tool_classifier_instance = None


def reset_classifier() -> None:
    """Reset the classifier singleton (for testing)."""
    global _tool_classifier_instance
    _tool_classifier_instance = None


def get_shared_tool_classifier():
    """
    Get or create the shared tool classifier singleton.

    Returns:
        FastTextToolClassifier instance or None if disabled/unavailable
    """
    global _tool_classifier_instance
    if _tool_classifier_instance is not None:
        return _tool_classifier_instance
    if os.getenv("JARVIS_TOOL_CLASSIFIER_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return None
    try:
        from app.core.tool_router.classifier import FastTextToolClassifier
    except Exception as exc:
        logger.warning("Tool classifier unavailable: %s", exc)
        return None
    classifier = FastTextToolClassifier.from_env()
    if classifier is None:
        return None
    _tool_classifier_instance = classifier
    logger.info("Tool classifier loaded (singleton)")
    return _tool_classifier_instance


def extract_tool_names(tools: Optional[List[Dict[str, Any]]]) -> Set[str]:
    """
    Extract tool names from a list of tool definitions.

    Handles both OpenAI function format and simple format.

    Args:
        tools: List of tool definitions

    Returns:
        Set of tool names
    """
    if not tools:
        return set()

    names = set()
    for tool in tools:
        if isinstance(tool.get("function"), dict):
            name = tool.get("function", {}).get("name")
        else:
            name = tool.get("name")
        if name:
            names.add(name)
    return names


def route_tool(
    utterance: str,
    tools: List[Dict[str, Any]],
    classifier
) -> Optional[Tuple[str, float]]:
    """
    Route an utterance to a tool using the classifier.

    Args:
        utterance: The user's voice command
        tools: Available tool definitions
        classifier: FastTextToolClassifier instance or None

    Returns:
        Tuple of (tool_name, confidence_score) or None if no prediction
    """
    if classifier is None or not utterance:
        return None

    tool_names = extract_tool_names(tools)
    prediction = classifier.predict(utterance)

    if not prediction.tool_name or prediction.tool_name not in tool_names:
        return None

    return prediction.tool_name, prediction.score


def filter_tools_for_utterance(
    voice_command: str,
    tools: List[Dict[str, Any]],
    available_commands: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Filter tools based on the utterance.

    Currently returns all tools (keyword-based filtering is disabled).

    Args:
        voice_command: The user's voice command
        tools: Available tool definitions
        available_commands: Optional available commands list

    Returns:
        Filtered list of tools (currently returns all)
    """
    # Keyword-based filtering disabled: always keep the full tool list
    return tools


# Minimum keyword length considered meaningful for utterance matching.
# Ultra-short keywords ("to", "in", "hi") appear in ordinary conversation
# and would make every chatty turn look tool-shaped.
_KEYWORD_MATCH_MIN_LEN = 3


def _normalize_for_keyword_match(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, pad with spaces.

    The space padding lets multi-word keywords match on word boundaries
    with a plain substring test ("took my" matches "Leo took my pills"
    but "night" does not match "goodnight").
    """
    collapsed = " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())
    return f" {collapsed} "


def collect_tool_keywords(
    *sources: Optional[Iterable[Dict[str, Any]]],
) -> List[str]:
    """
    Collect declared keywords from command/tool definition dicts.

    Accepts any number of iterables of dicts in either shape:
    - command definitions with a top-level ``keywords`` list
    - OpenAI-style tool defs with keywords under ``function``

    Returns:
        Deduplicated list of keyword strings (original casing preserved).
    """
    seen: Set[str] = set()
    keywords: List[str] = []
    for source in sources:
        for item in source or []:
            if not isinstance(item, dict):
                continue
            raw = item.get("keywords")
            if raw is None and isinstance(item.get("function"), dict):
                raw = item["function"].get("keywords")
            if not isinstance(raw, list):
                continue
            for kw in raw:
                if isinstance(kw, str) and kw.strip() and kw not in seen:
                    seen.add(kw)
                    keywords.append(kw)
    return keywords


def utterance_matches_keywords(
    utterance: Optional[str],
    keywords: Iterable[str],
) -> bool:
    """
    Check whether an utterance contains any command keyword.

    Deterministic word-boundary matching: keywords and utterance are both
    lowercased and punctuation-stripped, then keywords must appear as whole
    words (multi-word keywords as whole-word phrases). Keywords shorter than
    ``_KEYWORD_MATCH_MIN_LEN`` are ignored — they match ordinary chatter.

    Args:
        utterance: The user's voice command
        keywords: Keyword strings declared by commands/tools

    Returns:
        True if any meaningful keyword appears in the utterance.
    """
    if not utterance:
        return False
    normalized_utterance = _normalize_for_keyword_match(utterance)
    if not normalized_utterance.strip():
        return False
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        normalized_kw = _normalize_for_keyword_match(kw).strip()
        if len(normalized_kw) < _KEYWORD_MATCH_MIN_LEN:
            continue
        if f" {normalized_kw} " in normalized_utterance:
            return True
    return False


def prune_tools_by_confidence(
    tools: List[Dict[str, Any]],
    predicted_tool: str,
    confidence: float,
    confidence_threshold: float,
    server_tool_names: Set[str]
) -> Optional[List[Dict[str, Any]]]:
    """
    Prune tools to only include the predicted tool and server tools.

    Only prunes if confidence exceeds threshold. Used to reduce prompt size
    when the classifier is highly confident.

    Args:
        tools: List of all tool definitions
        predicted_tool: Name of the predicted tool
        confidence: Confidence score from classifier
        confidence_threshold: Minimum confidence to enable pruning
        server_tool_names: Set of server tool names to always include

    Returns:
        Pruned list of tools, or None if pruning would result in empty list
    """
    if confidence < confidence_threshold:
        return tools

    def _get_name(tool: Dict[str, Any]) -> Optional[str]:
        if isinstance(tool.get("function"), dict):
            return tool.get("function", {}).get("name")
        return tool.get("name")

    pruned = []
    for tool in tools:
        name = _get_name(tool)
        if not name:
            continue
        if name == predicted_tool or name in server_tool_names:
            pruned.append(tool)

    if not pruned:
        return None

    if len(pruned) < len(tools):
        logger.info(
            "Pruned tools from %d to %d (predicted=%s, confidence=%.3f)",
            len(tools),
            len(pruned),
            predicted_tool,
            confidence
        )

    return pruned
