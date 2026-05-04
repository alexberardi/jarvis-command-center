"""
Usage and trace logging for Jarvis voice assistant.

This module provides utilities for logging LLM usage statistics and
prompt/response traces for debugging and monitoring.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger("uvicorn")


def write_usage_log(
    conversation_id: str,
    turns_used: int,
    status: str,
    usage_totals: Dict[str, int]
) -> None:
    """
    Write a usage log entry.

    Args:
        conversation_id: The conversation ID
        turns_used: Number of turns used in the conversation
        status: Status of the conversation (complete, error, etc.)
        usage_totals: Dict with prompt_tokens, completion_tokens, total_tokens
    """
    log_path = os.getenv(
        "JARVIS_LLM_USAGE_LOG_PATH",
        "/app/temp/llm_usage.log"
    )

    timestamp = datetime.utcnow().isoformat() + "Z"
    entry = (
        f"{timestamp} conversation_id={conversation_id} "
        f"turns={turns_used} status={status} "
        f"prompt_tokens={usage_totals.get('prompt_tokens', 0)} "
        f"completion_tokens={usage_totals.get('completion_tokens', 0)} "
        f"total_tokens={usage_totals.get('total_tokens', 0)}\n"
    )

    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(entry)
    except Exception as exc:
        logger.warning("Failed to write usage log: %s", exc)


def write_prompt_response_log(
    conversation_id: str,
    prompt_messages: List[Dict[str, Any]],
    response_obj: Optional[Dict[str, Any]],
    raw_content: Optional[str],
    error: Optional[str],
    iteration: int
) -> None:
    """
    Write a prompt/response trace log entry.

    Args:
        conversation_id: The conversation ID
        prompt_messages: The prompt messages sent to the LLM
        response_obj: The full response object from the LLM
        raw_content: The raw content extracted from the response
        error: Error message if any
        iteration: The iteration number in the tool loop
    """
    log_path = os.getenv(
        "JARVIS_LLM_TRACE_LOG_PATH",
        "/app/temp/llm_trace.log"
    )

    timestamp = datetime.utcnow().isoformat() + "Z"
    entry = {
        "timestamp": timestamp,
        "conversation_id": conversation_id,
        "iteration": iteration,
        "prompt_messages": prompt_messages,
        "raw_content": raw_content,
        "response": response_obj,
        "error": error,
    }

    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write LLM trace log: %s", exc)


def write_metrics_log(
    conversation_id: str,
    total_duration_ms: float,
    iterations: List[Dict[str, Any]],
    tools_count: int,
    agent_context_chars: int = 0,
) -> None:
    """
    Write a per-request metrics log entry for LLM performance analysis.

    Each entry is a JSON line capturing per-iteration token counts, durations,
    and tool calls — designed for before/after comparison of prompt optimizations.

    Args:
        conversation_id: The conversation ID
        total_duration_ms: Total duration of the tool execution loop in ms
        iterations: List of per-iteration dicts, each with:
            - iteration (int)
            - prompt_tokens (int)
            - completion_tokens (int)
            - duration_ms (float)
            - tool_calls (list of tool names called)
            - finish_reason (str)
        tools_count: Number of tool definitions in the prompt
        agent_context_chars: Number of chars of agent context injected (0 = none)
    """
    log_path = os.getenv(
        "JARVIS_LLM_METRICS_LOG_PATH",
        "/app/temp/llm_metrics.log"
    )

    # Determine if the model answered purely from context
    # (stopped on first iteration with no tool calls)
    answered_from_context = (
        len(iterations) == 1
        and iterations[0].get("finish_reason") == "stop"
        and not iterations[0].get("tool_calls")
    )

    total_prompt = sum(it.get("prompt_tokens", 0) for it in iterations)
    total_completion = sum(it.get("completion_tokens", 0) for it in iterations)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "conversation_id": conversation_id,
        "total_duration_ms": round(total_duration_ms, 1),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "iterations_count": len(iterations),
        "tools_count": tools_count,
        "agent_context_chars": agent_context_chars,
        "answered_from_context": answered_from_context,
        "per_iteration": iterations,
    }

    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write metrics log: %s", exc)
