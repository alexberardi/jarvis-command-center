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
