"""
Tool Execution Engine for Jarvis Voice Assistant.

This module handles the tool execution loop: call LLM, execute server tools,
repeat until completion or client tool calls are needed.
"""

import json
import logging
import os
import time as _time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from app.core.conversation_cache import conversation_cache
from app.core.tool_executor import tool_executor
from app.core.tool_call_parser import tool_call_parser
from app.core.utils.latency_logger import latency_logger

# Extracted modules for cleaner composition
from app.core.date_resolution import (
    normalize_date_key,
    flatten_date_context,
    is_datetime_param,
    is_datetime_array,
    resolve_date_keys,
)
from app.core.param_validation import (
    is_iso_datetime,
    find_invalid_params,
)
from app.core.system_prompt_builder import get_response_format
from app.core.usage_logging import (
    write_usage_log,
    write_prompt_response_log,
)
from app.core.general_context import generate_date_context_object
from app.core.tools.resolve_relative_date_tool import ResolveRelativeDateTool

logger = logging.getLogger("uvicorn")


class ToolExecutionEngine:
    """
    Engine for executing the tool loop.

    Handles:
    - LLM calls with conversation context
    - Server tool execution
    - Client tool forwarding
    - Date key resolution
    - Parameter validation with retries
    - Must-call guard logic
    """

    def __init__(self, llm_client):
        """
        Initialize the tool execution engine.

        Args:
            llm_client: LLM proxy client for chat completions
        """
        self.llm_client = llm_client

    async def execute(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_iterations: int = 10,
        user_utterance: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the tool loop: call LLM, execute server tools, repeat until done.

        Args:
            conversation_id: Conversation ID
            messages: Current message history (will be modified in place)
            tools: Available tools
            max_iterations: Maximum tool execution iterations
            user_utterance: Original user voice command (for LLM fallback in date resolution)

        Returns:
            Response dict with stop_reason and relevant data
        """
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Build adapter_settings from node's adapter_hash if present
        node_context = conversation_cache.get_node_context(conversation_id) or {}
        adapter_settings = None
        adapter_hash = node_context.get("adapter_hash")
        if adapter_hash:
            adapter_settings = {"hash": adapter_hash, "enabled": True}
            logger.info(f"Using adapter {adapter_hash[:8]}... for tool loop")

        # Get timezone for date resolution
        timezone_str = conversation_cache.get_timezone(conversation_id)

        # Environment helpers
        small_mode = os.getenv("JARVIS_SMALL_MODEL_MODE", "").strip().lower() in {"1", "true", "yes"}

        def _env_int(name: str, default_value: int) -> int:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default_value
            try:
                return int(raw)
            except ValueError:
                return default_value

        # Helper functions using closures
        def _log_usage(turns_used: int, status: str) -> None:
            write_usage_log(conversation_id, turns_used, status, usage_totals)

        def _log_prompt_response(
            prompt_messages: List[Dict[str, Any]],
            response_obj: Optional[Dict[str, Any]],
            raw_content: Optional[str],
            error: Optional[str],
            iteration: int
        ) -> None:
            write_prompt_response_log(
                conversation_id, prompt_messages, response_obj, raw_content, error, iteration
            )

        def _get_must_call_tools() -> List[str]:
            must_call = set()
            available_commands = conversation_cache.get_available_commands(conversation_id) or []
            for cmd in available_commands:
                cmd_name = cmd.get("command_name")
                if cmd_name and cmd.get("allow_direct_answer") is False:
                    must_call.add(cmd_name)
            for tool in tools or []:
                tool_name = tool.get("function", {}).get("name") if isinstance(tool.get("function"), dict) else tool.get("name")
                if tool_name and tool.get("allow_direct_answer") is False:
                    must_call.add(tool_name)
            return sorted(must_call)

        def _must_call_retry_already_added() -> bool:
            retry_tag = "[MUST_CALL_RETRY]"
            return any(
                msg.get("role") == "system" and retry_tag in msg.get("content", "")
                for msg in messages
            )

        def _invalid_param_retry_count() -> int:
            retry_tag = "[INVALID_PARAM_RETRY]"
            return sum(
                1
                for msg in messages
                if msg.get("role") == "system" and retry_tag in msg.get("content", "")
            )

        def _build_param_type_map() -> Dict[str, Dict[str, str]]:
            """Build a map of tool_name -> {param_name: param_type} from available commands."""
            param_map: Dict[str, Dict[str, str]] = {}
            available_commands = conversation_cache.get_available_commands(conversation_id) or []
            for cmd in available_commands:
                cmd_name = cmd.get("command_name")
                if not cmd_name:
                    continue
                params = cmd.get("parameters") or []
                for param in params:
                    name = (param or {}).get("name")
                    ptype = (param or {}).get("type")
                    if name and ptype:
                        param_map.setdefault(cmd_name, {})[name] = ptype
            return param_map

        def _validate_client_tool_params(client_calls: List[Dict[str, Any]]) -> List[str]:
            """Validate client tool call parameters."""
            param_types = _build_param_type_map()
            return find_invalid_params(client_calls, param_types)

        async def _inject_date_keys(
            tool_calls: List[Dict[str, Any]],
            date_keys: List[str],
            tools_list: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Inject resolved date keys into tool call arguments."""
            if not tool_calls:
                return tool_calls

            date_context = generate_date_context_object(timezone_str)
            flat_context = flatten_date_context(date_context)
            available_keys = ResolveRelativeDateTool.get_available_keys(date_context)

            # Resolve date_keys from LLM response
            resolved_dates: List[str] = []
            if date_keys:
                resolved_dates, unresolved_keys = resolve_date_keys(date_keys, date_context)

                # LLM fallback for unresolved keys
                if unresolved_keys and user_utterance:
                    for unresolved_key in unresolved_keys:
                        logger.info(f"Unresolved date key '{unresolved_key}', attempting LLM fallback")
                        fallback_key = await ResolveRelativeDateTool.resolve_with_llm_fallback(
                            unrecognized_key=unresolved_key,
                            available_keys=available_keys,
                            user_utterance=user_utterance,
                            llm_client=self.llm_client,
                            timezone=timezone_str
                        )
                        if fallback_key:
                            value = flat_context.get(fallback_key)
                            if isinstance(value, list):
                                resolved_dates.extend([v for v in value if isinstance(v, str)])
                            elif isinstance(value, str):
                                resolved_dates.append(value)
                            logger.info(f"LLM fallback resolved '{unresolved_key}' -> '{fallback_key}'")

            async def _resolve_relative_datetime(relative_value: str) -> Optional[str]:
                """Attempt to resolve a relative datetime string to an ISO datetime."""
                normalized = normalize_date_key(relative_value)
                value = flat_context.get(normalized)
                if isinstance(value, str):
                    logger.info(f"Resolved relative datetime '{relative_value}' -> '{value}'")
                    return value
                if isinstance(value, list) and value:
                    resolved = value[0] if isinstance(value[0], str) else None
                    logger.info(f"Resolved relative datetime '{relative_value}' -> '{resolved}' (first of list)")
                    return resolved

                # LLM fallback
                if user_utterance:
                    logger.info(f"Relative datetime '{relative_value}' not in context, attempting LLM fallback")
                    fallback_key = await ResolveRelativeDateTool.resolve_with_llm_fallback(
                        unrecognized_key=relative_value,
                        available_keys=available_keys,
                        user_utterance=user_utterance,
                        llm_client=self.llm_client,
                        timezone=timezone_str
                    )
                    if fallback_key:
                        fb_value = flat_context.get(fallback_key)
                        if isinstance(fb_value, str):
                            logger.info(f"LLM fallback resolved '{relative_value}' -> '{fb_value}'")
                            return fb_value
                        if isinstance(fb_value, list) and fb_value:
                            resolved = fb_value[0] if isinstance(fb_value[0], str) else None
                            logger.info(f"LLM fallback resolved '{relative_value}' -> '{resolved}' (first of list)")
                            return resolved

                logger.warning(f"Could not resolve relative datetime '{relative_value}'")
                return None

            # Build tool map for schema lookup
            tool_map: Dict[str, Dict[str, Any]] = {}
            for tool in tools_list or []:
                if isinstance(tool.get("function"), dict):
                    name = tool.get("function", {}).get("name")
                    if name:
                        tool_map[name] = tool.get("function", {})
                elif isinstance(tool.get("name"), str):
                    tool_map[tool["name"]] = tool

            # Process each tool call
            for call in tool_calls:
                tool_name = call.get("function", {}).get("name")
                if not tool_name:
                    continue
                tool_def = tool_map.get(tool_name, {})
                param_schemas = tool_def.get("parameters", {}).get("properties", {})
                if not isinstance(param_schemas, dict):
                    continue

                args_raw = call.get("function", {}).get("arguments", "{}")
                try:
                    args_obj = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                except (json.JSONDecodeError, ValueError, TypeError):
                    args_obj = {}
                if not isinstance(args_obj, dict):
                    continue

                mutated = False
                for param_name, param_schema in param_schemas.items():
                    if not is_datetime_param(param_schema):
                        continue

                    existing = args_obj.get(param_name)
                    is_array = is_datetime_array(param_schema)

                    # Case 1: Empty - inject resolved_dates
                    if existing in (None, [], ""):
                        if resolved_dates:
                            if is_array:
                                args_obj[param_name] = resolved_dates
                            else:
                                args_obj[param_name] = resolved_dates[0]
                            mutated = True
                        continue

                    # Case 2: Has value - validate and fix if needed
                    if is_array and isinstance(existing, list):
                        cleaned: List[str] = []
                        for item in existing:
                            if isinstance(item, str) and is_iso_datetime(item):
                                cleaned.append(item)
                            elif isinstance(item, str):
                                resolved = await _resolve_relative_datetime(item)
                                if resolved:
                                    cleaned.append(resolved)
                        if cleaned != existing:
                            args_obj[param_name] = cleaned if cleaned else resolved_dates
                            mutated = True
                    elif not is_array and isinstance(existing, str):
                        if not is_iso_datetime(existing):
                            resolved = await _resolve_relative_datetime(existing)
                            if resolved:
                                args_obj[param_name] = resolved
                                mutated = True
                            elif resolved_dates:
                                args_obj[param_name] = resolved_dates[0]
                                mutated = True

                if mutated:
                    call["function"]["arguments"] = json.dumps(args_obj)

            return tool_calls

        # Get timing context
        timing = latency_logger.get_request(conversation_id)

        # Main execution loop
        for iteration in range(max_iterations):
            logger.info(f"Tool loop iteration {iteration + 1}/{max_iterations}")
            prompt_snapshot = [dict(m) for m in (messages or [])]

            # Call LLM
            try:
                response_format = get_response_format()
                _llm_start = _time.time()
                with timing.measure(f"llm_call_iter_{iteration+1}") if timing else nullcontext():
                    response = await self.llm_client.chat_completion(
                        messages=messages,
                        conversation_id=conversation_id,
                        response_format=response_format,
                        include_date_context=True,
                        adapter_settings=adapter_settings
                    )
                logger.debug(f"LLM call took {(_time.time()-_llm_start)*1000:.0f}ms")
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                _log_prompt_response(prompt_snapshot, None, None, str(e), iteration + 1)
                _log_usage(iteration + 1, "error")
                return {
                    "stop_reason": "error",
                    "error": str(e)
                }

            # Parse response
            logger.info(f"Full LLM proxy response structure: {list(response.keys())}")

            try:
                raw_content = response["choices"][0]["message"]["content"]
                finish_reason_raw = response["choices"][0].get("finish_reason", "stop")
                logger.info(f"Extracted content length: {len(raw_content)}, finish_reason: {finish_reason_raw}")
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to extract content from response: {e}")
                logger.error(f"Response structure: {response}")
                raw_content = ""

            # Extract date_keys
            date_keys: List[str] = []
            if isinstance(response, dict):
                raw_keys = response.get("date_keys")
                if isinstance(raw_keys, list):
                    date_keys = [normalize_date_key(key) for key in raw_keys if isinstance(key, str)]

            # Accumulate token usage
            if isinstance(response, dict):
                usage = response.get("usage")
                if isinstance(usage, dict):
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if isinstance(usage.get(key), int):
                            usage_totals[key] += usage[key]

            if not raw_content:
                logger.warning("Empty raw_content!")
            response_obj = response if isinstance(response, dict) else None
            _log_prompt_response(prompt_snapshot, response_obj, raw_content, None, iteration + 1)

            # Parse tool calls from JSON
            finish_reason, tool_calls, assistant_message = tool_call_parser.parse_response(raw_content)

            logger.info(f"LLM response parsed: finish_reason={finish_reason}, tool_calls={len(tool_calls)}")

            # Add assistant message to history
            assistant_msg = {"role": "assistant", "content": raw_content}
            messages.append(assistant_msg)

            # Handle different finish reasons
            if finish_reason == "stop":
                must_call_tools = _get_must_call_tools()
                if must_call_tools and not _must_call_retry_already_added():
                    retry_message = (
                        "[MUST_CALL_RETRY] Direct answers are not allowed for this request. "
                        "You must call exactly one tool next. "
                        f"Tools that require a call: {', '.join(must_call_tools)}."
                    )
                    messages.append({"role": "system", "content": retry_message})
                    logger.info("Must-call guard triggered; retrying tool selection")
                    continue
                # Conversation complete
                _log_usage(iteration + 1, "complete")
                return {
                    "stop_reason": "complete",
                    "assistant_message": assistant_message
                }

            elif finish_reason == "tool_calls":
                # Always validate/inject datetime params
                tool_calls = await _inject_date_keys(tool_calls, date_keys, tools)
                if date_keys:
                    logger.info("Processed date keys: %s", date_keys)

                # Execute tools
                with timing.measure(f"tool_exec_iter_{iteration+1}") if timing else nullcontext():
                    server_results, client_calls = tool_executor.execute_tool_calls(
                        tool_calls, conversation_id=conversation_id, user_utterance=user_utterance
                    )

                # Check if request_validation was called
                validation_called = any(
                    tool_call.get("function", {}).get("name") == "request_validation"
                    for tool_call in tool_calls
                )

                # Check if get_command_examples was called
                get_examples_called = any(
                    tool_call.get("function", {}).get("name") == "get_command_examples"
                    for tool_call in tool_calls
                )

                # Always add server results to messages first
                if server_results:
                    messages.extend(server_results)
                    logger.info(f"Executed {len(server_results)} server tools, added results to conversation")

                # If get_command_examples was called, continue loop
                if get_examples_called:
                    logger.info("get_command_examples was called, continuing loop so LLM can see examples")
                    continue

                # Handle validation requests
                other_tools_called = len(tool_calls) > 1 or client_calls or get_examples_called
                if validation_called and other_tools_called:
                    logger.info("request_validation called with other tools - continuing loop")
                    continue

                if validation_called and not other_tools_called:
                    for result in server_results:
                        try:
                            content = json.loads(result["content"])
                            if content.get("_validation_request"):
                                logger.info("Validation request detected (only tool call)")
                                _log_usage(iteration + 1, "validation_required")
                                return {
                                    "stop_reason": "validation_required",
                                    "validation_request": {
                                        "question": content["question"],
                                        "parameter_name": content["parameter_name"],
                                        "options": content.get("options", [])
                                    }
                                }
                        except (json.JSONDecodeError, KeyError):
                            pass

                # If server tools and client tools, continue loop
                if server_results and client_calls:
                    logger.info("Server tools executed, continuing loop so LLM can see results before client tools")
                    continue

                # Return client tool calls
                if client_calls:
                    invalid_params = _validate_client_tool_params(client_calls)
                    if invalid_params:
                        max_retries = _env_int("JARVIS_INVALID_PARAM_RETRY_MAX", 1 if small_mode else 2)
                        retry_count = _invalid_param_retry_count()
                        if retry_count < max_retries:
                            retry_message = (
                                f"[INVALID_PARAM_RETRY {retry_count + 1}/{max_retries}] "
                                "Some parameters have invalid types or formats. "
                                "Fix the parameters to match expected types and return absolute "
                                f"date/time values when required. Invalid: {', '.join(invalid_params)}"
                            )
                            messages.append({"role": "system", "content": retry_message})
                            logger.info("Invalid parameter guard triggered; retrying tool call formatting")
                            continue
                    logger.info(f"Returning {len(client_calls)} client tool calls (no server tools to wait for)")
                    _log_usage(iteration + 1, "tool_calls")
                    return {
                        "stop_reason": "tool_calls",
                        "tool_calls": client_calls,
                        "assistant_message": assistant_message
                    }

                # Only server tools - already added to messages, continue loop
                if server_results:
                    logger.info("Only server tools executed, continuing loop")

            else:
                # Unknown finish reason
                logger.warning(f"Unknown finish_reason: {finish_reason}")
                _log_usage(iteration + 1, "complete")
                return {
                    "stop_reason": "complete",
                    "assistant_message": assistant_message
                }

        # Max iterations reached
        logger.warning(f"Max tool loop iterations ({max_iterations}) reached")
        _log_usage(max_iterations, "max_iterations_exceeded")
        return {
            "stop_reason": "complete",
            "assistant_message": "Maximum tool execution iterations reached.",
            "error": "max_iterations_exceeded"
        }
