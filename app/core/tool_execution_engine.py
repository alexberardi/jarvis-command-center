"""
Tool Execution Engine for Jarvis Voice Assistant.

This module handles the tool execution loop: call LLM, execute server tools,
repeat until completion or client tool calls are needed.
"""

import json
import logging
import os
import re
import time as _time
import uuid
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from app.core.conversation_cache import conversation_cache
from app.services.settings_service import get_settings_service
from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.tool_builder import ToolBuilder
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
from app.core.param_refinement import refine_params
from app.core.tools.resolve_relative_date_tool import ResolveRelativeDateTool

try:
    from jarvis_mcp_client import resolve_date_keys as _mcp_resolve_date_keys
    _HAS_MCP_CLIENT = True
except ImportError:
    _HAS_MCP_CLIENT = False

logger = logging.getLogger("uvicorn")


def _normalize_native_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize native OpenAI-format tool_calls into internal format.

    Native format from LLM proxy:
        [{"id": "call_abc", "type": "function",
          "function": {"name": "x", "arguments": "{...}"}}]

    Internal format expected by tool_executor:
        [{"id": "call_abc", "type": "function",
          "function": {"name": "x", "arguments": "{...}"}}]

    The formats are compatible, but we ensure all fields are present
    and arguments is always a JSON string.
    """
    normalized: List[Dict[str, Any]] = []
    for call in raw_calls:
        func = call.get("function", {})
        name = func.get("name")
        if not name:
            continue

        arguments = func.get("arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = "{}"

        call_id = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        normalized.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        })
    return normalized


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

    def __init__(
        self,
        llm_client,
        prompt_provider: Optional[IJarvisPromptProvider] = None,
    ):
        """
        Initialize the tool execution engine.

        Args:
            llm_client: LLM proxy client for chat completions
            prompt_provider: Optional prompt provider for model-specific
                response format and parse_response hooks.
        """
        self.llm_client = llm_client
        self.prompt_provider = prompt_provider

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
        reasoning_parts: List[str] = []  # Accumulated <think> blocks across iterations

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
        small_mode = get_settings_service().get_bool("model.small_model_mode", True)

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

        def _must_call_retry_count() -> int:
            retry_tag = "[MUST_CALL_RETRY]"
            return sum(
                1
                for msg in messages
                if msg.get("role") == "system" and retry_tag in msg.get("content", "")
            )

        def _invalid_param_retry_count() -> int:
            retry_tag = "[INVALID_PARAM_RETRY]"
            return sum(
                1
                for msg in messages
                if msg.get("role") == "system" and retry_tag in msg.get("content", "")
            )

        def _iso_date_retry_count() -> int:
            retry_tag = "[ISO_DATE_RETRY]"
            return sum(
                1
                for msg in messages
                if msg.get("role") == "system" and retry_tag in msg.get("content", "")
            )

        def _try_fix_iso_dates(tool_calls_to_check: List[Dict[str, Any]]) -> str:
            """Check for ISO dates in resolved_datetimes; fix or flag them.

            Returns "clean" (no ISO dates found), "fixed" (converted to keys),
            or "bad" (ISO dates don't match any known key).
            """
            # 1. Collect all ISO dates from all tool calls
            iso_entries: List[tuple[int, List[str]]] = []
            for i, call in enumerate(tool_calls_to_check):
                args_raw = call.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                if not isinstance(args, dict):
                    continue
                datetimes = args.get("resolved_datetimes", [])
                if isinstance(datetimes, str):
                    datetimes = [datetimes]
                if not isinstance(datetimes, list):
                    continue
                iso_dates = [d for d in datetimes if isinstance(d, str) and is_iso_datetime(d)]
                if iso_dates:
                    iso_entries.append((i, iso_dates))

            if not iso_entries:
                return "clean"

            # 2. Build reverse map from date context
            date_context = generate_date_context_object(timezone_str)
            flat = flatten_date_context(date_context)
            reverse_single: Dict[str, str] = {}   # "2026-03-04T..." → "today"
            reverse_multi: Dict[tuple, str] = {}   # ("2026-03-07T...", "2026-03-08T...") → "this_weekend"
            for key, value in flat.items():
                if isinstance(value, list):
                    str_vals = [v for v in value if isinstance(v, str)]
                    if str_vals:
                        reverse_multi[tuple(sorted(str_vals))] = key
                        for v in str_vals:
                            reverse_single.setdefault(v, key)
                elif isinstance(value, str):
                    reverse_single.setdefault(value, key)

            # 3. Try to reverse-map each ISO date
            all_fixed = True
            for call_idx, iso_dates in iso_entries:
                # Try multi-date match first (e.g., this_weekend)
                multi_key = reverse_multi.get(tuple(sorted(iso_dates)))
                if multi_key:
                    _replace_datetimes(tool_calls_to_check[call_idx], [multi_key])
                    continue
                # Try individual lookups
                keys: List[str] = []
                matched = True
                for iso in iso_dates:
                    key = reverse_single.get(iso)
                    if key:
                        keys.append(key)
                    else:
                        matched = False
                        break
                if matched and keys:
                    _replace_datetimes(tool_calls_to_check[call_idx], keys)
                else:
                    all_fixed = False

            return "fixed" if all_fixed else "bad"

        def _replace_datetimes(tool_call: Dict[str, Any], new_values: List[str]) -> None:
            """Replace resolved_datetimes in a tool call's arguments."""
            func = tool_call.get("function", {})
            args_raw = func.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            except (json.JSONDecodeError, ValueError, TypeError):
                return
            args["resolved_datetimes"] = new_values
            func["arguments"] = json.dumps(args) if isinstance(args_raw, str) else args

        def _build_param_maps() -> tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, List[str]]]]:
            """Build type and enum maps from available commands.

            Returns:
                Tuple of (param_type_map, param_enum_map) where:
                - param_type_map: tool_name -> {param_name: param_type}
                - param_enum_map: tool_name -> {param_name: [allowed_values]}
            """
            type_map: Dict[str, Dict[str, str]] = {}
            enum_map: Dict[str, Dict[str, List[str]]] = {}
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
                        type_map.setdefault(cmd_name, {})[name] = ptype
                    enum_vals = (param or {}).get("enum_values")
                    if name and enum_vals and isinstance(enum_vals, list):
                        enum_map.setdefault(cmd_name, {})[name] = enum_vals
            return type_map, enum_map

        def _validate_client_tool_params(client_calls: List[Dict[str, Any]]) -> List[str]:
            """Validate client tool call parameters (types + enum values)."""
            param_types, param_enums = _build_param_maps()
            return find_invalid_params(client_calls, param_types, param_enums)

        async def _inject_date_keys(
            tool_calls: List[Dict[str, Any]],
            date_keys: List[str],
            tools_list: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Inject resolved date keys into tool call arguments."""
            if not tool_calls:
                return tool_calls

            # MCP-first date resolution with local fallback
            date_context = None
            flat_context = None
            available_keys = None
            resolved_dates: List[str] = []
            unresolved_keys: List[str] = []

            if date_keys:
                mcp_succeeded = False
                if _HAS_MCP_CLIENT:
                    try:
                        mcp_result = await _mcp_resolve_date_keys(date_keys, timezone=timezone_str)
                        if mcp_result is not None:
                            resolved_dates = mcp_result.get("resolved", [])
                            unresolved_keys = mcp_result.get("unresolved", [])
                            logger.info("MCP resolved %d/%d date keys", len(resolved_dates), len(date_keys))
                            mcp_succeeded = True
                    except (RuntimeError, ConnectionError, TimeoutError, OSError) as e:
                        logger.info("MCP date resolution unavailable (%s), using local fallback", e)

                if not mcp_succeeded:
                    # Local fallback — identical to previous behavior
                    date_context = generate_date_context_object(timezone_str)
                    flat_context = flatten_date_context(date_context)
                    available_keys = ResolveRelativeDateTool.get_available_keys(date_context)
                    resolved_dates, unresolved_keys = resolve_date_keys(date_keys, date_context)
                    logger.info("Local fallback resolved %d/%d date keys", len(resolved_dates), len(date_keys))

                # LLM fallback for unresolved keys (needs local context)
                if unresolved_keys and user_utterance:
                    if date_context is None:
                        date_context = generate_date_context_object(timezone_str)
                        flat_context = flatten_date_context(date_context)
                        available_keys = ResolveRelativeDateTool.get_available_keys(date_context)
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
                nonlocal date_context, flat_context, available_keys
                # Lazily initialize local context if MCP resolved everything
                if flat_context is None:
                    date_context = generate_date_context_object(timezone_str)
                    flat_context = flatten_date_context(date_context)
                    available_keys = ResolveRelativeDateTool.get_available_keys(date_context)
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

                    # Case 1: Empty - inject resolved_dates (default to today if none)
                    if existing in (None, [], ""):
                        dates_to_inject = resolved_dates
                        if not dates_to_inject:
                            # No date keys from LLM — default to today
                            if date_context is None:
                                date_context = generate_date_context_object(timezone_str)
                                flat_context = flatten_date_context(date_context)
                            today_val = flat_context.get("today")
                            if isinstance(today_val, str):
                                dates_to_inject = [today_val]
                            elif isinstance(today_val, list):
                                dates_to_inject = [v for v in today_val if isinstance(v, str)]
                            logger.info("No date keys from LLM, defaulting to today: %s", dates_to_inject)
                        if dates_to_inject:
                            if is_array:
                                args_obj[param_name] = dates_to_inject
                            else:
                                args_obj[param_name] = dates_to_inject[0]
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

        # Determine if we should use native tool calling
        use_native_tools: bool = (
            self.prompt_provider is not None
            and self.prompt_provider.supports_native_tools
        )
        native_tools: Optional[List[Dict[str, Any]]] = None
        if use_native_tools:
            native_tools = self.prompt_provider.build_tools(
                ToolBuilder.strip_jarvis_extensions(tools)
            )
            logger.info("Native tool calling enabled (%d tools)", len(native_tools or []))

        # Main execution loop
        for iteration in range(max_iterations):
            logger.info(f"Tool loop iteration {iteration + 1}/{max_iterations}")
            prompt_snapshot = [dict(m) for m in (messages or [])]

            # Call LLM — dual path: native tools vs text-based
            try:
                _llm_start = _time.time()
                with timing.measure(f"llm_call_iter_{iteration+1}") if timing else nullcontext():
                    if use_native_tools:
                        # Native path: pass tools via API, no response_format constraint
                        response = await self.llm_client.chat_completion(
                            messages=messages,
                            conversation_id=conversation_id,
                            tools=native_tools,
                            tool_choice="auto",
                            include_date_context=True,
                            adapter_settings=adapter_settings,
                            max_tokens=256,
                            temperature=0.4,
                        )
                    else:
                        # Text-based path: tools in system prompt, JSON response format
                        if self.prompt_provider is not None:
                            response_format = self.prompt_provider.get_response_format()
                            if response_format is None:
                                response_format = get_response_format()
                        else:
                            response_format = get_response_format()
                        response = await self.llm_client.chat_completion(
                            messages=messages,
                            conversation_id=conversation_id,
                            response_format=response_format,
                            include_date_context=True,
                            adapter_settings=adapter_settings,
                            max_tokens=256,
                            temperature=0.4,
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

            # Defensive extraction — llm-proxy can return 'message' as a bare
            # string in some response shapes; .get() on a string raises
            # AttributeError, which wasn't in the previous except clause.
            # Result: 500 on /voice/command/stream for any text-model turn.
            msg: Any = None
            try:
                choice = response["choices"][0] if isinstance(response, dict) else None
                msg = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(msg, dict):
                    raw_content = msg.get("content") or ""
                elif isinstance(msg, str):
                    raw_content = msg
                else:
                    raw_content = ""
                finish_reason_raw = choice.get("finish_reason", "stop") if isinstance(choice, dict) else "stop"
                logger.info(f"Extracted content length: {len(raw_content)}, finish_reason: {finish_reason_raw}")
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                logger.error(f"Failed to extract content from response: {e}")
                logger.error(f"Response structure: {response}")
                raw_content = ""
                finish_reason_raw = "stop"
                msg = None

            # Extract <think> content before any stripping/parsing
            _think_matches = re.findall(r"<think>(.*?)</think>", raw_content, re.DOTALL)
            if _think_matches:
                reasoning_parts.extend(m.strip() for m in _think_matches if m.strip())

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

            # Dual path: extract tool calls
            if use_native_tools:
                # Native path: tool_calls live on the message dict extracted above.
                # Only dicts carry tool_calls; bare-string messages have none.
                tool_calls_raw = msg.get("tool_calls") if isinstance(msg, dict) else None
                if finish_reason_raw == "tool_calls" and tool_calls_raw:
                    tool_calls = _normalize_native_tool_calls(tool_calls_raw)
                    finish_reason = "tool_calls"
                    assistant_message = raw_content
                else:
                    tool_calls = []
                    finish_reason = "stop"
                    assistant_message = raw_content
            else:
                # Text-based path: let prompt provider transform, then parse JSON
                content_for_parsing = raw_content
                if self.prompt_provider is not None:
                    transformed = self.prompt_provider.parse_response(raw_content)
                    if transformed is not None:
                        content_for_parsing = transformed

                # Parse tool calls from JSON
                finish_reason, tool_calls, assistant_message = tool_call_parser.parse_response(content_for_parsing)

            logger.info(f"LLM response parsed: finish_reason={finish_reason}, tool_calls={len(tool_calls)}")

            # Add assistant message to history
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": raw_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Handle different finish reasons
            if finish_reason == "stop":
                # Unconditional force-tool-calls guard (compressed provider).
                # Pop the bad assistant message so the model gets a clean
                # slate instead of being biased by its own failed response.
                # Allows up to 2 retries before giving up.
                force_tools = conversation_cache.get_force_tool_calls(conversation_id)
                retry_count = _must_call_retry_count()
                if force_tools and retry_count < 2:
                    messages.pop()  # Remove the assistant message with no tool call
                    retry_message = (
                        f"[MUST_CALL_RETRY] You MUST call a tool. Do NOT answer directly. "
                        f"Pick the best matching function from the available tools. "
                        f"(attempt {retry_count + 1}/2)"
                    )
                    messages.append({"role": "system", "content": retry_message})
                    logger.info("Force-tool-calls guard triggered (attempt %d/2); retrying", retry_count + 1)
                    continue

                # Per-command must-call guard (only when router had a confident match).
                # Conversational queries ("how are you") won't match any tool,
                # so forcing a retry just produces an empty second response.
                router_decision = conversation_cache.get_router_decision(conversation_id)
                router_matched = (
                    router_decision is not None
                    and router_decision.get("used", False)
                )
                must_call_tools = _get_must_call_tools()
                if must_call_tools and router_matched and retry_count < 1:
                    messages.pop()  # Remove the assistant message with no tool call
                    retry_message = (
                        "[MUST_CALL_RETRY] Direct answers are not allowed for this request. "
                        "You must call exactly one tool next. "
                        f"Tools that require a call: {', '.join(must_call_tools)}."
                    )
                    messages.append({"role": "system", "content": retry_message})
                    logger.info("Must-call guard triggered; retrying tool selection")
                    continue
                # Conversation complete — run provider.sanitize_text on the
                # direct-answer content so model-specific artifacts (Qwen3
                # <think> blocks, etc.) don't reach TTS.
                # assistant_message may be a dict {"content": "..."} (native
                # tool-calling path) or a bare string (text-model path) —
                # blindly calling .get() on the latter raises AttributeError
                # and crashes the whole stream with a 500.
                if isinstance(assistant_message, dict):
                    original_opt = assistant_message.get("content")
                    original: str | None = original_opt if isinstance(original_opt, str) else None
                elif isinstance(assistant_message, str):
                    original = assistant_message
                else:
                    original = None
                if original:
                    # Provider-specific scrub (model artifacts: think blocks
                    # on Qwen3, etc.), then universal TTS-safe scrub
                    # (emojis — no provider should bypass this).
                    if self.prompt_provider:
                        cleaned: str = self.prompt_provider.sanitize_text(original)
                    else:
                        cleaned = original
                    from app.core.tts_text import clean_for_tts
                    cleaned = clean_for_tts(cleaned)
                    if cleaned != original:
                        if isinstance(assistant_message, dict):
                            assistant_message["content"] = cleaned
                        else:
                            assistant_message = cleaned
                        logger.debug(
                            "tts sanitize trimmed %d chars off response",
                            len(original) - len(cleaned),
                        )
                _log_usage(iteration + 1, "complete")
                result = {
                    "stop_reason": "complete",
                    "assistant_message": assistant_message,
                }
                if reasoning_parts:
                    result["reasoning"] = "\n\n".join(reasoning_parts)
                return result

            elif finish_reason == "tool_calls":
                # ISO date guard: if the model emitted ISO timestamps in
                # resolved_datetimes (instead of date key strings like "today"),
                # try to reverse-map them to known date keys. Only reprompt
                # if the dates don't match any known key.
                iso_result = _try_fix_iso_dates(tool_calls)
                if iso_result == "fixed":
                    logger.info("ISO date guard: silently converted ISO timestamps to date keys")
                    # Fall through — tool_calls already patched
                elif iso_result == "bad" and _iso_date_retry_count() < 1:
                    messages.pop()
                    messages.append({
                        "role": "system",
                        "content": (
                            "[ISO_DATE_RETRY] The dates you provided were incorrect. "
                            "Use date KEY STRINGS only: today, tomorrow, yesterday, this_weekend, etc. "
                            'Example: {"resolved_datetimes": ["today"]}. Try again.'
                        ),
                    })
                    logger.info("ISO date guard: incorrect ISO dates, retrying")
                    continue

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
                                result = {
                                    "stop_reason": "validation_required",
                                    "validation_request": {
                                        "question": content["question"],
                                        "parameter_name": content["parameter_name"],
                                        "options": content.get("options", []),
                                    },
                                }
                                if reasoning_parts:
                                    result["reasoning"] = "\n\n".join(reasoning_parts)
                                return result
                        except (json.JSONDecodeError, KeyError):
                            pass

                # If server tools and client tools, continue loop
                if server_results and client_calls:
                    logger.info("Server tools executed, continuing loop so LLM can see results before client tools")
                    continue

                # Return client tool calls
                if client_calls:
                    # Refine params for commands with refinable parameters
                    # (only when compressed provider is active — gated on force_tools)
                    force_tools = conversation_cache.get_force_tool_calls(conversation_id)
                    if force_tools and user_utterance:
                        all_tools = conversation_cache.get_tools(conversation_id) or []
                        client_calls = await refine_params(
                            client_calls, all_tools, user_utterance, self.llm_client
                        )

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
                    result = {
                        "stop_reason": "tool_calls",
                        "tool_calls": client_calls,
                        "assistant_message": assistant_message,
                    }
                    if reasoning_parts:
                        result["reasoning"] = "\n\n".join(reasoning_parts)
                    return result

                # Only server tools - already added to messages
                if server_results:
                    if not use_native_tools:
                        # Text-based models can't process role="tool" messages:
                        # the chat template drops tool_call metadata, causing
                        # the LLM to re-call the same tool endlessly.  Return
                        # results so the handler can format with a clean call.
                        logger.info(
                            "Server-only tools in text-based mode — returning for formatting"
                        )
                        _log_usage(iteration + 1, "server_tool_complete")
                        result = {
                            "stop_reason": "server_tool_complete",
                            "server_tool_results": server_results,
                            "assistant_message": assistant_message,
                        }
                        if reasoning_parts:
                            result["reasoning"] = "\n\n".join(reasoning_parts)
                        return result
                    logger.info("Only server tools executed, continuing loop")

            else:
                # Unknown finish reason
                logger.warning(f"Unknown finish_reason: {finish_reason}")
                _log_usage(iteration + 1, "complete")
                result = {
                    "stop_reason": "complete",
                    "assistant_message": assistant_message,
                }
                if reasoning_parts:
                    result["reasoning"] = "\n\n".join(reasoning_parts)
                return result

        # Max iterations reached
        logger.warning(f"Max tool loop iterations ({max_iterations}) reached")
        _log_usage(max_iterations, "max_iterations_exceeded")
        result = {
            "stop_reason": "complete",
            "assistant_message": "Maximum tool execution iterations reached.",
            "error": "max_iterations_exceeded",
        }
        if reasoning_parts:
            result["reasoning"] = "\n\n".join(reasoning_parts)
        return result
