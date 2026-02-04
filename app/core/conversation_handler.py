"""
Conversation Handler for Jarvis Voice Assistant.

This module orchestrates conversation flow including warmup, command processing,
and continuation with tool results. Extracted from ModelService for better
separation of concerns.
"""

import json
import logging
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.imodel_interface import IModelInterface
from app.core.llm_proxy_client import LLMProxyClient
from app.core.tool_execution_engine import ToolExecutionEngine
from app.core.tool_registry import tool_registry
from app.core.tool_routing import (
    get_shared_tool_classifier,
    route_tool,
    filter_tools_for_utterance,
)
from app.core.utils.latency_logger import latency_logger
from app.core.voice_command_helpers import (
    get_tool_name,
    filter_available_commands,
    build_available_command_flags,
    apply_tool_routing,
    prune_tools_by_router_decision,
)
from app.core.warmup_service import warmup_service

logger = logging.getLogger("uvicorn")


def get_server_tool_names(tools: Optional[List[Dict[str, Any]]]) -> Set[str]:
    """
    Extract server tool names from a tools list.

    Server tools are those registered in the tool_registry.

    Args:
        tools: List of tool definitions

    Returns:
        Set of server tool names
    """
    if not tools:
        return set()

    return {
        name
        for name in (get_tool_name(t) for t in tools)
        if name and tool_registry.has_tool(name)
    }


class ConversationHandler:
    """
    Orchestrates conversation flow for voice commands.

    Handles:
    - Conversation warmup (initializing state)
    - Voice command processing
    - Continuation with tool results
    - Conversation cleanup

    This class coordinates between the conversation cache, tool execution engine,
    and model interface to process voice commands through the tool-based architecture.
    """

    def __init__(self, model: IModelInterface, llm_client: LLMProxyClient):
        """
        Initialize the conversation handler.

        Args:
            model: The model interface for building prompts and configuration
            llm_client: The LLM proxy client for API calls
        """
        self.model = model
        self.llm_client = llm_client

    async def cleanup_conversation(self, conversation_id: str) -> None:
        """
        Clean up a conversation by removing it from the cache.

        Args:
            conversation_id: The conversation ID to clean up
        """
        logger.info(f"🧹 Cleaning up conversation {conversation_id[:8]}...")
        conversation_cache.remove(conversation_id)

    async def warmup_conversation_with_tools(
        self,
        conversation_id: str,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        client_tools: Optional[List[Dict[str, Any]]],
        available_commands: Optional[List[Any]],
        adapter_settings: Optional[Dict[str, Any]] = None,
        skip_warmup_inference: bool = False,
    ) -> None:
        """
        Initialize a tool-based conversation with warmup.

        Sets up the conversation cache with system prompt, tools, and available commands.
        Optionally performs a warmup inference call to reduce first-response latency.

        Args:
            conversation_id: Unique conversation identifier
            node_context: Context about the node (room, user, etc.)
            timezone: User's timezone
            client_tools: Tools provided by the client
            available_commands: Command definitions for examples/antipatterns
            adapter_settings: Optional adapter configuration
            skip_warmup_inference: If True, skip the warmup LLM call
        """
        logger.info(f"🔥 Warming up tool-based conversation {conversation_id[:8]}...")

        # Get server tools from registry
        server_tools = tool_registry.get_tools_for_model(self.model.name)
        server_tool_names = {get_tool_name(t) for t in server_tools if get_tool_name(t)}

        # Merge server and client tools
        all_tools = warmup_service.merge_tools(server_tools, client_tools)

        # Get available command names for filtering
        available_command_names = warmup_service.get_available_command_names(
            all_tools, available_commands
        )

        # Process client tools (filter antipatterns, etc.)
        if client_tools:
            warmup_service.process_client_tools(client_tools, available_command_names)

        # Build examples map from client tools
        examples_map = warmup_service.build_examples_map(client_tools or [])

        # Merge available commands into examples
        command_examples = warmup_service.merge_available_commands(
            examples_map, available_commands, available_command_names
        )

        # Build available command flags for system prompt
        available_command_flags = build_available_command_flags(
            [cmd for cmd in command_examples if cmd.get("command_name")]
        )

        # Build system prompt
        if hasattr(self.model, "_build_system_prompt"):
            system_prompt = self.model._build_system_prompt(  # type: ignore[attr-defined]
                node_context, timezone, all_tools, available_command_flags
            )
        else:
            system_prompt = "You are a helpful voice assistant."

        messages = [{"role": "system", "content": system_prompt}]

        # Store in cache
        conversation_cache.set(
            conversation_id=conversation_id,
            messages=messages,
            available_commands=command_examples,
            timezone=timezone,
            tools=all_tools,
            node_context=node_context,
        )

        # Optional warmup inference (reduces first-response latency)
        if not skip_warmup_inference:
            try:
                await self.llm_client.chat_completion(
                    conversation_id=conversation_id,
                    messages=messages,
                    tools=all_tools,
                    adapter_settings=adapter_settings,
                )
                logger.info(f"✅ Warmup inference complete for {conversation_id[:8]}...")
            except Exception as e:
                logger.warning(f"⚠️ Warmup inference failed (non-fatal): {e}")

    async def process_voice_command_with_tools(
        self,
        voice_command: str,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """
        Process a voice command using tool-based architecture.

        Args:
            voice_command: The user's voice command text
            conversation_id: Conversation ID

        Returns:
            Response dict with:
            {
                "stop_reason": str,  # "complete", "tool_calls", "validation_required"
                "assistant_message": Optional[str],
                "tool_calls": Optional[List[Dict]],  # Client tool calls
                "validation_request": Optional[Dict]
            }
        """
        import time

        _t0 = time.time()
        timing = latency_logger.get_request(conversation_id)
        logger.info(f"🎯 Processing tool-based command: '{voice_command}'")

        # Get conversation state from cache
        with timing.measure("cache_lookups") if timing else nullcontext():
            messages = conversation_cache.get_messages(conversation_id)
            tools = conversation_cache.get_tools(conversation_id)
            available_commands = conversation_cache.get_available_commands(conversation_id) or []
            node_context = conversation_cache.get_node_context(conversation_id) or {}
            timezone = conversation_cache.get_timezone(conversation_id)

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Cache lookups done")

        if not messages:
            raise ValueError(f"Conversation {conversation_id} not found or expired")

        # Get server tool names for filtering
        server_tool_names = get_server_tool_names(tools)

        # Apply keyword-based tool filtering (if enabled)
        tools, messages = self._apply_tool_filtering(
            voice_command, tools, available_commands, messages,
            node_context, timezone, server_tool_names
        )
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool filtering done")

        # Apply tool routing classifier
        _router_t0 = time.time()
        router_decision = self._apply_tool_routing_with_cache(
            voice_command, tools or [], conversation_id
        )
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool routing done (router took {(time.time()-_router_t0)*1000:.0f}ms)")

        # Apply high-confidence tool pruning
        tools, messages = self._apply_high_confidence_pruning(
            tools, router_decision, available_commands, messages,
            node_context, timezone, server_tool_names
        )

        # Add router hint if decision was used
        if router_decision and router_decision.get("used"):
            hint_tool = router_decision.get("tool_name")
            messages.append({
                "role": "system",
                "content": f"Router hint: likely tool is '{hint_tool}'. Use it if it matches intent; otherwise choose the best tool."
            })

        # Add user message
        messages.append({"role": "user", "content": voice_command})
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Starting tool execution loop")

        # Execute tool loop
        with timing.measure("tool_execution_loop") if timing else nullcontext():
            engine = ToolExecutionEngine(self.llm_client)
            result = await engine.execute(
                conversation_id=conversation_id,
                messages=messages,
                tools=tools or [],
                user_utterance=voice_command,
            )

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool execution loop completed")

        # Update cache and return
        conversation_cache.update_messages(conversation_id, messages)
        if result.get("stop_reason") == "error":
            logger.error("❌ Tool loop returned stop_reason=error: %s", result)
        return result

    async def continue_conversation_with_tool_results(
        self,
        conversation_id: str,
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Continue a conversation by providing tool execution results.

        Args:
            conversation_id: Conversation ID
            tool_results: List of tool results in format:
                [{"tool_call_id": str, "output": Any}, ...]

        Returns:
            Response dict (same format as process_voice_command_with_tools)
        """
        logger.info(f"🔄 Continuing conversation {conversation_id[:8]} with {len(tool_results)} tool results")

        # Get conversation state
        messages = conversation_cache.get_messages(conversation_id)
        tools = conversation_cache.get_tools(conversation_id)

        if not messages:
            raise ValueError(f"Conversation {conversation_id} not found or expired")

        # Add tool result messages
        for result in tool_results:
            output = result["output"]
            if not isinstance(output, str):
                output = json.dumps(output)

            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": output,
            })

        # Extract last user utterance for potential LLM fallback
        last_user_utterance = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_utterance = msg.get("content")
                break

        # Continue with tool execution loop
        engine = ToolExecutionEngine(self.llm_client)
        result = await engine.execute(
            conversation_id=conversation_id,
            messages=messages,
            tools=tools or [],
            user_utterance=last_user_utterance,
        )

        # Update cache
        conversation_cache.update_messages(conversation_id, messages)

        return result

    def _apply_tool_filtering(
        self,
        voice_command: str,
        tools: Optional[List[Dict[str, Any]]],
        available_commands: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        node_context: Dict[str, Any],
        timezone: Optional[str],
        server_tool_names: Set[str],
    ) -> Tuple[Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Apply keyword-based tool filtering and rebuild system prompt if needed.

        Returns:
            Tuple of (filtered_tools, updated_messages)
        """
        if not tools:
            return tools, messages

        filtered_tools = filter_tools_for_utterance(voice_command, tools, available_commands)
        if filtered_tools is tools:
            return tools, messages

        # Tools were filtered - rebuild system prompt if we have context
        if node_context and messages and messages[0].get("role") == "system":
            filtered_commands = filter_available_commands(filtered_tools, available_commands, server_tool_names)
            command_flags = build_available_command_flags(filtered_commands)
            messages = self._rebuild_system_prompt(messages, node_context, timezone, filtered_tools, command_flags)
            logger.info("🧹 Rebuilt system prompt with filtered tools")

        return filtered_tools, messages

    def _apply_tool_routing_with_cache(
        self,
        voice_command: str,
        tools: List[Dict[str, Any]],
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply tool routing classifier and cache the decision.

        Returns:
            Router decision dict or None
        """
        if not self.model.use_tool_classifier:
            logger.info("🔇 Tool classifier disabled for %s", self.model.name)
            conversation_cache.set_router_decision(conversation_id, None)
            return None

        try:
            min_conf = float(os.getenv("JARVIS_TOOL_CLASSIFIER_MIN_CONFIDENCE", "0.6"))
        except ValueError:
            min_conf = 0.6

        classifier = get_shared_tool_classifier()
        router_decision = apply_tool_routing(
            voice_command=voice_command,
            tools=tools,
            use_classifier=True,
            route_tool_fn=lambda vc, t: route_tool(vc, t, classifier),
            min_confidence=min_conf,
        )

        conversation_cache.set_router_decision(conversation_id, router_decision)

        if router_decision:
            logger.info(
                "🧭 Router predicted tool=%s score=%.3f use_hint=%s",
                router_decision["tool_name"],
                router_decision["score"],
                router_decision["used"],
            )

        return router_decision

    def _apply_high_confidence_pruning(
        self,
        tools: Optional[List[Dict[str, Any]]],
        router_decision: Optional[Dict[str, Any]],
        available_commands: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        node_context: Dict[str, Any],
        timezone: Optional[str],
        server_tool_names: Set[str],
    ) -> Tuple[Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Prune tools based on high-confidence router decision.

        Returns:
            Tuple of (pruned_tools, updated_messages)
        """
        if not router_decision or not tools:
            return tools, messages

        # Determine pruning confidence threshold
        small_mode = os.getenv("JARVIS_SMALL_MODEL_MODE", "").strip().lower() in {"1", "true", "yes"}
        try:
            default_conf = 0.8 if small_mode else 0.85
            prune_conf = float(os.getenv("JARVIS_TOOL_ROUTER_FILTER_MIN_CONFIDENCE", str(default_conf)))
        except ValueError:
            prune_conf = 0.8 if small_mode else 0.85

        pruned_tools = prune_tools_by_router_decision(
            tools=tools,
            router_decision=router_decision,
            server_tool_names=server_tool_names,
            prune_confidence=prune_conf,
        )

        # If tools were actually pruned, rebuild system prompt
        if len(pruned_tools) < len(tools) and node_context and messages and messages[0].get("role") == "system":
            predicted_tool = router_decision.get("tool_name")
            filtered_commands = [
                cmd for cmd in available_commands
                if cmd.get("command_name") == predicted_tool
            ]
            command_flags = build_available_command_flags(filtered_commands)
            messages = self._rebuild_system_prompt(messages, node_context, timezone, pruned_tools, command_flags)
            logger.info("🧹 Rebuilt system prompt with high-confidence tool pruning")

        return pruned_tools, messages

    def _rebuild_system_prompt(
        self,
        messages: List[Dict[str, Any]],
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_command_flags: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rebuild the system prompt with updated tools and command flags.

        Returns:
            Updated messages list with new system prompt
        """
        if not hasattr(self.model, "_build_system_prompt"):
            return messages

        messages[0] = {
            "role": "system",
            "content": self.model._build_system_prompt(  # type: ignore[attr-defined]
                node_context, timezone, tools, available_command_flags
            ),
        }
        return messages
