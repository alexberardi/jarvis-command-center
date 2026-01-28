"""
Model Service for Jarvis Voice Assistant.

This service provides a clean interface to the model system and can
gradually replace the existing prompt provider architecture.
"""

import logging
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from app.core.model_factory import ModelFactory
from app.core.interfaces.imodel_interface import IModelInterface
from app.core.llm_proxy_client import LLMProxyClient
from app.core.tool_registry import tool_registry
from app.core.tool_executor import tool_executor
from app.core.tool_call_parser import tool_call_parser
from app.core.conversation_cache import conversation_cache
from app.core.utils.latency_logger import latency_logger
from app.request_models.voice_command_request import CommandDefinition
from contextlib import nullcontext

logger = logging.getLogger("uvicorn")

# Module-level singleton for tool classifier (expensive to load, reuse across requests)
_tool_classifier_instance = None


def _get_shared_tool_classifier():
    """Get or create the shared tool classifier singleton."""
    global _tool_classifier_instance
    if _tool_classifier_instance is not None:
        return _tool_classifier_instance
    if os.getenv("JARVIS_TOOL_CLASSIFIER_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return None
    try:
        from app.core.tool_router.classifier import FastTextToolClassifier
    except Exception as exc:
        logger.warning("⚠️ Tool classifier unavailable: %s", exc)
        return None
    classifier = FastTextToolClassifier.from_env()
    if classifier is None:
        return None
    _tool_classifier_instance = classifier
    logger.info("🔧 Tool classifier loaded (singleton)")
    return _tool_classifier_instance


class ModelService:
    """
    Service for managing model interactions.
    
    This service provides a clean interface to the model system and handles:
    1. Model instantiation and management
    2. Conversation warmup and cleanup
    3. Inference requests
    4. Error handling and logging
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the model service.
        
        Args:
            model_name: Specific model to use. If None, uses environment variable.
        """
        self.model: IModelInterface = ModelFactory.create_model(model_name)
        self.llm_client = LLMProxyClient()
        logger.info(f"🤖 ModelService initialized with {self.model.name}")

    def _get_tool_classifier(self):
        # Use module-level singleton for performance
        return _get_shared_tool_classifier()

    def _route_tool(self, utterance: str, tools: List[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
        classifier = self._get_tool_classifier()
        if classifier is None or not utterance:
            return None
        tool_names = set()
        for tool in tools or []:
            if isinstance(tool.get("function"), dict):
                name = tool.get("function", {}).get("name")
            else:
                name = tool.get("name")
            if name:
                tool_names.add(name)
        prediction = classifier.predict(utterance)
        if not prediction.tool_name or prediction.tool_name not in tool_names:
            return None
        return prediction.tool_name, prediction.score
    
    async def warmup_conversation(
        self,
        node_context: Dict[str, Any],
        available_commands: List[CommandDefinition],
        conversation_id: str,
        timezone: Optional[str] = None
    ) -> None:
        """
        Warm up a conversation with the model.
        
        Args:
            node_context: Node information (room, user, device, etc.)
            available_commands: List of commands available to this node
            conversation_id: Unique conversation identifier
            timezone: User's timezone for date calculations
        """
        logger.info(f"🚀 Warming up conversation {conversation_id[:8]} with {self.model.name}")
        
        try:
            await self.model.perform_warmup(
                node_context=node_context,
                available_commands=available_commands,
                conversation_id=conversation_id,
                timezone=timezone
            )
            logger.info(f"✅ Warmup completed for {conversation_id[:8]}")
            
        except Exception as e:
            logger.error(f"❌ Warmup failed for {conversation_id[:8]}: {e}")
            raise
    
    async def process_voice_command(
        self,
        voice_command: str,
        conversation_id: str,
        node_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a voice command and return the result.
        
        Args:
            voice_command: The user's voice command text
            conversation_id: Conversation ID (must match warmup call)
            node_context: Optional additional node context
            
        Returns:
            Standard Jarvis response format:
            {
                "s": bool,           # Success flag
                "n": str,            # Command name (or null)
                "p": Dict[str, Any], # Extracted parameters
                "e": Optional[Dict]  # Error object (if s=False)
            }
        """
        logger.info(f"🎯 Processing command with {self.model.name}: '{voice_command}'")
        
        try:
            result = await self.model.perform_inference(
                voice_command=voice_command,
                conversation_id=conversation_id,
                node_context=node_context
            )
            
            logger.info(f"✅ Command processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Command processing failed: {e}")
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {"type": "service_error", "message": str(e)}
            }
    
    async def cleanup_conversation(self, conversation_id: str) -> None:
        """
        Clean up a conversation.
        
        Args:
            conversation_id: Conversation to clean up
        """
        logger.info(f"🧹 Cleaning up conversation {conversation_id[:8]}")
        
        try:
            await self.model.cleanup_conversation(conversation_id)
            logger.info(f"✅ Cleanup completed for {conversation_id[:8]}")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed for {conversation_id[:8]}: {e}")
            # Don't raise - cleanup failures shouldn't break the system
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the model service.
        
        Returns:
            Health status information
        """
        try:
            model_health = await self.model.health_check()
            
            return {
                "status": "healthy",
                "service": "ModelService",
                "model": model_health,
                "capabilities": self.model.get_capabilities()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "ModelService",
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information
        """
        return {
            "name": self.model.name,
            "class": self.model.__class__.__name__,
            "capabilities": self.model.get_capabilities()
        }
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of available model names
        """
        return ModelFactory.get_available_models()
    
    @staticmethod
    def get_model_details(model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model details or None if not found
        """
        return ModelFactory.get_model_info(model_name)
    
    # Tool-Based Architecture Methods
    
    async def warmup_conversation_with_tools(
        self,
        node_context: Dict[str, Any],
        conversation_id: str,
        timezone: Optional[str] = None,
        client_tools: Optional[List[Dict[str, Any]]] = None,
        available_commands: Optional[List[CommandDefinition]] = None,
        skip_warmup_inference: bool = False
    ) -> None:
        """
        Warm up a conversation with tool support.

        Args:
            node_context: Node information (room, user, device, etc.)
            conversation_id: Unique conversation identifier
            timezone: User's timezone for date calculations
            client_tools: Client-provided tool definitions
        """
        timing = latency_logger.get_request(conversation_id)
        logger.info(f"🚀 Warming up tool-based conversation {conversation_id[:8]}")

        # Get server tools
        with timing.measure("get_server_tools") if timing else nullcontext():
            server_tools = tool_registry.get_tool_definitions()
        
        # Merge with client tools
        all_tools = server_tools + (client_tools or [])
        
        def _get_tool_name(tool: Dict[str, Any]) -> Optional[str]:
            if isinstance(tool.get("function"), dict):
                return tool.get("function", {}).get("name")
            return tool.get("name")
        
        available_command_names = {name for name in (_get_tool_name(t) for t in all_tools) if name}
        if available_commands:
            available_command_names.update(cmd.command_name for cmd in available_commands)
        
        def _filter_antipatterns(items: Optional[List[Dict[str, Any]]], source: str) -> List[Dict[str, Any]]:
            if not items:
                return []
            filtered = []
            dropped = 0
            for ap in items:
                if not isinstance(ap, dict):
                    dropped += 1
                    continue
                cmd_name = ap.get("command_name")
                desc = ap.get("description")
                if cmd_name in available_command_names and isinstance(desc, str) and desc.strip():
                    filtered.append({"command_name": cmd_name, "description": desc.strip()})
                else:
                    dropped += 1
            logger.debug(
                "🔎 Antipatterns filtered (%s): kept=%d dropped=%d",
                source,
                len(filtered),
                dropped
            )
            return filtered
        
        logger.info(f"🔧 Total tools: {len(all_tools)} ({len(server_tools)} server + {len(client_tools or [])} client)")

        # Extract examples from client_tools
        # Each client_tool has an "examples" array at the top level with objects containing:
        # voice_command, expected_parameters, and is_primary
        # Build a structure similar to available_commands format for get_command_examples tool
        command_examples_map = {}
        if client_tools:
            logger.info(f"🔍 Processing {len(client_tools)} client tool(s) to extract examples")
            for i, tool in enumerate(client_tools):
                # Extract tool name from OpenAI format
                tool_name = tool.get("function", {}).get("name") if isinstance(tool.get("function"), dict) else None
                if not tool_name:
                    # Try direct name if not in function format
                    tool_name = tool.get("name")
                
                logger.debug(f"   Tool {i+1}: name={tool_name}, keys={list(tool.keys())}")

                tool["antipatterns"] = _filter_antipatterns(tool.get("antipatterns"), f"client_tool:{tool_name or i}")
                if tool.get("antipatterns"):
                    logger.debug("✅ Antipatterns attached to client tool %s: %d", tool_name, len(tool["antipatterns"]))
                
                if tool_name:
                    # Check if this tool has an "examples" array at the top level
                    examples_array = tool.get("examples", [])
                    
                    if examples_array and isinstance(examples_array, list):
                        # Initialize command entry if not exists
                        if tool_name not in command_examples_map:
                            command_examples_map[tool_name] = {
                                "command_name": tool_name,
                                "examples": [],
                                "keywords": tool.get("keywords"),
                                "antipatterns": tool.get("antipatterns"),
                                "allow_direct_answer": tool.get("allow_direct_answer")
                            }
                        
                        # Add all examples from the array
                        for example in examples_array:
                            if isinstance(example, dict):
                                voice_command = example.get("voice_command")
                                expected_parameters = example.get("expected_parameters", {})
                                is_primary = example.get("is_primary", False)
                                
                                if voice_command:
                                    command_examples_map[tool_name]["examples"].append({
                                        "voice_command": voice_command,
                                        "expected_parameters": expected_parameters,
                                        "is_primary": is_primary
                                    })
                                    logger.debug(f"   └─ ✅ Added example for {tool_name}: '{voice_command[:50]}...'")
                                else:
                                    logger.debug("   └─ ⚠️  Example missing voice_command")
                        
                        logger.debug(f"   └─ Processed {len(examples_array)} example(s) for {tool_name}")
                    else:
                        logger.debug(f"   └─ ⚠️  No examples array found for {tool_name}")
                else:
                    logger.warning(f"   └─ ⚠️  Could not extract tool name from tool {i+1}")
            
            if command_examples_map:
                total_examples = sum(len(cmd["examples"]) for cmd in command_examples_map.values())
                logger.info(f"📚 Extracted {total_examples} example(s) from {len(command_examples_map)} client tool(s)")
            else:
                logger.warning(f"⚠️  No examples found in client_tools - get_command_examples tool will not work")
                logger.warning(f"   Expected structure: tool.examples = [{{voice_command, expected_parameters, is_primary}}]")
        
        # Convert to list format for cache
        available_commands_to_cache = list(command_examples_map.values())
        
        # Also include available_commands if provided (for examples/metadata)
        if available_commands:
            logger.info(f"📚 Also received {len(available_commands)} available commands")
            for cmd in available_commands:
                cmd_dict = cmd.model_dump()
                cmd_dict["antipatterns"] = _filter_antipatterns(cmd_dict.get("antipatterns"), f"available_command:{cmd_dict.get('command_name')}")
                if cmd_dict.get("antipatterns"):
                    logger.debug("✅ Antipatterns attached to command %s: %d", cmd_dict.get("command_name"), len(cmd_dict["antipatterns"]))
                if cmd_dict.get("keywords"):
                    logger.debug("🔑 Keywords attached to command %s: %s", cmd_dict.get("command_name"), cmd_dict.get("keywords"))
                cmd_name = cmd_dict.get("command_name")
                if cmd_name in command_examples_map:
                    # Merge examples
                    command_examples_map[cmd_name]["examples"].extend(cmd_dict.get("examples", []))
                else:
                    available_commands_to_cache.append(cmd_dict)
        
        # Build system message using the model's prompt builder when available
        available_commands_dicts = []
        if available_commands:
            available_commands_dicts.extend([cmd.model_dump() for cmd in available_commands])
        if client_tools:
            for tool in client_tools:
                tool_name = tool.get("function", {}).get("name") if isinstance(tool.get("function"), dict) else tool.get("name")
                if tool_name:
                    tool_flags = {
                        "command_name": tool_name,
                        "allow_direct_answer": tool.get("allow_direct_answer")
                    }
                    available_commands_dicts.append(tool_flags)

        with timing.measure("build_system_prompt") if timing else nullcontext():
            if hasattr(self.model, "_build_system_prompt"):
                system_message = self.model._build_system_prompt(
                    node_context,
                    timezone,
                    all_tools,
                    available_commands_dicts
                )  # type: ignore[attr-defined]
            else:
                # Fallback to default builder
                system_message = self._build_tool_system_message(node_context, timezone, all_tools, available_commands_dicts)

        messages = [
            {"role": "system", "content": system_message}
        ]

        # Store in cache (including command examples extracted from client_tools)
        with timing.measure("cache_set") if timing else nullcontext():
            conversation_cache.set(
                conversation_id=conversation_id,
                messages=messages,
                available_commands=available_commands_to_cache,
                timezone=timezone,
                tools=all_tools,
                node_context=node_context
            )
        if available_commands_to_cache:
            total_examples = sum(len(cmd.get("examples", [])) for cmd in available_commands_to_cache)
            logger.info(f"📚 Cached {len(available_commands_to_cache)} command(s) with {total_examples} total examples for get_command_examples tool")
        else:
            logger.warning(f"⚠️  Cached empty available_commands list - get_command_examples tool will return no_commands_available")
        
        # Warm up LLM proxy (tools are in system message, not passed separately)
        try:
            from pathlib import Path
            repo_root = Path(__file__).resolve().parents[2]
            prompt_path = repo_root / "temp" / "warmup_prompt.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(system_message, encoding="utf-8")
            logger.info("📝 Wrote warmup prompt to %s", prompt_path)
        except Exception as e:
            logger.warning("⚠️ Failed to write warmup prompt to file: %s", e)

        # Check if we should skip warmup inference
        # Skip if: explicitly requested OR engine doesn't benefit from prefix caching
        should_skip_warmup = skip_warmup_inference
        if not should_skip_warmup:
            # Check if engine benefits from warmup caching
            if not await self.llm_client.allows_warmup_caching():
                should_skip_warmup = True
                logger.info(f"⚡ Auto-skipping LLM warmup for {conversation_id[:8]} (engine has no prefix caching benefit)")

        if should_skip_warmup:
            logger.info(f"⏭️  Skipping LLM warmup inference for {conversation_id[:8]}")
            if timing:
                timing.checkpoint("warmup_skipped")
        else:
            try:
                # Build adapter_settings from node's adapter_hash if present
                adapter_settings = None
                adapter_hash = node_context.get("adapter_hash")
                if adapter_hash:
                    adapter_settings = {"hash": adapter_hash, "enabled": True}
                    logger.info(f"🔌 Using adapter {adapter_hash[:8]}... for warmup")

                logger.info(f"🔥 Calling LLM proxy warmup_conversation for {conversation_id[:8]}")
                with timing.measure("llm_warmup_call") if timing else nullcontext():
                    result = await self.llm_client.warmup_conversation(
                        conversation_id=conversation_id,
                        messages=messages,
                        adapter_settings=adapter_settings
                    )
                logger.info(f"✅ Tool-based warmup completed for {conversation_id[:8]}")
                logger.info(f"   Warmup result type: {type(result)}")
                if isinstance(result, dict):
                    logger.info(f"   Warmup result keys: {list(result.keys())}")
                    import json
                    result_str = json.dumps(result, indent=2)
                    if len(result_str) > 2000:
                        logger.info(f"   Warmup result (first 2000 chars): {result_str[:2000]}...")
                    else:
                        logger.info(f"   Warmup result: {result_str}")
                else:
                    logger.info(f"   Warmup result (not a dict): {result}")
            except Exception as e:
                logger.error(f"❌ Warmup failed for {conversation_id[:8]}: {e}")
                logger.error(f"   Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                raise

    def _filter_tools_for_utterance(
        self,
        voice_command: str,
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        # Keyword-based filtering disabled: always keep the full tool list.
        return tools
    
    async def process_voice_command_with_tools(
        self,
        voice_command: str,
        conversation_id: str
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

        # Get conversation state
        with timing.measure("cache_lookups") if timing else nullcontext():
            messages = conversation_cache.get_messages(conversation_id)
            tools = conversation_cache.get_tools(conversation_id)
            available_commands = conversation_cache.get_available_commands(conversation_id) or []
            node_context = conversation_cache.get_node_context(conversation_id) or {}
            timezone = conversation_cache.get_timezone(conversation_id)

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Cache lookups done")

        if not messages:
            raise ValueError(f"Conversation {conversation_id} not found or expired")
        if tools:
            filtered_tools = self._filter_tools_for_utterance(voice_command, tools, available_commands)
            if filtered_tools is not tools:
                tools = filtered_tools
                if node_context:
                    def _tool_name(tool: Dict[str, Any]) -> Optional[str]:
                        if isinstance(tool.get("function"), dict):
                            return tool.get("function", {}).get("name")
                        return tool.get("name")

                    filtered_command_names = {
                        _tool_name(tool) for tool in tools
                        if _tool_name(tool) and not tool_registry.has_tool(_tool_name(tool))
                    }
                    filtered_available = [
                        cmd for cmd in available_commands
                        if cmd.get("command_name") in filtered_command_names
                    ]
                    available_command_flags = [
                        {
                            "command_name": cmd.get("command_name"),
                            "allow_direct_answer": cmd.get("allow_direct_answer")
                        }
                        for cmd in filtered_available
                    ]
                    if hasattr(self.model, "_build_system_prompt") and messages and messages[0].get("role") == "system":
                        messages[0] = {
                            "role": "system",
                            "content": self.model._build_system_prompt(node_context, timezone, tools, available_command_flags)  # type: ignore[attr-defined]
                        }
                        logger.info("🧹 Rebuilt system prompt with filtered tools")

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool filtering done")

        # Optional tool routing classifier (phase 1)
        router_decision = None
        _router_t0 = time.time()
        if self.model.use_tool_classifier:
            routed = self._route_tool(voice_command, tools or [])
            if routed:
                tool_name, score = routed
                try:
                    min_conf = float(os.getenv("JARVIS_TOOL_CLASSIFIER_MIN_CONFIDENCE", "0.6"))
                except ValueError:
                    min_conf = 0.6
                use_hint = score >= min_conf
                router_decision = {"tool_name": tool_name, "score": score, "used": use_hint}
                conversation_cache.set_router_decision(conversation_id, router_decision)
                logger.info("🧭 Router predicted tool=%s score=%.3f use_hint=%s", tool_name, score, use_hint)
            else:
                conversation_cache.set_router_decision(conversation_id, None)
        else:
            logger.info("🔇 Tool classifier disabled for %s", self.model.name)
            conversation_cache.set_router_decision(conversation_id, None)

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool routing done (router took {(time.time()-_router_t0)*1000:.0f}ms)")

        # High-confidence tool pruning (reduce prompt size)
        if router_decision:
            small_mode = os.getenv("JARVIS_SMALL_MODEL_MODE", "").strip().lower() in {"1", "true", "yes"}
            try:
                default_conf = 0.8 if small_mode else 0.85
                prune_conf = float(os.getenv("JARVIS_TOOL_ROUTER_FILTER_MIN_CONFIDENCE", str(default_conf)))
            except ValueError:
                prune_conf = 0.8 if small_mode else 0.85
            if router_decision.get("score", 0.0) >= prune_conf:
                predicted_tool = router_decision.get("tool_name")
                def _tool_name(tool: Dict[str, Any]) -> Optional[str]:
                    if isinstance(tool.get("function"), dict):
                        return tool.get("function", {}).get("name")
                    return tool.get("name")
                pruned_tools = []
                for tool in tools or []:
                    name = _tool_name(tool)
                    if not name:
                        continue
                    if name == predicted_tool or tool_registry.has_tool(name):
                        pruned_tools.append(tool)
                if pruned_tools and len(pruned_tools) < len(tools or []):
                    tools = pruned_tools
                    if node_context and messages and messages[0].get("role") == "system":
                        filtered_available = [
                            cmd for cmd in (available_commands or [])
                            if cmd.get("command_name") == predicted_tool
                        ]
                        available_command_flags = [
                            {
                                "command_name": cmd.get("command_name"),
                                "allow_direct_answer": cmd.get("allow_direct_answer")
                            }
                            for cmd in filtered_available
                        ]
                        if hasattr(self.model, "_build_system_prompt"):
                            messages[0] = {
                                "role": "system",
                                "content": self.model._build_system_prompt(node_context, timezone, tools, available_command_flags)  # type: ignore[attr-defined]
                            }
                            logger.info("🧹 Rebuilt system prompt with high-confidence tool pruning")
        
        if router_decision and router_decision.get("used"):
            hint_tool = router_decision.get("tool_name")
            messages.append({
                "role": "system",
                "content": f"Router hint: likely tool is '{hint_tool}'. Use it if it matches intent; otherwise choose the best tool."
            })
        
        # Add user message
        messages.append({"role": "user", "content": voice_command})

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Starting tool execution loop")

        # Process with tool execution loop
        with timing.measure("tool_execution_loop") if timing else nullcontext():
            result = await self._tool_execution_loop(
                conversation_id=conversation_id,
                messages=messages,
                tools=tools
            )

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool execution loop completed")

        # Update cache with final messages
        conversation_cache.update_messages(conversation_id, messages)
        if result.get("stop_reason") == "error":
            logger.error("❌ Tool loop returned stop_reason=error: %s", result)
        return result
    
    async def continue_conversation_with_tool_results(
        self,
        conversation_id: str,
        tool_results: List[Dict[str, Any]]
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
            # Convert output to string if needed
            output = result["output"]
            if not isinstance(output, str):
                output = json.dumps(output)
            
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": output
            })
        
        # Continue with tool execution loop
        result = await self._tool_execution_loop(
            conversation_id=conversation_id,
            messages=messages,
            tools=tools
        )
        
        # Update cache
        conversation_cache.update_messages(conversation_id, messages)
        
        return result
    
    async def _tool_execution_loop(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Execute the tool loop: call LLM, execute server tools, repeat until done.
        
        Args:
            conversation_id: Conversation ID
            messages: Current message history (will be modified in place)
            tools: Available tools
            max_iterations: Maximum tool execution iterations
            
        Returns:
            Response dict with stop_reason and relevant data
        """
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

        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Build adapter_settings from node's adapter_hash if present
        node_context = conversation_cache.get_node_context(conversation_id) or {}
        adapter_settings = None
        adapter_hash = node_context.get("adapter_hash")
        if adapter_hash:
            adapter_settings = {"hash": adapter_hash, "enabled": True}
            logger.info(f"🔌 Using adapter {adapter_hash[:8]}... for tool loop")

        def _write_usage_log(turns_used: int, status: str) -> None:
            log_path = os.getenv(
                "JARVIS_LLM_USAGE_LOG_PATH",
                "/app/temp/llm_usage.log"
            )
            timestamp = datetime.utcnow().isoformat() + "Z"
            entry = (
                f"{timestamp} conversation_id={conversation_id} "
                f"turns={turns_used} status={status} "
                f"prompt_tokens={usage_totals['prompt_tokens']} "
                f"completion_tokens={usage_totals['completion_tokens']} "
                f"total_tokens={usage_totals['total_tokens']}\n"
            )
            try:
                log_dir = os.path.dirname(log_path)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(entry)
            except Exception as exc:
                logger.warning("⚠️ Failed to write usage log: %s", exc)

        def _write_prompt_response_log(
            prompt_messages: List[Dict[str, Any]],
            response_obj: Optional[Dict[str, Any]],
            raw_content: Optional[str],
            error: Optional[str],
            iteration: int
        ) -> None:
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
                logger.warning("⚠️ Failed to write LLM trace log: %s", exc)

        def _invalid_param_retry_count() -> int:
            retry_tag = "[INVALID_PARAM_RETRY]"
            return sum(
                1
                for msg in messages
                if msg.get("role") == "system" and retry_tag in msg.get("content", "")
            )

        small_mode = os.getenv("JARVIS_SMALL_MODEL_MODE", "").strip().lower() in {"1", "true", "yes"}

        def _env_int(name: str, default_value: int) -> int:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default_value
            try:
                return int(raw)
            except ValueError:
                return default_value

        timezone_str = conversation_cache.get_timezone(conversation_id)

        def _normalize_date_key(raw: str) -> str:
            text = raw.strip().lower()
            text = re.sub(r"\s+", "_", text)
            text = text.replace(":", "_")
            return text

        def _flatten_date_context(nested_context: Dict[str, Any]) -> Dict[str, Any]:
            flat: Dict[str, Any] = {}
            if not isinstance(nested_context, dict):
                return flat

            current = nested_context.get("current", {})
            if isinstance(current, dict) and isinstance(current.get("utc_start_of_day"), str):
                flat["today"] = current["utc_start_of_day"]

            relative = nested_context.get("relative_dates", {})
            if isinstance(relative, dict):
                for key, value in relative.items():
                    if not isinstance(value, dict):
                        continue
                    if isinstance(value.get("utc_start_of_day"), str):
                        flat[key] = value["utc_start_of_day"]
                    elif isinstance(value.get("datetime"), str):
                        flat[key] = value["datetime"]

            for bucket_name in ("weekend", "weeks", "months", "years"):
                bucket = nested_context.get(bucket_name, {})
                if not isinstance(bucket, dict):
                    continue
                for key, value in bucket.items():
                    if not isinstance(value, list):
                        continue
                    dates = [
                        item.get("utc_start_of_day")
                        for item in value
                        if isinstance(item, dict) and isinstance(item.get("utc_start_of_day"), str)
                    ]
                    if dates:
                        flat[key] = dates

            weekdays = nested_context.get("weekdays", {})
            if isinstance(weekdays, dict):
                for key, value in weekdays.items():
                    if isinstance(value, dict) and isinstance(value.get("utc_start_of_day"), str):
                        flat[key] = value["utc_start_of_day"]

            this_week = nested_context.get("weeks", {}).get("this_week", [])
            if isinstance(this_week, list):
                for entry in this_week:
                    if not isinstance(entry, dict):
                        continue
                    day = entry.get("day")
                    if isinstance(day, str) and isinstance(entry.get("utc_start_of_day"), str):
                        flat[f"this_{day.strip().lower()}"] = entry["utc_start_of_day"]

            time_expressions = nested_context.get("time_expressions", {})
            if isinstance(time_expressions, dict):
                for key, value in time_expressions.items():
                    if isinstance(value, str):
                        flat[_normalize_date_key(key)] = value

            return flat

        def _is_datetime_param(param_schema: Dict[str, Any]) -> bool:
            if not isinstance(param_schema, dict):
                return False
            if param_schema.get("format") == "date-time":
                return True
            if param_schema.get("type") == "array":
                items = param_schema.get("items", {})
                return isinstance(items, dict) and items.get("format") == "date-time"
            return False

        def _is_datetime_array(param_schema: Dict[str, Any]) -> bool:
            if not isinstance(param_schema, dict):
                return False
            if param_schema.get("type") != "array":
                return False
            items = param_schema.get("items", {})
            return isinstance(items, dict) and items.get("format") == "date-time"

        def _parse_time_string(time_str: str) -> Tuple[int, int]:
            match = re.match(r"(\d+)_(\d+)(am|pm)", time_str)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                if match.group(3) == "pm" and hour != 12:
                    hour += 12
                elif match.group(3) == "am" and hour == 12:
                    hour = 0
                return hour, minute

            match = re.match(r"(\d+)(am|pm)", time_str)
            if match:
                hour = int(match.group(1))
                if match.group(2) == "pm" and hour != 12:
                    hour += 12
                elif match.group(2) == "am" and hour == 12:
                    hour = 0
                return hour, 0

            return 0, 0

        def _apply_time_modifier(base_datetime: str, modifier: str) -> Optional[str]:
            try:
                normalized = base_datetime.replace("Z", "+00:00") if base_datetime.endswith("Z") else base_datetime
                dt = datetime.fromisoformat(normalized)
            except ValueError:
                return None

            time_map = {
                "morning": 7,
                "afternoon": 13,
                "evening": 18,
                "night": 21,
                "noon": 12,
                "midnight": 0,
            }
            if modifier in time_map:
                dt = dt.replace(hour=time_map[modifier], minute=0, second=0, microsecond=0)
            elif modifier.startswith("at_"):
                hour, minute = _parse_time_string(modifier[3:])
                dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)

            return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        def _resolve_date_keys(date_keys: List[str], date_context: Dict[str, Any]) -> List[str]:
            if not date_keys:
                return []
            normalized_keys = [_normalize_date_key(key) for key in date_keys if isinstance(key, str)]
            flat_context = _flatten_date_context(date_context)

            resolved: List[str] = []
            for key in normalized_keys:
                value = flat_context.get(key)
                if isinstance(value, list):
                    resolved.extend([v for v in value if isinstance(v, str)])
                elif isinstance(value, str):
                    resolved.append(value)

            date_key = None
            for key in normalized_keys:
                value = flat_context.get(key)
                if isinstance(value, str):
                    date_key = key
                    break

            time_key = None
            for key in normalized_keys:
                if key in {"morning", "afternoon", "evening", "night", "noon", "midnight"} or key.startswith("at_"):
                    time_key = key
                    break

            if date_key and time_key:
                base = flat_context.get(date_key)
                if isinstance(base, str):
                    combined = _apply_time_modifier(base, time_key)
                    if combined:
                        resolved.append(combined)
                elif isinstance(base, list):
                    for entry in base:
                        if not isinstance(entry, str):
                            continue
                        combined = _apply_time_modifier(entry, time_key)
                        if combined:
                            resolved.append(combined)

            seen = set()
            unique = []
            for entry in resolved:
                if entry in seen:
                    continue
                seen.add(entry)
                unique.append(entry)
            return unique

        def _inject_date_keys(
            tool_calls: List[Dict[str, Any]],
            date_keys: List[str],
            tools_list: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            if not tool_calls or not date_keys:
                return tool_calls
            from app.core.general_context import generate_date_context_object

            date_context = generate_date_context_object(timezone_str)
            resolved_dates = _resolve_date_keys(date_keys, date_context)
            if not resolved_dates:
                return tool_calls

            tool_map: Dict[str, Dict[str, Any]] = {}
            for tool in tools_list or []:
                if isinstance(tool.get("function"), dict):
                    name = tool.get("function", {}).get("name")
                    if name:
                        tool_map[name] = tool.get("function", {})
                elif isinstance(tool.get("name"), str):
                    tool_map[tool["name"]] = tool

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
                except Exception:
                    args_obj = {}
                if not isinstance(args_obj, dict):
                    continue

                mutated = False
                for param_name, param_schema in param_schemas.items():
                    if not _is_datetime_param(param_schema):
                        continue
                    existing = args_obj.get(param_name)
                    if existing not in (None, [], ""):
                        continue
                    if _is_datetime_array(param_schema):
                        args_obj[param_name] = resolved_dates
                    else:
                        args_obj[param_name] = resolved_dates[0]
                    mutated = True

                if mutated:
                    call["function"]["arguments"] = json.dumps(args_obj)

            return tool_calls

        def _normalize_param_type(param_type: Optional[str]) -> Tuple[Optional[str], bool]:
            if not param_type:
                return None, False
            raw = param_type.strip().lower()
            if raw.startswith("array<") and raw.endswith(">"):
                return raw[len("array<"):-1].strip(), True
            if raw.startswith("array[") and raw.endswith("]"):
                return raw[len("array["):-1].strip(), True
            if raw.endswith("[]"):
                return raw[:-2].strip(), True
            return raw, False

        def _is_iso_date(value: str) -> bool:
            return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", value))

        def _is_iso_datetime(value: str) -> bool:
            if "T" not in value:
                return False
            normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return False
            return parsed.tzinfo is not None

        def _validate_scalar(value: Any, base_type: str) -> bool:
            if base_type in {"string"}:
                return isinstance(value, str)
            if base_type in {"int", "integer"}:
                return isinstance(value, int) and not isinstance(value, bool)
            if base_type in {"float", "number", "double"}:
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if base_type in {"bool", "boolean"}:
                return isinstance(value, bool)
            if base_type == "date":
                return isinstance(value, str) and _is_iso_date(value)
            if base_type == "datetime":
                return isinstance(value, str) and _is_iso_datetime(value)
            return True

        def _validate_value(value: Any, base_type: str, is_array: bool) -> bool:
            if is_array:
                if not isinstance(value, list):
                    return False
                return all(_validate_scalar(item, base_type) for item in value)
            return _validate_scalar(value, base_type)

        def _build_param_type_map() -> Dict[str, Dict[str, str]]:
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

        def _find_invalid_params(client_calls: List[Dict[str, Any]]) -> List[str]:
            invalid: List[str] = []
            param_types = _build_param_type_map()
            if not param_types:
                return invalid
            for call in client_calls:
                tool_name = call.get("function", {}).get("name")
                if not tool_name or tool_name not in param_types:
                    continue
                args_raw = call.get("function", {}).get("arguments", "{}")
                try:
                    args_obj = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                except Exception:
                    args_obj = {}
                if not isinstance(args_obj, dict):
                    continue
                for param_name, param_type in param_types[tool_name].items():
                    if param_name not in args_obj:
                        continue
                    base_type, is_array = _normalize_param_type(param_type)
                    if not base_type:
                        continue
                    if not _validate_value(args_obj.get(param_name), base_type, is_array):
                        invalid.append(f"{tool_name}.{param_name} expected {param_type}")
            return invalid

        import time as _time
        timing = latency_logger.get_request(conversation_id)

        for iteration in range(max_iterations):
            logger.info(f"🔁 Tool loop iteration {iteration + 1}/{max_iterations}")
            prompt_snapshot = [dict(m) for m in (messages or [])]

            # Call LLM (tools are in system message, not passed separately)
            try:
                response_format = self._get_response_format()
                _llm_start = _time.time()
                with timing.measure(f"llm_call_iter_{iteration+1}") if timing else nullcontext():
                    response = await self.llm_client.chat_completion(
                        messages=messages,
                        conversation_id=conversation_id,
                        response_format=response_format,
                        include_date_context=True,
                        adapter_settings=adapter_settings
                    )
                logger.debug(f"⏱️  LLM call took {(_time.time()-_llm_start)*1000:.0f}ms")
            except Exception as e:
                logger.error(f"❌ LLM call failed: {e}")
                _write_prompt_response_log(prompt_snapshot, None, None, str(e), iteration + 1)
                _write_usage_log(iteration + 1, "error")
                return {
                    "stop_reason": "error",
                    "error": str(e)
                }
            
            # Parse response - LLM proxy returns OpenAI-compatible format
            logger.info(f"🔍 Full LLM proxy response structure: {list(response.keys())}")
            
            # Extract content from OpenAI format: choices[0].message.content
            try:
                raw_content = response["choices"][0]["message"]["content"]
                finish_reason_raw = response["choices"][0].get("finish_reason", "stop")
                logger.info(f"🔍 Extracted content length: {len(raw_content)}, finish_reason: {finish_reason_raw}")
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"❌ Failed to extract content from response: {e}")
                logger.error(f"Response structure: {response}")
                raw_content = ""

            date_keys: List[str] = []
            if isinstance(response, dict):
                raw_keys = response.get("date_keys")
                if isinstance(raw_keys, list):
                    date_keys = [_normalize_date_key(key) for key in raw_keys if isinstance(key, str)]

            # Accumulate token usage if present
            if isinstance(response, dict):
                usage = response.get("usage")
                if isinstance(usage, dict):
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if isinstance(usage.get(key), int):
                            usage_totals[key] += usage[key]
            
            if not raw_content:
                logger.warning(f"⚠️ Empty raw_content!")
            response_obj = response if isinstance(response, dict) else None
            _write_prompt_response_log(prompt_snapshot, response_obj, raw_content, None, iteration + 1)
            
            # Use tool call parser to extract tool calls from JSON
            finish_reason, tool_calls, assistant_message = tool_call_parser.parse_response(raw_content)
            
            logger.info(f"📥 LLM response parsed: finish_reason={finish_reason}, tool_calls={len(tool_calls)}")
            
            # Add assistant message to history (store the raw JSON output)
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
                    logger.info("🔁 Must-call guard triggered; retrying tool selection")
                    continue
                # Conversation complete
                _write_usage_log(iteration + 1, "complete")
                return {
                    "stop_reason": "complete",
                    "assistant_message": assistant_message
                }
            
            elif finish_reason == "tool_calls":
                if date_keys:
                    tool_calls = _inject_date_keys(tool_calls, date_keys, tools)
                    logger.info("📅 Injected date keys into tool calls: %s", date_keys)

                # Execute tools
                with timing.measure(f"tool_exec_iter_{iteration+1}") if timing else nullcontext():
                    server_results, client_calls = tool_executor.execute_tool_calls(tool_calls, conversation_id=conversation_id)
                
                # Check if request_validation was called
                validation_called = any(
                    tool_call.get("function", {}).get("name") == "request_validation"
                    for tool_call in tool_calls
                )
                
                # Check if get_command_examples was called - if so, we need to continue the loop
                # so the LLM can see the examples before making decisions
                get_examples_called = any(
                    tool_call.get("function", {}).get("name") == "get_command_examples"
                    for tool_call in tool_calls
                )
                
                # Always add server results to messages first (so LLM can see them)
                if server_results:
                    messages.extend(server_results)
                    logger.info(f"✅ Executed {len(server_results)} server tools, added results to conversation")

                # If get_command_examples was called, we MUST continue the loop so the LLM can see the examples
                # before making any decisions (including validation requests)
                if get_examples_called:
                    logger.info(f"🔄 get_command_examples was called, continuing loop so LLM can see examples before making decisions")
                    continue
                
                # If request_validation was called WITH other tools, continue the loop to let other tools execute first
                # Only return validation if it's the ONLY tool call
                other_tools_called = len(tool_calls) > 1 or client_calls or get_examples_called
                if validation_called and other_tools_called:
                    logger.info(f"🔄 request_validation called with other tools - continuing loop to execute other tools first")
                    continue
                
                # Check for validation requests in server results (only if it's the only tool call)
                if validation_called and not other_tools_called:
                    for result in server_results:
                        try:
                            content = json.loads(result["content"])
                            if content.get("_validation_request"):
                                logger.info(f"🔍 Validation request detected (only tool call)")
                                _write_usage_log(iteration + 1, "validation_required")
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
                
                # If there are server tools, we need to continue the loop so the LLM can see the results
                # and potentially update client tool calls with the new information
                if server_results and client_calls:
                    logger.info(f"🔄 Server tools executed, continuing loop so LLM can see results before client tools")
                    logger.info(f"   Client tools will be called in next iteration after LLM processes server results")
                    # Continue loop - LLM will see server results and can update client tool calls
                    continue
                
                # If there are client tool calls and NO server tools, return them immediately
                if client_calls:
                    invalid_params = _find_invalid_params(client_calls)
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
                            logger.info("🔁 Invalid parameter guard triggered; retrying tool call formatting")
                            continue
                    logger.info(f"📤 Returning {len(client_calls)} client tool calls (no server tools to wait for)")
                    _write_usage_log(iteration + 1, "tool_calls")
                    return {
                        "stop_reason": "tool_calls",
                        "tool_calls": client_calls,
                        "assistant_message": assistant_message
                    }
                
                # Only server tools - already added to messages, continue loop
                if server_results:
                    logger.info(f"✅ Only server tools executed, continuing loop")
                
            else:
                # Unknown finish reason
                logger.warning(f"⚠️ Unknown finish_reason: {finish_reason}")
                _write_usage_log(iteration + 1, "complete")
                return {
                    "stop_reason": "complete",
                    "assistant_message": assistant_message
                }
        
        # Max iterations reached
        logger.warning(f"⚠️ Max tool loop iterations ({max_iterations}) reached")
        _write_usage_log(max_iterations, "max_iterations_exceeded")
        return {
            "stop_reason": "complete",
            "assistant_message": "Maximum tool execution iterations reached.",
            "error": "max_iterations_exceeded"
        }

    def _get_response_format(self) -> Dict[str, Any]:
        return {
            "type": "json_object",
            "json_schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {"type": "object"},
                            },
                            "required": ["name", "arguments"],
                        },
                    },
                    "error": {
                        "type": ["object", "null"],
                        "properties": {
                            "type": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["type", "message"],
                    },
                },
                "required": ["message", "tool_calls"],
                "additionalProperties": True,
            },
        }
    
    def _build_tool_system_message(
        self, 
        node_context: Dict[str, Any], 
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build system message for tool-based conversations.
        
        Args:
            node_context: Node context information
            timezone: User's timezone
            tools: Available tools
            available_commands: Optional available commands for examples
            
        Returns:
            System message string
        """
        prompt_style = os.getenv("JARVIS_PROMPT_STYLE", "compact").strip().lower()
        
        # Format tools for prompt with primary examples
        tools_text = tool_call_parser.format_tools_with_examples(tools, available_commands)
        
        include_reason = os.getenv("JARVIS_INCLUDE_REASON", "").strip().lower() in {"1", "true", "yes"}
        reason_suffix_inline = ', "reason": "<optional>"' if include_reason else ""
        reason_suffix_block = '\n  "reason": "<optional>"' if include_reason else ""

        if prompt_style == "full":
            system_msg = f"""You are Jarvis, a voice-controlled assistant that operates BY CALLING TOOLS.

Node Context:
- Room: {node_context.get('room', 'unknown')}
- User: {node_context.get('user', 'default')}
- Voice Mode: {node_context.get('voice_mode', 'brief')}

YOUR PRIMARY ROLE: You are a tool router and parameter extractor.
- Analyze the user's request
- Determine which available tool(s) to call
- Extract parameters from their message
- Call the appropriate tool(s)

CRITICAL - Tool Execution Order:
- Call tools ONE AT A TIME in the order they need to be executed.
- Do NOT select the final command/tool until all parameters have been resolved.
- If parameters need resolution (dates, examples, etc.), resolve them FIRST before calling the final tool.
- Read tool descriptions to understand their purpose and when to use them.

CRITICAL: You do NOT have direct access to information or the ability to perform actions yourself.
- When a user asks a question or makes a request, you MUST use the available tools
- DO NOT attempt to answer questions, provide information, or perform actions without calling a tool
- Review the available tools and select the one that best matches the user's intent
- After you receive tool results, THEN you format the data into a natural spoken response for the user

Think of yourself as operating in two phases:
1. TOOL CALLING: Route the request to the appropriate tool(s) - you cannot answer directly
2. RESPONSE FORMATTING: Once you receive tool results, format them into a conversational response for the user

EXCEPTION: If it's something you can absolutely figure out without a tool (like basic math or who you are), feel free to respond directly.

CRITICAL - Response Format (VALID JSON ONLY):
You MUST respond with valid JSON only. No comments, no explanations, no markdown, no code blocks.
- Comments (like # ...) are NOT allowed in JSON
- Only valid JSON syntax is accepted - any other text will cause errors
{"- You may include an optional 'reason' field explaining tool/parameter choices." if include_reason else ""}

When calling a tool (ONE at a time):
{{
  "message": "Brief acknowledgment",
  "tool_call": {{"name": "tool_name", "arguments": {{"param_name": "param_value"}}}}{reason_suffix_block}
}}

When you have the final answer:
{{
  "message": "Your natural spoken response",
  "tool_call": null{reason_suffix_block}
}}

Available Tools:
{tools_text}

Key Guidelines:
- Extract parameters directly from user's message when they are clearly stated. If the user says "Seattle", extract "Seattle" as the city parameter - do not ask for validation when the information is already provided.
- Only ask for validation if a required parameter is truly missing or genuinely ambiguous after trying all other options.
- Be conversational and {node_context.get('voice_mode', 'brief')}"""
        else:
            system_msg = f"""You are Jarvis. Always call tools; respond with VALID JSON only.

Context: room={node_context.get('room', 'unknown')}, user={node_context.get('user', 'default')}, style={node_context.get('voice_mode', 'brief')}

Rules:
- Call ONE tool at a time.
- Relative dates are resolved automatically; use natural terms like 'tomorrow' or 'next_week' in date parameters.
- Ask for validation only if required params are missing/ambiguous.
{"- You may include an optional 'reason' field explaining tool/parameter choices." if include_reason else ""}

Format:
- Tool call: {{"message":"brief ack","tool_call":{{"name":"tool_name","arguments":{{...}}}}{reason_suffix_inline}}}
- Final: {{"message":"concise reply","tool_call":null{reason_suffix_inline}}}

Tools:
{tools_text}"""
        
        return system_msg
