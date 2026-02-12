"""
Jarvis Tool Model - Modern Tool-Based Conversation Model

This model implementation is designed for the tool-based architecture using
prompt-based tool calling. It works with generic open source models (Llama, Mistral, etc.)
that can follow JSON formatting instructions.

Key Features:
- Tool-aware system prompts
- JSON response parsing
- Natural language final responses
- Clarifying questions support
- Multi-turn conversation handling
"""

import logging
import os
from typing import Dict, Any, List, Optional

from app.core.interfaces.imodel_interface import IModelInterface
from app.core.llm_proxy_client import LLMProxyClient
from app.core.tool_call_parser import tool_call_parser
from app.core.tool_registry import tool_registry
from app.core.tool_executor import tool_executor
from app.core.conversation_cache import conversation_cache
from app.request_models.voice_command_request import CommandDefinition

logger = logging.getLogger("uvicorn")


class JarvisToolModel(IModelInterface):
    """
    Tool-based model implementation for Jarvis.
    
    This model uses prompt engineering to teach generic open source models
    how to use tools through JSON-formatted responses. No fine-tuning required.
    
    Best used with:
    - Llama 3 (8B, 70B)
    - Mistral 7B+
    - Mixtral 8x7B
    - Any instruct model with strong JSON following capabilities
    """
    
    @property
    def name(self) -> str:
        return "JarvisToolModel"
    
    def __init__(self):
        """Initialize the tool-based model."""
        self.llm_client = LLMProxyClient()
        logger.info(f"🤖 Initialized {self.name} for tool-based conversations")
    
    async def perform_warmup(
        self,
        node_context: Dict[str, Any],
        available_commands: List[CommandDefinition],
        conversation_id: str,
        timezone: Optional[str] = None
    ) -> None:
        """
        Warm up a tool-based conversation.
        
        Note: available_commands are converted to tools internally when provided.
        For pure tool-based usage, use warmup_conversation_with_tools instead.
        """
        logger.info(f"🚀 Warming up tool conversation {conversation_id[:8]}")
        
        # Convert CommandDefinition commands to tools (if provided)
        client_tools = self._convert_commands_to_tools(available_commands or [])
        
        # Get server tools and merge
        server_tools = tool_registry.get_tool_definitions()
        all_tools = server_tools + client_tools
        
        # Get available_commands for examples
        available_commands_dicts = [cmd.model_dump() for cmd in (available_commands or [])]
        
        # Build system message (includes tools + examples)
        system_message = self._build_system_prompt(node_context, timezone, all_tools, available_commands_dicts)

        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Store in cache
        conversation_cache.set(
            conversation_id=conversation_id,
            messages=messages,
            available_commands=available_commands or [],
            timezone=timezone,
            tools=all_tools,
            node_context=node_context
        )
        
        # Warm up LLM proxy
        try:
            await self.llm_client.warmup_conversation(
                conversation_id=conversation_id,
                messages=messages
            )
            logger.info(f"✅ Warmup completed for {conversation_id[:8]}")
        except Exception as e:
            logger.error(f"❌ Warmup failed: {e}")
            raise
    
    async def perform_inference(
        self,
        voice_command: str,
        conversation_id: str,
        node_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a voice command using tool-based conversation.
        
        This handles the complete flow:
        1. Add user message to history
        2. Call LLM with tools in prompt
        3. Parse JSON response
        4. Execute server tools automatically
        5. Loop until complete or client tool needed
        6. Return natural language response
        """
        logger.info(f"🎯 Processing: '{voice_command}'")
        
        # Get conversation state
        messages = conversation_cache.get_messages(conversation_id)
        tools = conversation_cache.get_tools(conversation_id)
        
        if not messages:
            return self._error_response("Conversation not found or expired")
        
        # Detect and resolve relative dates in the utterance
        timezone = conversation_cache.get_timezone(conversation_id)
        from app.core.date_detector import detect_and_resolve_relative_dates
        
        date_resolution = detect_and_resolve_relative_dates(voice_command, timezone)
        
        # If relative dates were detected and resolved, add them as a system message
        if date_resolution["summary"]:
            messages.append({
                "role": "system",
                "content": date_resolution["summary"]
            })
            logger.info(f"📅 Added resolved dates to prompt: {len(date_resolution['resolved_dates'])} date(s)")
        
        # Add user message
        messages.append({"role": "user", "content": voice_command})
        
        # Execute tool loop
        try:
            result = await self._tool_execution_loop(
                conversation_id=conversation_id,
                messages=messages,
                tools=tools
            )
            
            # Update cache
            conversation_cache.update_messages(conversation_id, messages)
            
            # Convert to Jarvis response format
            return self._convert_to_response_format(result)
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            return self._error_response(str(e))
    
    async def cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up conversation resources."""
        logger.info(f"🧹 Cleaning up conversation {conversation_id[:8]}")
        conversation_cache.remove(conversation_id)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return model capabilities."""
        return {
            "supports_single_shot": False,
            "supports_streaming": False,
            "supports_tools": True,
            "supports_multi_turn": True,
            "supports_clarification": True,
            "max_context_length": None,
            "supported_languages": ["en"],
            "custom_features": {
                "tool_based": True,
                "prompt_engineering": True,
                "json_parsing": True
            }
        }
    
    # Private Methods
    
    async def _tool_execution_loop(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Execute the tool loop: call LLM, parse JSON, execute tools, repeat.
        
        Returns final result when:
        - LLM provides final response (no tool calls)
        - Client tool execution needed
        - Validation required
        - Max iterations reached
        """
        # Build adapter_settings from node's adapter_hash if present
        node_context = conversation_cache.get_node_context(conversation_id) or {}
        adapter_settings = None
        adapter_hash = node_context.get("adapter_hash")
        if adapter_hash:
            adapter_settings = {"hash": adapter_hash, "enabled": True}
            logger.info(f"🔌 Using adapter {adapter_hash[:8]}... for tool loop")

        for iteration in range(max_iterations):
            logger.info(f"🔁 Tool loop iteration {iteration + 1}/{max_iterations}")

            # Call LLM
            try:
                response_format = self._get_response_format()
                response = await self.llm_client.chat_completion(
                    messages=messages,
                    conversation_id=conversation_id,
                    response_format=response_format,
                    adapter_settings=adapter_settings
                )
            except Exception as e:
                logger.error(f"❌ LLM call failed: {e}")
                return {"error": str(e), "iteration": iteration}
            
            # Parse response
            raw_content = ""
            if isinstance(response, dict):
                raw_content = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                ) or response.get("content", "") or response.get("response", "") or response.get("message", {}).get("content", "")
            elif isinstance(response, str):
                raw_content = response
            finish_reason, tool_calls, assistant_message = tool_call_parser.parse_response(raw_content)
            
            logger.info(f"📥 Parsed: finish_reason={finish_reason}, tools={len(tool_calls)}, message='{assistant_message[:50]}...'")
            
            # Add assistant message to history
            assistant_msg = {"role": "assistant", "content": raw_content}
            messages.append(assistant_msg)
            
            # Handle based on finish reason
            if finish_reason == "stop":
                # Conversation complete - return natural language response
                return {
                    "complete": True,
                    "message": assistant_message,
                    "iterations": iteration + 1
                }
            
            elif finish_reason == "tool_calls":
                # Execute tools
                server_results, client_calls = tool_executor.execute_tool_calls(tool_calls, conversation_id=conversation_id)
                
                # Check for validation requests
                validation = self._check_for_validation(server_results)
                if validation:
                    return {
                        "validation_required": True,
                        "validation": validation,
                        "iterations": iteration + 1
                    }
                
                # Always add server results to messages first (so LLM can see them)
                if server_results:
                    messages.extend(server_results)
                    logger.info(f"✅ Executed {len(server_results)} server tools, added results to conversation")
                
                # If there are server tools, we need to continue the loop so the LLM can see the results
                # and potentially update client tool calls with the new information
                if server_results and client_calls:
                    logger.info(f"🔄 Server tools executed, continuing loop so LLM can see results before client tools")
                    logger.info(f"   Client tools will be called in next iteration after LLM processes server results")
                    # Continue loop - LLM will see server results and can update client tool calls
                    continue
                
                # If there are client tool calls and NO server tools, return them immediately
                if client_calls:
                    logger.info(f"📤 Returning {len(client_calls)} client tool calls (no server tools to wait for)")
                    return {
                        "client_tools_needed": True,
                        "tool_calls": client_calls,
                        "message": assistant_message,
                        "iterations": iteration + 1
                    }
                
                # Only server tools - already added to messages, continue loop
                if server_results:
                    logger.info(f"✅ Only server tools executed, continuing loop")
            
            else:
                logger.warning(f"⚠️ Unknown finish_reason: {finish_reason}")
                return {
                    "complete": True,
                    "message": assistant_message,
                    "iterations": iteration + 1
                }
        
        # Max iterations reached
        logger.warning(f"⚠️ Max iterations ({max_iterations}) reached")
        return {
            "complete": True,
            "message": "I apologize, but I've reached my processing limit. Could you try rephrasing your request?",
            "error": "max_iterations",
            "iterations": max_iterations
        }
    
    def _build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build concise system prompt for tool usage with primary examples.
        """
        from app.core.tool_call_parser import tool_call_parser
        
        room = node_context.get("room", "unknown")
        user = node_context.get("user", "default")
        voice_mode = node_context.get("voice_mode", "brief")
        available_commands = available_commands or []

        must_call_tools = []
        direct_answer_allowed = []
        for cmd in available_commands:
            cmd_name = cmd.get("command_name")
            if not cmd_name:
                continue
            allow_direct = cmd.get("allow_direct_answer")
            if allow_direct is False:
                must_call_tools.append(cmd_name)
            elif allow_direct is True:
                direct_answer_allowed.append(cmd_name)

        direct_answer_section = ""
        if must_call_tools or direct_answer_allowed:
            direct_answer_section = "\nDirect Answer Policy:\n"
            if must_call_tools:
                direct_answer_section += f"- MUST call tools for: {', '.join(sorted(set(must_call_tools)))}\n"
            if direct_answer_allowed:
                direct_answer_section += f"- Direct answers allowed for: {', '.join(sorted(set(direct_answer_allowed)))}\n"

        tools_section = tool_call_parser.format_tools_for_prompt(tools, available_commands)

        system_prompt = f"""You are Jarvis, a voice assistant that uses tools.
Context: room={room}, user={user}, style={voice_mode}

Rules:
- Use tools to fulfill requests. Call ONE tool at a time.
- Pick the tool that best matches intent/keywords/examples; otherwise use get_command_utterance_examples.
- Extract parameters from the user's words; request validation only if required params are missing/ambiguous.
{direct_answer_section}

Response format: JSON ONLY (no extra text).
- Tool call: {{"message":"brief ack","tool_calls":[{{"name":"<tool>","arguments":{{...}}}}],"error":null}}
- Final: {{"message":"<concise spoken reply>","tool_calls":[],"error":null}}

Tools:
{tools_section}
"""

        logger.info(f"📝 Built system prompt: {len(system_prompt)} characters, {len(tools)} tools available")
        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("📝 System prompt (full):\n%s", system_prompt)

        return system_prompt

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


    def _check_for_validation(self, tool_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if any tool results contain validation requests."""
        import json
        
        for result in tool_results:
            try:
                content = json.loads(result["content"])
                if content.get("_validation_request"):
                    return {
                        "question": content["question"],
                        "parameter_name": content["parameter_name"],
                        "options": content.get("options", []),
                        "tool_call_id": result.get("tool_call_id"),
                    }
            except (json.JSONDecodeError, KeyError):
                pass
        
        return None
    
    def _convert_commands_to_tools(self, commands: List[CommandDefinition]) -> List[Dict[str, Any]]:
        """
        Convert CommandDefinition format to OpenAI tool format.
        
        This allows backward compatibility with existing command definitions.
        """
        def _normalize_param_type(param_type: Optional[str]) -> Dict[str, Any]:
            if not param_type:
                return {"type": "string"}
            raw = param_type.strip().lower()
            if raw.startswith("array<") and raw.endswith(">"):
                inner = raw[len("array<"):-1].strip()
                return {"type": "array", "items": _normalize_param_type(inner)}
            if raw.startswith("array[") and raw.endswith("]"):
                inner = raw[len("array["):-1].strip()
                return {"type": "array", "items": _normalize_param_type(inner)}
            if raw.endswith("[]"):
                inner = raw[:-2].strip()
                return {"type": "array", "items": _normalize_param_type(inner)}
            if raw in {"int", "integer"}:
                return {"type": "integer"}
            if raw in {"float", "number", "double"}:
                return {"type": "number"}
            if raw in {"bool", "boolean"}:
                return {"type": "boolean"}
            if raw == "datetime":
                return {"type": "string", "format": "date-time"}
            if raw == "date":
                return {"type": "string", "format": "date"}
            return {"type": "string"}

        tools = []
        
        force_required_commands = {"get_calendar_events", "get_sports_scores"}
        force_required_params = {"resolved_datetimes"}
        
        for cmd in commands:
            # Build parameters schema
            properties = {}
            required = []
            
            for param in cmd.parameters:
                schema = _normalize_param_type(param.type)
                properties[param.name] = {
                    **schema,
                    "description": param.description or f"Parameter {param.name}"
                }
                
                if param.enum_values:
                    if properties[param.name].get("type") == "array":
                        properties[param.name].setdefault("items", {})["enum"] = param.enum_values
                    else:
                        properties[param.name]["enum"] = param.enum_values
                
                is_forced_required = (
                    cmd.command_name in force_required_commands
                    and param.name in force_required_params
                )
                if param.required or is_forced_required:
                    required.append(param.name)
            
            # Create tool definition
            tool = {
                "type": "function",
                "function": {
                    "name": cmd.command_name,
                    "description": cmd.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                },
                "allow_direct_answer": cmd.allow_direct_answer,
                "keywords": cmd.keywords
            }
            if cmd.antipatterns:
                tool["antipatterns"] = [ap.model_dump() for ap in cmd.antipatterns]
            
            tools.append(tool)
        
        return tools
    
    def _convert_to_response_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert tool-based result to Jarvis response format.
        
        Response format:
        {
            "s": bool,
            "n": str or None,
            "p": dict,
            "e": dict or None
        }
        """
        if result.get("complete"):
            # Successful completion
            return {
                "s": True,
                "n": "voice_response",  # Generic command name
                "p": {"message": result["message"]},
                "e": None
            }
        
        elif result.get("client_tools_needed"):
            # Need client to execute tools - return as "pending"
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {
                    "type": "client_tools_needed",
                    "message": result.get("message", "Tools need execution"),
                    "tool_calls": result.get("tool_calls", [])
                }
            }
        
        elif result.get("validation_required"):
            # Need user clarification
            validation = result.get("validation", {})
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {
                    "type": "validation_required",
                    "message": validation.get("question", "Need clarification"),
                    "validation": validation
                }
            }
        
        else:
            # Error case
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {
                    "type": "processing_error",
                    "message": result.get("error", "Unknown error occurred")
                }
            }
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate error response in Jarvis response format."""
        return {
            "s": False,
            "n": None,
            "p": {},
            "e": {
                "type": "error",
                "message": message
            }
        }

