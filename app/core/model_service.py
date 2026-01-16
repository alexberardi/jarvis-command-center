"""
Model Service for Jarvis Voice Assistant.

This service provides a clean interface to the model system and can
gradually replace the existing prompt provider architecture.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple

from app.core.model_factory import ModelFactory
from app.core.interfaces.imodel_interface import IModelInterface
from app.core.llm_proxy_client import LLMProxyClient
from app.core.tool_registry import tool_registry
from app.core.tool_executor import tool_executor
from app.core.tool_call_parser import tool_call_parser
from app.core.conversation_cache import conversation_cache
from app.request_models.voice_command_request import CommandDefinition

logger = logging.getLogger("uvicorn")

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
        available_commands: Optional[List[CommandDefinition]] = None
    ) -> None:
        """
        Warm up a conversation with tool support.
        
        Args:
            node_context: Node information (room, user, device, etc.)
            conversation_id: Unique conversation identifier
            timezone: User's timezone for date calculations
            client_tools: Client-provided tool definitions
        """
        logger.info(f"🚀 Warming up tool-based conversation {conversation_id[:8]}")
        
        # Get server tools
        server_tools = tool_registry.get_tool_definitions()
        
        # Merge with client tools
        all_tools = server_tools + (client_tools or [])
        
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
                
                if tool_name:
                    # Check if this tool has an "examples" array at the top level
                    examples_array = tool.get("examples", [])
                    
                    if examples_array and isinstance(examples_array, list):
                        # Initialize command entry if not exists
                        if tool_name not in command_examples_map:
                            command_examples_map[tool_name] = {
                                "command_name": tool_name,
                                "examples": []
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
                                    logger.debug(f"   └─ ⚠️  Example missing voice_command")
                        
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
                cmd_name = cmd_dict.get("command_name")
                if cmd_name in command_examples_map:
                    # Merge examples
                    command_examples_map[cmd_name]["examples"].extend(cmd_dict.get("examples", []))
                else:
                    available_commands_to_cache.append(cmd_dict)
        
        # Build system message using the model's prompt builder when available
        # Pass empty list for available_commands_dicts since we're using client_tools examples now
        if hasattr(self.model, "_build_system_prompt"):
            system_message = self.model._build_system_prompt(node_context, timezone, all_tools, [])  # type: ignore[attr-defined]
        else:
            # Fallback to default builder
            system_message = self._build_tool_system_message(node_context, timezone, all_tools, [])

        # Format tools without examples (examples available via get_command_examples tool)
        tools_text = tool_call_parser.format_tools_for_prompt(all_tools)
        tools_message = f"Available tools (use only what you need):\n{tools_text}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": tools_message}
        ]
        
        # Store in cache (including command examples extracted from client_tools)
        conversation_cache.set(
            conversation_id=conversation_id,
            messages=messages,
            available_commands=available_commands_to_cache,
            timezone=timezone,
            tools=all_tools
        )
        if available_commands_to_cache:
            total_examples = sum(len(cmd.get("examples", [])) for cmd in available_commands_to_cache)
            logger.info(f"📚 Cached {len(available_commands_to_cache)} command(s) with {total_examples} total examples for get_command_examples tool")
        else:
            logger.warning(f"⚠️  Cached empty available_commands list - get_command_examples tool will return no_commands_available")
        
        # Warm up LLM proxy (tools are in system message, not passed separately)
        try:
            logger.info(f"🔥 Calling LLM proxy warmup_conversation for {conversation_id[:8]}")
            result = await self.llm_client.warmup_conversation(
                conversation_id=conversation_id,
                messages=messages
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
        logger.info(f"🎯 Processing tool-based command: '{voice_command}'")
        
        # Get conversation state
        messages = conversation_cache.get_messages(conversation_id)
        tools = conversation_cache.get_tools(conversation_id)
        
        if not messages:
            raise ValueError(f"Conversation {conversation_id} not found or expired")
        
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
        
        # Process with tool execution loop
        result = await self._tool_execution_loop(
            conversation_id=conversation_id,
            messages=messages,
            tools=tools
        )
        
        # Update cache with final messages
        conversation_cache.update_messages(conversation_id, messages)
        
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
        for iteration in range(max_iterations):
            logger.info(f"🔁 Tool loop iteration {iteration + 1}/{max_iterations}")
            
            # Call LLM (tools are in system message, not passed separately)
            try:
                response = await self.llm_client.chat_completion(
                    messages=messages,
                    conversation_id=conversation_id
                )
            except Exception as e:
                logger.error(f"❌ LLM call failed: {e}")
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
            
            if not raw_content:
                logger.warning(f"⚠️ Empty raw_content!")
            
            # Use tool call parser to extract tool calls from JSON
            finish_reason, tool_calls, assistant_message = tool_call_parser.parse_response(raw_content)
            
            logger.info(f"📥 LLM response parsed: finish_reason={finish_reason}, tool_calls={len(tool_calls)}")
            
            # Add assistant message to history (store the raw JSON output)
            assistant_msg = {"role": "assistant", "content": raw_content}
            messages.append(assistant_msg)
            
            # Handle different finish reasons
            if finish_reason == "stop":
                # Conversation complete
                return {
                    "stop_reason": "complete",
                    "assistant_message": assistant_message
                }
            
            elif finish_reason == "tool_calls":
                # Execute tools
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
                    logger.info(f"📤 Returning {len(client_calls)} client tool calls (no server tools to wait for)")
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
                return {
                    "stop_reason": "complete",
                    "assistant_message": assistant_message
                }
        
        # Max iterations reached
        logger.warning(f"⚠️ Max tool loop iterations ({max_iterations}) reached")
        return {
            "stop_reason": "complete",
            "assistant_message": "Maximum tool execution iterations reached.",
            "error": "max_iterations_exceeded"
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
        from app.core.general_context import get_general_context
        
        # Get date context
        date_context = get_general_context(timezone)
        
        # Format tools for prompt with primary examples
        tools_text = tool_call_parser.format_tools_with_examples(tools, available_commands)
        
        system_msg = f"""You are Jarvis, a voice-controlled assistant that operates BY CALLING TOOLS.

Node Context:
- Room: {node_context.get('room', 'unknown')}
- User: {node_context.get('user', 'default')}
- Voice Mode: {node_context.get('voice_mode', 'brief')}

{date_context}

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

When calling a tool (ONE at a time):
{{
  "message": "Brief acknowledgment",
  "tool_call": {{"name": "tool_name", "arguments": {{"param_name": "param_value"}}}}
}}

When you have the final answer:
{{
  "message": "Your natural spoken response",
  "tool_call": null
}}

Available Tools:
{tools_text}

Key Guidelines:
- Extract parameters directly from user's message when they are clearly stated. If the user says "Seattle", extract "Seattle" as the city parameter - do not ask for validation when the information is already provided.
- Only ask for validation if a required parameter is truly missing or genuinely ambiguous after trying all other options.
- Be conversational and {node_context.get('voice_mode', 'brief')}"""
        
        return system_msg
