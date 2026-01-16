"""
Parallel Double-Shot Llama 3B Model Implementation

This model uses a two-stage approach:
1. Command inference (single call)
2. Parallel parameter extraction - general and date parameters simultaneously

This strikes a balance between reliability and efficiency.
"""

import json
import asyncio
import uuid
import logging
import time
from typing import Dict, Any, List, Optional

from app.core.models.base_model import BaseModel
from app.core.llm_manager import LLMManager
from app.core.conversation_cache import conversation_cache
from app.core.general_context import get_general_context
from app.request_models.voice_command_request import CommandDefinition

logger = logging.getLogger("uvicorn")


class JarvisParallelLlama3B(BaseModel):
    """
    Parallel Double-Shot implementation using Llama 3.2 3B Instruct.
    
    Step 1: Command inference
    Step 2: Parallel general + date parameter extraction
    """


    @property
    def name(self) -> str:
        return "JarvisParallelLlama3B"

    async def perform_warmup(
        self,
        node_context: Dict[str, Any],
        available_commands: List[CommandDefinition],
        conversation_id: str,
        timezone: Optional[str] = None
    ) -> None:
        """
        Parallel model warmup - identical to JarvisLlama3B.
        
        The only difference from JarvisLlama3B should be in perform_inference.
        """
        start_time = time.time()
        
        logger.info(f"🚀 JarvisParallelLlama3B warmup for {conversation_id[:8]}...")
        
        try:
            # Build simplified but complete context
            system_prompt = self.build_warmup_prompt(
                node_context=node_context,
                available_commands=available_commands,
                conversation_id=conversation_id,
                timezone=timezone
            )
            
            # Simple warmup messages
            warmup_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "This is a warm-up message to establish context."},
                {"role": "assistant", "content": "I understand. I'm ready to help with command inference and parameter extraction."}
            ]
            
            # Cache conversation
            conversation_cache.set(
                conversation_id=conversation_id,
                messages=warmup_messages,
                available_commands=[cmd.model_dump() for cmd in available_commands],
                timezone=timezone
            )
            
            # Warmup the LLM
            await self.llm_manager.warmup_conversation(
                conversation_id=conversation_id,
                warmup_messages=warmup_messages
            )
            
            duration = time.time() - start_time
            logger.info(f"✅ JarvisParallelLlama3B warmup completed in {duration:.2f}s")

        except Exception as e:
            logger.error(f"❌ JarvisParallelLlama3B warmup failed: {e}")
            raise

    async def perform_inference(
        self,
        voice_command: str,
        conversation_id: str,
        node_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parallel double-shot inference - the key difference from JarvisLlama3B.
        
        This performs:
        1. Command inference (single call)
        2. Parallel general + date parameter extraction
        
        Returns standard Jarvis format: {"s": bool, "n": str, "p": dict, "e": dict}
        """
        start_time = time.time()
        
        logger.info(f"🎯 JarvisParallelLlama3B parallel inference: '{voice_command}' (conversation: {conversation_id[:8]})")
        
        try:
            # Get cached conversation (conversation_id is guaranteed to be non-None by FastAPI validation)
            cached_messages = conversation_cache.get_messages(conversation_id)
            cached_commands = conversation_cache.get_available_commands(conversation_id)
            
            if not cached_messages or not cached_commands:
                logger.error("❌ No cached conversation found")
                return {
                    "s": False,
                    "n": None,
                    "p": {},
                    "e": {"type": "cache_error", "message": "No conversation context found"}
                }

            # Step 1: Command Inference
            command_result = await self._infer_command(conversation_id, voice_command, cached_commands)
            
            if not command_result.get("s", False):
                # Command inference failed
                duration = time.time() - start_time
                logger.info(f"❌ JarvisParallelLlama3B command inference failed in {duration:.2f}s")
                return command_result

            # Step 2: Parallel Parameter Extraction using LLMManager
            command_name = command_result.get("n")
            if not command_name:
                # No command found, return the command result as-is
                return {
                    "s": command_result.get("s", False),
                    "n": command_result.get("n"),
                    "p": {},
                    "e": command_result.get("e", {})
                }
            
            # Find the command definition to determine what parameters to extract
            command_def = None
            for cmd in cached_commands:
                if cmd["command_name"] == command_name:
                    command_def = cmd
                    break
            
            if not command_def or not command_def.get("parameters"):
                # No parameters to extract
                return {
                    "s": command_result.get("s", False),
                    "n": command_result.get("n"),
                    "p": {},
                    "e": command_result.get("e", {})
                }
            
            # Build parallel query tasks
            query_tasks = []
            
            # Check what types of parameters this command has
            non_date_params = self.filter_parameters_by_type(command_def, include_date=False)
            date_params = self.filter_parameters_by_type(command_def, include_date=True)
            
            # Add general parameter extraction task if needed
            if non_date_params:
                params_text = self.build_parameter_description_text(non_date_params)
                examples_text = self._build_examples_text(command_def, "general")
                general_prompt = f"""TASK: Extract general parameters from: "{voice_command}"

You are Jarvis extracting parameters for the "{command_name}" command.

Parameters to extract (NON-DATE parameters only):
{params_text}

GENERIC RULES:
1. Return ONLY a JSON object with "parameters" key
2. Use the EXACT parameter names from the parameter list above
3. Match parameter types: if description mentions "Array" or "List", return an array; otherwise return a single value
4. Extract parameter values from the user's voice command
5. Only include parameters you can confidently extract from the input
6. Use null for missing optional parameters
7. Do NOT extract date/time parameters (handled separately){examples_text}"""
                
                # Debug: Log the prompt length
                logger.info(f"🐛 DEBUG: General prompt length: {len(general_prompt)} chars")
                logger.info(f"🐛 DEBUG: Examples text length: {len(examples_text)} chars")
                
                query_tasks.append({
                    "prompt": general_prompt,
                    "function_after": lambda content, result: self._parse_parameter_response(content, result)
                })
            
            # Add date parameter extraction task if needed
            if date_params:
                params_text = self.build_parameter_description_text(date_params)
                examples_text = self._build_examples_text(command_def, "date")
                date_prompt = f"""TASK: Extract date/time parameters from: "{voice_command}"

You are Jarvis extracting date/time parameters for the "{command_name}" command.

Date/Time parameters to extract:
{params_text}

GENERIC RULES:
1. Return ONLY a JSON object with "parameters" key
2. Use the EXACT parameter names from the parameter list above
3. Match parameter types: if description mentions "Array" or "List", return an array; otherwise return a single value
4. Use the absolute-to-relative date mapping from the conversation context above
5. Convert all dates to UTC ISO format (YYYY-MM-DDTHH:MM:SSZ)
6. For relative dates (today, tomorrow, next week, etc.), look up the exact mapping in the conversation context
7. Only include parameters you can confidently extract from the input

The conversation context contains current date/time and absolute-to-relative mappings for your reference.{examples_text}"""
                
                # Debug: Log the prompt length
                logger.info(f"🐛 DEBUG: Date prompt length: {len(date_prompt)} chars")
                logger.info(f"🐛 DEBUG: Date examples text length: {len(examples_text)} chars")
                
                query_tasks.append({
                    "prompt": date_prompt,
                    "function_after": lambda content, result: self._parse_parameter_response(content, result)
                })
            
            # Execute parallel queries if we have any
            merged_parameters = {}
            if query_tasks:
                logger.info(f"🔄 Starting {len(query_tasks)} parallel parameter extraction queries")
                
                # Debug: Log the exact time before calling parallel_queries_with_cache
                debug_start = time.time()
                logger.info(f"🐛 DEBUG: About to call parallel_queries_with_cache at {debug_start}")
                
                results = await self.llm_manager.parallel_queries_with_cache(
                    conversation_id, query_tasks
                )
                
                debug_end = time.time()
                logger.info(f"🐛 DEBUG: parallel_queries_with_cache completed at {debug_end}, took {debug_end - debug_start:.2f}s")
                
                # Merge all parameter results
                for result in results:
                    if result and not result.get("error") and result.get("parameters"):
                        merged_parameters.update(result["parameters"])
            
            # Merge results in standard Jarvis format
            final_result = {
                "s": command_result.get("s", False),
                "n": command_result.get("n"),
                "p": merged_parameters,
                "e": command_result.get("e", {})
            }

            duration = time.time() - start_time
            logger.info(f"✅ JarvisParallelLlama3B parallel inference completed in {duration:.2f}s")
            logger.info(f"   Parallel result: {final_result}")

            return final_result

        except Exception as e:
            logger.error(f"❌ JarvisParallelLlama3B inference failed: {e}")
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {"type": "inference_error", "message": str(e)}
            }


    def _build_examples_text(self, command_def: Dict[str, Any], param_type: str = "all") -> str:
        """Build examples text from command definition for parameter extraction."""
        if not command_def.get("examples"):
            return ""
        
        examples_list = []
        
        for example in command_def["examples"]:
            voice_cmd = example.get("voice_command", "")
            expected_params = example.get("expected_parameters", {})
            
            if param_type == "date":
                # Only show examples with date/time parameters
                date_params = {k: v for k, v in expected_params.items() 
                              if any(date_word in k.lower() for date_word in ["date", "time", "when", "schedule"])}
                if date_params:
                    params_json = json.dumps(date_params)
                    examples_list.append(f'Input: "{voice_cmd}"\nOutput: {{"parameters": {params_json}}}')
            elif param_type == "general":
                # Only show examples with non-date parameters
                non_date_params = {k: v for k, v in expected_params.items() 
                                  if not any(date_word in k.lower() for date_word in ["date", "time", "when", "schedule"])}
                if non_date_params:
                    params_json = json.dumps(non_date_params)
                    examples_list.append(f'Input: "{voice_cmd}"\nOutput: {{"parameters": {params_json}}}')
            else:
                # Show all examples
                params_json = json.dumps(expected_params)
                examples_list.append(f'Input: "{voice_cmd}"\nOutput: {{"parameters": {params_json}}}')
        
        if examples_list:
            return f"\n\nEXAMPLES:\n" + "\n\n".join(examples_list)
        return ""

    def _parse_parameter_response(self, content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse parameter extraction response from LLM."""
        try:
            # Try to extract JSON from the content
            json_result = self.extract_json_from_text(content)
            if json_result and "parameters" in json_result:
                return {"parameters": json_result["parameters"]}
            else:
                logger.warning(f"Could not extract parameters from response: {content[:100]}...")
                return {"parameters": {}}
        except Exception as e:
            logger.error(f"Error parsing parameter response: {e}")
            return {"parameters": {}}

    async def cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up any conversation-specific resources."""
        try:
            # Remove from cache if it exists
            if hasattr(conversation_cache, 'remove_conversation'):
                conversation_cache.remove_conversation(conversation_id)
        except Exception:
            pass

    async def _infer_command(
        self,
        conversation_id: str,
        user_input: str,
        available_commands: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Step 1: Infer the command from user input - returns Jarvis format."""
        
        try:
            # Use LLMManager's command inference method with cached conversation
            # LLMManager handles building the appropriate prompts, messages, and context
            parsed_response, content = await self.llm_manager.command_inference_with_cache(
                message=user_input,
                conversation_id=conversation_id,
                available_commands=available_commands
            )

            # Return the parsed response from LLMManager
            if parsed_response and isinstance(parsed_response, dict) and "s" in parsed_response:
                return parsed_response
            else:
                return {
                    "s": False,
                    "n": None,
                    "p": {},
                    "e": {"type": "parse_error", "message": "Could not parse command inference response"}
                }

        except Exception as e:
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {"type": "inference_error", "message": f"Command inference failed: {str(e)}"}
            }

    def build_warmup_prompt(
        self,
        node_context: Dict[str, Any],
        available_commands: List[CommandDefinition],
        conversation_id: str,
        timezone: Optional[str] = None
    ) -> str:
        """
        Build warmup system prompt for parallel model.
        Uses exact same format as GeneralPromptProvider.build_complete_static_prompt.
        """
        # Get basic date context using the timezone parameter (passed from client request)
        general_context_str = get_general_context(timezone)
        
        # Build node context string
        node_context_str = '\n'.join(f"{k}: {v}" for k, v in node_context.items()) if node_context else "No specific context"
        
        # Add conversation ID at the beginning if provided (matching old system)
        conversation_prefix = f"Conversation ID: {conversation_id}\n\n" if conversation_id else ""
        
        # Build available commands string using the same simple format as old system
        available_commands_str = "No commands available"
        if available_commands:
            available_commands_str = '\n'.join(
                f"- {cmd.command_name} - {cmd.description}"
                for cmd in available_commands
            )
        
        # Use exact same format as GeneralPromptProvider.build_complete_static_prompt
        system_prompt = f"""{conversation_prefix}You are an inference engine for a voice assistant. Ultimately, your job is to match a command name and parameters from a voice command to a set of available commands. This prompt establishes the base context and rules.

{general_context_str}

Node Context: {node_context_str}

You are a voice assistant command classifier. Identify which command the user wants.

COMMANDS & EXAMPLES:

{available_commands_str}

Always respond with JSON: {{"s":true,"n":"command_name","e":null}}"""
        
        return system_prompt
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return JarvisParallelLlama3B capabilities."""
        return {
            "supports_single_shot": False,  # Uses parallel double-shot approach
            "supports_streaming": False,
            "max_context_length": 8192,
            "supported_languages": ["en"],
            "custom_features": {
                "fine_tuned_for_jarvis": True,
                "parallel_double_shot_inference": True,
                "optimized_prompts": True,
                "fast_inference": True
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for JarvisParallelLlama3B."""
        return {
            "status": "healthy",
            "model_name": self.name,
            "details": {
                "model_type": "fine_tuned_llama_3.2_3b",
                "inference_mode": "parallel_double_shot",
                "training_data": "jarvis_command_inference_v1"
            }
        }
