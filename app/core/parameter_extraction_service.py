"""
Parameter extraction service for building context and extracting parameters from commands.

NOTE: This is part of the LEGACY flow (BaseModel).
The new tool-based architecture (JarvisToolModel) handles parameter extraction via tool calls.
"""

import json
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from app.request_models.voice_command_request import CommandDefinition
from app.context_providers.standard_parameter_inference_system_prompt_provider import StandardParameterInferenceSystemPromptProvider
from app.core.conversation_cache import ConversationCache
from app.core.llm_proxy_client import LLMProxyClient
from app.core.llm_manager import LLMManager
from app.core.json_utils import validate_and_clean_json_response
from app.core.general_context import generate_date_context_object

logger = logging.getLogger("uvicorn")


class ParameterExtractionService:
    """Service for building parameter extraction context and extracting parameters."""
    
    def __init__(self):
        self.parameter_provider = StandardParameterInferenceSystemPromptProvider()
        self.llm_client = LLMProxyClient()
        self.llm_manager = LLMManager()
    
    def build_parameter_context(
        self,
        selected_command: CommandDefinition,
        request_conversation_id: str,
        conversation_cache: ConversationCache,
        node_context_provider: Any,
        voice_command: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Builds the context needed for parameter extraction.
        
        Args:
            selected_command: The command to extract parameters for
            request_conversation_id: Conversation ID from request
            conversation_cache: Conversation cache instance
            node_context_provider: Node context provider
            voice_command: The user's voice command
            
        Returns:
            Tuple of (node_context, cached_timezone)
        """
        # Build node context from node properties
        node_context = {
            "room": node_context_provider.node.room,
            "node_id": node_context_provider.node.node_id,
            "user": node_context_provider.node.user,
            "voice_mode": node_context_provider.node.voice_mode
        }
        
        # Get timezone from cache for parameter inference
        cached_timezone = None
        if request_conversation_id:
            cached_timezone = conversation_cache.get_timezone(request_conversation_id)
            if cached_timezone:
                node_context["timezone"] = cached_timezone
        
        return node_context, cached_timezone
    
    async def extract_parameters_and_dates(
        self,
        selected_command: CommandDefinition,
        voice_command: str,
        cached_timezone: Optional[str],
        conversation_id: Optional[str],
        conversation_cache: ConversationCache
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extracts parameters for a command using parallel parameter and date extraction.
        
        Args:
            selected_command: The command to extract parameters for
            voice_command: The user's voice command
            cached_timezone: Cached timezone for date extraction
            conversation_id: LLM conversation ID
            conversation_cache: Conversation cache instance
            
        Returns:
            Tuple of (extracted_parameters, date_extraction_result)
        """
        try:
            # Step 1: Add command details to conversation cache so queries have context
            command_details_message = self._build_simple_command_details(selected_command, voice_command)
            if conversation_id and conversation_cache:
                conversation_cache.add_message(conversation_id, {"role": "user", "content": command_details_message})
                # Add a placeholder assistant response to maintain conversation flow
                conversation_cache.add_message(conversation_id, {"role": "assistant", "content": "Command details loaded. Ready for parameter and date extraction."})
            
            # Step 2: Build query tasks (always use parallel_queries_with_cache)
            query_tasks = []
            
            # Check what types of parameters this command has
            has_non_date_params = self._has_non_date_parameters(selected_command)
            has_date_params = self._has_date_parameters(selected_command)
            
            # Add parameter extraction task only if command has non-date parameters
            if has_non_date_params:
                parameter_prompt = self._build_focused_parameter_prompt(voice_command)
                query_tasks.append({
                    "prompt": parameter_prompt,
                    "function_after": lambda content, result: self._parse_parameter_response_wrapper(content, result)
                })
            
            # Add date extraction task if command has date parameters
            if has_date_params:
                date_prompt = self._build_date_extraction_request(selected_command, voice_command, cached_timezone)
                query_tasks.append({
                    "prompt": date_prompt,
                    "function_after": lambda content, result: self._parse_parameter_response_wrapper(content, result)
                })
            
            # Log what we're doing
            if has_non_date_params and has_date_params:
                logger.info(f"🔄 Parallel parameter and date extraction for {selected_command.command_name}")
            elif has_non_date_params:
                logger.info(f"🔄 Parameter extraction only for {selected_command.command_name}")
            elif has_date_params:
                logger.info(f"🔄 Date extraction only for {selected_command.command_name}")
            else:
                logger.info(f"🔄 No parameter extraction needed for {selected_command.command_name}")
            
            # Step 3: Execute queries with proper cache management
            results = await self.llm_manager.parallel_queries_with_cache(
                conversation_id, query_tasks
            )
            
            # Step 4: Extract and merge results based on what we requested
            extracted_parameters = {}
            date_extraction_result = {}
            result_index = 0
            
            # Handle parameter extraction results (if we requested them)
            if has_non_date_params:
                parameter_result = results[result_index] if result_index < len(results) else {}
                result_index += 1
                
                if "error" in parameter_result:
                    logger.error(f"❌ Parameter extraction failed: {parameter_result['error']}")
                    logger.error(f"❌ Parameter extraction raw content: {parameter_result.get('content', 'No content')}")
                else:
                    # Extract just the parameters from the parsed response
                    parsed_response = parameter_result.get("parsed_response", {})
                    extracted_parameters = parsed_response.get("p", {}) if isinstance(parsed_response, dict) else {}
                    logger.info(f"✅ Parameter extraction successful: {extracted_parameters}")
            
            # Handle date extraction results (if we requested them)
            if has_date_params:
                date_result = results[result_index] if result_index < len(results) else {}
                
                if "error" in date_result:
                    logger.error(f"❌ Date extraction failed: {date_result['error']}")
                else:
                    date_extraction_result = date_result.get("parsed_response", {})
                    # Merge date results into parameters
                    if date_extraction_result and "p" in date_extraction_result:
                        extracted_parameters.update(date_extraction_result["p"])
                        logger.info(f"✅ Merged {len(date_extraction_result['p'])} date parameters into result")
            
            return extracted_parameters, date_extraction_result
                
        except Exception as e:
            logger.warning(f"❌ Parameter extraction failed for {selected_command.command_name}: {e}, using empty parameters")
            return {}, {}
    
    def _has_date_parameters(self, selected_command: CommandDefinition) -> bool:
        """Check if the command has any date-related parameters."""
        if not selected_command.parameters:
            return False
        
        for param in selected_command.parameters:
            if self._is_date_parameter_type(param.type):
                return True
        return False
    
    def _has_non_date_parameters(self, selected_command: CommandDefinition) -> bool:
        """Check if the command has any non-date parameters."""
        if not selected_command.parameters:
            return False
        
        for param in selected_command.parameters:
            if not self._is_date_parameter_type(param.type):
                return True
        return False
    
    async def _extract_parameters(
        self,
        selected_command: CommandDefinition,
        voice_command: str,
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        """Extract non-date parameters from the voice command."""
        start_time = time.time()
        
        try:
            # Build command details message
            command_details = self._build_simple_command_details(selected_command, voice_command)
            logger.info(f"📋 Parameter inference prompt includes voice command: '{voice_command}'")
            
            # Use command_inference method - it will handle cache management internally
            if conversation_id:
                # Use cache-managed conversation
                parsed_response, content = await self.llm_manager.command_inference_with_cache(
                    message=command_details,
                    conversation_id=conversation_id
                )
            else:
                # No conversation ID, create new conversation
                node_context = {"timezone": "UTC"}
                base_prompt = self.parameter_provider.build_base_parameter_inference_prompt(node_context, conversation_id=conversation_id)
                simple_prompt = f"{base_prompt}\n\n{command_details}"
                messages = [{"role": "system", "content": simple_prompt}]
                parsed_response, content = await self.llm_manager.command_inference(
                    messages=messages,
                    conversation_id=conversation_id
                )
            
            duration = time.time() - start_time
            logger.info(f"⚡ Parameter inference took {duration:.2f} seconds")
            
            # Parse JSON response (LLMManager already extracted content)
            parsed_response = self._parse_parameter_response(content)
            
            # Extract parameters from response
            if "p" in parsed_response:
                return parsed_response["p"]
            else:
                logger.warning("No parameters found in response")
                return {}
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Parameter inference failed after {duration:.2f}s: {e}")
            return {}
    
    async def _extract_dates(
        self,
        selected_command: CommandDefinition,
        voice_command: str,
        cached_timezone: Optional[str],
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        """Extract date parameters from the voice command."""
        start_time = time.time()
        
        try:
            # Build date extraction request with actual parameter names
            simple_date_request = self._build_date_extraction_request(selected_command, voice_command, cached_timezone)
            
            logger.info(f"📋 Date extraction prompt includes voice command: '{voice_command}'")
            
            # Use command_inference method - it will handle cache management internally
            if conversation_id:
                # Use cache-managed conversation
                parsed_response, content = await self.llm_manager.command_inference_with_cache(
                    message=simple_date_request,
                    conversation_id=conversation_id
                )
            else:
                # No conversation ID, create new conversation with full prompt
                date_context = generate_date_context_object()
                date_prompt = self._build_date_extraction_prompt(selected_command, voice_command, date_context, cached_timezone)
                messages = [{"role": "system", "content": date_prompt}]
                parsed_response, content = await self.llm_manager.command_inference(
                    messages=messages,
                    conversation_id=conversation_id
                )
            
            duration = time.time() - start_time
            logger.info(f"⚡ Date extraction took {duration:.2f} seconds")
            
            # Parse JSON response (LLMManager already extracted content)
            parsed_response = self._parse_parameter_response(content)
            return parsed_response
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Date extraction failed after {duration:.2f}s: {e}")
            return {}
    
    def _build_date_extraction_prompt(
        self,
        selected_command: CommandDefinition,
        voice_command: str,
        date_context: Dict[str, Any],
        cached_timezone: Optional[str]
    ) -> str:
        """Build the date extraction prompt."""
        timezone = cached_timezone or "UTC"
        
        # Build parameter descriptions for date parameters only
        date_param_descriptions = []
        if selected_command.parameters:
            for param_info in selected_command.parameters:
                param_type = param_info.type
                if self._is_date_parameter_type(param_type):
                    param_name = param_info.name
                    param_desc = param_info.description or 'No description available'
                    param_required = param_info.required
                    
                    param_line = f"- {param_name} ({param_type})"
                    if param_required:
                        param_line += " [REQUIRED]"
                    param_line += f": {param_desc}"
                    
                    date_param_descriptions.append(param_line)
        
        # Build a simple date extraction prompt since date context is already in warmup
        date_prompt = f"""Extract date/time references from: "{voice_command}"

Current time: {date_context.get('current', {}).get('date', 'Unknown')} at {date_context.get('current', {}).get('time', 'Unknown')} ({timezone})

Use the RELATIVE DATE MAPPING from the cached context to convert relative dates.
Examples:
- "next week" → use cached mapping value {date_context.get('weeks', {}).get('next_week', [])}
- "tomorrow at 3pm" → "2025-09-05T19:00:00Z" (3pm tomorrow in UTC)
- "this weekend" → use cached mapping value {date_context.get('weekend', {}).get('this_weekend', [])}

Response: {{"s":true,"p":{{}},"e":null}}"""
        
        return date_prompt
    
    def _build_date_extraction_request(self, selected_command: CommandDefinition, voice_command: str, cached_timezone: Optional[str] = None) -> str:
        """Build a focused date extraction request with actual parameter names and real date examples."""
        # Get real date context for examples
        date_context = generate_date_context_object(cached_timezone)
        
        # Get date parameters with their types
        date_params = []
        if selected_command.parameters:
            for param_info in selected_command.parameters:
                param_type = param_info.type
                if self._is_date_parameter_type(param_type):
                    param_name = param_info.name
                    param_desc = param_info.description or 'No description available'
                    param_required = param_info.required
                    
                    param_line = f"- {param_name} ({param_type})"
                    if param_required:
                        param_line += " [REQUIRED]"
                    param_line += f": {param_desc}"
                    
                    date_params.append(param_line)
        
        # Build rules sections
        rules_str = ""
        if hasattr(selected_command, 'rules') and selected_command.rules:
            rules_str += f"\n\nCOMMAND RULES:\n" + "\n".join(f"- {rule}" for rule in selected_command.rules)
        
        critical_rules_str = ""
        if hasattr(selected_command, 'critical_rules') and selected_command.critical_rules:
            critical_rules_str += f"\n\nCOMMAND CRITICAL RULES:\n" + "\n".join(f"- {rule}" for rule in selected_command.critical_rules)
        
        return f"""🚨 CRITICAL: Return ONLY valid JSON. NO explanations, descriptions, or additional text.

Extract date parameters from the voice command and return ONLY valid JSON.

REQUIRED OUTPUT FORMAT:
{{"s":true,"p":{{...}},"e":null}}

DATE PARAMETERS TO EXTRACT:
{chr(10).join(date_params) if date_params else "No date parameters"}{rules_str}{critical_rules_str}

CRITICAL DATE EXTRACTION INSTRUCTIONS:
- Look at the voice command and identify any relative date words
- Go to the RELATIVE TO ABSOLUTE DATE MAPPING section in the system prompt above
- Find the EXACT key that matches the relative date phrase
- Copy the COMPLETE value (array or single) from that mapping entry
- Put that value in the "datetimes" parameter

STEP BY STEP:
1. Voice command: "What meetings do I have next week?"
2. Relative date phrase: "next week"  
3. Look up "next week" in the mapping above
4. Find: "next week": ["2025-09-07T04:00:00Z","2025-09-08T04:00:00Z",...] 
5. Use the COMPLETE array: ["2025-09-07T04:00:00Z","2025-09-08T04:00:00Z","2025-09-09T04:00:00Z","2025-09-10T04:00:00Z","2025-09-11T04:00:00Z","2025-09-12T04:00:00Z","2025-09-13T04:00:00Z"]

IMPORTANT: Use the COMPLETE value from the mapping. Do not create your own dates.

Voice Command: {voice_command}

Remember: ONLY JSON, no explanations. Look up the EXACT mapping value."""
    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """Extract content from LLM response, handling different formats."""
        # Try OpenAI-style format first
        choices = response.get('choices', [{}])
        if choices:
            first_choice = choices[0] if choices else {}
            message = first_choice.get('message', {})
            content = message.get('content', '')
            if content:
                return content
        
        # Try alternative formats
        content = response.get('content', '') or response.get('response', '')
        if content:
            return content
        
        return ""
    
    def _parse_parameter_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate parameter extraction response."""
        # Log the raw response for debugging
        logger.info(f"📋 Raw parameter extraction response: {content}")
        
        # Simple JSON validation - no cleaning or fixing
        try:
            response_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON response: {e}")
            logger.error(f"❌ Raw content: {content}")
            raise ValueError(f"Invalid JSON response: {e}")
        
        # Validate structure
        if not isinstance(response_data, dict):
            raise ValueError("Parameter extraction response is not a dictionary")
        
        return response_data
    
    def _build_simple_command_details(self, selected_command: CommandDefinition, voice_command: str) -> str:
        """Build a simple command details section without verbose examples."""
        # Get command details
        command_name = selected_command.command_name
        command_description = selected_command.description
        parameters = selected_command.parameters
        
        # Build parameter descriptions (excluding date parameters)
        parameter_descriptions = []
        if parameters:
            for param_info in parameters:
                param_type = param_info.type
                # Skip date parameters - they'll be handled by date extraction
                if self._is_date_parameter_type(param_type):
                    continue
                    
                param_name = param_info.name
                param_desc = param_info.description or 'No description available'
                param_required = param_info.required
                enum_values = param_info.enum_values or []
                
                # Build parameter line
                param_line = f"- {param_name} ({param_type})"
                if param_required:
                    param_line += " [REQUIRED]"
                param_line += f": {param_desc}"
                
                # Add enum values if available
                if enum_values:
                    param_line += f" [Allowed values: {', '.join(enum_values)}]"
                
                parameter_descriptions.append(param_line)
        
        # Build rules sections
        rules_str = ""
        if hasattr(selected_command, 'rules') and selected_command.rules:
            rules_str += f"\n\nRULES:\n" + "\n".join(f"- {rule}" for rule in selected_command.rules)
        
        critical_rules_str = ""
        if hasattr(selected_command, 'critical_rules') and selected_command.critical_rules:
            critical_rules_str += f"\n\nCRITICAL RULES:\n" + "\n".join(f"- {rule}" for rule in selected_command.critical_rules)
        
        # Build examples using structured format
        examples_str = ""
        if hasattr(selected_command, 'examples') and selected_command.examples:
            import json
            examples_list = []
            for example in selected_command.examples:
                # Format the expected parameters as JSON output format
                params_json = json.dumps(example.expected_parameters) if example.expected_parameters else "{}"
                examples_list.append(f'Input: "{example.voice_command}"\nOutput: {{"s": true, "p": {params_json}, "e": null}}')
            examples_str = f"\n\nEXAMPLES:\n" + "\n\n".join(examples_list)
        
        # Build the simple command details message
        command_details = f"""COMMAND DETAILS:

Command: {command_name}
Description: {command_description}

PARAMETERS:
{chr(10).join(parameter_descriptions) if parameter_descriptions else "No non-date parameters required"}{rules_str}{critical_rules_str}{examples_str}

🚨 CRITICAL: Return ONLY valid JSON. NO explanations, descriptions, or additional text.

Voice Command: {voice_command}

NOTE: Date parameters (datetime, date, time, timedelta) will be extracted separately. Focus on extracting non-date parameters that you can confidently identify from the voice command."""
        
        return command_details
    
    def _build_focused_parameter_prompt(self, voice_command: str) -> str:
        """Build a focused parameter extraction prompt that references command details from cache."""
        return f"""🚨 CRITICAL: Return ONLY valid JSON. NO explanations, descriptions, or additional text.

Extract the non-date parameters from the voice command. Command details are available above.

REQUIRED OUTPUT FORMAT:
{{"s": true, "p": {{...}}, "e": null}}

Where:
- "s": success status (true/false)
- "p": object containing extracted parameters
- "e": error object (null for success)

Voice Command: {voice_command}

Remember: ONLY JSON, no explanations."""
    
    def _is_date_parameter_type(self, param_type: str) -> bool:
        """Check if a parameter type is a date-related type (including arrays)."""
        if not param_type:
            return False
        
        # Basic date types
        basic_date_types = ["datetime", "date", "time", "timedelta"]
        if param_type in basic_date_types:
            return True
        
        # Array date types (Python typing syntax)
        array_date_types = [
            "List[datetime]", "list[datetime]",
            "List[date]", "list[date]", 
            "List[time]", "list[time]",
            "List[timedelta]", "list[timedelta]",
            "Array[datetime]", "array[datetime]",
            "Array[date]", "array[date]",
            "Array[time]", "array[time]", 
            "Array[timedelta]", "array[timedelta]"
        ]
        
        if param_type in array_date_types:
            return True
        
        # Check for generic array patterns
        if ("[" in param_type and "]" in param_type and 
            any(date_type in param_type.lower() for date_type in ["datetime", "date", "time", "timedelta"])):
            return True
            
        return False
    
    def _parse_parameter_response_wrapper(self, content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper function for _parse_parameter_response to match the expected signature."""
        try:
            parsed_response = self._parse_parameter_response(content)
            result["parsed_response"] = parsed_response
            return result
        except Exception as e:
            logger.error(f"❌ Wrapper failed to parse response: {e}")
            logger.error(f"❌ Raw content that failed to parse: {content}")
            # Return the result with an error so the main flow can handle it
            result["error"] = str(e)
            result["content"] = content
            return result


# Legacy class for backward compatibility - DO NOT DELETE
class LegacyParameterExtractionService:
    """
    Legacy parameter extraction service for parallel date and parameter extraction.
    This is kept for backward compatibility and future reference.
    """
    
    def __init__(self):
        self.llm_client = None  # Will be set when needed
    
    async def extract_parameters_and_dates(
        self,
        selected_command: CommandDefinition,
        param_inference_messages: List[Dict[str, str]],
        voice_command: str,
        cached_timezone: Optional[str],
        conversation_id: Optional[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Legacy method for parallel parameter and date extraction.
        This method is kept for reference but should not be used in new code.
        """
        # DO NOT DELETE - Legacy implementation for future reference
        # This was the original parallel approach that was replaced by the LLM manager
        pass
