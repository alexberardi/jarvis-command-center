"""
LLM Manager for handling LLM responses and conversation management.
"""

import json
import logging
import time
import os
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from app.core.llm_proxy_client import LLMProxyClient
from app.core.json_utils import validate_and_clean_json_response
from app.core.conversation_cache import conversation_cache

logger = logging.getLogger("uvicorn")

class LLMManager:
    """Manages LLM interactions and response processing."""
    
    def __init__(self):
        self.llm_client = LLMProxyClient()
        self.llm_api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
    
    def get_chat_url(self) -> str:
        """Get the chat URL for parameter extraction."""
        return f"{self.llm_api_url.rstrip('/')}/v1/chat/completions"
    
    async def transcription_cleanup(
        self, 
        voice_command: str, 
        cleanup_prompt: str
    ) -> str:
        """
        Perform transcription cleanup using the LLM.
        
        Args:
            voice_command: The original voice command
            cleanup_prompt: The cleanup prompt to use
            
        Returns:
            Cleaned voice command
        """
        cleanup_messages = [
            {"role": "system", "content": cleanup_prompt},
            {"role": "user", "content": voice_command}
        ]
        
        try:
            cleanup_response = await self.llm_client.lightweight_chat(messages=cleanup_messages)
            cleanup_content = self._extract_content_from_response(cleanup_response)
            
            if cleanup_content and cleanup_content.strip():
                cleaned_command = cleanup_content.strip()
                logger.info(f"✅ Transcription cleaned: '{voice_command}' → '{cleaned_command}'")
                return cleaned_command
            else:
                logger.warning("⚠️ Transcription cleanup returned empty response, using original")
                return voice_command
                
        except Exception as e:
            logger.warning(f"⚠️ Transcription cleanup failed: {e}, using original command")
            return voice_command
    
    async def command_inference(
        self, 
        messages: List[Dict], 
        conversation_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Perform command inference using the LLM.
        
        Args:
            messages: The messages to send to the LLM
            conversation_id: Optional conversation ID for multi-turn conversations
            
        Returns:
            Tuple of (parsed_response, raw_content)
        """
        start_time = time.time()
        
        try:
            response = await self.llm_client.chat_completion(
                messages=messages,
                conversation_id=conversation_id
            )
            
            duration = time.time() - start_time
            logger.info(f"⚡ Command inference took {duration:.2f} seconds")
            
            # Extract content from response
            content = self._extract_content_from_response(response)
            if not content:
                raise ValueError("Empty response from command inference LLM")
            
            # Update conversation cache if conversation_id is provided
            if conversation_id:
                # Add assistant response to cache (user message was already added by caller)
                conversation_cache.add_message(conversation_id, {"role": "assistant", "content": content})
            
            # Parse and validate the response
            parsed_response = self._parse_command_inference_response(content)
            
            return parsed_response, content
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Command inference failed after {duration:.2f}s: {e}")
            raise
    
    async def command_inference_with_cache(
        self, 
        message: str, 
        conversation_id: str,
        available_commands: List[Dict] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Perform command inference with conversation cache management.
        
        Args:
            message: The user message to send
            conversation_id: Conversation ID for cache management
            
        Returns:
            Tuple of (parsed_response, raw_content)
        """
        start_time = time.time()
        
        try:
            # Get existing conversation from cache
            existing_messages = conversation_cache.get_messages(conversation_id)
            logger.info(f"🔍 Cache lookup for {conversation_id[:8]}: {'found' if existing_messages else 'not found'}")
            if existing_messages:
                logger.info(f"🔍 Found {len(existing_messages)} cached messages")
            
            # Debug: Check what's actually in the cache
            if conversation_cache.has_conversation(conversation_id):
                logger.info(f"🔍 Cache entry exists for {conversation_id[:8]}")
                cached_commands = conversation_cache.get_available_commands(conversation_id)
                logger.info(f"🔍 Cached commands: {'found' if cached_commands else 'not found'}")
                if cached_commands:
                    logger.info(f"🔍 Cached commands count: {len(cached_commands)}")
            else:
                logger.info(f"🔍 No cache entry exists for {conversation_id[:8]}")
            
            if not existing_messages:
                # Wait for cache to be ready (up to 2.5 seconds)
                logger.info(f"⏳ No cached conversation found, waiting for cache to be ready...")
                cache_ready = await self.wait_for_cache_ready(conversation_id)
                
                if cache_ready:
                    # Try to get messages again after waiting
                    existing_messages = conversation_cache.get_messages(conversation_id)
                    if existing_messages:
                        logger.info(f"✅ Cache is now ready, found {len(existing_messages)} messages")
                    else:
                        logger.error(f"❌ Cache reported ready but no messages found for {conversation_id}")
                        raise ValueError(f"No cached conversation found for {conversation_id} - cache may be corrupted")
                else:
                    # Cache not ready after waiting, this should not happen if warmup completed
                    logger.error(f"❌ No cached conversation found for {conversation_id} - warmup may not have completed")
                    raise ValueError(f"No cached conversation found for {conversation_id}")
            
            # Add user message to conversation
            messages = existing_messages.copy()
            messages.append({"role": "user", "content": message})
            
            logger.info(f"🔍 Sending {len(messages)} messages to LLM (cached conversation)")
            logger.info(f"🔍 First message role: {messages[0]['role']}, content length: {len(messages[0]['content'])}")
            
            # Add user message to cache
            conversation_cache.add_message(conversation_id, {"role": "user", "content": message})
            
            # Make LLM call
            response = await self.llm_client.chat_completion(
                messages=messages,
                conversation_id=conversation_id
            )
            
            duration = time.time() - start_time
            logger.info(f"⚡ Command inference took {duration:.2f} seconds")
            
            # Extract content from response
            content = self._extract_content_from_response(response)
            if not content:
                raise ValueError("Empty response from command inference LLM")
            
            # Add assistant response to cache
            conversation_cache.add_message(conversation_id, {"role": "assistant", "content": content})
            
            # Parse and validate the response
            parsed_response = self._parse_command_inference_response(content)
            
            return parsed_response, content
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Command inference failed after {duration:.2f}s: {e}")
            raise
    
    async def chat_completion_with_cache(
        self, 
        message: str, 
        conversation_id: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Perform chat completion with conversation cache management.
        
        Args:
            message: The user message to send
            conversation_id: Conversation ID for cache management
            
        Returns:
            Tuple of (parsed_response, raw_content)
        """
        start_time = time.time()
        
        try:
            # Get existing conversation from cache
            existing_messages = conversation_cache.get_command_inference_messages(conversation_id)
            if not existing_messages:
                raise ValueError(f"No cached conversation found for {conversation_id}")
            
            # Add user message to conversation
            messages = existing_messages.copy()
            messages.append({"role": "user", "content": message})
            
            # Add user message to cache
            conversation_cache.add_to_conversation(conversation_id, {"role": "user", "content": message})
            
            # Make LLM call
            response = await self.llm_client.chat_completion(
                messages=messages,
                conversation_id=conversation_id
            )
            
            duration = time.time() - start_time
            logger.info(f"⚡ Chat completion took {duration:.2f} seconds")
            
            # Extract content from response
            content = self._extract_content_from_response(response)
            if not content:
                raise ValueError("Empty response from chat completion LLM")
            
            # Add assistant response to cache
            conversation_cache.add_to_conversation(conversation_id, {"role": "assistant", "content": content})
            
            # Parse and validate the response
            parsed_response = self._parse_command_inference_response(content)
            
            return parsed_response, content
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Chat completion failed after {duration:.2f}s: {e}")
            raise
    

    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """
        Extract content from LLM response, handling different formats.
        
        Args:
            response: The LLM response
            
        Returns:
            Extracted content string
        """

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
    
    def _parse_command_inference_response(self, content: str) -> Dict[str, Any]:
        """
        Parse and validate command inference response.
        
        Args:
            content: The raw response content
            
        Returns:
            Parsed response data
        """
        # Check for HTML encoding issues and fix them
        if content:
            has_html_entities = '&#34;' in content or '&quot;' in content
            if has_html_entities:
                logger.warning("🔧 Fixing HTML entities in response...")
                original_content = content
                content = content.replace('&#34;', '"').replace('&quot;', '"')
                logger.info(f"🔧 Fixed response: {original_content} → {content}")
        
        # Check for potential truncation indicators
        if content and content.count('{') != content.count('}'):
            logger.warning(f"Brace mismatch detected! Opening braces: {content.count('{')}, Closing braces: {content.count('}')}")
            logger.warning(f"This suggests the response may be truncated")
        
        # Validate JSON structure
        json_valid, validation_issues = validate_and_clean_json_response(content)
        if not json_valid:
            for issue in validation_issues:
                logger.warning(f"Command inference JSON validation issue: {issue}")
            raise ValueError(f"Invalid command inference JSON response: {validation_issues[0] if validation_issues else 'Unknown error'}")
        
        # Parse the JSON
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_end <= json_start:
            raise ValueError("JSON response appears to be truncated")
            
        json_portion = content[json_start:json_end + 1]
        
        try:
            response_data = json.loads(json_portion)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
        
        # Validate structure
        if not isinstance(response_data, dict):
            raise ValueError("Command inference response is not a dictionary")
        
        return response_data
    
    def process_command_inference_response(
        self, 
        response_data: Dict[str, Any], 
        raw_content: str
    ) -> List[Dict[str, Any]]:
        """
        Process the parsed command inference response.
        
        Args:
            response_data: The parsed response data
            raw_content: The raw response content
            
        Returns:
            List of commands to process
        """
        # Check for invalid responses first
        command_name = response_data.get("n")
        
        # Handle null, empty, or invalid command names
        if not command_name or command_name == "" or command_name is None:
            error_type = "invalid_command_detected"
            if command_name is None:
                error_message = "LLM proxy returned null command name"
            elif command_name == "":
                error_message = "LLM proxy returned empty string command name"
            else:
                error_message = "LLM proxy returned invalid command name"
            
            logger.error(f"❌ {error_message}: {response_data}")
            
            # Return error response
            error_command = {
                "s": False,
                "n": None,
                "p": {},
                "e": {
                    "type": error_type,
                    "message": error_message
                }
            }
            return [error_command]
        
        # Check if it's the new single command format or old array format
        if "c" in response_data:
            # Old array format - convert to single command
            commands = response_data.get("c", [])
            if not commands:
                raise ValueError("Command inference response has empty 'c' array")
            command = commands[0]  # Take first command
            logger.info(f"📋 Using old array format, extracted command: {command}")
        else:
            # New single command format
            command = response_data
            commands = [command]  # Wrap in array for compatibility
            logger.info(f"📋 Using new single command format: {command}")
        
        return commands
    
    def add_to_conversation_history(
        self, 
        messages: List[Dict], 
        response_content: str, 
        conversation_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Add LLM response to conversation history.
        
        Args:
            messages: Current conversation messages
            response_content: The LLM response content
            conversation_id: Optional conversation ID
            
        Returns:
            Updated messages list
        """
        # For parameter inference, omit the "p" property to avoid parameter leakage
        try:
            response_data = json.loads(response_content)
            if "p" in response_data:
                # Create a copy without the "p" property
                clean_response = response_data.copy()
                del clean_response["p"]
                clean_content = json.dumps(clean_response)
                messages.append({"role": "assistant", "content": clean_content})
                logger.info(f"📋 Added cleaned command inference response (without parameters) to messages")
            else:
                messages.append({"role": "assistant", "content": response_content})
                logger.info(f"📋 Added command inference response (no parameters to clean) to messages")
        except (json.JSONDecodeError, TypeError):
            # If we can't parse as JSON, just append the original content
            messages.append({"role": "assistant", "content": response_content})
            logger.info(f"📋 Added command inference response (JSON parsing failed) to messages")
        
        return messages
    
    async def warmup_conversation(
        self, 
        conversation_id: str, 
        warmup_messages: List[Dict]
    ) -> None:
        """
        Warm up a conversation with the LLM.
        
        Args:
            conversation_id: The conversation ID
            warmup_messages: The messages to send for warmup
        """
        try:
            await self.llm_client.warmup_conversation(
                conversation_id=conversation_id,
                messages=warmup_messages
            )
            logger.info(f"✅ Successfully warmed up LLM with complete static prompt")
        except Exception as e:
            logger.error(f"❌ Background LLM warm-up failed for conversation {conversation_id}: {e}")
            raise
    
    async def warmup_conversation_background(
        self,
        conversation_id: str,
        llm_conversation_id: str,
        warmup_messages: List[Dict[str, str]],
        available_commands: List[Dict] = None,
        timezone: str = None
    ) -> None:
        """Background task to warm up the LLM and populate conversation cache."""
        try:
            # First, warm up the LLM
            await self.warmup_conversation(
                conversation_id=llm_conversation_id,
                warmup_messages=warmup_messages
            )
            
            # Then populate the conversation cache with the warmup messages
            
            logger.info(f"🔧 Populating cache for conversation {conversation_id[:8]}... with {len(warmup_messages)} messages")
            
            conversation_cache.set(
                conversation_id, 
                warmup_messages,  # messages (full conversation)
                available_commands, 
                timezone
            )
            
            # Verify cache was populated
            cached_messages = conversation_cache.get_messages(conversation_id)
            if cached_messages:
                logger.info(f"✅ Cache verification successful: {len(cached_messages)} messages cached")
            else:
                logger.error(f"❌ Cache verification failed: no messages found after population")
            
            logger.info(f"✅ Successfully warmed up LLM and populated cache for conversation {conversation_id[:8]}...")
            logger.info(f"   Messages: {len(warmup_messages)}")
            
        except Exception as e:
            logger.error(f"❌ Background LLM warm-up failed for conversation {conversation_id}: {e}")
            raise
    
    async def wait_for_cache_ready(
        self, 
        conversation_id: str,
        max_wait_time: float = 2.5  # 5 attempts * 0.5 seconds
    ) -> bool:
        """
        Wait for the LLM proxy cache to be ready.
        
        Args:
            conversation_id: The conversation ID
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            True if cache is ready, False if timeout reached
        """
        logger.info(f"⏳ Waiting for cache to be ready for conversation {conversation_id[:8]}...")
        retry_interval = 0.5
        attempts = int(max_wait_time / retry_interval)

        for attempt in range(attempts):
            has_conversation = conversation_cache.has_conversation(conversation_id)
            messages = conversation_cache.get_messages(conversation_id) if has_conversation else []

            if has_conversation and messages:
                logger.info(f"✅ Cache is ready for conversation {conversation_id[:8]} (local)")
                return True

            if attempt < attempts - 1:
                logger.info(
                    f"⏳ Cache not ready (attempt {attempt + 1}/{attempts}) for {conversation_id[:8]}, "
                    f"waiting {retry_interval}s..."
                )
                await asyncio.sleep(retry_interval)

        logger.warning(f"⏰ Cache not ready after {attempts} attempts for conversation {conversation_id[:8]}")
        return False

    async def check_conversation_status(
        self, 
        conversation_id: str, 
        inference_type: str
    ) -> bool:
        """
        Check if LLM proxy cache is ready.
        
        Args:
            conversation_id: The conversation ID
            inference_type: Type of inference (e.g., "command inference")
            
        Returns:
            True if cache is ready, False otherwise
        """
        
        has_conversation = conversation_cache.has_conversation(conversation_id)
        messages = conversation_cache.get_messages(conversation_id) if has_conversation else []

        if has_conversation and messages:
            logger.info(f"✅ {inference_type} cache is ready locally for session {conversation_id[:8]}")
            return True

        logger.warning(
            f"⚠️ {inference_type} cache not ready locally for session {conversation_id[:8]} "
            f"(has_conversation={has_conversation}, messages={len(messages) if messages else 0})"
        )
        return False
    
    async def parallel_queries_with_cache(
        self,
        conversation_id: str,
        query_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple LLM queries in parallel with proper cache management.
        
        Args:
            conversation_id: The conversation ID to use
            query_tasks: List of query task objects with structure:
                {
                    "function_before": Optional[Callable] - function to run before the query
                    "prompt": str - the prompt to send
                    "function_after": Optional[Callable] - function to run after getting response
                }
        
        Returns:
            List of results from each query task
        """
        try:
            import time
            debug_start = time.time()
            logger.info(f"🐛 DEBUG: parallel_queries_with_cache entered at {debug_start}")
            
            # Retrieve the conversation cache
            base_messages = conversation_cache.get_messages(conversation_id)
            if not base_messages:
                raise ValueError(f"No cached messages found for {conversation_id}")
            logger.info(f"✅ Using cached messages for parallel queries")
            
            debug_after_cache = time.time()
            logger.info(f"🐛 DEBUG: Retrieved cache at {debug_after_cache}, took {debug_after_cache - debug_start:.3f}s")
            
            logger.info(f"🔄 Starting {len(query_tasks)} parallel queries for conversation {conversation_id}")
            
            # Create individual query functions
            async def execute_single_query(task: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    query_start = time.time()
                    logger.info(f"🐛 DEBUG: Starting single query at {query_start}")
                    
                    # Create a copy of the base messages for this query
                    messages = base_messages.copy()
                    
                    # Run pre-query function if provided
                    if "function_before" in task and task["function_before"]:
                        if asyncio.iscoroutinefunction(task["function_before"]):
                            await task["function_before"](messages)
                        else:
                            task["function_before"](messages)
                    
                    # Add the prompt to the messages
                    messages.append({"role": "user", "content": task["prompt"]})
                    
                    # Execute the LLM query with timeout
                    try:
                        llm_start = time.time()
                        logger.info(f"🐛 DEBUG: About to call LLM at {llm_start}")
                        
                        response = await asyncio.wait_for(
                            self.llm_client.chat_completion(
                                messages=messages,
                                conversation_id=conversation_id
                            ),
                            timeout=30.0  # 30 second timeout
                        )
                        
                        llm_end = time.time()
                        logger.info(f"🐛 DEBUG: LLM call completed at {llm_end}, took {llm_end - llm_start:.3f}s")
                        
                    except asyncio.TimeoutError:
                        raise ValueError("LLM query timed out after 30 seconds")
                    
                    # Extract content from response
                    content = self._extract_content_from_response(response)
                    if not content:
                        raise ValueError("Empty response from LLM")
                    
                    # Run post-query function if provided
                    result = {"content": content, "raw_response": response}
                    if "function_after" in task and task["function_after"]:
                        if asyncio.iscoroutinefunction(task["function_after"]):
                            result = await task["function_after"](content, result)
                        else:
                            result = task["function_after"](content, result)
                    
                    query_end = time.time()
                    logger.info(f"🐛 DEBUG: Single query completed at {query_end}, total took {query_end - query_start:.3f}s")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ Single query failed: {e}")
                    logger.error(f"❌ Exception type: {type(e).__name__}")
                    logger.error(f"❌ Exception args: {e.args}")
                    logger.error(f"❌ Task prompt: {task.get('prompt', 'No prompt')[:100]}...")
                    return {"error": str(e) if str(e) else f"{type(e).__name__}: {e.args}", "content": None}
            
            # Execute all queries in parallel
            debug_before_gather = time.time()
            logger.info(f"🐛 DEBUG: About to call asyncio.gather at {debug_before_gather}")
            
            results = await asyncio.gather(
                *[execute_single_query(task) for task in query_tasks],
                return_exceptions=True
            )
            
            debug_after_gather = time.time()
            logger.info(f"🐛 DEBUG: asyncio.gather completed at {debug_after_gather}, took {debug_after_gather - debug_before_gather:.3f}s")
            
            # Add all successful results to the cache
            for i, result in enumerate(results):
                if not isinstance(result, Exception) and "content" in result and result["content"]:
                    # Add user message
                    conversation_cache.add_message(
                        conversation_id, 
                        {"role": "user", "content": query_tasks[i]["prompt"]}
                    )
                    # Add assistant response
                    conversation_cache.add_message(
                        conversation_id, 
                        {"role": "assistant", "content": result["content"]}
                    )
            
            logger.info(f"✅ Completed {len(query_tasks)} parallel queries for conversation {conversation_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Parallel queries failed for conversation {conversation_id}: {e}")
            return [{"error": str(e), "content": None} for _ in query_tasks]
