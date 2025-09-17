"""
Base Model Implementation for Jarvis Voice Assistant.

This is the reference implementation that demonstrates the full complexity
of the current system. It shows exactly what needs to be handled:

- Complete date context generation
- Node context management
- Multi-step inference pipeline
- Complex prompt building
- Conversation caching
- Error handling

Custom implementations can inherit from this class and override specific
methods to optimize for their model's capabilities.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from app.core.interfaces.imodel_interface import IModelInterface
from app.core.llm_manager import LLMManager
from app.core.conversation_cache import conversation_cache
from app.core.general_context import get_general_context
from app.core.parameter_extraction_service import ParameterExtractionService
from app.core.command_validation_service import CommandValidationService
from app.context_providers.general_prompt_provider import GeneralPromptProvider
from app.request_models.voice_command_request import CommandDefinition

logger = logging.getLogger("uvicorn")

class BaseModel(IModelInterface):
    """
    Base model implementation showing the full current system complexity.
    
    This implementation demonstrates:
    1. Complete context building (dates, nodes, commands)
    2. Multi-step inference pipeline
    3. Comprehensive prompt management
    4. Conversation caching
    5. Error handling and validation
    
    Custom model implementations can inherit from this and override
    specific methods to optimize for their capabilities.
    """
    
    def __init__(self):
        """Initialize the base model with all required services."""
        self.llm_manager = LLMManager()
        self.general_provider = GeneralPromptProvider()
        self.parameter_extraction_service = ParameterExtractionService()
        self.command_validation_service = CommandValidationService()
        
        # BaseModel is self-contained - it uses its own internal services
        # No need for external prompt providers
        
        logger.info(f"🤖 Initialized {self.name} with all services")
    
    @property
    def name(self) -> str:
        return "BASE_MODEL"
    
    async def perform_warmup(
        self,
        node_context: Dict[str, Any],
        available_commands: List[CommandDefinition],
        conversation_id: str,
        timezone: Optional[str] = None
    ) -> None:
        """
        Comprehensive warmup showing the full system complexity.
        
        This method demonstrates everything needed for a complete warmup:
        1. Build complete date context with timezone handling
        2. Format node context with all available fields
        3. Generate comprehensive command descriptions
        4. Create complete static prompt with all rules and examples
        5. Cache conversation with full context
        """
        start_time = time.time()
        
        logger.info(f"🚀 Starting comprehensive warmup for {conversation_id[:8]}...")
        logger.info(f"   Node context: {node_context}")
        logger.info(f"   Available commands: {len(available_commands)}")
        logger.info(f"   Timezone: {timezone}")
        
        try:
            # Step 1: Build complete static prompt with all context
            complete_static_prompt = self.general_provider.build_complete_static_prompt(
                node_context=node_context,
                timezone=timezone,
                available_commands=available_commands,
                conversation_id=conversation_id
            )
            
            logger.info(f"📝 Built complete static prompt ({len(complete_static_prompt)} chars)")
            
            # Step 2: Build warm-up messages with full context
            warmup_messages = [
                {"role": "system", "content": complete_static_prompt},
                {"role": "user", "content": "This is a warm-up message to establish context."},
                {"role": "assistant", "content": "I understand. I'm ready to help with command inference and parameter extraction."}
            ]
            
            # Step 3: Cache conversation with complete context
            conversation_cache.set(
                conversation_id=conversation_id,
                messages=warmup_messages,
                available_commands=[cmd.model_dump() for cmd in available_commands],
                timezone=timezone
            )
            
            # Step 4: Perform actual LLM warmup
            await self.llm_manager.warmup_conversation(
                conversation_id=conversation_id,
                warmup_messages=warmup_messages
            )
            
            duration = time.time() - start_time
            logger.info(f"✅ Warmup completed in {duration:.2f}s for {conversation_id[:8]}")
            
        except Exception as e:
            logger.error(f"❌ Warmup failed for {conversation_id[:8]}: {e}")
            raise
    
    async def perform_inference(
        self,
        voice_command: str,
        conversation_id: str,
        node_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete multi-step inference pipeline showing full system complexity.
        
        This demonstrates the current 3-step process:
        1. Command inference (identify which command)
        2. Parameter extraction (extract non-date parameters)
        3. Date parameter extraction (handle relative dates)
        4. Result combination and validation
        """
        start_time = time.time()
        
        logger.info(f"🎯 Starting inference for: '{voice_command}'")
        logger.info(f"   Conversation: {conversation_id[:8]}")
        
        try:
            # Step 1: Command Inference
            command_result = await self._perform_command_inference(voice_command, conversation_id)
            
            if not command_result["s"]:
                logger.info("❌ Command inference failed")
                return command_result
            
            command_name = command_result["n"]
            logger.info(f"✅ Command identified: {command_name}")
            
            # Step 2: Get available commands for validation
            available_commands = conversation_cache.get_available_commands(conversation_id)
            if not available_commands:
                logger.error("❌ No available commands in cache")
                return {
                    "s": False,
                    "n": None,
                    "p": {},
                    "e": {"type": "cache_error", "message": "No available commands found"}
                }
            
            # Find the selected command definition
            selected_command = None
            for cmd_dict in available_commands:
                if cmd_dict["command_name"] == command_name:
                    selected_command = CommandDefinition(**cmd_dict)
                    break
            
            if not selected_command:
                logger.error(f"❌ Command {command_name} not found in available commands")
                return {
                    "s": False,
                    "n": None,
                    "p": {},
                    "e": {"type": "command_not_found", "message": f"Command {command_name} not available"}
                }
            
            # Step 3: Parameter Extraction (both non-date and date parameters)
            final_parameters = await self._perform_complete_parameter_extraction(
                selected_command=selected_command,
                voice_command=voice_command,
                conversation_id=conversation_id,
                node_context=node_context
            )
            
            # Step 4: Build final result
            final_result = {
                "s": True,
                "n": command_name,
                "p": final_parameters,
                "e": None
            }
            
            duration = time.time() - start_time
            logger.info(f"✅ Inference completed in {duration:.2f}s")
            logger.info(f"   Result: {final_result}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {"type": "inference_error", "message": str(e)}
            }
    
    async def cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up conversation resources."""
        logger.info(f"🧹 Cleaning up conversation {conversation_id[:8]}")
        
        # Remove from conversation cache
        conversation_cache.remove(conversation_id)
        
        # Additional cleanup can be added here for model-specific resources
        logger.info(f"✅ Cleanup completed for {conversation_id[:8]}")
    
    # Protected methods that can be overridden by implementations
    
    async def _perform_command_inference(
        self,
        voice_command: str,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Perform command inference using the current system approach.
        
        This shows the full complexity of command inference:
        1. Use cached conversation context
        2. Apply command inference prompt provider
        3. Process LLM response
        4. Validate and return result
        """
        logger.info("🔍 Performing command inference...")
        
        try:
            # Get available commands from cache for context
            available_commands = conversation_cache.get_available_commands(conversation_id)
            
            # Use LLM manager's cache-aware command inference
            command_response_data, command_response_content = await self.llm_manager.command_inference_with_cache(
                message=voice_command,
                conversation_id=conversation_id,
                available_commands=available_commands
            )
            
            # Process the response using LLM manager's processing logic
            commands = self.llm_manager.process_command_inference_response(
                command_response_data, 
                command_response_content
            )
            
            if commands and len(commands) > 0:
                return commands[0]  # Return first command result
            else:
                return {
                    "s": False,
                    "n": None,
                    "p": {},
                    "e": {"type": "no_command_match", "message": "No matching command found"}
                }
                
        except Exception as e:
            logger.error(f"❌ Command inference error: {e}")
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {"type": "command_inference_error", "message": str(e)}
            }
    
    async def _perform_complete_parameter_extraction(
        self,
        selected_command: CommandDefinition,
        voice_command: str,
        conversation_id: str,
        node_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform complete parameter extraction showing full system complexity.
        
        This demonstrates the current parameter extraction process:
        1. Extract non-date parameters
        2. Extract date parameters
        3. Combine results
        4. Validate parameters
        """
        logger.info(f"🔧 Performing parameter extraction for {selected_command.command_name}")
        
        try:
            # Use the parameter extraction service to handle the full complexity
            node_context_provider = self._build_mock_node_context_provider(node_context or {})
            
            extracted_params, cached_timezone = await self.parameter_extraction_service.build_parameter_context(
                selected_command=selected_command,
                request_conversation_id=conversation_id,
                conversation_cache=conversation_cache,
                node_context_provider=node_context_provider,
                voice_command=voice_command
            )
            
            return extracted_params
            
        except Exception as e:
            logger.error(f"❌ Parameter extraction error: {e}")
            return {}
    
    def _build_mock_node_context_provider(self, node_context: Dict[str, Any]):
        """Build a mock node context provider for parameter extraction."""
        class MockNode:
            def __init__(self, context):
                self.room = context.get("room", "unknown")
                self.node_id = context.get("node_id", "unknown")
                self.user = context.get("user", "default")
                self.voice_mode = context.get("voice_mode", "brief")
        
        class MockNodeContextProvider:
            def __init__(self, context):
                self.node = MockNode(context)
        
        return MockNodeContextProvider(node_context)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return capabilities of the base model."""
        return {
            "supports_single_shot": False,
            "supports_streaming": False,
            "max_context_length": None,
            "supported_languages": ["en"],
            "custom_features": {
                "multi_step_inference": True,
                "date_parameter_extraction": True,
                "conversation_caching": True,
                "command_validation": True
            }
        }
