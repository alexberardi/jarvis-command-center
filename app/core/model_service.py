"""
Model Service for Jarvis Voice Assistant.

This service provides a clean interface to the model system and can
gradually replace the existing prompt provider architecture.
"""

import logging
from typing import Dict, Any, List, Optional

from app.core.model_factory import ModelFactory
from app.core.interfaces.imodel_interface import IModelInterface
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
