"""
Example Custom Model Implementation.

This is an example of how to create a custom model that extends
the BaseModel with specific optimizations or changes.
"""

import logging
from typing import Dict, Any, List, Optional

from app.core.models.base_model import BaseModel

logger = logging.getLogger("uvicorn")

class ExampleCustomModel(BaseModel):
    """
    Example custom model that extends BaseModel.
    
    This shows how you can:
    1. Inherit from BaseModel to get all the complex functionality
    2. Override specific methods to customize behavior
    3. Add your own custom features
    
    To use this model, set:
    JARVIS_MODEL_INTERFACE=ExampleCustomModel
    """
    
    @property
    def name(self) -> str:
        return "ExampleCustomModel"
    
    async def perform_inference(
        self,
        voice_command: str,
        conversation_id: str,
        node_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Custom inference that adds logging and preprocessing.
        """
        logger.info(f"🎨 ExampleCustomModel processing: '{voice_command}'")
        
        # Add custom preprocessing
        processed_command = self._preprocess_command(voice_command)
        
        # Use base model's inference with processed command
        result = await super().perform_inference(
            voice_command=processed_command,
            conversation_id=conversation_id,
            node_context=node_context
        )
        
        # Add custom postprocessing
        result = self._postprocess_result(result)
        
        logger.info(f"✅ ExampleCustomModel result: {result}")
        return result
    
    def _preprocess_command(self, voice_command: str) -> str:
        """Custom preprocessing - example: normalize common phrases."""
        # Example: Convert casual phrases to more formal ones
        replacements = {
            "hey": "please",
            "gimme": "give me",
            "what's": "what is"
        }
        
        processed = voice_command.lower()
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        return processed
    
    def _postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Custom postprocessing - example: add metadata."""
        if result.get("s"):  # If successful
            result["_metadata"] = {
                "processed_by": "ExampleCustomModel",
                "version": "1.0.0"
            }
        
        return result
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return custom model capabilities."""
        capabilities = super().get_capabilities()
        capabilities["custom_features"].update({
            "command_preprocessing": True,
            "result_postprocessing": True,
            "metadata_injection": True
        })
        return capabilities
