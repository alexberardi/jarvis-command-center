from app.context_providers.general_prompt_provider import GeneralPromptProvider
from app.request_models.voice_command_request import CommandDefinition
from typing import Dict, List
import logging

logger = logging.getLogger("uvicorn")

class SystemPromptProvider:
    """Provider for general system prompt information used to warm up the LLM."""
    
    def __init__(self):
        self.general_provider = GeneralPromptProvider()
    
    @property
    def name(self) -> str:
        return "SYSTEM"
    
    def build_warmup_prompt(self, node_context: Dict[str, str], timezone: str = None, available_commands: List[CommandDefinition] = None) -> str:
        """Build the warm-up system prompt for command inference only."""
        logger.info(f"🔍 SystemPromptProvider received timezone: {timezone}")
        
        # Use the general prompt provider to build the command inference prompt
        warmup_prompt = self.general_provider.build_command_inference_prompt(node_context, timezone, available_commands)
        
        # Log the final prompt details
        logger.info(f"   Full warm-up prompt length: {len(warmup_prompt)} characters")
        logger.info(f"   Sample of warm-up prompt (first 500 chars): {warmup_prompt[:500]}")
        
        return warmup_prompt
