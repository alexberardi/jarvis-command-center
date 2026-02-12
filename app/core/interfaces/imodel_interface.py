"""
Model Interface ABC for Jarvis Voice Assistant.

This interface defines the contract for different model implementations.
The base implementation shows the full complexity of the current system,
and custom implementations can override methods to optimize for their specific model.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.request_models.voice_command_request import CommandDefinition

class IModelInterface(ABC):
    """
    Abstract base class for model implementations.
    
    This interface allows different models to implement their own inference strategies:
    - Tool-based inference with JSON tool calls
    - Custom pipelines (whatever works for the model)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Model implementation name for environment variable matching.
        
        Examples:
        - "JarvisToolModel" (tool-based model)
        - "CustomLlama7B" (third-party implementation)
        - "OpenAIGPT4" (API-based implementation)
        
        This name is used with JARVIS_MODEL_INTERFACE environment variable.
        """
        ...
    
    @abstractmethod
    async def perform_warmup(
        self,
        node_context: Dict[str, Any],
        available_commands: List[CommandDefinition],
        conversation_id: str,
        timezone: Optional[str] = None
    ) -> None:
        """
        Initialize the model with context and available commands/tools.
        
        This method should:
        1. Build the complete system context (date context, node context, etc.)
        2. Format available commands for the model
        3. Cache the conversation state for future inference calls
        4. Prepare any model-specific context or prompts
        
        Args:
            node_context: Node information (room, user, device, etc.)
            available_commands: List of commands available to this node
            conversation_id: Unique conversation identifier for caching
            timezone: User's timezone for date calculations
        """
        ...
    
    @abstractmethod
    async def perform_inference(
        self,
        voice_command: str,
        conversation_id: str,
        node_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform tool-based inference and return model results.
        
        This is the main entry point for processing voice commands.
        Implementation can use any strategy:
        - Single LLM call with complete context
        - Multi-step inference pipeline
        - Custom processing logic
        
        Args:
            voice_command: The user's voice command text
            conversation_id: Conversation ID (must match perform_warmup call)
            node_context: Optional additional node context
            
        Returns:
            Standard Jarvis response format:
            {
                "s": bool,           # Success flag
                "n": str,            # Command name (or null)
                "p": Dict[str, Any], # Extracted parameters
                "e": Optional[Dict]  # Error object (if s=False)
            }
            
            Success example:
            {"s": true, "n": "weather_command", "p": {"location": "New York"}, "e": null}
            
            Error example:
            {"s": false, "n": null, "p": {}, "e": {"type": "no_command_match", "message": "..."}}
        """
        ...
    
    @abstractmethod
    async def cleanup_conversation(self, conversation_id: str) -> None:
        """
        Clean up resources associated with a conversation.
        
        This method should:
        1. Remove conversation from cache
        2. Free any model-specific resources
        3. Clean up temporary state
        
        Args:
            conversation_id: Conversation to clean up
        """
        ...
    
    # Optional methods that implementations can override for customization
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the model implementation is healthy and ready.
        
        Returns:
            Health status information:
            {
                "status": "healthy|degraded|unhealthy",
                "model_name": str,
                "details": Dict[str, Any]
            }
        """
        return {
            "status": "healthy",
            "model_name": self.name,
            "details": {}
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return information about this model's capabilities.

        Returns:
            Capability information:
            {
                "supports_single_shot": bool,
                "supports_streaming": bool,
                "max_context_length": Optional[int],
                "supported_languages": List[str],
                "custom_features": Dict[str, Any]
            }
        """
        return {
            "supports_single_shot": False,
            "supports_streaming": False,
            "max_context_length": None,
            "supported_languages": ["en"],
            "custom_features": {}
        }

    @property
    def use_tool_classifier(self) -> bool:
        """
        Whether to use the fastText tool classifier for routing hints.

        Override to return False for models that should rely purely on
        LLM reasoning (e.g., adapter-tuned models).
        """
        return True
    
    # Shared utility methods for all model implementations
    
    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text response.
        
        This is a shared utility method that all model implementations can use
        to parse JSON from LLM responses.
        
        Args:
            text: Text containing JSON response
            
        Returns:
            Parsed JSON dictionary or None if extraction fails
        """
        try:
            # Find JSON boundaries
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                return None
                
            # Extract JSON portion
            json_str = text[start_idx:end_idx + 1]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None
    
    def build_command_list_text(self, available_commands: List[Dict[str, Any]]) -> str:
        """
        Build formatted command list text for prompts.
        
        Args:
            available_commands: List of command definitions
            
        Returns:
            Formatted string describing available commands
        """
        command_list = []
        for cmd in available_commands:
            cmd_info = f"- {cmd['command_name']}: {cmd.get('description', 'No description')}"
            if cmd.get('parameters'):
                params = [f"{p['name']} ({p['type']})" for p in cmd['parameters']]
                cmd_info += f" [Parameters: {', '.join(params)}]"
            command_list.append(cmd_info)
        
        return "\n".join(command_list)
    
    def filter_parameters_by_type(self, command_def: Dict[str, Any], include_date: bool = True) -> List[Dict[str, Any]]:
        """
        Filter command parameters by type (date vs non-date).
        
        Args:
            command_def: Command definition with parameters
            include_date: If True, include date/time parameters; if False, exclude them
            
        Returns:
            Filtered list of parameters
        """
        if not command_def.get("parameters"):
            return []
        
        filtered_params = []
        for param in command_def["parameters"]:
            param_type = param.get("type", "").lower()
            param_name = param.get("name", "").lower()
            
            # Check if this is a date/time parameter
            is_date_param = (
                any(date_word in param_type for date_word in ["date", "time", "datetime"]) or
                any(date_word in param_name for date_word in ["date", "time", "when", "schedule"])
            )
            
            # Include based on filter criteria
            if include_date == is_date_param:
                filtered_params.append(param)
        
        return filtered_params
    
    def build_parameter_description_text(self, parameters: List[Dict[str, Any]]) -> str:
        """
        Build formatted parameter description text for prompts.
        
        Args:
            parameters: List of parameter definitions
            
        Returns:
            Formatted string describing parameters
        """
        param_descriptions = []
        for param in parameters:
            desc = f"- {param['name']} ({param['type']}): {param.get('description', 'No description')}"
            if param.get('required', False):
                desc += " [REQUIRED]"
            param_descriptions.append(desc)
        
        return "\n".join(param_descriptions)
