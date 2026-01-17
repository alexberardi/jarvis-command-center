"""
Interface for Server-Side Tools.

Server tools are functions that can be executed by the LLM during conversations.
Tools are automatically discovered and registered by scanning the tools directory.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IServerTool(ABC):
    """
    Abstract base class for server-side tools.
    
    Each tool implementation should:
    1. Inherit from this interface
    2. Implement the required properties and methods
    3. Be placed in app/core/tools/ or app/core/tools/custom/
    4. Will be automatically discovered and registered
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique tool name (used in tool calls).
        
        Examples: "resolve_relative_date", "request_validation"
        """
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Tool description for the LLM.
        
        Should explain:
        - What the tool does
        - When to use it
        - What it returns
        """
        ...
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        JSON Schema for tool parameters (OpenAI format).
        
        Example:
        {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param_name"]
        }
        """
        ...
    
    @property
    def included_system_prompt_text(self) -> Optional[str]:
        """
        Optional extra guidance to include in the main system prompt.
        Use this for global instructions that shouldn't live in the tool description.
        """
        return None

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with provided arguments.
        
        Args:
            **kwargs: Tool parameters as specified in the parameters schema
            
        Returns:
            Dict with tool results. Should be JSON-serializable.
            
        Example:
            {
                "result": "some value",
                "status": "success"
            }
            
        Or for errors:
            {
                "error": "error_code",
                "message": "Human readable error"
            }
        """
        ...
    
    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI function calling format.
        
        Returns:
            Tool definition in OpenAI format
        """
        tool_def = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
        if self.included_system_prompt_text:
            tool_def["included_system_prompt_text"] = self.included_system_prompt_text
        return tool_def

