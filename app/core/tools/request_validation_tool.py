"""
Request Validation Tool.

Requests clarification or validation from the user when parameters are ambiguous.
"""

import logging
from typing import Dict, Any, Optional, List

from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class RequestValidationTool(IServerTool):
    """
    Tool for requesting user validation or clarification.
    
    This is a special tool that triggers a validation flow rather than
    returning data directly. The execution service handles this specially.
    """
    
    @property
    def name(self) -> str:
        return "request_validation"
    
    @property
    def description(self) -> str:
        return (
            "Ask the user to clarify missing or ambiguous required parameters. "
            "Use as a last resort only when you cannot extract required parameters from the command."
        )

    @property
    def included_system_prompt_text(self) -> str:
        return ""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarification question to ask the user"
                },
                "parameter_name": {
                    "type": "string",
                    "description": "The parameter that needs clarification"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of options for the user to choose from"
                }
            },
            "required": ["question", "parameter_name"]
        }
    
    def execute(
        self,
        question: str,
        parameter_name: str,
        options: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute validation request.
        
        This returns a special marker that the tool executor recognizes
        and handles as a validation request.
        
        Args:
            question: Clarification question
            parameter_name: Parameter needing clarification
            options: Optional list of choices
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Validation request object with special marker
        """
        logger.info(f"      🔍 Requesting validation for: {parameter_name}")
        logger.debug(f"      └─ Question: {question}")
        
        return {
            "_validation_request": True,
            "question": question,
            "parameter_name": parameter_name,
            "options": options or []
        }

