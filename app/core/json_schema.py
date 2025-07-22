"""
Single source of truth for JSON response format used by LLM providers.
This module defines the schema and format for voice command responses.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class CommandError(BaseModel):
    """Error details for failed commands"""
    type: str = Field(..., description="Error type: no_command_match, ambiguous_command, missing_parameters, invalid_parameters")
    message: str = Field(..., description="Human-readable error message")
    missing_parameters: Optional[List[str]] = Field(None, description="List of missing parameter names")
    suggestions: Optional[List[str]] = Field(None, description="Suggested alternatives or fixes")
    clarification_question: Optional[str] = Field(None, description="Question to ask user for clarification")

class CommandResponse(BaseModel):
    """Single command response"""
    success: bool = Field(..., description="Whether the command was successfully identified")
    command_name: Optional[str] = Field(None, description="Name of the identified command")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters (never null)")
    errors: Optional[CommandError] = Field(None, description="Error details if success=False")

class VoiceCommandResponse(BaseModel):
    """Complete response for voice command processing"""
    commands: List[CommandResponse] = Field(..., description="List of identified commands")

# Compressed key format for token efficiency
COMPRESSED_KEYS = {
    "commands": "c",
    "success": "s", 
    "command_name": "n",
    "parameters": "p",
    "errors": "e"
}

def get_json_format_examples() -> Dict[str, Dict[str, Any]]:
    """Get example JSON responses for LLM prompting"""
    return {
        "success_example": {
            "c": [
                {
                    "s": True,
                    "n": "turn_on_lights", 
                    "p": {"room": "bedroom"},
                    "e": None
                }
            ]
        },
        "error_example": {
            "c": [
                {
                    "s": False,
                    "n": None,
                    "p": {},
                    "e": {
                        "type": "no_command_match",
                        "message": "Could not identify what you want to turn on",
                        "clarification_question": "What would you like to turn on?"
                    }
                }
            ]
        },
        "multi_command_example": {
            "c": [
                {
                    "s": True,
                    "n": "turn_on_lights",
                    "p": {"room": "bedroom"},
                    "e": None
                },
                {
                    "s": True,
                    "n": "play_music",
                    "p": {"song": "jazz"},
                    "e": None
                }
            ]
        }
    }

def get_json_format_instructions() -> str:
    """Get formatted instructions for JSON response format"""
    return """
JSON Response Format (ONLY this format):
{"c": [{"s": true, "n": "name", "p": {...}, "e": null}, ...]}

Error Response Format (ONLY this format):
{"s": false, "n": null, "p": null, "e": {"type": "error_type", "message": "desc", "missing_parameters": [...], "suggestions": [...], "clarification_question": "..."}}

Key Legend: c=commands, s=success, n=command_name, p=parameters, e=errors

Error types:
- no_command_match
- ambiguous_command  
- missing_parameters
- invalid_parameters

Parameters must be objects, never null (use {} for empty parameters)
All boolean values must be true/false, never null
"""

def create_fallback_prompt(bad_json: str, good_example: str) -> str:
    """Create a prompt for the fallback LLM to extract JSON"""
    error_response = '{"c":[{"s":false,"n":null,"p":{},"e":{"type":"invalid_parameters","message":"Could not extract valid JSON"}}]}'
    
    return f"""We're having trouble extracting JSON from this response:

{bad_json}

Can you extract the valid JSON and return it in this exact format:

{good_example}

CRITICAL INSTRUCTIONS:
- Respond ONLY with the valid JSON object - nothing else
- Use the compressed key format (c, s, n, p, e)
- NO explanations, NO comments, NO additional text
- NO line breaks after the JSON
- NO trailing characters or punctuation
- If you can't extract valid JSON, return ONLY: {error_response}
- Your response must be parseable by json.loads() immediately
""" 