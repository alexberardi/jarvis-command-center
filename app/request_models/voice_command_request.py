from pydantic import BaseModel
from typing import List, Optional

class CommandParameter(BaseModel):
    name: str
    type: str
    required: bool = True  # Default to required for backward compatibility
    description: Optional[str] = None
    enum_values: Optional[List[str]] = None  # Enum values for this parameter

class CommandDefinition(BaseModel):
    command_name: str
    description: str
    parameters: List[CommandParameter]
    keywords: Optional[List[str]] = None  # Keywords for command filtering
    enum_values: Optional[List[str]] = None  # Enum values for command filtering
    example: Optional[str] = None  # Example usage for the LLM to follow
    rules: Optional[List[str]] = None  # General rules for this command
    critical_rules: Optional[List[str]] = None  # Critical rules that must be followed

class VoiceCommandRequest(BaseModel):
    voice_command: str
    conversation_id: Optional[str] = None
