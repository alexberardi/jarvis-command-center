from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class CommandParameter(BaseModel):
    name: str
    type: str
    required: bool = True  # Default to required for backward compatibility
    description: Optional[str] = None
    enum_values: Optional[List[str]] = None  # Enum values for this parameter
    refinable: bool = False  # If True, param is stripped from Stage 1 and resolved via refinement

class CommandExample(BaseModel):
    voice_command: str
    expected_parameters: Dict[str, Any]
    is_primary: bool = False

class CommandAntipattern(BaseModel):
    command_name: str
    description: str

class CommandDefinition(BaseModel):
    command_name: str
    description: str
    parameters: List[CommandParameter]
    keywords: Optional[List[str]] = None  # Keywords for command filtering
    enum_values: Optional[List[str]] = None  # Enum values for command filtering
    examples: Optional[List[CommandExample]] = None  # Structured examples format
    rules: Optional[List[str]] = None  # General rules for this command
    critical_rules: Optional[List[str]] = None  # Critical rules that must be followed
    allow_direct_answer: Optional[bool] = None  # If False, must call tool for this command
    antipatterns: Optional[List[CommandAntipattern]] = None  # Anti-patterns for tool selection

class VoiceCommandRequest(BaseModel):
    voice_command: str
    conversation_id: str  # Required in new architecture
    speaker_user_id: Optional[int] = None  # Actual speaker from STT (for mismatch detection with warmup)
