from pydantic import BaseModel
from typing import List, Dict, Optional

class CommandParameter(BaseModel):
    name: str
    type: str

class CommandDefinition(BaseModel):
    command_name: str
    description: str
    parameters: List[CommandParameter]
    keywords: Optional[List[str]] = None  # Keywords for command filtering

class VoiceCommandRequest(BaseModel):
    voice_command: str
    node_context: Dict[str, str]
    available_commands: List[CommandDefinition]
