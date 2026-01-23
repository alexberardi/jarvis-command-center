"""
Conversation start request model.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from .voice_command_request import CommandDefinition

class ConversationStartRequest(BaseModel):
    conversation_id: str
    node_context: Optional[dict] = None  # Keep for backward compatibility but we'll use provider context
    available_commands: Optional[List[CommandDefinition]] = None  # Available commands for this conversation
    client_tools: Optional[List[Dict[str, Any]]] = None  # Client-side tool definitions in OpenAI format
    skip_warmup_inference: bool = False  # Skip the LLM warmup call (useful for vLLM which doesn't cache KV)
