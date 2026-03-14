"""
Chat completion request model for LLM interactions.
"""

from pydantic import BaseModel
from typing import List, Optional
from .message import Message

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "live"
    temperature: Optional[float] = 0.7
    messages: List[Message]
