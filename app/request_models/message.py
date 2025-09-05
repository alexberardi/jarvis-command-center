"""
Message model for chat interactions.
"""

from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str
