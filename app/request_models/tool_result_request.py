"""
Tool Result Request Model.

Used when clients submit tool execution results to continue a conversation.
"""

from pydantic import BaseModel
from typing import List, Any


class ToolResult(BaseModel):
    """Result from a tool execution."""
    tool_call_id: str
    output: Any  # Can be string, dict, list, etc.


class ToolResultRequest(BaseModel):
    """Request to continue a conversation with tool execution results."""
    conversation_id: str
    tool_results: List[ToolResult]

