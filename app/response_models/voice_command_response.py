from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from enum import Enum


class StopReason(str, Enum):
    """Reason why the LLM stopped generating."""
    COMPLETE = "complete"
    TOOL_CALLS = "tool_calls"
    VALIDATION_REQUIRED = "validation_required"
    ERROR = "error"


class VoiceCommandError(BaseModel):
    type: str
    message: str
    missing_parameters: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    clarification_question: Optional[str] = None


class SingleCommandResponse(BaseModel):
    """Response for a single command"""
    success: bool
    command_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    errors: Optional[VoiceCommandError] = None


class RequestInformation(BaseModel):
    """Information about the original request"""
    voice_command: str
    conversation_id: Optional[str] = None


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""
    id: str
    type: str = "function"
    function: Dict[str, Any]
    failure_message: Optional[str] = None


class ValidationRequest(BaseModel):
    """Validation/clarification request from the server (stub for future implementation)."""
    question: str
    parameter_name: str
    options: Optional[List[str]] = None
    tool_call_id: Optional[str] = None


class VoiceCommandResponse(BaseModel):
    """Response that can contain one or multiple commands"""
    commands: List[SingleCommandResponse]
    request_information: Optional[RequestInformation] = None
    
    # New fields for tool-based architecture
    stop_reason: Optional[StopReason] = None
    tool_calls: Optional[List[ToolCall]] = None
    validation_request: Optional[ValidationRequest] = None
    assistant_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Returns True if all commands succeeded"""
        return all(cmd.success for cmd in self.commands)
    
    @property
    def has_errors(self) -> bool:
        """Returns True if any command has errors"""
        return any(cmd.errors is not None for cmd in self.commands)
    
    @property
    def errors(self) -> List[VoiceCommandError]:
        """Returns all errors from all commands"""
        return [cmd.errors for cmd in self.commands if cmd.errors is not None] 