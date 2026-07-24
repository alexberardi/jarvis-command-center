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
    # Seconds of speech-like audio detected in the fixed window immediately
    # before the wake word fired. The node computes this from a rolling RMS
    # ring buffer kept alongside the wake-detect loop. Strong signal for
    # not_for_me decisions: ~0s before wake → likely directed; multiple
    # seconds → wake fired mid-conversation, lean toward silent abort.
    # None when the node didn't report it (old client, alternate entry path).
    pre_wake_speech_seconds: Optional[float] = None
    # Acoustic affect read from whisper (``{read, arousal, confidence}``) when
    # voice.emotion_enabled is on, forwarded verbatim by the node. Surfaced to
    # the LLM as a per-turn tone hint (shape, don't announce). None when the
    # feature is off or the read was withheld.
    affect: Optional[Dict[str, Any]] = None
