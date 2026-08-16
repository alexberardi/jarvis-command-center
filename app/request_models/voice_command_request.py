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
    # Callbacks this command opts in to being proposed as a tap-to-confirm card
    # by any agent (the proposable-action contract). Opaque passthrough of the
    # SDK ProposableAction.to_dict() wire form; the capability registry + the
    # proposable-action dispatcher read it. None/absent = nothing proposable.
    proposable_actions: Optional[List[Dict[str, Any]]] = None

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
    # Turn provenance — how the mic came to be open for this transcript.
    # "wake": a wake-word fire captured it (wake_confidence carries the OWW
    # detection score). "follow_up": no wake word — the node kept the mic
    # open after TTS (follow_up_iteration is the 1-based window iteration).
    # Selects the not_for_me posture for the turn (see core/turn_context.py).
    # All None on old clients → CC falls back to inferring wake mode from
    # pre_wake_speech_seconds (only the wake path measures it).
    turn_source: Optional[str] = None
    wake_confidence: Optional[float] = None
    follow_up_iteration: Optional[int] = None
    # Self-playback context — the node was playing media out of its OWN
    # speaker when the wake fired (it hard-pauses/ducks on wake, so the
    # command audio itself is clean; the signal is about the moments BEFORE).
    # During self-playback the node's pre-wake VAD calibrates against the
    # music bleed, so pre_wake_speech_seconds reads ~0 on a REAL mid-music
    # wake — the quiet-room fingerprint — and residual music in the wake clip
    # can degrade verification phrase-match. CC treats the VAD signal as
    # uninformative and softens clip-verdict leans for these turns (see
    # core/direction_hint.py, core/turn_context.py, core/wake_verification.py).
    # None/absent on old nodes → behavior unchanged.
    self_playback: Optional[bool] = None
    # What kind of media was playing. Only "music" today; the media-aware
    # hint rules key on it so a future kind doesn't silently inherit
    # music-specific posture (music-control commands, lyric bleed).
    self_playback_kind: Optional[str] = None
