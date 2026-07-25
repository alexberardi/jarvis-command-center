"""
Conversation start request model.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from .voice_command_request import CommandDefinition


class AdapterOverride(BaseModel):
    """Optional adapter override for a conversation (Phase 2 eval gate).

    Honored only when JARVIS_TEST_MODE=1 is set on the server. Allows eval
    harnesses to target a specific trained adapter without touching node DB
    state. Falls back to the node's configured adapter when absent or ignored.

    To force a baseline run (no adapter at all — overrides the Phase 5
    household adapter), pass hash="" and enabled=False.
    """
    hash: str = ""
    scale: float = 1.0
    enabled: bool = True


class ConversationStartRequest(BaseModel):
    conversation_id: str
    node_context: Optional[dict] = None  # Keep for backward compatibility but we'll use provider context
    available_commands: Optional[List[CommandDefinition]] = None  # Available commands for this conversation
    client_tools: Optional[List[Dict[str, Any]]] = None  # Client-side tool definitions in OpenAI format
    # DEPRECATED / IGNORED: the warmup inference now ALWAYS runs and is never
    # skippable. Field kept only so older nodes that still send it don't 422
    # during rollout. CC does not act on this value.
    skip_warmup_inference: bool = False
    adapter_settings: Optional[AdapterOverride] = None  # Phase 2: test-mode adapter override
