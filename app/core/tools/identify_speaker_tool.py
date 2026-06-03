"""Identify Speaker Tool — answers "who am I?" / "do you know who I am?".

The speaker's name already lives in the system prompt (built by
``system_prompt_builder`` from ``node_context["speaker_name"]``), but a
strict not_for_me prompt makes the LLM hedge on these questions — the
asker isn't naming Jarvis, isn't issuing an imperative, and the question
doesn't obviously map to a tool. Surfacing identity as a real tool gives
the LLM (and the fastText router) an unambiguous target, so the
not_for_me prompt can stay strict on borderline ambient without
silencing the "who am I" class.

Returns ``{"speaker_name": <name>}`` on success or
``{"speaker_name": null, "message": ...}`` when voice recognition didn't
resolve a user. The LLM phrases the spoken answer from this output via
the standard tool-result formatting call.
"""

import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


# Placeholder ``speaker_name`` the system-prompt builder falls back to
# when nothing is set. Treat as "unknown" — same as a missing key.
_UNKNOWN_SPEAKER_PLACEHOLDERS = {"default", "user", ""}


class IdentifySpeakerTool(IServerTool):
    """Server-side tool for "who am I?" identity questions."""

    @property
    def name(self) -> str:
        return "identify_speaker"

    @property
    def description(self) -> str:
        return (
            "Identify who is currently speaking to you. Use when the user "
            "asks who they are, what their name is, or whether you know "
            "them: \"who am I?\", \"do you know who I am?\", \"what's my "
            "name?\", \"who is this?\" (when asked about themselves). "
            "Takes no arguments — the answer comes from the active voice "
            "session's recognized speaker. Returns the speaker's name or "
            "a not-recognized signal when voice ID didn't match a profile."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @property
    def included_system_prompt_text(self) -> Optional[str]:
        return (
            "Identity questions are valid commands and satisfy the "
            "positive-evidence requirement: \"who am I?\", \"do you know "
            "who I am?\", \"what's my name?\", \"who's speaking?\" all "
            "map to the identify_speaker tool. Call it directly; do not "
            "answer from the speaker block in the system prompt and do "
            "not emit <not_for_me/> for these."
        )

    def execute(
        self,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not conversation_id:
            return {
                "speaker_name": None,
                "error": "no_conversation",
                "message": "I don't have an active session to identify you from.",
            }

        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return {
                "speaker_name": None,
                "error": "no_context",
                "message": "I don't have an active session to identify you from.",
            }

        speaker_user_id = node_context.get("speaker_user_id")
        raw_name = node_context.get("speaker_name")
        name = (raw_name or "").strip()

        if not speaker_user_id or name.lower() in _UNKNOWN_SPEAKER_PLACEHOLDERS:
            logger.info(
                "🔎 identify_speaker: unknown speaker (speaker_user_id=%s, raw_name=%r)",
                speaker_user_id,
                raw_name,
            )
            return {
                "speaker_name": None,
                "message": (
                    "I don't recognize your voice yet. Enroll a voice "
                    "profile and I'll know you next time."
                ),
            }

        logger.info("🔎 identify_speaker: %s (user_id=%s)", name, speaker_user_id)
        return {"speaker_name": name}
