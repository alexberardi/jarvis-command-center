"""Forget Tool — removes a user memory via voice commands.

Example: "Forget my coffee preference"
"""

import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class ForgetTool(IServerTool):
    """Tool for removing user memories via voice (e.g. 'forget my coffee preference')."""

    @property
    def name(self) -> str:
        return "forget"

    @property
    def description(self) -> str:
        return (
            "Remove a previously saved piece of information about the user. "
            "Use when the user says 'forget that...', 'remove...', or 'delete...'. "
            "Requires speaker identification."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content_match": {
                    "type": "string",
                    "description": "What to forget — matches against stored memories (e.g. 'coffee preference')",
                },
            },
            "required": ["content_match"],
        }

    def execute(
        self,
        content_match: str,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Remove memories matching the given description.

        Args:
            content_match: Substring to match against stored memories
            conversation_id: Passed by tool executor for context lookup
        """
        if not conversation_id:
            return {"error": "no_conversation", "message": "No conversation context available"}

        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return {"error": "no_context", "message": "No node context available"}

        speaker_user_id = node_context.get("speaker_user_id")
        household_id = node_context.get("household_id")

        if not speaker_user_id:
            return {
                "error": "no_speaker",
                "message": "I need to know who's speaking to forget things. Voice identification is required.",
            }

        if not household_id:
            return {"error": "no_household", "message": "No household context available"}

        try:
            from app.db import get_session_local
            from app.services.memory_service import MemoryService

            SessionLocal = get_session_local()
            db = SessionLocal()
            try:
                svc = MemoryService(db)
                count = svc.forget_memory(
                    user_id=speaker_user_id,
                    household_id=household_id,
                    content_match=content_match,
                )
                speaker_name = node_context.get("speaker_name", f"user {speaker_user_id}")
                if count > 0:
                    logger.info(f"🗑️ Forgot {count} memories for {speaker_name} matching '{content_match}'")
                    return {"status": "forgotten", "count": count, "match": content_match}
                else:
                    return {"status": "not_found", "count": 0, "message": f"No memories found matching '{content_match}'"}
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to forget memory: {e}")
            return {"error": "forget_failed", "message": f"Failed to forget memory: {e}"}
