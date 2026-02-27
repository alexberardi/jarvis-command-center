"""Remember Tool — saves a user memory from voice commands.

Example: "Remember that I like my coffee black"
"""

import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class RememberTool(IServerTool):
    """Tool for saving user memories via voice (e.g. 'remember that I like X')."""

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Save a piece of information about the user for future conversations. "
            "Use when the user says 'remember that...', 'note that...', or 'save that...'. "
            "Requires speaker identification."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember (e.g. 'likes coffee black')",
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "fact", "note"],
                    "description": "Type of memory: preference (likes/dislikes), fact (about the person), note (general)",
                },
                "key": {
                    "type": "string",
                    "description": "Optional key for updating existing memories (e.g. 'coffee_preference')",
                },
            },
            "required": ["content"],
        }

    def execute(
        self,
        content: str,
        category: str = "general",
        key: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Save a memory for the identified speaker.

        Args:
            content: The memory text to save
            category: Memory category (preference, fact, note)
            key: Optional structured key for upsert
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
                "message": "I need to know who's speaking to remember things. Voice identification is required.",
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
                memory = svc.save_memory(
                    user_id=speaker_user_id,
                    household_id=household_id,
                    content=content,
                    category=category,
                    key=key,
                    source="voice",
                )
                speaker_name = node_context.get("speaker_name", f"user {speaker_user_id}")
                logger.info(f"💾 Remembered for {speaker_name}: {content}")

                # Best-effort: generate embedding for the saved memory
                self._generate_embedding(memory.id, content)

                return {"status": "remembered", "content": content, "category": category}
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return {"error": "save_failed", "message": f"Failed to save memory: {e}"}

    def _generate_embedding(self, memory_id: int, content: str) -> None:
        """Generate and store embedding for a memory (best-effort).

        Uses the sync embedding client to avoid event-loop conflicts
        inside FastAPI. If embedding fails, the memory is still saved —
        it just won't be searchable via vector similarity.
        """
        try:
            from app.core.llm_proxy_client import LLMProxyClient
            from app.db import get_session_local
            from app.services.memory_service import MemoryService

            client = LLMProxyClient()
            vectors = client.create_embeddings_sync([content])

            if vectors and vectors[0]:
                SessionLocal = get_session_local()
                db = SessionLocal()
                try:
                    svc = MemoryService(db)
                    svc.update_embedding(memory_id, vectors[0])
                    logger.info(f"🔢 Generated embedding for memory {memory_id}")
                finally:
                    db.close()

        except Exception as e:
            logger.warning(f"⚠️ Failed to generate embedding for memory {memory_id}: {e}")
