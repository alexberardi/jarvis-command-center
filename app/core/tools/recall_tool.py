"""Recall Tool — semantic search over user memories.

Example: "What are my hobbies?" → LLM calls recall(query="hobbies interests activities")
"""

import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class RecallTool(IServerTool):
    """Tool for searching user memories via semantic similarity."""

    @property
    def name(self) -> str:
        return "recall"

    @property
    def description(self) -> str:
        return (
            "Search your memory for information about the user. "
            "Use when the user asks about their preferences, past conversations, "
            "or personal details that aren't in the current conversation. "
            "Provide a descriptive search query. Requires speaker identification."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g. 'coffee preferences', 'travel habits', 'dietary restrictions')",
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "fact", "note", "general"],
                    "description": "Optional: filter by memory category",
                },
            },
            "required": ["query"],
        }

    def execute(
        self,
        query: str,
        category: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search user memories by semantic similarity.

        Args:
            query: Search text to find relevant memories
            category: Optional category filter
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
                "message": "I need to know who's speaking to search memories. Voice identification is required.",
            }

        if not household_id:
            return {"error": "no_household", "message": "No household context available"}

        # Get settings for recall
        limit, threshold = self._get_recall_settings()

        try:
            from app.db import get_session_local
            from app.services.memory_service import MemoryService

            SessionLocal = get_session_local()
            db = SessionLocal()
            try:
                svc = MemoryService(db)

                # Try vector search first
                results = self._vector_search(
                    svc, speaker_user_id, household_id, query,
                    limit=limit, category=category, threshold=threshold,
                )

                # Fallback to substring search if vector search fails or returns nothing
                if results is None:
                    results = svc.search_memories_substring(
                        user_id=speaker_user_id,
                        household_id=household_id,
                        query=query,
                        limit=limit,
                        category=category,
                    )
                    search_method = "substring"
                else:
                    search_method = "vector"

                if not results:
                    return {
                        "status": "no_results",
                        "message": f"No memories found matching '{query}'",
                    }

                memories = []
                for memory, score in results:
                    memories.append({
                        "content": memory.content,
                        "category": memory.category,
                        "similarity": round(score, 3),
                    })

                logger.info(
                    f"🔍 Recall for user {speaker_user_id}: "
                    f"'{query}' → {len(memories)} results ({search_method})"
                )

                return {
                    "status": "found",
                    "count": len(memories),
                    "memories": memories,
                    "search_method": search_method,
                }
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return {"error": "recall_failed", "message": f"Failed to search memories: {e}"}

    def _vector_search(
        self,
        svc: Any,
        user_id: int,
        household_id: str,
        query: str,
        limit: int,
        category: str | None,
        threshold: float,
    ) -> list[tuple[Any, float]] | None:
        """Attempt vector search. Returns None if embedding service is unavailable."""
        try:
            from app.core.llm_proxy_client import LLMProxyClient

            client = LLMProxyClient()
            vectors = client.create_embeddings_sync([query])

            if not vectors or not vectors[0]:
                logger.warning("⚠️ Embedding service returned empty result, falling back to substring")
                return None

            return svc.search_memories(
                user_id=user_id,
                household_id=household_id,
                query_embedding=vectors[0],
                limit=limit,
                category=category,
                similarity_threshold=threshold,
            )

        except Exception as e:
            logger.warning(f"⚠️ Vector search failed, falling back to substring: {e}")
            return None

    @staticmethod
    def _get_recall_settings() -> tuple[int, float]:
        """Get recall settings (max_results, similarity_threshold)."""
        limit = 5
        threshold = 0.3
        try:
            from app.services.settings_service import get_settings_service

            settings = get_settings_service()
            val = settings.get("memory.recall_max_results")
            if val is not None:
                limit = int(val)
            val = settings.get("memory.recall_similarity_threshold")
            if val is not None:
                threshold = float(val)
        except Exception:
            pass
        return limit, threshold
