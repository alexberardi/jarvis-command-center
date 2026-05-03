"""Agent context retrieval for voice command enrichment.

Searches household-wide memories (user_id IS NULL) injected by background
agents (calendar, news, weather) and returns the most relevant ones for
a given voice command.  Used during conversation processing to inject
a "Current context" section into the LLM prompt.

Retrieval strategy:
1. Embed the user's query (single text, ~10-20ms via local model)
2. pgvector cosine search against pre-embedded agent memories (~1-5ms)
3. Fallback to word-overlap substring search if embedding fails
"""

import logging

from sqlalchemy.orm import Session

from app.services.memory_service import MemoryService

logger = logging.getLogger("uvicorn")


class AgentContextService:
    """Retrieve agent-injected context relevant to a voice command."""

    def __init__(self, db: Session):
        self.db = db

    def get_relevant_context(
        self,
        household_id: str,
        query: str,
        max_results: int = 5,
        max_chars: int = 500,
        similarity_threshold: float = 0.25,
    ) -> str:
        """Search agent memories and format for prompt injection.

        Args:
            household_id: The household scope
            query: The user's voice command text
            max_results: Maximum number of context items to return
            max_chars: Maximum total characters for the formatted output
            similarity_threshold: Minimum cosine similarity (0-1)

        Returns:
            Formatted string like "Current context:\\n- item1\\n- item2"
            or empty string if no relevant context found.
        """
        results = self._search_vector(
            household_id, query, max_results, similarity_threshold
        )

        if not results:
            results = self._search_substring(household_id, query, max_results)

        if not results:
            return ""

        return self._format_results(results, max_chars)

    def _search_vector(
        self,
        household_id: str,
        query: str,
        limit: int,
        threshold: float,
    ) -> list[str]:
        """Try vector similarity search. Returns list of content strings."""
        try:
            from app.core.llm_proxy_client import LLMProxyClient

            client = LLMProxyClient()
            vectors = client.create_embeddings_sync([query])

            if not vectors or not vectors[0]:
                return []

            service = MemoryService(self.db)
            matches = service.search_household_memories(
                household_id=household_id,
                query_embedding=vectors[0],
                limit=limit,
                similarity_threshold=threshold,
            )

            return [m.content for m, _score in matches]

        except Exception as e:
            logger.debug("Agent context vector search failed, will try substring: %s", e)
            return []

    def _search_substring(
        self,
        household_id: str,
        query: str,
        limit: int,
    ) -> list[str]:
        """Fallback to word-overlap substring search."""
        try:
            service = MemoryService(self.db)
            matches = service.search_household_memories_substring(
                household_id=household_id,
                query=query,
                limit=limit,
            )
            return [m.content for m, _score in matches]

        except Exception as e:
            logger.warning("Agent context substring search failed: %s", e)
            return []

    @staticmethod
    def _format_results(contents: list[str], max_chars: int) -> str:
        """Format results as a 'Current context' block for prompt injection."""
        lines: list[str] = []
        header = "Current context (use this information to answer directly when relevant — no need to call a tool if the answer is here):"
        total_chars = len(header) + 1  # +1 for newline after header

        for content in contents:
            line = f"- {content}"
            if total_chars + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 1  # +1 for newline

        if not lines:
            return ""

        return header + "\n" + "\n".join(lines)
