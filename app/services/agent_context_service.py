"""Agent context retrieval for voice command enrichment.

Searches household-wide memories (user_id IS NULL) injected by background
agents (calendar, news, weather) and returns the most relevant ones for
a given voice command.  Used during conversation processing to inject
a "Current context" section into the LLM prompt.

Retrieval strategy:
1. Always include the latest from priority categories (weather, calendar)
2. Fill remaining slots with vector-search results from other categories
3. Fallback to word-overlap substring search if embedding fails
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.models import UserMemory
from app.services.memory_service import MemoryService

logger = logging.getLogger("uvicorn")

# Categories that are always included (most recent entry per category)
# regardless of vector similarity — they're universally relevant context.
PRIORITY_CATEGORIES = ["weather", "calendar"]


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

        Always includes the latest weather/calendar context, then fills
        remaining slots with vector-search results from other categories.

        Args:
            household_id: The household scope
            query: The user's voice command text
            max_results: Maximum number of context items to return
            max_chars: Maximum total characters for the formatted output
            similarity_threshold: Minimum cosine similarity (0-1)

        Returns:
            Formatted string for prompt injection, or empty string.
        """
        results: list[str] = []

        # Step 1: Always include latest from priority categories
        priority_contents = self._get_priority_context(household_id)
        results.extend(priority_contents)

        # Step 2: Fill remaining slots with vector search (non-priority)
        remaining = max_results - len(results)
        if remaining > 0:
            vector_results = self._search_vector(
                household_id, query, remaining, similarity_threshold,
                exclude_categories=PRIORITY_CATEGORIES,
            )
            if not vector_results:
                vector_results = self._search_substring(
                    household_id, query, remaining,
                    exclude_categories=PRIORITY_CATEGORIES,
                )
            results.extend(vector_results)

        if not results:
            return ""

        return self._format_results(results, max_chars)

    def _get_priority_context(self, household_id: str) -> list[str]:
        """Get the latest memory from each priority category."""
        contents: list[str] = []
        now = datetime.utcnow()

        for category in PRIORITY_CATEGORIES:
            memory = (
                self.db.query(UserMemory)
                .filter(
                    UserMemory.user_id.is_(None),
                    UserMemory.household_id == household_id,
                    UserMemory.category == category,
                    UserMemory.is_active == True,  # noqa: E712
                )
                .filter(
                    (UserMemory.expires_at == None) | (UserMemory.expires_at > now)  # noqa: E711
                )
                .order_by(UserMemory.updated_at.desc())
                .first()
            )
            if memory:
                contents.append(memory.content)

        return contents

    def _search_vector(
        self,
        household_id: str,
        query: str,
        limit: int,
        threshold: float,
        exclude_categories: list[str] | None = None,
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

            results = []
            for m, _score in matches:
                if exclude_categories and m.category in exclude_categories:
                    continue
                results.append(m.content)

            return results[:limit]

        except Exception as e:
            logger.debug("Agent context vector search failed, will try substring: %s", e)
            return []

    def _search_substring(
        self,
        household_id: str,
        query: str,
        limit: int,
        exclude_categories: list[str] | None = None,
    ) -> list[str]:
        """Fallback to word-overlap substring search."""
        try:
            service = MemoryService(self.db)
            matches = service.search_household_memories_substring(
                household_id=household_id,
                query=query,
                limit=limit,
            )

            results = []
            for m, _score in matches:
                if exclude_categories and m.category in exclude_categories:
                    continue
                results.append(m.content)

            return results[:limit]

        except Exception as e:
            logger.warning("Agent context substring search failed: %s", e)
            return []

    @staticmethod
    def _format_results(contents: list[str], max_chars: int) -> str:
        """Format results as a 'Current context' block for prompt injection."""
        lines: list[str] = []
        header = (
            "LIVE DATA (already fetched — DO NOT call get_weather, get_news, "
            "or get_calendar_events for information listed below. Instead, "
            "use the chat tool and include this data in your response):"
        )
        total_chars = len(header) + 1

        for content in contents:
            line = f"- {content}"
            if total_chars + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 1

        if not lines:
            return ""

        return header + "\n" + "\n".join(lines)
