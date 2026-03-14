"""Deep Research Tool — kicks off web research in the background.

Example: "Research the best ergonomic keyboards"
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class DeepResearchTool(IServerTool):
    """Tool for performing deep web research on a topic."""

    @property
    def name(self) -> str:
        return "deep_research"

    @property
    def description(self) -> str:
        return (
            "Research a topic by searching the web, reading top results, and "
            "synthesizing a comprehensive summary. Results are delivered via "
            "push notification and saved to the inbox. Use when the user asks "
            "to research, investigate, or compare something in depth."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research topic or question to investigate",
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "thorough"],
                    "description": "Research depth: quick (3 sources) or thorough (5-8 sources)",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Kick off deep research in the background.

        Returns immediately with an acknowledgment. Results are delivered
        asynchronously via push notification + inbox.
        """
        query: str = kwargs.get("query", "")
        depth: str = kwargs.get("depth", "quick")
        conversation_id: Optional[str] = kwargs.get("conversation_id")

        if not query:
            return {"error": "missing_query", "message": "A research query is required"}

        if not conversation_id:
            return {"error": "no_conversation", "message": "No conversation context available"}

        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return {"error": "no_context", "message": "No node context available"}

        household_id = node_context.get("household_id")
        if not household_id:
            return {"error": "no_household", "message": "No household context available"}

        speaker_user_id = node_context.get("speaker_user_id")

        # Validate depth
        if depth not in ("quick", "thorough"):
            depth = "quick"

        # Spawn background task
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(
                _run_research_background(
                    query=query,
                    depth=depth,
                    household_id=household_id,
                    speaker_user_id=speaker_user_id,
                )
            )
            logger.info(
                "Started deep research: query=%r, depth=%s, household=%s",
                query, depth, household_id,
            )
        except RuntimeError as e:
            logger.error("Failed to start research task: %s", e)
            return {"error": "task_failed", "message": f"Could not start research: {e}"}

        return {
            "status": "accepted",
            "message": f"Research started on: {query}. I'll send you a notification when the results are ready.",
        }


async def _run_research_background(
    query: str,
    depth: str,
    household_id: str,
    speaker_user_id: int | None,
) -> None:
    """Background coroutine that performs the actual research pipeline."""
    try:
        from app.services.deep_research_service import run_research

        await run_research(
            query=query,
            depth=depth,
            household_id=household_id,
            speaker_user_id=speaker_user_id,
        )
    except Exception as e:
        logger.error("Deep research failed for query=%r: %s", query, e, exc_info=True)
        # Best-effort: send failure notification
        try:
            await _send_failure_notification(query, str(e), household_id)
        except Exception as notify_err:
            logger.error("Failed to send research failure notification: %s", notify_err)


async def _send_failure_notification(query: str, error: str, household_id: str) -> None:
    """Send a push notification about a failed research attempt."""
    from app.services.deep_research_service import _send_notification

    await _send_notification(
        household_id=household_id,
        title="Research Failed",
        body=f"Research on \"{query}\" failed: {error}",
        data={"type": "deep_research_failed"},
    )
