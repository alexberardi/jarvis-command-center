"""Deep Research Tool — kicks off web research in the background.

Example: "Research the best ergonomic keyboards"
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


def _web_search_enabled(household_id: str) -> bool:
    """Fail-closed check of the household ``web_search.enabled`` setting.

    Defense-in-depth: the warmup tool whitelist already excludes deep_research
    when web search is disabled, but this re-check at execution time closes the
    gap where a model hallucinates the call for a tool it was never offered.
    Any error → disabled (no outbound egress).
    """
    try:
        from app.services.settings_service import get_settings_service

        val = get_settings_service().get(
            "web_search.enabled", household_id=str(household_id)
        )
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)
    except Exception as e:
        logger.warning(
            "deep_research web_search gate check failed, defaulting to DISABLED: %s", e
        )
        return False


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

        # Defense-in-depth: never egress for a household that opted out, even
        # if this tool was somehow invoked without being in the prompt.
        if not _web_search_enabled(household_id):
            logger.info(
                "🚫 deep_research blocked — web_search disabled for household %s",
                household_id,
            )
            return {
                "error": "web_search_disabled",
                "message": "Web search is disabled for this household.",
            }

        speaker_user_id = node_context.get("speaker_user_id")

        if depth not in ("quick", "thorough"):
            depth = "quick"

        # Spawn background task — search + scrape + enqueue LLM job
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
    """Background coroutine: search, scrape, enqueue LLM summarization."""
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
            from app.services.deep_research_service import _send_notification

            await _send_notification(
                household_id=household_id,
                title="Research Failed",
                body=f"Research on \"{query}\" failed: {e}",
                data={"type": "deep_research_failed"},
            )
        except Exception as notify_err:
            logger.error("Failed to send research failure notification: %s", notify_err)
