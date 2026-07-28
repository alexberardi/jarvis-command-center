"""Run Errand Tool — drafts a background errand plan from a spoken goal.

Example: "Run an errand: check the weather and remind me to buy milk tomorrow"

Voice-inline entry for the Errand Runner (errand-runner PRD §2-§3 POC): the
spoken goal → a background planner draft → a plan card on the user's phone with
Run/Cancel. Nothing runs until the user taps Run. The errand itself is HEADLESS
— it reports via the completion card, it is never spoken back on the node.

Mirrors make_phone_call_tool: derive household/node/speaker from node_context,
fire the plan-draft as a detached task, and return a natural-language spoken ack.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")

# asyncio only weakly references fire-and-forget tasks; hold a strong ref so a
# detached draft isn't garbage-collected mid-flight.
_ERRAND_TASKS: set[asyncio.Task] = set()


class RunErrandTool(IServerTool):
    """Draft a background 'errand' plan card from a spoken goal."""

    @property
    def name(self) -> str:
        return "run_errand"

    @property
    def description(self) -> str:
        return (
            "Draft a multi-step 'errand' to run in the background on the user's "
            "behalf, then show them a plan card to review and approve. Use when "
            "the user asks to 'run an errand', or to 'take care of' / 'handle' a "
            "multi-step background task (e.g. 'run an errand: check the weather "
            "and remind me to buy milk tomorrow'). Nothing runs until the user "
            "taps Run on the plan card. Do NOT use for a single immediate command "
            "the user wants done right now — use the normal command for that."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "What the errand should accomplish, in the user's own "
                        "words and with every detail they gave."
                    ),
                },
            },
            "required": ["goal"],
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        goal: str = (kwargs.get("goal") or "").strip()
        conversation_id: Optional[str] = kwargs.get("conversation_id")

        if not goal:
            return {"error": "missing_goal", "message": "I need to know what the errand should do."}
        if not conversation_id:
            return {"error": "no_conversation", "message": "No conversation context available"}

        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return {"error": "no_context", "message": "No node context available"}
        household_id = node_context.get("household_id")
        node_id = node_context.get("node_id")
        if not household_id or not node_id:
            return {
                "error": "no_context",
                "message": "I couldn't tell which node to run this on.",
            }
        # An errand plan card is fine as a household broadcast, so an unidentified
        # speaker is OK — the tap is authorized on a JWT device either way.
        speaker_user_id = node_context.get("speaker_user_id")

        try:
            from app.services.errand_service import draft_errand_plan_detached

            loop = asyncio.get_event_loop()
            task = loop.create_task(
                draft_errand_plan_detached(
                    household_id=household_id,
                    node_id=node_id,
                    goal=goal,
                    user_id=speaker_user_id,
                )
            )
            _ERRAND_TASKS.add(task)
            task.add_done_callback(_ERRAND_TASKS.discard)
            logger.info(
                "🗒️ Errand draft started: household=%s node=%s goal=%r",
                household_id, node_id, goal,
            )
        except RuntimeError as e:
            logger.error("Failed to start errand-draft task: %s", e)
            return {"error": "task_failed", "message": f"I couldn't start planning that errand: {e}"}

        return {
            "status": "accepted",
            "message": (
                "On it — I'm putting together a plan for that errand and I'll send "
                "it to your phone to review. Tap Run when you're ready."
            ),
        }
