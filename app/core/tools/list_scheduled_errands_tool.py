"""List Scheduled Errands Tool — show the household's scheduled/recurring errands.

Voice: "what errands do I have scheduled?", "list my scheduled errands", "what's
Jarvis going to do later?". Posts a management card to the user's phone listing every
LIVE schedule (one-time + recurring) with a per-errand **Cancel** button, and speaks a
short count. Cancel is a TAP (server_callback_registry → schedule_service) so nothing
is stopped by a mis-heard "cancel the weather one".

Read-only on the voice side: the tool never cancels anything itself — the card's
buttons do, safely and explicitly.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")

_LIST_TASKS: set[asyncio.Task] = set()


class ListScheduledErrandsTool(IServerTool):
    """List the household's scheduled + recurring errands (and offer to cancel each)."""

    @property
    def name(self) -> str:
        return "list_scheduled_errands"

    @property
    def description(self) -> str:
        return (
            "Use this when the user asks what errands/tasks are SCHEDULED or set to REPEAT "
            "— 'what errands do I have scheduled?', 'list my scheduled errands', 'what's set "
            "to run later?', 'what recurring errands do I have?', 'what is Jarvis going to do "
            "later?'. It shows every upcoming one-time and repeating errand and lets the user "
            "cancel any of them. Do NOT use it to CREATE an errand (use schedule_errand/"
            "run_errand for that)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs) -> Dict[str, Any]:
        conversation_id: Optional[str] = kwargs.get("conversation_id")
        if not conversation_id:
            return {"error": "no_conversation", "message": "No conversation context available"}

        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return {"error": "no_context", "message": "No node context available"}
        household_id = node_context.get("household_id")
        if not household_id:
            return {"error": "no_context", "message": "I couldn't tell which household to check."}
        user_id = node_context.get("speaker_user_id")

        from app.services.schedule_service import list_schedules, post_schedule_list_card

        schedules = list_schedules(household_id)  # fast local query — for the count/ack
        if not schedules:
            return {"status": "ok", "message": "You don't have any errands scheduled right now."}

        # Post the management card off the request path — it's a network POST to the
        # notifications service; don't block the voice loop on it.
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(
                asyncio.to_thread(post_schedule_list_card, household_id, user_id)
            )
            _LIST_TASKS.add(task)
            task.add_done_callback(_LIST_TASKS.discard)
        except RuntimeError:
            # No running loop (e.g. a direct call) — post inline.
            post_schedule_list_card(household_id, user_id)

        n = len(schedules)
        if n == 1:
            msg = ("You have one errand scheduled. I've put it on your phone, where you "
                   "can cancel it.")
        else:
            msg = (f"You have {n} errands scheduled. I've sent the full list to your phone, "
                   "where you can cancel any of them.")
        return {"status": "ok", "message": msg}
