"""Make Phone Call Tool — plans an AI phone call to a business.

Example: "Call Tony's Pizzeria and book a table for four on Friday at 7"

Flow (phone-calls PRD): spoken ack now → background resolve + plan draft →
confirm card on the initiating user's phone (editable number + details,
TTL'd, single-use) → tap dials via the phone gateway. The call itself never
starts from this tool — user confirmation on a JWT device is the
authorization (decision 8).

ALWAYS offered in the prompt, even when disabled: a call-shaped utterance
with no phone affordance makes the model improvise with whatever tools it
has (observed live 2026-07-19: hallucinated thermostat/TV/lights control
for "Call Tony's Pizzeria"). When ``phone_calls.enabled`` is off, execute()
returns an honest refusal for the model to speak instead.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class MakePhoneCallTool(IServerTool):
    """Tool for placing phone calls to businesses on the user's behalf."""

    @property
    def name(self) -> str:
        return "make_phone_call"

    @property
    def description(self) -> str:
        return (
            "Place a phone call to a business on the user's behalf (book a "
            "reservation, order food for pickup, ask about hours or "
            "availability, schedule an appointment). Use when the user asks "
            "to call, phone, book, or order from a business. The user "
            "confirms the call plan on their phone before anything is dialed."
        )

    @property
    def included_system_prompt_text(self) -> Optional[str]:
        return (
            "make_phone_call: use it whenever the user wants a business "
            "called (reservations, orders, appointments, questions). Never "
            "try to accomplish a phone call with other tools."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "business": {
                    "type": "string",
                    "description": "Name of the business to call, e.g. \"Tony's Pizzeria\"",
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "What the call should accomplish, with every detail "
                        "the user gave (party size, time, order items, name)"
                    ),
                },
            },
            "required": ["business", "goal"],
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        business: str = (kwargs.get("business") or "").strip()
        goal: str = (kwargs.get("goal") or "").strip()
        conversation_id: Optional[str] = kwargs.get("conversation_id")

        if not business or not goal:
            return {
                "error": "missing_params",
                "message": "I need the business name and what the call should accomplish.",
            }
        if not conversation_id:
            return {"error": "no_conversation", "message": "No conversation context available"}

        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return {"error": "no_context", "message": "No node context available"}
        household_id = node_context.get("household_id")
        if not household_id:
            return {"error": "no_household", "message": "No household context available"}

        # The honest refusal (fail-closed, execute-time — the tool is always
        # in the prompt precisely so this branch exists).
        from app.services.phone_call_service import phone_calls_enabled

        if not phone_calls_enabled(household_id):
            logger.info(
                "🚫 make_phone_call refused — phone_calls disabled for household %s",
                household_id,
            )
            return {
                "error": "phone_calls_disabled",
                "message": (
                    "Phone calls aren't set up on this Jarvis. A household "
                    "admin can enable them in Household Settings."
                ),
            }

        speaker_user_id = node_context.get("speaker_user_id")
        if speaker_user_id is None:
            # Identity model (decision 8): the confirm card must land on an
            # identified user's phone. No identified speaker → fail closed,
            # point at the app path.
            return {
                "error": "no_identified_speaker",
                "message": (
                    "I couldn't tell who's asking, so I can't send the call "
                    "plan to your phone. Try again from the Jarvis app."
                ),
            }

        try:
            from app.services.phone_call_service import create_call_plan

            loop = asyncio.get_event_loop()
            loop.create_task(
                create_call_plan(
                    business=business,
                    goal=goal,
                    household_id=household_id,
                    user_id=speaker_user_id,
                )
            )
            logger.info(
                "📞 Call plan started: business=%r household=%s user=%s",
                business, household_id, speaker_user_id,
            )
        except RuntimeError as e:
            logger.error("Failed to start call-plan task: %s", e)
            return {"error": "task_failed", "message": f"Could not plan the call: {e}"}

        return {
            "status": "accepted",
            "message": (
                f"I've sent the call plan for {business} to your phone — "
                "review it and tap Call now to dial."
            ),
        }
