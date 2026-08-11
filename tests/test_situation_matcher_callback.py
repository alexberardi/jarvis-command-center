"""The proactive reasoner's callback side: parse the background-model result,
finalize the matches, and emit tap-to-confirm cards — subject to anti-nag.
"""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import app.services.situation_matcher_service as sm

_ACTIONS = [{"command": "add_event", "callback": "create_event",
             "params": [{"name": "title", "required": True},
                        {"name": "start", "required": True},
                        {"name": "idempotency_key", "required": True}],
             "idempotency_param": "idempotency_key", "card_title": "Add?"}]


def _payload(content, status="succeeded"):
    return {
        "status": status,
        "result": {"content": content},
        "metadata": {"household_id": "hh-1", "node_id": "node-7",
                     "bundle": [{"source_key": "cal@x",
                                 "data": {"title": "D", "start": "2026-08-10T09:00:00"}}]},
    }


def _match_content():
    return json.dumps({"matches": [{"command": "add_event", "action": "create_event",
                                    "args": {"title": "D", "start": "2026-08-10T09:00:00"},
                                    "sources": [0]}]})


def _reset():
    sm._fired_keys.clear()
    sm._daily.clear()
    sm._last_fired.clear()


def test_emits_card_on_success():
    _reset()
    async def go():
        with patch("app.services.capability_registry.list_proposable_actions",
                   new=AsyncMock(return_value=_ACTIONS)), \
             patch("app.services.proposal_card.emit_proposal_card", return_value=True) as emit:
            await sm.handle_match_callback(_payload(_match_content()))
        emit.assert_called_once()
        assert emit.call_args.kwargs["command"] == "add_event"
        assert emit.call_args.kwargs["callback"] == "create_event"
    asyncio.run(go())


def test_ignores_failed_status():
    _reset()
    async def go():
        with patch("app.services.proposal_card.emit_proposal_card") as emit:
            await sm.handle_match_callback({"status": "failed"})
        emit.assert_not_called()
    asyncio.run(go())


def test_ignores_unparseable_content_without_raising():
    _reset()
    async def go():
        with patch("app.services.capability_registry.list_proposable_actions",
                   new=AsyncMock(return_value=_ACTIONS)), \
             patch("app.services.proposal_card.emit_proposal_card") as emit:
            await sm.handle_match_callback(_payload("this is not json at all"))
        emit.assert_not_called()
    asyncio.run(go())


def test_anti_nag_dedups_repeat_situation():
    _reset()
    async def go():
        with patch("app.services.capability_registry.list_proposable_actions",
                   new=AsyncMock(return_value=_ACTIONS)), \
             patch("app.services.proposal_card.emit_proposal_card", return_value=True) as emit:
            await sm.handle_match_callback(_payload(_match_content()))
            await sm.handle_match_callback(_payload(_match_content()))   # same situation again
        emit.assert_called_once()          # second is deduped by idempotency key
    asyncio.run(go())
