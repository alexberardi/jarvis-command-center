"""Unit tests for the open-mode capability matcher (data → which command?)."""

import asyncio
import json
from unittest.mock import AsyncMock

from app.services import proposal_matcher as pm


def _fetch_report():
    return AsyncMock(
        return_value={
            "available_commands": [
                {
                    "command_name": "add_event",
                    "proposable_actions": [
                        {
                            "callback": "create_event",
                            "params": [
                                {"name": "title", "required": True},
                                {"name": "start", "required": True},
                                {"name": "end", "required": False},
                                {"name": "idempotency_key", "required": True},
                            ],
                            "idempotency_param": "idempotency_key",
                            "card_title": "Add to your calendar?",
                        }
                    ],
                }
            ]
        }
    )


def _llm(matches):
    client = AsyncMock()
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": json.dumps({"matches": matches})}}]}
    )
    return client


def test_matches_data_to_advertised_action_and_injects_idempotency():
    llm = _llm([{"command": "add_event", "action": "create_event",
                 "args": {"title": "Dentist", "start": "2026-08-10T09:00:00"}}])
    out = asyncio.run(pm.match_proposals(
        data={"title": "Dentist", "start": "2026-08-10T09:00:00"},
        node_id="node-7", llm_client=llm, fetch=_fetch_report(),
    ))
    assert len(out) == 1
    m = out[0]
    assert m["command"] == "add_event" and m["action"] == "create_event"
    assert m["args"]["title"] == "Dentist"
    # System injects the idempotency key (not asked of the LLM); it's stable.
    assert m["idempotency_key"].startswith("match:")
    assert m["args"]["idempotency_key"] == m["idempotency_key"]
    # The LLM prompt never lists the injected param as fillable.
    prompt = llm.chat_completion.call_args.kwargs["messages"][0]["content"]
    assert "idempotency_key" not in prompt


def test_no_advertised_actions_returns_empty():
    out = asyncio.run(pm.match_proposals(
        data={"x": 1}, node_id="n", llm_client=_llm([]),
        fetch=AsyncMock(return_value={"available_commands": []}),
    ))
    assert out == []


def test_hallucinated_action_is_dropped():
    llm = _llm([{"command": "wire_money", "action": "send", "args": {}}])
    out = asyncio.run(pm.match_proposals(
        data={"amount": 1000}, node_id="node-7", llm_client=llm, fetch=_fetch_report(),
    ))
    assert out == []  # not advertised → refused


def test_invalid_args_dropped():
    # LLM omits the required non-injected `start`.
    llm = _llm([{"command": "add_event", "action": "create_event", "args": {"title": "x"}}])
    out = asyncio.run(pm.match_proposals(
        data={"title": "x"}, node_id="node-7", llm_client=llm, fetch=_fetch_report(),
    ))
    assert out == []


def test_empty_matches_returns_empty():
    out = asyncio.run(pm.match_proposals(
        data={"nothing": True}, node_id="node-7", llm_client=_llm([]), fetch=_fetch_report(),
    ))
    assert out == []
