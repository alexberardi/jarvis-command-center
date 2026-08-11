"""Unit tests for match_situation — the bundle generalization of match_proposals.

match_situation takes a BUNDLE of Signals ({source_key, data}) instead of one
opaque dict, and returns which Signals drove each match (contributing
source_keys) plus an idempotency key scoped to ONLY those contributing items.
match_proposals becomes a one-line adapter, so the shipped email→calendar pilot
(tests/test_proposal_matcher.py) must keep passing unchanged.
"""
import asyncio
import json
from unittest.mock import AsyncMock

from app.services import proposal_matcher as pm


# _fetch_report / _llm mirror tests/test_proposal_matcher.py verbatim.
def _fetch_report():
    return AsyncMock(return_value={"available_commands": [
        {"command_name": "add_event", "proposable_actions": [
            {"callback": "create_event",
             "params": [{"name": "title", "required": True},
                        {"name": "start", "required": True},
                        {"name": "end", "required": False},
                        {"name": "idempotency_key", "required": True}],
             "idempotency_param": "idempotency_key",
             "card_title": "Add to your calendar?"}]}]})


def _llm(matches):
    client = AsyncMock()
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": json.dumps({"matches": matches})}}]}
    )
    return client


def _bundle():
    return [
        {"source_key": "cal@x", "data": {"title": "Dentist", "start": "2026-08-10T09:00:00"}},
        {"source_key": "weather@y", "data": {"summary": "sunny"}},
    ]


def _appt_match(sources=None):
    m = {"command": "add_event", "action": "create_event",
         "args": {"title": "Dentist", "start": "2026-08-10T09:00:00"}}
    if sources is not None:
        m["sources"] = sources
    return m


def test_returns_contributing_source_keys():
    out = asyncio.run(pm.match_situation(
        bundle=_bundle(), node_id="n", llm_client=_llm([_appt_match(sources=[0])]),
        fetch=_fetch_report()))
    assert len(out) == 1
    assert out[0]["source_keys"] == ["cal@x"]          # only the cited item
    assert "weather@y" not in out[0]["source_keys"]


def test_idempotency_scoped_to_contributing_items():
    # same contributing item (#0) + different noise (#1) → identical idem key
    b1 = [{"source_key": "cal@x", "data": {"title": "Dentist", "start": "2026-08-10T09:00:00"}},
          {"source_key": "n1", "data": {"noise": "a"}}]
    b2 = [{"source_key": "cal@x", "data": {"title": "Dentist", "start": "2026-08-10T09:00:00"}},
          {"source_key": "n2", "data": {"noise": "b"}}]
    o1 = asyncio.run(pm.match_situation(bundle=b1, node_id="n",
                                        llm_client=_llm([_appt_match(sources=[0])]), fetch=_fetch_report()))
    o2 = asyncio.run(pm.match_situation(bundle=b2, node_id="n",
                                        llm_client=_llm([_appt_match(sources=[0])]), fetch=_fetch_report()))
    assert o1[0]["idempotency_key"] == o2[0]["idempotency_key"]


def test_out_of_range_source_index_dropped():
    out = asyncio.run(pm.match_situation(
        bundle=_bundle(), node_id="n", llm_client=_llm([_appt_match(sources=[0, 5])]),
        fetch=_fetch_report()))
    assert out[0]["source_keys"] == ["cal@x"]          # #5 is out of range → dropped


def test_injects_idempotency_and_lists_indices_omits_injected_param():
    llm = _llm([_appt_match(sources=[0])])
    out = asyncio.run(pm.match_situation(bundle=_bundle(), node_id="n", llm_client=llm, fetch=_fetch_report()))
    m = out[0]
    assert m["idempotency_key"].startswith("match:")
    assert m["args"]["idempotency_key"] == m["idempotency_key"]
    prompt = llm.chat_completion.call_args.kwargs["messages"][0]["content"]
    assert "idempotency_key" not in prompt              # injected param never offered to the LLM
    assert "#0" in prompt and "#1" in prompt            # indices listed so the LLM can cite sources


def test_hallucinated_action_dropped():
    out = asyncio.run(pm.match_situation(
        bundle=_bundle(), node_id="n",
        llm_client=_llm([{"command": "wire_money", "action": "send", "args": {}, "sources": [0]}]),
        fetch=_fetch_report()))
    assert out == []


def test_invalid_args_dropped():
    out = asyncio.run(pm.match_situation(
        bundle=_bundle(), node_id="n",
        llm_client=_llm([{"command": "add_event", "action": "create_event",
                          "args": {"title": "x"}, "sources": [0]}]),   # missing required start
        fetch=_fetch_report()))
    assert out == []


def test_no_advertised_actions_empty():
    out = asyncio.run(pm.match_situation(
        bundle=_bundle(), node_id="n", llm_client=_llm([]),
        fetch=AsyncMock(return_value={"available_commands": []})))
    assert out == []


def test_empty_matches_empty():
    out = asyncio.run(pm.match_situation(
        bundle=_bundle(), node_id="n", llm_client=_llm([]), fetch=_fetch_report()))
    assert out == []


def test_match_proposals_adapter_preserves_behavior():
    # match_proposals (no 'sources' cited, single logical item) still works.
    out = asyncio.run(pm.match_proposals(
        data={"title": "Dentist", "start": "2026-08-10T09:00:00"},
        node_id="n", llm_client=_llm([_appt_match()]), fetch=_fetch_report()))
    assert len(out) == 1
    assert out[0]["command"] == "add_event"
    assert out[0]["idempotency_key"].startswith("match:")
    assert out[0]["args"]["idempotency_key"] == out[0]["idempotency_key"]
