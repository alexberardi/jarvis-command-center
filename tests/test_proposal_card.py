"""Unit tests for the server-side proposable-card emitter.

emit_proposal_card mirrors the SDK inbox.propose_action card shape server-side so
command-center can raise a tap-to-confirm card that routes a Confirm to the
shipped jarvis.proposable_action.execute dispatcher.
"""
from unittest.mock import patch

from app.services import proposal_card


def _emit_and_capture(**over):
    kwargs = dict(
        household_id="hh-1", node_id="node-7", command="add_event",
        callback="create_event", params={"title": "Dentist"},
        idempotency_key="match:abc", card_title="Add to your calendar?",
    )
    kwargs.update(over)
    with patch.object(proposal_card, "post_inbox_item_sync", return_value="item-1") as post:
        ok = proposal_card.emit_proposal_card(**kwargs)
    return ok, post


def test_emit_posts_confirm_card_routing_to_dispatcher():
    ok, post = _emit_and_capture()
    assert ok is True
    post.assert_called_once()
    elems = post.call_args.kwargs["metadata"]["interactive_elements"]
    confirm = next(e for e in elems if e["kind"] == "confirm")
    assert confirm["command"] == "jarvis.proposable_action"
    assert confirm["callback"] == "execute"
    act = confirm["data"]["_action"]
    assert act["target_command"] == "add_event"
    assert act["target_callback"] == "create_event"
    assert act["idempotency_key"] == "match:abc"
    assert act["node_id"] == "node-7"
    # declared params ride at the top level, stringified (so mobile edits merge)
    assert confirm["data"]["title"] == "Dentist"
    # a Dismiss element is always present
    assert any(e["kind"] == "dismiss" for e in elems)


def test_no_suppress_element_without_source():
    _, post = _emit_and_capture()
    elems = post.call_args.kwargs["metadata"]["interactive_elements"]
    assert not any(e["kind"] == "suppress" for e in elems)


def test_suppress_element_when_source_given():
    _, post = _emit_and_capture(source="presence:alex", descriptor="d")
    elems = post.call_args.kwargs["metadata"]["interactive_elements"]
    sup = next(e for e in elems if e["kind"] == "suppress")
    assert sup["callback"] == "suppress"
    assert sup["data"]["source"] == "presence:alex"


def test_returns_false_when_post_raises():
    with patch.object(proposal_card, "post_inbox_item_sync", side_effect=RuntimeError("boom")):
        ok = proposal_card.emit_proposal_card(
            household_id="hh-1", node_id=None, command="c", callback="cb",
            params={}, idempotency_key="k", card_title="t")
    assert ok is False


def test_returns_false_when_post_returns_none():
    with patch.object(proposal_card, "post_inbox_item_sync", return_value=None):
        ok = proposal_card.emit_proposal_card(
            household_id="hh-1", node_id=None, command="c", callback="cb",
            params={}, idempotency_key="k", card_title="t")
    assert ok is False
