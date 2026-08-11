"""_emit_directed_proposal upgrade: resolve the named command against the node's
advertisement, then post a real card (Phase 4a) — refuse (no card) if unadvertised.
"""
from unittest.mock import AsyncMock, patch

from app.api import signals as signals_api


def test_directed_emits_card_when_advertised():
    # The node advertises a proposable action for the named command.
    actions = [{"command": "add_event", "callback": "create_event",
                "params": [{"name": "title"}], "card_title": "Add to your calendar?"}]
    with patch("app.api.signals.capability_registry.list_proposable_actions",
               new=AsyncMock(return_value=actions)), \
         patch("app.api.signals.emit_proposal_card", return_value=True) as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id="node-7", command="add_event",
            args={"title": "Dentist"}, source_key="cal@x")
    assert ok is True
    emit.assert_called_once()
    # the callback from the advertisement is what gets proposed
    assert emit.call_args.kwargs["callback"] == "create_event"


def test_directed_falls_back_to_default_node_when_no_node_in_scope():
    # External/app directed signal (no node_id) → resolve against the household's
    # default node.
    actions = [{"command": "add_event", "callback": "create_event",
                "params": [{"name": "title"}], "card_title": "Add?"}]
    with patch("app.services.proposable_action_service._first_household_node_id",
               return_value="default-node"), \
         patch("app.api.signals.capability_registry.list_proposable_actions",
               new=AsyncMock(return_value=actions)) as lp, \
         patch("app.api.signals.emit_proposal_card", return_value=True) as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id=None, command="add_event",
            args={"title": "x"}, source_key="k")
    assert ok is True
    lp.assert_awaited_once_with("default-node")     # resolved against the default node
    emit.assert_called_once()


def test_directed_refused_when_not_advertised():
    with patch("app.api.signals.capability_registry.list_proposable_actions",
               new=AsyncMock(return_value=[])), \
         patch("app.api.signals.emit_proposal_card") as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id="node-7", command="unlock_door",
            args={}, source_key="x")
    assert ok is False
    emit.assert_not_called()
