"""_emit_directed_proposal: resolve the named command against a node's advertisement,
then post a real card (Phase 4a) — refuse (no card) if unadvertised. An explicit
scope.node_id is honored only if it belongs to the household (client-supplied,
unvalidated upstream); otherwise it falls through to the command-aware resolver.
"""
from unittest.mock import AsyncMock, patch

from app.api import signals as signals_api

# Where the household-membership check reads its node list from.
HH_NODES = "app.services.proposable_action_service._household_node_ids_by_recency"


def test_directed_emits_card_when_advertised():
    # The node advertises a proposable action for the named command.
    actions = [{"command": "add_event", "callback": "create_event",
                "params": [{"name": "title"}], "card_title": "Add to your calendar?"}]
    with patch(HH_NODES, return_value=["node-7"]), \
         patch("app.api.signals.capability_registry.list_proposable_actions",
               new=AsyncMock(return_value=actions)), \
         patch("app.api.signals.emit_proposal_card", return_value=True) as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id="node-7", command="add_event",
            args={"title": "Dentist"}, source_key="cal@x")
    assert ok is True
    emit.assert_called_once()
    # the callback from the advertisement is what gets proposed
    assert emit.call_args.kwargs["callback"] == "create_event"


def test_directed_falls_back_to_node_that_advertises_when_no_node_in_scope():
    # External/app directed signal (no node_id) → resolve against a household node
    # that actually ADVERTISES the command (multi-node correct), not just the most
    # recent one.
    actions = [{"command": "add_event", "callback": "create_event",
                "params": [{"name": "title"}], "card_title": "Add?"}]
    with patch("app.services.proposable_action_service.resolve_household_node_for_command",
               new=AsyncMock(return_value="calendar-node")), \
         patch("app.api.signals.capability_registry.list_proposable_actions",
               new=AsyncMock(return_value=actions)) as lp, \
         patch("app.api.signals.emit_proposal_card", return_value=True) as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id=None, command="add_event",
            args={"title": "x"}, source_key="k")
    assert ok is True
    assert lp.await_args.args == ("calendar-node",)   # listed against the node that advertises it
    emit.assert_called_once()
    assert emit.call_args.kwargs["node_id"] == "calendar-node"


def test_directed_foreign_scope_node_id_is_not_honored():
    # A scope.node_id that isn't in the household must NOT be contacted — it falls
    # through to the command-aware resolver instead (cross-household / bogus-id guard).
    resolver = AsyncMock(return_value=None)
    lp = AsyncMock(return_value=[])
    with patch(HH_NODES, return_value=["node-7"]), \
         patch("app.services.proposable_action_service.resolve_household_node_for_command",
               new=resolver), \
         patch("app.api.signals.capability_registry.list_proposable_actions", new=lp), \
         patch("app.api.signals.emit_proposal_card") as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id="attacker-node", command="add_event",
            args={"title": "x"}, source_key="k")
    assert ok is False
    resolver.assert_awaited_once_with("hh-1", "add_event", None)   # fell through to the resolver
    # the foreign node was never contacted
    for call in lp.await_args_list:
        assert call.args[0] != "attacker-node"
    emit.assert_not_called()


def test_directed_refused_when_no_household_node_advertises():
    # No node in scope AND no household node advertises the command → refuse, no card.
    with patch("app.services.proposable_action_service.resolve_household_node_for_command",
               new=AsyncMock(return_value=None)), \
         patch("app.api.signals.emit_proposal_card") as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id=None, command="add_event",
            args={"title": "x"}, source_key="k")
    assert ok is False
    emit.assert_not_called()


def test_directed_dotted_command_callback_selects_that_callback():
    # "command.callback" form must resolve to that exact callback, not the command's
    # first proposable action.
    actions = [{"command": "add_event", "callback": "update_event", "params": [], "card_title": "Update?"},
               {"command": "add_event", "callback": "create_event",
                "params": [{"name": "title"}], "card_title": "Add?"}]
    with patch(HH_NODES, return_value=["node-7"]), \
         patch("app.api.signals.capability_registry.list_proposable_actions",
               new=AsyncMock(return_value=actions)), \
         patch("app.api.signals.emit_proposal_card", return_value=True) as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id="node-7",
            command="add_event.create_event", args={"title": "x"}, source_key="k")
    assert ok is True
    assert emit.call_args.kwargs["command"] == "add_event"
    assert emit.call_args.kwargs["callback"] == "create_event"   # the named callback, not update_event


def test_directed_fallback_threads_parsed_command_and_callback_to_resolver():
    # No node in scope + dotted form → the parsed (command, callback) is what the
    # command-aware resolver is asked to find.
    actions = [{"command": "add_event", "callback": "create_event",
                "params": [{"name": "title"}], "card_title": "Add?"}]
    resolver = AsyncMock(return_value="calendar-node")
    with patch("app.services.proposable_action_service.resolve_household_node_for_command",
               new=resolver), \
         patch("app.api.signals.capability_registry.list_proposable_actions",
               new=AsyncMock(return_value=actions)), \
         patch("app.api.signals.emit_proposal_card", return_value=True):
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id=None,
            command="add_event.create_event", args={"title": "x"}, source_key="k")
    assert ok is True
    resolver.assert_awaited_once_with("hh-1", "add_event", "create_event")


def test_directed_refused_when_not_advertised():
    with patch(HH_NODES, return_value=["node-7"]), \
         patch("app.api.signals.capability_registry.list_proposable_actions",
               new=AsyncMock(return_value=[])), \
         patch("app.api.signals.emit_proposal_card") as emit:
        ok = signals_api._emit_directed_proposal(
            db=None, household_id="hh-1", node_id="node-7", command="unlock_door",
            args={}, source_key="x")
    assert ok is False
    emit.assert_not_called()
