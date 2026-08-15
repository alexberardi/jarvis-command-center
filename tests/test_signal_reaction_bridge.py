"""SignalReactionBridge — appt.upcoming → DETERMINISTIC leave-by CARD.

The bridge computes drive time on the node (injected here), computes the absolute
leave-by instant in code, and proposes a reminder.set_at card (emitter injected).
No LLM, no workflow. These tests pin the DECISION logic: the gate, dedup,
suppression, the strict-drive-time skip, the absolute due_at math, and the card
params (esp. that idempotency_key rides IN params so the confirm validates).
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

import app.services.signal_reaction_bridge as bridge
from app.services.signal_reaction_registry import ReactionContext

_FAR_START = datetime(2999, 1, 1, 12, 0, tzinfo=timezone.utc)


def _ctx(facts=None, node_id="node-7", user_id=7):
    return ReactionContext(
        household_id="hh-1", node_id=node_id, user_id=user_id, kind="appt.upcoming",
        facts=facts if facts is not None else _facts(),
    )


def _facts(**over):
    f = {
        "title": "Dentist",
        "start_iso": _FAR_START.isoformat(),
        "start_display": "12:00 PM",
        "location": "123 Main St",
        "event_id": "evt-1",
    }
    f.update(over)
    return f


def _drive(**over):
    d = {
        "success": True,
        "duration_minutes": 10,
        "duration_text": "10 mins",
        "destination": "123 Main St, Testville CA 90000",
    }
    d.update(over)
    return d


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Every test starts with a clean dedup set, no suppression, and no DB write."""
    bridge._fired.clear()
    monkeypatch.setattr(bridge, "_is_suppressed", lambda hh, uid: False)
    monkeypatch.setattr(bridge, "_save_leave_by_signal", lambda **kw: None)


def _run(
    facts=None,
    node_id="node-7",
    user_id=7,
    enabled=True,
    names=None,
    dispatch=None,
    emit_card=None,
):
    return asyncio.run(
        bridge.react_to_appt_upcoming(
            _ctx(facts, node_id, user_id),
            enabled_check=lambda h: enabled,
            menu_fetch=AsyncMock(
                return_value=names if names is not None else {"get_drive_time", "reminder"}
            ),
            dispatch=dispatch or AsyncMock(return_value=_drive()),
            emit_card=emit_card or AsyncMock(return_value=True),
        )
    )


# ── the guards ────────────────────────────────────────────────────────────────
def test_disabled_household_is_a_noop():
    dispatch = AsyncMock()
    assert _run(enabled=False, dispatch=dispatch) == "disabled"
    dispatch.assert_not_awaited()


def test_missing_node_skipped():
    assert _run(node_id=None) == "no_node"


def test_missing_location_skipped():
    assert _run(facts=_facts(location=None)) == "no_location"


def test_missing_start_skipped():
    assert _run(facts=_facts(start_iso=None)) == "no_start"


def test_missing_user_skipped():
    assert _run(user_id=None) == "no_user"


def test_suppressed_household_is_a_noop(monkeypatch):
    monkeypatch.setattr(bridge, "_is_suppressed", lambda hh, uid: True)
    dispatch = AsyncMock()
    assert _run(dispatch=dispatch) == "suppressed"
    dispatch.assert_not_awaited()


def test_node_without_drive_time_skipped():
    dispatch = AsyncMock()
    assert _run(names={"reminder"}, dispatch=dispatch) == "no_drive_time"
    dispatch.assert_not_awaited()


# ── drive-time outcomes ─────────────────────────────────────────────────────────
def test_generic_location_strict_skip_yields_no_route():
    # strict resolution on the node returns success=False for a generic location
    dispatch = AsyncMock(return_value={"success": False, "reason": "generic_location"})
    emit = AsyncMock()
    assert _run(dispatch=dispatch, emit_card=emit) == "no_route"
    emit.assert_not_awaited()


def test_non_numeric_drive_minutes_yields_no_route():
    dispatch = AsyncMock(return_value=_drive(duration_minutes=None))
    assert _run(dispatch=dispatch) == "no_route"


def test_departure_already_passed_is_a_noop():
    past = datetime.now(timezone.utc) + timedelta(minutes=5)  # only 5 min out
    dispatch = AsyncMock(return_value=_drive(duration_minutes=30))  # leave 35 min ago
    emit = AsyncMock()
    out = _run(facts=_facts(start_iso=past.isoformat()), dispatch=dispatch, emit_card=emit)
    assert out == "departure_passed"
    emit.assert_not_awaited()


# ── the happy path ──────────────────────────────────────────────────────────────
def test_proposes_card_with_absolute_due_at_and_idempotency_in_params():
    emit = AsyncMock(return_value=True)
    out = _run(emit_card=emit)
    assert out == "proposed"
    kw = emit.await_args.kwargs
    # absolute due_at = event_start − (drive 10 + buffer 5) = start − 15 min
    expected_due = (_FAR_START - timedelta(minutes=15)).isoformat()
    assert kw["due_at_iso"] == expected_due
    assert kw["title"] == "Dentist"
    assert kw["user_id"] == 7
    # geocoded address from the node result rides through to the card
    assert kw["address"] == "123 Main St, Testville CA 90000"


def test_emit_card_builds_setat_params_including_idempotency_key(monkeypatch):
    # Drive the REAL _emit_leave_by_card to prove idempotency_key rides IN params
    # (so the confirm-time validate_against_params passes) + the node is resolved.
    captured = {}

    async def fake_resolve_node(hh, cmd, cb):
        return "node-7"

    async def fake_resolve_action(node, cmd, cb):
        return {"callback": "set_at", "card_title": "Set a reminder?",
                "confirm_label": "Set reminder", "params": [
                    {"name": "text", "required": True},
                    {"name": "due_at_iso", "required": True},
                    {"name": "idempotency_key", "required": True}]}

    def fake_emit(**kw):
        captured.update(kw)
        return True

    monkeypatch.setattr(
        "app.services.proposable_action_service.resolve_household_node_for_command",
        fake_resolve_node,
    )
    monkeypatch.setattr(
        "app.services.capability_registry.resolve_proposable_action", fake_resolve_action
    )
    monkeypatch.setattr("app.services.proposal_card.emit_proposal_card", fake_emit)

    ok = asyncio.run(
        bridge._emit_leave_by_card(
            household_id="hh-1", user_id=7, event_id="evt-1", title="Dentist",
            due_at_iso="2999-01-01T11:45:00+00:00", address="123 Main St",
            start_display="12:00 PM", duration_text="10 mins",
        )
    )
    assert ok is True
    assert captured["command"] == "reminder"
    assert captured["callback"] == "set_at"
    # idempotency_key MUST be present in params (required declared param) AND as the
    # card's idempotency_key — else the confirm dispatcher rejects it.
    assert captured["params"]["idempotency_key"] == "leaveby:evt-1"
    assert captured["params"]["due_at_iso"] == "2999-01-01T11:45:00+00:00"
    assert captured["params"]["text"] == "Leave for Dentist"
    assert captured["idempotency_key"] == "leaveby:evt-1"
    # stable category source powers "Never suggest this" across all leave-by cards
    assert captured["source"] == "leaveby"


def test_emit_card_refused_when_no_node_advertises_reminder():
    emit = AsyncMock(return_value=False)  # emitter reports no advertised node
    assert _run(emit_card=emit) == "not_advertised"


def test_dedup_second_reaction_for_same_event_is_a_noop():
    dispatch = AsyncMock(return_value=_drive())
    emit = AsyncMock(return_value=True)

    async def go():
        common = dict(
            enabled_check=lambda h: True,
            menu_fetch=AsyncMock(return_value={"get_drive_time", "reminder"}),
            dispatch=dispatch,
            emit_card=emit,
        )
        first = await bridge.react_to_appt_upcoming(_ctx(), **common)
        second = await bridge.react_to_appt_upcoming(_ctx(), **common)
        return first, second

    first, second = asyncio.run(go())
    assert first == "proposed"
    assert second == "duplicate"
    assert emit.await_count == 1


def test_register_leave_by_reaction_wires_appt_upcoming():
    # The reaction self-registers for its kind on the generic registry — ingest and
    # dispatch name no kind. (Generic dispatch itself is covered in the registry test.)
    from app.services import signal_reaction_registry as reg
    reg.clear_reactions()
    bridge.register_leave_by_reaction()
    handlers = reg.reactions_for("appt.upcoming")
    assert [name for name, _ in handlers] == ["leave_by"]
    assert reg.reactions_for("presence.seen") == []  # no reaction for other kinds
