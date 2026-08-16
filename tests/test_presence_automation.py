"""Presence → smart-lock automation reaction.

``presence.left`` → LOCK, ``presence.seen`` → UNLOCK the household's lock devices,
each gated on a per-direction fail-closed setting. No LLM. These tests pin the
DECISION logic via dependency injection: the kind→(action, gate) mapping, the
fail-closed gate, the no-lock / no-node paths, and the confirmed/total status.
The MQTT actuation transport is injected, not exercised.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

import app.services.presence_automation as pa
from app.services.signal_reaction_registry import ReactionContext


@pytest.fixture(autouse=True)
def _clean_state():
    """The per-(household, user) transition dedup is module-level — reset it around
    every test so one test's latched state can't mask another's actuation."""
    pa.clear_state()
    yield
    pa.clear_state()


def _ctx(kind="presence.left", household_id="hh-1", user_id=7):
    return ReactionContext(
        household_id=household_id,
        node_id=None,  # phone-originated presence carries no node
        user_id=user_id,
        kind=kind,
        facts={"user_id": user_id, "state": "away" if kind == "presence.left" else "home"},
    )


def _actuation(node_id="node-7", entity_id="lock.front_door"):
    return (node_id, {"command_name": "control_device"}, "req-1", entity_id)


def _run(ctx, *, gate=True, plan=None, dispatch=None):
    return asyncio.run(
        pa.react_to_presence(
            ctx,
            gate_check=lambda hh, key: gate,
            plan=plan if plan is not None else (lambda hh, action: [_actuation()]),
            dispatch=dispatch or AsyncMock(return_value=True),
        )
    )


# ── kind routing ────────────────────────────────────────────────────────────
def test_non_presence_kind_is_ignored():
    assert _run(_ctx(kind="appt.upcoming")) == "ignored"


def test_left_plans_the_lock_action():
    seen = {}

    def _plan(hh, action):
        seen["action"] = action
        return [_actuation()]

    assert _run(_ctx(kind="presence.left"), plan=_plan) == "lock:1/1"
    assert seen["action"] == "lock"


def test_seen_plans_the_unlock_action():
    seen = {}

    def _plan(hh, action):
        seen["action"] = action
        return [_actuation()]

    assert _run(_ctx(kind="presence.seen"), plan=_plan) == "unlock:1/1"
    assert seen["action"] == "unlock"


# ── the per-direction gate ──────────────────────────────────────────────────
def test_gate_off_disables_and_does_no_work():
    called = {"plan": False}

    def _plan(hh, action):
        called["plan"] = True
        return [_actuation()]

    dispatch = AsyncMock(return_value=True)
    assert _run(_ctx(), gate=False, plan=_plan, dispatch=dispatch) == "disabled"
    assert called["plan"] is False
    dispatch.assert_not_awaited()


def test_left_checks_the_auto_lock_gate_key():
    seen = {}
    asyncio.run(
        pa.react_to_presence(
            _ctx(kind="presence.left"),
            gate_check=lambda hh, key: seen.setdefault("key", key) and False,
            plan=lambda h, a: [],
            dispatch=AsyncMock(),
        )
    )
    assert seen["key"] == "presence.auto_lock_enabled"


def test_seen_checks_the_auto_unlock_gate_key():
    seen = {}
    asyncio.run(
        pa.react_to_presence(
            _ctx(kind="presence.seen"),
            gate_check=lambda hh, key: seen.setdefault("key", key) and False,
            plan=lambda h, a: [],
            dispatch=AsyncMock(),
        )
    )
    assert seen["key"] == "presence.auto_unlock_enabled"


# ── device / node resolution ────────────────────────────────────────────────
def test_no_lock_device_configured():
    assert _run(_ctx(), plan=lambda h, a: []) == "no_lock_device"


def test_device_exists_but_no_node_reaches_it():
    dispatch = AsyncMock(return_value=True)
    result = _run(_ctx(), plan=lambda h, a: [_actuation(node_id=None)], dispatch=dispatch)
    assert result == "no_node"
    dispatch.assert_not_awaited()


def test_dispatch_receives_the_actuation_fields():
    dispatch = AsyncMock(return_value=True)
    act = ("node-9", {"command_name": "control_device"}, "req-9", "lock.front_door")
    _run(_ctx(kind="presence.left"), plan=lambda h, a: [act], dispatch=dispatch)
    dispatch.assert_awaited_once_with(
        "node-9", {"command_name": "control_device"}, "req-9", "lock.front_door"
    )


def test_partial_confirmation_counts_confirmed_over_total():
    dispatch = AsyncMock(side_effect=[True, False])  # front confirms, back times out
    plan = lambda h, a: [  # noqa: E731
        _actuation(entity_id="lock.front"),
        _actuation(entity_id="lock.back"),
    ]
    assert _run(_ctx(kind="presence.left"), plan=plan, dispatch=dispatch) == "lock:1/2"
    assert dispatch.await_count == 2


# ── transition dedup (heartbeat re-asserts must not re-actuate) ──────────────
def test_repeated_same_state_is_deduped():
    dispatch = AsyncMock(return_value=True)
    plan = lambda h, a: [_actuation()]  # noqa: E731
    # First arrival unlocks; a heartbeat re-assert of "home" must NOT re-unlock.
    assert _run(_ctx(kind="presence.seen"), plan=plan, dispatch=dispatch) == "unlock:1/1"
    assert _run(_ctx(kind="presence.seen"), plan=plan, dispatch=dispatch) == "unchanged"
    assert dispatch.await_count == 1


def test_genuine_transition_after_dedup_actuates():
    dispatch = AsyncMock(return_value=True)
    plan = lambda h, a: [_actuation()]  # noqa: E731
    assert _run(_ctx(kind="presence.left"), plan=plan, dispatch=dispatch) == "lock:1/1"
    # away → home is a real transition → unlock.
    assert _run(_ctx(kind="presence.seen"), plan=plan, dispatch=dispatch) == "unlock:1/1"
    assert dispatch.await_count == 2


def test_no_node_does_not_latch_state():
    # A transient no-node blip must not suppress the real actuation once a node
    # returns — the state only latches after a dispatch.
    with_node = AsyncMock(return_value=True)
    r1 = _run(_ctx(kind="presence.left"), plan=lambda h, a: [_actuation(node_id=None)])
    assert r1 == "no_node"
    r2 = _run(_ctx(kind="presence.left"), plan=lambda h, a: [_actuation()], dispatch=with_node)
    assert r2 == "lock:1/1"
    with_node.assert_awaited_once()


def test_separate_users_do_not_dedupe_each_other():
    dispatch = AsyncMock(return_value=True)
    plan = lambda h, a: [_actuation()]  # noqa: E731
    assert _run(_ctx(kind="presence.left", user_id=7), plan=plan, dispatch=dispatch) == "lock:1/1"
    # A different user's identical "away" is an independent transition.
    assert _run(_ctx(kind="presence.left", user_id=99), plan=plan, dispatch=dispatch) == "lock:1/1"
    assert dispatch.await_count == 2


# ── resilience ──────────────────────────────────────────────────────────────
def test_a_planning_error_never_raises():
    def _plan(hh, action):
        raise RuntimeError("boom")

    assert _run(_ctx(), plan=_plan) == "error"


# ── the fail-closed setting read ────────────────────────────────────────────
def test_gate_read_is_fail_closed(monkeypatch):
    import app.services.settings_service as ss

    class _Broken:
        def get(self, key, household_id=None):
            raise RuntimeError("settings down")

    monkeypatch.setattr(ss, "get_settings_service", lambda: _Broken())
    assert pa._gate_enabled("hh-1", "presence.auto_lock_enabled") is False


def test_gate_read_parses_bool_and_string(monkeypatch):
    import app.services.settings_service as ss

    class _Fixed:
        def __init__(self, v):
            self.v = v

        def get(self, key, household_id=None):
            return self.v

    monkeypatch.setattr(ss, "get_settings_service", lambda: _Fixed(True))
    assert pa._gate_enabled("hh", "k") is True
    monkeypatch.setattr(ss, "get_settings_service", lambda: _Fixed("on"))
    assert pa._gate_enabled("hh", "k") is True
    monkeypatch.setattr(ss, "get_settings_service", lambda: _Fixed("false"))
    assert pa._gate_enabled("hh", "k") is False


# ── registration ────────────────────────────────────────────────────────────
def test_registers_both_presence_edges():
    from app.services import signal_reaction_registry as reg

    reg.clear_reactions()
    try:
        pa.register_presence_automation()
        left = [n for n, _ in reg.reactions_for("presence.left")]
        seen = [n for n, _ in reg.reactions_for("presence.seen")]
        assert "auto_lock" in left
        assert "auto_unlock" in seen
    finally:
        reg.clear_reactions()
