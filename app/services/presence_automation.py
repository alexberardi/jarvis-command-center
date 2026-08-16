"""Presence → smart-lock automation (a deterministic signal reaction).

The phone is a presence producer on the Signal Bus; a geofence crossing emits a
presence signal, and this reaction actuates the household's lock-domain devices:

    presence.left  → LOCK    (gated on ``presence.auto_lock_enabled``)
    presence.seen  → UNLOCK  (gated on ``presence.auto_unlock_enabled``)

No LLM is in the loop — the presence→lock/unlock mapping is 1:1. Actuation reuses
the EXACT path the smart-home device-control endpoint uses: the ``action`` MQTT
verb carrying the device's addressing context (:func:`build_control_details`) to
a protocol-matched node, whose ``control_device.handle_action`` calls Home
Assistant ``lock.lock`` / ``lock.unlock``.

Both gates default OFF (fail-safe): deploying this reaction actuates nothing
until a household explicitly opts in. Auto-UNLOCK is a SEPARATE gate from
auto-LOCK because it physically opens a door on a geofence event — a household
can enable auto-lock (fail-safe) without ever enabling auto-unlock (risky under a
GPS/geofence false-positive).

Registered on the generic :mod:`app.services.signal_reaction_registry`; this
module owns only the reaction's logic and names its trigger kinds in one line at
the bottom (no ingest changes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Callable
from uuid import uuid4

from app.services.signal_reaction_registry import ReactionContext, register_signal_reaction

logger = logging.getLogger("uvicorn")

# presence kind → (control action, the per-direction gate that must be enabled).
_KIND_ACTION: dict[str, tuple[str, str]] = {
    "presence.left": ("lock", "presence.auto_lock_enabled"),
    "presence.seen": ("unlock", "presence.auto_unlock_enabled"),
}

# Best-effort wait for the node's control result — publishing already actuated the
# lock, so this only confirms success (for logging) and cleans up the result file.
_RESULT_WAIT_S = 8.0
_RESULT_POLL_S = 0.2

# Last-actuated presence state per (household, user) — so only a genuine transition
# actuates. The foreground presence sampler re-asserts the SAME state on a ~90-min
# heartbeat; without this, an arrival heartbeat would UNLOCK a door the user had
# manually locked while home (locking an already-locked door is a harmless no-op,
# but re-unlocking is not). Keyed per-user (each phone's transitions are tracked
# independently); a multi-user household still actuates on any member's transition —
# richer "everyone out" aggregation is left to the natural-language builder. Resets
# on restart: the first post-restart signal actuates once (a re-lock/re-unlock,
# almost always a no-op).
_last_state: dict[tuple[str, str], str] = {}


def clear_state() -> None:
    """Test hook — drop the per-(household, user) dedup memory."""
    _last_state.clear()

# One planned actuation: the node to run it, the ready MQTT payload, the request id
# (to poll the result file), and the entity id (for logging). node_id is None when
# no active node can reach the device's protocol.
Actuation = tuple[str | None, dict[str, Any], str, str]


def _gate_enabled(household_id: str, key: str) -> bool:
    """Fail-closed per-household read of a bool setting (default False) — mirrors the
    proposals gate. Any settings error → disabled: never actuate a door on a broken
    read."""
    try:
        from app.services.settings_service import get_settings_service

        value = get_settings_service().get(key, household_id=str(household_id))
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return False
    except Exception:  # noqa: BLE001 — fail closed
        return False


def _plan_actuations(household_id: str, action: str) -> list[Actuation]:
    """Build one actuation per controllable lock device in the household. All DB /
    node-pick work happens here (synchronously, session open, no awaits); the async
    dispatch below consumes plain tuples. Empty list ⇒ no lock configured."""
    from app.api.smart_home import _pick_node_for_protocol, build_control_details
    from app.db import get_session_local
    from app.models import Device

    db = get_session_local()()
    try:
        rows = (
            db.query(Device)
            .filter(
                Device.household_id == household_id,
                Device.domain == "lock",
                Device.is_active.is_(True),
                Device.is_controllable.is_(True),
            )
            .all()
        )
        plan: list[Actuation] = []
        for dev in rows:
            request_id = str(uuid4())
            node = _pick_node_for_protocol(db, household_id, dev.protocol)
            details = build_control_details(dev, action, request_id)
            plan.append((node.node_id if node else None, details, request_id, dev.entity_id))
        return plan
    finally:
        db.close()


async def _dispatch(node_id: str, details: dict[str, Any], request_id: str, entity_id: str) -> bool:
    """Publish the control action and best-effort await the node's result (log +
    file cleanup). Returns True only on a confirmed success; publishing alone has
    already actuated the lock, so a timeout is logged, not fatal."""
    from app.api.smart_home import _RESULT_DIR
    from app.node_settings import get_mqtt_client
    from app.services.node_command_service import get_node_command_service

    if get_mqtt_client() is None:
        logger.warning("presence automation: MQTT unavailable — cannot actuate %s", entity_id)
        return False

    result_file = os.path.join(_RESULT_DIR, f"{request_id}.json")
    get_node_command_service().publish_command_with_id(node_id, "action", details, request_id)

    loop = asyncio.get_event_loop()
    deadline = loop.time() + _RESULT_WAIT_S
    while loop.time() < deadline:
        if os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    result = json.load(f)
                os.unlink(result_file)
                ok = bool(result.get("success"))
                logger.info(
                    "presence automation: %s -> success=%s err=%s",
                    entity_id, ok, result.get("error"),
                )
                return ok
            except (json.JSONDecodeError, OSError):
                pass
        await asyncio.sleep(_RESULT_POLL_S)

    try:
        os.unlink(result_file)
    except OSError:
        pass
    logger.info(
        "presence automation: %s dispatched (no node result within %.0fs)",
        entity_id, _RESULT_WAIT_S,
    )
    return False


async def react_to_presence(
    ctx: ReactionContext,
    *,
    gate_check: Callable[[str, str], bool] | None = None,
    plan: Callable[[str, str], list[Actuation]] | None = None,
    dispatch: Callable[..., Awaitable[bool]] | None = None,
) -> str:
    """React to a presence signal by locking/unlocking the household's lock devices.

    A registered handler on the generic signal-reaction registry (dispatched by
    kind). Returns a short status string (for logging/tests): ``ignored`` (kind not
    a presence edge), ``disabled`` (per-direction gate off), ``no_lock_device``,
    ``no_node`` (a device exists but no node can reach it), ``error``, or
    ``"<action>:<confirmed>/<total>"``. Never raises.
    """
    mapping = _KIND_ACTION.get(ctx.kind)
    if mapping is None:
        return "ignored"
    action, gate_key = mapping

    gate_check = gate_check or _gate_enabled
    if not gate_check(ctx.household_id, gate_key):
        return "disabled"

    # Only a genuine home↔away transition actuates — skip heartbeat re-asserts.
    desired = "away" if ctx.kind == "presence.left" else "home"
    dkey = (str(ctx.household_id), str(ctx.user_id))
    if _last_state.get(dkey) == desired:
        return "unchanged"

    try:
        plan = plan or _plan_actuations
        actuations = plan(ctx.household_id, action)
        if not actuations:
            return "no_lock_device"  # nothing to remember: a later device still acts

        dispatch = dispatch or _dispatch
        confirmed = 0
        for node_id, details, request_id, entity_id in actuations:
            if node_id is None:
                logger.info("presence automation: no node can reach lock %s", entity_id)
                continue
            if await dispatch(node_id, details, request_id, entity_id):
                confirmed += 1

        if all(node_id is None for node_id, *_ in actuations):
            return "no_node"  # a node may come online later — don't latch the state
        # Latch the state only once we've dispatched to a node, so a transient
        # no-node blip doesn't suppress the real actuation when the node returns.
        _last_state[dkey] = desired
        return f"{action}:{confirmed}/{len(actuations)}"
    except Exception:  # noqa: BLE001 — a reaction must never break signal ingest
        logger.warning("presence automation failed for %s", ctx.household_id, exc_info=True)
        return "error"


def register_presence_automation() -> None:
    """Register the presence→lock/unlock reaction (called once at startup).

    Same handler for both edges; it derives the action + gate from ``ctx.kind`` via
    :data:`_KIND_ACTION`. Adding this needs no ingest or dispatcher change."""
    register_signal_reaction("presence.left", "auto_lock", react_to_presence)
    register_signal_reaction("presence.seen", "auto_unlock", react_to_presence)
