"""Capability registry — what proposable actions each node advertises.

The proposable-action dispatcher must answer "does node N offer command C's
action A, and with what typed params?" before it will run anything. Commands
advertise their ``proposable_actions`` (SDK ``ProposableAction.to_dict()`` wire
form) inside the ``available_commands`` they report to command-center via the
``report_tools`` MQTT round-trip (node-setup builds these with
``get_command_schema()``, which now includes ``proposable_actions``).

v0 resolves on demand: the dispatcher fetches the node's advertised commands at
tap time (the node is online — proposals are online-only) and finds the matching
proposable action. This is the *authoritative opt-in gate*: a callback the
command did not advertise as proposable resolves to ``None`` and the dispatcher
refuses it, so an agent (or a crafted tap) cannot invoke an un-opted-in callback.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger("uvicorn")

# Injectable fetch boundary (patched in tests). Returns the node's tool report
# ({"available_commands": [...], ...}) or None if the node didn't answer.
NodeToolsFetcher = Callable[[str], Awaitable[dict[str, Any] | None]]


async def _default_fetch(node_id: str) -> dict[str, Any] | None:
    from app.api.node_tools import _request_tools_from_node

    return await _request_tools_from_node(node_id)


async def resolve_proposable_action(
    node_id: str,
    command_name: str,
    action_name: str,
    *,
    fetch: NodeToolsFetcher | None = None,
) -> dict[str, Any] | None:
    """Return the advertised ProposableAction wire-dict for
    ``node_id → command_name → action_name``, or ``None`` if the node does not
    currently advertise it (unreachable node, command not installed, or the
    callback is not opted in as proposable).

    ``None`` is the refusal signal — the dispatcher turns it into a visible
    "no device here can do that" failure, never a silent no-op.
    """
    fetch = fetch or _default_fetch
    try:
        report = await fetch(node_id)
    except Exception:  # noqa: BLE001 — a fetch failure is a resolution failure
        logger.warning("capability_registry: fetch failed for node=%s", node_id)
        return None
    if not report:
        return None
    for cmd in report.get("available_commands", []) or []:
        if cmd.get("command_name") != command_name:
            continue
        for action in cmd.get("proposable_actions", []) or []:
            if action.get("callback") == action_name:
                return action
    return None


async def list_proposable_actions(
    node_id: str, *, fetch: NodeToolsFetcher | None = None
) -> list[dict[str, Any]]:
    """All proposable actions a node currently advertises, each stamped with its
    owning ``command`` — the menu the capability matcher plans over.
    """
    fetch = fetch or _default_fetch
    try:
        report = await fetch(node_id)
    except Exception:  # noqa: BLE001
        return []
    if not report:
        return []
    out: list[dict[str, Any]] = []
    for cmd in report.get("available_commands", []) or []:
        name = cmd.get("command_name")
        for action in cmd.get("proposable_actions", []) or []:
            out.append({**action, "command": name})
    return out
