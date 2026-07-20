"""Plan-time context queries against a household's nodes.

Server-side planners call `query_context(...)` to ask an installed node
command for typed, read-only context BEFORE acting. First consumer: the
phone-call plan-draft step asking the calendar command for `availability`
so the confirm card carries a real constraint envelope instead of a
fill-me-in placeholder.

Transport is the proven subscribe-before-publish round-trip
(`MQTTClient.request_response`) already carrying the mobile command-data
browser — here against `jarvis/nodes/{node_id}/context/query`, with the
context OPERATION in the payload rather than the topic so a newly
installed provider needs no CC change.

Contract (phone-calls PRD, cross-agent-context):
- Plan-time only. Never called from inside a live phone call: the callee is
  untrusted input, and a live context tool would be both an exfiltration
  channel and a re-opened injection surface.
- The provider stays node-side; credentials never leave the LAN edge.
- **Degrades, never blocks.** Node offline, no provider installed, MQTT
  down, malformed answer — every path returns a ContextAnswer with
  `ok=False` and a human-readable reason. Callers proceed without the
  context; they must never fail the user's request over it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.models import Node

logger = logging.getLogger("uvicorn")

# Plan drafting already costs a background-model call; keep the context
# round-trip well under a user's patience for the confirm card to arrive.
DEFAULT_TIMEOUT_SECONDS = 8.0


@dataclass
class ContextAnswer:
    """Result of a plan-time context query — never raises to the caller."""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    node_id: str | None = None
    command_name: str | None = None

    @classmethod
    def unavailable(cls, error: str, node_id: str | None = None) -> "ContextAnswer":
        return cls(ok=False, data={}, error=error, node_id=node_id)


def _candidate_nodes(db, household_id: str) -> list[Node]:
    """Active nodes in the household, most-recently-seen first.

    Online nodes lead, but offline ones are still tried afterwards: a node
    whose heartbeat lapsed may well answer MQTT (the command-data browser
    relies on exactly that), and the cost of trying is one timeout.
    """
    nodes = (
        db.query(Node)
        .filter(Node.household_id == household_id, Node.is_active.is_(True))
        .all()
    )
    return sorted(
        nodes,
        key=lambda n: (0 if n.is_online() else 1, -(n.last_seen.timestamp() if n.last_seen else 0)),
    )


async def _round_trip(
    node_id: str, payload: dict[str, Any], timeout_s: float
) -> dict[str, Any] | None:
    """One subscribe-before-publish exchange; None on timeout/transport error."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("Context query skipped: MQTT client unavailable")
        return None

    correlation_id = payload.setdefault("correlation_id", str(uuid.uuid4()))
    request_topic = f"jarvis/nodes/{node_id}/context/query"
    response_topic = f"{request_topic}/response/{correlation_id}"

    raw = await asyncio.to_thread(
        client.request_response,
        request_topic,
        response_topic,
        json.dumps(payload),
        timeout_s,
    )
    if raw is None:
        return None
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Node %s returned non-JSON context response", node_id)
        return None
    return body if isinstance(body, dict) else None


async def query_context(
    household_id: str,
    operation: str,
    params: dict[str, Any] | None = None,
    *,
    command: str | None = None,
    user_id: int | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    db=None,
) -> ContextAnswer:
    """Ask the household's nodes for one typed context operation.

    Tries nodes in liveness order and returns the first real answer. A node
    that reports `no_provider` is not a failure of that node — it simply
    doesn't have the command installed, so the next node is tried.

    ``user_id`` identifies the speaker the context is being gathered FOR.
    Providers whose data is per-person (calendar, email) resolve their
    credentials from it; omitting it makes them refuse rather than answer
    with someone else's data.
    """
    own_session = db is None
    if own_session:
        from app.db import get_session_local

        db = get_session_local()()
    try:
        nodes = _candidate_nodes(db, household_id)
    finally:
        if own_session:
            db.close()

    if not nodes:
        return ContextAnswer.unavailable("no nodes in this household")

    last_error = "no node answered"
    for node in nodes:
        payload: dict[str, Any] = {"operation": operation, "params": params or {}}
        if command:
            payload["command"] = command
        if user_id is not None:
            payload["user_id"] = user_id

        body = await _round_trip(node.node_id, payload, timeout_s)
        if body is None:
            last_error = "node did not respond"
            continue

        if body.get("ok"):
            return ContextAnswer(
                ok=True,
                data=body.get("data") or {},
                node_id=node.node_id,
                command_name=body.get("command_name"),
            )

        # Honest node-side refusal. "no_provider" just means this node
        # doesn't have the command — keep looking.
        last_error = str(body.get("error") or "context query failed")
        if body.get("code") == "no_provider":
            continue
        return ContextAnswer.unavailable(last_error, node_id=node.node_id)

    return ContextAnswer.unavailable(last_error)
