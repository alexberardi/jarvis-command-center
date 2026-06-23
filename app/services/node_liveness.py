"""Node liveness tracking.

A node is shown "online" in the app/admin when its ``last_seen`` timestamp is
within ``ONLINE_THRESHOLD_MINUTES`` (see ``app/models.py:Node.is_online``).

Historically ``last_seen`` was written by *exactly one* thing: the dedicated
HTTP heartbeat (``POST /api/v0/admin/nodes/heartbeat``, every 300s). That made
"online" depend on a single fragile channel. If a node's heartbeat POSTs fail
for ~15 min — e.g. a flaky WAN / Cloudflare-tunnel leg between the node and CC,
even while the node is alive, on the LAN, and actively serving voice + MQTT —
the node is marked OFFLINE. (This was the kitchen-node incident: 48/48
heartbeats ``ReadTimeout``'d over the WAN leg while MQTT round-trips and SSH
kept working.)

This module lets *any* proof-of-life from a node refresh ``last_seen``, so
"online" reflects actual reachability on any channel rather than one HTTP TTL:

- every node-authenticated HTTP request (``verify_api_key``) — voice commands,
  ``/continue``, ``conversation/start``, and the ``/nodes/{id}/.../results``
  callbacks that carry node responses to CC's MQTT commands;
- a successful MQTT request/response round-trip with the node
  (``mobile_command_data._mqtt_request``), whose response never touches HTTP.

Writes are debounced (``LIVENESS_DEBOUNCE_SECONDS``) so a busy node doesn't
write ``last_seen`` on every request, and every entry point is fully
best-effort: a liveness bump must NEVER raise into an auth path or a request
handler. It is purely additive — the dedicated heartbeat still runs and still
updates ``last_seen`` (plus version / busy / protocols) as before.
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models import Node

logger = logging.getLogger("uvicorn")

# Don't rewrite last_seen more often than this per node. Far below the 15-min
# online threshold, so liveness stays fresh while keeping DB writes bounded
# (at most ~1 write/min/node from the hot path).
LIVENESS_DEBOUNCE_SECONDS = 60


def touch_node_last_seen(
    db: Session,
    node: Node | None,
    *,
    debounce_seconds: int = LIVENESS_DEBOUNCE_SECONDS,
) -> bool:
    """Best-effort refresh of ``node.last_seen`` from any proof-of-life.

    Updates ``last_seen`` to now and commits, unless it was already updated
    within ``debounce_seconds`` (in which case this is a no-op). Returns True
    iff it wrote.

    Never raises: a liveness bump must never break the request that triggered
    it. On any error the session is rolled back and False is returned.
    """
    if node is None:
        return False
    try:
        now = datetime.utcnow()
        last_seen = node.last_seen
        if last_seen is not None and last_seen > now - timedelta(seconds=debounce_seconds):
            return False  # seen within the debounce window — skip the write
        node.last_seen = now
        db.commit()
        return True
    except Exception as exc:  # pragma: no cover - defensive, must not propagate
        logger.debug(
            "touch_node_last_seen failed for %s: %s",
            getattr(node, "node_id", "?"),
            exc,
        )
        try:
            db.rollback()
        except Exception:
            pass
        return False


def record_node_seen(
    node_id: str,
    *,
    debounce_seconds: int = LIVENESS_DEBOUNCE_SECONDS,
) -> bool:
    """Refresh ``last_seen`` for ``node_id`` using a dedicated short session.

    For callers that have proof-of-life but no request DB session / loaded
    Node — e.g. the MQTT request/response round-trip path. Opens its own
    session so it never commits a caller's transaction. Best-effort; returns
    True iff it wrote.
    """
    try:
        from app.db import get_session_local

        db = get_session_local()()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("record_node_seen could not open session for %s: %s", node_id, exc)
        return False
    try:
        node = db.query(Node).filter(Node.node_id == node_id).first()
        return touch_node_last_seen(db, node, debounce_seconds=debounce_seconds)
    finally:
        db.close()
