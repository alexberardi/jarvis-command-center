"""Shared signal fan-out — the generic post-ingest dispatch planes.

A just-persisted Signal drives two GENERIC (kind-agnostic) planes: the proactive
matcher edge and the deterministic reaction registry. Both the ``/signals`` ingest
(app/api/signals.py) and the mobile presence endpoint call this one helper so the
wiring lives in a single place instead of being duplicated per producer.

Fire-and-forget and best-effort — never raises into the caller, never blocks it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("uvicorn")


def dispatch_signal_edges(
    *,
    household_id: str,
    node_id: str | None,
    user_id: int | None,
    kind: str,
    facts: dict[str, Any] | None,
) -> None:
    """Fan a persisted Signal out to the matcher edge + the reaction registry.

    Both planes are kind-agnostic — they name no signal kind. Each is guarded so one
    failing never affects the other or the caller.
    """
    try:
        from app.services.situation_matcher_service import signal_situation_edge

        signal_situation_edge(household_id, node_id)
    except Exception:  # noqa: BLE001 — matcher edge is best-effort
        logger.debug("dispatch_signal_edges: situation edge failed", exc_info=True)
    try:
        from app.services.signal_reaction_registry import (
            ReactionContext,
            schedule_signal_reactions,
        )

        schedule_signal_reactions(
            ReactionContext(
                household_id=household_id,
                node_id=node_id,
                user_id=user_id,
                kind=kind,
                facts=facts or {},
            )
        )
    except Exception:  # noqa: BLE001 — reaction scheduling is best-effort
        logger.debug("dispatch_signal_edges: reaction scheduling failed", exc_info=True)
