"""Signal-reaction registry — the generic dispatch plane for DETERMINISTIC reactions.

A *reaction* is server-side code that runs when a Signal of a given kind is ingested
(compute something, propose a card, …). Reactions register themselves for a kind
(once, at startup) and the ``/signals`` ingest fans EVERY ingested signal out to the
handlers registered for its kind — fire-and-forget, on the main loop, never raising
into ingest. This keeps ingest kind-AGNOSTIC: adding a reaction is "write a handler +
one ``register_signal_reaction(...)`` call", with zero ingest changes, and several
reactions may subscribe to the same kind.

Mirrors :mod:`app.services.server_callback_registry` (the proposable-action
dispatcher's registry). This is the DETERMINISTIC plane; the LLM-routed plane is the
situation matcher (``signal_situation_edge``), which the ingest path fires in
parallel. BOTH are generic — ingest names no signal kind and no reaction.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger("uvicorn")


@dataclass(frozen=True)
class ReactionContext:
    """Everything a reaction is handed about an ingested Signal — a uniform handler
    argument (like ``ServerCallbackContext``) so ONE generic dispatcher can invoke
    any reaction without knowing its kind."""

    household_id: str
    node_id: str | None
    user_id: int | None
    kind: str
    facts: dict[str, Any]


# A reaction returns a short status string (for logging/tests); it must never raise.
SignalReactionHandler = Callable[["ReactionContext"], Awaitable[str]]

_reactions: dict[str, list[tuple[str, SignalReactionHandler]]] = {}
_main_loop: asyncio.AbstractEventLoop | None = None


def register_signal_reaction(kind: str, name: str, handler: SignalReactionHandler) -> None:
    """Register a reaction for a signal ``kind``. Idempotent per ``(kind, name)`` so a
    re-import / double-registration never double-fires."""
    handlers = _reactions.setdefault(kind, [])
    if any(existing == name for existing, _ in handlers):
        return
    handlers.append((name, handler))
    logger.info("Registered signal reaction '%s' for kind=%s", name, kind)


def reactions_for(kind: str) -> list[tuple[str, SignalReactionHandler]]:
    """The reactions registered for ``kind`` (a copy — callers must not mutate it)."""
    return list(_reactions.get(kind, ()))


def clear_reactions() -> None:
    """Test hook — drop all registrations."""
    _reactions.clear()


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Capture the main loop so the SYNC ``/signals`` threadpool worker (which has no
    running loop) can still schedule reactions onto it."""
    global _main_loop
    _main_loop = loop


async def _run_reaction(name: str, handler: SignalReactionHandler, ctx: ReactionContext) -> None:
    try:
        status = await handler(ctx)
        logger.info("signal reaction '%s' (kind=%s) -> %s", name, ctx.kind, status)
    except Exception:  # noqa: BLE001 — a reaction must never break anything
        logger.warning("signal reaction '%s' failed (kind=%s)", name, ctx.kind, exc_info=True)


def schedule_signal_reactions(ctx: ReactionContext) -> None:
    """Fan an ingested Signal out to every reaction registered for its kind —
    fire-and-forget on the main loop. Never blocks or raises into the (sync) ingest
    path; a kind with no registered reactions is a silent no-op."""
    handlers = reactions_for(ctx.kind)
    if not handlers:
        return
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = _main_loop
        if loop is None:
            return
        for name, handler in handlers:
            loop.call_soon_threadsafe(
                lambda n=name, h=handler: loop.create_task(_run_reaction(n, h, ctx))
            )
    except Exception:  # noqa: BLE001 — scheduling must never break ingest
        pass
