"""Registry for CC-plane (server-side) interactive-notification callbacks.

The callback flow was node-plane on both ends: mobile tap → CallbackJob →
MQTT → node executes the command's @callback method → node posts the result.
That works only when a node owns the originating command. Server tools
(deep research follow-ups, phone-call confirm/escalation cards) have no node
— CC itself must execute the callback.

This module is the server-side analogue of the node's command/callback
dispatch table: a tool registers a handler per (command_name, callback_name)
pair at import/startup time, and ``app/api/callbacks.py`` invokes it in a
background task when a tap arrives with no ``target_node_id``. The handler's
result flows through the exact same machinery as a node-posted result
(status polling, inbox fan-out on navigation_type=new_notification).

Handlers may be sync or async. They must be self-contained: they receive a
context snapshot, not a DB session — anything they persist happens through
their own services. Raising marks the job failed with the exception text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

logger = logging.getLogger("uvicorn")


@dataclass
class ServerCallbackContext:
    """Snapshot handed to a server-side callback handler."""

    job_id: str
    household_id: str
    user_id: int | None
    data: dict[str, Any] = field(default_factory=dict)
    navigation_type: str = "new_notification"


@dataclass
class ServerCallbackResult:
    """What a handler returns — mirrors the node's POST /result body."""

    success: bool
    error: str | None = None
    context_data: dict[str, Any] | None = None


ServerCallbackHandler = Callable[
    [ServerCallbackContext],
    Union[ServerCallbackResult, Awaitable[ServerCallbackResult]],
]

_registry: dict[tuple[str, str], ServerCallbackHandler] = {}


def register_server_callback(
    command_name: str,
    callback_name: str,
    handler: ServerCallbackHandler,
) -> None:
    """Register a handler for (command_name, callback_name).

    Re-registering the same pair overwrites (module reload during dev) but
    logs, so a genuine collision between two tools is visible.
    """
    key = (command_name, callback_name)
    if key in _registry and _registry[key] is not handler:
        logger.warning(
            "Server callback %s.%s re-registered (overwriting previous handler)",
            command_name,
            callback_name,
        )
    _registry[key] = handler


def unregister_server_callback(command_name: str, callback_name: str) -> None:
    _registry.pop((command_name, callback_name), None)


def get_server_callback(
    command_name: str, callback_name: str
) -> ServerCallbackHandler | None:
    return _registry.get((command_name, callback_name))


def registered_server_callbacks() -> list[tuple[str, str]]:
    return sorted(_registry.keys())
