"""In-memory store for pending node factory-reset tokens.

When a node is deleted via the admin API, a short-lived token is stored here
and sent to the node via MQTT. The node then calls back to verify the token
before actually resetting — this prevents spoofed MQTT messages from
triggering factory resets.

Tokens expire after RESET_TOKEN_TTL_SECONDS (default 5 minutes).
"""

import time
import threading
import uuid

RESET_TOKEN_TTL_SECONDS: int = 300  # 5 minutes

_lock = threading.Lock()
_pending: dict[str, float] = {}  # token -> expiry timestamp


def create_reset_token() -> str:
    """Generate a reset token and store it with a TTL.

    Returns:
        The generated token string (UUID4).
    """
    token: str = uuid.uuid4().hex
    expiry: float = time.time() + RESET_TOKEN_TTL_SECONDS

    with _lock:
        _pending[token] = expiry
        _cleanup_expired()

    return token


def verify_and_consume(token: str) -> bool:
    """Verify a reset token is valid and consume it (single-use).

    Returns:
        True if the token was valid and not expired, False otherwise.
    """
    with _lock:
        _cleanup_expired()
        expiry = _pending.pop(token, None)

    if expiry is None:
        return False

    return time.time() < expiry


def verify_token(token: str) -> bool:
    """Verify a reset token is valid WITHOUT consuming it.

    Used by status-update endpoints during a factory reset, where the node
    needs to PATCH the task multiple times (in_progress, success). The token
    expires naturally after RESET_TOKEN_TTL_SECONDS — no need to consume on
    each call.
    """
    with _lock:
        _cleanup_expired()
        expiry = _pending.get(token)

    if expiry is None:
        return False

    return time.time() < expiry


def _cleanup_expired() -> None:
    """Remove expired tokens. Must be called under _lock."""
    now: float = time.time()
    expired = [t for t, exp in _pending.items() if now >= exp]
    for t in expired:
        del _pending[t]
