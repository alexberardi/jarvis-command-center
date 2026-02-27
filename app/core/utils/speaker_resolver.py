"""Resolve speaker_user_id to a display name via jarvis-auth batch lookup.

Uses a TTL cache to avoid repeated auth-service calls for the same user.
"""

import logging
import time
from typing import Optional

from app.core.utils.rest_client import get

logger = logging.getLogger("uvicorn")

# Simple TTL cache: {user_id: (display_name, expiry_timestamp)}
_speaker_cache: dict[int, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


async def resolve_speaker_name(
    auth_base_url: str, user_id: int
) -> Optional[str]:
    """Resolve a user_id to a display name via jarvis-auth.

    Args:
        auth_base_url: Base URL of jarvis-auth (e.g. "http://localhost:7701")
        user_id: The user ID to resolve

    Returns:
        Display name (username) or None if resolution fails
    """
    now = time.time()

    # Check cache
    cached = _speaker_cache.get(user_id)
    if cached and cached[1] > now:
        return cached[0]

    try:
        url = f"{auth_base_url}/internal/users/batch?user_ids={user_id}"
        result = await get(url, timeout=5)

        if isinstance(result, dict) and "users" in result:
            users = result["users"]
            name = users.get(str(user_id))
            if name:
                _speaker_cache[user_id] = (name, now + _CACHE_TTL_SECONDS)
                return name

        logger.warning("Could not resolve speaker name", extra={"user_id": user_id})
        return None

    except Exception as e:
        logger.warning(f"Failed to resolve speaker name for user_id={user_id}: {e}")
        return None
