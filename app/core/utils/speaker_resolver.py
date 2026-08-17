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


async def resolve_member_names(
    auth_base_url: str, user_ids: list[int] | None
) -> list[str]:
    """Batch-resolve household member user_ids to display names.

    One auth round-trip for all cache misses (same ``/internal/users/batch``
    endpoint and TTL cache as ``resolve_speaker_name``). Unresolvable ids
    are dropped — callers treat the names as a best-effort hint signal
    (the follow-up named-person addressing lean), so this NEVER raises:
    any failure degrades to fewer/no names.
    """
    ids: list[int] = []
    for uid in user_ids or []:
        try:
            ids.append(int(uid))
        except (TypeError, ValueError):
            continue
    if not ids:
        return []

    now = time.time()
    resolved: dict[int, str] = {}
    missing: list[int] = []
    for uid in ids:
        cached = _speaker_cache.get(uid)
        if cached and cached[1] > now:
            resolved[uid] = cached[0]
        else:
            missing.append(uid)

    if missing:
        try:
            joined = ",".join(str(u) for u in missing)
            url = f"{auth_base_url}/internal/users/batch?user_ids={joined}"
            result = await get(url, timeout=5)
            users = result.get("users") if isinstance(result, dict) else None
            if isinstance(users, dict):
                for uid in missing:
                    name = users.get(str(uid))
                    if name:
                        _speaker_cache[uid] = (name, now + _CACHE_TTL_SECONDS)
                        resolved[uid] = name
        except Exception as e:
            logger.warning(f"Failed to batch-resolve member names: {e}")

    return [resolved[uid] for uid in ids if uid in resolved]
