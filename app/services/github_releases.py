"""Thin GitHub Releases client with a 5-minute in-process cache.

We only need "what's the latest tag for `alexberardi/jarvis-node-setup`?"
so this deliberately stays tiny — no pagination, no asset listing, no auth.
Rate-limited GET (60 req/hr anonymous) is plenty with caching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx


logger = logging.getLogger("uvicorn")

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "alexberardi/jarvis-node-setup"
_CACHE_TTL_SECONDS = 300


@dataclass
class ReleaseInfo:
    tag: str           # e.g. "v0.3.0"
    version: str       # tag without the leading "v" (matches VERSION file)
    published_at: Optional[str] = None


_cache: dict[str, tuple[float, ReleaseInfo]] = {}


def _updates_allowed(household_id: str | None = None) -> bool:
    """Fail-closed check of the ``updates.allow_check`` setting.

    Gates the outbound lookup to api.github.com. Default OFF so a local-only
    deployment never egresses to GitHub for update checks. Any error (settings
    outage, typo'd key, missing service) → disabled, never enabled.
    """
    try:
        from app.services.settings_service import get_settings_service

        val = get_settings_service().get(
            "updates.allow_check",
            household_id=str(household_id) if household_id is not None else None,
        )
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)
    except Exception as e:
        logger.warning(
            "updates.allow_check gate check failed, defaulting to DISABLED: %s", e
        )
        return False


def latest_release(
    repo: str = DEFAULT_REPO, household_id: str | None = None
) -> ReleaseInfo | None:
    """Return the repo's latest published release, or None on failure.

    Gated on ``updates.allow_check`` (default OFF): when disabled we return None
    WITHOUT touching the cache or the network — preserving the documented
    None-on-failure contract so callers fail soft (no outbound egress).

    Cached for _CACHE_TTL_SECONDS per repo. Network errors fall through as
    None so callers can decide how to fail soft (we don't want update-UI
    outages to take down unrelated endpoints).
    """
    if not _updates_allowed(household_id):
        return None

    now = time.time()
    cached = _cache.get(repo)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{GITHUB_API}/repos/{repo}/releases/latest")
        if resp.status_code != 200:
            return None
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    tag = body.get("tag_name")
    if not tag:
        return None

    info = ReleaseInfo(
        tag=tag,
        version=tag[1:] if tag.startswith("v") else tag,
        published_at=body.get("published_at"),
    )
    _cache[repo] = (now, info)
    return info


def resolve_target_version(
    requested: str | None, household_id: str | None = None
) -> str | None:
    """Resolve an "upgrade target" to a concrete version string.

    - "latest" or None → latest release version (without v prefix), or None
      if no release is available (or update checks are disabled — the GitHub
      lookup is gated on ``updates.allow_check``).
    - Anything else → returned as-is (with any leading v stripped), so
      callers can request a specific version. This path never touches GitHub,
      so an explicit version still installs even when checks are disabled.
    """
    if requested is None or requested.lower() == "latest":
        info = latest_release(household_id=household_id)
        return info.version if info else None
    return requested[1:] if requested.startswith("v") else requested
