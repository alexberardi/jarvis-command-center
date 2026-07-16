"""Household-scoped feature settings, controllable by household admins from mobile.

The shared ``/settings/*`` router (jarvis-settings-client) requires a GLOBAL
``is_superuser`` to write — the wrong scope for a household-level toggle. This
router lets a household's own admin read and flip an *allowlisted* set of
household-controllable settings (currently just the web-search master toggle),
authorized by their role IN THAT HOUSEHOLD via ``verify_household_role`` rather
than a global superuser flag.

Powers the "Household Settings" screen in jarvis-node-mobile.
"""
import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path

from app.deps import (
    AuthenticatedUser,
    verify_household_role,
    verify_user_jwt,
)
from app.services.settings_service import get_settings_service

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-household-settings"])

# Settings a household admin may control from the mobile Household Settings
# screen. This explicit allowlist is the security boundary — it stops this
# endpoint from becoming a household-admin write to ANY command-center setting.
HOUSEHOLD_CONTROLLABLE_SETTINGS: dict[str, str] = {
    "web_search.enabled": "bool",
    "web_scraping.allow_external": "bool",
}

# Role required to CHANGE a household setting. Reads are open to any member;
# writes match the household-management tier (same as editing the name).
_WRITE_ROLE = "admin"
_READ_ROLE = "member"


def _coerce(value: Any, value_type: str) -> Any:
    """Coerce a raw settings value to its declared type (settings store strings)."""
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    return value


@router.get("/household/{household_id}/settings")
async def get_household_settings(
    household_id: str = Path(..., description="Household UUID"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Return the household-controllable settings + current values.

    Any member of the household may read. Values fall back to the code default
    (web_search.enabled → False) when no override row exists, so no seed data is
    required for new households.
    """
    verify_household_role(user.user_id, household_id, required_role=_READ_ROLE)

    settings = get_settings_service()
    values: dict[str, Any] = {}
    for key, vtype in HOUSEHOLD_CONTROLLABLE_SETTINGS.items():
        raw = settings.get(key, household_id=str(household_id))
        values[key] = _coerce(raw, vtype)

    return {"household_id": household_id, "settings": values}


@router.put("/household/{household_id}/settings/{key:path}")
async def put_household_setting(
    household_id: str = Path(..., description="Household UUID"),
    key: str = Path(..., description="Setting key, e.g. web_search.enabled"),
    value: Any = Body(..., embed=True, description="New value"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Set a household-controllable setting. Requires household admin.

    Rejects any key not on the allowlist (404) so this can't be used to write
    arbitrary command-center settings.
    """
    vtype = HOUSEHOLD_CONTROLLABLE_SETTINGS.get(key)
    if vtype is None:
        raise HTTPException(
            status_code=404,
            detail=f"Setting is not household-controllable: {key}",
        )

    verify_household_role(user.user_id, household_id, required_role=_WRITE_ROLE)

    coerced = _coerce(value, vtype)
    settings = get_settings_service()
    ok = settings.set(key, coerced, household_id=str(household_id))
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to update {key}")

    logger.info(
        "Household %s set %s=%r (by user %s)", household_id, key, coerced, user.user_id
    )
    return {"success": True, "key": key, "value": coerced}
