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
from app.services.persona_presets import (
    DEFAULT_PERSONA,
    DEFAULT_PERSONA_PRESET_ID,
    PERSONA_MAX_CHARS,
    PERSONA_PRESETS,
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
    # Phone calls (phone-calls PRD): the gate + user-tunable caps. The
    # attempt cap stays out — it's a compliance posture knob, not a
    # household preference.
    "phone_calls.enabled": "bool",
    "phone_calls.plan_ttl_minutes": "int",
    "phone_calls.audio_retention_days": "int",
    "phone_calls.max_call_seconds": "int",
    "phone_calls.calls_per_day": "int",
    "phone_calls.monthly_minutes_cap": "int",
    "phone_calls.max_concurrent_calls": "int",
    # Locality used to disambiguate business searches (e.g. "Springfield, IL").
    # A household preference, not a compliance knob — and getting it wrong
    # calls the wrong business, so the household must own it.
    "household.location": "string",
    # Household speaking-voice persona (the "voice" layer). Free text, shapes
    # tone ONLY — the CC prompt fences it in <personality> so it can't touch
    # tools/safety. Length-capped on write (PERSONA_MAX_CHARS) to bound the
    # cached prefix. Starter presets served via GET .../persona/presets.
    "persona.household_prompt": "string",
}

# Role required to CHANGE a household setting. Reads are open to any member;
# writes match the household-management tier (same as editing the name).
_WRITE_ROLE = "admin"
_READ_ROLE = "member"


def _coerce(value: Any, value_type: str) -> Any:
    """Coerce a raw settings value to its declared type (settings store strings).

    bool never raises (anything truthy-ish coerces). int raises ValueError on
    garbage — the write endpoint turns that into a 400; reads fall back to
    None rather than 500ing the whole settings screen on one bad row.
    """
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if value_type == "int":
        # bool is an int subclass — reject it explicitly, "true" is not a count.
        if isinstance(value, bool):
            raise ValueError(f"expected an integer, got boolean {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(f"expected an integer, got {value!r}")
        if isinstance(value, str):
            return int(value.strip())  # ValueError on garbage
        raise ValueError(f"expected an integer, got {type(value).__name__}")
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
        try:
            values[key] = _coerce(raw, vtype)
        except (ValueError, TypeError):
            logger.warning(
                "Household %s setting %s has uncoercible value %r", household_id, key, raw
            )
            values[key] = None

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

    try:
        coerced = _coerce(value, vtype)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value for {key}: {e}",
        )

    # The persona rides the cached prompt on every LLM call, so a paste-bomb
    # would balloon each turn. Cap its length (an empty string is allowed — it
    # clears the voice layer back to the flat identity line).
    if key == "persona.household_prompt" and len(str(coerced)) > PERSONA_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Persona is too long ({len(str(coerced))} chars); "
                f"max is {PERSONA_MAX_CHARS}."
            ),
        )

    settings = get_settings_service()
    ok = settings.set(key, coerced, household_id=str(household_id))
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to update {key}")

    logger.info(
        "Household %s set %s=%r (by user %s)", household_id, key, coerced, user.user_id
    )
    return {"success": True, "key": key, "value": coerced}


@router.get("/household/{household_id}/persona/presets")
async def get_persona_presets(
    household_id: str = Path(..., description="Household UUID"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Return the starter persona presets + the default, for the mobile chips.

    Any household member may read. The UI renders each preset as a tappable chip
    that loads its ``text`` into the editable persona box; ``default_preset_id``
    marks which chip is "the default" and doubles as the reset affordance.
    ``max_chars`` bounds the editor input.
    """
    verify_household_role(user.user_id, household_id, required_role=_READ_ROLE)
    return {
        "presets": PERSONA_PRESETS,
        "default_preset_id": DEFAULT_PERSONA_PRESET_ID,
        "default_text": DEFAULT_PERSONA,
        "max_chars": PERSONA_MAX_CHARS,
    }
