"""Mobile call-context editor — user-JWT authenticated, USER-scoped.

Mounted at /api/v0/mobile/call-context. Backs jarvis-node-mobile's call
context grid: the per-user details the phone agent may use (name, callback
number, insurance, plus custom fields).

USER-scoped, not household — unlike the phonebook. Insurance member IDs and
callback numbers are personal, so the scope is the JWT's user and there is
no household in the path or the role check. A member does not need anyone's
permission to edit their own details, and no other member should see them.

This is PII: anything stored here can be spoken on a call and lands in the
transcript and recording. The tier control (state vs give-if-asked) governs
whether the agent may volunteer a field; the gateway's spoken-output guard
enforces it. This endpoint only reads and writes the store.

Storage is a single JSON blob (the well-known + custom fields are a short
list the user edits as a unit), held in the `phone_calls.call_context`
setting. GET returns the current fields plus the static catalog the grid
renders; PUT replaces the whole list. Full-replace rather than per-row CRUD
because the blob has no stable row ids — the grid owns the list and saves it
whole.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.deps import AuthenticatedUser, verify_user_jwt
from app.services.call_context import (
    SETTING_KEY,
    catalog,
    field_to_dict,
    load_call_context,
    prepare_for_storage,
    serialize_fields,
)
from app.services.settings_service import get_settings_service

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-call-context"])


class CallContextRow(BaseModel):
    """One row from the grid. `key` is optional — a custom field arrives as a
    label the user typed, and the server mints the key. Category and tier are
    optional too; the server coerces anything unrecognized to a safe default
    (General, and the private give-if-asked side)."""

    key: str | None = None
    label: str = ""
    value: str = ""
    category: str | None = None
    tier: str | None = None


class CallContextWrite(BaseModel):
    fields: list[CallContextRow] = Field(default_factory=list)


def _fields_response(user_id: int) -> dict:
    fields = load_call_context(user_id)
    return {
        "fields": [field_to_dict(f) for f in fields],
        "catalog": catalog(),
    }


@router.get("/call-context")
def get_call_context(user: AuthenticatedUser = Depends(verify_user_jwt)) -> dict:
    """The caller's stored fields plus the grid's static vocabulary.

    Reads degrade to an empty list inside `load_call_context`, so a settings
    outage yields an editable-but-empty grid, never a 500 that blocks the
    screen from opening.
    """
    return _fields_response(user.user_id)


@router.put("/call-context")
def put_call_context(
    body: CallContextWrite,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict:
    """Replace the caller's stored fields with the grid's current list.

    The payload is run through the same coercion and first-wins dedup as a
    read before it is stored, so the response is the canonical result — what
    a later call will actually see — not a mirror of what was sent. A row the
    user left blank, or a duplicate key, simply will not come back.

    Serialized to a JSON string here on purpose: the setting is declared
    `value_type="string"`, so the settings client stores the value verbatim
    (it does not JSON-encode a string-typed value). Handing it a dict would
    persist a Python repr that the reader's json.loads then rejects.
    """
    fields = prepare_for_storage([row.model_dump() for row in body.fields])
    ok = get_settings_service().set(
        SETTING_KEY, serialize_fields(fields), user_id=user.user_id
    )
    if not ok:
        logger.error("call context write failed for user %s", user.user_id)
        # Surface the failure rather than pretending it saved — a silent drop
        # of PII the user thinks is stored is the worse outcome here.
        raise HTTPException(status_code=500, detail="Could not save call context")

    return {
        "fields": [field_to_dict(f) for f in fields],
        "catalog": catalog(),
    }
