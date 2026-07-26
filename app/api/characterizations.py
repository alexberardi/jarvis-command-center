"""Characterization API — read + manually trigger the per-person synthesized view.

Slice 1 surface for inspecting quality before any injection work:
  - GET lets us eyeball what the synthesis job produced for a person.
  - POST /synthesize triggers a synthesis immediately so we don't have to wait on
    the background interval during testing.

Auth mirrors the memory API: admin API key OR a household-member JWT.
"""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.deps import get_db
from app.provisioning import (
    ProvisioningAuthContext,
    require_household_access,
    verify_provisioning_auth,
)
from app.services.characterization_service import CharacterizationService
from app.services.characterization_synthesis_service import (
    synthesize_for_user,
    synthesize_for_user_sync,
)

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/characterizations", tags=["characterizations"])


class CharacterizationResponse(BaseModel):
    user_id: int
    household_id: str
    body: dict
    rendered: str | None = None
    confidence: float | None = None
    version: int
    last_transcript_at: datetime | None = None
    model: str | None = None
    created_at: datetime
    updated_at: datetime


def _to_response(row) -> CharacterizationResponse:
    try:
        body = json.loads(row.body) if row.body else {}
    except (json.JSONDecodeError, TypeError):
        body = {}
    return CharacterizationResponse(
        user_id=row.user_id,
        household_id=row.household_id,
        body=body,
        rendered=row.rendered,
        confidence=row.confidence,
        version=row.version,
        last_transcript_at=row.last_transcript_at,
        model=row.model,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("")
def get_characterization(
    user_id: int,
    household_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> CharacterizationResponse:
    """Return the current synthesized characterization for a person."""
    require_household_access(household_id, auth)
    row = CharacterizationService(db).get(user_id, household_id)
    if not row:
        raise HTTPException(status_code=404, detail="No characterization yet for this user")
    return _to_response(row)


@router.post("/synthesize", status_code=202)
async def trigger_synthesis(
    user_id: int,
    household_id: str,
    response: Response,
    sync: bool = False,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
) -> dict:
    """Synthesize a person's characterization (dev/inspection).

    Default (async): enqueues on the background model via the LLM queue; the
    result lands via /characterization-synthesis/callback and can then be read
    with GET (202 Accepted).

    ?sync=true: runs the background model inline and returns the characterization
    directly (200) — handy for testing and for setups where the queue callback
    isn't wired.
    """
    require_household_access(household_id, auth)

    if sync:
        try:
            result = await synthesize_for_user_sync(user_id, household_id)
        except Exception as e:
            logger.error("Sync synthesis failed for user %s: %s", user_id, e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Synthesis failed: {e}")
        if result is None:
            raise HTTPException(status_code=502, detail="Model returned unparseable output")
        response.status_code = 200
        return result

    await synthesize_for_user(user_id, household_id)
    return {"status": "enqueued", "user_id": user_id, "household_id": household_id}
