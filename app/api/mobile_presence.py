"""Mobile presence — the phone as a Signal Bus presence producer.

The mobile app can't hit ``POST /api/v0/signals`` (that ingress takes node or
app-to-app auth, not the user JWT the phone carries). This JWT-authed endpoint is
the bridge: it verifies the caller is a member of the household, then writes a
presence Signal SERVER-side via the shared :func:`record_presence` — ``user_id``
and ``source_key`` come from the trusted token, never the client, so a phone can
only ever report *its own* user's presence — and fans it out to the same
reaction/matcher planes as ``/signals`` via the shared dispatch helper.

Open presence only: no directed/command path is exposed here.
Mounted at ``/api/v0/mobile`` (see main.py), like the other mobile routers.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.deps import (
    AuthenticatedUser,
    get_db,
    verify_household_role,
    verify_user_jwt,
)
from app.services.signal_dispatch import dispatch_signal_edges
from app.services.signal_service import record_presence

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-presence"])

# Any household MEMBER may report their own presence (it's self-scoped by the JWT).
_PRESENCE_ROLE = "member"
_VALID_STATES = {"home", "away"}
# Default lifetime for a phone-reported "home" when the client doesn't specify one:
# long enough not to expire mid-stay, short enough to self-heal if the phone stops
# reporting (e.g. app removed) without an explicit "away". The geofence producer
# (a later phase) will tune this against its refresh cadence / exit edges.
_DEFAULT_MOBILE_TTL_SECONDS = 4 * 3600


class PresenceIn(BaseModel):
    household_id: str = Field(..., max_length=255)
    state: str = Field("home")                # "home" (arrived) | "away" (left)
    room: str | None = Field(None, max_length=255)
    ttl_seconds: int | None = Field(None, gt=0, le=604800)  # <= 7 days


@router.post("/presence", status_code=200)
def post_mobile_presence(
    body: PresenceIn,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict:
    """Report the calling user's home/away presence from their phone."""
    if body.state not in _VALID_STATES:
        raise HTTPException(
            status_code=422, detail=f"state must be one of {sorted(_VALID_STATES)}"
        )
    # Membership check — the user must belong to the household they name. The
    # user_id comes from the JWT (self-presence only; a phone can't report someone
    # else), so there's nothing to spoof beyond the household, which this guards.
    verify_household_role(user.user_id, body.household_id, required_role=_PRESENCE_ROLE)

    sig = record_presence(
        db,
        household_id=body.household_id,
        user_id=user.user_id,
        state=body.state,
        source_agent="mobile",
        room=body.room,
        ttl_seconds=body.ttl_seconds or _DEFAULT_MOBILE_TTL_SECONDS,
    )
    if sig is None:  # user_id is always set here, so this is defensive
        raise HTTPException(status_code=400, detail="could not record presence")

    # Drive the same generic planes /signals does, so presence can fire deterministic
    # reactions and feed the matcher. node_id is None — mobile presence is user-scoped,
    # not tied to a node.
    dispatch_signal_edges(
        household_id=body.household_id,
        node_id=None,
        user_id=user.user_id,
        kind=sig.kind,
        facts={"user_id": user.user_id, "state": body.state, "room": body.room},
    )
    return {"ok": True, "signal_id": sig.id, "kind": sig.kind}
