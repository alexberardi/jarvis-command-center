"""Proposal-suppression endpoints ('never suggest this again' blocklist).

- node plane: the detector agent fetches its blocklist *signals* (source_keys to
  skip deterministically + descriptors to inject into its extraction prompt).
- mobile plane: the user lists and un-blocks their own suppressions.

The suppress *write* happens through the proposable-action dispatcher's
``suppress`` server callback (a card tap), not here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import (
    AuthenticatedUser,
    verify_api_key,
    verify_household_role,
    verify_user_jwt,
)
from app.services.proposal_suppressions import (
    delete_suppression,
    list_suppressions,
    suppression_signals,
)

logger = logging.getLogger("uvicorn")

# ── node plane: the detector agent reads its blocklist ───────────────────────
node_router = APIRouter(tags=["proposals"])


@node_router.get("/proposals/suppressions")
def get_suppression_signals(
    command: str = Query(..., description="Target command, e.g. 'add_event'"),
    user_id: int = Query(..., description="The user the agent is scanning for"),
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict:
    """Blocklist signals for (this node's household, user_id, command):
    ``{source_keys: [...], descriptors: [...]}``. Household is taken from the
    validated node — the agent supplies the per-user id it's scanning for."""
    household_id = node_context.household_id
    if not household_id:
        raise HTTPException(status_code=400, detail="Node has no household")
    return suppression_signals(
        household_id=household_id, user_id=user_id, command=command
    )


# ── mobile plane: the user manages their own blocklist ───────────────────────
mobile_router = APIRouter(tags=["proposals"])


@mobile_router.get("/mobile/proposal-suppressions")
def list_my_suppressions(
    household_id: str = Query(...),
    command: str | None = Query(None),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict:
    verify_household_role(user.user_id, household_id, required_role="member")
    return {
        "suppressions": list_suppressions(
            household_id=household_id, user_id=user.user_id, command=command
        )
    }


@mobile_router.delete("/mobile/proposal-suppressions/{suppression_id}")
def delete_my_suppression(
    suppression_id: str,
    household_id: str = Query(...),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict:
    verify_household_role(user.user_id, household_id, required_role="member")
    ok = delete_suppression(
        suppression_id=suppression_id,
        household_id=household_id,
        user_id=user.user_id,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Suppression not found")
    return {"deleted": True}
