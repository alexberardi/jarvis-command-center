"""Mobile schedules screen — user-JWT authenticated, HOUSEHOLD-scoped.

Mounted at /api/v0/mobile. Backs jarvis-node-mobile's Schedules screen: the
always-there list of the household's scheduled + recurring errands (the same rows
the "what errands do I have scheduled?" voice card lists), with a per-errand cancel.

HOUSEHOLD-scoped: a schedule belongs to a household, so the caller must be a member
of it (``verify_household_role`` at the ``member`` level — a member creates schedules
freely by voice, so listing/cancelling their household's schedules needs no elevated
role). The presentation strings (cadence, next-fire) come from ``schedule_service`` so
this screen and the voice card read a cadence identically.

Cancel returns the FRESH list so the screen updates in place off one round-trip.
"""

import logging

from fastapi import APIRouter, Depends, Path

from app.deps import AuthenticatedUser, verify_household_role, verify_user_jwt
from app.services import schedule_service

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-schedules"])

# A member can create schedules by voice, so listing + cancelling their own
# household's schedules is a member-level action (not admin).
_ROLE = "member"


@router.get("/household/{household_id}/schedules")
def list_household_schedules(
    household_id: str = Path(..., description="Household UUID"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict:
    """The household's live (active/paused) schedules, enriched for display.

    Reads never 500 the screen: ``list_schedule_views`` degrades to an empty list on
    a DB hiccup, so the screen opens empty rather than erroring."""
    verify_household_role(user.user_id, household_id, required_role=_ROLE)
    return {
        "household_id": household_id,
        "schedules": schedule_service.list_schedule_views(household_id),
    }


@router.post("/household/{household_id}/schedules/{schedule_id}/cancel")
def cancel_household_schedule(
    household_id: str = Path(..., description="Household UUID"),
    schedule_id: str = Path(..., description="Schedule id to cancel"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict:
    """Cancel a schedule (household-scoped + idempotent) and return the refreshed
    list so the screen updates in place. ``cancelled`` is False if it was already
    terminal (nothing to do) — the screen still gets the current list."""
    verify_household_role(user.user_id, household_id, required_role=_ROLE)
    cancelled = schedule_service.cancel_schedule(schedule_id, household_id)
    logger.info(
        "🗓️ Mobile cancel schedule %s (household=%s user=%s) → %s",
        schedule_id, household_id, user.user_id, cancelled,
    )
    return {
        "cancelled": cancelled,
        "schedules": schedule_service.list_schedule_views(household_id),
    }
