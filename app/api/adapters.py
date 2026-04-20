"""Adapter proposal / deployment API (Phase 7.1).

Mobile-facing endpoints for the user-approved adapter deploy flow. Each
proposal — created by the scheduler after a successful training+eval run
(Phase 7.2) — is surfaced as an inbox item with Apply/Preview/Dismiss
actions; taps land here.

Auth: user JWT (jarvis-auth) + household-role check. Every route that
mutates adapter state confirms the caller is at least `power_user` in the
household the proposal belongs to.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.deps import (
    AuthenticatedUser,
    get_db,
    verify_household_role,
    verify_user_jwt,
)
from app.services.adapter_proposal_service import (
    AdapterProposalService,
    HashMismatchError,
    NoActiveAdapterError,
    ProposalExpiredError,
    ProposalNotFoundError,
    ProposalNotPendingError,
    ProposalView,
)
from app.services.inbox_notification_service import (
    push_adapter_deployed_to_inbox,
    push_adapter_reverted_to_inbox,
)
from app.services.settings_service import get_settings_service


router = APIRouter(prefix="/adapters", tags=["adapters"])
logger = logging.getLogger("uvicorn")


# --------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------


class ProposalResponse(BaseModel):
    id: str
    household_id: str
    adapter_hash: str
    provider_name_before: str | None = None
    provider_name_after: str | None = None
    pass_rate_before: float | None = None
    pass_rate_after: float | None = None
    latency_before_s: float | None = None
    latency_after_s: float | None = None
    per_command_delta: dict[str, Any] | None = None
    trained_on_examples: int | None = None
    status: str
    inbox_item_id: str | None = None
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ApplyResponse(BaseModel):
    proposal: ProposalResponse
    adapter_hash: str
    pass_rate: float | None = None
    trained_on_examples: int | None = None
    deployed_at: datetime
    provider_name: str | None = None
    deployed_inbox_item_id: str | None = None


class RevertResponse(BaseModel):
    restored_adapter_hash: str | None = None
    restored_pass_rate: float | None = None
    restored_provider_name: str | None = None
    reverted_inbox_item_id: str | None = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _view_to_response(view: ProposalView) -> ProposalResponse:
    return ProposalResponse(
        id=view.id,
        household_id=view.household_id,
        adapter_hash=view.adapter_hash,
        provider_name_before=view.provider_name_before,
        provider_name_after=view.provider_name_after,
        pass_rate_before=view.pass_rate_before,
        pass_rate_after=view.pass_rate_after,
        latency_before_s=view.latency_before_s,
        latency_after_s=view.latency_after_s,
        per_command_delta=view.per_command_delta,
        trained_on_examples=view.trained_on_examples,
        status=view.status,
        inbox_item_id=view.inbox_item_id,
        created_at=view.created_at,
        expires_at=view.expires_at,
        decided_at=view.decided_at,
    )


def _load_proposal_view(
    service: AdapterProposalService, proposal_id: str
) -> ProposalView:
    view = service.get(proposal_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return view


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@router.get("/proposals/{proposal_id}", response_model=ProposalResponse)
def get_proposal(
    proposal_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> ProposalResponse:
    """Return a single proposal. 404 unless the caller has household access."""
    service = AdapterProposalService(db)
    view = _load_proposal_view(service, proposal_id)
    verify_household_role(user.user_id, view.household_id, required_role="power_user")
    return _view_to_response(view)


@router.post("/proposals/{proposal_id}/apply", response_model=ApplyResponse)
def apply_proposal(
    proposal_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> ApplyResponse:
    """Deploy the proposed adapter and flip llm.interface to the paired provider."""
    service = AdapterProposalService(db)
    view = _load_proposal_view(service, proposal_id)
    verify_household_role(user.user_id, view.household_id, required_role="power_user")

    previous_hash = service.registry.get_active(view.household_id)
    previous_adapter_hash = (
        previous_hash.adapter_hash if previous_hash is not None else None
    )

    try:
        applied_view, dep = service.apply(
            proposal_id=proposal_id, household_id=view.household_id
        )
    except ProposalNotPendingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProposalExpiredError as exc:
        db.commit()  # persist the pending → expired flip
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except ProposalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Flip llm.interface in the same transaction so adapter + provider stay
    # consistent. Null/blank means "don't change interface" — Phase 5
    # compatibility for proposals that didn't pair a new provider.
    settings_service = get_settings_service()
    if applied_view.provider_name_after:
        settings_service.set("llm.interface", applied_view.provider_name_after)
    db.commit()

    deployed_inbox_id = push_adapter_deployed_to_inbox(
        household_id=view.household_id,
        adapter_hash=dep.adapter_hash,
        pass_rate=dep.pass_rate,
        latency_s=applied_view.latency_after_s,
        provider_name=applied_view.provider_name_after,
        previous_adapter_hash=previous_adapter_hash,
        user_id=user.user_id,
    )

    logger.info(
        "proposal applied: id=%s household=%s hash=%s provider=%s user=%s",
        proposal_id,
        view.household_id,
        dep.adapter_hash[:12],
        applied_view.provider_name_after,
        user.user_id,
    )
    return ApplyResponse(
        proposal=_view_to_response(applied_view),
        adapter_hash=dep.adapter_hash,
        pass_rate=dep.pass_rate,
        trained_on_examples=dep.trained_on_examples,
        deployed_at=dep.deployed_at,
        provider_name=applied_view.provider_name_after,
        deployed_inbox_item_id=deployed_inbox_id,
    )


@router.post("/proposals/{proposal_id}/dismiss", response_model=ProposalResponse)
def dismiss_proposal(
    proposal_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> ProposalResponse:
    """Dismiss a pending proposal without deploying."""
    service = AdapterProposalService(db)
    view = _load_proposal_view(service, proposal_id)
    verify_household_role(user.user_id, view.household_id, required_role="power_user")

    try:
        dismissed = service.dismiss(
            proposal_id=proposal_id, household_id=view.household_id
        )
    except ProposalNotPendingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProposalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db.commit()
    logger.info(
        "proposal dismissed: id=%s household=%s user=%s",
        proposal_id, view.household_id, user.user_id,
    )
    return _view_to_response(dismissed)


class RevertRequest(BaseModel):
    household_id: str


@router.post(
    "/deployments/{adapter_hash}/revert",
    response_model=RevertResponse,
)
def revert_deployment(
    adapter_hash: str,
    body: RevertRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> RevertResponse:
    """Roll back from the currently-active adapter for a household.

    `household_id` is in the body (not the URL) because revert is driven from
    the inbox item's stored metadata and we want the check to fail if the
    user somehow got a stale item referencing another household.
    """
    verify_household_role(user.user_id, body.household_id, required_role="power_user")

    service = AdapterProposalService(db)
    try:
        restored, provider_to_restore = service.revert(
            adapter_hash=adapter_hash, household_id=body.household_id
        )
    except HashMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NoActiveAdapterError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if restored is None:
        db.commit()
        raise HTTPException(
            status_code=404, detail="no prior adapter to roll back to"
        )

    settings_service = get_settings_service()
    if provider_to_restore:
        settings_service.set("llm.interface", provider_to_restore)
    db.commit()

    reverted_inbox_id = push_adapter_reverted_to_inbox(
        household_id=body.household_id,
        from_adapter_hash=adapter_hash,
        to_adapter_hash=restored.adapter_hash,
        from_provider_name=None,
        to_provider_name=provider_to_restore,
        user_id=user.user_id,
    )

    logger.info(
        "adapter reverted: household=%s from=%s to=%s provider=%s user=%s",
        body.household_id,
        adapter_hash[:12],
        restored.adapter_hash[:12],
        provider_to_restore,
        user.user_id,
    )
    return RevertResponse(
        restored_adapter_hash=restored.adapter_hash,
        restored_pass_rate=restored.pass_rate,
        restored_provider_name=provider_to_restore,
        reverted_inbox_item_id=reverted_inbox_id,
    )
