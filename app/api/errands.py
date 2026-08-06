"""Errands API — ``POST /errands``: plan a goal into a reviewable plan card.

Errand Runner POC chunk 3 (prds/errand-runner.md §2-§3). Node-authenticated (the
voice-launcher path): household + executing node come from the validated node
row; the speaker (the completion-notify target) is passed in the body, exactly
like ``/routines/run-background``. Orchestration lives in ``errand_service``.

Nothing runs from this endpoint — it drafts a plan and posts a plan card with
Run/Revise/Cancel. **Run** dispatches the plan step-by-step via the errand
executor (node commands over MQTT + server tools in-process).
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import get_db, verify_api_key
from app.models import Node
from app.services.errand_service import create_errand_plan

router = APIRouter()
logger = logging.getLogger("uvicorn")


class ErrandCreate(BaseModel):
    goal: str
    # Initiating speaker — the completion-notify target (like run-background's
    # speaker_user_id). Absent → the whole household is notified.
    user_id: int | None = None
    # Target node to run the errand on. Absent → the calling node. Must belong to
    # the caller's household.
    node_id: str | None = None


@router.post("/errands")
async def create_errand(
    body: ErrandCreate,
    node_ctx: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Plan an errand from a goal and post a plan card for review.

    Returns the DRAFT plan; the card carries Run/Cancel. Node-authenticated:
    household + executing node come from the validated node row, not the payload.
    """
    goal = (body.goal or "").strip()
    if not goal:
        raise HTTPException(status_code=422, detail="goal is required")

    household_id = node_ctx.household_id
    node = getattr(node_ctx, "node", None)
    if not household_id or node is None:
        raise HTTPException(
            status_code=403, detail="Node is not associated with a household"
        )

    # Target node: the caller's node by default; an override must be in the
    # caller's household (no cross-household targeting).
    target_node_id = node.node_id
    if body.node_id and body.node_id != node.node_id:
        target = (
            db.query(Node)
            .filter(
                Node.node_id == body.node_id,
                Node.household_id == household_id,
                Node.is_active.is_(True),  # never target a factory-reset node
            )
            .first()
        )
        if target is None:
            raise HTTPException(status_code=404, detail="Target node not in this household")
        target_node_id = target.node_id

    try:
        row = await create_errand_plan(
            db, household_id, target_node_id, goal, user_id=body.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Couldn't plan that errand: {e}")

    return {
        "errand_plan_id": row.id,
        "state": row.state,
        "summary": row.summary,
        "steps": json.loads(row.steps or "[]"),
    }
