"""Attention broker API — event ingest + journal read (prds/attention-broker.md).

POST /attention/events — record + gate proactive events (node X-API-Key auth).
    Default requested_rung is "journal": the phase-2 node shim ships already-
    handled local alerts here purely for visibility. Callers may request
    inbox/push, in which case the broker gates and this endpoint delivers via
    the same downstream services as the interposed /node/* endpoints.
GET /attention/journal — recent deliveries with gate trails (admin token).
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import get_db, verify_admin_key, verify_api_key
from app.models import AttentionDelivery, AttentionEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attention", tags=["attention"])


class AttentionEventIn(BaseModel):
    source: str
    category: str = "general"
    title: str
    summary: str = ""
    dedupe_key: str | None = None
    target_user_id: int | None = None
    requested_rung: Literal["journal", "inbox", "push"] = "journal"


class AttentionEventsRequest(BaseModel):
    events: list[AttentionEventIn]


class AttentionEventResult(BaseModel):
    event_id: str
    rung: str
    delivered: bool
    withheld_by: str | None = None


class AttentionEventsResponse(BaseModel):
    enabled: bool
    results: list[AttentionEventResult]


@router.post("/events", response_model=AttentionEventsResponse)
async def post_attention_events(
    request: AttentionEventsRequest,
    node_context: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> AttentionEventsResponse:
    from app.services.attention_broker import attention_enabled, mark_outcome, record_and_route
    from app.services.inbox_notification_service import post_inbox_item_sync

    household_id = node_context.node.household_id or ""
    if not household_id or not attention_enabled(household_id):
        return AttentionEventsResponse(enabled=False, results=[])

    results: list[AttentionEventResult] = []
    for item in request.events[:50]:  # per-request cap; a node has no business batching more
        decision = record_and_route(
            db,
            household_id=household_id,
            source=item.source,
            category=item.category,
            title=item.title,
            summary=item.summary,
            requested_rung=item.requested_rung,
            dedupe_key=item.dedupe_key,
            target_user_id=item.target_user_id,
            origin_node_id=node_context.node.node_id,
        )
        if decision.deliver:
            inbox_id = await asyncio.to_thread(
                post_inbox_item_sync,
                household_id=household_id,
                user_id=item.target_user_id,
                title=item.title[:500],
                summary=item.summary,
                body=item.summary,
                category=item.category[:50],
                metadata={"node_id": node_context.node.node_id},
                push=decision.rung == "push",
                target_type="user" if item.target_user_id is not None else "household",
            )
            mark_outcome(
                db,
                decision.delivery_id,
                outcome="delivered" if inbox_id else "failed",
                inbox_item_id=inbox_id,
            )
        results.append(
            AttentionEventResult(
                event_id=decision.event_id,
                rung=decision.rung,
                delivered=decision.deliver,
                withheld_by=decision.withheld_by,
            )
        )
    return AttentionEventsResponse(enabled=True, results=results)


@router.get("/journal", dependencies=[Depends(verify_admin_key)])
def get_attention_journal(
    household_id: str = Query(...),
    hours: int = Query(24, le=24 * 30),
    db: Session = Depends(get_db),
) -> dict:
    """Admin/debug view of recent deliveries with their gate trails."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(AttentionDelivery, AttentionEvent)
        .join(AttentionEvent, AttentionDelivery.event_id == AttentionEvent.id)
        .filter(
            AttentionDelivery.household_id == household_id,
            AttentionDelivery.created_at >= cutoff,
        )
        .order_by(AttentionDelivery.created_at.desc())
        .limit(500)
        .all()
    )
    return {
        "household_id": household_id,
        "hours": hours,
        "deliveries": [
            {
                "delivery_id": d.id,
                "event_id": e.id,
                "source": e.source,
                "category": e.category,
                "title": e.title,
                "rung": d.rung,
                "withheld_by": d.withheld_by,
                "outcome": d.outcome,
                "gate_trail": json.loads(d.gate_trail_json or "[]"),
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d, e in rows
        ],
    }
