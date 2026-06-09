"""
Request trace API — admin list/detail and mobile per-conversation lookups.

Serves the trace visualizer in both the admin page and mobile app.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.deps import get_db, verify_user_jwt, AuthenticatedUser
from app.models import RequestTrace

logger = logging.getLogger("uvicorn")

admin_router = APIRouter()
mobile_router = APIRouter()


# ─── Admin endpoints ─────────────────────────────────────────────────────────

# Admin GET endpoints follow the same pattern as admin.py — no per-endpoint
# auth for reads; the admin page handles auth at login via JWT.


@admin_router.get("/traces")
def list_traces(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    household_id: Optional[str] = Query(None),
    node_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """List recent request traces (newest first).

    Returns rows without spans to keep the response lightweight.
    Use GET /traces/{trace_id} for full span detail.
    """
    query = db.query(RequestTrace).order_by(desc(RequestTrace.created_at))

    if status:
        query = query.filter(RequestTrace.status == status)
    if source:
        query = query.filter(RequestTrace.source == source)
    if household_id:
        query = query.filter(RequestTrace.household_id == household_id)
    if node_id:
        query = query.filter(RequestTrace.node_id == node_id)

    total = query.count()
    rows = query.offset(offset).limit(limit).all()

    traces = []
    for row in rows:
        try:
            span_count = len(json.loads(row.spans_json)) if row.spans_json else 0
        except (json.JSONDecodeError, TypeError):
            span_count = 0

        traces.append({
            "id": row.id,
            "conversation_id": row.conversation_id,
            "request_type": row.request_type,
            "source": row.source,
            "node_id": row.node_id,
            "household_id": row.household_id,
            "user_command": row.user_command,
            "assistant_message": row.assistant_message,
            "status": row.status,
            "total_duration_ms": row.total_duration_ms,
            "span_count": span_count,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        })

    return {"traces": traces, "total": total}


@admin_router.get("/traces/{trace_id}")
def get_trace(
    trace_id: str,
    db: Session = Depends(get_db),
):
    """Get a single trace with full span detail."""
    row = db.query(RequestTrace).filter(RequestTrace.id == trace_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Trace not found")

    try:
        spans = json.loads(row.spans_json) if row.spans_json else []
    except (json.JSONDecodeError, TypeError):
        spans = []

    return {
        "id": row.id,
        "conversation_id": row.conversation_id,
        "request_type": row.request_type,
        "source": row.source,
        "node_id": row.node_id,
        "household_id": row.household_id,
        "user_command": row.user_command,
        "assistant_message": row.assistant_message,
        "status": row.status,
        "error_message": row.error_message,
        "total_duration_ms": row.total_duration_ms,
        "spans": spans,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ─── Mobile endpoint ─────────────────────────────────────────────────────────


@mobile_router.get("/traces/{conversation_id}")
def get_trace_by_conversation(
    conversation_id: str,
    _user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
):
    """Get the most recent trace for a conversation (mobile).

    Returns the trace with full spans so the mobile waterfall can render.
    """
    row = (
        db.query(RequestTrace)
        .filter(RequestTrace.conversation_id == conversation_id)
        .order_by(desc(RequestTrace.created_at))
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="No trace for this conversation")

    try:
        spans = json.loads(row.spans_json) if row.spans_json else []
    except (json.JSONDecodeError, TypeError):
        spans = []

    return {
        "id": row.id,
        "conversation_id": row.conversation_id,
        "request_type": row.request_type,
        "source": row.source,
        "status": row.status,
        "error_message": row.error_message,
        "total_duration_ms": row.total_duration_ms,
        "spans": spans,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
