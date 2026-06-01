"""Interactive-notification callback API.

Three endpoints, three audiences:

  POST /api/v0/callbacks                  (mobile, user JWT)
    Mobile user taps a tappable element in a rich inbox item. The body carries
    {command, callback, data, target_node_id}. We create a CallbackJob row,
    publish [{"command": "callback", "details": {"request_id": id}}] to the
    node's commands topic via NodeCommandService, and return the job id.

  GET /api/v0/callbacks/{job_id}          (node, X-API-Key)
    Node received an opaque request_id over MQTT. It fetches the full payload
    here. We verify the authenticating node owns the job and return
    {command_name, callback_name, data, user_id, ...}.

  POST /api/v0/callbacks/{job_id}/result  (node, X-API-Key)
    Node has dispatched the callback and posts the result back. Body is
    {success, error, context_data}. We mark the job completed/failed.

The MQTT layer carries only the opaque id — every sensitive piece of state
flows over authenticated HTTPS. Follow-up inbox creation (e.g., a new card
with an actor's filmography after expand_actor) is the callback method's
responsibility via the existing notifications service — this route only
records that the dispatch happened.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

NavigationType = Literal["stack", "new_notification", "popover"]

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import (
    AuthenticatedUser,
    get_db,
    verify_api_key,
    verify_household_role,
    verify_user_jwt,
)
from app.models import CallbackJob, Node
from app.services.node_command_service import get_node_command_service

router = APIRouter()
logger = logging.getLogger("uvicorn")


CALLBACK_TTL = timedelta(minutes=5)


# =============================================================================
# Request / Response models
# =============================================================================


class CallbackCreateBody(BaseModel):
    command_name: str = Field(..., min_length=1, max_length=128)
    callback_name: str = Field(..., min_length=1, max_length=128)
    data: dict[str, Any] = Field(default_factory=dict)
    target_node_id: str = Field(..., min_length=1)
    # Mobile renderer hint — see `CallbackJob.navigation_type`. Mobile is
    # the source of truth; CC persists the choice so the result endpoint
    # knows whether to also create an inbox item server-side.
    navigation_type: NavigationType = "new_notification"


class CallbackCreateResponse(BaseModel):
    id: str
    status: str
    navigation_type: str
    created_at: datetime


class CallbackPayloadResponse(BaseModel):
    """Returned to the node when it GETs the full payload by id."""
    job_id: str
    command_name: str
    callback_name: str
    data: dict[str, Any]
    user_id: int | None = None
    voice_command: str | None = None
    conversation_id: str | None = None


class CallbackResultBody(BaseModel):
    success: bool
    error: str | None = None
    context_data: dict[str, Any] | None = None


class CallbackResultResponse(BaseModel):
    id: str
    status: str
    completed_at: datetime


class CallbackStatusResponse(BaseModel):
    """User-JWT'd mobile poll of a callback's progress + final result.

    Mobile pushes a screen on tap (for navigation_type=stack/popover),
    polls this until status leaves "pending", then renders the
    ``context_data`` inline. For navigation_type=new_notification the
    follow-up inbox item is the canonical surface; this endpoint still
    works for status checks but mobile typically doesn't need to poll.
    """
    id: str
    status: str
    navigation_type: str
    completed_at: datetime | None = None
    error_message: str | None = None
    context_data: dict[str, Any] | None = None


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/callbacks",
    response_model=CallbackCreateResponse,
    status_code=201,
)
def create_callback(
    body: CallbackCreateBody,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> CallbackCreateResponse:
    """Mobile-app entry point: register a callback job and signal the node."""
    node = db.query(Node).filter(Node.node_id == body.target_node_id).first()
    if node is None:
        raise HTTPException(status_code=404, detail="Target node not found")

    household_id = node.household_id or ""
    if not household_id:
        # Defense in depth — a node without a household can't route callbacks.
        raise HTTPException(status_code=400, detail="Target node has no household")

    # Membership check: the user must belong to the node's household.
    verify_household_role(user.user_id, household_id, required_role="member")

    now = datetime.utcnow()
    job = CallbackJob(
        id=str(uuid4()),
        node_id=body.target_node_id,
        household_id=household_id,
        user_id=user.user_id,
        command_name=body.command_name,
        callback_name=body.callback_name,
        data_json=json.dumps(body.data),
        navigation_type=body.navigation_type,
        status="pending",
        created_at=now,
        expires_at=now + CALLBACK_TTL,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Publish [{"command": "callback", "details": {"request_id": job.id}}] —
    # zero payload beyond the opaque id. Node will GET the full body.
    get_node_command_service().publish_command_with_id(
        node_id=body.target_node_id,
        command="callback",
        details=None,
        request_id=job.id,
    )

    logger.info(
        "Callback job created job_id=%s node=%s command=%s callback=%s nav=%s user=%s",
        job.id[:8], body.target_node_id, body.command_name, body.callback_name,
        body.navigation_type, user.user_id,
    )

    return CallbackCreateResponse(
        id=job.id,
        status=job.status,
        navigation_type=job.navigation_type,
        created_at=job.created_at,
    )


@router.get(
    "/callbacks/{job_id}",
    response_model=CallbackPayloadResponse,
)
def get_callback_payload(
    job_id: str,
    node_ctx: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> CallbackPayloadResponse:
    """Node fetches the full payload it needs to dispatch.

    Ownership check: only the node named in ``target_node_id`` can read it.
    Returning 404 (rather than 403) avoids leaking job-id existence to
    unauthorized nodes.
    """
    job = db.query(CallbackJob).filter(CallbackJob.id == job_id).first()
    if job is None or job.node_id != node_ctx.node.node_id:
        raise HTTPException(status_code=404, detail="Callback job not found")

    if job.status not in ("pending", "completed", "failed"):
        # "expired" or any unknown future state — treat as gone.
        raise HTTPException(status_code=410, detail=f"Callback job is {job.status}")

    if datetime.utcnow() > job.expires_at and job.status == "pending":
        # Mark expired on read so the dashboard / cleanup task sees it.
        job.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Callback job expired")

    try:
        data: dict[str, Any] = json.loads(job.data_json) if job.data_json else {}
    except json.JSONDecodeError:
        logger.error("Callback job %s has malformed data_json", job_id[:8])
        data = {}

    return CallbackPayloadResponse(
        job_id=job.id,
        command_name=job.command_name,
        callback_name=job.callback_name,
        data=data,
        user_id=job.user_id,
        voice_command=f"cb:{job.callback_name}",
        conversation_id=f"callback:{job.id}",
    )


@router.post(
    "/callbacks/{job_id}/result",
    response_model=CallbackResultResponse,
)
def post_callback_result(
    job_id: str,
    body: CallbackResultBody,
    node_ctx: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> CallbackResultResponse:
    """Node reports the dispatch outcome.

    For navigation_type="new_notification" we *also* create an inbox item
    server-side from the result's ``context_data["inbox"]`` block — the
    callback method only returns the renderable content; CC decides
    whether to fan it out to the inbox surface based on the original
    mobile tap's navigation choice. For "stack"/"popover" the mobile is
    polling the status endpoint and renders the same content inline.
    """
    job = db.query(CallbackJob).filter(CallbackJob.id == job_id).first()
    if job is None or job.node_id != node_ctx.node.node_id:
        raise HTTPException(status_code=404, detail="Callback job not found")

    now = datetime.utcnow()
    job.status = "completed" if body.success else "failed"
    job.error_message = body.error
    job.completed_at = now
    if body.context_data is not None:
        try:
            job.result_context_data_json = json.dumps(body.context_data)
        except (TypeError, ValueError):
            logger.warning(
                "Callback job %s result context_data not JSON-serializable, dropping",
                job_id[:8],
            )
            job.result_context_data_json = None

    db.commit()
    db.refresh(job)

    # navigation_type drives whether to also fan out to the inbox surface.
    # "new_notification" is the async path — produce a separate inbox row
    # so the user sees it land in the feed. "stack"/"popover" are inline
    # surfaces; the mobile screen polls this job and renders directly.
    if (
        job.navigation_type == "new_notification"
        and body.success
        and isinstance(body.context_data, dict)
    ):
        inbox = body.context_data.get("inbox")
        if isinstance(inbox, dict) and isinstance(inbox.get("title"), str) and inbox["title"]:
            try:
                from app.services.inbox_notification_service import post_inbox_item_sync
                metadata = dict(inbox.get("metadata") or {})
                metadata.setdefault("node_id", node_ctx.node.node_id)
                post_inbox_item_sync(
                    household_id=job.household_id,
                    user_id=job.user_id,
                    title=inbox["title"],
                    summary=inbox.get("summary") or "",
                    body=inbox.get("body") or "",
                    category=inbox.get("category") or "callback_result",
                    metadata=metadata,
                    push=bool(inbox.get("create_push_notification", False)),
                )
            except Exception as e:
                # Inbox creation is best-effort — the result still records.
                logger.warning(
                    "Callback %s: inbox fan-out failed: %s", job_id[:8], e,
                )

    logger.info(
        "Callback job result job_id=%s status=%s nav=%s error=%r",
        job_id[:8], job.status, job.navigation_type, body.error,
    )

    return CallbackResultResponse(
        id=job.id,
        status=job.status,
        completed_at=job.completed_at,
    )


@router.get(
    "/callbacks/{job_id}/status",
    response_model=CallbackStatusResponse,
)
def get_callback_status(
    job_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> CallbackStatusResponse:
    """Mobile poll of a callback's progress.

    Distinct from the node-authed GET /callbacks/{job_id} (which returns
    the dispatch payload). Mobile pushes a screen on tap (when
    navigation_type is "stack" or "popover"), then polls here until the
    job leaves "pending" and renders ``context_data`` inline.
    """
    job = db.query(CallbackJob).filter(CallbackJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Callback job not found")

    # Membership gate — user must belong to the job's household.
    verify_household_role(user.user_id, job.household_id, required_role="member")

    if datetime.utcnow() > job.expires_at and job.status == "pending":
        job.status = "expired"
        db.commit()

    context_data: dict[str, Any] | None = None
    if job.result_context_data_json:
        try:
            context_data = json.loads(job.result_context_data_json)
        except json.JSONDecodeError:
            logger.warning(
                "Callback job %s has malformed result_context_data_json", job_id[:8],
            )

    return CallbackStatusResponse(
        id=job.id,
        status=job.status,
        navigation_type=job.navigation_type,
        completed_at=job.completed_at,
        error_message=job.error_message,
        context_data=context_data,
    )
