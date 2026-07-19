"""Interactive-notification callback API.

Two dispatch planes, selected by the create body:

**Node plane** (``target_node_id`` set — the original flow): the tapped
element belongs to a command that lives on a node. We create a CallbackJob
row, publish [{"command": "callback", "details": {"request_id": id}}] to the
node's commands topic via NodeCommandService; the node GETs the payload here
(X-API-Key), dispatches the command's @callback method, and POSTs the result
back. MQTT carries only the opaque id — every sensitive piece of state flows
over authenticated HTTPS.

**Server plane** (``target_node_id`` omitted): the element belongs to a CC
server tool (deep-research follow-ups, phone-call confirm/escalation cards)
— there is no node. The create body must carry ``household_id`` instead
(membership verified the same way), the handler must be pre-registered in
``app.services.server_callback_registry``, and CC executes it in a
background task immediately after the 201. The handler's result is recorded
through the same machinery as a node-posted result, so the mobile status
poll and the navigation_type=new_notification inbox fan-out behave
identically on both planes. Nodes can never read or complete a server-plane
job (the ownership check treats node_id=NULL as "no node matches").

Endpoints:

  POST /api/v0/callbacks                  (mobile, user JWT) — both planes
  GET  /api/v0/callbacks/{job_id}         (node, X-API-Key) — node plane only
  POST /api/v0/callbacks/{job_id}/result  (node, X-API-Key) — node plane only
  GET  /api/v0/callbacks/{job_id}/status  (mobile, user JWT) — both planes

Follow-up inbox creation (e.g., a new card with an actor's filmography after
expand_actor) is the callback handler's responsibility via the existing
notifications service — this route only records that the dispatch happened.
"""

import inspect
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
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
from app.services.server_callback_registry import (
    ServerCallbackContext,
    ServerCallbackResult,
    get_server_callback,
)

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
    # Node plane: the node that owns the command executes the callback.
    # Omitted → server plane: CC executes a registered handler in-process,
    # and household_id becomes required (no node row to derive it from).
    target_node_id: str | None = Field(default=None, min_length=1)
    household_id: str | None = Field(default=None, min_length=1)
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


def _create_job_row(
    db: Session,
    body: CallbackCreateBody,
    *,
    node_id: str | None,
    household_id: str,
    user_id: int,
) -> CallbackJob:
    now = datetime.utcnow()
    job = CallbackJob(
        id=str(uuid4()),
        node_id=node_id,
        household_id=household_id,
        user_id=user_id,
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
    return job


@router.post(
    "/callbacks",
    response_model=CallbackCreateResponse,
    status_code=201,
)
def create_callback(
    body: CallbackCreateBody,
    background_tasks: BackgroundTasks,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> CallbackCreateResponse:
    """Mobile-app entry point: register a callback job and dispatch it.

    target_node_id set → node plane (MQTT signal, node executes).
    target_node_id omitted → server plane (CC executes a registered handler
    in a background task; household_id required in the body).
    """
    if body.target_node_id is None:
        return _create_server_callback(body, background_tasks, user, db)

    node = db.query(Node).filter(Node.node_id == body.target_node_id).first()
    if node is None:
        raise HTTPException(status_code=404, detail="Target node not found")

    household_id = node.household_id or ""
    if not household_id:
        # Defense in depth — a node without a household can't route callbacks.
        raise HTTPException(status_code=400, detail="Target node has no household")

    # Membership check: the user must belong to the node's household.
    verify_household_role(user.user_id, household_id, required_role="member")

    job = _create_job_row(
        db, body, node_id=body.target_node_id,
        household_id=household_id, user_id=user.user_id,
    )

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


def _create_server_callback(
    body: CallbackCreateBody,
    background_tasks: BackgroundTasks,
    user: AuthenticatedUser,
    db: Session,
) -> CallbackCreateResponse:
    """Server-plane branch: CC executes a registered handler itself.

    Handler existence is checked BEFORE the row is created — an unroutable
    tap should fail loudly at the tap, not sit pending until TTL.
    """
    if get_server_callback(body.command_name, body.callback_name) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No server-side handler registered for "
                f"{body.command_name}.{body.callback_name}"
            ),
        )

    if not body.household_id:
        raise HTTPException(
            status_code=400,
            detail="household_id is required when target_node_id is omitted",
        )

    # Same membership bar as the node plane.
    verify_household_role(user.user_id, body.household_id, required_role="member")

    job = _create_job_row(
        db, body, node_id=None,
        household_id=body.household_id, user_id=user.user_id,
    )

    background_tasks.add_task(_run_server_callback_job, job.id)

    logger.info(
        "Server callback job created job_id=%s command=%s callback=%s nav=%s user=%s",
        job.id[:8], body.command_name, body.callback_name,
        body.navigation_type, user.user_id,
    )

    return CallbackCreateResponse(
        id=job.id,
        status=job.status,
        navigation_type=job.navigation_type,
        created_at=job.created_at,
    )


def _server_callback_session_factory() -> Session:
    """Fresh session for the background task (module-level so tests patch it)."""
    from app.db import get_session_local

    return get_session_local()()


async def _run_server_callback_job(job_id: str) -> None:
    """Background task: execute the registered handler and record the result.

    Never raises — any failure lands on the job row as status=failed so the
    mobile status poll always terminates (the anti-vanishing rule).
    """
    db = _server_callback_session_factory()
    try:
        job = db.query(CallbackJob).filter(CallbackJob.id == job_id).first()
        if job is None:
            logger.error("Server callback job %s vanished before execution", job_id[:8])
            return

        handler = get_server_callback(job.command_name, job.callback_name)
        if handler is None:
            # Registration disappeared between create and execution (module
            # reload) — record the failure rather than leaving it pending.
            _record_callback_result(
                db, job, success=False,
                error="Server-side handler no longer registered",
                context_data=None, source_node_id=None,
            )
            return

        try:
            data: dict[str, Any] = json.loads(job.data_json) if job.data_json else {}
        except json.JSONDecodeError:
            logger.error("Server callback job %s has malformed data_json", job_id[:8])
            data = {}

        try:
            outcome = handler(
                ServerCallbackContext(
                    job_id=job.id,
                    household_id=job.household_id,
                    user_id=job.user_id,
                    data=data,
                    navigation_type=job.navigation_type,
                )
            )
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if not isinstance(outcome, ServerCallbackResult):
                raise TypeError(
                    f"handler returned {type(outcome).__name__}, "
                    "expected ServerCallbackResult"
                )
        except Exception as e:  # noqa: BLE001 — handler failures land on the row
            logger.exception(
                "Server callback %s.%s failed (job %s)",
                job.command_name, job.callback_name, job_id[:8],
            )
            _record_callback_result(
                db, job, success=False, error=str(e),
                context_data=None, source_node_id=None,
            )
            return

        _record_callback_result(
            db, job,
            success=outcome.success,
            error=outcome.error,
            context_data=outcome.context_data,
            source_node_id=None,
        )
    finally:
        db.close()


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

    _record_callback_result(
        db, job,
        success=body.success,
        error=body.error,
        context_data=body.context_data,
        source_node_id=node_ctx.node.node_id,
    )

    return CallbackResultResponse(
        id=job.id,
        status=job.status,
        completed_at=job.completed_at,
    )


def _record_callback_result(
    db: Session,
    job: CallbackJob,
    *,
    success: bool,
    error: str | None,
    context_data: dict[str, Any] | None,
    source_node_id: str | None,
) -> None:
    """Shared result recorder for both planes: mark the row, then fan out.

    ``source_node_id`` is the executing node for node-plane results (stamped
    into inbox metadata); None for server-plane results.
    """
    job.status = "completed" if success else "failed"
    job.error_message = error
    job.completed_at = datetime.utcnow()
    if context_data is not None:
        try:
            job.result_context_data_json = json.dumps(context_data)
        except (TypeError, ValueError):
            logger.warning(
                "Callback job %s result context_data not JSON-serializable, dropping",
                job.id[:8],
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
        and success
        and isinstance(context_data, dict)
    ):
        inbox = context_data.get("inbox")
        if isinstance(inbox, dict) and isinstance(inbox.get("title"), str) and inbox["title"]:
            try:
                from app.services.inbox_notification_service import post_inbox_item_sync
                metadata = dict(inbox.get("metadata") or {})
                if source_node_id is not None:
                    metadata.setdefault("node_id", source_node_id)
                # Callbacks always have a user_id (the tapping user from
                # the JWT on the original POST /callbacks). Push them
                # user-scoped by default so only the user who tapped
                # gets the buzz, not the whole household.
                target_type = inbox.get("target_type")
                if target_type not in ("user", "household"):
                    target_type = "user" if job.user_id is not None else "household"
                post_inbox_item_sync(
                    household_id=job.household_id,
                    user_id=job.user_id,
                    title=inbox["title"],
                    summary=inbox.get("summary") or "",
                    body=inbox.get("body") or "",
                    category=inbox.get("category") or "callback_result",
                    metadata=metadata,
                    push=bool(inbox.get("create_push_notification", False)),
                    target_type=target_type,
                )
            except Exception as e:
                # Inbox creation is best-effort — the result still records.
                logger.warning(
                    "Callback %s: inbox fan-out failed: %s", job.id[:8], e,
                )

    logger.info(
        "Callback job result job_id=%s status=%s nav=%s error=%r",
        job.id[:8], job.status, job.navigation_type, error,
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
