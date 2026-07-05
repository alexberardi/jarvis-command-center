"""Node update dispatch + task status API.

Consumed by the mobile app. The actual update is delivered to the node via
the existing heartbeat channel (see admin.node_heartbeat) — this module
creates the task row; the heartbeat handler returns it as `pending_update`
the next time the node checks in, and reconciles success when the node
reports its new version.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ..context_providers.node_context_provider import NodeContextProvider
from ..deps import (
    AuthenticatedUser,
    get_db,
    verify_admin_key,
    verify_api_key,
    verify_household_role,
    verify_user_jwt,
)
from ..models import Node, NodeTask
from ..services.github_releases import latest_release, resolve_target_version


router = APIRouter()


# ── Schemas ─────────────────────────────────────────────────────────────

class UpdateNodeRequest(BaseModel):
    target_version: str | None = None  # "latest" (default), or "vX.Y.Z" / "X.Y.Z"


class NodeTaskResponse(BaseModel):
    id: str
    node_id: str
    kind: str
    target_version: str | None
    state: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class LatestReleaseResponse(BaseModel):
    tag: str
    version: str
    published_at: str | None


class NodeTaskStatusRequest(BaseModel):
    # Nodes may only report terminal FAILURE (e.g. the allow_updates consent
    # gate refused the task). Success stays inferred from the post-upgrade
    # heartbeat version — a node must not be able to fake a completed update.
    state: Literal["failed"]
    error_message: str | None = None


# ── Helpers ─────────────────────────────────────────────────────────────

def _require_node_household(db: Session, node_id: str, user: AuthenticatedUser) -> None:
    """A JWT user may only touch a node/task in their own household."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(404, "Node not found")
    if node.household_id:
        verify_household_role(user.user_id, node.household_id, required_role="member")


def _dependent_task(db: Session, node_id: str) -> NodeTask | None:
    """Return an open update task for this node, if any.

    Used to reject new update requests while one is already queued or in
    flight — the mobile UI should fall back to polling the existing task.
    """
    return (
        db.query(NodeTask)
        .filter(
            NodeTask.node_id == node_id,
            NodeTask.kind == "update",
            NodeTask.state.in_(["pending", "dispatched", "in_progress"]),
        )
        .order_by(NodeTask.created_at.desc())
        .first()
    )


# ── Routes ──────────────────────────────────────────────────────────────

@router.get("/releases/latest", response_model=LatestReleaseResponse | None)
def get_latest_release():
    """Thin passthrough to jarvis-node-setup's latest GitHub release.

    Returns None if GitHub is unreachable or no release exists; the mobile
    app can fall back to just showing current_version.
    """
    info = latest_release()
    if info is None:
        return None
    return LatestReleaseResponse(
        tag=info.tag,
        version=info.version,
        published_at=info.published_at,
    )


@router.post(
    "/nodes/{node_id}/update",
    response_model=NodeTaskResponse,
    dependencies=[Depends(verify_user_jwt)],
)
def request_node_update(
    node_id: str,
    body: UpdateNodeRequest | None = None,
    db: Session = Depends(get_db),
    user: AuthenticatedUser = Depends(verify_user_jwt),
):
    """Queue an update for a node. Node picks it up on its next heartbeat."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(404, "Node not found")
    if node.household_id:
        verify_household_role(user.user_id, node.household_id, required_role="member")

    existing = _dependent_task(db, node_id)
    if existing is not None:
        raise HTTPException(
            409,
            {
                "message": "An update is already queued for this node.",
                "task_id": existing.id,
                "state": existing.state,
            },
        )

    requested = (body.target_version if body else None) or "latest"
    # An explicit "vX.Y.Z" / "X.Y.Z" bypasses the GitHub lookup entirely (no
    # egress); only "latest"/None hits api.github.com, gated per-household on
    # updates.allow_check (503 below when disabled/unreachable and no explicit
    # version was requested).
    target_version = resolve_target_version(requested, household_id=node.household_id)
    if target_version is None:
        raise HTTPException(
            503,
            "Could not determine target version (GitHub releases unreachable "
            "and no explicit version was requested).",
        )

    task = NodeTask(
        node_id=node_id,
        kind="update",
        target_version=target_version,
        state="pending",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.post(
    "/nodes/tasks/{task_id}/status",
    response_model=NodeTaskResponse,
)
def report_task_status(
    task_id: str,
    body: NodeTaskStatusRequest,
    node_context: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Node-reported terminal task status.

    Lets a node surface WHY a task ended — e.g. the ``allow_updates``
    consent gate refused it — instead of leaving the task to die as a
    sweeper timeout with a misleading "Timeout: no heartbeat" message
    (Jul-2026 prod). A node may only fail its OWN update tasks, and
    terminal tasks are immutable (a late refusal must not overwrite
    "Cancelled by user").
    """
    task = db.query(NodeTask).filter(NodeTask.id == task_id).first()
    if (
        not task
        or task.node_id != node_context.node.node_id
        or task.kind != "update"
    ):
        # One 404 for all three: don't leak other nodes' task IDs.
        raise HTTPException(404, "Task not found")
    if task.state in ("success", "failed"):
        raise HTTPException(409, f"Task is already {task.state}")
    task.state = "failed"
    task.error_message = (body.error_message or "Failed (reported by node)")[:1000]
    task.finished_at = datetime.utcnow()
    task.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(task)
    return task


@router.get(
    "/tasks/{task_id}",
    response_model=NodeTaskResponse,
    dependencies=[Depends(verify_user_jwt)],
)
def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    user: AuthenticatedUser = Depends(verify_user_jwt),
):
    task = db.query(NodeTask).filter(NodeTask.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")
    _require_node_household(db, task.node_id, user)
    return task


@router.post(
    "/nodes/{node_id}/tasks/{task_id}/cancel",
    response_model=NodeTaskResponse,
    dependencies=[Depends(verify_user_jwt)],
)
def cancel_node_task(
    node_id: str,
    task_id: str,
    db: Session = Depends(get_db),
    user: AuthenticatedUser = Depends(verify_user_jwt),
):
    """Manually cancel an open node task.

    The sweeper already times out stale tasks, but waiting 10-15 min is
    bad UX when the user knows the install is dead. This lets the mobile
    "Cancel" button immediately mark the task ``failed`` so a follow-up
    update request isn't rejected with 409.
    """
    task = (
        db.query(NodeTask)
        .filter(NodeTask.id == task_id, NodeTask.node_id == node_id)
        .first()
    )
    if not task:
        raise HTTPException(404, "Task not found")
    _require_node_household(db, node_id, user)
    if task.state in ("success", "failed"):
        raise HTTPException(409, f"Task is already {task.state}")
    task.state = "failed"
    task.error_message = "Cancelled by user"
    task.finished_at = datetime.utcnow()
    task.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(task)
    return task


@router.get(
    "/nodes/{node_id}/tasks",
    response_model=list[NodeTaskResponse],
    dependencies=[Depends(verify_user_jwt)],
)
def list_node_tasks(
    node_id: str,
    limit: int = 20,
    db: Session = Depends(get_db),
    user: AuthenticatedUser = Depends(verify_user_jwt),
):
    """Most recent tasks for a node. Used by the mobile Update History view."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(404, "Node not found")
    if node.household_id:
        verify_household_role(user.user_id, node.household_id, required_role="member")

    limit = max(1, min(limit, 100))
    rows = (
        db.query(NodeTask)
        .filter(NodeTask.node_id == node_id)
        .order_by(NodeTask.created_at.desc())
        .limit(limit)
        .all()
    )
    return rows


# ── Admin helpers (for CC internal use) ─────────────────────────────────

def dispatch_pending_task(db: Session, node: Node) -> dict | None:
    """If the node has a pending update task and isn't busy, flip it to
    "dispatched" and return the payload the node should act on. Returns
    None if there's nothing to do.

    Called from the heartbeat handler — this is the channel that actually
    gets an update to the node.
    """
    if node.is_busy:
        return None

    task = (
        db.query(NodeTask)
        .filter(
            NodeTask.node_id == node.node_id,
            NodeTask.kind == "update",
            NodeTask.state == "pending",
        )
        .order_by(NodeTask.created_at.asc())
        .first()
    )
    if task is None:
        return None

    task.state = "dispatched"
    task.updated_at = datetime.utcnow()
    db.commit()
    return {
        "task_id": task.id,
        "target_version": task.target_version,
    }


def reconcile_open_task(db: Session, node: Node, reported_version: Optional[str]) -> None:
    """Match post-upgrade heartbeat to any open task for this node.

    Three outcomes:
      - Node reports the target version → mark task success.
      - Node has been in `dispatched` state for >30s and is heartbeating
        with the OLD version → transition to `in_progress` (one-time state
        change, signals to the mobile app that the install is underway).
      - Already `in_progress` and still reporting the OLD version → do
        nothing. Critically, this path does NOT bump updated_at: if the
        installer has died silently (OOM, power loss, ...) the node keeps
        heartbeating with the old version indefinitely, and we need the
        sweeper's staleness check to be able to see that nothing is
        actually progressing.
    """
    if reported_version is None:
        return

    open_task = (
        db.query(NodeTask)
        .filter(
            NodeTask.node_id == node.node_id,
            NodeTask.kind == "update",
            NodeTask.state.in_(["dispatched", "in_progress"]),
        )
        .order_by(NodeTask.created_at.desc())
        .first()
    )
    if open_task is None:
        return

    if open_task.target_version and reported_version == open_task.target_version:
        open_task.state = "success"
        open_task.finished_at = datetime.utcnow()
        open_task.updated_at = datetime.utcnow()
        db.commit()
        return

    # Node is heartbeating with the old version. Only transition the state
    # machine on the first such heartbeat after dispatch; every subsequent
    # heartbeat that reports the same old version is a non-event.
    if open_task.state == "dispatched":
        if (datetime.utcnow() - open_task.updated_at) < timedelta(seconds=30):
            # Very recent dispatch: node may still be stopping the old
            # service before the installer starts — do nothing.
            return
        open_task.state = "in_progress"
        open_task.updated_at = datetime.utcnow()
        db.commit()
        return

    # Already in_progress, still reporting old version — no state change,
    # no updated_at bump. Sweeper decides when this counts as stale.
