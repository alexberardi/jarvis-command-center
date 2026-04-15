"""Node update dispatch + task status API.

Consumed by the mobile app. The actual update is delivered to the node via
the existing heartbeat channel (see admin.node_heartbeat) — this module
creates the task row; the heartbeat handler returns it as `pending_update`
the next time the node checks in, and reconciles success when the node
reports its new version.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ..deps import get_db, verify_admin_key, verify_user_jwt
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


# ── Helpers ─────────────────────────────────────────────────────────────

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
):
    """Queue an update for a node. Node picks it up on its next heartbeat."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(404, "Node not found")

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
    target_version = resolve_target_version(requested)
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


@router.get(
    "/tasks/{task_id}",
    response_model=NodeTaskResponse,
    dependencies=[Depends(verify_user_jwt)],
)
def get_task(task_id: str, db: Session = Depends(get_db)):
    task = db.query(NodeTask).filter(NodeTask.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")
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
):
    """Most recent tasks for a node. Used by the mobile Update History view."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(404, "Node not found")

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

    If an update was dispatched and the node now reports the target version
    (or at least a different version from before), mark it complete. Tasks
    stuck in `dispatched` / `in_progress` for more than TASK_TIMEOUT are
    marked failed by the background sweeper, not here.
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

    # Node has come back online but not at the expected version yet — keep
    # the task open. If it stays in this state past the timeout, the
    # sweeper marks it failed.
    if (datetime.utcnow() - open_task.updated_at) < timedelta(seconds=30):
        # Very recent dispatch: the node may still be mid-upgrade, do nothing.
        return
    open_task.state = "in_progress"
    open_task.updated_at = datetime.utcnow()
    db.commit()
