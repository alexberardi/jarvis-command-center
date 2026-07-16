"""Node-reported task status: POST /api/v0/nodes/tasks/{task_id}/status.

A node that refuses an update (the ``allow_updates`` consent gate) must be
able to mark its own task failed with the reason — otherwise the task dies
~15 minutes later as a misleading sweeper "Timeout: no heartbeat..." and
the operator never learns why (Jul-2026 prod kitchen incident).

Security pins:
  - node auth (X-API-Key) required; a node may only touch ITS OWN update task
  - state whitelist: nodes may only report ``failed``
  - terminal tasks are immutable (a late refusal must not overwrite
    "Cancelled by user")
  - cross-node / wrong-kind / unknown task all 404 identically (no
    existence leak)
"""
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.deps import get_db, verify_api_key
from app.main import app
from app.models import Node, NodeTask


_NODE_ID = "task-status-node"


@pytest.fixture
def node_client(test_db):
    """Client authenticated as node ``_NODE_ID`` via X-API-Key override."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_api_key():
        return SimpleNamespace(node=SimpleNamespace(node_id=_NODE_ID))

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_api_key] = override_api_key
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _make_node(test_db, node_id=_NODE_ID):
    node = Node(node_id=node_id, api_key=f"key-{node_id}", room="kitchen", user="alex")
    test_db.add(node)
    test_db.commit()
    return node


def _make_task(test_db, node_id=_NODE_ID, kind="update", state="dispatched"):
    task = NodeTask(
        id=str(uuid.uuid4()),
        node_id=node_id,
        kind=kind,
        target_version="0.2.0",
        state=state,
    )
    test_db.add(task)
    test_db.commit()
    test_db.refresh(task)
    return task


class TestReportTaskStatus:
    def test_node_marks_own_task_failed_with_reason(self, node_client, test_db):
        _make_node(test_db)
        task = _make_task(test_db)

        resp = node_client.post(
            f"/api/v0/nodes/tasks/{task.id}/status",
            json={"state": "failed", "error_message": "Updates are disabled on this node"},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "failed"
        assert "disabled" in body["error_message"]
        test_db.refresh(task)
        assert task.state == "failed"
        assert task.finished_at is not None

    def test_cannot_touch_another_nodes_task(self, node_client, test_db):
        _make_node(test_db, node_id="other-node")
        task = _make_task(test_db, node_id="other-node")

        resp = node_client.post(
            f"/api/v0/nodes/tasks/{task.id}/status",
            json={"state": "failed", "error_message": "x"},
        )

        assert resp.status_code == 404
        test_db.refresh(task)
        assert task.state == "dispatched"  # untouched

    def test_wrong_kind_404s_identically(self, node_client, test_db):
        _make_node(test_db)
        task = _make_task(test_db, kind="factory_reset")

        resp = node_client.post(
            f"/api/v0/nodes/tasks/{task.id}/status",
            json={"state": "failed", "error_message": "x"},
        )

        assert resp.status_code == 404

    def test_unknown_task_404s(self, node_client, test_db):
        _make_node(test_db)
        resp = node_client.post(
            f"/api/v0/nodes/tasks/{uuid.uuid4()}/status",
            json={"state": "failed", "error_message": "x"},
        )
        assert resp.status_code == 404

    def test_only_failed_is_accepted(self, node_client, test_db):
        _make_node(test_db)
        task = _make_task(test_db)

        resp = node_client.post(
            f"/api/v0/nodes/tasks/{task.id}/status",
            json={"state": "success", "error_message": "trust me"},
        )

        # Literal["failed"] rejects it at validation (CC maps validation
        # errors to 400 via its custom handler).
        assert resp.status_code in (400, 422)
        test_db.refresh(task)
        assert task.state == "dispatched"

    def test_terminal_task_is_immutable(self, node_client, test_db):
        _make_node(test_db)
        task = _make_task(test_db, state="failed")
        task.error_message = "Cancelled by user"
        task.finished_at = datetime.utcnow()
        test_db.commit()

        resp = node_client.post(
            f"/api/v0/nodes/tasks/{task.id}/status",
            json={"state": "failed", "error_message": "Updates are disabled"},
        )

        assert resp.status_code == 409
        test_db.refresh(task)
        assert task.error_message == "Cancelled by user"  # not overwritten
