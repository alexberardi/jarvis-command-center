"""Tests for the factory-reset flow.

Covers:
- POST /admin/nodes/{node_id}/factory-reset (mobile-facing, JWT-auth)
- POST /nodes/factory-reset/{task_id}/status (node-facing, token-auth)
- list_nodes filtering by is_active
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.deps import get_db, verify_user_jwt
from app.main import app
from app.models import Node, NodeTask
from app.core import pending_resets


# =============================================================================
# Fixtures
# =============================================================================


class _FakeUser:
    """Stand-in for AuthenticatedUser in tests."""
    def __init__(self, *, user_id: str = "test-user-1", is_superuser: bool = True):
        self.user_id = user_id
        self.is_superuser = is_superuser


@pytest.fixture
def reset_client(test_db):
    """TestClient with DB + JWT overrides + household-role no-op + leftover-token cleanup.

    `verify_household_role` is a plain function call (not a FastAPI dep),
    so we patch the symbol in `app.admin` rather than using
    `dependency_overrides`. MQTT is patched in each test that cares.
    """
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_user():
        return _FakeUser()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_user_jwt] = override_user

    with patch("app.admin.verify_household_role", lambda *a, **k: None):
        try:
            with TestClient(app) as client:
                yield client
        finally:
            app.dependency_overrides.clear()
            with pending_resets._lock:
                pending_resets._pending.clear()


def _create_node(db, *, household_id: str | None = "hh-test", is_active: bool = True) -> Node:
    uid = uuid.uuid4().hex[:8]
    node = Node(
        node_id=f"test-node-{uid}",
        api_key=f"test-key-{uid}",
        room="office",
        user="test-user",
        household_id=household_id,
        is_active=is_active,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return node


# =============================================================================
# POST /admin/nodes/{node_id}/factory-reset
# =============================================================================


class TestCreateFactoryReset:

    def test_creates_task_and_publishes_mqtt(self, reset_client, test_db):
        node = _create_node(test_db)
        mqtt = MagicMock()
        with patch("app.admin.get_mqtt_client", return_value=mqtt, create=True), \
             patch("app.node_settings.get_mqtt_client", return_value=mqtt):
            resp = reset_client.post(
                f"/api/v0/admin/nodes/{node.node_id}/factory-reset"
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "task_id" in body
        assert "reset_token" in body

        # Task row exists with kind=factory_reset, state=dispatched (MQTT succeeded).
        task = test_db.query(NodeTask).filter(NodeTask.id == body["task_id"]).one()
        assert task.kind == "factory_reset"
        assert task.state == "dispatched"
        assert task.node_id == node.node_id

        # MQTT was published to the right topic.
        assert mqtt.publish.called
        topic = mqtt.publish.call_args[0][0]
        assert topic == f"jarvis/nodes/{node.node_id}/factory-reset"

        # Token is registered in pending_resets store.
        assert pending_resets.verify_token(body["reset_token"])

    def test_task_stays_pending_when_mqtt_unavailable(self, reset_client, test_db):
        node = _create_node(test_db)
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            resp = reset_client.post(
                f"/api/v0/admin/nodes/{node.node_id}/factory-reset"
            )
        assert resp.status_code == 200
        task = test_db.query(NodeTask).filter(
            NodeTask.id == resp.json()["task_id"]
        ).one()
        assert task.state == "pending"

    def test_404_for_unknown_node(self, reset_client):
        resp = reset_client.post("/api/v0/admin/nodes/does-not-exist/factory-reset")
        assert resp.status_code == 404

    def test_409_when_reset_already_in_flight(self, reset_client, test_db):
        node = _create_node(test_db)
        # Pre-seed a pending factory-reset task.
        existing = NodeTask(
            node_id=node.node_id, kind="factory_reset", state="pending"
        )
        test_db.add(existing)
        test_db.commit()

        with patch("app.node_settings.get_mqtt_client", return_value=None):
            resp = reset_client.post(
                f"/api/v0/admin/nodes/{node.node_id}/factory-reset"
            )
        assert resp.status_code == 409
        body = resp.json()
        assert body["detail"]["task_id"] == existing.id


# =============================================================================
# POST /nodes/factory-reset/{task_id}/status
# =============================================================================


class TestUpdateFactoryResetStatus:

    def _begin_reset(self, client, db, *, household_id: str | None = "hh-test") -> tuple[str, str, Node]:
        """Helper: create node, kick off factory reset, return (task_id, token, node)."""
        node = _create_node(db, household_id=household_id)
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            resp = client.post(
                f"/api/v0/admin/nodes/{node.node_id}/factory-reset"
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        return body["task_id"], body["reset_token"], node

    def test_in_progress_updates_state(self, reset_client, test_db):
        task_id, token, _ = self._begin_reset(reset_client, test_db)
        resp = reset_client.post(
            f"/api/v0/nodes/factory-reset/{task_id}/status",
            json={"state": "in_progress"},
            headers={"X-Reset-Token": token},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "in_progress"
        assert body["finished_at"] is None

    def test_success_marks_node_inactive_and_deactivates_auth(self, reset_client, test_db):
        task_id, token, node = self._begin_reset(reset_client, test_db)
        with patch("app.admin._deactivate_node_with_auth", return_value=True) as deact:
            resp = reset_client.post(
                f"/api/v0/nodes/factory-reset/{task_id}/status",
                json={"state": "success"},
                headers={"X-Reset-Token": token},
            )
        assert resp.status_code == 200
        assert resp.json()["state"] == "success"
        assert resp.json()["finished_at"] is not None

        test_db.refresh(node)
        assert node.is_active is False
        deact.assert_called_once_with(node.node_id)

    def test_failed_marks_finished_but_keeps_node_active(self, reset_client, test_db):
        task_id, token, node = self._begin_reset(reset_client, test_db)
        with patch("app.admin._deactivate_node_with_auth") as deact:
            resp = reset_client.post(
                f"/api/v0/nodes/factory-reset/{task_id}/status",
                json={"state": "failed", "error_message": "wifi reset failed"},
                headers={"X-Reset-Token": token},
            )
        assert resp.status_code == 200
        assert resp.json()["state"] == "failed"
        assert resp.json()["error_message"] == "wifi reset failed"

        test_db.refresh(node)
        # Failed reset must NOT mark the node inactive — operator may retry.
        assert node.is_active is True
        deact.assert_not_called()

    def test_invalid_state_400(self, reset_client, test_db):
        task_id, token, _ = self._begin_reset(reset_client, test_db)
        resp = reset_client.post(
            f"/api/v0/nodes/factory-reset/{task_id}/status",
            json={"state": "weird"},
            headers={"X-Reset-Token": token},
        )
        assert resp.status_code == 400

    def test_missing_token_401(self, reset_client, test_db):
        task_id, _, _ = self._begin_reset(reset_client, test_db)
        resp = reset_client.post(
            f"/api/v0/nodes/factory-reset/{task_id}/status",
            json={"state": "in_progress"},
        )
        assert resp.status_code == 401

    def test_unknown_token_401(self, reset_client, test_db):
        task_id, _, _ = self._begin_reset(reset_client, test_db)
        resp = reset_client.post(
            f"/api/v0/nodes/factory-reset/{task_id}/status",
            json={"state": "in_progress"},
            headers={"X-Reset-Token": "deadbeef"},
        )
        assert resp.status_code == 401

    def test_unknown_task_404(self, reset_client, test_db):
        # Issue a token that's NOT tied to any task — can still be used to
        # call status, but the task itself doesn't exist.
        token = pending_resets.create_reset_token()
        resp = reset_client.post(
            "/api/v0/nodes/factory-reset/does-not-exist/status",
            json={"state": "in_progress"},
            headers={"X-Reset-Token": token},
        )
        assert resp.status_code == 404

    def test_terminal_state_is_idempotent(self, reset_client, test_db):
        """Late-arriving status updates after success should be no-ops, not errors."""
        task_id, token, _ = self._begin_reset(reset_client, test_db)
        with patch("app.admin._deactivate_node_with_auth", return_value=True):
            r1 = reset_client.post(
                f"/api/v0/nodes/factory-reset/{task_id}/status",
                json={"state": "success"},
                headers={"X-Reset-Token": token},
            )
            assert r1.status_code == 200

            # A duplicate "in_progress" landing late shouldn't roll the state back.
            r2 = reset_client.post(
                f"/api/v0/nodes/factory-reset/{task_id}/status",
                json={"state": "in_progress"},
                headers={"X-Reset-Token": token},
            )
            assert r2.status_code == 200
            assert r2.json()["state"] == "success"


# =============================================================================
# list_nodes filtering
# =============================================================================


class TestListNodesIsActive:

    def test_inactive_excluded_by_default(self, reset_client, test_db):
        active = _create_node(test_db, is_active=True)
        inactive = _create_node(test_db, is_active=False)

        resp = reset_client.get("/api/v0/admin/nodes")
        assert resp.status_code == 200
        ids = [n["node_id"] for n in resp.json()]
        assert active.node_id in ids
        assert inactive.node_id not in ids

    def test_include_inactive_returns_both(self, reset_client, test_db):
        active = _create_node(test_db, is_active=True)
        inactive = _create_node(test_db, is_active=False)

        resp = reset_client.get("/api/v0/admin/nodes?include_inactive=true")
        assert resp.status_code == 200
        ids = [n["node_id"] for n in resp.json()]
        assert active.node_id in ids
        assert inactive.node_id in ids
