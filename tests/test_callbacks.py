"""Tests for the interactive-notification callback endpoints.

  POST /api/v0/callbacks                 (mobile, user JWT)
  GET  /api/v0/callbacks/{job_id}        (node, X-API-Key)
  POST /api/v0/callbacks/{job_id}/result (node, X-API-Key)
"""

import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import (
    AuthenticatedUser,
    get_db,
    verify_api_key,
    verify_user_jwt,
)
from app.main import app
from app.models import CallbackJob, Node


# =============================================================================
# Helpers
# =============================================================================


TEST_USER_ID = 42
TEST_HOUSEHOLD = "hh-cb-test"


def _create_node(db, *, household_id: str = TEST_HOUSEHOLD) -> Node:
    uid = str(uuid.uuid4())[:8]
    node = Node(
        node_id=f"test-cb-node-{uid}",
        api_key=f"test-cb-key-{uid}",
        room="kitchen",
        user="test-user",
        household_id=household_id,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return node


def _create_job(
    db,
    *,
    node: Node,
    status: str = "pending",
    command_name: str = "movie_knowledge",
    callback_name: str = "expand_actor",
    data: dict | None = None,
    expires_in: timedelta = timedelta(minutes=5),
    user_id: int = TEST_USER_ID,
) -> CallbackJob:
    now = datetime.utcnow()
    job = CallbackJob(
        id=str(uuid.uuid4()),
        node_id=node.node_id,
        household_id=node.household_id or "",
        user_id=user_id,
        command_name=command_name,
        callback_name=callback_name,
        data_json=json.dumps(data or {}),
        status=status,
        created_at=now,
        expires_at=now + expires_in,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@pytest.fixture
def cb_user_client(test_db):
    """Client for POST /callbacks — user-JWT-authed endpoint."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_user_jwt():
        return AuthenticatedUser(user_id=TEST_USER_ID, email="t@x", is_superuser=False)

    def override_household_role(*args, **kwargs):
        return None  # always allow

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_user_jwt] = override_user_jwt
    # verify_household_role is a regular function not a Depends — patch by import
    with patch("app.api.callbacks.verify_household_role", side_effect=override_household_role):
        try:
            with TestClient(app) as client:
                yield client
        finally:
            app.dependency_overrides.clear()


def _node_client_for(test_db, node: Node) -> TestClient:
    """Build a client that authenticates as the given node for node-keyed endpoints."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_verify_api_key():
        return NodeContextProvider(
            node, household_id=node.household_id, household_member_ids=[TEST_USER_ID],
        )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_api_key] = override_verify_api_key
    return TestClient(app)


# =============================================================================
# POST /callbacks  (user-authed)
# =============================================================================


class TestPostCreateCallback:
    def test_creates_job_and_publishes_mqtt(self, cb_user_client, test_db):
        node = _create_node(test_db)

        with patch("app.api.callbacks.get_node_command_service") as svc_factory:
            svc = MagicMock()
            svc_factory.return_value = svc

            resp = cb_user_client.post(
                "/api/v0/callbacks",
                json={
                    "command_name": "movie_knowledge",
                    "callback_name": "expand_actor",
                    "data": {"actor_id": "nm0000158"},
                    "target_node_id": node.node_id,
                },
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["status"] == "pending"
        assert "id" in body

        # MQTT was triggered with the job id as the request_id and "callback" command.
        svc.publish_command_with_id.assert_called_once()
        kwargs = svc.publish_command_with_id.call_args.kwargs
        assert kwargs["node_id"] == node.node_id
        assert kwargs["command"] == "callback"
        assert kwargs["request_id"] == body["id"]

        # Row was persisted with all the right fields.
        job = test_db.query(CallbackJob).filter(CallbackJob.id == body["id"]).one()
        assert job.command_name == "movie_knowledge"
        assert job.callback_name == "expand_actor"
        assert json.loads(job.data_json) == {"actor_id": "nm0000158"}
        assert job.user_id == TEST_USER_ID
        assert job.node_id == node.node_id
        assert job.household_id == TEST_HOUSEHOLD

    def test_target_node_not_found_returns_404(self, cb_user_client):
        with patch("app.api.callbacks.get_node_command_service"):
            resp = cb_user_client.post(
                "/api/v0/callbacks",
                json={
                    "command_name": "movie_knowledge",
                    "callback_name": "expand_actor",
                    "data": {},
                    "target_node_id": "no-such-node",
                },
            )
        assert resp.status_code == 404

    def test_target_node_without_household_returns_400(self, cb_user_client, test_db):
        node = _create_node(test_db, household_id="")
        with patch("app.api.callbacks.get_node_command_service"):
            resp = cb_user_client.post(
                "/api/v0/callbacks",
                json={
                    "command_name": "movie_knowledge",
                    "callback_name": "expand_actor",
                    "data": {},
                    "target_node_id": node.node_id,
                },
            )
        assert resp.status_code == 400

    def test_validation_rejects_empty_command_name(self, cb_user_client, test_db):
        node = _create_node(test_db)
        resp = cb_user_client.post(
            "/api/v0/callbacks",
            json={
                "command_name": "",
                "callback_name": "expand_actor",
                "data": {},
                "target_node_id": node.node_id,
            },
        )
        # CC remaps FastAPI's 422 → 400 via a global handler; either is "validation rejected".
        assert resp.status_code in (400, 422)


# =============================================================================
# GET /callbacks/{id}  (node-authed)
# =============================================================================


class TestGetCallbackPayload:
    def test_owning_node_can_fetch_pending_payload(self, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node, data={"actor_id": "nm0000158"})

        client = _node_client_for(test_db, node)
        try:
            resp = client.get(f"/api/v0/callbacks/{job.id}")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["job_id"] == job.id
        assert body["command_name"] == "movie_knowledge"
        assert body["callback_name"] == "expand_actor"
        assert body["data"] == {"actor_id": "nm0000158"}
        assert body["user_id"] == TEST_USER_ID

    def test_different_node_gets_404(self, test_db):
        owner = _create_node(test_db)
        intruder = _create_node(test_db)
        job = _create_job(test_db, node=owner)

        client = _node_client_for(test_db, intruder)
        try:
            resp = client.get(f"/api/v0/callbacks/{job.id}")
        finally:
            app.dependency_overrides.clear()

        # 404 (not 403) to avoid leaking job-id existence to non-owning nodes.
        assert resp.status_code == 404

    def test_nonexistent_job_returns_404(self, test_db):
        node = _create_node(test_db)
        client = _node_client_for(test_db, node)
        try:
            resp = client.get(f"/api/v0/callbacks/{uuid.uuid4()}")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 404

    def test_expired_job_returns_410_and_marks_expired(self, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node, expires_in=timedelta(minutes=-1))

        client = _node_client_for(test_db, node)
        try:
            resp = client.get(f"/api/v0/callbacks/{job.id}")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 410
        refreshed = test_db.query(CallbackJob).filter(CallbackJob.id == job.id).one()
        assert refreshed.status == "expired"


# =============================================================================
# POST /callbacks/{id}/result  (node-authed)
# =============================================================================


class TestPostCallbackResult:
    def test_success_marks_completed_and_stores_context_data(self, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node)

        client = _node_client_for(test_db, node)
        try:
            resp = client.post(
                f"/api/v0/callbacks/{job.id}/result",
                json={
                    "success": True,
                    "context_data": {"actor": "Tom Hanks", "films": ["A", "B"]},
                },
            )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "completed"

        refreshed = test_db.query(CallbackJob).filter(CallbackJob.id == job.id).one()
        assert refreshed.status == "completed"
        assert refreshed.completed_at is not None
        assert json.loads(refreshed.result_context_data_json) == {
            "actor": "Tom Hanks", "films": ["A", "B"],
        }

    def test_failure_marks_failed_with_error(self, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node)

        client = _node_client_for(test_db, node)
        try:
            resp = client.post(
                f"/api/v0/callbacks/{job.id}/result",
                json={"success": False, "error": "TMDB lookup failed"},
            )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"

        refreshed = test_db.query(CallbackJob).filter(CallbackJob.id == job.id).one()
        assert refreshed.status == "failed"
        assert refreshed.error_message == "TMDB lookup failed"

    def test_different_node_cannot_post_result(self, test_db):
        owner = _create_node(test_db)
        intruder = _create_node(test_db)
        job = _create_job(test_db, node=owner)

        client = _node_client_for(test_db, intruder)
        try:
            resp = client.post(
                f"/api/v0/callbacks/{job.id}/result",
                json={"success": True},
            )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 404
        # Job state must be unchanged.
        refreshed = test_db.query(CallbackJob).filter(CallbackJob.id == job.id).one()
        assert refreshed.status == "pending"


# =============================================================================
# navigation_type: how the result gets surfaced
# =============================================================================


class TestNavigationType:
    def test_default_is_new_notification(self, cb_user_client, test_db):
        node = _create_node(test_db)
        with patch("app.api.callbacks.get_node_command_service"):
            resp = cb_user_client.post(
                "/api/v0/callbacks",
                json={
                    "command_name": "movie_knowledge",
                    "callback_name": "expand_actor",
                    "data": {},
                    "target_node_id": node.node_id,
                },
            )
        assert resp.status_code == 201
        assert resp.json()["navigation_type"] == "new_notification"
        job = test_db.query(CallbackJob).filter(CallbackJob.id == resp.json()["id"]).one()
        assert job.navigation_type == "new_notification"

    def test_stack_navigation_type_persisted(self, cb_user_client, test_db):
        node = _create_node(test_db)
        with patch("app.api.callbacks.get_node_command_service"):
            resp = cb_user_client.post(
                "/api/v0/callbacks",
                json={
                    "command_name": "movie_knowledge",
                    "callback_name": "expand_actor",
                    "data": {},
                    "target_node_id": node.node_id,
                    "navigation_type": "stack",
                },
            )
        assert resp.status_code == 201
        assert resp.json()["navigation_type"] == "stack"
        job = test_db.query(CallbackJob).filter(CallbackJob.id == resp.json()["id"]).one()
        assert job.navigation_type == "stack"

    def test_invalid_navigation_type_rejected(self, cb_user_client, test_db):
        node = _create_node(test_db)
        with patch("app.api.callbacks.get_node_command_service"):
            resp = cb_user_client.post(
                "/api/v0/callbacks",
                json={
                    "command_name": "movie_knowledge",
                    "callback_name": "expand_actor",
                    "data": {},
                    "target_node_id": node.node_id,
                    "navigation_type": "carousel",  # not in the Literal
                },
            )
        assert resp.status_code in (400, 422)


class TestResultFansOutOnNewNotification:
    """When navigation_type=new_notification, posting a successful result
    with context_data.inbox should also create an inbox item server-side."""

    def test_new_notification_creates_inbox(self, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node)
        job.navigation_type = "new_notification"
        test_db.commit()

        client = _node_client_for(test_db, node)
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            mock_post.return_value = "inbox-xyz"
            try:
                resp = client.post(
                    f"/api/v0/callbacks/{job.id}/result",
                    json={
                        "success": True,
                        "context_data": {
                            "inbox": {
                                "title": "Tom Hanks — Films",
                                "summary": "American actor",
                                "body": "Bio markdown",
                                "category": "movie_knowledge",
                                "metadata": {"interactive_elements": [], "tmdb_person_id": 31},
                                "create_push_notification": False,
                            }
                        },
                    },
                )
            finally:
                app.dependency_overrides.clear()

        assert resp.status_code == 200
        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        assert kwargs["title"] == "Tom Hanks — Films"
        assert kwargs["household_id"] == job.household_id
        # CC injects node_id into the metadata server-side.
        assert kwargs["metadata"]["node_id"] == node.node_id
        assert kwargs["push"] is False

    def test_stack_does_not_create_inbox(self, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node)
        job.navigation_type = "stack"
        test_db.commit()

        client = _node_client_for(test_db, node)
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            try:
                client.post(
                    f"/api/v0/callbacks/{job.id}/result",
                    json={
                        "success": True,
                        "context_data": {
                            "inbox": {"title": "anything", "body": "x"},
                        },
                    },
                )
            finally:
                app.dependency_overrides.clear()
        mock_post.assert_not_called()

    def test_new_notification_pushes_to_user(self, test_db):
        """Callbacks always have a user_id (the JWT-authenticated tapper),
        so the fan-out push should land on that user — not the household."""
        node = _create_node(test_db)
        job = _create_job(test_db, node=node, user_id=42)
        job.navigation_type = "new_notification"
        test_db.commit()

        client = _node_client_for(test_db, node)
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            mock_post.return_value = "inbox-x"
            try:
                client.post(
                    f"/api/v0/callbacks/{job.id}/result",
                    json={
                        "success": True,
                        "context_data": {"inbox": {"title": "Tom Hanks"}},
                    },
                )
            finally:
                app.dependency_overrides.clear()
        assert mock_post.call_args.kwargs["target_type"] == "user"
        assert mock_post.call_args.kwargs["user_id"] == 42

    def test_failed_result_does_not_create_inbox(self, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node)
        job.navigation_type = "new_notification"
        test_db.commit()

        client = _node_client_for(test_db, node)
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            try:
                client.post(
                    f"/api/v0/callbacks/{job.id}/result",
                    json={
                        "success": False,
                        "error": "TMDB lookup failed",
                        "context_data": {"inbox": {"title": "leftover"}},
                    },
                )
            finally:
                app.dependency_overrides.clear()
        mock_post.assert_not_called()


# =============================================================================
# GET /callbacks/{job_id}/status — user-JWT mobile polling
# =============================================================================


class TestGetCallbackStatus:
    def test_pending_status_returned(self, cb_user_client, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node)
        resp = cb_user_client.get(f"/api/v0/callbacks/{job.id}/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "pending"
        assert body["navigation_type"] == "new_notification"
        assert body["context_data"] is None

    def test_completed_status_returns_context_data(self, cb_user_client, test_db):
        node = _create_node(test_db)
        job = _create_job(test_db, node=node)
        job.status = "completed"
        job.navigation_type = "stack"
        job.result_context_data_json = json.dumps({
            "inbox": {"title": "Tom Hanks — Films", "body": "..."}
        })
        job.completed_at = datetime.utcnow()
        test_db.commit()

        resp = cb_user_client.get(f"/api/v0/callbacks/{job.id}/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["navigation_type"] == "stack"
        assert body["context_data"]["inbox"]["title"] == "Tom Hanks — Films"

    def test_nonexistent_job_returns_404(self, cb_user_client):
        resp = cb_user_client.get(f"/api/v0/callbacks/{uuid.uuid4()}/status")
        assert resp.status_code == 404

    def test_expired_pending_is_marked_expired(self, cb_user_client, test_db):
        node = _create_node(test_db)
        job = _create_job(
            test_db, node=node,
            expires_in=timedelta(minutes=-1),  # already expired
        )
        resp = cb_user_client.get(f"/api/v0/callbacks/{job.id}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "expired"
