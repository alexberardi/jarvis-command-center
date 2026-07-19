"""Server-plane (node-less) interactive-notification callbacks.

POST /api/v0/callbacks with target_node_id omitted → CC executes a handler
registered in app.services.server_callback_registry in a background task and
records the result through the same machinery as a node-posted result.

Prereq for the phone-calls PRD: confirm cards and mid-call escalation are
produced by a CC tool and answered by CC — no node exists to route through.
"""

import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

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
from app.services.server_callback_registry import (
    ServerCallbackResult,
    register_server_callback,
    unregister_server_callback,
)


TEST_USER_ID = 77
TEST_HOUSEHOLD = "hh-server-cb"

COMMAND = "test_server_tool"
CALLBACK = "confirm_action"


class _NonClosingSession:
    """Wraps the test session so the background task's close() is a no-op
    (the fixture owns the session/transaction lifecycle)."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        pass


@pytest.fixture
def server_cb_client(test_db):
    """User-JWT client with the background task bound to the test session."""

    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_user_jwt():
        return AuthenticatedUser(user_id=TEST_USER_ID, email="t@x", is_superuser=False)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_user_jwt] = override_user_jwt
    with (
        patch("app.api.callbacks.verify_household_role", return_value=None),
        patch(
            "app.api.callbacks._server_callback_session_factory",
            side_effect=lambda: _NonClosingSession(test_db),
        ),
    ):
        try:
            with TestClient(app) as client:
                yield client
        finally:
            app.dependency_overrides.clear()


@pytest.fixture
def handler_registry():
    """Auto-unregister anything a test registers."""
    registered: list[tuple[str, str]] = []

    def _register(command, callback, handler):
        register_server_callback(command, callback, handler)
        registered.append((command, callback))

    yield _register
    for command, callback in registered:
        unregister_server_callback(command, callback)


def _create_body(**overrides) -> dict:
    body = {
        "command_name": COMMAND,
        "callback_name": CALLBACK,
        "data": {"choice": "6:30 works"},
        "household_id": TEST_HOUSEHOLD,
    }
    body.update(overrides)
    return body


# =============================================================================
# Happy path
# =============================================================================


class TestServerDispatch:
    def test_handler_runs_and_job_completes(self, server_cb_client, test_db, handler_registry):
        seen = {}

        def handler(ctx):
            seen["ctx"] = ctx
            return ServerCallbackResult(
                success=True, context_data={"echo": ctx.data["choice"]},
            )

        handler_registry(COMMAND, CALLBACK, handler)

        resp = server_cb_client.post("/api/v0/callbacks", json=_create_body())

        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]

        # BackgroundTasks run before TestClient returns — the job is final.
        job = test_db.query(CallbackJob).filter(CallbackJob.id == job_id).one()
        assert job.status == "completed"
        assert job.node_id is None
        assert job.household_id == TEST_HOUSEHOLD
        assert job.user_id == TEST_USER_ID
        assert json.loads(job.result_context_data_json) == {"echo": "6:30 works"}

        ctx = seen["ctx"]
        assert ctx.job_id == job_id
        assert ctx.household_id == TEST_HOUSEHOLD
        assert ctx.user_id == TEST_USER_ID
        assert ctx.data == {"choice": "6:30 works"}

    def test_async_handler_supported(self, server_cb_client, test_db, handler_registry):
        async def handler(ctx):
            return ServerCallbackResult(success=True, context_data={"ok": True})

        handler_registry(COMMAND, CALLBACK, handler)

        resp = server_cb_client.post("/api/v0/callbacks", json=_create_body())
        assert resp.status_code == 201
        job = test_db.query(CallbackJob).filter(CallbackJob.id == resp.json()["id"]).one()
        assert job.status == "completed"
        assert json.loads(job.result_context_data_json) == {"ok": True}

    def test_status_endpoint_returns_result(self, server_cb_client, test_db, handler_registry):
        handler_registry(
            COMMAND, CALLBACK,
            lambda ctx: ServerCallbackResult(success=True, context_data={"n": 1}),
        )

        create = server_cb_client.post(
            "/api/v0/callbacks", json=_create_body(navigation_type="stack"),
        )
        job_id = create.json()["id"]

        status = server_cb_client.get(f"/api/v0/callbacks/{job_id}/status")
        assert status.status_code == 200
        body = status.json()
        assert body["status"] == "completed"
        assert body["context_data"] == {"n": 1}


# =============================================================================
# Failure paths
# =============================================================================


class TestServerDispatchFailures:
    def test_handler_exception_marks_failed(self, server_cb_client, test_db, handler_registry):
        def handler(ctx):
            raise RuntimeError("gateway unreachable")

        handler_registry(COMMAND, CALLBACK, handler)

        resp = server_cb_client.post("/api/v0/callbacks", json=_create_body())
        assert resp.status_code == 201
        job = test_db.query(CallbackJob).filter(CallbackJob.id == resp.json()["id"]).one()
        assert job.status == "failed"
        assert "gateway unreachable" in job.error_message

    def test_handler_returning_failure_recorded(self, server_cb_client, test_db, handler_registry):
        handler_registry(
            COMMAND, CALLBACK,
            lambda ctx: ServerCallbackResult(success=False, error="plan expired"),
        )

        resp = server_cb_client.post("/api/v0/callbacks", json=_create_body())
        job = test_db.query(CallbackJob).filter(CallbackJob.id == resp.json()["id"]).one()
        assert job.status == "failed"
        assert job.error_message == "plan expired"

    def test_unregistered_handler_rejected_no_row(self, server_cb_client, test_db):
        resp = server_cb_client.post(
            "/api/v0/callbacks",
            json=_create_body(command_name="nobody_home"),
        )
        assert resp.status_code == 400
        assert "No server-side handler" in resp.text
        count = (
            test_db.query(CallbackJob)
            .filter(CallbackJob.command_name == "nobody_home")
            .count()
        )
        assert count == 0

    def test_missing_household_id_rejected(self, server_cb_client, handler_registry):
        handler_registry(
            COMMAND, CALLBACK, lambda ctx: ServerCallbackResult(success=True),
        )
        body = _create_body()
        del body["household_id"]
        resp = server_cb_client.post("/api/v0/callbacks", json=body)
        assert resp.status_code == 400
        assert "household_id" in resp.text

    def test_membership_check_enforced(self, test_db, handler_registry):
        """verify_household_role raising must propagate as the response."""
        from fastapi import HTTPException

        handler_registry(
            COMMAND, CALLBACK, lambda ctx: ServerCallbackResult(success=True),
        )

        def override_get_db():
            try:
                yield test_db
            finally:
                pass

        def override_user_jwt():
            return AuthenticatedUser(user_id=999, email="i@x", is_superuser=False)

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[verify_user_jwt] = override_user_jwt
        try:
            with patch(
                "app.api.callbacks.verify_household_role",
                side_effect=HTTPException(status_code=403, detail="not a member"),
            ):
                with TestClient(app) as client:
                    resp = client.post("/api/v0/callbacks", json=_create_body())
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 403


# =============================================================================
# Plane isolation: nodes can never touch server-plane jobs
# =============================================================================


class TestPlaneIsolation:
    def _server_job(self, db) -> CallbackJob:
        now = datetime.utcnow()
        job = CallbackJob(
            id=str(uuid.uuid4()),
            node_id=None,
            household_id=TEST_HOUSEHOLD,
            user_id=TEST_USER_ID,
            command_name=COMMAND,
            callback_name=CALLBACK,
            data_json="{}",
            status="pending",
            created_at=now,
            expires_at=now + timedelta(minutes=5),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def _node_client(self, test_db) -> TestClient:
        uid = str(uuid.uuid4())[:8]
        node = Node(
            node_id=f"test-cb-node-{uid}",
            api_key=f"test-cb-key-{uid}",
            room="kitchen",
            user="test-user",
            household_id=TEST_HOUSEHOLD,
        )
        test_db.add(node)
        test_db.commit()

        def override_get_db():
            try:
                yield test_db
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[verify_api_key] = lambda: NodeContextProvider(
            node, household_id=node.household_id, household_member_ids=[TEST_USER_ID],
        )
        return TestClient(app)

    def test_node_cannot_fetch_server_job(self, test_db):
        job = self._server_job(test_db)
        client = self._node_client(test_db)
        try:
            resp = client.get(f"/api/v0/callbacks/{job.id}")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 404

    def test_node_cannot_complete_server_job(self, test_db):
        job = self._server_job(test_db)
        client = self._node_client(test_db)
        try:
            resp = client.post(
                f"/api/v0/callbacks/{job.id}/result", json={"success": True},
            )
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 404
        refreshed = test_db.query(CallbackJob).filter(CallbackJob.id == job.id).one()
        assert refreshed.status == "pending"


# =============================================================================
# Inbox fan-out parity with the node plane
# =============================================================================


class TestServerResultFanOut:
    def test_new_notification_creates_inbox_without_node_id(
        self, server_cb_client, test_db, handler_registry,
    ):
        handler_registry(
            COMMAND, CALLBACK,
            lambda ctx: ServerCallbackResult(
                success=True,
                context_data={
                    "inbox": {
                        "title": "Call booked",
                        "summary": "Friday 7pm, party of 4",
                        "metadata": {"session_id": "s-1"},
                    }
                },
            ),
        )

        with patch(
            "app.services.inbox_notification_service.post_inbox_item_sync"
        ) as mock_post:
            mock_post.return_value = "inbox-1"
            resp = server_cb_client.post(
                "/api/v0/callbacks", json=_create_body(),
            )

        assert resp.status_code == 201
        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        assert kwargs["title"] == "Call booked"
        assert kwargs["household_id"] == TEST_HOUSEHOLD
        assert kwargs["target_type"] == "user"
        assert kwargs["user_id"] == TEST_USER_ID
        # No node in this plane — CC must not stamp a node_id.
        assert "node_id" not in kwargs["metadata"]

    def test_stack_navigation_skips_inbox(self, server_cb_client, handler_registry):
        handler_registry(
            COMMAND, CALLBACK,
            lambda ctx: ServerCallbackResult(
                success=True, context_data={"inbox": {"title": "x"}},
            ),
        )
        with patch(
            "app.services.inbox_notification_service.post_inbox_item_sync"
        ) as mock_post:
            server_cb_client.post(
                "/api/v0/callbacks", json=_create_body(navigation_type="stack"),
            )
        mock_post.assert_not_called()


# =============================================================================
# Node plane unchanged (back-compat pin)
# =============================================================================


class TestNodePlaneBackCompat:
    def test_node_path_still_requires_existing_node(self, server_cb_client):
        with patch("app.api.callbacks.get_node_command_service"):
            resp = server_cb_client.post(
                "/api/v0/callbacks",
                json=_create_body(target_node_id="no-such-node"),
            )
        assert resp.status_code == 404
