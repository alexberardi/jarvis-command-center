"""Tests for package install/uninstall/revert endpoints.

Covers the request/poll pattern endpoints in app/api/package_install.py:
1. POST /nodes/{node_id}/package-install            — mobile requests an install
2. POST /nodes/{node_id}/package-install/{id}/results — node uploads results
3. GET  /nodes/{node_id}/package-install/{id}        — mobile polls for results
plus the uninstall and revert variants.

Focus areas (resilient-package-installs):
- "restarting" status: non-terminal, extends expiry, passes through polling,
  and is overwritten by the node's deferred result post after boot.
- Revert flow mirroring the uninstall flow (same table, no repo URL).

Follows the same patterns as test_device_list_endpoints.py.
"""

import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Node, PackageInstallRequest
from app.deps import get_db, verify_api_key
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext
from app.api.package_install import RESTART_EXPIRY_EXTENSION_SECONDS


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def package_client(test_db):
    """Test client with DB + auth overrides for package install/uninstall/revert.

    Request and poll endpoints use verify_provisioning_auth (mobile/admin).
    Results-upload endpoints use verify_api_key (node auth).
    """
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_auth():
        return ProvisioningAuthContext(auth_type="admin_key")

    def override_api_key():
        return MagicMock()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_provisioning_auth] = override_auth
    app.dependency_overrides[verify_api_key] = override_api_key

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _create_node(db, household_id: str = "hh-test-001") -> Node:
    """Create a test node with household_id."""
    uid = str(uuid.uuid4())[:8]
    node = Node(
        node_id=f"test-node-{uid}",
        api_key=f"test-key-{uid}",
        room="living room",
        user="test-user",
        household_id=household_id,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return node


def _create_package_request(
    db,
    node: Node,
    *,
    status: str = "pending",
    command_name: str = "test_command",
    github_repo_url: str = "https://github.com/example/test-command",
    expired: bool = False,
    results_json: str | None = None,
    error_message: str | None = None,
) -> PackageInstallRequest:
    """Create a PackageInstallRequest record directly in the DB.

    Install, uninstall, and revert all share this table (uninstall/revert rows
    carry an empty github_repo_url).
    """
    now = datetime.utcnow()
    expires_at = now - timedelta(minutes=1) if expired else now + timedelta(minutes=2)

    req = PackageInstallRequest(
        id=str(uuid.uuid4()),
        node_id=node.node_id,
        household_id=node.household_id or "",
        command_name=command_name,
        github_repo_url=github_repo_url,
        status=status,
        results_json=results_json,
        error_message=error_message,
        created_at=now,
        expires_at=expires_at,
        completed_at=now if status in ("completed", "failed") else None,
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req


# =============================================================================
# Request Endpoint: POST /nodes/{node_id}/package-install
# =============================================================================


class TestRequestPackageInstall:
    """Tests for POST /api/v0/nodes/{node_id}/package-install"""

    @patch("app.api.package_install._publish_package_install_mqtt")
    def test_request_creates_record_and_returns_201(
        self, mock_mqtt, package_client, test_db
    ):
        """Successful request creates a PackageInstallRequest and returns 201."""
        node = _create_node(test_db)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install",
            json={
                "command_name": "weather",
                "github_repo_url": "https://github.com/example/weather",
                "git_tag": "v1.0.0",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert "id" in data
        assert "created_at" in data

        db_req = test_db.query(PackageInstallRequest).filter(
            PackageInstallRequest.id == data["id"]
        ).first()
        assert db_req is not None
        assert db_req.node_id == node.node_id
        assert db_req.status == "pending"
        assert db_req.command_name == "weather"
        assert db_req.git_tag == "v1.0.0"

        mock_mqtt.assert_called_once()
        assert mock_mqtt.call_args[0][0] == node.node_id

    @patch("app.api.package_install._publish_package_install_mqtt")
    def test_request_node_not_found(self, mock_mqtt, package_client):
        """Returns 404 for nonexistent node."""
        resp = package_client.post(
            "/api/v0/nodes/nonexistent-node/package-install",
            json={
                "command_name": "weather",
                "github_repo_url": "https://github.com/example/weather",
            },
        )
        assert resp.status_code == 404
        mock_mqtt.assert_not_called()


# =============================================================================
# Results Endpoint: POST /nodes/{node_id}/package-install/{request_id}/results
# =============================================================================


class TestUploadPackageInstallResults:
    """Tests for POST /api/v0/nodes/{node_id}/package-install/{request_id}/results"""

    def test_upload_success(self, package_client, test_db):
        """Node uploads success, request status becomes completed."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}/results",
            json={"success": True, "details": {"version": "1.0.0"}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        test_db.refresh(req)
        assert req.status == "completed"
        assert req.completed_at is not None
        assert json.loads(req.results_json) == {"version": "1.0.0"}

    def test_upload_failure(self, package_client, test_db):
        """Node uploads failure, request status becomes failed."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}/results",
            json={"success": False, "error": "Package failed validation: boom"},
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        assert req.status == "failed"
        assert req.error_message == "Package failed validation: boom"
        assert req.completed_at is not None

    def test_upload_restarting_sets_non_terminal_status(self, package_client, test_db):
        """restarting=true marks the request restarting without completing it."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}/results",
            json={"success": True, "restarting": True, "details": {"version": "1.0.0"}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        test_db.refresh(req)
        assert req.status == "restarting"
        assert req.completed_at is None
        assert json.loads(req.results_json) == {"version": "1.0.0"}

    def test_upload_restarting_extends_expiry(self, package_client, test_db):
        """Entering restarting extends expires_at so a slow boot doesn't expire it."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node)
        old_expires = req.expires_at

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}/results",
            json={"success": True, "restarting": True},
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        expected = old_expires + timedelta(seconds=RESTART_EXPIRY_EXTENSION_SECONDS)
        assert abs((req.expires_at - expected).total_seconds()) < 1

    def test_upload_after_restarting_overwrites_to_completed(self, package_client, test_db):
        """The deferred result post after boot overwrites restarting with completed."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, status="restarting")

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}/results",
            json={"success": True, "details": {"version": "1.0.0"}},
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        assert req.status == "completed"
        assert req.completed_at is not None

    def test_upload_after_restarting_overwrites_to_failed(self, package_client, test_db):
        """A failed deferred result after boot overwrites restarting with failed."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, status="restarting")

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}/results",
            json={"success": False, "error": "Installed but components failed to load"},
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        assert req.status == "failed"
        assert req.error_message == "Installed but components failed to load"

    def test_upload_expired_request(self, package_client, test_db):
        """Returns 410 for expired request."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, expired=True)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}/results",
            json={"success": True},
        )
        assert resp.status_code == 410

        test_db.refresh(req)
        assert req.status == "expired"

    def test_upload_request_not_found(self, package_client, test_db):
        """Returns 404 for nonexistent request."""
        node = _create_node(test_db)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install/{uuid.uuid4()}/results",
            json={"success": True},
        )
        assert resp.status_code == 404

    def test_upload_wrong_node(self, package_client, test_db):
        """Returns 404 when node_id in URL does not match request's node."""
        node_a = _create_node(test_db)
        node_b = _create_node(test_db)
        req = _create_package_request(test_db, node_a)

        resp = package_client.post(
            f"/api/v0/nodes/{node_b.node_id}/package-install/{req.id}/results",
            json={"success": True},
        )
        assert resp.status_code == 404


# =============================================================================
# Poll Endpoint: GET /nodes/{node_id}/package-install/{request_id}
# =============================================================================


class TestPollPackageInstall:
    """Tests for GET /api/v0/nodes/{node_id}/package-install/{request_id}"""

    def test_poll_pending(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(test_db, node)

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["command_name"] == "test_command"

    def test_poll_restarting_passthrough(self, package_client, test_db):
        """Poll passes the non-terminal restarting status through to mobile."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, status="restarting")

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "restarting"
        assert data["error_message"] is None

        test_db.refresh(req)
        assert req.status == "restarting"  # not expired while within window

    def test_poll_restarting_expires_like_pending(self, package_client, test_db):
        """A restarting request past expires_at is marked expired on poll."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, status="restarting", expired=True)

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}"
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "expired"

        test_db.refresh(req)
        assert req.status == "expired"

    def test_poll_pending_expires(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, expired=True)

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "expired"
        assert "expired" in data["error_message"].lower()

    def test_poll_completed_with_details(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(
            test_db, node,
            status="completed",
            results_json=json.dumps({"version": "1.0.0"}),
        )

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["details"] == {"version": "1.0.0"}

    def test_poll_failed(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(
            test_db, node, status="failed", error_message="boom"
        )

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error_message"] == "boom"

    def test_poll_request_not_found(self, package_client, test_db):
        node = _create_node(test_db)

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-install/{uuid.uuid4()}"
        )
        assert resp.status_code == 404


# =============================================================================
# Uninstall: restarting handling
# =============================================================================


class TestPackageUninstallRestarting:
    """Restarting handling on the uninstall results/poll endpoints.

    Uninstall also defers its result across a node restart, so the uninstall
    endpoints accept the same restarting marker as install.
    """

    def test_upload_uninstall_restarting(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, github_repo_url="")
        old_expires = req.expires_at

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-uninstall/{req.id}/results",
            json={"success": True, "restarting": True},
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        assert req.status == "restarting"
        assert req.completed_at is None
        expected = old_expires + timedelta(seconds=RESTART_EXPIRY_EXTENSION_SECONDS)
        assert abs((req.expires_at - expected).total_seconds()) < 1

    def test_poll_uninstall_restarting_passthrough(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(
            test_db, node, status="restarting", github_repo_url=""
        )

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-uninstall/{req.id}"
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "restarting"

    def test_poll_uninstall_restarting_expires_like_pending(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(
            test_db, node, status="restarting", github_repo_url="", expired=True
        )

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-uninstall/{req.id}"
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "expired"


# =============================================================================
# Revert Request: POST /nodes/{node_id}/package-revert
# =============================================================================


class TestRequestPackageRevert:
    """Tests for POST /api/v0/nodes/{node_id}/package-revert"""

    @patch("app.api.package_install._publish_package_revert_mqtt")
    def test_request_creates_record_and_returns_201(
        self, mock_mqtt, package_client, test_db
    ):
        """Successful revert request creates a record (no repo URL) and returns 201."""
        node = _create_node(test_db)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert",
            json={"command_name": "weather"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert "id" in data

        db_req = test_db.query(PackageInstallRequest).filter(
            PackageInstallRequest.id == data["id"]
        ).first()
        assert db_req is not None
        assert db_req.node_id == node.node_id
        assert db_req.command_name == "weather"
        assert db_req.github_repo_url == ""  # not needed for revert
        assert db_req.status == "pending"

        mock_mqtt.assert_called_once()
        assert mock_mqtt.call_args[0][0] == node.node_id

    @patch("app.api.package_install._publish_package_revert_mqtt")
    def test_request_accepts_package_name(self, mock_mqtt, package_client, test_db):
        """Body may use package_name instead of command_name."""
        node = _create_node(test_db)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert",
            json={"package_name": "weather"},
        )
        assert resp.status_code == 201

        db_req = test_db.query(PackageInstallRequest).filter(
            PackageInstallRequest.id == resp.json()["id"]
        ).first()
        assert db_req.command_name == "weather"

    @patch("app.api.package_install._publish_package_revert_mqtt")
    def test_request_requires_a_name(self, mock_mqtt, package_client, test_db):
        """Returns 422 when neither command_name nor package_name is given."""
        node = _create_node(test_db)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert",
            json={},
        )
        assert resp.status_code == 422
        mock_mqtt.assert_not_called()

    @patch("app.api.package_install._publish_package_revert_mqtt")
    def test_request_node_not_found(self, mock_mqtt, package_client):
        """Returns 404 for nonexistent node."""
        resp = package_client.post(
            "/api/v0/nodes/nonexistent-node/package-revert",
            json={"command_name": "weather"},
        )
        assert resp.status_code == 404
        mock_mqtt.assert_not_called()

    @patch("app.node_settings.get_mqtt_client")
    def test_request_publishes_mqtt_topic_and_payload(
        self, mock_get_client, package_client, test_db
    ):
        """MQTT publish goes to the package-revert topic with request_id + name."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        node = _create_node(test_db)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert",
            json={"command_name": "weather"},
        )
        assert resp.status_code == 201
        request_id = resp.json()["id"]

        mock_client.publish.assert_called_once()
        topic, payload = mock_client.publish.call_args[0]
        assert topic == f"jarvis/nodes/{node.node_id}/package-revert"
        data = json.loads(payload)
        assert data["request_id"] == request_id
        assert data["command_name"] == "weather"
        assert data["package_name"] == "weather"


# =============================================================================
# Revert Results: POST /nodes/{node_id}/package-revert/{request_id}/results
# =============================================================================


class TestUploadPackageRevertResults:
    """Tests for POST /api/v0/nodes/{node_id}/package-revert/{request_id}/results"""

    def test_upload_success(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, github_repo_url="")

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}/results",
            json={"success": True, "details": {"restored_version": "1.0.1"}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        test_db.refresh(req)
        assert req.status == "completed"
        assert req.completed_at is not None
        assert json.loads(req.results_json) == {"restored_version": "1.0.1"}

    def test_upload_failure(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, github_repo_url="")

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}/results",
            json={"success": False, "error": "No previous version to revert to"},
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        assert req.status == "failed"
        assert req.error_message == "No previous version to revert to"

    def test_upload_restarting(self, package_client, test_db):
        """Revert also restarts the node — restarting is accepted and extends expiry."""
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, github_repo_url="")
        old_expires = req.expires_at

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}/results",
            json={"success": True, "restarting": True},
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        assert req.status == "restarting"
        assert req.completed_at is None
        expected = old_expires + timedelta(seconds=RESTART_EXPIRY_EXTENSION_SECONDS)
        assert abs((req.expires_at - expected).total_seconds()) < 1

    def test_upload_expired_request(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, github_repo_url="", expired=True)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}/results",
            json={"success": True},
        )
        assert resp.status_code == 410

        test_db.refresh(req)
        assert req.status == "expired"

    def test_upload_request_not_found(self, package_client, test_db):
        node = _create_node(test_db)

        resp = package_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert/{uuid.uuid4()}/results",
            json={"success": True},
        )
        assert resp.status_code == 404

    def test_upload_wrong_node(self, package_client, test_db):
        node_a = _create_node(test_db)
        node_b = _create_node(test_db)
        req = _create_package_request(test_db, node_a, github_repo_url="")

        resp = package_client.post(
            f"/api/v0/nodes/{node_b.node_id}/package-revert/{req.id}/results",
            json={"success": True},
        )
        assert resp.status_code == 404


# =============================================================================
# Revert Poll: GET /nodes/{node_id}/package-revert/{request_id}
# =============================================================================


class TestPollPackageRevert:
    """Tests for GET /api/v0/nodes/{node_id}/package-revert/{request_id}"""

    def test_poll_pending(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(test_db, node, github_repo_url="")

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["command_name"] == "test_command"

    def test_poll_restarting_passthrough(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(
            test_db, node, status="restarting", github_repo_url=""
        )

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}"
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "restarting"

    def test_poll_restarting_expires_like_pending(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(
            test_db, node, status="restarting", github_repo_url="", expired=True
        )

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}"
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "expired"

        test_db.refresh(req)
        assert req.status == "expired"

    def test_poll_completed_with_details(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(
            test_db, node,
            status="completed",
            github_repo_url="",
            results_json=json.dumps({"restored_version": "1.0.1"}),
        )

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["details"] == {"restored_version": "1.0.1"}

    def test_poll_failed(self, package_client, test_db):
        node = _create_node(test_db)
        req = _create_package_request(
            test_db, node,
            status="failed",
            github_repo_url="",
            error_message="No previous version to revert to",
        )

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-revert/{req.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error_message"] == "No previous version to revert to"

    def test_poll_request_not_found(self, package_client, test_db):
        node = _create_node(test_db)

        resp = package_client.get(
            f"/api/v0/nodes/{node.node_id}/package-revert/{uuid.uuid4()}"
        )
        assert resp.status_code == 404
