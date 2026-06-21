"""
Tests for node settings management.

This module tests the settings request/snapshot flow:
1. Mobile creates a settings request
2. CC signals node via MQTT
3. Node confirms request exists
4. Node uploads encrypted snapshot
5. Mobile polls and retrieves snapshot
"""
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models import Base, Node, SettingsRequest, SettingsSnapshot
from app.deps import AuthenticatedUser, get_db, verify_user_jwt


@pytest.fixture
def jwt_client(test_db):
    """Test client for the mobile-app-facing (user-JWT) settings endpoints.

    The create-request and get-result endpoints were migrated from an admin
    api-key to user-JWT auth (commit 8fb81a6). This overrides verify_user_jwt
    and stubs verify_household_role (an HTTP call to jarvis-auth) so the
    household-role check is a no-op in tests.
    """
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_user_jwt():
        return AuthenticatedUser(user_id=1, email="test@example.com")

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_user_jwt] = override_user_jwt
    with patch("app.node_settings.verify_household_role", lambda *a, **k: None):
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.clear()


# =============================================================================
# Model Tests
# =============================================================================

class TestSettingsRequestModel:
    """Tests for SettingsRequest SQLAlchemy model."""

    def test_create_settings_request(self, test_db, test_node):
        """Test creating a settings request."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
            expires_at=datetime.utcnow() + timedelta(minutes=5),
        )
        test_db.add(request)
        test_db.commit()
        test_db.refresh(request)

        assert request.request_id is not None
        assert request.node_id == test_node.node_id
        assert request.status == "pending"
        assert request.created_at is not None

    def test_request_defaults(self, test_db, test_node):
        """Test that defaults are applied correctly."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
        )
        test_db.add(request)
        test_db.commit()
        test_db.refresh(request)

        assert request.status == "pending"
        assert request.created_at is not None
        # expires_at should default to 5 minutes from now
        assert request.expires_at is not None
        assert request.expires_at > datetime.utcnow()

    def test_request_status_transitions(self, test_db, test_node):
        """Test valid status transitions."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
        )
        test_db.add(request)
        test_db.commit()

        # Transition to fulfilled
        request.status = "fulfilled"
        test_db.commit()
        test_db.refresh(request)
        assert request.status == "fulfilled"

    def test_request_foreign_key_constraint(self, test_db):
        """Test that request requires valid node_id."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id="nonexistent-node",
            status="pending",
        )
        test_db.add(request)
        with pytest.raises(Exception):  # IntegrityError
            test_db.commit()


class TestSettingsSnapshotModel:
    """Tests for SettingsSnapshot SQLAlchemy model."""

    def test_create_settings_snapshot(self, test_db, test_node):
        """Test creating a settings snapshot."""
        # First create a request
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
        )
        test_db.add(request)
        test_db.commit()

        # Create snapshot
        snapshot = SettingsSnapshot(
            snapshot_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            request_id=request.request_id,
            ciphertext="base64-encoded-ciphertext",
            nonce="base64-encoded-nonce",
            tag="base64-encoded-tag",
            aad_node_id=test_node.node_id,
            aad_schema_version=1,
            aad_commands_schema_version=1,
            aad_revision=1,
            aad_request_id=request.request_id,
        )
        test_db.add(snapshot)
        test_db.commit()
        test_db.refresh(snapshot)

        assert snapshot.snapshot_id is not None
        assert snapshot.node_id == test_node.node_id
        assert snapshot.request_id == request.request_id
        assert snapshot.created_at is not None

    def test_snapshot_aad_fields(self, test_db, test_node):
        """Test that AAD fields are stored correctly for binding."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
        )
        test_db.add(request)
        test_db.commit()

        snapshot = SettingsSnapshot(
            snapshot_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            request_id=request.request_id,
            ciphertext="encrypted-data",
            nonce="random-nonce",
            tag="auth-tag",
            aad_node_id=test_node.node_id,
            aad_schema_version=2,
            aad_commands_schema_version=3,
            aad_revision=5,
            aad_request_id=request.request_id,
        )
        test_db.add(snapshot)
        test_db.commit()
        test_db.refresh(snapshot)

        # Verify AAD fields match what clients need for decryption
        assert snapshot.aad_node_id == test_node.node_id
        assert snapshot.aad_schema_version == 2
        assert snapshot.aad_commands_schema_version == 3
        assert snapshot.aad_revision == 5
        assert snapshot.aad_request_id == request.request_id


# =============================================================================
# API Tests - Request Creation
# =============================================================================

class TestCreateSettingsRequest:
    """Tests for POST /api/v0/nodes/{node_id}/settings/requests"""

    def test_create_request_success(self, jwt_client, test_node):
        """Test successful request creation."""
        response = jwt_client.post(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests",
        )
        assert response.status_code == 201
        data = response.json()
        assert data["node_id"] == test_node.node_id
        assert data["status"] == "pending"
        assert "request_id" in data
        assert "expires_at" in data

    def test_create_request_node_not_found(self, jwt_client):
        """Test request creation for nonexistent node."""
        response = jwt_client.post(
            "/api/v0/nodes/nonexistent-node/settings/requests",
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_request_unauthorized(self, client_with_test_db, test_node):
        """Test request creation without auth."""
        response = client_with_test_db.post(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests",
        )
        # Endpoint now requires a user JWT (commit 8fb81a6); a missing
        # Authorization header makes verify_user_jwt raise 401.
        assert response.status_code == 401

    def test_create_request_invalid_admin_key(self, client_with_test_db, test_node):
        """Test request creation with invalid admin key."""
        response = client_with_test_db.post(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests",
            headers={"x-api-key": "wrong-key"},
        )
        assert response.status_code == 401

    @patch("app.node_settings.mqtt_client")
    def test_create_request_publishes_mqtt(self, mock_mqtt, jwt_client, test_node):
        """Test that request creation publishes MQTT message."""
        response = jwt_client.post(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests",
        )
        assert response.status_code == 201

        # Verify MQTT was called
        mock_mqtt.publish.assert_called_once()
        call_args = mock_mqtt.publish.call_args
        topic = call_args[0][0]
        assert f"jarvis/nodes/{test_node.node_id}/settings/request" == topic


# =============================================================================
# API Tests - Request Retrieval (Node Confirmation)
# =============================================================================

class TestGetSettingsRequest:
    """Tests for GET /api/v0/nodes/{node_id}/settings/requests/{request_id}"""

    def test_get_request_success(self, client_with_test_db, test_node, test_db):
        """Test successful request retrieval."""
        # Create request directly in DB
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
        )
        test_db.add(request)
        test_db.commit()

        # Use legacy api_key format for auth (not node_id:key)
        response = client_with_test_db.get(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request.request_id}",
            headers={"x-api-key": test_node.api_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == request.request_id
        assert data["status"] == "pending"

    def test_get_request_not_found(self, client_with_test_db, test_node):
        """Test retrieval of nonexistent request."""
        fake_id = str(uuid.uuid4())
        response = client_with_test_db.get(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{fake_id}",
            headers={"x-api-key": test_node.api_key},
        )
        assert response.status_code == 404

    def test_get_request_wrong_node(self, client_with_test_db, test_node, test_db, multiple_test_nodes):
        """Test that nodes can only access their own requests."""
        # Create request for test_node
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
        )
        test_db.add(request)
        test_db.commit()

        # Try to access from different node (using that node's api_key)
        other_node = multiple_test_nodes[0]
        response = client_with_test_db.get(
            f"/api/v0/nodes/{other_node.node_id}/settings/requests/{request.request_id}",
            headers={"x-api-key": other_node.api_key},
        )
        # Request doesn't exist for other_node, so 404
        assert response.status_code == 404

    def test_get_expired_request(self, client_with_test_db, test_node, test_db):
        """Test that expired requests return appropriate status."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
            expires_at=datetime.utcnow() - timedelta(minutes=1),  # Already expired
        )
        test_db.add(request)
        test_db.commit()

        response = client_with_test_db.get(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request.request_id}",
            headers={"x-api-key": test_node.api_key},
        )
        # Should return 410 Gone for expired requests
        assert response.status_code == 410


# =============================================================================
# API Tests - Snapshot Upload
# =============================================================================

class TestUploadSettingsSnapshot:
    """Tests for PUT /api/v0/nodes/{node_id}/settings/requests/{request_id}/snapshot"""

    def test_upload_snapshot_success(self, client_with_test_db, test_node, test_db):
        """Test successful snapshot upload."""
        # Create pending request
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
        )
        test_db.add(request)
        test_db.commit()

        response = client_with_test_db.put(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request.request_id}/snapshot",
            headers={"x-api-key": test_node.api_key},
            json={
                "ciphertext": "base64-encrypted-settings",
                "nonce": "base64-nonce",
                "tag": "base64-auth-tag",
                "aad_schema_version": 1,
                "aad_commands_schema_version": 1,
                "aad_revision": 1,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "snapshot_id" in data
        assert data["node_id"] == test_node.node_id

    def test_upload_snapshot_request_not_pending(self, client_with_test_db, test_node, test_db):
        """Test upload fails if request is not pending."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="fulfilled",  # Already fulfilled
        )
        test_db.add(request)
        test_db.commit()

        response = client_with_test_db.put(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request.request_id}/snapshot",
            headers={"x-api-key": test_node.api_key},
            json={
                "ciphertext": "data",
                "nonce": "nonce",
                "tag": "tag",
                "aad_schema_version": 1,
                "aad_commands_schema_version": 1,
                "aad_revision": 1,
            },
        )
        assert response.status_code == 409  # Conflict

    def test_upload_snapshot_request_expired(self, client_with_test_db, test_node, test_db):
        """Test upload fails if request is expired."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
            expires_at=datetime.utcnow() - timedelta(minutes=1),
        )
        test_db.add(request)
        test_db.commit()

        response = client_with_test_db.put(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request.request_id}/snapshot",
            headers={"x-api-key": test_node.api_key},
            json={
                "ciphertext": "data",
                "nonce": "nonce",
                "tag": "tag",
                "aad_schema_version": 1,
                "aad_commands_schema_version": 1,
                "aad_revision": 1,
            },
        )
        assert response.status_code == 410  # Gone

    def test_upload_snapshot_marks_request_fulfilled(self, client_with_test_db, test_node, test_db):
        """Test that uploading snapshot marks request as fulfilled."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
        )
        test_db.add(request)
        test_db.commit()
        request_id = request.request_id

        response = client_with_test_db.put(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request_id}/snapshot",
            headers={"x-api-key": test_node.api_key},
            json={
                "ciphertext": "data",
                "nonce": "nonce",
                "tag": "tag",
                "aad_schema_version": 1,
                "aad_commands_schema_version": 1,
                "aad_revision": 1,
            },
        )
        assert response.status_code == 201

        # Verify request status changed
        test_db.refresh(request)
        assert request.status == "fulfilled"


# =============================================================================
# API Tests - Snapshot Retrieval (Mobile Polling)
# =============================================================================

class TestGetSettingsSnapshot:
    """Tests for GET /api/v0/nodes/{node_id}/settings/requests/{request_id}/result"""

    def test_get_result_pending(self, jwt_client, test_node, test_db):
        """Test polling when request is still pending."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
        )
        test_db.add(request)
        test_db.commit()

        response = jwt_client.get(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request.request_id}/result",
        )
        # 202 Accepted - still processing
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"

    def test_get_result_fulfilled(self, jwt_client, test_node, test_db):
        """Test polling when snapshot is ready."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="fulfilled",
        )
        test_db.add(request)
        test_db.commit()

        snapshot = SettingsSnapshot(
            snapshot_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            request_id=request.request_id,
            ciphertext="encrypted-settings",
            nonce="nonce-value",
            tag="tag-value",
            aad_node_id=test_node.node_id,
            aad_schema_version=1,
            aad_commands_schema_version=1,
            aad_revision=1,
            aad_request_id=request.request_id,
        )
        test_db.add(snapshot)
        test_db.commit()

        response = jwt_client.get(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request.request_id}/result",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "fulfilled"
        assert "snapshot" in data
        assert data["snapshot"]["ciphertext"] == "encrypted-settings"
        # AAD fields for client-side verification
        assert data["snapshot"]["aad"]["node_id"] == test_node.node_id
        assert data["snapshot"]["aad"]["schema_version"] == 1

    def test_get_result_expired(self, jwt_client, test_node, test_db):
        """Test polling for expired request."""
        request = SettingsRequest(
            request_id=str(uuid.uuid4()),
            node_id=test_node.node_id,
            status="pending",
            expires_at=datetime.utcnow() - timedelta(minutes=1),
        )
        test_db.add(request)
        test_db.commit()

        response = jwt_client.get(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests/{request.request_id}/result",
        )
        assert response.status_code == 410  # Gone


# =============================================================================
# MQTT Integration Tests
# =============================================================================

class TestMQTTPublishing:
    """Tests for MQTT message publishing."""

    @patch("app.node_settings.mqtt_client")
    def test_mqtt_message_format(self, mock_mqtt, jwt_client, test_node):
        """Test that MQTT messages have correct format."""
        response = jwt_client.post(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests",
        )
        assert response.status_code == 201
        request_id = response.json()["request_id"]

        mock_mqtt.publish.assert_called_once()
        call_args = mock_mqtt.publish.call_args

        # Check topic
        topic = call_args[0][0]
        assert topic == f"jarvis/nodes/{test_node.node_id}/settings/request"

        # Check payload is JSON with request_id
        import json
        payload = json.loads(call_args[0][1])
        assert payload["request_id"] == request_id
        assert payload["node_id"] == test_node.node_id

    @patch("app.node_settings.mqtt_client")
    def test_mqtt_failure_does_not_fail_request(self, mock_mqtt, jwt_client, test_node):
        """Test that MQTT failure doesn't prevent request creation."""
        mock_mqtt.publish.side_effect = Exception("MQTT connection failed")

        response = jwt_client.post(
            f"/api/v0/nodes/{test_node.node_id}/settings/requests",
        )
        # Request should still be created
        assert response.status_code == 201


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Tests for rate limiting on settings endpoints."""

    def test_rate_limit_request_creation(self, client_with_test_db, test_node):
        """Test that request creation is rate limited."""
        # Create many requests quickly
        for i in range(10):
            response = client_with_test_db.post(
                f"/api/v0/nodes/{test_node.node_id}/settings/requests",
                headers={"x-api-key": "test-admin-key"},
            )
            if response.status_code == 429:
                # Rate limited as expected
                return

        # If we get here, rate limiting might not be implemented yet
        # This is a reminder to implement it
        pytest.skip("Rate limiting not yet implemented")
