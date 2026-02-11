"""Tests for provisioning token endpoints.

Covers token creation (admin key + JWT), node registration, token consumption,
expiration, cleanup, room resolution, and refresh flows.
"""
import hashlib
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Node, ProvisioningToken
from app.provisioning import _hash_token, cleanup_expired_tokens


# ============================================================
# Fixtures
# ============================================================

ADMIN_KEY = "test-admin-key"
FAKE_NODE_KEY = "generated-node-key-abc123"
FAKE_JWT_USER = {"id": 42, "email": "user@example.com"}


@pytest.fixture
def prov_client(test_db):
    """Test client with DB override and mocked external deps."""
    from app.deps import get_db

    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _admin_headers() -> dict[str, str]:
    return {"X-API-Key": ADMIN_KEY}


def _jwt_headers() -> dict[str, str]:
    return {"Authorization": "Bearer fake-jwt-token"}


def _create_token(client: TestClient, headers: dict, **kwargs) -> dict:
    """Helper to create a provisioning token."""
    body = {"household_id": "hh-001", **kwargs}
    resp = client.post("/api/v0/provisioning/token", json=body, headers=headers)
    return resp.json(), resp.status_code


# ============================================================
# Token Creation Tests
# ============================================================


class TestCreateToken:
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_create_token_with_admin_key(self, prov_client):
        """201: admin key auth returns token with prov_ prefix and valid UUID node_id."""
        data, status = _create_token(prov_client, _admin_headers())

        assert status == 201
        assert data["token"].startswith("prov_")
        assert data["expires_in"] == 600
        # node_id should be a valid UUID
        uuid.UUID(data["node_id"])

    @patch("app.provisioning._validate_jwt", return_value=FAKE_JWT_USER)
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_create_token_with_jwt(self, mock_jwt, prov_client):
        """201: JWT auth returns token with valid UUID node_id."""
        data, status = _create_token(prov_client, _jwt_headers())

        assert status == 201
        uuid.UUID(data["node_id"])
        mock_jwt.assert_called_once()

    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_create_token_no_auth(self, prov_client):
        """401: no auth header."""
        resp = prov_client.post(
            "/api/v0/provisioning/token",
            json={"household_id": "hh-001"},
        )
        assert resp.status_code == 401

    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_create_token_invalid_admin_key(self, prov_client):
        """401: wrong admin key."""
        resp = prov_client.post(
            "/api/v0/provisioning/token",
            json={"household_id": "hh-001"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_create_token_generates_unique_guids(self, prov_client):
        """Two token creations produce different node_ids."""
        data1, _ = _create_token(prov_client, _admin_headers())
        data2, _ = _create_token(prov_client, _admin_headers())

        assert data1["node_id"] != data2["node_id"]


# ============================================================
# Node Registration Tests
# ============================================================


class TestRegisterNode:
    @patch("app.provisioning._register_node_with_auth", return_value=FAKE_NODE_KEY)
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_register_with_valid_token(self, mock_auth, prov_client, test_db):
        """201: valid token registers node, returns node_key."""
        data, _ = _create_token(prov_client, _admin_headers())
        resp = prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
        })

        assert resp.status_code == 201
        body = resp.json()
        assert body["node_key"] == FAKE_NODE_KEY
        assert body["node_id"] == data["node_id"]

        # Verify node exists in DB
        node = test_db.query(Node).filter(Node.node_id == data["node_id"]).first()
        assert node is not None

    @patch("app.provisioning._register_node_with_auth", return_value=FAKE_NODE_KEY)
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_register_consumes_token(self, mock_auth, prov_client):
        """Second registration with same token returns 401."""
        data, _ = _create_token(prov_client, _admin_headers())
        # First registration succeeds
        resp1 = prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
        })
        assert resp1.status_code == 201

        # Second attempt fails — token already consumed so validation fails first
        resp2 = prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
        })
        assert resp2.status_code == 401

    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_register_expired_token(self, prov_client, test_db):
        """401: expired token is rejected."""
        # Create token then manually expire it
        data, _ = _create_token(prov_client, _admin_headers())
        token_hash = _hash_token(data["token"])
        prov_record = test_db.query(ProvisioningToken).filter(
            ProvisioningToken.token_hash == token_hash
        ).first()
        prov_record.expires_at = datetime.utcnow() - timedelta(minutes=1)
        test_db.commit()

        resp = prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
        })
        assert resp.status_code == 401

    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_register_wrong_node_id(self, prov_client):
        """401: token used with wrong node_id."""
        data, _ = _create_token(prov_client, _admin_headers())
        resp = prov_client.post("/api/v0/nodes/register", json={
            "node_id": str(uuid.uuid4()),  # Different node_id
            "provisioning_token": data["token"],
        })
        assert resp.status_code == 401

    @patch("app.provisioning._register_node_with_auth", return_value=FAKE_NODE_KEY)
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_register_node_already_exists(self, mock_auth, prov_client, test_db):
        """400: reject if node already registered in nodes table."""
        data, _ = _create_token(prov_client, _admin_headers())

        # Pre-create the node
        existing = Node(
            node_id=data["node_id"],
            api_key="existing-key",
            room="kitchen",
        )
        test_db.add(existing)
        test_db.commit()

        resp = prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
        })
        assert resp.status_code == 400
        assert "already registered" in resp.json()["detail"].lower()


# ============================================================
# Token Storage Tests
# ============================================================


class TestTokenStorage:
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_token_hash_not_stored_raw(self, prov_client, test_db):
        """DB stores SHA-256 hash, not the raw token."""
        data, _ = _create_token(prov_client, _admin_headers())
        raw_token = data["token"]
        expected_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        prov_record = test_db.query(ProvisioningToken).filter(
            ProvisioningToken.node_id == data["node_id"]
        ).first()

        assert prov_record is not None
        assert prov_record.token_hash != raw_token
        assert prov_record.token_hash == expected_hash


# ============================================================
# Cleanup Tests
# ============================================================


class TestCleanup:
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_cleanup_expired_tokens(self, prov_client, test_db):
        """Expired tokens older than 24h are removed."""
        data, _ = _create_token(prov_client, _admin_headers())

        # Expire it far in the past
        token_hash = _hash_token(data["token"])
        prov_record = test_db.query(ProvisioningToken).filter(
            ProvisioningToken.token_hash == token_hash
        ).first()
        prov_record.expires_at = datetime.utcnow() - timedelta(hours=25)
        test_db.commit()

        removed = cleanup_expired_tokens(test_db)
        assert removed >= 1

        # Verify it's gone
        remaining = test_db.query(ProvisioningToken).filter(
            ProvisioningToken.token_hash == token_hash
        ).first()
        assert remaining is None

    @patch("app.provisioning._register_node_with_auth", return_value=FAKE_NODE_KEY)
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_cleanup_consumed_tokens(self, mock_auth, prov_client, test_db):
        """Consumed tokens older than 24h are removed."""
        data, _ = _create_token(prov_client, _admin_headers())

        # Register (consumes token)
        prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
        })

        # Backdate consumed_at
        token_hash = _hash_token(data["token"])
        prov_record = test_db.query(ProvisioningToken).filter(
            ProvisioningToken.token_hash == token_hash
        ).first()
        prov_record.consumed_at = datetime.utcnow() - timedelta(hours=25)
        test_db.commit()

        removed = cleanup_expired_tokens(test_db)
        assert removed >= 1

    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_cleanup_preserves_active_tokens(self, prov_client, test_db):
        """Active (unexpired, unconsumed) tokens are not removed."""
        data, _ = _create_token(prov_client, _admin_headers())

        removed = cleanup_expired_tokens(test_db)

        # Token should still exist
        token_hash = _hash_token(data["token"])
        prov_record = test_db.query(ProvisioningToken).filter(
            ProvisioningToken.token_hash == token_hash
        ).first()
        assert prov_record is not None


# ============================================================
# Refresh Token Tests
# ============================================================


class TestRefreshToken:
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_refresh_token_same_guid(self, prov_client, test_db):
        """Refreshing reuses the same GUID and invalidates old token."""
        data1, _ = _create_token(prov_client, _admin_headers())
        node_id = data1["node_id"]

        # Refresh: pass node_id back
        data2, status = _create_token(prov_client, _admin_headers(), node_id=node_id)
        assert status == 201
        assert data2["node_id"] == node_id
        assert data2["token"] != data1["token"]

        # Old token should be invalidated (consumed_at is still None but it was deleted)
        old_hash = _hash_token(data1["token"])
        old_record = test_db.query(ProvisioningToken).filter(
            ProvisioningToken.token_hash == old_hash
        ).first()
        assert old_record is None

    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_refresh_token_registered_node_fails(self, prov_client, test_db):
        """400: can't refresh token for an already-registered node."""
        node_id = str(uuid.uuid4())
        # Pre-create the node
        existing = Node(node_id=node_id, api_key="key", room="room")
        test_db.add(existing)
        test_db.commit()

        resp = prov_client.post(
            "/api/v0/provisioning/token",
            json={"household_id": "hh-001", "node_id": node_id},
            headers=_admin_headers(),
        )
        assert resp.status_code == 400
        assert "already registered" in resp.json()["detail"].lower()


# ============================================================
# Room Resolution Tests
# ============================================================


class TestRoomResolution:
    @patch("app.provisioning._register_node_with_auth", return_value=FAKE_NODE_KEY)
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_register_with_room_override(self, mock_auth, prov_client, test_db):
        """Request room overrides token room."""
        data, _ = _create_token(prov_client, _admin_headers(), room="kitchen")

        resp = prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
            "room": "bedroom",
        })
        assert resp.status_code == 201
        assert resp.json()["room"] == "bedroom"

    @patch("app.provisioning._register_node_with_auth", return_value=FAKE_NODE_KEY)
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_register_without_room_uses_token_room(self, mock_auth, prov_client, test_db):
        """When request has no room, token room is used."""
        data, _ = _create_token(prov_client, _admin_headers(), room="kitchen")

        resp = prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
        })
        assert resp.status_code == 201
        assert resp.json()["room"] == "kitchen"

    @patch("app.provisioning._register_node_with_auth", return_value=FAKE_NODE_KEY)
    @patch("app.provisioning.ADMIN_API_KEY", ADMIN_KEY)
    def test_register_without_any_room_uses_default(self, mock_auth, prov_client):
        """When neither request nor token has a room, 'default' is used."""
        data, _ = _create_token(prov_client, _admin_headers())  # No room

        resp = prov_client.post("/api/v0/nodes/register", json={
            "node_id": data["node_id"],
            "provisioning_token": data["token"],
        })
        assert resp.status_code == 201
        assert resp.json()["room"] == "default"
