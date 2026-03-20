"""Tests for device-list endpoints (smart home).

Covers the three device-list endpoints:
1. POST /nodes/{node_id}/device-list/request — mobile requests a device list
2. POST /nodes/{node_id}/device-list/{request_id}/results — node uploads results
3. GET  /nodes/{node_id}/device-list/{request_id} — mobile polls for results

Follows the same request/poll pattern as settings (test_node_settings.py).
"""

import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Device, DeviceListRequest, Node, Room
from app.deps import get_db, verify_api_key
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device_list_client(test_db):
    """Test client with DB + auth overrides for device-list endpoints.

    Request and poll endpoints use verify_provisioning_auth (mobile/admin).
    Upload endpoint uses verify_api_key (node auth).
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


def _create_device_list_request(
    db,
    node: Node,
    *,
    status: str = "pending",
    manager_name: str = "jarvis_direct",
    expired: bool = False,
    results_json: str | None = None,
    error_message: str | None = None,
    can_edit_devices: bool | None = None,
    device_count: int | None = None,
) -> DeviceListRequest:
    """Create a DeviceListRequest record directly in the DB."""
    now = datetime.utcnow()
    expires_at = now - timedelta(minutes=1) if expired else now + timedelta(minutes=2)

    req = DeviceListRequest(
        id=str(uuid.uuid4()),
        node_id=node.node_id,
        household_id=node.household_id or "",
        status=status,
        manager_name=manager_name,
        can_edit_devices=can_edit_devices,
        results_json=results_json,
        device_count=device_count,
        error_message=error_message,
        created_at=now,
        expires_at=expires_at,
        completed_at=now if status in ("completed", "failed") else None,
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req


def _sample_devices() -> list[dict]:
    """Sample device list as the node would upload."""
    return [
        {
            "name": "Living Room Light",
            "domain": "light",
            "entity_id": "light.living_room",
            "is_controllable": True,
            "manufacturer": "LIFX",
            "protocol": "lifx",
            "mac_address": "AA:BB:CC:DD:EE:01",
            "source": "direct",
        },
        {
            "name": "Front Door Lock",
            "domain": "lock",
            "entity_id": "lock.front_door",
            "is_controllable": True,
            "manufacturer": "Schlage",
            "cloud_id": "schlage-001",
            "source": "cloud",
        },
    ]


# =============================================================================
# Request Endpoint: POST /nodes/{node_id}/device-list/request
# =============================================================================


class TestRequestDeviceList:
    """Tests for POST /api/v0/nodes/{node_id}/device-list/request"""

    @patch("app.api.smart_home._publish_device_list_mqtt")
    @patch("app.services.settings_service.get_settings_service")
    def test_request_creates_record_and_returns_201(
        self, mock_settings, mock_mqtt, device_list_client, test_db
    ):
        """Successful request creates a DeviceListRequest and returns 201."""
        mock_svc = MagicMock()
        mock_svc.get.return_value = "jarvis_direct"
        mock_settings.return_value = mock_svc

        node = _create_node(test_db)

        resp = device_list_client.post(
            f"/api/v0/nodes/{node.node_id}/device-list/request",
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert "id" in data
        assert "created_at" in data

        # Verify DB record
        db_req = test_db.query(DeviceListRequest).filter(
            DeviceListRequest.id == data["id"]
        ).first()
        assert db_req is not None
        assert db_req.node_id == node.node_id
        assert db_req.status == "pending"
        assert db_req.manager_name == "jarvis_direct"

    @patch("app.api.smart_home._publish_device_list_mqtt")
    @patch("app.services.settings_service.get_settings_service")
    def test_request_publishes_mqtt(
        self, mock_settings, mock_mqtt, device_list_client, test_db
    ):
        """Request endpoint publishes MQTT to notify the node."""
        mock_svc = MagicMock()
        mock_svc.get.return_value = "home_assistant"
        mock_settings.return_value = mock_svc

        node = _create_node(test_db)

        resp = device_list_client.post(
            f"/api/v0/nodes/{node.node_id}/device-list/request",
        )
        assert resp.status_code == 201

        mock_mqtt.assert_called_once()
        call_args = mock_mqtt.call_args
        assert call_args[0][0] == node.node_id
        assert call_args[0][1] == resp.json()["id"]
        assert call_args[0][2] == "home_assistant"

    @patch("app.api.smart_home._publish_device_list_mqtt")
    @patch("app.services.settings_service.get_settings_service")
    def test_request_node_not_found(
        self, mock_settings, mock_mqtt, device_list_client
    ):
        """Returns 404 for nonexistent node."""
        resp = device_list_client.post(
            "/api/v0/nodes/nonexistent-node/device-list/request",
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
        mock_mqtt.assert_not_called()


# =============================================================================
# Upload Endpoint: POST /nodes/{node_id}/device-list/{request_id}/results
# =============================================================================


class TestUploadDeviceListResults:
    """Tests for POST /api/v0/nodes/{node_id}/device-list/{request_id}/results"""

    def test_upload_success(self, device_list_client, test_db):
        """Node uploads devices, request status becomes completed."""
        node = _create_node(test_db)
        req = _create_device_list_request(test_db, node)
        devices = _sample_devices()

        resp = device_list_client.post(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}/results",
            json={
                "devices": devices,
                "manager_name": "jarvis_direct",
                "can_edit_devices": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify DB
        test_db.refresh(req)
        assert req.status == "completed"
        assert req.device_count == 2
        assert req.manager_name == "jarvis_direct"
        assert req.can_edit_devices is True
        assert req.completed_at is not None

        stored = json.loads(req.results_json)
        assert len(stored) == 2
        assert stored[0]["entity_id"] == "light.living_room"

    def test_upload_with_error(self, device_list_client, test_db):
        """Node uploads an error, request status becomes failed."""
        node = _create_node(test_db)
        req = _create_device_list_request(test_db, node)

        resp = device_list_client.post(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}/results",
            json={
                "devices": [],
                "error": "Connection refused by Home Assistant",
            },
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        assert req.status == "failed"
        assert req.error_message == "Connection refused by Home Assistant"

    def test_upload_updates_manager_name(self, device_list_client, test_db):
        """Node can override manager_name from the original request."""
        node = _create_node(test_db)
        req = _create_device_list_request(test_db, node, manager_name="jarvis_direct")

        resp = device_list_client.post(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}/results",
            json={
                "devices": _sample_devices(),
                "manager_name": "home_assistant",
                "can_edit_devices": False,
            },
        )
        assert resp.status_code == 200

        test_db.refresh(req)
        assert req.manager_name == "home_assistant"
        assert req.can_edit_devices is False

    def test_upload_request_not_found(self, device_list_client, test_db):
        """Returns 404 for nonexistent request."""
        node = _create_node(test_db)
        fake_id = str(uuid.uuid4())

        resp = device_list_client.post(
            f"/api/v0/nodes/{node.node_id}/device-list/{fake_id}/results",
            json={"devices": []},
        )
        assert resp.status_code == 404

    def test_upload_expired_request(self, device_list_client, test_db):
        """Returns 410 for expired request."""
        node = _create_node(test_db)
        req = _create_device_list_request(test_db, node, expired=True)

        resp = device_list_client.post(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}/results",
            json={"devices": _sample_devices()},
        )
        assert resp.status_code == 410

        test_db.refresh(req)
        assert req.status == "expired"

    def test_upload_wrong_node(self, device_list_client, test_db):
        """Returns 404 when node_id in URL does not match request's node."""
        node_a = _create_node(test_db)
        node_b = _create_node(test_db)
        req = _create_device_list_request(test_db, node_a)

        resp = device_list_client.post(
            f"/api/v0/nodes/{node_b.node_id}/device-list/{req.id}/results",
            json={"devices": []},
        )
        assert resp.status_code == 404


# =============================================================================
# Poll Endpoint: GET /nodes/{node_id}/device-list/{request_id}
# =============================================================================


class TestPollDeviceList:
    """Tests for GET /api/v0/nodes/{node_id}/device-list/{request_id}"""

    def test_poll_pending(self, device_list_client, test_db):
        """Returns pending status when node has not yet responded."""
        node = _create_node(test_db)
        req = _create_device_list_request(test_db, node, manager_name="jarvis_direct")

        resp = device_list_client.get(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["request_id"] == req.id
        assert data["manager_name"] == "jarvis_direct"
        assert data["devices"] is None

    def test_poll_completed_with_enrichment(self, device_list_client, test_db):
        """Returns enriched devices with room assignments and already_registered flags."""
        node = _create_node(test_db, household_id="hh-enrich")

        # Create a room
        room = Room(
            id=str(uuid.uuid4()),
            household_id="hh-enrich",
            name="Living Room",
            normalized_name="living room",
        )
        test_db.add(room)
        test_db.commit()

        # Create an existing registered device (matches by entity_id)
        existing_device = Device(
            id=str(uuid.uuid4()),
            household_id="hh-enrich",
            room_id=room.id,
            entity_id="light.living_room",
            name="Living Room Light",
            domain="light",
            is_active=True,
        )
        test_db.add(existing_device)
        test_db.commit()

        devices = _sample_devices()
        req = _create_device_list_request(
            test_db,
            node,
            status="completed",
            results_json=json.dumps(devices),
            device_count=len(devices),
            manager_name="jarvis_direct",
            can_edit_devices=True,
        )

        resp = device_list_client.get(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["request_id"] == req.id
        assert data["manager_name"] == "jarvis_direct"
        assert data["can_edit_devices"] is True
        assert data["device_count"] == 2

        # First device should be matched (already_registered=True, with room info)
        light = data["devices"][0]
        assert light["entity_id"] == "light.living_room"
        assert light["already_registered"] is True
        assert light["existing_device_id"] == existing_device.id
        assert light["room_id"] == room.id
        assert light["room_name"] == "Living Room"

        # Second device should not be matched
        lock = data["devices"][1]
        assert lock["entity_id"] == "lock.front_door"
        assert lock["already_registered"] is False
        assert lock["existing_device_id"] is None
        assert lock["room_id"] is None

    def test_poll_completed_match_by_cloud_id(self, device_list_client, test_db):
        """Devices matched by cloud_id are marked as already_registered."""
        node = _create_node(test_db, household_id="hh-cloud")

        existing_device = Device(
            id=str(uuid.uuid4()),
            household_id="hh-cloud",
            entity_id="lock.schlage_front",
            name="Front Door Lock",
            domain="lock",
            cloud_id="schlage-001",
            is_active=True,
        )
        test_db.add(existing_device)
        test_db.commit()

        devices = [
            {
                "name": "Front Door Lock",
                "domain": "lock",
                "entity_id": "lock.front_door",  # different entity_id
                "cloud_id": "schlage-001",  # matches by cloud_id
                "source": "cloud",
            }
        ]
        req = _create_device_list_request(
            test_db,
            node,
            status="completed",
            results_json=json.dumps(devices),
            device_count=1,
        )

        resp = device_list_client.get(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}",
        )
        assert resp.status_code == 200
        lock = resp.json()["devices"][0]
        assert lock["already_registered"] is True
        assert lock["existing_device_id"] == existing_device.id

    def test_poll_completed_match_by_mac_address(self, device_list_client, test_db):
        """Devices matched by MAC address are marked as already_registered."""
        node = _create_node(test_db, household_id="hh-mac")

        existing_device = Device(
            id=str(uuid.uuid4()),
            household_id="hh-mac",
            entity_id="light.lifx_bulb",
            name="LIFX Bulb",
            domain="light",
            mac_address="AA:BB:CC:DD:EE:01",
            is_active=True,
        )
        test_db.add(existing_device)
        test_db.commit()

        devices = [
            {
                "name": "Living Room Light",
                "domain": "light",
                "entity_id": "light.different_id",  # different entity_id
                "mac_address": "aa:bb:cc:dd:ee:01",  # same MAC, different case
                "source": "direct",
            }
        ]
        req = _create_device_list_request(
            test_db,
            node,
            status="completed",
            results_json=json.dumps(devices),
            device_count=1,
        )

        resp = device_list_client.get(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}",
        )
        assert resp.status_code == 200
        light = resp.json()["devices"][0]
        assert light["already_registered"] is True
        assert light["existing_device_id"] == existing_device.id

    def test_poll_failed(self, device_list_client, test_db):
        """Returns error_message when request has failed status."""
        node = _create_node(test_db)
        req = _create_device_list_request(
            test_db,
            node,
            status="failed",
            error_message="Device manager not found",
            manager_name="jarvis_direct",
        )

        resp = device_list_client.get(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error_message"] == "Device manager not found"
        assert data["devices"] is None

    def test_poll_expired(self, device_list_client, test_db):
        """Returns 410 for expired request."""
        node = _create_node(test_db)
        req = _create_device_list_request(test_db, node, expired=True)

        resp = device_list_client.get(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}",
        )
        assert resp.status_code == 410

    def test_poll_marks_pending_as_expired_when_past_deadline(
        self, device_list_client, test_db
    ):
        """Pending request past expires_at is transitioned to expired and returns 410."""
        node = _create_node(test_db)
        req = _create_device_list_request(test_db, node, expired=True)

        resp = device_list_client.get(
            f"/api/v0/nodes/{node.node_id}/device-list/{req.id}",
        )
        assert resp.status_code == 410

        # Verify DB was updated
        test_db.refresh(req)
        assert req.status == "expired"

    def test_poll_request_not_found(self, device_list_client, test_db):
        """Returns 404 for nonexistent request."""
        node = _create_node(test_db)
        fake_id = str(uuid.uuid4())

        resp = device_list_client.get(
            f"/api/v0/nodes/{node.node_id}/device-list/{fake_id}",
        )
        assert resp.status_code == 404

    def test_poll_wrong_node(self, device_list_client, test_db):
        """Returns 404 when node_id does not own the request."""
        node_a = _create_node(test_db)
        node_b = _create_node(test_db)
        req = _create_device_list_request(test_db, node_a)

        resp = device_list_client.get(
            f"/api/v0/nodes/{node_b.node_id}/device-list/{req.id}",
        )
        assert resp.status_code == 404


# =============================================================================
# MQTT Helper Tests
# =============================================================================


class TestPublishDeviceListMqtt:
    """Tests for _publish_device_list_mqtt helper."""

    @patch("app.node_settings.get_mqtt_client")
    def test_publishes_correct_topic_and_payload(self, mock_get_mqtt):
        """MQTT message has the right topic and JSON payload."""
        from app.api.smart_home import _publish_device_list_mqtt

        mock_client = MagicMock()
        mock_get_mqtt.return_value = mock_client

        _publish_device_list_mqtt("node-123", "req-abc", "home_assistant")

        mock_client.publish.assert_called_once()
        topic, payload = mock_client.publish.call_args[0]
        assert topic == "jarvis/nodes/node-123/device-list"

        parsed = json.loads(payload)
        assert parsed["request_id"] == "req-abc"
        assert parsed["manager_name"] == "home_assistant"

    @patch("app.node_settings.get_mqtt_client")
    def test_mqtt_unavailable_does_not_raise(self, mock_get_mqtt):
        """When MQTT client is None, the function logs a warning but does not raise."""
        from app.api.smart_home import _publish_device_list_mqtt

        mock_get_mqtt.return_value = None

        # Should not raise
        _publish_device_list_mqtt("node-123", "req-abc", "jarvis_direct")

    @patch("app.node_settings.get_mqtt_client")
    def test_mqtt_publish_failure_does_not_raise(self, mock_get_mqtt):
        """If MQTT publish throws, the error is caught (not propagated)."""
        from app.api.smart_home import _publish_device_list_mqtt

        mock_client = MagicMock()
        mock_client.publish.side_effect = ConnectionError("broker down")
        mock_get_mqtt.return_value = mock_client

        # Should not raise
        _publish_device_list_mqtt("node-123", "req-abc", "jarvis_direct")
