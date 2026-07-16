"""Bluetooth management API: scan, pair, disconnect, discoverable, status.

Mobile → CC → MQTT → node → CC → mobile polling pattern.
Follows the same request/poll lifecycle as device-scan in smart_home.py.
"""

import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.deps import get_db, verify_api_key
from app.models import BluetoothPairRequest, BluetoothScanRequest, Node
from app.provisioning import ProvisioningAuthContext, verify_provisioning_auth, require_household_access

router = APIRouter()
logger = logging.getLogger("uvicorn")


# =============================================================================
# Request/Response Models
# =============================================================================


class BluetoothScanRequestBody(BaseModel):
    role: str = "source"  # "source" (speaker) or "sink" (audio input)
    source: str = "mobile"  # "mobile" or "voice"
    user_id: int | None = None  # speaker user ID (for voice-triggered deep link push)


class BluetoothScanResponse(BaseModel):
    id: str
    status: str
    created_at: datetime


class BluetoothDeviceItem(BaseModel):
    name: str
    mac_address: str
    device_type: str = "unknown"
    paired: bool = False
    connected: bool = False


class BluetoothScanPollResponse(BaseModel):
    status: str
    request_id: str
    devices: list[BluetoothDeviceItem] | None = None
    device_count: int | None = None
    error_message: str | None = None


class BluetoothScanResultUpload(BaseModel):
    devices: list[dict] = []
    error: str | None = None


class BluetoothPairRequestBody(BaseModel):
    mac_address: str
    role: str = "source"


class BluetoothPairResponse(BaseModel):
    id: str
    status: str
    created_at: datetime


class BluetoothPairPollResponse(BaseModel):
    status: str
    request_id: str
    device_name: str | None = None
    error_message: str | None = None


class BluetoothPairResultUpload(BaseModel):
    success: bool
    device_name: str | None = None
    error: str | None = None


class BluetoothDisconnectBody(BaseModel):
    mac_address: str


class BluetoothStatusResponse(BaseModel):
    available: bool = True
    connected: list[BluetoothDeviceItem] = []
    paired: list[BluetoothDeviceItem] = []


# =============================================================================
# Scan Endpoints
# =============================================================================


@router.post(
    "/nodes/{node_id}/bluetooth-scan/request",
    response_model=BluetoothScanResponse,
    status_code=201,
)
def request_bluetooth_scan(
    node_id: str,
    body: BluetoothScanRequestBody,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> BluetoothScanResponse:
    """Request a Bluetooth scan on a node. Mobile/voice calls this, CC notifies node via MQTT."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    require_household_access(node.household_id, auth)

    now = datetime.utcnow()
    scan_request = BluetoothScanRequest(
        id=str(uuid4()),
        node_id=node_id,
        household_id=node.household_id or "",
        role=body.role,
        source=body.source,
        user_id=body.user_id,
        status="pending",
        created_at=now,
        expires_at=now + timedelta(minutes=2),
    )
    db.add(scan_request)
    db.commit()
    db.refresh(scan_request)

    logger.info("Bluetooth scan requested for node %s, request_id=%s", node_id, scan_request.id[:8])

    # Notify node via MQTT
    _publish_bluetooth_mqtt(
        node_id,
        "bluetooth-scan",
        {"request_id": scan_request.id, "role": body.role},
    )

    # If voice-triggered with a known user, send push notification with deep link
    if body.source == "voice" and body.user_id:
        _send_bluetooth_deep_link_push(
            household_id=node.household_id or "",
            user_id=body.user_id,
            node_id=node_id,
        )

    return BluetoothScanResponse(
        id=scan_request.id,
        status=scan_request.status,
        created_at=scan_request.created_at,
    )


@router.post("/nodes/{node_id}/bluetooth-scan/{request_id}/results")
def upload_bluetooth_scan_results(
    node_id: str,
    request_id: str,
    body: BluetoothScanResultUpload,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Node uploads Bluetooth scan results after running bluetoothctl scan."""
    scan_request = db.query(BluetoothScanRequest).filter(
        BluetoothScanRequest.id == request_id,
        BluetoothScanRequest.node_id == node_id,
    ).first()
    if not scan_request:
        raise HTTPException(status_code=404, detail="Scan request not found")

    now = datetime.utcnow()
    if scan_request.expires_at and scan_request.expires_at < now:
        scan_request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Scan request expired")

    if body.error:
        scan_request.status = "failed"
        scan_request.error_message = body.error
    else:
        scan_request.status = "completed"
        scan_request.results_json = json.dumps(body.devices)
        scan_request.device_count = len(body.devices)

    scan_request.completed_at = now
    db.commit()

    logger.info(
        "Bluetooth scan results uploaded: request=%s status=%s devices=%d",
        request_id[:8], scan_request.status, len(body.devices),
    )
    return {"status": "ok"}


@router.get(
    "/nodes/{node_id}/bluetooth-scan/{request_id}",
    response_model=BluetoothScanPollResponse,
)
def poll_bluetooth_scan(
    node_id: str,
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> BluetoothScanPollResponse:
    """Mobile polls for Bluetooth scan results."""
    scan_request = db.query(BluetoothScanRequest).filter(
        BluetoothScanRequest.id == request_id,
        BluetoothScanRequest.node_id == node_id,
    ).first()
    if not scan_request:
        raise HTTPException(status_code=404, detail="Scan request not found")

    now = datetime.utcnow()

    # Check expiration
    if scan_request.status == "pending" and scan_request.expires_at and scan_request.expires_at < now:
        scan_request.status = "expired"
        db.commit()

    if scan_request.status == "expired":
        raise HTTPException(status_code=410, detail="Scan request expired")

    if scan_request.status == "pending":
        return BluetoothScanPollResponse(status="pending", request_id=request_id)

    if scan_request.status == "failed":
        return BluetoothScanPollResponse(
            status="failed",
            request_id=request_id,
            error_message=scan_request.error_message,
        )

    # Completed
    raw_devices: list[dict] = json.loads(scan_request.results_json or "[]")
    devices = [
        BluetoothDeviceItem(
            name=d.get("name", "Unknown"),
            mac_address=d.get("mac_address", ""),
            device_type=d.get("device_type", "unknown"),
            paired=d.get("paired", False),
            connected=d.get("connected", False),
        )
        for d in raw_devices
    ]

    return BluetoothScanPollResponse(
        status="completed",
        request_id=request_id,
        devices=devices,
        device_count=len(devices),
    )


# =============================================================================
# Pair Endpoints
# =============================================================================


@router.post(
    "/nodes/{node_id}/bluetooth/pair",
    response_model=BluetoothPairResponse,
    status_code=201,
)
def request_bluetooth_pair(
    node_id: str,
    body: BluetoothPairRequestBody,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> BluetoothPairResponse:
    """Request pairing with a specific Bluetooth device on a node."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    require_household_access(node.household_id, auth)

    now = datetime.utcnow()
    pair_request = BluetoothPairRequest(
        id=str(uuid4()),
        node_id=node_id,
        household_id=node.household_id or "",
        mac_address=body.mac_address,
        role=body.role,
        status="pending",
        created_at=now,
        expires_at=now + timedelta(minutes=2),
    )
    db.add(pair_request)
    db.commit()
    db.refresh(pair_request)

    logger.info(
        "Bluetooth pair requested: node=%s mac=%s role=%s request_id=%s",
        node_id, body.mac_address, body.role, pair_request.id[:8],
    )

    _publish_bluetooth_mqtt(
        node_id,
        "bluetooth-pair",
        {"request_id": pair_request.id, "mac_address": body.mac_address, "role": body.role},
    )

    return BluetoothPairResponse(
        id=pair_request.id,
        status=pair_request.status,
        created_at=pair_request.created_at,
    )


@router.post("/nodes/{node_id}/bluetooth/pair/{request_id}/results")
def upload_bluetooth_pair_results(
    node_id: str,
    request_id: str,
    body: BluetoothPairResultUpload,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Node uploads Bluetooth pair result."""
    pair_request = db.query(BluetoothPairRequest).filter(
        BluetoothPairRequest.id == request_id,
        BluetoothPairRequest.node_id == node_id,
    ).first()
    if not pair_request:
        raise HTTPException(status_code=404, detail="Pair request not found")

    now = datetime.utcnow()
    if pair_request.expires_at and pair_request.expires_at < now:
        pair_request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Pair request expired")

    if body.success:
        pair_request.status = "completed"
        pair_request.device_name = body.device_name
    else:
        pair_request.status = "failed"
        pair_request.error_message = body.error

    pair_request.completed_at = now
    db.commit()

    logger.info(
        "Bluetooth pair result: request=%s status=%s",
        request_id[:8], pair_request.status,
    )
    return {"status": "ok"}


@router.get(
    "/nodes/{node_id}/bluetooth/pair/{request_id}",
    response_model=BluetoothPairPollResponse,
)
def poll_bluetooth_pair(
    node_id: str,
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> BluetoothPairPollResponse:
    """Mobile polls for Bluetooth pair result."""
    pair_request = db.query(BluetoothPairRequest).filter(
        BluetoothPairRequest.id == request_id,
        BluetoothPairRequest.node_id == node_id,
    ).first()
    if not pair_request:
        raise HTTPException(status_code=404, detail="Pair request not found")

    now = datetime.utcnow()

    if pair_request.status == "pending" and pair_request.expires_at and pair_request.expires_at < now:
        pair_request.status = "expired"
        db.commit()

    if pair_request.status == "expired":
        raise HTTPException(status_code=410, detail="Pair request expired")

    if pair_request.status == "pending":
        return BluetoothPairPollResponse(status="pending", request_id=request_id)

    if pair_request.status == "failed":
        return BluetoothPairPollResponse(
            status="failed",
            request_id=request_id,
            error_message=pair_request.error_message,
        )

    return BluetoothPairPollResponse(
        status="completed",
        request_id=request_id,
        device_name=pair_request.device_name,
    )


# =============================================================================
# Disconnect / Discoverable / Status
# =============================================================================


@router.post("/nodes/{node_id}/bluetooth/disconnect", status_code=202)
def request_bluetooth_disconnect(
    node_id: str,
    body: BluetoothDisconnectBody,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    """Disconnect a Bluetooth device on a node (fire-and-forget)."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    require_household_access(node.household_id, auth)

    _publish_bluetooth_mqtt(
        node_id,
        "bluetooth-disconnect",
        {"mac_address": body.mac_address},
    )

    logger.info("Bluetooth disconnect requested: node=%s mac=%s", node_id, body.mac_address)
    return {"status": "accepted"}


@router.post("/nodes/{node_id}/bluetooth/discoverable", status_code=202)
def request_bluetooth_discoverable(
    node_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    """Make a node discoverable for Bluetooth pairing (phone → Pi flow)."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    require_household_access(node.household_id, auth)

    _publish_bluetooth_mqtt(
        node_id,
        "bluetooth-discoverable",
        {"timeout": 120},
    )

    logger.info("Bluetooth discoverable requested: node=%s", node_id)
    return {"status": "accepted", "timeout_seconds": 120}


@router.get(
    "/nodes/{node_id}/bluetooth/status",
    response_model=BluetoothStatusResponse,
)
def get_bluetooth_status(
    node_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> BluetoothStatusResponse:
    """Get current Bluetooth status for a node.

    Returns the most recent completed scan results as a proxy for current state.
    For real-time status, the mobile app should trigger a fresh scan.
    """
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    require_household_access(node.household_id, auth)

    # Return most recent completed scan as cached status
    recent_scan = db.query(BluetoothScanRequest).filter(
        BluetoothScanRequest.node_id == node_id,
        BluetoothScanRequest.status == "completed",
    ).order_by(BluetoothScanRequest.completed_at.desc()).first()

    if not recent_scan or not recent_scan.results_json:
        return BluetoothStatusResponse()

    raw_devices: list[dict] = json.loads(recent_scan.results_json)
    connected = [
        BluetoothDeviceItem(
            name=d.get("name", "Unknown"),
            mac_address=d.get("mac_address", ""),
            device_type=d.get("device_type", "unknown"),
            paired=True,
            connected=True,
        )
        for d in raw_devices if d.get("connected")
    ]
    paired = [
        BluetoothDeviceItem(
            name=d.get("name", "Unknown"),
            mac_address=d.get("mac_address", ""),
            device_type=d.get("device_type", "unknown"),
            paired=True,
            connected=False,
        )
        for d in raw_devices if d.get("paired") and not d.get("connected")
    ]

    return BluetoothStatusResponse(connected=connected, paired=paired)


# =============================================================================
# Internal Helpers
# =============================================================================


def _publish_bluetooth_mqtt(node_id: str, action: str, payload_data: dict) -> None:
    """Publish an MQTT message to the node's bluetooth topic."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, node %s cannot receive bluetooth request", node_id)
        return

    topic = f"jarvis/nodes/{node_id}/{action}"
    payload = json.dumps(payload_data)

    try:
        client.publish(topic, payload)
        logger.debug("Published MQTT %s: %s", topic, payload)
    except Exception as e:
        logger.error("Failed to publish MQTT bluetooth message: %s", e)


def _send_bluetooth_deep_link_push(household_id: str, user_id: int, node_id: str) -> None:
    """Send a push notification with deep link to the Hardware/Bluetooth screen."""
    from app.services.inbox_notification_service import send_deep_link_push_sync

    send_deep_link_push_sync(
        household_id=household_id,
        user_id=user_id,
        title="Bluetooth scan ready",
        body="Tap to see nearby devices",
        data={
            "type": "bluetooth_scan",
            "node_id": node_id,
        },
    )
