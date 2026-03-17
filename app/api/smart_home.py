"""Smart home API: rooms, devices, and encrypted config push to nodes.

Room/Device endpoints use JWT auth (same as provisioning).
Config push uses the phone→CC→MQTT→node encrypted relay pattern.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.deps import get_db, verify_api_key
from app.models import ConfigPush, Device, DeviceScanRequest, Node, Room
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext

router = APIRouter()
logger = logging.getLogger("uvicorn")


# =============================================================================
# Request/Response Models
# =============================================================================


class RoomCreate(BaseModel):
    name: str
    icon: str | None = None
    ha_area_id: str | None = None


class RoomUpdate(BaseModel):
    name: str | None = None
    icon: str | None = None


class RoomResponse(BaseModel):
    id: str
    household_id: str
    name: str
    normalized_name: str
    icon: str | None = None
    ha_area_id: str | None = None
    device_count: int = 0
    node_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DeviceImportItem(BaseModel):
    entity_id: str
    name: str
    domain: str
    room_id: str | None = None
    device_class: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    ha_device_id: str | None = None
    source: str = "home_assistant"
    protocol: str | None = None      # e.g., "lifx", "kasa", "tuya"
    local_ip: str | None = None      # LAN address
    mac_address: str | None = None   # MAC for stable identity
    cloud_id: str | None = None      # Cloud-only device ID (Govee, Nest, Schlage)


class DeviceImportRequest(BaseModel):
    devices: list[DeviceImportItem]


class DeviceUpdate(BaseModel):
    name: str | None = None
    room_id: str | None = None
    is_active: bool | None = None


class JarvisButtonResponse(BaseModel):
    button_text: str
    button_action: str
    button_type: str
    button_icon: str | None = None


class DeviceResponse(BaseModel):
    id: str
    household_id: str
    room_id: str | None = None
    entity_id: str
    name: str
    domain: str
    device_class: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    source: str
    protocol: str | None = None
    local_ip: str | None = None
    mac_address: str | None = None
    cloud_id: str | None = None
    ha_device_id: str | None = None
    is_controllable: bool
    is_active: bool
    room_name: str | None = None
    supported_actions: list[JarvisButtonResponse] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DeviceRoomAssignment(BaseModel):
    device_id: str
    room_id: str | None  # null to unassign


class DeviceRoomAssignRequest(BaseModel):
    assignments: list[DeviceRoomAssignment]


class ConfigPushRequest(BaseModel):
    """Encrypted config blob from mobile to relay to a node."""
    config_type: str  # e.g. "home_assistant"
    ciphertext: str   # base64url
    nonce: str        # base64url
    tag: str          # base64url


class ConfigPushResponse(BaseModel):
    id: str
    node_id: str
    config_type: str
    status: str
    created_at: datetime


# Non-controllable HA domains (sensors/read-only)
_SENSOR_DOMAINS = {"sensor", "binary_sensor", "weather", "sun", "zone", "person", "device_tracker"}

# Protocol → default actions mapping.
# Mirrors DeviceProtocol.supported_actions from each node adapter.
# CC uses this to return actions without needing access to node adapter code.
_PROTOCOL_ACTIONS: dict[str, list[dict[str, str]]] = {
    "lifx": [
        {"button_text": "Turn On", "button_action": "turn_on", "button_type": "primary", "button_icon": "lightbulb-on"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "lightbulb-off"},
        {"button_text": "Toggle", "button_action": "toggle", "button_type": "secondary", "button_icon": "lightbulb-outline"},
    ],
    "kasa": [
        {"button_text": "Turn On", "button_action": "turn_on", "button_type": "primary", "button_icon": "power"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "power-off"},
        {"button_text": "Toggle", "button_action": "toggle", "button_type": "secondary", "button_icon": "toggle-switch"},
    ],
    "govee": [
        {"button_text": "Turn On", "button_action": "turn_on", "button_type": "primary", "button_icon": "power"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "power-off"},
    ],
    "apple": [
        {"button_text": "Play", "button_action": "play", "button_type": "primary", "button_icon": "play"},
        {"button_text": "Pause", "button_action": "pause", "button_type": "secondary", "button_icon": "pause"},
        {"button_text": "Power On", "button_action": "turn_on", "button_type": "primary", "button_icon": "power"},
        {"button_text": "Power Off", "button_action": "turn_off", "button_type": "destructive", "button_icon": "power-off"},
        {"button_text": "Vol Up", "button_action": "volume_up", "button_type": "secondary", "button_icon": "volume-plus"},
        {"button_text": "Vol Down", "button_action": "volume_down", "button_type": "secondary", "button_icon": "volume-minus"},
    ],
}

# Domain-based fallback actions for HA devices without a specific protocol.
_DOMAIN_ACTIONS: dict[str, list[dict[str, str]]] = {
    "light": [
        {"button_text": "Turn On", "button_action": "turn_on", "button_type": "primary", "button_icon": "lightbulb-on"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "lightbulb-off"},
    ],
    "switch": [
        {"button_text": "Turn On", "button_action": "turn_on", "button_type": "primary", "button_icon": "power"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "power-off"},
    ],
    "lock": [
        {"button_text": "Lock", "button_action": "lock", "button_type": "primary", "button_icon": "lock"},
        {"button_text": "Unlock", "button_action": "unlock", "button_type": "destructive", "button_icon": "lock-open"},
    ],
    "climate": [
        {"button_text": "Turn On", "button_action": "turn_on", "button_type": "primary", "button_icon": "thermostat"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "power-off"},
    ],
    "fan": [
        {"button_text": "Turn On", "button_action": "turn_on", "button_type": "primary", "button_icon": "fan"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "fan-off"},
    ],
    "cover": [
        {"button_text": "Open", "button_action": "open_cover", "button_type": "primary", "button_icon": "blinds-open"},
        {"button_text": "Close", "button_action": "close_cover", "button_type": "secondary", "button_icon": "blinds"},
    ],
    "media_player": [
        {"button_text": "Play", "button_action": "media_play", "button_type": "primary", "button_icon": "play"},
        {"button_text": "Pause", "button_action": "media_pause", "button_type": "secondary", "button_icon": "pause"},
    ],
}


def _get_device_actions(dev: "Device") -> list[JarvisButtonResponse] | None:
    """Resolve supported actions for a device based on protocol or domain."""
    if not dev.is_controllable:
        return None

    # Protocol-specific actions take priority
    if dev.protocol and dev.protocol in _PROTOCOL_ACTIONS:
        return [JarvisButtonResponse(**a) for a in _PROTOCOL_ACTIONS[dev.protocol]]

    # Fall back to domain-based actions
    if dev.domain in _DOMAIN_ACTIONS:
        return [JarvisButtonResponse(**a) for a in _DOMAIN_ACTIONS[dev.domain]]

    # Generic fallback for controllable devices
    return [
        JarvisButtonResponse(button_text="Turn On", button_action="turn_on", button_type="primary", button_icon="power"),
        JarvisButtonResponse(button_text="Turn Off", button_action="turn_off", button_type="secondary", button_icon="power-off"),
    ]


# =============================================================================
# Room Endpoints
# =============================================================================


@router.get("/households/{household_id}/rooms", response_model=list[RoomResponse])
def list_rooms(
    household_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> list[RoomResponse]:
    rooms = db.query(Room).filter(Room.household_id == household_id).all()
    result = []
    for room in rooms:
        device_count = db.query(Device).filter(
            Device.room_id == room.id, Device.is_active.is_(True),
        ).count()
        node_count = db.query(Node).filter(Node.room_id == room.id).count()
        resp = RoomResponse(
            id=room.id,
            household_id=room.household_id,
            name=room.name,
            normalized_name=room.normalized_name,
            icon=room.icon,
            ha_area_id=room.ha_area_id,
            device_count=device_count,
            node_count=node_count,
            created_at=room.created_at,
            updated_at=room.updated_at,
        )
        result.append(resp)
    return result


@router.post("/households/{household_id}/rooms", response_model=RoomResponse, status_code=201)
def create_room(
    household_id: str,
    body: RoomCreate,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> RoomResponse:
    normalized = body.name.strip().lower()
    existing = db.query(Room).filter(
        Room.household_id == household_id,
        Room.normalized_name == normalized,
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Room '{body.name}' already exists")

    room = Room(
        id=str(uuid4()),
        household_id=household_id,
        name=body.name.strip(),
        normalized_name=normalized,
        icon=body.icon,
        ha_area_id=body.ha_area_id,
    )
    db.add(room)
    db.commit()
    db.refresh(room)
    logger.info("Room created: %s in household %s", room.name, household_id)

    return RoomResponse(
        id=room.id,
        household_id=room.household_id,
        name=room.name,
        normalized_name=room.normalized_name,
        icon=room.icon,
        ha_area_id=room.ha_area_id,
        device_count=0,
        node_count=0,
        created_at=room.created_at,
        updated_at=room.updated_at,
    )


@router.patch("/households/{household_id}/rooms/{room_id}", response_model=RoomResponse)
def update_room(
    household_id: str,
    room_id: str,
    body: RoomUpdate,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> RoomResponse:
    room = db.query(Room).filter(Room.id == room_id, Room.household_id == household_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    if body.name is not None:
        normalized = body.name.strip().lower()
        conflict = db.query(Room).filter(
            Room.household_id == household_id,
            Room.normalized_name == normalized,
            Room.id != room_id,
        ).first()
        if conflict:
            raise HTTPException(status_code=409, detail=f"Room '{body.name}' already exists")
        room.name = body.name.strip()
        room.normalized_name = normalized
    if body.icon is not None:
        room.icon = body.icon

    db.commit()
    db.refresh(room)

    device_count = db.query(Device).filter(
        Device.room_id == room.id, Device.is_active.is_(True),
    ).count()
    node_count = db.query(Node).filter(Node.room_id == room.id).count()

    return RoomResponse(
        id=room.id,
        household_id=room.household_id,
        name=room.name,
        normalized_name=room.normalized_name,
        icon=room.icon,
        ha_area_id=room.ha_area_id,
        device_count=device_count,
        node_count=node_count,
        created_at=room.created_at,
        updated_at=room.updated_at,
    )


@router.delete("/households/{household_id}/rooms/{room_id}", status_code=204)
def delete_room(
    household_id: str,
    room_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> None:
    room = db.query(Room).filter(Room.id == room_id, Room.household_id == household_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    db.delete(room)
    db.commit()
    logger.info("Room deleted: %s", room.name)


# =============================================================================
# Device Endpoints
# =============================================================================


@router.get("/households/{household_id}/devices", response_model=list[DeviceResponse])
def list_devices(
    household_id: str,
    room_id: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> list[DeviceResponse]:
    query = db.query(Device).filter(Device.household_id == household_id)
    if room_id:
        query = query.filter(Device.room_id == room_id)
    if domain:
        query = query.filter(Device.domain == domain)
    if source:
        query = query.filter(Device.source == source)

    devices = query.all()
    result = []
    for dev in devices:
        room_name = dev.room.name if dev.room else None
        resp = DeviceResponse(
            id=dev.id,
            household_id=dev.household_id,
            room_id=dev.room_id,
            entity_id=dev.entity_id,
            name=dev.name,
            domain=dev.domain,
            device_class=dev.device_class,
            manufacturer=dev.manufacturer,
            model=dev.model,
            source=dev.source,
            protocol=dev.protocol,
            local_ip=dev.local_ip,
            mac_address=dev.mac_address,
            cloud_id=dev.cloud_id,
            ha_device_id=dev.ha_device_id,
            is_controllable=dev.is_controllable,
            is_active=dev.is_active,
            room_name=room_name,
            supported_actions=_get_device_actions(dev),
            created_at=dev.created_at,
            updated_at=dev.updated_at,
        )
        result.append(resp)
    return result


@router.post("/households/{household_id}/devices/import", status_code=201)
def import_devices(
    household_id: str,
    body: DeviceImportRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    created = 0
    updated = 0
    for item in body.devices:
        is_controllable = item.domain not in _SENSOR_DOMAINS
        existing = db.query(Device).filter(
            Device.household_id == household_id,
            Device.entity_id == item.entity_id,
        ).first()

        if existing:
            existing.name = item.name
            existing.domain = item.domain
            existing.device_class = item.device_class
            existing.manufacturer = item.manufacturer
            existing.model = item.model
            existing.ha_device_id = item.ha_device_id
            existing.source = item.source
            existing.protocol = item.protocol
            existing.local_ip = item.local_ip
            existing.mac_address = item.mac_address
            existing.cloud_id = item.cloud_id
            existing.is_controllable = is_controllable
            if item.room_id:
                existing.room_id = item.room_id
            existing.is_active = True
            updated += 1
        else:
            dev = Device(
                id=str(uuid4()),
                household_id=household_id,
                room_id=item.room_id,
                entity_id=item.entity_id,
                name=item.name,
                domain=item.domain,
                device_class=item.device_class,
                manufacturer=item.manufacturer,
                model=item.model,
                source=item.source,
                protocol=item.protocol,
                local_ip=item.local_ip,
                mac_address=item.mac_address,
                cloud_id=item.cloud_id,
                ha_device_id=item.ha_device_id,
                is_controllable=is_controllable,
            )
            db.add(dev)
            created += 1

    db.commit()
    logger.info("Device import: %d created, %d updated in household %s", created, updated, household_id)
    return {"created": created, "updated": updated}


@router.patch("/households/{household_id}/devices/{device_id}", response_model=DeviceResponse)
def update_device(
    household_id: str,
    device_id: str,
    body: DeviceUpdate,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> DeviceResponse:
    dev = db.query(Device).filter(Device.id == device_id, Device.household_id == household_id).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Device not found")

    data = body.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(dev, key, value)

    db.commit()
    db.refresh(dev)

    room_name = dev.room.name if dev.room else None
    return DeviceResponse(
        id=dev.id,
        household_id=dev.household_id,
        room_id=dev.room_id,
        entity_id=dev.entity_id,
        name=dev.name,
        domain=dev.domain,
        device_class=dev.device_class,
        manufacturer=dev.manufacturer,
        model=dev.model,
        source=dev.source,
        protocol=dev.protocol,
        local_ip=dev.local_ip,
        mac_address=dev.mac_address,
        cloud_id=dev.cloud_id,
        ha_device_id=dev.ha_device_id,
        is_controllable=dev.is_controllable,
        is_active=dev.is_active,
        room_name=room_name,
        created_at=dev.created_at,
        updated_at=dev.updated_at,
    )


@router.delete("/households/{household_id}/devices/{device_id}", status_code=204)
def delete_device(
    household_id: str,
    device_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> None:
    dev = db.query(Device).filter(Device.id == device_id, Device.household_id == household_id).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Device not found")
    db.delete(dev)
    db.commit()


@router.post("/households/{household_id}/devices/assign-rooms")
def assign_device_rooms(
    household_id: str,
    body: DeviceRoomAssignRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    updated = 0
    for assignment in body.assignments:
        dev = db.query(Device).filter(
            Device.id == assignment.device_id,
            Device.household_id == household_id,
        ).first()
        if dev:
            dev.room_id = assignment.room_id
            updated += 1
    db.commit()
    return {"updated": updated}


# =============================================================================
# Device Control (synchronous: mobile -> CC -> device API -> mobile)
# =============================================================================


class DeviceControlRequest(BaseModel):
    action: str  # e.g. "turn_on", "turn_off", "toggle"
    data: dict | None = None  # Optional action-specific data


class DeviceControlResponse(BaseModel):
    success: bool
    entity_id: str
    action: str
    error: str | None = None


import os as _os
import tempfile as _tempfile

# File-based result passing — avoids in-memory dict issues with reload workers.
_RESULT_DIR = _os.path.join(_tempfile.gettempdir(), "jarvis-device-control")
_os.makedirs(_RESULT_DIR, exist_ok=True)


@router.post(
    "/households/{household_id}/devices/{device_id}/control",
    response_model=DeviceControlResponse,
)
def control_device(
    household_id: str,
    device_id: str,
    body: DeviceControlRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> DeviceControlResponse:
    """Control a device via a node. Sends MQTT command, waits for HTTP callback.

    Flow: CC publishes MQTT → node executes → node POSTs result back to CC.
    """
    import time

    dev = db.query(Device).filter(
        Device.id == device_id, Device.household_id == household_id,
    ).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Device not found")

    if not dev.is_controllable:
        raise HTTPException(status_code=400, detail="Device is not controllable")

    # Find a node for this household
    nodes = db.query(Node).filter(Node.household_id == household_id).all()
    node = nodes[-1] if nodes else None
    if not node:
        raise HTTPException(status_code=400, detail="No node available in household")

    from app.node_settings import get_mqtt_client
    from app.services.node_command_service import get_node_command_service

    mqtt = get_mqtt_client()
    if mqtt is None:
        raise HTTPException(status_code=503, detail="MQTT not available")

    request_id = str(uuid4())
    result_file = _os.path.join(_RESULT_DIR, f"{request_id}.json")

    service = get_node_command_service()
    details = {
        "command_name": "control_device",
        "action_name": body.action,
        "context": {
            "entity_id": dev.entity_id,
            "protocol": dev.protocol,
            "cloud_id": dev.cloud_id,
            "model": dev.model,
            "local_ip": dev.local_ip,
            "mac_address": dev.mac_address,
            "source": dev.source,
            **(body.data or {}),
        },
        "trusted": True,
        "reply_request_id": request_id,
    }
    service.publish_command_with_id(node.node_id, "action", details, request_id)

    # Poll for result file (node writes it via POST callback)
    deadline = time.time() + 10.0
    result = None
    while time.time() < deadline:
        if _os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    result = json.load(f)
                _os.unlink(result_file)
                break
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.1)

    if result is None:
        # Clean up stale file if it appeared after timeout
        try:
            _os.unlink(result_file)
        except OSError:
            pass
        return DeviceControlResponse(
            success=False, entity_id=dev.entity_id, action=body.action,
            error="Timed out waiting for node response",
        )

    return DeviceControlResponse(
        success=result.get("success", False),
        entity_id=dev.entity_id,
        action=body.action,
        error=result.get("error"),
    )


@router.post("/device-control-results/{request_id}")
def post_device_control_result(
    request_id: str,
    body: dict,
) -> dict:
    """Node POSTs device control result back to CC. Written to file for cross-process access."""
    result_file = _os.path.join(_RESULT_DIR, f"{request_id}.json")
    with open(result_file, "w") as f:
        json.dump(body, f)
    return {"status": "ok"}


# =============================================================================
# Device Scan Endpoints (user-driven: mobile -> CC -> MQTT -> node -> CC -> mobile)
# =============================================================================


class DeviceScanResponse(BaseModel):
    id: str
    status: str
    created_at: datetime


class DeviceScanResultUpload(BaseModel):
    devices: list[dict]
    error: str | None = None


class DiscoveredDeviceItem(BaseModel):
    name: str
    domain: str
    manufacturer: str | None = None
    model: str | None = None
    protocol: str | None = None
    entity_id: str
    local_ip: str | None = None
    mac_address: str | None = None
    cloud_id: str | None = None
    device_class: str | None = None
    is_controllable: bool = True
    already_registered: bool = False
    existing_device_id: str | None = None


class DeviceScanPollResponse(BaseModel):
    status: str
    request_id: str
    devices: list[DiscoveredDeviceItem] | None = None
    device_count: int | None = None
    error_message: str | None = None


@router.post(
    "/nodes/{node_id}/device-scan/request",
    response_model=DeviceScanResponse,
    status_code=201,
)
def request_device_scan(
    node_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> DeviceScanResponse:
    """Request a device scan on a node. Mobile calls this, CC notifies node via MQTT."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    now = datetime.utcnow()
    scan_request = DeviceScanRequest(
        id=str(uuid4()),
        node_id=node_id,
        household_id=node.household_id or "",
        status="pending",
        created_at=now,
        expires_at=now + timedelta(minutes=2),
    )
    db.add(scan_request)
    db.commit()
    db.refresh(scan_request)

    logger.info("Device scan requested for node %s, request_id=%s", node_id, scan_request.id[:8])

    # Notify node via MQTT
    _publish_device_scan_mqtt(node_id, scan_request.id)

    return DeviceScanResponse(
        id=scan_request.id,
        status=scan_request.status,
        created_at=scan_request.created_at,
    )


@router.post("/nodes/{node_id}/device-scan/{request_id}/results")
def upload_device_scan_results(
    node_id: str,
    request_id: str,
    body: DeviceScanResultUpload,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Node uploads scan results after running protocol adapters."""
    scan_request = db.query(DeviceScanRequest).filter(
        DeviceScanRequest.id == request_id,
        DeviceScanRequest.node_id == node_id,
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
        "Device scan results uploaded: request=%s status=%s devices=%d",
        request_id[:8], scan_request.status, len(body.devices),
    )
    return {"status": "ok"}


@router.get(
    "/nodes/{node_id}/device-scan/{request_id}",
    response_model=DeviceScanPollResponse,
)
def poll_device_scan(
    node_id: str,
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> DeviceScanPollResponse:
    """Mobile polls for scan results. CC enriches with already_registered flags."""
    scan_request = db.query(DeviceScanRequest).filter(
        DeviceScanRequest.id == request_id,
        DeviceScanRequest.node_id == node_id,
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
        return DeviceScanPollResponse(
            status="pending",
            request_id=request_id,
        )

    if scan_request.status == "failed":
        return DeviceScanPollResponse(
            status="failed",
            request_id=request_id,
            error_message=scan_request.error_message,
        )

    # Completed — enrich with already_registered flags
    raw_devices: list[dict] = json.loads(scan_request.results_json or "[]")
    household_id = scan_request.household_id

    # Load existing devices for matching
    existing_devices = db.query(Device).filter(
        Device.household_id == household_id,
        Device.is_active.is_(True),
    ).all()

    existing_entity_ids = {d.entity_id for d in existing_devices}
    existing_cloud_ids = {d.cloud_id for d in existing_devices if d.cloud_id}
    existing_macs = {d.mac_address.lower() for d in existing_devices if d.mac_address}
    # Map for getting device IDs
    entity_to_id = {d.entity_id: d.id for d in existing_devices}
    cloud_to_id = {d.cloud_id: d.id for d in existing_devices if d.cloud_id}
    mac_to_id = {d.mac_address.lower(): d.id for d in existing_devices if d.mac_address}

    enriched: list[DiscoveredDeviceItem] = []
    for dev in raw_devices:
        entity_id = dev.get("entity_id", "")
        cloud_id = dev.get("cloud_id")
        mac = dev.get("mac_address", "")

        already_registered = False
        existing_device_id = None

        if entity_id in existing_entity_ids:
            already_registered = True
            existing_device_id = entity_to_id.get(entity_id)
        elif cloud_id and cloud_id in existing_cloud_ids:
            already_registered = True
            existing_device_id = cloud_to_id.get(cloud_id)
        elif mac and mac.lower() in existing_macs:
            already_registered = True
            existing_device_id = mac_to_id.get(mac.lower())

        enriched.append(DiscoveredDeviceItem(
            name=dev.get("name", "Unknown"),
            domain=dev.get("domain", "unknown"),
            manufacturer=dev.get("manufacturer"),
            model=dev.get("model"),
            protocol=dev.get("protocol"),
            entity_id=entity_id,
            local_ip=dev.get("local_ip"),
            mac_address=dev.get("mac_address"),
            cloud_id=cloud_id,
            device_class=dev.get("device_class"),
            is_controllable=dev.get("is_controllable", True),
            already_registered=already_registered,
            existing_device_id=existing_device_id,
        ))

    return DeviceScanPollResponse(
        status="completed",
        request_id=request_id,
        devices=enriched,
        device_count=len(enriched),
    )


def _publish_device_scan_mqtt(node_id: str, request_id: str) -> None:
    """Publish MQTT message to tell node to run a device scan."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, node %s cannot receive scan request", node_id)
        return

    topic = f"jarvis/nodes/{node_id}/device-scan"
    payload = json.dumps({"request_id": request_id})

    try:
        client.publish(topic, payload)
        logger.info("Published device scan request to %s", topic)
    except Exception as e:
        logger.error("Failed to publish device scan MQTT: %s", e)


# =============================================================================
# Config Push Endpoints (encrypted relay: phone -> CC -> MQTT -> node)
# =============================================================================


@router.post(
    "/nodes/{node_id}/config/push",
    response_model=ConfigPushResponse,
    status_code=201,
)
def create_config_push(
    node_id: str,
    body: ConfigPushRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> ConfigPushResponse:
    """Store encrypted config and notify node via MQTT to pick it up."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    # Auth pushes (config_type starts with "auth:") get a 5-minute TTL
    expires_at = None
    if body.config_type.startswith("auth:"):
        expires_at = datetime.utcnow() + timedelta(minutes=5)

    push = ConfigPush(
        id=str(uuid4()),
        node_id=node_id,
        config_type=body.config_type,
        ciphertext=body.ciphertext,
        nonce=body.nonce,
        tag=body.tag,
        expires_at=expires_at,
    )
    db.add(push)
    db.commit()
    db.refresh(push)

    logger.info("Config push created: %s type=%s for node %s", push.id[:8], body.config_type, node_id)

    # Notify node via MQTT
    _publish_config_push_mqtt(node_id, push.id, body.config_type)

    return ConfigPushResponse(
        id=push.id,
        node_id=node_id,
        config_type=body.config_type,
        status=push.status,
        created_at=push.created_at,
    )


@router.get("/nodes/{node_id}/config/pending")
def get_pending_config(
    node_id: str,
    db: Session = Depends(get_db),
) -> list[dict]:
    """Get pending config pushes for a node (called by node after MQTT notification).

    Filters out expired rows and deletes them. Returns only pending, non-expired pushes.
    No auth required beyond node API key (handled by verify_api_key at caller's discretion
    or by the node knowing its own ID from the MQTT topic).
    """
    now = datetime.utcnow()

    # Delete expired pushes
    db.query(ConfigPush).filter(
        ConfigPush.node_id == node_id,
        ConfigPush.expires_at.isnot(None),
        ConfigPush.expires_at < now,
    ).delete(synchronize_session="fetch")

    # Fetch pending, non-expired pushes
    from sqlalchemy import or_
    pushes = db.query(ConfigPush).filter(
        ConfigPush.node_id == node_id,
        ConfigPush.status == "pending",
        or_(ConfigPush.expires_at.is_(None), ConfigPush.expires_at >= now),
    ).all()

    db.commit()

    return [
        {
            "id": p.id,
            "config_type": p.config_type,
            "ciphertext": p.ciphertext,
            "nonce": p.nonce,
            "tag": p.tag,
            "created_at": p.created_at.isoformat(),
        }
        for p in pushes
    ]


@router.post("/nodes/{node_id}/config/{push_id}/ack")
def ack_config_push(
    node_id: str,
    push_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Node acknowledges it has received and processed the config.

    Auth pushes (config_type starts with "auth:") are deleted on ack
    to avoid keeping sensitive token data in the database.
    """
    push = db.query(ConfigPush).filter(
        ConfigPush.id == push_id,
        ConfigPush.node_id == node_id,
    ).first()
    if not push:
        raise HTTPException(status_code=404, detail="Config push not found")
    if push.status == "consumed":
        return {"status": "already_consumed"}

    # Auth pushes: delete entirely to avoid keeping tokens in DB
    if push.config_type.startswith("auth:"):
        db.delete(push)
        db.commit()
        logger.info("Auth config push deleted after ack: %s by node %s", push_id[:8], node_id)
        return {"status": "consumed_and_deleted"}

    push.status = "consumed"
    push.consumed_at = datetime.utcnow()
    db.commit()
    logger.info("Config push consumed: %s by node %s", push_id[:8], node_id)
    return {"status": "consumed"}


def _publish_config_push_mqtt(node_id: str, push_id: str, config_type: str) -> None:
    """Publish MQTT message to notify node about pending config."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, node %s must poll for config", node_id)
        return

    topic = f"jarvis/nodes/{node_id}/config/push"
    payload = json.dumps({
        "push_id": push_id,
        "config_type": config_type,
        "node_id": node_id,
    })

    try:
        client.publish(topic, payload)
        logger.info("Published config push notification to %s", topic)
    except Exception as e:
        logger.error("Failed to publish config push MQTT: %s", e)
