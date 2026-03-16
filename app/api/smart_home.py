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

from app.deps import get_db
from app.models import ConfigPush, Device, Node, Room
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


class DeviceImportRequest(BaseModel):
    devices: list[DeviceImportItem]


class DeviceUpdate(BaseModel):
    name: str | None = None
    room_id: str | None = None
    is_active: bool | None = None


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
    ha_device_id: str | None = None
    is_controllable: bool
    is_active: bool
    room_name: str | None = None
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
            ha_device_id=dev.ha_device_id,
            is_controllable=dev.is_controllable,
            is_active=dev.is_active,
            room_name=room_name,
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
