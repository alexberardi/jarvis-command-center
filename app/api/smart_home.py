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
from app.models import ConfigPush, Device, DeviceListRequest, DeviceScanRequest, Node, Room
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext

router = APIRouter()
logger = logging.getLogger("uvicorn")


# =============================================================================
# Smart Home Config (device manager selection + primary node)
# =============================================================================


class SmartHomeConfigResponse(BaseModel):
    device_manager: str
    primary_node_id: str
    use_external_devices: bool


class SmartHomeConfigUpdate(BaseModel):
    device_manager: str | None = None
    primary_node_id: str | None = None
    use_external_devices: bool | None = None


class NodeOption(BaseModel):
    node_id: str
    room: str | None = None
    online: bool = False
    last_seen: datetime | None = None


class SmartHomeConfigWithNodes(BaseModel):
    device_manager: str
    primary_node_id: str
    use_external_devices: bool
    nodes: list[NodeOption]


@router.get(
    "/households/{household_id}/smart-home/config",
    response_model=SmartHomeConfigWithNodes,
)
def get_smart_home_config(
    household_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> SmartHomeConfigWithNodes:
    """Get smart home configuration for a household (device manager + primary node + node list)."""
    from app.services.settings_service import get_settings_service

    settings = get_settings_service()
    device_manager: str = settings.get("smart_home.device_manager", household_id=household_id) or "jarvis_direct"
    primary_node_id: str = settings.get("smart_home.primary_node_id", household_id=household_id) or ""
    use_external_devices: bool = settings.get("smart_home.use_external_devices", household_id=household_id) or False

    # Get active nodes in this household for the dropdown
    nodes = db.query(Node).filter(Node.household_id == household_id, Node.is_active.is_(True)).all()
    node_options = [
        NodeOption(node_id=n.node_id, room=n.room, online=n.is_online(), last_seen=n.last_seen)
        for n in nodes
    ]

    return SmartHomeConfigWithNodes(
        device_manager=device_manager,
        primary_node_id=primary_node_id,
        use_external_devices=use_external_devices,
        nodes=node_options,
    )


@router.put(
    "/households/{household_id}/smart-home/config",
    response_model=SmartHomeConfigResponse,
)
def update_smart_home_config(
    household_id: str,
    body: SmartHomeConfigUpdate,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> SmartHomeConfigResponse:
    """Update smart home configuration for a household."""
    from app.services.settings_service import get_settings_service

    settings = get_settings_service()

    if body.device_manager is not None:
        settings.set("smart_home.device_manager", body.device_manager, household_id=household_id)

    if body.primary_node_id is not None:
        # Validate node exists in household (allow empty to clear)
        if body.primary_node_id:
            node = db.query(Node).filter(
                Node.node_id == body.primary_node_id,
                Node.household_id == household_id,
            ).first()
            if not node:
                raise HTTPException(status_code=400, detail="Node not in this household")
        settings.set("smart_home.primary_node_id", body.primary_node_id, household_id=household_id)

    if body.use_external_devices is not None:
        settings.set("smart_home.use_external_devices", body.use_external_devices, household_id=household_id)

        # Toggle the built-in control_device on all household nodes:
        # external=True → disable built-in (Pantry package provides it)
        # external=False → enable built-in (no external package)
        _toggle_builtin_control_device(household_id, enabled=not body.use_external_devices, db=db)

    # Return current values
    device_manager: str = settings.get("smart_home.device_manager", household_id=household_id) or "jarvis_direct"
    primary_node_id: str = settings.get("smart_home.primary_node_id", household_id=household_id) or ""
    use_external_devices: bool = settings.get("smart_home.use_external_devices", household_id=household_id) or False

    return SmartHomeConfigResponse(
        device_manager=device_manager,
        primary_node_id=primary_node_id,
        use_external_devices=use_external_devices,
    )


def _toggle_builtin_control_device(household_id: str, *, enabled: bool, db: Session) -> None:
    """Publish toggle_command to all nodes in a household via MQTT.

    Enables/disables the built-in control_device command so it doesn't
    conflict with a Pantry package's control_device.
    """
    from app.services.node_command_service import get_node_command_service

    nodes = db.query(Node).filter(Node.household_id == household_id, Node.is_active.is_(True)).all()
    if not nodes:
        return

    service = get_node_command_service()
    for node in nodes:
        service.publish_command(
            node.node_id,
            "toggle_command",
            {"command_name": "control_device", "enabled": enabled},
        )
    logger.info(
        "Toggled built-in control_device on %d nodes: enabled=%s",
        len(nodes), enabled,
    )


# =============================================================================
# Request/Response Models
# =============================================================================


class RoomCreate(BaseModel):
    name: str
    icon: str | None = None
    ha_area_id: str | None = None
    parent_room_id: str | None = None


class RoomUpdate(BaseModel):
    name: str | None = None
    icon: str | None = None
    parent_room_id: str | None = None


class RoomResponse(BaseModel):
    id: str
    household_id: str
    name: str
    normalized_name: str
    icon: str | None = None
    ha_area_id: str | None = None
    parent_room_id: str | None = None
    device_count: int = 0
    node_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RoomTreeNode(BaseModel):
    id: str
    household_id: str
    name: str
    normalized_name: str
    icon: str | None = None
    ha_area_id: str | None = None
    parent_room_id: str | None = None
    device_count: int = 0
    node_count: int = 0
    created_at: datetime
    updated_at: datetime
    children: list["RoomTreeNode"] = []

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
    "nest": [
        {"button_text": "Set Temperature", "button_action": "set_temperature", "button_type": "primary", "button_icon": "thermometer"},
        {"button_text": "Set Mode", "button_action": "set_mode", "button_type": "secondary", "button_icon": "thermostat"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "power-off"},
    ],
    "apple": [
        {"button_text": "Pair", "button_action": "pair_start", "button_type": "primary", "button_icon": "link"},
        {"button_text": "Play", "button_action": "play", "button_type": "primary", "button_icon": "play"},
        {"button_text": "Pause", "button_action": "pause", "button_type": "secondary", "button_icon": "pause"},
        {"button_text": "Power On", "button_action": "turn_on", "button_type": "primary", "button_icon": "power"},
        {"button_text": "Power Off", "button_action": "turn_off", "button_type": "destructive", "button_icon": "power-off"},
        {"button_text": "Vol Up", "button_action": "volume_up", "button_type": "secondary", "button_icon": "volume-plus"},
        {"button_text": "Vol Down", "button_action": "volume_down", "button_type": "secondary", "button_icon": "volume-minus"},
    ],
    "homeconnect": [
        {"button_text": "Auto Wash", "button_action": "start_auto", "button_type": "primary", "button_icon": "dishwasher"},
        {"button_text": "Eco 50\u00b0", "button_action": "start_eco", "button_type": "primary", "button_icon": "leaf"},
        {"button_text": "Quick 65\u00b0", "button_action": "start_quick", "button_type": "secondary", "button_icon": "lightning-bolt"},
        {"button_text": "Glass 40\u00b0", "button_action": "start_glass", "button_type": "secondary", "button_icon": "glass-fragile"},
        {"button_text": "Pre-Rinse", "button_action": "start_prerinse", "button_type": "secondary", "button_icon": "water"},
        {"button_text": "Stop", "button_action": "stop", "button_type": "destructive", "button_icon": "stop"},
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
        {"button_text": "Set Temperature", "button_action": "set_temperature", "button_type": "primary", "button_icon": "thermometer"},
        {"button_text": "Set Mode", "button_action": "set_mode", "button_type": "secondary", "button_icon": "thermostat"},
        {"button_text": "Turn Off", "button_action": "turn_off", "button_type": "secondary", "button_icon": "power-off"},
    ],
    "camera": [
        {"button_text": "Get Stream", "button_action": "get_stream", "button_type": "primary", "button_icon": "video"},
    ],
    "kettle": [
        {"button_text": "Boil", "button_action": "turn_on", "button_type": "primary", "button_icon": "kettle"},
        {"button_text": "Set Temperature", "button_action": "set_temperature", "button_type": "secondary", "button_icon": "thermometer"},
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


def _get_actions_for_raw(
    domain: str, protocol: str | None, is_controllable: bool,
) -> list[JarvisButtonResponse] | None:
    """Resolve supported actions based on protocol or domain (no DB model needed)."""
    if not is_controllable:
        return None

    # Protocol-specific actions take priority
    if protocol and protocol in _PROTOCOL_ACTIONS:
        return [JarvisButtonResponse(**a) for a in _PROTOCOL_ACTIONS[protocol]]

    # Fall back to domain-based actions
    if domain in _DOMAIN_ACTIONS:
        return [JarvisButtonResponse(**a) for a in _DOMAIN_ACTIONS[domain]]

    # Generic fallback for controllable devices
    return [
        JarvisButtonResponse(button_text="Turn On", button_action="turn_on", button_type="primary", button_icon="power"),
        JarvisButtonResponse(button_text="Turn Off", button_action="turn_off", button_type="secondary", button_icon="power-off"),
    ]


def _get_device_actions(dev: "Device") -> list[JarvisButtonResponse] | None:
    """Resolve supported actions for a DB device."""
    return _get_actions_for_raw(dev.domain, dev.protocol, dev.is_controllable)


def _pick_node_for_protocol(
    db: Session,
    household_id: str,
    protocol: str | None,
    primary_node_id: str = "",
) -> Node | None:
    """Pick the best active node for a device control command.

    Priority:
    1. Node that reports having the device's protocol installed.
    2. primary_node_id (if set and active).
    3. Any active node (fallback for old nodes with protocols=NULL).
    """
    nodes = db.query(Node).filter(
        Node.household_id == household_id,
        Node.is_active.is_(True),
    ).all()
    if not nodes:
        return None

    if protocol:
        matching: list[Node] = []
        for n in nodes:
            if n.protocols:
                try:
                    node_protocols = json.loads(n.protocols)
                    if protocol in node_protocols:
                        matching.append(n)
                except (json.JSONDecodeError, TypeError):
                    pass
        if matching:
            if primary_node_id:
                for n in matching:
                    if n.node_id == primary_node_id:
                        return n
            return max(matching, key=lambda n: n.last_seen or datetime.min)

    # Fallback: prefer primary_node_id, then any node
    if primary_node_id:
        for n in nodes:
            if n.node_id == primary_node_id:
                return n
    return nodes[-1]


# =============================================================================
# Room Helpers
# =============================================================================


def get_descendant_room_ids(db: Session, room_id: str, household_id: str) -> list[str]:
    """BFS to collect room_id + all descendant IDs within a household."""
    result: list[str] = [room_id]
    queue = [room_id]
    while queue:
        parent_id = queue.pop(0)
        children = db.query(Room.id).filter(
            Room.parent_room_id == parent_id,
            Room.household_id == household_id,
        ).all()
        for (child_id,) in children:
            result.append(child_id)
            queue.append(child_id)
    return result


def _would_create_cycle(db: Session, room_id: str, proposed_parent_id: str, household_id: str) -> bool:
    """Walk up from proposed_parent; if we reach room_id, it's a cycle."""
    current = proposed_parent_id
    while current is not None:
        if current == room_id:
            return True
        parent = db.query(Room.parent_room_id).filter(
            Room.id == current,
            Room.household_id == household_id,
        ).scalar()
        current = parent
    return False


def _build_room_response(room: Room, db: Session) -> RoomResponse:
    """Build a RoomResponse with device/node counts."""
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
        parent_room_id=room.parent_room_id,
        device_count=device_count,
        node_count=node_count,
        created_at=room.created_at,
        updated_at=room.updated_at,
    )


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
    return [_build_room_response(room, db) for room in rooms]


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

    # Validate parent exists in same household
    if body.parent_room_id:
        parent = db.query(Room).filter(
            Room.id == body.parent_room_id,
            Room.household_id == household_id,
        ).first()
        if not parent:
            raise HTTPException(status_code=400, detail="Parent room not found in this household")

    room = Room(
        id=str(uuid4()),
        household_id=household_id,
        name=body.name.strip(),
        normalized_name=normalized,
        icon=body.icon,
        ha_area_id=body.ha_area_id,
        parent_room_id=body.parent_room_id,
    )
    db.add(room)
    db.commit()
    db.refresh(room)
    logger.info("Room created: %s in household %s", room.name, household_id)

    return _build_room_response(room, db)


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

    # Handle parent_room_id update (explicit None clears the parent)
    if "parent_room_id" in (body.model_dump(exclude_unset=True)):
        new_parent = body.parent_room_id
        if new_parent is not None:
            # Validate parent exists in same household
            parent = db.query(Room).filter(
                Room.id == new_parent, Room.household_id == household_id,
            ).first()
            if not parent:
                raise HTTPException(status_code=400, detail="Parent room not found in this household")
            # Reject cycles: cannot set parent to self or any descendant
            if _would_create_cycle(db, room_id, new_parent, household_id):
                raise HTTPException(status_code=400, detail="Cannot set parent: would create a cycle")
        room.parent_room_id = new_parent

    db.commit()
    db.refresh(room)

    return _build_room_response(room, db)


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


@router.get("/households/{household_id}/rooms/tree", response_model=list[RoomTreeNode])
def get_room_tree(
    household_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> list[RoomTreeNode]:
    """Return rooms as a nested tree structure."""
    rooms = db.query(Room).filter(Room.household_id == household_id).all()

    # Build lookup and count maps
    room_map: dict[str, RoomTreeNode] = {}
    for room in rooms:
        device_count = db.query(Device).filter(
            Device.room_id == room.id, Device.is_active.is_(True),
        ).count()
        node_count = db.query(Node).filter(Node.room_id == room.id).count()
        room_map[room.id] = RoomTreeNode(
            id=room.id,
            household_id=room.household_id,
            name=room.name,
            normalized_name=room.normalized_name,
            icon=room.icon,
            ha_area_id=room.ha_area_id,
            parent_room_id=room.parent_room_id,
            device_count=device_count,
            node_count=node_count,
            created_at=room.created_at,
            updated_at=room.updated_at,
        )

    # Build tree: attach children to parents
    roots: list[RoomTreeNode] = []
    for node in room_map.values():
        if node.parent_room_id and node.parent_room_id in room_map:
            room_map[node.parent_room_id].children.append(node)
        else:
            roots.append(node)

    return roots


# =============================================================================
# Device Endpoints
# =============================================================================


@router.get("/households/{household_id}/devices", response_model=list[DeviceResponse])
def list_devices(
    household_id: str,
    room_id: Optional[str] = Query(None),
    recursive: bool = Query(False),
    domain: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> list[DeviceResponse]:
    query = db.query(Device).filter(Device.household_id == household_id)
    if room_id:
        if recursive:
            all_room_ids = get_descendant_room_ids(db, room_id, household_id)
            query = query.filter(Device.room_id.in_(all_room_ids))
        else:
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


def _broadcast_device_cache_invalidate(db: Session, household_id: str) -> None:
    """Tell every node in the household to drop its DirectDeviceService cache.

    Why: each node holds a local cache of the household's devices that was
    seeded by ``GET /api/v0/node/devices`` and only refreshed every 5 min
    by DeviceDiscoveryAgent. Without this signal, a device added or
    removed via the mobile app is invisible to ``control_device`` on a
    given node until that node's next periodic refresh — up to 5 min of
    "device not found" errors after a successful add. Best-effort: if
    MQTT is down, the next periodic refresh still picks up the change.
    """
    try:
        from app.services.node_command_service import get_node_command_service
        nodes = db.query(Node).filter(Node.household_id == household_id, Node.is_active.is_(True)).all()
        if not nodes:
            return
        service = get_node_command_service()
        for node in nodes:
            service.publish_command(str(node.node_id), "invalidate_device_cache", {})
    except Exception as e:
        logger.warning("Failed to broadcast device-cache invalidate: %s", e)


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
    if created or updated:
        _broadcast_device_cache_invalidate(db, household_id)
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
    _broadcast_device_cache_invalidate(db, household_id)

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
    _broadcast_device_cache_invalidate(db, household_id)


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
    input_required: dict | None = None


@router.get("/node/devices")
def list_devices_for_node(
    node_ctx=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> list[dict]:
    """List devices for the node's household (node API key auth).

    Used by the built-in control_device command to populate DirectDeviceService.
    """
    household_id: str = node_ctx.household_id or ""
    if not household_id:
        return []

    from sqlalchemy.orm import joinedload

    devices = db.query(Device).options(
        joinedload(Device.room),
    ).filter(
        Device.household_id == household_id,
        Device.is_active == True,  # noqa: E712
    ).all()

    return [
        {
            "entity_id": d.entity_id,
            "name": d.name,
            "domain": d.domain,
            "source": d.source,
            "protocol": d.protocol,
            "local_ip": d.local_ip or "",
            "mac_address": d.mac_address or "",
            "cloud_id": d.cloud_id or "",
            "model": d.model or "",
            "room_name": d.room.name if d.room else "",
        }
        for d in devices
    ]


@router.get("/node/devices/{entity_id:path}")
def get_device_for_node(
    entity_id: str,
    node_ctx=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Get a single device by entity_id (node API key auth).

    Used by control_device command to resolve entity_id → protocol/IP details.
    """
    household_id: str = node_ctx.household_id or ""
    if not household_id:
        raise HTTPException(status_code=400, detail="No household context")

    device = db.query(Device).filter(
        Device.household_id == household_id,
        Device.entity_id == entity_id,
        Device.is_active == True,  # noqa: E712
    ).first()

    if not device:
        raise HTTPException(status_code=404, detail=f"Device '{entity_id}' not found")

    return {
        "entity_id": device.entity_id,
        "name": device.name,
        "domain": device.domain,
        "source": device.source,
        "protocol": device.protocol,
        "local_ip": device.local_ip or "",
        "mac_address": device.mac_address or "",
        "cloud_id": device.cloud_id or "",
        "model": device.model or "",
    }


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

    # Pick a node that has the device's protocol installed
    node = _pick_node_for_protocol(db, household_id, dev.protocol)
    if not node:
        raise HTTPException(status_code=400, detail="No active node available in household")

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
            "domain": dev.domain,
            "name": dev.name,
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
    logger.info("DEVICE CONTROL: action=%s node=%s device=%s mqtt=%s", body.action, node.node_id[:12], dev.entity_id, mqtt.is_connected if mqtt else "N/A")
    service.publish_command_with_id(node.node_id, "action", details, request_id)
    logger.info("DEVICE CONTROL: published, polling for %s", request_id[:8])

    # Poll for result file (node writes it via POST callback).
    # Pairing actions need longer: pair_start does device scan + SRP handshake,
    # pair_finish sends PIN verification. Regular control actions are fast.
    timeout: float = 20.0 if body.action.startswith("pair") else 10.0
    deadline = time.time() + timeout
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
        input_required=result.get("input_required"),
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
# External Device Control (non-persisted devices via MQTT)
# =============================================================================


class ExternalDeviceControlRequest(BaseModel):
    entity_id: str
    action: str
    source: str = "direct"
    protocol: str | None = None
    cloud_id: str | None = None
    model: str | None = None
    local_ip: str | None = None
    mac_address: str | None = None
    data: dict | None = None


@router.post(
    "/households/{household_id}/devices/control-external",
    response_model=DeviceControlResponse,
)
def control_external_device(
    household_id: str,
    body: ExternalDeviceControlRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> DeviceControlResponse:
    """Control a non-persisted (external) device via a node.

    Same MQTT flow as control_device but skips the DB device lookup.
    Uses the primary_node_id from settings, falling back to any household node.
    """
    import time

    from app.services.settings_service import get_settings_service

    settings = get_settings_service()
    primary_node_id: str = settings.get("smart_home.primary_node_id", household_id=household_id) or ""

    # Pick a node that has the device's protocol installed
    node = _pick_node_for_protocol(db, household_id, body.protocol, primary_node_id)
    if not node:
        raise HTTPException(status_code=400, detail="No active node available in household")

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
            "entity_id": body.entity_id,
            "protocol": body.protocol,
            "cloud_id": body.cloud_id,
            "model": body.model,
            "local_ip": body.local_ip,
            "mac_address": body.mac_address,
            "source": body.source,
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
        try:
            _os.unlink(result_file)
        except OSError:
            pass
        return DeviceControlResponse(
            success=False, entity_id=body.entity_id, action=body.action,
            error="Timed out waiting for node response",
        )

    return DeviceControlResponse(
        success=result.get("success", False),
        entity_id=body.entity_id,
        action=body.action,
        error=result.get("error"),
        input_required=result.get("input_required"),
    )


# =============================================================================
# Device State Endpoint (mobile -> CC -> MQTT -> node -> CC -> mobile)
# =============================================================================


class DeviceStateResponse(BaseModel):
    entity_id: str
    domain: str
    state: dict | None = None
    ui_hints: dict | None = None
    error: str | None = None


@router.get(
    "/households/{household_id}/devices/{device_id}/state",
    response_model=DeviceStateResponse,
)
def get_device_state(
    household_id: str,
    device_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> DeviceStateResponse:
    """Query current device state via a node. Sends MQTT, waits for HTTP callback.

    Flow: CC publishes MQTT → node queries adapter → node POSTs result → CC returns.
    """
    import time

    dev = db.query(Device).filter(
        Device.id == device_id, Device.household_id == household_id,
    ).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Device not found")

    node = _pick_node_for_protocol(db, household_id, dev.protocol)
    if not node:
        raise HTTPException(status_code=400, detail="No active node available in household")

    from app.node_settings import get_mqtt_client
    from app.services.node_command_service import get_node_command_service

    mqtt = get_mqtt_client()
    if mqtt is None:
        raise HTTPException(status_code=503, detail="MQTT not available")

    request_id = str(uuid4())
    result_file = _os.path.join(_RESULT_DIR, f"state-{request_id}.json")

    # Prefer stored domain column, fall back to entity_id prefix
    domain = dev.domain or (dev.entity_id.split(".", 1)[0] if "." in dev.entity_id else "")

    # Publish MQTT request to node
    service = get_node_command_service()
    topic = f"jarvis/nodes/{node.node_id}/device-state"
    payload = json.dumps({
        "request_id": request_id,
        "entity_id": dev.entity_id,
        "domain": domain,
        "protocol": dev.protocol,
        "source": dev.source,
        "cloud_id": dev.cloud_id,
        "local_ip": dev.local_ip,
        "mac_address": dev.mac_address,
    })
    mqtt.publish(topic, payload)

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
        try:
            _os.unlink(result_file)
        except OSError:
            pass
        return DeviceStateResponse(
            entity_id=dev.entity_id, domain=domain,
            error="Timed out waiting for node response",
        )

    return DeviceStateResponse(
        entity_id=result.get("entity_id", dev.entity_id),
        domain=result.get("domain", domain),
        state=result.get("state"),
        ui_hints=result.get("ui_hints"),
        error=result.get("error"),
    )


@router.post("/device-state-results/{request_id}")
def post_device_state_result(
    request_id: str,
    body: dict,
) -> dict:
    """Node POSTs device state result back to CC."""
    result_file = _os.path.join(_RESULT_DIR, f"state-{request_id}.json")
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
    supported_actions: list[dict[str, str]] | None = None


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
            supported_actions=dev.get("supported_actions"),
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
# Device List Request/Poll (mobile → CC → MQTT → node → CC → mobile)
# =============================================================================


class DeviceListResponse(BaseModel):
    id: str
    status: str
    created_at: datetime


class DeviceListResultUpload(BaseModel):
    devices: list[dict]
    manager_name: str | None = None
    can_edit_devices: bool | None = None
    error: str | None = None


class DeviceListItem(BaseModel):
    name: str
    domain: str
    entity_id: str
    is_controllable: bool = True
    manufacturer: str | None = None
    model: str | None = None
    protocol: str | None = None
    local_ip: str | None = None
    mac_address: str | None = None
    cloud_id: str | None = None
    device_class: str | None = None
    source: str = "direct"
    area: str | None = None
    state: str | None = None
    already_registered: bool = False
    existing_device_id: str | None = None
    room_id: str | None = None
    room_name: str | None = None
    supported_actions: list[JarvisButtonResponse] | None = None


class DeviceListPollResponse(BaseModel):
    status: str
    request_id: str
    manager_name: str | None = None
    can_edit_devices: bool | None = None
    devices: list[DeviceListItem] | None = None
    device_count: int | None = None
    error_message: str | None = None


@router.post(
    "/nodes/{node_id}/device-list/request",
    response_model=DeviceListResponse,
    status_code=201,
)
def request_device_list(
    node_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> DeviceListResponse:
    """Request a full device list from a node.  Mobile calls this, CC notifies node via MQTT."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    # Aggregate devices from all enabled managers on the node
    manager_name = "all"

    now = datetime.utcnow()
    list_request = DeviceListRequest(
        id=str(uuid4()),
        node_id=node_id,
        household_id=node.household_id or "",
        status="pending",
        manager_name=manager_name,
        created_at=now,
        expires_at=now + timedelta(minutes=2),
    )
    db.add(list_request)
    db.commit()
    db.refresh(list_request)

    logger.info("Device list requested for node %s, request_id=%s, manager=%s", node_id, list_request.id[:8], manager_name)

    _publish_device_list_mqtt(node_id, list_request.id, manager_name)

    return DeviceListResponse(
        id=list_request.id,
        status=list_request.status,
        created_at=list_request.created_at,
    )


@router.post("/nodes/{node_id}/device-list/{request_id}/results")
def upload_device_list_results(
    node_id: str,
    request_id: str,
    body: DeviceListResultUpload,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Node uploads device list results after running the selected device manager."""
    list_request = db.query(DeviceListRequest).filter(
        DeviceListRequest.id == request_id,
        DeviceListRequest.node_id == node_id,
    ).first()
    if not list_request:
        raise HTTPException(status_code=404, detail="Device list request not found")

    now = datetime.utcnow()
    if list_request.expires_at and list_request.expires_at < now:
        list_request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Device list request expired")

    if body.error:
        list_request.status = "failed"
        list_request.error_message = body.error
    else:
        list_request.status = "completed"
        list_request.results_json = json.dumps(body.devices)
        list_request.device_count = len(body.devices)

    if body.manager_name:
        list_request.manager_name = body.manager_name
    if body.can_edit_devices is not None:
        list_request.can_edit_devices = body.can_edit_devices

    list_request.completed_at = now
    db.commit()

    logger.info(
        "Device list results uploaded: request=%s status=%s devices=%d",
        request_id[:8], list_request.status, len(body.devices),
    )
    return {"status": "ok"}


@router.get(
    "/nodes/{node_id}/device-list/{request_id}",
    response_model=DeviceListPollResponse,
)
def poll_device_list(
    node_id: str,
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> DeviceListPollResponse:
    """Mobile polls for device list results.  CC enriches with room assignments."""
    list_request = db.query(DeviceListRequest).filter(
        DeviceListRequest.id == request_id,
        DeviceListRequest.node_id == node_id,
    ).first()
    if not list_request:
        raise HTTPException(status_code=404, detail="Device list request not found")

    now = datetime.utcnow()

    # Check expiration
    if list_request.status == "pending" and list_request.expires_at and list_request.expires_at < now:
        list_request.status = "expired"
        db.commit()

    if list_request.status == "expired":
        raise HTTPException(status_code=410, detail="Device list request expired")

    if list_request.status == "pending":
        return DeviceListPollResponse(
            status="pending",
            request_id=request_id,
            manager_name=list_request.manager_name,
        )

    if list_request.status == "failed":
        return DeviceListPollResponse(
            status="failed",
            request_id=request_id,
            manager_name=list_request.manager_name,
            error_message=list_request.error_message,
        )

    # Completed — enrich with room assignments and already_registered flags
    raw_devices: list[dict] = json.loads(list_request.results_json or "[]")
    household_id: str = list_request.household_id

    # Load existing devices for matching
    existing_devices = db.query(Device).filter(
        Device.household_id == household_id,
        Device.is_active.is_(True),
    ).all()

    existing_entity_ids = {d.entity_id: d for d in existing_devices}
    existing_cloud_ids = {d.cloud_id: d for d in existing_devices if d.cloud_id}
    existing_macs = {d.mac_address.lower(): d for d in existing_devices if d.mac_address}

    # Load rooms for enrichment
    rooms = db.query(Room).filter(Room.household_id == household_id).all()
    room_by_id = {r.id: r for r in rooms}
    # Build device_id → room mapping
    device_room_map: dict[str, Room] = {}
    for d in existing_devices:
        if d.room_id and d.room_id in room_by_id:
            device_room_map[d.id] = room_by_id[d.room_id]

    enriched: list[DeviceListItem] = []
    for dev in raw_devices:
        entity_id: str = dev.get("entity_id", "")
        cloud_id: str | None = dev.get("cloud_id")
        mac: str = dev.get("mac_address") or ""

        already_registered = False
        existing_device_id: str | None = None
        room_id: str | None = None
        room_name: str | None = None

        # Match against existing devices
        matched_device: Device | None = None
        if entity_id in existing_entity_ids:
            matched_device = existing_entity_ids[entity_id]
        elif cloud_id and cloud_id in existing_cloud_ids:
            matched_device = existing_cloud_ids[cloud_id]
        elif mac and mac.lower() in existing_macs:
            matched_device = existing_macs[mac.lower()]

        if matched_device:
            already_registered = True
            existing_device_id = matched_device.id
            if matched_device.id in device_room_map:
                room_obj = device_room_map[matched_device.id]
                room_id = room_obj.id
                room_name = room_obj.name

        dev_domain = dev.get("domain", "unknown")
        dev_protocol = dev.get("protocol")
        dev_controllable = dev.get("is_controllable", True)

        enriched.append(DeviceListItem(
            name=dev.get("name", "Unknown"),
            domain=dev_domain,
            entity_id=entity_id,
            is_controllable=dev_controllable,
            manufacturer=dev.get("manufacturer"),
            model=dev.get("model"),
            protocol=dev_protocol,
            local_ip=dev.get("local_ip"),
            mac_address=dev.get("mac_address"),
            cloud_id=cloud_id,
            device_class=dev.get("device_class"),
            source=dev.get("source", "direct"),
            area=dev.get("area"),
            state=dev.get("state"),
            already_registered=already_registered,
            existing_device_id=existing_device_id,
            room_id=room_id,
            room_name=room_name,
            supported_actions=_get_actions_for_raw(dev_domain, dev_protocol, dev_controllable),
        ))

    return DeviceListPollResponse(
        status="completed",
        request_id=request_id,
        manager_name=list_request.manager_name,
        can_edit_devices=list_request.can_edit_devices,
        devices=enriched,
        device_count=len(enriched),
    )


def _publish_device_list_mqtt(node_id: str, request_id: str, manager_name: str) -> None:
    """Publish MQTT message to tell node to collect a device list."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, node %s cannot receive device list request", node_id)
        return

    topic = f"jarvis/nodes/{node_id}/device-list"
    payload = json.dumps({"request_id": request_id, "manager_name": manager_name})

    try:
        client.publish(topic, payload)
        logger.info("Published device list request to %s", topic)
    except Exception as e:
        logger.error("Failed to publish device list MQTT: %s", e)


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
