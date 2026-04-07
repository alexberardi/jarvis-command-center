"""Control Device Tool — controls smart home devices via voice/chat.

Sends device control commands (turn on/off, play, pause, etc.) through
the same MQTT flow as the mobile app's device detail screen.
"""

import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from app.core.conversation_cache import conversation_cache
from app.core.interfaces.iserver_tool import IServerTool
from app.db import get_session_local
from app.models import Device, Node

logger = logging.getLogger("uvicorn")

_RESULT_DIR = os.path.join(tempfile.gettempdir(), "jarvis-device-control")


class ControlDeviceTool(IServerTool):
    """Server tool for controlling smart home devices from chat/voice.

    DISABLED — control_device is now a built-in client command on the node
    (commands/control_device_command.py). This server tool is kept for
    reference but not registered.
    """

    @property
    def enabled(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "control_device"

    @property
    def description(self) -> str:
        return (
            "Control a smart home device: turn on/off, play, pause, volume, etc. "
            "MUST be used for any request to turn on, turn off, play, pause, or "
            "control a physical device (TV, light, speaker, lock, thermostat). "
            "Common actions: turn_on, turn_off, play, pause, volume_up, volume_down, "
            "next, previous, lock, unlock."
        )

    @property
    def included_system_prompt_text(self) -> str:
        return (
            "DEVICE CONTROL: When the user asks to turn on/off, play, pause, or "
            "control ANY physical device (TV, light, speaker, lock, thermostat), "
            "ALWAYS use the control_device tool. Do NOT use 'routine' for device control."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "device_name": {
                    "type": "string",
                    "description": "Name of the device to control (e.g. 'Apple TV', 'living room light').",
                },
                "action": {
                    "type": "string",
                    "description": "Action to perform (e.g. turn_on, turn_off, play, pause, volume_up, volume_down, next, previous, lock, unlock).",
                },
            },
            "required": ["device_name", "action"],
        }

    def to_openai_format(self) -> Dict[str, Any]:
        base = super().to_openai_format()
        base["keywords"] = [
            "turn on", "turn off", "power on", "power off",
            "play", "pause", "stop", "volume", "next", "previous",
            "lock", "unlock", "device", "tv", "light", "speaker",
        ]
        base["examples"] = [
            {"voice_command": "Turn on the Apple TV", "expected_parameters": {"device_name": "Apple TV", "action": "turn_on"}, "is_primary": True},
            {"voice_command": "Turn off the office light", "expected_parameters": {"device_name": "office light", "action": "turn_off"}, "is_primary": True},
            {"voice_command": "Pause the TV", "expected_parameters": {"device_name": "TV", "action": "pause"}, "is_primary": False},
            {"voice_command": "Turn up the volume on the speaker", "expected_parameters": {"device_name": "speaker", "action": "volume_up"}, "is_primary": False},
        ]
        return base

    def execute(
        self,
        device_name: str,
        action: str,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        node_context = conversation_cache.get_node_context(conversation_id) if conversation_id else {}
        household_id: str = node_context.get("household_id", "")

        if not household_id:
            return {"error": "No household context available."}

        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            # Find device by name (case-insensitive fuzzy match)
            devices = db.query(Device).filter(
                Device.household_id == household_id,
                Device.is_active == True,  # noqa: E712
            ).all()

            dev = self._match_device(devices, device_name)
            if not dev:
                available = [d.name for d in devices] if devices else []
                return {"error": f"Device '{device_name}' not found.", "available_devices": available}

            if not dev.is_controllable:
                return {"error": f"Device '{dev.name}' is not controllable."}

            # Find a node in this household
            nodes = db.query(Node).filter(Node.household_id == household_id).all()
            node = nodes[-1] if nodes else None
            if not node:
                return {"error": "No node available in this household."}

            return self._send_control(dev, node, action)
        finally:
            db.close()

    def _match_device(self, devices: list, query: str) -> Optional[Device]:
        """Match a device by name. Tries exact, then case-insensitive, then substring."""
        query_lower = query.lower().strip()

        # Exact match
        for d in devices:
            if d.name.lower() == query_lower:
                return d

        # Substring match
        for d in devices:
            if query_lower in d.name.lower() or d.name.lower() in query_lower:
                return d

        # Entity ID match
        for d in devices:
            if query_lower in d.entity_id.lower():
                return d

        return None

    def _send_control(self, dev: Device, node: Node, action: str) -> Dict[str, Any]:
        """Send control command via MQTT and wait for result."""
        from app.node_settings import get_mqtt_client
        from app.services.node_command_service import get_node_command_service

        mqtt = get_mqtt_client()
        if mqtt is None:
            return {"error": "MQTT not available."}

        request_id = str(uuid4())
        result_file = os.path.join(_RESULT_DIR, f"{request_id}.json")

        service = get_node_command_service()
        details = {
            "command_name": "control_device",
            "action_name": action,
            "context": {
                "entity_id": dev.entity_id,
                "protocol": dev.protocol,
                "cloud_id": dev.cloud_id,
                "model": dev.model,
                "local_ip": dev.local_ip,
                "mac_address": dev.mac_address,
                "source": dev.source,
            },
            "trusted": True,
            "reply_request_id": request_id,
        }

        service.publish_command_with_id(str(node.node_id), "action", details, request_id)

        # Poll for result (node POSTs back to CC)
        deadline = time.time() + 10.0
        result = None
        while time.time() < deadline:
            if os.path.exists(result_file):
                try:
                    with open(result_file) as f:
                        result = json.load(f)
                    os.unlink(result_file)
                    break
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(0.1)

        if result is None:
            try:
                os.unlink(result_file)
            except OSError:
                pass
            return {"error": f"Timed out waiting for device response."}

        if result.get("success"):
            return {"success": True, "device": dev.name, "action": action}
        else:
            return {"error": result.get("error", "Unknown device error."), "device": dev.name, "action": action}
