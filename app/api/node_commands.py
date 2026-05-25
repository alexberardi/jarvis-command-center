"""
API endpoints for sending commands to nodes and verifying command requests.

POST /nodes/{node_id}/commands — admin sends a command to a node via MQTT
POST /nodes/{node_id}/actions — mobile forwards a user action (e.g. Send/Cancel) to a node via MQTT
POST /commands/{request_id}/verify — node verifies a command is legitimate
"""
import json
import logging
import os
import tempfile
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import AuthenticatedUser, verify_admin_key, verify_api_key, verify_user_jwt
from app.services.node_command_service import get_node_command_service

logger = logging.getLogger("uvicorn")

router = APIRouter()

_RESULT_DIR = os.path.join(tempfile.gettempdir(), "jarvis-device-control")


class SendCommandRequest(BaseModel):
    command: str
    details: dict | None = None


class SendCommandResponse(BaseModel):
    status: str
    request_id: str


class ActionRequest(BaseModel):
    """Request to forward a user action (button tap) to a node."""
    command_name: str
    action_name: str
    context: dict | None = None


class ActionResponse(BaseModel):
    status: str
    request_id: str
    success: bool | None = None
    error: str | None = None


class VerifyResponse(BaseModel):
    valid: bool


@router.post(
    "/nodes/{node_id}/actions",
    response_model=ActionResponse,
)
def send_node_action(
    node_id: str,
    body: ActionRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> ActionResponse:
    """Forward a user action (e.g. Send/Cancel button tap) to a node via MQTT.

    Authenticated via JWT (mobile app). Publishes an MQTT message with
    command="action" so the node's MQTT listener can dispatch to the
    correct command's handle_action() method.

    Waits up to 10s for the node to POST its result back, then returns
    success/failure to the caller.
    """
    service = get_node_command_service()
    request_id = service.publish_command(node_id, "action", {
        "command_name": body.command_name,
        "action_name": body.action_name,
        "context": body.context or {},
        "trusted": True,
        "user_id": user.user_id,
    })

    # Wait for node to POST result back via /device-control-results/{request_id}
    result_file = os.path.join(_RESULT_DIR, f"{request_id}.json")
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
        return ActionResponse(
            status="timeout", request_id=request_id,
            success=None, error="Node did not respond in time",
        )

    return ActionResponse(
        status="completed",
        request_id=request_id,
        success=result.get("success", True),
        error=result.get("error"),
    )


@router.post(
    "/nodes/{node_id}/commands",
    response_model=SendCommandResponse,
    dependencies=[Depends(verify_admin_key)],
)
def send_node_command(node_id: str, body: SendCommandRequest) -> SendCommandResponse:
    """Send a command to a node via MQTT. Admin-only."""
    service = get_node_command_service()
    request_id = service.publish_command(node_id, body.command, body.details)
    return SendCommandResponse(status="sent", request_id=request_id)


class NodeConfigUpdateRequest(BaseModel):
    """Update node config.json settings from mobile app."""
    settings: dict[str, int | float | str | bool]
    restart: bool = True  # restart node service to apply module-level settings


@router.post(
    "/nodes/{node_id}/node-config",
    response_model=SendCommandResponse,
)
def update_node_config(
    node_id: str,
    body: NodeConfigUpdateRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> SendCommandResponse:
    """Update a node's config.json from the mobile app (JWT auth).

    Publishes an MQTT command to the node with the settings to merge
    into its config.json. If restart=True (default), the node restarts
    its service after applying so module-level settings take effect.
    """
    service = get_node_command_service()
    request_id = service.publish_command(
        node_id,
        "update_node_config",
        {"settings": body.settings, "restart": body.restart},
    )
    return SendCommandResponse(status="sent", request_id=request_id)


class LedPreviewRequest(BaseModel):
    """Briefly show ``pattern`` on the node's LED chain, then auto-revert."""
    pattern: str
    duration_seconds: float = 3.0


@router.post(
    "/nodes/{node_id}/led/preview",
    response_model=SendCommandResponse,
)
def preview_led_pattern(
    node_id: str,
    body: LedPreviewRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> SendCommandResponse:
    """Preview an LED pattern on the node, then auto-revert (JWT auth).

    Drives the Test-LEDs picker on the mobile Hardware tab. Mirrors
    ``update_node_config``: publishes an MQTT command, the node's
    ``handle_preview_led_pattern`` handler (mqtt_tts_listener) flashes
    the pattern for ``duration_seconds`` and reverts to its stable
    state. No persisted change.

    Endpoint was missing in v0.1.x — the mobile app would hit 404 when
    tapping any LED-state chip even though brightness/toggle worked
    (those went through the existing update_node_config route).
    """
    service = get_node_command_service()
    request_id = service.publish_command(
        node_id,
        "preview_led_pattern",
        {"pattern": body.pattern, "duration_seconds": body.duration_seconds},
    )
    return SendCommandResponse(status="sent", request_id=request_id)


@router.post(
    "/commands/{request_id}/verify",
    response_model=VerifyResponse,
)
def verify_command(
    request_id: str,
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> VerifyResponse:
    """Verify that a command request_id is legitimate. Node-only."""
    service = get_node_command_service()
    valid = service.verify_command(request_id, node_context.node.node_id)
    return VerifyResponse(valid=valid)


# ── Node Push Notification ────────────────────────────────────────────


class PushNotificationRequest(BaseModel):
    title: str
    body: str
    priority: str = "default"
    category: str = "alert"


class PushNotificationResponse(BaseModel):
    sent: bool
    inbox_item_id: str | None = None


@router.post(
    "/node/push-notification",
    response_model=PushNotificationResponse,
)
async def node_push_notification(
    request: PushNotificationRequest,
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> PushNotificationResponse:
    """Push a notification from a node to the household's mobile devices.

    Used by agents (e.g., reminder_agent) to send push notifications
    when a setting like REMINDER_PUSH_NOTIFICATIONS is enabled.
    """
    from app.services.inbox_notification_service import push_confirmation_to_inbox

    household_id = node_context.node.household_id
    node_id = node_context.node.node_id

    inbox_item_id = await push_confirmation_to_inbox(
        household_id=household_id,
        user_id=None,  # Broadcast to household
        node_id=node_id,
        title=request.title,
        summary=request.body,
        body=request.body,
        command_name="reminder",
        actions=[],  # No actions — informational only
    )

    return PushNotificationResponse(
        sent=inbox_item_id is not None,
        inbox_item_id=inbox_item_id,
    )
