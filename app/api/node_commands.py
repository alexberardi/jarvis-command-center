"""
API endpoints for sending commands to nodes and verifying command requests.

POST /nodes/{node_id}/commands — admin sends a command to a node via MQTT
POST /nodes/{node_id}/actions — mobile forwards a user action (e.g. Send/Cancel) to a node via MQTT
POST /commands/{request_id}/verify — node verifies a command is legitimate
"""
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import AuthenticatedUser, verify_admin_key, verify_api_key, verify_user_jwt
from app.services.node_command_service import get_node_command_service

logger = logging.getLogger("uvicorn")

router = APIRouter()


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
    """
    service = get_node_command_service()
    details = {
        "command_name": body.command_name,
        "action_name": body.action_name,
        "context": body.context or {},
    }
    request_id = service.publish_command(node_id, "action", details)
    return ActionResponse(status="sent", request_id=request_id)


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
