"""Node tool definition fetch endpoint.

Fetches client_tools and available_commands from nodes via MQTT on every request.
No caching — always returns fresh data from the node.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.deps import get_db, verify_user_jwt, verify_household_role, AuthenticatedUser
from app.models import Node

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["node-tools"])

_RESULT_DIR = "/tmp/jarvis-node-tools"


@router.get("/nodes/{node_id}/tools")
async def get_node_tools(
    node_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get a node's tool definitions via MQTT.

    Returns:
        {
            client_tools: [...],
            available_commands: [...],
            installed_packages: [{name, version}, ...],
        }
    """
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    if node.household_id:
        verify_household_role(user.user_id, node.household_id, required_role="member")

    tools = await _request_tools_from_node(node_id)
    if tools:
        return {
            "client_tools": tools["client_tools"],
            "available_commands": tools["available_commands"],
            "installed_packages": tools.get("installed_packages", []),
        }

    return {
        "client_tools": [],
        "available_commands": [],
        "installed_packages": [],
    }


async def _request_tools_from_node(
    node_id: str, timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Send MQTT request to node for its tool definitions, wait for response."""
    try:
        from app.services.node_command_service import get_node_command_service
    except ImportError:
        return None

    request_id = str(uuid4())
    os.makedirs(_RESULT_DIR, exist_ok=True)
    result_file = os.path.join(_RESULT_DIR, f"{request_id}.json")

    service = get_node_command_service()
    details = {
        "reply_request_id": request_id,
        "trusted": True,
    }
    service.publish_command_with_id(node_id, "report_tools", details, request_id)

    # Poll for result
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    result = json.load(f)
                os.unlink(result_file)
                return result
            except (json.JSONDecodeError, OSError):
                pass
        await asyncio.sleep(0.1)

    # Clean up
    try:
        os.unlink(result_file)
    except OSError:
        pass
    return None


@router.post("/node-tool-reports/{request_id}")
def post_node_tool_report(request_id: str, body: dict) -> dict:
    """Callback for nodes to POST their tool definitions.

    The node receives a 'report_tools' MQTT command, builds its tool schemas,
    and POSTs them here. Written to file for the polling endpoint to pick up.
    """
    os.makedirs(_RESULT_DIR, exist_ok=True)
    result_file = os.path.join(_RESULT_DIR, f"{request_id}.json")
    with open(result_file, "w") as f:
        json.dump(body, f)
    return {"status": "ok"}
