"""
Service for publishing verified commands to nodes via MQTT.

Implements a verify-callback pattern:
1. Admin triggers a command → service generates request_id, publishes via MQTT
2. Node receives MQTT message, extracts request_id
3. Node calls back to verify the request_id is legitimate
4. Only then does the node execute the command

This prevents forged MQTT messages from triggering actions on nodes.
"""
import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from uuid import uuid4

logger = logging.getLogger("uvicorn")

# The node POSTs a command's result to CC's POST /device-control-results/{request_id},
# which writes it here. All single-command dispatch paths (mobile chat, errands,
# device control) share this dir + the request_id-as-filename correlation.
_RESULT_DIR = os.path.join(tempfile.gettempdir(), "jarvis-device-control")

# Singleton instance
_service: "NodeCommandService | None" = None


class NodeCommandService:
    """Publish commands to nodes via MQTT with verify-callback security."""

    def __init__(self) -> None:
        # In-memory store: {request_id: {node_id, command, created_at, expires_at}}
        self._pending_commands: dict[str, dict] = {}

    def publish_command_with_id(
        self, node_id: str, command: str, details: dict | None, request_id: str,
    ) -> str:
        """Publish a command with a caller-supplied request_id. Returns request_id."""
        return self._publish(node_id, command, details, request_id)

    def publish_command(self, node_id: str, command: str, details: dict | None = None) -> str:
        """Publish a command to a node via MQTT. Returns request_id."""
        return self._publish(node_id, command, details, str(uuid4()))

    def _publish(self, node_id: str, command: str, details: dict | None, request_id: str) -> str:
        """Internal: register and publish a command."""
        from app.node_settings import get_mqtt_client

        now = datetime.utcnow()
        self._pending_commands[request_id] = {
            "node_id": node_id,
            "command": command,
            "created_at": now,
            "expires_at": now + timedelta(minutes=5),
        }

        # Clean up expired entries while we're here
        self._cleanup_expired()

        topic = f"jarvis/nodes/{node_id}/commands"
        payload = json.dumps([{
            "command": command,
            "details": {**(details or {}), "request_id": request_id},
        }])

        client = get_mqtt_client()
        if client is None:
            logger.warning("MQTT not available, command %s for node %s stored but not delivered", command, node_id)
            return request_id

        try:
            client.publish(topic, payload)
            logger.info("Published command %s to node %s (request_id=%s)", command, node_id, request_id[:8])
        except Exception as e:
            logger.error("Failed to publish MQTT command: %s", e)

        return request_id

    def verify_command(self, request_id: str, node_id: str) -> bool:
        """Verify a command was issued by this service for this node. One-time use."""
        entry = self._pending_commands.get(request_id)
        if not entry:
            return False
        if entry["node_id"] != node_id:
            logger.warning(
                "Command verify mismatch: request %s belongs to %s, not %s",
                request_id[:8], entry["node_id"], node_id,
            )
            return False
        if datetime.utcnow() > entry["expires_at"]:
            del self._pending_commands[request_id]
            return False

        # Valid — remove to prevent replay
        del self._pending_commands[request_id]
        return True

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the pending store."""
        now = datetime.utcnow()
        expired = [rid for rid, entry in self._pending_commands.items() if now > entry["expires_at"]]
        for rid in expired:
            del self._pending_commands[rid]


def get_node_command_service() -> NodeCommandService:
    """Get the global NodeCommandService singleton."""
    global _service
    if _service is None:
        _service = NodeCommandService()
    return _service


async def _await_result_file(request_id: str, timeout: float) -> dict | None:
    """Poll <tmpdir>/jarvis-device-control/{request_id}.json for the node's reply.

    The node POSTs its result to CC's /device-control-results/{request_id}; that
    handler writes the file. Returns the parsed body (and deletes the file) or
    None on timeout. Correlation is purely the request_id in the filename.
    """
    os.makedirs(_RESULT_DIR, exist_ok=True)
    result_file = os.path.join(_RESULT_DIR, f"{request_id}.json")
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
    try:
        os.unlink(result_file)
    except OSError:
        pass
    return None


async def dispatch_node_command(
    node_id: str,
    command_name: str,
    arguments: dict | None = None,
    *,
    user_id: int | None = None,
    voice_command: str | None = None,
    tool_call_id: str | None = None,
    timeout: float = 10.0,
) -> dict:
    """Run ONE command on a node headlessly and await its structured output.

    Publishes the ``tool_call`` MQTT verb (the same one mobile chat uses — the
    node's ``handle_tool_call`` looks the command up in its local registry, runs
    ``cmd.execute(...)``, and POSTs ``{"output": {...}}`` back) and awaits the
    result file. This is the per-step primitive the errand executor dispatches a
    NODE step through — no transient routine, no pre-pull.

    Returns the node's ``output`` dict on success — ``{...context_data, "success":
    bool, "error"?: str, "message"?: str, "actions"?: [...]}``. On a publish
    failure or a timeout (node offline / slow) returns a synthetic failure
    ``{"success": False, "error": ..., "timeout": True?}`` — never raises for
    those normal cases, so a caller looping over steps always gets a dict.
    """
    request_id = str(uuid4())
    details: dict = {
        "command_name": command_name,
        "arguments": arguments or {},
        "tool_call_id": tool_call_id or request_id,
        "reply_request_id": request_id,
        "trusted": True,
    }
    if user_id is not None:
        details["user_id"] = user_id
    if voice_command:
        details["voice_command"] = voice_command

    try:
        get_node_command_service().publish_command_with_id(
            node_id, "tool_call", details, request_id
        )
    except Exception as exc:  # noqa: BLE001 — surface as a failed step, don't crash the loop
        logger.error("Node command publish failed for %s on %s: %s", command_name, node_id, exc)
        return {"success": False, "error": f"could not dispatch to node: {exc}"}

    result = await _await_result_file(request_id, timeout)
    if result is None:
        return {"success": False, "error": "the node didn't respond in time", "timeout": True}
    output = result.get("output", result)
    if not isinstance(output, dict):
        output = {"success": True, "result": output}
    return output
