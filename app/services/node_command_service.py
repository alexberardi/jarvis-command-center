"""
Service for publishing verified commands to nodes via MQTT.

Implements a verify-callback pattern:
1. Admin triggers a command → service generates request_id, publishes via MQTT
2. Node receives MQTT message, extracts request_id
3. Node calls back to verify the request_id is legitimate
4. Only then does the node execute the command

This prevents forged MQTT messages from triggering actions on nodes.
"""
import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

logger = logging.getLogger("uvicorn")

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
