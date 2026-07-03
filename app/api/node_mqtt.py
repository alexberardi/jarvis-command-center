"""Node MQTT credential handout.

A node connects to the broker over the LAN at an address it already knows (from
provisioning); it just needs the shared broker credential once broker auth is
enabled. It fetches that here over its authenticated HTTP channel (X-API-Key) —
never over MQTT itself, which would be circular.

Transition-safe: while broker auth is not yet enabled (credentials unset), this
returns nulls and the node connects anonymously, exactly as today. Only once the
broker password file + MQTT_USERNAME/MQTT_PASSWORD are in place does it hand back
a real credential.
"""
from fastapi import APIRouter, Depends

from app.context_providers.node_context_provider import NodeContextProvider
from app.core.mqtt_client import get_mqtt_credentials
from app.deps import verify_api_key

router = APIRouter(prefix="/node", tags=["node"])


@router.get("/mqtt-credentials")
def node_mqtt_credentials(
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict[str, str | None]:
    """Return the shared MQTT broker credentials for the authenticated node.

    Response: ``{"username": str | None, "password": str | None}``. Nulls mean
    "no broker auth yet — connect anonymously".
    """
    username, password = get_mqtt_credentials()
    return {"username": username, "password": password}
