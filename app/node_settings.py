"""
Node settings management API.

Implements the settings request/snapshot flow for encrypted node settings
as specified in jarvis-node-settings.md.

Flow:
1. Mobile creates request -> CC stores request, publishes MQTT signal
2. Node confirms request -> CC returns request details
3. Node uploads snapshot -> CC stores encrypted snapshot, marks request fulfilled
4. Mobile polls result -> CC returns snapshot with AAD for decryption
"""
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from .deps import get_db, verify_api_key, verify_user_jwt, verify_household_role, AuthenticatedUser
from .models import Node, SettingsRequest, SettingsSnapshot
from .context_providers.node_context_provider import NodeContextProvider

if TYPE_CHECKING:
    from app.core.mqtt_client import MQTTClient

router = APIRouter()
logger = logging.getLogger("uvicorn")

# MQTT client - initialized lazily
mqtt_client: Optional["MQTTClient"] = None


def get_mqtt_client() -> Optional["MQTTClient"]:
    """Get or create MQTT client.

    If the existing client has disconnected, cleanly shuts it down
    (stopping its background reconnect thread) before creating a new one.
    This prevents duplicate client IDs from fighting on the broker.
    """
    global mqtt_client
    if mqtt_client is not None and mqtt_client.is_connected:
        return mqtt_client

    # Tear down old client to stop its loop_start() thread
    if mqtt_client is not None:
        try:
            mqtt_client.disconnect()
        except Exception:
            pass
        mqtt_client = None

    try:
        from .core.mqtt_client import MQTTClient, get_mqtt_broker_url
        broker_url = get_mqtt_broker_url()
        if broker_url:
            mqtt_client = MQTTClient(broker_url)
            mqtt_client.connect()
            logger.info(f"✅ MQTT client connected to {broker_url}")
    except Exception as e:
        mqtt_client = None
        logger.warning(f"⚠️  MQTT client not available: {e}")
    return mqtt_client


# =============================================================================
# Pydantic Models
# =============================================================================

class SettingsRequestResponse(BaseModel):
    """Response model for settings request."""
    request_id: str
    node_id: str
    status: str
    created_at: datetime
    expires_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SnapshotUpload(BaseModel):
    """Request body for uploading encrypted snapshot."""
    ciphertext: str  # base64url encoded
    nonce: str       # base64url encoded
    tag: str         # base64url encoded
    aad_schema_version: int
    aad_commands_schema_version: int
    aad_revision: int


class SnapshotResponse(BaseModel):
    """Response model for snapshot."""
    snapshot_id: str
    node_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AADResponse(BaseModel):
    """AAD fields for client-side decryption."""
    node_id: str
    schema_version: int
    commands_schema_version: int
    revision: int
    request_id: str


class SnapshotDetailResponse(BaseModel):
    """Full snapshot with AAD for mobile decryption."""
    snapshot_id: str
    ciphertext: str
    nonce: str
    tag: str
    aad: AADResponse
    created_at: datetime


class ResultResponse(BaseModel):
    """Response for polling result endpoint."""
    status: str
    request_id: str
    snapshot: Optional[SnapshotDetailResponse] = None


class PendingResponse(BaseModel):
    """Response when request is still pending."""
    status: str
    request_id: str
    message: str


# =============================================================================
# Helper Functions
# =============================================================================

def _check_request_expired(request: SettingsRequest) -> bool:
    """Check if a request has expired."""
    return request.expires_at < datetime.utcnow()


def _publish_settings_request_mqtt(
    node_id: str, request_id: str, include_values: bool = False,
    user_id: int | None = None,
) -> None:
    """Publish MQTT message to signal node about settings request."""
    client = get_mqtt_client()
    if client is None:
        logger.warning(f"⚠️  MQTT not available, node {node_id} must poll for requests")
        return

    topic = f"jarvis/nodes/{node_id}/settings/request"
    mqtt_payload: dict = {
        "request_id": request_id,
        "node_id": node_id,
    }
    if include_values:
        mqtt_payload["include_values"] = True
    if user_id is not None:
        mqtt_payload["user_id"] = user_id
    payload = json.dumps(mqtt_payload)

    try:
        client.publish(topic, payload)
        logger.info(f"📤 Published settings request to {topic}")
    except Exception as e:
        logger.error(f"❌ Failed to publish MQTT message: {e}")
        # Don't fail the request - node can poll


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/nodes/{node_id}/settings/requests",
    response_model=SettingsRequestResponse,
    status_code=201,
)
def create_settings_request(
    node_id: str,
    include_values: bool = False,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
):
    """
    Create a new settings request for a node.

    This is called by the mobile app when it wants to retrieve node settings.
    Requires power_user role in the node's household.
    CC will publish an MQTT message to signal the node.

    Args:
        include_values: If True, the snapshot will include sensitive secret values
            (for secret sync to other nodes). Default False for normal settings view.
    """
    # Verify node exists
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    # Verify user has power_user role in the node's household
    if node.household_id:
        verify_household_role(user.user_id, node.household_id)

    # Create request with UUIDv4 (unguessable)
    request = SettingsRequest(
        request_id=str(uuid.uuid4()),
        node_id=node_id,
        status="pending",
    )
    db.add(request)
    db.commit()
    db.refresh(request)

    logger.info(f"📝 Settings request created: {request.request_id[:8]}... for node {node_id}")

    # Publish MQTT signal (non-blocking, best effort)
    _publish_settings_request_mqtt(node_id, request.request_id, include_values=include_values, user_id=user.user_id)

    return request


@router.get(
    "/nodes/{node_id}/settings/requests",
    response_model=list[SettingsRequestResponse],
)
def list_pending_settings_requests(
    node_id: str,
    node_context: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return the node's still-pending, non-expired settings requests.

    Backstops the MQTT publish path: a settings_request published
    while the node was offline beyond Mosquitto's persistence window
    (broker restart, fresh-provision before the first persistent
    session is established) is otherwise lost forever and the mobile
    snapshot poll spins on 202 indefinitely. The node calls this on
    every MQTT (re)connect and re-runs the snapshot pipeline for each
    pending ID, so the worst-case time-to-recovery for a dropped
    publish is bounded by the next reconnect — typically seconds.
    """
    if node_context.node.node_id != node_id:
        raise HTTPException(status_code=403, detail="Cannot access other node's requests")

    pending = (
        db.query(SettingsRequest)
        .filter(
            SettingsRequest.node_id == node_id,
            SettingsRequest.status == "pending",
            SettingsRequest.expires_at > datetime.utcnow(),
        )
        .order_by(SettingsRequest.created_at.asc())
        .all()
    )
    return pending


@router.get(
    "/nodes/{node_id}/settings/requests/{request_id}",
    response_model=SettingsRequestResponse,
)
def get_settings_request(
    node_id: str,
    request_id: str,
    node_context: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Get details of a settings request.

    This is called by the node to confirm a request exists before responding.
    Node must authenticate with its API key.
    """
    # Verify node is accessing its own request
    if node_context.node.node_id != node_id:
        raise HTTPException(status_code=403, detail="Cannot access other node's requests")

    request = db.query(SettingsRequest).filter(
        SettingsRequest.request_id == request_id,
        SettingsRequest.node_id == node_id,
    ).first()

    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    # Check expiration
    if _check_request_expired(request):
        raise HTTPException(status_code=410, detail="Request expired")

    return request


@router.put(
    "/nodes/{node_id}/settings/requests/{request_id}/snapshot",
    response_model=SnapshotResponse,
    status_code=201,
)
def upload_settings_snapshot(
    node_id: str,
    request_id: str,
    payload: SnapshotUpload,
    node_context: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Upload encrypted settings snapshot for a request.

    This is called by the node after confirming the request.
    The snapshot is encrypted with K2 using AES-256-GCM.
    """
    # Verify node is uploading to its own request
    if node_context.node.node_id != node_id:
        raise HTTPException(status_code=403, detail="Cannot upload to other node's requests")

    request = db.query(SettingsRequest).filter(
        SettingsRequest.request_id == request_id,
        SettingsRequest.node_id == node_id,
    ).first()

    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    # Check expiration
    if _check_request_expired(request):
        raise HTTPException(status_code=410, detail="Request expired")

    # Check status
    if request.status != "pending":
        raise HTTPException(status_code=409, detail=f"Request already {request.status}")

    # Create snapshot with AAD fields for client-side verification
    snapshot = SettingsSnapshot(
        snapshot_id=str(uuid.uuid4()),
        node_id=node_id,
        request_id=request_id,
        ciphertext=payload.ciphertext,
        nonce=payload.nonce,
        tag=payload.tag,
        aad_node_id=node_id,
        aad_schema_version=payload.aad_schema_version,
        aad_commands_schema_version=payload.aad_commands_schema_version,
        aad_revision=payload.aad_revision,
        aad_request_id=request_id,
    )
    db.add(snapshot)

    # Mark request as fulfilled
    request.status = "fulfilled"
    db.commit()
    db.refresh(snapshot)

    logger.info(f"✅ Snapshot uploaded for request {request_id[:8]}...")

    return snapshot


@router.get("/nodes/{node_id}/settings/requests/{request_id}/result")
def get_settings_result(
    node_id: str,
    request_id: str,
    response: Response,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
):
    """
    Poll for settings request result.

    This is called by the mobile app to check if the node has responded.
    Requires power_user role in the node's household.
    Returns 202 if still pending, 200 with snapshot if fulfilled.
    """
    request = db.query(SettingsRequest).filter(
        SettingsRequest.request_id == request_id,
        SettingsRequest.node_id == node_id,
    ).first()

    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    # Check expiration
    if _check_request_expired(request) and request.status == "pending":
        raise HTTPException(status_code=410, detail="Request expired")

    # Still pending
    if request.status == "pending":
        response.status_code = 202
        return PendingResponse(
            status="pending",
            request_id=request_id,
            message="Waiting for node response",
        ).model_dump()

    # Fulfilled - return snapshot
    snapshot = db.query(SettingsSnapshot).filter(
        SettingsSnapshot.request_id == request_id,
    ).first()

    if not snapshot:
        raise HTTPException(status_code=500, detail="Snapshot missing for fulfilled request")

    return {
        "status": "fulfilled",
        "request_id": request_id,
        "snapshot": {
            "snapshot_id": snapshot.snapshot_id,
            "ciphertext": snapshot.ciphertext,
            "nonce": snapshot.nonce,
            "tag": snapshot.tag,
            "aad": {
                "node_id": snapshot.aad_node_id,
                "schema_version": snapshot.aad_schema_version,
                "commands_schema_version": snapshot.aad_commands_schema_version,
                "revision": snapshot.aad_revision,
                "request_id": snapshot.aad_request_id,
            },
            "created_at": snapshot.created_at.isoformat(),
        },
    }


# =============================================================================
# K2 Provisioning via MQTT
# =============================================================================


class K2ProvisionRequest(BaseModel):
    k2: str
    kid: str
    created_at: str


_K2_RESULT_DIR = os.path.join(tempfile.gettempdir(), "jarvis-k2-provision")
os.makedirs(_K2_RESULT_DIR, exist_ok=True)

# Short-lived store for K2 key material awaiting an authenticated node pull.
# The key lives here only for the duration of a provision_k2 call (one-time
# read by the node, else cleaned up in the finally). It is never persisted to
# the DB — the node pulls it over its X-API-Key channel, not via the broker.
_K2_PENDING_DIR = os.path.join(tempfile.gettempdir(), "jarvis-k2-pending")
os.makedirs(_K2_PENDING_DIR, exist_ok=True)


@router.post("/nodes/{node_id}/k2")
def provision_k2(
    node_id: str,
    body: K2ProvisionRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
):
    """Relay K2 encryption key to a node via MQTT and wait for ack.

    Used by the mobile app to provision K2 for Docker/headless nodes
    that aren't reachable via direct AP connection. Waits up to 15s
    for the node to acknowledge receipt.
    """
    import time as _time

    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    if node.household_id:
        verify_household_role(user.user_id, node.household_id, required_role="member")

    client = get_mqtt_client()
    if client is None:
        raise HTTPException(status_code=503, detail="MQTT not available")

    request_id = str(uuid.uuid4())

    # Zero-trust delivery: stash the key material in a short-lived pending file
    # and publish only a NUDGE (request_id, no key) over the anonymous MQTT
    # broker. The node fetches the key from GET /nodes/{id}/k2/provision/{rid}
    # over its authenticated (X-API-Key) channel, which one-time-reads and
    # deletes the pending file. A spoofed broker nudge finds nothing to pull.
    pending_file = os.path.join(_K2_PENDING_DIR, f"{request_id}.json")
    with open(pending_file, "w") as f:
        json.dump({
            "node_id": node_id,
            "k2": body.k2,
            "kid": body.kid,
            "created_at": body.created_at,
        }, f)

    result_file = os.path.join(_K2_RESULT_DIR, f"{request_id}.json")
    try:
        topic = f"jarvis/nodes/{node_id}/k2/provision"
        client.publish(topic, json.dumps({"request_id": request_id}))
        logger.info("K2 provision nudge sent to node %s (kid=%s, request_id=%s)", node_id, body.kid, request_id[:8])

        # Poll for ack from node
        deadline = _time.time() + 15.0
        while _time.time() < deadline:
            if os.path.exists(result_file):
                try:
                    with open(result_file) as f:
                        result = json.load(f)
                    os.unlink(result_file)
                    if result.get("success"):
                        logger.info("K2 provision acknowledged by node %s", node_id)
                        return {"ok": True, "node_id": node_id, "kid": body.kid}
                    else:
                        raise HTTPException(status_code=502, detail=result.get("error", "Node failed to store K2"))
                except (json.JSONDecodeError, OSError):
                    pass
            _time.sleep(0.3)

        # Timeout — node didn't ack
        try:
            os.unlink(result_file)
        except OSError:
            pass
        raise HTTPException(status_code=504, detail="Node did not acknowledge K2 — it may not be online yet. Try again in a few seconds.")
    finally:
        # Never let key material outlive the provisioning call (the node's
        # authenticated pull normally deletes it first).
        try:
            os.unlink(pending_file)
        except OSError:
            pass


@router.get("/nodes/{node_id}/k2/provision/{request_id}")
def fetch_k2_provision(
    node_id: str,
    request_id: str,
    node_context=Depends(verify_api_key),
):
    """Node pulls the authoritative K2 key over its authenticated channel.

    Zero-trust gate for K2: the MQTT k2/provision message is only a nudge; the
    key material is delivered here — one-time, to the authenticated node it was
    provisioned for. A spoofed nudge (no matching pending file) 404s, so an
    attacker who can publish to the anonymous broker cannot inject a K2 key.
    """
    if getattr(node_context, "node", None) is None or node_context.node.node_id != node_id:
        raise HTTPException(status_code=403, detail="Node mismatch")

    pending_file = os.path.join(_K2_PENDING_DIR, f"{request_id}.json")
    if not os.path.exists(pending_file):
        raise HTTPException(status_code=404, detail="No pending K2 for this request")

    try:
        with open(pending_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = None
    finally:
        # One-time read — remove regardless so the key can't be pulled twice.
        try:
            os.unlink(pending_file)
        except OSError:
            pass

    if not data or data.get("node_id") != node_id:
        raise HTTPException(status_code=404, detail="No pending K2 for this request")

    return {"k2": data["k2"], "kid": data["kid"], "created_at": data.get("created_at", "")}


@router.post("/nodes/{node_id}/k2/ack/{request_id}")
def ack_k2_provision(
    node_id: str,
    request_id: str,
    body: dict,
    node_context=Depends(verify_api_key),
):
    """Node acknowledges K2 receipt. Writes result file for polling."""
    result_file = os.path.join(_K2_RESULT_DIR, f"{request_id}.json")
    with open(result_file, "w") as f:
        json.dump(body, f)
    return {"status": "ok"}
