"""Camera streaming API: list cameras, register go2rtc streams, proxy HLS.

Flow:
1. Mobile calls GET /households/{id}/cameras to list camera devices
2. Mobile calls POST /households/{id}/cameras/{device_id}/stream (no credentials)
3. CC fetches credentials from the node via MQTT (same pattern as device state)
4. CC registers the stream in go2rtc via its REST API
5. CC proxies HLS segments back to mobile (go2rtc is internal-only)
6. Mobile plays the proxied HLS URL via react-native-video

go2rtc runs on the Docker network with no exposed ports. CC is the only
path to camera streams, providing JWT auth on every request.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import get_db, verify_api_key
from app.models import Device, Node
from app.provisioning import (
    ProvisioningAuthContext,
    require_household_access,
    verify_provisioning_auth,
)

router = APIRouter()
logger = logging.getLogger("uvicorn")

GO2RTC_BASE_URL: str = os.getenv("GO2RTC_URL", "http://jarvis-go2rtc:1984")

# Track active streams: device_id → stream_name
_active_streams: dict[str, str] = {}
# stream_name -> owning household_id. The HLS proxy authorizes against this,
# NOT the global _active_streams set (which is not a tenant boundary).
_stream_households: dict[str, str] = {}

# The HLS proxy forwards {path} to go2rtc; restrict it to media files so it
# cannot be pivoted to go2rtc control endpoints like /api/config (which would
# disclose every household's camera/OAuth credentials).
_ALLOWED_STREAM_SUFFIXES = (".m3u8", ".ts", ".mp4", ".m4s", ".aac", ".vtt", ".key")


def _is_allowed_stream_path(path: str) -> bool:
    return ".." not in path and path.lower().endswith(_ALLOWED_STREAM_SUFFIXES)

# Temp directory for MQTT credential responses (shared with smart_home.py pattern)
_CREDS_DIR: str = os.path.join(tempfile.gettempdir(), "jarvis-device-control")
os.makedirs(_CREDS_DIR, exist_ok=True)

_CREDS_TIMEOUT_SECONDS: float = 10.0


# =============================================================================
# Models
# =============================================================================


class CameraResponse(BaseModel):
    device_id: str
    entity_id: str
    name: str
    protocol: str | None = None
    cloud_id: str | None = None
    room_name: str | None = None
    is_streaming: bool = False


class StartStreamRequest(BaseModel):
    """Empty body: the node builds the go2rtc source; CC needs no client input."""


class StartStreamResponse(BaseModel):
    stream_name: str
    hls_url: str


# =============================================================================
# MQTT Credential Retrieval
# =============================================================================


async def _fetch_credentials_from_node(
    household_id: str, device: Device, db: Session,
) -> dict[str, str]:
    """Ask the node (over MQTT) to build a go2rtc stream source and wait for it.

    The node's device-protocol plugin owns the source format and the choice of
    streaming transport; CC just relays the device identity and registers
    whatever ``stream_source`` string comes back.
    """
    from app.services.settings_service import get_settings_service

    settings = get_settings_service()
    primary_node_id: str = settings.get(
        "smart_home.primary_node_id", household_id=household_id,
    ) or ""

    # Find target node: prefer primary, fall back to any household node
    node = None
    if primary_node_id:
        node = db.query(Node).filter(
            Node.node_id == primary_node_id, Node.household_id == household_id,
            Node.is_active.is_(True),
        ).first()
    if not node:
        nodes = db.query(Node).filter(Node.household_id == household_id, Node.is_active.is_(True)).all()
        node = nodes[-1] if nodes else None
    if not node:
        raise HTTPException(status_code=400, detail="No active node available in household")

    from app.node_settings import get_mqtt_client

    mqtt = get_mqtt_client()
    if mqtt is None:
        raise HTTPException(status_code=503, detail="MQTT not available")

    request_id: str = str(uuid4())
    result_file: str = os.path.join(_CREDS_DIR, f"creds-{request_id}.json")

    # Publish MQTT request to node. The node needs the device identity (protocol
    # + full unstripped cloud_id) so its plugin can build the go2rtc source.
    topic: str = f"jarvis/nodes/{node.node_id}/camera-credentials"
    payload: str = json.dumps({
        "request_id": request_id,
        "protocol": device.protocol or "nest",
        "cloud_id": device.cloud_id,
        "entity_id": device.entity_id,
        "domain": device.domain,
    })
    mqtt.publish(topic, payload)

    logger.info(
        "Camera stream source requested from node: request=%s node=%s protocol=%s",
        request_id[:8], node.node_id[:8], device.protocol or "nest",
    )

    # Poll for result file (node writes it via POST callback)
    deadline: float = time.time() + _CREDS_TIMEOUT_SECONDS
    result: dict | None = None
    while time.time() < deadline:
        if os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    result = json.load(f)
                os.unlink(result_file)
                break
            except (json.JSONDecodeError, OSError):
                pass
        await asyncio.sleep(0.1)

    if result is None:
        try:
            os.unlink(result_file)
        except OSError:
            pass
        raise HTTPException(
            status_code=504,
            detail="Node did not respond with camera credentials. Is it online?",
        )

    # Check for error from node
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/households/{household_id}/cameras", response_model=list[CameraResponse])
def list_cameras(
    household_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> list[CameraResponse]:
    """List camera devices for a household."""
    require_household_access(household_id, auth)
    devices = (
        db.query(Device)
        .filter(
            Device.household_id == household_id,
            Device.domain == "camera",
            Device.is_active.is_(True),
        )
        .all()
    )

    cameras: list[CameraResponse] = []
    for d in devices:
        room_name: str | None = d.room.name if d.room else None
        cameras.append(
            CameraResponse(
                device_id=d.id,
                entity_id=d.entity_id,
                name=d.name,
                protocol=d.protocol,
                cloud_id=d.cloud_id,
                room_name=room_name,
                is_streaming=d.id in _active_streams,
            )
        )

    return cameras


@router.post(
    "/households/{household_id}/cameras/{device_id}/stream",
    response_model=StartStreamResponse,
)
async def start_camera_stream(
    household_id: str,
    device_id: str,
    body: StartStreamRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> StartStreamResponse:
    """Register a camera stream in go2rtc.

    The node's device-protocol plugin builds the go2rtc source string (owning the
    protocol/transport choice); CC fetches it via MQTT and registers it verbatim.
    """
    require_household_access(household_id, auth)
    device = (
        db.query(Device)
        .filter(
            Device.id == device_id,
            Device.household_id == household_id,
            Device.domain == "camera",
        )
        .first()
    )
    if not device:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Ask the node's device-protocol plugin to build the go2rtc source. CC owns
    # no protocol specifics — it registers whatever source string comes back.
    result = await _fetch_credentials_from_node(household_id, device, db)
    stream_url: str = result.get("stream_source", "")
    if not stream_url:
        raise HTTPException(status_code=400, detail="No stream source returned by node")

    stream_name: str = f"cam_{device.entity_id}"

    # Register stream in go2rtc via its REST API.
    # go2rtc may return 400 from config file persistence even when the stream
    # registers successfully in memory. We verify by checking the streams list.
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.put(
                f"{GO2RTC_BASE_URL}/api/streams",
                params={"name": stream_name, "src": stream_url},
            )
            if resp.status_code not in (200, 201):
                # Check if stream registered despite the config persistence error
                verify = await client.get(f"{GO2RTC_BASE_URL}/api/streams")
                if verify.status_code == 200 and stream_name in verify.text:
                    logger.info("go2rtc stream registered (config persistence skipped)")
                else:
                    logger.error("go2rtc stream registration failed: %d %s", resp.status_code, resp.text)
                    raise HTTPException(status_code=502, detail="Failed to register stream")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Camera streaming service not available")

    _active_streams[device_id] = stream_name
    _stream_households[stream_name] = household_id

    # Return proxied HLS URL (relative to CC)
    hls_url: str = f"/api/v0/cameras/stream/{stream_name}/stream.m3u8"

    logger.info("Camera stream started: device=%s stream=%s", device_id, stream_name)
    return StartStreamResponse(stream_name=stream_name, hls_url=hls_url)


@router.post("/camera-credentials/{request_id}")
def post_camera_credentials_result(
    request_id: str,
    body: dict,
    _node: NodeContextProvider = Depends(verify_api_key),
) -> dict:
    """Node POSTs camera credentials back to CC (MQTT callback).

    Node-authenticated (X-API-Key): an unauthenticated caller must not be able to
    poison the credentials a node fetch is waiting on. `request_id` must be the
    UUID CC issued — validating it also stops a crafted id from path-traversing
    the creds filename into an arbitrary-file write.
    """
    try:
        UUID(str(request_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request_id")

    result_file: str = os.path.join(_CREDS_DIR, f"creds-{request_id}.json")
    with open(result_file, "w") as f:
        json.dump(body, f)
    return {"status": "ok"}


@router.delete("/households/{household_id}/cameras/{device_id}/stream")
async def stop_camera_stream(
    household_id: str,
    device_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
) -> dict:
    """Stop a camera stream and remove it from go2rtc."""
    require_household_access(household_id, auth)
    stream_name: str | None = _active_streams.pop(device_id, None)
    if not stream_name:
        return {"status": "not_streaming"}
    _stream_households.pop(stream_name, None)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.delete(
                f"{GO2RTC_BASE_URL}/api/streams",
                params={"src": stream_name},
            )
    except httpx.ConnectError:
        pass  # go2rtc down, stream will expire anyway

    logger.info("Camera stream stopped: device=%s stream=%s", device_id, stream_name)
    return {"status": "stopped"}


@router.get("/cameras/stream/{stream_name}/{path:path}")
async def proxy_stream(
    stream_name: str,
    path: str,
    request: Request,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
) -> StreamingResponse:
    """Proxy HLS/MP4 streams from go2rtc. JWT-authenticated."""
    if stream_name not in _active_streams.values():
        raise HTTPException(status_code=404, detail="Stream not found")

    # Only a member of the stream's owning household may proxy it (admin
    # bypasses). Resolving from _stream_households, never a client field.
    require_household_access(_stream_households.get(stream_name), auth)

    # Never let the proxied path reach a go2rtc control endpoint (/api/config).
    if not _is_allowed_stream_path(path):
        raise HTTPException(status_code=404, detail="Stream not found")

    query_params: dict[str, str] = dict(request.query_params)
    if "src" not in query_params:
        query_params["src"] = stream_name

    go2rtc_url: str = f"{GO2RTC_BASE_URL}/api/{path}"

    try:
        client = httpx.AsyncClient(timeout=30.0)
        req = client.build_request("GET", go2rtc_url, params=query_params)
        resp = await client.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aclose()
            await client.aclose()
            raise HTTPException(status_code=resp.status_code, detail="Stream error")

        content_type: str = resp.headers.get("content-type", "application/octet-stream")

        async def stream_body():
            try:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(stream_body(), media_type=content_type)

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Camera streaming service not available")
