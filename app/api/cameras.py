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

import json
import logging
import os
import tempfile
import time
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models import Device, Node
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext

router = APIRouter()
logger = logging.getLogger("uvicorn")

GO2RTC_BASE_URL: str = os.getenv("GO2RTC_URL", "http://jarvis-go2rtc:1984")

# Track active streams: device_id → stream_name
_active_streams: dict[str, str] = {}

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
    # Legacy fields (optional for backward compat with old mobile versions)
    refresh_token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    project_id: str | None = None
    protocols: str = "RTSP"


class StartStreamResponse(BaseModel):
    stream_name: str
    hls_url: str


# =============================================================================
# MQTT Credential Retrieval
# =============================================================================


def _fetch_credentials_from_node(
    household_id: str, protocol: str, db: Session,
) -> dict[str, str]:
    """Request camera credentials from the node via MQTT and wait for response."""
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
        ).first()
    if not node:
        nodes = db.query(Node).filter(Node.household_id == household_id).all()
        node = nodes[-1] if nodes else None
    if not node:
        raise HTTPException(status_code=400, detail="No node available in household")

    from app.node_settings import get_mqtt_client

    mqtt = get_mqtt_client()
    if mqtt is None:
        raise HTTPException(status_code=503, detail="MQTT not available")

    request_id: str = str(uuid4())
    result_file: str = os.path.join(_CREDS_DIR, f"creds-{request_id}.json")

    # Publish MQTT request to node
    topic: str = f"jarvis/nodes/{node.node_id}/camera-credentials"
    payload: str = json.dumps({
        "request_id": request_id,
        "protocol": protocol,
    })
    mqtt.publish(topic, payload)

    logger.info(
        "Camera credentials requested from node: request=%s node=%s protocol=%s",
        request_id[:8], node.node_id[:8], protocol,
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
        time.sleep(0.1)

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

    Credentials are fetched from the node via MQTT. Legacy mobile versions
    may still send credentials in the body — those are used directly.
    """
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

    # Extract Nest device ID from cloud_id (full SDM path → just the ID suffix)
    nest_device_id: str = device.cloud_id or ""
    if "/" in nest_device_id:
        nest_device_id = nest_device_id.rsplit("/", 1)[-1]
    if not nest_device_id:
        raise HTTPException(status_code=400, detail="Device has no cloud_id")

    # Resolve credentials: use legacy body fields or fetch from node via MQTT
    has_legacy_creds: bool = all([
        body.refresh_token, body.client_id, body.client_secret, body.project_id,
    ])
    if has_legacy_creds:
        creds = {
            "refresh_token": body.refresh_token,
            "client_id": body.client_id,
            "client_secret": body.client_secret,
            "project_id": body.project_id,
        }
    else:
        protocol: str = device.protocol or "nest"
        creds = _fetch_credentials_from_node(household_id, protocol, db)

    # Build go2rtc nest stream URL
    stream_name: str = f"cam_{device.entity_id}"
    nest_params: dict[str, str] = {
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "project_id": creds["project_id"],
        "device_id": nest_device_id,
        "protocols": body.protocols,
    }
    stream_url: str = f"nest:?{urlencode(nest_params)}"

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

    # Return proxied HLS URL (relative to CC)
    hls_url: str = f"/api/v0/cameras/stream/{stream_name}/stream.m3u8"

    logger.info("Camera stream started: device=%s stream=%s", device_id, stream_name)
    return StartStreamResponse(stream_name=stream_name, hls_url=hls_url)


@router.post("/camera-credentials/{request_id}")
def post_camera_credentials_result(
    request_id: str,
    body: dict,
) -> dict:
    """Node POSTs camera credentials back to CC (MQTT callback)."""
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
    stream_name: str | None = _active_streams.pop(device_id, None)
    if not stream_name:
        return {"status": "not_streaming"}

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
