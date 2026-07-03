"""Test install API: mobile -> CC -> MQTT nudge -> node verifies -> Pantry download.

Zero-trust flow:
1. Mobile POSTs share_code + node_id (JWT auth)
2. CC fetches package info from Pantry to validate share code
3. CC creates TestInstallRequest, publishes MQTT with just request_id
4. Node receives MQTT nudge, calls /verify to confirm + get download info
5. Node downloads from Pantry, installs to test_commands/, POSTs results
6. Mobile polls for status
"""

import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.service_config import get_pantry_url
from app.deps import get_db, verify_api_key
from app.models import Node, TestInstallRequest
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext, require_household_access

router = APIRouter()
logger = logging.getLogger("uvicorn")


# =============================================================================
# Request/Response Models
# =============================================================================


class TestInstallBody(BaseModel):
    share_code: str


class TestInstallResponse(BaseModel):
    id: str
    status: str
    package_name: str
    created_at: datetime


class TestInstallResultUpload(BaseModel):
    success: bool
    error: str | None = None
    details: dict | None = None


class TestInstallPollResponse(BaseModel):
    status: str
    request_id: str
    package_name: str
    error_message: str | None = None
    details: dict | None = None


class TestInstallVerifyResponse(BaseModel):
    """Returned to the node when it verifies a test install request."""
    confirmed: bool
    package_name: str
    pantry_download_url: str


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/nodes/{node_id}/test-install",
    response_model=TestInstallResponse,
    status_code=201,
)
def request_test_install(
    node_id: str,
    body: TestInstallBody,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> TestInstallResponse:
    """Request a test install on a node. Mobile calls this with a share code."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    require_household_access(node.household_id, auth)

    share_code = body.share_code.strip().upper()
    if not share_code or len(share_code) != 6:
        raise HTTPException(status_code=400, detail="Invalid share code")

    # Validate share code with Pantry
    try:
        pantry_url = get_pantry_url()
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{pantry_url}/v1/forge/drafts/{share_code}")
    except Exception as e:
        logger.error("Failed to reach Pantry: %s", e)
        raise HTTPException(status_code=502, detail="Could not reach Pantry service")

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Share code not found or expired")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Pantry returned an error")

    draft = resp.json()
    package_name = draft.get("package_name", "unknown")

    now = datetime.utcnow()
    install_request = TestInstallRequest(
        id=str(uuid4()),
        node_id=node_id,
        household_id=node.household_id or "",
        share_code=share_code,
        package_name=package_name,
        status="pending",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    db.add(install_request)
    db.commit()
    db.refresh(install_request)

    logger.info(
        "Test install requested: node=%s package=%s code=%s request_id=%s",
        node_id, package_name, share_code, install_request.id[:8],
    )

    # Publish lightweight MQTT nudge (just request_id, no code or files)
    _publish_test_install_mqtt(node_id, install_request.id)

    return TestInstallResponse(
        id=install_request.id,
        status=install_request.status,
        package_name=package_name,
        created_at=install_request.created_at,
    )


@router.get(
    "/nodes/{node_id}/test-install/{request_id}/verify",
    response_model=TestInstallVerifyResponse,
)
def verify_test_install(
    node_id: str,
    request_id: str,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> TestInstallVerifyResponse:
    """Node calls this to verify a test install request and get the download URL.

    This is the zero-trust gate: spoofed MQTT messages with fake request_ids
    will fail here with 404 because no matching row exists in the database.
    """
    install_request = db.query(TestInstallRequest).filter(
        TestInstallRequest.id == request_id,
        TestInstallRequest.node_id == node_id,
    ).first()
    if not install_request:
        raise HTTPException(status_code=404, detail="Test install request not found")

    now = datetime.utcnow()
    if install_request.expires_at and install_request.expires_at < now:
        install_request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Test install request expired")

    if install_request.status != "pending":
        raise HTTPException(status_code=409, detail=f"Request already {install_request.status}")

    try:
        pantry_url = get_pantry_url()
    except ValueError:
        raise HTTPException(status_code=502, detail="Pantry service not configured")

    return TestInstallVerifyResponse(
        confirmed=True,
        package_name=install_request.package_name,
        pantry_download_url=f"{pantry_url}/v1/forge/drafts/{install_request.share_code}",
    )


@router.post("/nodes/{node_id}/test-install/{request_id}/results")
def upload_test_install_results(
    node_id: str,
    request_id: str,
    body: TestInstallResultUpload,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Node uploads test install results after completing the install."""
    install_request = db.query(TestInstallRequest).filter(
        TestInstallRequest.id == request_id,
        TestInstallRequest.node_id == node_id,
    ).first()
    if not install_request:
        raise HTTPException(status_code=404, detail="Test install request not found")

    now = datetime.utcnow()
    if install_request.expires_at and install_request.expires_at < now:
        install_request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Test install request expired")

    if body.success:
        install_request.status = "completed"
        if body.details:
            install_request.results_json = json.dumps(body.details)
    else:
        install_request.status = "failed"
        install_request.error_message = body.error or "Unknown error"

    install_request.completed_at = now
    db.commit()

    logger.info(
        "Test install results: request=%s status=%s package=%s",
        request_id[:8], install_request.status, install_request.package_name,
    )
    return {"status": "ok"}


@router.get(
    "/nodes/{node_id}/test-install/{request_id}",
    response_model=TestInstallPollResponse,
)
def poll_test_install(
    node_id: str,
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> TestInstallPollResponse:
    """Mobile polls for test install results."""
    install_request = db.query(TestInstallRequest).filter(
        TestInstallRequest.id == request_id,
        TestInstallRequest.node_id == node_id,
    ).first()
    if not install_request:
        raise HTTPException(status_code=404, detail="Test install request not found")

    now = datetime.utcnow()

    # Check expiration
    if install_request.status == "pending" and install_request.expires_at and install_request.expires_at < now:
        install_request.status = "expired"
        db.commit()

    if install_request.status == "expired":
        return TestInstallPollResponse(
            status="expired",
            request_id=request_id,
            package_name=install_request.package_name,
            error_message="Test install request expired — node may be offline",
        )

    if install_request.status == "pending":
        return TestInstallPollResponse(
            status="pending",
            request_id=request_id,
            package_name=install_request.package_name,
        )

    if install_request.status == "failed":
        return TestInstallPollResponse(
            status="failed",
            request_id=request_id,
            package_name=install_request.package_name,
            error_message=install_request.error_message,
        )

    # Completed
    details = None
    if install_request.results_json:
        details = json.loads(install_request.results_json)

    return TestInstallPollResponse(
        status="completed",
        request_id=request_id,
        package_name=install_request.package_name,
        details=details,
    )


# =============================================================================
# MQTT helper
# =============================================================================


def _publish_test_install_mqtt(node_id: str, request_id: str) -> None:
    """Publish lightweight MQTT nudge to tell node it has a test install waiting.

    Only sends the request_id — no share code, no files, no URLs.
    The node must verify with CC to get download info (zero-trust).
    """
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, node %s cannot receive test install", node_id)
        return

    topic = f"jarvis/nodes/{node_id}/test-install"
    payload = json.dumps({"request_id": request_id})

    try:
        client.publish(topic, payload)
        logger.info("Published test install nudge to %s", topic)
    except Exception as e:
        logger.error("Failed to publish test install MQTT: %s", e)
