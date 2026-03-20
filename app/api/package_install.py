"""Package install API: mobile -> CC -> MQTT -> node -> CC -> mobile.

Follows the same request/poll pattern as device-scan (smart_home.py).
Also handles prompt provider installs (CC-local, no MQTT).
"""

import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.deps import get_db, verify_api_key
from app.models import Node, PackageInstallRequest
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext

router = APIRouter()
logger = logging.getLogger("uvicorn")


# =============================================================================
# Request/Response Models
# =============================================================================


class PackageInstallBody(BaseModel):
    command_name: str
    github_repo_url: str
    git_tag: str | None = None


class PackageInstallResponse(BaseModel):
    id: str
    status: str
    created_at: datetime


class PackageInstallResultUpload(BaseModel):
    success: bool
    error: str | None = None
    details: dict | None = None


class PackageInstallPollResponse(BaseModel):
    status: str
    request_id: str
    command_name: str
    error_message: str | None = None
    details: dict | None = None


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/nodes/{node_id}/package-install",
    response_model=PackageInstallResponse,
    status_code=201,
)
def request_package_install(
    node_id: str,
    body: PackageInstallBody,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> PackageInstallResponse:
    """Request a package install on a node. Mobile calls this, CC notifies node via MQTT."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    now = datetime.utcnow()
    install_request = PackageInstallRequest(
        id=str(uuid4()),
        node_id=node_id,
        household_id=node.household_id or "",
        command_name=body.command_name,
        github_repo_url=body.github_repo_url,
        git_tag=body.git_tag,
        status="pending",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    db.add(install_request)
    db.commit()
    db.refresh(install_request)

    logger.info(
        "Package install requested: node=%s command=%s request_id=%s",
        node_id, body.command_name, install_request.id[:8],
    )

    # Notify node via MQTT
    _publish_package_install_mqtt(node_id, install_request)

    return PackageInstallResponse(
        id=install_request.id,
        status=install_request.status,
        created_at=install_request.created_at,
    )


@router.post("/nodes/{node_id}/package-install/{request_id}/results")
def upload_package_install_results(
    node_id: str,
    request_id: str,
    body: PackageInstallResultUpload,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Node uploads install results after running the install pipeline."""
    install_request = db.query(PackageInstallRequest).filter(
        PackageInstallRequest.id == request_id,
        PackageInstallRequest.node_id == node_id,
    ).first()
    if not install_request:
        raise HTTPException(status_code=404, detail="Install request not found")

    now = datetime.utcnow()
    if install_request.expires_at and install_request.expires_at < now:
        install_request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Install request expired")

    if body.success:
        install_request.status = "completed"
        if body.details:
            install_request.results_json = json.dumps(body.details)

        # No tool cache to invalidate — tools are fetched fresh from node via MQTT.
    else:
        install_request.status = "failed"
        install_request.error_message = body.error or "Unknown error"

    install_request.completed_at = now
    db.commit()

    logger.info(
        "Package install results uploaded: request=%s status=%s command=%s",
        request_id[:8], install_request.status, install_request.command_name,
    )
    return {"status": "ok"}


@router.get(
    "/nodes/{node_id}/package-install/{request_id}",
    response_model=PackageInstallPollResponse,
)
def poll_package_install(
    node_id: str,
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> PackageInstallPollResponse:
    """Mobile polls for install results."""
    install_request = db.query(PackageInstallRequest).filter(
        PackageInstallRequest.id == request_id,
        PackageInstallRequest.node_id == node_id,
    ).first()
    if not install_request:
        raise HTTPException(status_code=404, detail="Install request not found")

    now = datetime.utcnow()

    # Check expiration
    if install_request.status == "pending" and install_request.expires_at and install_request.expires_at < now:
        install_request.status = "expired"
        db.commit()

    if install_request.status == "expired":
        return PackageInstallPollResponse(
            status="expired",
            request_id=request_id,
            command_name=install_request.command_name,
            error_message="Install request expired — node may be offline",
        )

    if install_request.status == "pending":
        return PackageInstallPollResponse(
            status="pending",
            request_id=request_id,
            command_name=install_request.command_name,
        )

    if install_request.status == "failed":
        return PackageInstallPollResponse(
            status="failed",
            request_id=request_id,
            command_name=install_request.command_name,
            error_message=install_request.error_message,
        )

    # Completed
    details = None
    if install_request.results_json:
        details = json.loads(install_request.results_json)

    return PackageInstallPollResponse(
        status="completed",
        request_id=request_id,
        command_name=install_request.command_name,
        details=details,
    )


# =============================================================================
# Prompt Provider Install (CC-local, no MQTT)
# =============================================================================


class PromptProviderInstallBody(BaseModel):
    github_repo_url: str
    git_tag: str | None = None


@router.post("/prompt-providers/install", status_code=201)
def install_prompt_provider_endpoint(
    body: PromptProviderInstallBody,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
) -> dict:
    """Install a prompt provider to the command center from GitHub."""
    from app.services.prompt_provider_installer import (
        install_prompt_provider,
        PromptProviderInstallError,
    )

    try:
        result = install_prompt_provider(body.github_repo_url, body.git_tag)
        return {"status": "installed", **result}
    except PromptProviderInstallError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/prompt-providers")
def list_prompt_providers(
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
) -> dict:
    """List installed custom prompt providers."""
    from app.services.prompt_provider_installer import list_custom_providers

    providers = list_custom_providers()
    return {"providers": providers, "count": len(providers)}


@router.delete("/prompt-providers/{name}")
def uninstall_prompt_provider_endpoint(
    name: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
) -> dict:
    """Uninstall a custom prompt provider."""
    from app.services.prompt_provider_installer import uninstall_prompt_provider

    if uninstall_prompt_provider(name):
        return {"status": "uninstalled", "package_name": name}
    raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")


# =============================================================================
# MQTT helpers
# =============================================================================


def _publish_package_install_mqtt(node_id: str, request: PackageInstallRequest) -> None:
    """Publish MQTT message to tell node to install a package."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, node %s cannot receive install request", node_id)
        return

    topic = f"jarvis/nodes/{node_id}/package-install"
    payload = json.dumps({
        "request_id": request.id,
        "command_name": request.command_name,
        "github_repo_url": request.github_repo_url,
        "git_tag": request.git_tag,
    })

    try:
        client.publish(topic, payload)
        logger.info("Published package install request to %s", topic)
    except Exception as e:
        logger.error("Failed to publish package install MQTT: %s", e)
