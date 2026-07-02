"""Package install API: mobile -> CC -> MQTT -> node -> CC -> mobile.

Follows the same request/poll pattern as device-scan (smart_home.py).
Also handles prompt provider installs (CC-local, async via BackgroundTasks).
"""

import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_session_local
from app.deps import get_db, verify_api_key
from app.models import Node, PackageInstallRequest, PromptProviderInstallRequest
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
    restarting: bool = False  # node is about to restart to load the change — not terminal


class PackageInstallPollResponse(BaseModel):
    status: str
    request_id: str
    command_name: str
    error_message: str | None = None
    details: dict | None = None


class PackageInstallVerifyResponse(BaseModel):
    confirmed: bool
    command_name: str
    github_repo_url: str
    git_tag: str | None = None


# =============================================================================
# Helpers
# =============================================================================

# Extra time granted when a node reports it is restarting, so a slow boot
# doesn't expire the request mid-restart.
RESTART_EXPIRY_EXTENSION_SECONDS = 120


def _mark_restarting(
    request: PackageInstallRequest,
    body: PackageInstallResultUpload,
    now: datetime,
) -> None:
    """Mark a request as restarting (non-terminal) and extend its expiry.

    The node POSTs {"success": true, "restarting": true} right before it
    restarts to load the change. The real result arrives after boot via the
    deferred-result flush, which overwrites this with completed/failed.
    """
    request.status = "restarting"
    request.expires_at = (request.expires_at or now) + timedelta(
        seconds=RESTART_EXPIRY_EXTENSION_SECONDS
    )
    if body.details:
        request.results_json = json.dumps(body.details)


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


@router.get(
    "/nodes/{node_id}/package-install/{request_id}/verify",
    response_model=PackageInstallVerifyResponse,
)
def verify_package_install(
    node_id: str,
    request_id: str,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> PackageInstallVerifyResponse:
    """Node calls this to verify a package install request and get the repo URL.

    This is the zero-trust gate (mirrors verify_test_install): a spoofed MQTT
    nudge with a fake request_id fails here with 404 because no matching row
    exists for this node. The node installs from the URL returned here — never
    from the (untrusted) URL in the MQTT payload.
    """
    install_request = db.query(PackageInstallRequest).filter(
        PackageInstallRequest.id == request_id,
        PackageInstallRequest.node_id == node_id,
    ).first()
    if not install_request:
        raise HTTPException(status_code=404, detail="Package install request not found")

    now = datetime.utcnow()
    if install_request.expires_at and install_request.expires_at < now:
        install_request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Package install request expired")

    if install_request.status != "pending":
        raise HTTPException(status_code=409, detail=f"Request already {install_request.status}")

    return PackageInstallVerifyResponse(
        confirmed=True,
        command_name=install_request.command_name,
        github_repo_url=install_request.github_repo_url,
        git_tag=install_request.git_tag,
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

    if body.restarting:
        # Non-terminal: node is restarting to load the package. The deferred
        # result POST after boot overwrites this with completed/failed.
        _mark_restarting(install_request, body, now)
        db.commit()
        logger.info(
            "Package install restarting: request=%s command=%s",
            request_id[:8], install_request.command_name,
        )
        return {"status": "ok"}

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

    # Check expiration (restarting is non-terminal, treated like pending)
    if (
        install_request.status in ("pending", "restarting")
        and install_request.expires_at
        and install_request.expires_at < now
    ):
        install_request.status = "expired"
        db.commit()

    if install_request.status == "expired":
        return PackageInstallPollResponse(
            status="expired",
            request_id=request_id,
            command_name=install_request.command_name,
            error_message="Install request expired — node may be offline",
        )

    if install_request.status in ("pending", "restarting"):
        return PackageInstallPollResponse(
            status=install_request.status,
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
# Package Uninstall
# =============================================================================


class PackageUninstallBody(BaseModel):
    command_name: str
    component_type: str


@router.post(
    "/nodes/{node_id}/package-uninstall",
    response_model=PackageInstallResponse,
    status_code=201,
)
def request_package_uninstall(
    node_id: str,
    body: PackageUninstallBody,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> PackageInstallResponse:
    """Request a package uninstall on a node. Mobile calls this, CC notifies node via MQTT."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    now = datetime.utcnow()
    uninstall_request = PackageInstallRequest(
        id=str(uuid4()),
        node_id=node_id,
        household_id=node.household_id or "",
        command_name=body.command_name,
        github_repo_url="",  # not needed for uninstall
        status="pending",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    db.add(uninstall_request)
    db.commit()
    db.refresh(uninstall_request)

    logger.info(
        "Package uninstall requested: node=%s command=%s request_id=%s",
        node_id, body.command_name, uninstall_request.id[:8],
    )

    _publish_package_uninstall_mqtt(node_id, uninstall_request, component_type=body.component_type)

    return PackageInstallResponse(
        id=uninstall_request.id,
        status=uninstall_request.status,
        created_at=uninstall_request.created_at,
    )


@router.post("/nodes/{node_id}/package-uninstall/{request_id}/results")
def upload_package_uninstall_results(
    node_id: str,
    request_id: str,
    body: PackageInstallResultUpload,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Node uploads uninstall results."""
    request = db.query(PackageInstallRequest).filter(
        PackageInstallRequest.id == request_id,
        PackageInstallRequest.node_id == node_id,
    ).first()
    if not request:
        raise HTTPException(status_code=404, detail="Uninstall request not found")

    now = datetime.utcnow()
    if request.expires_at and request.expires_at < now:
        request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Uninstall request expired")

    if body.restarting:
        _mark_restarting(request, body, now)
        db.commit()
        logger.info(
            "Package uninstall restarting: request=%s command=%s",
            request_id[:8], request.command_name,
        )
        return {"status": "ok"}

    if body.success:
        request.status = "completed"
        if body.details:
            request.results_json = json.dumps(body.details)
    else:
        request.status = "failed"
        request.error_message = body.error or "Unknown error"

    request.completed_at = now
    db.commit()

    logger.info(
        "Package uninstall results uploaded: request=%s status=%s command=%s",
        request_id[:8], request.status, request.command_name,
    )
    return {"status": "ok"}


@router.get(
    "/nodes/{node_id}/package-uninstall/{request_id}",
    response_model=PackageInstallPollResponse,
)
def poll_package_uninstall(
    node_id: str,
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> PackageInstallPollResponse:
    """Mobile polls for uninstall results."""
    request = db.query(PackageInstallRequest).filter(
        PackageInstallRequest.id == request_id,
        PackageInstallRequest.node_id == node_id,
    ).first()
    if not request:
        raise HTTPException(status_code=404, detail="Uninstall request not found")

    now = datetime.utcnow()
    if (
        request.status in ("pending", "restarting")
        and request.expires_at
        and request.expires_at < now
    ):
        request.status = "expired"
        db.commit()

    return PackageInstallPollResponse(
        status=request.status,
        request_id=request_id,
        command_name=request.command_name,
        error_message=request.error_message if request.status in ("failed", "expired") else None,
        details=json.loads(request.results_json) if request.results_json else None,
    )


def _publish_package_uninstall_mqtt(
    node_id: str,
    request: PackageInstallRequest,
    component_type: str | None = None,
) -> None:
    """Publish MQTT message to tell node to uninstall a package."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, node %s cannot receive uninstall request", node_id)
        return

    topic = f"jarvis/nodes/{node_id}/package-uninstall"
    mqtt_payload: dict[str, str] = {
        "request_id": request.id,
        "command_name": request.command_name,
    }
    if component_type:
        mqtt_payload["component_type"] = component_type
    payload = json.dumps(mqtt_payload)

    try:
        client.publish(topic, payload)
        logger.info("Published package uninstall request to %s", topic)
    except Exception as e:
        logger.error("Failed to publish package uninstall MQTT: %s", e)


# =============================================================================
# Package Revert
# =============================================================================


class PackageRevertBody(BaseModel):
    """Body for a revert request. Accepts either field name for the package."""

    command_name: str | None = None
    package_name: str | None = None


@router.post(
    "/nodes/{node_id}/package-revert",
    response_model=PackageInstallResponse,
    status_code=201,
)
def request_package_revert(
    node_id: str,
    body: PackageRevertBody,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> PackageInstallResponse:
    """Request a package revert (restore .previous) on a node. Mobile calls this, CC notifies node via MQTT."""
    name = body.command_name or body.package_name
    if not name:
        raise HTTPException(status_code=422, detail="command_name or package_name is required")

    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    now = datetime.utcnow()
    revert_request = PackageInstallRequest(
        id=str(uuid4()),
        node_id=node_id,
        household_id=node.household_id or "",
        command_name=name,
        github_repo_url="",  # not needed for revert
        status="pending",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    db.add(revert_request)
    db.commit()
    db.refresh(revert_request)

    logger.info(
        "Package revert requested: node=%s command=%s request_id=%s",
        node_id, name, revert_request.id[:8],
    )

    _publish_package_revert_mqtt(node_id, revert_request)

    return PackageInstallResponse(
        id=revert_request.id,
        status=revert_request.status,
        created_at=revert_request.created_at,
    )


@router.post("/nodes/{node_id}/package-revert/{request_id}/results")
def upload_package_revert_results(
    node_id: str,
    request_id: str,
    body: PackageInstallResultUpload,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Node uploads revert results."""
    request = db.query(PackageInstallRequest).filter(
        PackageInstallRequest.id == request_id,
        PackageInstallRequest.node_id == node_id,
    ).first()
    if not request:
        raise HTTPException(status_code=404, detail="Revert request not found")

    now = datetime.utcnow()
    if request.expires_at and request.expires_at < now:
        request.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Revert request expired")

    if body.restarting:
        _mark_restarting(request, body, now)
        db.commit()
        logger.info(
            "Package revert restarting: request=%s command=%s",
            request_id[:8], request.command_name,
        )
        return {"status": "ok"}

    if body.success:
        request.status = "completed"
        if body.details:
            request.results_json = json.dumps(body.details)
    else:
        request.status = "failed"
        request.error_message = body.error or "Unknown error"

    request.completed_at = now
    db.commit()

    logger.info(
        "Package revert results uploaded: request=%s status=%s command=%s",
        request_id[:8], request.status, request.command_name,
    )
    return {"status": "ok"}


@router.get(
    "/nodes/{node_id}/package-revert/{request_id}",
    response_model=PackageInstallPollResponse,
)
def poll_package_revert(
    node_id: str,
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> PackageInstallPollResponse:
    """Mobile polls for revert results."""
    request = db.query(PackageInstallRequest).filter(
        PackageInstallRequest.id == request_id,
        PackageInstallRequest.node_id == node_id,
    ).first()
    if not request:
        raise HTTPException(status_code=404, detail="Revert request not found")

    now = datetime.utcnow()
    if (
        request.status in ("pending", "restarting")
        and request.expires_at
        and request.expires_at < now
    ):
        request.status = "expired"
        db.commit()

    return PackageInstallPollResponse(
        status=request.status,
        request_id=request_id,
        command_name=request.command_name,
        error_message=request.error_message if request.status in ("failed", "expired") else None,
        details=json.loads(request.results_json) if request.results_json else None,
    )


def _publish_package_revert_mqtt(
    node_id: str,
    request: PackageInstallRequest,
) -> None:
    """Publish MQTT message to tell node to revert a package to its .previous version."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, node %s cannot receive revert request", node_id)
        return

    topic = f"jarvis/nodes/{node_id}/package-revert"
    payload = json.dumps({
        "request_id": request.id,
        "command_name": request.command_name,
        "package_name": request.command_name,
    })

    try:
        client.publish(topic, payload)
        logger.info("Published package revert request to %s", topic)
    except Exception as e:
        logger.error("Failed to publish package revert MQTT: %s", e)


# =============================================================================
# Prompt Provider Install (CC-local, async via BackgroundTasks)
# =============================================================================


class PromptProviderInstallBody(BaseModel):
    github_repo_url: str
    git_tag: str | None = None


class PromptProviderInstallPollResponse(BaseModel):
    status: str
    request_id: str
    package_name: str | None = None
    provider_name: str | None = None
    error_message: str | None = None
    details: dict | None = None


def _run_prompt_provider_install(request_id: str, repo_url: str, git_tag: str | None) -> None:
    """Background task: clone, validate, install prompt provider, update DB record."""
    from app.services.prompt_provider_installer import (
        install_prompt_provider,
        PromptProviderInstallError,
    )

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        result = install_prompt_provider(repo_url, git_tag)

        request = db.query(PromptProviderInstallRequest).get(request_id)
        if request and request.status == "pending":
            request.status = "completed"
            request.package_name = result.get("package_name")
            request.results_json = json.dumps(result)
            request.completed_at = datetime.utcnow()
            db.commit()
            logger.info("Prompt provider installed: %s", result.get("provider_name"))

    except (PromptProviderInstallError, Exception) as e:
        request = db.query(PromptProviderInstallRequest).get(request_id)
        if request and request.status == "pending":
            request.status = "failed"
            request.error_message = str(e)
            request.completed_at = datetime.utcnow()
            db.commit()
        logger.error("Prompt provider install failed: %s", e)
    finally:
        db.close()


@router.post("/prompt-providers/install", status_code=201)
def install_prompt_provider_endpoint(
    body: PromptProviderInstallBody,
    background_tasks: BackgroundTasks,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    """Start async prompt provider install. Returns request_id for polling."""
    request = PromptProviderInstallRequest(
        id=str(uuid4()),
        household_id=auth.household_id,
        github_repo_url=body.github_repo_url,
        git_tag=body.git_tag,
        status="pending",
        expires_at=datetime.utcnow() + timedelta(minutes=5),
    )
    db.add(request)
    db.commit()

    background_tasks.add_task(
        _run_prompt_provider_install,
        request.id, body.github_repo_url, body.git_tag,
    )

    logger.info("Prompt provider install queued: request=%s repo=%s", request.id[:8], body.github_repo_url)
    return {"id": request.id, "status": "pending", "created_at": request.created_at.isoformat()}


@router.get(
    "/prompt-providers/install/{request_id}",
    response_model=PromptProviderInstallPollResponse,
)
def poll_prompt_provider_install(
    request_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> PromptProviderInstallPollResponse:
    """Poll for prompt provider install status."""
    request = db.query(PromptProviderInstallRequest).filter(
        PromptProviderInstallRequest.id == request_id,
        PromptProviderInstallRequest.household_id == auth.household_id,
    ).first()
    if not request:
        raise HTTPException(status_code=404, detail="Install request not found")

    now = datetime.utcnow()

    if request.status == "pending" and request.expires_at and request.expires_at < now:
        request.status = "expired"
        db.commit()

    if request.status == "expired":
        return PromptProviderInstallPollResponse(
            status="expired",
            request_id=request_id,
            error_message="Install request expired",
        )

    if request.status == "pending":
        return PromptProviderInstallPollResponse(
            status="pending",
            request_id=request_id,
        )

    if request.status == "failed":
        return PromptProviderInstallPollResponse(
            status="failed",
            request_id=request_id,
            package_name=request.package_name,
            error_message=request.error_message,
        )

    # Completed
    details = None
    provider_name = None
    if request.results_json:
        details = json.loads(request.results_json)
        provider_name = details.get("provider_name")

    return PromptProviderInstallPollResponse(
        status="completed",
        request_id=request_id,
        package_name=request.package_name,
        provider_name=provider_name,
        details=details,
    )


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
