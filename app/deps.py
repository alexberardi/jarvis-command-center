import os
import sys
import logging
import time

import httpx
from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from app.context_providers.node_context_provider import NodeContextProvider
from app.db import get_session_local
from app.models import Node
from app.core.model_service import ModelService
from dotenv import load_dotenv

# Add the root project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

logger = logging.getLogger("uvicorn")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
JARVIS_APP_ID = os.getenv("JARVIS_APP_ID", "command-center")
JARVIS_APP_KEY = os.getenv("JARVIS_APP_KEY")


def _get_auth_base_url() -> str:
    """Get auth service URL from service discovery or fallback to env var."""
    try:
        from app.core import service_config
        if service_config.is_initialized():
            return service_config.get_auth_url()
    except (ImportError, AttributeError):
        pass
    # Fallback to env var
    return os.getenv("JARVIS_AUTH_BASE_URL", "http://localhost:8007")

# Cache for node validation results
_node_validation_cache: dict[str, tuple[dict, float]] = {}
NODE_AUTH_CACHE_TTL = int(os.getenv("NODE_AUTH_CACHE_TTL", "60"))


def get_db():
    """Get database session with dynamic configuration"""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_cached_validation(api_key: str) -> dict | None:
    """Get cached validation result if not expired."""
    if api_key in _node_validation_cache:
        result, timestamp = _node_validation_cache[api_key]
        if time.time() - timestamp < NODE_AUTH_CACHE_TTL:
            return result
        del _node_validation_cache[api_key]
    return None


def _cache_validation(api_key: str, result: dict) -> None:
    """Cache a validation result."""
    _node_validation_cache[api_key] = (result, time.time())


def _validate_node_with_auth_service(node_id: str, node_key: str) -> dict:
    """Validate node credentials with jarvis-auth service (synchronous)."""
    if not JARVIS_APP_KEY:
        logger.error("JARVIS_APP_KEY not configured for node validation")
        return {"valid": False, "reason": "Auth not configured"}

    validate_url = _get_auth_base_url().rstrip("/") + "/internal/validate-node"

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                validate_url,
                headers={
                    "X-Jarvis-App-Id": JARVIS_APP_ID,
                    "X-Jarvis-App-Key": JARVIS_APP_KEY,
                },
                json={
                    "node_id": node_id,
                    "node_key": node_key,
                    "service_id": JARVIS_APP_ID,
                },
            )
    except httpx.RequestError as exc:
        logger.error("Failed to reach jarvis-auth: %s", exc)
        return {"valid": False, "reason": f"Auth service unavailable: {exc}"}

    if resp.status_code != 200:
        logger.error("jarvis-auth returned %d", resp.status_code)
        return {"valid": False, "reason": f"Auth service error: {resp.status_code}"}

    return resp.json()


def verify_api_key(x_api_key: str = Header(...), db: Session = Depends(get_db)):
    """
    Verify node API key against jarvis-auth service.

    The x_api_key header should be in format "node_id:node_key" for centralized auth,
    or just the legacy api_key for backwards compatibility with local nodes table.
    """
    # Check cache first
    cached = _get_cached_validation(x_api_key)
    if cached is not None:
        if not cached.get("valid"):
            raise HTTPException(status_code=401, detail=cached.get("reason", "Invalid API Key"))
        # Get node from local DB for additional context
        node = db.query(Node).filter(Node.node_id == cached.get("node_id")).first()
        if node:
            return NodeContextProvider(
                node,
                household_id=cached.get("household_id"),
                household_member_ids=cached.get("household_member_ids"),
            )

    # Try centralized auth first (format: "node_id:node_key")
    if ":" in x_api_key:
        node_id, node_key = x_api_key.split(":", 1)
        result = _validate_node_with_auth_service(node_id, node_key)
        _cache_validation(x_api_key, result)

        if result.get("valid"):
            # Get node from local DB for additional context (room, voice_mode, etc.)
            node = db.query(Node).filter(Node.node_id == node_id).first()
            if node:
                return NodeContextProvider(
                    node,
                    household_id=result.get("household_id"),
                    household_member_ids=result.get("household_member_ids"),
                )
            else:
                # Node validated by jarvis-auth but not in local DB - create minimal context
                logger.warning("Node %s validated but not in local DB", node_id)
                raise HTTPException(status_code=401, detail="Node not configured locally")
        else:
            logger.warning("Node auth failed for %s: %s", node_id, result.get("reason"))
            raise HTTPException(status_code=401, detail=result.get("reason", "Invalid API Key"))

    # Fallback to legacy local DB lookup (backwards compatibility)
    node = db.query(Node).filter(Node.api_key == x_api_key).first()
    if not node:
        logger.warning("Unauthorized attempt with API key: %s", x_api_key[:8] + "...")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return NodeContextProvider(node)

def verify_admin_key(x_api_key: str = Header(...)):
    if x_api_key != ADMIN_API_KEY:
        logger.warning("Unauthorized admin access attempt with API key: %s", x_api_key)
        raise HTTPException(status_code=401, detail="Invalid Admin API Key")


def get_model_service() -> ModelService:
    """
    Get the configured model service instance.

    Uses JARVIS_MODEL_INTERFACE environment variable to determine which model to use.
    Defaults to BASE_MODEL if not specified.

    Available models:
    - JarvisToolModel: Tool-based model using JSON tool calls
    - JarvisAdapterModel: Slim prompt model for adapter-tuned usage
    - Custom models can be added to app/core/models/custom/
    """
    return ModelService()


async def require_app_auth(
    x_jarvis_app_id: str = Header(None),
    x_jarvis_app_key: str = Header(None),
):
    """
    Validate app-to-app authentication by calling jarvis-auth /internal/app-ping.

    Headers required:
    - X-Jarvis-App-Id: App identifier
    - X-Jarvis-App-Key: App secret key
    """
    if not x_jarvis_app_id or not x_jarvis_app_key:
        raise HTTPException(status_code=401, detail="Missing app credentials")

    jarvis_auth_base = _get_auth_base_url()
    app_ping_url = jarvis_auth_base.rstrip("/") + "/internal/app-ping"

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(
                app_ping_url,
                headers={
                    "X-Jarvis-App-Id": x_jarvis_app_id,
                    "X-Jarvis-App-Key": x_jarvis_app_key,
                },
            )
        except httpx.RequestError as exc:
            logger.error("Failed to reach jarvis-auth for app auth: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"Auth service unavailable: {exc}",
            ) from exc

    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid app credentials")
    elif resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="App auth failed")

    # Auth succeeded
    return None

