import os
import sys
import logging
import time
from dataclasses import dataclass

import httpx
from fastapi import Header, HTTPException, Depends
from jose import ExpiredSignatureError, JWTError, jwt
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
JARVIS_AUTH_SECRET_KEY = os.getenv("JARVIS_AUTH_SECRET_KEY", "")
JARVIS_AUTH_ALGORITHM = os.getenv("JARVIS_AUTH_ALGORITHM", "HS256")


def _get_auth_base_url() -> str:
    """Get auth service URL from service discovery or fallback to env var."""
    try:
        from app.core import service_config
        if service_config.is_initialized():
            return service_config.get_auth_url()
    except (ImportError, AttributeError):
        pass
    except Exception as e:
        logger.warning("Service discovery failed for auth, falling back to env var: %s", e)
    # Fallback to env var
    return os.getenv("JARVIS_AUTH_BASE_URL", "http://localhost:7701")

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

    Uses the ``llm.interface`` database setting to determine which model to use.

    Available models:
    - JarvisToolModel: Tool-based model using JSON tool calls
    - JarvisAdapterModel: Slim prompt model for adapter-tuned usage
    - Custom models can be added to app/core/models/custom/
    - Prompt providers can be added to app/core/prompt_providers/
    """
    try:
        return ModelService()
    except (ValueError, AttributeError) as e:
        logger.error("Invalid LLM interface configuration: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Invalid LLM Interface: {e}",
        ) from e


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


# =============================================================================
# User JWT Authentication (for mobile app)
# =============================================================================


@dataclass
class AuthenticatedUser:
    """User info extracted from a validated JWT."""
    user_id: int
    email: str | None = None


def verify_user_jwt(authorization: str | None = Header(None)) -> AuthenticatedUser:
    """Validate a Bearer JWT and return the user info.

    Does NOT require superuser — any valid user token is accepted.
    Use verify_household_role() afterwards to check permission level.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not JARVIS_AUTH_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="JARVIS_AUTH_SECRET_KEY not configured",
        )

    token = authorization[7:]
    try:
        payload = jwt.decode(token, JARVIS_AUTH_SECRET_KEY, algorithms=[JARVIS_AUTH_ALGORITHM])
    except ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token has expired") from exc
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing user ID")

    return AuthenticatedUser(user_id=int(user_id), email=payload.get("email"))


def verify_household_role(
    user_id: int,
    household_id: str,
    required_role: str = "power_user",
) -> None:
    """Check that user has the required role in a household via jarvis-auth.

    Raises HTTPException(403) if the user lacks permission.
    """
    if not JARVIS_APP_KEY:
        logger.warning("JARVIS_APP_KEY not set — skipping household role check")
        return

    auth_url = _get_auth_base_url().rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{auth_url}/internal/validate-household-access",
                headers={
                    "X-Jarvis-App-Id": JARVIS_APP_ID,
                    "X-Jarvis-App-Key": JARVIS_APP_KEY,
                },
                json={
                    "user_id": user_id,
                    "household_id": household_id,
                    "required_role": required_role,
                },
            )
    except httpx.RequestError as exc:
        logger.error("Failed to reach jarvis-auth for role check: %s", exc)
        raise HTTPException(status_code=502, detail="Auth service unavailable") from exc

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Auth service error")

    result = resp.json()
    if not result.get("valid"):
        raise HTTPException(
            status_code=403,
            detail=result.get("reason", "Insufficient permissions"),
        )

