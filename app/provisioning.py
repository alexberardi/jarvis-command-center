"""Provisioning token endpoints for node self-registration.

Flow:
1. Authenticated user (admin key or JWT) requests POST /api/v0/provisioning/token
2. CC generates a GUID (node_id) + short-lived token, stores SHA-256 hash
3. Mobile app passes node_id + raw token to the physical node
4. Node calls POST /api/v0/nodes/register with node_id + raw token
5. CC validates token, registers node with jarvis-auth, creates local record
"""
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.admin import NodeCreateResponse, _register_node_with_auth
from app.deps import ADMIN_API_KEY, _get_auth_base_url, get_db
from app.models import Node, ProvisioningToken

router = APIRouter()
logger = logging.getLogger("uvicorn")

TOKEN_TTL_SECONDS = 600  # 10 minutes


# ============================================================
# Request/Response Models
# ============================================================


class ProvisioningTokenRequest(BaseModel):
    household_id: str
    room: str | None = None
    name: str | None = None
    node_id: str | None = None  # Omit for new, provide for refresh


class ProvisioningTokenResponse(BaseModel):
    token: str
    node_id: str
    expires_at: datetime
    expires_in: int


class NodeRegisterRequest(BaseModel):
    node_id: str
    provisioning_token: str
    room: str | None = None


# ============================================================
# Auth helpers
# ============================================================


@dataclass
class ProvisioningAuthContext:
    auth_type: str  # "admin_key" or "jwt"
    user_id: int | None = None


def _validate_jwt(token: str) -> dict | None:
    """Validate a JWT by calling jarvis-auth GET /auth/me.

    Returns the user dict on success, None on failure.
    """
    auth_url = _get_auth_base_url().rstrip("/") + "/auth/me"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(auth_url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code == 200:
            return resp.json()
    except httpx.RequestError as exc:
        logger.error("Failed to reach jarvis-auth for JWT validation: %s", exc)
    return None


def verify_provisioning_auth(
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> ProvisioningAuthContext:
    """Authenticate via admin API key or JWT Bearer token."""
    # Try admin API key first
    if x_api_key:
        if ADMIN_API_KEY and x_api_key == ADMIN_API_KEY:
            return ProvisioningAuthContext(auth_type="admin_key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Try JWT Bearer token
    if authorization and authorization.startswith("Bearer "):
        jwt_token = authorization[7:]
        user_info = _validate_jwt(jwt_token)
        if user_info:
            return ProvisioningAuthContext(
                auth_type="jwt",
                user_id=user_info.get("id"),
            )
        raise HTTPException(status_code=401, detail="Invalid or expired JWT")

    raise HTTPException(status_code=401, detail="Authentication required")


# ============================================================
# Token helpers
# ============================================================


def _hash_token(raw_token: str) -> str:
    """SHA-256 hash a raw token."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


def _validate_provisioning_token(
    db: Session, node_id: str, raw_token: str
) -> ProvisioningToken | None:
    """Look up a valid (unconsumed, unexpired) provisioning token."""
    token_hash = _hash_token(raw_token)
    return (
        db.query(ProvisioningToken)
        .filter(
            ProvisioningToken.node_id == node_id,
            ProvisioningToken.token_hash == token_hash,
            ProvisioningToken.consumed_at.is_(None),
            ProvisioningToken.expires_at > datetime.utcnow(),
        )
        .first()
    )


# ============================================================
# Endpoints
# ============================================================


@router.post("/provisioning/token", status_code=201, response_model=ProvisioningTokenResponse)
def create_provisioning_token(
    body: ProvisioningTokenRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> ProvisioningTokenResponse:
    """Generate a provisioning token for a new or existing (unconsumed) node."""
    node_id = body.node_id

    if node_id:
        # Refresh: reuse GUID but reject if node already registered
        existing_node = db.query(Node).filter(Node.node_id == node_id).first()
        if existing_node:
            raise HTTPException(status_code=400, detail="Node already registered")
        # Invalidate any previous unconsumed tokens for this node_id
        db.query(ProvisioningToken).filter(
            ProvisioningToken.node_id == node_id,
            ProvisioningToken.consumed_at.is_(None),
        ).delete(synchronize_session=False)
    else:
        node_id = str(uuid4())

    raw_token = f"prov_{secrets.token_urlsafe(32)}"
    token_hash = _hash_token(raw_token)
    expires_at = datetime.utcnow() + timedelta(seconds=TOKEN_TTL_SECONDS)

    prov = ProvisioningToken(
        token_hash=token_hash,
        node_id=node_id,
        household_id=body.household_id,
        room=body.room,
        name=body.name,
        created_by_user_id=auth.user_id,
        expires_at=expires_at,
    )
    db.add(prov)
    db.commit()

    logger.info("Provisioning token created for node_id=%s by %s", node_id, auth.auth_type)

    return ProvisioningTokenResponse(
        token=raw_token,
        node_id=node_id,
        expires_at=expires_at,
        expires_in=TOKEN_TTL_SECONDS,
    )


@router.post("/nodes/register", status_code=201, response_model=NodeCreateResponse)
def register_node(
    body: NodeRegisterRequest,
    db: Session = Depends(get_db),
) -> NodeCreateResponse:
    """Self-register a node using a provisioning token."""
    token_record = _validate_provisioning_token(db, body.node_id, body.provisioning_token)
    if not token_record:
        raise HTTPException(status_code=401, detail="Invalid or expired provisioning token")

    # Reject if node already exists
    existing = db.query(Node).filter(Node.node_id == body.node_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Node already registered")

    # Register with jarvis-auth
    name = token_record.name or body.node_id
    node_key = _register_node_with_auth(body.node_id, token_record.household_id, name)

    # Determine room: request body > token > "default"
    room = body.room or token_record.room or "default"

    # Create local node record
    db_node = Node(
        node_id=body.node_id,
        api_key=node_key,
        room=room,
        user="default",
        voice_mode="brief",
        household_id=token_record.household_id,
    )
    db.add(db_node)

    # Consume the token
    token_record.consumed_at = datetime.utcnow()
    db.commit()
    db.refresh(db_node)

    logger.info("Node registered via provisioning: %s in room %s", db_node.node_id, room)

    return NodeCreateResponse(
        node_id=db_node.node_id,
        room=db_node.room,
        user=db_node.user,
        voice_mode=db_node.voice_mode,
        node_key=node_key,
    )


# ============================================================
# Cleanup
# ============================================================


def cleanup_expired_tokens(db: Session) -> int:
    """Delete expired and old consumed provisioning tokens.

    Removes tokens that are:
    - Expired (expires_at < 24 hours ago), OR
    - Consumed more than 24 hours ago
    """
    cutoff = datetime.utcnow() - timedelta(hours=24)
    count = db.query(ProvisioningToken).filter(
        (ProvisioningToken.expires_at < cutoff)
        | (
            ProvisioningToken.consumed_at.isnot(None)
            & (ProvisioningToken.consumed_at < cutoff)
        )
    ).delete(synchronize_session=False)
    db.commit()
    return count
