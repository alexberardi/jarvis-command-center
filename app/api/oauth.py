"""OAuth session management: JCC as redirect authority.

Two redirect modes:

1. **Direct (local providers like HA):** Provider redirects to JCC's
   ``/oauth/callback``, JCC exchanges the code, stores tokens, redirects to
   ``jarvis://auth-complete``.

2. **Relay bounce (external providers like Google):** Provider redirects to the
   cloud relay's ``/oauth/bounce``, which 302s to ``jarvis://auth-complete``
   with the auth code.  Mobile extracts the code and POSTs it to JCC's
   ``/oauth/sessions/{id}/exchange``.  JCC exchanges the code for tokens and
   stores them.  This avoids requiring JCC to be internet-accessible.

The mode is chosen automatically: if the provider supplies a full
``authorize_url`` (external) AND ``JARVIS_RELAY_URL`` is set, the relay path is
used.  Otherwise JCC handles the callback directly.
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.deps import get_db, require_app_auth, verify_api_key
from app.models import AuthSession, Node
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext

router = APIRouter()
logger = logging.getLogger("uvicorn")

SESSION_TTL_MINUTES = 10


# =============================================================================
# Request/Response Models
# =============================================================================


class AuthConfigPayload(BaseModel):
    """AuthenticationConfig subset sent by mobile when creating a session."""
    type: str = "oauth"
    provider: str
    client_id: str
    keys: list[str]
    authorize_url: str | None = None
    exchange_url: str | None = None
    authorize_path: str | None = None
    exchange_path: str | None = None
    scopes: list[str] = []
    extra_authorize_params: dict[str, str] = {}
    extra_exchange_params: dict[str, str] = {}
    send_redirect_uri_in_exchange: bool = True
    client_secret: str | None = None
    supports_pkce: bool = False
    native_redirect_uri: str | None = None


class CreateAuthSessionRequest(BaseModel):
    provider: str
    node_id: str
    provider_base_url: str | None = None
    auth_config: AuthConfigPayload


class CreateAuthSessionResponse(BaseModel):
    session_id: str
    authorize_url: str
    requires_code_exchange: bool = False  # True = mobile must POST code to /exchange


class AuthSessionStatusResponse(BaseModel):
    session_id: str
    status: str
    provider: str


class ProviderCredentialsResponse(BaseModel):
    access_token: str | None = None
    refresh_token: str | None = None
    token_data: dict | None = None
    base_url: str | None = None
    # The user who initiated the OAuth flow — owner of user-scoped tokens.
    # None for admin-key flows and sessions created before this field existed;
    # nodes fall back to legacy (no user context) storage in that case.
    user_id: int | None = None


# =============================================================================
# Token Encryption Helpers (AES-256-GCM)
# =============================================================================


def _get_encryption_key() -> bytes:
    """Get or derive the 32-byte encryption key for token storage."""
    key_hex = os.getenv("JARVIS_TOKEN_ENCRYPTION_KEY")
    if key_hex:
        key_bytes = bytes.fromhex(key_hex)
        if len(key_bytes) != 32:
            raise ValueError("JARVIS_TOKEN_ENCRYPTION_KEY must be 32 bytes (64 hex chars)")
        return key_bytes

    # Derive from SECRET_KEY if no dedicated key is set
    secret_key = os.getenv("SECRET_KEY") or os.getenv("JARVIS_AUTH_SECRET_KEY")
    if not secret_key:
        raise ValueError("JARVIS_TOKEN_ENCRYPTION_KEY, SECRET_KEY, or JARVIS_AUTH_SECRET_KEY must be set for token encryption")
    return hashlib.sha256(secret_key.encode()).digest()


def _encrypt_value(plaintext: str) -> str:
    """Encrypt a string value, returning base64url(nonce + ciphertext)."""
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.urlsafe_b64encode(nonce + ct).decode()


def _decrypt_value(encrypted: str) -> str:
    """Decrypt a base64url(nonce + ciphertext) value."""
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    raw = base64.urlsafe_b64decode(encrypted)
    nonce, ct = raw[:12], raw[12:]
    return aesgcm.decrypt(nonce, ct, None).decode()


# =============================================================================
# PKCE Helpers
# =============================================================================


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)[:128]
    challenge_bytes = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode("ascii")
    return verifier, challenge


# =============================================================================
# Relay State Encoding
# =============================================================================


def _encode_relay_state(csrf_token: str, redirect_uri: str) -> str:
    """Encode CSRF token + redirect URI into the relay's expected state format.

    Format: base64url({"t": "<csrf>", "r": "<redirect_uri>"})
    """
    payload = json.dumps({"t": csrf_token, "r": redirect_uri})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _get_relay_url() -> str | None:
    """Get the relay base URL for OAuth bounce, if configured.

    Reads from the ``oauth.relay_url`` setting (DB → JARVIS_RELAY_URL env
    fallback).  Returns *None* when unset so callers fall back to the direct
    callback flow.
    """
    from app.services.settings_service import get_settings_service

    url: str = get_settings_service().get("oauth.relay_url", "")
    return url or None


# =============================================================================
# External URL Resolution
# =============================================================================


def _get_external_url(request: Request) -> str:
    """Get the external base URL for building callback URIs.

    Reads from the ``oauth.external_url`` setting (DB → JARVIS_EXTERNAL_URL
    env fallback), then falls back to request headers.
    """
    from app.services.settings_service import get_settings_service

    external_url: str = get_settings_service().get("oauth.external_url", "")
    if external_url:
        return external_url.rstrip("/")

    # Auto-detect from request (works behind Caddy reverse proxy)
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    return f"{scheme}://{host}"


# =============================================================================
# MQTT Notification
# =============================================================================


def _publish_auth_ready(
    provider: str, node_id: str, session_id: str, user_id: int | None = None
) -> None:
    """Publish MQTT message: jarvis/auth/{provider}/ready."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT not available, cannot publish auth ready for %s", provider)
        return

    topic = f"jarvis/auth/{provider}/ready"
    payload = json.dumps({
        "provider": provider,
        "node_id": node_id,
        "session_id": session_id,
        # Nullable — owner of user-scoped tokens (the user who ran the flow).
        "user_id": user_id,
    })

    try:
        client.publish(topic, payload)
        logger.info("Published auth ready notification to %s", topic)
    except Exception as e:
        logger.error("Failed to publish auth ready MQTT: %s", e)


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/oauth/sessions", response_model=CreateAuthSessionResponse, status_code=201)
def create_auth_session(
    body: CreateAuthSessionRequest,
    request: Request,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> CreateAuthSessionResponse:
    """Create an OAuth auth session. Mobile calls this to start the flow.

    Returns the session_id and the authorize_url for the mobile to open.
    """
    cfg = body.auth_config
    node = db.query(Node).filter(Node.node_id == body.node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    # Generate state (CSRF) token
    state = secrets.token_urlsafe(32)

    # Generate PKCE if supported
    code_verifier = None
    code_challenge = None
    if cfg.supports_pkce:
        code_verifier, code_challenge = _generate_pkce()

    # Build authorize URL
    base_url = body.provider_base_url
    if cfg.authorize_url:
        auth_url = cfg.authorize_url
    elif base_url and cfg.authorize_path:
        auth_url = f"{base_url.rstrip('/')}{cfg.authorize_path}"
    else:
        raise HTTPException(status_code=400, detail="Cannot build authorize URL: need authorize_url or provider_base_url + authorize_path")

    # Build exchange URL
    if cfg.exchange_url:
        exchange_endpoint = cfg.exchange_url
    elif base_url and cfg.exchange_path:
        exchange_endpoint = f"{base_url.rstrip('/')}{cfg.exchange_path}"
    else:
        raise HTTPException(status_code=400, detail="Cannot build exchange URL: need exchange_url or provider_base_url + exchange_path")

    # Determine redirect mode:
    #   1. Native app redirect (iOS/Android custom URL scheme)
    #   2. Relay bounce (external providers via cloud relay)
    #   3. Direct callback (local providers like HA)
    relay_url = _get_relay_url()
    use_relay = bool(relay_url and cfg.authorize_url)  # external provider + relay configured

    if cfg.native_redirect_uri:
        # Native app redirect: provider redirects to the app via custom URL
        # scheme. Mobile catches it, extracts the code, and POSTs to /exchange.
        # No client_secret needed — PKCE only.
        redirect_uri = cfg.native_redirect_uri
        encoded_state = state  # plain CSRF, not relay-encoded
        effective_client_id = cfg.client_id
        use_relay = True  # mobile must POST code to /exchange (same as relay flow)
    elif use_relay:
        # External provider (e.g. Google): redirect to relay bounce
        redirect_uri = f"{relay_url.rstrip('/')}/oauth/bounce"
        # Encode CSRF + app callback into state so relay can extract it
        encoded_state = _encode_relay_state(state, "jarvis://auth-complete")
        effective_client_id = cfg.client_id
    else:
        # Local provider (e.g. HA): redirect to JCC's own callback
        external_base = _get_external_url(request)
        redirect_uri = f"{external_base}/api/v0/oauth/callback"
        encoded_state = state
        # For local providers, use JCC's external URL as client_id
        # so the origin matches redirect_uri (HA requires this).
        effective_client_id = external_base if cfg.authorize_path else cfg.client_id

    # Build authorize URL with query params
    params: dict[str, str] = {
        "response_type": "code",
        "client_id": effective_client_id,
        "redirect_uri": redirect_uri,
        "state": encoded_state,
    }
    if cfg.scopes:
        params["scope"] = " ".join(cfg.scopes)
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    if cfg.extra_authorize_params:
        params.update(cfg.extra_authorize_params)

    full_authorize_url = f"{auth_url}?{urlencode(params)}"

    # Encrypt client_secret if provided (Web Application OAuth)
    client_secret_enc = None
    if getattr(cfg, "client_secret", None):
        client_secret_enc = _encrypt_value(cfg.client_secret)

    # Create session. The JWT caller (mobile) is the owner of any
    # user-scoped tokens this flow produces; admin-key callers have no user.
    session = AuthSession(
        id=str(uuid4()),
        provider=body.provider,
        node_id=body.node_id,
        status="pending",
        state=state,
        code_verifier=code_verifier,
        provider_base_url=base_url,
        authorize_url=full_authorize_url,
        exchange_url=exchange_endpoint,
        client_id=effective_client_id,
        client_secret_enc=client_secret_enc,
        redirect_uri=redirect_uri,
        user_id=auth.user_id,
        expires_at=datetime.utcnow() + timedelta(minutes=SESSION_TTL_MINUTES),
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info(
        "Auth session created: %s provider=%s node=%s",
        session.id[:8], body.provider, body.node_id,
    )

    return CreateAuthSessionResponse(
        session_id=session.id,
        authorize_url=full_authorize_url,
        requires_code_exchange=use_relay,
    )


@router.get("/oauth/callback")
async def oauth_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """OAuth provider redirects here after user authenticates.

    Validates state, exchanges code for tokens, encrypts and stores them,
    marks session active, and publishes MQTT notification.
    Returns a redirect to jarvis://auth-complete so mobile WebView closes.
    """
    # Find session by state
    session = db.query(AuthSession).filter(AuthSession.state == state).first()
    if not session:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    if session.status != "pending":
        raise HTTPException(status_code=400, detail=f"Session already {session.status}")

    if datetime.utcnow() > session.expires_at:
        session.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Auth session expired")

    # Build token exchange request — use stored redirect_uri if available,
    # otherwise fall back to building it from the request (backwards compat)
    if session.redirect_uri:
        redirect_uri = session.redirect_uri
    else:
        external_base = _get_external_url(request)
        redirect_uri = f"{external_base}/api/v0/oauth/callback"

    exchange_data: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": session.client_id,
    }

    # Include client_secret for Web Application OAuth (server-side flow)
    if session.client_secret_enc:
        exchange_data["client_secret"] = _decrypt_value(session.client_secret_enc)

    # Include redirect_uri in exchange (most providers require it)
    exchange_data["redirect_uri"] = redirect_uri

    # Include PKCE verifier if present
    if session.code_verifier:
        exchange_data["code_verifier"] = session.code_verifier

    # Exchange code for tokens
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                session.exchange_url,
                data=exchange_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.RequestError as exc:
        logger.error("Token exchange failed for session %s: %s", session.id[:8], exc)
        raise HTTPException(status_code=502, detail=f"Token exchange failed: {exc}") from exc

    if resp.status_code != 200:
        logger.error(
            "Token exchange returned %d for session %s: %s",
            resp.status_code, session.id[:8], resp.text[:200],
        )
        raise HTTPException(
            status_code=502,
            detail=f"Token exchange failed with status {resp.status_code}",
        )

    token_data = resp.json()

    # Encrypt and store tokens
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")

    if access_token:
        session.access_token_enc = _encrypt_value(access_token)
    if refresh_token:
        session.refresh_token_enc = _encrypt_value(refresh_token)

    # Store full token response (encrypted)
    session.token_data_enc = _encrypt_value(json.dumps(token_data))
    session.status = "active"
    session.completed_at = datetime.utcnow()
    db.commit()

    logger.info(
        "Auth session completed: %s provider=%s node=%s",
        session.id[:8], session.provider, session.node_id,
    )

    # Publish MQTT notification
    _publish_auth_ready(session.provider, session.node_id, session.id, session.user_id)

    # Redirect to custom scheme so mobile WebView detects completion
    return RedirectResponse(
        url=f"jarvis://auth-complete?session_id={session.id}",
        status_code=302,
    )


class CodeExchangeRequest(BaseModel):
    """Mobile sends the auth code after relay bounce."""
    code: str


@router.post("/oauth/sessions/{session_id}/exchange")
async def exchange_code(
    session_id: str,
    body: CodeExchangeRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
):
    """Exchange an auth code for tokens (relay bounce flow).

    After the relay bounces the OAuth callback to the mobile app, the app
    extracts the code and POSTs it here. JCC exchanges the code with the
    provider's token endpoint, encrypts and stores the tokens, and marks
    the session active.
    """
    session = db.query(AuthSession).filter(AuthSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Auth session not found")

    if session.status != "pending":
        raise HTTPException(status_code=400, detail=f"Session already {session.status}")

    if datetime.utcnow() > session.expires_at:
        session.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Auth session expired")

    # Build token exchange request
    exchange_data: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": body.code,
        "client_id": session.client_id,
    }

    # Include client_secret for Web Application OAuth (server-side flow)
    if session.client_secret_enc:
        exchange_data["client_secret"] = _decrypt_value(session.client_secret_enc)

    # Include redirect_uri (provider requires it to match the authorize request)
    if session.redirect_uri:
        exchange_data["redirect_uri"] = session.redirect_uri

    # Include PKCE verifier if present
    if session.code_verifier:
        exchange_data["code_verifier"] = session.code_verifier

    # Exchange code for tokens
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                session.exchange_url,
                data=exchange_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.RequestError as exc:
        logger.error("Token exchange failed for session %s: %s", session.id[:8], exc)
        raise HTTPException(status_code=502, detail=f"Token exchange failed: {exc}") from exc

    if resp.status_code != 200:
        logger.error(
            "Token exchange returned %d for session %s: %s",
            resp.status_code, session.id[:8], resp.text[:200],
        )
        raise HTTPException(
            status_code=502,
            detail=f"Token exchange failed with status {resp.status_code}",
        )

    token_data = resp.json()

    # Encrypt and store tokens
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")

    if access_token:
        session.access_token_enc = _encrypt_value(access_token)
    if refresh_token:
        session.refresh_token_enc = _encrypt_value(refresh_token)

    session.token_data_enc = _encrypt_value(json.dumps(token_data))
    session.status = "active"
    session.completed_at = datetime.utcnow()
    db.commit()

    logger.info(
        "Auth session completed (relay exchange): %s provider=%s node=%s",
        session.id[:8], session.provider, session.node_id,
    )

    # Publish MQTT notification
    _publish_auth_ready(session.provider, session.node_id, session.id, session.user_id)

    return {"status": "ok", "session_id": session_id}


@router.get("/oauth/sessions/{session_id}", response_model=AuthSessionStatusResponse)
def get_auth_session_status(
    session_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> AuthSessionStatusResponse:
    """Mobile polls this to check if OAuth flow completed successfully."""
    session = db.query(AuthSession).filter(AuthSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Auth session not found")

    # Check expiry for pending sessions
    if session.status == "pending" and datetime.utcnow() > session.expires_at:
        session.status = "expired"
        db.commit()

    return AuthSessionStatusResponse(
        session_id=session.id,
        status=session.status,
        provider=session.provider,
    )


@router.get("/oauth/provider/{provider}/credentials", response_model=ProviderCredentialsResponse)
async def get_provider_credentials(
    provider: str,
    node_context=Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> ProviderCredentialsResponse:
    """Node pulls credentials after MQTT auth-ready notification.

    Requires node auth (X-API-Key: node_id:node_key).
    Returns decrypted tokens. Marks session as consumed.
    """
    node_id = node_context.node.node_id
    session = db.query(AuthSession).filter(
        AuthSession.provider == provider,
        AuthSession.node_id == node_id,
        AuthSession.status == "active",
    ).order_by(AuthSession.completed_at.desc()).first()

    if not session:
        raise HTTPException(status_code=404, detail="No active auth session found for this provider/node")

    # Decrypt tokens
    access_token = _decrypt_value(session.access_token_enc) if session.access_token_enc else None
    refresh_token = _decrypt_value(session.refresh_token_enc) if session.refresh_token_enc else None
    token_data = json.loads(_decrypt_value(session.token_data_enc)) if session.token_data_enc else None

    # Mark consumed
    session.status = "consumed"
    db.commit()

    logger.info(
        "Credentials consumed: session %s provider=%s node=%s",
        session.id[:8], provider, node_id,
    )

    return ProviderCredentialsResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_data=token_data,
        base_url=session.provider_base_url,
        user_id=session.user_id,
    )
