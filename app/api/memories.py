"""Memory CRUD API endpoints.

REST endpoints for managing user memories (preferences, facts, notes).
Consumed by the admin dashboard, mobile app, and other UIs.

Also provides a batch inject endpoint for background agents (calendar,
news, weather) to push ephemeral context with TTL.
"""

import logging
from datetime import datetime, timedelta

from dataclasses import dataclass

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.deps import _get_auth_base_url, get_db, verify_api_key
from app.models import UserMemory
from app.provisioning import ProvisioningAuthContext, verify_provisioning_auth
from app.services.memory_service import MemoryService

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/memories", tags=["memories"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class MemoryCreate(BaseModel):
    content: str
    category: str = "general"
    key: str | None = None
    source: str = "ui"
    is_pinned: bool = False


class MemoryUpdate(BaseModel):
    content: str | None = None
    category: str | None = None
    is_active: bool | None = None
    is_pinned: bool | None = None


class MemoryResponse(BaseModel):
    id: int
    user_id: int | None
    household_id: str
    category: str
    key: str | None = None
    content: str
    source: str
    is_active: bool
    is_pinned: bool
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
def list_memories(
    user_id: int,
    household_id: str,
    category: str | None = Query(None),
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> list[MemoryResponse]:
    """List active memories for a user in a household."""
    service = MemoryService(db)
    categories = [category] if category else None
    memories = service.get_active_memories(
        user_id=user_id,
        household_id=household_id,
        categories=categories,
    )
    return [MemoryResponse.model_validate(m) for m in memories]


@router.post("", status_code=201)
def create_memory(
    user_id: int,
    household_id: str,
    body: MemoryCreate,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> MemoryResponse:
    """Create a new memory (upserts if key matches an existing one)."""
    service = MemoryService(db)
    memory = service.save_memory(
        user_id=user_id,
        household_id=household_id,
        content=body.content,
        category=body.category,
        key=body.key,
        source=body.source,
        is_pinned=body.is_pinned,
    )
    return MemoryResponse.model_validate(memory)


@router.get("/{memory_id}")
def get_memory(
    memory_id: int,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> MemoryResponse:
    """Get a single memory by ID."""
    memory = db.query(UserMemory).filter(UserMemory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse.model_validate(memory)


@router.put("/{memory_id}")
def update_memory(
    memory_id: int,
    body: MemoryUpdate,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> MemoryResponse:
    """Update a memory."""
    memory = db.query(UserMemory).filter(UserMemory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    data = body.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(memory, field, value)
    memory.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(memory)
    return MemoryResponse.model_validate(memory)


@router.delete("/{memory_id}", status_code=200)
def delete_memory(
    memory_id: int,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    """Soft-delete a memory (set is_active=False)."""
    memory = db.query(UserMemory).filter(UserMemory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    memory.is_active = False
    memory.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "deleted", "id": memory_id}


# ---------------------------------------------------------------------------
# Agent memory injection (batch endpoint for background agents)
# ---------------------------------------------------------------------------


class MemoryInjectItem(BaseModel):
    """A single memory to inject from a background agent."""
    content: str = Field(..., max_length=2000)
    category: str = "agent_context"
    key: str = Field(..., max_length=500, description="Stable key for dedup (e.g. news:nba:warriors-lakers:2026-05-01)")
    user_id: int | None = None  # NULL = household-wide
    ttl_hours: float = Field(default=24.0, gt=0, le=720)  # max 30 days
    source: str = Field(default="agent", max_length=100)


class MemoryInjectRequest(BaseModel):
    """Batch inject request from a background agent."""
    household_id: str | None = None  # optional — auto-derived from node auth
    memories: list[MemoryInjectItem] = Field(..., max_length=100)


class MemoryInjectResponse(BaseModel):
    """Result of a batch inject operation."""
    injected: int
    updated: int
    deduplicated: int
    errors: list[str]


# Combined auth for inject: accepts node auth (X-Api-Key) OR app-to-app
@dataclass
class InjectAuthContext:
    """Auth result for the inject endpoint."""
    auth_type: str  # "node" or "app"
    household_id: str | None = None  # auto-derived for node auth


def _verify_inject_auth(
    x_api_key: str | None = Header(None),
    x_jarvis_app_id: str | None = Header(None),
    x_jarvis_app_key: str | None = Header(None),
    db: Session = Depends(get_db),
) -> InjectAuthContext:
    """Authenticate via node API key OR app-to-app credentials.

    Node auth: X-Api-Key header (node_id:node_key format).
      → household_id auto-derived from node context.
    App-to-app auth: X-Jarvis-App-Id + X-Jarvis-App-Key headers.
      → household_id must be in request body.
    """
    # Try node auth first (X-Api-Key with node_id:node_key)
    if x_api_key:
        try:
            node_ctx = verify_api_key(x_api_key=x_api_key, db=db)
            return InjectAuthContext(
                auth_type="node",
                household_id=node_ctx.household_id,
            )
        except HTTPException:
            pass  # fall through to app-to-app

    # Try app-to-app auth
    if x_jarvis_app_id and x_jarvis_app_key:
        import httpx as _httpx

        app_ping_url = _get_auth_base_url().rstrip("/") + "/internal/app-ping"
        try:
            resp = _httpx.get(
                app_ping_url,
                headers={
                    "X-Jarvis-App-Id": x_jarvis_app_id,
                    "X-Jarvis-App-Key": x_jarvis_app_key,
                },
                timeout=5.0,
            )
            if resp.status_code == 200:
                return InjectAuthContext(auth_type="app")
        except _httpx.RequestError as exc:
            logger.error("Failed to reach jarvis-auth for app auth: %s", exc)
            raise HTTPException(status_code=502, detail="Auth service unavailable") from exc

        raise HTTPException(status_code=401, detail="Invalid app credentials")

    raise HTTPException(status_code=401, detail="Authentication required (X-Api-Key or X-Jarvis-App-Id/Key)")


@router.post("/inject", status_code=200)
def inject_memories(
    body: MemoryInjectRequest,
    auth: InjectAuthContext = Depends(_verify_inject_auth),
    db: Session = Depends(get_db),
) -> MemoryInjectResponse:
    """Batch inject memories from a background agent.

    Supports 3-layer deduplication:
    1. In-batch: same (key, user_id) → last one wins
    2. Cross-batch: key-based upsert in save_memory()
    3. Content similarity: cosine > 0.9 against existing memories

    Auth: node API key (auto-derives household) OR app-to-app.
    """
    if len(body.memories) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 memories per batch")

    # Resolve household_id: node auth auto-derives, app auth requires it in body
    household_id = body.household_id or auth.household_id
    if not household_id:
        raise HTTPException(
            status_code=400,
            detail="household_id required (provide in body or use node auth)",
        )

    # Node auth: verify body household_id matches node's household (if provided)
    if auth.auth_type == "node" and body.household_id and auth.household_id:
        if body.household_id != auth.household_id:
            raise HTTPException(
                status_code=403,
                detail="household_id does not match node's household",
            )

    # Check that memory system is enabled for this household
    try:
        from app.services.settings_service import get_settings_service

        settings = get_settings_service()
        memory_enabled = settings.get(
            "memory.enabled", household_id=household_id
        )
        if memory_enabled is not None and str(memory_enabled).lower() in ("false", "0"):
            raise HTTPException(
                status_code=409, detail="Memory system is disabled for this household"
            )

        # Agent context injection is gated behind model.advanced_thinking.
        # When disabled, skip storage entirely to save embedding compute.
        adv_thinking = settings.get(
            "model.advanced_thinking", household_id=household_id
        )
        if adv_thinking is not None and str(adv_thinking).lower() in ("false", "0"):
            logger.info("Inject skipped: model.advanced_thinking is disabled for %s", household_id)
            return MemoryInjectResponse(
                injected=0, updated=0, deduplicated=0, errors=[],
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Could not check memory.enabled setting, proceeding: %s", e)

    service = MemoryService(db)

    # Layer 1: In-batch dedup by (key, user_id) — last one wins
    dedup_map: dict[tuple[str, int | None], MemoryInjectItem] = {}
    for item in body.memories:
        dedup_map[(item.key, item.user_id)] = item
    deduped_items = list(dedup_map.values())
    in_batch_deduped = len(body.memories) - len(deduped_items)

    # Save all memories and collect contents for batch embedding
    injected = 0
    updated = 0
    content_deduped = 0
    errors: list[str] = []
    saved_memories: list[tuple[UserMemory, str]] = []  # (memory, content) for embedding

    for item in deduped_items:
        try:
            expires_at = datetime.utcnow() + timedelta(hours=item.ttl_hours)

            # Check if key-based upsert will match an existing row
            is_update = False
            if item.key:
                filters = [
                    UserMemory.household_id == household_id,
                    UserMemory.key == item.key,
                    UserMemory.is_active == True,  # noqa: E712
                ]
                if item.user_id is not None:
                    filters.append(UserMemory.user_id == item.user_id)
                else:
                    filters.append(UserMemory.user_id.is_(None))
                is_update = db.query(UserMemory).filter(*filters).first() is not None

            memory = service.save_memory(
                user_id=item.user_id,
                household_id=household_id,
                content=item.content,
                category=item.category,
                key=item.key,
                source=item.source,
                expires_at=expires_at,
            )

            saved_memories.append((memory, item.content))
            if is_update:
                updated += 1
            else:
                injected += 1

        except Exception as e:
            logger.warning("Failed to inject memory key=%s: %s", item.key, e)
            errors.append(f"key={item.key}: {e}")

    # Batch-generate embeddings (best-effort — agent isn't latency-sensitive)
    if saved_memories:
        try:
            from app.core.llm_proxy_client import LLMProxyClient

            client = LLMProxyClient()
            contents = [content for _, content in saved_memories]
            vectors = client.create_embeddings_sync(contents)

            if vectors and len(vectors) == len(saved_memories):
                for i, (memory, _content) in enumerate(saved_memories):
                    if vectors[i]:
                        service.update_embedding(memory.id, vectors[i])

                # Layer 3: Content similarity guard for newly inserted memories
                # Check if any new (non-update) memory is near-duplicate of
                # an older existing memory. If so, deactivate the duplicate.
                for i, (memory, _content) in enumerate(saved_memories):
                    if not vectors[i]:
                        continue
                    existing = service.check_content_similarity(
                        household_id=household_id,
                        query_embedding=vectors[i],
                        threshold=0.9,
                    )
                    if existing and existing.id != memory.id:
                        # Near-duplicate found — keep the newer one, deactivate old
                        existing.is_active = False
                        existing.updated_at = datetime.utcnow()
                        db.commit()
                        content_deduped += 1
                        logger.info(
                            "Content dedup: deactivated memory %d (sim>0.9 with %d)",
                            existing.id, memory.id,
                        )
        except Exception as e:
            logger.warning("Embedding generation failed (non-fatal): %s", e)

    return MemoryInjectResponse(
        injected=injected,
        updated=updated,
        deduplicated=in_batch_deduped + content_deduped,
        errors=errors,
    )
