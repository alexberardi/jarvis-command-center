"""Memory CRUD API endpoints.

REST endpoints for managing user memories (preferences, facts, notes).
Consumed by the admin dashboard, mobile app, and other UIs.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models import UserMemory
from app.provisioning import ProvisioningAuthContext, verify_provisioning_auth
from app.services.memory_service import MemoryService

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
    user_id: int
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
