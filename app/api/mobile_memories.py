"""Mobile memory CRUD — user-JWT authenticated, role-scoped.

Mounted at /api/v0/mobile/memories. Used by jarvis-node-mobile.

Permission matrix (resolved per-request via jarvis-auth /internal/validate-household-access):

  MEMBER         sees / writes own only (user_id == jwt.user_id)
  POWER_USER     sees own + household-wide; writes own + non-agent household-wide
                 (blocked from source="agent" OR category="agent_context")
  ADMIN          full read+write on everything in their household
  is_superuser   same as ADMIN, globally

Source is forced to "ui" on create — users can't impersonate "voice"/"agent" origins.
"""

import logging
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.deps import (
    AuthenticatedUser,
    get_db,
    resolve_household_role,
    verify_user_jwt,
)
from app.models import UserMemory
from app.services.memory_service import MemoryService

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/mobile/memories", tags=["mobile-memories"])

# Roles that can see + manage household-wide memories
_ELEVATED_ROLES = {"power_user", "admin"}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class MobileMemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    category: str = "general"
    key: str | None = Field(default=None, max_length=500)
    is_pinned: bool = False
    scope: Literal["user", "household"] = "user"


class MobileMemoryUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=2000)
    category: str | None = None
    is_active: bool | None = None
    is_pinned: bool | None = None


class MobileMemoryResponse(BaseModel):
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
    # Mobile-only hint so the UI can grey out edit/delete on agent-injected entries
    editable: bool = True

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------


def _is_elevated(role: str, is_superuser: bool) -> bool:
    return is_superuser or role in _ELEVATED_ROLES


def _is_admin(role: str, is_superuser: bool) -> bool:
    return is_superuser or role == "admin"


def _is_agent_injected(memory: UserMemory) -> bool:
    return memory.source == "agent" or memory.category == "agent_context"


def _can_read(memory: UserMemory, user_id: int, role: str, is_superuser: bool) -> bool:
    """Read scope: MEMBER sees own only; POWER_USER+ also sees household-wide."""
    if memory.user_id == user_id:
        return True
    if memory.user_id is None and _is_elevated(role, is_superuser):
        return True
    return False


def _can_write(memory: UserMemory, user_id: int, role: str, is_superuser: bool) -> bool:
    """Write scope: MEMBER on own only. POWER_USER on own + non-agent household-wide.
    ADMIN/superuser on anything in household.
    """
    if _is_admin(role, is_superuser):
        return True
    if memory.user_id == user_id:
        return True
    if memory.user_id is None and _is_elevated(role, is_superuser):
        return not _is_agent_injected(memory)
    return False


def _editable_flag(memory: UserMemory, user_id: int, role: str, is_superuser: bool) -> bool:
    """The 'editable' field on responses — drives mobile UI affordances."""
    return _can_write(memory, user_id, role, is_superuser)


def _serialize(memory: UserMemory, user_id: int, role: str, is_superuser: bool) -> MobileMemoryResponse:
    resp = MobileMemoryResponse.model_validate(memory)
    resp.editable = _editable_flag(memory, user_id, role, is_superuser)
    return resp


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
def list_memories(
    household_id: str = Query(..., description="Household scope"),
    category: str | None = Query(None, description="Optional category filter"),
    include_household: bool = Query(
        True,
        description="Include household-wide memories (POWER_USER+ only; ignored for MEMBER)",
    ),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> list[MobileMemoryResponse]:
    """List memories the caller can see.

    MEMBER     → own only
    POWER_USER+ → own + household-wide (when include_household=True)
    """
    role = resolve_household_role(user.user_id, household_id)
    elevated = _is_elevated(role, user.is_superuser)

    query = db.query(UserMemory).filter(
        UserMemory.household_id == household_id,
        UserMemory.is_active == True,  # noqa: E712
    )
    query = query.filter(
        (UserMemory.expires_at == None)  # noqa: E711
        | (UserMemory.expires_at > datetime.utcnow())
    )

    if elevated and include_household:
        query = query.filter(
            (UserMemory.user_id == user.user_id) | (UserMemory.user_id.is_(None))
        )
    else:
        query = query.filter(UserMemory.user_id == user.user_id)

    if category:
        query = query.filter(UserMemory.category == category)

    memories = query.order_by(UserMemory.updated_at.desc()).all()
    return [_serialize(m, user.user_id, role, user.is_superuser) for m in memories]


@router.post("", status_code=201)
def create_memory(
    body: MobileMemoryCreate,
    household_id: str = Query(..., description="Household scope"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> MobileMemoryResponse:
    """Create a new memory.

    scope="user" sets user_id=jwt.user_id. scope="household" sets user_id=NULL
    and requires POWER_USER+. category="agent_context" is reserved for ADMIN.
    Source is always forced to "ui".
    """
    role = resolve_household_role(user.user_id, household_id)
    elevated = _is_elevated(role, user.is_superuser)
    admin = _is_admin(role, user.is_superuser)

    target_user_id: int | None
    if body.scope == "household":
        if not elevated:
            raise HTTPException(
                status_code=403,
                detail="Only POWER_USER or higher can create household-wide memories",
            )
        target_user_id = None
    else:
        target_user_id = user.user_id

    if body.category == "agent_context" and not admin:
        raise HTTPException(
            status_code=403,
            detail="Only ADMIN can create agent_context memories",
        )

    service = MemoryService(db)
    memory = service.save_memory(
        user_id=target_user_id,
        household_id=household_id,
        content=body.content,
        category=body.category,
        key=body.key,
        source="ui",
        is_pinned=body.is_pinned,
    )
    return _serialize(memory, user.user_id, role, user.is_superuser)


def _load_memory_for_caller(
    memory_id: int,
    household_id: str,
    user: AuthenticatedUser,
    db: Session,
    require_write: bool,
) -> tuple[UserMemory, str]:
    """Load a memory and enforce read- or write-scope. Returns (memory, role)."""
    role = resolve_household_role(user.user_id, household_id)

    memory = db.query(UserMemory).filter(UserMemory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    if memory.household_id != household_id:
        # Don't leak whether the memory exists in a different household
        raise HTTPException(status_code=404, detail="Memory not found")

    if not _can_read(memory, user.user_id, role, user.is_superuser):
        # Don't leak existence to users who can't see it
        raise HTTPException(status_code=404, detail="Memory not found")

    if require_write and not _can_write(memory, user.user_id, role, user.is_superuser):
        if _is_agent_injected(memory):
            raise HTTPException(
                status_code=403,
                detail="Agent-injected memories are read-only for non-admin users",
            )
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    return memory, role


@router.get("/{memory_id}")
def get_memory(
    memory_id: int,
    household_id: str = Query(..., description="Household scope"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> MobileMemoryResponse:
    memory, role = _load_memory_for_caller(
        memory_id, household_id, user, db, require_write=False,
    )
    return _serialize(memory, user.user_id, role, user.is_superuser)


@router.put("/{memory_id}")
def update_memory(
    memory_id: int,
    body: MobileMemoryUpdate,
    household_id: str = Query(..., description="Household scope"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> MobileMemoryResponse:
    memory, role = _load_memory_for_caller(
        memory_id, household_id, user, db, require_write=True,
    )

    data = body.model_dump(exclude_unset=True)

    # Block non-admin from re-categorizing into agent_context
    if (
        "category" in data
        and data["category"] == "agent_context"
        and not _is_admin(role, user.is_superuser)
    ):
        raise HTTPException(
            status_code=403,
            detail="Only ADMIN can set category=agent_context",
        )

    for field, value in data.items():
        setattr(memory, field, value)
    memory.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(memory)
    return _serialize(memory, user.user_id, role, user.is_superuser)


@router.delete("/{memory_id}", status_code=200)
def delete_memory(
    memory_id: int,
    household_id: str = Query(..., description="Household scope"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict:
    """Soft-delete a memory (sets is_active=False)."""
    memory, _ = _load_memory_for_caller(
        memory_id, household_id, user, db, require_write=True,
    )

    memory.is_active = False
    memory.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "deleted", "id": memory_id}
