"""Mobile phonebook CRUD — user-JWT authenticated, household-scoped.

Mounted at /api/v0/mobile/household/{household_id}/phone-contacts.
Used by jarvis-node-mobile's phonebook screen.

Entries are **household-scoped** (a business's number is a household fact,
per the phone-calls PRD data model), so any member of the household may
read and write them — matching how household settings and rooms behave
rather than the per-user ownership model memories use. Per-user overlays
(usual order, personal history) are PRD open question 3 and deliberately
not implemented here.

Cross-household ids return 404 rather than 403: a wrong-household caller
should not be able to probe which contact ids exist.

Numbers are normalized and validated through the same path the dialer uses
(`normalize_us_number`), so emergency, short-code, and premium-rate numbers
are rejected at write time — a contact that could never legally be dialed
should never make it into the book.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.deps import (
    AuthenticatedUser,
    get_db,
    verify_household_role,
    verify_user_jwt,
)
from app.models import PhoneContact
from app.services.phone_call_service import (
    NumberValidationError,
    _normalize_name,
    normalize_us_number,
)

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-phone-contacts"])

# Household-scoped facts: members read and write, same bar as reading them.
_READ_ROLE = "member"
_WRITE_ROLE = "member"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PhoneContactCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    number: str = Field(..., min_length=1, max_length=32)
    address: str | None = None
    notes: str | None = None


class PhoneContactUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    number: str | None = Field(default=None, min_length=1, max_length=32)
    address: str | None = None
    notes: str | None = None
    do_not_call: bool | None = None


class PhoneContactResponse(BaseModel):
    id: str
    name: str
    number: str
    address: str | None = None
    source: str
    line_type: str | None = None
    do_not_call: bool
    notes: str | None = None
    verified_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PhoneContactListResponse(BaseModel):
    contacts: list[PhoneContactResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validated_number(raw: str) -> str:
    """Normalize to E.164 or 400 with the validator's own message."""
    try:
        return normalize_us_number(raw)
    except NumberValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _get_owned(db: Session, household_id: str, contact_id: str) -> PhoneContact:
    """Fetch a contact scoped to the household, else 404.

    404 (not 403) on a foreign id: existence itself shouldn't leak.
    """
    contact = (
        db.query(PhoneContact)
        .filter(
            PhoneContact.id == contact_id,
            PhoneContact.household_id == household_id,
        )
        .first()
    )
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact


def _name_taken(
    db: Session, household_id: str, normalized: str, exclude_id: str | None = None
) -> bool:
    q = db.query(PhoneContact).filter(
        PhoneContact.household_id == household_id,
        PhoneContact.normalized_name == normalized,
    )
    if exclude_id:
        q = q.filter(PhoneContact.id != exclude_id)
    return db.query(q.exists()).scalar()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/household/{household_id}/phone-contacts",
    response_model=PhoneContactListResponse,
)
def list_phone_contacts(
    household_id: str = Path(..., description="Household UUID"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> PhoneContactListResponse:
    """Every phonebook entry for the household, sorted by name."""
    verify_household_role(user.user_id, household_id, required_role=_READ_ROLE)

    contacts = (
        db.query(PhoneContact)
        .filter(PhoneContact.household_id == household_id)
        .order_by(PhoneContact.name.asc())
        .all()
    )
    return PhoneContactListResponse(
        contacts=[PhoneContactResponse.model_validate(c) for c in contacts]
    )


@router.post(
    "/household/{household_id}/phone-contacts",
    response_model=PhoneContactResponse,
    status_code=201,
)
def create_phone_contact(
    body: PhoneContactCreate,
    household_id: str = Path(..., description="Household UUID"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> PhoneContactResponse:
    """Add a business to the household phonebook."""
    verify_household_role(user.user_id, household_id, required_role=_WRITE_ROLE)

    number = _validated_number(body.number)
    name = body.name.strip()
    normalized = _normalize_name(name)
    if not normalized:
        raise HTTPException(status_code=400, detail="Name must contain letters or digits")
    if _name_taken(db, household_id, normalized):
        raise HTTPException(
            status_code=409,
            detail=f"'{name}' is already in this household's phonebook",
        )

    now = datetime.utcnow()
    contact = PhoneContact(
        household_id=household_id,
        name=name,
        normalized_name=normalized,
        number=number,
        address=body.address,
        notes=body.notes,
        source="manual",
        do_not_call=False,
        verified_at=now,
        created_at=now,
    )
    db.add(contact)
    db.commit()
    db.refresh(contact)
    logger.info(
        "Phone contact created household=%s name=%r by user=%s",
        household_id, name, user.user_id,
    )
    return PhoneContactResponse.model_validate(contact)


@router.patch(
    "/household/{household_id}/phone-contacts/{contact_id}",
    response_model=PhoneContactResponse,
)
def update_phone_contact(
    body: PhoneContactUpdate,
    household_id: str = Path(..., description="Household UUID"),
    contact_id: str = Path(..., description="Contact id"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> PhoneContactResponse:
    """Edit one entry. Setting do_not_call blocks future calls to it."""
    verify_household_role(user.user_id, household_id, required_role=_WRITE_ROLE)
    contact = _get_owned(db, household_id, contact_id)

    if body.number is not None:
        contact.number = _validated_number(body.number)
        # A hand-entered number is a fresh assertion about this business.
        contact.verified_at = datetime.utcnow()
        contact.source = "manual"
    if body.name is not None:
        name = body.name.strip()
        normalized = _normalize_name(name)
        if not normalized:
            raise HTTPException(
                status_code=400, detail="Name must contain letters or digits"
            )
        if _name_taken(db, household_id, normalized, exclude_id=contact.id):
            raise HTTPException(
                status_code=409,
                detail=f"'{name}' is already in this household's phonebook",
            )
        contact.name = name
        contact.normalized_name = normalized
    if body.address is not None:
        contact.address = body.address
    if body.notes is not None:
        contact.notes = body.notes
    if body.do_not_call is not None:
        contact.do_not_call = body.do_not_call

    db.commit()
    db.refresh(contact)
    logger.info(
        "Phone contact updated household=%s id=%s by user=%s dnc=%s",
        household_id, contact_id[:8], user.user_id, contact.do_not_call,
    )
    return PhoneContactResponse.model_validate(contact)


@router.delete(
    "/household/{household_id}/phone-contacts/{contact_id}",
    status_code=204,
)
def delete_phone_contact(
    household_id: str = Path(..., description="Household UUID"),
    contact_id: str = Path(..., description="Contact id"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> None:
    """Remove an entry. Calls will fall back to search/manual entry after."""
    verify_household_role(user.user_id, household_id, required_role=_WRITE_ROLE)
    contact = _get_owned(db, household_id, contact_id)
    db.delete(contact)
    db.commit()
    logger.info(
        "Phone contact deleted household=%s id=%s by user=%s",
        household_id, contact_id[:8], user.user_id,
    )
