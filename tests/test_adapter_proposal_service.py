"""Tests for AdapterProposalService (Phase 7.1)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import ActiveAdapter, AdapterProposal, Base
from app.services.adapter_proposal_service import (
    STATUS_APPLIED,
    STATUS_DISMISSED,
    STATUS_EXPIRED,
    STATUS_PENDING,
    STATUS_SUPERSEDED,
    AdapterProposalService,
    HashMismatchError,
    NoActiveAdapterError,
    ProposalExpiredError,
    ProposalNotFoundError,
    ProposalNotPendingError,
)
from app.services.adapter_registry import AdapterRegistry


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def svc(db):
    return AdapterProposalService(db)


def _mk_proposal(svc, **overrides):
    defaults = dict(
        household_id="h1",
        adapter_hash="a" * 32,
        provider_name_before="OldProvider",
        provider_name_after="NewProvider",
        pass_rate_before=0.78,
        pass_rate_after=0.86,
        latency_before_s=5.88,
        latency_after_s=4.18,
        per_command_delta={"calculate": {"before": 0.875, "after": 1.0}},
        trained_on_examples=489,
    )
    defaults.update(overrides)
    return svc.create(**defaults)


# ---------- create ----------


def test_create_persists_row(svc, db):
    view = _mk_proposal(svc)
    db.commit()

    assert view.status == STATUS_PENDING
    assert view.pass_rate_after == 0.86
    assert view.per_command_delta == {"calculate": {"before": 0.875, "after": 1.0}}
    assert view.expires_at > view.created_at

    row = db.query(AdapterProposal).one()
    assert row.status == STATUS_PENDING
    assert row.per_command_delta_json is not None


def test_create_supersedes_prior_pending(svc, db):
    first = _mk_proposal(svc, adapter_hash="a" * 32)
    second = _mk_proposal(svc, adapter_hash="b" * 32)
    db.commit()

    first_row = db.query(AdapterProposal).filter_by(id=first.id).one()
    second_row = db.query(AdapterProposal).filter_by(id=second.id).one()
    assert first_row.status == STATUS_SUPERSEDED
    assert second_row.status == STATUS_PENDING


def test_create_does_not_supersede_applied(svc, db):
    """Applied proposals stay applied even when new pending one is created."""
    earlier = _mk_proposal(svc, adapter_hash="a" * 32)
    db.commit()
    svc.apply(earlier.id, household_id="h1")
    db.commit()

    _mk_proposal(svc, adapter_hash="b" * 32)
    db.commit()

    earlier_row = db.query(AdapterProposal).filter_by(id=earlier.id).one()
    assert earlier_row.status == STATUS_APPLIED


# ---------- apply ----------


def test_apply_deploys_and_marks_applied(svc, db):
    view = _mk_proposal(svc)
    db.commit()

    applied, dep = svc.apply(view.id, household_id="h1")
    db.commit()

    assert applied.status == STATUS_APPLIED
    assert applied.decided_at is not None
    assert dep.adapter_hash == view.adapter_hash
    assert dep.pass_rate == view.pass_rate_after

    active = db.query(ActiveAdapter).filter_by(household_id="h1").one()
    assert active.adapter_hash == view.adapter_hash


def test_apply_unknown_proposal_raises_not_found(svc):
    with pytest.raises(ProposalNotFoundError):
        svc.apply("does-not-exist", household_id="h1")


def test_apply_wrong_household_raises_not_found(svc, db):
    view = _mk_proposal(svc, household_id="h1")
    db.commit()

    with pytest.raises(ProposalNotFoundError):
        svc.apply(view.id, household_id="h2")


def test_apply_twice_raises_not_pending(svc, db):
    view = _mk_proposal(svc)
    db.commit()
    svc.apply(view.id, household_id="h1")
    db.commit()

    with pytest.raises(ProposalNotPendingError):
        svc.apply(view.id, household_id="h1")


def test_apply_expired_proposal_marks_expired(svc, db):
    view = _mk_proposal(svc)
    # Force expiry: rewrite the row to be already expired.
    row = db.query(AdapterProposal).filter_by(id=view.id).one()
    row.expires_at = datetime.utcnow() - timedelta(hours=1)
    db.commit()

    with pytest.raises(ProposalExpiredError):
        svc.apply(view.id, household_id="h1")

    db.commit()
    row_after = db.query(AdapterProposal).filter_by(id=view.id).one()
    assert row_after.status == STATUS_EXPIRED
    assert row_after.decided_at is not None


# ---------- dismiss ----------


def test_dismiss_marks_dismissed_without_deploy(svc, db):
    view = _mk_proposal(svc)
    db.commit()

    dismissed = svc.dismiss(view.id, household_id="h1")
    db.commit()

    assert dismissed.status == STATUS_DISMISSED
    assert dismissed.decided_at is not None

    # No active adapter was touched
    active = db.query(ActiveAdapter).filter_by(household_id="h1").one_or_none()
    assert active is None


def test_dismiss_wrong_household_raises_not_found(svc, db):
    view = _mk_proposal(svc, household_id="h1")
    db.commit()

    with pytest.raises(ProposalNotFoundError):
        svc.dismiss(view.id, household_id="other")


def test_dismiss_already_applied_raises_not_pending(svc, db):
    view = _mk_proposal(svc)
    db.commit()
    svc.apply(view.id, household_id="h1")
    db.commit()

    with pytest.raises(ProposalNotPendingError):
        svc.dismiss(view.id, household_id="h1")


# ---------- revert ----------


def test_revert_rolls_back_adapter_and_returns_provider(svc, db):
    """Apply p1 (→ B), apply p2 (→ C), then revert C; expect B restored
    and provider_name_before from p2 returned."""
    p1 = _mk_proposal(
        svc,
        adapter_hash="b" * 32,
        provider_name_before="OriginalProvider",
        provider_name_after="ProviderAfterP1",
    )
    db.commit()
    svc.apply(p1.id, household_id="h1")
    db.commit()

    p2 = _mk_proposal(
        svc,
        adapter_hash="c" * 32,
        provider_name_before="ProviderAfterP1",
        provider_name_after="ProviderAfterP2",
    )
    db.commit()
    svc.apply(p2.id, household_id="h1")
    db.commit()

    restored, provider = svc.revert(adapter_hash="c" * 32, household_id="h1")
    db.commit()

    assert restored is not None
    assert restored.adapter_hash == "b" * 32
    assert provider == "ProviderAfterP1"


def test_revert_no_active_raises(svc):
    with pytest.raises(NoActiveAdapterError):
        svc.revert(adapter_hash="a" * 32, household_id="h-empty")


def test_revert_hash_mismatch_raises(svc, db):
    view = _mk_proposal(svc, adapter_hash="b" * 32)
    db.commit()
    svc.apply(view.id, household_id="h1")
    db.commit()

    with pytest.raises(HashMismatchError):
        svc.revert(adapter_hash="z" * 32, household_id="h1")


def test_revert_adapter_without_proposal_returns_none_provider(svc, db):
    """Adapters deployed before Phase 7 (no paired proposal) revert cleanly
    but return None for the provider — caller leaves llm.interface alone."""
    # Simulate a pre-Phase-7 deploy by writing active_adapter directly.
    reg = AdapterRegistry(db)
    reg.deploy("h1", "a" * 32, pass_rate=0.70, trained_on_examples=50)
    reg.deploy("h1", "b" * 32, pass_rate=0.75, trained_on_examples=60)
    db.commit()

    restored, provider = svc.revert(adapter_hash="b" * 32, household_id="h1")
    db.commit()

    assert restored is not None
    assert restored.adapter_hash == "a" * 32
    assert provider is None


# ---------- expiry ----------


def test_expire_stale_flips_overdue_pending(svc, db):
    fresh = _mk_proposal(svc, adapter_hash="a" * 32)
    stale = _mk_proposal(svc, adapter_hash="b" * 32)
    db.commit()

    # Back-date the stale one AND the fresh one needs to stay pending. But
    # creating two rows in the same household auto-supersedes the first.
    # Work with the status we have: stale is pending, fresh is pending.
    # (Second _mk_proposal superseded the first.)
    # Re-pend the first one manually for this test:
    first_row = db.query(AdapterProposal).filter_by(id=fresh.id).one()
    first_row.status = STATUS_PENDING
    stale_row = db.query(AdapterProposal).filter_by(id=stale.id).one()
    stale_row.expires_at = datetime.utcnow() - timedelta(days=1)
    db.commit()

    touched = svc.expire_stale()
    db.commit()

    assert touched == 1
    assert db.query(AdapterProposal).filter_by(id=stale.id).one().status == STATUS_EXPIRED
    assert db.query(AdapterProposal).filter_by(id=fresh.id).one().status == STATUS_PENDING


# ---------- list / get ----------


def test_list_for_household_filters_and_orders(svc, db):
    a = _mk_proposal(svc, adapter_hash="a" * 32)
    b = _mk_proposal(svc, adapter_hash="b" * 32)  # supersedes a
    db.commit()

    pending = svc.list_for_household("h1", status=STATUS_PENDING)
    assert [p.id for p in pending] == [b.id]

    superseded = svc.list_for_household("h1", status=STATUS_SUPERSEDED)
    assert [p.id for p in superseded] == [a.id]

    all_for_h1 = svc.list_for_household("h1")
    assert [p.id for p in all_for_h1] == [b.id, a.id]  # desc by created_at


def test_attach_inbox_item_updates_row(svc, db):
    view = _mk_proposal(svc)
    db.commit()

    svc.attach_inbox_item(view.id, "inbox-abc")
    db.commit()

    row = db.query(AdapterProposal).filter_by(id=view.id).one()
    assert row.inbox_item_id == "inbox-abc"


def test_attach_inbox_item_on_missing_id_is_noop(svc):
    svc.attach_inbox_item("does-not-exist", "inbox-xyz")  # must not raise
