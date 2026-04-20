"""Tests for the Phase 5 AdapterRegistry."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import ActiveAdapter, AdapterHistory, Base
from app.services.adapter_registry import (
    TRIGGER_MANUAL,
    TRIGGER_ROLLBACK,
    TRIGGER_SCHEDULER,
    AdapterRegistry,
)


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
def reg(db):
    return AdapterRegistry(db)


def test_get_active_missing_returns_none(reg):
    assert reg.get_active("h-none") is None


def test_deploy_inserts_first_adapter(reg, db):
    dep = reg.deploy("h1", "aaaa" * 8, pass_rate=0.9, trained_on_examples=120)
    assert dep.adapter_hash == "aaaa" * 8
    assert dep.pass_rate == 0.9
    assert dep.trained_on_examples == 120
    assert isinstance(dep.deployed_at, datetime)

    # Still in session — roll-forward
    db.commit()
    fetched = reg.get_active("h1")
    assert fetched is not None
    assert fetched.adapter_hash == "aaaa" * 8


def test_deploy_replaces_and_archives_prior(reg, db):
    reg.deploy("h1", "a" * 32, pass_rate=0.80, trained_on_examples=50)
    db.commit()

    reg.deploy("h1", "b" * 32, pass_rate=0.92, trained_on_examples=200)
    db.commit()

    active = reg.get_active("h1")
    assert active is not None
    assert active.adapter_hash == "b" * 32
    assert active.pass_rate == 0.92

    history = db.query(AdapterHistory).all()
    assert len(history) == 1
    h = history[0]
    assert h.adapter_hash == "a" * 32
    assert h.pass_rate == 0.80
    assert h.replaced_at is not None
    assert h.trigger == TRIGGER_SCHEDULER


def test_deploy_independent_across_households(reg, db):
    reg.deploy("h1", "a" * 32, pass_rate=0.9, trained_on_examples=10)
    reg.deploy("h2", "b" * 32, pass_rate=0.8, trained_on_examples=20)
    db.commit()

    assert reg.get_active("h1").adapter_hash == "a" * 32
    assert reg.get_active("h2").adapter_hash == "b" * 32
    assert db.query(AdapterHistory).count() == 0


def test_deploy_rejects_unknown_trigger(reg):
    with pytest.raises(ValueError):
        reg.deploy("h1", "a" * 32, pass_rate=0.9, trained_on_examples=1, trigger="bogus")


def test_deploy_records_trigger_on_history(reg, db):
    reg.deploy("h1", "a" * 32, pass_rate=0.8, trained_on_examples=5)
    reg.deploy(
        "h1",
        "b" * 32,
        pass_rate=0.85,
        trained_on_examples=15,
        trigger=TRIGGER_MANUAL,
    )
    db.commit()

    (h,) = db.query(AdapterHistory).all()
    assert h.trigger == TRIGGER_MANUAL


def test_rollback_restores_prior(reg, db):
    reg.deploy("h1", "a" * 32, pass_rate=0.80, trained_on_examples=50)
    reg.deploy("h1", "b" * 32, pass_rate=0.92, trained_on_examples=200)
    db.commit()

    dep = reg.rollback("h1")
    db.commit()

    assert dep is not None
    assert dep.adapter_hash == "a" * 32
    assert dep.pass_rate == 0.80

    active = db.query(ActiveAdapter).filter_by(household_id="h1").one()
    assert active.adapter_hash == "a" * 32

    # Rollback itself writes a history row stamped with the rollback trigger,
    # since the previously-active "b" was displaced.
    history = (
        db.query(AdapterHistory)
        .filter_by(household_id="h1")
        .order_by(AdapterHistory.id)
        .all()
    )
    assert len(history) == 2
    assert history[-1].adapter_hash == "b" * 32
    assert history[-1].trigger == TRIGGER_ROLLBACK


def test_rollback_with_no_history_returns_none(reg, db):
    reg.deploy("h1", "a" * 32, pass_rate=0.8, trained_on_examples=5)
    db.commit()

    assert reg.rollback("h1") is None


def test_rollback_missing_household_returns_none(reg):
    assert reg.rollback("nope") is None


def test_list_history_is_descending(reg, db):
    reg.deploy("h1", "a" * 32, pass_rate=0.8, trained_on_examples=1)
    reg.deploy("h1", "b" * 32, pass_rate=0.85, trained_on_examples=2)
    reg.deploy("h1", "c" * 32, pass_rate=0.9, trained_on_examples=3)
    db.commit()

    history = reg.list_history("h1")
    assert len(history) == 2
    # Most-recently replaced first
    assert history[0].adapter_hash == "b" * 32
    assert history[1].adapter_hash == "a" * 32


def test_list_history_respects_limit(reg, db):
    reg.deploy("h1", "x" * 32, pass_rate=0.8, trained_on_examples=1)
    for i in range(5):
        reg.deploy("h1", f"{i}" * 32, pass_rate=0.85, trained_on_examples=2)
        db.commit()

    history = reg.list_history("h1", limit=2)
    assert len(history) == 2


def test_ensure_state_creates_then_returns_existing(reg, db):
    state = reg.ensure_state("h1")
    db.commit()
    assert state.household_id == "h1"
    assert state.is_training is False
    assert state.last_example_count == 0

    state2 = reg.ensure_state("h1")
    assert state2 is state


def test_try_acquire_training_lock_blocks_second_caller(reg, db):
    assert reg.try_acquire_training_lock("h1") is True
    db.commit()
    assert reg.try_acquire_training_lock("h1") is False


def test_release_training_lock_clears_and_records_progress(reg, db):
    reg.try_acquire_training_lock("h1")
    db.commit()

    cutoff = datetime(2026, 4, 19, 12, 0, 0)
    reg.release_training_lock(
        "h1",
        last_trained_at=cutoff,
        last_cutoff_at=cutoff,
        last_example_count=42,
    )
    db.commit()

    state = reg.get_state("h1")
    assert state.is_training is False
    assert state.last_trained_at == cutoff
    assert state.last_cutoff_at == cutoff
    assert state.last_example_count == 42

    assert reg.try_acquire_training_lock("h1") is True


def test_release_training_lock_without_progress_keeps_counters(reg, db):
    reg.try_acquire_training_lock("h1")
    reg.release_training_lock(
        "h1",
        last_trained_at=datetime(2026, 4, 19),
        last_cutoff_at=datetime(2026, 4, 19),
        last_example_count=7,
    )
    db.commit()

    # Acquire + release without progress fields → counters preserved
    reg.try_acquire_training_lock("h1")
    reg.release_training_lock("h1")
    db.commit()

    state = reg.get_state("h1")
    assert state.last_example_count == 7
    assert state.is_training is False


def test_deploy_returned_dto_fields_match_row(reg, db):
    dep = reg.deploy("h1", "z" * 32, pass_rate=0.77, trained_on_examples=99)
    db.commit()
    assert dep.household_id == "h1"
    assert dep.adapter_hash == "z" * 32
    assert dep.pass_rate == 0.77
    assert dep.trained_on_examples == 99
