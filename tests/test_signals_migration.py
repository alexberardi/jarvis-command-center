"""DB-backed: the signals table migration + the (household_id, source_key) constraint.

Requires Postgres — run via ``python run_database_tests.py --type postgres``.
Proves the migration applies and that a duplicate (household, source_key) is
rejected at the DB level, so flaps physically collapse to one row even if the
application-side upsert is ever bypassed.
"""
from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError

from app.models import Signal


def test_unique_household_source(test_db):
    now = datetime.utcnow()
    first = Signal(
        household_id="hh-1", source_key="presence:alex", kind="presence.seen",
        is_active=True, created_at=now, updated_at=now,
    )
    test_db.add(first)
    test_db.commit()

    dup = Signal(
        household_id="hh-1", source_key="presence:alex", kind="presence.seen",
        is_active=True, created_at=now, updated_at=now,
    )
    test_db.add(dup)
    with pytest.raises(IntegrityError):
        test_db.commit()
    test_db.rollback()


def test_same_source_key_other_household_allowed(test_db):
    now = datetime.utcnow()
    test_db.add(Signal(household_id="hh-1", source_key="presence:alex", kind="presence.seen",
                       is_active=True, created_at=now, updated_at=now))
    test_db.add(Signal(household_id="hh-2", source_key="presence:alex", kind="presence.seen",
                       is_active=True, created_at=now, updated_at=now))
    test_db.commit()
    assert test_db.query(Signal).filter(Signal.source_key == "presence:alex").count() == 2
