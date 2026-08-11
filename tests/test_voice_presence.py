"""Unit tests for the voice-derived presence producer.

When a voice command resolves a confident speaker at a node, that is a free
"person X is here" observation. record_voice_presence writes it as a
presence.seen Signal so "who's home?" works with zero new hardware — the first
real producer on the Signal Bus, proving genericity alongside synthetic signals.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Signal
from app.services.signal_service import record_voice_presence


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_records_presence_seen(db):
    sig = record_voice_presence(
        db, household_id="hh-1", user_id=7, node_id="kitchen", speaker_name="Alex"
    )
    assert sig is not None
    assert sig.kind == "presence.seen"
    assert sig.source_key == "presence:7"      # source_key = suppression/dedup key per user
    assert sig.household_id == "hh-1"
    assert sig.user_id == 7
    assert sig.node_id == "kitchen"
    assert sig.summary and "Alex" in sig.summary
    assert sig.expires_at is not None          # presence decays (has a TTL)


def test_upserts_on_repeat(db):
    record_voice_presence(db, household_id="hh-1", user_id=7, node_id="kitchen", speaker_name="Alex")
    record_voice_presence(db, household_id="hh-1", user_id=7, node_id="bedroom", speaker_name="Alex")
    rows = db.query(Signal).filter(Signal.source_key == "presence:7").all()
    assert len(rows) == 1                       # same person → one row, moved room
    assert rows[0].node_id == "bedroom"


def test_no_user_id_is_noop(db):
    # an unresolved speaker can't be attributed → no presence written
    sig = record_voice_presence(
        db, household_id="hh-1", user_id=None, node_id="kitchen", speaker_name=None
    )
    assert sig is None
    assert db.query(Signal).count() == 0
