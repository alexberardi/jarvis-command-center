"""Tests for app/core/utils/speaker_stickiness.py."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.utils import speaker_stickiness as sm
from app.core.utils.speaker_stickiness import (
    DEFAULT_STICKINESS_MIN_CONFIDENCE,
    DEFAULT_STICKINESS_TTL_SECONDS,
    inherit_speaker_for_node,
    record_speaker_for_node,
    reset_node_history,
)


@pytest.fixture(autouse=True)
def _reset_history():
    sm.clear_all()
    yield
    sm.clear_all()


class TestRecordAndInherit:
    def test_high_confidence_record_is_inheritable(self):
        record_speaker_for_node("node-1", user_id=42, confidence=0.95)
        assert inherit_speaker_for_node("node-1") == 42

    def test_low_confidence_does_not_record(self):
        # Use a value well below the default threshold (0.55) so the test
        # doesn't break if the default is tuned upward later.
        record_speaker_for_node("node-1", user_id=42, confidence=0.30)
        assert inherit_speaker_for_node("node-1") is None

    def test_threshold_is_inclusive(self):
        record_speaker_for_node("node-1", user_id=42, confidence=DEFAULT_STICKINESS_MIN_CONFIDENCE)
        assert inherit_speaker_for_node("node-1") == 42

    def test_none_user_id_skipped(self):
        record_speaker_for_node("node-1", user_id=None, confidence=0.95)
        assert inherit_speaker_for_node("node-1") is None

    def test_none_confidence_skipped(self):
        record_speaker_for_node("node-1", user_id=42, confidence=None)
        assert inherit_speaker_for_node("node-1") is None

    def test_no_record_returns_none(self):
        assert inherit_speaker_for_node("unknown-node") is None

    def test_per_node_isolation(self):
        record_speaker_for_node("node-1", user_id=42, confidence=0.95)
        record_speaker_for_node("node-2", user_id=99, confidence=0.95)
        assert inherit_speaker_for_node("node-1") == 42
        assert inherit_speaker_for_node("node-2") == 99


class TestTTL:
    def test_stale_snapshot_returns_none(self, monkeypatch):
        record_speaker_for_node("node-1", user_id=42, confidence=0.95)
        # Pretend we're past the TTL by mutating the stored timestamp
        sm._node_speaker_history["node-1"].timestamp = (
            datetime.now(timezone.utc) - timedelta(seconds=DEFAULT_STICKINESS_TTL_SECONDS + 5)
        )
        assert inherit_speaker_for_node("node-1") is None

    def test_stale_snapshot_is_evicted(self):
        record_speaker_for_node("node-1", user_id=42, confidence=0.95)
        sm._node_speaker_history["node-1"].timestamp = (
            datetime.now(timezone.utc) - timedelta(seconds=DEFAULT_STICKINESS_TTL_SECONDS + 5)
        )
        inherit_speaker_for_node("node-1")
        # Stale entry should be cleaned up on lookup
        assert "node-1" not in sm._node_speaker_history

    def test_fresh_snapshot_persists(self):
        record_speaker_for_node("node-1", user_id=42, confidence=0.95)
        # Just inside TTL
        sm._node_speaker_history["node-1"].timestamp = (
            datetime.now(timezone.utc) - timedelta(seconds=DEFAULT_STICKINESS_TTL_SECONDS - 1)
        )
        assert inherit_speaker_for_node("node-1") == 42


class TestReset:
    def test_reset_node_history_clears_one_node(self):
        record_speaker_for_node("node-1", user_id=42, confidence=0.95)
        record_speaker_for_node("node-2", user_id=99, confidence=0.95)
        reset_node_history("node-1")
        assert inherit_speaker_for_node("node-1") is None
        assert inherit_speaker_for_node("node-2") == 99

    def test_reset_unknown_node_is_safe(self):
        # Should not raise
        reset_node_history("ghost-node")
