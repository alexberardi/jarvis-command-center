"""record_presence — the canonical presence writer shared by voice + mobile.

Pins the unified shape (kind mapping, source_key, cacheable, subject+facts both
set) so every producer stays consistent, and that record_voice_presence is a thin
wrapper over it. Mocks SignalService.save_signal — no DB.
"""
from unittest.mock import MagicMock, patch

from app.services import signal_service
from app.services.signal_service import record_presence, record_voice_presence


def _save_mock(kind="presence.seen"):
    return MagicMock(return_value=MagicMock(id=1, kind=kind))


def test_home_writes_presence_seen_with_unified_shape():
    with patch.object(signal_service.SignalService, "save_signal", _save_mock()) as save:
        record_presence(
            MagicMock(), household_id="h1", user_id=42, state="home",
            source_agent="mobile", room="kitchen",
        )
    kw = save.call_args.kwargs
    assert kw["kind"] == "presence.seen"
    assert kw["source_key"] == "presence:42"
    assert kw["cacheable"] is False
    assert kw["subject"] == "user:42"          # subject AND facts both set (drift fix)
    assert kw["facts"] == {"user_id": 42, "state": "home", "room": "kitchen"}
    assert kw["source_agent"] == "mobile"
    assert kw["household_id"] == "h1"


def test_away_maps_to_presence_left():
    with patch.object(signal_service.SignalService, "save_signal", _save_mock("presence.left")) as save:
        record_presence(MagicMock(), household_id="h1", user_id=42, state="away", source_agent="mobile")
    assert save.call_args.kwargs["kind"] == "presence.left"


def test_unidentified_user_returns_none_without_writing():
    with patch.object(signal_service.SignalService, "save_signal", _save_mock()) as save:
        out = record_presence(MagicMock(), household_id="h1", user_id=None, source_agent="mobile")
    assert out is None
    save.assert_not_called()


def test_record_voice_presence_is_a_thin_wrapper():
    with patch.object(signal_service.SignalService, "save_signal", _save_mock()) as save:
        record_voice_presence(
            MagicMock(), household_id="h1", user_id=7, node_id="kitchen", speaker_name="Alex",
        )
    kw = save.call_args.kwargs
    assert kw["kind"] == "presence.seen"
    assert kw["source_key"] == "presence:7"
    assert kw["source_agent"] == "voice"
    assert kw["subject"] == "user:7"
    assert "recently heard" in kw["summary"] and "kitchen" in kw["summary"]  # voice summary preserved
