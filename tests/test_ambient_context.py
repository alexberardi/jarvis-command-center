"""Ambient context bundle — the always-on situational snapshot injected into the
cached prompt prefix.

Guards the two invariants that make it cache-safe: an empty bundle yields an empty
block (so a household not using it keeps a byte-identical prefix), and the assembled
snapshot is DETERMINISTIC (time quantized, no live seconds) so it doesn't differ
turn-to-turn. Weather/calendar come from the household's priority agent memories.
"""
import re
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.conversation_handler import ConversationHandler
from app.core.prompt_providers.shared.core_rules import build_ambient_context_block
from app.models import Base, UserMemory


def _sqlite_sessionmaker(seed=()):
    # StaticPool: one shared connection so the assembler's own session sees seeded rows.
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    if seed:
        s = Session()
        for category, content in seed:
            s.add(UserMemory(user_id=None, household_id="hh-1", category=category,
                             content=content, is_active=True))
        s.commit()
        s.close()
    return Session


class TestBuildAmbientBlock:
    def test_empty_yields_empty_block(self):
        # The safe fallback: no bundle -> no block -> prefix byte-identical to today.
        assert build_ambient_context_block("") == ""
        assert build_ambient_context_block(None) == ""
        assert build_ambient_context_block("   ") == ""

    def test_nonempty_is_fenced_and_instructive(self):
        block = build_ambient_context_block("As of 3:15 PM, Wed Jul 30.\nWeather: 72F clear")
        assert block.startswith("<ambient_context>") and block.endswith("</ambient_context>")
        assert "72F clear" in block
        assert "never invent" in block.lower()  # the anti-hallucination instruction


class TestAssembleBundle:
    def _handler(self):
        # The assembler uses no __init__ state, so bypass the heavy constructor.
        return ConversationHandler.__new__(ConversationHandler)

    def test_snapshot_is_deterministic_labeled_and_carries_all_categories(self):
        Session = _sqlite_sessionmaker(seed=[
            ("weather", "60°F, overcast — 72% chance of rain after 3 PM"),
            ("calendar", "Standup 9:00 AM; Dentist 4:30 PM downtown"),
            ("reminder", "Leo's prescription refill is due this week"),
        ])
        h = self._handler()
        with patch("app.db.get_session_local", return_value=Session):
            out1 = h._assemble_ambient_bundle("hh-1", "UTC")
            out2 = h._assemble_ambient_bundle("hh-1", "UTC")

        assert out1 == out2                       # byte-stable within the 15-min bucket
        assert out1.startswith("As of ")          # absolute clock line
        assert "Weather: 60°F, overcast — 72% chance of rain after 3 PM" in out1
        assert "Today: Standup 9:00 AM; Dentist 4:30 PM downtown" in out1
        assert "Reminders: Leo's prescription refill is due this week" in out1
        # Quantized to 15 minutes — no live seconds could survive into the prefix.
        minute = re.search(r"As of \d{1,2}:(\d{2}) ", out1)
        assert minute and int(minute.group(1)) % 15 == 0

    def test_missing_sources_leave_time_only(self):
        Session = _sqlite_sessionmaker()  # empty DB — no household memories yet
        h = self._handler()
        with patch("app.db.get_session_local", return_value=Session):
            out = h._assemble_ambient_bundle("hh-1", "UTC")
        assert out.startswith("As of ") and "\n" not in out  # time-only, single line

    def test_fail_open_on_error(self):
        h = self._handler()
        # A DB failure must never break warmup — return "" (no block).
        with patch("app.db.get_session_local", side_effect=RuntimeError("db down")):
            out = h._assemble_ambient_bundle("hh-1", "UTC")
        assert out == ""
