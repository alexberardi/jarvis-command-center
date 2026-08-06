"""Ambient context bundle — the always-on situational snapshot injected as a
TRAILING per-turn system message, kept OUT of the cached warmup prefix.

Guards the invariants that make it cache-safe:
  - an empty bundle yields an empty block (a household not using it is a no-op);
  - the assembled snapshot is DETERMINISTIC (time quantized, no live seconds);
  - it does NOT enter the cached prefix (``build_context_header`` is byte-identical
    with or without ambient) — putting it at the top of the prefix cold-prefilled
    the whole prompt on every new conversation (prod TTFS 3s→5s, 2026-08-06);
  - its block is transient (stripped + rebuilt each turn, never accumulates).
Weather/calendar come from the household's priority agent memories.
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


class TestAmbientStaysOutOfCachedPrefix:
    """The regression guard (prod TTFS 3s→5s, 2026-08-06): ambient is situational,
    so it must NOT ride the byte-stable warmup prefix — else every new conversation
    cold-prefills the whole ~9k-token prompt. It now rides a trailing per-turn block.
    """

    def _provider(self):
        from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider

        class _P(IJarvisPromptProvider):
            @property
            def name(self) -> str:
                return "TestProvider"

            def build_system_prompt(self, node_context, timezone, tools, available_commands=None):
                return "unused"

        return _P()

    def test_ambient_does_not_change_the_cached_header(self):
        p = self._provider()
        base = {"room": "kitchen", "voice_mode": "brief"}
        with_ambient = {**base, "ambient_context": "As of 3:15 PM.\nWeather: 72°F clear"}
        # Byte-identical with or without ambient → warmup prefix stays cacheable.
        assert p.build_context_header(with_ambient) == p.build_context_header(base)
        assert "<ambient_context>" not in p.build_context_header(with_ambient)

    def test_ambient_block_is_a_transient_per_turn_block(self):
        from app.core.conversation_handler import _is_transient_system_block

        block = build_ambient_context_block("As of 3:15 PM.\nWeather: 72°F clear")
        # Stripped + rebuilt each turn (like the speaker block) so it never accumulates.
        assert _is_transient_system_block({"role": "system", "content": block}) is True
        assert _is_transient_system_block({"role": "user", "content": block}) is False
