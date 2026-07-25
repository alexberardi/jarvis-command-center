"""Tests for the household speaking-voice persona (the "voice" layer).

The persona is a per-household, free-text speaking style injected into the cached
voice prompt as a fenced ``<personality>`` block. It governs TONE/word choice ONLY
— never tool-calling or safety, which stay non-overridable. This suite locks:

  1. The presets module (default is a real preset; ids unique; caps sane).
  2. The seam: ``build_context_header`` appends the block when node_context carries
     a persona, and is byte-identical to the pre-persona header when it doesn't
     (the safe fallback) — and identical across speakers (rule #1: cached prefix).
  3. The warmup resolver ``_get_household_persona``: default flows through, empty
     is honored, fails SAFE to the warm default on a settings error, household-scoped.

Hermetic: no DB, no model, no network. Settings are stubbed.
"""
from unittest.mock import MagicMock, patch

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.services.persona_presets import (
    DEFAULT_PERSONA,
    DEFAULT_PERSONA_PRESET_ID,
    PERSONA_FRAME,
    PERSONA_MAX_CHARS,
    PERSONA_PRESETS,
)

SETTINGS = "app.services.settings_service.get_settings_service"


class _Provider(IJarvisPromptProvider):
    """Minimal concrete provider so we can exercise the base build_context_header
    (the shared seam every real provider inherits)."""

    @property
    def name(self) -> str:
        return "TestProvider"

    def build_system_prompt(self, node_context, timezone, tools, available_commands=None):
        return "unused"


def _settings_returning(value):
    svc = MagicMock()
    svc.get.return_value = value
    return MagicMock(return_value=svc)


def _settings_raising():
    return MagicMock(side_effect=RuntimeError("settings service down"))


def _handler():
    from app.core.conversation_handler import ConversationHandler
    return ConversationHandler(model=MagicMock(), llm_client=MagicMock())


class TestPersonaPresets:
    """The presets module is the single source of truth for the voice layer."""

    def test_default_persona_is_nonempty(self):
        assert DEFAULT_PERSONA.strip()

    def test_default_persona_is_within_cap(self):
        assert len(DEFAULT_PERSONA) <= PERSONA_MAX_CHARS

    def test_max_chars_is_sane(self):
        assert PERSONA_MAX_CHARS > 0

    def test_default_preset_id_is_a_real_preset(self):
        ids = {p["id"] for p in PERSONA_PRESETS}
        assert DEFAULT_PERSONA_PRESET_ID in ids

    def test_default_preset_text_matches_default_persona(self):
        # Re-loading the default preset IS the "reset to default" affordance, so
        # its text must equal the setting's default.
        default = next(p for p in PERSONA_PRESETS if p["id"] == DEFAULT_PERSONA_PRESET_ID)
        assert default["text"] == DEFAULT_PERSONA

    def test_preset_ids_are_unique(self):
        ids = [p["id"] for p in PERSONA_PRESETS]
        assert len(ids) == len(set(ids))

    def test_every_preset_has_id_label_text_within_cap(self):
        for p in PERSONA_PRESETS:
            assert p["id"] and p["label"] and p["text"].strip()
            assert len(p["text"]) <= PERSONA_MAX_CHARS

    def test_expected_starter_presets_present(self):
        ids = {p["id"] for p in PERSONA_PRESETS}
        assert {"warm_folksy", "terse", "dry_witty", "classic_jarvis"} <= ids


class TestBuildContextHeaderPersona:
    """The seam: build_context_header injects the persona into the cached header."""

    def _ctx(self, **extra):
        base = {"room": "kitchen", "voice_mode": "brief"}
        base.update(extra)
        return base

    def test_no_persona_is_byte_identical_to_pre_persona_header(self):
        # The exact string real providers relied on before personas existed.
        out = _Provider().build_context_header(self._ctx())
        assert out == (
            "You are Jarvis, a function calling voice assistant.\n"
            "Context: room=kitchen, style=brief"
        )

    def test_empty_persona_is_byte_identical(self):
        out = _Provider().build_context_header(self._ctx(household_persona=""))
        assert "<personality>" not in out
        assert out.endswith("style=brief")

    def test_persona_appends_fenced_block_after_identity(self):
        out = _Provider().build_context_header(
            self._ctx(household_persona="You're warm and folksy.")
        )
        # Identity line + context still lead (load-bearing, kept intact)...
        assert out.startswith("You are Jarvis, a function calling voice assistant.\n")
        assert "Context: room=kitchen, style=brief" in out
        # ...then the fenced persona, with the voice-only frame.
        assert "<personality>" in out and "</personality>" in out
        assert PERSONA_FRAME in out
        assert "You're warm and folksy." in out
        # Persona comes AFTER the identity/context header.
        assert out.index("<personality>") > out.index("Context: room=kitchen")

    def test_identical_across_speakers_rule_one(self):
        # Persona is per-household → the cached prefix must be byte-identical
        # regardless of who is speaking. build_context_header ignores speaker.
        persona = "You're dry and witty."
        alex = _Provider().build_context_header(
            self._ctx(household_persona=persona, speaker_name="alex")
        )
        sam = _Provider().build_context_header(
            self._ctx(household_persona=persona, speaker_name="sam")
        )
        assert alex == sam

    def test_none_context_is_safe(self):
        # Defensive: build_context_header(None) must not crash.
        out = _Provider().build_context_header(None)
        assert out.startswith("You are Jarvis")


class TestGetHouseholdPersona:
    """The warmup resolver that stashes the persona into node_context."""

    def test_returns_setting_value(self):
        with patch(SETTINGS, _settings_returning("Be terse.")):
            assert _handler()._get_household_persona({"household_id": "hh-1"}) == "Be terse."

    def test_strips_whitespace(self):
        with patch(SETTINGS, _settings_returning("  Be terse.  ")):
            assert _handler()._get_household_persona({"household_id": "hh-1"}) == "Be terse."

    def test_empty_stored_value_is_honored(self):
        # A household that cleared the box gets NO persona (not the default) —
        # honoring an explicit opt-out of the voice layer.
        with patch(SETTINGS, _settings_returning("")):
            assert _handler()._get_household_persona({"household_id": "hh-1"}) == ""

    def test_default_flows_through_when_no_override(self):
        # settings.get returns the SettingDefinition default when no row exists.
        with patch(SETTINGS, _settings_returning(DEFAULT_PERSONA)):
            assert _handler()._get_household_persona({"household_id": "hh-1"}) == DEFAULT_PERSONA

    def test_fails_safe_to_default_on_settings_error(self):
        # Voice has no egress/safety risk, so a settings outage must NOT strip
        # Jarvis to the flat identity line — it falls back to the warm default.
        with patch(SETTINGS, _settings_raising()):
            assert _handler()._get_household_persona({"household_id": "hh-1"}) == DEFAULT_PERSONA

    def test_passes_household_id_to_settings(self):
        svc = MagicMock()
        svc.get.return_value = "x"
        with patch(SETTINGS, MagicMock(return_value=svc)):
            _handler()._get_household_persona({"household_id": "hh-xyz"})
            svc.get.assert_called_once_with(
                "persona.household_prompt", household_id="hh-xyz"
            )

    def test_none_value_returns_empty(self):
        with patch(SETTINGS, _settings_returning(None)):
            assert _handler()._get_household_persona({"household_id": "hh-1"}) == ""


class TestEndOfPromptReminder:
    """The recency fix: the persona is restated at the END of the assembled
    prompt (after the not-for-me wall), so a small model that lost the
    top-of-prompt <personality> block still carries the voice into generation."""

    def _handler(self):
        import types

        class _P:
            def build_system_prompt(self, nc, tz, tools, flags):
                return "PROVIDER_BASE"

        return types.SimpleNamespace(
            prompt_provider=_P(),
            model=object(),
            _get_characterization_injection_enabled=lambda nc: False,
        )

    def test_reminder_is_last_and_after_not_for_me(self):
        from app.core import conversation_handler as ch

        nc = {"household_persona": "You're dry and witty."}
        out = ch.ConversationHandler._get_system_prompt(self._handler(), nc, None, [], None)
        # The not-for-me wall is still there...
        assert "<not_for_me/>" in out
        # ...and the VOICE reminder is the final instruction, restating the persona.
        assert "YOUR VOICE" in out
        assert "You're dry and witty." in out
        assert out.rstrip().endswith("just speak in it.")
        # Recency: the voice reminder comes AFTER the not-for-me block.
        assert out.index("YOUR VOICE") > out.rindex("<not_for_me/>")

    def test_no_reminder_when_persona_absent(self):
        from app.core import conversation_handler as ch

        nc = {}
        out = ch.ConversationHandler._get_system_prompt(self._handler(), nc, None, [], None)
        assert "YOUR VOICE" not in out


class TestSettingDefinition:
    def test_persona_setting_defined_with_folksy_default(self):
        from app.services.settings_definitions import SETTINGS_DEFINITIONS

        d = next(x for x in SETTINGS_DEFINITIONS if x.key == "persona.household_prompt")
        assert d.value_type == "string"
        assert d.default == DEFAULT_PERSONA
