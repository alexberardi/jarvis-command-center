"""Guards the split of the old coupled `model.advanced_thinking` flag into
`model.include_thinking` (slow chain-of-thought) and `model.advanced_context`
(proactive context injection) — so a household can have fast proactive context
WITHOUT paying for <think> latency.
"""
from app.core.prompt_providers.medium.untrained.qwen3_8b_compressed import Qwen3_8B_Compressed
from app.services.settings_definitions import SETTINGS_DEFINITIONS

_BY_KEY = {d.key: d for d in SETTINGS_DEFINITIONS}


def test_settings_are_split_and_deprecated_removed():
    assert "model.advanced_context" in _BY_KEY
    assert "model.include_thinking" in _BY_KEY
    assert "model.advanced_thinking" not in _BY_KEY  # deprecated → deleted by migration


def test_both_default_off():
    assert _BY_KEY["model.advanced_context"].default is False
    assert _BY_KEY["model.include_thinking"].default is False


def test_think_suffix_follows_include_thinking_only():
    # The /think vs /no_think suffix (the slow part) is driven by include_thinking,
    # independent of context injection.
    p = Qwen3_8B_Compressed.__new__(Qwen3_8B_Compressed)
    p.include_thinking = False
    assert p.user_message_suffix == "/no_think"
    p.include_thinking = True
    assert p.user_message_suffix == "/think"
