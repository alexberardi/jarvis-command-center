"""
Tests for the provider-level NOT_FOR_ME opt-out (``bakes_not_for_me``).

ConversationHandler._get_system_prompt appends NOT_FOR_ME_INSTRUCTION to
every provider's prompt so the false-wake gating policy lives in exactly
one place. A fine-tuned provider whose model has that policy trained into
its weights must be able to opt out — appending the instruction there
would duplicate the policy and waste ~4.6k prompt chars per conversation.

These tests pin the append byte-for-byte for the default case (so the
opt-out change is provably a no-op for every existing provider) and pin
the opt-out output to be identical apart from the instruction block.
"""
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from app.core.conversation_handler import ConversationHandler
from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.shared.core_rules import NOT_FOR_ME_INSTRUCTION


PROVIDER_BODY = "provider body\n\nRules:\n- be brief\n"


class _MinimalProvider(IJarvisPromptProvider):
    """Smallest concrete provider: only the abstract members implemented.

    Inherits every default — including bakes_not_for_me — so it stands in
    for 'every existing provider' in the no-op proof.
    """

    @property
    def name(self) -> str:
        return "MinimalTestProvider"

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return PROVIDER_BODY


class _BakedProvider(_MinimalProvider):
    """Provider whose model bakes the not_for_me policy into its weights."""

    @property
    def name(self) -> str:
        return "BakedTestProvider"

    @property
    def bakes_not_for_me(self) -> bool:
        return True


def _handler(provider: Optional[IJarvisPromptProvider]) -> ConversationHandler:
    return ConversationHandler(
        model=MagicMock(), llm_client=MagicMock(), prompt_provider=provider
    )


class TestInterfaceDefault:
    """The property must exist on the interface and default to False."""

    def test_minimal_concrete_subclass_defaults_to_false(self):
        assert _MinimalProvider().bakes_not_for_me is False

    def test_override_to_true_is_respected(self):
        assert _BakedProvider().bakes_not_for_me is True


class TestDefaultProviderStillGetsInstruction:
    """bakes_not_for_me=False (the default) must be byte-identical to the
    pre-change behavior: base.rstrip() + blank line + instruction + newline."""

    def test_instruction_appended_exactly_once_at_exact_position(self):
        result = _handler(_MinimalProvider())._get_system_prompt(
            {"room": "kitchen"}, "UTC", []
        )

        expected = f"{PROVIDER_BODY.rstrip()}\n\n{NOT_FOR_ME_INSTRUCTION}\n"
        assert result == expected
        assert result.count(NOT_FOR_ME_INSTRUCTION) == 1

    def test_legacy_model_path_still_gets_instruction(self):
        """No provider → legacy model._build_system_prompt path is untouched."""
        mock_model = MagicMock()
        mock_model._build_system_prompt = MagicMock(return_value="legacy body")
        handler = ConversationHandler(model=mock_model, llm_client=MagicMock())

        result = handler._get_system_prompt({}, "UTC", [])

        assert result == f"legacy body\n\n{NOT_FOR_ME_INSTRUCTION}\n"

    def test_minimal_fallback_path_still_gets_instruction(self):
        """No provider, no _build_system_prompt → fallback path is untouched."""
        mock_model = MagicMock(spec=["name", "use_tool_classifier"])
        handler = ConversationHandler(model=mock_model, llm_client=MagicMock())

        result = handler._get_system_prompt({}, "UTC", [])

        assert result == (
            f"You are a helpful voice assistant.\n\n{NOT_FOR_ME_INSTRUCTION}\n"
        )


class TestOptOutProviderSkipsInstruction:
    """bakes_not_for_me=True must omit the instruction block and nothing else."""

    def test_instruction_absent(self):
        result = _handler(_BakedProvider())._get_system_prompt(
            {"room": "kitchen"}, "UTC", []
        )

        assert NOT_FOR_ME_INSTRUCTION not in result

    def test_everything_else_identical_to_default_output(self):
        """Opt-out output == default output minus the instruction block.

        Default: base.rstrip() + "\\n\\n" + INSTRUCTION + "\\n"
        Opt-out: base.rstrip() + "\\n"
        Same rstrip normalization, same trailing newline — the only delta
        is the "\\n" + INSTRUCTION block itself.
        """
        default_result = _handler(_MinimalProvider())._get_system_prompt(
            {"room": "kitchen"}, "UTC", []
        )
        opt_out_result = _handler(_BakedProvider())._get_system_prompt(
            {"room": "kitchen"}, "UTC", []
        )

        assert opt_out_result == f"{PROVIDER_BODY.rstrip()}\n"
        assert default_result == (
            f"{opt_out_result.rstrip()}\n\n{NOT_FOR_ME_INSTRUCTION}\n"
        )


class TestNodeContextUntouched:
    """_get_system_prompt is a pure read of node_context in both modes —
    it must not stash derived keys (e.g. a '_system_prompt_base') or
    otherwise mutate the dict the conversation cache will hold."""

    def test_node_context_not_mutated_default(self):
        node_context = {"room": "kitchen", "household_id": "hh-1"}
        snapshot = dict(node_context)

        _handler(_MinimalProvider())._get_system_prompt(node_context, "UTC", [])

        assert node_context == snapshot

    def test_node_context_not_mutated_opt_out(self):
        node_context = {"room": "kitchen", "household_id": "hh-1"}
        snapshot = dict(node_context)

        _handler(_BakedProvider())._get_system_prompt(node_context, "UTC", [])

        assert node_context == snapshot
