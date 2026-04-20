"""Tests for the Phase 3.5 command-example expander."""

from __future__ import annotations

import json
from typing import Callable

import pytest

from app.services.command_example_expander import (
    CommandExampleExpander,
    _extract_canonical_shapes,
    _params_equal,
    _validate_generation,
)


# --------------------------------------------------------------------------
# Schemas used across tests
# --------------------------------------------------------------------------


MUSIC_SCHEMA = {
    "command_name": "music",
    "description": "Control music playback.",
    "parameters": [
        {"name": "action", "type": "enum"},
        {"name": "query", "type": "string", "optional": True},
    ],
    "examples": [
        {"voice_command": "Play Radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}, "is_primary": True},
        {"voice_command": "Pause", "expected_parameters": {"action": "pause"}, "is_primary": False},
    ],
}


TIMER_SCHEMA = {
    "command_name": "set_timer",
    "description": "Start a timer.",
    "parameters": [
        {"name": "duration_minutes", "type": "integer"},
    ],
    "examples": [
        {"voice_command": "5 minute timer", "expected_parameters": {"duration_minutes": 5}, "is_primary": True},
    ],
}


# --------------------------------------------------------------------------
# Fake expander — returns canned JSONL
# --------------------------------------------------------------------------


def _fake_expander(output: str) -> Callable[[str, int], str]:
    def _fn(prompt: str, n: int) -> str:
        return output
    return _fn


# --------------------------------------------------------------------------
# Unit tests for helpers
# --------------------------------------------------------------------------


class TestParamsEqual:
    def test_exact(self):
        assert _params_equal({"a": 1}, {"a": 1})

    def test_case_insensitive_strings(self):
        assert _params_equal({"q": "JAZZ"}, {"q": "jazz"})

    def test_key_mismatch(self):
        assert not _params_equal({"a": 1}, {"a": 1, "b": 2})

    def test_value_mismatch(self):
        assert not _params_equal({"a": 1}, {"a": 2})


class TestValidateGeneration:
    def test_accepts_matching_params(self):
        shapes = [{"action": "play", "query": "Radiohead"}]
        assert _validate_generation(
            {"voice_command": "Put on Radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}},
            shapes,
        ) is None

    def test_rejects_no_voice(self):
        assert _validate_generation(
            {"voice_command": "", "expected_parameters": {"action": "play"}}, [{"action": "play"}],
        ) == "missing_or_empty_voice"

    def test_rejects_bad_params(self):
        assert _validate_generation(
            {"voice_command": "Play", "expected_parameters": "not a dict"},
            [{"action": "play"}],
        ) == "missing_or_bad_params"

    def test_rejects_mismatched_params(self):
        """LLM must paraphrase only, not change semantics."""
        shapes = [{"action": "play", "query": "Radiohead"}]
        # Generator altered the query slot — reject
        assert _validate_generation(
            {"voice_command": "Play Nirvana", "expected_parameters": {"action": "play", "query": "Nirvana"}},
            shapes,
        ) == "params_do_not_match_any_canonical"

    def test_accepts_second_canonical(self):
        """Any of the canonical shapes is allowed to match."""
        shapes = [
            {"action": "play", "query": "Radiohead"},
            {"action": "pause"},
        ]
        assert _validate_generation(
            {"voice_command": "hold on", "expected_parameters": {"action": "pause"}}, shapes,
        ) is None


class TestExtractShapes:
    def test_gets_only_dict_params(self):
        canonical = [
            {"voice_command": "a", "expected_parameters": {"x": 1}},
            {"voice_command": "b", "expected_parameters": None},
        ]
        assert _extract_canonical_shapes(canonical) == [{"x": 1}]


# --------------------------------------------------------------------------
# End-to-end expand_command tests
# --------------------------------------------------------------------------


class TestExpandCommand:
    def test_basic_happy_path(self):
        output = "\n".join([
            '{"voice_command": "Put on Radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}}',
            '{"voice_command": "I wanna hear Radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}}',
            '{"voice_command": "Hold the music", "expected_parameters": {"action": "pause"}}',
        ])
        exp = CommandExampleExpander(expander_fn=_fake_expander(output))
        rows, stats = exp.expand_command(MUSIC_SCHEMA, target_count=3)
        assert len(rows) == 3
        assert stats.after_validation == 3
        assert {r.tool_call["name"] for r in rows} == {"music"}
        assert all(r.source == "synthetic_expansion" for r in rows)

    def test_rejects_drifted_params(self):
        """Generator altered the query value — drop those rows."""
        output = "\n".join([
            '{"voice_command": "Play Nirvana", "expected_parameters": {"action": "play", "query": "Nirvana"}}',
            '{"voice_command": "Put on Radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}}',
        ])
        exp = CommandExampleExpander(expander_fn=_fake_expander(output))
        rows, stats = exp.expand_command(MUSIC_SCHEMA, target_count=2)
        assert len(rows) == 1
        assert rows[0].user_message == "Put on Radiohead"
        assert stats.rejected_reasons.get("params_do_not_match_any_canonical") == 1

    def test_tolerant_of_code_fences(self):
        output = "```json\n" + "\n".join([
            '{"voice_command": "Put on Radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}}',
            '{"voice_command": "Hold the music", "expected_parameters": {"action": "pause"}}',
        ]) + "\n```"
        exp = CommandExampleExpander(expander_fn=_fake_expander(output))
        rows, _stats = exp.expand_command(MUSIC_SCHEMA, target_count=2)
        assert len(rows) == 2

    def test_tolerant_of_leading_commentary(self):
        output = (
            "Sure! Here are some variations:\n"
            '{"voice_command": "Put on Radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}}\n'
            "Let me know if you'd like more.\n"
        )
        exp = CommandExampleExpander(expander_fn=_fake_expander(output))
        rows, _ = exp.expand_command(MUSIC_SCHEMA, target_count=1)
        assert len(rows) == 1

    def test_dedup_duplicate_phrasings(self):
        output = "\n".join([
            '{"voice_command": "Put on Radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}}',
            '{"voice_command": "put on radiohead", "expected_parameters": {"action": "play", "query": "Radiohead"}}',
            '{"voice_command": "PUT ON RADIOHEAD", "expected_parameters": {"action": "play", "query": "Radiohead"}}',
        ])
        exp = CommandExampleExpander(expander_fn=_fake_expander(output))
        rows, stats = exp.expand_command(MUSIC_SCHEMA, target_count=3)
        assert len(rows) == 1
        assert stats.rejected_reasons.get("duplicate_phrasing") == 2

    def test_no_baked_examples_returns_empty(self):
        schema = {"command_name": "empty", "description": "", "parameters": [], "examples": []}
        exp = CommandExampleExpander(expander_fn=_fake_expander(""))
        rows, stats = exp.expand_command(schema, target_count=5)
        assert rows == []
        assert stats.rejected_reasons == {"no_baked_examples": 1}


class TestExpandMany:
    def test_mixes_commands(self):
        outputs_by_cmd = {
            "music": '{"voice_command": "Hold the music", "expected_parameters": {"action": "pause"}}',
            "set_timer": '{"voice_command": "Timer 5", "expected_parameters": {"duration_minutes": 5}}',
        }
        # Figure out which command the prompt is for, return the matching output
        def dispatch(prompt: str, n: int) -> str:
            for cmd in outputs_by_cmd:
                if f"called `{cmd}`" in prompt:
                    return outputs_by_cmd[cmd]
            return ""
        exp = CommandExampleExpander(expander_fn=dispatch)
        rows, stats = exp.expand_many([MUSIC_SCHEMA, TIMER_SCHEMA], target_count_per_command=1)
        assert len(rows) == 2
        assert {r.tool_call["name"] for r in rows} == {"music", "set_timer"}
        assert [s.command for s in stats] == ["music", "set_timer"]


# --------------------------------------------------------------------------
# Prompt-assembly smoke test
# --------------------------------------------------------------------------


class TestPromptAssembly:
    def test_prompt_mentions_command_and_examples(self):
        captured = {"prompt": None}
        def capture(prompt: str, n: int) -> str:
            captured["prompt"] = prompt
            return ""
        exp = CommandExampleExpander(expander_fn=capture)
        exp.expand_command(MUSIC_SCHEMA, target_count=5)
        p = captured["prompt"]
        assert p is not None
        assert "called `music`" in p
        assert "Play Radiohead" in p  # canonical present
        assert "5 ADDITIONAL" in p  # target count present
