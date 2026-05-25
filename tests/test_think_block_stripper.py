"""Tests for ThinkBlockStripper."""

import pytest

from app.core.utils.think_block_stripper import (
    DEFAULT_THINK_DELIMITERS,
    ThinkBlockStripper,
)


@pytest.fixture
def qwen_stripper():
    return ThinkBlockStripper.from_pair(DEFAULT_THINK_DELIMITERS)


@pytest.fixture
def llama33_stripper():
    return ThinkBlockStripper("[[[thinking start]]]", "[[[thinking end]]]")


class TestStripCompleteBlocks:
    def test_strips_single_block(self, qwen_stripper):
        assert qwen_stripper.strip_complete_blocks(
            "hello <think>foo</think> world"
        ) == "hello world"

    def test_strips_multiple_blocks(self, qwen_stripper):
        assert qwen_stripper.strip_complete_blocks(
            "a <think>x</think> b <think>y</think> c"
        ) == "a b c"

    def test_leaves_unclosed_block(self, qwen_stripper):
        # An open block with no end marker stays — caller decides whether
        # to wait or strip aggressively.
        assert qwen_stripper.strip_complete_blocks(
            "hello <think>truncated"
        ) == "hello <think>truncated"

    def test_spans_newlines(self, qwen_stripper):
        # Qwen3 emits `<think>\n\n</think>\n\n...` even under /no_think.
        assert qwen_stripper.strip_complete_blocks(
            "<think>\n\nfoo\n\n</think>\n\nreal answer"
        ) == "real answer"

    def test_custom_delimiters(self, llama33_stripper):
        assert llama33_stripper.strip_complete_blocks(
            "pre [[[thinking start]]]reasoning[[[thinking end]]] post"
        ) == "pre post"


class TestHasOpenBlock:
    def test_returns_true_for_unclosed(self, qwen_stripper):
        assert qwen_stripper.has_open_block("hello <think>foo")

    def test_returns_false_for_balanced(self, qwen_stripper):
        assert not qwen_stripper.has_open_block(
            "hello <think>foo</think>"
        )

    def test_returns_false_for_no_marker(self, qwen_stripper):
        assert not qwen_stripper.has_open_block("hello world")

    def test_custom_delimiters(self, llama33_stripper):
        assert llama33_stripper.has_open_block(
            "pre [[[thinking start]]]reason"
        )
        assert not llama33_stripper.has_open_block(
            "pre [[[thinking start]]]r[[[thinking end]]]"
        )


class TestStripAll:
    def test_removes_unclosed_open(self, qwen_stripper):
        # max_tokens cut off mid-think — strip_all should clear it.
        assert qwen_stripper.strip_all(
            "hello <think>truncated"
        ) == "hello "

    def test_combines_complete_and_unclosed(self, qwen_stripper):
        assert qwen_stripper.strip_all(
            "a <think>x</think> b <think>cut"
        ) == "a b "
