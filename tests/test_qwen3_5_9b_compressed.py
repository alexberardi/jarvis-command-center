"""Qwen3.5-9B provider — native tool calling.

Qwen3.5's native tool syntax is handled end-to-end by llama.cpp (tools via the API →
structured tool_calls back), so this provider goes native and must NOT re-declare a
text <tools>/<tool_call> format in the system prompt (that fought the native format and
dropped tool calls to direct answers — the live run_errand miss).
"""

from app.core.prompt_providers.medium.untrained.qwen3_5_9b_compressed import (
    Qwen3_5_9B_Compressed,
)


def test_qwen35_declares_native_tools():
    assert Qwen3_5_9B_Compressed().supports_native_tools is True


def test_qwen35_prompt_omits_the_text_tool_format():
    # The native tools go via the API; a text <tools> block + JSON <tool_call>
    # instruction here would conflict with the model's native syntax.
    sp = Qwen3_5_9B_Compressed().build_system_prompt({}, "UTC", [], [])
    assert "<tools>" not in sp
    assert "<tool_call>" not in sp


def test_qwen35_prompt_keeps_routing_guidance():
    # It should still guide: the function-calling preamble survives (identity/rules
    # too) even though the tool-format block is gone.
    sp = Qwen3_5_9B_Compressed().build_system_prompt({}, "UTC", [], [])
    assert "function calling AI model" in sp


def test_qwen35_keeps_no_think_control():
    # /no_think (latency + <think> stripping) rides on user_message_suffix, inherited
    # from the 8B provider — the native override must not lose it.
    assert Qwen3_5_9B_Compressed().user_message_suffix in ("/no_think", "/think")
