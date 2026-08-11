"""Mistral7bMediumUntrained — single-render tool block.

The provider used to render the toolset TWICE in build_system_prompt: once as
pretty-printed (indent=2) JSON inside <tools>, and again as a human-readable
"Tools:" section. For a 30-37 tool prod set that was ~11k of a ~14k-token
prompt (~79%) — the dominant cold-prefill cost. These tests pin the toolset to
a SINGLE compact render (matching the Qwen "compressed" providers) while
proving the callable contract and cross-tool disambiguation hints survive.

Ministral14bLargeUntrained subclasses this and inherits build_system_prompt, so
it gets the same single render for free (asserted indirectly — the base is the
only thing that renders tools).
"""

import json

from app.core.prompt_providers.medium.untrained.mistral7b_medium_untrained import (
    Mistral7bMediumUntrained,
)


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "control_device",
                "description": "Turn devices or lights on/off, lock/unlock, set volume.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device": {"type": "string", "description": "The device or room to control."},
                        "action": {"type": "string", "enum": ["on", "off", "toggle"], "description": "What to do."},
                    },
                    "required": ["device"],
                },
            },
            "antipatterns": [
                {"command_name": "get_device_state", "description": "for asking whether something is on, not changing it"}
            ],
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather forecast for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name."},
                        "resolved_datetimes": {"type": "array", "items": {"type": "string"}, "format": "date-time", "description": "Natural date keys."},
                    },
                    "required": ["city"],
                },
            },
        },
    ]


def test_toolset_renders_exactly_once():
    # The <tools> schema BLOCK (opened with a newline) must appear once — the
    # old redundant "Tools:" section is gone. (The prose instruction mentions
    # the bare <tools></tools> tag inline; that's not the schema block.)
    sp = Mistral7bMediumUntrained().build_system_prompt({}, "UTC", _tools(), [])
    assert sp.count("<tools>\n") == 1
    assert "\nTools:\n" not in sp


def test_callable_contract_survives():
    # Single render still carries the callable schema: names, required, enums,
    # param names. Without these the model can't emit a valid <tool_call>.
    sp = Mistral7bMediumUntrained().build_system_prompt({}, "UTC", _tools(), [])
    assert "control_device" in sp and "get_weather" in sp
    assert '"required"' in sp and '"device"' in sp and '"city"' in sp
    assert '"enum"' in sp  # action enum preserved


def test_compact_render_strips_param_descriptions_and_indent():
    # Compact = minified JSON (no indent=2 whitespace) and per-parameter
    # "description" text stripped to save tokens (type/enum/required survive).
    sp = Mistral7bMediumUntrained().build_system_prompt({}, "UTC", _tools(), [])
    assert "The device or room to control." not in sp  # param description gone
    assert '  "name":' not in sp  # no 2-space pretty-printed JSON


def test_antipattern_hints_preserved():
    # The one routing signal the schema block lacks — cross-tool "NOT X → use Y"
    # hints — must survive the removal of the "Tools:" section.
    sp = Mistral7bMediumUntrained().build_system_prompt({}, "UTC", _tools(), [])
    assert "Disambiguation:" in sp
    assert "NOT control_device → use get_device_state" in sp


def test_signature_and_example_both_present():
    # A signature (the <tools> block) AND a concrete <tool_call> example both
    # remain — the example demonstrates the output format the parser expects.
    sp = Mistral7bMediumUntrained().build_system_prompt({}, "UTC", _tools(), [])
    assert "<tools>\n" in sp  # signatures
    assert "<tool_call>" in sp and "get_weather" in sp  # worked example


def test_no_tools_is_graceful():
    sp = Mistral7bMediumUntrained().build_system_prompt({}, "UTC", [], [])
    assert "<tools>\n</tools>" in sp
    assert "Disambiguation:" not in sp  # no antipatterns → no hint block


def test_stays_text_based():
    # Text <tool_call> path, not native API tools.
    assert Mistral7bMediumUntrained().supports_native_tools is False


def test_build_tools_xml_is_minified_and_lossless():
    xml = Mistral7bMediumUntrained._build_tools_xml(_tools())
    body = xml[len("<tools>\n"):-len("\n</tools>")]
    lines = [ln for ln in body.splitlines() if ln.strip()]
    assert len(lines) == 2  # one minified JSON object per tool
    for ln in lines:
        obj = json.loads(ln)  # each line is valid JSON
        assert obj["type"] == "function"
        assert "," in ln and ": " not in ln  # minified (no ", " / ": " spacing)
