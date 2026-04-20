"""Prompt-variant rendering for training data.

Previously lived in `scripts/extract_training_data.py`. Moved here so the
Phase 5 scheduler (inside the command-center container) can render variants
without needing the top-level `scripts/` directory on its import path.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

_REL = re.compile(r"^(\d+)([dhm])$")


def parse_since(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    m = _REL.match(value.strip())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "m": timedelta(minutes=n)}[unit]
        return datetime.utcnow() - delta
    iso = value.replace("Z", "+00:00") if value.endswith("Z") else value
    dt = datetime.fromisoformat(iso)
    # Store as naive UTC because conversation_transcripts.created_at is naive
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# --- variant rendering ----------------------------------------------------


def _format_tool_block(cmd: dict, include_description: bool, include_examples: bool) -> str:
    lines = [f"- {cmd['name']}"]
    if include_description and cmd.get("description"):
        lines.append(f"    description: {cmd['description']}")
    param_lines = []
    for p in cmd.get("parameters", []):
        parts = [f"{p['name']} ({p['type']}"]
        if p.get("optional"):
            parts.append(", optional")
        parts.append(")")
        line = "".join(parts)
        if p["type"] == "enum" and p.get("values"):
            line += ": " + " | ".join(f'"{v}"' for v in p["values"])
        param_lines.append("      " + line)
    if param_lines:
        lines.append("    parameters:")
        lines.extend(param_lines)
    if include_examples and cmd.get("templates"):
        seen_shapes: set = set()
        example_patterns: list[str] = []
        for tpl in cmd["templates"]:
            shape = frozenset(tpl["args"].keys())
            if shape in seen_shapes:
                continue
            seen_shapes.add(shape)
            example_patterns.append(tpl["pattern"])
            if len(example_patterns) >= 3:
                break
        if example_patterns:
            lines.append("    examples: " + "; ".join(f'"{p}"' for p in example_patterns))
    return "\n".join(lines)


def _preamble_without_rules(preamble: str) -> str:
    idx = preamble.find("CRITICAL - Response Format")
    if idx == -1:
        return preamble
    return preamble[:idx].rstrip()


def _render_tools(cmds: list[dict], *, desc: bool, ex: bool) -> str:
    return "\n\n".join(
        _format_tool_block(c, include_description=desc, include_examples=ex) for c in cmds
    )


def build_prompt_variants(spec: dict) -> dict[str, str]:
    """Return {variant_name: prompt} for the 5 training-time variants."""
    preamble = spec["system_prompt_preamble"]
    preamble_no_rules = _preamble_without_rules(preamble)
    cmds = spec["commands"]
    return {
        "full": "\n\n".join([
            preamble, "Available Tools:",
            _render_tools(cmds, desc=True, ex=True),
        ]),
        "no_examples": "\n\n".join([
            preamble, "Available Tools:",
            _render_tools(cmds, desc=True, ex=False),
        ]),
        "no_descriptions": "\n\n".join([
            preamble, "Available Tools:",
            _render_tools(cmds, desc=False, ex=False),
        ]),
        "no_critical_rules": "\n\n".join([
            preamble_no_rules, "Available Tools:",
            _render_tools(cmds, desc=False, ex=False),
        ]),
        "schema_only": "\n\n".join([
            "Available Tools:",
            _render_tools(cmds, desc=False, ex=False),
        ]),
    }
