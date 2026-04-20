"""Per-canonical example expansion for adapter training.

Called from CC's /adapters/train handler before dataset formatting.
For each baked adapter example, asks the active llm-proxy to generate
N colloquial rephrasings that must resolve to the SAME parameters.

Deliberately kept narrow:
  - One LLM backend (the configured llm-proxy — same one serving
    inference), no dedicated Claude path. The user routes cloud Claude
    via llm-proxy's cloud-model config if they want higher quality
    expansion, so this service doesn't need its own cloud integration.
  - Per-canonical anchoring: every generated variant is paired with
    the originating canonical's params, so we never get
    (voice="eleven minus three", params={num1:7,num2:9}) shape errors
    that killed an earlier session iteration.
  - Tolerant parsing: regex fallback if the model emits malformed JSON.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger("uvicorn")


# --------------------------------------------------------------------------
# Settings — read from settings_service by the caller
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionConfig:
    """Runtime config for the expansion pass."""
    enabled: bool = False
    expand_n: int = 2                    # variants per canonical
    max_canonicals: int = 10             # cap canonicals per command
    temperature: float = 0.3
    max_tokens: int = 1500
    timeout_seconds: float = 120.0


# --------------------------------------------------------------------------
# LLM call — pluggable for tests
# --------------------------------------------------------------------------


class LLMCaller(Protocol):
    """Abstracts the actual LLM call so tests can inject fakes."""
    async def __call__(self, system: str, user: str, cfg: ExpansionConfig) -> str: ...


# Default LLM caller routes through the CC's LLMProxyClient
async def default_llm_caller(system: str, user: str, cfg: ExpansionConfig) -> str:
    """Use the command-center's LLMProxyClient to reach the active llm-proxy."""
    from app.core.llm_proxy_client import LLMProxyClient
    client = LLMProxyClient()
    response = await client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model="live",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    # response shape: {"choices":[{"message":{"content":"..."}}], ...}
    choices = response.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


# --------------------------------------------------------------------------
# Prompt + parsing
# --------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You generate natural-language voice command variations. "
    "Given one canonical example, produce N colloquial rephrasings "
    "a user might say to mean THE SAME THING. "
    "CRITICAL: preserve every literal value from the canonical — do not change "
    "numbers, names, durations, units, or specific terms. Only vary phrasing, "
    "politeness, word order. "
    "Output ONLY a JSON list of strings, no prose, no commentary."
)


def _build_user_prompt(
    command_name: str,
    description: str,
    canonical_voice: str,
    canonical_params: dict,
    n: int,
) -> str:
    return (
        f"Command: {command_name}\n"
        f"Description: {description}\n\n"
        f"Canonical: {canonical_voice!r}\n"
        f"Fixed parameters (must match across all variants): {json.dumps(canonical_params)}\n\n"
        f"Produce {n} rephrasings. JSON list of strings only."
    )


_BAD_KEYS = {"text", "name", "description", "command", "value", "type"}


def _extract_variants(content: str, max_n: int) -> list[str]:
    """Parse LLM output into candidate variants.

    Tries strict JSON list first, falls back to regex-extracting quoted
    strings, filters obvious JSON-key noise, dedupes.
    """
    variants: list[str] = []
    start, end = content.find("["), content.rfind("]")
    if 0 <= start < end:
        try:
            arr = json.loads(content[start : end + 1])
            variants = [str(v).strip() for v in arr if isinstance(v, str) and v.strip()]
        except json.JSONDecodeError:
            pass

    if len(variants) < 3:
        quoted = re.findall(r'"((?:[^"\\]|\\.)+)"', content)
        quoted += re.findall(r"'((?:[^'\\]|\\.)+)'", content)
        variants = [
            q.strip()
            for q in quoted
            if len(q.strip()) >= 5 and q.strip().lower() not in _BAD_KEYS
        ]

    seen: set[str] = set()
    deduped: list[str] = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped[:max_n]


# --------------------------------------------------------------------------
# Main API
# --------------------------------------------------------------------------


@dataclass
class ExpansionResult:
    """What the expansion pass produced for one command."""
    command_name: str
    baked_count: int
    expanded_count: int


async def expand_examples(
    commands: list,
    cfg: ExpansionConfig,
    llm_caller: Optional[LLMCaller] = None,
) -> list[ExpansionResult]:
    """Expand the `examples` list on each command in-place.

    Each baked example (up to cfg.max_canonicals per command) gets cfg.expand_n
    new phrasings appended with the same params. Commands are modified in
    place; the returned list is stats-only for logging/auditing.

    Args:
        commands: list of CommandDefinition Pydantic objects (same shape the
            /adapters/train handler already uses). Must have `.examples` and
            `.command_name`; each example has `.voice_command`,
            `.expected_parameters`, `.is_primary`.
        cfg: ExpansionConfig (typically populated from settings).
        llm_caller: optional injection point for tests.

    Returns:
        One ExpansionResult per command.
    """
    caller = llm_caller or default_llm_caller
    results: list[ExpansionResult] = []

    for cmd in commands:
        baked = list(cmd.examples or [])
        baked_count = len(baked)
        expanded_count = 0

        if not cfg.enabled or cfg.expand_n <= 0 or not baked:
            results.append(ExpansionResult(
                command_name=cmd.command_name,
                baked_count=baked_count,
                expanded_count=0,
            ))
            continue

        # Build new examples object(s) — same shape as input .examples items
        example_cls = type(baked[0])
        new_examples = []
        description = getattr(cmd, "description", "") or ""

        for canonical in baked[:cfg.max_canonicals]:
            try:
                raw = await caller(
                    _SYSTEM_PROMPT,
                    _build_user_prompt(
                        command_name=cmd.command_name,
                        description=description,
                        canonical_voice=canonical.voice_command,
                        canonical_params=canonical.expected_parameters or {},
                        n=cfg.expand_n,
                    ),
                    cfg,
                )
            except Exception as e:
                logger.warning(
                    "expansion call failed for %s (canonical %r): %s",
                    cmd.command_name, canonical.voice_command, e,
                )
                continue

            variants = _extract_variants(raw, cfg.expand_n)
            for new_voice in variants:
                new_examples.append(example_cls(
                    voice_command=new_voice,
                    expected_parameters=canonical.expected_parameters,
                    is_primary=False,
                ))

        cmd.examples = baked + new_examples
        expanded_count = len(new_examples)

        logger.info(
            "adapter expansion: %s baked=%d expanded=%d",
            cmd.command_name, baked_count, expanded_count,
        )
        results.append(ExpansionResult(
            command_name=cmd.command_name,
            baked_count=baked_count,
            expanded_count=expanded_count,
        ))

    return results
