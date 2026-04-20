"""Command-example expander (Phase 3.5).

Takes a command schema (with its baked-in canonical examples) and asks an
LLM to generate N additional natural-language variations that resolve to
the same parameter output. The expanded set feeds the Phase 4 training
orchestrator alongside Phase 3's organic-trace data.

Two expander backends:
  • ClaudeExpander — preferred. Uses the Anthropic Messages API via httpx.
    Activated when ANTHROPIC_API_KEY is set. Higher-quality phrasing
    diversity, ~$0.01 per command.
  • LocalExpander — fallback. Uses the same llm-proxy `/internal/model/chat`
    endpoint the voice path uses. Free but lower phrasing diversity
    (risk: the expander model is the SAME one we'll later train, so we'd
    be training on its own output — mild feedback-loop concern).

Design note: the LLM call is injectable (a callable `ExpanderFn`), so tests
can pass a fake expander that returns canned results without burning
real API credits.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol

import httpx

from app.services.training_data_extractor import ExtractedRow

logger = logging.getLogger("uvicorn")


# --------------------------------------------------------------------------
# Expander protocol — a pluggable LLM call
# --------------------------------------------------------------------------


class ExpanderFn(Protocol):
    """A function that takes a prompt + target count and returns raw LLM text.

    The service parses the text as JSONL. If the LLM sometimes wraps output
    in code fences or prose, the parser is tolerant (extracts lines that
    look like {"voice_command": ..., "expected_parameters": ...}).
    """
    def __call__(self, prompt: str, n: int) -> str: ...


# --------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------


@dataclass
class ExpansionStats:
    command: str
    asked_for: int
    got_raw: int
    after_validation: int
    rejected_reasons: dict[str, int]


class CommandExampleExpander:
    def __init__(self, expander_fn: ExpanderFn):
        self.expander_fn = expander_fn

    # --- entrypoints --------------------------------------------------

    def expand_command(
        self,
        schema: dict,
        *,
        target_count: int = 15,
        household_id: str = "synthetic",
        user_id: int = 0,
    ) -> tuple[list[ExtractedRow], ExpansionStats]:
        """Generate `target_count` new examples for one command.

        Returns the synthetic rows + per-command stats (so the CLI can
        report coverage).
        """
        canonical = schema.get("examples") or []
        if not canonical:
            logger.warning(
                "expander: command %r has no baked examples; skipping",
                schema.get("command_name"),
            )
            return [], ExpansionStats(
                command=schema.get("command_name", "?"),
                asked_for=target_count, got_raw=0, after_validation=0,
                rejected_reasons={"no_baked_examples": 1},
            )

        prompt = self._build_prompt(schema, target_count)
        raw = self.expander_fn(prompt, target_count)
        parsed = self._parse_jsonl(raw)

        logger.info(
            "expander: command=%s  asked=%d  parsed=%d",
            schema.get("command_name"), target_count, len(parsed),
        )

        rejected: dict[str, int] = {}
        allowed_param_shapes = _extract_canonical_shapes(canonical)
        out: list[ExtractedRow] = []
        seen_texts: set[str] = set()
        now = datetime.utcnow()
        for gen in parsed:
            reason = _validate_generation(gen, allowed_param_shapes)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            voice = gen["voice_command"].strip()
            norm = voice.lower()
            if norm in seen_texts:
                rejected["duplicate_phrasing"] = rejected.get("duplicate_phrasing", 0) + 1
                continue
            seen_texts.add(norm)
            out.append(ExtractedRow(
                user_id=user_id,
                household_id=household_id,
                conversation_id=f"synth-{schema['command_name']}",
                created_at=now,
                user_message=voice,
                tool_call={
                    "name": schema["command_name"],
                    "arguments": gen["expected_parameters"],
                },
                source="synthetic_expansion",
                user_rating=None,
            ))

        stats = ExpansionStats(
            command=schema.get("command_name", "?"),
            asked_for=target_count,
            got_raw=len(parsed),
            after_validation=len(out),
            rejected_reasons=rejected,
        )
        return out, stats

    def expand_many(
        self,
        schemas: list[dict],
        *,
        target_count_per_command: int = 15,
    ) -> tuple[list[ExtractedRow], list[ExpansionStats]]:
        """Expand a whole set of commands."""
        all_rows: list[ExtractedRow] = []
        all_stats: list[ExpansionStats] = []
        for schema in schemas:
            rows, stats = self.expand_command(
                schema, target_count=target_count_per_command,
            )
            all_rows.extend(rows)
            all_stats.append(stats)
        return all_rows, all_stats

    # --- prompt assembly ---------------------------------------------

    def _build_prompt(self, schema: dict, target_count: int) -> str:
        name = schema["command_name"]
        desc = schema.get("description", "")
        canonical = schema["examples"]
        param_summary = ", ".join(
            f"{p['name']} ({p.get('type', '?')})" + ("?" if p.get("optional") else "")
            for p in (schema.get("parameters") or [])
        )

        canonical_lines = []
        for i, ex in enumerate(canonical, 1):
            canonical_lines.append(
                f"[{i}] \"{ex['voice_command']}\" → {json.dumps(ex['expected_parameters'], ensure_ascii=False)}"
            )

        return f"""You are generating synthetic voice-command training data for a parser
called `{name}`.

Command description: {desc}
Parameters: {param_summary}

Canonical examples:
{chr(10).join(canonical_lines)}

Generate {target_count} ADDITIONAL natural phrasings. Distribute roughly
evenly across the canonical examples. Each generated phrasing MUST produce
EXACTLY the same `expected_parameters` as the canonical it paraphrases.

Rules:
- Do not change parameter values.
- Do not invent new parameters or omit required ones.
- Do not repeat any canonical example verbatim.
- Include casual, formal, and disfluency-style phrasings (e.g. "uh", "please", "can you").
- Output ONLY JSONL — one JSON object per line, nothing else, no markdown fences.

Each line must look like:
{{"voice_command": "…", "expected_parameters": {{…}}}}
"""

    # --- output parsing / validation ---------------------------------

    _JSON_LINE_RX = re.compile(r"\{[^\n]*\}")

    def _parse_jsonl(self, raw: str) -> list[dict]:
        """Extract valid JSON objects, one per line. Tolerant of noise."""
        out: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Try the line as-is, then as a best-effort substring match
            candidate = line
            if not candidate.startswith("{"):
                m = self._JSON_LINE_RX.search(line)
                if not m:
                    continue
                candidate = m.group(0)
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _extract_canonical_shapes(canonical: list[dict]) -> list[dict]:
    """Return the raw canonical parameter payloads so generated examples can
    be compared against them for shape-matching."""
    out: list[dict] = []
    for ex in canonical:
        params = ex.get("expected_parameters")
        if isinstance(params, dict):
            out.append(params)
    return out


def _validate_generation(gen: dict, allowed_shapes: list[dict]) -> str | None:
    """Reject generations that fail structural checks.

    Returns None if valid, else a short reason string used for stats.
    """
    if not isinstance(gen, dict):
        return "not_dict"
    voice = gen.get("voice_command")
    if not isinstance(voice, str) or not voice.strip():
        return "missing_or_empty_voice"
    params = gen.get("expected_parameters")
    if not isinstance(params, dict):
        return "missing_or_bad_params"
    # The generated params must exactly match ONE of the canonical shapes.
    # This is a strong constraint — the LLM is supposed to paraphrase the
    # utterance only, not change the semantics.
    if not any(_params_equal(params, canon) for canon in allowed_shapes):
        return "params_do_not_match_any_canonical"
    return None


def _params_equal(a: dict, b: dict) -> bool:
    """Strict equality check (order-insensitive, type-insensitive for common cases)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if isinstance(a[k], str) and isinstance(b[k], str):
            if a[k].strip().lower() != b[k].strip().lower():
                return False
        else:
            if a[k] != b[k]:
                return False
    return True


# --------------------------------------------------------------------------
# Concrete expander backends
# --------------------------------------------------------------------------


def build_claude_expander(
    api_key: str | None = None,
    *,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 2000,
) -> ExpanderFn:
    """Expander that calls the Anthropic Messages API directly via httpx."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    def _call(prompt: str, n: int) -> str:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        with httpx.Client(timeout=60) as client:
            r = client.post(
                "https://api.anthropic.com/v1/messages",
                json=body, headers=headers,
            )
            r.raise_for_status()
            data = r.json()
        # Messages API returns content as a list of content-block dicts
        blocks = data.get("content") or []
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        return "\n".join(texts)

    return _call


def build_local_expander(
    model_service_url: str = "http://localhost:7705",
    internal_token: str | None = None,
    *,
    max_tokens: int = 2000,
) -> ExpanderFn:
    """Expander that calls the local llm-proxy. Free but quality varies."""
    tok = internal_token or os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if not tok:
        raise RuntimeError("MODEL_SERVICE_TOKEN / LLM_PROXY_INTERNAL_TOKEN not set")

    def _call(prompt: str, n: int) -> str:
        body = {
            "model": "live",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        headers = {"X-Internal-Token": tok, "content-type": "application/json"}
        with httpx.Client(timeout=180) as client:
            r = client.post(
                f"{model_service_url.rstrip('/')}/internal/model/chat",
                json=body, headers=headers,
            )
            r.raise_for_status()
            return r.json().get("content", "")

    return _call


def default_expander() -> ExpanderFn:
    """Pick Claude if configured, otherwise fall back to local llm-proxy."""
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("expander: using Claude (ANTHROPIC_API_KEY set)")
        return build_claude_expander()
    logger.info("expander: using local llm-proxy (no ANTHROPIC_API_KEY)")
    return build_local_expander()
