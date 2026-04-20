"""Training data extractor (Phase 3).

Pulls clean voice→tool-call pairs from `conversation_transcripts`, applies
filters + dedup + per-user balance, and emits row-level records ready for
the Phase 4 training orchestrator to format as llm-proxy `dataset_ref` shape.

Core rules (from the plan):

• Positive set = rows where EITHER
    - `user_rating == +1` (explicit thumbs-up from mobile Feedback tab), OR
    - `tool_calls_json` parses cleanly AND `user_rating` is not −1 AND no
      same-intent retry later in the same `conversation_id` within 60s.

• Drop rows where `user_id` is null (unresolved speaker — useless signal).

• Dedup per-user on `(normalized_user_message, tool_name)`. Same phrase from
  two different users is NOT a dup (per the Ultraplan refinement).

• Per-user balance:
    - Users with fewer than `min_per_user` examples → excluded entirely
      (cold-start, will be served by the base model until they accumulate).
    - Top user capped at `max_per_user_multiplier × median_count` via random
      downsample. Prevents a single heavy speaker from monopolizing the adapter.

Consumer-friendly output shape (ExtractedRow dataclass — keeps the service
stateless about prompt format; the CLI layer stamps in variant-randomized
system prompts when it writes JSONL).
"""

from __future__ import annotations

import json
import logging
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models import ConversationTranscript

logger = logging.getLogger("uvicorn")


# --------------------------------------------------------------------------
# Row shape returned by the service. Opaque to consumers.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedRow:
    """One clean training example (pre-format).

    The Phase 4 orchestrator converts these into dataset_ref rows with
    variant-randomized `formatted_system_prompt` and `formatted_completion`.
    """
    user_id: int
    household_id: str
    conversation_id: str
    created_at: datetime
    user_message: str
    tool_call: dict  # {"name": str, "arguments": dict}
    source: str  # "explicit_positive" | "implicit_clean"
    user_rating: int | None


# --------------------------------------------------------------------------
# Normalization + retry detection
# --------------------------------------------------------------------------


_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def normalize_utterance(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used for dedup."""
    if not text:
        return ""
    text = text.lower().strip()
    text = _NON_WORD.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def _first_tool_call(tool_calls_json: str | None) -> dict | None:
    if not tool_calls_json:
        return None
    try:
        parsed = json.loads(tool_calls_json)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict) and first.get("name"):
            return {
                "name": first["name"],
                "arguments": first.get("arguments", {}) or {},
            }
    return None


def _same_intent_retry(
    candidate: ConversationTranscript,
    others_in_convo: list[ConversationTranscript],
    window_seconds: int = 60,
) -> bool:
    """True if the SAME user repeated the same normalized utterance + tool
    within `window_seconds` later in the same conversation.

    When this fires, the candidate is treated as noise (user probably had to
    re-say it because the first parse was wrong). The later row wins.

    Retry detection is scoped per user_id because a multi-speaker household
    may have two users in one conversation echoing similar phrases
    independently — those aren't retries.
    """
    cand_norm = normalize_utterance(candidate.user_message)
    cand_tool = (_first_tool_call(candidate.tool_calls_json) or {}).get("name")
    if not cand_tool:
        return False
    cand_ts = candidate.created_at
    for other in others_in_convo:
        if other.id == candidate.id:
            continue
        if other.user_id != candidate.user_id:
            continue
        if other.created_at <= cand_ts:
            continue
        dt = (other.created_at - cand_ts).total_seconds()
        if dt > window_seconds:
            continue
        other_norm = normalize_utterance(other.user_message)
        other_tool = (_first_tool_call(other.tool_calls_json) or {}).get("name")
        if other_norm == cand_norm and other_tool == cand_tool:
            return True
    return False


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------


class TrainingDataExtractor:
    def __init__(self, db: Session):
        self.db = db

    def extract(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        household_id: str | None = None,
        min_per_user: int = 10,
        max_per_user_multiplier: float = 4.0,
        rng_seed: int = 42,
    ) -> list[ExtractedRow]:
        """Query, filter, dedup, balance. Returns rows ready for formatting."""
        raw = self._query(since=since, until=until, household_id=household_id)
        logger.info("extractor: %d raw rows in window", len(raw))

        positives = self._filter_positive(raw)
        logger.info("extractor: %d after positive-set filter", len(positives))

        deduped = self._dedup_per_user(positives)
        logger.info("extractor: %d after per-user dedup", len(deduped))

        rng = random.Random(rng_seed)
        balanced = self._balance_per_user(
            deduped,
            min_per_user=min_per_user,
            max_per_user_multiplier=max_per_user_multiplier,
            rng=rng,
        )
        logger.info("extractor: %d after per-user balance", len(balanced))

        return balanced

    # --- steps ---------------------------------------------------------

    def _query(
        self,
        *,
        since: datetime | None,
        until: datetime | None,
        household_id: str | None,
    ) -> list[ConversationTranscript]:
        q = self.db.query(ConversationTranscript).filter(
            ConversationTranscript.user_id.isnot(None),
        )
        if since is not None:
            q = q.filter(ConversationTranscript.created_at >= since)
        if until is not None:
            q = q.filter(ConversationTranscript.created_at < until)
        if household_id is not None:
            q = q.filter(ConversationTranscript.household_id == household_id)
        # Thumbs-down is an explicit reject — never include
        q = q.filter(
            (ConversationTranscript.user_rating.is_(None))
            | (ConversationTranscript.user_rating != -1)
        )
        return q.order_by(ConversationTranscript.created_at.asc()).all()

    def _filter_positive(
        self,
        rows: list[ConversationTranscript],
    ) -> list[ExtractedRow]:
        """Apply the positive-set rules + drop unparseable tool calls."""
        # Group by conversation so retry-detection can peek at siblings
        by_convo: dict[str, list[ConversationTranscript]] = {}
        for r in rows:
            by_convo.setdefault(r.conversation_id, []).append(r)

        out: list[ExtractedRow] = []
        for r in rows:
            tc = _first_tool_call(r.tool_calls_json)
            if tc is None:
                continue

            if r.user_rating == 1:
                source = "explicit_positive"
            else:
                # Implicit path: must pass retry check
                if _same_intent_retry(r, by_convo[r.conversation_id]):
                    continue
                source = "implicit_clean"

            if not r.user_message or not r.user_message.strip():
                continue

            out.append(ExtractedRow(
                user_id=r.user_id,
                household_id=r.household_id,
                conversation_id=r.conversation_id,
                created_at=r.created_at,
                user_message=r.user_message.strip(),
                tool_call=tc,
                source=source,
                user_rating=r.user_rating,
            ))
        return out

    def _dedup_per_user(self, rows: list[ExtractedRow]) -> list[ExtractedRow]:
        """Dedup on (user_id, normalized_text, tool_name). Keep the newest."""
        # We iterate newest-last since input is created_at ascending, so
        # later writes into the dict naturally win.
        seen: dict[tuple[int, str, str], ExtractedRow] = {}
        for r in rows:
            key = (
                r.user_id,
                normalize_utterance(r.user_message),
                r.tool_call["name"],
            )
            seen[key] = r
        return list(seen.values())

    def _balance_per_user(
        self,
        rows: list[ExtractedRow],
        *,
        min_per_user: int,
        max_per_user_multiplier: float,
        rng: random.Random,
    ) -> list[ExtractedRow]:
        """Drop cold-start users + downsample the heaviest user."""
        if not rows:
            return rows

        by_user: dict[int, list[ExtractedRow]] = {}
        for r in rows:
            by_user.setdefault(r.user_id, []).append(r)

        # 1. Drop users with < min_per_user
        eligible = {uid: rs for uid, rs in by_user.items() if len(rs) >= min_per_user}
        dropped = set(by_user) - set(eligible)
        if dropped:
            logger.info(
                "extractor: dropped %d cold-start users (< %d examples): %s",
                len(dropped), min_per_user, sorted(dropped),
            )
        if not eligible:
            return []

        # 2. Cap at max_per_user_multiplier × median
        counts = sorted(len(rs) for rs in eligible.values())
        median_count = statistics.median(counts) if counts else 0
        cap = int(max_per_user_multiplier * median_count) if median_count else None

        out: list[ExtractedRow] = []
        for uid, rs in eligible.items():
            if cap is not None and len(rs) > cap:
                sampled = rs.copy()
                rng.shuffle(sampled)
                sampled = sampled[:cap]
                logger.info(
                    "extractor: user %d had %d rows → downsampled to cap %d (median=%d, mult=%s)",
                    uid, len(rs), cap, median_count, max_per_user_multiplier,
                )
                out.extend(sampled)
            else:
                out.extend(rs)
        # Stable sort by time so downstream consumers get deterministic ordering
        out.sort(key=lambda r: (r.created_at, r.user_id))
        return out


# --------------------------------------------------------------------------
# Convenience: format an ExtractedRow as a Phase 4 dataset_ref row.
# Kept separate from the service so the service stays side-effect-free.
# --------------------------------------------------------------------------


def build_response_json(tool_call: dict) -> str:
    """The JSON the model should emit — matches the production response format."""
    payload = {
        "message": "",
        "tool_call": tool_call,
    }
    return " " + json.dumps(payload, ensure_ascii=False)


def to_dataset_ref_row(
    row: ExtractedRow,
    *,
    system_prompt: str,
) -> dict[str, Any]:
    """Convert an ExtractedRow into llm-proxy dataset_ref shape."""
    return {
        "voice_command": row.user_message,
        "expected_tool_call": row.tool_call,
        "formatted_system_prompt": system_prompt,
        "formatted_completion": build_response_json(row.tool_call),
        # Audit metadata — training scripts ignore unknown fields.
        "_meta": {
            "user_id": row.user_id,
            "household_id": row.household_id,
            "conversation_id": row.conversation_id,
            "created_at": row.created_at.isoformat(),
            "source": row.source,
            "user_rating": row.user_rating,
        },
    }
