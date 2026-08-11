"""SignalService — the Signal Bus store.

UPSERT-on-``source_key`` (flaps collapse to one row), TTL cleanup, and the
cacheable write-boundary rule that keeps live values out of the byte-stable
cached prefix. Session-per-call shape mirrors MemoryService.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.models import Signal

logger = logging.getLogger("uvicorn")

# A cacheable Signal may ride the byte-stable prefix, so it must be quantized /
# absolute. These patterns are the things that would vary the prefix per request
# and poison the prefix cache / TTFS (the regression fixed in #100 / 35c9b8d).
_FLOAT_RE = re.compile(r"\d+\.\d+")
_SECONDS_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")
_RELATIVE_TIME_RE = re.compile(
    r"\b(ago|just now|right now|moments? ago|minutes?|seconds?|in \d+ min)\b",
    re.IGNORECASE,
)


class CacheableSignalError(ValueError):
    """Raised when a cacheable=True Signal carries a non-quantized / live value."""


class SignalService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def save_signal(
        self,
        *,
        household_id: str,
        source_key: str,
        kind: str,
        subject: str | None = None,
        summary: str | None = None,
        facts: dict[str, Any] | None = None,
        user_id: int | None = None,
        node_id: str | None = None,
        room: str | None = None,
        ttl_seconds: int | None = None,
        cacheable: bool = False,
        salience: float | None = None,
        source_agent: str | None = None,
        observed_at: datetime | None = None,
    ) -> Signal:
        """Insert or UPSERT (on household_id + source_key) a Signal.

        A cacheable Signal is validated at this write boundary — a live float,
        a seconds-precision time, or a relative-time phrase raises
        CacheableSignalError.
        """
        if cacheable:
            self._reject_non_quantized(summary, facts)

        facts_json = json.dumps(facts) if facts is not None else None
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None
        observed = observed_at or now

        existing = (
            self.db.query(Signal)
            .filter(
                Signal.household_id == household_id,
                Signal.source_key == source_key,
                Signal.is_active == True,  # noqa: E712
            )
            .first()
        )
        if existing is not None:
            existing.kind = kind
            existing.subject = subject
            existing.summary = summary
            existing.facts = facts_json
            existing.user_id = user_id
            existing.node_id = node_id
            existing.room = room
            existing.cacheable = cacheable
            existing.salience = salience
            existing.source_agent = source_agent
            existing.observed_at = observed
            existing.expires_at = expires_at
            existing.updated_at = now
            self.db.commit()
            self.db.refresh(existing)
            return existing

        signal = Signal(
            household_id=household_id,
            source_key=source_key,
            kind=kind,
            subject=subject,
            summary=summary,
            facts=facts_json,
            user_id=user_id,
            node_id=node_id,
            room=room,
            cacheable=cacheable,
            salience=salience,
            source_agent=source_agent,
            observed_at=observed,
            expires_at=expires_at,
        )
        self.db.add(signal)
        self.db.commit()
        self.db.refresh(signal)
        return signal

    def cleanup_expired(self) -> int:
        """Hard-delete expired Signals (expires_at in the past). NULL never expires."""
        now = datetime.utcnow()
        q = self.db.query(Signal).filter(
            Signal.expires_at.isnot(None),
            Signal.expires_at <= now,
        )
        count = q.count()
        if count:
            q.delete(synchronize_session=False)
            self.db.commit()
            logger.info("Cleaned up %d expired signals", count)
        return count

    @staticmethod
    def _reject_non_quantized(
        summary: str | None, facts: dict[str, Any] | None
    ) -> None:
        blob = " ".join(
            part
            for part in (summary, json.dumps(facts) if facts is not None else None)
            if part
        )
        if (
            _FLOAT_RE.search(blob)
            or _SECONDS_TIME_RE.search(blob)
            or _RELATIVE_TIME_RE.search(blob)
        ):
            raise CacheableSignalError(
                "cacheable Signal must be quantized/absolute — it may not contain a "
                "live float, a seconds-precision time, or a relative-time phrase"
            )
