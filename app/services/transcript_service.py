"""Transcript buffer service for passive memory extraction.

Manages conversation_transcripts rows: logging, batch retrieval,
in-flight tracking, and TTL cleanup.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models import ConversationTranscript

logger = logging.getLogger("uvicorn")


class TranscriptService:
    def __init__(self, db: Session):
        self.db = db

    def log_transcript(
        self,
        user_id: int,
        household_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> ConversationTranscript:
        """Insert a transcript row. Called fire-and-forget from the voice path."""
        row = ConversationTranscript(
            user_id=user_id,
            household_id=household_id,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
            tool_calls_json=json.dumps(tool_calls) if tool_calls else None,
            created_at=datetime.utcnow(),
        )
        self.db.add(row)
        self.db.commit()
        return row

    def get_users_with_unprocessed(self) -> list[tuple[int, str]]:
        """Return distinct (user_id, household_id) pairs with unprocessed transcripts
        that are not already in-flight."""
        rows = (
            self.db.query(
                ConversationTranscript.user_id,
                ConversationTranscript.household_id,
            )
            .filter(
                and_(
                    ConversationTranscript.is_processed.is_(False),
                    ConversationTranscript.extraction_job_id.is_(None),
                )
            )
            .distinct()
            .all()
        )
        return [(r.user_id, r.household_id) for r in rows]

    def get_unprocessed_for_user(
        self, user_id: int, household_id: str, limit: int = 20
    ) -> list[ConversationTranscript]:
        """Fetch unprocessed transcripts for a user, oldest first."""
        return (
            self.db.query(ConversationTranscript)
            .filter(
                and_(
                    ConversationTranscript.user_id == user_id,
                    ConversationTranscript.household_id == household_id,
                    ConversationTranscript.is_processed.is_(False),
                    ConversationTranscript.extraction_job_id.is_(None),
                )
            )
            .order_by(ConversationTranscript.created_at.asc())
            .limit(limit)
            .all()
        )

    def mark_extraction_in_flight(
        self, transcript_ids: list[int], job_id: str
    ) -> None:
        """Stamp extraction_job_id so these transcripts are not re-picked-up."""
        (
            self.db.query(ConversationTranscript)
            .filter(ConversationTranscript.id.in_(transcript_ids))
            .update(
                {ConversationTranscript.extraction_job_id: job_id},
                synchronize_session="fetch",
            )
        )
        self.db.commit()

    def mark_processed(self, job_id: str) -> int:
        """Mark all transcripts with this job_id as processed. Returns count."""
        count = (
            self.db.query(ConversationTranscript)
            .filter(ConversationTranscript.extraction_job_id == job_id)
            .update(
                {
                    ConversationTranscript.is_processed: True,
                    ConversationTranscript.processed_at: datetime.utcnow(),
                },
                synchronize_session="fetch",
            )
        )
        self.db.commit()
        return count

    def reset_stale_jobs(self, max_age_minutes: int = 30) -> int:
        """Reset extraction_job_id for jobs that never completed (callback never arrived)."""
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        count = (
            self.db.query(ConversationTranscript)
            .filter(
                and_(
                    ConversationTranscript.extraction_job_id.isnot(None),
                    ConversationTranscript.is_processed.is_(False),
                    ConversationTranscript.created_at < cutoff,
                )
            )
            .update(
                {ConversationTranscript.extraction_job_id: None},
                synchronize_session="fetch",
            )
        )
        if count:
            self.db.commit()
            logger.info("Reset %d stale extraction jobs", count)
        return count

    def cleanup_expired(self, ttl_days: int) -> int:
        """Delete transcripts older than ttl_days. Returns count deleted."""
        cutoff = datetime.utcnow() - timedelta(days=ttl_days)
        count = (
            self.db.query(ConversationTranscript)
            .filter(ConversationTranscript.created_at < cutoff)
            .delete(synchronize_session="fetch")
        )
        if count:
            self.db.commit()
        return count

    # ------------------------------------------------------------------
    # Phase 1: user-facing rating API
    # ------------------------------------------------------------------

    def list_recent_for_user(
        self,
        user_id: int,
        limit: int = 50,
        since: datetime | None = None,
    ) -> list[ConversationTranscript]:
        """Return the user's most-recent transcripts, newest first.

        Used by the mobile "Recent Commands" feedback screen. Caller is responsible
        for authorizing the request (verify_user_jwt → user_id).
        """
        q = self.db.query(ConversationTranscript).filter(
            ConversationTranscript.user_id == user_id
        )
        if since is not None:
            q = q.filter(ConversationTranscript.created_at >= since)
        return (
            q.order_by(ConversationTranscript.created_at.desc())
            .limit(limit)
            .all()
        )

    def rate_transcript(
        self,
        transcript_id: int,
        user_id: int,
        rating: int,
        notes: str | None = None,
    ) -> ConversationTranscript | None:
        """Record an explicit rating on a transcript.

        `rating` must be −1 / 0 / +1. Returns the updated row, or None if the
        transcript doesn't exist or doesn't belong to `user_id` (opaque 404 at the
        API layer — don't leak existence of other users' rows).
        """
        if rating not in (-1, 0, 1):
            raise ValueError(f"rating must be -1, 0, or 1; got {rating!r}")

        row = (
            self.db.query(ConversationTranscript)
            .filter(
                and_(
                    ConversationTranscript.id == transcript_id,
                    ConversationTranscript.user_id == user_id,
                )
            )
            .first()
        )
        if not row:
            return None

        row.user_rating = rating
        row.rating_notes = notes
        row.rated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(row)
        return row
