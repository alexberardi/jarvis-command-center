"""Characterization service — CRUD for the per-person synthesized view.

One `PersonCharacterization` row per (user_id, household_id): a single evolving
document (structured JSON body + rendered prose) that a background job
regenerates from the user's facts + recent transcripts. This module only
persists/reads — it never calls the LLM (see characterization_synthesis_service
for the synthesis job).
"""

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models import PersonCharacterization

logger = logging.getLogger("uvicorn")


class CharacterizationService:
    """CRUD for a person's synthesized characterization."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, user_id: int, household_id: str) -> PersonCharacterization | None:
        """Return the current characterization for a person, or None."""
        return (
            self.db.query(PersonCharacterization)
            .filter(
                PersonCharacterization.user_id == user_id,
                PersonCharacterization.household_id == household_id,
            )
            .first()
        )

    def upsert(
        self,
        user_id: int,
        household_id: str,
        body: dict[str, Any],
        rendered: str | None = None,
        confidence: float | None = None,
        last_transcript_at: datetime | None = None,
        model: str | None = None,
    ) -> PersonCharacterization:
        """Create or update a person's characterization, bumping `version` on update.

        `body` is the structured characterization dict; it is JSON-serialized for
        storage. Because the characterization is always re-derived from the
        immutable facts, an overwrite here is safe — history is captured only via
        the monotonically increasing `version`.
        """
        body_json = json.dumps(body, ensure_ascii=False)
        existing = self.get(user_id, household_id)

        if existing:
            existing.body = body_json
            existing.rendered = rendered
            existing.confidence = confidence
            existing.last_transcript_at = last_transcript_at
            existing.model = model
            existing.version = (existing.version or 1) + 1
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            logger.info(
                "Updated characterization for user_id=%s -> v%d", user_id, existing.version
            )
            return existing

        row = PersonCharacterization(
            user_id=user_id,
            household_id=household_id,
            body=body_json,
            rendered=rendered,
            confidence=confidence,
            last_transcript_at=last_transcript_at,
            model=model,
            version=1,
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        logger.info("Created characterization for user_id=%s (v1)", user_id)
        return row
