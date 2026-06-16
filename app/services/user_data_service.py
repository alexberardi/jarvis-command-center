"""Self-scoped user-data purge.

Deletes every row in command-center keyed to a single ``user_id``. Used by the
account-deletion flow: jarvis-auth orchestrates the delete and fans out to this
service via ``DELETE /api/v0/me/data`` (forwarding the user's Bearer token).

The purge is idempotent — a user with no rows still completes cleanly — and runs
in a single transaction so a mid-purge failure rolls everything back rather than
leaving the user's data half-deleted.
"""

import logging

from sqlalchemy.orm import Session

from app.models import AuthSession, ConversationTranscript, Setting, UserMemory

logger = logging.getLogger("uvicorn")


def purge_user_data(db: Session, user_id: int) -> dict[str, int]:
    """Delete all command-center rows keyed to ``user_id``.

    Removes the user's memories, conversation transcripts, user-scoped settings,
    and OAuth auth-sessions (whose ``user_id`` owns the user-scoped tokens). House-
    hold-wide and node-scoped rows (``user_id IS NULL``) are deliberately left
    untouched.

    Wrapped in a single transaction: on any failure the whole purge rolls back.
    Returns a per-table count of deleted rows (useful for logging / assertions).
    """
    try:
        counts = {
            "user_memories": (
                db.query(UserMemory)
                .filter(UserMemory.user_id == user_id)
                .delete(synchronize_session=False)
            ),
            "conversation_transcripts": (
                db.query(ConversationTranscript)
                .filter(ConversationTranscript.user_id == user_id)
                .delete(synchronize_session=False)
            ),
            "settings": (
                db.query(Setting)
                .filter(Setting.user_id == user_id)
                .delete(synchronize_session=False)
            ),
            "auth_sessions": (
                db.query(AuthSession)
                .filter(AuthSession.user_id == user_id)
                .delete(synchronize_session=False)
            ),
        }
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to purge data for user_id=%s", user_id)
        raise

    logger.info(
        "Purged user data for user_id=%s: %s",
        user_id,
        ", ".join(f"{table}={n}" for table, n in counts.items()),
    )
    return counts
