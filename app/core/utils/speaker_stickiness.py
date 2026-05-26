"""Per-node speaker stickiness for short follow-up utterances.

When a wake-word + utterance is too short for the speaker recognizer to ID
confidently (Whisper returns user_id=None), but the same node identified
speaker X with high confidence a moment ago, inherit X for this turn. This
solves the "Hey Jarvis, delete it" follow-up to "Hey Jarvis, do I have any
emails" — where the second utterance is too short to recognize on its own
but the user is obviously the same person.

Scoped per-node because each node is room-scoped, so the "same speaker as
30 seconds ago" assumption is sound. Cross-node identification would have
to deal with people moving between rooms, which we're not solving here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


# Fallback defaults — used when the settings service isn't reachable
# (tests, early startup, etc.). The live values are read from the
# voice.stickiness_min_confidence and voice.stickiness_ttl_seconds
# settings on every call so admin changes apply without a restart.
DEFAULT_STICKINESS_TTL_SECONDS: float = 30.0
DEFAULT_STICKINESS_MIN_CONFIDENCE: float = 0.55


def _stickiness_min_confidence() -> float:
    try:
        from app.services.settings_service import get_settings_service
        return get_settings_service().get_float(
            "voice.stickiness_min_confidence", DEFAULT_STICKINESS_MIN_CONFIDENCE
        )
    except Exception:
        return DEFAULT_STICKINESS_MIN_CONFIDENCE


def _stickiness_ttl_seconds() -> float:
    try:
        from app.services.settings_service import get_settings_service
        return get_settings_service().get_float(
            "voice.stickiness_ttl_seconds", DEFAULT_STICKINESS_TTL_SECONDS
        )
    except Exception:
        return DEFAULT_STICKINESS_TTL_SECONDS


@dataclass
class SpeakerSnapshot:
    user_id: int
    confidence: float
    timestamp: datetime


# Module-level history. CC is single-process per the conversation_cache
# pattern, so this is safe without locking.
_node_speaker_history: dict[str, SpeakerSnapshot] = {}


def record_speaker_for_node(node_id: str, user_id: int | None, confidence: float | None) -> None:
    """Remember a high-confidence speaker identification on this node.

    Stores nothing if confidence is below threshold — we only want to
    inherit identifications we trust. If user_id is None (no match),
    nothing is recorded.
    """
    if user_id is None:
        return
    min_conf = _stickiness_min_confidence()
    if confidence is None or confidence < min_conf:
        logger.debug(
            "Not recording speaker for stickiness: confidence %s below %.2f",
            confidence,
            min_conf,
        )
        return
    _node_speaker_history[node_id] = SpeakerSnapshot(
        user_id=user_id,
        confidence=float(confidence),
        timestamp=datetime.now(timezone.utc),
    )
    logger.debug(
        "Recorded speaker for stickiness: node=%s user_id=%s confidence=%.3f",
        node_id,
        user_id,
        confidence,
    )


def inherit_speaker_for_node(node_id: str) -> int | None:
    """Return a sticky speaker user_id if recent and confident enough.

    Returns None if no snapshot exists, the snapshot is stale, or the
    recorded confidence was below threshold. Stale entries are removed
    on access (lazy cleanup).
    """
    snap = _node_speaker_history.get(node_id)
    if snap is None:
        return None
    age = datetime.now(timezone.utc) - snap.timestamp
    if age > timedelta(seconds=_stickiness_ttl_seconds()):
        _node_speaker_history.pop(node_id, None)
        return None
    if snap.confidence < _stickiness_min_confidence():
        # Belt-and-suspenders — record_speaker_for_node already gates on this
        return None
    return snap.user_id


def reset_node_history(node_id: str) -> None:
    """Forget any sticky speaker for a node — used on long silence or room change."""
    _node_speaker_history.pop(node_id, None)


def clear_all() -> None:
    """Wipe all node history. For tests and service restarts."""
    _node_speaker_history.clear()
