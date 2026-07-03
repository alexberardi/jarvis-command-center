"""Validate a client-asserted ``speaker_user_id`` against household membership.

The node asserts ``speaker_user_id`` from on-device voice ID, and command-center
uses it to load that user's persistent memories into the prompt and to attribute
transcripts (which feed passive memory extraction). Those are per-user scopes, so
a compromised or misconfigured node must not be able to read or write another
user's memories/transcripts by claiming an arbitrary ``user_id``.

We only honor a speaker who is a member of the node's *validated* household. The
member list comes from the jarvis-auth node-validation response
(``household_member_ids`` on ``NodeContextProvider``) — an in-memory check with no
extra round-trip on the hot path. When membership is unknown (legacy local-DB
nodes carry an empty list), we fail open to preserve existing behavior.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validated_speaker_user_id(
    speaker_user_id: object,
    household_member_ids: list[int] | None,
    node_id: str | None = None,
) -> object:
    """Return ``speaker_user_id`` only if it belongs to the node's household.

    Args:
        speaker_user_id: The client-asserted speaker id (may be int, str, or None).
        household_member_ids: User ids that belong to the node's validated
            household. Empty/None means membership is unknown (legacy node).
        node_id: For logging the rejected node.

    Returns:
        The validated speaker id as an ``int`` when membership is known and the
        speaker is a member; the original value when membership is unknown
        (fail-open); or ``None`` when the speaker is not a household member.
    """
    if speaker_user_id is None:
        return None

    member_ids = household_member_ids or []
    if not member_ids:
        # Membership unknown (e.g. legacy local node) — preserve prior behavior.
        return speaker_user_id

    try:
        sid = int(speaker_user_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        sid = None

    if sid is None or sid not in member_ids:
        logger.warning(
            "🚫 Ignoring speaker_user_id=%r — not a member of node %s's household %r",
            speaker_user_id,
            node_id,
            member_ids,
        )
        return None

    return sid
