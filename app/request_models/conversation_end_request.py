"""Conversation end request model.

Sent by the node when a wake cycle completes (no further follow-up
expected within this conversation). Currently used to clear per-node
speaker stickiness so identifications don't leak across wake events,
but the endpoint exists as a generic conversation-lifecycle hook —
future cleanups (cache eviction, transcript finalization, etc.) can
piggy-back here without adding a new endpoint.
"""

from pydantic import BaseModel


class ConversationEndRequest(BaseModel):
    conversation_id: str
