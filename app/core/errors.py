"""Typed domain errors for the core conversation flow.

Kept deliberately small. `ConversationPreconditionError` lets the blocking
command endpoints distinguish a *whole-request precondition failure*
(e.g. an un-started / expired conversation) — which must surface as HTTP 422 —
from a genuine internal/processing error, without string-matching on message
text. It subclasses `ValueError` so existing `except ValueError` callers
continue to handle it.
"""


class ConversationPreconditionError(ValueError):
    """Raised when a request cannot be processed because a conversation-level
    precondition is not met (e.g. the conversation was never started, or has
    expired). The blocking endpoints map this to HTTP 422 Unprocessable Entity.
    """
