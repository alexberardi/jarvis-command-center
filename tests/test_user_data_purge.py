"""Account-deletion purge must also remove the user's request traces.

Privacy (P2.4): request_traces rows hold the raw spoken utterance in
``user_command``. The purge used to delete memories/transcripts/settings but
left traces behind, so a "permanent delete" still retained raw utterances. Traces
have no user_id — only conversation_id — so they're mapped to the user via their
transcripts and purged for those conversations.
"""
from app.models import (
    ConversationTranscript,
    RequestTrace,
    Setting,
    UserMemory,
)
from app.services.user_data_service import purge_user_data

HH = "hh-purge-test"


def _transcript(db, user_id, conversation_id):
    db.add(ConversationTranscript(
        user_id=user_id, household_id=HH, conversation_id=conversation_id,
        user_message="turn on the lights",
    ))


def _trace(db, trace_id, conversation_id):
    db.add(RequestTrace(
        id=trace_id, conversation_id=conversation_id, request_type="voice_command",
        source="node", total_duration_ms=12.0, spans_json="[]",
    ))


def test_purge_deletes_users_request_traces(test_db):
    _transcript(test_db, user_id=100, conversation_id="conv-100")
    _trace(test_db, "trace-100", "conv-100")
    test_db.add(UserMemory(user_id=100, household_id=HH, content="my name is alex"))
    test_db.add(Setting(key="k", value_type="string", category="general", user_id=100))
    test_db.commit()

    counts = purge_user_data(test_db, user_id=100)

    assert counts["request_traces"] == 1
    assert test_db.query(RequestTrace).filter(RequestTrace.id == "trace-100").first() is None
    assert test_db.query(ConversationTranscript).filter(
        ConversationTranscript.user_id == 100
    ).first() is None
    assert test_db.query(UserMemory).filter(UserMemory.user_id == 100).first() is None


def test_purge_leaves_other_users_traces(test_db):
    _transcript(test_db, user_id=100, conversation_id="conv-100")
    _trace(test_db, "trace-100", "conv-100")
    _transcript(test_db, user_id=200, conversation_id="conv-200")
    _trace(test_db, "trace-200", "conv-200")
    test_db.commit()

    purge_user_data(test_db, user_id=100)

    # The other user's trace + transcript survive.
    assert test_db.query(RequestTrace).filter(RequestTrace.id == "trace-200").first() is not None
    assert test_db.query(ConversationTranscript).filter(
        ConversationTranscript.user_id == 200
    ).first() is not None
    assert test_db.query(RequestTrace).filter(RequestTrace.id == "trace-100").first() is None


def test_purge_with_no_conversations_is_clean(test_db):
    # A user with memories but no transcripts/traces still purges cleanly.
    test_db.add(UserMemory(user_id=300, household_id=HH, content="note"))
    test_db.commit()

    counts = purge_user_data(test_db, user_id=300)

    assert counts["request_traces"] == 0
    assert counts["user_memories"] == 1
