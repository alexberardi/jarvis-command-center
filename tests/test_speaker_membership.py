"""Tests for app/core/utils/speaker_membership.py.

Security (intra-household IDOR half in CC): a node asserts speaker_user_id and CC
scopes memories + transcripts to it. Validate the asserted id against the node's
validated household membership so a node can't attribute activity to — or read
the memories of — a user outside its household.
"""
from __future__ import annotations

from app.core.utils.speaker_membership import validated_speaker_user_id


class TestValidatedSpeakerUserId:
    def test_member_is_accepted(self):
        assert validated_speaker_user_id(2, [1, 2, 3], "node-1") == 2

    def test_non_member_is_rejected(self):
        assert validated_speaker_user_id(99, [1, 2, 3], "node-1") is None

    def test_string_member_is_coerced_to_int(self):
        assert validated_speaker_user_id("2", [1, 2, 3], "node-1") == 2

    def test_string_non_member_is_rejected(self):
        assert validated_speaker_user_id("99", [1, 2, 3], "node-1") is None

    def test_none_speaker_passes_through_as_none(self):
        assert validated_speaker_user_id(None, [1, 2, 3], "node-1") is None

    def test_empty_membership_fails_open(self):
        # Legacy local node — membership unknown, preserve prior behavior.
        assert validated_speaker_user_id(99, [], "node-1") == 99

    def test_none_membership_fails_open(self):
        assert validated_speaker_user_id(7, None, "node-1") == 7

    def test_unparseable_speaker_with_known_membership_is_rejected(self):
        assert validated_speaker_user_id("not-a-number", [1, 2, 3], "node-1") is None
