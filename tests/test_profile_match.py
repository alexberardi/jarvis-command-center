"""Tests for the profile-match hint — restating matched memories at recency.

2026-07-27 prod, "Who is Leo?": the 379-char User Profile block sat directly
adjacent to the question — "Has golden doodle dogs named Leo and Groot who
are brothers" — and the quantized no-think 14B still answered "I don't have
any information about someone named Leo in your profile." Its /think trace
even narrated "let me check the profile" without reading it. Against a 38KB
system prompt the model's mid-context attention is unreliable; the only
dependable position is inside the LAST message. So: when the utterance's
content words overlap a profile line, restate that line in the user turn.
"""

from app.core.profile_match import build_profile_match_hint


PROFILE_BLOCK = (
    "You are speaking with Alex.\n\n"
    "User Profile - If user asks a question about one of these items "
    "you must respond with an answer:\n"
    "- Interested in Witcher 3 builds\n"
    "- Enjoys Red Hot Chili Peppers\n"
    "- Follows the New York Yankees\n"
    "- Has golden doodle dogs named Leo and Groot who are brothers\n"
    "- Kaitlyn is Alex's wife.\n"
    "- Miles is Alex's son and he is one years old.\n"
    "- Resident of Brick, NJ"
)


class TestMatching:
    def test_who_is_leo_matches_the_dog_line(self):
        hint = build_profile_match_hint("Who is Leo?", PROFILE_BLOCK)
        assert hint is not None
        assert hint.startswith("[profile match:")
        assert "Leo and Groot" in hint

    def test_sons_name_matches_the_son_line(self):
        hint = build_profile_match_hint("What's my son's name?", PROFILE_BLOCK)
        assert hint is not None
        assert "Miles" in hint

    def test_dogs_named_matches(self):
        hint = build_profile_match_hint("What are my dogs named?", PROFILE_BLOCK)
        assert hint is not None
        assert "Leo and Groot" in hint

    def test_possessive_still_matches(self):
        # "Leo's" must match the "Leo" token.
        hint = build_profile_match_hint("When is Leo's birthday?", PROFILE_BLOCK)
        assert hint is not None
        assert "Leo and Groot" in hint

    def test_case_insensitive(self):
        hint = build_profile_match_hint("who is LEO", PROFILE_BLOCK)
        assert hint is not None

    def test_multiple_matches_included(self):
        hint = build_profile_match_hint(
            "Tell me about Kaitlyn and Miles", PROFILE_BLOCK
        )
        assert hint is not None
        assert "Kaitlyn" in hint and "Miles" in hint


class TestNoMatch:
    def test_unrelated_question_returns_none(self):
        # No profile term in the utterance → no hint, prompt unchanged.
        assert build_profile_match_hint("Turn on the lights", PROFILE_BLOCK) is None

    def test_stopwords_never_match(self):
        # "who", "is", "the", "and" appear in profile lines but are noise.
        assert build_profile_match_hint("who is the and", PROFILE_BLOCK) is None

    def test_empty_inputs(self):
        assert build_profile_match_hint("", PROFILE_BLOCK) is None
        assert build_profile_match_hint("Who is Leo?", "") is None
        assert build_profile_match_hint("Who is Leo?", None) is None

    def test_block_without_profile_lines(self):
        assert build_profile_match_hint("Who is Leo?", "You are speaking with Alex.") is None


class TestShape:
    def test_single_bracketed_line(self):
        hint = build_profile_match_hint("Who is Leo?", PROFILE_BLOCK)
        assert hint.startswith("[profile match:") and hint.endswith("]")
        assert "\n" not in hint

    def test_capped_line_count(self):
        # A word that hits many lines must not paste the whole profile back.
        block = "User Profile - answer:\n" + "\n".join(
            f"- Fact number {i} about zebras" for i in range(10)
        )
        hint = build_profile_match_hint("tell me about zebras", block)
        assert hint is not None
        assert hint.count("zebras") <= 3
