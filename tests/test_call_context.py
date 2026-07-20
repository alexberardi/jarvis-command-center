"""Per-user call context: what gets loaded, and what may be said.

Two independent controls, tested independently:
  category — is the field in the brief at all (blast radius)
  tier     — once there, may the agent volunteer it
"""

import json

import pytest

from app.services.call_context import (
    GENERAL,
    IF_ASKED,
    MEDICAL,
    STATE,
    build_context_block,
    parse_call_context,
    restricted_values,
    select_fields,
)


def _blob(*entries):
    return json.dumps({"fields": list(entries)})


NAME = {"key": "full_name", "value": "Alex B"}
ADDRESS = {"key": "address", "value": "742 Evergreen Ave, Springfield, IL 62704"}
MEMBER_ID = {"key": "insurance_member_id", "value": "XZ-9912345"}


class TestParsing:
    def test_well_known_keys_get_their_label_category_and_tier(self):
        (f,) = parse_call_context(_blob(MEMBER_ID))
        assert f.label == "Insurance member ID"
        assert f.category == MEDICAL
        assert f.tier == IF_ASKED

    def test_custom_field_defaults_to_if_asked(self):
        """A field someone typed themselves is far likelier to be an account
        number than a pleasantry, so the private side is the safe default."""
        (f,) = parse_call_context(_blob({"key": "gate_code", "value": "4412"}))
        assert f.tier == IF_ASKED
        assert f.category == GENERAL
        assert f.label == "gate_code"

    def test_explicit_values_win_over_defaults(self):
        (f,) = parse_call_context(
            _blob({"key": "gate_code", "value": "4412", "label": "Gate code",
                   "category": MEDICAL, "tier": STATE})
        )
        assert (f.label, f.category, f.tier) == ("Gate code", MEDICAL, STATE)

    def test_bogus_category_or_tier_falls_back_rather_than_raising(self):
        (f,) = parse_call_context(
            _blob({"key": "x", "value": "y", "category": "nope", "tier": "nope"})
        )
        assert f.category == GENERAL and f.tier == IF_ASKED

    @pytest.mark.parametrize("raw", ["", None, "{not json", "[]", '{"fields": {}}'])
    def test_unusable_blobs_yield_nothing(self, raw):
        assert parse_call_context(raw) == []

    def test_one_bad_row_does_not_lose_the_others(self):
        """This blob is user-edited; a single malformed row must not cost
        every other field on the call."""
        fields = parse_call_context(_blob(NAME, "garbage", {"key": "", "value": "x"},
                                          {"key": "k"}, MEMBER_ID))
        assert [f.key for f in fields] == ["full_name", "insurance_member_id"]

    def test_duplicate_keys_keep_the_first(self):
        fields = parse_call_context(
            _blob(NAME, {"key": "full_name", "value": "Someone Else"})
        )
        assert [f.value for f in fields] == ["Alex B"]

    def test_accepts_a_decoded_object_too(self):
        """Settings backends differ on whether they deserialize."""
        assert len(parse_call_context({"fields": [NAME]})) == 1


class TestCategorySelection:
    def test_general_is_always_included(self):
        fields = parse_call_context(_blob(NAME, MEMBER_ID))
        assert [f.key for f in select_fields(fields, None)] == ["full_name"]

    def test_medical_only_when_that_call_asks_for_it(self):
        fields = parse_call_context(_blob(NAME, MEMBER_ID))
        keys = [f.key for f in select_fields(fields, [MEDICAL])]
        assert keys == ["full_name", "insurance_member_id"]

    def test_a_pizza_order_never_carries_the_policy_number(self):
        """The whole point of categories: not in the brief means no prompt
        rule has to hold it back."""
        fields = parse_call_context(_blob(NAME, MEMBER_ID))
        block = build_context_block(select_fields(fields, [])) or ""
        assert "XZ-9912345" not in block

    def test_unknown_category_names_are_ignored_not_fatal(self):
        fields = parse_call_context(_blob(NAME, MEMBER_ID))
        assert [f.key for f in select_fields(fields, ["nonsense", None])] == ["full_name"]


class TestBriefBlock:
    def test_groups_are_labelled_the_way_the_prompt_rules_expect(self):
        fields = parse_call_context(_blob(NAME, ADDRESS, MEMBER_ID))
        block = build_context_block(select_fields(fields, [MEDICAL]))

        assert "you may state these" in block
        assert "never volunteer these" in block
        # Name is statable; address and member id are not.
        state_part, private_part = block.split("Give ONLY if they ask")
        assert "Alex B" in state_part
        assert "Evergreen" in private_part and "XZ-9912345" in private_part

    def test_no_fields_means_no_scaffolding(self):
        assert build_context_block([]) is None

    def test_only_private_fields_still_renders(self):
        fields = parse_call_context(_blob(MEMBER_ID))
        block = build_context_block(select_fields(fields, [MEDICAL]))
        assert "never volunteer" in block and "XZ-9912345" in block


class TestRestrictedValues:
    def test_lists_exactly_the_values_that_must_not_be_volunteered(self):
        """The spoken-output guard consumes this; the prompt asking nicely is
        not the enforcement layer."""
        fields = parse_call_context(_blob(NAME, ADDRESS, MEMBER_ID))
        vals = restricted_values(select_fields(fields, [MEDICAL]))

        assert "XZ-9912345" in vals
        assert "742 Evergreen Ave, Springfield, IL 62704" in vals
        assert "Alex B" not in vals


class TestBriefAssembly:
    """apply_call_context appends to the brief without ever failing the plan."""

    def _patch(self, monkeypatch, blob):
        from app.services import call_context as cc

        monkeypatch.setattr(
            cc, "load_call_context", lambda user_id: cc.parse_call_context(blob)
        )

    def test_general_fields_are_appended_to_the_brief(self, monkeypatch):
        from app.services.phone_call_service import apply_call_context

        self._patch(monkeypatch, _blob(NAME, MEMBER_ID))
        out = apply_call_context("order a pizza", user_id=7)

        assert out.startswith("order a pizza")
        assert "Alex B" in out
        assert "XZ-9912345" not in out  # medical was not requested

    def test_requested_category_reaches_the_brief(self, monkeypatch):
        from app.services.phone_call_service import apply_call_context

        self._patch(monkeypatch, _blob(NAME, MEMBER_ID))
        out = apply_call_context("book a check-up", user_id=7, categories=[MEDICAL])

        assert "XZ-9912345" in out

    def test_no_context_leaves_the_brief_untouched(self, monkeypatch):
        from app.services.phone_call_service import apply_call_context

        self._patch(monkeypatch, "")
        assert apply_call_context("order a pizza", user_id=7) == "order a pizza"

    def test_a_broken_lookup_never_fails_the_plan(self, monkeypatch):
        """Missing context is a worse call; a raised exception is no call."""
        from app.services import call_context as cc
        from app.services.phone_call_service import apply_call_context

        def boom(user_id):
            raise RuntimeError("settings down")

        monkeypatch.setattr(cc, "load_call_context", boom)
        assert apply_call_context("order a pizza", user_id=7) == "order a pizza"

    def test_anonymous_caller_gets_nothing(self, monkeypatch):
        from app.services.phone_call_service import apply_call_context

        assert apply_call_context("order a pizza", user_id=None) == "order a pizza"
