"""Per-user call context: what gets loaded, and what may be said.

Two independent controls, tested independently:
  category — is the field in the brief at all (blast radius)
  tier     — once there, may the agent volunteer it
"""

import json

import pytest

from app.services.call_context import (
    CATEGORIES,
    GENERAL,
    IF_ASKED,
    MEDICAL,
    STATE,
    TIERS,
    build_context_block,
    catalog,
    parse_call_context,
    prepare_for_storage,
    restricted_fields,
    select_fields,
    serialize_fields,
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


PHARMACY = {"key": "pharmacy", "value": "Rite Aid on Main"}  # medical, STATE tier


class TestBriefBlock:
    def test_drops_full_name_and_frames_as_assistant(self):
        """full_name repeated as a caller detail made the model introduce
        itself AS the caller ("I'm Alex Berardi", 5/5 on the box). It is
        already in the disclosure, so it is dropped and the block states the
        model is the caller's assistant, not the caller."""
        fields = parse_call_context(_blob(NAME, ADDRESS, MEMBER_ID))
        block = build_context_block(select_fields(fields, [MEDICAL]))

        assert "Alex B" not in block  # full_name dropped
        assert "you are" in block and "assistant" in block
        assert "never claim to be them" in block

    def test_give_when_asked_group_still_renders(self):
        """The private group leads with "give when asked", not a refusal — a
        refusal-forward header made the model refuse legitimate asks (live
        2026-07-20). It must still forbid volunteering."""
        fields = parse_call_context(_blob(ADDRESS, MEMBER_ID))
        block = build_context_block(select_fields(fields, [MEDICAL]))

        assert "give it directly" in block
        assert "Never volunteer them" in block
        assert "Evergreen" in block and "XZ-9912345" in block

    def test_statable_non_name_details_still_render(self):
        fields = parse_call_context(_blob(PHARMACY))
        block = build_context_block(select_fields(fields, [MEDICAL]))
        assert "You may state these" in block
        assert "Rite Aid on Main" in block

    def test_no_fields_means_no_scaffolding(self):
        assert build_context_block([]) is None

    def test_a_block_with_only_full_name_disappears(self):
        """Dropping full_name can empty the block — then there is nothing to
        render, not empty scaffolding."""
        assert build_context_block(parse_call_context(_blob(NAME))) is None

    def test_only_private_fields_still_renders(self):
        fields = parse_call_context(_blob(MEMBER_ID))
        block = build_context_block(select_fields(fields, [MEDICAL]))
        assert "Never volunteer them" in block and "XZ-9912345" in block


class TestRestrictedFields:
    def test_lists_exactly_the_fields_that_must_not_be_volunteered(self):
        """The spoken-output guard consumes this; the prompt asking nicely is
        not the enforcement layer."""
        fields = parse_call_context(_blob(NAME, ADDRESS, MEMBER_ID))
        restricted = restricted_fields(select_fields(fields, [MEDICAL]))
        values = [f.value for f in restricted]

        assert "XZ-9912345" in values
        assert "742 Evergreen Ave, Springfield, IL 62704" in values
        assert "Alex B" not in values

    def test_carries_labels_so_the_guard_never_needs_the_value(self):
        """The guard asks "did they ask for Insurance member ID?" — a question
        answerable from the label alone. That is what keeps the secret out of
        the classifier prompt, so a callee who manipulates it learns nothing."""
        restricted = restricted_fields(parse_call_context(_blob(MEMBER_ID)))

        assert [f.label for f in restricted] == ["Insurance member ID"]
        assert [f.key for f in restricted] == ["insurance_member_id"]

    def test_a_custom_field_is_restricted_by_default(self):
        """Unknown keys default to IF_ASKED, so a hand-typed "Gate code" is
        guarded without the user having to know the tier system exists."""
        custom = {"key": "gate_code", "label": "Gate code", "value": "4417"}
        restricted = restricted_fields(parse_call_context(_blob(custom)))

        assert [(f.label, f.value) for f in restricted] == [("Gate code", "4417")]


class TestBriefAssembly:
    """apply_call_context appends to the brief without ever failing the plan."""

    def _patch(self, monkeypatch, blob):
        from app.services import call_context as cc

        monkeypatch.setattr(
            cc, "load_call_context", lambda user_id: cc.parse_call_context(blob)
        )

    def test_general_fields_are_appended_to_the_brief(self, monkeypatch):
        from app.services.phone_call_service import apply_call_context

        callback = {"key": "callback_number", "value": "+15555550123"}
        self._patch(monkeypatch, _blob(callback, MEMBER_ID))
        out = apply_call_context("order a pizza", user_id=7)

        assert out.startswith("order a pizza")
        assert "+15555550123" in out  # general field reaches the brief
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


class TestWritePath:
    """The grid saves rows; storage needs keys and a canonical blob."""

    def test_custom_field_gets_a_key_derived_from_its_label(self):
        """The grid lets a user type just a label and a value for a custom
        field. Storage needs a stable key, and deriving it server-side keeps
        one source of truth."""
        fields = prepare_for_storage([{"label": "Gate code", "value": "4417"}])
        assert [(f.key, f.label, f.value) for f in fields] == [
            ("gate_code", "Gate code", "4417")
        ]

    def test_a_well_known_key_survives_unchanged(self):
        fields = prepare_for_storage(
            [{"key": "insurance_member_id", "value": "XZ-1"}]
        )
        assert fields[0].key == "insurance_member_id"
        assert fields[0].label == "Insurance member ID"
        assert fields[0].category == MEDICAL
        assert fields[0].tier == IF_ASKED

    def test_blank_and_keyless_rows_drop_out(self):
        fields = prepare_for_storage(
            [
                {"label": "", "value": ""},           # nothing to key on
                {"label": "  ", "value": "x"},         # slug empties -> no key
                {"label": "Real", "value": "keep me"},
            ]
        )
        assert [f.label for f in fields] == ["Real"]

    def test_serialize_round_trips_through_parse(self):
        fields = prepare_for_storage(
            [
                {"key": "full_name", "value": "Alex B"},
                {"label": "Rewards number", "value": "99887766"},
            ]
        )
        reparsed = parse_call_context(serialize_fields(fields))
        assert [f.key for f in reparsed] == ["full_name", "rewards_number"]
        assert [f.value for f in reparsed] == ["Alex B", "99887766"]

    def test_serialize_emits_a_json_string_not_a_dict(self):
        """The setting is value_type=string, so the client stores the value
        verbatim — it must already be JSON text or the reader can't parse it."""
        blob = serialize_fields(prepare_for_storage([{"key": "x", "value": "y"}]))
        assert isinstance(blob, str)
        assert json.loads(blob) == {
            "fields": [
                {"key": "x", "label": "x", "value": "y",
                 "category": GENERAL, "tier": IF_ASKED}
            ]
        }


class TestCatalog:
    """The static vocabulary served to the grid, so the app can't drift."""

    def test_covers_every_category_and_tier(self):
        cat = catalog()
        assert {c["value"] for c in cat["categories"]} == set(CATEGORIES)
        assert {t["value"] for t in cat["tiers"]} == set(TIERS)

    def test_well_known_fields_carry_their_controls(self):
        by_key = {f["key"]: f for f in catalog()["well_known"]}
        assert by_key["insurance_member_id"]["category"] == MEDICAL
        assert by_key["insurance_member_id"]["tier"] == IF_ASKED
        assert by_key["full_name"]["tier"] == STATE
