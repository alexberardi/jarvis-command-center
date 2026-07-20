"""Web-search fallback for a business's phone number.

The governing rule: this saves the user typing, it is never trusted. Every
path either produces a number the user will eyeball on an editable card, or
an honest reason there isn't one — never a blocked plan, never a guess.
"""

from unittest.mock import patch

import pytest

from app.services.phone_number_search import (
    NumberSearchResult,
    extract_address,
    extract_number,
    find_business_number,
    web_search_enabled,
)

HH = "hh-search"
SEARCH = "app.services.phone_number_search"


def _settings(value):
    """Fake get_settings_service factory returning `value` for the gate key."""
    class _S:
        def get(self, key, household_id=None):
            return value

    return lambda: _S()


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class TestGate:
    def test_enabled_when_true(self):
        with patch("app.services.settings_service.get_settings_service", _settings(True)):
            assert web_search_enabled(HH) is True

    def test_string_true_accepted(self):
        with patch("app.services.settings_service.get_settings_service", _settings("true")):
            assert web_search_enabled(HH) is True

    def test_disabled_when_false(self):
        with patch("app.services.settings_service.get_settings_service", _settings(False)):
            assert web_search_enabled(HH) is False

    def test_no_household_is_disabled(self):
        assert web_search_enabled(None) is False
        assert web_search_enabled("") is False

    def test_settings_error_fails_closed(self):
        """Any doubt means no outbound egress."""
        def _boom():
            raise RuntimeError("settings down")

        with patch("app.services.settings_service.get_settings_service", _boom):
            assert web_search_enabled(HH) is False


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------


class TestExtractNumber:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Call us at (732) 592-4183 today", "+17325924183"),
            ("Phone: 732-592-4183", "+17325924183"),
            ("Tel 732.592.4183", "+17325924183"),
            ("+1 732 592 4183", "+17325924183"),
            ("Reach us: 1-732-592-4183", "+17325924183"),
        ],
    )
    def test_common_listing_formats(self, text, expected):
        assert extract_number(text) == expected

    def test_first_valid_number_wins(self):
        text = "Fax 900-555-1212 or call (732) 592-4183"
        # The premium-rate fax is rejected by the shared validator, so the
        # real number is returned rather than the first regex hit.
        assert extract_number(text) == "+17325924183"

    def test_emergency_number_never_returned(self):
        assert extract_number("In an emergency dial 911") is None

    def test_bare_digit_runs_ignored(self):
        """Order numbers and zip+4 must not read as phone numbers."""
        assert extract_number("Order 7325924183 shipped") is None

    def test_no_number_returns_none(self):
        assert extract_number("We are open 9 to 5 daily") is None
        assert extract_number("") is None


class TestExtractAddress:
    def test_street_address_found(self):
        text = "Visit us at 33 National Ave, Brick, NJ 08724 today"
        assert "33 National Ave" in (extract_address(text) or "")

    def test_no_address_returns_none(self):
        assert extract_address("Call for hours") is None


# ---------------------------------------------------------------------------
# find_business_number — the degradation matrix
# ---------------------------------------------------------------------------


class TestFindBusinessNumber:
    def test_gate_off_short_circuits_without_searching(self):
        with patch(f"{SEARCH}.web_search_enabled", return_value=False), patch(
            "app.core.tools.quick_search_tool._search_web"
        ) as search:
            result = find_business_number("Tony's Pizza", HH)
        assert not result.found
        assert result.reason == "web_search_disabled"
        search.assert_not_called()  # no egress for an opted-out household

    def test_number_found_in_scraped_page(self):
        sources = [{"url": "https://tonys.example", "content": "Call (732) 592-4183"}]
        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            "app.core.tools.quick_search_tool._search_web",
            return_value=[{"url": "https://tonys.example"}],
        ), patch(
            "app.core.tools.quick_search_tool._scrape_results", return_value=sources
        ):
            result = find_business_number("Tony's Pizza", HH)
        assert result.found
        assert result.number == "+17325924183"
        assert result.source_url == "https://tonys.example"

    def test_address_captured_when_present(self):
        sources = [{
            "url": "https://tonys.example",
            "content": "Tony's, 33 National Ave, Brick, NJ 08724. Call 732-592-4183",
        }]
        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            "app.core.tools.quick_search_tool._search_web", return_value=[{"url": "x"}]
        ), patch(
            "app.core.tools.quick_search_tool._scrape_results", return_value=sources
        ):
            result = find_business_number("Tony's", HH)
        assert "33 National Ave" in (result.address or "")

    def test_later_page_used_when_first_has_no_number(self):
        sources = [
            {"url": "https://a.example", "content": "Open daily"},
            {"url": "https://b.example", "content": "Phone 732-592-4183"},
        ]
        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            "app.core.tools.quick_search_tool._search_web",
            return_value=[{"url": "a"}, {"url": "b"}],
        ), patch(
            "app.core.tools.quick_search_tool._scrape_results", return_value=sources
        ):
            result = find_business_number("Tony's", HH)
        assert result.number == "+17325924183"
        assert result.source_url == "https://b.example"

    def test_no_search_results(self):
        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            "app.core.tools.quick_search_tool._search_web", return_value=[]
        ):
            result = find_business_number("Nowhere Inc", HH)
        assert not result.found and result.reason == "no_results"

    def test_results_but_no_parseable_number(self):
        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            "app.core.tools.quick_search_tool._search_web", return_value=[{"url": "x"}]
        ), patch(
            "app.core.tools.quick_search_tool._scrape_results",
            return_value=[{"url": "x", "content": "Hours: 9-5"}],
        ):
            result = find_business_number("Tony's", HH)
        assert not result.found and result.reason == "no_number_found"

    def test_scraper_exception_degrades(self):
        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            "app.core.tools.quick_search_tool._search_web",
            side_effect=RuntimeError("network down"),
        ):
            result = find_business_number("Tony's", HH)
        assert not result.found and result.reason == "search_failed"

    def test_result_dataclass_found_property(self):
        assert NumberSearchResult(number="+17325924183").found is True
        assert NumberSearchResult(reason="no_results").found is False


# ---------------------------------------------------------------------------
# Household location: biasing + mismatch detection
# ---------------------------------------------------------------------------


def _multi_settings(values: dict):
    """Fake settings service resolving several keys."""
    class _S:
        def get(self, key, household_id=None):
            return values.get(key)

    return lambda: _S()


class TestHouseholdLocation:
    def test_reads_the_setting(self):
        from app.services.phone_number_search import household_location

        with patch(
            "app.services.settings_service.get_settings_service",
            _multi_settings({"household.location": "Brick, NJ 08724"}),
        ):
            assert household_location(HH) == "Brick, NJ 08724"

    def test_unset_is_empty(self):
        from app.services.phone_number_search import household_location

        with patch(
            "app.services.settings_service.get_settings_service",
            _multi_settings({"household.location": ""}),
        ):
            assert household_location(HH) == ""

    def test_no_household_is_empty(self):
        from app.services.phone_number_search import household_location

        assert household_location(None) == ""

    def test_settings_error_degrades_to_empty(self):
        """An unbiased search is worse, not fatal — never fail the plan."""
        from app.services.phone_number_search import household_location

        def _boom():
            raise RuntimeError("settings down")

        with patch("app.services.settings_service.get_settings_service", _boom):
            assert household_location(HH) == ""


class TestQueryBiasing:
    def _run_and_capture_query(self, location):
        captured = {}

        def _fake_search(query):
            captured["query"] = query
            return []

        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            f"{SEARCH}.household_location", return_value=location
        ), patch(
            "app.core.tools.quick_search_tool._search_web", _fake_search
        ):
            find_business_number("Tony's Pizzeria", HH)
        return captured["query"]

    def test_location_appended_when_set(self):
        assert (
            self._run_and_capture_query("Brick, NJ")
            == "Tony's Pizzeria Brick, NJ phone number"
        )

    def test_no_location_is_unchanged_behavior(self):
        """Regression guard: empty setting must search exactly as before."""
        assert self._run_and_capture_query("") == "Tony's Pizzeria phone number"

    def test_searched_near_recorded_on_result(self):
        sources = [{"url": "https://t.example", "content": "Call 732-592-4183"}]
        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            f"{SEARCH}.household_location", return_value="Brick, NJ"
        ), patch(
            "app.core.tools.quick_search_tool._search_web", return_value=[{"url": "x"}]
        ), patch(
            "app.core.tools.quick_search_tool._scrape_results", return_value=sources
        ):
            result = find_business_number("Tony's", HH)
        assert result.searched_near == "Brick, NJ"

    def test_searched_near_none_without_location(self):
        sources = [{"url": "https://t.example", "content": "Call 732-592-4183"}]
        with patch(f"{SEARCH}.web_search_enabled", return_value=True), patch(
            f"{SEARCH}.household_location", return_value=""
        ), patch(
            "app.core.tools.quick_search_tool._search_web", return_value=[{"url": "x"}]
        ), patch(
            "app.core.tools.quick_search_tool._scrape_results", return_value=sources
        ):
            result = find_business_number("Tony's", HH)
        assert result.searched_near is None


class TestLocationMismatch:
    """The live failure: a Maryland listing dialed for a New Jersey household.

    Silence beats crying wolf — every ambiguous case must return None.
    """

    def test_different_state_warns(self):
        from app.services.phone_number_search import location_mismatch

        warning = location_mismatch(
            "12800 Frederick Rd, West Friendship, MD 21794", "Brick, NJ 08724"
        )
        assert warning is not None
        assert "MD" in warning and "NJ" in warning

    def test_same_state_is_silent(self):
        from app.services.phone_number_search import location_mismatch

        assert (
            location_mismatch("33 National Ave, Brick, NJ 08724", "Brick, NJ 08724")
            is None
        )

    def test_same_state_distant_town_still_silent(self):
        """Two towns far apart in one state is a judgement call, not an error."""
        from app.services.phone_number_search import location_mismatch

        assert location_mismatch("1 Beach Ave, Cape May, NJ 08204", "Newark, NJ") is None

    def test_missing_address_is_silent(self):
        from app.services.phone_number_search import location_mismatch

        assert location_mismatch(None, "Brick, NJ") is None
        assert location_mismatch("", "Brick, NJ") is None

    def test_missing_location_is_silent(self):
        from app.services.phone_number_search import location_mismatch

        assert location_mismatch("33 National Ave, Brick, NJ", None) is None
        assert location_mismatch("33 National Ave, Brick, NJ", "") is None

    def test_address_without_a_state_is_silent(self):
        from app.services.phone_number_search import location_mismatch

        assert location_mismatch("33 National Ave", "Brick, NJ") is None

    def test_location_without_a_state_is_silent(self):
        """A bare ZIP household location can't be compared — say nothing."""
        from app.services.phone_number_search import location_mismatch

        assert location_mismatch("33 National Ave, Brick, NJ 08724", "08724") is None

    def test_lowercase_words_are_not_read_as_states(self):
        """'in', 'me', 'or' are ordinary words — a false state reading here
        would put a false alarm on a card the user is meant to trust."""
        from app.services.phone_number_search import location_mismatch

        assert location_mismatch("12 Main St in the plaza", "Brick, NJ") is None

    def test_trailing_state_wins_over_earlier_token(self):
        from app.services.phone_number_search import location_mismatch

        # "OR" appears early as a word-ish token; the real state is trailing.
        assert location_mismatch("5 Water St, Portland, OR 97204", "Portland, OR") is None
