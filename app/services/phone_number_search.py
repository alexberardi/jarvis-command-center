"""Web-search fallback for resolving a business's phone number.

Used by the phone-call plan step after a phonebook miss (phone-calls PRD,
UX flow step 2 "Resolve"). Reuses quick_search's search + SSRF-guarded
scrape machinery rather than duplicating the provider/scraper wiring.

Governing rules:
- **Gated on `web_search.enabled` for the household**, checked explicitly
  with a household_id. quick_search's own gate helper resolves the
  household from the conversation cache, which the plan step does not have
  — reusing it verbatim would silently fail closed for every call.
- **Confidence is never assumed.** Whatever is found lands in the confirm
  card's *editable* tel field, attributed to its source, and a human
  eyeballs it before it dials. That is the PRD's stale-search-results
  mitigation; this module's job is to save typing, not to be trusted.
- **Degrades in every direction** — gate off, no results, no parseable
  number, scraper error — to "no number found". Never blocks the plan,
  never invents a number.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("uvicorn")

# Enough pages to beat a single bad listing without turning a plan step into
# a research project (each page is an HTTP fetch on the user's wait).
_MAX_PAGES = 3

# US phone shapes as they appear in listing prose: (732) 592-4183,
# 732-592-4183, 732.592.4183, +1 732 592 4183. Deliberately NOT matching
# bare 10-digit runs — those collide with zip+4, order numbers, and prices.
_PHONE_RE = re.compile(
    r"""(?<![\d-])            # not mid-number
    (?:\+?1[\s.\-]*)?         # optional country code
    (?:\((\d{3})\)|(\d{3}))   # area code, parenthesised or not
    [\s.\-]+                  # REQUIRED separator (rejects bare digit runs)
    (\d{3})
    [\s.\-]+
    (\d{4})
    (?![\d-])""",
    re.VERBOSE,
)

# Address-ish line near a hit, best-effort — nice to have, never required.
_ADDRESS_RE = re.compile(
    r"\d{1,6}\s+[A-Za-z0-9.\- ]{3,40}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|"
    r"Highway|Hwy|Route|Rt|Place|Pl|Court|Ct|Parkway|Pkwy)\.?"
    r"(?:,\s*[A-Za-z .]{2,30})?(?:,\s*[A-Z]{2})?(?:\s+\d{5})?",
    re.IGNORECASE,
)


@dataclass
class NumberSearchResult:
    """What the search found. ``number`` is None when nothing usable turned up."""

    number: str | None = None
    source_url: str | None = None
    address: str | None = None
    reason: str | None = None  # why there's no number, for the card's note
    searched_near: str | None = None  # household location used to bias, if any

    @property
    def found(self) -> bool:
        return self.number is not None


def web_search_enabled(household_id: str | None) -> bool:
    """Fail-closed read of the household's ``web_search.enabled`` setting.

    Explicitly household-scoped: unlike quick_search's helper this takes the
    id directly, because the plan step runs outside any conversation cache.
    """
    if not household_id:
        return False
    try:
        from app.services.settings_service import get_settings_service

        val = get_settings_service().get(
            "web_search.enabled", household_id=str(household_id)
        )
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)
    except Exception as e:  # noqa: BLE001 — any doubt means no egress
        logger.warning(
            "phone number search gate check failed, defaulting to DISABLED: %s", e
        )
        return False


def household_location(household_id: str | None) -> str:
    """The household's locality string, or "" when unset.

    Explicitly household-scoped for the same reason as the gate above: the
    plan step runs outside any conversation cache. Any failure degrades to
    "" — an unbiased search is worse than a biased one, but it is not a
    reason to fail the plan.
    """
    if not household_id:
        return ""
    try:
        from app.services.settings_service import get_settings_service

        val = get_settings_service().get(
            "household.location", household_id=str(household_id)
        )
        return str(val).strip() if val else ""
    except Exception as e:  # noqa: BLE001 — location is an enhancement
        logger.warning("household location lookup failed: %s", e)
        return ""


# US state abbreviations, for the "is this result even near me?" check.
_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}

_STATE_RE = re.compile(r"\b([A-Z]{2})\b")
_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def _state_of(text: str) -> str | None:
    """Last US state abbreviation in ``text``, if any.

    Last, not first: addresses read "33 National Ave, Brick, NJ 08724", so
    the trailing token is the state. Case-sensitive on purpose — lowercase
    "in"/"me"/"or" are ordinary words, and a false state reading here would
    produce a false alarm on the card.
    """
    found = [m.group(1) for m in _STATE_RE.finditer(text or "") if m.group(1) in _STATES]
    return found[-1] if found else None


def _zip_of(text: str) -> str | None:
    match = _ZIP_RE.search(text or "")
    return match.group(1) if match else None


def location_mismatch(address: str | None, location: str | None) -> str | None:
    """Human-readable warning when ``address`` is clearly not near ``location``.

    Deliberately cheap and dependency-free — no geocoding. It answers only
    the question that actually burned us: "is this result in a different
    STATE than the household?" Anything less clear-cut returns None.

    False-alarm guards (silence beats crying wolf on a card the user is
    meant to trust):
    - Either side missing a state → None.
    - Same state → None, regardless of ZIP or city distance. Two towns 200
      miles apart in the same state are a judgement call, not an error.
    - ZIP is only used to corroborate a state we already read.
    """
    if not address or not location:
        return None
    addr_state = _state_of(address)
    home_state = _state_of(location)
    if not addr_state or not home_state:
        return None
    if addr_state == home_state:
        return None
    return (
        f"\u26a0\ufe0f This result is in {addr_state} but your household is in "
        f"{home_state} \u2014 check it's the right location before calling."
    )


def extract_number(text: str) -> str | None:
    """First E.164-normalizable US number in ``text``, else None.

    Candidates are validated through the same `normalize_us_number` the dial
    path uses, so emergency/short/premium numbers are rejected here too —
    a scraped 911 or 900-number never reaches the card.
    """
    from app.services.phone_call_service import (
        NumberValidationError,
        normalize_us_number,
    )

    for match in _PHONE_RE.finditer(text or ""):
        area = match.group(1) or match.group(2)
        candidate = f"{area}{match.group(3)}{match.group(4)}"
        try:
            return normalize_us_number(candidate)
        except NumberValidationError:
            continue  # premium/invalid shape — keep looking
    return None


def extract_address(text: str) -> str | None:
    match = _ADDRESS_RE.search(text or "")
    return match.group(0).strip() if match else None


def find_business_number(business: str, household_id: str) -> NumberSearchResult:
    """Search the web for ``business``'s phone number.

    Synchronous (quick_search's machinery is sync); callers should offload
    with ``asyncio.to_thread``. Never raises.
    """
    if not web_search_enabled(household_id):
        return NumberSearchResult(reason="web_search_disabled")

    # Bias the query toward the household's own area. Without this a search
    # for a common business name returns whichever listing ranks highest
    # nationally — live 2026-07-19 that dialed a Tony's Pizzeria in Maryland
    # for a household in New Jersey.
    location = household_location(household_id)
    query = (
        f"{business} {location} phone number" if location
        else f"{business} phone number"
    )
    try:
        from app.core.tools.quick_search_tool import _scrape_results, _search_web

        results = _search_web(query)
        if not results:
            logger.info("[phone_search] no results for %r", query)
            return NumberSearchResult(reason="no_results")

        sources = _scrape_results(results[:_MAX_PAGES])
    except Exception as e:  # noqa: BLE001 — search is best-effort
        logger.warning("[phone_search] search failed for %r: %s", business, e)
        return NumberSearchResult(reason="search_failed")

    for source in sources:
        content = source.get("content") or ""
        number = extract_number(content)
        if number:
            logger.info(
                "[phone_search] FOUND %s for %r via %s",
                number, business, source.get("url"),
            )
            return NumberSearchResult(
                number=number,
                source_url=source.get("url"),
                address=extract_address(content),
                searched_near=location or None,
            )

    logger.info("[phone_search] no parseable number for %r", business)
    return NumberSearchResult(reason="no_number_found")
