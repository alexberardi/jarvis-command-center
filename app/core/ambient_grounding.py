"""Deterministic grounding guard for ambient-context responses.

The ambient bundle is the ONLY situational data the model is given, so a reply that
states a **temperature or percentage** not present in the bundle is a fabrication — the
exact "we're lying" failure a proactive assistant must never make. This flags those
cheaply, with no model call: a hard backstop underneath the GPT faithfulness judge in
the eval, and a candidate runtime guard.

Scope is deliberately narrow — temperatures and percentages are stated *exactly*, so a
mismatch is unambiguous. Fuzzier claims (event names, phrasing of clock times) are left
to the LLM judge; over-flagging a legitimate paraphrase would be worse than a miss.
"""
import re

# Temperatures ("58°F", "58 F", "58 degrees") and percentages ("72%") — stated exactly.
_SPECIFIC_RE = re.compile(
    r"\d{1,3}\s?%"
    r"|\d{1,3}\s?°\s?[Ff]?"
    r"|\d{1,3}\s?°?\s?[Ff]\b"
    r"|\d{1,3}\s?degrees?\b",
    re.IGNORECASE,
)


def unsupported_numbers(response: str, bundle: str) -> list[str]:
    """Temperature/percentage specifics stated in `response` whose number is absent from
    `bundle`. Empty list ⇒ numerically grounded.

    A response specific is allowed if its number appears ANYWHERE in the bundle (the
    bundle may state it without a unit, e.g. "high 61"). This favors a rare miss over a
    false alarm — the LLM faithfulness judge is the primary grounding check.
    """
    allowed = set(re.findall(r"\d+", bundle))
    out: list[str] = []
    for m in _SPECIFIC_RE.findall(response):
        core = re.sub(r"\D", "", m)
        token = m.strip()
        if core and core not in allowed and token not in out:
            out.append(token)
    return out


def is_number_grounded(response: str, bundle: str) -> bool:
    """True when the response invents no temperature/percentage absent from the bundle."""
    return not unsupported_numbers(response, bundle)
