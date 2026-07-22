"""Deterministic time-window validation for phone-call scheduling.

The live model (Qwen3-8B, /no_think) cannot reliably decide whether a
proposed clock time falls inside an availability window — it false-declines
valid times ("Thursday at noon" against "Thu 9am-8pm") and false-accepts
invalid ones ("Wednesday at 12" against "Wed 5-8pm"), and prompt wording only
moves the failure around (verified against the box, 2026-07-21). This is
interval arithmetic, so it belongs in code — the same call the codebase
already made for date-key extraction: deterministic, instant, testable.

The bounds come from the confirmed brief's constraint envelope, the exact
strings `extract_constraint_envelope` produces:

    Acceptable times: Tue 4-8pm; Wed 5-8pm; Thu 9am-8pm
    Do not book: Wed 7am-5pm (Work); Tue 8am-4pm (Same day Epic training)

A time is available when it lands inside an Acceptable window on that day and
inside no Do-not-book window. The gateway asks per turn; the verdict rides
into the model as a trailing note, so the model states availability instead
of computing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Mon=0 … Sun=6, matched on the first three letters so "Tue"/"Tuesday"/
# "tues" all resolve.
_DAYS = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}
_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
              "Saturday", "Sunday")

_ACCEPTABLE_PREFIX = "acceptable times:"
_BLOCKED_PREFIX = "do not book:"


@dataclass(frozen=True)
class Interval:
    """A window on one weekday, in minutes past midnight (end exclusive)."""

    day: int
    start: int
    end: int

    def contains(self, day: int, minute: int) -> bool:
        return day == self.day and self.start <= minute < self.end


@dataclass(frozen=True)
class Windows:
    acceptable: tuple[Interval, ...]
    blocked: tuple[Interval, ...]

    def is_open(self, day: int, minute: int) -> bool:
        in_acceptable = any(w.contains(day, minute) for w in self.acceptable)
        in_blocked = any(w.contains(day, minute) for w in self.blocked)
        return in_acceptable and not in_blocked


@dataclass(frozen=True)
class ProposedTime:
    """A day plus one or more candidate minutes.

    Candidates exist because a bare hour ("6", "12") has no am/pm — both
    readings are carried and the envelope decides which one the speaker
    plainly meant ("6" on a 5-8pm day is 6pm, not 6am).
    """

    day: int
    minutes: tuple[int, ...]
    label: str


@dataclass(frozen=True)
class CheckResult:
    time_detected: bool
    # None when a time was detected but could not be validated (e.g. no day),
    # so the caller can fall back to letting the model handle it.
    available: bool | None
    proposed_label: str | None
    acceptable_summary: str | None


# --------------------------------------------------------------- time parsing

def _parse_clock(hour: int, minute: int, meridiem: str | None) -> int | None:
    """(hour, minute, am/pm) -> minutes past midnight, or None if impossible."""
    if not (0 <= minute < 60) or hour < 0 or hour > 23:
        return None
    if meridiem == "am":
        if hour == 12:
            hour = 0
        elif hour > 12:
            return None
    elif meridiem == "pm":
        if hour < 12:
            hour += 12
        elif hour > 12:
            return None
    return hour * 60 + minute


def _candidate_minutes(hour: int, minute: int, meridiem: str | None) -> tuple[int, ...]:
    """All plausible minutes-past-midnight for a parsed clock face.

    With an explicit am/pm there is one. Without, a 1-12 hour has two (am and
    pm) and the envelope picks; a 13-23 hour is already unambiguous.
    """
    if meridiem is not None:
        m = _parse_clock(hour, minute, meridiem)
        return (m,) if m is not None else ()
    if hour > 12:
        m = _parse_clock(hour, minute, None)
        return (m,) if m is not None else ()
    out = []
    for mer in ("am", "pm"):
        m = _parse_clock(hour, minute, mer)
        if m is not None:
            out.append(m)
    return tuple(dict.fromkeys(out))


# "3pm", "10 am", "9:30pm", "12", "6", "2 p.m."
_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?",
    re.IGNORECASE,
)
_NOON_RE = re.compile(r"\bnoon\b", re.IGNORECASE)
_MIDNIGHT_RE = re.compile(r"\bmidnight\b", re.IGNORECASE)
_DAY_RE = re.compile(
    r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b", re.IGNORECASE
)


def _first_day(text: str) -> int | None:
    m = _DAY_RE.search(text)
    return _DAYS[m.group(1).lower()] if m else None


def parse_proposed(utterance: str) -> ProposedTime | None:
    """Pull a proposed day+time out of a callee utterance, or None.

    Requires BOTH a weekday and a time — a lone "3pm" with no day can't be
    checked against a per-day envelope, and guessing the day is exactly the
    kind of inference that belongs to the model, not this validator.
    """
    text = utterance.strip()
    if not text:
        return None
    day = _first_day(text)
    if day is None:
        return None

    if _NOON_RE.search(text):
        return ProposedTime(day, (12 * 60,), f"{_DAY_NAMES[day]} at noon")
    if _MIDNIGHT_RE.search(text):
        return ProposedTime(day, (0,), f"{_DAY_NAMES[day]} at midnight")

    for m in _TIME_RE.finditer(text):
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        raw_mer = m.group(3)
        meridiem = None
        if raw_mer:
            meridiem = "am" if raw_mer.lower().startswith("a") else "pm"
        candidates = _candidate_minutes(hour, minute, meridiem)
        if not candidates:
            continue
        label = _format_label(day, hour, minute, meridiem)
        return ProposedTime(day, candidates, label)
    return None


def _format_label(day: int, hour: int, minute: int, meridiem: str | None) -> str:
    clock = f"{hour}:{minute:02d}" if minute else str(hour)
    suffix = f" {meridiem}" if meridiem else ""
    return f"{_DAY_NAMES[day]} at {clock}{suffix}".strip()


# ------------------------------------------------------------ window parsing

def _parse_range(text: str) -> tuple[int, int] | None:
    """"4-8pm" / "9am-8pm" / "7am-5pm" -> (start_min, end_min).

    A missing am/pm on the start inherits the end's, so "4-8pm" reads as
    4pm-8pm — the way the envelope writes and a person reads it.
    """
    parts = text.split("-", 1)
    if len(parts) != 2:
        return None
    start_raw, end_raw = parts[0].strip(), parts[1].strip()

    def face(s: str) -> tuple[int, int, str | None] | None:
        mm = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s, re.IGNORECASE)
        if not mm:
            return None
        return (
            int(mm.group(1)),
            int(mm.group(2)) if mm.group(2) else 0,
            mm.group(3).lower() if mm.group(3) else None,
        )

    s_face, e_face = face(start_raw), face(end_raw)
    if not s_face or not e_face:
        return None
    end = _parse_clock(e_face[0], e_face[1], e_face[2] or "pm")
    start_mer = s_face[2] or e_face[2]
    start = _parse_clock(s_face[0], s_face[1], start_mer)
    if start is None or end is None or start >= end:
        return None
    return start, end


def _parse_entries(line_body: str) -> list[Interval]:
    """"Tue 4-8pm; Wed 5-8pm; ..." -> intervals. Bad entries drop silently."""
    out: list[Interval] = []
    for chunk in line_body.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Strip a trailing "(Work)"-style label that Do-not-book lines carry.
        chunk = re.sub(r"\s*\([^)]*\)\s*$", "", chunk).strip()
        dm = _DAY_RE.match(chunk)
        if not dm:
            continue
        day = _DAYS[dm.group(1).lower()]
        rng = _parse_range(chunk[dm.end():].strip())
        if rng is None:
            continue
        out.append(Interval(day, rng[0], rng[1]))
    return out


def parse_windows(envelope: str | None) -> Windows:
    """Parse an envelope's Acceptable/Do-not-book lines into intervals."""
    acceptable: list[Interval] = []
    blocked: list[Interval] = []
    for line in (envelope or "").splitlines():
        low = line.strip().lower()
        if low.startswith(_ACCEPTABLE_PREFIX):
            acceptable += _parse_entries(line.strip()[len(_ACCEPTABLE_PREFIX):])
        elif low.startswith(_BLOCKED_PREFIX):
            blocked += _parse_entries(line.strip()[len(_BLOCKED_PREFIX):])
    return Windows(tuple(acceptable), tuple(blocked))


# --------------------------------------------------------------------- check

def _acceptable_summary(envelope: str | None) -> str | None:
    for line in (envelope or "").splitlines():
        if line.strip().lower().startswith(_ACCEPTABLE_PREFIX):
            return line.strip()
    return None


def check_time(envelope: str | None, utterance: str) -> CheckResult:
    """Is the time proposed in ``utterance`` available under ``envelope``?

    ``available`` is True/False only when a day+time was found AND the
    envelope has real acceptable windows; otherwise None so the caller lets
    the model carry the turn unaided.
    """
    proposed = parse_proposed(utterance)
    if proposed is None:
        return CheckResult(False, None, None, None)

    windows = parse_windows(envelope)
    summary = _acceptable_summary(envelope)
    if not windows.acceptable:
        # Nothing to check against — a verdict here would be a guess.
        return CheckResult(True, None, proposed.label, summary)

    # Any candidate reading that lands open wins: it is the one the speaker
    # plainly meant ("6" on a 5-8pm day is 6pm).
    available = any(
        windows.is_open(proposed.day, minute) for minute in proposed.minutes
    )
    return CheckResult(True, available, proposed.label, summary)
