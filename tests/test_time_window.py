"""Deterministic time-window validation.

The cases here are exactly the ones the live model got wrong under /no_think
(false-declining valid times, false-accepting invalid ones) — this is the
code that replaces that guesswork, so those cases are the point of the file.
"""

import pytest

from app.services.time_window import (
    check_time,
    parse_proposed,
    parse_windows,
)

# The real envelope from the live calls (session 804c0806 / b6c51d78).
ENVELOPE = (
    "Acceptable times: Tue 4-8pm; Wed 5-8pm; Thu 9am-8pm; Fri 9am-8pm; "
    "Sat 9am-8pm; Sun 9am-8pm\n"
    "Do not book: Mon 7am-5pm (Work); Tue 8am-4pm (Same day Epic training); "
    "Wed 7am-5pm (Work)"
)


class TestTheCasesTheModelFailed:
    @pytest.mark.parametrize(
        "utterance, available",
        [
            # Live 2026-07-20: model said "Wednesday at 12 is available" — noon
            # is before Wed's 5pm open. Must be declined.
            ("How about Wednesday at 12?", False),
            # Model false-declined this repeatedly; noon is inside Thu 9am-8pm.
            ("Can you do Thursday at noon?", True),
            # 3pm is before Tue's 4pm open.
            ("Is Tuesday at 3pm okay?", False),
            ("Does Friday at 10am work?", True),
            # Bare "6" on a 5-8pm day is 6pm, and 6pm is open.
            ("How about Wednesday at 6?", True),
            ("Could I come Thursday at 2pm?", True),
            # 8am is before Sat's 9am open.
            ("Is Saturday at 8am possible?", False),
            # Monday isn't an acceptable day at all.
            ("How about Monday at 6pm?", False),
        ],
    )
    def test_verdicts(self, utterance, available):
        result = check_time(ENVELOPE, utterance)
        assert result.time_detected
        assert result.available is available, (
            f"{utterance!r}: expected available={available}, got {result.available}"
        )


class TestBareHourDisambiguation:
    def test_bare_hour_reads_as_the_one_that_fits(self):
        # 6 -> 6pm on a 5-8pm day (6am would be closed).
        assert check_time("Acceptable times: Wed 5-8pm", "Wednesday at 6").available
        # 10 -> 10am on a 9am-8pm day (10pm would be closed).
        assert check_time("Acceptable times: Thu 9am-8pm", "Thursday at 10").available

    def test_bare_hour_with_no_fitting_reading_is_declined(self):
        # 12 -> neither noon nor midnight fits a 5-8pm day.
        assert check_time("Acceptable times: Wed 5-8pm", "Wednesday at 12").available is False

    def test_explicit_meridiem_is_honoured_over_the_fitting_reading(self):
        # 6am is explicit and does not fit 5-8pm, even though 6pm would.
        assert check_time("Acceptable times: Wed 5-8pm", "Wednesday at 6am").available is False


class TestDoNotBook:
    def test_a_blocked_window_overrides_an_acceptable_one(self):
        env = "Acceptable times: Wed 9am-8pm\nDo not book: Wed 12-1pm (Lunch)"
        assert check_time(env, "Wednesday at 12:30pm").available is False
        # Just outside the block is fine.
        assert check_time(env, "Wednesday at 2pm").available is True


class TestBoundaries:
    def test_start_is_inclusive_end_is_exclusive(self):
        env = "Acceptable times: Thu 9am-5pm"
        assert check_time(env, "Thursday at 9am").available is True
        # 5pm is the exclusive end — a 5pm appointment runs past close.
        assert check_time(env, "Thursday at 5pm").available is False
        assert check_time(env, "Thursday at 4pm").available is True


class TestParsingProposed:
    def test_requires_a_day(self):
        # A lone time with no day can't be checked against a per-day envelope.
        assert parse_proposed("How about 3pm?") is None

    def test_requires_a_time(self):
        assert parse_proposed("How about Thursday?") is None

    @pytest.mark.parametrize(
        "text, label_contains",
        [
            ("thursday at noon", "noon"),
            ("Friday at 10am", "10 am"),
            ("wed at 6", "6"),
            ("Tuesday at 2:30pm", "2:30 pm"),
        ],
    )
    def test_label_is_human_readable(self, text, label_contains):
        p = parse_proposed(text)
        assert p is not None
        assert label_contains in p.label


class TestParsingWindows:
    def test_inherits_start_meridiem_from_end(self):
        (w,) = parse_windows("Acceptable times: Tue 4-8pm").acceptable
        assert (w.start, w.end) == (16 * 60, 20 * 60)  # 4pm-8pm

    def test_explicit_am_and_pm(self):
        (w,) = parse_windows("Acceptable times: Thu 9am-8pm").acceptable
        assert (w.start, w.end) == (9 * 60, 20 * 60)

    def test_strips_do_not_book_labels(self):
        (w,) = parse_windows("Do not book: Wed 7am-5pm (Work)").blocked
        assert (w.day, w.start, w.end) == (2, 7 * 60, 17 * 60)

    def test_bad_entries_drop_without_killing_the_line(self):
        w = parse_windows("Acceptable times: Wed nonsense; Thu 9am-8pm")
        assert len(w.acceptable) == 1
        assert w.acceptable[0].day == 3


class TestDegradation:
    def test_no_time_in_utterance(self):
        r = check_time(ENVELOPE, "Okay, and the patient's name?")
        assert r.time_detected is False
        assert r.available is None

    def test_empty_envelope_detects_but_does_not_judge(self):
        # A time was proposed but there is nothing to check it against — the
        # model should carry the turn, so available is None, not a guess.
        r = check_time("", "Thursday at noon")
        assert r.time_detected is True
        assert r.available is None

    def test_placeholder_only_envelope_does_not_judge(self):
        r = check_time("Acceptable times: (fill in your availability)", "Thursday at noon")
        assert r.available is None


class TestFallOpenHardening:
    """Realistic envelope shapes that used to fall open to the model (available=None)
    and let a bad slot through — the live 'booked during a bad Tuesday window' bug."""

    def test_blocked_slot_vetoed_even_when_acceptable_did_not_parse(self):
        # SAFETY: a freeform 'Acceptable times:' can't be parsed, but a proposed slot
        # inside a real 'Do not book:' window must still be declined — never fall open.
        env = "Acceptable times: whenever works for you\nDo not book: Tue 8am-4pm"
        r = check_time(env, "Can you do Tuesday at noon?")
        assert r.available is False

    def test_date_stamped_windows_parse(self):
        env = "Acceptable times: Tue Aug 5 4-8pm\nDo not book: Tue Aug 5 8am-4pm"
        assert check_time(env, "Tuesday at 1pm?").available is False
        assert check_time(env, "Tuesday at 6pm?").available is True

    def test_to_separator_and_dotted_meridiem(self):
        env = ("Acceptable times: Tuesday 4 p.m. to 8 p.m.\n"
               "Do not book: Tuesday 8 a.m. to 4 p.m.")
        assert check_time(env, "Tuesday at 10am?").available is False
        assert check_time(env, "Tuesday at 5pm?").available is True

    def test_good_slot_unaffected_by_the_blocked_veto(self):
        env = "Acceptable times: Tue 4-8pm\nDo not book: Tue 8am-4pm"
        assert check_time(env, "How about Tuesday at 6pm?").available is True
        assert check_time(env, "Tuesday at 6?").available is True  # bare hour -> 6pm

    def test_fully_freeform_still_falls_open(self):
        # Nothing parseable at all — the model legitimately carries the turn.
        env = "Acceptable times: Tuesday afternoons\nDo not book: Tuesday mornings"
        assert check_time(env, "Tuesday at 9am?").available is None
