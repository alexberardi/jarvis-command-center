"""Unit tests for render_signal_block — Signals → deterministic ambient text.

The renderer turns a list of Signal rows into the situational lines that fold
into the household ambient bundle. It must be DETERMINISTIC (stable-sorted by
kind, subject) so the bundle doesn't churn, and must preserve multiple facts of
the same kind (e.g. weather current vs forecast).
"""
from app.core.prompt_providers.shared.core_rules import render_signal_block


class _Sig:
    """Duck-typed stand-in for a Signal row (kind / subject / summary)."""
    def __init__(self, kind, subject=None, summary=None):
        self.kind = kind
        self.subject = subject
        self.summary = summary


def test_empty_returns_empty():
    assert render_signal_block([]) == ""


def test_all_empty_summaries_returns_empty():
    assert render_signal_block([
        _Sig("presence.seen", summary=None),
        _Sig("weather", summary="   "),
    ]) == ""


def test_renders_summary():
    out = render_signal_block([_Sig("presence.seen", "user:alex", "Alex is home")])
    assert "Alex is home" in out


def test_stable_ordering_byte_identical():
    a = _Sig("weather", "current", "Currently 72 and sunny")
    b = _Sig("presence.seen", "user:alex", "Alex is home")
    c = _Sig("device.state", "garage", "Garage door is open")
    assert render_signal_block([a, b, c]) == render_signal_block([c, b, a])


def test_multi_fact_same_kind_both_render():
    cur = _Sig("weather", "current", "Currently 72 and sunny")
    fc = _Sig("weather", "forecast", "Tomorrow: rain")
    out = render_signal_block([fc, cur])            # deliberately reversed
    assert "Currently 72 and sunny" in out
    assert "Tomorrow: rain" in out
    # deterministic order by (kind, subject): 'current' sorts before 'forecast'
    assert out.index("Currently") < out.index("Tomorrow")


def test_skips_empty_keeps_others():
    out = render_signal_block([
        _Sig("a", summary="keep me"),
        _Sig("b", summary=None),
    ])
    assert out == "keep me"
