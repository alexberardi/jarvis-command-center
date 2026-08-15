"""Signal-reaction registry — the generic, kind-agnostic dispatch plane.

Reactions register for a kind; ingest fans each signal out to the handlers for its
kind (fire-and-forget), never naming a kind itself. These pin: registration +
lookup, idempotency, multiple reactions per kind, kind-scoped dispatch, and that a
failing/unknown reaction can never break the ingest path.
"""
import asyncio

import pytest

from app.services import signal_reaction_registry as reg
from app.services.signal_reaction_registry import (
    ReactionContext,
    reactions_for,
    register_signal_reaction,
    schedule_signal_reactions,
)


@pytest.fixture(autouse=True)
def _clean():
    reg.clear_reactions()
    yield
    reg.clear_reactions()


def _ctx(kind):
    return ReactionContext(household_id="hh", node_id="n", user_id=1, kind=kind, facts={"x": 1})


def test_register_and_lookup_is_idempotent_per_kind_name():
    async def h(ctx):
        return "ok"

    register_signal_reaction("appt.upcoming", "leave_by", h)
    register_signal_reaction("appt.upcoming", "leave_by", h)  # dup → ignored
    assert [n for n, _ in reactions_for("appt.upcoming")] == ["leave_by"]


def test_multiple_reactions_per_kind():
    async def a(ctx):
        return "a"

    async def b(ctx):
        return "b"

    register_signal_reaction("appt.upcoming", "one", a)
    register_signal_reaction("appt.upcoming", "two", b)
    assert [n for n, _ in reactions_for("appt.upcoming")] == ["one", "two"]


def test_dispatch_runs_only_the_reactions_for_that_kind():
    seen = []

    async def leave_by(ctx):
        seen.append(("leave_by", ctx.kind))
        return "proposed"

    async def weather(ctx):
        seen.append(("weather", ctx.kind))
        return "warned"

    register_signal_reaction("appt.upcoming", "leave_by", leave_by)
    register_signal_reaction("weather.severe", "weather", weather)

    async def go():
        schedule_signal_reactions(_ctx("appt.upcoming"))
        await asyncio.sleep(0.02)  # let the fire-and-forget task run

    asyncio.run(go())
    assert seen == [("leave_by", "appt.upcoming")]  # kind-scoped: weather did NOT run


def test_dispatch_fans_out_to_all_reactions_of_a_kind():
    seen = []

    async def one(ctx):
        seen.append("one")

    async def two(ctx):
        seen.append("two")

    register_signal_reaction("appt.upcoming", "one", one)
    register_signal_reaction("appt.upcoming", "two", two)

    async def go():
        schedule_signal_reactions(_ctx("appt.upcoming"))
        await asyncio.sleep(0.02)

    asyncio.run(go())
    assert sorted(seen) == ["one", "two"]


def test_unknown_kind_is_a_silent_noop():
    async def go():
        schedule_signal_reactions(_ctx("nothing.here"))  # no reaction registered
        await asyncio.sleep(0.01)

    asyncio.run(go())  # must not raise


def test_reaction_exception_is_contained():
    async def boom(ctx):
        raise RuntimeError("kaboom")

    register_signal_reaction("appt.upcoming", "boom", boom)

    async def go():
        schedule_signal_reactions(_ctx("appt.upcoming"))
        await asyncio.sleep(0.02)  # the failing task must not propagate out

    asyncio.run(go())  # must not raise
