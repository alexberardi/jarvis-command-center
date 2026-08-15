"""Plan-start blast gate — the autonomy dial the engine deferred (workflow_engine
``widens_envelope`` only guards amendments mid-run; nothing classified a FRESH plan
for autorun).

A signal-triggered plan may execute WITHOUT a human tap only if it is entirely
low-blast. This is the security boundary, so it is deliberately tiny and explicit:
an allowlist of read-only / node-local-write commands, no risky step, no outbound
counterparty, and a small step count. Anything outside it falls back to a
tap-to-confirm card — never autorun.
"""

from __future__ import annotations

from typing import Any

# Commands allowed to run autonomously. Keep this SMALL and explicit — it is the
# trust boundary. get_drive_time is read-only; reminder writes only to the speaker's
# own reminder store (reversible, node-local). Nothing that spends money, contacts a
# stranger, or takes an irreversible/external action belongs here.
AUTORUN_ALLOWLIST: frozenset[str] = frozenset({"get_drive_time", "reminder"})

_MAX_AUTORUN_STEPS = 3


def is_autorun_eligible(steps: list[dict[str, Any]]) -> tuple[bool, str]:
    """Whether a fresh node-native plan (``[{command, args, label, is_risky}]``) may
    autorun. Returns ``(eligible, reason)``. Fail-closed: any doubt → not eligible."""
    if not steps:
        return False, "empty plan"
    if len(steps) > _MAX_AUTORUN_STEPS:
        return False, f"too many steps for autorun ({len(steps)} > {_MAX_AUTORUN_STEPS})"
    for step in steps:
        command = (step.get("command") or "").strip()
        if command not in AUTORUN_ALLOWLIST:
            return False, f"command '{command}' is not autorun-allowlisted"
        if step.get("is_risky"):
            return False, f"step '{command}' is marked risky"
        args = step.get("args") or {}
        # A 'business' arg means an outbound counterparty (a call/contact) — hard block
        # regardless of the command, so autorun can never originate outbound comms.
        if isinstance(args, dict) and args.get("business"):
            return False, f"step '{command}' targets an outbound counterparty"
    return True, "eligible"
