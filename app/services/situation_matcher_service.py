"""The proactive Signal reasoner — event-driven, background-model, propose-only.

A Signal edge schedules a DEBOUNCED reason pass per household. The pass gathers
the household's live Signal bundle + the node's advertised proposable actions,
builds the situation prompt, and enqueues a BACKGROUND-model job (never the live
voice slot). The queue callback finalizes the matches and emits tap-to-confirm
cards, gated by anti-nag (idempotency dedup + per-source cooldown + per-household
daily cap). The reasoner ONLY proposes — it never executes a tool.

OFF by default: both ``proposals.enabled`` AND ``proposals.proactive_enabled``
must be truthy for a household (fail-closed). Keep it dormant until the offline
precision harness clears the 9B worst case.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("uvicorn")

# Coalesce edges arriving within this window into ONE reason pass. Module-level
# so tests can shrink it.
DEBOUNCE_SECONDS = 5.0
_COOLDOWN_SECONDS = 6 * 3600.0   # per (household, source_key, command)
_DAILY_CAP = 12                  # per household

# In-process state (single-worker; a restart simply re-arms).
_main_loop: asyncio.AbstractEventLoop | None = None
_pending: dict[str, "asyncio.Task[Any]"] = {}          # household -> debounce task
_fired_keys: set[str] = set()                          # idempotency keys already proposed
_last_fired: dict[tuple[str, str, str], float] = {}    # (household, source_key, command) -> monotonic
_daily: dict[str, tuple[str, int]] = {}                # household -> (yyyy-mm-dd, count)


# ── gating ──────────────────────────────────────────────────────────────────
def _truthy(v: Any) -> bool:
    return v is not None and str(v).lower() not in ("false", "0", "")


def _proactive_enabled(household_id: str) -> bool:
    """Fail-closed: BOTH proposals.enabled and proposals.proactive_enabled must
    be explicitly truthy for this household."""
    try:
        from app.services.settings_service import get_settings_service

        s = get_settings_service()
        base = s.get("proposals.enabled", household_id=household_id)
        pro = s.get("proposals.proactive_enabled", household_id=household_id)
        return _truthy(base) and _truthy(pro)
    except Exception:  # noqa: BLE001 — any settings error → disabled
        return False


# ── url helpers (reuse the memory-extraction discovery, fall back to env) ─────
def _cc_url() -> str:
    try:
        from app.services.memory_extraction_service import _get_command_center_url

        return _get_command_center_url()
    except Exception:  # noqa: BLE001
        return os.getenv("JARVIS_CC_URL", "http://localhost:7703")


def _llm_proxy_url() -> str:
    try:
        from app.services.memory_extraction_service import _get_llm_proxy_url

        return _get_llm_proxy_url()
    except Exception:  # noqa: BLE001
        return os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:7704")


# ── the reason pass (enqueue a background job) ────────────────────────────────
def _gather_bundle(household_id: str) -> list[dict[str, Any]]:
    """Every non-expired household Signal as a match_situation bundle item."""
    from app.db import get_session_local
    from app.models import Signal

    db = get_session_local()()
    try:
        now = datetime.utcnow()
        rows = (
            db.query(Signal)
            .filter(
                Signal.household_id == household_id,
                Signal.is_active == True,  # noqa: E712
                (Signal.expires_at.is_(None)) | (Signal.expires_at > now),
            )
            .all()
        )
        out: list[dict[str, Any]] = []
        for s in rows:
            try:
                data = json.loads(s.facts) if s.facts else {}
            except Exception:  # noqa: BLE001
                data = {}
            if s.summary and "summary" not in data:
                data = {**data, "summary": s.summary}
            out.append({"source_key": s.source_key, "kind": s.kind, "data": data})
        return out
    finally:
        db.close()


async def run_match_batch(household_id: str, node_id: str | None = None) -> None:
    """Gate → gather → enqueue a background-model situation-match job. Never
    touches the live inference slot; best-effort, never raises."""
    if not _proactive_enabled(household_id):
        return
    try:
        if not node_id:
            from app.services.proposable_action_service import _first_household_node_id

            node_id = _first_household_node_id(household_id)
        if not node_id:
            return

        bundle = _gather_bundle(household_id)
        if not bundle:
            return

        from app.services.capability_registry import list_proposable_actions

        actions = await list_proposable_actions(node_id)
        if not actions:
            return

        from app.services.proposal_matcher import _build_menu, _build_situation_prompt

        prompt = _build_situation_prompt(bundle, _build_menu(actions))
        await _enqueue(household_id, node_id, bundle, prompt)
    except Exception:  # noqa: BLE001
        logger.warning("run_match_batch failed for %s", household_id, exc_info=True)


async def _enqueue(
    household_id: str, node_id: str, bundle: list[dict[str, Any]], prompt: str
) -> None:
    from app.core.utils.rest_client import post

    job_id = f"situation-{uuid.uuid4().hex[:16]}"
    callback: dict[str, str] = {
        "url": f"{_cc_url().rstrip('/')}/api/v0/situation-matcher/callback"
    }
    token = os.getenv("JARVIS_ADAPTER_CALLBACK_TOKEN")
    if token:
        callback["auth_type"] = "bearer"
        callback["token"] = token

    payload = {
        "job_id": job_id,
        "job_type": "chat",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "trace_id": job_id,
        "idempotency_key": job_id,
        "job_type_version": "v1",
        "ttl_seconds": 600,
        "metadata": {
            "type": "situation_match",
            "household_id": household_id,
            "node_id": node_id,
            "bundle": bundle,
        },
        "request": {
            "model": "background",  # <-- off the live voice slot
            "messages": [{"role": "user", "content": prompt}],
            "sampling": {"temperature": 0.0},
        },
        "callback": callback,
    }

    headers: dict[str, str] = {}
    internal = os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if internal:
        headers["X-Internal-Token"] = internal

    queue_url = f"{_llm_proxy_url().rstrip('/')}/internal/queue/enqueue"
    await post(url=queue_url, json_data=payload, headers=headers)
    logger.info(
        "Enqueued situation match job_id=%s household=%s bundle=%d",
        job_id, household_id, len(bundle),
    )


# ── the queue callback (finalize → anti-nag → emit cards) ─────────────────────
def _parse_matches(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001 — tolerate stray prose / fences
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


async def handle_match_callback(payload: dict[str, Any]) -> None:
    """Background-job callback: parse the LLM output, finalize, and emit cards
    (subject to anti-nag). Best-effort — never raises."""
    if payload.get("status") != "succeeded":
        return
    content = (payload.get("result") or {}).get("content", "")
    if not content:
        return
    metadata = payload.get("metadata") or {}
    household_id = metadata.get("household_id", "")
    node_id = metadata.get("node_id")
    bundle = metadata.get("bundle") or []

    try:
        parsed = _parse_matches(content)
    except Exception:  # noqa: BLE001
        logger.warning("situation callback: unparseable content", exc_info=True)
        return

    from app.services.capability_registry import list_proposable_actions
    from app.services.proposal_card import emit_proposal_card
    from app.services.proposal_matcher import finalize_situation_matches

    try:
        actions = await list_proposable_actions(node_id) if node_id else []
    except Exception:  # noqa: BLE001
        actions = []
    if not actions:
        return

    for r in finalize_situation_matches(parsed, bundle, actions):
        if not _anti_nag_ok(household_id, r):
            continue
        spec = r.get("spec") or {}
        try:
            emit_proposal_card(
                household_id=household_id,
                node_id=node_id,
                command=r["command"],
                callback=r["action"],
                params=r["args"],
                idempotency_key=r["idempotency_key"],
                card_title=spec.get("card_title") or f"Run {r['command']}?",
                source=(r["source_keys"][0] if r.get("source_keys") else None),
            )
        except Exception:  # noqa: BLE001 — one bad card must not sink the batch
            logger.warning("situation callback: emit failed for %s", r.get("command"), exc_info=True)
            continue
        _record_fired(household_id, r)


# ── anti-nag ──────────────────────────────────────────────────────────────────
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _anti_nag_ok(household_id: str, r: dict[str, Any]) -> bool:
    if r["idempotency_key"] in _fired_keys:
        return False  # already proposed this exact situation→action
    date, count = _daily.get(household_id, (_today(), 0))
    if date != _today():
        count = 0
    if count >= _DAILY_CAP:
        return False
    sk = r["source_keys"][0] if r.get("source_keys") else ""
    last = _last_fired.get((household_id, sk, r["command"]))
    if last is not None and (time.monotonic() - last) < _COOLDOWN_SECONDS:
        return False
    return True


def _record_fired(household_id: str, r: dict[str, Any]) -> None:
    _fired_keys.add(r["idempotency_key"])
    today = _today()
    date, count = _daily.get(household_id, (today, 0))
    if date != today:
        count = 0
    _daily[household_id] = (today, count + 1)
    sk = r["source_keys"][0] if r.get("source_keys") else ""
    _last_fired[(household_id, sk, r["command"])] = time.monotonic()


# ── the debounced edge ────────────────────────────────────────────────────────
def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called once at startup so a SYNC endpoint (threadpool, no running loop)
    can still schedule the debounce onto the main loop."""
    global _main_loop
    _main_loop = loop


def signal_situation_edge(household_id: str, node_id: str | None = None) -> None:
    """Schedule a debounced proactive reason pass for the household. Fire-and-
    forget: coalesces edges within DEBOUNCE_SECONDS into ONE run. Never blocks or
    raises into the caller (safe to call from the ingest hot path)."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = _main_loop
        if loop is None:
            return  # no event loop available → skip
        loop.call_soon_threadsafe(_schedule_debounce, household_id, node_id, loop)
    except Exception:  # noqa: BLE001 — never let edge scheduling break the caller
        pass


def _schedule_debounce(
    household_id: str, node_id: str | None, loop: asyncio.AbstractEventLoop
) -> None:
    existing = _pending.get(household_id)
    if existing and not existing.done():
        existing.cancel()  # coalesce — the latest edge wins
    _pending[household_id] = loop.create_task(_debounced_run(household_id, node_id))


async def _debounced_run(household_id: str, node_id: str | None) -> None:
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)
        await run_match_batch(household_id, node_id)
    except asyncio.CancelledError:  # superseded by a newer edge
        pass
    except Exception:  # noqa: BLE001
        logger.warning("debounced situation run failed for %s", household_id, exc_info=True)
    finally:
        if _pending.get(household_id) is not None and _pending[household_id].done():
            _pending.pop(household_id, None)
