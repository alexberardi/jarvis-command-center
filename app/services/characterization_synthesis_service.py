"""Characterization synthesis — regenerate the per-person *view* off the hot path.

Two-phase flow (same pattern as memory_extraction_service / deep_research_service):
  1. run_synthesis_batch() — for each identifiable user with NEW transcripts since
     their last synthesis, gather (facts + prior view + recent transcripts) and
     enqueue a background-model LLM job.
  2. handle_synthesis_callback() — parse the returned characterization document and
     upsert it via CharacterizationService.

There is also synthesize_for_user_sync() — a synchronous variant that calls the
background model inline and returns the result. It exists for dev/testing (the
`?sync=true` trigger) and for setups where the async queue callback isn't wired.

Design notes:
  - Anti-drift by grounding: the raw user_memories facts are the immutable ground
    truth; each run RE-DERIVES the document from them (plus the prior view for
    episodic carry-forward), so it never photocopy-degrades across runs.
  - Reads transcripts NON-destructively (never marks them processed) so it never
    interferes with passive extraction, which consumes them.
  - Entirely off the voice hot path — the live inference path imports nothing here.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from uuid import uuid4

from app.services.characterization_service import CharacterizationService

logger = logging.getLogger("uvicorn")

_SYNTHESIS_SYSTEM_PROMPT = """You are the characterization synthesizer for a private voice assistant named Jarvis. \
Your job is to maintain a single evolving *view* of one person — who they are, what they care about right now, \
and how Jarvis should talk to them — so future conversations feel like they come from someone who genuinely knows them.

You are given THREE inputs:
1. FACTS — durable, verified facts already stored about this person. Treat these as GROUND TRUTH; never contradict them.
2. PRIOR VIEW — your previous characterization (may be empty on the first run). Use it to carry forward things that \
recent conversation no longer mentions, but revise it freely — it is not sacred.
3. RECENT CONVERSATIONS — the latest transcripts. Use these to update what the person is currently focused on and how \
they communicate.

Produce an updated characterization as a SINGLE JSON object with these fields:
- "summary": 2-4 sentences capturing who this person is, grounded in the facts.
- "current_focus": array of {"item": string, "confidence": number 0-1} — what they're working on or thinking about \
lately. RETIRE items that recent conversations have clearly moved past; do NOT let this list only grow.
- "communication_style": array of short strings — how they talk and how Jarvis should talk back \
(e.g. "prefers directness", "enjoys playful banter", "wants brevity").
- "relationships": array of {"name": string, "relation": string} — people or pets that matter to them.
- "constraints": array of {"item": string, "confidence": number 0-1} — allergies, dislikes, or hard limits Jarvis \
should quietly respect.
- "confidence": number 0-1 — your overall confidence in this view.
- "rendered": 3-5 sentences written TO Jarvis, briefing it on this person the way a colleague would before you meet them.

Rules:
- Ground everything in the inputs. Invent NOTHING. If you do not know something, leave it out.
- Prefer a smaller, truer view over a padded one.
- Confidence must reflect how well-supported a claim is by the facts and conversations.
- Return ONLY the JSON object — no prose, no markdown fences, no <think> content."""


# ---------------------------------------------------------------------------
# Batch driver + single-user entry points
# ---------------------------------------------------------------------------


async def run_synthesis_batch(max_users: int = 50) -> None:
    """For each user with transcripts newer than their last synthesis, enqueue a job."""
    from app.db import get_session_local
    from app.models import ConversationTranscript

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        char_svc = CharacterizationService(db)
        max_transcripts = _get_max_transcripts()

        pairs = (
            db.query(ConversationTranscript.user_id, ConversationTranscript.household_id)
            .distinct()
            .all()
        )
        if not pairs:
            return

        enqueued = 0
        for user_id, household_id in pairs:
            if enqueued >= max_users:
                break

            newest = (
                db.query(ConversationTranscript.created_at)
                .filter(
                    ConversationTranscript.user_id == user_id,
                    ConversationTranscript.household_id == household_id,
                )
                .order_by(ConversationTranscript.created_at.desc())
                .first()
            )
            if not newest:
                continue
            newest_at = newest[0]

            char = char_svc.get(user_id, household_id)
            if char and char.last_transcript_at and newest_at <= char.last_transcript_at:
                continue  # nothing new since last synthesis

            await _synthesize_one(db, user_id, household_id, max_transcripts)
            enqueued += 1

        if enqueued:
            logger.info("Characterization synthesis: enqueued %d user(s)", enqueued)
    except Exception as e:
        logger.error("Characterization synthesis batch failed: %s", e, exc_info=True)
    finally:
        db.close()


async def synthesize_for_user(user_id: int, household_id: str) -> None:
    """Enqueue a single user's synthesis immediately (async — result lands via callback)."""
    from app.db import get_session_local

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        await _synthesize_one(db, user_id, household_id, _get_max_transcripts())
    finally:
        db.close()


async def synthesize_for_user_sync(user_id: int, household_id: str) -> dict[str, Any] | None:
    """Synthesize inline via the background model and persist — returns the result dict.

    Dev/testing path (`?sync=true`) and a fallback where the async queue callback
    isn't wired. Returns None if the model output can't be parsed.
    """
    from app.db import get_session_local
    from app.core.llm_proxy_client import LLMProxyClient

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        facts_text, transcripts, prior_text = _gather(db, user_id, household_id, _get_max_transcripts())
        messages = _build_messages(facts_text, transcripts, prior_text)

        result = await LLMProxyClient().chat_completion(
            messages=messages,
            model="background",
            temperature=0.4,
            include_date_context=False,
            max_tokens=3000,
        )
        content = _extract_content(result)
        parsed = _parse_synthesis_response(content)
        if not parsed:
            logger.warning("Sync synthesis: unparseable model output for user %s", user_id)
            return None

        char = _apply_parsed(db, user_id, household_id, parsed, _newest_at(transcripts))
        return {
            "user_id": char.user_id,
            "household_id": char.household_id,
            "body": parsed,
            "rendered": char.rendered,
            "confidence": char.confidence,
            "version": char.version,
            "last_transcript_at": char.last_transcript_at.isoformat() if char.last_transcript_at else None,
            "model": char.model,
            "updated_at": char.updated_at.isoformat() if char.updated_at else None,
        }
    finally:
        db.close()


async def _synthesize_one(
    db: Any, user_id: int, household_id: str, max_transcripts: int
) -> None:
    """Gather one user's inputs, build the prompt, and enqueue an async job."""
    facts_text, transcripts, prior_text = _gather(db, user_id, household_id, max_transcripts)
    messages = _build_messages(facts_text, transcripts, prior_text)
    await _enqueue_synthesis(user_id, household_id, messages, _newest_at(transcripts), len(transcripts))


# ---------------------------------------------------------------------------
# Input gathering + prompt building (shared by async + sync paths)
# ---------------------------------------------------------------------------


def _gather(
    db: Any, user_id: int, household_id: str, max_transcripts: int
) -> tuple[str, list, str]:
    """Return (facts_text, recent_transcripts_oldest_first, prior_view_text)."""
    from app.models import ConversationTranscript
    from app.services.memory_service import MemoryService

    facts = MemoryService(db).get_active_memories(user_id, household_id)
    facts_text = (
        "\n".join(f"- [{m.category}] {m.content}" for m in facts) or "None stored yet."
    )

    transcripts = (
        db.query(ConversationTranscript)
        .filter(
            ConversationTranscript.user_id == user_id,
            ConversationTranscript.household_id == household_id,
        )
        .order_by(ConversationTranscript.created_at.desc())
        .limit(max_transcripts)
        .all()
    )
    transcripts = list(reversed(transcripts))  # oldest-first for natural reading

    prior = CharacterizationService(db).get(user_id, household_id)
    prior_text = prior.body if prior else "None yet — this is the first synthesis for this person."
    return facts_text, transcripts, prior_text


def _build_messages(facts_text: str, transcripts: list, prior_text: str) -> list[dict[str, str]]:
    lines: list[str] = []
    for t in transcripts:
        line = f"User: {t.user_message}"
        if t.assistant_message:
            line += f"\nJarvis: {t.assistant_message}"
        lines.append(line)
    transcript_text = "\n---\n".join(lines) or "None."

    return [
        {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"FACTS (ground truth):\n{facts_text}\n\n"
                f"PRIOR VIEW:\n{prior_text}\n\n"
                f"RECENT CONVERSATIONS:\n{transcript_text}"
            ),
        },
    ]


def _newest_at(transcripts: list) -> datetime | None:
    return max((t.created_at for t in transcripts), default=None)


# ---------------------------------------------------------------------------
# Async enqueue + callback
# ---------------------------------------------------------------------------


async def _enqueue_synthesis(
    user_id: int,
    household_id: str,
    messages: list,
    newest_at: datetime | None,
    transcript_count: int,
) -> None:
    """Enqueue a prebuilt synthesis prompt via the LLM proxy queue."""
    from app.core.utils.rest_client import post
    from app.services.memory_extraction_service import (
        _get_command_center_url,
        _get_llm_proxy_url,
    )

    job_id = str(uuid4())

    cc_base_url = _get_command_center_url()
    callback_url = f"{cc_base_url}/api/v0/characterization-synthesis/callback"

    callback: dict[str, str] = {"url": callback_url}
    callback_token = os.getenv("JARVIS_ADAPTER_CALLBACK_TOKEN")
    if callback_token:
        callback["auth_type"] = "bearer"
        callback["token"] = callback_token

    queue_payload = {
        "job_id": job_id,
        "job_type": "chat",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "trace_id": job_id,
        "idempotency_key": job_id,
        "job_type_version": "v1",
        "ttl_seconds": 600,
        "metadata": {
            "type": "characterization_synthesis",
            "user_id": user_id,
            "household_id": household_id,
            "last_transcript_at": newest_at.isoformat() if newest_at else None,
            "transcript_count": transcript_count,
        },
        "request": {
            "model": "background",
            "messages": messages,
            "sampling": {
                "temperature": 0.4,
            },
        },
        "callback": callback,
    }

    llm_proxy_url = _get_llm_proxy_url()
    queue_url = f"{llm_proxy_url.rstrip('/')}/internal/queue/enqueue"

    headers: dict[str, str] = {}
    internal_token = os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if internal_token:
        headers["X-Internal-Token"] = internal_token

    await post(url=queue_url, json_data=queue_payload, headers=headers)
    logger.info(
        "Enqueued characterization synthesis: job_id=%s, user_id=%s, transcripts=%d",
        job_id, user_id, transcript_count,
    )


async def handle_synthesis_callback(payload: dict[str, Any]) -> None:
    """Parse the LLM response and upsert the characterization."""
    from app.db import get_session_local

    job_id: str = payload.get("job_id", "unknown")
    status: str = payload.get("status", "")
    metadata: dict[str, Any] = payload.get("metadata", {})
    user_id: int = metadata.get("user_id", 0)
    household_id: str = metadata.get("household_id", "")

    if status != "succeeded":
        error = payload.get("error", {})
        logger.error(
            "Characterization synthesis failed for job %s (user %s): %s",
            job_id, user_id, error.get("message", "Unknown error"),
        )
        return

    result = payload.get("result", {})
    content = result.get("content", "")
    if not content:
        logger.warning("Empty synthesis result for job %s", job_id)
        return

    parsed = _parse_synthesis_response(content)
    if not parsed:
        logger.warning("Unparseable synthesis result for job %s (user %s)", job_id, user_id)
        return

    last_transcript_at = _parse_iso(metadata.get("last_transcript_at"))

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        char = _apply_parsed(db, user_id, household_id, parsed, last_transcript_at)
        logger.info(
            "Synthesized characterization for user %s (job %s) -> v%d",
            user_id, job_id, char.version,
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Persistence + parsing helpers
# ---------------------------------------------------------------------------


def _apply_parsed(
    db: Any,
    user_id: int,
    household_id: str,
    parsed: dict[str, Any],
    last_transcript_at: datetime | None,
):
    """Map a parsed characterization dict onto an upsert (shared by sync + callback)."""
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None
    rendered = parsed.get("rendered") if isinstance(parsed.get("rendered"), str) else None

    return CharacterizationService(db).upsert(
        user_id=user_id,
        household_id=household_id,
        body=parsed,
        rendered=rendered,
        confidence=float(confidence) if confidence is not None else None,
        last_transcript_at=last_transcript_at,
        model="background",
    )


def _extract_content(result: Any) -> str:
    """Pull assistant text out of a chat-completion result (OpenAI shape, with fallbacks)."""
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
    msg = result.get("message")
    if isinstance(msg, dict) and msg.get("content"):
        return msg["content"]
    return result.get("content") or ""


def _parse_synthesis_response(content: str) -> dict[str, Any] | None:
    """Parse the LLM synthesis response into a characterization dict.

    Handles raw JSON, JSON wrapped in markdown fences, and <think> preambles.
    """
    import re

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            logger.warning("No JSON object found in synthesis response: %s", cleaned[:200])
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse synthesis response: %s", cleaned[:200])
            return None

    if not isinstance(parsed, dict):
        logger.warning("Synthesis response is not an object: %s", type(parsed))
        return None
    return parsed


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO datetime string back into a naive datetime, tolerating None."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    # Store naive UTC to match the rest of the schema (columns are naive).
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _get_max_transcripts() -> int:
    """Read the synthesis transcript window from settings (default 50)."""
    from app.services.settings_service import get_settings_service

    try:
        return int(get_settings_service().get("characterization.max_transcripts") or 50)
    except (TypeError, ValueError):
        return 50
