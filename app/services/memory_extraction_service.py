"""Passive memory extraction — batch transcripts, extract via background LLM, upsert.

Two-phase flow (same pattern as deep_research_service):
  1. run_extraction_batch() — find unprocessed transcripts, enqueue LLM extraction job
  2. handle_extraction_callback() — parse extracted memories, upsert via MemoryService
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger("uvicorn")

_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant for a voice assistant called Jarvis. \
Given conversation transcripts between Jarvis and a user, extract personal facts, preferences, \
and habits worth remembering for future conversations.

What TO extract:
- Names of family members, pets, friends mentioned naturally — capture the name even when it arrives indirectly across the conversation
- Location preferences (city they check weather for = likely where they live)
- Food preferences, dietary info, and allergies
- Music/entertainment preferences
- Hobbies, activities, and recurring routines
- Work/schedule patterns
- Dated commitments the user mentions — appointments, trips, deadlines, visitors, events tied to a day or date ("dentist Friday", "flying to Denver Thursday", "brother staying this week"). Capture these with a short ttl_days so they expire after they pass.

What to SKIP:
- The specific request/command itself ("set a timer", "check the weather", "remind me to…") — those are ephemeral
- One-time facts with no future value
- Verification codes, passwords, or other one-time secrets — never store these
- Information already in the existing memories listed below

Each memory can optionally include "ttl_days" — how long it stays relevant (choose deliberately):
- Durable identity facts & preferences (family names, allergies, "likes coffee black"): omit ttl_days (permanent)
- Recurring habits ("picks up Emma from soccer Tuesdays"): ttl_days: 30
- Dated one-offs ("dentist appointment Friday", "flight Thursday"): ttl_days: 7
Do NOT store a transient or administrative event (a one-off meeting, a today-only note) as a permanent fact — give it a short ttl_days, or skip it.

Example — if the user says "Set a timer for 10 minutes, I am grilling steaks for my brother Mike":
[{"category": "fact", "key": "brother_name", "content": "Has a brother named Mike"}, \
{"category": "preference", "key": "cooking_style", "content": "Enjoys grilling"}]

Output ONLY the JSON array — no reasoning, no explanation, and no <think> block. If nothing is worth remembering, output []."""


def _expires_at_from_ttl(ttl_days: Any) -> datetime | None:
    """Convert an optional ``ttl_days`` into a naive-UTC expiry ("for how long").

    None/absent → permanent (no expiry). A positive int (or int-like string) → now
    + that many days. Anything unparseable → permanent — fail SAFE by keeping the
    memory rather than expiring it instantly.
    """
    if ttl_days is None:
        return None
    try:
        return datetime.utcnow() + timedelta(days=int(ttl_days))
    except (TypeError, ValueError):
        return None


async def run_extraction_batch() -> None:
    """For each user with unprocessed transcripts, batch and enqueue extraction."""
    from app.db import get_session_local
    from app.services.transcript_service import TranscriptService
    from app.services.memory_service import MemoryService

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        transcript_svc = TranscriptService(db)
        memory_svc = MemoryService(db)

        # Reset stale in-flight jobs (callback never arrived)
        transcript_svc.reset_stale_jobs(max_age_minutes=30)

        users = transcript_svc.get_users_with_unprocessed()
        if not users:
            return

        logger.info("Memory extraction: found %d users with unprocessed transcripts", len(users))

        for user_id, household_id in users:
            transcripts = transcript_svc.get_unprocessed_for_user(user_id, household_id, limit=20)
            if not transcripts:
                continue

            # Build existing memories context for dedup
            existing_memories = memory_svc.get_memories_for_prompt(
                user_id, household_id, max_chars=1000
            )

            await _enqueue_extraction(
                user_id=user_id,
                household_id=household_id,
                transcripts=transcripts,
                existing_memories=existing_memories or "None stored yet.",
                transcript_svc=transcript_svc,
            )
    except Exception as e:
        logger.error("Memory extraction batch failed: %s", e, exc_info=True)
    finally:
        db.close()


async def _enqueue_extraction(
    user_id: int,
    household_id: str,
    transcripts: list,
    existing_memories: str,
    transcript_svc: Any,
) -> None:
    """Build extraction prompt, enqueue via LLM proxy Redis queue."""
    from app.core.utils.rest_client import post

    job_id = str(uuid4())

    # Format transcript batch
    transcript_lines: list[str] = []
    for t in transcripts:
        line = f"User: {t.user_message}"
        if t.assistant_message:
            line += f"\nJarvis: {t.assistant_message}"
        if t.tool_calls_json:
            try:
                calls = json.loads(t.tool_calls_json)
                tool_names = [c.get("name", "?") for c in calls if isinstance(c, dict)]
                if tool_names:
                    line += f"\n[Tools called: {', '.join(tool_names)}]"
            except (json.JSONDecodeError, TypeError):
                pass
        transcript_lines.append(line)

    transcript_text = "\n---\n".join(transcript_lines)

    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Existing memories for this user:\n{existing_memories}\n\n"
                f"Recent conversations:\n{transcript_text}"
            ),
        },
    ]

    # Build callback URL
    cc_base_url = _get_command_center_url()
    callback_url = f"{cc_base_url}/api/v0/memory-extraction/callback"

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
            "type": "memory_extraction",
            "user_id": user_id,
            "household_id": household_id,
            "transcript_count": len(transcripts),
        },
        "request": {
            "model": "background",
            "messages": messages,
            "sampling": {
                # Greedy: extraction is a structured factual task, and the eval showed
                # temp 0.3 swung the pass rate 56–72% run-to-run while temp 0 was stable
                # at ~67%. Consistency matters more than sampling diversity here.
                "temperature": 0.0,
            },
        },
        "callback": callback,
    }

    # Enqueue to LLM proxy
    llm_proxy_url = _get_llm_proxy_url()
    queue_url = f"{llm_proxy_url.rstrip('/')}/internal/queue/enqueue"

    headers: dict[str, str] = {}
    internal_token = os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if internal_token:
        headers["X-Internal-Token"] = internal_token

    result = await post(url=queue_url, json_data=queue_payload, headers=headers)

    # Mark transcripts as in-flight only after successful enqueue
    transcript_ids = [t.id for t in transcripts]
    transcript_svc.mark_extraction_in_flight(transcript_ids, job_id)

    deduped = result.get("deduped", False) if isinstance(result, dict) else False
    logger.info(
        "Enqueued memory extraction: job_id=%s, user_id=%d, transcripts=%d, deduped=%s",
        job_id, user_id, len(transcripts), deduped,
    )


async def handle_extraction_callback(payload: dict[str, Any]) -> None:
    """Parse LLM response, upsert memories, mark transcripts processed."""
    from app.db import get_session_local
    from app.services.transcript_service import TranscriptService
    from app.services.memory_service import MemoryService

    job_id: str = payload.get("job_id", "unknown")
    status: str = payload.get("status", "")
    metadata: dict[str, Any] = payload.get("metadata", {})
    user_id: int = metadata.get("user_id", 0)
    household_id: str = metadata.get("household_id", "")

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        transcript_svc = TranscriptService(db)

        if status != "succeeded":
            error = payload.get("error", {})
            logger.error(
                "Memory extraction failed for job %s (user %d): %s",
                job_id, user_id, error.get("message", "Unknown error"),
            )
            # Don't mark as processed — stale job reset will allow retry
            return

        result = payload.get("result", {})
        content = result.get("content", "")
        if not content:
            logger.warning("Empty extraction result for job %s", job_id)
            transcript_svc.mark_processed(job_id)
            return

        # Parse extracted memories from LLM response
        memories = _parse_extraction_response(content)

        if memories:
            memory_svc = MemoryService(db)
            for mem in memories:
                expires_at = _expires_at_from_ttl(mem.get("ttl_days"))

                memory_svc.save_memory(
                    user_id=user_id,
                    household_id=household_id,
                    content=mem["content"],
                    category=mem.get("category", "general"),
                    key=mem.get("key"),
                    source="passive",
                    expires_at=expires_at,
                )
            logger.info(
                "Extracted %d memories for user %d (job %s)",
                len(memories), user_id, job_id,
            )
        else:
            logger.info("No new memories extracted for user %d (job %s)", user_id, job_id)

        transcript_svc.mark_processed(job_id)

    finally:
        db.close()


def _parse_extraction_response(content: str) -> list[dict[str, str]]:
    """Parse LLM extraction response into memory dicts.

    Handles: raw JSON array, JSON wrapped in markdown fences, or
    content with <think> blocks that need stripping.
    """
    import re

    # Strip <think> blocks (Qwen3 chain-of-thought)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

    # A TRUNCATED <think> (opened, never closed) leaves reasoning text with no usable
    # JSON — salvage by dropping everything before the first array bracket so a partial
    # response that still reached the JSON can parse.
    if "<think>" in cleaned:
        start = cleaned.find("[")
        cleaned = cleaned[start:] if start != -1 else ""

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON array in the content
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Failed to parse extraction response: %s", cleaned[:200])
                return []
        else:
            logger.warning("No JSON array found in extraction response: %s", cleaned[:200])
            return []

    if not isinstance(parsed, list):
        logger.warning("Extraction response is not a list: %s", type(parsed))
        return []

    # Validate each memory has at least content
    valid: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict) and item.get("content"):
            valid.append(item)

    return valid


def _get_command_center_url() -> str:
    """Get the CC's own URL for callbacks."""
    from app.services.settings_service import get_settings_service
    url = get_settings_service().get("network.public_url")
    if url:
        return url.rstrip("/")
    return os.getenv("CC_PUBLIC_URL", "http://localhost:7703")


def _get_llm_proxy_url() -> str:
    """Get LLM proxy URL from service discovery or env."""
    try:
        from app.core import service_config
        if service_config.is_initialized():
            return service_config.get_llm_proxy_url()
    except (ImportError, AttributeError):
        pass
    return os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:7704")
