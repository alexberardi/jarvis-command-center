"""Deep research pipeline — search, scrape, enqueue LLM summarization, deliver.

Two-phase flow:
  1. run_research() — search + scrape + enqueue LLM job via Redis queue
  2. handle_summarization_callback() — called by queue callback, stores inbox + sends push
"""

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

logger = logging.getLogger("uvicorn")

# Search result counts by depth
_RESULT_COUNTS = {"quick": 3, "thorough": 6}

_SUMMARIZE_SYSTEM_PROMPT = """You are a research analyst. Synthesize the following web sources into a comprehensive, well-organized summary.

Guidelines:
- Use markdown formatting with headers for major themes
- Cite sources inline using [Source Title](URL) format
- Note areas of agreement and disagreement between sources
- Highlight key findings, recommendations, and caveats
- Keep the summary focused and actionable
- If sources are insufficient, acknowledge limitations"""


async def run_research(
    query: str,
    depth: str,
    household_id: str,
    speaker_user_id: int | None,
) -> None:
    """Phase 1: Search the web, scrape results, enqueue LLM summarization."""
    start_time = time.monotonic()
    logger.info("Starting deep research: query=%r, depth=%s", query, depth)

    # Step 1: Search
    num_results = _RESULT_COUNTS.get(depth, 3)
    search_results = await _search_web(query, num_results)
    if not search_results:
        raise ValueError("No search results found")

    urls = [r["url"] for r in search_results]
    logger.info("Found %d search results for %r", len(urls), query)

    # Step 2: Scrape
    scraped_pages = await _scrape_urls(urls)
    successful = [p for p in scraped_pages if p.ok]
    if not successful:
        raise ValueError("Could not scrape any of the search results")

    logger.info("Scraped %d/%d pages successfully", len(successful), len(urls))

    # Step 3: Build LLM messages and enqueue via Redis queue
    source_texts: list[str] = []
    for i, page in enumerate(successful, 1):
        title = page.title or f"Source {i}"
        source_texts.append(
            f"## Source {i}: {title}\nURL: {page.url}\n\n{page.text_content}"
        )

    user_content = (
        f"Research query: {query}\n\n"
        f"{'---'.join(source_texts)}\n\n"
        f"Please synthesize these {len(successful)} sources into a comprehensive research summary."
    )

    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    scrape_elapsed = time.monotonic() - start_time

    # Metadata to pass through the queue callback
    research_metadata = {
        "query": query,
        "depth": depth,
        "household_id": household_id,
        "speaker_user_id": speaker_user_id,
        "sources": [
            {"title": r.get("title", ""), "url": r["url"]}
            for r in search_results
        ],
        "pages_scraped": len(successful),
        "pages_attempted": len(urls),
        "scrape_elapsed_seconds": round(scrape_elapsed, 1),
    }

    await _enqueue_summarization(messages, research_metadata)
    logger.info(
        "Enqueued LLM summarization for %r (scraped in %.1fs)",
        query, scrape_elapsed,
    )


async def handle_summarization_callback(payload: dict[str, Any]) -> None:
    """Phase 2: Queue callback — extract summary, store inbox item, send push.

    Called by the /api/v0/deep-research/callback endpoint.
    """
    job_id: str = payload.get("job_id", "unknown")
    status: str = payload.get("status", "")
    metadata: dict[str, Any] = payload.get("metadata", {})
    query: str = metadata.get("query", "unknown")
    household_id: str = metadata.get("household_id", "")

    if status != "succeeded":
        error = payload.get("error", {})
        error_msg = error.get("message", "Unknown error")
        logger.error("Research summarization failed for job %s: %s", job_id, error_msg)
        if household_id:
            await _send_notification(
                household_id=household_id,
                title="Research Failed",
                body=f"Research on \"{query}\" failed: {error_msg}",
                data={"type": "deep_research_failed"},
            )
        return

    # Extract the LLM summary from the queue result
    result = payload.get("result", {})
    summary = result.get("content", "")
    if not summary:
        logger.error("Empty summary from LLM for research job %s", job_id)
        if household_id:
            await _send_notification(
                household_id=household_id,
                title="Research Failed",
                body=f"Research on \"{query}\" produced an empty summary",
                data={"type": "deep_research_failed"},
            )
        return

    speaker_user_id: int | None = metadata.get("speaker_user_id")
    depth: str = metadata.get("depth", "quick")
    scrape_elapsed: float = metadata.get("scrape_elapsed_seconds", 0)

    # Add LLM timing from the queue result
    timing = payload.get("timing", {})
    llm_elapsed_ms: int = timing.get("processing_ms", 0)
    total_elapsed = scrape_elapsed + (llm_elapsed_ms / 1000)

    # Strip <think> blocks before generating preview
    clean_summary = re.sub(r"<think>[\s\S]*?</think>", "", summary).strip()
    preview = clean_summary[:200].rsplit(" ", 1)[0] + "..." if len(clean_summary) > 200 else clean_summary

    inbox_metadata = {
        "query": query,
        "depth": depth,
        "sources": metadata.get("sources", []),
        "pages_scraped": metadata.get("pages_scraped", 0),
        "pages_attempted": metadata.get("pages_attempted", 0),
        "elapsed_seconds": round(total_elapsed, 1),
    }

    inbox_item_id = await _store_inbox_item(
        household_id=household_id,
        user_id=speaker_user_id,
        title=f"Research: {query}",
        summary=preview,
        body=summary,
        metadata=inbox_metadata,
    )

    await _send_notification(
        household_id=household_id,
        title="Research Complete",
        body=f"Results ready: {query}",
        data={
            "type": "deep_research",
            "inbox_item_id": inbox_item_id,
        },
    )

    logger.info(
        "Research delivered: query=%r, inbox_item=%s, total=%.1fs (scrape=%.1fs, llm=%.1fs)",
        query, inbox_item_id, total_elapsed, scrape_elapsed, llm_elapsed_ms / 1000,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _search_web(query: str, num_results: int) -> list[dict[str, str]]:
    """Search using DuckDuckGo via the ddgs package."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return []


async def _scrape_urls(urls: list[str]) -> list:
    """Scrape URLs using jarvis-web-scraper."""
    from jarvis_web_scraper import WebScraper

    scraper = WebScraper()
    return await scraper.batch_fetch(urls, max_concurrent=3, max_chars=6000)


async def _enqueue_summarization(
    messages: list[dict[str, str]],
    research_metadata: dict[str, Any],
) -> None:
    """Submit LLM summarization job to the Redis queue."""
    from app.core.utils.rest_client import post, build_jarvis_app_headers

    job_id = str(uuid4())

    # Build callback URL
    cc_base_url = _get_command_center_url()
    callback_url = f"{cc_base_url}/api/v0/deep-research/callback"

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
        "ttl_seconds": 600,  # 10 min — research shouldn't take longer
        "metadata": research_metadata,
        "request": {
            "model": "background",
            "messages": messages,
            "sampling": {
                "temperature": 0.3,
            },
        },
        "callback": callback,
    }

    # Get LLM proxy queue URL
    llm_proxy_url = _get_llm_proxy_url()
    queue_url = f"{llm_proxy_url.rstrip('/')}/internal/queue/enqueue"

    headers: dict[str, str] = {}
    internal_token = os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if internal_token:
        headers["X-Internal-Token"] = internal_token

    result = await post(url=queue_url, json_data=queue_payload, headers=headers)

    deduped = result.get("deduped", False)
    logger.info(
        "Enqueued research summarization: job_id=%s, deduped=%s",
        job_id, deduped,
    )


async def _store_inbox_item(
    household_id: str,
    user_id: int | None,
    title: str,
    summary: str,
    body: str,
    metadata: dict[str, Any],
) -> str:
    """Store research results in the notifications inbox via API."""
    notifications_url = _get_notifications_url()

    payload: dict[str, Any] = {
        "household_id": household_id,
        "title": title,
        "summary": summary,
        "body": body,
        "category": "deep_research",
        "source_service": "jarvis-command-center",
        "user_id": user_id,
        "metadata": metadata,
    }

    app_headers = _get_app_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{notifications_url}/api/v0/inbox",
            json=payload,
            headers=app_headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["id"]


async def _send_notification(
    household_id: str,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Send a push notification via jarvis-notifications."""
    notifications_url = _get_notifications_url()

    payload: dict[str, Any] = {
        "target_type": "household",
        "target_id": household_id,
        "title": title,
        "body": body,
        "data": data,
        "priority": "default",
        "category": "deep_research",
    }

    app_headers = _get_app_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{notifications_url}/api/v0/notify",
            json=payload,
            headers=app_headers,
        )
        if resp.status_code != 200:
            logger.warning("Push notification failed: %s %s", resp.status_code, resp.text)


def _get_llm_proxy_url() -> str:
    """Get LLM proxy URL from service discovery or env."""
    try:
        from app.core import service_config
        if service_config.is_initialized():
            return service_config.get_llm_proxy_url()
    except (ImportError, AttributeError):
        pass
    return os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:7704")


def _get_command_center_url() -> str:
    """Get command-center's externally-reachable URL for callback registration.

    Checks the ``network.public_url`` setting first (configurable at runtime),
    then falls back to env vars, then localhost.
    """
    try:
        from app.services.settings_service import get_settings_service
        public_url = get_settings_service().get("network.public_url")
        if public_url:
            return public_url.rstrip("/")
    except Exception:
        pass
    return os.getenv(
        "CC_PUBLIC_URL",
        os.getenv("JARVIS_COMMAND_CENTER_URL", "http://localhost:7703"),
    )


def _get_notifications_url() -> str:
    """Get jarvis-notifications service URL."""
    try:
        from app.core import service_config
        if service_config.is_initialized():
            url = service_config.get_service_url("notifications")
            if url:
                return url
    except (ImportError, AttributeError, Exception):
        pass
    return os.getenv("JARVIS_NOTIFICATIONS_URL", "http://localhost:7712")


def _get_app_headers() -> dict[str, str]:
    """Get app-to-app auth headers."""
    from app.core.utils.rest_client import build_jarvis_app_headers
    return build_jarvis_app_headers()
