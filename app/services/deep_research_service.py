"""Deep research pipeline — search, scrape, summarize, deliver.

Called as a background task from deep_research_tool.py.
"""

import logging
import os
import time
from typing import Any

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
    """Execute the full research pipeline."""
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

    # Step 3: Summarize with LLM
    summary = await _summarize(query, successful, search_results)

    elapsed_s = time.monotonic() - start_time
    logger.info("Research complete in %.1fs: %r", elapsed_s, query)

    # Step 4: Store in inbox
    # Build summary preview (first 200 chars of the LLM summary)
    preview = summary[:200].rsplit(" ", 1)[0] + "..." if len(summary) > 200 else summary

    metadata = {
        "query": query,
        "depth": depth,
        "sources": [
            {"title": r.get("title", ""), "url": r["url"]}
            for r in search_results
        ],
        "pages_scraped": len(successful),
        "pages_attempted": len(urls),
        "elapsed_seconds": round(elapsed_s, 1),
    }

    inbox_item_id = await _store_inbox_item(
        household_id=household_id,
        user_id=speaker_user_id,
        title=f"Research: {query}",
        summary=preview,
        body=summary,
        metadata=metadata,
    )

    # Step 5: Send push notification
    await _send_notification(
        household_id=household_id,
        title="Research Complete",
        body=f"Results ready: {query}",
        data={
            "type": "deep_research",
            "inbox_item_id": inbox_item_id,
        },
    )


async def _search_web(query: str, num_results: int) -> list[dict[str, str]]:
    """Search using DuckDuckGo via the ddgs package."""
    try:
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


async def _summarize(query: str, pages: list, search_results: list[dict]) -> str:  # noqa: ARG001
    """Summarize scraped content using the LLM proxy."""
    from app.core.llm_proxy_client import LLMProxyClient

    # Build source context
    source_texts: list[str] = []
    for i, page in enumerate(pages, 1):
        title = page.title or f"Source {i}"
        source_texts.append(
            f"## Source {i}: {title}\nURL: {page.url}\n\n{page.text_content}"
        )

    user_content = (
        f"Research query: {query}\n\n"
        f"{'---'.join(source_texts)}\n\n"
        f"Please synthesize these {len(pages)} sources into a comprehensive research summary."
    )

    client = LLMProxyClient()
    response = await client.chat_completion(
        messages=[
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("LLM returned empty summary")

    return content


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
