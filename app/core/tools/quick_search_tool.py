"""Quick Search Tool — inline web search during conversation.

Searches the web, scrapes the top 2 results, and returns the content
directly as a tool result so the LLM can synthesize an answer inline.
Unlike deep_research, this blocks until results are ready (~5-10s).

Example: "What's the latest on the Mars mission?"
"""

import logging
import time
from typing import Any, Dict

import httpx

from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")

_NUM_RESULTS = 2
_MAX_CHARS_PER_PAGE = 4000
_SCRAPE_TIMEOUT = 8


class QuickSearchTool(IServerTool):
    """Tool for inline web search during conversation."""

    @property
    def name(self) -> str:
        return "quick_search"

    @property
    def description(self) -> str:
        return (
            "Search the web and return results inline. ALWAYS use this tool "
            "when the user asks about current events, recent news, today's "
            "information, who holds a position now, scores, prices, weather, "
            "or anything that may have changed since your training data. "
            "Returns content from the top 2 web results for you to synthesize "
            "into an answer. For deeper research across many sources, use "
            "deep_research instead."
        )

    @property
    def included_system_prompt_text(self) -> str:
        return (
            "You have access to quick_search for real-time web lookups. "
            "Your training data has a cutoff date — for ANY question about "
            "current events, recent developments, who currently holds a "
            "position, live scores, or anything time-sensitive, you MUST "
            "call quick_search rather than guessing from potentially "
            "outdated knowledge."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Run search + scrape synchronously and return results inline."""
        query: str = kwargs.get("query", "")
        if not query:
            return {"error": "missing_query", "message": "A search query is required"}

        start_time = time.monotonic()

        try:
            # Step 1: Search (sync — ddgs uses requests internally)
            search_results = _search_web(query)
            if not search_results:
                return {
                    "error": "no_results",
                    "message": f"No search results found for: {query}",
                }

            logger.info("Quick search: found %d results for %r", len(search_results), query)

            # Step 2: Scrape (sync via httpx)
            sources = _scrape_results(search_results)

            elapsed = time.monotonic() - start_time
            logger.info("Quick search: completed in %.1fs for %r", elapsed, query)

            return {
                "query": query,
                "sources": sources,
                "elapsed_seconds": round(elapsed, 1),
            }

        except Exception as e:
            logger.error("Quick search failed for query=%r: %s", query, e, exc_info=True)
            return {"error": "search_failed", "message": f"Search failed: {e}"}


def _search_web(query: str) -> list[dict[str, str]]:
    """Search using DuckDuckGo (sync)."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=_NUM_RESULTS):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        logger.error("Quick search web search failed: %s", e)
        return []


def _scrape_results(search_results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Scrape search result URLs synchronously via httpx.

    Falls back to search snippets for pages that fail to scrape.
    """
    sources: list[dict[str, str]] = []

    with httpx.Client(timeout=_SCRAPE_TIMEOUT, follow_redirects=True) as client:
        for r in search_results:
            url = r["url"]
            title = r.get("title", "Untitled")
            try:
                resp = client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0)",
                })
                if resp.status_code == 200:
                    content = _extract_text(resp.text)
                    if content:
                        sources.append({
                            "title": title,
                            "url": url,
                            "content": content[:_MAX_CHARS_PER_PAGE],
                        })
                        continue
            except Exception as e:
                logger.debug("Quick search scrape failed for %s: %s", url, e)

            # Fallback to snippet
            snippet = r.get("snippet", "")
            if snippet:
                sources.append({
                    "title": title,
                    "url": url,
                    "content": snippet,
                })

    return sources


def _extract_text(html: str) -> str:
    """Extract readable text from HTML, stripping tags and excess whitespace."""
    import re

    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
