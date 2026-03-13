"""Agent service — stateless utilities for background agent data fetching.

Provides RSS parsing and calendar context for node-side agents
via the /api/v0/agents/* endpoints.
"""

import calendar as cal_mod
from datetime import datetime
from typing import Any

import feedparser
from jarvis_log_client import JarvisLogger

logger = JarvisLogger(service="jarvis-command-center")

# Default RSS feeds by category
_DEFAULT_FEEDS: dict[str, list[str]] = {
    "general": [
        "https://feeds.apnews.com/rss/apf-topnews",
        "https://feeds.bbci.co.uk/news/rss.xml",
    ],
    "tech": [
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    ],
    "sports": [
        "https://www.espn.com/espn/rss/ncb/news",
        "https://feeds.apnews.com/rss/apf-sports",
    ],
    "business": [
        "https://feeds.reuters.com/reuters/businessNews",
    ],
    "science": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    ],
    "health": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
    ],
}


def fetch_news_headlines(
    category: str = "general",
    count: int = 5,
) -> list[dict[str, Any]]:
    """Fetch RSS headlines for a category.

    Returns a list of article dicts sorted newest-first.
    """
    feed_urls = list(_DEFAULT_FEEDS.get(category, _DEFAULT_FEEDS["general"]))

    all_articles: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for url in feed_urls:
        try:
            parsed = feedparser.parse(url)
            source = getattr(parsed.feed, "title", url)

            for entry in parsed.entries:
                title = getattr(entry, "title", None)
                if not title:
                    continue

                title_key = title.strip().lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                published_parsed = getattr(entry, "published_parsed", None)
                published_ts = cal_mod.timegm(published_parsed) if published_parsed else 0

                pub_str = None
                if published_parsed:
                    try:
                        pub_str = datetime(*published_parsed[:6]).strftime("%Y-%m-%d")
                    except (TypeError, ValueError):
                        pass

                all_articles.append({
                    "title": title,
                    "summary": getattr(entry, "summary", ""),
                    "source": source,
                    "published": pub_str,
                    "_sort_ts": published_ts,
                })
        except Exception as e:
            logger.warning("Failed to fetch RSS feed", url=url, error=str(e))

    all_articles.sort(key=lambda a: a["_sort_ts"], reverse=True)

    # Remove internal sort key
    for article in all_articles:
        article.pop("_sort_ts", None)

    return all_articles[:count]
