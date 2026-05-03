"""Background agents that periodically inject context into the memory system.

Each agent fetches data from an external source (RSS feeds, Google Calendar,
etc.) and pushes it into the user_memories table via MemoryService so that
Jarvis has proactive awareness of current events, calendar, weather, etc.

Agents run as asyncio tasks created in main.py startup.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.services.agent_service import fetch_news_headlines
from app.services.memory_service import MemoryService

logger = logging.getLogger("uvicorn")


# =========================================================================
# News Agent
# =========================================================================


def refresh_news_context(
    db: Session,
    household_id: str,
    categories: list[str] | None = None,
    headlines_per_category: int = 5,
) -> int:
    """Fetch RSS headlines and inject as household-wide memories.

    Args:
        db: Database session
        household_id: Target household
        categories: RSS categories to fetch (default: ["general"])
        headlines_per_category: Max headlines per category

    Returns:
        Number of memories injected/updated
    """
    if not categories:
        categories = ["general"]

    service = MemoryService(db)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    count = 0

    for category in categories:
        try:
            articles = fetch_news_headlines(
                category=category, count=headlines_per_category
            )
        except Exception as e:
            logger.warning(
                "News agent: failed to fetch %s headlines: %s", category, e
            )
            continue

        for article in articles:
            title = article.get("title", "").strip()
            if not title:
                continue

            source = article.get("source", "")
            summary = article.get("summary", "")

            # Build concise memory content
            content = title
            if summary:
                # Truncate long summaries
                clean_summary = summary[:200].strip()
                if clean_summary and clean_summary != title:
                    content = f"{title} — {clean_summary}"

            # Stable key: category + date + title hash
            title_hash = hashlib.md5(title.lower().encode()).hexdigest()[:8]
            key = f"news:{category}:{today}:{title_hash}"

            try:
                service.save_memory(
                    user_id=None,  # household-wide
                    household_id=household_id,
                    content=content,
                    category="news",
                    key=key,
                    source=f"news-agent:{source}",
                    expires_at=datetime.utcnow() + timedelta(hours=24),
                )
                count += 1
            except Exception as e:
                logger.warning("News agent: failed to save headline: %s", e)

    # Best-effort embedding generation for all new/updated news memories
    _embed_recent_household_memories(db, household_id, source_prefix="news-agent")

    if count:
        logger.info(
            "News agent: injected %d headlines for household %s",
            count, household_id,
        )
    return count


# =========================================================================
# Calendar Agent
# =========================================================================

GOOGLE_CALENDAR_API = "https://www.googleapis.com/calendar/v3"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def refresh_calendar_context(
    db: Session,
    household_id: str,
    look_ahead_days: int = 2,
) -> int:
    """Fetch calendar events from linked Google Calendar sessions and inject.

    Queries AuthSession for active/consumed Google Calendar sessions linked
    to nodes in the given household. Refreshes expired access tokens
    automatically.

    Args:
        db: Database session
        household_id: Target household
        look_ahead_days: How many days ahead to fetch events

    Returns:
        Number of event memories injected/updated
    """
    from app.models import AuthSession, Node

    # Find Google Calendar sessions for nodes in this household
    sessions = (
        db.query(AuthSession)
        .join(Node, AuthSession.node_id == Node.node_id)
        .filter(
            Node.household_id == household_id,
            AuthSession.provider == "google_calendar",
            AuthSession.status.in_(["active", "consumed"]),
            AuthSession.access_token_enc.isnot(None),
        )
        .all()
    )

    if not sessions:
        return 0

    service = MemoryService(db)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    count = 0

    # Deduplicate by node_id (use most recent session per node)
    seen_nodes: set[str] = set()
    for session in sessions:
        if session.node_id in seen_nodes:
            continue
        seen_nodes.add(session.node_id)

        try:
            access_token = _get_valid_access_token(db, session)
            if not access_token:
                continue

            events = _fetch_google_events(
                access_token, look_ahead_days=look_ahead_days
            )

            for event in events:
                summary = event.get("summary", "No Title")
                start = event.get("start_display", "")
                end = event.get("end_display", "")
                location = event.get("location", "")
                is_all_day = event.get("is_all_day", False)

                # Format as natural language
                if is_all_day:
                    content = f"{start}: {summary} (all day)"
                else:
                    content = f"{start} - {end}: {summary}"
                if location:
                    content += f" at {location}"

                event_id = event.get("id", "")
                key = f"calendar:{today}:{event_id}"

                # Calendar events expire at end of event + 1 hour buffer
                event_end_dt = event.get("end_dt")
                if event_end_dt and isinstance(event_end_dt, datetime):
                    expires_at = event_end_dt + timedelta(hours=1)
                else:
                    expires_at = datetime.utcnow() + timedelta(
                        hours=24 * look_ahead_days
                    )

                try:
                    service.save_memory(
                        user_id=None,  # household-wide
                        household_id=household_id,
                        content=content,
                        category="calendar",
                        key=key,
                        source="calendar-agent",
                        expires_at=expires_at,
                    )
                    count += 1
                except Exception as e:
                    logger.warning(
                        "Calendar agent: failed to save event: %s", e
                    )

        except Exception as e:
            logger.warning(
                "Calendar agent: failed for node %s: %s",
                session.node_id, e,
            )

    _embed_recent_household_memories(db, household_id, source_prefix="calendar-agent")

    if count:
        logger.info(
            "Calendar agent: injected %d events for household %s",
            count, household_id,
        )
    return count


def _get_valid_access_token(db: Session, session: Any) -> str | None:
    """Decrypt and optionally refresh the Google OAuth access token.

    If the token is expired (or returns 401), attempts a refresh using
    the stored refresh_token. Updates the stored encrypted tokens on
    successful refresh.

    Returns:
        Valid access token string, or None if unavailable.
    """
    from app.api.oauth import _decrypt_value, _encrypt_value

    if not session.access_token_enc:
        return None

    access_token = _decrypt_value(session.access_token_enc)

    # Quick validity check — try a lightweight API call
    try:
        resp = httpx.get(
            f"{GOOGLE_CALENDAR_API}/users/me/calendarList",
            params={"maxResults": "1"},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            return access_token
    except httpx.RequestError:
        pass

    # Token expired or invalid — try refresh
    if not session.refresh_token_enc:
        logger.debug("Calendar agent: no refresh token for session %s", session.id[:8])
        return None

    refresh_token = _decrypt_value(session.refresh_token_enc)

    try:
        resp = httpx.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": session.client_id,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        token_data = resp.json()
    except Exception as e:
        logger.warning(
            "Calendar agent: token refresh failed for session %s: %s",
            session.id[:8], e,
        )
        return None

    new_access_token = token_data.get("access_token")
    if not new_access_token:
        return None

    # Update stored encrypted token
    session.access_token_enc = _encrypt_value(new_access_token)
    if token_data.get("refresh_token"):
        session.refresh_token_enc = _encrypt_value(token_data["refresh_token"])
    session.token_data_enc = _encrypt_value(json.dumps(token_data))
    db.commit()

    logger.info("Calendar agent: refreshed access token for session %s", session.id[:8])
    return new_access_token


def _fetch_google_events(
    access_token: str,
    look_ahead_days: int = 2,
) -> list[dict[str, Any]]:
    """Fetch events from Google Calendar REST API.

    Returns a simplified list of event dicts with display-friendly fields.
    """
    now = datetime.utcnow()
    time_min = now.strftime("%Y-%m-%dT00:00:00Z")
    time_max = (now + timedelta(days=look_ahead_days)).strftime("%Y-%m-%dT00:00:00Z")

    resp = httpx.get(
        f"{GOOGLE_CALENDAR_API}/calendars/primary/events",
        params={
            "timeMin": time_min,
            "timeMax": time_max,
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": "50",
        },
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])

    events: list[dict[str, Any]] = []
    for item in items:
        try:
            start_raw = item.get("start", {})
            end_raw = item.get("end", {})
            is_all_day = "date" in start_raw and "dateTime" not in start_raw

            if is_all_day:
                start_dt = datetime.strptime(start_raw["date"], "%Y-%m-%d")
                end_dt = datetime.strptime(
                    end_raw.get("date", start_raw["date"]), "%Y-%m-%d"
                )
                start_display = start_dt.strftime("%A %b %d")
                end_display = end_dt.strftime("%A %b %d")
            else:
                start_dt = datetime.fromisoformat(
                    start_raw["dateTime"].replace("Z", "+00:00")
                )
                end_dt = datetime.fromisoformat(
                    end_raw["dateTime"].replace("Z", "+00:00")
                )
                start_display = start_dt.strftime("%A %b %d %I:%M%p")
                end_display = end_dt.strftime("%I:%M%p")

            events.append({
                "id": item.get("id", ""),
                "summary": item.get("summary", "No Title"),
                "start_display": start_display,
                "end_display": end_display,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "location": item.get("location", ""),
                "is_all_day": is_all_day,
            })
        except Exception as e:
            logger.debug("Calendar agent: skipping unparseable event: %s", e)

    return events


# =========================================================================
# Shared helpers
# =========================================================================


def _embed_recent_household_memories(
    db: Session,
    household_id: str,
    source_prefix: str,
) -> None:
    """Best-effort batch embedding for recently injected agent memories."""
    try:
        from app.core.llm_proxy_client import LLMProxyClient
        from app.models import UserMemory

        # Find agent memories without embeddings
        memories = (
            db.query(UserMemory)
            .filter(
                UserMemory.user_id.is_(None),
                UserMemory.household_id == household_id,
                UserMemory.is_active == True,  # noqa: E712
                UserMemory.embedding.is_(None),
                UserMemory.source.like(f"{source_prefix}%"),
            )
            .limit(50)
            .all()
        )

        if not memories:
            return

        client = LLMProxyClient()
        contents = [m.content for m in memories]
        vectors = client.create_embeddings_sync(contents)

        if vectors and len(vectors) == len(memories):
            service = MemoryService(db)
            for i, memory in enumerate(memories):
                if vectors[i]:
                    service.update_embedding(memory.id, vectors[i])

        logger.debug(
            "Embedded %d %s memories for household %s",
            len(memories), source_prefix, household_id,
        )
    except Exception as e:
        logger.debug("Agent embedding generation failed (non-fatal): %s", e)


def get_active_household_ids(db: Session) -> list[str]:
    """Get household IDs that have at least one active node.

    Used by background loops to determine which households to refresh.
    """
    from app.models import Node

    rows = (
        db.query(Node.household_id)
        .filter(
            Node.household_id.isnot(None),
            Node.is_active == True,  # noqa: E712
        )
        .distinct()
        .all()
    )
    return [r[0] for r in rows if r[0]]
