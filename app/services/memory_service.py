"""User memory service for persistent personalization.

Manages CRUD operations on user memories (preferences, facts, notes)
and formats them for injection into the LLM system prompt.

Supports:
- Pinned memories (always in prompt — identity facts like name, age)
- Vector search via pgvector for semantic recall
- Embedding generation on save (best-effort)
"""

import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import UserMemory

logger = logging.getLogger("uvicorn")

# Priority order for prompt formatting
_CATEGORY_PRIORITY = {"preference": 0, "fact": 1, "note": 2, "general": 3}


class MemoryService:
    """Service for managing user memories."""

    def __init__(self, db: Session):
        self.db = db

    def get_active_memories(
        self,
        user_id: int,
        household_id: str,
        categories: list[str] | None = None,
    ) -> list[UserMemory]:
        """Get all active (non-expired) memories for a user.

        Args:
            user_id: The user's ID
            household_id: The household scope
            categories: Optional filter by category

        Returns:
            List of active UserMemory objects
        """
        query = self.db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.household_id == household_id,
            UserMemory.is_active == True,  # noqa: E712
        )

        # Exclude expired memories
        query = query.filter(
            (UserMemory.expires_at == None) | (UserMemory.expires_at > datetime.utcnow())  # noqa: E711
        )

        if categories:
            query = query.filter(UserMemory.category.in_(categories))

        return query.order_by(UserMemory.updated_at.desc()).all()

    def get_pinned_memories(
        self,
        user_id: int,
        household_id: str,
    ) -> list[UserMemory]:
        """Get only pinned, active memories for a user.

        Pinned memories are identity facts (name, age, location) that should
        always be included in the system prompt.

        Args:
            user_id: The user's ID
            household_id: The household scope

        Returns:
            List of pinned UserMemory objects
        """
        return self.db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.household_id == household_id,
            UserMemory.is_active == True,  # noqa: E712
            UserMemory.is_pinned == True,  # noqa: E712
        ).order_by(UserMemory.updated_at.desc()).all()

    def save_memory(
        self,
        user_id: int | None,
        household_id: str,
        content: str,
        category: str = "general",
        key: str | None = None,
        source: str = "voice",
        is_pinned: bool = False,
        expires_at: datetime | None = None,
    ) -> UserMemory:
        """Save a memory, upserting if a matching key exists.

        Args:
            user_id: The user's ID, or None for household-wide memories
            household_id: The household scope
            content: The memory text
            category: Category (preference, fact, note, general, agent_context)
            key: Optional structured key for upsert matching
            source: Origin (voice, ui, system, agent)
            is_pinned: Whether this is a pinned identity fact
            expires_at: Optional expiration (memory auto-excluded after this date)

        Returns:
            The created or updated UserMemory
        """
        # Upsert: if key is provided, check for existing entry
        if key:
            filters = [
                UserMemory.household_id == household_id,
                UserMemory.key == key,
                UserMemory.is_active == True,  # noqa: E712
            ]
            # SQL NULL semantics: = NULL never matches, must use IS NULL
            if user_id is not None:
                filters.append(UserMemory.user_id == user_id)
            else:
                filters.append(UserMemory.user_id.is_(None))
            existing = self.db.query(UserMemory).filter(*filters).first()

            if existing:
                # If the text changed, drop the stale vector so recall can't match
                # the OLD content — the embedding sweep re-embeds it off the hot path.
                if existing.content != content:
                    existing.embedding = None
                existing.content = content
                existing.category = category
                existing.source = source
                existing.is_pinned = is_pinned
                existing.expires_at = expires_at
                existing.updated_at = datetime.utcnow()
                self.db.commit()
                self.db.refresh(existing)
                logger.info(f"Updated memory key={key} for user_id={user_id}")
                return existing

        memory = UserMemory(
            user_id=user_id,
            household_id=household_id,
            category=category,
            key=key,
            content=content,
            source=source,
            is_pinned=is_pinned,
            expires_at=expires_at,
        )
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        logger.info(f"Saved new memory for user_id={user_id}, category={category}")
        return memory

    def update_embedding(self, memory_id: int, embedding: list[float]) -> None:
        """Set the embedding vector for a memory.

        Args:
            memory_id: The memory's ID
            embedding: The embedding vector (384 floats)
        """
        self.db.execute(
            text("UPDATE user_memories SET embedding = :emb WHERE id = :mid"),
            {"emb": str(embedding), "mid": memory_id},
        )
        self.db.commit()

    def embed_missing(self, limit: int = 100) -> int:
        """Embed active memories that have no vector yet, from ANY write path —
        passive extraction, agent contributions, or a ``remember`` embed that
        failed. Semantic recall filters ``embedding IS NOT NULL``, so an unembedded
        memory is invisible to recall; this is the sweep that guarantees everything
        Jarvis learns becomes recallable. Runs OFF the hot path (periodic task in a
        thread). One batched embedding call for all pending rows. Returns the count
        embedded."""
        pending = self.get_memories_without_embeddings(limit=limit)
        if not pending:
            return 0

        from app.core.llm_proxy_client import LLMProxyClient

        vectors = LLMProxyClient().create_embeddings_sync([m.content for m in pending])
        embedded = 0
        for mem, vec in zip(pending, vectors or []):
            if vec:
                self.update_embedding(mem.id, vec)
                embedded += 1
        return embedded

    def forget_memory(
        self,
        user_id: int,
        household_id: str,
        key: str | None = None,
        content_match: str | None = None,
    ) -> int:
        """Soft-delete memories matching key or content.

        Args:
            user_id: The user's ID
            household_id: The household scope
            key: Optional key to match exactly
            content_match: Optional content substring to match (case-insensitive)

        Returns:
            Number of memories deactivated
        """
        query = self.db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.household_id == household_id,
            UserMemory.is_active == True,  # noqa: E712
        )

        if key:
            query = query.filter(UserMemory.key == key)
        elif content_match:
            query = query.filter(UserMemory.content.ilike(f"%{content_match}%"))
        else:
            return 0  # Must provide at least one filter

        memories = query.all()
        count = 0
        for memory in memories:
            memory.is_active = False
            memory.updated_at = datetime.utcnow()
            count += 1

        if count > 0:
            self.db.commit()
            logger.info(f"Deactivated {count} memories for user_id={user_id}")

        return count

    def search_memories(
        self,
        user_id: int,
        household_id: str,
        query_embedding: list[float],
        limit: int = 5,
        category: str | None = None,
        similarity_threshold: float = 0.3,
    ) -> list[tuple[UserMemory, float]]:
        """Search memories by cosine similarity using pgvector.

        Args:
            user_id: The user's ID
            household_id: The household scope
            query_embedding: The query vector (384 floats)
            limit: Maximum number of results
            category: Optional category filter
            similarity_threshold: Minimum cosine similarity (0-1)

        Returns:
            List of (memory, similarity_score) tuples, highest similarity first
        """
        # pgvector cosine distance: 1 - cosine_similarity
        # So similarity = 1 - distance
        sql = """
            SELECT id, 1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM user_memories
            WHERE user_id = :uid
              AND household_id = :hid
              AND is_active = true
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> CAST(:query_vec AS vector)) >= :threshold
        """
        params: dict = {
            "query_vec": str(query_embedding),
            "uid": user_id,
            "hid": household_id,
            "threshold": similarity_threshold,
        }

        if category:
            sql += " AND category = :cat"
            params["cat"] = category

        sql += " ORDER BY similarity DESC LIMIT :lim"
        params["lim"] = limit

        rows = self.db.execute(text(sql), params).fetchall()

        if not rows:
            return []

        # Fetch full UserMemory objects for the matching IDs
        memory_ids = [row[0] for row in rows]
        similarity_map = {row[0]: row[1] for row in rows}

        memories = self.db.query(UserMemory).filter(
            UserMemory.id.in_(memory_ids)
        ).all()

        # Pair with scores and sort by similarity (descending)
        results = [(m, similarity_map[m.id]) for m in memories]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_memories_substring(
        self,
        user_id: int,
        household_id: str,
        query: str,
        limit: int = 5,
        category: str | None = None,
    ) -> list[tuple[UserMemory, float]]:
        """Fallback substring search when embeddings are unavailable.

        Uses ILIKE matching with simple word-overlap scoring.

        Args:
            user_id: The user's ID
            household_id: The household scope
            query: Search text
            limit: Maximum number of results
            category: Optional category filter

        Returns:
            List of (memory, score) tuples
        """
        db_query = self.db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.household_id == household_id,
            UserMemory.is_active == True,  # noqa: E712
        )

        if category:
            db_query = db_query.filter(UserMemory.category == category)

        # Exclude expired
        db_query = db_query.filter(
            (UserMemory.expires_at == None) | (UserMemory.expires_at > datetime.utcnow())  # noqa: E711
        )

        # ILIKE search for any word in the query
        words = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
        if not words:
            return []

        # Match any word
        from sqlalchemy import or_
        conditions = [UserMemory.content.ilike(f"%{w}%") for w in words]
        db_query = db_query.filter(or_(*conditions))

        memories = db_query.all()

        # Score by word overlap
        def score(memory: UserMemory) -> float:
            content_lower = memory.content.lower()
            matches = sum(1 for w in words if w in content_lower)
            return matches / len(words) if words else 0.0

        scored = [(m, score(m)) for m in memories]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Household-wide memory search (agent-injected context, user_id NULL)
    # ------------------------------------------------------------------

    def search_household_memories(
        self,
        household_id: str,
        query_embedding: list[float],
        limit: int = 5,
        similarity_threshold: float = 0.25,
    ) -> list[tuple[UserMemory, float]]:
        """Search household-wide memories by cosine similarity.

        Searches memories where user_id IS NULL (agent-injected context
        like news, calendar, weather that applies to all speakers).

        Args:
            household_id: The household scope
            query_embedding: The query vector (384 floats)
            limit: Maximum number of results
            similarity_threshold: Minimum cosine similarity (0-1)

        Returns:
            List of (memory, similarity_score) tuples, highest first
        """
        sql = """
            SELECT id, 1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM user_memories
            WHERE user_id IS NULL
              AND household_id = :hid
              AND is_active = true
              AND embedding IS NOT NULL
              AND (expires_at IS NULL OR expires_at > NOW())
              AND 1 - (embedding <=> CAST(:query_vec AS vector)) >= :threshold
            ORDER BY similarity DESC
            LIMIT :lim
        """
        params: dict = {
            "query_vec": str(query_embedding),
            "hid": household_id,
            "threshold": similarity_threshold,
            "lim": limit,
        }

        rows = self.db.execute(text(sql), params).fetchall()

        if not rows:
            return []

        memory_ids = [row[0] for row in rows]
        similarity_map = {row[0]: row[1] for row in rows}

        memories = self.db.query(UserMemory).filter(
            UserMemory.id.in_(memory_ids)
        ).all()

        results = [(m, similarity_map[m.id]) for m in memories]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_household_memories_substring(
        self,
        household_id: str,
        query: str,
        limit: int = 5,
    ) -> list[tuple[UserMemory, float]]:
        """Fallback substring search for household-wide memories.

        Uses ILIKE matching with word-overlap scoring against memories
        where user_id IS NULL (agent-injected context).

        Args:
            household_id: The household scope
            query: Search text
            limit: Maximum number of results

        Returns:
            List of (memory, score) tuples
        """
        from sqlalchemy import or_

        db_query = self.db.query(UserMemory).filter(
            UserMemory.user_id.is_(None),
            UserMemory.household_id == household_id,
            UserMemory.is_active == True,  # noqa: E712
        )

        # Exclude expired
        db_query = db_query.filter(
            (UserMemory.expires_at == None) | (UserMemory.expires_at > datetime.utcnow())  # noqa: E711
        )

        words = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
        if not words:
            return []

        conditions = [UserMemory.content.ilike(f"%{w}%") for w in words]
        db_query = db_query.filter(or_(*conditions))

        memories = db_query.all()

        def score(memory: UserMemory) -> float:
            content_lower = memory.content.lower()
            matches = sum(1 for w in words if w in content_lower)
            return matches / len(words) if words else 0.0

        scored = [(m, score(m)) for m in memories]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def check_content_similarity(
        self,
        household_id: str,
        query_embedding: list[float],
        threshold: float = 0.9,
        user_id: int | None = None,
    ) -> UserMemory | None:
        """Check if a very similar memory already exists (content dedup).

        Prevents near-duplicate memories from different keys (the same fact rephrased,
        or the same news story). Scope: pass ``user_id`` to dedup a USER'S own memories
        (the passive-extraction path); omit it (default) to dedup the household-wide
        ``user_id IS NULL`` pool (agent injections).

        Args:
            household_id: The household scope
            query_embedding: Embedding of the new content
            threshold: Minimum similarity to consider a duplicate (default 0.9)
            user_id: Dedup within this user's memories; None = household-wide pool

        Returns:
            The existing duplicate memory, or None if no match
        """
        user_clause = "user_id = :uid" if user_id is not None else "user_id IS NULL"
        sql = f"""
            SELECT id, 1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM user_memories
            WHERE {user_clause}
              AND household_id = :hid
              AND is_active = true
              AND embedding IS NOT NULL
              AND (expires_at IS NULL OR expires_at > NOW())
              AND 1 - (embedding <=> CAST(:query_vec AS vector)) >= :threshold
            ORDER BY similarity DESC
            LIMIT 1
        """
        params: dict = {
            "query_vec": str(query_embedding),
            "hid": household_id,
            "threshold": threshold,
        }
        if user_id is not None:
            params["uid"] = user_id

        row = self.db.execute(text(sql), params).fetchone()
        if not row:
            return None

        return self.db.query(UserMemory).filter(UserMemory.id == row[0]).first()

    def get_memories_without_embeddings(self, limit: int = 100) -> list[UserMemory]:
        """Get active memories that don't have embeddings yet (for backfill).

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of UserMemory objects without embeddings
        """
        return self.db.query(UserMemory).filter(
            UserMemory.is_active == True,  # noqa: E712
            UserMemory.embedding == None,  # noqa: E711
        ).limit(limit).all()

    def get_memories_for_prompt(
        self,
        user_id: int,
        household_id: str,
        max_chars: int = 500,
    ) -> str:
        """Format pinned memories for inclusion in the system prompt.

        Only includes pinned memories (identity facts). General memories
        are accessed via the recall tool for semantic search.

        Args:
            user_id: The user's ID
            household_id: The household scope
            max_chars: Maximum characters for the formatted output

        Returns:
            Formatted string of pinned memories, or empty string if none
        """
        memories = self.get_pinned_memories(user_id, household_id)

        # Fall back to all active memories if no pinned ones exist yet
        # (backwards compatibility during migration)
        if not memories:
            memories = self.get_active_memories(user_id, household_id)

        if not memories:
            return ""

        # Sort by category priority, then most recently updated first
        memories.sort(key=lambda m: (
            _CATEGORY_PRIORITY.get(m.category, 99),
            -(m.updated_at.timestamp() if m.updated_at else 0),
        ))

        lines: list[str] = []
        total_chars = 0
        for memory in memories:
            line = f"- {memory.content}"
            if total_chars + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 1  # +1 for newline

        return "\n".join(lines)

    def cleanup_expired(self) -> int:
        """Deactivate memories past their expiration time.

        Returns:
            Number of memories deactivated
        """
        now = datetime.utcnow()
        expired = self.db.query(UserMemory).filter(
            UserMemory.is_active == True,  # noqa: E712
            UserMemory.expires_at != None,  # noqa: E711
            UserMemory.expires_at <= now,
        ).all()

        count = 0
        for memory in expired:
            memory.is_active = False
            memory.updated_at = now
            count += 1

        if count > 0:
            self.db.commit()
            logger.info(f"Cleaned up {count} expired memories")

        return count
