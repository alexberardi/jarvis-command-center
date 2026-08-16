"""Full-flow tests for the memory feature: extraction → save → embed → recall,
plus identity scoping.

These validate the pipeline Part 1 fixed end-to-end. The critical bug was that
passively-learned memories were saved with a NULL embedding and semantic recall
filters ``embedding IS NOT NULL`` — so everything Jarvis passively learned was
invisible to recall until a manual backfill. The periodic sweep
(``MemoryService.embed_missing``) now closes that gap on the hot-path-free
background task.

Two layers:
  * SQLite classes — deterministic, no infra, always run. Cover the save→embed
    flow, extraction parsing, and cross-user scoping.
  * ``TestSemanticRecallRealPgvector`` — needs a real pgvector Postgres (set
    ``TEST_PGVECTOR_URL``); skips cleanly otherwise. This is the ONLY place the
    real ``embedding <=> vector`` path is exercised — SQLite can't, so the older
    integration suite forces the substring fallback and never proves that an
    embedded memory is actually semantically recallable (or that an unembedded
    one is invisible).
"""
import os
import uuid

import pytest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.services.memory_service import MemoryService
from app.services.memory_extraction_service import _parse_extraction_response

HH = "flow-hh"
ALICE = 900001
BOB = 900002


# --------------------------------------------------------------------------- #
# SQLite fixtures (fast, deterministic)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def service(db):
    return MemoryService(db)


def _embedder(dim: int = 384):
    """A mock LLMProxyClient that returns one deterministic vector per input."""
    fake = MagicMock()
    fake.create_embeddings_sync.side_effect = lambda texts: [[0.1] * dim for _ in texts]
    return fake


# --------------------------------------------------------------------------- #
# The passive-learning flow (the Part 1 regression)
# --------------------------------------------------------------------------- #
class TestPassiveLearningFlow:
    def test_passive_write_starts_unembedded(self, service):
        """The bug precondition: the extractor saves with NO embedding."""
        m = service.save_memory(ALICE, HH, "is allergic to peanuts", source="passive")
        assert m.embedding is None
        assert [x.id for x in service.get_memories_without_embeddings()] == [m.id]

    def test_sweep_embeds_passive_memory_and_makes_it_findable(self, service):
        service.save_memory(ALICE, HH, "is allergic to peanuts", category="fact", source="passive")
        service.save_memory(ALICE, HH, "takes coffee black", category="preference", source="passive")
        assert len(service.get_memories_without_embeddings()) == 2  # both invisible to recall

        with patch("app.core.llm_proxy_client.LLMProxyClient", return_value=_embedder()):
            embedded = service.embed_missing()

        assert embedded == 2
        assert service.get_memories_without_embeddings() == []  # now recallable
        # and the content is retrievable (substring path works in SQLite)
        hits = service.search_memories_substring(ALICE, HH, "peanuts allergy")
        assert any("peanuts" in m.content for m, _ in hits)

    def test_sweep_is_the_safety_net_for_a_failed_remember_embed(self, service):
        """The remember tool embeds inline; if that call FAILS the row is still
        saved with a NULL vector. The sweep must catch it regardless of source."""
        m = service.save_memory(ALICE, HH, "has a dog named Rex", source="voice")
        assert m.embedding is None  # inline embed didn't happen / failed

        with patch("app.core.llm_proxy_client.LLMProxyClient", return_value=_embedder()):
            assert service.embed_missing() == 1
        assert service.get_memories_without_embeddings() == []


# --------------------------------------------------------------------------- #
# Extraction quality — the parser turning LLM output into memories
# --------------------------------------------------------------------------- #
class TestExtractionParsing:
    def test_parses_a_raw_json_array(self):
        out = _parse_extraction_response(
            '[{"category":"fact","key":"brother","content":"Has a brother named Mike"}]'
        )
        assert out == [{"category": "fact", "key": "brother", "content": "Has a brother named Mike"}]

    def test_parses_json_wrapped_in_markdown_fences(self):
        out = _parse_extraction_response('```json\n[{"content":"Enjoys grilling"}]\n```')
        assert out == [{"content": "Enjoys grilling"}]

    def test_strips_qwen_think_block_before_parsing(self):
        out = _parse_extraction_response(
            '<think>the user mentioned a pet</think>\n[{"content":"Has a cat named Luna"}]'
        )
        assert out == [{"content": "Has a cat named Luna"}]

    def test_finds_the_array_amid_prose(self):
        out = _parse_extraction_response('Sure! Here you go: [{"content":"Likes hiking"}] — hope that helps')
        assert out == [{"content": "Likes hiking"}]

    def test_malformed_output_yields_no_memories(self):
        assert _parse_extraction_response("I couldn't find anything useful.") == []
        assert _parse_extraction_response("[not, valid, json") == []

    def test_items_without_content_are_dropped(self):
        out = _parse_extraction_response('[{"category":"fact"},{"content":"Real fact"},{"content":""}]')
        assert out == [{"content": "Real fact"}]


# --------------------------------------------------------------------------- #
# Identity scoping — memory must belong to the speaker, not bleed across a household
# --------------------------------------------------------------------------- #
class TestCrossUserIsolation:
    def test_reads_are_scoped_to_the_speaker(self, service):
        # Same household, same key — the ONLY thing separating them is user_id.
        service.save_memory(ALICE, HH, "allergic to peanuts", key="allergy", source="passive")
        service.save_memory(BOB, HH, "allergic to shellfish", key="allergy", source="passive")

        alice_view = service.get_memories_for_prompt(ALICE, HH, max_chars=500)
        bob_view = service.get_memories_for_prompt(BOB, HH, max_chars=500)
        assert "peanuts" in alice_view and "shellfish" not in alice_view
        assert "shellfish" in bob_view and "peanuts" not in bob_view

    def test_substring_recall_never_crosses_users(self, service):
        service.save_memory(ALICE, HH, "drives a red truck", source="passive")
        # Bob searching for Alice's fact gets nothing back.
        assert service.search_memories_substring(BOB, HH, "red truck") == []

    def test_pinned_reads_are_scoped(self, service):
        service.save_memory(ALICE, HH, "name is Jordan", key="name", source="voice", is_pinned=True)
        service.save_memory(BOB, HH, "name is Riley", key="name", source="voice", is_pinned=True)
        alice_pinned = [m.content for m in service.get_pinned_memories(ALICE, HH)]
        assert alice_pinned == ["name is Jordan"]

    def test_sweep_embeds_every_users_rows(self, service):
        service.save_memory(ALICE, HH, "likes tea", source="passive")
        service.save_memory(BOB, HH, "likes coffee", source="passive")
        with patch("app.core.llm_proxy_client.LLMProxyClient", return_value=_embedder()):
            assert service.embed_missing() == 2
        assert service.get_memories_without_embeddings() == []


# --------------------------------------------------------------------------- #
# Real semantic recall — the pgvector `<=>` path SQLite cannot exercise.
# Set TEST_PGVECTOR_URL to a pgvector Postgres (e.g. the jarvis_command_center_test DB).
# --------------------------------------------------------------------------- #
def _close_vec(dim: int = 384):
    return [1.0] + [0.0] * (dim - 1)


def _far_vec(dim: int = 384):
    return [0.0] * (dim - 1) + [1.0]


@pytest.fixture(scope="module")
def pg_engine():
    """Connect once per module; skip cleanly if there's no reachable pgvector DB."""
    url = os.environ.get("TEST_PGVECTOR_URL") or os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("no TEST_PGVECTOR_URL / TEST_DATABASE_URL — skipping real-pgvector recall tests")
    try:
        engine = create_engine(url)
        with engine.connect() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            c.commit()
        Base.metadata.create_all(bind=engine)  # idempotent
    except Exception as e:  # noqa: BLE001 — DB not reachable / not pgvector
        pytest.skip(f"pgvector test DB unavailable: {type(e).__name__}: {str(e)[:120]}")
    return engine


@pytest.fixture()
def pg_service(pg_engine):
    """Fresh session + a UNIQUE household per test, so tests don't see each
    other's rows. Cleans up its own rows on teardown."""
    session = sessionmaker(bind=pg_engine)()
    hh = f"flow-pg-{uuid.uuid4().hex[:8]}"
    try:
        yield MemoryService(session), hh
    finally:
        session.execute(text("DELETE FROM user_memories WHERE household_id = :h"), {"h": hh})
        session.commit()
        session.close()


class TestSemanticRecallRealPgvector:
    def test_unembedded_is_invisible_then_embedded_is_recallable(self, pg_service):
        """The exact bug + fix: an unembedded memory is invisible to semantic
        recall (embedding IS NOT NULL filter); once the sweep embeds it, the same
        query finds it."""
        svc, hh = pg_service
        m = svc.save_memory(ALICE, hh, "is allergic to peanuts", category="fact", source="passive")

        # BEFORE embedding: semantic recall cannot see it — this is why passive
        # memories felt forgotten.
        assert svc.search_memories(ALICE, hh, _close_vec(), similarity_threshold=0.1) == []

        # The sweep embeds it (here we set the vector directly to keep the test
        # independent of the embedding model).
        svc.update_embedding(m.id, _close_vec())

        hits = svc.search_memories(ALICE, hh, _close_vec(), similarity_threshold=0.1)
        assert [mm.id for mm, _ in hits] == [m.id]

    def test_semantic_recall_ranks_relevant_over_irrelevant(self, pg_service):
        svc, hh = pg_service
        near = svc.save_memory(ALICE, hh, "loves black coffee", category="preference", source="passive")
        far = svc.save_memory(ALICE, hh, "parked in lot B", category="note", source="passive")
        svc.update_embedding(near.id, _close_vec())
        svc.update_embedding(far.id, _far_vec())

        hits = svc.search_memories(ALICE, hh, _close_vec(), similarity_threshold=0.0, limit=5)
        ids = [mm.id for mm, _ in hits]
        assert ids and ids[0] == near.id  # the close memory ranks first

    def test_semantic_recall_is_user_scoped(self, pg_service):
        """Two members with an identical embedded fact — a query as ALICE must
        never surface BOB's row."""
        svc, hh = pg_service
        a = svc.save_memory(ALICE, hh, "has a standing 9am meeting", source="passive")
        b = svc.save_memory(BOB, hh, "has a standing 9am meeting", source="passive")
        svc.update_embedding(a.id, _close_vec())
        svc.update_embedding(b.id, _close_vec())

        hits = svc.search_memories(ALICE, hh, _close_vec(), similarity_threshold=0.1)
        ids = [mm.id for mm, _ in hits]
        assert a.id in ids and b.id not in ids

    def test_dedup_guard_is_user_scoped(self, pg_service):
        """check_content_similarity(user_id=...) must match within a user's OWN
        memories (the extraction dedup path), not across users or the household pool."""
        svc, hh = pg_service
        m = svc.save_memory(ALICE, hh, "likes coffee black", source="passive")
        svc.update_embedding(m.id, _close_vec())
        # Same user, near-identical vector → duplicate found.
        assert svc.check_content_similarity(hh, _close_vec(), threshold=0.5, user_id=ALICE) is not None
        # Different user → not a duplicate for them (scoped).
        assert svc.check_content_similarity(hh, _close_vec(), threshold=0.5, user_id=BOB) is None
        # Household pool (user_id IS NULL) → does not match a USER memory.
        assert svc.check_content_similarity(hh, _close_vec(), threshold=0.5) is None


class TestExtractionPromptGuardrails:
    """The relationship guardrails exist because the extractor once stored
    'Brother Leo takes Keppra' from a conversation about the family dog —
    and the live model then told the user's wife that Leo was her brother
    (prod 2026-08-15). Keep the rules pinned in the prompt."""

    def test_prompt_forbids_inferred_relationships(self):
        from app.services.memory_extraction_service import _EXTRACTION_SYSTEM_PROMPT
        assert "NEVER infer" in _EXTRACTION_SYSTEM_PROMPT
        assert "pet" in _EXTRACTION_SYSTEM_PROMPT.lower()
        # The literal failure case stays documented as a counter-example.
        assert "Counter-example" in _EXTRACTION_SYSTEM_PROMPT

    def test_prompt_pins_user_turns_as_sole_source(self):
        from app.services.memory_extraction_service import _EXTRACTION_SYSTEM_PROMPT
        assert "ONLY from what the" in _EXTRACTION_SYSTEM_PROMPT
        assert "NEVER store a claim" in _EXTRACTION_SYSTEM_PROMPT
