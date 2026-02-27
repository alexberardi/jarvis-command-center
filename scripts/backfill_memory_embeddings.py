#!/usr/bin/env python3
"""Backfill embeddings for existing user memories.

Finds active memories without embeddings and generates them
via the LLM proxy /v1/embeddings endpoint.

Usage:
    python scripts/backfill_memory_embeddings.py
    python scripts/backfill_memory_embeddings.py --dry-run
    python scripts/backfill_memory_embeddings.py --batch-size 50
"""

import argparse
import asyncio
import os
import sys

# Add parent dir to path for app imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


async def backfill(batch_size: int, dry_run: bool) -> None:
    from app.db import get_session_local
    from app.core.llm_proxy_client import LLMProxyClient
    from app.services.memory_service import MemoryService

    SessionLocal = get_session_local()
    db = SessionLocal()

    try:
        svc = MemoryService(db)
        memories = svc.get_memories_without_embeddings(limit=batch_size)

        if not memories:
            print("No memories need embedding backfill.")
            return

        print(f"Found {len(memories)} memories without embeddings.")

        if dry_run:
            for m in memories:
                print(f"  [DRY RUN] Would embed memory {m.id}: {m.content[:80]}...")
            return

        client = LLMProxyClient()

        # Batch embed all texts at once
        texts = [m.content for m in memories]
        print(f"Generating embeddings for {len(texts)} memories...")

        vectors = await client.create_embeddings(texts)

        if len(vectors) != len(memories):
            print(f"WARNING: Got {len(vectors)} vectors for {len(memories)} memories")

        updated = 0
        for memory, vector in zip(memories, vectors):
            try:
                svc.update_embedding(memory.id, vector)
                updated += 1
                print(f"  ✓ Memory {memory.id}: {memory.content[:60]}...")
            except Exception as e:
                print(f"  ✗ Memory {memory.id} failed: {e}")

        print(f"\nDone. Updated {updated}/{len(memories)} memories with embeddings.")

    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill memory embeddings")
    parser.add_argument("--batch-size", type=int, default=100, help="Max memories to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    asyncio.run(backfill(batch_size=args.batch_size, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
