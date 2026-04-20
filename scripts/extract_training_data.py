"""CLI wrapper for Phase 3 training-data extraction.

Queries `conversation_transcripts` via TrainingDataExtractor, applies all
filter/dedup/balance rules, stamps each row with a variant-randomized
system prompt (per Phase 2.5 lesson), and writes the result as JSONL in
llm-proxy's `dataset_ref` row shape.

Usage:
    # Last 14 days, all households, at least 10 examples per user
    python scripts/extract_training_data.py --since 14d \
        --commands-spec /path/to/spec.json \
        --out /tmp/training_data.jsonl

    # Scoped to one household, fail if fewer than N usable rows
    python scripts/extract_training_data.py --household-id h1 \
        --since 2026-04-01T00:00:00Z \
        --commands-spec spec.json \
        --out data.jsonl --min-rows 200

Relative --since shorthand: 7d, 48h, 30m.

The commands spec is the same JSON shape as
jarvis-llm-proxy-api/poc/synth_commands_spec.json — used here to generate
the 5 prompt variants (full / no_examples / no_descriptions /
no_critical_rules / schema_only) that get randomly assigned per row.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Add repo root to path so "from app..." imports work when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_session_local  # noqa: E402
from app.services.prompt_variant_builder import (  # noqa: E402
    build_prompt_variants,
    parse_since,
)
from app.services.training_data_extractor import (  # noqa: E402
    TrainingDataExtractor,
    to_dataset_ref_row,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("extract_training_data")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract training data from conversation_transcripts")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO-8601 or relative shorthand (7d, 48h, 30m)")
    ap.add_argument("--until", type=str, default=None,
                    help="ISO-8601 or relative shorthand (exclusive upper bound)")
    ap.add_argument("--household-id", type=str, default=None,
                    help="Scope to a single household")
    ap.add_argument("--commands-spec", type=Path, required=True,
                    help="Path to commands spec JSON (for building prompt variants)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path (dataset_ref row shape)")
    ap.add_argument("--min-per-user", type=int, default=10,
                    help="Exclude users with fewer than N examples (default 10)")
    ap.add_argument("--max-per-user-multiplier", type=float, default=4.0,
                    help="Cap top speaker at MULT × median count (default 4.0)")
    ap.add_argument("--min-rows", type=int, default=0,
                    help="Fail (exit 1) if fewer than N rows emitted. Default 0 = no check.")
    ap.add_argument("--no-variant-randomize", action="store_true",
                    help="Use only the 'full' system prompt (V1 behavior). "
                         "Default is to randomize across 5 variants per Phase 2.5 lesson.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.commands_spec.is_file():
        print(f"ERROR: commands spec not found at {args.commands_spec}", file=sys.stderr)
        return 2
    spec = json.loads(args.commands_spec.read_text(encoding="utf-8"))
    variants = build_prompt_variants(spec)

    try:
        since_dt = parse_since(args.since) if args.since else None
        until_dt = parse_since(args.until) if args.until else None
    except ValueError as e:
        print(f"ERROR: bad time value: {e}", file=sys.stderr)
        return 2

    SessionLocal = get_session_local()
    rng = random.Random(args.seed)

    with SessionLocal() as db:
        extractor = TrainingDataExtractor(db)
        rows = extractor.extract(
            since=since_dt,
            until=until_dt,
            household_id=args.household_id,
            min_per_user=args.min_per_user,
            max_per_user_multiplier=args.max_per_user_multiplier,
            rng_seed=args.seed,
        )

    log.info("extracted %d rows for window [%s, %s) household=%s",
             len(rows), since_dt, until_dt, args.household_id)

    if args.min_rows and len(rows) < args.min_rows:
        print(
            f"ERROR: only {len(rows)} rows extracted; --min-rows was {args.min_rows}",
            file=sys.stderr,
        )
        return 1

    # Stamp each row with a system prompt (variant-randomized unless --no-variant-randomize)
    variant_names = list(variants.keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    variant_counts: dict[str, int] = {n: 0 for n in variant_names}
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            if args.no_variant_randomize:
                variant = "full"
            else:
                variant = rng.choice(variant_names)
            variant_counts[variant] += 1
            dataset_row = to_dataset_ref_row(row, system_prompt=variants[variant])
            f.write(json.dumps(dataset_row, ensure_ascii=False) + "\n")

    log.info("wrote %d rows → %s", len(rows), args.out)
    if not args.no_variant_randomize:
        log.info("variant distribution: %s", variant_counts)

    # Summarize for the operator
    by_source: dict[str, int] = {}
    by_user: dict[int, int] = {}
    for r in rows:
        by_source[r.source] = by_source.get(r.source, 0) + 1
        by_user[r.user_id] = by_user.get(r.user_id, 0) + 1
    print("\nSummary:")
    print(f"  total rows:       {len(rows)}")
    print(f"  by source:        {by_source}")
    print(f"  users included:   {len(by_user)}")
    if by_user:
        print(f"  per-user counts:  {dict(sorted(by_user.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
