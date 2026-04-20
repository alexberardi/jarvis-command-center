"""Phase 4 CLI — one-command end-to-end adapter training.

Composes:
  • Phase 3: organic transcript extraction
  • Phase 3.5: synthetic command-example expansion
  • llm-proxy adapter training (via queue API)
  • Phase 2 eval gate (via docker exec into jarvis-node)

Prints a PASS/FAIL verdict + adapter hash. No auto-deploy (that's Phase 5).

Usage:
    # Organic only (requires rated/clean traces in conversation_transcripts)
    python scripts/train_adapter.py \
        --organic --since 14d --household-id h1 \
        --commands-spec /path/to/spec.json \
        --base-model-id .models/Hermes-3-Llama-3.1-8B-4bit-mlx \
        --eval-via-docker

    # Synthetic cold-start (works even with zero real traces)
    ANTHROPIC_API_KEY=... python scripts/train_adapter.py \
        --synthetic --commands-spec /path/to/spec.json \
        --expand-count 15 \
        --base-model-id .models/Hermes-3-Llama-3.1-8B-4bit-mlx \
        --eval-via-docker

    # Both (recommended — best of both)
    python scripts/train_adapter.py --organic --since 14d --synthetic \
        --commands-spec /path/to/spec.json \
        --base-model-id ... --eval-via-docker

Output:
  stdout: verdict line + JSON trailer
  exit code: 0 on success, 1 on training/eval failure, 2 on config error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_session_local  # noqa: E402
from app.services.command_example_expander import (  # noqa: E402
    CommandExampleExpander,
    default_expander,
)
from app.services.training_data_extractor import (  # noqa: E402
    TrainingDataExtractor,
    to_dataset_ref_row,
)
from app.services.training_orchestrator import (  # noqa: E402
    invoke_eval_via_docker,
    orchestrate_training,
)

# Reuse the prompt-variant builder + spec-to-schema helpers from the extract CLI
from scripts.extract_training_data import (  # noqa: E402
    build_prompt_variants,
    parse_since,
)
from scripts.expand_command_examples import _spec_command_to_schema  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train_adapter")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 end-to-end adapter trainer")

    # Source selection (at least one)
    ap.add_argument("--organic", action="store_true",
                    help="Include data from conversation_transcripts (Phase 3)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Include LLM-expanded command examples (Phase 3.5)")

    # Phase 3 inputs
    ap.add_argument("--since", type=str, default="14d")
    ap.add_argument("--until", type=str, default=None)
    ap.add_argument("--household-id", type=str, default=None)
    ap.add_argument("--min-per-user", type=int, default=10)
    ap.add_argument("--max-per-user-multiplier", type=float, default=4.0)

    # Phase 3.5 inputs
    ap.add_argument("--expand-count", type=int, default=15,
                    help="Synthetic examples per command (default 15)")
    ap.add_argument("--synth-backend", choices=["auto", "claude", "local", "mock"],
                    default="auto")
    ap.add_argument("--synth-mock", type=Path, default=None,
                    help="(with --synth-backend mock) path to canned LLM output")

    # Shared
    ap.add_argument("--commands-spec", type=Path, required=True,
                    help="Commands spec JSON — drives prompt variants + synth schemas")
    ap.add_argument("--no-variant-randomize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # llm-proxy + training
    ap.add_argument("--llm-proxy-url", type=str, default="http://localhost:7704")
    ap.add_argument("--base-model-id", type=str, required=True)
    ap.add_argument("--node-id", type=str, default="phase4-orchestrator")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--learning-rate", type=float, default=2e-4)

    # Eval
    ap.add_argument("--eval-via-docker", action="store_true",
                    help="After training, invoke eval_gate.py inside jarvis-node container")
    ap.add_argument("--eval-container", type=str, default="jarvis-node")
    ap.add_argument("--eval-threshold", type=float, default=None)
    ap.add_argument("--eval-tolerance", type=float, default=1.0)

    # Housekeeping
    ap.add_argument("--out-dataset", type=Path, default=None,
                    help="(optional) Also write the merged dataset JSONL for audit")
    args = ap.parse_args()

    if not (args.organic or args.synthetic):
        print("ERROR: must pass --organic and/or --synthetic", file=sys.stderr)
        return 2

    if not args.commands_spec.is_file():
        print(f"ERROR: commands spec not found at {args.commands_spec}", file=sys.stderr)
        return 2
    spec = json.loads(args.commands_spec.read_text(encoding="utf-8"))
    variants = build_prompt_variants(spec)
    variant_names = list(variants.keys())

    app_id = os.getenv("JARVIS_APP_ID") or "jarvis-command-center"
    app_key = os.getenv("JARVIS_APP_KEY")
    if not app_key:
        print("ERROR: JARVIS_APP_KEY env var required (app-to-app auth for llm-proxy)",
              file=sys.stderr)
        return 2

    # --- wire sources ------------------------------------------------

    def organic_source():
        if not args.organic:
            return []
        since_dt = parse_since(args.since) if args.since else None
        until_dt = parse_since(args.until) if args.until else None
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            extractor = TrainingDataExtractor(db)
            return extractor.extract(
                since=since_dt, until=until_dt,
                household_id=args.household_id,
                min_per_user=args.min_per_user,
                max_per_user_multiplier=args.max_per_user_multiplier,
                rng_seed=args.seed,
            )

    def synthetic_source():
        if not args.synthetic:
            return []
        backend = args.synth_backend
        if backend == "auto":
            backend = "claude" if os.getenv("ANTHROPIC_API_KEY") else "local"
        if backend == "claude":
            from app.services.command_example_expander import build_claude_expander
            expander_fn = build_claude_expander()
        elif backend == "local":
            from app.services.command_example_expander import build_local_expander
            expander_fn = build_local_expander()
        elif backend == "mock":
            if not args.synth_mock or not args.synth_mock.is_file():
                raise RuntimeError("--synth-backend mock requires --synth-mock <path>")
            content = args.synth_mock.read_text(encoding="utf-8")
            def expander_fn(prompt: str, n: int) -> str:  # noqa: E306
                return content
        else:
            raise RuntimeError(f"unknown synth backend: {backend}")
        log.info("synth backend: %s", backend)
        expander = CommandExampleExpander(expander_fn=expander_fn)
        schemas = [_spec_command_to_schema(c) for c in spec["commands"]]
        rows, stats = expander.expand_many(schemas, target_count_per_command=args.expand_count)
        for s in stats:
            log.info("  %s: asked %d → kept %d  (rejected: %s)",
                     s.command, s.asked_for, s.after_validation, s.rejected_reasons)
        return rows

    # --- formatter (variant-randomized by default) -------------------

    rng = random.Random(args.seed)
    def formatter(row):
        variant = "full" if args.no_variant_randomize else rng.choice(variant_names)
        return to_dataset_ref_row(row, system_prompt=variants[variant])

    # --- eval runner (optional) --------------------------------------

    eval_runner = None
    if args.eval_via_docker:
        def eval_runner(adapter_hash: str):
            return invoke_eval_via_docker(
                adapter_hash=adapter_hash,
                node_container=args.eval_container,
                threshold=args.eval_threshold,
                tolerance=args.eval_tolerance,
            )

    # --- orchestrate -------------------------------------------------

    training_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
        "learning_rate": args.learning_rate,
    }

    result = orchestrate_training(
        organic_source=organic_source,
        synthetic_source=synthetic_source,
        formatter=formatter,
        llm_proxy_url=args.llm_proxy_url,
        base_model_id=args.base_model_id,
        node_id=args.node_id,
        app_id=app_id, app_key=app_key,
        training_params=training_params,
        eval_runner=eval_runner,
    )

    # --- report ------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"Phase 4 orchestrator — {result.status.upper()}")
    print("=" * 60)
    print(f"dataset rows:       {result.dataset_rows}")
    print(f"by source:          {result.by_source}")
    print(f"adapter hash:       {result.adapter_hash or '—'}")
    print(f"training time:      {result.training_seconds:.1f}s" if result.training_seconds else "training time:      —")
    if result.timings:
        print(f"timings:            {result.timings}")
    if result.eval:
        trailer = result.eval.get("trailer") or {}
        print("eval:")
        print(f"  exit_code:        {result.eval.get('exit_code')}")
        if trailer:
            print(f"  verdict:          {trailer.get('verdict')}")
            print(f"  pass_rate:        {trailer.get('pass_rate')}%")
            print(f"  threshold:        {trailer.get('threshold')}%")
            print(f"  delta_pp:         {trailer.get('delta_pp')}")
    if result.error:
        print(f"error:              {result.error}")
    print()
    # JSON trailer for orchestrator chains
    print(json.dumps(asdict(result)))

    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
