"""CLI wrapper for Phase 3.5 command-example expansion.

Reads a commands spec JSON (same shape as `synth_commands_spec.json`),
feeds each command's canonical examples to an LLM expander, writes the
result as JSONL in llm-proxy's `dataset_ref` row shape — same output
format as Phase 3's `extract_training_data.py` so Phase 4 orchestrator
can concatenate organic + synthetic data transparently.

Backend selection:
  ANTHROPIC_API_KEY set      → use Claude (preferred, ~$0.01 per command)
  MODEL_SERVICE_TOKEN set    → use local llm-proxy as fallback
  --mock output-path         → read canned JSONL from a file (no LLM call)

Usage:
    # Claude (needs ANTHROPIC_API_KEY in env)
    python scripts/expand_command_examples.py \
        --commands-spec poc/synth_commands_spec.json \
        --out /tmp/synthetic.jsonl \
        --target-count 15

    # Local llm-proxy
    MODEL_SERVICE_TOKEN=... python scripts/expand_command_examples.py \
        --commands-spec spec.json --out synth.jsonl --backend local

    # Offline / dev — no LLM call
    python scripts/expand_command_examples.py \
        --commands-spec spec.json --out synth.jsonl \
        --mock fixtures/expander_output.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the variant builder from the Phase 3 extractor CLI
from scripts.extract_training_data import build_prompt_variants  # noqa: E402
from app.services.command_example_expander import (  # noqa: E402
    CommandExampleExpander,
    build_claude_expander,
    build_local_expander,
)
from app.services.training_data_extractor import to_dataset_ref_row  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("expand_command_examples")


def _mock_expander(path: Path):
    """Returns a canned string, same for every call. Useful for dry-runs."""
    content = path.read_text(encoding="utf-8")
    def _call(prompt: str, n: int) -> str:
        return content
    return _call


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3.5 command-example expander")
    ap.add_argument("--commands-spec", type=Path, required=True,
                    help="Path to commands spec JSON (same shape as synth_commands_spec.json)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path (dataset_ref row shape)")
    ap.add_argument("--target-count", type=int, default=15,
                    help="How many extra examples to ask for per command (default 15)")
    ap.add_argument("--backend", choices=["auto", "claude", "local", "mock"], default="auto",
                    help="Expander backend selection. 'auto' picks Claude if "
                         "ANTHROPIC_API_KEY is set, else local. Default: auto.")
    ap.add_argument("--mock", type=Path, default=None,
                    help="(with --backend mock) path to a file whose contents are returned as raw LLM text")
    ap.add_argument("--claude-model", type=str, default="claude-sonnet-4-6")
    ap.add_argument("--model-service-url", type=str, default="http://localhost:7705")
    ap.add_argument("--no-variant-randomize", action="store_true",
                    help="Stamp all rows with the 'full' prompt variant instead of randomizing.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.commands_spec.is_file():
        print(f"ERROR: commands spec not found at {args.commands_spec}", file=sys.stderr)
        return 2
    spec = json.loads(args.commands_spec.read_text(encoding="utf-8"))
    variants = build_prompt_variants(spec)
    variant_names = list(variants.keys())

    # Build the expander
    backend = args.backend
    if backend == "auto":
        backend = "claude" if os.getenv("ANTHROPIC_API_KEY") else "local"
    log.info("using backend: %s", backend)

    if backend == "claude":
        expander_fn = build_claude_expander(model=args.claude_model)
    elif backend == "local":
        expander_fn = build_local_expander(model_service_url=args.model_service_url)
    elif backend == "mock":
        if not args.mock or not args.mock.is_file():
            print("ERROR: --backend mock requires --mock <path>", file=sys.stderr)
            return 2
        expander_fn = _mock_expander(args.mock)
    else:
        print(f"ERROR: unknown backend {backend!r}", file=sys.stderr)
        return 2

    expander = CommandExampleExpander(expander_fn=expander_fn)

    # Run expansion across every command in the spec. The spec's commands
    # here follow our internal template format; we need to translate each
    # into the shape the expander expects (which mirrors IJarvisCommand.get_command_schema).
    schemas = [_spec_command_to_schema(c) for c in spec["commands"]]
    log.info("expanding %d commands, target %d each (~$%.2f if Claude)",
             len(schemas), args.target_count,
             0.01 * len(schemas) if backend == "claude" else 0.0)

    all_rows = []
    all_stats = []
    for schema in schemas:
        rows, stats = expander.expand_command(schema, target_count=args.target_count)
        all_rows.extend(rows)
        all_stats.append(stats)
        log.info("  %s: asked %d → parsed %d → kept %d  (rejected: %s)",
                 stats.command, stats.asked_for, stats.got_raw,
                 stats.after_validation, stats.rejected_reasons)

    # Write dataset_ref rows, variant-randomized unless disabled
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    variant_counts: dict[str, int] = {n: 0 for n in variant_names}
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            variant = "full" if args.no_variant_randomize else rng.choice(variant_names)
            variant_counts[variant] += 1
            ds_row = to_dataset_ref_row(row, system_prompt=variants[variant])
            f.write(json.dumps(ds_row, ensure_ascii=False) + "\n")

    print("\nSummary:")
    print(f"  total rows:            {len(all_rows)}")
    print(f"  commands expanded:     {len(schemas)}")
    print(f"  variant distribution:  {variant_counts}")
    print(f"  wrote:                 {args.out}")
    return 0


def _spec_command_to_schema(cmd: dict) -> dict:
    """Translate a synth_commands_spec.json command block into an
    IJarvisCommand-style schema dict (what the expander consumes).

    The spec format has `templates` + `slots` for programmatic expansion;
    the expander wants a flat `examples` list. We convert the FIRST distinct
    template-shape per command into a canonical example with default slot
    values — same logic used by Phase 2.5's `_format_tool_block`.
    """
    examples: list[dict] = []
    seen_shapes: set[frozenset] = set()
    slots = cmd.get("slots", {})
    for tpl in cmd.get("templates", []):
        shape = frozenset(tpl["args"].keys())
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        # Substitute default (first) slot values into pattern + args
        pattern = tpl["pattern"]
        resolved_args: dict = {}
        rendered = pattern
        for k, v in tpl["args"].items():
            if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                slot = v[1:-1]
                default = slots.get(slot, ["x"])[0] if slots.get(slot) else "x"
                resolved_args[k] = default
            else:
                resolved_args[k] = v
        # Substitute slot names inside the pattern using first value
        import re
        rendered = re.sub(
            r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}",
            lambda m: str(slots.get(m.group(1), ["x"])[0]) if slots.get(m.group(1)) else m.group(0),
            pattern,
        )
        examples.append({
            "voice_command": rendered,
            "expected_parameters": resolved_args,
            "is_primary": not examples,  # first shape marked primary
        })
    return {
        "command_name": cmd["name"],
        "description": cmd.get("description", ""),
        "parameters": cmd.get("parameters", []),
        "examples": examples,
    }


if __name__ == "__main__":
    raise SystemExit(main())
