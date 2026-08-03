"""Extraction-validity eval — the "what's saved / for how long / whether valid" test.

Runs each scenario's transcript through the REAL extraction model (the same
`model="background"` the passive extractor uses), parses it with the REAL
`_parse_extraction_response`, then has an independent GPT judge grade: were the
expected facts captured, was anything saved that shouldn't be (ephemeral commands,
junk, unsafe), and are the ttl assignments sane?

    python tools/memory_extraction_eval.py            # all scenarios
    python tools/memory_extraction_eval.py --limit 3  # quick smoke
"""
import argparse
import asyncio
import json
import os
import sys

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                     # _memory_eval_common
sys.path.insert(0, os.path.dirname(_HERE))    # CC root, for app.*

import _memory_eval_common as C  # noqa: E402
from app.services.memory_extraction_service import (  # noqa: E402
    _EXTRACTION_SYSTEM_PROMPT,
    _parse_extraction_response,
)

EXTRACT_MODEL = os.environ.get("SIM_EXTRACT_MODEL", "background")
EXTRACT_TEMP = float(os.environ.get("SIM_EXTRACT_TEMP", "0.0"))  # greedy, matches prod; stable numbers

_JUDGE_SYS = (
    "You grade a passive memory-extraction system for a PRIVATE voice assistant. It should "
    "remember DURABLE personal facts/preferences/habits and IGNORE ephemeral requests. "
    "Match by MEANING, not exact wording: an expected fact is CAPTURED if any extracted item "
    "conveys the same information, even if paraphrased or split across items. Each extracted "
    "item is EITHER a valid capture OR a bad_save — never both. A durable, harmless extra fact "
    "is NOT a bad_save. Return ONLY JSON."
)


def _build_messages(transcript: str) -> list:
    # Mirrors memory_extraction_service.py:120-129 (system + user message).
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Existing memories for this user:\nNone stored yet.\n\n"
            f"Recent conversations:\n{transcript}"
        )},
    ]


def _judge_prompt(sc: dict, extracted: list) -> str:
    return (
        f"TRANSCRIPT:\n{sc['transcript']}\n\n"
        f"Facts that SHOULD be saved (with intended permanence): "
        f"{json.dumps(sc['expected_saves'])}\n"
        f"Things that should be SKIPPED (never stored): {json.dumps(sc.get('expected_skips', []))}\n\n"
        f"What the system ACTUALLY extracted: {json.dumps(extracted)}\n\n"
        "ttl_days convention: absent=permanent, ~30=recurring habit, ~7=time-bound note.\n"
        "Grade by MEANING (paraphrase = still a capture). Return JSON:\n"
        "{\n"
        '  "captured": [expected facts covered by the extraction],\n'
        '  "missed": [expected facts NOTHING in the extraction conveys],\n'
        '  "bad_saves": [extracted items that should NOT be stored — an ephemeral request/command, junk, or unsafe/PII. A durable harmless extra fact is NOT a bad_save],\n'
        '  "extra_saves": [durable-but-not-expected facts it stored — informational, not a failure],\n'
        '  "ttl_reasonable": true or false (permanent facts have no/large ttl; genuinely time-bound notes have short ttl; minor differences are fine),\n'
        '  "verdict": "pass" or "fail"  (pass = missed is empty or only minor, AND no bad_saves, AND ttl broadly reasonable),\n'
        '  "notes": "one short sentence"\n'
        "}"
    )


async def _run_one(client: httpx.AsyncClient, sc: dict) -> dict:
    # Generous max_tokens so a multi-fact JSON array isn't truncated mid-object —
    # we're testing extraction QUALITY, not the server's default token cap. (NOTE:
    # prod's enqueue sets no max_tokens at all — a truncation risk worth flagging.)
    raw = await C.proxy_chat(client, _build_messages(sc["transcript"]),
                             model=EXTRACT_MODEL, temperature=EXTRACT_TEMP, max_tokens=3000)
    extracted = _parse_extraction_response(raw)
    verdict = await C.openai_judge(client, _JUDGE_SYS, _judge_prompt(sc, extracted))
    return {"name": sc["name"], "extracted": extracted, "verdict": verdict}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=os.path.join(_HERE, "extraction_eval_out.json"))
    args = ap.parse_args()

    C.require_keys()
    scenarios = C.load_scenarios()["extraction_scenarios"]
    if args.limit:
        scenarios = scenarios[:args.limit]

    print(f"Extraction eval: {len(scenarios)} scenarios | model={EXTRACT_MODEL} | "
          f"proxy={C.PROXY_URL} | judge={C.JUDGE_MODEL}")
    sem = asyncio.Semaphore(3)

    async with httpx.AsyncClient() as client:
        async def guarded(sc):
            async with sem:
                try:
                    return await _run_one(client, sc)
                except Exception as e:  # noqa: BLE001
                    return {"name": sc["name"], "error": f"{type(e).__name__}: {str(e)[:200]}"}
        results = await asyncio.gather(*[guarded(sc) for sc in scenarios])

    passed = sum(1 for r in results if r.get("verdict", {}).get("verdict") == "pass")
    errored = sum(1 for r in results if "error" in r)
    print(f"\n=== EXTRACTION: {passed}/{len(results)} pass"
          + (f", {errored} errored" if errored else "") + " ===")
    for r in results:
        if "error" in r:
            print(f"  ⚠️  {r['name']}: {r['error']}")
            continue
        v = r["verdict"]
        mark = "✅" if v.get("verdict") == "pass" else "❌"
        print(f"  {mark} {r['name']}")
        if v.get("missed"):
            print(f"       missed: {v['missed']}")
        if v.get("bad_saves"):
            print(f"       bad_saves: {v['bad_saves']}")
        if not v.get("ttl_reasonable", True):
            print("       ttl: unreasonable")
        if v.get("notes"):
            print(f"       — {v['notes']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
