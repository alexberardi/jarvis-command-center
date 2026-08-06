"""Recall-usage eval — the "does the model pick memories up from various utterances?" test.

For each scenario we seed the memory into the REAL profile block the live path injects
(build_speaker_block), then send each utterance to the REAL "live" model and have a GPT
judge decide whether the answer USED the seeded fact. It also checks the negative
utterances (should_not_recall) to catch OVER-recall — shoehorning the fact where it's
irrelevant.

This is prompt-level (pinned-injection) recall, which is how the live path surfaces
identity facts; it does not exercise the semantic `recall` tool (that needs the full
voice/node pipeline).

    python tools/memory_recall_eval.py            # all scenarios
    python tools/memory_recall_eval.py --limit 2  # quick smoke
"""
import argparse
import asyncio
import json
import os
import sys

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import _memory_eval_common as C  # noqa: E402

CHAT_MODEL = os.environ.get("SIM_CHAT_MODEL", "live")
SPEAKER = "Jordan"

try:  # use the exact production profile-block wording if importable
    from app.core.prompt_providers.shared.core_rules import build_speaker_block
except Exception:  # noqa: BLE001 — fall back to the verbatim format (core_rules.py:265)
    def build_speaker_block(name: str, user_memories: str) -> str:
        return (f"You are speaking with {name}.\n\nUser Profile - If user asks a question "
                f"about one of these items you must respond with an answer:\n{user_memories}")

_PERSONA = "You are Jarvis, a concise, helpful voice assistant. Answer naturally and briefly."

_JUDGE_SYS = ("You grade whether a voice assistant USED a known fact about the user in its "
              "reply. 'Used' means the answer reflects/applies the fact (personalizes to it), "
              "not merely that it's plausible. Return ONLY JSON.")


def _system_prompt(seed_memories: list) -> str:
    lines = "\n".join(f"- {m['content']}" for m in seed_memories)
    return _PERSONA + "\n\n" + build_speaker_block(SPEAKER, lines)


def _judge_prompt(sc: dict, pos: list, neg: list) -> str:
    return (
        f"Known fact about the user (seeded into the assistant's profile): {sc['expected_fact']}\n"
        f"Full seeded profile: {json.dumps(sc['seed_memories'])}\n\n"
        "SHOULD-USE turns — the reply SHOULD reflect/apply the fact:\n"
        f"{json.dumps(pos, indent=1)}\n\n"
        "SHOULD-NOT turns — the fact is irrelevant; a good reply does NOT force it in:\n"
        f"{json.dumps(neg, indent=1)}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "used": [ {"utterance": ..., "used": true/false} for each SHOULD-USE turn ],\n'
        '  "over_recalled": [ {"utterance": ..., "over": true/false} for each SHOULD-NOT turn ],\n'
        '  "verdict": "pass" or "fail"  (pass = used on all/most SHOULD-USE AND no over_recall),\n'
        '  "notes": "one short sentence"\n'
        "}"
    )


async def _answer(client, system, utterance):
    ans = await C.proxy_chat(client, [{"role": "system", "content": system},
                                      {"role": "user", "content": utterance}],
                             model=CHAT_MODEL, temperature=0.0, max_tokens=200)
    return {"utterance": utterance, "answer": ans}


async def _run_one(client, sc):
    system = _system_prompt(sc["seed_memories"])
    pos = [await _answer(client, system, u) for u in sc.get("should_recall", [])]
    neg = [await _answer(client, system, u) for u in sc.get("should_not_recall", [])]
    verdict = await C.openai_judge(client, _JUDGE_SYS, _judge_prompt(sc, pos, neg))
    return {"name": sc["name"], "answers": {"should_recall": pos, "should_not_recall": neg},
            "verdict": verdict}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=os.path.join(_HERE, "recall_eval_out.json"))
    args = ap.parse_args()

    C.require_keys()
    scenarios = C.load_scenarios()["recall_scenarios"]
    if args.limit:
        scenarios = scenarios[:args.limit]

    print(f"Recall eval: {len(scenarios)} scenarios | model={CHAT_MODEL} | "
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
    # aggregate per-utterance recall rate
    used_hits = used_total = over = 0
    for r in results:
        v = r.get("verdict", {})
        for u in v.get("used", []):
            used_total += 1
            used_hits += 1 if u.get("used") else 0
        over += sum(1 for u in v.get("over_recalled", []) if u.get("over"))
    print(f"\n=== RECALL: {passed}/{len(results)} scenarios pass"
          + (f", {errored} errored" if errored else "") + " ===")
    if used_total:
        print(f"    utterance-level: {used_hits}/{used_total} should-use utterances recalled "
              f"({100*used_hits//used_total}%); {over} over-recalls")
    for r in results:
        if "error" in r:
            print(f"  ⚠️  {r['name']}: {r['error']}")
            continue
        v = r["verdict"]
        mark = "✅" if v.get("verdict") == "pass" else "❌"
        misses = [u["utterance"] for u in v.get("used", []) if not u.get("used")]
        overs = [u["utterance"] for u in v.get("over_recalled", []) if u.get("over")]
        print(f"  {mark} {r['name']}")
        if misses:
            print(f"       did NOT recall on: {misses}")
        if overs:
            print(f"       OVER-recalled on: {overs}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
