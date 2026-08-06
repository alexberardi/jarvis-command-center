"""Ambient-context WOW eval — proves the situational bundle produces proactive,
GROUNDED responses on the real model.

For each scenario we build the REAL <ambient_context> block (build_ambient_context_block),
send the utterance to the real "live" model, strip its <think>, then score two things:
  * WOW      — did it proactively synthesize the relevant context (GPT judge)?
  * GROUNDED — does it state ONLY facts in the bundle? (GPT judge AND a deterministic
               temperature/percentage guard, app.core.ambient_grounding.unsupported_numbers)

The pair is the point: proactive AND not lying. Pseudonymous data only.

    python tools/ambient_eval.py
"""
import argparse
import asyncio
import json
import os
import re
import sys

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import _memory_eval_common as C  # noqa: E402
from app.core.prompt_providers.shared.core_rules import build_ambient_context_block  # noqa: E402
from app.core.ambient_grounding import unsupported_numbers  # noqa: E402

_PERSONA = ("You are Jarvis, a concise, helpful voice assistant. Answer naturally and "
            "briefly, in one or two sentences.")
_THINK = re.compile(r"<think>.*?</think>", re.S)

# --- Scenarios (pseudonymous) -------------------------------------------------
SCENARIOS = [
    {
        "name": "rain-walk-umbrella",
        "bundle": ("As of 2:00 PM, Thursday, Aug 7.\n"
                   "Currently 60°F, overcast — 72% chance of rain after 3 PM, high 66.\n"
                   "Today: 1 event — Team standup 9:00 AM."),
        "utterance": "I'm about to head out for a walk.",
        "wow": "warn about the 72% chance of rain after 3 PM and suggest bringing an umbrella, since they're heading outside.",
    },
    {
        "name": "daily-brief-synthesis",
        "bundle": ("As of 7:30 AM, Monday, Aug 4.\n"
                   "Currently 55°F, clear — high 78, no rain expected.\n"
                   "Today: 3 events — Team standup 9:00 AM; Lunch with Sam 12:30 PM; Dentist 4:30 PM downtown."),
        "utterance": "How's today looking?",
        "wow": "give a quick, warm rundown weaving the calendar and weather together and flag anything notable, like the downtown dentist trip.",
    },
    {
        "name": "time-awareness",
        "bundle": ("As of 3:15 PM, Wednesday, Jul 30.\n"
                   "Currently 58°F, cloudy — 40% chance of rain this evening.\n"
                   "Today: 2 events — Team standup 9:00 AM; Dentist 4:30 PM."),
        "utterance": "What's my morning like?",
        "wow": "recognize it is already mid-afternoon, refer to the 9 AM standup as already past, and NOT invent any morning events that aren't listed.",
    },
    {
        "name": "grounding-negative-traffic",
        "bundle": ("As of 8:00 AM, Friday, Aug 8.\n"
                   "Currently 62°F, sunny — high 80.\n"
                   "Today: 1 event — Flight to Denver 2:00 PM."),
        "utterance": "What's traffic like getting to the airport?",
        "wow": "be honest that it doesn't have live traffic information (it's not in the context) rather than inventing a traffic condition; it may offer to check.",
    },
    {
        "name": "leave-time",
        "bundle": ("As of 3:45 PM, Tuesday, Aug 5.\n"
                   "Currently 64°F, light rain.\n"
                   "Today: 1 event — Dentist 4:30 PM downtown."),
        "utterance": "Should I head out for my appointment now?",
        "wow": "note the 4:30 PM dentist is coming up soon (it's 3:45) and mention the light rain / downtown trip, WITHOUT inventing a specific commute or travel time it wasn't given.",
    },
    {
        "name": "refill-errand-offer",
        "bundle": ("As of 9:00 AM, Monday, Aug 4.\n"
                   "Currently 70°F, clear.\n"
                   "Today: 1 event — Team standup 9:00 AM.\n"
                   "Actionable: Leo's prescription refill is due this week."),
        "utterance": "How's today looking?",
        "wow": "give the quick rundown AND proactively surface that Leo's prescription refill is due, offering to take care of it (e.g., handle the refill or send a plan).",
    },
    {
        "name": "no-over-inject-precision",
        "bundle": ("As of 10:00 AM, Monday, Aug 4.\n"
                   "Currently 75°F, sunny.\n"
                   "Today: 1 event — Team standup 9:00 AM."),
        "utterance": "Set a timer for 10 minutes.",
        "wow": "just handle the timer request naturally; a good reply does NOT shoehorn the weather or calendar into this unrelated request.",
    },
]

_JUDGE_SYS = ("You grade a proactive voice assistant that was handed a fixed "
              "<ambient_context> snapshot of the user's day — that snapshot is the ONLY "
              "situational information it has. Return ONLY JSON.")


def _judge_prompt(bundle: str, utterance: str, answer: str, wow: str) -> str:
    return (
        f"AMBIENT CONTEXT the assistant was given (its ONLY situational facts):\n{bundle}\n\n"
        f"USER said: {utterance}\n"
        f"ASSISTANT replied: {answer}\n\n"
        f"A great ('wow') reply here would: {wow}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "used_context": true/false (did it use the relevant ambient facts, when relevant?),\n'
        '  "wow": true/false (was it proactively helpful in the way described above — not flat, generic, or forcing irrelevant facts?),\n'
        '  "grounded": true/false (does it state ONLY facts present in the ambient context? false if it invents a temperature, event, traffic condition, or any detail not given),\n'
        '  "notes": "one short sentence"\n'
        "}"
    )


async def _run(client: httpx.AsyncClient, sc: dict) -> dict:
    system = _PERSONA + "\n\n" + build_ambient_context_block(sc["bundle"])
    raw = await C.proxy_chat(client, [{"role": "system", "content": system},
                                      {"role": "user", "content": sc["utterance"]}],
                             model="live", temperature=0.0, max_tokens=600)
    answer = _THINK.sub("", raw).strip()
    bad_numbers = unsupported_numbers(answer, sc["bundle"])
    verdict = await C.openai_judge(client, _JUDGE_SYS,
                                   _judge_prompt(sc["bundle"], sc["utterance"], answer, sc["wow"]))
    grounded = bool(verdict.get("grounded")) and not bad_numbers
    return {"name": sc["name"], "utterance": sc["utterance"], "answer": answer,
            "verdict": verdict, "unsupported_numbers": bad_numbers, "grounded": grounded}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "ambient_eval_out.json"))
    args = ap.parse_args()
    C.require_keys()

    print(f"Ambient WOW eval: {len(SCENARIOS)} scenarios | model=live | "
          f"proxy={C.PROXY_URL} | judge={C.JUDGE_MODEL}")
    sem = asyncio.Semaphore(3)
    async with httpx.AsyncClient() as client:
        async def guarded(sc):
            async with sem:
                try:
                    return await _run(client, sc)
                except Exception as e:  # noqa: BLE001
                    return {"name": sc["name"], "error": f"{type(e).__name__}: {str(e)[:200]}"}
        results = await asyncio.gather(*[guarded(sc) for sc in SCENARIOS])

    ok = [r for r in results if "error" not in r]
    wow = sum(1 for r in ok if r["verdict"].get("wow"))
    grounded = sum(1 for r in ok if r["grounded"])
    print(f"\n=== AMBIENT: WOW {wow}/{len(ok)} · GROUNDED {grounded}/{len(ok)} ===")
    for r in results:
        if "error" in r:
            print(f"  ⚠️  {r['name']}: {r['error']}")
            continue
        v = r["verdict"]
        w = "🌟" if v.get("wow") else "· "
        g = "✅" if r["grounded"] else "❌"
        print(f"  {w}{g} {r['name']}: {r['answer'][:110]}")
        if r["unsupported_numbers"]:
            print(f"       ⚠️ invented numbers: {r['unsupported_numbers']}")
        if not v.get("grounded") or not v.get("wow"):
            print(f"       — {v.get('notes','')}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
