"""Shared plumbing for the memory behavioral evals.

Two model surfaces:
  * The REAL Jarvis LLM proxy (the same models the assistant uses) for the
    system-under-test — chat completions over /v1/chat/completions.
  * OpenAI (via CHAT_GPT_TOKEN) as an independent GPT JUDGE.

Creds are read from jarvis-command-center/.env (JARVIS_APP_ID / JARVIS_APP_KEY for
app-to-app auth to the proxy; CHAT_GPT_TOKEN for the judge). NO real PII belongs in
any scenario — this data is sent to external models.
"""
import json
import os

import httpx
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # jarvis-command-center
load_dotenv(os.path.join(_ROOT, ".env"))

# The proxy isn't local on this laptop — config-service resolves it to the GPU host.
PROXY_URL = os.environ.get("JARVIS_LLM_PROXY_API_URL", "http://10.0.0.122:7704").rstrip("/")
APP_ID = os.environ.get("JARVIS_APP_ID", "")
APP_KEY = os.environ.get("JARVIS_APP_KEY", "")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_KEY = os.environ.get("CHAT_GPT_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
JUDGE_MODEL = os.environ.get("SIM_JUDGE_MODEL", "gpt-4o")


def require_keys() -> None:
    missing = [n for n, v in [("JARVIS_APP_ID", APP_ID), ("JARVIS_APP_KEY", APP_KEY),
                              ("CHAT_GPT_TOKEN", OPENAI_KEY)] if not v]
    if missing:
        raise SystemExit(f"Missing env {missing} — expected in jarvis-command-center/.env")


def load_scenarios() -> dict:
    with open(os.path.join(_ROOT, "tools", "memory_eval_scenarios.json")) as f:
        return json.load(f)


def _extract_content(result: dict) -> str:
    """Pull the assistant text out of the proxy's response (shape varies)."""
    ch = result.get("choices")
    if isinstance(ch, list) and ch:
        msg = ch[0].get("message") or {}
        if msg.get("content"):
            return msg["content"]
    m = result.get("message")
    if isinstance(m, dict) and m.get("content"):
        return m["content"]
    if isinstance(m, str):
        return m
    return result.get("content") or ""


async def proxy_chat(client: httpx.AsyncClient, messages: list, *, model: str,
                     temperature: float = 0.0, response_format: dict | None = None,
                     max_tokens: int | None = None) -> str:
    """One chat completion against the REAL Jarvis LLM proxy."""
    payload: dict = {"model": model, "temperature": temperature, "messages": messages}
    if response_format:
        payload["response_format"] = response_format
    if max_tokens:
        payload["max_tokens"] = max_tokens
    r = await client.post(f"{PROXY_URL}/v1/chat/completions",
                          headers={"X-Jarvis-App-Id": APP_ID, "X-Jarvis-App-Key": APP_KEY,
                                   "Content-Type": "application/json"},
                          json=payload, timeout=120)
    r.raise_for_status()
    return _extract_content(r.json())


async def openai_judge(client: httpx.AsyncClient, system: str, user: str) -> dict:
    """An independent GPT verdict (JSON mode)."""
    payload = {"model": JUDGE_MODEL, "temperature": 0,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    r = await client.post(OPENAI_URL,
                          headers={"Authorization": f"Bearer {OPENAI_KEY}",
                                   "Content-Type": "application/json"},
                          json=payload, timeout=120)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])
