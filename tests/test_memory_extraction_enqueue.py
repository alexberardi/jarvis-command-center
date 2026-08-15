"""Memory-extraction enqueue payload: thinking OFF via reasoning_budget=0.

The old mechanism was a Qwen3 `/no_think` token appended to the user turn;
newer Qwen generations ignore that soft token, so the job now carries the
proxy's backend-agnostic ``reasoning_budget: 0`` (REST → enable_thinking=false,
in-process GGUF → empty-<think> prefill) instead.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app.services.memory_extraction_service as mes


def _transcript():
    return SimpleNamespace(
        id=1, user_message="hi", assistant_message="hello", tool_calls_json=None
    )


def _enqueue_and_capture(monkeypatch):
    monkeypatch.setattr(mes, "_get_command_center_url", lambda: "http://cc")
    monkeypatch.setattr(mes, "_get_llm_proxy_url", lambda: "http://proxy")
    captured = {}

    async def _fake_post(url, json_data=None, headers=None):
        captured["url"] = url
        captured["payload"] = json_data
        return {}

    with patch("app.core.utils.rest_client.post", new=_fake_post):
        asyncio.run(
            mes._enqueue_extraction(
                user_id=1,
                household_id="hh",
                transcripts=[_transcript()],
                existing_memories="none",
                transcript_svc=MagicMock(),
            )
        )
    return captured


def test_extraction_job_disables_thinking_via_reasoning_budget(monkeypatch):
    captured = _enqueue_and_capture(monkeypatch)
    request = captured["payload"]["request"]
    assert request["reasoning_budget"] == 0
    assert request["model"] == "background"


def test_extraction_prompt_no_longer_carries_no_think_token(monkeypatch):
    captured = _enqueue_and_capture(monkeypatch)
    request = captured["payload"]["request"]
    joined = " ".join(m["content"] for m in request["messages"])
    assert "/no_think" not in joined
