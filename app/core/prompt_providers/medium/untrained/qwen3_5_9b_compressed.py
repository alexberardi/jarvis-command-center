"""
Qwen3_5_9B_Compressed - Prompt provider for Qwen3.5-9B (Q4_K_M GGUF, MTP).

Qwen3.5-9B is the same ChatML + ``<think>`` reasoning family as Qwen3-8B
(verified from the GGUF's own chat_template: ``<|im_start|>``/``<|im_end|>``
turns, ``<think>…</think>`` reasoning blocks). So this reuses the proven
``Qwen3_8B_Compressed`` prompt wholesale — compressed ``<tools>`` block, JSON
``<tool_call>`` format, ``/think`` vs ``/no_think`` control, and
``<think>``-block stripping — and only overrides ``name`` so it can be selected
independently via the ``llm.interface`` setting and tuned on its own later.

Serving notes (llama.cpp side, independent of this prompt formatting):
  * arch ``qwen35``; loads on llama-cpp-python 0.3.20 as a normal 9B.
  * 256K native context (``qwen35.context_length = 262144``).
  * carries 1 MTP ``nextn`` prediction layer — usable for ~1.5-2x faster
    speculative decoding ONLY on a llama.cpp build that includes the MTP merge
    (merged upstream 2026-05-16). Without that, it runs as a standard 9B.

KNOWN RISK — tool-call format: Qwen3.5's NATIVE tool-call syntax (per its
chat_template) is XML — ``<tool_call><function=NAME><parameter=P>…`` — NOT the
JSON-in-``<tool_call>`` object this provider (inherited) instructs and parses.
We keep the JSON format on purpose: it's proven at 95%+ on Qwen3-8B via
instruction-following, and the parser already understands it. If Qwen3.5 ignores
the instruction and emits its native XML function calls, tool-based commands will
silently fall through to direct-answer — that's the signal to add a native
XML-format ``build_system_prompt`` + ``parse_response`` here (subclass, don't
touch the 8B provider).
"""

from typing import Any, Dict

from app.core.prompt_providers.medium.untrained.qwen3_8b_compressed import (
    Qwen3_8B_Compressed,
)


class Qwen3_5_9B_Compressed(Qwen3_8B_Compressed):
    """Qwen3.5-9B (MTP GGUF) — reuses the Qwen3-8B compressed prompt verbatim."""

    @property
    def name(self) -> str:
        return "Qwen3_5_9B_Compressed"

    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["provider_name"] = self.name
        caps["model_family"] = "qwen3.5"
        return caps
