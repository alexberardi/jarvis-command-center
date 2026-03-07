"""
Parameter Refinement — secondary LLM call for refinable params.

When a command marks parameters as ``_refinable``, they are stripped from
the Stage 1 tool schema so the primary (small) model only handles intent
classification.  After the model picks the right tool, this module makes a
focused secondary LLM call to fill in those parameters using a compact
prompt with only the relevant enum options.

Same two-stage pattern as ``ha_entity_resolver``.
"""

import json
import logging
from typing import Any, Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.llm_proxy_client import LLMProxyClient

logger = logging.getLogger("uvicorn")

# Tools handled by dedicated resolvers — skip generic refinement.
# The HA entity resolver already resolves action + entity_id with
# device-domain context, which is more accurate than generic refinement.
_RESOLVER_HANDLED_TOOLS: Set[str] = {"control_device", "get_device_status"}


async def refine_params(
    client_calls: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    user_utterance: str,
    llm_client: "LLMProxyClient",
) -> List[Dict[str, Any]]:
    """Refine parameters for tool calls that have refinable params.

    Scans *client_calls* for commands whose full tool definition (in *tools*)
    contains ``_refinable`` properties.  For each match, builds a focused
    prompt and makes a secondary LLM call to resolve the missing params,
    then merges them back into the tool call arguments.

    Tools in ``_RESOLVER_HANDLED_TOOLS`` are skipped — their refinable
    params are stripped from Stage 1 for token savings, but a dedicated
    resolver (e.g. HA entity resolver) handles them instead.

    Args:
        client_calls: Tool calls produced by Stage 1.
        tools: Full tool definitions (with ``_refinable`` markers intact).
        user_utterance: The original user voice command.
        llm_client: LLM proxy client for the secondary call.

    Returns:
        The same *client_calls* list, potentially with refined params merged in.
    """
    for idx, call in enumerate(client_calls):
        tool_name: str = call.get("function", {}).get("name", "")

        # Skip tools with dedicated resolvers
        if tool_name in _RESOLVER_HANDLED_TOOLS:
            continue

        refinable_params: Dict[str, Dict[str, Any]] | None = _get_refinable_params(tool_name, tools)
        if not refinable_params:
            continue

        logger.info(
            "Param refinement: refining %d param(s) for '%s'",
            len(refinable_params), tool_name,
        )

        # Extract already-set args to provide category context to refinement
        args_raw: str = call.get("function", {}).get("arguments", "{}")
        try:
            existing_args: Dict[str, Any] = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            existing_args = {}

        examples: List[Dict[str, Any]] = _get_refinement_examples(tool_name, tools, refinable_params)
        prompt: str = _build_refinement_prompt(user_utterance, tool_name, refinable_params, examples, existing_args)
        resolved: Dict[str, Any] | None = await _call_llm(prompt, llm_client)
        if resolved:
            _merge_refined_params(client_calls[idx], resolved)
            logger.info("Param refinement: merged %s for '%s'", resolved, tool_name)
        else:
            logger.warning("Param refinement: LLM returned no usable result for '%s'", tool_name)

    return client_calls


def _get_refinable_params(
    tool_name: str,
    tools: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]] | None:
    """Find params marked ``_refinable`` for *tool_name* in the full tools list."""
    for tool in tools:
        func: Dict[str, Any] = tool.get("function", {})
        if func.get("name") != tool_name:
            continue
        props: Dict[str, Any] = func.get("parameters", {}).get("properties", {})
        refinable: Dict[str, Dict[str, Any]] = {}
        for pname, pschema in props.items():
            if isinstance(pschema, dict) and pschema.get("_refinable"):
                refinable[pname] = pschema
        return refinable if refinable else None
    return None


def _get_refinement_examples(
    tool_name: str,
    tools: List[Dict[str, Any]],
    refinable_params: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract few-shot examples from the tool definition, filtered to refinable params.

    Reads the ``examples`` list from the cached tool schema and returns
    each example with only the refinable parameter values (setting missing
    ones to ``None`` so the model sees explicit null outputs).
    """
    for tool in tools:
        func: Dict[str, Any] = tool.get("function", {})
        if func.get("name") != tool_name:
            continue
        raw_examples: List[Dict[str, Any]] = tool.get("examples", [])
        if not raw_examples:
            return []
        result: List[Dict[str, Any]] = []
        for ex in raw_examples:
            utterance: str = ex.get("voice_command", "")
            all_params: Dict[str, Any] = ex.get("expected_parameters", {})
            if not utterance:
                continue
            # Filter to only refinable param keys
            filtered: Dict[str, Any] = {
                pname: all_params.get(pname) for pname in refinable_params
            }
            result.append({"utterance": utterance, "params": filtered})
        return result
    return []


def _build_refinement_prompt(
    utterance: str,
    tool_name: str,
    refinable_params: Dict[str, Dict[str, Any]],
    examples: List[Dict[str, Any]] | None = None,
    existing_args: Dict[str, Any] | None = None,
) -> str:
    """Build a focused prompt for parameter refinement.

    When few-shot examples are available, uses them as the primary signal
    (small models pattern-match better from examples than from descriptions).
    Falls back to descriptions only when no examples exist.

    Args:
        existing_args: Already-selected params from Stage 1 (e.g., action category).
            Included as context so the model can narrow down conditional enums.
    """
    param_lines: List[str] = []
    for pname, pschema in refinable_params.items():
        desc: str = pschema.get("description", "")
        enum: List[str] | None = pschema.get("enum")
        ptype: str = pschema.get("type", "string")
        if enum:
            param_lines.append(f"- {pname} ({ptype}): one of [{', '.join(enum)}]")
        else:
            param_lines.append(f"- {pname} ({ptype}): {desc}")

    json_keys: str = ", ".join(f'"{p}": ...' for p in refinable_params)

    prompt: str = (
        f"Given this voice command, determine the parameters for {tool_name}.\n\n"
        f"Parameters:\n" + "\n".join(param_lines) + "\n\n"
    )
    if existing_args:
        prompt += f"Already selected: {json.dumps(existing_args)}\n\n"
    if examples:
        prompt += "Examples:\n"
        for ex in examples:
            prompt += f'"{ex["utterance"]}" → {json.dumps(ex["params"])}\n'
        prompt += "\n"
    prompt += (
        f'Command: "{utterance}"\n'
        f"Return ONLY JSON: {{{json_keys}}}\n"
        f"Use null for parameters that don't apply."
    )
    return prompt


async def _call_llm(
    prompt: str,
    llm_client: "LLMProxyClient",
) -> Dict[str, Any] | None:
    """Make a secondary LLM call for param refinement."""
    try:
        response: Dict[str, Any] = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            include_date_context=False,
            max_tokens=256,
        )

        raw_content: str = ""
        if isinstance(response, dict):
            choices: List[Dict[str, Any]] = response.get("choices", [])
            if choices:
                raw_content = choices[0].get("message", {}).get("content", "")
            if not raw_content:
                raw_content = response.get("message", "")

        if not raw_content:
            logger.warning("Param refinement: empty LLM response")
            return None

        # Extract JSON (handle markdown code fences)
        content: str = raw_content.strip()
        if content.startswith("```"):
            lines: List[str] = content.split("\n")
            json_lines: List[str] = [
                line for line in lines
                if not line.strip().startswith("```")
            ]
            content = "\n".join(json_lines).strip()

        result: Any = json.loads(content)
        if not isinstance(result, dict):
            logger.warning("Param refinement: response is not a dict: %s", content[:200])
            return None

        return result

    except json.JSONDecodeError as e:
        logger.warning("Param refinement: JSON parse error: %s", e)
        return None
    except Exception as e:
        logger.error("Param refinement: error: %s", e)
        return None


def _merge_refined_params(
    call: Dict[str, Any],
    resolved: Dict[str, Any],
) -> None:
    """Merge refined params into tool call arguments."""
    args_raw: str = call["function"].get("arguments", "{}")
    try:
        args: Dict[str, Any] = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        args = {}
    for key, value in resolved.items():
        if value is not None:
            args[key] = value
    call["function"]["arguments"] = json.dumps(args)
