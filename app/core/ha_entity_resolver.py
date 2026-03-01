"""
Home Assistant Entity Resolver.

Resolves HA entity_id + action via a secondary LLM call when the primary
model produces invalid or placeholder entity IDs.  Follows the same pattern
as ResolveRelativeDateTool.resolve_with_llm_fallback().
"""

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.llm_proxy_client import LLMProxyClient

logger = logging.getLogger("uvicorn")

# Map common short/incorrect action names → valid HA service actions.
# Applied as a fast fixup before enum validation to avoid wasting LLM retries.
ACTION_ALIASES: Dict[str, str] = {
    "on": "turn_on",
    "off": "turn_off",
    "open": "open_cover",
    "close": "close_cover",
    "stop": "stop_cover",
    "play": "media_play",
    "pause": "media_pause",
}

# Hallucinated tool names that should be redirected to control_device.
# The 3B model sometimes invents HA-sounding tools that don't exist.
_HA_TOOL_ALIASES: Dict[str, Dict[str, str]] = {
    "set_temperature": {"action": "set_temperature"},
    "set_climate": {"action": "set_temperature"},
    "set_climate_mode": {"action": "set_hvac_mode"},
    "set_thermostat": {"action": "set_temperature"},
    "turn_on_light": {"action": "turn_on"},
    "turn_off_light": {"action": "turn_off"},
    "lock_door": {"action": "lock"},
    "unlock_door": {"action": "unlock"},
    "open_garage": {"action": "open_cover"},
    "close_garage": {"action": "close_cover"},
}


def normalize_ha_action(action: Optional[str]) -> Optional[str]:
    """Normalize a short or incorrect action name to a valid HA service action.

    Returns the normalized action, or the original if no alias found.
    """
    if not action or not isinstance(action, str):
        return action
    return ACTION_ALIASES.get(action.strip().lower(), action)


def get_ha_tool_redirect(tool_name: str) -> Optional[Dict[str, str]]:
    """Check if a hallucinated tool name should redirect to control_device.

    Returns dict with default args (e.g. {"action": "set_temperature"}) or None.
    """
    return _HA_TOOL_ALIASES.get(tool_name)


# Valid actions per domain for the resolution prompt
_DOMAIN_ACTIONS: Dict[str, List[str]] = {
    "light": ["turn_on", "turn_off", "toggle"],
    "switch": ["turn_on", "turn_off", "toggle"],
    "lock": ["lock", "unlock"],
    "cover": ["open_cover", "close_cover", "stop_cover"],
    "climate": ["set_temperature", "turn_on", "turn_off"],
    "fan": ["turn_on", "turn_off", "toggle"],
    "scene": ["turn_on"],
    "script": ["turn_on"],
    "media_player": ["turn_on", "turn_off", "media_play", "media_pause"],
    "vacuum": ["start", "stop", "return_to_base"],
}


def _needs_resolution(entity_id: Optional[str]) -> bool:
    """Return True if the entity_id looks invalid or is a placeholder.

    Invalid patterns:
    - None / empty
    - Contains spaces or braces (LLM placeholder like "{entity_id}")
    - Missing a dot (all HA entity IDs are domain.object_id)
    - Starts with "get_ha" (model tried to nest the lookup tool name)
    """
    if not entity_id or not isinstance(entity_id, str):
        return True
    entity_id = entity_id.strip()
    if not entity_id:
        return True
    if " " in entity_id or "{" in entity_id or "}" in entity_id:
        return True
    if "." not in entity_id:
        return True
    if entity_id.lower().startswith("get_ha"):
        return True
    return False


def _build_device_catalog(ha_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flatten light_controls + device_controls into a compact device list.

    Returns a list of dicts with keys: entity_id, name, area, state, domain.
    """
    catalog: List[Dict[str, str]] = []

    light_controls: Dict[str, Any] = ha_data.get("light_controls", {})
    for name, info in light_controls.items():
        catalog.append({
            "entity_id": info.get("entity_id", ""),
            "name": name,
            "area": info.get("area", name),
            "state": info.get("state", "unknown"),
            "domain": "light",
        })

    device_controls: Dict[str, Any] = ha_data.get("device_controls", {})
    room_group_ids: set[str] = {
        info.get("entity_id") for info in light_controls.values()
    }

    for domain, devices in device_controls.items():
        if not isinstance(devices, list):
            continue
        for dev in devices:
            eid = dev.get("entity_id", "")
            if domain == "light" and eid in room_group_ids:
                continue
            if dev.get("state") == "unavailable":
                continue
            catalog.append({
                "entity_id": eid,
                "name": dev.get("name", ""),
                "area": dev.get("area", ""),
                "state": dev.get("state", "unknown"),
                "domain": domain,
            })

    return catalog


def _build_resolution_prompt(
    utterance: str,
    catalog: List[Dict[str, str]],
    floors: Dict[str, List[str]],
    tool_name: str,
) -> str:
    """Build a compact prompt for the secondary LLM call."""
    device_lines: List[str] = []
    for dev in catalog:
        device_lines.append(
            f"{dev['entity_id']} | {dev['area']} | {dev['name']} | {dev['state']}"
        )

    floor_lines: List[str] = []
    for fname, areas in floors.items():
        floor_lines.append(f"{fname}=[{', '.join(areas)}]")

    action_lines: List[str] = []
    for domain, actions in _DOMAIN_ACTIONS.items():
        action_lines.append(f"{domain}={','.join(actions)}")

    if tool_name == "control_device":
        instruction = (
            'Return ONLY JSON: {"entity_id": "...", "action": "..."}\n'
            'If a value is needed (e.g. temperature), add a "value" key.'
        )
    else:
        instruction = 'Return ONLY JSON: {"entity_id": "..."}'

    prompt = (
        f"Given this voice command and device list, determine the device"
        f"{' and action' if tool_name == 'control_device' else ''}.\n\n"
        f'Command: "{utterance}"\n\n'
        f"Devices:\n{chr(10).join(device_lines)}\n\n"
    )
    if floor_lines:
        prompt += f"Floors: {', '.join(floor_lines)}\n\n"
    if tool_name == "control_device":
        prompt += f"Valid actions: {' | '.join(action_lines)}\n\n"
    prompt += instruction

    return prompt


async def resolve_ha_control(
    utterance: str,
    ha_data: Dict[str, Any],
    llm_client: "LLMProxyClient",
) -> Optional[Dict[str, Any]]:
    """Resolve entity_id + action for a control_device call.

    Returns dict with at least {"entity_id": str, "action": str}, or None
    on failure.
    """
    return await _resolve(utterance, ha_data, llm_client, "control_device")


async def resolve_ha_status(
    utterance: str,
    ha_data: Dict[str, Any],
    llm_client: "LLMProxyClient",
) -> Optional[Dict[str, Any]]:
    """Resolve entity_id for a get_device_status call.

    Returns dict with at least {"entity_id": str}, or None on failure.
    """
    return await _resolve(utterance, ha_data, llm_client, "get_device_status")


async def _resolve(
    utterance: str,
    ha_data: Dict[str, Any],
    llm_client: "LLMProxyClient",
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    """Internal resolver: build prompt, call LLM, parse JSON response."""
    catalog = _build_device_catalog(ha_data)
    if not catalog:
        logger.warning("🏠 HA resolver: no devices in catalog")
        return None

    floors: Dict[str, List[str]] = ha_data.get("floors", {})
    prompt = _build_resolution_prompt(utterance, catalog, floors, tool_name)

    try:
        logger.info(
            "🏠 HA resolver: resolving %s for utterance '%s' (%d devices)",
            tool_name, utterance[:60], len(catalog),
        )
        response = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            include_date_context=False,
            max_tokens=256,
        )

        raw_content: str = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                raw_content = choices[0].get("message", {}).get("content", "")
            if not raw_content:
                raw_content = response.get("message", "")

        if not raw_content:
            logger.warning("🏠 HA resolver: empty LLM response")
            return None

        # Extract JSON from response (handle markdown code fences)
        content = raw_content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            json_lines = [
                l for l in lines
                if not l.strip().startswith("```")
            ]
            content = "\n".join(json_lines).strip()

        result = json.loads(content)
        if not isinstance(result, dict):
            logger.warning("🏠 HA resolver: response is not a dict: %s", content[:200])
            return None

        # Validate entity_id exists in catalog
        resolved_eid = result.get("entity_id", "")
        catalog_ids = {d["entity_id"] for d in catalog}
        if resolved_eid not in catalog_ids:
            logger.warning(
                "🏠 HA resolver: resolved entity_id '%s' not in catalog",
                resolved_eid,
            )
            return None

        logger.info("🏠 HA resolver: resolved → %s", result)
        return result

    except json.JSONDecodeError as e:
        logger.warning("🏠 HA resolver: JSON parse error: %s", e)
        return None
    except Exception as e:
        logger.error("🏠 HA resolver: error: %s", e)
        return None
