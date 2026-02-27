"""
Get Home Assistant Entities Tool.

Looks up Home Assistant devices from the node's cached context,
filtered by domain and optionally by area or floor.  The LLM calls
this on demand instead of having every device dumped into the system
prompt, saving ~1,350 tokens per request.
"""

import logging
from typing import Any, Dict, List, Optional

from app.core.interfaces.iserver_tool import IServerTool
from app.core.conversation_cache import conversation_cache
from app.core.prompt_providers.shared.context_builders import _format_device_line

logger = logging.getLogger("uvicorn")

# Domains the tool can query
_SUPPORTED_DOMAINS = ("light", "switch", "lock", "cover", "climate", "fan", "scene")


class GetHAEntitiesTool(IServerTool):
    """Tool for looking up Home Assistant devices by domain/area/floor."""

    @property
    def name(self) -> str:
        return "get_ha_entities"

    @property
    def description(self) -> str:
        return (
            "Look up Home Assistant devices by domain and optionally by area or floor. "
            "Call this BEFORE control_device or get_device_status to find entity IDs."
        )

    @property
    def included_system_prompt_text(self) -> str:
        return ""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "enum": list(_SUPPORTED_DOMAINS),
                    "description": "Device domain to look up (e.g. light, switch, lock).",
                },
                "area": {
                    "type": "string",
                    "description": "Optional area/room name to filter by (e.g. 'Living Room').",
                },
                "floor": {
                    "type": "string",
                    "description": "Optional floor name to filter by (e.g. 'Downstairs'). Resolves to all areas on that floor.",
                },
            },
            "required": ["domain"],
        }

    def execute(
        self,
        domain: str,
        area: Optional[str] = None,
        floor: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Return matching HA entities from the node's cached context.

        Args:
            domain: Device domain (light, switch, lock, etc.)
            area: Optional area name filter (case-insensitive)
            floor: Optional floor name filter (resolves to areas via floors dict)
            conversation_id: Conversation ID (passed by tool executor)

        Returns:
            Dict with list of matching devices or an error.
        """
        logger.info(
            "      🏠 get_ha_entities called: domain=%s area=%s floor=%s",
            domain, area, floor,
        )

        if not conversation_id:
            logger.warning("      └─ ⚠️  Missing conversation_id")
            return {
                "error": "missing_conversation_id",
                "message": "Conversation ID is required but was not provided",
            }

        node_context: Optional[Dict[str, Any]] = conversation_cache.get_node_context(
            conversation_id
        )
        if not node_context:
            logger.warning("      └─ ⚠️  No node context for %s", conversation_id[:8])
            return {
                "error": "no_node_context",
                "message": "No node context found for this conversation",
            }

        ha_data: Dict[str, Any] = (
            node_context.get("agents", {}).get("home_assistant", {})
        )
        if not ha_data:
            logger.warning("      └─ ⚠️  No HA data in node context")
            return {
                "error": "no_ha_data",
                "message": "This node has no Home Assistant data",
            }

        # ------------------------------------------------------------------
        # Resolve floor → set of area names (case-insensitive)
        # ------------------------------------------------------------------
        allowed_areas: Optional[set[str]] = None
        floors: Dict[str, List[str]] = ha_data.get("floors", {})

        if floor:
            floor_lower = floor.lower()
            for floor_name, area_list in floors.items():
                if floor_name.lower() == floor_lower:
                    allowed_areas = {a.lower() for a in area_list}
                    break
            if allowed_areas is None:
                return {
                    "error": "floor_not_found",
                    "message": f"Floor '{floor}' not found. Available floors: {', '.join(floors.keys())}",
                }

        if area:
            area_lower = {area.lower()}
            allowed_areas = area_lower if allowed_areas is None else allowed_areas & area_lower

        # ------------------------------------------------------------------
        # Collect devices for the requested domain
        # ------------------------------------------------------------------
        light_controls: Dict[str, Any] = ha_data.get("light_controls", {})
        device_controls: Dict[str, Any] = ha_data.get("device_controls", {})
        devices: List[Dict[str, Any]] = []

        if domain == "light":
            # Room-group lights from light_controls
            for lc_name, lc_info in light_controls.items():
                devices.append({
                    "entity_id": lc_info.get("entity_id", ""),
                    "name": lc_name,
                    "state": lc_info.get("state", "unknown"),
                    "area": lc_info.get("area", ""),
                })

            # Individual lights from device_controls, excluding room groups
            room_group_ids: set[str] = {
                info.get("entity_id") for info in light_controls.values()
            }
            for dev in device_controls.get("light", []):
                if dev.get("entity_id") in room_group_ids:
                    continue
                if dev.get("state") == "unavailable":
                    continue
                devices.append({
                    "entity_id": dev.get("entity_id", ""),
                    "name": dev.get("name", ""),
                    "state": dev.get("state", "unknown"),
                    "area": dev.get("area", ""),
                })
        else:
            for dev in device_controls.get(domain, []):
                if dev.get("state") == "unavailable":
                    continue
                devices.append({
                    "entity_id": dev.get("entity_id", ""),
                    "name": dev.get("name", ""),
                    "state": dev.get("state", "unknown"),
                    "area": dev.get("area", ""),
                })

        # ------------------------------------------------------------------
        # Apply area filter
        # ------------------------------------------------------------------
        if allowed_areas is not None:
            devices = [
                d for d in devices
                if d.get("area", "").lower() in allowed_areas
            ]

        logger.info(
            "      └─ ✅ Returning %d %s device(s)%s",
            len(devices),
            domain,
            f" (area filter: {area or floor})" if (area or floor) else "",
        )

        return {
            "domain": domain,
            "count": len(devices),
            "devices": [
                _format_device_line(d) for d in devices
            ],
        }
