"""
Shared context-building utilities for prompt providers.

Extracted from JarvisToolModel._build_system_prompt and
JarvisAdapterModel._build_agent_context_section to avoid duplication
across prompt providers.
"""

from typing import Any, Dict, List


def build_direct_answer_section(
    available_commands: List[Dict[str, Any]],
) -> str:
    """
    Build the direct answer policy section from command flags.

    Separates commands into "must call tools" vs "direct answer allowed"
    based on the allow_direct_answer flag on each command.

    Args:
        available_commands: Command flag dicts with command_name and
            allow_direct_answer fields.

    Returns:
        Formatted policy string, or empty string if no policies apply.
    """
    must_call_tools: list[str] = []
    direct_answer_allowed: list[str] = []

    for cmd in available_commands:
        cmd_name = cmd.get("command_name")
        if not cmd_name:
            continue
        allow_direct = cmd.get("allow_direct_answer")
        if allow_direct is False:
            must_call_tools.append(cmd_name)
        elif allow_direct is True:
            direct_answer_allowed.append(cmd_name)

    if not must_call_tools and not direct_answer_allowed:
        return ""

    section = "\nDirect Answer Policy:\n"
    if must_call_tools:
        section += f"- MUST call tools for: {', '.join(sorted(set(must_call_tools)))}\n"
    if direct_answer_allowed:
        section += f"- Direct answers allowed for: {', '.join(sorted(set(direct_answer_allowed)))}\n"

    return section


def _format_device_line(dev: Dict[str, Any]) -> str:
    """Format a device as a context line with area as the primary identifier.

    Shows: "Area — Name: entity_id (currently state)" so the LLM sees the
    area first, avoiding mix-ups between devices with identical names.
    """
    entity_id = dev.get("entity_id", "")
    dev_name = dev.get("name", "")
    state = dev.get("state", "unknown")
    area = dev.get("area", "")
    if area:
        return f"- {area} — {dev_name}: {entity_id} (currently {state})"
    return f"- {dev_name}: {entity_id} (currently {state})"


def build_agent_context_summary(node_context: Dict[str, Any]) -> str:
    """
    Build a compact summary of HA capabilities (~50 tokens) for the system prompt.

    Instead of listing every device, this outputs device counts per domain
    and the floor layout, then instructs the LLM to call get_ha_entities
    to look up specific entity IDs on demand.

    Args:
        node_context: Node context dict, may contain an "agents" key with
            Home Assistant device data.

    Returns:
        Compact summary string, or empty string if no agent data.
    """
    agents_data = node_context.get("agents", {})
    if not agents_data:
        return ""

    ha_data = agents_data.get("home_assistant", {})
    if not ha_data:
        return ""

    # Count devices per domain
    light_controls: Dict[str, Any] = ha_data.get("light_controls", {})
    device_controls: Dict[str, Any] = ha_data.get("device_controls", {})

    # Lights: room groups + individual (deduplicated)
    room_group_ids: set[str] = {
        info.get("entity_id") for info in light_controls.values()
    }
    individual_lights: List[Any] = [
        d for d in device_controls.get("light", [])
        if d.get("entity_id") not in room_group_ids
        and d.get("state") != "unavailable"
    ]
    light_count: int = len(light_controls) + len(individual_lights)

    domain_counts: List[str] = []
    if light_count:
        domain_counts.append(f"{light_count} lights")

    for domain_name, label in [
        ("switch", "switches"),
        ("lock", "locks"),
        ("cover", "covers"),
        ("climate", "climate"),
        ("fan", "fans"),
        ("scene", "scenes"),
    ]:
        items = [
            d for d in device_controls.get(domain_name, [])
            if d.get("state") != "unavailable"
        ]
        if items:
            domain_counts.append(f"{len(items)} {label}")

    if not domain_counts:
        return ""

    section = f"\nHome Assistant: {', '.join(domain_counts)}"

    # Floor layout
    floors: Dict[str, List[str]] = ha_data.get("floors", {})
    if floors:
        floor_parts: List[str] = [
            f"{fname} ({', '.join(areas)})" for fname, areas in floors.items()
        ]
        section += f"\nFloors: {', '.join(floor_parts)}"

    section += (
        "\nCall control_device to control devices. Call get_device_status to check device state.\n"
    )

    return section


def build_agent_context_section(node_context: Dict[str, Any]) -> str:
    """
    Build agent context section (e.g., Home Assistant devices) from node context.

    Extracted from JarvisAdapterModel._build_agent_context_section.

    Args:
        node_context: Node context dict, may contain an "agents" key with
            Home Assistant device data.

    Returns:
        Formatted agent context string, or empty string if no agent data.
    """
    agents_data = node_context.get("agents", {})
    if not agents_data:
        return ""

    ha_data = agents_data.get("home_assistant", {})
    if not ha_data:
        return ""

    section = ""

    # Floor → area groupings (e.g., "Downstairs" → ["Living Room", "Kitchen"])
    floors = ha_data.get("floors", {})
    if floors:
        section += "\nFloor Layout (use to resolve 'upstairs'/'downstairs'/etc.):\n"
        for floor_name, areas in floors.items():
            section += f"- {floor_name}: {', '.join(areas)}\n"
        section += (
            "When user references a floor (e.g., 'turn off lights downstairs'), "
            "make a SEPARATE control_device call for EACH device in EVERY area "
            "on that floor. Do NOT pick just one device.\n"
        )

    # Light controls (room groups)
    light_controls = ha_data.get("light_controls", {})
    if light_controls:
        section += "\nAvailable Light Controls (room groups):\n"
        for name, info in light_controls.items():
            entity_id = info.get("entity_id", "")
            state = info.get("state", "unknown")
            section += f"- {name}: {entity_id} (currently {state})\n"

    # Device controls (individual devices)
    device_controls = ha_data.get("device_controls", {})
    if device_controls:
        # Individual lights (exclude room group entity_ids)
        lights = device_controls.get("light", [])
        room_group_ids = (
            {info.get("entity_id") for info in light_controls.values()}
            if light_controls
            else set()
        )
        individual_lights = [
            l for l in lights
            if l.get("entity_id") not in room_group_ids
            and l.get("state") != "unavailable"
        ]
        if individual_lights:
            section += "\nAvailable Individual Lights:\n"
            for dev in individual_lights[:20]:
                section += _format_device_line(dev) + "\n"

        # Switches
        switches = [
            s for s in device_controls.get("switch", [])
            if s.get("state") != "unavailable"
        ]
        if switches:
            section += "\nAvailable Switches (HA devices - use control_device/get_device_status):\n"
            for dev in switches[:10]:
                section += _format_device_line(dev) + "\n"

        # Locks (use lock/unlock actions, NOT turn_on/turn_off)
        locks = device_controls.get("lock", [])
        if locks:
            section += "\nAvailable Locks (use lock/unlock actions):\n"
            for dev in locks[:10]:
                section += _format_device_line(dev) + "\n"

        # Covers (use open_cover/close_cover actions)
        covers = device_controls.get("cover", [])
        if covers:
            section += "\nAvailable Covers (use open_cover/close_cover actions):\n"
            for dev in covers[:10]:
                section += _format_device_line(dev) + "\n"

        # Climate
        climate = device_controls.get("climate", [])
        if climate:
            section += "\nAvailable Climate Controls:\n"
            for dev in climate[:10]:
                section += _format_device_line(dev) + "\n"

        # Fans
        fans = device_controls.get("fan", [])
        if fans:
            section += "\nAvailable Fans:\n"
            for dev in fans[:10]:
                section += _format_device_line(dev) + "\n"

        # Scenes
        scenes = device_controls.get("scene", [])
        if scenes:
            section += "\nAvailable Scenes:\n"
            for dev in scenes[:20]:
                entity_id = dev.get("entity_id", "")
                dev_name = dev.get("name", "")
                section += f"- {dev_name}: {entity_id}\n"

    return section
