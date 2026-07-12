"""
Shared context-building utilities for prompt providers.

Extracted from JarvisToolModel._build_system_prompt and
JarvisAdapterModel._build_agent_context_section to avoid duplication
across prompt providers.
"""

from typing import Any, Dict, List


def device_agent_data(agents_data: Dict[str, Any]) -> Dict[str, Any]:
    """Device context published by whichever device agent is active.

    The Home Assistant Pantry agent publishes under ``home_assistant``; the
    built-in agent publishes under ``device_agent`` (renamed off ``home_assistant``
    so a core agent isn't named after a third party). Both can be present at once
    (different agent names → no override), so the HA integration takes precedence
    when installed and the built-in is the fallback. HA-present preserves the
    original ``get("home_assistant", {})`` behavior exactly (including empty).
    """
    if "home_assistant" in agents_data:
        return agents_data.get("home_assistant") or {}
    return agents_data.get("device_agent") or {}


def build_tool_guidance_section(
    tools: List[Dict[str, Any]],
) -> str:
    """
    Aggregate per-tool ``included_system_prompt_text`` into a guidance section.

    Each server tool may declare routing guidance that should live in the
    system prompt rather than its description (e.g., "MUST call X when
    user asks about current events"). This pulls those strings out and
    renders them as a bulleted "Tool Guidance:" section.

    Args:
        tools: Tool definitions (post ``to_openai_format``). May contain
            an ``included_system_prompt_text`` top-level key.

    Returns:
        Formatted section string, or empty string if no tools declare guidance.
    """
    hints: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        text = tool.get("included_system_prompt_text")
        if not text:
            continue
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        hints.append(text)

    if not hints:
        return ""

    return "\nTool Guidance:\n" + "\n".join(f"- {h}" for h in hints) + "\n"


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
        # SDK default for allow_direct_answer is False. When upstream
        # merge/filter layers lose track of the flag (None), treat as False
        # so the command still ends up in MUST-call — matching the SDK
        # default rather than silently dropping the command from the policy.
        if allow_direct is True:
            direct_answer_allowed.append(cmd_name)
        else:
            must_call_tools.append(cmd_name)

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

    ha_data = device_agent_data(agents_data)
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
        ("media_player", "media players"),
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

    # Count total devices
    total_devices: int = sum(
        len([d for d in device_controls.get(dom, []) if d.get("state") != "unavailable"])
        for dom in device_controls
    ) + len(light_controls)

    # Small device lists (≤ 20): show full room-grouped list so the LLM
    # can resolve names and entity_ids directly — no get_ha_entities needed.
    if total_devices <= 20:
        return build_agent_context_by_room(node_context)

    # Large device lists: compact summary + instruct LLM to call get_ha_entities
    section = f"\nHome Assistant: {', '.join(domain_counts)}"

    # Floor layout
    floors: Dict[str, List[str]] = ha_data.get("floors", {})
    if floors:
        floor_parts: List[str] = [
            f"{fname} ({', '.join(areas)})" for fname, areas in floors.items()
        ]
        section += f"\nFloors: {', '.join(floor_parts)}"

    room = node_context.get("room", "")
    section += "\nCall control_device to control devices. Call get_device_status to check device state."
    if room:
        section += (
            f"\nWhen the user doesn't specify a room, prefer devices in '{room}' (the current room)."
        )
    section += "\n"

    # Append room hierarchy if available
    section += build_room_hierarchy_section(node_context)

    return section


def build_agent_context_by_room(node_context: Dict[str, Any]) -> str:
    """
    Build HA context grouped by room/area so the LLM can match voice
    commands like "turn on the play room lights" to the correct entity.

    Output format:
        Home Assistant Devices:
        Office:
        - light.office: Office Lights (on)
        - switch.office_fan: Fan (off)
        Play Room:
        - light.play_room: Play Room Lights (off)
        ...

    Args:
        node_context: Node context dict with "agents" → "home_assistant" data.

    Returns:
        Formatted string grouped by room, or empty string if no data.
    """
    agents_data = node_context.get("agents", {})
    if not agents_data:
        return ""

    ha_data = device_agent_data(agents_data)
    if not ha_data:
        return ""

    # Collect all entities into room → list[entity_info]
    room_entities: Dict[str, List[Dict[str, Any]]] = {}

    # Light controls (room groups — e.g., Hue rooms)
    light_controls: Dict[str, Any] = ha_data.get("light_controls", {})
    room_group_ids: set[str] = set()
    for name, info in light_controls.items():
        entity_id = info.get("entity_id", "")
        state = info.get("state", "unknown")
        room_group_ids.add(entity_id)
        # Use the friendly name as a rough room guess if no area
        room_entities.setdefault("Room Groups", []).append({
            "entity_id": entity_id,
            "name": name,
            "state": state,
        })

    # Device controls (individual devices grouped by domain)
    device_controls: Dict[str, List[Dict[str, Any]]] = ha_data.get("device_controls", {})
    for devices in device_controls.values():
        for dev in devices:
            entity_id = dev.get("entity_id", "")
            if dev.get("state") == "unavailable":
                continue
            # Skip room group duplicates
            if entity_id in room_group_ids:
                continue
            area = dev.get("area", "Unassigned")
            room_entities.setdefault(area, []).append({
                "entity_id": entity_id,
                "name": dev.get("name", ""),
                "state": dev.get("state", "unknown"),
            })

    if not room_entities:
        return ""

    # Floor layout for floor-level commands
    floors: Dict[str, List[str]] = ha_data.get("floors", {})
    section = "\nHome Assistant Devices:\n"

    if floors:
        floor_parts: List[str] = [
            f"{fname} ({', '.join(areas)})" for fname, areas in floors.items()
        ]
        section += f"Floors: {', '.join(floor_parts)}\n"
        section += (
            "Floor commands (e.g., 'turn off lights downstairs') → "
            "call control_device for EACH device in EVERY area on that floor.\n"
        )

    # Room groups first (if any)
    if "Room Groups" in room_entities:
        section += "\nRoom Groups (control all lights in a room):\n"
        for dev in room_entities.pop("Room Groups"):
            section += f"- {dev['entity_id']}: {dev['name']} ({dev['state']})\n"

    # Remaining rooms sorted alphabetically, Unassigned last
    sorted_rooms = sorted(
        room_entities.keys(),
        key=lambda r: (r == "Unassigned", r),
    )
    for room_name in sorted_rooms:
        devices = room_entities[room_name]
        section += f"\n{room_name}:\n"
        for dev in devices:
            section += f"- {dev['entity_id']}: {dev['name']} ({dev['state']})\n"

    room = node_context.get("room", "")
    section += "\nUse control_device to control devices. "
    section += "Use get_device_status to check state. "
    section += "Copy entity_id EXACTLY as shown above."
    if room:
        section += (
            f"\nWhen the user doesn't specify a room, prefer devices in '{room}' (the current room)."
        )
    section += "\n"

    return section


def build_room_hierarchy_section(node_context: Dict[str, Any]) -> str:
    """
    Build a room hierarchy section for the system prompt so the LLM knows
    which rooms contain sub-rooms.

    Output format:
        Room Hierarchy:
        Upstairs → Bedroom 1, Bedroom 2, Hallway
        Bedroom 1 → Bedroom Bathroom 1
        When user references a room with sub-rooms, target ALL devices in that room and its sub-rooms.

    Args:
        node_context: Node context dict, may contain a "room_hierarchy" key.

    Returns:
        Formatted hierarchy string, or empty string if no hierarchy data.
    """
    hierarchy: List[Dict[str, Any]] = node_context.get("room_hierarchy", [])
    if not hierarchy:
        return ""

    # Build parent → children mapping
    parent_children: Dict[str, List[str]] = {}
    name_map: Dict[str, str] = {}  # id → name
    for room in hierarchy:
        name_map[room["id"]] = room["name"]

    for room in hierarchy:
        parent_id = room.get("parent_room_id")
        if parent_id and parent_id in name_map:
            parent_children.setdefault(parent_id, []).append(room["name"])

    if not parent_children:
        return ""

    section = "\nRoom Hierarchy:\n"
    for parent_id, children in parent_children.items():
        parent_name = name_map[parent_id]
        section += f"{parent_name} → {', '.join(children)}\n"
    section += (
        "When user references a room with sub-rooms, target ALL devices "
        "in that room AND its sub-rooms.\n"
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

    ha_data = device_agent_data(agents_data)
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

        # Media players (TVs, speakers, etc.)
        media_players = [
            m for m in device_controls.get("media_player", [])
            if m.get("state") != "unavailable"
        ]
        if media_players:
            section += "\nAvailable Media Players (use turn_on/turn_off/play/pause/stop actions):\n"
            for dev in media_players[:10]:
                section += _format_device_line(dev) + "\n"

        # Scenes
        scenes = device_controls.get("scene", [])
        if scenes:
            section += "\nAvailable Scenes:\n"
            for dev in scenes[:20]:
                entity_id = dev.get("entity_id", "")
                dev_name = dev.get("name", "")
                section += f"- {dev_name}: {entity_id}\n"

    room = node_context.get("room", "")
    if room and section:
        section += (
            f"\nWhen the user doesn't specify a room, prefer devices in '{room}' (the current room).\n"
        )

    return section
