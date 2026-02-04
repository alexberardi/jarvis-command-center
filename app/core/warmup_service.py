"""
Warmup Service for Jarvis Voice Assistant.

This module provides helper functions for processing client tools,
extracting examples, filtering antipatterns, and preparing data
for conversation warmup.
"""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("uvicorn")


class WarmupService:
    """
    Service for processing warmup-related data.

    Handles:
    - Client tools processing
    - Examples extraction
    - Antipatterns filtering
    - Tool merging (server + client)
    """

    def _get_tool_name(self, tool: Dict[str, Any]) -> Optional[str]:
        """
        Extract tool name from a tool definition.

        Handles both OpenAI function format and simple format.

        Args:
            tool: Tool definition dict

        Returns:
            Tool name or None if not found
        """
        if isinstance(tool.get("function"), dict):
            return tool.get("function", {}).get("name")
        return tool.get("name")

    def build_examples_map(
        self,
        client_tools: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a map of command_name -> command info with examples.

        Args:
            client_tools: List of client tool definitions

        Returns:
            Dict mapping tool names to their examples and metadata
        """
        examples_map: Dict[str, Dict[str, Any]] = {}

        if not client_tools:
            return examples_map

        for tool in client_tools:
            tool_name = self._get_tool_name(tool)
            if not tool_name:
                continue

            examples_array = tool.get("examples", [])
            if not examples_array or not isinstance(examples_array, list):
                continue

            # Initialize command entry
            examples_map[tool_name] = {
                "command_name": tool_name,
                "examples": [],
                "keywords": tool.get("keywords"),
                "antipatterns": tool.get("antipatterns"),
                "allow_direct_answer": tool.get("allow_direct_answer")
            }

            # Add all examples from the array
            for example in examples_array:
                if isinstance(example, dict):
                    voice_command = example.get("voice_command")
                    if voice_command:
                        examples_map[tool_name]["examples"].append({
                            "voice_command": voice_command,
                            "expected_parameters": example.get("expected_parameters", {}),
                            "is_primary": example.get("is_primary", False)
                        })

        return examples_map

    def filter_antipatterns(
        self,
        antipatterns: Optional[List[Dict[str, Any]]],
        available_command_names: Set[str],
        source: str
    ) -> List[Dict[str, Any]]:
        """
        Filter antipatterns to only include valid ones.

        Args:
            antipatterns: List of antipattern dicts
            available_command_names: Set of valid command names
            source: Source identifier for logging

        Returns:
            Filtered list of antipatterns
        """
        if not antipatterns:
            return []

        filtered = []
        dropped = 0

        for ap in antipatterns:
            if not isinstance(ap, dict):
                dropped += 1
                continue
            cmd_name = ap.get("command_name")
            desc = ap.get("description")
            if cmd_name in available_command_names and isinstance(desc, str) and desc.strip():
                filtered.append({"command_name": cmd_name, "description": desc.strip()})
            else:
                dropped += 1

        logger.debug(
            "Antipatterns filtered (%s): kept=%d dropped=%d",
            source,
            len(filtered),
            dropped
        )
        return filtered

    def merge_tools(
        self,
        server_tools: List[Dict[str, Any]],
        client_tools: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Merge server and client tools.

        Args:
            server_tools: List of server tool definitions
            client_tools: Optional list of client tool definitions

        Returns:
            Combined list of all tools
        """
        return server_tools + (client_tools or [])

    def get_available_command_names(
        self,
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Any]]
    ) -> Set[str]:
        """
        Build a set of all available command names.

        Args:
            tools: List of tool definitions
            available_commands: Optional list of CommandDefinition objects

        Returns:
            Set of command names
        """
        names = set()

        for tool in tools:
            name = self._get_tool_name(tool)
            if name:
                names.add(name)

        if available_commands:
            for cmd in available_commands:
                if hasattr(cmd, "command_name"):
                    names.add(cmd.command_name)

        return names

    def process_client_tools(
        self,
        client_tools: List[Dict[str, Any]],
        available_command_names: Set[str]
    ) -> None:
        """
        Process client tools in place, filtering antipatterns.

        Args:
            client_tools: List of client tool definitions (modified in place)
            available_command_names: Set of valid command names
        """
        for i, tool in enumerate(client_tools):
            tool_name = self._get_tool_name(tool) or str(i)
            tool["antipatterns"] = self.filter_antipatterns(
                tool.get("antipatterns"),
                available_command_names,
                f"client_tool:{tool_name}"
            )

    def merge_available_commands(
        self,
        examples_map: Dict[str, Dict[str, Any]],
        available_commands: Optional[List[Any]],
        available_command_names: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Merge available_commands into examples map and return list for caching.

        Args:
            examples_map: Dict of command_name -> command info
            available_commands: Optional list of CommandDefinition objects
            available_command_names: Set of valid command names

        Returns:
            List of command dicts for caching
        """
        result = list(examples_map.values())

        if not available_commands:
            return result

        for cmd in available_commands:
            cmd_dict = cmd.model_dump()
            cmd_dict["antipatterns"] = self.filter_antipatterns(
                cmd_dict.get("antipatterns"),
                available_command_names,
                f"available_command:{cmd_dict.get('command_name')}"
            )

            cmd_name = cmd_dict.get("command_name")
            if cmd_name in examples_map:
                # Merge examples into existing entry
                examples_map[cmd_name]["examples"].extend(cmd_dict.get("examples", []))
            else:
                result.append(cmd_dict)

        return result


# Create singleton instance
warmup_service = WarmupService()
