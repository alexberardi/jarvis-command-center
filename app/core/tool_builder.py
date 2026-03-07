"""
ToolBuilder - Utility for building OpenAI-format tool definitions.

Standalone utility, not tied to any provider. Converts raw Jarvis tool
definitions into clean OpenAI function-calling format.
"""

from typing import Any, Dict, List, Optional


# Keys that are Jarvis-specific extensions (not part of OpenAI spec)
_JARVIS_EXTENSION_KEYS = {
    "allow_direct_answer",
    "is_server_tool",
    "command_name",
    "keywords",
    "examples",
    "_refinable",
}


class ToolBuilder:
    """Builds clean OpenAI-format tool definitions from Jarvis tool definitions."""

    @staticmethod
    def build(
        tools: List[Dict[str, Any]],
        include_descriptions: bool = True,
        max_description_length: Optional[int] = None,
        include_param_descriptions: bool = True,
        include_format_hints: bool = True,
        exclude_refinable: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Build clean OpenAI-format tool definitions.

        Args:
            tools: Raw tool definitions (may include Jarvis extensions).
            include_descriptions: Whether to include tool-level descriptions.
            max_description_length: Truncate descriptions to this length.
            include_param_descriptions: Whether to include "description" on
                individual parameter properties. When False, strips param
                descriptions to save tokens (keeps type, enum, required).
            include_format_hints: Whether to keep "format" keys (e.g.,
                "format": "date-time") in parameter schemas. When False,
                strips them to avoid conflicting with datekey instructions.
            exclude_refinable: Whether to strip properties marked with
                ``_refinable: true``.  When True, those params are removed
                from the schema so the primary model only handles intent
                classification; a secondary LLM call fills them in later.

        Returns:
            List of OpenAI-format tool definitions:
            [{"type": "function", "function": {"name", "description", "parameters"}}]
        """
        result: List[Dict[str, Any]] = []
        for tool in tools:
            func: Dict[str, Any] = tool.get("function", {})
            if not func or not func.get("name"):
                continue

            clean_func: Dict[str, Any] = {"name": func["name"]}

            # Description
            if include_descriptions and func.get("description"):
                desc: str = func["description"]
                if max_description_length and len(desc) > max_description_length:
                    desc = desc[:max_description_length - 3].rstrip() + "..."
                clean_func["description"] = desc

            # Parameters
            if func.get("parameters"):
                params = func["parameters"]
                if not include_param_descriptions:
                    params = ToolBuilder._strip_param_descriptions(params)
                if not include_format_hints:
                    params = ToolBuilder._strip_format_keys(params)
                if exclude_refinable:
                    params = ToolBuilder._strip_refinable_params(params)
                clean_func["parameters"] = params

            result.append({"type": "function", "function": clean_func})

        return result

    @staticmethod
    def _strip_param_descriptions(params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove 'description' from each property in a parameters schema.

        Preserves type, enum, required, default, and all structural keys.
        Works on a shallow copy to avoid mutating the original.
        """
        result: Dict[str, Any] = dict(params)
        properties: Dict[str, Any] | None = params.get("properties")
        if not properties:
            return result
        clean_props: Dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                clean_props[prop_name] = {
                    k: v for k, v in prop_schema.items() if k != "description"
                }
            else:
                clean_props[prop_name] = prop_schema
        result["properties"] = clean_props
        return result

    @staticmethod
    def _strip_refinable_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove properties marked ``_refinable`` and drop them from ``required``.

        Used by the compressed provider so the primary model only handles
        intent classification without complex enum parameters.
        """
        result: Dict[str, Any] = dict(params)
        properties: Dict[str, Any] | None = params.get("properties")
        if not properties:
            return result
        clean_props: Dict[str, Any] = {}
        removed: set[str] = set()
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict) and prop_schema.get("_refinable"):
                removed.add(prop_name)
            else:
                clean_props[prop_name] = prop_schema
        result["properties"] = clean_props
        if "required" in result and removed:
            result["required"] = [r for r in result["required"] if r not in removed]
        return result

    @staticmethod
    def _strip_format_keys(params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove 'format' from each property (and array items) in a parameters schema.

        Prevents 'format: date-time' from conflicting with datekey instructions.
        Works on a shallow copy to avoid mutating the original.
        """
        result: Dict[str, Any] = dict(params)
        properties: Dict[str, Any] | None = params.get("properties")
        if not properties:
            return result
        clean_props: Dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                cleaned: Dict[str, Any] = {
                    k: v for k, v in prop_schema.items() if k != "format"
                }
                # Also strip format from array items
                if "items" in cleaned and isinstance(cleaned["items"], dict):
                    cleaned["items"] = {
                        k: v for k, v in cleaned["items"].items() if k != "format"
                    }
                clean_props[prop_name] = cleaned
            else:
                clean_props[prop_name] = prop_schema
        result["properties"] = clean_props
        return result

    @staticmethod
    def strip_jarvis_extensions(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove Jarvis-specific keys, return pure OpenAI format.

        Strips top-level keys like allow_direct_answer, is_server_tool, etc.
        that are Jarvis conventions but not part of the OpenAI tool spec.
        Also strips ``_refinable`` markers from parameter property schemas.

        Args:
            tools: Tool definitions potentially containing Jarvis extensions.

        Returns:
            Clean OpenAI-format tool definitions.
        """
        result: List[Dict[str, Any]] = []
        for tool in tools:
            clean: Dict[str, Any] = {}
            for key, value in tool.items():
                if key not in _JARVIS_EXTENSION_KEYS:
                    clean[key] = value
            # Strip _refinable from parameter property schemas
            func = clean.get("function", {})
            properties = func.get("parameters", {}).get("properties", {})
            if properties:
                for prop_schema in properties.values():
                    if isinstance(prop_schema, dict):
                        prop_schema.pop("_refinable", None)
            result.append(clean)
        return result
