"""
Tool Registry for Server-Side Tools.

This module dynamically discovers and manages server-side tools that can be exposed to the LLM.
Tools are automatically loaded from app/core/tools/ and app/core/tools/custom/.
"""

import os
import logging
import importlib
import inspect
import pkgutil
from typing import Dict, Any, List, Optional

from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class ToolRegistry:
    """
    Registry for server-side tools available to the LLM.
    
    Automatically discovers and registers tools by scanning:
    - app/core/tools/ (built-in tools)
    - app/core/tools/custom/ (custom tools)
    
    Tools must implement the IServerTool interface.
    """
    
    def __init__(self):
        self._tools: Dict[str, IServerTool] = {}
        self._discover_and_register_tools()
    
    def _discover_and_register_tools(self):
        """
        Discover and register all tools implementing IServerTool.
        
        Scans:
        - app/core/tools/
        - app/core/tools/custom/
        """
        base_paths = [
            os.path.join(os.path.dirname(__file__), "tools"),
            os.path.join(os.path.dirname(__file__), "tools", "custom")
        ]
        
        prefix = "app.core.tools."
        
        for path in base_paths:
            if not os.path.exists(path):
                logger.debug(f"Tool path does not exist: {path}")
                continue
            
            for finder, module_name, ispkg in pkgutil.walk_packages(path=[path], prefix=prefix):
                try:
                    imported_module = importlib.import_module(module_name)
                except Exception as e:
                    logger.warning(f"Failed to import tool module {module_name}: {e}")
                    continue
                
                # Look for classes that implement IServerTool
                for class_name, cls in inspect.getmembers(imported_module, inspect.isclass):
                    if issubclass(cls, IServerTool) and cls is not IServerTool:
                        try:
                            # Instantiate the tool
                            tool_instance = cls()
                            tool_name = tool_instance.name

                            # Skip disabled tools
                            if not tool_instance.enabled:
                                logger.info(f"⏭️  Skipping disabled tool: {tool_name} (from {cls.__name__})")
                                continue

                            # Register it
                            self._tools[tool_name] = tool_instance
                            logger.info(f"🔧 Registered tool: {tool_name} (from {cls.__name__})")
                            
                        except Exception as e:
                            logger.warning(f"Failed to instantiate tool {cls.__name__}: {e}")
                            continue
        
        logger.info(f"✅ Discovered and registered {len(self._tools)} tool(s)")
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def get_tools_for_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Get tool definitions for a specific model. Currently returns all tools."""
        return self.get_tool_definitions()
    
    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_tool(self, name: str) -> Optional[IServerTool]:
        """Get a tool instance by name."""
        return self._tools.get(name)
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        tool = self.get_tool(name)
        if not tool:
            logger.error(f"❌ Tool not found: {name}")
            return {
                "error": "tool_not_found",
                "message": f"Tool '{name}' is not registered"
            }
        
        try:
            logger.debug(f"   🔨 Executing {name}.execute(**kwargs)")
            result = tool.execute(**kwargs)
            return result
        except Exception as e:
            logger.error(f"   ❌ Tool execution failed: {e}")
            return {
                "error": "execution_error",
                "message": f"Tool execution failed: {str(e)}"
            }
    

# Global registry instance
tool_registry = ToolRegistry()

