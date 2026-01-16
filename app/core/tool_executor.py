"""
Tool Executor Service.

Handles execution of server-side tools and manages tool call results.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from app.core.tool_registry import tool_registry

logger = logging.getLogger("uvicorn")


class ToolExecutor:
    """Service for executing server-side tools."""
    
    def __init__(self):
        self.registry = tool_registry
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], conversation_id: Optional[str] = None) -> str:
        """
        Execute a single tool and return the result as a JSON string.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            conversation_id: Optional conversation ID to pass to tool
            
        Returns:
            JSON string with tool execution result
        """
        logger.info(f"⚙️ Executing tool: {tool_name}")
        logger.debug(f"Tool arguments: {arguments}")
        
        # Execute via registry (pass conversation_id if provided)
        if conversation_id:
            result = self.registry.execute_tool(tool_name, conversation_id=conversation_id, **arguments)
        else:
            result = self.registry.execute_tool(tool_name, **arguments)
        
        # Check if this is a validation request
        if isinstance(result, dict) and result.get("_validation_request"):
            logger.info(f"🔍 Tool returned validation request")
        elif "error" not in result:
            logger.info(f"✅ Tool executed successfully: {tool_name}")
        
        return json.dumps(result)
    
    def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        conversation_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute multiple tool calls and separate server vs client tools.
        
        Args:
            tool_calls: List of tool calls from LLM response
            
        Returns:
            Tuple of (server_tool_results, client_tool_calls)
            - server_tool_results: List of executed server tool results (tool message format)
            - client_tool_calls: List of tool calls that need client execution
        """
        server_tool_results = []
        client_tool_calls = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            tool_call_id = tool_call.get("id")
            
            # Check if this is a server-side tool
            if self.registry.has_tool(tool_name):
                logger.info(f"🔧 [SERVER TOOL] Executing: {tool_name}")
                
                # Execute server-side tool
                arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except json.JSONDecodeError:
                    arguments = {}
                
                logger.debug(f"   └─ Arguments: {arguments}")
                result = self.execute_tool(tool_name, arguments, conversation_id=conversation_id)
                
                # Check if result contains validation request
                try:
                    result_obj = json.loads(result)
                    if result_obj.get("_validation_request"):
                        # This is a validation request, treat it specially
                        logger.info(f"   └─ 🔍 Validation request: {result_obj.get('question', 'N/A')}")
                        # For now, we'll still add it as a tool result
                        # The caller will need to check for validation requests
                    elif result_obj.get("error"):
                        logger.warning(f"   └─ ⚠️  Tool error: {result_obj.get('error')}")
                    else:
                        logger.info(f"   └─ ✅ Success")
                except json.JSONDecodeError:
                    logger.info(f"   └─ ✅ Success")
                
                # Add to server results in tool message format
                server_tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result
                })
            else:
                # This is a client-side tool
                client_tool_calls.append(tool_call)
                logger.info(f"📲 [CLIENT TOOL] Forwarding to client: {tool_name}")
        
        # Summary
        if server_tool_results:
            logger.info(f"📊 Executed {len(server_tool_results)} server tool(s)")
        if client_tool_calls:
            logger.info(f"📊 Forwarded {len(client_tool_calls)} client tool(s)")
        
        return server_tool_results, client_tool_calls
    
    def is_server_tool(self, tool_name: str) -> bool:
        """Check if a tool is server-side."""
        return self.registry.has_tool(tool_name)
    
    def get_available_server_tools(self) -> List[str]:
        """Get list of available server-side tool names."""
        return self.registry.get_tool_names()


# Global executor instance
tool_executor = ToolExecutor()

