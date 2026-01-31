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

    def _compact_tool_result(self, tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result

        if tool_name == "resolve_relative_date":
            term = result.get("term")
            if result.get("error"):
                return {
                    "term": term,
                    "error": result.get("error"),
                    "message": result.get("message", "Failed to resolve relative date")
                }
            if "utc_start_of_day" in result:
                return {"term": term, "utc_start_of_day": result.get("utc_start_of_day")}
            if "dates" in result and isinstance(result.get("dates"), list):
                dates = [
                    day.get("utc_start_of_day")
                    for day in result.get("dates", [])
                    if isinstance(day, dict) and day.get("utc_start_of_day")
                ]
                return {"term": term, "utc_start_of_day": dates}
            if "datetime" in result:
                return {"term": term, "datetime": result.get("datetime")}
            if result.get("candidates"):
                return {
                    "term": term,
                    "error": "unresolved_relative_date",
                    "message": result.get("message", "Could not resolve term")
                }

        return result
    
    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        conversation_id: Optional[str] = None,
        user_utterance: Optional[str] = None
    ) -> str:
        """
        Execute a single tool and return the result as a JSON string.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            conversation_id: Optional conversation ID to pass to tool
            user_utterance: Optional user's spoken message for context

        Returns:
            JSON string with tool execution result
        """
        logger.info(f"⚙️ Executing tool: {tool_name}")
        logger.debug(f"Tool arguments: {arguments}")

        # Build kwargs to pass to tool
        kwargs = {**arguments}
        if conversation_id:
            kwargs["conversation_id"] = conversation_id
        if user_utterance:
            kwargs["user_utterance"] = user_utterance

        # Execute via registry
        result = self.registry.execute_tool(tool_name, **kwargs)

        # Compact large server tool outputs before sending to the LLM
        result = self._compact_tool_result(tool_name, result)
        
        # Check if this is a validation request
        if isinstance(result, dict) and result.get("_validation_request"):
            logger.info(f"🔍 Tool returned validation request")
        elif "error" not in result:
            logger.info(f"✅ Tool executed successfully: {tool_name}")
        
        return json.dumps(result)
    
    def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
        user_utterance: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute multiple tool calls and separate server vs client tools.

        Args:
            tool_calls: List of tool calls from LLM response
            conversation_id: Optional conversation ID to pass to tools
            user_utterance: Optional user's spoken message for context

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
                result = self.execute_tool(
                    tool_name, arguments, conversation_id=conversation_id, user_utterance=user_utterance
                )
                
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

