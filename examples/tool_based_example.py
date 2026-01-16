"""
Example: Tool-Based Architecture Usage

This script demonstrates how to use the new tool-based conversation API.
It shows:
1. Starting a conversation with client tools
2. Sending voice commands
3. Handling tool execution loops
4. Processing different stop reasons
"""

import requests
import json
import uuid
from typing import Dict, Any, List


class JarvisToolClient:
    """Example client for tool-based Jarvis API."""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
    
    def start_conversation(
        self,
        conversation_id: str,
        client_tools: List[Dict[str, Any]],
        timezone: str = "America/New_York"
    ) -> Dict[str, Any]:
        """
        Start a new conversation with tool definitions.
        
        Args:
            conversation_id: Unique conversation identifier
            client_tools: List of tool definitions in OpenAI format
            timezone: User's timezone
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/api/v0/conversation/start",
            headers=self.headers,
            json={
                "conversation_id": conversation_id,
                "node_context": {"timezone": timezone},
                "client_tools": client_tools
            }
        )
        response.raise_for_status()
        return response.json()
    
    def send_command(
        self,
        command: str,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Send a voice command.
        
        Args:
            command: Voice command text
            conversation_id: Conversation ID
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/api/v0/voice/command",
            headers=self.headers,
            json={
                "voice_command": command,
                "conversation_id": conversation_id
            }
        )
        response.raise_for_status()
        return response.json()
    
    def continue_with_results(
        self,
        conversation_id: str,
        tool_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Continue conversation with tool execution results.
        
        Args:
            conversation_id: Conversation ID
            tool_results: List of tool results
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/api/v0/voice/command/continue",
            headers=self.headers,
            json={
                "conversation_id": conversation_id,
                "tool_results": tool_results
            }
        )
        response.raise_for_status()
        return response.json()
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a client-side tool.
        
        Override this method or provide tool implementations.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Example implementations
        if tool_name == "turn_on_light":
            room = arguments.get("room", "unknown")
            brightness = arguments.get("brightness", 100)
            print(f"  [TOOL] Turning on light in {room} at {brightness}% brightness")
            return {
                "success": True,
                "message": f"Light in {room} turned on at {brightness}%"
            }
        
        elif tool_name == "set_temperature":
            room = arguments.get("room", "unknown")
            temperature = arguments.get("temperature")
            print(f"  [TOOL] Setting {room} temperature to {temperature}°F")
            return {
                "success": True,
                "message": f"Temperature in {room} set to {temperature}°F"
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
    
    def process_command_with_loop(
        self,
        command: str,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Process a command with automatic tool execution loop.
        
        Args:
            command: Voice command text
            conversation_id: Conversation ID
            
        Returns:
            Final response dictionary
        """
        print(f"\n🎤 User: {command}")
        response = self.send_command(command, conversation_id)
        
        # Loop until complete
        max_iterations = 10
        iteration = 0
        
        while response.get("stop_reason") == "tool_calls" and iteration < max_iterations:
            iteration += 1
            
            # Show assistant message
            if response.get("assistant_message"):
                print(f"🤖 Assistant: {response['assistant_message']}")
            
            # Execute tools
            tool_calls = response.get("tool_calls", [])
            print(f"⚙️  Executing {len(tool_calls)} tool(s)...")
            
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                print(f"  → {tool_name}({json.dumps(arguments)})")
                result = self.execute_tool(tool_name, arguments)
                
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "output": result
                })
            
            # Continue conversation
            response = self.continue_with_results(conversation_id, tool_results)
        
        # Handle final result
        stop_reason = response.get("stop_reason")
        
        if stop_reason == "complete":
            if response.get("assistant_message"):
                print(f"🤖 Assistant: {response['assistant_message']}")
            print("✅ Complete!")
        
        elif stop_reason == "validation_required":
            validation = response.get("validation_request")
            print(f"🔍 Clarification needed: {validation['question']}")
            if validation.get("options"):
                print(f"   Options: {', '.join(validation['options'])}")
        
        else:
            print(f"⚠️  Unexpected stop reason: {stop_reason}")
        
        return response


def main():
    """Example usage."""
    
    # Initialize client
    client = JarvisToolClient(
        api_key="your-api-key-here",
        base_url="http://localhost:8000"
    )
    
    # Define client tools
    client_tools = [
        {
            "type": "function",
            "function": {
                "name": "turn_on_light",
                "description": "Turn on a light in the specified room",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "The room where the light is located",
                            "enum": ["bedroom", "living_room", "kitchen", "bathroom"]
                        },
                        "brightness": {
                            "type": "number",
                            "description": "Brightness level from 0-100",
                            "default": 100
                        }
                    },
                    "required": ["room"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_temperature",
                "description": "Set the temperature in a room",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "The room to set temperature for"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Target temperature in Fahrenheit"
                        }
                    },
                    "required": ["room", "temperature"]
                }
            }
        }
    ]
    
    # Start conversation
    conversation_id = str(uuid.uuid4())
    print(f"Starting conversation: {conversation_id[:8]}...")
    
    try:
        client.start_conversation(
            conversation_id=conversation_id,
            client_tools=client_tools,
            timezone="America/New_York"
        )
        print("✅ Conversation started!")
        
        # Example commands
        commands = [
            "Turn on the bedroom light at 50%",
            "Set the living room temperature to 72 degrees",
            "What time is it?",  # This will use server-side tool
        ]
        
        for command in commands:
            client.process_command_with_loop(command, conversation_id)
            print()
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Note: Update the API key before running
    print("=" * 60)
    print("Jarvis Tool-Based Architecture Example")
    print("=" * 60)
    main()

