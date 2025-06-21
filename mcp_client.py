import json
import requests
from typing import Dict, Any, Optional

class MCPClient:
    """Client for communicating with the MCP (Model Context Protocol) server."""
    
    def __init__(self, server_url: str = "http://localhost:3000"):
        """
        Initialize the MCP client.
        
        Args:
            server_url: URL of the MCP server (default: http://localhost:3000)
        """
        self.server_url = server_url
        self.base_url = f"{server_url}/tools"
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool
            
        Returns:
            Response from the tool execution
        """
        payload = {
            "tool_name": tool_name,
            "parameters": parameters
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/{tool_name}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to call tool {tool_name}: {str(e)}")
    
    def send_code_to_revit(self, code: str, parameters: Optional[list] = None) -> Dict[str, Any]:
        """
        Send C# code to Revit for execution.
        
        Args:
            code: The C# code to execute in Revit
            parameters: Optional parameters to pass to the code
            
        Returns:
            Response from the tool execution
        """
        return self.call_tool("send_code_to_revit", {
            "code": code,
            "parameters": parameters or []
        })

# Example usage function
def execute_revit_code(code: str, parameters: Optional[list] = None) -> str:
    """
    Execute C# code in Revit using the MCP server.
    
    Args:
        code: The C# code to execute
        parameters: Optional parameters
        
    Returns:
        Result message from the execution
    """
    try:
        client = MCPClient()
        result = client.send_code_to_revit(code, parameters)
        
        # Extract the text content from the response
        if "content" in result and len(result["content"]) > 0:
            return result["content"][0]["text"]
        else:
            return str(result)
            
    except Exception as e:
        return f"Error executing code: {str(e)}" 