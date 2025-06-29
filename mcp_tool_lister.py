import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

try:
    import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("Required packages not installed. Install with:")
    print("pip install mcp httpx")
    sys.exit(1)


class MCPToolLister:
    def __init__(self):
        self.client_session: Optional[ClientSession] = None
    
    async def connect_stdio_server(self, command: str, args: List[str] = None) -> bool:
        """Connect to an MCP server via stdio"""
        try:
            if args is None:
                args = []
            
            server_params = StdioServerParameters(
                command=command,
                args=args
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.client_session = session
                    await self._initialize_session()
                    await self._list_and_print_tools()
                    return True
                    
        except Exception as e:
            print(f"Error connecting to stdio server: {e}")
            return False
    
    async def connect_http_server(self, base_url: str) -> bool:
        """Connect to an MCP server via HTTP"""
        try:
            async with httpx.AsyncClient() as http_client:
                # This is a simplified HTTP connection example
                # Actual HTTP MCP implementation may vary
                response = await http_client.post(
                    f"{base_url}/initialize",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "mcp-tool-lister",
                                "version": "1.0.0"
                            }
                        }
                    }
                )
                
                if response.status_code == 200:
                    await self._list_tools_http(http_client, base_url)
                    return True
                else:
                    print(f"Failed to connect: HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"Error connecting to HTTP server: {e}")
            return False
    
    async def _initialize_session(self):
        """Initialize the MCP session"""
        try:
            # Initialize the session with server capabilities
            await self.client_session.initialize()
            print("✅ Successfully connected to MCP server")
        except Exception as e:
            print(f"❌ Failed to initialize session: {e}")
            raise
    
    async def _list_and_print_tools(self):
        """List and print all available tools"""
        try:
            # Get the list of available tools
            tools_result = await self.client_session.list_tools()
            
            if hasattr(tools_result, 'tools') and tools_result.tools:
                print(f"\n📋 Found {len(tools_result.tools)} tools:")
                print("=" * 50)
                
                for i, tool in enumerate(tools_result.tools, 1):
                    print(f"{i}. {tool.name}")
                    if hasattr(tool, 'description') and tool.description:
                        print(f"   Description: {tool.description}")
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        print(f"   Input Schema: {json.dumps(tool.inputSchema, indent=2)}")
                    print("-" * 30)
            else:
                print("📭 No tools found on this MCP server")
                
        except Exception as e:
            print(f"❌ Error listing tools: {e}")
    
    async def _list_tools_http(self, client: httpx.AsyncClient, base_url: str):
        """List tools via HTTP connection"""
        try:
            response = await client.post(
                f"{base_url}/tools/list",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'tools' in data['result']:
                    tools = data['result']['tools']
                    print(f"\n📋 Found {len(tools)} tools:")
                    print("=" * 50)
                    
                    for i, tool in enumerate(tools, 1):
                        print(f"{i}. {tool.get('name', 'Unknown')}")
                        if 'description' in tool:
                            print(f"   Description: {tool['description']}")
                        print("-" * 30)
                else:
                    print("📭 No tools found on this MCP server")
            else:
                print(f"❌ Failed to list tools: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error listing tools via HTTP: {e}")


async def main():
    """Main function to demonstrate usage"""
    lister = MCPToolLister()
    
    print("MCP Tool Lister")
    print("===============")
    
    # Example 1: Connect to a stdio-based MCP server
    print("\n1. Connecting to stdio-based MCP server...")
    print("   Example: python -m your_mcp_server")
    
    # Uncomment and modify the following line to connect to your actual MCP server
    # success = await lister.connect_stdio_server("python", ["-m", "your_mcp_server"])

    success = await lister.connect_stdio_server("node", ["D:\\mcp_workspace\\custom\\click-up_mcp_servers.js"])
    
    # Example 2: Connect to an HTTP-based MCP server
    print("\n2. Connecting to HTTP-based MCP server...")
    print("   Example: http://localhost:8000")
    
    # Uncomment and modify the following line to connect to your HTTP MCP server
    # success = await lister.connect_http_server("http://localhost:8000")
    
    # For demonstration, let's show how to use it:
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    print("\n1. For stdio-based MCP server:")
    print("   await lister.connect_stdio_server('python', ['-m', 'my_mcp_server'])")
    print("\n2. For HTTP-based MCP server:")
    print("   await lister.connect_http_server('http://localhost:8000')")
    print("\n3. For executable MCP server:")
    print("   await lister.connect_stdio_server('./my_mcp_server')")
    
    print("\n" + "="*60)
    print("NOTE: Modify the connection parameters above to match your MCP server setup")


if __name__ == "__main__":
    asyncio.run(main())