#!/usr/bin/env python3

import asyncio
from mistralrs import Runner, Which, ChatCompletionRequest


async def main():
    """
    Example demonstrating how to configure mistral.rs as an MCP client
    to connect to external MCP servers and use their tools automatically.
    
    This example shows how to configure MCP servers via environment
    variables or configuration files, since the Python bindings may
    expose MCP configuration differently than the Rust API.
    """
    
    # Note: MCP client configuration in Python will likely be exposed
    # through environment variables, configuration files, or builder
    # methods once the Python bindings are updated.
    
    # For now, this demonstrates the expected usage pattern:
    
    runner = Runner(
        which=Which.Plain(
            model_id="microsoft/Phi-3.5-mini-instruct",
            arch=None,
            dtype=None,
        ),
        max_seqs=10,
        no_kv_cache=False,
        throughput_logging_enabled=True,
        # Future MCP configuration might look like:
        # mcp_servers=[
        #     {
        #         "id": "filesystem_server",
        #         "name": "Filesystem MCP Server", 
        #         "source": {
        #             "type": "process",
        #             "command": "mcp-server-filesystem",
        #             "args": ["--root", "/tmp"]
        #         },
        #         "enabled": True,
        #         "tool_prefix": "fs"
        #     },
        #     {
        #         "id": "example_server",
        #         "name": "Example HTTP MCP Server",
        #         "source": {
        #             "type": "http", 
        #             "url": "http://localhost:8080/mcp",
        #             "timeout_secs": 30
        #         },
        #         "enabled": True,
        #         "tool_prefix": "example"
        #     }
        # ]
    )
    
    # Create a conversation that could use MCP tools
    request = ChatCompletionRequest(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant with access to external tools via MCP servers. "
                          "You can search the web, access filesystem operations, and use other tools "
                          "provided by connected MCP servers. Use these tools when appropriate to "
                          "help answer user questions."
            },
            {
                "role": "user", 
                "content": "Hello! Can you help me list the files in the /tmp directory and then "
                          "search for information about Rust programming language?"
            }
        ],
        max_tokens=1000,
        temperature=0.1,
        # Tools would be automatically discovered from MCP servers
        # No need to manually specify tools when using MCP client
        tool_choice="auto"  # Enable automatic tool calling
    )
    
    print("Sending chat request with MCP tool support...")
    response = await runner.send_chat_completion_request(request)
    
    print(f"\nResponse: {response.choices[0].message.content}")
    
    # Display any tool calls that were made
    if response.choices[0].message.tool_calls:
        print("\nTool calls made:")
        for tool_call in response.choices[0].message.tool_calls:
            print(f"- {tool_call.function.name}: {tool_call.function.arguments}")
    
    print(f"\nTokens used: {response.usage.total_tokens}")


if __name__ == "__main__":
    print("MCP Client Example for mistral.rs")
    print("==================================")
    print()
    print("This example demonstrates how mistral.rs can act as an MCP client")
    print("to connect to external MCP servers and automatically use their tools.")
    print()
    print("Note: Full MCP client support in Python bindings is coming soon.")
    print("This example shows the expected usage pattern.")
    print()
    
    asyncio.run(main())