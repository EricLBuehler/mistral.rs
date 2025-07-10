#!/usr/bin/env python3
"""
Example demonstrating MCP (Model Context Protocol) client usage with mistral.rs Python API

This example shows how to:
- Configure MCP servers with different transport protocols (HTTP, WebSocket, Process)
- Set up Bearer token authentication for secure connections
- Integrate MCP tools with model tool calling

The MCP client automatically discovers tools from connected servers and makes them
available for the model to use during conversations.

Note: This is a simpler example. For full API usage, see mcp_client_example.py
"""

import asyncio
import mistralrs


async def main():
    # Create MCP client configuration using the Hugging Face MCP server
    # Note: Replace 'hf_xxx' with your actual Hugging Face token

    # Process example using defaults (enabled=True, UUID for id/prefix, no timeouts)
    filesystem_server_simple = mistralrs.McpServerConfigPy(
        name="Filesystem Tools",
        source=mistralrs.McpServerSourcePy.Process(
            command="npx",
            args=["@modelcontextprotocol/server-filesystem", "."],
            work_dir=None,
            env=None,
        ),
    )

    # Alternative HTTP example: Hugging Face MCP Server (disabled by default)
    hf_server = mistralrs.McpServerConfigPy(
        id="hf_server",
        name="Hugging Face MCP",
        source=mistralrs.McpServerSourcePy.Http(
            url="https://hf.co/mcp", timeout_secs=30, headers=None
        ),
        enabled=False,  # Disabled by default
        tool_prefix="hf",  # Prefixes tool names to avoid conflicts
        resources=None,
        bearer_token="hf_xxx",  # Replace with your actual Hugging Face token
    )

    # Alternative WebSocket example (disabled by default)
    websocket_server = mistralrs.McpServerConfigPy(
        id="websocket_server",
        name="WebSocket Example",
        source=mistralrs.McpServerSourcePy.WebSocket(
            url="wss://api.example.com/mcp", timeout_secs=30, headers=None
        ),
        enabled=False,  # Disabled by default
        tool_prefix="ws",
        resources=None,
        bearer_token="your-websocket-token",
    )

    # Simple MCP client configuration using defaults
    # (auto_register_tools=True, no timeouts, max_concurrent_calls=1)
    mcp_config_simple = mistralrs.McpClientConfigPy(servers=[filesystem_server_simple])

    # Alternative: Full MCP client configuration with multiple servers
    mcp_config_full = mistralrs.McpClientConfigPy(
        servers=[
            filesystem_server_simple,
            hf_server,
            websocket_server,
        ],  # filesystem enabled, others disabled
        auto_register_tools=True,
        tool_timeout_secs=30,
        max_concurrent_calls=5,
    )

    # Use the simple configuration for this example
    mcp_config = mcp_config_simple

    print("Building model with MCP client support...")

    # Build the model with MCP client configuration
    runner = mistralrs.Runner(
        which=mistralrs.Which.Plain(
            model_id="Qwen/Qwen3-4B",
            arch=mistralrs.Architecture.Qwen3,
        ),
        max_seqs=10,
        no_kv_cache=False,
        throughput_logging_enabled=True,
        mcp_client_config=mcp_config,
    )

    print("Model built successfully! MCP servers connected and tools registered.")

    # Create a conversation that demonstrates MCP tool usage
    request = mistralrs.ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant with access to external tools via MCP servers. "
                "You can access filesystem operations and other external services. "
                "Use these tools when appropriate to help answer user questions.",
            },
            {
                "role": "user",
                "content": "Hello! Can you list the files in the current directory and create a test.txt file?",
            },
        ],
        max_tokens=1000,
        temperature=0.1,
        tool_choice="auto",  # Enable automatic tool calling
    )

    print("\nSending chat request with MCP tool support...")
    print(
        "The model will automatically use MCP tools if needed to answer the question."
    )

    response = await runner.send_chat_completion_request(request)

    print(f"\nResponse: {response.choices[0].message.content}")

    # Display any tool calls that were made
    if (
        hasattr(response.choices[0].message, "tool_calls")
        and response.choices[0].message.tool_calls
    ):
        print("\nMCP tool calls made:")
        for tool_call in response.choices[0].message.tool_calls:
            print(
                f"- Tool: {tool_call.function.name} | Arguments: {tool_call.function.arguments}"
            )
    else:
        print("\nNo tool calls were made for this request.")

    print(f"\nTokens used: {response.usage.total_tokens}")


if __name__ == "__main__":
    print("MCP Client Example for mistral.rs")
    print("==================================")
    print()
    print("This example demonstrates how mistral.rs can act as an MCP client")
    print(
        "to connect to external MCP servers (like filesystem tools) and automatically use their tools."
    )
    print(
        "Note: Install the filesystem server with: npx @modelcontextprotocol/server-filesystem . -y"
    )
    print()

    asyncio.run(main())
