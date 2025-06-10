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
    
    # Primary example: Hugging Face MCP server (HTTP)
    hf_server = mistralrs.McpServerConfigPy(
        id="hf_server",
        name="Hugging Face MCP",
        source=mistralrs.McpServerSourcePy.Http(
            url="https://hf.co/mcp",
            timeout_secs=30,
            headers=None
        ),
        enabled=True,
        tool_prefix="hf",  # Prefixes tool names to avoid conflicts
        resources=None,
        bearer_token="hf_xxx"  # Replace with your actual Hugging Face token
    )
    
    # Additional examples (commented out for demonstration):
    
    # # Example: Process-based MCP server (local filesystem tools)
    # filesystem_server = mistralrs.McpServerConfigPy(
    #     id="filesystem_server",
    #     name="Filesystem MCP Server",
    #     source=mistralrs.McpServerSourcePy.Process(
    #         command="mcp-server-filesystem",
    #         args=["--root", "/tmp"],
    #         work_dir=None,
    #         env=None
    #     ),
    #     enabled=False,  # Disabled for this example
    #     tool_prefix="fs",
    #     resources=["file://**"],
    #     bearer_token=None
    # )
    
    # # Example: WebSocket-based MCP server (real-time data)
    # websocket_server = mistralrs.McpServerConfigPy(
    #     id="websocket_server",
    #     name="WebSocket MCP Server",
    #     source=mistralrs.McpServerSourcePy.WebSocket(
    #         url="wss://api.example.com/mcp",
    #         timeout_secs=30,
    #         headers=None
    #     ),
    #     enabled=False,  # Disabled for this example
    #     tool_prefix="ws",
    #     resources=None,
    #     bearer_token="your-websocket-token"
    # )
    
    # Create MCP client configuration
    mcp_config = mistralrs.McpClientConfigPy(
        servers=[hf_server],  # Add filesystem_server, websocket_server if enabled
        auto_register_tools=True,
        tool_timeout_secs=30,
        max_concurrent_calls=5
    )

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
        mcp_client_config=mcp_config
    )

    print("Model built successfully! MCP servers connected and tools registered.")

    # Create a conversation that demonstrates MCP tool usage
    request = mistralrs.ChatCompletionRequest(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant with access to external tools via MCP servers. "
                "You can access Hugging Face tools and other external services. "
                "Use these tools when appropriate to help answer user questions.",
            },
            {
                "role": "user",
                "content": "Hello! Can you help me get the top 10 HF models right now?",
            },
        ],
        max_tokens=1000,
        temperature=0.1,
        tool_choice="auto",  # Enable automatic tool calling
    )

    print("\nSending chat request with MCP tool support...")
    print("The model will automatically use MCP tools if needed to answer the question.")
    
    response = await runner.send_chat_completion_request(request)

    print(f"\nResponse: {response.choices[0].message.content}")

    # Display any tool calls that were made
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        print("\nMCP tool calls made:")
        for tool_call in response.choices[0].message.tool_calls:
            print(f"- Tool: {tool_call.function.name} | Arguments: {tool_call.function.arguments}")
    else:
        print("\nNo tool calls were made for this request.")

    print(f"\nTokens used: {response.usage.total_tokens}")


if __name__ == "__main__":
    print("MCP Client Example for mistral.rs")
    print("==================================")
    print()
    print("This example demonstrates how mistral.rs can act as an MCP client")
    print("to connect to external MCP servers (like Hugging Face) and automatically use their tools.")
    print()

    asyncio.run(main())
