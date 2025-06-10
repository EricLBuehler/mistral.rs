#!/usr/bin/env python3
"""
Example demonstrating MCP (Model Context Protocol) client usage with mistral.rs Python API

This example shows how to:
- Configure MCP servers with different transport protocols (HTTP, WebSocket, Process)
- Set up Bearer token authentication for secure connections
- Integrate MCP tools with model tool calling

The MCP client automatically discovers tools from connected servers and makes them
available for the model to use during conversations.
"""

import mistralrs

def main():
    # Create MCP client configuration with example servers
    # This configuration demonstrates different transport types and authentication methods
    
    # Example: HTTP-based MCP server with Bearer token authentication
    http_server = mistralrs.McpServerConfigPy(
        id="example_server",
        name="Hugging Face MCP",
        source=mistralrs.McpServerSourcePy.Http(
            url="https://hf.co/mcp",
            timeout_secs=30,
            headers=None  # Additional headers can be specified here if needed
        ),
        enabled=True,
        tool_prefix="example",  # Prefixes tool names to avoid conflicts
        resources=None,
        bearer_token="hf_xxx"  # Replace with your actual Hugging Face token
    )
    
    # Example: Process-based MCP server (commented out for demonstration)
    # process_server = mistralrs.McpServerConfigPy(
    #     id="filesystem_server",
    #     name="Filesystem MCP Server",
    #     source=mistralrs.McpServerSourcePy.Process(
    #         command="mcp-server-filesystem",
    #         args=["--root", "/tmp"],
    #         work_dir=None,
    #         env=None
    #     ),
    #     enabled=True,
    #     tool_prefix="fs",
    #     resources=["file://**"],
    #     bearer_token=None  # Process servers typically don't need Bearer tokens
    # )
    
    # Example: WebSocket-based MCP server (commented out for demonstration)
    # websocket_server = mistralrs.McpServerConfigPy(
    #     id="websocket_server",
    #     name="WebSocket MCP Server",
    #     source=mistralrs.McpServerSourcePy.WebSocket(
    #         url="wss://api.example.com/mcp",
    #         timeout_secs=30,
    #         headers=None
    #     ),
    #     enabled=False,  # Disabled for example - change to True to use
    #     tool_prefix="ws",
    #     resources=None,
    #     bearer_token="your-websocket-token"  # WebSocket Bearer token support
    # )
    
    # Create MCP client configuration
    mcp_config = mistralrs.McpClientConfigPy(
        servers=[http_server],  # Add process_server, websocket_server if enabled
        tool_timeout_secs=30,      # Timeout for individual tool calls (30 seconds)
        max_concurrent_calls=5     # Maximum concurrent tool calls across all servers
    )
    
    print("Building model with MCP client support...")
    
    # Build the model with MCP client configuration
    # The MCP client will automatically connect to configured servers and discover available tools
    runner = mistralrs.Runner(
        which=mistralrs.Which.GGUF(
            tok_model_id="../hf_models/qwen3_4b",
            quantized_model_id="../hf_models/qwen3_4b",
            quantized_filename=["model.gguf"]
        ),
        mcp_client_config=mcp_config  # This automatically connects to MCP servers and registers tools
    )
    
    print("Model built successfully! MCP servers connected and tools registered.")
    print("MCP tools are now available for automatic tool calling during conversations.")
    
    # Create a conversation that demonstrates MCP tool usage
    # The system message informs the model about available external tools
    res = runner.send_chat_completion_request(
        mistralrs.ChatCompletionRequest(
            model="mistral",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an AI assistant with access to external tools via MCP servers. 
                                 You can search the web, access filesystem operations, and use other tools 
                                 provided by connected MCP servers. Use these tools when appropriate to 
                                 help answer user questions. Tools are automatically available and you 
                                 can call them as needed."""
                },
                {
                    "role": "user",
                    "content": "Hello! Can you help me get the top 10 HF models right now?"
                }
            ],
            max_tokens=1000,
            temperature=0.7,
            stream=False
        )
    )
    
    print("\nResponse:")
    print(res.choices[0].message.content)
    
    # Display performance metrics
    print(f"\nPerformance metrics:")
    print(f"Prompt tokens/sec: {res.usage.avg_prompt_tok_per_sec:.2f}")
    print(f"Completion tokens/sec: {res.usage.avg_compl_tok_per_sec:.2f}")
    
    # Display any MCP tool calls that were made during the conversation
    if hasattr(res.choices[0].message, 'tool_calls') and res.choices[0].message.tool_calls:
        print("\nMCP tool calls made:")
        for tool_call in res.choices[0].message.tool_calls:
            print(f"- Tool: {tool_call.function.name} | Arguments: {tool_call.function.arguments}")
    else:
        print("\nNo tool calls were made for this request.")

if __name__ == "__main__":
    main()