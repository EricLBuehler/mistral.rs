#!/usr/bin/env python3
"""
Example demonstrating how to start a mistral.rs HTTP server with MCP client support
and then interact with it using the OpenAI API format.

This example shows how to:
1. Start the mistral.rs server with MCP configuration via JSON config file
2. Send chat requests that can automatically use MCP tools
3. Parse responses to see tool calls made to MCP servers

Usage:
1. First, start the mistral.rs server with MCP config:
   mistralrs serve -p 1234 --mcp-config examples/mcp-simple-config.json -m Qwen/Qwen3-4B

2. Then run this script:
   python examples/server/mcp_chat.py
"""

from openai import OpenAI
import json


def main():
    # Connect to the mistral.rs server
    # Note: Make sure to start the server with MCP configuration first!
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="placeholder",  # mistral.rs doesn't require a real API key
    )

    print("MCP Client HTTP Server Example")
    print("==============================")
    print()
    print(
        "This example demonstrates using mistral.rs HTTP server with MCP client support."
    )
    print("The server should be started with MCP configuration like:")
    print(
        "mistralrs serve -p 1234 --mcp-config examples/mcp-simple-config.json -m Qwen/Qwen3-4B"
    )
    print("or for more advanced configuration:")
    print(
        "mistralrs serve -p 1234 --mcp-config examples/mcp-server-config.json -m Qwen/Qwen3-4B"
    )
    print(
        "Note: Install filesystem server with: npx @modelcontextprotocol/server-filesystem . -y"
    )
    print()

    # Create a chat completion request that can trigger MCP tool usage
    messages = [
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
    ]

    print("Sending chat request to mistral.rs server with MCP support...")
    print(
        "The model will automatically use MCP tools if needed to answer the question."
    )
    print()

    try:
        # Send the chat completion request
        response = client.chat.completions.create(
            model="default",  # This will be handled by the configured model
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
            tool_choice="auto",  # Enable automatic tool calling
        )

        print("Response:")
        print("=" * 50)
        print(response.choices[0].message.content)
        print()

        # Display any tool calls that were made
        if response.choices[0].message.tool_calls:
            print("MCP tool calls made:")
            print("-" * 30)
            for tool_call in response.choices[0].message.tool_calls:
                print(f"Tool: {tool_call.function.name}")
                print(f"Arguments: {tool_call.function.arguments}")
                print()
        else:
            print("No tool calls were made for this request.")
            print(
                "(This could mean MCP servers aren't configured or the question didn't require tool usage)"
            )
            print(
                "Make sure filesystem server is installed: npx @modelcontextprotocol/server-filesystem . -y"
            )

        # Display usage information
        if hasattr(response, "usage") and response.usage:
            print(f"Tokens used: {response.usage.total_tokens}")
            if hasattr(response.usage, "completion_tokens"):
                print(f"Completion tokens: {response.usage.completion_tokens}")
            if hasattr(response.usage, "prompt_tokens"):
                print(f"Prompt tokens: {response.usage.prompt_tokens}")

    except Exception as e:
        print(f"Error making request: {e}")
        print()
        print("Make sure the mistral.rs server is running with MCP configuration:")
        print(
            "mistralrs serve -p 1234 --mcp-config examples/mcp-simple-config.json -m Qwen/Qwen3-4B"
        )
        print("or for advanced configuration:")
        print(
            "mistralrs serve -p 1234 --mcp-config examples/mcp-server-config.json -m Qwen/Qwen3-4B"
        )
        print()
        print("And that the MCP configuration file exists and is properly configured.")


if __name__ == "__main__":
    main()
