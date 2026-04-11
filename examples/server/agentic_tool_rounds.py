#!/usr/bin/env python3
"""
Example demonstrating server-side agentic loop with max_tool_rounds.

When the server has tool callbacks registered (via MCP, web search, or custom
callbacks), setting max_tool_rounds enables the agentic loop: the model calls
tools, the server executes them, feeds results back, and repeats until the
model produces a final text response or the round limit is reached.

Usage:
1. Start the server with MCP tools and max_tool_rounds:
   mistralrs serve -p 1234 --mcp-config examples/mcp-simple-config.json --max-tool-rounds 5 -m Qwen/Qwen3-4B

   Or with web search:
   mistralrs serve -p 1234 --enable-search --max-tool-rounds 5 -m Qwen/Qwen3-4B

2. Then run this script:
   python examples/server/agentic_tool_rounds.py
"""

from openai import OpenAI


def main():
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="placeholder",
    )

    print("Agentic Tool Rounds Example")
    print("===========================")
    print()
    print("The server will auto-execute tools and loop until a final answer.")
    print("max_tool_rounds can be set server-side (--max-tool-rounds CLI flag)")
    print("or per-request via extra_body. Per-request values take priority.")
    print()

    try:
        # max_tool_rounds can be set per-request (overrides server default)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": "List the files in the current directory and tell me what you find.",
                }
            ],
            tool_choice="auto",
            extra_body={"max_tool_rounds": 5},
        )

        print("Final response (after tool execution loop):")
        print("=" * 50)
        print(response.choices[0].message.content)
        print()

        if response.usage:
            print(f"Total tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Make sure the server is running with tool callbacks registered:")
        print(
            "mistralrs serve -p 1234 --mcp-config examples/mcp-simple-config.json "
            "--max-tool-rounds 5 -m Qwen/Qwen3-4B"
        )


if __name__ == "__main__":
    main()
