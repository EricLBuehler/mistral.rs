#!/usr/bin/env python3
"""
Example demonstrating tool dispatch URL with mistral.rs server-side agentic loop.

The server POSTs tool calls to your endpoint, executes them, and feeds results
back to the model automatically.

Usage:
1. Start the mistral.rs server with tool dispatch URL:
   mistralrs serve -p 1234 --tool-dispatch-url http://localhost:8787/tools --max-tool-rounds 5 -m Qwen/Qwen3-4B

2. Then run this script (it starts a local tool server and sends a chat request):
   python examples/server/tool_dispatch.py
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from openai import OpenAI

TOOL_SERVER_PORT = 8787


# --- Tool implementations ---

def get_weather(city: str, units: str = "celsius") -> str:
    """Simulated weather lookup."""
    weather_data = {
        "tokyo": {"temp": 22, "condition": "Sunny"},
        "london": {"temp": 15, "condition": "Cloudy"},
        "new york": {"temp": 28, "condition": "Partly cloudy"},
    }
    data = weather_data.get(city.lower(), {"temp": 20, "condition": "Unknown"})
    return f"{data['condition']}, {data['temp']}{'C' if units == 'celsius' else 'F'}"


TOOLS = {
    "get_weather": lambda args: get_weather(
        args.get("city", ""), args.get("units", "celsius")
    ),
}


# --- Lightweight HTTP tool server ---

class ToolHandler(BaseHTTPRequestHandler):
    """Handles POST requests from mistral.rs tool dispatch.

    Receives: {"name": "tool_name", "arguments": {...}}
    Returns:  {"content": "result string"}
    """

    def do_POST(self, *_args):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        name = body.get("name", "")
        arguments = body.get("arguments", {})

        print(f"  Tool call: {name}({json.dumps(arguments)})")

        if name in TOOLS:
            result = TOOLS[name](arguments)
        else:
            result = f"Unknown tool: {name}"

        response = json.dumps({"content": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logs


def start_tool_server():
    server = HTTPServer(("localhost", TOOL_SERVER_PORT), ToolHandler)
    server.serve_forever()


# --- Main ---

def main():
    # Start the tool dispatch server in a background thread
    thread = threading.Thread(target=start_tool_server, daemon=True)
    thread.start()
    print(f"Tool server listening on http://localhost:{TOOL_SERVER_PORT}/tools")
    print()

    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="placeholder",
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units",
                        },
                    },
                    "required": ["city"],
                },
                "strict": True,
            },
        }
    ]

    print("Sending chat request (server will auto-execute tools via dispatch URL)...")
    print()

    try:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo and London?"}
            ],
            tools=tools,
            tool_choice="auto",
            # max_tool_rounds can also be set per-request via extra_body
            # (if not already configured server-side with --max-tool-rounds)
            extra_body={"max_tool_rounds": 5},
        )

        print("Final response:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Make sure the server is running with tool dispatch URL:")
        print(
            f"mistralrs serve -p 1234 --tool-dispatch-url http://localhost:{TOOL_SERVER_PORT}/tools "
            "--max-tool-rounds 5 -m Qwen/Qwen3-4B"
        )


if __name__ == "__main__":
    main()
