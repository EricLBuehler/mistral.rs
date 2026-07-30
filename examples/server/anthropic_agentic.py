"""
Anthropic-style server tools mapped to mistral.rs agentic features.

Run the server with search and code execution enabled:
    mistralrs serve --agent -p 1234 -m Qwen/Qwen3-4B

Then run:
    python3 examples/server/anthropic_agentic.py
"""

import json
import os
import urllib.request

BASE_URL = os.environ.get("MISTRALRS_BASE_URL", "http://localhost:1234")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "local")


def sse_events(response):
    event = None
    data = []
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line:
            if event and data:
                yield event, json.loads("".join(data))
            event = None
            data = []
        elif line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").strip())


payload = {
    "model": "default",
    "max_tokens": 1024,
    "stream": True,
    "session_id": "anthropic-agentic-demo",
    "agent_permission": "auto",
    "max_tool_rounds": 6,
    "tools": [
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "code_execution_20250825", "name": "code_execution"},
    ],
    "messages": [
        {
            "role": "user",
            "content": (
                "Search for the current Rust stable version, then use Python "
                "to print a one-row markdown table with the version and today's date."
            ),
        }
    ],
}

request = urllib.request.Request(
    f"{BASE_URL}/v1/messages",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "content-type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "code-execution-2025-08-25",
    },
    method="POST",
)

with urllib.request.urlopen(request) as response:
    for event, payload in sse_events(response):
        if event == "content_block_delta":
            delta = payload["delta"]
            if delta["type"] == "text_delta":
                print(delta["text"], end="", flush=True)
            elif delta["type"] == "thinking_delta":
                print(delta["thinking"], end="", flush=True)
        elif event == "agentic_tool_call_progress":
            phase = payload.get("phase")
            tool_name = payload.get("tool_name")
            tool_type = payload.get("data", {}).get("tool_type", "custom")
            print(f"\n[{tool_type}] {tool_name}: {phase}")
        elif event == "agentic_tool_approval_required":
            print(f"\nApproval required: {payload['approval_id']}")
        elif event == "file_produced":
            print(f"\nFile produced: {payload['id']} {payload['name']}")
        elif event == "message_stop":
            print()
