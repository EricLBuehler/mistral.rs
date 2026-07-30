"""
Streaming Anthropic Messages API example for the mistral.rs server.

Run the server:
    mistralrs serve -p 1234 -m Qwen/Qwen3-4B

Then run:
    python3 examples/server/anthropic_streaming.py
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


request = urllib.request.Request(
    f"{BASE_URL}/v1/messages",
    data=json.dumps(
        {
            "model": "default",
            "max_tokens": 512,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a short haiku about local inference.",
                }
            ],
        }
    ).encode("utf-8"),
    headers={
        "content-type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
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
        elif event == "message_stop":
            print()
