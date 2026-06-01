"""
Anthropic Messages API example for the mistral.rs server.

Run the server:
    mistralrs serve -p 1234 -m Qwen/Qwen3-4B

Then run:
    python3 examples/server/anthropic_chat.py
"""

import json
import os
import urllib.request

BASE_URL = os.environ.get("MISTRALRS_BASE_URL", "http://localhost:1234")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "local")


def post(path, payload):
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


response = post(
    "/v1/messages",
    {
        "model": "default",
        "max_tokens": 256,
        "system": "You are concise.",
        "messages": [
            {
                "role": "user",
                "content": "Explain what mistral.rs is in one paragraph.",
            }
        ],
    },
)

for block in response["content"]:
    if block["type"] == "text":
        print(block["text"])
