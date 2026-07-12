"""
Basic client-side tool calling with the Anthropic Messages API.

Run the server:
    mistralrs serve -p 1234 --quant 4 -m Qwen/Qwen3-4B

Then run:
    python3 examples/server/anthropic_tool_calling.py
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


def get_weather(city):
    data = {"tokyo": "Sunny, 22C", "london": "Cloudy, 15C"}
    return data.get(city.lower(), f"Unknown city: {city}")


tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    }
]

messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

# Step 1: Model generates a tool call.
first = post(
    "/v1/messages",
    {
        "model": "default",
        "max_tokens": 256,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "auto"},
    },
)

tool_uses = [block for block in first["content"] if block["type"] == "tool_use"]
if not tool_uses:
    raise RuntimeError(f"Expected a tool_use block, got: {json.dumps(first, indent=2)}")

tool_use = tool_uses[0]
print(f"Model wants to call: {tool_use['name']}")

# Step 2: Execute the tool locally.
result = get_weather(tool_use["input"]["city"])
print(f"Tool result: {result}")

# Step 3: Send the result back.
messages.append({"role": "assistant", "content": first["content"]})
messages.append(
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use["id"],
                "content": result,
            }
        ],
    }
)

# Step 4: Model produces the final answer.
second = post(
    "/v1/messages",
    {
        "model": "default",
        "max_tokens": 256,
        "messages": messages,
        "tools": tools,
    },
)

for block in second["content"]:
    if block["type"] == "text":
        print(block["text"])
