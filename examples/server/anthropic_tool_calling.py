"""
Client-side tool calling with the Anthropic Messages API.

Run the server:
    mistralrs serve -p 1234 -m meta-llama/Meta-Llama-3.1-8B-Instruct

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
    return f"The weather in {city} is 72 F and clear."


tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "What is the weather in Paris? Use the tool.",
    }
]

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

messages.append({"role": "assistant", "content": first["content"]})

for block in first["content"]:
    if block["type"] == "tool_use" and block["name"] == "get_weather":
        result = get_weather(block["input"]["city"])
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": result,
                    }
                ],
            }
        )

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
