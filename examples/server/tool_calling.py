"""
Basic client-side tool calling with the HTTP API.

Start the server:
    mistralrs serve -p 1234 --isq 4 -m Qwen/Qwen3-4B

Then run this script:
    python examples/server/tool_calling.py
"""

import json
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

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
                },
                "required": ["city"],
            },
            "strict": True,
        },
    }
]


def get_weather(city: str) -> str:
    """Simulated weather lookup."""
    data = {"tokyo": "Sunny, 22C", "london": "Cloudy, 15C"}
    return data.get(city.lower(), f"Unknown city: {city}")


messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

# Step 1: Model generates a tool call
completion = client.chat.completions.create(
    model="default", messages=messages, tools=tools, tool_choice="auto"
)
msg = completion.choices[0].message
print(f"Model wants to call: {msg.tool_calls[0].function.name}")

# Step 2: Execute the tool locally
tool_call = msg.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = get_weather(**args)
print(f"Tool result: {result}")

# Step 3: Send the result back
messages.append(
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    }
)
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": result,
    }
)

# Step 4: Model produces the final answer
completion = client.chat.completions.create(
    model="default", messages=messages, tools=tools, tool_choice="auto"
)
print(completion.choices[0].message.content)
