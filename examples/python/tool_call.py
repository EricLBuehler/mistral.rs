"""
Basic client-side tool calling with the Python SDK.

Usage:
    python examples/python/tool_call.py
"""

import json
from mistralrs import Runner, ToolChoice, Which, ChatCompletionRequest, Architecture

tools = [
    json.dumps(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["city"],
                },
                "strict": True,
            },
        }
    )
]


def get_weather(city: str) -> str:
    """Simulated weather lookup."""
    data = {"tokyo": "Sunny, 22C", "london": "Cloudy, 15C"}
    return data.get(city.lower(), f"Unknown city: {city}")


messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B", arch=Architecture.Qwen3),
)

# Step 1: Model generates a tool call
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=messages,
        max_tokens=256,
        tool_schemas=tools,
        tool_choice=ToolChoice.Auto,
    )
)

tool_called = res.choices[0].message.tool_calls[0].function
args = json.loads(tool_called.arguments)
result = get_weather(**args)
print(f"Called tool `{tool_called.name}`: {result}")

# Step 2: Send the result back
messages.append(
    {
        "role": "assistant",
        "content": json.dumps({"name": tool_called.name, "parameters": args}),
    }
)
messages.append({"role": "tool", "content": result})

# Step 3: Model produces the final answer
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=messages,
        max_tokens=256,
        tool_schemas=tools,
        tool_choice=ToolChoice.Auto,
    )
)
print(res.choices[0].message.content)
