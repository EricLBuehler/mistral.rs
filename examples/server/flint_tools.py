"""Flint reasoning and multi-turn tool calling.

Start the server with `mistralrs serve -p 1234 -m RecursiveSI/flint`, then run
`python examples/server/flint_tools.py`. The example preserves reasoning and
handles multiple tool calls in a turn.
"""

import json

from openai import OpenAI

MAX_TOOL_ROUNDS = 8
MAX_OUTPUT_TOKENS = 32768

client = OpenAI(api_key="unused", base_url="http://localhost:1234/v1")
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]
messages = [
    {"role": "user", "content": "Compare the weather in Paris and Tokyo using the tool."}
]
weather = {"paris": "Sunny, 21C", "tokyo": "Cloudy, 24C"}

for _ in range(MAX_TOOL_ROUNDS):
    response = client.chat.completions.create(
        model="default",
        messages=messages,
        tools=tools,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    choice = response.choices[0]
    message = choice.message
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        print(f"Reasoning: {reasoning}")
    if message.content:
        print(message.content)
    if choice.finish_reason == "length":
        raise RuntimeError("Output budget exhausted; increase MAX_OUTPUT_TOKENS.")
    if not message.tool_calls:
        break

    assistant = {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [call.model_dump() for call in message.tool_calls],
    }
    if reasoning:
        assistant["reasoning_content"] = reasoning
    messages.append(assistant)
    for call in message.tool_calls:
        if call.function.name != "get_weather":
            raise ValueError(f"Unexpected tool: {call.function.name}")
        arguments = json.loads(call.function.arguments)
        city = arguments["city"]
        result = weather.get(city.lower(), f"No weather available for {city}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": result,
            }
        )
else:
    raise RuntimeError("Tool round limit reached.")
