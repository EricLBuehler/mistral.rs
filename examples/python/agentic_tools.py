"""
Example demonstrating the Python SDK's agentic tool callback system.

The Python SDK lets you register tool callbacks directly on the Runner.
When combined with max_tool_rounds on the request, the engine automatically
executes your callbacks and feeds results back to the model in a loop.

You can also set tool_dispatch_url on the request to POST unhandled tool
calls to an external HTTP endpoint for execution.

Usage:
    python examples/python/agentic_tools.py
"""

import json
from mistralrs import (
    Runner,
    Which,
    Architecture,
    ChatCompletionRequest,
    ToolChoice,
)


def tool_callback(name: str, args: dict) -> str:
    """Dispatch tool calls to local implementations."""
    if name == "get_weather":
        city = args.get("city", "unknown")
        return json.dumps({"city": city, "temp": 22, "condition": "Sunny"})
    if name == "calculate":
        expression = args.get("expression", "0")
        try:
            result = eval(expression)  # noqa: S307 — example only
        except Exception as e:
            result = str(e)
        return json.dumps({"result": result})
    return json.dumps({"error": f"Unknown tool: {name}"})


def main():
    # Register tool callbacks at Runner level — these are available to all requests
    runner = Runner(
        which=Which.Plain(
            model_id="Qwen/Qwen3-4B",
            arch=Architecture.Qwen3,
        ),
        tool_callbacks={
            "get_weather": tool_callback,
            "calculate": tool_callback,
        },
    )

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
        ),
        json.dumps(
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a math expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression to evaluate",
                            },
                        },
                        "required": ["expression"],
                    },
                    "strict": True,
                },
            }
        ),
    ]

    # max_tool_rounds enables the agentic loop: model calls tools,
    # engine executes callbacks, feeds results back, repeats.
    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Tokyo? Also calculate 42 * 17.",
            }
        ],
        model="default",
        tool_schemas=tools,
        tool_choice=ToolChoice.Auto,
        max_tool_rounds=5,
    )

    response = runner.send_chat_completion_request(request)

    for choice in response.choices:
        print(choice.message.content)


if __name__ == "__main__":
    main()
