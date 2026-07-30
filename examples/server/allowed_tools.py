"""
OpenAI-compatible allowed_tools tool choice.

Start the server:
    mistralrs serve -p 1234 --quant 4 -m Qwen/Qwen3-4B

Then run this script:
    python examples/server/allowed_tools.py
"""

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
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight to a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": True,
        },
    },
]

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": "Use a tool to help me plan for Tokyo. Do not answer directly.",
        }
    ],
    tools=tools,
    tool_choice={
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "get_weather"}],
    },
)

tool_call = completion.choices[0].message.tool_calls[0]
print(f"Model called: {tool_call.function.name}")
print(f"Arguments: {tool_call.function.arguments}")
