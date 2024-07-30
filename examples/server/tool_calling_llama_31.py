import openai
import json

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:1234/v1/"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "string",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

completion = openai.chat.completions.create(
    model="llama-31", messages=messages, tools=tools, tool_choice="auto"
)

print(completion.usage)

tool_called = completion.choices[0].message.tool_calls[0].function


def get_current_weather(unit: str = None, location: str = None) -> str:
    return f"The weather in {location} is 40 degrees {unit}."


if tool_called.name == "get_current_weather":
    args = json.loads(tool_called.arguments)
    result = get_current_weather(**args)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps({"name": tool_called.name, "parameters": args}),
        }
    )
    messages.append({"role": "ipython", "content": result})
    print(messages)
    completion = openai.chat.completions.create(
        model="llama-31", messages=messages, tools=tools, tool_choice="auto"
    )
    print(completion.usage)
    print(completion.choices[0].message)
