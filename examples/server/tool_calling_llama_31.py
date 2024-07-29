import openai

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

print(completion)
