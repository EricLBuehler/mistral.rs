"""
https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/

Llama 3.1 may be used:
```
cargo run --release --features cuda -- --port 1234 --isq Q4K plain -m meta-llama/Meta-Llama-3.1-8B-Instruct -a llama
```

And then:
```
python3 examples/server/tool_calling_llama_31.py
```
"""

import json
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

tools = [
    {
        "type": "function",
        "function": {
            "name": "add_2_numbers",
            "description": "Add two numbers, floating point or integer",
            "parameters": {
                "type": "string",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number.",
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number.",
                    },
                },
                "required": ["x", "y"],
            },
        },
    }
]


def add_2_numbers(x, y):
    return x + y


functions = {
    "add_2_numbers": add_2_numbers,
}

messages = [
    {
        "role": "user",
        "content": "Please add 1234513543214 and 1111998778.",
    }
]

completion = client.chat.completions.create(
    model="llama-3.1", messages=messages, tools=tools, tool_choice="auto"
)

# print(completion.usage)
# print(completion.choices[0].message)

tool_called = completion.choices[0].message.tool_calls[0].function

if tool_called.name in functions:
    args = json.loads(tool_called.arguments)
    result = functions[tool_called.name](**args)
    print(f"Called tool `{tool_called.name}`")

    messages.append(
        {
            "role": "assistant",
            "content": json.dumps({"name": tool_called.name, "parameters": args}),
        }
    )

    messages.append({"role": "tool", "content": result})

    completion = client.chat.completions.create(
        model="llama-3.1", messages=messages, tools=tools, tool_choice="auto"
    )
    # print(completion.usage)
    print(completion.choices[0].message.content)
