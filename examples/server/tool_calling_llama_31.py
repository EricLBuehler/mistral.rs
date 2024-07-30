"""
Important note:

We recommend you use Llama 3.1 70B or larger for this example as it incorporates multi-turn chat.

Llama 3.1 may also be used:
```
cargo run --release --features cuda -- --port 1234 --isq Q4K plain -m meta-llama/Meta-Llama-3.1-8B-Instruct -a llama
```

And then:
```
python3 examples/server/tool_calling_llama_31.py
```
"""

import openai
import json

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:1234/v1/"

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run some Python code",
            "parameters": {
                "type": "string",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to evaluate. Returns a JSON of the local variables.",
                    },
                },
                "required": ["code"],
            },
        },
    }
]


def custom_serializer(obj):
    try:
        res = json.dumps(obj)
    except:
        # Handle serializing, for example, an imported module
        res = None
    return res


def run_python(code: str) -> str:
    lcls = dict()
    # No opening of files
    glbls = {"open": None}
    exec(code, glbls, lcls)
    res = {
        "locals": lcls,
    }
    return json.dumps(res, default=custom_serializer)


functions = {
    "run_python": run_python,
}

messages = [
    {
        "role": "user",
        "content": "Write some Python code to calculate the area of a circle with radius 4. Store the result in `A`. What is `A`?",
    }
]

completion = openai.chat.completions.create(
    model="llama-3.1", messages=messages, tools=tools, tool_choice="auto"
)

print(completion.usage)
print(completion.choices[0].message)

tool_called = completion.choices[0].message.tool_calls[0].function


if tool_called.name in functions:
    args = json.loads(tool_called.arguments)
    result = functions[tool_called.name](**args)
    print(f"Called {tool_called.name}, result is {result}")

    messages.append(
        {
            "role": "assistant",
            "content": json.dumps({"name": tool_called.name, "parameters": args}),
        }
    )

    messages.append({"role": "ipython", "content": result})

    completion = openai.chat.completions.create(
        model="llama-3.1", messages=messages, tools=tools, tool_choice="auto"
    )
    print(completion.usage)
    print(completion.choices[0].message)
