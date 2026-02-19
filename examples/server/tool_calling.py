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

The output should be something like:
```
Called tool `run_python`
The final answer is $\boxed{50.26548245743669}$.
```
"""

import json
import sys
from io import StringIO
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

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
                        "description": "The Python code to evaluate. The return value whatever was printed out from `print`.",
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

    print(f"Running:\n```py\n{code}\n```")

    old_stdout = sys.stdout
    out = StringIO()
    sys.stdout = out
    exec(code, glbls, lcls)
    sys.stdout = old_stdout

    return out.getvalue()


functions = {
    "run_python": run_python,
}

messages = [
    {
        "role": "user",
        "content": "Please write and run a python script to do a matmul. Then tell me JUST the result of the matmul.",
    }
]

completion = client.chat.completions.create(
    model="default",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

# print(completion.usage)
print(completion.choices[0].message)

tool_call = completion.choices[0].message.tool_calls[0]
tool_called = tool_call.function

if tool_called.name in functions:
    args = json.loads(tool_called.arguments)
    result = functions[tool_called.name](**args)
    print(f"Called tool `{tool_called.name}`")

    # Append the assistant message with tool_calls (must include the original message)
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_called.name,
                        "arguments": tool_called.arguments,
                    },
                }
            ],
        }
    )

    # Append the tool response with tool_call_id
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_called.name,
            "content": result,
        }
    )

    completion = client.chat.completions.create(
        model="default", messages=messages, tools=tools, tool_choice="auto"
    )
    # print(completion.usage)
    print(completion.choices[0].message)
