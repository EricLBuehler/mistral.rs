from io import StringIO
import json
import sys
from mistralrs import Runner, ToolChoice, Which, ChatCompletionRequest, Architecture

tools = [
    json.dumps(
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
    )
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
        "content": "What is the value of the area of a circle with radius 4?",
    }
]

runner = Runner(
    which=Which.Plain(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        arch=Architecture.Llama,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="llama-3.1",
        messages=messages,
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
        tool_schemas=tools,
        tool_choice=ToolChoice.Auto,
    )
)
# print(res.choices[0].message)
# print(res.usage)

tool_called = res.choices[0].message.tool_calls[0].function

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

    res = runner.send_chat_completion_request(
        ChatCompletionRequest(
            model="llama-3.1",
            messages=messages,
            max_tokens=256,
            presence_penalty=1.0,
            top_p=0.1,
            temperature=0.1,
            tool_schemas=tools,
            tool_choice=ToolChoice.Auto,
        )
    )
    # print(res.usage)
    print(res.choices[0].message.content)
