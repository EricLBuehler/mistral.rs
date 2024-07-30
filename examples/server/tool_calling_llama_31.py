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
                        "description": "The Python code to evaluate. Returns a JSON of the global and local variables.",
                    },
                },
                "required": ["code"],
            },
        },
    }
]


def run_python(code: str) -> str:
    glbls = dict()
    lcls = dict()
    exec(code, glbls, lcls)
    res = {
        "globals": glbls,
        "locals": lcls,
    }
    return json.dumps(res)


functions = {
    "run_python": run_python,
}

messages = [{"role": "user", "content": "Write some Python code to calculate the arctan of 1rad."}]

completion = openai.chat.completions.create(
    model="llama-3.1", messages=messages, tools=tools, tool_choice="auto"
)

print(completion.usage)
print(completion.choices[0].message)

tool_called = completion.choices[0].message.tool_calls[0].function


if tool_called.name in functions:
    args = json.loads(tool_called.arguments)
    result = functions[functions](**args)

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
