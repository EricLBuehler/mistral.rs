import json
from mistralrs import Runner, ToolChoice, Which, ChatCompletionRequest, Architecture

tools = [
    json.dumps(
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
    )
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

runner = Runner(
    which=Which.Plain(
        model_id="lamm-mit/Bioinspired-Llama-3-1-8B-128k-gamma",
        arch=Architecture.Llama,
    ),
    in_situ_quant="Q4K",
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
