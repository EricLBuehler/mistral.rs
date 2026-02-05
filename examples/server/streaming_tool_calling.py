"""
Streaming tool calling example.

This demonstrates how to use streaming with tool calls in mistral.rs.
Tool calls are accumulated during streaming and sent in the final chunk
with finish_reason="tool_calls".

Usage:
```
cargo run --release --features cuda -- --port 1234 --isq Q4K plain -m meta-llama/Meta-Llama-3.1-8B-Instruct -a llama
```

And then:
```
python3 examples/server/streaming_tool_calling.py
```

Note: Some models may make multiple tool calls before providing a final response.
This example handles up to MAX_TOOL_ROUNDS of tool calling iterations.
"""

import json
import sys
from io import StringIO
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

MAX_TOOL_ROUNDS = 10

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run some Python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to evaluate. The return value is whatever was printed out from `print`.",
                    },
                },
                "required": ["code"],
            },
        },
    }
]


def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    lcls = dict()
    glbls = {"open": None}  # No opening of files

    print(f"Running:\n```py\n{code}\n```")

    old_stdout = sys.stdout
    out = StringIO()
    sys.stdout = out
    try:
        exec(code, glbls, lcls)
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {e}"
    sys.stdout = old_stdout

    return out.getvalue()


functions = {
    "run_python": run_python,
}


def do_streaming_request(messages, tools, round_num):
    """Make a streaming request and handle the response."""
    print(f"\n{'=' * 60}")
    print(f"Round {round_num}: Making streaming request...")
    print(f"{'=' * 60}")

    stream = client.chat.completions.create(
        model="default",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )

    collected_content = ""
    collected_tool_calls = []
    finish_reason = None

    print("Assistant: ", end="", flush=True)
    collected_reasoning = ""

    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        # Print reasoning content in gray (if present)
        # The field may be 'reasoning_content' or accessible via getattr
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            # Print reasoning in gray/dim
            print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
            collected_reasoning += reasoning

        # Print and collect content as it streams
        if delta.content:
            print(delta.content, end="", flush=True)
            collected_content += delta.content

        # Collect tool calls from delta
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                # Find or create the tool call entry
                while len(collected_tool_calls) <= tool_call.index:
                    collected_tool_calls.append(
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )

                tc = collected_tool_calls[tool_call.index]
                if tool_call.id:
                    tc["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        tc["function"]["name"] = tool_call.function.name
                    if tool_call.function.arguments:
                        tc["function"]["arguments"] += tool_call.function.arguments

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    print()  # newline after streaming
    print(f"[Finish reason: {finish_reason}]")
    if collected_reasoning:
        print(f"[Reasoning: {len(collected_reasoning)} chars]")

    return collected_content, collected_tool_calls, finish_reason, collected_reasoning


def execute_tool_calls(tool_calls):
    """Execute tool calls and return results."""
    results = []
    for tool_call in tool_calls:
        func_name = tool_call["function"]["name"]
        func_args = tool_call["function"]["arguments"]

        print(f"  Calling {func_name}({func_args})")

        if func_name in functions:
            try:
                args = json.loads(func_args) if func_args else {}
                result = functions[func_name](**args)
                print(f"  -> {result}")
                results.append((tool_call, result))
            except Exception as e:
                error_result = json.dumps({"error": str(e)})
                print(f"  -> Error: {e}")
                results.append((tool_call, error_result))
        else:
            error_result = json.dumps({"error": f"Unknown function: {func_name}"})
            print(f"  -> Error: Unknown function")
            results.append((tool_call, error_result))

    return results


def main():
    messages = [
        {
            "role": "user",
            "content": "Please write and run a python script to do a matmul of 2 random integer matrices. Then tell me JUST the result of the matmul.",
        }
    ]

    print(
        "User: Please write and run a python script to do a matmul of 2 random integer matrices. Then tell me JUST the result of the matmul."
    )

    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        content, tool_calls, finish_reason, reasoning = do_streaming_request(
            messages, tools, round_num
        )

        if finish_reason == "tool_calls" and tool_calls:
            print(f"\nTool calls detected ({len(tool_calls)}):")

            # Add assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": content if content else None,
                    "tool_calls": tool_calls,
                }
            )

            # Execute tool calls and add results
            results = execute_tool_calls(tool_calls)
            for tool_call, result in results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": result,
                    }
                )

            print(f"\nContinuing to round {round_num + 1}...")
        else:
            # No tool calls, we're done
            print(f"\nFinal response received (finish_reason: {finish_reason})")
            if content:
                print(f"Content: {content}")
            break
    else:
        print(f"\nReached maximum tool rounds ({MAX_TOOL_ROUNDS})")


if __name__ == "__main__":
    main()
