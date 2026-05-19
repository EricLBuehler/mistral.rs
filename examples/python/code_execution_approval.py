"""
Python SDK code-execution approval callback.

Run with:
    pip install -e mistralrs-pyo3 --features code-execution
    python examples/python/code_execution_approval.py
"""

from mistralrs import ChatCompletionRequest, CodeExecutionConfig, Runner, Which


def approve(call):
    print("\nAgent action approval required")
    print(f"approval_id: {call['approval_id']}")
    print(f"session_id: {call['session_id']}")
    print(f"tool: {call['tool']['label']}")
    print("\nCode:")
    print(call.get("code", "<no code>"))

    decision = input("\nRun this Python code? [y/N] ").strip().lower()
    return decision in {"y", "yes"}


def main():
    runner = Runner(
        which=Which.Plain(model_id="Qwen/Qwen3-4B"),
        code_execution_config=CodeExecutionConfig(),
    )

    response = runner.send_chat_completion_request(
        ChatCompletionRequest(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": "Use Python to calculate the first 20 Fibonacci numbers.",
                }
            ],
            enable_code_execution=True,
            agent_permission="ask",
            agent_approval_callback=approve,
            max_tool_rounds=4,
        )
    )

    for choice in response.choices:
        print(choice.message.content)


if __name__ == "__main__":
    main()
