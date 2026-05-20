"""
Python SDK code-execution approval callback.

Run with:
    pip install -e mistralrs-pyo3 --features code-execution
    python examples/python/code_execution_approval.py
"""

from mistralrs import (
    AgentPermission,
    AgentToolApproval,
    AgentToolApprovalDecision,
    AgentToolKind,
    ChatCompletionRequest,
    CodeExecutionConfig,
    Runner,
    Which,
)


def approve(call: AgentToolApproval):
    print("\nAgent action approval required")
    print(f"approval_id: {call.approval_id}")
    print(f"session_id: {call.session_id}")
    print(f"tool: {call.tool.label} ({call.tool.kind})")
    if call.tool.kind == AgentToolKind.CodeExecution:
        print("\nCode:")
        print(call.code or "<no code>")
    else:
        print("\nArguments:")
        print(call.arguments_json)

    while True:
        decision = (
            input("\nRun this Python code? [y]es / [n]o / [a]lways: ").strip().lower()
        )
        if decision in {"y", "yes"}:
            return AgentToolApprovalDecision.approve()
        if decision in {"a", "always"}:
            return AgentToolApprovalDecision.approve(remember_for_session=True)
        if decision in {"", "n", "no"}:
            return AgentToolApprovalDecision.deny("The user denied this action.")
        print("Please enter y, n, or a.")


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
            agent_permission=AgentPermission.Ask,
            agent_approval_callback=approve,
            max_tool_rounds=4,
        )
    )

    for choice in response.choices:
        print(choice.message.content)


if __name__ == "__main__":
    main()
