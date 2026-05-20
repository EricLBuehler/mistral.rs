"""
Python code execution with the OS-level sandbox enabled.

Equivalent of `mistralrs/examples/advanced/code_execution/main.rs` for the
Python SDK. The model is given the `execute_python` tool and runs Python
inside a per-session subprocess that is hardened with rlimits + seccomp +
namespaces + Landlock on Linux (Seatbelt + rlimits on macOS).

Run with:
    pip install -e mistralrs-pyo3 --features code-execution
    python examples/python/code_execution.py
"""

from mistralrs import (
    ChatCompletionRequest,
    CodeExecutionConfig,
    NetworkMode,
    Runner,
    SandboxPolicy,
    Which,
)


def main():
    # Construct an OS-level sandbox policy. Pass `sandbox_policy=None`
    # (or omit it) to disable the sandbox - the spawned interpreter will
    # then have full filesystem and network access.
    sandbox = SandboxPolicy(
        max_memory_mb=1024,
        max_cpu_secs=120,
        max_procs=32,
        network=NetworkMode.NoNetwork,
    )

    runner = Runner(
        which=Which.Plain(model_id="Qwen/Qwen3-4B"),
        code_execution_config=CodeExecutionConfig(sandbox_policy=sandbox),
    )

    response = runner.send_chat_completion_request(
        ChatCompletionRequest(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": "Use Python to calculate the first 20 prime numbers and their sum.",
                }
            ],
            enable_code_execution=True,
            max_tool_rounds=4,
        )
    )

    for choice in response.choices:
        print(choice.message.content)


if __name__ == "__main__":
    main()
