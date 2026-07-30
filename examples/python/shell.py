"""
Shell execution with the OS-level sandbox enabled.

The model is given a shell tool and can run commands in a per-session working
directory.

Run with:
    pip install -e mistralrs-pyo3 --features code-execution
    python examples/python/shell.py
"""

from mistralrs import (
    ChatCompletionRequest,
    NetworkMode,
    Runner,
    SandboxPolicy,
    ShellConfig,
    Which,
)


def main():
    sandbox = SandboxPolicy(
        max_memory_mb=1024,
        max_cpu_secs=120,
        max_procs=32,
        network=NetworkMode.Loopback,
    )

    runner = Runner(
        which=Which.Plain(model_id="Qwen/Qwen3-4B"),
        shell_config=ShellConfig(sandbox_policy=sandbox),
    )

    response = runner.send_chat_completion_request(
        ChatCompletionRequest(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": "Use the shell to print the current directory and list its files.",
                }
            ],
            enable_shell=True,
            max_tool_rounds=4,
        )
    )

    for choice in response.choices:
        print(choice.message.content)


if __name__ == "__main__":
    main()
