"""
User-provided input files with the Python SDK.

The request attaches a CSV as an input file. Text-like files are previewed in
the prompt and can be paginated with the built-in file tools when the agentic
runtime is active.

Usage:
    python examples/python/file_inputs.py
"""

from mistralrs import Architecture, ChatCompletionRequest, InputFile, Runner, Which


def main():
    runner = Runner(
        which=Which.Plain(
            model_id="Qwen/Qwen3-4B",
            arch=Architecture.Qwen3,
        ),
    )

    sales = InputFile.from_text(
        "sales.csv",
        "region,revenue\nnorth,120\nsouth,95\nwest,180\n",
        "text/csv",
    )
    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": "Which region has the highest revenue? Use the attached CSV.",
            }
        ],
        model="default",
        input_files=[sales],
    )

    response = runner.send_chat_completion_request(request)
    for choice in response.choices:
        print(choice.message.content)


if __name__ == "__main__":
    main()
