# Tool calling

Tool calling makes LLMs smarter.

LLMs use tool calling to interact with the outside world. Mistral.rs has OpenAI compatible support for tool calling in all APIs, HTTP, Python, and Rust.

OpenAI docs: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

## OpenAI compatible HTTP example
Please see [our example here](../examples/server/tool_calling.py).

> OpenAI docs: https://platform.openai.com/docs/api-reference/chat/create?lang=curl

## Rust example
Please see [our example here](../mistralrs/examples/tools/main.rs).

## Python example
Please see [our notebook here](../examples/python/tool_calling.ipynb).
