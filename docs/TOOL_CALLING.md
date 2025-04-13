# Tool calling

Tool calling makes LLMs smarter.

LLMs use tool calling to interact with the outside world. Mistral.rs has OpenAI compatible support for tool calling in all APIs, HTTP, Python, and Rust.

Note that some models, such as Mistral Small/Nemo models, require a chat template to be specified. For example:

```bash
./mistralrs-server --port 1234 --isq q4k --jinja-explicit chat_templates/mistral_small_tool_call.jinja vision-plain -m mistralai/Mistral-Small-3.1-24B-Instruct-2503 -a mistral3  
```

OpenAI docs: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

We support the following models' tool calling in OpenAI-compatible and parse native tool calling:

- Llama 4
- Llama 3.1/3.2/3.3
- Mistral Small (including 3.1 + multimodal)
- Mistral Nemo
- Hermes 2 Pro
- Hermes 3
- DeepSeeek V2/V3/R1

All models that support tool calling will respond according to the OpenAI tool calling API.

## OpenAI compatible HTTP example
Please see [our example here](../examples/server/tool_calling.py).

> OpenAI docs: https://platform.openai.com/docs/api-reference/chat/create?lang=curl

## Rust example
Please see [our example here](../mistralrs/examples/tools/main.rs).

## Python example
Please see [our notebook here](../examples/python/tool_calling.ipynb).
