# Tool calling

Tool calling makes LLMs smarter.

LLMs use tool calling to interact with the outside world. Mistral.rs has OpenAI compatible support for tool calling in all APIs, HTTP, Python, and Rust.

Note that some models, such as Mistral Small/Nemo models, require a chat template to be specified. For example:

```bash
mistralrs serve -p 1234 --isq 4 --jinja-explicit chat_templates/mistral_small_tool_call.jinja -m mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

OpenAI docs: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

We support the following models' tool calling in OpenAI-compatible and parse native tool calling:

- Llama 4
- Llama 3.1/3.2/3.3
- Mistral Small (including 3.1 + multimodal)
- Mistral Nemo
- Hermes 2 Pro
- Hermes 3
- DeepSeek V2/V3/R1
- Qwen 3

All models that support tool calling will respond according to the OpenAI tool calling API.

## OpenAI compatible HTTP example
Please see [our example here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/tool_calling.py).

> OpenAI docs: https://platform.openai.com/docs/api-reference/chat/create?lang=curl

## Rust example
Please see [our example here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/tools/main.rs).

## Python example
Please see [our notebook here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/tool_calling.ipynb).

## Tool callbacks

You can override tool execution using a **tool callback**. The callback receives
the tool name and a dictionary of arguments and must return the tool output as a
string.

### Python

```py
def tool_cb(name: str, args: dict) -> str:
    if name == "local_search":
        return json.dumps(local_search(args.get("query", "")))
    return ""

runner = Runner(
    which=Which.Plain(model_id="YourModel/ID", arch=Architecture.Llama),
    tool_callback=tool_cb,
)
```

See [custom_search.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/custom_search.py) for a full
example. In Rust pass `.with_tool_callback(...)` to the builder as demonstrated
in [tool_callback/main.rs](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/tool_callback/main.rs).

## Search callbacks

Web search uses a DuckDuckGo-based callback by default. Provide your own search
function with `search_callback` in Python or `.with_search_callback(...)` in
Rust. Each callback should return a list of results with `title`, `description`,
`url` and `content` fields. See [WEB_SEARCH.md](WEB_SEARCH.md) for more details
and examples.
