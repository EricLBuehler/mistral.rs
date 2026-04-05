# Tool calling

Tool calling makes LLMs smarter.

LLMs use tool calling to interact with the outside world. Mistral.rs has OpenAI compatible support for tool calling in all APIs, HTTP, Python, and Rust.

Note that some models, such as Mistral Small/Nemo models, require a chat template to be specified. For example:

```bash
mistralrs serve -p 1234 --isq 4 --jinja-explicit chat_templates/mistral_small_tool_call.jinja -m mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

OpenAI docs: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

We support the following models' tool calling in OpenAI-compatible and parse native tool calling:

- Gemma 4
- Llama 4
- Llama 3.1/3.2/3.3
- Mistral Small (including 3.1 + multimodal)
- Mistral Nemo
- Hermes 2 Pro
- Hermes 3
- DeepSeek V2/V3/R1
- Qwen 3
- GPT-OSS

All models that support tool calling will respond according to the OpenAI tool calling API.

## Tool call grammar enforcement

When tools are provided in a request, mistral.rs automatically enforces constrained decoding on tool call output. When the model begins generating a tool call, a grammar is activated mid-stream that constrains subsequent tokens to valid tool call syntax. This prevents malformed JSON, hallucinated tool names, and missing closing delimiters.

**How it works:**
1. The model generates normally until a tool call prefix is detected.
2. A format-specific grammar activates and constrains all subsequent tokens to valid tool call structure.
3. When the tool call body (and closing delimiter, if applicable) is complete, the grammar deactivates.
4. For multi-tool-call turns, the grammar re-activates when the next prefix is detected.

This feature is automatic, and no configuration is needed. It uses the same [llguidance](https://github.com/guidance-ai/llguidance) infrastructure as user-specified grammar constraints. If a user-specified grammar is already active on the request, tool call grammar activation is skipped.

## Strict mode

By default, the tool call grammar enforces valid syntax (correct delimiters, valid tool names, well-formed key-value pairs) but allows any argument keys and value types. **Strict mode** goes further: it enforces the tool's `parameters` JSON schema on the generated arguments.

Set `"strict": true` on the function definition to enable it:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": { "type": "string" },
        "units": { "type": "string", "enum": ["celsius", "fahrenheit"] }
      },
      "required": ["city"]
    },
    "strict": true
  }
}
```

**What strict mode enforces:**
- Only declared property names are accepted as argument keys
- Value types match the schema (string, number, integer, boolean, null)
- Enum values are constrained to the declared set
- Nested objects and typed arrays follow their sub-schemas
- Required fields must appear; optional fields may be omitted

**Notes:**
- Strict and non-strict tools can be mixed in the same request. Each tool is enforced independently.
- The built-in web search tools (`search_the_web`, `website_content_extractor`) use strict mode automatically.
- For Gemma 4, arguments are emitted in alphabetical key order (matching the model's native `dictsort` convention), which allows required-field enforcement without combinatorial grammar blowup.

In the Rust SDK, set `strict: Some(true)` on the `Function` struct. In the Python SDK, include `"strict": true` in the tool JSON string passed to `tool_schemas`.

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
