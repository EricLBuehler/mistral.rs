# Tool calling

Mistral.rs supports OpenAI-compatible tool calling across all APIs (HTTP, Python SDK, Rust SDK). The model decides which tools to call and generates structured arguments; you decide how the tools get executed: either client-side or automatically on the server.

## Supported models

All supported models respond according to the [OpenAI tool calling API](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models):

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

> Some models (e.g. Mistral Small/Nemo) require a specific chat template:
> ```bash
> mistralrs serve -p 1234 --isq 4 --jinja-explicit chat_templates/mistral_small_tool_call.jinja -m mistralai/Mistral-Small-3.1-24B-Instruct-2503
> ```

## Two modes of tool calling

### Basic (client-side execution)

The model generates a tool call, the server returns it to the client, and your code executes the tool and sends the result back in a follow-up request. This is the standard OpenAI flow.

### Agentic (server-side execution)

The server executes tools automatically and feeds results back to the model in a loop without any client round-trips. The model calls a tool, the server runs it, appends the result, and lets the model continue until it produces a final text answer or hits the round limit.

To use the agentic loop, you need two things:
1. **A way to execute tools**: register callbacks (Python/Rust SDK), connect MCP servers, enable web search, or set a tool dispatch URL
2. **`max_tool_rounds`**: tells the server how many loop iterations to allow

## Grammar enforcement

When tools are provided in a request, mistral.rs automatically constrains the model's output to valid tool call syntax using [llguidance](https://github.com/guidance-ai/llguidance). This prevents malformed JSON, hallucinated tool names, and missing delimiters.

1. The model generates normally until a tool call prefix is detected.
2. A grammar activates mid-stream and constrains tokens to valid tool call structure.
3. When the tool call is complete, the grammar deactivates.
4. For multi-tool turns, the grammar re-activates for each subsequent tool call.

This is automatic and requires no configuration. If a user-specified grammar is already active on the request, tool call grammar is skipped.

### Strict mode

By default, grammar enforcement ensures valid syntax but allows any argument keys and types. **Strict mode** additionally enforces the tool's JSON schema on the arguments.

Set `"strict": true` on the function definition:

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

What strict mode enforces:
- Only declared property names are accepted as argument keys
- Value types match the schema (string, number, integer, boolean, null)
- Enum values are constrained to the declared set
- Nested objects and typed arrays follow their sub-schemas
- Required fields must appear; optional fields may be omitted

Notes:
- Strict and non-strict tools can be mixed in the same request.
- Built-in web search tools use strict mode automatically.
- MCP tools are automatically promoted to strict mode when they provide an input schema.
- For Gemma 4, arguments are emitted in alphabetical key order (matching the model's native `dictsort` convention).

In the Rust SDK, set `strict: Some(true)` on the `Function` struct. In the Python SDK, include `"strict": true` in the tool JSON string passed to `tool_schemas`.

## Agentic loop

### Enabling the loop

Set `max_tool_rounds` to activate server-side tool execution. The server will loop until the model stops calling tools or the limit is reached.

**HTTP API:**
```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
  "tools": [...],
  "tool_choice": "auto",
  "max_tool_rounds": 5
}
```

**Rust SDK:**
```rust
let response = model.send_chat_request(
    RequestBuilder::new()
        .add_message(TextMessageRole::User, "What's the weather in Tokyo?")
        .set_tools(tools)
        .set_tool_choice(ToolChoice::Auto)
        .set_max_tool_rounds(5)
).await?;
```

**Python SDK:**
```python
request = ChatCompletionRequest(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    model="default",
    tool_schemas=tools,
    tool_choice=ToolChoice.Auto,
    max_tool_rounds=5,
)
response = runner.send_chat_completion_request(request)
```

**CLI default:** You can set a server-wide default with `--max-tool-rounds <N>`. Per-request values from the HTTP API override it. Safety cap is 16 rounds if unset.

### How tools get executed

When the model calls a tool during the agentic loop, the server tries these dispatch methods in order:

1. **Built-in search tools** (`search_the_web`, `website_content_extractor`): if web search is enabled
2. **Registered callbacks**: tool callbacks from the Python/Rust SDK or MCP servers
3. **Tool dispatch URL**: POSTs the tool call to your HTTP endpoint
4. **No handler found**: the loop stops and the un-executed tool call is returned to the client

Streaming is supported: tool call chunks are forwarded to the client so you can see which tools are being called mid-loop.

### Tool dispatch URL

The tool dispatch URL lets the server POST unhandled tool calls to your HTTP endpoint. This is how HTTP API users get server-side tool execution without registering callbacks.

**Security:** To prevent SSRF, `tool_dispatch_url` cannot be set per-request via the HTTP API. Configure it server-side or in trusted SDK code:

```bash
# CLI: applies to all requests
mistralrs serve -p 1234 --tool-dispatch-url https://my-service.com/tools --max-tool-rounds 5 -m google/gemma-4-E4B-it
```

```rust
// Rust SDK: per-request (trusted code)
RequestBuilder::new()
    .set_tool_dispatch_url("https://my-service.com/tools")
    .set_max_tool_rounds(5)
```

```python
# Python SDK: per-request (trusted code)
ChatCompletionRequest(
    ...,
    tool_dispatch_url="https://my-service.com/tools",
    max_tool_rounds=5,
)
```

The server POSTs to your URL with:
```json
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
```

Your endpoint returns:
```json
{"content": "Sunny, 22°C"}
```

The response can also be a bare string. One URL handles all tools, dispatched by `name`.

### Tool callbacks

Register custom functions to handle tool calls directly in your code. The callback receives the tool name and arguments, and returns the result as a string.

**Python SDK:**
```python
def tool_cb(name: str, args: dict) -> str:
    if name == "get_weather":
        return json.dumps({"temp": 22, "condition": "Sunny"})
    return ""

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B", arch=Architecture.Qwen3),
    tool_callbacks={"get_weather": tool_cb},
)
```

**Rust SDK:** Pass `.with_tool_callback(name, callback)` to the builder. See [tool_callback/main.rs](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/tool_callback/main.rs).

### Search callbacks

Web search uses DuckDuckGo by default. Override it with a custom search function using `search_callback` in Python or `.with_search_callback(...)` in Rust. Each callback should return a list of results with `title`, `description`, `url`, and `content` fields. See [WEB_SEARCH.md](WEB_SEARCH.md) for details.

## Examples

### HTTP API
- [Basic tool calling (single round)](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/tool_calling.py)
- [Tool dispatch URL (agentic loop)](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/tool_dispatch.py)
- [Agentic tool rounds with MCP](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/agentic_tool_rounds.py)

> OpenAI API reference: https://platform.openai.com/docs/api-reference/chat/create

### Rust SDK
- [Tool calling](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/tools/main.rs)
- [Tool callbacks](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/tool_callback/main.rs)

### Python SDK
- [Tool calling notebook](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/tool_calling.ipynb)
- [Agentic tools (callbacks + max_tool_rounds)](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/agentic_tools.py)
- [Custom tool callback](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/custom_tool_call.py)
- [Custom search callback](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/custom_search.py)

## See Also

- [Agentic Features Guide](AGENTS.md): Server-side tool execution, web search, MCP, and tool dispatch
- [Web Search](WEB_SEARCH.md): Built-in web search integration
- [MCP Client](MCP/client.md): Connect to external tool servers
