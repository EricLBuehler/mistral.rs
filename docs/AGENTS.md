# Agentic Features Guide

mistral.rs can execute tools on behalf of the model in a server-side loop, eliminating client round-trips. This guide walks through each agentic capability from simplest to most advanced.

Give a local model web search in one command:
```bash
mistralrs run --enable-search -m Qwen/Qwen3-4B
```

Or use the standard OpenAI Python SDK. No custom client needed:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1/", api_key="foobar")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What happened in the news today?"}],
    web_search_options={},
)
```

| I want to... | Use... | Section |
|---|---|---|
| Add search to any model, zero code | Web Search | [Jump](#web-search) |
| Run my own functions server-side | Tool Callbacks | [Jump](#tool-callbacks) |
| Build a full agent in Rust | Agent Builder | [Jump](#agent-builder-rust-sdk) |
| Connect to external tool servers | MCP Client | [Jump](#mcp-client) |
| Dispatch tools over HTTP in production | Tool Dispatch URLs | [Jump](#tool-dispatch-urls) |

## Table of Contents

1. [Overview](#overview)
   - [What makes a request "agentic"](#what-makes-a-request-agentic)
   - [The agentic loop](#the-agentic-loop)
   - [Dispatch order](#dispatch-order)
   - [Supported models](#supported-models)
2. [Build a Full Agent in 5 Minutes](#build-a-full-agent-in-5-minutes)
3. [Web Search](#web-search)
   - [Enabling web search](#enabling-web-search)
   - [Custom search callbacks](#custom-search-callbacks)
4. [Tool Callbacks](#tool-callbacks)
   - [Python SDK](#tool-callbacks-python-sdk)
   - [Rust SDK](#tool-callbacks-rust-sdk)
5. [Agent Builder (Rust SDK)](#agent-builder-rust-sdk)
   - [The `#[tool]` macro](#the-tool-macro)
   - [Building an agent](#building-an-agent)
   - [Non-streaming execution](#non-streaming-execution)
   - [Streaming execution](#streaming-execution)
6. [MCP Client](#mcp-client)
   - [Quick start with the CLI](#quick-start-with-the-cli)
   - [Rust SDK](#mcp-rust-sdk)
   - [Python SDK](#mcp-python-sdk)
   - [HTTP API](#mcp-http-api)
   - [Multi-server and authentication](#multi-server-and-authentication)
7. [Tool Dispatch URLs](#tool-dispatch-urls)
   - [Security model](#security-model)
   - [Setting up a tool server](#setting-up-a-tool-server)
   - [Configuration](#tool-dispatch-configuration)
8. [Common Configuration](#common-configuration)
   - [`max_tool_rounds`](#max_tool_rounds)
   - [Grammar enforcement and strict mode](#grammar-enforcement-and-strict-mode)
   - [Streaming support](#streaming-support)
9. [Combining Features](#combining-features)
10. [Further Reading](#further-reading)

---

## Overview

### What makes a request "agentic"

In **basic** tool calling, the model generates a tool call, the server returns it to your code, and you execute the tool yourself before sending the result back. This is the standard OpenAI flow.

In **agentic** tool calling, the server executes tools automatically and feeds results back to the model in a loop. No client round-trips needed. You get a final text answer instead of intermediate tool calls.

For basic tool calling details, see [TOOL_CALLING.md](TOOL_CALLING.md).

### The agentic loop

1. Model generates a response. If it includes tool calls, continue; otherwise return the response.
2. Server dispatches each tool call to the appropriate handler.
3. Tool results are appended to the conversation as `tool` messages.
4. Model generates again with the updated conversation.
5. Repeat until the model produces a final text response or `max_tool_rounds` is reached.

### Dispatch order

When the model calls a tool, the server tries these handlers in order:

1. **Built-in search tools**: `search_the_web` and `website_content_extractor` (if web search is enabled)
2. **Registered callbacks**: tool callbacks from the Python/Rust SDK or auto-registered MCP tools
3. **Tool dispatch URL**: POSTs the tool call to your HTTP endpoint
4. **No handler found**: the loop stops and the un-executed tool call is returned to the client

### Supported models

Agentic features work with any model that supports tool calling. See [TOOL_CALLING.md](TOOL_CALLING.md#supported-models) for the full list.

---

## Web Search

The simplest agentic feature: flip one flag and the model can search the web. mistral.rs uses DuckDuckGo for search and [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) for result reranking. No tool schemas needed from the user; the search tools are injected automatically.

### Enabling web search

**CLI:**
```bash
mistralrs run --enable-search --isq 4 -m Qwen/Qwen3-4B
```

**HTTP API:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1/", api_key="foobar")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is the weather forecast for Boston?"}],
    tool_choice="auto",
    max_tokens=1024,
    web_search_options={},
)
print(response.choices[0].message.content)
```

**Python SDK:**
```python
from mistralrs import Runner, Which, Architecture, ChatCompletionRequest, WebSearchOptions

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B", arch=Architecture.Qwen3),
    enable_search=True,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "What is the weather forecast for Boston?"}],
        max_tokens=256,
        web_search_options=WebSearchOptions(),
    )
)
print(res.choices[0].message.content)
```

**Rust SDK:**
```rust
use mistralrs::{
    IsqBits, ModelBuilder, RequestBuilder, SearchEmbeddingModel,
    TextMessageRole, TextMessages, WebSearchOptions,
};

let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_auto_isq(IsqBits::Four)
    .with_search(SearchEmbeddingModel::default())
    .build()
    .await?;

let messages = TextMessages::new()
    .add_message(TextMessageRole::User, "What is the weather forecast for Boston?");
let request = RequestBuilder::from(messages)
    .with_web_search_options(WebSearchOptions::default());

let response = model.send_chat_request(request).await?;
println!("{}", response.choices[0].message.content.as_ref().unwrap());
```

### Custom search callbacks

Override the default DuckDuckGo search with your own function. The callback receives a query and returns a list of results with `title`, `description`, `url`, and `content` fields.

**Python SDK:**
```python
def my_search(query: str) -> list[dict[str, str]]:
    return [{"title": "Result", "description": "...", "url": "https://...", "content": "..."}]

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B", arch=Architecture.Qwen3),
    enable_search=True,
    search_callback=my_search,
)
```

**Rust SDK:**
```rust
let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_auto_isq(IsqBits::Four)
    .with_search_callback(Arc::new(|params: &mistralrs::SearchFunctionParameters| {
        // Return Vec<SearchResult> with title, description, url, content
        my_custom_search(&params.query)
    }))
    .build()
    .await?;
```

For full details, see [WEB_SEARCH.md](WEB_SEARCH.md).

---

## Tool Callbacks

Register named functions at model-build time. When the model calls a tool by that name during the agentic loop, your function runs server-side. Set `max_tool_rounds` on the request to activate the loop.

### Tool callbacks: Python SDK

```python
import json
from mistralrs import Runner, Which, Architecture, ChatCompletionRequest, ToolChoice

def tool_callback(name: str, args: dict) -> str:
    """Dispatch tool calls to local implementations."""
    if name == "get_weather":
        city = args.get("city", "unknown")
        return json.dumps({"city": city, "temp": 22, "condition": "Sunny"})
    if name == "calculate":
        expression = args.get("expression", "0")
        return json.dumps({"result": eval(expression)})
    return json.dumps({"error": f"Unknown tool: {name}"})

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B", arch=Architecture.Qwen3),
    tool_callbacks={"get_weather": tool_callback, "calculate": tool_callback},
)

tools = [
    json.dumps({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
            "strict": True,
        },
    }),
    json.dumps({
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Expression"}},
                "required": ["expression"],
            },
            "strict": True,
        },
    }),
]

request = ChatCompletionRequest(
    messages=[{"role": "user", "content": "What's the weather in Tokyo? Also calculate 42 * 17."}],
    model="default",
    tool_schemas=tools,
    tool_choice=ToolChoice.Auto,
    max_tool_rounds=5,
)
response = runner.send_chat_completion_request(request)
print(response.choices[0].message.content)
```

### Tool callbacks: Rust SDK

```rust
use mistralrs::{
    CalledFunction, IsqBits, ModelBuilder, RequestBuilder,
    TextMessageRole, TextMessages, Tool, ToolChoice, ToolType,
};
use std::sync::Arc;

let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_auto_isq(IsqBits::Four)
    .with_tool_callback(
        "get_weather",
        Arc::new(|f: &CalledFunction| {
            let args: serde_json::Value = serde_json::from_str(&f.arguments)?;
            let city = args["city"].as_str().unwrap_or("unknown");
            Ok(format!(r#"{{"city":"{}","temp":22,"condition":"Sunny"}}"#, city))
        }),
    )
    .build()
    .await?;

let tool = Tool {
    tp: ToolType::Function,
    function: mistralrs::Function {
        description: Some("Get the current weather for a city.".to_string()),
        name: "get_weather".to_string(),
        parameters: Some(std::collections::HashMap::from([(
            "city".to_string(),
            serde_json::json!({"type": "string", "description": "City name"}),
        )])),
        strict: Some(true),
    },
};

let messages = TextMessages::new()
    .add_message(TextMessageRole::User, "What's the weather in Tokyo?");
let request = RequestBuilder::from(messages)
    .set_tools(vec![tool])
    .set_tool_choice(ToolChoice::Auto)
    .set_max_tool_rounds(5);

let response = model.send_chat_request(request).await?;
println!("{}", response.choices[0].message.content.as_ref().unwrap());
```

---

## Agent Builder (Rust SDK)

The Agent Builder is a Rust SDK-only abstraction that wraps the model, tool definitions, and the agentic loop into a single `Agent` object. It provides a higher-level API compared to `RequestBuilder` with manual tool callbacks.

### The `#[tool]` macro

Annotate a function with `#[tool]` to generate tool definitions and callbacks automatically. The macro produces three items: `<name>_tool()`, `<name>_callback()`, and `<name>_tool_with_callback()`.

Both sync and async functions are supported. Sync tools run in `spawn_blocking`; async tools run natively.

```rust
use mistralrs::tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherInfo {
    temperature: f32,
    conditions: String,
    humidity: u8,
}

#[tool(description = "Get the current weather for a location")]
fn get_weather(
    #[description = "The city name to get weather for"] city: String,
    #[description = "Temperature unit: 'celsius' or 'fahrenheit'"]
    #[default = "celsius"]
    unit: Option<String>,
) -> Result<WeatherInfo> {
    let temp = match unit.as_deref() {
        Some("fahrenheit") => 72.5,
        _ => 22.5,
    };
    Ok(WeatherInfo {
        temperature: temp,
        conditions: format!("Sunny with clear skies in {}", city),
        humidity: 45,
    })
}

// Async tools are also supported:
#[tool(description = "Search the web for information on a topic")]
async fn web_search(
    #[description = "The search query"] query: String,
    #[description = "Maximum number of results"]
    #[default = 3u32]
    max_results: Option<u32>,
) -> Result<Vec<SearchResult>> {
    // async implementation...
}
```

### Building an agent

```rust
use mistralrs::{AgentBuilder, IsqBits, ModelBuilder, PagedAttentionMetaBuilder};

let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_auto_isq(IsqBits::Four)
    .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
    .build()
    .await?;

let agent = AgentBuilder::new(model)
    .with_system_prompt(
        "You are a helpful assistant with access to weather and web search tools.",
    )
    .with_max_iterations(5)
    .with_parallel_tool_execution(true)
    .register_tool(get_weather_tool_with_callback())
    .register_tool(web_search_tool_with_callback())
    .build();
```

### Non-streaming execution

```rust
let response = agent.run("What's the weather in Boston?").await?;

if let Some(text) = &response.final_response {
    println!("{}", text);
}

println!("Iterations: {}", response.iterations);
println!("Stop reason: {:?}", response.stop_reason);

for (i, step) in response.steps.iter().enumerate() {
    println!("Step {}: {} tool call(s)", i + 1, step.tool_calls.len());
    for call in &step.tool_calls {
        println!("  - {}: {}", call.function.name, call.function.arguments);
    }
}
```

`AgentStopReason` tells you why the agent stopped:
- `TextResponse`: model produced a final text answer
- `MaxIterations`: hit the iteration limit
- `NoAction`: model produced no response
- `Error(e)`: an error occurred

### Streaming execution

```rust
use mistralrs::AgentEvent;
use std::io::Write;

let mut stream = agent.run_stream("What's the weather in Boston?").await?;

while let Some(event) = stream.next().await {
    match event {
        AgentEvent::TextDelta(text) => {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
        AgentEvent::ToolCallsStart(calls) => {
            println!("\n[Calling {} tool(s)...]", calls.len());
            for call in &calls {
                println!("  - {}: {}", call.function.name, call.function.arguments);
            }
        }
        AgentEvent::ToolResult(result) => {
            let status = if result.result.is_ok() { "OK" } else { "ERROR" };
            println!("  [Tool {} completed: {}]", result.tool_name, status);
        }
        AgentEvent::ToolCallsComplete => {
            println!("[All tools completed, continuing...]");
        }
        AgentEvent::Complete(response) => {
            println!("\nDone in {} iteration(s)", response.iterations);
        }
    }
}
```

---

## MCP Client

Connect to external [MCP](https://modelcontextprotocol.io/) servers to give the model access to tools like filesystem operations, databases, APIs, and more. Tools are auto-discovered from connected servers at startup.

Three transport types are supported:
- **Process**: local command (stdin/stdout communication)
- **HTTP**: remote JSON-RPC over HTTP
- **WebSocket**: remote real-time communication

### Quick start with the CLI

Create `mcp-config.json`:
```json
{
  "servers": [
    {
      "name": "Filesystem Tools",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "."]
      }
    }
  ]
}
```

Start the server:
```bash
mistralrs serve -p 1234 --mcp-config mcp-config.json --max-tool-rounds 5 -m Qwen/Qwen3-4B
```

MCP tools are now available to the model automatically.

### MCP: Rust SDK

```rust
use mistralrs::{
    IsqBits, McpClientConfig, McpServerConfig, McpServerSource,
    ModelBuilder, TextMessageRole, TextMessages,
};

let mcp_config = McpClientConfig {
    servers: vec![McpServerConfig {
        name: "Filesystem Tools".to_string(),
        source: McpServerSource::Process {
            command: "npx".to_string(),
            args: vec![
                "@modelcontextprotocol/server-filesystem".to_string(),
                ".".to_string(),
            ],
            work_dir: None,
            env: None,
        },
        ..Default::default()
    }],
    ..Default::default()
};

let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_auto_isq(IsqBits::Four)
    .with_mcp_client(mcp_config)
    .build()
    .await?;

let messages = TextMessages::new()
    .add_message(TextMessageRole::User, "List the files in the current directory.");
let response = model.send_chat_request(messages).await?;
println!("{}", response.choices[0].message.content.as_ref().unwrap());
```

### MCP: Python SDK

```python
import asyncio
import mistralrs

mcp_config = mistralrs.McpClientConfigPy(
    servers=[
        mistralrs.McpServerConfigPy(
            name="Filesystem Tools",
            source=mistralrs.McpServerSourcePy.Process(
                command="npx",
                args=["@modelcontextprotocol/server-filesystem", "."],
                work_dir=None,
                env=None,
            ),
        )
    ]
)

runner = mistralrs.Runner(
    which=mistralrs.Which.Plain(
        model_id="Qwen/Qwen3-4B",
        arch=mistralrs.Architecture.Qwen3,
    ),
    mcp_client_config=mcp_config,
)

request = mistralrs.ChatCompletionRequest(
    model="default",
    messages=[{"role": "user", "content": "List the files in the current directory."}],
    max_tokens=1000,
    tool_choice="auto",
)

response = asyncio.run(runner.send_chat_completion_request(request))
print(response.choices[0].message.content)
```

### MCP: HTTP API

Once the server is started with `--mcp-config`, MCP tools are available to all requests:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="placeholder")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List the files in the current directory."}],
    tool_choice="auto",
    extra_body={"max_tool_rounds": 5},
)
print(response.choices[0].message.content)
```

### Multi-server and authentication

Connect to multiple MCP servers with different transports. Use `tool_prefix` to avoid naming conflicts and `bearer_token` for authentication.

```rust
let mcp_config = McpClientConfig {
    servers: vec![
        McpServerConfig {
            name: "Filesystem Tools".to_string(),
            source: McpServerSource::Process {
                command: "npx".to_string(),
                args: vec!["@modelcontextprotocol/server-filesystem".to_string(), ".".to_string()],
                work_dir: None,
                env: None,
            },
            tool_prefix: Some("fs".to_string()),
            ..Default::default()
        },
        McpServerConfig {
            id: "hf_server".to_string(),
            name: "Hugging Face MCP".to_string(),
            source: McpServerSource::Http {
                url: "https://hf.co/mcp".to_string(),
                timeout_secs: Some(30),
                headers: None,
            },
            tool_prefix: Some("hf".to_string()),
            bearer_token: Some("hf_xxx".to_string()),
            ..Default::default()
        },
        McpServerConfig {
            id: "ws_server".to_string(),
            name: "WebSocket Server".to_string(),
            source: McpServerSource::WebSocket {
                url: "wss://api.example.com/mcp".to_string(),
                timeout_secs: Some(30),
                headers: None,
            },
            tool_prefix: Some("ws".to_string()),
            ..Default::default()
        },
    ],
    auto_register_tools: true,
    tool_timeout_secs: Some(30),
    max_concurrent_calls: Some(5),
};
```

For full MCP configuration details, see [MCP/client.md](MCP/client.md), [MCP/configuration.md](MCP/configuration.md), and [MCP/transports.md](MCP/transports.md).

---

## Tool Dispatch URLs

The tool dispatch URL lets the server POST unhandled tool calls to your HTTP endpoint. This is how HTTP API users get server-side tool execution without registering SDK-level callbacks or MCP servers.

### Security model

To prevent SSRF, `tool_dispatch_url` **cannot** be set per-request via the HTTP API. It can only be configured:
- Server-side via the `--tool-dispatch-url` CLI flag
- Per-request in trusted Rust/Python SDK code

### Setting up a tool server

The server POSTs to your endpoint with this format:

**Request:**
```json
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
```

**Response:**
```json
{"content": "Sunny, 22C"}
```

The response can also be a bare string instead of a JSON object.

Here is a minimal Python tool server:

```python
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

def get_weather(args):
    city = args.get("city", "unknown")
    return f"Sunny, 22C in {city}"

TOOLS = {"get_weather": get_weather}

class ToolHandler(BaseHTTPRequestHandler):
    def do_POST(self, *_args):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = body["name"]
        result = TOOLS.get(name, lambda a: f"Unknown tool: {name}")(body.get("arguments", {}))
        response = json.dumps({"content": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

HTTPServer(("localhost", 8787), ToolHandler).serve_forever()
```

### Tool dispatch configuration

**CLI:**
```bash
mistralrs serve -p 1234 \
    --tool-dispatch-url http://localhost:8787/tools \
    --max-tool-rounds 5 \
    -m Qwen/Qwen3-4B
```

**HTTP API** (with the server started above):
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="placeholder")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo and London?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": True,
        },
    }],
    tool_choice="auto",
    extra_body={"max_tool_rounds": 5},
)
print(response.choices[0].message.content)
```

**Rust SDK** (per-request, trusted code):
```rust
let request = RequestBuilder::from(messages)
    .set_tools(tools)
    .set_tool_choice(ToolChoice::Auto)
    .set_tool_dispatch_url("http://localhost:8787/tools")
    .set_max_tool_rounds(5);
```

**Python SDK** (per-request, trusted code):
```python
request = ChatCompletionRequest(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    model="default",
    tool_schemas=tools,
    tool_choice=ToolChoice.Auto,
    tool_dispatch_url="http://localhost:8787/tools",
    max_tool_rounds=5,
)
```

---

## Common Configuration

### `max_tool_rounds`

Controls how many iterations the agentic loop can run. The loop terminates early if the model stops calling tools.

| Setting | Value |
|---------|-------|
| Default (unset) | 16 (safety cap) |
| CLI flag | `--max-tool-rounds <N>` |
| Per-request (HTTP) | `"max_tool_rounds": N` in request body |
| Per-request (Rust) | `.set_max_tool_rounds(N)` |
| Per-request (Python) | `max_tool_rounds=N` |

Per-request values override the server default.

### Grammar enforcement and strict mode

When tools are provided, mistral.rs automatically constrains output to valid tool call syntax using [llguidance](https://github.com/guidance-ai/llguidance). No configuration needed.

For stricter argument validation, set `"strict": true` on the function definition. This enforces the tool's JSON schema on the generated arguments: only declared property names, correct types, valid enum values, and required fields. See [TOOL_CALLING.md](TOOL_CALLING.md#strict-mode) for details.

### Streaming support

All agentic features (web search, callbacks, MCP, dispatch URLs) support streaming. Tool call chunks are forwarded to the client during the loop so you can observe which tools are being called in real time.

---

## Combining Features

You can enable multiple agentic features on the same model. The [dispatch order](#dispatch-order) governs which handler runs for each tool call.

**Rust SDK:**
```rust
let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_auto_isq(IsqBits::Four)
    .with_search(SearchEmbeddingModel::default())      // Built-in web search
    .with_tool_callback("local_db", Arc::new(db_cb))   // Custom callback
    .with_mcp_client(mcp_config)                        // MCP servers
    .build()
    .await?;
```

**CLI:**
```bash
mistralrs serve -p 1234 \
    --enable-search \
    --mcp-config mcp-config.json \
    --tool-dispatch-url http://localhost:8787/tools \
    --max-tool-rounds 10 \
    -m Qwen/Qwen3-4B
```

---

## Further Reading

| Topic | Link |
|-------|------|
| Tool calling reference | [TOOL_CALLING.md](TOOL_CALLING.md) |
| Web search reference | [WEB_SEARCH.md](WEB_SEARCH.md) |
| MCP client reference | [MCP/client.md](MCP/client.md) |
| MCP configuration | [MCP/configuration.md](MCP/configuration.md) |
| MCP transports | [MCP/transports.md](MCP/transports.md) |
| MCP advanced usage | [MCP/advanced.md](MCP/advanced.md) |

### Examples

**Rust SDK:**
- [Agent (non-streaming)](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/agent/main.rs)
- [Agent (streaming)](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/agent_streaming/main.rs)
- [Tool callbacks](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/tool_callback/main.rs)
- [Web search](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/web_search/main.rs)
- [Custom search callback](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/search_callback/main.rs)
- [MCP client](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/mcp_client/main.rs)

**Python SDK:**
- [Agentic tools (callbacks)](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/agentic_tools.py)
- [Web search](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/web_search.py)
- [Custom search callback](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/custom_search.py)
- [MCP client](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/mcp_client.py)

**HTTP API:**
- [Basic tool calling](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/tool_calling.py)
- [Tool dispatch URL](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/tool_dispatch.py)
- [Agentic tool rounds](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/agentic_tool_rounds.py)
