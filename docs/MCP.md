# MCP protocol support

`mistralrs-server` can serve **MCP (Model Control Protocol)** traffic next to the regular OpenAI-compatible HTTP interface.  

MCP is an open, tool-based protocol that lets clients interact with models through structured *tool calls* instead of free-form HTTP routes.  

Under the hood the server uses [`rust-mcp-sdk`](https://crates.io/crates/rust-mcp-sdk) and exposes a single tool called **`chat`** that mirrors the behaviour of the `/v1/chat/completions` endpoint.

---

## 1. Building

Support for MCP is compiled in by default because the workspace enables the `server` and `hyper-server` features of `rust-mcp-sdk`.  
When you only compile the `mistralrs-server` crate outside the workspace enable the `mcp-server` Cargo feature manually:

```bash
cargo build -p mistralrs-server --release --features "mcp-server"
```

## 2. Running

Start the normal HTTP server and add the `--mcp-port` flag to spin up an MCP server on a separate port:

```bash
./target/release/mistralrs-server \
  --port 1234            # OpenAI compatible HTTP API
  --mcp-port 4321        # MCP protocol endpoint (SSE over HTTP)
  plain -m mistralai/Mistral-7B-Instruct-v0.3
```

* `--mcp-port` takes precedence over `--port` – you can run the HTTP and MCP servers on totally independent ports or omit `--port` when you only need MCP.*

## 3. Capabilities announced to clients

At start-up the MCP handler advertises the following `InitializeResult` (abridged):

```jsonc
{
  "server_info": { "name": "mistralrs", "version": "<crate-version>" },
  "protocol_version": "2025-03-26",            // latest spec version from rust-mcp-sdk
  "instructions": "use tool 'chat'",
  "capabilities": {
    "tools": {}
  }
}
```

Only one tool is currently exposed:

| tool | description                                          |
|------|------------------------------------------------------|
| `chat` | Wraps the OpenAI `/v1/chat/completions` endpoint. |

## 4. Calling the `chat` tool

Clients send a [`CallToolRequest`](https://docs.rs/rust-mcp-schema/latest/rust_mcp_schema/struct.CallToolRequest.html) event where `params.name` is `"chat"` and `params.arguments` contains a standard MCP [`CreateMessageRequest`](https://docs.rs/rust-mcp-schema/latest/rust_mcp_schema/struct.CreateMessageRequest.html).

Example request (sent as SSE `POST /mcp/stream` or via the convenience helpers in `rust-mcp-sdk`):

```jsonc
{
  "kind": "callToolRequest",
  "id": "123",
  "params": {
    "name": "chat",
    "arguments": {
      "model": "mistralai/Mistral-7B-Instruct-v0.3",
      "messages": [
        { "role": "user", "content": "Explain Rust ownership." }
      ]
    }
  }
}
```

The response is a `CallToolResult` event whose `content` array contains a single `TextContent` item with the assistant response.

```jsonc
{
  "kind": "callToolResult",
  "id": "123",
  "content": [
    { "type": "text", "text": "Rust’s ownership system ..." }
  ]
}
```

Error cases are mapped to `CallToolError` with `is_error = true`.

## 5. Example clients

### Rust

```rust
use anyhow::Result;
use rust_mcp_sdk::{
    mcp_client::client_runtime,
    schema::{
        CallToolRequestParams, ClientCapabilities, CreateMessageRequest,
        Implementation, InitializeRequestParams, Message, LATEST_PROTOCOL_VERSION,
    },
    ClientSseTransport, ClientSseTransportOptions,
};

struct Handler;
#[async_trait::async_trait]
impl rust_mcp_sdk::mcp_client::ClientHandler for Handler {}

#[tokio::main]
async fn main() -> Result<()> {
    let transport = ClientSseTransport::new(
        "http://localhost:4321/mcp/stream",
        ClientSseTransportOptions::default(),
    )?;

    let details = InitializeRequestParams {
        capabilities: ClientCapabilities::default(),
        client_info: Implementation { name: "mcp-client".into(), version: "0.1".into() },
        protocol_version: LATEST_PROTOCOL_VERSION.into(),
    };

    let client = client_runtime::create_client(details, transport, Handler);
    client.clone().start().await?;

    let req = CreateMessageRequest {
        model: "mistralai/Mistral-7B-Instruct-v0.3".into(),
        messages: vec![Message::user("Explain Rust ownership.")],
        ..Default::default()
    };

    let result = client
        .call_tool(CallToolRequestParams::new("chat", req.into()))
        .await?;

    println!("{}", result.content[0].as_text_content()?.text);
    client.shut_down().await?;
    Ok(())
}
```

### Python

```py
import json
import requests
from sseclient import SSEClient

payload = {
    "kind": "callToolRequest",
    "id": "123",
    "params": {
        "name": "chat",
        "arguments": {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [
                {"role": "user", "content": "Explain Rust ownership."}
            ],
        },
    },
}

resp = requests.post(
    "http://localhost:4321/mcp/stream",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
    stream=True,
)
for event in SSEClient(resp):
    print(event.data)
```

### HTTP

```bash
curl -N -X POST http://localhost:4321/mcp/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "callToolRequest",
    "id": "123",
    "params": {
      "name": "chat",
      "arguments": {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "Explain Rust ownership."}]
      }
    }
  }'
```

## 6. Limitations & future work

• Only synchronous, single-shot requests are supported right now.  
• Streaming responses (`partialCallToolResult`) are not yet implemented.  
• No authentication layer is provided – run the MCP port behind a reverse proxy if you need auth.

Contributions to extend MCP coverage (streaming, more tools, auth hooks) are welcome!
