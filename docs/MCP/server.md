# MCP protocol support

`mistralrs serve` can speak the **MCP – Model-Control-Protocol** in addition to the regular OpenAI-compatible REST API.

At a high-level, MCP is an opinionated, tool-based JSON-RPC 2.0 protocol that lets clients interact with models through structured *tool calls* instead of specialised HTTP routes.  
The implementation in Mistral.rs is powered by [`rust-mcp-sdk`](https://crates.io/crates/rust-mcp-sdk) and automatically registers tools based on the modalities supported by the loaded model (text, vision, …).

Exposed tools:

| Tool | Minimum `input` -> `output` modalities | Description |
| -- | -- | -- |
| `chat` | `Text` -> `Text` | Wraps the OpenAI `/v1/chat/completions` endpoint |


---

## ToC
- [MCP protocol support](#mcp-protocol-support)
  - [ToC](#toc)
  - [Running](#running)
  - [Check if it's working](#check-if-its-working)
  - [Example clients](#example-clients)
    - [Python](#python)
    - [Rust](#rust)
    - [HTTP](#http)
  - [Limitations \& roadmap](#limitations--roadmap)

---

## Running

Start the normal HTTP server and add the `--mcp-port` flag to expose an MCP endpoint **in parallel** on a separate port:

```bash
mistralrs serve \
  -p 1234 \
  --mcp-port 4321 \
  -m mistralai/Mistral-7B-Instruct-v0.3
```

## Check if it's working

The following `curl` command lists the tools advertised by the server and therefore serves as a quick smoke-test:

```
curl -X POST http://localhost:4321/mcp \
-H "Content-Type: application/json" \
-d '{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}'      
```

## Example clients


### Python

The [reference Python SDK](https://pypi.org/project/mcp/) can be installed via:

```bash
pip install --upgrade mcp
```

Here is a minimal end-to-end example that initialises a session, lists the available tools and finally sends a chat request:

```python
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


SERVER_URL = "http://localhost:4321/mcp"


async def main() -> None:
    # The helper creates an SSE (Server-Sent-Events) transport under the hood
    async with streamablehttp_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:

            # --- INITIALIZE ---
            init_result = await session.initialize()
            print("Server info:", init_result.serverInfo)

            # --- LIST TOOLS ---
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            # --- CALL TOOL ---
            resp = await session.call_tool(
                "chat",
                arguments={
                    "messages": [
                        {"role": "user", "content": "Hello MCP 👋"},
                        {"role": "assistant", "content": "Hi there!"}
                    ],
                    "maxTokens": 50,
                    "temperature": 0.7,
                },
            )
            # resp.content is a list[CallToolResultContentItem]; extract text parts
            text = "\n".join(c.text for c in resp.content if c.type == "text")
            print("Model replied:", text)

if __name__ == "__main__":
    asyncio.run(main())
```

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
        "http://localhost:4321/mcp",
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

### HTTP

**Call a tool:**
```bash
curl -X POST http://localhost:4321/mcp \
-H "Content-Type: application/json" \
-d '{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "chat",
    "arguments": {
    "messages": [
      { "role": "system",    "content": "You are a helpful assistant." },
      { "role": "user",      "content": "Hello, what’s the time?" }
    ],
    "maxTokens": 50,
    "temperature": 0.7
  }
  }
}'
```

**Initialize:**
```bash
curl -X POST http://localhost:4321/mcp \
-H "Content-Type: application/json" \
-d '{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {}
}'         
```

**List tools:**
```bash
curl -X POST http://localhost:4321/mcp \
-H "Content-Type: application/json" \
-d '{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}'      
```

## Limitations & roadmap

The MCP support that ships with the current Mistral.rs release focuses on the **happy-path**.  A few niceties have not yet been implemented and PRs are more than welcome:

1. Streaming token responses (similar to the `stream=true` flag in the OpenAI API).
2. An authentication layer – if you are exposing the MCP port publicly run it behind a reverse-proxy that handles auth (e.g.  nginx + OIDC).
3. Additional tools for other modalities such as vision or audio once the underlying crates stabilise.

If you would like to work on any of the above please open an issue first so the work can be coordinated.
