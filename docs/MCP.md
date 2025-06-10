# MCP protocol support

`mistralrs-server` can serve **MCP (Model Control Protocol)** traffic next to the regular OpenAI-compatible HTTP interface!

MCP is an open, tool-based protocol that lets clients interact with models through structured *tool calls* instead of free-form HTTP routes.  

Under the hood the server uses [`rust-mcp-sdk`](https://crates.io/crates/rust-mcp-sdk) and exposes tools based on the supported modalities of the loaded model.

Exposed tools:

| Tool | Minimum `input` -> `output` modalities | Description |
| -- | -- | -- |
| `chat` | `Text` -> `Text` | Wraps the OpenAI `/v1/chat/completions` endpoint. |


---

## ToC
- [MCP protocol support](#mcp-protocol-support)
  - [ToC](#toc)
  - [Running](#running)
  - [Check if it's working](#check-if-its-working)
  - [Example clients](#example-clients)
    - [Rust](#rust)
    - [Python](#python)
    - [HTTP](#http)
  - [Limitations](#limitations)

---

## Running

Start the normal HTTP server and add the `--mcp-port` flag to spin up an MCP server on a separate port:

```bash
./target/release/mistralrs-server \
  --port 1234            # OpenAI compatible HTTP API
  --mcp-port 4321        # MCP protocol endpoint (Streamable HTTP)
  plain -m mistralai/Mistral-7B-Instruct-v0.3
```

## Check if it's working

Run this `curl` command to check the available tools:

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

### Python

```py
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SERVER_URL = "http://localhost:4321/mcp"

async def main() -> None:
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

## Limitations

- Streaming requests are not implemented.
- No authentication layer is provided – run the MCP port behind a reverse proxy if you need auth.

Contributions to extend MCP coverage (streaming, more tools, auth hooks) are welcome!
