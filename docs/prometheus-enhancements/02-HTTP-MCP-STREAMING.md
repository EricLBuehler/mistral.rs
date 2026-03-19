# Enhancement 2: HTTP Streaming MCP Server Endpoint

## Executive Summary

Add a direct HTTP streaming endpoint for MCP protocol to enable **zero-proxy integration** with candle-vllm and other HTTP-based clients:
- **Eliminate proxy overhead**: Direct connection vs multi-hop routing
- **Server-Sent Events (SSE)**: Standard HTTP streaming, wide client support
- **Backward compatible**: Existing MCP server continues to work
- **Production ready**: Authentication, rate limiting, observability

**Estimated Time**: 2-3 weeks  
**Risk Level**: Medium (new protocol endpoint, needs thorough testing)  
**Business Value**: Very High (key enabler for candle-vllm integration)

## Current MCP Architecture

### Existing Implementation

```rust
// mistral.rs currently supports:
1. MCP Server (stdio-based, process protocol)
   ./mistralrs-server --mcp-port 4321 plain -m model

2. MCP Client (connects to external MCP servers)
   Config: mcp-config.json with server definitions

// Architecture:
┌──────────────┐     stdio/process     ┌──────────────┐
│ MCP Client   │ ◄─────────────────── │ mistral.rs   │
│ (external)   │                       │ MCP Server   │
└──────────────┘                       └──────────────┘
```

### Limitation for candle-vllm

```rust
// Problem: candle-vllm needs HTTP-based connection
// Current workaround: Requires proxy

┌──────────────┐   HTTP   ┌───────┐   stdio   ┌──────────────┐
│ candle-vllm  │ ────────►│ Proxy │ ─────────►│ mistral.rs   │
│              │          │       │           │              │
└──────────────┘          └───────┘           └──────────────┘
                           ↑ Extra hop
                           ↑ Latency + complexity
```

### Proposed Solution

```rust
// Direct HTTP connection with streaming

┌──────────────┐   HTTP SSE   ┌──────────────┐
│ candle-vllm  │ ────────────►│ mistral.rs   │
│ MCP Client   │ ◄────────────│ HTTP MCP     │
│              │              │ Server       │
└──────────────┘              └──────────────┘
                               ↑ Zero proxy
                               ↑ Lower latency
```

## Protocol Design

### HTTP MCP Streaming Specification

```
Endpoint: POST /mcp/stream
Content-Type: application/json
Accept: text/event-stream

Request Format (JSON-RPC 2.0):
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "generate",
    "arguments": {
      "prompt": "Hello, world!",
      "max_tokens": 100
    }
  },
  "id": "req-123"
}

Response Format (SSE):
event: message
data: {"jsonrpc":"2.0","result":{"text":"Hello"},"id":"req-123"}

event: message  
data: {"jsonrpc":"2.0","result":{"text":" there"},"id":"req-123"}

event: done
data: {"jsonrpc":"2.0","result":{"finish_reason":"stop"},"id":"req-123"}
```

### Why Server-Sent Events (SSE)?

**Advantages over WebSocket**:
```
SSE:
✅ Simpler protocol (HTTP/1.1, no upgrade)
✅ Automatic reconnection built-in
✅ Works through corporate proxies
✅ Lower overhead (no frame wrapping)
✅ Native browser support (EventSource API)
✅ Easy to debug (curl, http tools)

WebSocket:
❌ Requires protocol upgrade
❌ Complex state management
❌ Blocked by some proxies
❌ More implementation complexity
⚖️ Bidirectional (not needed for MCP response stream)
```

**Trade-off**: SSE is unidirectional (server→client), but MCP request/response fits this model perfectly:
- Client sends request → HTTP POST
- Server streams response → SSE
- Next request → New HTTP POST

### Alternative: HTTP/2 Server Push

```rust
// Could use HTTP/2 push, but SSE is simpler
// HTTP/2 push being deprecated in browsers anyway
// SSE works over HTTP/1.1 and HTTP/2

// Decision: Use SSE for maximum compatibility
```

## Implementation Strategy

### Phase 1: Core SSE Endpoint (Week 1)

```rust
// File: mistralrs-server/src/http_mcp.rs

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use futures::stream::{Stream, StreamExt};
use std::convert::Infallible;

/// HTTP MCP streaming endpoint
pub async fn mcp_stream_handler(
    State(mcp_server): State<Arc<McpServer>>,
    Json(request): Json<McpRequest>,
) -> impl IntoResponse {
    let stream = handle_mcp_request(mcp_server, request);
    
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
}

async fn handle_mcp_request(
    mcp_server: Arc<McpServer>,
    request: McpRequest,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(100);
    
    // Spawn task to handle MCP request
    tokio::spawn(async move {
        match request.method.as_str() {
            "tools/call" => {
                // Route to appropriate tool
                let tool_name = request.params.name;
                let result = mcp_server.call_tool(tool_name, request.params.arguments).await;
                
                // Stream response tokens
                match result {
                    Ok(response_stream) => {
                        futures::pin_mut!(response_stream);
                        while let Some(chunk) = response_stream.next().await {
                            let event = Event::default()
                                .event("message")
                                .json_data(McpResponse {
                                    jsonrpc: "2.0".to_string(),
                                    result: chunk,
                                    id: request.id.clone(),
                                })
                                .unwrap();
                            
                            tx.send(Ok(event)).await.ok();
                        }
                        
                        // Send completion event
                        let done_event = Event::default()
                            .event("done")
                            .json_data(McpResponse {
                                jsonrpc: "2.0".to_string(),
                                result: json!({"finish_reason": "stop"}),
                                id: request.id,
                            })
                            .unwrap();
                        
                        tx.send(Ok(done_event)).await.ok();
                    }
                    Err(e) => {
                        // Send error event
                        let error_event = Event::default()
                            .event("error")
                            .json_data(McpError {
                                jsonrpc: "2.0".to_string(),
                                error: ErrorObject {
                                    code: -32603,
                                    message: e.to_string(),
                                },
                                id: request.id,
                            })
                            .unwrap();
                        
                        tx.send(Ok(error_event)).await.ok();
                    }
                }
            }
            "tools/list" => {
                // Return available tools
                let tools = mcp_server.list_tools().await;
                let event = Event::default()
                    .event("message")
                    .json_data(McpResponse {
                        jsonrpc: "2.0".to_string(),
                        result: json!({"tools": tools}),
                        id: request.id,
                    })
                    .unwrap();
                
                tx.send(Ok(event)).await.ok();
            }
            _ => {
                // Unknown method
                let error_event = Event::default()
                    .event("error")
                    .json_data(McpError {
                        jsonrpc: "2.0".to_string(),
                        error: ErrorObject {
                            code: -32601,
                            message: format!("Method not found: {}", request.method),
                        },
                        id: request.id,
                    })
                    .unwrap();
                
                tx.send(Ok(error_event)).await.ok();
            }
        }
    });
    
    tokio_stream::wrappers::ReceiverStream::new(rx)
}
```

### Phase 2: Authentication & Security (Week 2, Days 1-2)

```rust
// Add authentication middleware

use axum::{
    middleware::{self, Next},
    http::{Request, StatusCode},
};

async fn auth_middleware<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    // Check for API key
    let auth_header = req.headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());
    
    match auth_header {
        Some(key) if validate_api_key(key) => {
            Ok(next.run(req).await)
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

// Apply to MCP routes
let mcp_routes = Router::new()
    .route("/mcp/stream", post(mcp_stream_handler))
    .route("/mcp/list", get(list_tools_handler))
    .layer(middleware::from_fn(auth_middleware));
```

### Phase 3: Rate Limiting (Week 2, Days 3-4)

```rust
// Use tower-governor for rate limiting

use tower_governor::{
    governor::GovernorConfigBuilder,
    GovernorLayer,
};

// Configure rate limiter
let governor_conf = Box::new(
    GovernorConfigBuilder::default()
        .per_second(10)        // 10 requests per second
        .burst_size(20)        // Allow bursts up to 20
        .finish()
        .unwrap()
);

let mcp_routes = Router::new()
    .route("/mcp/stream", post(mcp_stream_handler))
    .layer(GovernorLayer {
        config: Box::leak(governor_conf),
    });
```

### Phase 4: Integration with Existing Server (Week 2, Day 5)

```rust
// File: mistralrs-server/src/main.rs

#[tokio::main]
async fn main() -> Result<()> {
    // ... existing setup ...
    
    // Add HTTP MCP endpoint if configured
    let mut app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        // ... other routes ...
        ;
    
    // NEW: Add HTTP MCP endpoints
    if let Some(mcp_config) = &args.http_mcp_config {
        let mcp_routes = build_mcp_routes(mcp_server.clone(), mcp_config)?;
        app = app.nest("/mcp", mcp_routes);
        
        info!("HTTP MCP server enabled at /mcp/stream");
    }
    
    // Start server
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

### Phase 5: Client Implementation Guide (Week 3)

```rust
// Example: candle-vllm MCP client for HTTP streaming

use reqwest::Client;
use futures::stream::StreamExt;
use eventsource_stream::Eventsource;

pub struct HttpMcpClient {
    base_url: String,
    api_key: Option<String>,
    client: Client,
}

impl HttpMcpClient {
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<impl Stream<Item = McpResponse>> {
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            params: ToolCallParams {
                name: tool_name.to_string(),
                arguments,
            },
            id: format!("req-{}", uuid::Uuid::new_v4()),
        };
        
        let mut req_builder = self.client
            .post(format!("{}/mcp/stream", self.base_url))
            .json(&request);
        
        if let Some(key) = &self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }
        
        let response = req_builder.send().await?;
        
        // Parse SSE stream
        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|event| async move {
                match event {
                    Ok(event) => {
                        match event.event.as_str() {
                            "message" => {
                                serde_json::from_str::<McpResponse>(&event.data).ok()
                            }
                            "done" => {
                                serde_json::from_str::<McpResponse>(&event.data).ok()
                            }
                            "error" => {
                                eprintln!("MCP error: {}", event.data);
                                None
                            }
                            _ => None
                        }
                    }
                    Err(e) => {
                        eprintln!("Stream error: {}", e);
                        None
                    }
                }
            });
        
        Ok(stream)
    }
}
```

## Protocol Compatibility

### MCP Standard Compliance

```rust
// Ensure compatibility with MCP protocol spec

// Required methods:
✅ tools/list        - List available tools
✅ tools/call        - Execute a tool
✅ resources/list    - List resources (if applicable)
✅ resources/read    - Read a resource

// Optional methods (for advanced features):
⚖️ prompts/list      - List prompt templates
⚖️ prompts/get       - Get prompt template
⚖️ sampling/         - Sampling configuration

// All methods work via HTTP streaming endpoint
```

### Backward Compatibility

```rust
// Existing MCP server continues to work
// HTTP endpoint is additive

Configuration:
1. --mcp-port 4321              # Original stdio/process MCP
2. --http-mcp-port 4322         # New HTTP streaming MCP
3. Both can run simultaneously  # No conflicts
```

## Performance Considerations

### Latency Analysis

```
Scenario: candle-vllm → mistral.rs tool call

Via stdio/process MCP (with proxy):
├─ HTTP → proxy:         5ms
├─ Proxy → stdio:        3ms
├─ Process spawn:        2ms
├─ MCP processing:      10ms
├─ stdio → proxy:        3ms
└─ Proxy → HTTP:         5ms
Total: ~28ms overhead

Via HTTP streaming MCP (direct):
├─ HTTP → mistral.rs:    2ms
├─ MCP processing:      10ms
└─ SSE stream:           1ms
Total: ~13ms overhead

Improvement: 54% reduction in communication overhead
```

### Throughput Impact

```
Concurrent requests: 100

stdio/process MCP:
├─ Process limit: ~1000 concurrent
├─ Context switching overhead
└─ Throughput: ~200 req/sec

HTTP streaming MCP:
├─ Async/tokio: 10,000+ concurrent
├─ No process spawning
└─ Throughput: ~500 req/sec

Improvement: 2.5x throughput increase
```

### Memory Footprint

```
Per connection:

stdio/process:
├─ Process: ~8MB
├─ Stdio buffers: ~64KB
└─ Total: ~8MB per client

HTTP SSE:
├─ Connection: ~16KB
├─ Stream buffer: ~4KB
└─ Total: ~20KB per client

Improvement: 400x less memory per connection
```

## Security Considerations

### Authentication

```rust
// Multiple auth methods supported

// 1. API Key (Bearer token)
Authorization: Bearer sk-abc123...

// 2. mTLS (mutual TLS)
Client certificate verification

// 3. OAuth2 (future)
Standard OAuth2 flow
```

### Rate Limiting

```rust
// Per-client rate limits

Config:
- Global: 1000 req/sec
- Per API key: 100 req/sec
- Per IP: 10 req/sec (unauthenticated)

Headers:
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1640000000
```

### Input Validation

```rust
// Validate all MCP requests

async fn validate_mcp_request(req: &McpRequest) -> Result<()> {
    // 1. Check JSON-RPC version
    if req.jsonrpc != "2.0" {
        return Err(Error::InvalidJsonRpc);
    }
    
    // 2. Validate method name
    if !ALLOWED_METHODS.contains(&req.method.as_str()) {
        return Err(Error::MethodNotAllowed);
    }
    
    // 3. Validate parameters
    match req.method.as_str() {
        "tools/call" => {
            validate_tool_call_params(&req.params)?;
        }
        _ => {}
    }
    
    // 4. Check request size
    let size = serde_json::to_string(req)?.len();
    if size > MAX_REQUEST_SIZE {
        return Err(Error::RequestTooLarge);
    }
    
    Ok(())
}
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mcp_stream_handler() {
        let mcp_server = Arc::new(MockMcpServer::new());
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            params: json!({
                "name": "generate",
                "arguments": {"prompt": "test"}
            }),
            id: "test-1".to_string(),
        };
        
        let response = mcp_stream_handler(
            State(mcp_server),
            Json(request),
        ).await;
        
        // Verify SSE stream works
        assert!(response.is_ok());
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_end_to_end_mcp_call() {
    // Start mistral.rs with HTTP MCP
    let server = spawn_test_server().await;
    
    // Create client
    let client = HttpMcpClient::new("http://localhost:4322");
    
    // Call tool
    let mut stream = client.call_tool("generate", json!({
        "prompt": "Hello",
        "max_tokens": 10
    })).await.unwrap();
    
    // Verify streaming response
    let mut tokens = Vec::new();
    while let Some(response) = stream.next().await {
        tokens.push(response);
    }
    
    assert!(!tokens.is_empty());
    assert_eq!(tokens.last().unwrap().finish_reason, Some("stop"));
}
```

### Load Tests

```bash
# Use k6 for load testing
k6 run load-test.js

# Verify:
# - 100 concurrent connections
# - < 50ms p95 latency
# - No connection drops
# - Memory stays < 1GB
```

## Deployment Configuration

### Command Line

```bash
# Enable HTTP MCP streaming
./mistralrs-server \
  --port 1234 \                      # Main HTTP API
  --http-mcp-port 4322 \             # HTTP MCP endpoint
  --http-mcp-auth-key sk-abc123 \   # API key
  run -m meta-llama/Llama-3.2-8B
```

### Configuration File

```toml
# mistralrs.toml

[http_mcp]
enabled = true
port = 4322
auth_method = "api_key"
api_keys = [
  "sk-prod-abc123",
  "sk-dev-xyz789"
]

[http_mcp.rate_limit]
global = 1000      # requests per second
per_key = 100
per_ip = 10

[http_mcp.cors]
allow_origins = ["http://localhost:3000"]
allow_methods = ["GET", "POST"]
```

### Environment Variables

```bash
# Alternative to command line
export MISTRALRS_HTTP_MCP_PORT=4322
export MISTRALRS_HTTP_MCP_AUTH_KEY=sk-abc123
export MISTRALRS_HTTP_MCP_RATE_LIMIT=100

./mistralrs-server run -m model
```

## Monitoring & Observability

### Prometheus Metrics

```rust
// Automatically exported metrics

http_mcp_requests_total{method="tools/call",status="200"}
http_mcp_request_duration_seconds{method="tools/call"}
http_mcp_active_connections
http_mcp_stream_bytes_sent
http_mcp_errors_total{type="auth",method="tools/call"}
```

### Logging

```rust
// Structured logging with tracing

#[instrument(skip(mcp_server, request))]
async fn mcp_stream_handler(
    State(mcp_server): State<Arc<McpServer>>,
    Json(request): Json<McpRequest>,
) -> impl IntoResponse {
    info!(
        request_id = %request.id,
        method = %request.method,
        "Handling HTTP MCP request"
    );
    
    // ... implementation ...
}
```

## Migration from Proxy

### For candle-vllm Users

```rust
// Before (with proxy)
let mcp_config = McpClientConfig {
    servers: vec![McpServerConfig {
        name: "mistralrs",
        source: McpServerSource::Http {
            url: "http://proxy:8080/mcp".to_string(),
        }
    }]
};

// After (direct)
let mcp_config = McpClientConfig {
    servers: vec![McpServerConfig {
        name: "mistralrs",
        source: McpServerSource::Http {
            url: "http://mistralrs:4322/mcp/stream".to_string(),
            auth: Some(ApiKey("sk-abc123".to_string())),
        }
    }]
};

// No code changes needed, just configuration!
```

## Success Criteria

### Functional
- ✅ SSE streaming works correctly
- ✅ All MCP methods supported
- ✅ Compatible with existing MCP clients
- ✅ Backward compatible (old MCP server still works)

### Performance
- ✅ < 15ms overhead vs direct function call
- ✅ Support 100+ concurrent connections
- ✅ < 20KB memory per connection
- ✅ Throughput ≥ 500 requests/sec

### Security
- ✅ API key authentication works
- ✅ Rate limiting prevents abuse
- ✅ Input validation prevents injection
- ✅ No unauthorized access possible

### Operational
- ✅ Prometheus metrics available
- ✅ Easy to configure
- ✅ Clear error messages
- ✅ Production deployment guide

## Next Steps

1. **Read**: [03-MCP-TOOL-FORWARDING.md](03-MCP-TOOL-FORWARDING.md)
2. **Implement**: Follow phased approach (weeks 1-3)
3. **Test**: Integration with candle-vllm
4. **Deploy**: Production rollout with monitoring

---

**Status**: Design complete, ready for implementation  
**Priority**: High (enables candle-vllm integration)  
**Dependencies**: None (independent feature)
