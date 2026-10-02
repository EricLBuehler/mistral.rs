# MCP Transport Types

mistral.rs supports three transport types for connecting to MCP servers, each optimized for different use cases.

## HTTP Transport

Best for public APIs, RESTful services, and servers behind load balancers.

### Configuration

```json
{
  "source": {
    "type": "Http",
    "url": "https://api.example.com/mcp",
    "timeout_secs": 30,
    "headers": {
      "X-API-Version": "v1",
      "User-Agent": "mistral-rs/0.6.0"
    }
  },
  "bearer_token": "your-api-token"
}
```

### Features
- Server-Sent Events (SSE) support for streaming responses
- Custom headers for API versioning or client identification
- Bearer token authentication (added as `Authorization: Bearer <token>`)
- Configurable timeouts
- Standard HTTP semantics

### Example: Hugging Face MCP
```rust
McpServerSource::Http {
    url: "https://hf.co/mcp".to_string(),
    timeout_secs: Some(30),
    headers: None,
}
```

## WebSocket Transport

Best for real-time applications, bidirectional communication, and low-latency requirements.

### Configuration

```json
{
  "source": {
    "type": "WebSocket",
    "url": "wss://realtime.example.com/mcp",
    "timeout_secs": 60,
    "headers": {
      "Origin": "https://mistral.rs",
      "Sec-WebSocket-Protocol": "mcp"
    }
  },
  "bearer_token": "your-websocket-token"
}
```

### Features
- Persistent connections reduce handshake overhead
- Server-initiated notifications
- Lower latency for frequent tool calls
- Automatic reconnection handling
- WebSocket-specific headers support

### Example: Real-time Data Feed
```rust
McpServerSource::WebSocket {
    url: "wss://data.example.com/mcp".to_string(),
    timeout_secs: Some(60),
    headers: Some(headers),
}
```

## Process Transport

Best for local tools, development servers, and sandboxed environments.

### Configuration

```json
{
  "source": {
    "type": "Process",
    "command": "mcp-server-filesystem",
    "args": ["--root", "/tmp", "--readonly"],
    "work_dir": "/home/user/workspace",
    "env": {
      "MCP_LOG_LEVEL": "info",
      "MCP_TIMEOUT": "30"
    }
  }
}
```

### Features
- No network overhead
- Process isolation for security
- Direct stdin/stdout communication
- Environment variable configuration
- Working directory control
- No authentication needed (process inherits permissions)

### Example: Filesystem Server
```rust
McpServerSource::Process {
    command: "mcp-server-filesystem".to_string(),
    args: vec!["--root".to_string(), "/tmp".to_string()],
    work_dir: None,
    env: None,
}
```

## Transport Selection Guide

| Use Case | Recommended Transport | Why |
|----------|---------------------|-----|
| Public APIs | HTTP | Standard auth, caching, load balancing |
| Local tools | Process | No network, process isolation |
| Real-time data | WebSocket | Low latency, server push |
| Corporate proxies | HTTP | Proxy support, standard ports |
| Development | Process | Easy debugging, no network setup |
| Interactive apps | WebSocket | Bidirectional, persistent connection |

## Security Considerations

### HTTP
- Always use HTTPS in production
- Bearer tokens transmitted with each request
- Consider token rotation strategies

### WebSocket
- Use WSS (WebSocket Secure) in production
- Bearer token sent during handshake
- Connection persists with authenticated state

### Process
- Inherits user permissions
- Sandboxing via work_dir and env
- No network exposure

## Performance Tips

1. **HTTP**: Enable keep-alive, use connection pooling
2. **WebSocket**: Reuse connections, handle reconnection gracefully
3. **Process**: Minimize startup time, use long-running processes

## Error Handling

All transports implement automatic retry with exponential backoff:
- Initial retry: 1 second
- Max retry: 60 seconds
- Max attempts: 5

Custom retry behavior can be configured per server.