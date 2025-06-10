# Advanced MCP Usage

This guide covers advanced MCP client configurations and usage patterns.

## Multi-Server Configuration

Connect to multiple MCP servers simultaneously to access different tool sets:

```rust
let mcp_config = McpClientConfig {
    servers: vec![
        // Hugging Face for ML tools
        McpServerConfig {
            id: "hf_server".to_string(),
            name: "Hugging Face MCP".to_string(),
            source: McpServerSource::Http {
                url: "https://hf.co/mcp".to_string(),
                timeout_secs: Some(30),
                headers: None,
            },
            enabled: true,
            tool_prefix: Some("hf".to_string()),
            resources: None,
            bearer_token: Some("hf_xxx".to_string()),
        },
        // Local filesystem access
        McpServerConfig {
            id: "fs_server".to_string(),
            name: "Filesystem MCP".to_string(),
            source: McpServerSource::Process {
                command: "mcp-server-filesystem".to_string(),
                args: vec!["--root".to_string(), "/data".to_string()],
                work_dir: None,
                env: None,
            },
            enabled: true,
            tool_prefix: Some("fs".to_string()),
            resources: Some(vec!["file://**".to_string()]),
            bearer_token: None,
        },
        // GitHub API access
        McpServerConfig {
            id: "github_server".to_string(),
            name: "GitHub MCP".to_string(),
            source: McpServerSource::Http {
                url: "https://api.github.com/mcp".to_string(),
                timeout_secs: Some(45),
                headers: Some(HashMap::from([
                    ("Accept".to_string(), "application/vnd.github.v3+json".to_string()),
                ])),
            },
            enabled: true,
            tool_prefix: Some("gh".to_string()),
            resources: None,
            bearer_token: Some("ghp_xxx".to_string()),
        },
    ],
    auto_register_tools: true,
    tool_timeout_secs: Some(30),
    max_concurrent_calls: Some(10),
};
```

## Tool Prefixing Strategy

When using multiple servers, tool prefixes prevent naming conflicts:

```json
{
  "servers": [
    {
      "id": "server1",
      "tool_prefix": "s1",
      // Tool "search" becomes "s1_search"
    },
    {
      "id": "server2", 
      "tool_prefix": "s2",
      // Tool "search" becomes "s2_search"
    }
  ]
}
```

## Custom Headers and Authentication

### API Key in Headers
```rust
let mut headers = HashMap::new();
headers.insert("X-API-Key".to_string(), "your-api-key".to_string());
headers.insert("X-Client-Version".to_string(), "1.0.0".to_string());

McpServerSource::Http {
    url: "https://api.example.com/mcp".to_string(),
    timeout_secs: Some(30),
    headers: Some(headers),
}
```

### OAuth2 Bearer Token
```rust
McpServerConfig {
    // ...
    bearer_token: Some("your-oauth2-token".to_string()),
    // Automatically added as: Authorization: Bearer your-oauth2-token
}
```

## Resource Subscriptions

Subscribe to specific resource patterns from MCP servers:

```rust
McpServerConfig {
    id: "data_server".to_string(),
    // ...
    resources: Some(vec![
        "file://data/**/*.json".to_string(),  // All JSON files in data/
        "db://users/*".to_string(),            // All user records
        "api://v1/metrics".to_string(),        // Specific API endpoint
    ]),
    // ...
}
```

## Concurrency and Rate Limiting

### Global Concurrency Control
```rust
McpClientConfig {
    // ...
    max_concurrent_calls: Some(5),  // Max 5 tools executing simultaneously
}
```

### Per-Tool Timeouts
```rust
McpClientConfig {
    // ...
    tool_timeout_secs: Some(30),  // Each tool call times out after 30s
}
```

### Custom Rate Limiting
```python
# Python example with custom rate limiting
import time
from collections import deque

class RateLimitedMcpRunner:
    def __init__(self, runner, max_calls_per_minute=60):
        self.runner = runner
        self.max_calls = max_calls_per_minute
        self.call_times = deque()
    
    def send_chat_completion_request(self, request):
        # Remove calls older than 1 minute
        now = time.time()
        while self.call_times and self.call_times[0] < now - 60:
            self.call_times.popleft()
        
        # Check rate limit
        if len(self.call_times) >= self.max_calls:
            sleep_time = 60 - (now - self.call_times[0])
            time.sleep(sleep_time)
        
        # Make the call
        self.call_times.append(now)
        return self.runner.send_chat_completion_request(request)
```

## Environment-Specific Configuration

### Development vs Production
```rust
let mcp_config = if cfg!(debug_assertions) {
    McpClientConfig {
        servers: vec![/* development servers */],
        tool_timeout_secs: Some(60),  // Longer timeouts for debugging
        max_concurrent_calls: Some(1), // Sequential execution for debugging
        // ...
    }
} else {
    McpClientConfig {
        servers: vec![/* production servers */],
        tool_timeout_secs: Some(10),   // Strict timeouts
        max_concurrent_calls: Some(20), // Higher concurrency
        // ...
    }
};
```

### Environment Variables
```rust
let mcp_config = McpClientConfig {
    servers: vec![
        McpServerConfig {
            // ...
            bearer_token: std::env::var("HF_TOKEN").ok(),
            source: McpServerSource::Http {
                url: std::env::var("MCP_SERVER_URL")
                    .unwrap_or_else(|_| "https://hf.co/mcp".to_string()),
                // ...
            },
            // ...
        },
    ],
    // ...
};
```

## Error Handling and Fallbacks

### Graceful Degradation
```rust
let mcp_config = McpClientConfig {
    servers: vec![
        // Primary server
        McpServerConfig {
            id: "primary".to_string(),
            enabled: true,
            // ...
        },
        // Fallback server
        McpServerConfig {
            id: "fallback".to_string(),
            enabled: check_primary_health().is_err(),
            // ...
        },
    ],
    // ...
};
```

### Tool-Specific Error Handling
```python
# Handle specific tool errors
try:
    response = runner.send_chat_completion_request(request)
except Exception as e:
    if "tool_timeout" in str(e):
        print("Tool execution timed out, trying with longer timeout...")
        # Retry with extended timeout
    elif "tool_not_found" in str(e):
        print("Tool not available, falling back to built-in response...")
        # Fallback logic
```

## Monitoring and Debugging

### Enable Debug Logging
```rust
std::env::set_var("RUST_LOG", "mistralrs_mcp=debug");
env_logger::init();
```

### Tool Call Inspection
```rust
let response = model.send_chat_request(messages).await?;

// Check if tools were called
if let Some(tool_calls) = &response.choices[0].message.tool_calls {
    for call in tool_calls {
        println!("Tool: {}", call.function.name);
        println!("Args: {}", call.function.arguments);
        println!("ID: {}", call.id);
    }
}
```

## Performance Optimization

### Connection Pooling
HTTP and WebSocket transports automatically use connection pooling. Configure pool size:

```rust
// Set via environment variable
std::env::set_var("MCP_POOL_SIZE", "10");
```

### Caching Tool Responses
```python
from functools import lru_cache
import json

@lru_cache(maxsize=100)
def cached_tool_call(tool_name, args_json):
    args = json.loads(args_json)
    # Tool execution logic
    return result

# Use with MCP tools that have deterministic outputs
```

## Security Best Practices

1. **Token Rotation**: Implement automatic token refresh for long-running applications
2. **Least Privilege**: Only enable required tools and resources
3. **Audit Logging**: Log all tool calls for security monitoring
4. **Network Isolation**: Use Process transport for sensitive local operations
5. **Input Validation**: MCP servers should validate all tool inputs