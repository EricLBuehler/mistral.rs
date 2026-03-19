# Prometheus Enhancements - Quick Reference

**Quick navigation and TL;DR for busy developers**

## What Are These Enhancements?

Three strategic improvements to mistral.rs enabling hybrid deployment with candle-vllm:

1. **🚀 prometheus-parking-lot** - 10-40% faster threading + built-in metrics
2. **🔌 HTTP MCP Streaming** - Zero-proxy integration with candle-vllm  
3. **🔧 MCP Tool Forwarding** - Transparent tool capability expansion

## Quick Start

### Want Better Performance?
→ Read [01-PROMETHEUS-PARKING-LOT.md](01-PROMETHEUS-PARKING-LOT.md)
- Replace locks, get 10-20% latency reduction
- Built-in Prometheus metrics dashboard
- 1-2 weeks implementation

### Want candle-vllm Integration?
→ Read [02-HTTP-MCP-STREAMING.md](02-HTTP-MCP-STREAMING.md)
- Direct HTTP connection (no proxy needed)
- SSE streaming protocol
- 2-3 weeks implementation

### Want Tool Ecosystem Expansion?
→ Read [03-MCP-TOOL-FORWARDING.md](03-MCP-TOOL-FORWARDING.md)
- Forward all MCP client tools through MCP server
- candle-vllm inherits all tools automatically
- 2-3 weeks implementation

### Want Implementation Plan?
→ Read [IMPLEMENTATION-ROADMAP.md](IMPLEMENTATION-ROADMAP.md)
- 8-week phased approach
- Detailed timeline and milestones
- Risk management and rollback plans

## File Guide

| File | Purpose | Length | Priority |
|------|---------|---------|----------|
| [README.md](README.md) | Overview & strategy | 10 min | **START HERE** |
| [01-PROMETHEUS-PARKING-LOT.md](01-PROMETHEUS-PARKING-LOT.md) | Threading refactor | 20 min | High |
| [02-HTTP-MCP-STREAMING.md](02-HTTP-MCP-STREAMING.md) | HTTP endpoint design | 25 min | High |
| [03-MCP-TOOL-FORWARDING.md](03-MCP-TOOL-FORWARDING.md) | Tool forwarding | 30 min | High |
| [IMPLEMENTATION-ROADMAP.md](IMPLEMENTATION-ROADMAP.md) | Execution plan | 20 min | **MUST READ** |
| [QUICK-REFERENCE.md](QUICK-REFERENCE.md) | This file | 5 min | Optional |

## Code Locations

### prometheus-parking-lot Changes
```
mistralrs-core/src/scheduler.rs        # Request scheduling
mistralrs-paged-attn/src/cache.rs      # KV cache
mistralrs-core/src/model_router.rs     # Model routing
mistralrs-core/src/xlora.rs            # Adapter management
mistralrs-core/src/sampling.rs         # Token generation
```

### HTTP MCP Streaming Files (New)
```
mistralrs-server/src/http_mcp/mod.rs   # Main SSE endpoint
mistralrs-server/src/http_mcp/auth.rs  # Authentication
mistralrs-server/src/metrics.rs         # Prometheus metrics
```

### MCP Tool Forwarding Files (New)
```
mistralrs-core/src/mcp_server/tool_forwarder.rs  # Core logic
mistralrs-core/src/mcp_server/security.rs        # Security policy
mistralrs-core/src/mcp_server/cache.rs           # Result caching
```

## Commands Cheat Sheet

### Build with Enhancements
```bash
# With all features
cargo build --release --features "prometheus-locks,http-mcp-stream,tool-forwarding"

# Individual features
cargo build --release --features "prometheus-locks"
cargo build --release --features "http-mcp-stream"
cargo build --release --features "tool-forwarding"

# Rollback (disable all)
cargo build --release --no-default-features
```

### Run with Enhancements
```bash
# Enable everything
./mistralrs-server \
  --port 1234 \
  --http-mcp-port 4322 \
  --http-mcp-auth-key sk-abc123 \
  run -m meta-llama/Llama-3.2-8B

# Metrics endpoint
curl http://localhost:1234/metrics

# Test HTTP MCP
curl -X POST http://localhost:4322/mcp/stream \
  -H "Authorization: Bearer sk-abc123" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":"1"}'
```

### Testing
```bash
# Unit tests
cargo test --features prometheus-locks
cargo test --features http-mcp-stream
cargo test --features tool-forwarding

# Integration tests
cargo test --test integration_http_mcp
cargo test --test integration_tool_forwarding

# Benchmarks
cargo bench --features prometheus-locks

# Load testing
k6 run load-tests/http-mcp-stream.js
```

### Monitoring
```bash
# Watch Prometheus metrics
watch -n 1 'curl -s http://localhost:1234/metrics | grep mistralrs'

# Check lock contention
curl -s http://localhost:1234/metrics | grep lock_wait_seconds

# Check HTTP MCP stats
curl -s http://localhost:1234/metrics | grep http_mcp

# Check tool forwarding
curl -s http://localhost:1234/metrics | grep tool_forwarded
```

## Key Metrics to Watch

### Phase 1: prometheus-parking-lot
```prometheus
# Target: < 1ms average
avg(rate(mistralrs_lock_wait_seconds_sum[5m]) / rate(mistralrs_lock_wait_seconds_count[5m]))

# Target: 10-20% reduction
histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))

# Target: 12-18% increase
rate(requests_total[1m])
```

### Phase 2: HTTP MCP Streaming
```prometheus
# Target: < 15ms overhead
histogram_quantile(0.95, rate(http_mcp_request_duration_seconds_bucket[5m]))

# Target: > 100 concurrent
http_mcp_active_connections

# Target: < 1% error rate
rate(http_mcp_errors_total[5m]) / rate(http_mcp_requests_total[5m])
```

### Phase 3: MCP Tool Forwarding
```prometheus
# Target: < 1ms overhead
histogram_quantile(0.95, rate(mistralrs_tool_forwarded_duration_seconds_bucket[5m]))

# Target: > 80% hit rate
rate(mistralrs_tool_cache_hits[5m]) / (rate(mistralrs_tool_cache_hits[5m]) + rate(mistralrs_tool_cache_misses[5m]))

# Target: < 1% error rate
rate(mistralrs_tool_errors_total[5m]) / rate(mistralrs_tool_calls_total[5m])
```

## Timeline Summary

```
Week 1-2: prometheus-parking-lot
├─ Replace std::sync locks
├─ Add Prometheus metrics
└─ Validate 10-20% improvement

Week 3-5: HTTP MCP Streaming
├─ Implement SSE endpoint
├─ Add auth & rate limiting
└─ Test with candle-vllm

Week 6-8: MCP Tool Forwarding
├─ Build tool forwarder
├─ Add caching & security
└─ End-to-end testing

Week 9: Buffer & Production Deployment
├─ Final testing
├─ Documentation polish
└─ Production rollout
```

## Common Issues & Solutions

### Issue: Compilation Errors
**Solution**: Ensure feature flags are correct
```bash
# Check enabled features
cargo metadata --format-version 1 | jq '.resolve.nodes[] | select(.id | contains("mistralrs")) | .features'

# Clean rebuild
cargo clean && cargo build --release --features "prometheus-locks"
```

### Issue: Performance Regression
**Solution**: Use feature flags to isolate issue
```bash
# Test each feature independently
cargo bench --no-default-features  # Baseline
cargo bench --features "prometheus-locks"
cargo bench --features "http-mcp-stream"
cargo bench --features "tool-forwarding"
```

### Issue: MCP Connection Fails
**Solution**: Check authentication and network
```bash
# Test without auth
curl -v http://localhost:4322/mcp/stream

# Test with auth
curl -v -H "Authorization: Bearer sk-abc123" http://localhost:4322/mcp/stream

# Check server logs
tail -f mistralrs-server.log | grep mcp
```

### Issue: Tool Forwarding Not Working
**Solution**: Verify MCP client configuration
```bash
# List available tools
curl -X POST http://localhost:4322/mcp/stream \
  -H "Authorization: Bearer sk-abc123" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":"1"}'

# Check forwarded tools appear with namespace
# e.g., "filesystem::read_file", "web_search::query"

# Verify MCP client is configured
cat mistralrs-mcp-client.toml
```

## Configuration Templates

### Minimal Configuration
```toml
# mistralrs.toml - Minimal setup

[http_mcp]
enabled = true
port = 4322
auth_key = "sk-abc123"

[tool_forwarding]
enabled = false  # Disable for minimal setup
```

### Full Configuration
```toml
# mistralrs.toml - Full featured

[prometheus_locks]
enabled = true
metrics_endpoint = "/metrics"

[http_mcp]
enabled = true
port = 4322
auth_method = "api_key"
api_keys = ["sk-prod-abc123", "sk-dev-xyz789"]

[http_mcp.rate_limit]
global = 1000
per_key = 100
per_ip = 10

[mcp_client]
enabled = true

[[mcp_client.servers]]
name = "filesystem"
command = "npx"
args = ["@modelcontextprotocol/server-filesystem", "/tmp", "-y"]

[[mcp_client.servers]]
name = "web_search"
url = "http://search-server:5000"

[tool_forwarding]
enabled = true
namespace_strategy = "server_prefix"

[tool_forwarding.security]
allowed_namespaces = ["filesystem", "web_search"]
blocked_tools = ["filesystem::delete"]

[tool_forwarding.cache]
enabled = true
ttl_seconds = 60
max_size = 1000
```

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│           candle-vllm (Client)              │
│  ├─ Standard transformers                   │
│  ├─ Continuous batching                     │
│  └─ MCP Client ──────────────┐              │
└──────────────────────────────│──────────────┘
                                │
                          HTTP MCP Stream
                          (SSE, no proxy)
                                │
┌───────────────────────────────▼─────────────┐
│        mistral.rs (Enhanced)                │
│  ┌───────────────────────────────────────┐  │
│  │ prometheus-parking-lot                │  │
│  │ ├─ Faster locks (10-40% improvement)  │  │
│  │ └─ Prometheus metrics dashboard       │  │
│  └───────────────────────────────────────┘  │
│  ┌───────────────────────────────────────┐  │
│  │ HTTP MCP Server                       │  │
│  │ ├─ SSE streaming endpoint             │  │
│  │ ├─ Authentication & rate limiting     │  │
│  │ └─ Tool Forwarder ──────────────┐     │  │
│  └─────────────────────────────────│─────┘  │
│                                    │         │
│  ┌─────────────────────────────────▼─────┐  │
│  │ MCP Client (Internal)                 │  │
│  │ ├─ Filesystem tools                   │  │
│  │ ├─ Web search tools                   │  │
│  │ └─ Custom API tools                   │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Success Criteria Checklist

### Phase 1: prometheus-parking-lot
- [ ] Average lock wait time < 1ms
- [ ] Request latency reduced 10-20%
- [ ] Throughput increased 12-18%
- [ ] Prometheus dashboard operational
- [ ] Zero regressions in benchmarks

### Phase 2: HTTP MCP Streaming
- [ ] candle-vllm connects without proxy
- [ ] Latency overhead < 15ms
- [ ] Support 100+ concurrent connections
- [ ] Authentication working
- [ ] Rate limiting prevents abuse

### Phase 3: MCP Tool Forwarding
- [ ] All MCP client tools exposed
- [ ] Forwarding overhead < 1ms
- [ ] Cache hit rate > 80%
- [ ] Security isolation working
- [ ] E2E tests passing

## Next Actions

### Just Starting?
1. Read [README.md](README.md) (10 min)
2. Read [IMPLEMENTATION-ROADMAP.md](IMPLEMENTATION-ROADMAP.md) (20 min)
3. Choose which phase to implement first

### Ready to Code?
1. Set up development environment
2. Read detailed docs for chosen phase
3. Follow implementation checklist
4. Run tests continuously

### Deploying?
1. Review rollback plan
2. Deploy to staging first
3. Monitor metrics closely
4. Gradual production rollout

## Getting Help

### Documentation
- Full docs in this directory
- Upstream mistral.rs: https://github.com/EricLBuehler/mistral.rs
- MCP Specification: https://spec.modelcontextprotocol.io/

### Community
- Upstream Issues: https://github.com/EricLBuehler/mistral.rs/issues
- Discussions: https://github.com/EricLBuehler/mistral.rs/discussions

### Internal
- Project Notes: /Users/gqadonis/Projects/references/candle-vllm/docs/llm-architectures/
- Code: /Users/gqadonis/Projects/references/mistral.rs/

---

**Last Updated**: December 8, 2024  
**Status**: Design complete, ready for implementation  
**Maintainer**: Travis James (Prometheus Agentic Growth Solutions)
