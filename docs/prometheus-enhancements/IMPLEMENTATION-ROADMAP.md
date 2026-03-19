# Implementation Roadmap

## Executive Summary

This document provides a **concrete, phased implementation plan** for all three prometheus enhancements to mistral.rs:

1. **prometheus-parking-lot** (Weeks 1-2): 10-40% performance improvement
2. **HTTP MCP Streaming** (Weeks 3-5): Zero-proxy candle-vllm integration
3. **MCP Tool Forwarding** (Weeks 6-8): Transparent tool ecosystem expansion

**Total Timeline**: 8 weeks (with 1 week buffer)  
**Team Size**: 1-2 developers  
**Risk Level**: Medium (phased approach mitigates risk)

## Phase Overview

```
Week 1-2: prometheus-parking-lot
├─ Day 1-2: Profiling & hotspot identification
├─ Day 3-5: Lock replacement & testing
├─ Day 6-8: Metrics integration
└─ Day 9-10: Validation & documentation

Week 3-5: HTTP MCP Streaming
├─ Week 3: Core SSE endpoint implementation
├─ Week 4: Security (auth, rate limiting)
└─ Week 5: Integration & testing

Week 6-8: MCP Tool Forwarding
├─ Week 6: Basic forwarding architecture
├─ Week 7: Performance & security
└─ Week 8: Integration & polish

Week 9: Buffer for unexpected issues
```

## Phase 1: prometheus-parking-lot (Weeks 1-2)

### Week 1: Core Implementation

#### Day 1-2: Profiling & Analysis

**Objectives**:
- Identify top 10 lock contention hotspots
- Establish performance baselines
- Document current threading architecture

**Tasks**:
```bash
# Profile with perf
cargo build --release --features cuda
perf record --call-graph dwarf -F 99 -p $(pgrep mistralrs-server)
perf report --stdio > baseline-profile.txt

# Profile with flamegraph
cargo flamegraph --bin mistralrs-server -- run -m model

# Analyze mutex usage
rg "Mutex<" --type rust | wc -l
rg "RwLock<" --type rust | wc -l
```

**Deliverables**:
- [ ] Profiling report with top 10 hotspots
- [ ] Baseline performance metrics (requests/sec, latency p50/p95/p99)
- [ ] Threading architecture diagram

#### Day 3-5: Lock Replacement

**Objectives**:
- Replace std::sync locks with prometheus-parking-lot
- Maintain 100% backward compatibility
- Zero performance regression

**Tasks**:
```bash
# Add dependency
echo 'prometheus-parking-lot = "0.6"' >> Cargo.toml

# Replace imports (example files)
# 1. Request scheduler
vim mistralrs-core/src/scheduler.rs
# 2. KV cache
vim mistralrs-paged-attn/src/cache.rs
# 3. Model router
vim mistralrs-core/src/model_router.rs
# 4. Adapter manager
vim mistralrs-core/src/xlora.rs
# 5. Sampling state
vim mistralrs-core/src/sampling.rs
```

**Code Changes Template**:
```rust
// Before:
use std::sync::{Mutex, RwLock};

// After:
use prometheus_parking_lot::{Mutex, RwLock};

// Feature flag support:
#[cfg(feature = "prometheus-locks")]
use prometheus_parking_lot::{Mutex, RwLock};

#[cfg(not(feature = "prometheus-locks"))]
use std::sync::{Mutex, RwLock};
```

**Deliverables**:
- [ ] All hotspot locks replaced
- [ ] Feature flag configuration working
- [ ] Compilation successful with new locks
- [ ] All existing tests pass

#### Day 6-8: Metrics Integration

**Objectives**:
- Expose Prometheus metrics endpoint
- Create Grafana dashboard
- Validate metrics accuracy

**Tasks**:
```rust
// Add metrics endpoint
// File: mistralrs-server/src/metrics.rs

use prometheus::{Encoder, TextEncoder};

pub async fn metrics_handler() -> impl warp::Reply {
    let encoder = TextEncoder::new();
    let registry = prometheus_parking_lot::registry();
    let metric_families = registry.gather();
    
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    
    warp::reply::with_header(
        buffer,
        "Content-Type",
        encoder.format_type(),
    )
}

// Add to server routes
app.route("/metrics", get(metrics_handler))
```

**Deliverables**:
- [ ] `/metrics` endpoint working
- [ ] Grafana dashboard JSON created
- [ ] Metrics validated against actual workload
- [ ] Documentation updated

#### Day 9-10: Validation & Documentation

**Objectives**:
- Benchmark performance improvement
- Run stress tests
- Document changes

**Tasks**:
```bash
# Benchmark suite
cargo bench --features prometheus-locks > results-new.txt
cargo bench --no-default-features > results-old.txt

# Compare results
python scripts/compare_benchmarks.py results-old.txt results-new.txt

# Stress test
./scripts/stress_test.sh --duration 3600 --concurrency 100
```

**Deliverables**:
- [ ] Performance report showing 10-20% improvement
- [ ] Zero regressions in any test
- [ ] PR ready for review
- [ ] Documentation updated in docs/

### Week 2: Production Validation

#### Day 1-3: Real Workload Testing

**Tasks**:
- Deploy to staging environment
- Run production-like workloads
- Monitor metrics for anomalies
- Collect feedback

**Success Metrics**:
- ✅ Average lock wait time < 1ms
- ✅ Request latency reduced by 10-15%
- ✅ Throughput increased by 12-18%
- ✅ No deadlocks or crashes

#### Day 4-5: Rollout Preparation

**Tasks**:
- Create rollback plan
- Write operational runbook
- Train team on metrics dashboard
- Schedule production deployment

**Deliverables**:
- [ ] Rollback procedure documented
- [ ] Ops runbook complete
- [ ] Team trained
- [ ] Deployment scheduled

## Phase 2: HTTP MCP Streaming (Weeks 3-5)

### Week 3: Core SSE Implementation

#### Day 1-2: SSE Endpoint

**Objectives**:
- Implement basic SSE streaming
- Handle MCP JSON-RPC protocol
- Test with simple tool calls

**Tasks**:
```rust
// File: mistralrs-server/src/http_mcp/mod.rs

pub async fn mcp_stream_handler(
    State(mcp): State<Arc<McpServer>>,
    Json(req): Json<McpRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Implementation from design doc
}

// Add route
app.route("/mcp/stream", post(mcp_stream_handler))
```

**Deliverables**:
- [ ] SSE endpoint responds to requests
- [ ] JSON-RPC protocol working
- [ ] Basic tool calls succeed
- [ ] Unit tests passing

#### Day 3-5: Protocol Compliance

**Objectives**:
- Implement all MCP methods
- Handle errors gracefully
- Support tool listing and execution

**Tasks**:
```rust
// Implement MCP methods
impl McpServer {
    async fn handle_tools_list(&self) -> Result<Vec<Tool>>;
    async fn handle_tools_call(&self, name: &str, args: Value) -> Result<Value>;
    async fn handle_resources_list(&self) -> Result<Vec<Resource>>;
    async fn handle_resources_read(&self, uri: &str) -> Result<Value>;
}
```

**Deliverables**:
- [ ] All MCP methods implemented
- [ ] Error handling robust
- [ ] Integration tests passing
- [ ] MCP spec compliance verified

### Week 4: Security & Authentication

#### Day 1-2: Authentication

**Objectives**:
- Implement API key authentication
- Support multiple auth methods
- Integrate with existing auth system

**Tasks**:
```rust
// Authentication middleware
async fn auth_middleware<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    // Check Authorization header
    // Validate API key
    // Add auth context to request
}
```

**Deliverables**:
- [ ] API key auth working
- [ ] Bearer token support
- [ ] Auth tests passing
- [ ] Documentation updated

#### Day 3-5: Rate Limiting & Input Validation

**Objectives**:
- Prevent abuse via rate limiting
- Validate all input
- Implement request size limits

**Tasks**:
```rust
// Rate limiting with tower-governor
let governor_conf = GovernorConfigBuilder::default()
    .per_second(10)
    .burst_size(20)
    .finish()
    .unwrap();

// Input validation
fn validate_mcp_request(req: &McpRequest) -> Result<()> {
    // Check JSON-RPC version
    // Validate method name
    // Check parameter types
    // Limit request size
}
```

**Deliverables**:
- [ ] Rate limiting working
- [ ] Input validation comprehensive
- [ ] Security tests passing
- [ ] Documented security model

### Week 5: Integration & Testing

#### Day 1-3: candle-vllm Integration

**Objectives**:
- Test with candle-vllm MCP client
- Verify end-to-end workflow
- Benchmark latency

**Tasks**:
```bash
# Start mistral.rs
./mistralrs-server --port 1234 --http-mcp-port 4322 \
  gguf -m model.gguf -f model-q4.gguf

# Configure candle-vllm
# (in candle-vllm MCP client config)
{
  "servers": [{
    "name": "mistralrs",
    "url": "http://localhost:4322/mcp/stream"
  }]
}

# Test tool calls
curl -X POST http://localhost:4322/mcp/stream \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call",...}'
```

**Deliverables**:
- [ ] candle-vllm successfully connects
- [ ] Tool calls work correctly
- [ ] Latency < 15ms overhead
- [ ] Stress test passes (100+ concurrent)

#### Day 4-5: Documentation & Examples

**Objectives**:
- Write comprehensive documentation
- Create example clients
- Document API surface

**Deliverables**:
- [ ] HTTP MCP API documented
- [ ] Example clients (Python, Rust, curl)
- [ ] Deployment guide
- [ ] Troubleshooting guide

## Phase 3: MCP Tool Forwarding (Weeks 6-8)

### Week 6: Core Architecture

#### Day 1-3: Tool Forwarder Implementation

**Objectives**:
- Implement ToolForwarder struct
- Support basic forwarding
- Handle namespace prefixing

**Tasks**:
```rust
// File: mistralrs-core/src/mcp_server/tool_forwarder.rs

pub struct ToolForwarder {
    native_tools: HashMap<String, Box<dyn Tool>>,
    mcp_client: Arc<McpClient>,
    forwarded_tools: Arc<RwLock<HashMap<String, ForwardedTool>>>,
}

impl ToolForwarder {
    pub async fn list_all_tools(&self) -> Vec<ToolDescriptor>;
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value>;
    pub async fn refresh_forwarded_tools(&self) -> Result<()>;
}
```

**Deliverables**:
- [ ] ToolForwarder compiles
- [ ] Basic forwarding works
- [ ] Namespace prefixing correct
- [ ] Unit tests passing

#### Day 4-5: Integration with MCP Server

**Objectives**:
- Connect ToolForwarder to MCP server
- Expose forwarded tools via HTTP endpoint
- Test with external MCP servers

**Tasks**:
```rust
// Update MCP server to use ToolForwarder
pub struct McpServer {
    tool_forwarder: Arc<ToolForwarder>,
}

impl McpServer {
    pub async fn list_tools(&self) -> Vec<ToolDescriptor> {
        self.tool_forwarder.list_all_tools().await
    }
    
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value> {
        self.tool_forwarder.call_tool(name, args).await
    }
}
```

**Deliverables**:
- [ ] Tools listed via HTTP MCP endpoint
- [ ] Forwarded tools callable
- [ ] Integration tests passing
- [ ] Basic E2E test working

### Week 7: Performance & Security

#### Day 1-2: Performance Optimization

**Objectives**:
- Implement result caching
- Add request batching
- Minimize forwarding overhead

**Tasks**:
```rust
// Add caching
pub struct ToolCache {
    descriptor_cache: LruCache<String, ToolDescriptor>,
    result_cache: TtlCache<(String, Value), Value>,
}

// Add batching
pub async fn call_tools_batch(
    &self,
    calls: Vec<(String, Value)>,
) -> Vec<Result<Value>>;
```

**Deliverables**:
- [ ] Cache hit rate > 80%
- [ ] Batching reduces latency by 50%+
- [ ] Overhead < 1ms
- [ ] Benchmark results documented

#### Day 3-5: Security Controls

**Objectives**:
- Implement namespace isolation
- Add input validation
- Create security policy engine

**Tasks**:
```rust
// Security policy
pub struct ToolSecurityPolicy {
    allowed_namespaces: HashSet<String>,
    blocked_tools: HashSet<String>,
    rate_limits: HashMap<String, RateLimit>,
}

// Validation
fn validate_arguments(&self, tool: &str, args: &Value) -> Result<()> {
    // Prevent path traversal
    // Prevent SQL injection
    // Check argument types
}
```

**Deliverables**:
- [ ] Security policy enforced
- [ ] Input validation comprehensive
- [ ] Audit logging working
- [ ] Security tests passing

### Week 8: Polish & Production

#### Day 1-3: End-to-End Testing

**Objectives**:
- Test complete candle-vllm → mistral.rs → external tools flow
- Verify all three enhancements work together
- Load testing

**Tasks**:
```bash
# Setup test environment
# 1. Start external MCP servers (filesystem, web search)
# 2. Start mistral.rs with all enhancements
# 3. Connect candle-vllm
# 4. Run comprehensive test suite

./scripts/e2e_test.sh --full-stack
```

**Success Criteria**:
- ✅ All tool calls succeed
- ✅ Performance within targets
- ✅ No memory leaks
- ✅ Stress test passes (1000 req/sec for 1 hour)

#### Day 4-5: Documentation & Release

**Objectives**:
- Complete all documentation
- Create migration guide
- Prepare release notes

**Deliverables**:
- [ ] All docs updated
- [ ] Migration guide complete
- [ ] Release notes written
- [ ] Ready for production deployment

## Testing Strategy

### Unit Tests

```bash
# Run unit tests for each component
cargo test --package mistralrs-core -- prometheus_parking_lot
cargo test --package mistralrs-server -- http_mcp
cargo test --package mistralrs-core -- tool_forwarder
```

### Integration Tests

```bash
# Test interactions between components
cargo test --test integration_http_mcp
cargo test --test integration_tool_forwarding
cargo test --test integration_end_to_end
```

### Performance Tests

```bash
# Benchmark suite
cargo bench --features prometheus-locks

# Load testing
k6 run load-tests/http-mcp-stream.js
k6 run load-tests/tool-forwarding.js

# Memory profiling
valgrind --tool=massif ./target/release/mistralrs-server
```

### Security Tests

```bash
# Run security test suite
cargo test --test security_auth
cargo test --test security_rate_limit
cargo test --test security_input_validation

# Penetration testing (manual)
./scripts/pentest.sh
```

## Risk Management

### High-Risk Items

#### Risk 1: Performance Regression
**Mitigation**:
- Comprehensive benchmarking before/after
- Feature flags for easy rollback
- Gradual rollout (staging → canary → production)

#### Risk 2: Concurrency Bugs
**Mitigation**:
- ThreadSanitizer during development
- Extensive concurrency tests
- Stress testing with high parallelism

#### Risk 3: Security Vulnerabilities
**Mitigation**:
- Security review at each phase
- Input validation on all paths
- Audit logging for forensics

### Medium-Risk Items

#### Risk 4: API Compatibility
**Mitigation**:
- Backward compatibility tests
- Feature flags for old behavior
- Clear migration documentation

#### Risk 5: Integration Issues
**Mitigation**:
- Early integration testing with candle-vllm
- Mock MCP servers for testing
- Comprehensive E2E tests

## Deployment Strategy

### Staging Deployment (Week 9)

```bash
# Week 9, Day 1-3: Staging deployment
# 1. Deploy to staging environment
# 2. Run production-like workloads
# 3. Monitor metrics closely
# 4. Fix any issues found
```

### Canary Deployment (Week 9, Day 4-5)

```bash
# Deploy to 10% of production traffic
# Monitor for:
# - Error rate increase
# - Latency regression
# - Memory leaks
# - CPU usage increase

# If successful, increase to 50%, then 100%
```

### Rollback Plan

```bash
# If issues detected:
# 1. Immediate: Switch feature flags off
# 2. If that fails: Rollback to previous version
# 3. Post-mortem: Identify root cause

# Feature flag rollback:
mistralrs-server --features "no-prometheus-locks,no-http-mcp,no-tool-forwarding"
```

## Success Metrics

### Phase 1: prometheus-parking-lot
- ✅ 10-20% latency reduction
- ✅ 12-18% throughput increase
- ✅ < 1ms average lock wait time
- ✅ Prometheus dashboard operational

### Phase 2: HTTP MCP Streaming
- ✅ candle-vllm connects without proxy
- ✅ < 15ms latency overhead
- ✅ 100+ concurrent connections
- ✅ Zero auth bypasses in security tests

### Phase 3: MCP Tool Forwarding
- ✅ All MCP client tools forwarded
- ✅ < 1ms forwarding overhead
- ✅ Cache hit rate > 80%
- ✅ Zero security incidents

### Overall
- ✅ Production-ready by end of Week 9
- ✅ All documentation complete
- ✅ Zero critical bugs
- ✅ Positive feedback from early adopters

## Resource Requirements

### Developer Time
- **Lead Developer**: 8 weeks full-time
- **Code Reviewer**: 2 days/week for 8 weeks
- **QA Engineer**: 1 week (during Phase 3)

### Infrastructure
- **Staging Environment**: 2 GPU servers + 1 CPU server
- **Monitoring**: Prometheus + Grafana instance
- **Test Infrastructure**: Load testing cluster

### External Dependencies
- MCP specification compliance
- candle-vllm MCP client (external project)
- External MCP servers for testing (filesystem, web search)

## Timeline Contingency

### If Behind Schedule

**Option 1: Reduce Scope**
- Skip Phase 3 (tool forwarding) initially
- Deploy Phases 1-2 first
- Add Phase 3 in follow-up release

**Option 2: Extend Timeline**
- Add 1-2 weeks buffer
- Prioritize quality over speed
- Maintain all features

**Option 3: Add Resources**
- Bring in second developer
- Focus on parallel workstreams
- Maintain timeline

### If Ahead of Schedule

**Option 1: Additional Polish**
- Improve documentation
- Add more examples
- Create video tutorials

**Option 2: Additional Features**
- Add WebSocket support (alternative to SSE)
- Implement advanced caching strategies
- Add more security features

## Communication Plan

### Weekly Updates
- Monday: Planning meeting (1 hour)
- Wednesday: Progress check-in (30 min)
- Friday: Demo + retrospective (1 hour)

### Stakeholder Communication
- End of Phase 1: Performance report to leadership
- End of Phase 2: Integration demo with candle-vllm
- End of Phase 3: Final presentation + documentation

### Documentation Updates
- Update docs/ after each phase
- Blog post after Phase 2 (HTTP MCP streaming)
- Conference talk proposal after Phase 3

## Post-Launch

### Monitoring (First 2 Weeks)
- Daily metrics review
- Weekly performance analysis
- Bi-weekly stakeholder updates

### Iteration (Weeks 10-12)
- Address user feedback
- Fix any discovered bugs
- Performance tuning based on real usage

### Long-term Maintenance
- Monthly dependency updates
- Quarterly performance reviews
- Bi-annual feature enhancements

## Conclusion

This roadmap provides a structured, phased approach to implementing three major enhancements to mistral.rs. The 8-week timeline includes:

1. **2 weeks**: prometheus-parking-lot (performance + observability)
2. **3 weeks**: HTTP MCP streaming (candle-vllm integration)
3. **3 weeks**: MCP tool forwarding (ecosystem expansion)
4. **1 week**: Buffer for unexpected issues

Success depends on:
- Disciplined execution of phases
- Comprehensive testing at each stage
- Clear communication with stakeholders
- Willingness to adjust scope if needed

**Next Steps**:
1. Review and approve roadmap
2. Allocate resources (developer time, infrastructure)
3. Begin Phase 1 (prometheus-parking-lot profiling)

---

**Status**: Roadmap complete, ready for execution  
**Start Date**: TBD  
**Target Completion**: 9 weeks from start  
**Risk Level**: Medium (mitigated by phased approach)
