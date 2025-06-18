# mistralrs-paged-attn

Paged attention implementation for mistral.rs with support for CUDA and Metal backends.

## Features

- **CUDA Backend**: High-performance paged attention for NVIDIA GPUs
- **Metal Backend**: Optimized implementation for Apple Silicon
- **OpenTelemetry Support**: Built-in observability with metrics and structured logging

## OpenTelemetry Observability

The paged attention module includes comprehensive telemetry capabilities for monitoring and debugging:

### Metrics
- Operation counters and timing histograms
- Tensor size tracking
- Support for OTLP and Prometheus exporters

### Structured Logging
- JSON-formatted debug logs for all tensor operations
- Ready for ingestion into Elasticsearch, OpenSearch, or Meilisearch
- Minimal performance overhead

### Usage

Enable telemetry by adding the `telemetry` feature:

```toml
[dependencies]
mistralrs-paged-attn = { version = "*", features = ["cuda", "telemetry"] }
```

Initialize telemetry in your application:

```rust
#[cfg(feature = "telemetry")]
mistralrs_paged_attn::telemetry::init_telemetry()
    .expect("Failed to initialize telemetry");
```

Enable debug logging:

```bash
RUST_LOG=paged_attention=debug,paged_attention_metrics=debug cargo run
```

For detailed telemetry documentation, see [TELEMETRY.md](TELEMETRY.md).