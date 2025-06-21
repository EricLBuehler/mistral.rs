use opentelemetry::{
    global,
    metrics::{Counter, Histogram, Meter},
    KeyValue,
};
use serde::Serialize;
use std::sync::LazyLock;
use tracing::{debug, instrument};

#[derive(Serialize, Debug)]
pub struct PagedAttentionMetrics {
    pub num_sequences: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub max_context_len: usize,
    pub softmax_scale: f32,
    pub softcapping: f32,
    pub use_v1: bool,
}

#[derive(Serialize, Debug)]
pub struct CacheUpdateMetrics {
    pub num_tokens: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub block_size: usize,
}

static METER: LazyLock<Meter> = LazyLock::new(|| global::meter("mistralrs_paged_attention"));

static PAGED_ATTENTION_CALLS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER
        .u64_counter("paged_attention_calls")
        .with_description("Number of paged attention forward calls")
        .with_unit("calls")
        .build()
});

static PAGED_ATTENTION_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    METER
        .f64_histogram("paged_attention_duration_ms")
        .with_description("Duration of paged attention forward calls")
        .with_unit("ms")
        .build()
});

static CACHE_UPDATE_CALLS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER
        .u64_counter("cache_update_calls")
        .with_description("Number of cache update operations")
        .with_unit("calls")
        .build()
});

static CACHE_UPDATE_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    METER
        .f64_histogram("cache_update_duration_ms")
        .with_description("Duration of cache update operations")
        .with_unit("ms")
        .build()
});

static TENSOR_SIZE_BYTES: LazyLock<Histogram<u64>> = LazyLock::new(|| {
    METER
        .u64_histogram("tensor_size_bytes")
        .with_description("Size of tensors in bytes")
        .with_unit("bytes")
        .build()
});

pub fn record_paged_attention_call(metrics: &PagedAttentionMetrics) {
    let attributes = vec![
        KeyValue::new("num_sequences", metrics.num_sequences as i64),
        KeyValue::new("num_heads", metrics.num_heads as i64),
        KeyValue::new("head_size", metrics.head_size as i64),
        KeyValue::new("use_v1", metrics.use_v1),
    ];

    PAGED_ATTENTION_CALLS.add(1, &attributes);

    // Log as structured JSON for analysis
    debug!(
        target: "paged_attention_metrics",
        "{}",
        serde_json::to_string(metrics).unwrap_or_default()
    );
}

pub fn record_paged_attention_duration(duration_ms: f64, use_v1: bool) {
    let attributes = vec![KeyValue::new("use_v1", use_v1)];
    PAGED_ATTENTION_DURATION.record(duration_ms, &attributes);
}

pub fn record_cache_update(metrics: &CacheUpdateMetrics) {
    let attributes = vec![
        KeyValue::new("num_tokens", metrics.num_tokens as i64),
        KeyValue::new("num_heads", metrics.num_heads as i64),
    ];

    CACHE_UPDATE_CALLS.add(1, &attributes);

    // Log as structured JSON
    debug!(
        target: "cache_update_metrics",
        "{}",
        serde_json::to_string(metrics).unwrap_or_default()
    );
}

pub fn record_cache_update_duration(duration_ms: f64) {
    CACHE_UPDATE_DURATION.record(duration_ms, &[]);
}

pub fn record_tensor_size(name: &str, size_bytes: u64) {
    let attributes = vec![KeyValue::new("tensor_name", name.to_string())];
    TENSOR_SIZE_BYTES.record(size_bytes, &attributes);

    debug!(
        target: "tensor_metrics",
        "{}",
        serde_json::json!({
            "tensor_name": name,
            "size_bytes": size_bytes
        })
    );
}

// Initialize telemetry with OTLP exporter
#[cfg(feature = "telemetry")]
pub fn init_telemetry() -> Result<(), Box<dyn std::error::Error>> {
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::metrics::PeriodicReader;
    use opentelemetry_sdk::runtime;

    // Setup OTLP exporter
    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint("http://localhost:4317")
        .build()?;

    // Create a periodic reader that exports metrics every 10 seconds
    let reader = PeriodicReader::builder(exporter, runtime::Tokio)
        .with_interval(std::time::Duration::from_secs(10))
        .build();

    // Build the meter provider
    let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_reader(reader)
        .build();

    // Set as global provider
    global::set_meter_provider(provider);

    Ok(())
}

// Initialize Prometheus exporter
#[cfg(feature = "telemetry")]
pub fn init_prometheus() -> Result<prometheus::Registry, Box<dyn std::error::Error>> {
    use opentelemetry_prometheus::exporter;

    let registry = prometheus::Registry::new();
    let exporter = exporter()
        .with_registry(registry.clone())
        .build()?;

    // Build the meter provider
    let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_reader(exporter)
        .build();

    // Set as global provider
    global::set_meter_provider(provider);

    Ok(registry)
}