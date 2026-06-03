//! Prometheus metrics support for the mistral.rs server.
//! Installs a process-wide Prometheus recorder and exposes a `/metrics`
//! endpoint rendering metrics in the Prometheus text exposition format.
use axum::{
    extract::{MatchedPath, Request},
    middleware::Next,
    response::Response,
};
use axum::{http::StatusCode, response::IntoResponse};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;
use std::time::Instant;
static PROMETHEUS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
/// Install the global Prometheus recorder. Safe to call once at startup.
pub fn install_prometheus_recorder() {
    if PROMETHEUS_HANDLE.get().is_some() {
        return;
    }
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");
    let _ = PROMETHEUS_HANDLE.set(handle);
}
/// Axum handler for `GET /metrics`. Renders the Prometheus exposition format.
pub async fn metrics() -> impl IntoResponse {
    match PROMETHEUS_HANDLE.get() {
        Some(handle) => (StatusCode::OK, handle.render()).into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            "metrics recorder not initialized",
        )
            .into_response(),
    }
}
/// Axum middleware that records per-request metrics: a request counter and a
/// latency histogram, both labeled by method, path, and response status.
pub async fn track_metrics(req: Request, next: Next) -> Response {
    let method = req.method().to_string();
    // Use the matched route pattern (e.g. "/v1/files/{id}") rather than the
    // raw URI path so that high-cardinality segments (ids, session keys) and
    // arbitrary 404 paths do not each create a separate Prometheus time series.
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    let start = Instant::now();
    let response = next.run(req).await;
    let latency = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();
    let labels = [("method", method), ("path", path), ("status", status)];
    metrics::counter!("http_requests_total", &labels).increment(1);
    metrics::histogram!("http_request_duration_seconds", &labels).record(latency);
    response
}
