//! Server observability: access logs, request ids, and Prometheus metrics.
use axum::{
    extract::{MatchedPath, Request, State},
    http::{header::HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use axum::{http::StatusCode, response::IntoResponse};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;
use std::time::Instant;
use tracing::{debug, info};

static PROMETHEUS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
const REQUEST_ID_HEADER: &str = "x-request-id";
const UNMATCHED_ROUTE: &str = "<unmatched>";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AccessLogFormat {
    #[default]
    Text,
    Json,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ObservabilityConfig {
    #[serde(default = "default_true")]
    pub access_log: bool,
    #[serde(default)]
    pub access_log_health: bool,
    #[serde(default)]
    pub access_log_format: AccessLogFormat,
    #[serde(default = "default_true")]
    pub request_id_header: bool,
    #[serde(default = "default_true")]
    pub metrics: bool,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            access_log: true,
            access_log_health: false,
            access_log_format: AccessLogFormat::Text,
            request_id_header: true,
            metrics: true,
        }
    }
}

fn default_true() -> bool {
    true
}

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
#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/metrics",
    responses(
        (status = 200, description = "Prometheus text exposition format", content_type = "text/plain"),
        (status = 503, description = "Metrics recorder not initialized or metrics disabled"),
    )
)]
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

pub async fn metrics_disabled() -> impl IntoResponse {
    (StatusCode::SERVICE_UNAVAILABLE, "metrics disabled")
}

pub async fn observe_http(
    State(config): State<ObservabilityConfig>,
    mut req: Request,
    next: Next,
) -> Response {
    let method = req.method().to_string();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| UNMATCHED_ROUTE.to_string());
    let uri_path = req.uri().path().to_string();
    let content_length = req
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let request_id = request_id(&mut req);
    let log_access = config.access_log && (config.access_log_health || !is_housekeeping(&route));

    if log_access {
        log_request_start(
            config.access_log_format,
            &request_id,
            &method,
            &route,
            &uri_path,
            content_length,
        );
    }

    if config.metrics && !is_housekeeping(&route) {
        let labels = [("method", method.clone()), ("path", route.clone())];
        metrics::gauge!("http_requests_in_flight", &labels).increment(1.0);
        if let Some(bytes) = content_length {
            metrics::histogram!("http_request_body_bytes", &labels).record(bytes as f64);
        }
    }

    let start = Instant::now();
    let mut response = next.run(req).await;
    let latency = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    if config.request_id_header {
        if let Ok(value) = HeaderValue::from_str(&request_id) {
            response
                .headers_mut()
                .insert(HeaderName::from_static(REQUEST_ID_HEADER), value);
        }
    }

    if config.metrics && !is_housekeeping(&route) {
        let active_labels = [("method", method.clone()), ("path", route.clone())];
        metrics::gauge!("http_requests_in_flight", &active_labels).decrement(1.0);
        let labels = [
            ("method", method.clone()),
            ("path", route.clone()),
            ("status", status.clone()),
        ];
        metrics::counter!("http_requests_total", &labels).increment(1);
        metrics::histogram!("http_request_duration_seconds", &labels).record(latency);
    }

    if log_access {
        log_request_done(
            config.access_log_format,
            &request_id,
            &method,
            &route,
            &status,
            latency,
        );
    } else {
        debug!(
            request_id = %request_id,
            method = %method,
            route = %route,
            status = %status,
            duration_ms = latency * 1000.0,
            "HTTP request completed"
        );
    }

    response
}

fn request_id(req: &mut Request) -> String {
    let request_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("req_{}", uuid::Uuid::new_v4().simple()));
    req.extensions_mut().insert(RequestId(request_id.clone()));
    request_id
}

#[derive(Clone, Debug)]
pub struct RequestId(pub String);

fn is_housekeeping(route: &str) -> bool {
    matches!(
        route,
        "/" | "/health"
            | "/metrics"
            | "/docs"
            | "/docs/"
            | "/docs/{*rest}"
            | "/api-doc/openapi.json"
    ) || route.starts_with("/ui")
}

fn log_request_start(
    format: AccessLogFormat,
    request_id: &str,
    method: &str,
    route: &str,
    uri_path: &str,
    content_length: Option<u64>,
) {
    match format {
        AccessLogFormat::Text => info!(
            request_id = %request_id,
            method = %method,
            route = %route,
            path = %uri_path,
            content_length,
            "HTTP request started"
        ),
        AccessLogFormat::Json => info!(
            "{}",
            serde_json::json!({
                "event": "http_request_started",
                "request_id": request_id,
                "method": method,
                "route": route,
                "path": uri_path,
                "content_length": content_length,
            })
        ),
    }
}

fn log_request_done(
    format: AccessLogFormat,
    request_id: &str,
    method: &str,
    route: &str,
    status: &str,
    latency: f64,
) {
    match format {
        AccessLogFormat::Text => info!(
            request_id = %request_id,
            method = %method,
            route = %route,
            status = %status,
            duration_ms = latency * 1000.0,
            "HTTP request completed"
        ),
        AccessLogFormat::Json => info!(
            "{}",
            serde_json::json!({
                "event": "http_request_completed",
                "request_id": request_id,
                "method": method,
                "route": route,
                "status": status,
                "duration_ms": latency * 1000.0,
            })
        ),
    }
}
