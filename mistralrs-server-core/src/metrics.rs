//! Server observability: access logs, request ids, and Prometheus metrics.
use axum::{
    body::{to_bytes, Body},
    extract::{MatchedPath, Request, State},
    http::{
        header::{HeaderName, CONTENT_LENGTH},
        HeaderValue,
    },
    middleware::Next,
    response::Response,
};
use axum::{http::StatusCode, response::IntoResponse};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;
use std::time::Instant;
use tracing::{debug, info};

use crate::{mistralrs_server_router_builder::DEFAULT_MAX_BODY_LIMIT, types::SharedMistralRsState};

static PROMETHEUS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
const REQUEST_ID_HEADER: &str = "x-request-id";
const UNMATCHED_ROUTE: &str = "<unmatched>";
const NO_MODEL: &str = "none";
const UNKNOWN_MODEL: &str = "unknown";
const DEFAULT_MODEL: &str = "default";
const MILLIS_PER_SECOND: f64 = 1_000.0;
const ACCESS_LOG_MS_ROUNDING: f64 = 1_000.0;

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

#[derive(Clone)]
pub struct ObservabilityState {
    config: ObservabilityConfig,
    mistralrs: SharedMistralRsState,
    max_body_bytes: usize,
}

impl ObservabilityState {
    pub fn new(config: ObservabilityConfig, mistralrs: SharedMistralRsState) -> Self {
        Self::with_max_body_bytes(config, mistralrs, DEFAULT_MAX_BODY_LIMIT)
    }

    pub fn with_max_body_bytes(
        config: ObservabilityConfig,
        mistralrs: SharedMistralRsState,
        max_body_bytes: usize,
    ) -> Self {
        Self {
            config,
            mistralrs,
            max_body_bytes,
        }
    }

    pub fn metrics_enabled(&self) -> bool {
        self.config.metrics
    }
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
    State(observability): State<ObservabilityState>,
    req: Request,
    next: Next,
) -> Response {
    let config = observability.config.clone();
    let start = Instant::now();
    let method = req.method().to_string();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| UNMATCHED_ROUTE.to_string());
    let uri_path = req.uri().path().to_string();
    let content_length_header = req
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let (mut req, model, body_bytes) = match extract_model(req, &route, &observability).await {
        Ok(result) => result,
        Err(response) => return response,
    };
    let request_body_bytes = body_bytes.or(content_length_header);
    let request_id = request_id(&mut req);
    let log_access = config.access_log && (config.access_log_health || !is_housekeeping(&route));

    if log_access {
        log_request_start(
            config.access_log_format,
            &request_id,
            &method,
            &route,
            &uri_path,
            &model,
            request_body_bytes,
        );
    }

    if config.metrics && !is_housekeeping(&route) {
        let labels = [
            ("method", method.clone()),
            ("path", route.clone()),
            ("model", model.clone()),
        ];
        metrics::gauge!("http_requests_in_flight", &labels).increment(1.0);
        if let Some(bytes) = request_body_bytes {
            metrics::histogram!("http_request_body_bytes", &labels).record(bytes as f64);
        }
    }

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
        let active_labels = [
            ("method", method.clone()),
            ("path", route.clone()),
            ("model", model.clone()),
        ];
        metrics::gauge!("http_requests_in_flight", &active_labels).decrement(1.0);
        let labels = [
            ("method", method.clone()),
            ("path", route.clone()),
            ("model", model.clone()),
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
            &model,
            &status,
            latency,
        );
    } else {
        let duration_ms = rounded_duration_ms(latency);
        debug!(
            "request completed: request_id={} method={} route={} model={} status={} duration_ms={:.3}",
            request_id, method, route, model, status, duration_ms
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

async fn extract_model(
    req: Request,
    route: &str,
    observability: &ObservabilityState,
) -> Result<(Request, String, Option<u64>), Response> {
    if !is_model_route(route) {
        return Ok((req, NO_MODEL.to_string(), None));
    }

    let (parts, body) = req.into_parts();
    let bytes = to_bytes(body, observability.max_body_bytes)
        .await
        .map_err(|_| (StatusCode::PAYLOAD_TOO_LARGE, "request body too large").into_response())?;
    let body_bytes = Some(bytes.len() as u64);
    let model = match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(value) => {
            let model = value.get("model").and_then(|model| model.as_str());
            resolve_model_label(model, observability)
        }
        Err(_) => UNKNOWN_MODEL.to_string(),
    };
    Ok((
        Request::from_parts(parts, Body::from(bytes)),
        model,
        body_bytes,
    ))
}

fn resolve_model_label(model: Option<&str>, observability: &ObservabilityState) -> String {
    match model.filter(|model| !model.is_empty()) {
        Some(DEFAULT_MODEL) | None => observability
            .mistralrs
            .get_default_model_id()
            .ok()
            .flatten()
            .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        Some(model) => model.to_string(),
    }
}

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

fn is_model_route(route: &str) -> bool {
    matches!(
        route,
        "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/responses"
            | "/v1/messages"
            | "/v1/messages/count_tokens"
            | "/v1/embeddings"
            | "/v1/images/generations"
            | "/v1/audio/speech"
    )
}

fn log_request_start(
    format: AccessLogFormat,
    request_id: &str,
    method: &str,
    route: &str,
    uri_path: &str,
    model: &str,
    content_length: Option<u64>,
) {
    match format {
        AccessLogFormat::Text => match content_length {
            Some(content_length) => info!(
                "request started: request_id={} method={} route={} path={} model={} content_length={}",
                request_id, method, route, uri_path, model, content_length
            ),
            None => info!(
                "request started: request_id={} method={} route={} path={} model={}",
                request_id, method, route, uri_path, model
            ),
        },
        AccessLogFormat::Json => info!(
            "{}",
            serde_json::json!({
                "event": "request_started",
                "request_id": request_id,
                "method": method,
                "route": route,
                "path": uri_path,
                "model": model,
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
    model: &str,
    status: &str,
    latency: f64,
) {
    let duration_ms = rounded_duration_ms(latency);
    match format {
        AccessLogFormat::Text => info!(
            "request completed: request_id={} method={} route={} model={} status={} duration_ms={:.3}",
            request_id, method, route, model, status, duration_ms
        ),
        AccessLogFormat::Json => info!(
            "{}",
            serde_json::json!({
                "event": "request_completed",
                "request_id": request_id,
                "method": method,
                "route": route,
                "model": model,
                "status": status,
                "duration_ms": duration_ms,
            })
        ),
    }
}

fn rounded_duration_ms(latency_seconds: f64) -> f64 {
    let millis = latency_seconds * MILLIS_PER_SECOND;
    (millis * ACCESS_LOG_MS_ROUNDING).round() / ACCESS_LOG_MS_ROUNDING
}
