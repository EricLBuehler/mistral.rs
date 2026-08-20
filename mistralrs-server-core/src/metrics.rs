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
use axum::{
    http::{header::CONTENT_TYPE, StatusCode},
    response::IntoResponse,
};
use http_body::{Body as HttpBody, Frame};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll};
use std::time::Instant;
use tracing::{debug, info};

use crate::{
    handler_core::ResponseErrorMessage,
    lora_adapters::{
        is_resolvable_lora_adapter_model, lifecycle_body_too_large_response,
        list_lora_adapter_models,
    },
    mistralrs_server_router_builder::DEFAULT_MAX_BODY_LIMIT,
    streaming::{StreamOutcome, StreamOutcomeHandle},
    types::SharedMistralRsState,
};

static PROMETHEUS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
const REQUEST_ID_HEADER: &str = "x-request-id";
const UNMATCHED_ROUTE: &str = "<unmatched>";
const NO_MODEL: &str = "none";
const UNKNOWN_MODEL: &str = "unknown";
const DEFAULT_MODEL: &str = "default";
const OPTIONS_METHOD: &str = "OPTIONS";
const SSE_CONTENT_TYPE: &str = "text/event-stream";
// Error bodies are small; cap what we buffer to recover a message for the access log
const MAX_LOGGED_ERROR_BODY_BYTES: usize = 4 * 1024;
const MILLIS_PER_SECOND: f64 = 1_000.0;
const ACCESS_LOG_MS_ROUNDING: f64 = 1_000.0;
const HTTP_REQUEST_DURATION_BUCKETS: [f64; 18] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    600.0, 1_200.0,
];
const HTTP_REQUEST_BODY_BYTE_BUCKETS: [f64; 12] = [
    128.0,
    512.0,
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    52_428_800.0,
    104_857_600.0,
];
// Bucket lists taken from vLLM's Prometheus exporter
const TTFT_BUCKETS: [f64; 22] = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2_560.0,
];
const ITL_BUCKETS: [f64; 19] = [
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
    40.0, 80.0,
];
pub(crate) const TTFT_METRIC: &str = "mistralrs_time_to_first_token_seconds";
pub(crate) const ITL_METRIC: &str = "mistralrs_inter_token_latency_seconds";

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
}

/// Install the global Prometheus recorder. Safe to call once at startup.
pub fn install_prometheus_recorder() {
    if PROMETHEUS_HANDLE.get().is_some() {
        return;
    }
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full("http_request_duration_seconds".to_string()),
            &HTTP_REQUEST_DURATION_BUCKETS,
        )
        .expect("valid HTTP request duration buckets")
        .set_buckets_for_metric(
            Matcher::Full("http_request_body_bytes".to_string()),
            &HTTP_REQUEST_BODY_BYTE_BUCKETS,
        )
        .expect("valid HTTP request body byte buckets")
        .set_buckets_for_metric(Matcher::Full(TTFT_METRIC.to_string()), &TTFT_BUCKETS)
        .expect("valid TTFT buckets")
        .set_buckets_for_metric(Matcher::Full(ITL_METRIC.to_string()), &ITL_BUCKETS)
        .expect("valid ITL buckets")
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

struct InFlightGuard {
    labels: [(&'static str, String); 3],
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        metrics::gauge!("http_requests_in_flight", &self.labels).decrement(1.0);
    }
}

pub async fn observe_http(
    State(observability): State<ObservabilityState>,
    mut req: Request,
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
    let request_id = request_id(&mut req);
    let (req, model, body_bytes, early_response) =
        match extract_model(req, &route, &observability).await {
            Ok((req, model, body_bytes)) => (Some(req), model, body_bytes, None),
            Err(response) => (None, UNKNOWN_MODEL.to_string(), None, Some(response)),
        };
    let request_body_bytes = body_bytes.or(content_length_header);
    let housekeeping = is_housekeeping(&method, &route, &uri_path);
    let log_access = config.access_log && (config.access_log_health || !housekeeping);

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

    let in_flight = if config.metrics && !housekeeping {
        let labels = [
            ("method", method.clone()),
            ("path", route.clone()),
            ("model", model.clone()),
        ];
        metrics::gauge!("http_requests_in_flight", &labels).increment(1.0);
        if let Some(bytes) = request_body_bytes {
            metrics::histogram!("http_request_body_bytes", &labels).record(bytes as f64);
        }
        Some(InFlightGuard { labels })
    } else {
        None
    };

    let outcome_handle = StreamOutcomeHandle::default();
    let mut response = match early_response {
        Some(response) => response,
        None => {
            let mut req = req.expect("request exists without early response");
            req.extensions_mut().insert(outcome_handle.clone());
            next.run(req).await
        }
    };
    let status = response.status().as_u16().to_string();

    if config.request_id_header {
        if let Ok(value) = HeaderValue::from_str(&request_id) {
            response
                .headers_mut()
                .insert(HeaderName::from_static(REQUEST_ID_HEADER), value);
        }
    }

    let completion = RequestCompletion {
        config,
        request_id,
        method,
        route,
        model,
        status,
        start,
        housekeeping,
        log_access,
        in_flight,
    };

    // SSE bodies keep working long after the handler returns; finish accounting when the body ends
    if is_sse(&response) {
        completion.log_stream_accepted();
        // Labels and start are fixed here; the streamer is only polled once this body is consumed
        if completion.config.metrics {
            outcome_handle.set_latency_labels(
                [
                    ("method", completion.method.clone()),
                    ("path", completion.route.clone()),
                    ("model", completion.model.clone()),
                    ("status", completion.status.clone()),
                ],
                completion.start,
            );
        }
        let (parts, body) = response.into_parts();
        let body = Body::new(ObservedBody {
            inner: body,
            completion: Some(completion),
            outcome: outcome_handle,
            ended: false,
        });
        return Response::from_parts(parts, body);
    }

    let error = if response.status().is_client_error() || response.status().is_server_error() {
        match response.extensions().get::<ResponseErrorMessage>() {
            Some(ResponseErrorMessage(message)) => Some(message.clone()),
            None => {
                let (parts, body) = response.into_parts();
                let (message, body) = match to_bytes(body, MAX_LOGGED_ERROR_BODY_BYTES).await {
                    Ok(bytes) => (error_message_from_body(&bytes), Body::from(bytes)),
                    Err(_) => (None, Body::empty()),
                };
                response = Response::from_parts(parts, body);
                message
            }
        }
    } else {
        None
    };
    completion.finish(error.map(RequestError::Message));
    response
}

/// Axum rejections and our JSON errors carry the message as `{"message": ...}` or raw text.
fn error_message_from_body(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?.trim();
    if text.is_empty() {
        return None;
    }
    let message = serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| {
            value
                .get("message")
                .or_else(|| value.get("error").and_then(|e| e.get("message")))
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| text.to_string());
    Some(message)
}

fn is_sse(response: &Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with(SSE_CONTENT_TYPE))
}

/// Everything needed to emit the "request completed" line and metrics once, whenever the request truly ends.
struct RequestCompletion {
    config: ObservabilityConfig,
    request_id: String,
    method: String,
    route: String,
    model: String,
    status: String,
    start: Instant,
    housekeeping: bool,
    log_access: bool,
    in_flight: Option<InFlightGuard>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamEnd {
    Completed,
    Error,
    ClientDisconnected,
}

impl StreamEnd {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Error => "error",
            Self::ClientDisconnected => "client_disconnected",
        }
    }
}

struct StreamStats {
    end: StreamEnd,
    outcome: StreamOutcome,
}

enum RequestError {
    Message(String),
    Stream(StreamStats),
}

impl RequestCompletion {
    fn log_stream_accepted(&self) {
        debug!(
            "stream accepted: request_id={} method={} route={} model={} status={} accepted_ms={:.3}",
            self.request_id,
            self.method,
            self.route,
            self.model,
            self.status,
            rounded_duration_ms(self.start.elapsed().as_secs_f64())
        );
    }

    fn finish(self, detail: Option<RequestError>) {
        let latency = self.start.elapsed().as_secs_f64();
        if self.config.metrics && !self.housekeeping {
            let labels = [
                ("method", self.method.clone()),
                ("path", self.route.clone()),
                ("model", self.model.clone()),
                ("status", self.status.clone()),
            ];
            metrics::counter!("http_requests_total", &labels).increment(1);
            metrics::histogram!("http_request_duration_seconds", &labels).record(latency);
        }
        drop(self.in_flight);

        if self.log_access {
            log_request_done(
                self.config.access_log_format,
                &self.request_id,
                &self.method,
                &self.route,
                &self.model,
                &self.status,
                latency,
                detail.as_ref(),
            );
        } else {
            debug!(
                "request completed: request_id={} method={} route={} model={} status={} duration_ms={:.3}",
                self.request_id,
                self.method,
                self.route,
                self.model,
                self.status,
                rounded_duration_ms(latency)
            );
        }
    }
}

/// Wraps an SSE body so request accounting fires when the stream ends or the client goes away.
struct ObservedBody {
    inner: Body,
    completion: Option<RequestCompletion>,
    outcome: StreamOutcomeHandle,
    ended: bool,
}

impl ObservedBody {
    fn finish(&mut self, end: StreamEnd) {
        if let Some(completion) = self.completion.take() {
            let outcome = self.outcome.snapshot();
            let end = if outcome.error.is_some() {
                StreamEnd::Error
            } else {
                end
            };
            completion.finish(Some(RequestError::Stream(StreamStats { end, outcome })));
        }
    }
}

impl HttpBody for ObservedBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = &mut *self;
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(None) => {
                this.ended = true;
                this.finish(StreamEnd::Completed);
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(err))) => {
                this.ended = true;
                this.finish(StreamEnd::Error);
                Poll::Ready(Some(Err(err)))
            }
            other => other,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for ObservedBody {
    fn drop(&mut self) {
        if !self.ended {
            self.finish(StreamEnd::ClientDisconnected);
        }
    }
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
    let Some(field) = model_label_field(route) else {
        return Ok((req, NO_MODEL.to_string(), None));
    };

    if matches!(field, ModelLabelField::LoraModelQuery) {
        let model = query_model(req.uri().query());
        return Ok((
            req,
            resolve_normalized_model_label(model.as_deref(), observability),
            None,
        ));
    }

    let (parts, body) = req.into_parts();
    let bytes = to_bytes(body, observability.max_body_bytes)
        .await
        .map_err(|_| body_too_large_response(route))?;
    let body_bytes = Some(bytes.len() as u64);
    let model = match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(value) => resolve_model_label(&value, field, observability),
        Err(_) => UNKNOWN_MODEL.to_string(),
    };
    Ok((
        Request::from_parts(parts, Body::from(bytes)),
        model,
        body_bytes,
    ))
}

fn body_too_large_response(route: &str) -> Response {
    if matches!(route, "/v1/load_lora_adapter" | "/v1/unload_lora_adapter") {
        lifecycle_body_too_large_response()
    } else {
        (StatusCode::PAYLOAD_TOO_LARGE, "request body too large").into_response()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelLabelField {
    Model,
    ModelId,
    LoraModel,
    LoraModelQuery,
}

fn resolve_model_label(
    value: &serde_json::Value,
    field: ModelLabelField,
    observability: &ObservabilityState,
) -> String {
    let model = match field {
        ModelLabelField::Model | ModelLabelField::LoraModel | ModelLabelField::LoraModelQuery => {
            value.get("model").and_then(|model| model.as_str())
        }
        ModelLabelField::ModelId => value.get("model_id").and_then(|model| model.as_str()),
    };

    match field {
        ModelLabelField::Model => resolve_defaultable_model_label(model, observability),
        ModelLabelField::LoraModel | ModelLabelField::LoraModelQuery => {
            resolve_normalized_model_label(model, observability)
        }
        ModelLabelField::ModelId => resolve_explicit_model_label(model, observability),
    }
}

fn query_model(query: Option<&str>) -> Option<String> {
    url::form_urlencoded::parse(query?.as_bytes())
        .find_map(|(key, value)| (key == "model" && !value.is_empty()).then(|| value.into_owned()))
}

fn resolve_defaultable_model_label(
    model: Option<&str>,
    observability: &ObservabilityState,
) -> String {
    match model.filter(|model| !model.is_empty()) {
        Some(DEFAULT_MODEL) | None => observability
            .mistralrs
            .get_default_model_id()
            .ok()
            .flatten()
            .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        Some(model) => known_model_label(model, observability),
    }
}

fn resolve_explicit_model_label(model: Option<&str>, observability: &ObservabilityState) -> String {
    model
        .filter(|model| !model.is_empty())
        .map(|model| known_model_label(model, observability))
        .unwrap_or_else(|| UNKNOWN_MODEL.to_string())
}

fn resolve_normalized_model_label(
    model: Option<&str>,
    observability: &ObservabilityState,
) -> String {
    resolve_defaultable_model_label(normalize_model_label_input(model), observability)
}

fn normalize_model_label_input(model: Option<&str>) -> Option<&str> {
    model.map(str::trim).filter(|model| !model.is_empty())
}

fn known_model_label(model: &str, observability: &ObservabilityState) -> String {
    let base_model_exists = observability
        .mistralrs
        .get_model_status(model)
        .ok()
        .flatten()
        .is_some();
    let adapter_model_exists = list_lora_adapter_models(&observability.mistralrs)
        .ok()
        .is_some_and(|models| adapter_model_label_is_known(model, &models));
    if base_model_exists || adapter_model_exists {
        model.to_string()
    } else {
        UNKNOWN_MODEL.to_string()
    }
}

fn adapter_model_label_is_known(
    model: &str,
    models: &[crate::lora_adapters::LoraAdapterModel],
) -> bool {
    is_resolvable_lora_adapter_model(models, model)
}

fn is_housekeeping(method: &str, route: &str, uri_path: &str) -> bool {
    if method == OPTIONS_METHOD {
        return true;
    }

    matches!(
        route,
        "/" | "/health"
            | "/metrics"
            | "/docs"
            | "/docs/"
            | "/docs/{*rest}"
            | "/api-doc/openapi.json"
    ) || route.starts_with("/ui")
        || uri_path.starts_with("/ui")
}

fn model_label_field(route: &str) -> Option<ModelLabelField> {
    if matches!(
        route,
        "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/responses"
            | "/v1/messages"
            | "/v1/messages/count_tokens"
            | "/v1/embeddings"
            | "/v1/images/generations"
            | "/v1/audio/speech"
    ) {
        return Some(ModelLabelField::Model);
    }

    if matches!(route, "/v1/load_lora_adapter" | "/v1/unload_lora_adapter") {
        return Some(ModelLabelField::LoraModel);
    }

    if route == "/v1/lora_adapters" {
        return Some(ModelLabelField::LoraModelQuery);
    }

    if matches!(
        route,
        "/v1/models/unload" | "/v1/models/reload" | "/v1/models/status" | "/v1/models/tune"
    ) {
        return Some(ModelLabelField::ModelId);
    }

    None
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

#[allow(clippy::too_many_arguments)]
fn log_request_done(
    format: AccessLogFormat,
    request_id: &str,
    method: &str,
    route: &str,
    model: &str,
    status: &str,
    latency: f64,
    detail: Option<&RequestError>,
) {
    let duration_ms = rounded_duration_ms(latency);
    let stream = match detail {
        Some(RequestError::Stream(stats)) => Some(stats),
        _ => None,
    };
    let usage = stream.and_then(|stats| stats.outcome.usage.as_ref());
    let error = match detail {
        Some(RequestError::Message(message)) => Some(message.as_str()),
        Some(RequestError::Stream(stats)) => stats.outcome.error.as_deref(),
        None => None,
    };
    match format {
        AccessLogFormat::Text => {
            let mut line = format!(
                "request completed: request_id={request_id} method={method} route={route} model={model} status={status}"
            );
            if let Some(stats) = stream {
                line.push_str(&format!(" outcome={}", stats.end.as_str()));
            }
            line.push_str(&format!(" duration_ms={duration_ms:.3}"));
            if let Some(usage) = usage {
                line.push_str(&format!(
                    " prompt_tokens={} completion_tokens={} prefill_tok_s={:.1} decode_tok_s={:.1}",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.avg_prompt_tok_per_sec,
                    usage.avg_compl_tok_per_sec
                ));
            }
            if let Some(error) = error {
                line.push_str(&format!(" error={error:?}"));
            }
            info!("{line}");
        }
        AccessLogFormat::Json => {
            let mut record = serde_json::json!({
                "event": "request_completed",
                "request_id": request_id,
                "method": method,
                "route": route,
                "model": model,
                "status": status,
                "duration_ms": duration_ms,
            });
            if let Some(stats) = stream {
                record["outcome"] = serde_json::Value::String(stats.end.as_str().to_string());
            }
            if let Some(usage) = usage {
                record["prompt_tokens"] = usage.prompt_tokens.into();
                record["completion_tokens"] = usage.completion_tokens.into();
                record["prefill_tok_s"] = usage.avg_prompt_tok_per_sec.into();
                record["decode_tok_s"] = usage.avg_compl_tok_per_sec.into();
            }
            if let Some(error) = error {
                record["error"] = serde_json::Value::String(error.to_string());
            }
            info!("{record}");
        }
    }
}

fn rounded_duration_ms(latency_seconds: f64) -> f64 {
    let millis = latency_seconds * MILLIS_PER_SECOND;
    (millis * ACCESS_LOG_MS_ROUNDING).round() / ACCESS_LOG_MS_ROUNDING
}

#[cfg(test)]
mod tests {
    use super::{
        adapter_model_label_is_known, body_too_large_response, model_label_field,
        normalize_model_label_input, query_model, ModelLabelField,
    };
    use crate::lora_adapters::LoraAdapterModel;
    use axum::body::to_bytes;
    use mistralrs_core::{AdapterGenerationId, LoraAdapterInfo};

    fn adapter_model(id: &str, parent: &str, alias: &str) -> LoraAdapterModel {
        LoraAdapterModel {
            id: id.to_string(),
            parent: parent.to_string(),
            adapter: LoraAdapterInfo {
                alias: alias.to_string(),
                source: "source".to_string(),
                revision: None,
                generation: AdapterGenerationId::from_bytes([1; 32]),
                rank: 8,
                bytes: 16,
            },
        }
    }

    #[test]
    fn extracts_model_from_adapter_list_query() {
        assert_eq!(
            query_model(Some("ignored=value&model=org%2Fmodel")),
            Some("org/model".to_string())
        );
        assert_eq!(query_model(Some("model=")), None);
        assert_eq!(query_model(None), None);
    }

    #[test]
    fn model_labels_use_trimmed_request_ids() {
        assert_eq!(normalize_model_label_input(None), None);
        assert_eq!(normalize_model_label_input(Some("   ")), None);
        assert_eq!(normalize_model_label_input(Some(" model ")), Some("model"));
        assert_eq!(
            normalize_model_label_input(Some(" default ")),
            Some("default")
        );
    }

    #[test]
    fn only_lora_management_routes_trim_model_ids() {
        assert_eq!(
            model_label_field("/v1/load_lora_adapter"),
            Some(ModelLabelField::LoraModel)
        );
        assert_eq!(
            model_label_field("/v1/lora_adapters"),
            Some(ModelLabelField::LoraModelQuery)
        );
        assert_eq!(
            model_label_field("/v1/chat/completions"),
            Some(ModelLabelField::Model)
        );
        assert_eq!(
            model_label_field("/v1/models/status"),
            Some(ModelLabelField::ModelId)
        );
    }

    #[test]
    fn adapter_model_labels_are_bounded_by_resolvable_cards() {
        let models = vec![
            adapter_model("base-a::code", "base-a", "code"),
            adapter_model("base-b::code", "base-b", "code"),
            adapter_model("base-a::math", "base-a", "math"),
        ];

        assert!(adapter_model_label_is_known("base-a::code", &models));
        assert!(adapter_model_label_is_known("math", &models));
        assert!(!adapter_model_label_is_known("code", &models));
        assert!(!adapter_model_label_is_known(
            "unbounded-user-value",
            &models
        ));
    }

    #[tokio::test]
    async fn lifecycle_body_limit_uses_the_lora_error_envelope() {
        let response = body_too_large_response("/v1/load_lora_adapter");
        assert_eq!(response.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "request_body_too_large");
        assert_eq!(value["error"]["type"], "invalid_request_error");
    }
}
