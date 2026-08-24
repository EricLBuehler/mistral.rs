//! Core functionality for handlers.

use anyhow::Result;
use axum::{
    extract::{rejection::JsonRejection, Json},
    http::StatusCode,
    response::IntoResponse,
};
use mistralrs_core::{
    LoraAdapterError, MistralRsError, Request, Response, ServiceUnavailableError,
};
use serde::Serialize;
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::types::SharedMistralRsState;

/// Default buffer size for the response channel used in streaming operations.
///
/// This constant defines the maximum number of response messages that can be buffered
/// in the channel before backpressure is applied. A larger buffer reduces the likelihood
/// of blocking but uses more memory.
pub const DEFAULT_CHANNEL_BUFFER_SIZE: usize = 10_000;

pub(crate) const INTERNAL_ERROR_MESSAGE: &str = "Internal server error.";
pub(crate) const MODEL_ERROR_MESSAGE: &str = "The model failed to process the request.";
pub(crate) const SERVICE_UNAVAILABLE_MESSAGE: &str = "The service is temporarily unavailable.";

/// Error message attached to a failed response so the access log can report it.
#[derive(Clone, Debug)]
pub struct ResponseErrorMessage(pub String);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ApiErrorKind {
    InvalidRequest,
    NotFound,
    Conflict,
    PayloadTooLarge,
    UnsupportedMediaType,
    RateLimited,
    Unavailable,
    Overloaded,
    Internal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ApiError {
    pub(crate) kind: ApiErrorKind,
    pub(crate) message: String,
    pub(crate) code: Option<String>,
    pub(crate) param: Option<String>,
}

impl ApiError {
    pub(crate) fn new(
        kind: ApiErrorKind,
        message: impl Into<String>,
        code: Option<&str>,
        param: Option<&str>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            code: code.map(ToString::to_string),
            param: param.map(ToString::to_string),
        }
    }

    pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::InvalidRequest, message, None, None)
    }

    pub(crate) fn internal() -> Self {
        Self::new(
            ApiErrorKind::Internal,
            INTERNAL_ERROR_MESSAGE,
            Some("internal_error"),
            None,
        )
    }

    pub(crate) fn model_error() -> Self {
        Self::new(
            ApiErrorKind::Internal,
            MODEL_ERROR_MESSAGE,
            Some("model_error"),
            None,
        )
    }

    pub(crate) fn from_status(status: StatusCode, message: impl Into<String>) -> Self {
        let message = message.into();
        match status {
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
                Self::invalid_request(message)
            }
            StatusCode::NOT_FOUND => {
                Self::new(ApiErrorKind::NotFound, message, Some("not_found"), None)
            }
            StatusCode::CONFLICT => {
                Self::new(ApiErrorKind::Conflict, message, Some("conflict"), None)
            }
            StatusCode::PAYLOAD_TOO_LARGE => Self::new(
                ApiErrorKind::PayloadTooLarge,
                message,
                Some("request_body_too_large"),
                None,
            ),
            StatusCode::UNSUPPORTED_MEDIA_TYPE => Self::new(
                ApiErrorKind::UnsupportedMediaType,
                message,
                Some("invalid_content_type"),
                None,
            ),
            StatusCode::TOO_MANY_REQUESTS => Self::new(
                ApiErrorKind::RateLimited,
                message,
                Some("rate_limit_exceeded"),
                None,
            ),
            StatusCode::SERVICE_UNAVAILABLE => Self::new(
                ApiErrorKind::Unavailable,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            ),
            _ if status.is_server_error() => Self::internal(),
            _ => Self::invalid_request(message),
        }
    }

    pub(crate) fn from_error(
        error: &(dyn std::error::Error + 'static),
        fallback: ApiErrorKind,
    ) -> Self {
        if let Some(error) = find_error::<ApiError>(error) {
            return error.clone();
        }
        if let Some(error) = find_error::<MistralRsError>(error) {
            return Self::from_mistralrs_error(error);
        }
        if find_error::<ServiceUnavailableError>(error).is_some() {
            return Self::new(
                ApiErrorKind::Overloaded,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            );
        }

        match fallback {
            ApiErrorKind::Internal => Self::internal(),
            ApiErrorKind::Unavailable => Self::new(
                ApiErrorKind::Unavailable,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            ),
            ApiErrorKind::Overloaded => Self::new(
                ApiErrorKind::Overloaded,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            ),
            kind => Self::new(kind, error.to_string(), None, None),
        }
    }

    pub(crate) fn from_json_rejection(error: JsonRejection) -> Self {
        let status = error.status();
        let code = match status {
            StatusCode::PAYLOAD_TOO_LARGE => "request_body_too_large",
            StatusCode::UNSUPPORTED_MEDIA_TYPE => "invalid_content_type",
            StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_body",
            _ => "malformed_json",
        };
        let kind = match status {
            StatusCode::PAYLOAD_TOO_LARGE => ApiErrorKind::PayloadTooLarge,
            StatusCode::UNSUPPORTED_MEDIA_TYPE => ApiErrorKind::UnsupportedMediaType,
            _ => ApiErrorKind::InvalidRequest,
        };
        Self::new(kind, error.body_text(), Some(code), None)
    }

    fn from_mistralrs_error(error: &MistralRsError) -> Self {
        match error {
            MistralRsError::ModelNotFound(_) => Self::new(
                ApiErrorKind::NotFound,
                error.to_string(),
                Some("model_not_found"),
                Some("model"),
            ),
            MistralRsError::ModelReloading(_)
            | MistralRsError::ModelAlreadyLoaded(_)
            | MistralRsError::ModelAlreadyUnloaded(_) => Self::new(
                ApiErrorKind::Conflict,
                error.to_string(),
                Some("model_state_conflict"),
                Some("model"),
            ),
            MistralRsError::NoLoaderConfig(_) => Self::new(
                ApiErrorKind::InvalidRequest,
                error.to_string(),
                Some("invalid_model_operation"),
                Some("model"),
            ),
            MistralRsError::LoraAdapter(error) => Self::from_lora_error(error),
            MistralRsError::EnginePoisoned
            | MistralRsError::ReloadFailed(_)
            | MistralRsError::Other(_) => Self::internal(),
            MistralRsError::SenderPoisoned => Self::new(
                ApiErrorKind::Unavailable,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            ),
        }
    }

    fn from_lora_error(error: &LoraAdapterError) -> Self {
        let (kind, code) = match error {
            LoraAdapterError::RuntimeUnavailable { .. }
            | LoraAdapterError::TensorParallelUnsupported { .. }
            | LoraAdapterError::RuntimeChanged { .. } => {
                (ApiErrorKind::Conflict, "lora_runtime_unavailable")
            }
            LoraAdapterError::InvalidAlias | LoraAdapterError::AliasTooLong { .. } => {
                (ApiErrorKind::InvalidRequest, "invalid_lora_name")
            }
            LoraAdapterError::LoadBusy => (ApiErrorKind::RateLimited, "lora_load_busy"),
            LoraAdapterError::NotFound { .. } | LoraAdapterError::GenerationNotFound { .. } => {
                (ApiErrorKind::NotFound, "lora_adapter_not_found")
            }
            LoraAdapterError::FileTooLarge { .. } => {
                (ApiErrorKind::PayloadTooLarge, "lora_adapter_file_too_large")
            }
            LoraAdapterError::Io { source, .. }
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                (ApiErrorKind::NotFound, "adapter_file_not_found")
            }
            LoraAdapterError::Io { source, .. }
                if matches!(
                    source.kind(),
                    std::io::ErrorKind::InvalidData
                        | std::io::ErrorKind::InvalidInput
                        | std::io::ErrorKind::UnexpectedEof
                ) =>
            {
                (ApiErrorKind::InvalidRequest, "invalid_lora_adapter")
            }
            LoraAdapterError::Io { .. } | LoraAdapterError::Load(_) => {
                (ApiErrorKind::Internal, "internal_error")
            }
            LoraAdapterError::Config { .. } | LoraAdapterError::Format(_) => {
                (ApiErrorKind::InvalidRequest, "invalid_lora_adapter")
            }
            LoraAdapterError::AlreadyLoaded { .. }
            | LoraAdapterError::GenerationMismatch { .. }
            | LoraAdapterError::GenerationConflict { .. }
            | LoraAdapterError::AliasLimit { .. }
            | LoraAdapterError::RankLimit { .. }
            | LoraAdapterError::AdapterLimit { .. }
            | LoraAdapterError::ByteLimit { .. }
            | LoraAdapterError::SlotExhausted => (ApiErrorKind::Conflict, "lora_state_conflict"),
            LoraAdapterError::SizeOverflow => {
                (ApiErrorKind::InvalidRequest, "invalid_lora_adapter")
            }
            LoraAdapterError::InvalidRuntimeConfig(_) | LoraAdapterError::Task(_) => {
                (ApiErrorKind::Internal, "internal_error")
            }
            _ => (ApiErrorKind::Internal, "internal_error"),
        };
        let message = if kind == ApiErrorKind::Internal {
            INTERNAL_ERROR_MESSAGE.to_string()
        } else if matches!(kind, ApiErrorKind::Unavailable | ApiErrorKind::Overloaded) {
            SERVICE_UNAVAILABLE_MESSAGE.to_string()
        } else {
            error.to_string()
        };
        Self::new(kind, message, Some(code), Some("adapter"))
    }

    pub(crate) fn status(&self) -> StatusCode {
        match self.kind {
            ApiErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
            ApiErrorKind::NotFound => StatusCode::NOT_FOUND,
            ApiErrorKind::Conflict => StatusCode::CONFLICT,
            ApiErrorKind::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            ApiErrorKind::UnsupportedMediaType => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ApiErrorKind::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            ApiErrorKind::Unavailable | ApiErrorKind::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
            ApiErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ApiError {}

fn find_error<'a, E: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a E> {
    loop {
        if let Some(error) = error.downcast_ref::<E>() {
            return Some(error);
        }
        error = error.source()?;
    }
}

#[derive(Debug, Serialize)]
struct OpenAiErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiErrorResponse {
    error: OpenAiErrorBody,
}

pub(crate) fn openai_error_response(error: ApiError) -> axum::response::Response {
    let error_type = match error.kind {
        ApiErrorKind::RateLimited => "rate_limit_error",
        ApiErrorKind::Unavailable | ApiErrorKind::Overloaded | ApiErrorKind::Internal => {
            "server_error"
        }
        _ => "invalid_request_error",
    };
    let status = error.status();
    let message = error.message;
    let mut response = Json(OpenAiErrorResponse {
        error: OpenAiErrorBody {
            message: message.clone(),
            error_type,
            param: error.param,
            code: error.code,
        },
    })
    .into_response();
    *response.status_mut() = status;
    response
        .extensions_mut()
        .insert(ResponseErrorMessage(message));
    response
}

pub(crate) fn openai_error_from_error(
    error: &(dyn std::error::Error + 'static),
    fallback: ApiErrorKind,
) -> axum::response::Response {
    openai_error_response(ApiError::from_error(error, fallback))
}

/// Standard JSON error response structure.
#[derive(Serialize, Debug)]
pub(crate) struct JsonError {
    pub(crate) message: String,
}

impl JsonError {
    /// Creates a new JSON error with the specified message.
    pub(crate) fn new(message: String) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for JsonError {}

/// Internal error type for model-related errors with a descriptive message.
///
/// This struct wraps error messages from the underlying model and implements
/// the standard error traits for proper error handling and display.
#[derive(Debug)]
pub(crate) struct ModelErrorMessage(pub(crate) String);

impl std::fmt::Display for ModelErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ModelErrorMessage {}

/// Creates a channel for response communication.
pub fn create_response_channel(
    buffer_size: Option<usize>,
) -> (Sender<Response>, Receiver<Response>) {
    let channel_buffer_size = buffer_size.unwrap_or(DEFAULT_CHANNEL_BUFFER_SIZE);
    channel(channel_buffer_size)
}

/// Sends a request to the model processing pipeline.
pub async fn send_request(
    state: &SharedMistralRsState,
    request: Request,
) -> Result<(), MistralRsError> {
    send_request_with_model(state, request, None).await
}

pub async fn send_request_with_model(
    state: &SharedMistralRsState,
    mut request: Request,
    model_id: Option<&str>,
) -> Result<(), MistralRsError> {
    if let Some(model_id) = model_id {
        if let Request::Normal(request) = &mut request {
            request.model_id = Some(model_id.to_string());
        } else {
            return state
                .get_sender(Some(model_id))?
                .send(request)
                .await
                .map_err(|_| MistralRsError::SenderPoisoned);
        }
    }
    state.send_request_async(request).await
}

pub(crate) fn request_model_override(
    requested_model: String,
    routed_model: &str,
) -> Option<String> {
    (requested_model != routed_model).then_some(requested_model)
}

pub(crate) fn apply_model_override(model: &mut String, model_override: Option<&str>) {
    if let Some(model_override) = model_override {
        *model = model_override.to_string();
    }
}

/// Generic function to process non-streaming responses.
pub(crate) async fn base_process_non_streaming_response<R, M, E>(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
    match_fn: M,
    error_handler: E,
) -> R
where
    M: FnOnce(SharedMistralRsState, Response) -> R,
    E: FnOnce(SharedMistralRsState, Box<dyn std::error::Error + Send + Sync + 'static>) -> R,
{
    loop {
        match rx.recv().await {
            Some(Response::AgenticToolCallProgress { .. }) => continue,
            Some(Response::BlockDenoisingProgress(_)) => continue,
            Some(Response::AgenticToolApprovalRequired { .. }) => continue,
            Some(Response::File(_)) => continue,
            Some(response) => return match_fn(state, response),
            None => {
                let error = anyhow::Error::msg("No response received from the model.");
                return error_handler(state, error.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[test]
    fn request_model_override_only_preserves_routed_aliases() {
        assert_eq!(
            request_model_override("code".to_string(), "base"),
            Some("code".to_string())
        );
        assert_eq!(request_model_override("base".to_string(), "base"), None);

        let mut response_model = "base".to_string();
        apply_model_override(&mut response_model, Some("code"));
        assert_eq!(response_model, "code");
    }

    #[test]
    fn classifies_core_errors_without_losing_wrapped_sources() {
        let error = anyhow::Error::new(MistralRsError::ModelNotFound("missing".to_string()))
            .context("failed to dispatch request");
        let classified = ApiError::from_error(error.as_ref(), ApiErrorKind::Internal);
        assert_eq!(classified.kind, ApiErrorKind::NotFound);
        assert_eq!(classified.status(), StatusCode::NOT_FOUND);
        assert_eq!(classified.code.as_deref(), Some("model_not_found"));
        assert_eq!(classified.param.as_deref(), Some("model"));

        let classified = ApiError::from_error(
            &MistralRsError::ModelReloading("busy".to_string()),
            ApiErrorKind::Internal,
        );
        assert_eq!(classified.kind, ApiErrorKind::Conflict);
        assert_eq!(classified.status(), StatusCode::CONFLICT);

        let classified =
            ApiError::from_error(&MistralRsError::SenderPoisoned, ApiErrorKind::Internal);
        assert_eq!(classified.kind, ApiErrorKind::Unavailable);
        assert_eq!(classified.status(), StatusCode::SERVICE_UNAVAILABLE);

        let error = anyhow::Error::new(ServiceUnavailableError("private detail".to_string()))
            .context("allocation failed");
        let classified = ApiError::from_error(error.as_ref(), ApiErrorKind::Internal);
        assert_eq!(classified.kind, ApiErrorKind::Overloaded);
        assert_eq!(classified.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(classified.message, SERVICE_UNAVAILABLE_MESSAGE);

        let classified = ApiError::from_error(
            &MistralRsError::LoraAdapter(LoraAdapterError::SizeOverflow),
            ApiErrorKind::Internal,
        );
        assert_eq!(classified.kind, ApiErrorKind::InvalidRequest);
        assert_eq!(classified.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn does_not_expose_internal_core_errors() {
        let error = MistralRsError::Other("secret backend detail".to_string());
        let classified = ApiError::from_error(&error, ApiErrorKind::InvalidRequest);
        assert_eq!(classified.kind, ApiErrorKind::Internal);
        assert_eq!(classified.message, INTERNAL_ERROR_MESSAGE);
        assert!(!classified.message.contains("secret"));
    }

    #[tokio::test]
    async fn serializes_openai_error_envelope() {
        let response = openai_error_response(ApiError::new(
            ApiErrorKind::NotFound,
            "model `missing` was not found",
            Some("model_not_found"),
            Some("model"),
        ));
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
        assert_eq!(body["error"]["message"], "model `missing` was not found");
    }

    #[tokio::test]
    async fn replaces_server_error_details_with_stable_message() {
        let response = openai_error_response(ApiError::from_status(
            StatusCode::INTERNAL_SERVER_ERROR,
            "database password leaked",
        ));
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["message"], INTERNAL_ERROR_MESSAGE);
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "internal_error");
    }
}
