//! Core functionality for handlers.

use anyhow::{Context, Result};
use axum::{extract::Json, http::StatusCode, response::IntoResponse};
use mistralrs_core::{Request, Response};
use serde::Serialize;
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::types::SharedMistralRsState;

/// Default buffer size for the response channel used in streaming operations.
///
/// This constant defines the maximum number of response messages that can be buffered
/// in the channel before backpressure is applied. A larger buffer reduces the likelihood
/// of blocking but uses more memory.
pub const DEFAULT_CHANNEL_BUFFER_SIZE: usize = 10_000;

/// Trait for converting errors to HTTP responses with appropriate status codes.
pub(crate) trait ErrorToResponse: Serialize {
    /// Converts the error to an HTTP response with the specified status code.
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut response = Json(self).into_response();
        *response.status_mut() = code;
        response
    }
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
impl ErrorToResponse for JsonError {}

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

/// Generic JSON error response structure
#[derive(Serialize, Debug)]
pub(crate) struct BaseJsonModelError<T> {
    pub(crate) message: String,
    pub(crate) partial_response: T,
}

impl<T> BaseJsonModelError<T> {
    pub(crate) fn new(message: String, partial_response: T) -> Self {
        Self {
            message,
            partial_response,
        }
    }
}

/// Creates a channel for response communication.
pub fn create_response_channel(
    buffer_size: Option<usize>,
) -> (Sender<Response>, Receiver<Response>) {
    let channel_buffer_size = buffer_size.unwrap_or(DEFAULT_CHANNEL_BUFFER_SIZE);
    channel(channel_buffer_size)
}

/// Sends a request to the model processing pipeline.
pub async fn send_request(state: &SharedMistralRsState, request: Request) -> Result<()> {
    send_request_with_model(state, request, None).await
}

pub async fn send_request_with_model(
    state: &SharedMistralRsState,
    request: Request,
    model_id: Option<&str>,
) -> Result<()> {
    let sender = state
        .get_sender(model_id)
        .context("mistral.rs sender not available.")?;

    sender
        .send(request)
        .await
        .context("Failed to send request to model pipeline")
}

/// Generic function to process non-streaming responses.
pub(crate) async fn base_process_non_streaming_response<R>(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
    match_fn: fn(SharedMistralRsState, Response) -> R,
    error_handler: fn(
        SharedMistralRsState,
        Box<dyn std::error::Error + Send + Sync + 'static>,
    ) -> R,
) -> R {
    match rx.recv().await {
        Some(response) => match_fn(state, response),
        None => {
            let error = anyhow::Error::msg("No response received from the model.");
            error_handler(state, error.into())
        }
    }
}
