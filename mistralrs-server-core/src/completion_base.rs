//! Base functionality for completions.

use std::error::Error;

use axum::response::Sse;
use mistralrs_core::MistralRs;

use crate::types::SharedMistralRsState;

/// Generic responder enum for different completion types
#[derive(Debug)]
pub enum BaseCompletionResponder<R, S> {
    /// Server-Sent Events streaming response
    Sse(Sse<S>),
    /// Complete JSON response for non-streaming requests
    Json(R),
    /// Model error with partial response data
    ModelError(String, R),
    /// Internal server error
    InternalError(Box<dyn Error>),
    /// Request validation error
    ValidationError(Box<dyn Error>),
}

/// Generic function to handle completion errors and logging them.
pub(crate) fn base_handle_completion_error<R, S>(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> BaseCompletionResponder<R, S> {
    let e = anyhow::Error::msg(e.to_string());
    MistralRs::maybe_log_error(state, &*e);
    BaseCompletionResponder::InternalError(e.into())
}
