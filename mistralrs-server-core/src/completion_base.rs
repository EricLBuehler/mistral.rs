//! Base functionality for completions.

use std::error::Error;

use anyhow::Result;
use axum::response::Sse;
use mistralrs_core::{DrySamplingParams, MistralRs, StopTokens as InternalStopTokens};

use crate::{openai::StopTokens, types::SharedMistralRsState};

/// Generic responder enum for different completion types.
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
pub(crate) fn handle_completion_error<R, S>(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> BaseCompletionResponder<R, S> {
    let e = anyhow::Error::msg(e.to_string());
    MistralRs::maybe_log_error(state, &*e);
    BaseCompletionResponder::InternalError(e.into())
}

/// Helper function to convert from the OpenAI stop tokens to the mistral.rs
/// internal stop tokens.
pub(crate) fn convert_stop_tokens(stop_seqs: Option<StopTokens>) -> Option<InternalStopTokens> {
    match stop_seqs {
        Some(StopTokens::Multi(m)) => Some(InternalStopTokens::Seqs(m)),
        Some(StopTokens::Single(s)) => Some(InternalStopTokens::Seqs(vec![s])),
        None => None,
    }
}

/// Helper function to get the dry sampling params.
pub(crate) fn get_dry_sampling_params(
    dry_multiplier: Option<f32>,
    dry_sequence_breakers: Option<Vec<String>>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<usize>,
) -> Result<Option<DrySamplingParams>> {
    if let Some(dry_multiplier) = dry_multiplier {
        let dry_sampling_params = DrySamplingParams::new_with_defaults(
            dry_multiplier,
            dry_sequence_breakers,
            dry_base,
            dry_allowed_length,
        )?;

        Ok(Some(dry_sampling_params))
    } else {
        Ok(None)
    }
}
