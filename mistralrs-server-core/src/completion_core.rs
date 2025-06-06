//! Core functionality for completions.

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
    let error = anyhow::Error::msg(e.to_string());
    MistralRs::maybe_log_error(state, &*error);
    BaseCompletionResponder::InternalError(error.into())
}

/// Helper function to convert from the OpenAI stop tokens to the mistral.rs
/// internal stop tokens.
pub(crate) fn convert_stop_tokens(stop_seqs: Option<StopTokens>) -> Option<InternalStopTokens> {
    match stop_seqs {
        Some(StopTokens::Multi(sequences)) => Some(InternalStopTokens::Seqs(sequences)),
        Some(StopTokens::Single(sequence)) => Some(InternalStopTokens::Seqs(vec![sequence])),
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
    match dry_multiplier {
        Some(multiplier) => {
            let params = DrySamplingParams::new_with_defaults(
                multiplier,
                dry_sequence_breakers,
                dry_base,
                dry_allowed_length,
            )?;
            Ok(Some(params))
        }
        None => Ok(None),
    }
}
