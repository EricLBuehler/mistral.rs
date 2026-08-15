//! SSE streaming utilities.

use std::{
    env,
    sync::{Arc, Mutex},
};

use mistralrs_core::{Response, Usage};
use tokio::sync::mpsc::Receiver;

use crate::{types::SharedMistralRsState, util::sanitize_error_message};

/// What a streaming request produced, filled in by the streamer and read once the SSE body ends.
#[derive(Debug, Clone, Default)]
pub struct StreamOutcome {
    pub usage: Option<Usage>,
    /// The error message sent to the client, if the request failed
    pub error: Option<String>,
}

/// Shared between the observability middleware and the streamer that serves the request.
#[derive(Debug, Clone, Default)]
pub struct StreamOutcomeHandle(Arc<Mutex<StreamOutcome>>);

impl StreamOutcomeHandle {
    pub fn snapshot(&self) -> StreamOutcome {
        self.0.lock().expect("stream outcome poisoned").clone()
    }

    /// Record usage or errors from any engine response as it flows through a streamer.
    pub fn observe(&self, response: &Response) {
        let mut outcome = self.0.lock().expect("stream outcome poisoned");
        match response {
            Response::Chunk(chunk) => {
                if let Some(usage) = &chunk.usage {
                    outcome.usage = Some(usage.clone());
                }
            }
            Response::Done(done) => outcome.usage = Some(done.usage.clone()),
            Response::CompletionDone(done) => outcome.usage = Some(done.usage.clone()),
            Response::ModelError(msg, _) | Response::CompletionModelError(msg, _) => {
                outcome.error = Some(msg.clone())
            }
            Response::InternalError(e) | Response::ValidationError(e) => {
                outcome.error = Some(sanitize_error_message(e.as_ref()))
            }
            _ => {}
        }
    }
}

/// Convenience for streamers holding an optional handle.
pub(crate) fn observe_response(handle: &Option<StreamOutcomeHandle>, response: &Response) {
    if let Some(handle) = handle {
        handle.observe(response);
    }
}

/// Default keep-alive interval for Server-Sent Events (SSE) streams in milliseconds.
pub const DEFAULT_KEEP_ALIVE_INTERVAL_MS: u64 = 10_000;

/// Represents the current state of a streaming response.
pub enum DoneState {
    /// The stream is actively processing and sending response chunks
    Running,
    /// The stream has finished processing and is about to send the `[DONE]` message
    SendingDone,
    /// The stream has completed entirely
    Done,
}

/// A streaming response handler.
///
/// It processes incoming response chunks from a model and converts them
/// into Server-Sent Events (SSE) format for real-time streaming to clients.
pub struct BaseStreamer<R, C, D> {
    /// Channel receiver for incoming model responses
    pub rx: Receiver<Response>,
    /// Current state of the streaming operation
    pub done_state: DoneState,
    /// Underlying mistral.rs instance
    pub state: SharedMistralRsState,
    /// Whether to store chunks for the completion callback
    pub store_chunks: bool,
    /// All chunks received during streaming (if `store_chunks` is true)
    pub chunks: Vec<R>,
    /// Optional callback to process each chunk before sending
    pub on_chunk: Option<C>,
    /// Optional callback to execute when streaming completes
    pub on_done: Option<D>,
    /// Where usage and errors are reported for the end-of-stream access log
    pub outcome: Option<StreamOutcomeHandle>,
}

/// Generic function to create a SSE streamer with optional callbacks.
pub(crate) fn base_create_streamer<R, C, D>(
    rx: Receiver<Response>,
    state: SharedMistralRsState,
    on_chunk: Option<C>,
    on_done: Option<D>,
    outcome: Option<StreamOutcomeHandle>,
) -> BaseStreamer<R, C, D> {
    let store_chunks = on_done.is_some();

    BaseStreamer {
        rx,
        done_state: DoneState::Running,
        store_chunks,
        state,
        chunks: Vec::new(),
        on_chunk,
        on_done,
        outcome,
    }
}

/// Gets the keep-alive interval for SSE streams from environment or default.
pub fn get_keep_alive_interval() -> u64 {
    env::var("KEEP_ALIVE_INTERVAL")
        .map(|val| {
            val.parse::<u64>().unwrap_or_else(|e| {
                tracing::warn!("Failed to parse KEEP_ALIVE_INTERVAL: {}. Using default.", e);
                DEFAULT_KEEP_ALIVE_INTERVAL_MS
            })
        })
        .unwrap_or(DEFAULT_KEEP_ALIVE_INTERVAL_MS)
}
