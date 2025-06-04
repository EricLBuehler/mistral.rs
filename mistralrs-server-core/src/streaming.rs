//! ## Streaming utils and functionality.

use std::env;

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
