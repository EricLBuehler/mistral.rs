//! ## Streaming utils and functionality.

use std::env;

use mistralrs_core::Response;
use tokio::sync::mpsc::{channel, Receiver, Sender};

/// Default buffer size for the response channel used in streaming operations.
///
/// This constant defines the maximum number of response messages that can be buffered
/// in the channel before backpressure is applied. A larger buffer reduces the likelihood
/// of blocking but uses more memory.
pub const DEFAULT_CHANNEL_BUFFER_SIZE: usize = 10_000;

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

/// Creates a channel for response communication.
pub fn create_response_channel(
    buffer_size: Option<usize>,
) -> (Sender<Response>, Receiver<Response>) {
    let channel_buffer_size = buffer_size.unwrap_or(DEFAULT_CHANNEL_BUFFER_SIZE);

    channel(channel_buffer_size)
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
