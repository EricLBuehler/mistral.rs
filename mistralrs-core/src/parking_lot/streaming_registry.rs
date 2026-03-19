//! Streaming channel registry for prometheus-parking-lot integration.
//!
//! This module provides a thread-safe registry for storing and retrieving
//! streaming token channels, allowing the WorkerPool to return serializable
//! mailbox keys while keeping actual channels in memory.

use super::StreamingTokenResult;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Entry in the streaming registry with metadata
struct StreamingEntry {
    /// The channel receiver for streaming tokens
    receiver: flume::Receiver<Result<StreamingTokenResult, String>>,
    /// When this entry was created
    created_at: Instant,
    /// Request ID for logging
    request_id: String,
}

/// Thread-safe registry for streaming token channels.
///
/// This registry allows the WorkerPool to return serializable mailbox keys
/// for streaming requests, while the actual channels are stored in memory.
#[derive(Clone)]
pub struct StreamingRegistry {
    /// Map from channel key to streaming receiver
    channels: Arc<RwLock<HashMap<String, StreamingEntry>>>,
    /// How long to keep entries before cleanup (default: 1 hour)
    retention_duration: Duration,
}

impl StreamingRegistry {
    /// Create a new streaming registry.
    #[must_use]
    pub fn new(retention_duration: Duration) -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
            retention_duration,
        }
    }

    /// Create with default 1-hour retention.
    #[must_use]
    pub fn with_default_retention() -> Self {
        Self::new(Duration::from_secs(3600))
    }

    /// Register a streaming channel and return the key.
    pub fn register(
        &self,
        channel_key: String,
        request_id: String,
        receiver: flume::Receiver<Result<StreamingTokenResult, String>>,
    ) {
        let entry = StreamingEntry {
            receiver,
            created_at: Instant::now(),
            request_id: request_id.clone(),
        };

        self.channels.write().insert(channel_key.clone(), entry);

        info!(
            "Registered streaming channel - request_id={}, key={}",
            request_id, channel_key
        );
    }

    /// Retrieve a streaming channel by key.
    pub fn retrieve(
        &self,
        channel_key: &str,
    ) -> Option<flume::Receiver<Result<StreamingTokenResult, String>>> {
        let channels = self.channels.read();
        if let Some(entry) = channels.get(channel_key) {
            debug!(
                "Retrieved streaming channel - request_id={}, key={}",
                entry.request_id, channel_key
            );
            Some(entry.receiver.clone())
        } else {
            warn!(
                "Streaming channel not found - key={}",
                channel_key
            );
            None
        }
    }

    /// Remove a streaming channel by key.
    pub fn remove(&self, channel_key: &str) -> bool {
        let removed = self.channels.write().remove(channel_key);
        if let Some(entry) = removed {
            info!(
                "Removed streaming channel - request_id={}, key={}",
                entry.request_id, channel_key
            );
            true
        } else {
            false
        }
    }

    /// Clean up expired entries.
    pub fn cleanup_expired(&self) -> usize {
        let mut channels = self.channels.write();
        let now = Instant::now();
        let before_count = channels.len();

        channels.retain(|key, entry| {
            let age = now.duration_since(entry.created_at);
            if age > self.retention_duration {
                warn!(
                    "Cleaning up expired streaming channel - request_id={}, key={}, age={}s",
                    entry.request_id,
                    key,
                    age.as_secs()
                );
                false
            } else {
                true
            }
        });

        let removed = before_count - channels.len();
        if removed > 0 {
            info!(
                "Cleanup complete - removed {} expired channels",
                removed
            );
        }
        removed
    }

    /// Get the current number of registered channels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.read().len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.read().is_empty()
    }

    /// Start a background cleanup task that runs periodically.
    pub fn start_cleanup_task(self, cleanup_interval: Duration) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                self.cleanup_expired();
            }
        })
    }
}

impl Default for StreamingRegistry {
    fn default() -> Self {
        Self::with_default_retention()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_retrieve() {
        let registry = StreamingRegistry::with_default_retention();
        let (_tx, rx) = flume::unbounded();
        let key = "test-key".to_string();
        let request_id = "test-request".to_string();

        registry.register(key.clone(), request_id, rx);
        assert_eq!(registry.len(), 1);

        let retrieved = registry.retrieve(&key);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_remove() {
        let registry = StreamingRegistry::with_default_retention();
        let (_tx, rx) = flume::unbounded();
        let key = "test-key".to_string();
        let request_id = "test-request".to_string();

        registry.register(key.clone(), request_id, rx);
        assert_eq!(registry.len(), 1);

        let removed = registry.remove(&key);
        assert!(removed);
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_cleanup_expired() {
        let registry = StreamingRegistry::new(Duration::from_millis(100));
        let (_tx, rx) = flume::unbounded();
        let key = "test-key".to_string();
        let request_id = "test-request".to_string();

        registry.register(key, request_id, rx);
        assert_eq!(registry.len(), 1);

        // Should not clean up immediately
        let removed = registry.cleanup_expired();
        assert_eq!(removed, 0);
        assert_eq!(registry.len(), 1);

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should clean up now
        let removed = registry.cleanup_expired();
        assert_eq!(removed, 1);
        assert_eq!(registry.len(), 0);
    }
}
