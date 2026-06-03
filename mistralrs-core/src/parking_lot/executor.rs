//! LLM inference executor placeholder for the parking-lot scheduler.
//!
//! # Why this is (intentionally) not an inference engine
//!
//! Earlier iterations tried to run inference *inside* the prometheus
//! `WorkerPool` executor. That cannot work correctly here: token generation in
//! mistral.rs is owned by [`crate::engine::Engine`]'s batched run loop, which
//! holds the single `Pipeline` (behind a mutex), the KV cache, prefix caching,
//! and paged-attention block management. There is exactly one pipeline, so N
//! pool worker threads could not parallelize generation regardless.
//!
//! The integration therefore uses the worker pool as an **admission gate**
//! (see [`crate::parking_lot::InferenceWorkerPool::admit`]) placed *in front of*
//! the engine: it bounds concurrency, accounts for in-flight work, and applies
//! backpressure, while the engine's existing scheduler performs the actual
//! inference and streams responses on the caller's own channel.
//!
//! Because inference no longer flows through this executor, its `execute`
//! methods are never invoked on the hot path. They are retained only so the
//! `LlmExecutor` type continues to satisfy the trait bounds required to
//! construct the underlying `prometheus_parking_lot::WorkerPool`, and they fail
//! loudly if ever called.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tracing::error;

use super::job::{InferenceJob, InferenceResult};
use super::types::{ParkingLotTaskMetadata, PrometheusWorkerExecutor, TaskExecutor, TaskMetadata};
use crate::pipeline::Pipeline;

/// Message returned when the executor is invoked, which should never happen in
/// the admission-gate model.
const UNREACHABLE_EXECUTOR: &str =
    "LlmExecutor::execute was called, but inference must run through the engine's scheduler. \
     The worker pool is an admission gate (see InferenceWorkerPool::admit); jobs are not \
     executed inside the pool.";

/// Placeholder executor required to construct the prometheus `WorkerPool`.
///
/// It holds a handle to the pipeline purely so the type composes with the
/// pool's generics; it does not drive inference (see the module docs).
#[derive(Clone)]
pub struct LlmExecutor {
    /// Pipeline handle (unused on the hot path; kept for type compatibility and
    /// potential future introspection).
    #[allow(dead_code)]
    pipeline: Arc<TokioMutex<dyn Pipeline + Send + Sync>>,
}

impl LlmExecutor {
    /// Create a new executor wrapping the given pipeline handle.
    #[must_use]
    pub fn new(pipeline: Arc<TokioMutex<dyn Pipeline + Send + Sync>>) -> Self {
        Self { pipeline }
    }
}

// Implement prometheus_parking_lot's WorkerExecutor trait. Not used on the hot
// path: inference is gated by `InferenceWorkerPool::admit` and executed by the
// engine, not submitted to the pool for execution.
#[async_trait]
impl PrometheusWorkerExecutor<InferenceJob, InferenceResult> for LlmExecutor {
    async fn execute(
        &self,
        payload: InferenceJob,
        meta: ParkingLotTaskMetadata,
    ) -> InferenceResult {
        error!(
            task_id = %meta.id,
            request_id = %payload.request_id,
            "{UNREACHABLE_EXECUTOR}"
        );
        InferenceResult::error(UNREACHABLE_EXECUTOR)
    }
}

// Local TaskExecutor trait — same contract, same (unused) behavior.
#[async_trait]
impl TaskExecutor<InferenceJob, InferenceResult> for LlmExecutor {
    async fn execute(&self, payload: InferenceJob, meta: TaskMetadata) -> InferenceResult {
        error!(
            task_id = %meta.id,
            request_id = %payload.request_id,
            "{UNREACHABLE_EXECUTOR}"
        );
        InferenceResult::error(UNREACHABLE_EXECUTOR)
    }
}
