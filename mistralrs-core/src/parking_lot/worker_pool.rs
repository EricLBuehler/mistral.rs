//! Worker pool wrapper for prometheus-parking-lot.
//!
//! This module provides `InferenceWorkerPool`, which wraps prometheus-parking-lot's
//! `WorkerPool` and integrates it with mistral.rs inference pipeline.

use super::{
    InferenceJob, InferenceResult, LlmExecutor, ParkingLotSchedulerConfig,
    SerializableInferenceResult, StreamingRegistry, TaskMetadata,
};
use prometheus_parking_lot::config::WorkerPoolConfig as PrometheusWorkerPoolConfig;
use prometheus_parking_lot::core::{PoolError, WorkerPool};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{error, info};

/// RAII admission permit held for the lifetime of one in-flight inference.
///
/// In the "admission gate" model, the pool does not execute inference itself
/// (there is a single pipeline behind a mutex driven by the engine's batched
/// run loop). Instead, [`InferenceWorkerPool::admit`] hands out one of these
/// permits to bound concurrency and account for in-flight work. The permit is
/// released — freeing a slot for a queued request — when it is dropped, which
/// the caller arranges to coincide with the request reaching a terminal
/// response.
#[derive(Debug)]
pub struct AdmissionPermit {
    _permit: OwnedSemaphorePermit,
    in_flight: Arc<AtomicUsize>,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::Release);
    }
}

/// Decrements an [`AtomicUsize`] when dropped.
///
/// Used to keep the `waiting` counter accurate even if the task awaiting a
/// permit is cancelled (e.g. an HTTP request times out while queued). Without
/// this, a cancellation between the `fetch_add` and the permit acquisition
/// would leak the increment and eventually wedge the admission gate shut.
struct WaitingGuard(Arc<AtomicUsize>);

impl Drop for WaitingGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

/// Error returned when admission is refused.
#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
    /// The wait queue is already at `max_queue_depth`.
    #[error("worker pool queue is full ({waiting} waiting, capacity {capacity})")]
    QueueFull { waiting: usize, capacity: usize },
    /// The pool was shut down while waiting for a slot.
    #[error("worker pool is shutting down")]
    ShuttingDown,
}

/// Configuration for the inference worker pool.
#[derive(Debug, Clone)]
pub struct InferenceWorkerPoolConfig {
    /// Number of dedicated worker threads (default: num_cpus)
    pub worker_count: usize,

    /// Stack size per worker thread in bytes (default: 2MB, native only)
    pub thread_stack_size: Option<usize>,

    /// Maximum resource units (GPU VRAM in MB or KV cache blocks)
    pub max_units: u32,

    /// Maximum queue depth before rejection
    pub max_queue_depth: usize,

    /// Default timeout for job execution in seconds
    pub timeout_secs: u64,
}

impl Default for InferenceWorkerPoolConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get(),
            thread_stack_size: None, // Use prometheus_parking_lot default (2MB)
            max_units: 16384,        // ~256K tokens with 16-token blocks
            max_queue_depth: 1000,
            timeout_secs: 120,
        }
    }
}

impl InferenceWorkerPoolConfig {
    /// Create a new config with explicit values.
    #[must_use]
    pub fn new(worker_count: usize, max_units: u32, max_queue_depth: usize) -> Self {
        Self {
            worker_count,
            thread_stack_size: None,
            max_units,
            max_queue_depth,
            timeout_secs: 120,
        }
    }

    /// Set the timeout in seconds.
    #[must_use]
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Create from YAML scheduler configuration with defaults.
    ///
    /// This method creates an `InferenceWorkerPoolConfig` from a
    /// `ParkingLotSchedulerConfig` loaded from YAML, applying defaults
    /// for any unspecified values.
    ///
    /// # Arguments
    ///
    /// * `config` - The scheduler configuration from YAML
    ///
    /// # Returns
    ///
    /// A fully populated `InferenceWorkerPoolConfig` ready to use.
    #[must_use]
    pub fn from_scheduler_config(config: ParkingLotSchedulerConfig) -> Self {
        Self {
            worker_count: config.pool.worker_threads.unwrap_or_else(num_cpus::get),
            thread_stack_size: config.pool.thread_stack_size,
            max_units: config.limits.max_units.unwrap_or(16384),
            max_queue_depth: config.limits.max_queue_depth.unwrap_or(1000),
            timeout_secs: config.limits.timeout_secs.unwrap_or(120),
        }
    }
}

/// Convert to prometheus-parking-lot's WorkerPoolConfig.
impl From<InferenceWorkerPoolConfig> for PrometheusWorkerPoolConfig {
    fn from(config: InferenceWorkerPoolConfig) -> Self {
        let mut pool_config = PrometheusWorkerPoolConfig::new()
            .with_worker_count(config.worker_count)
            .with_max_units(config.max_units)
            .with_max_queue_depth(config.max_queue_depth)
            .with_timeout_ms(config.timeout_secs * 1000); // Convert seconds to milliseconds

        // Set thread stack size if provided (native only, ignored on WASM)
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(stack_size) = config.thread_stack_size {
            pool_config = pool_config.with_thread_stack_size(stack_size);
        }

        pool_config
    }
}

/// Pool statistics for monitoring.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of worker threads
    pub worker_threads: usize,
    
    /// Currently executing tasks
    pub active_tasks: usize,
    
    /// Tasks waiting in queue
    pub queued_tasks: usize,
    
    /// Used resource units (VRAM MB or KV cache blocks)
    pub used_units: usize,
    
    /// Total resource units
    pub total_units: usize,
    
    /// Tasks completed successfully
    pub completed_tasks: u64,
    
    /// Tasks that failed
    pub failed_tasks: u64,
}

impl PoolStats {
    /// Calculate resource utilization as a percentage (0-100).
    #[must_use]
    pub fn utilization_percent(&self) -> f64 {
        if self.total_units == 0 {
            0.0
        } else {
            (self.used_units as f64 / self.total_units as f64) * 100.0
        }
    }

    /// Check if pool is at or near capacity.
    #[must_use]
    pub fn is_near_capacity(&self) -> bool {
        self.utilization_percent() > 80.0
    }
}

/// Worker pool for LLM inference using prometheus-parking-lot.
///
/// This pool provides:
/// - Dedicated OS threads for CPU/GPU-bound inference work
/// - Resource tracking (GPU VRAM / KV cache blocks)
/// - Priority-based queue management
/// - Graceful degradation under load
/// - Integration with StreamingRegistry for non-serializable results
pub struct InferenceWorkerPool {
    /// The underlying prometheus-parking-lot WorkerPool
    pool: Arc<WorkerPool<InferenceJob, InferenceResult, LlmExecutor>>,

    /// Streaming channel registry for non-serializable results
    streaming_registry: Arc<StreamingRegistry>,

    /// Configuration
    config: InferenceWorkerPoolConfig,

    /// Concurrency gate: at most `worker_count` requests run at once. Excess
    /// requests wait here, bounded by `max_queue_depth`.
    admission: Arc<Semaphore>,

    /// Number of requests currently holding a permit (executing).
    in_flight: Arc<AtomicUsize>,

    /// Number of requests currently waiting for a permit.
    waiting: Arc<AtomicUsize>,
}

impl InferenceWorkerPool {
    /// Create a new inference worker pool.
    ///
    /// # Arguments
    ///
    /// * `executor` - The LLM executor that processes inference jobs
    /// * `streaming_registry` - Registry for managing streaming token channels
    /// * `config` - Pool configuration
    ///
    /// # Errors
    ///
    /// Returns error if pool creation fails (e.g., unable to spawn worker threads).
    pub fn new(
        executor: LlmExecutor,
        streaming_registry: StreamingRegistry,
        config: InferenceWorkerPoolConfig,
    ) -> Result<Self, PoolError> {
        info!(
            "🏗️ WORKER_POOL: Creating inference worker pool - workers={}, max_units={}, max_queue={}",
            config.worker_count, config.max_units, config.max_queue_depth
        );

        // The prometheus WorkerPool is retained only for its type/stats surface;
        // it never executes inference in the admission-gate model (see
        // `executor` docs). Force it to a single backing thread so we don't
        // spawn `num_cpus` idle OS threads. Real concurrency is governed by the
        // `admission` semaphore below, which uses the configured `worker_count`.
        let mut gated_config = config.clone();
        gated_config.worker_count = 1;
        let pool_config: PrometheusWorkerPoolConfig = gated_config.into();
        let pool = WorkerPool::new(pool_config, executor)?;

        info!(
            "✅ WORKER_POOL: Worker pool created successfully - {} dedicated worker threads",
            config.worker_count
        );

        // The admission gate caps concurrent in-flight inferences at
        // `worker_count`. Inference itself runs on the engine's batched run
        // loop; this gate provides backpressure and resource accounting in
        // front of it (see `admit`).
        let admission = Arc::new(Semaphore::new(config.worker_count.max(1)));

        Ok(Self {
            pool: Arc::new(pool),
            streaming_registry: Arc::new(streaming_registry),
            config,
            admission,
            in_flight: Arc::new(AtomicUsize::new(0)),
            waiting: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Acquire an admission permit, applying backpressure.
    ///
    /// At most `worker_count` requests may hold a permit simultaneously. When
    /// all slots are taken, the call waits until one frees up — unless the
    /// number of already-waiting requests has reached `max_queue_depth`, in
    /// which case [`AdmissionError::QueueFull`] is returned immediately so the
    /// caller can shed load.
    ///
    /// The returned [`AdmissionPermit`] must be kept alive for the duration of
    /// the request; dropping it releases the slot for a queued request.
    ///
    /// # Errors
    ///
    /// Returns [`AdmissionError::QueueFull`] if the wait queue is saturated, or
    /// [`AdmissionError::ShuttingDown`] if the pool's semaphore was closed.
    pub async fn admit(&self) -> Result<AdmissionPermit, AdmissionError> {
        // Atomically reserve a queue slot: check the cap and increment in one
        // CAS so concurrent callers cannot all pass the guard at once and
        // overshoot `max_queue_depth`.
        let mut current = self.waiting.load(Ordering::Acquire);
        loop {
            if current >= self.config.max_queue_depth {
                return Err(AdmissionError::QueueFull {
                    waiting: current,
                    capacity: self.config.max_queue_depth,
                });
            }
            match self.waiting.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }

        // `_waiting` decrements the reservation on drop — including if this task
        // is cancelled while awaiting the permit — so the counter never leaks.
        let _waiting = WaitingGuard(Arc::clone(&self.waiting));
        let permit = Arc::clone(&self.admission)
            .acquire_owned()
            .await
            .map_err(|_| AdmissionError::ShuttingDown)?;

        self.in_flight.fetch_add(1, Ordering::AcqRel);
        Ok(AdmissionPermit {
            _permit: permit,
            in_flight: Arc::clone(&self.in_flight),
        })
    }

    /// Number of requests currently executing (holding a permit).
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Number of requests currently waiting for a permit.
    #[must_use]
    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }

    /// Submit an inference job to the pool.
    ///
    /// This method submits a job and waits for the result. For streaming jobs,
    /// the result will contain a channel key that can be used to retrieve the
    /// streaming token channel from the registry.
    ///
    /// # Arguments
    ///
    /// * `job` - The inference job to execute
    /// * `meta` - Task metadata (priority, cost, etc.)
    ///
    /// # Returns
    ///
    /// Returns a `SerializableInferenceResult` which is either:
    /// - `Completion` - Contains the full generated text and usage info
    /// - `StreamingChannel` - Contains a channel key for retrieving tokens
    /// - `Error` - Contains error message if inference failed
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Queue is full (queue at capacity)
    /// - Job times out (exceeds configured timeout)
    /// - Pool is shutting down
    pub async fn submit(
        &self,
        job: InferenceJob,
        meta: TaskMetadata,
    ) -> Result<SerializableInferenceResult, PoolError> {
        let request_id = job.request_id.clone();
        let is_streaming = job.is_streaming;

        info!(
            "📥 WORKER_POOL: Submitting job - request_id={}, streaming={}, cost={}",
            request_id, is_streaming, meta.cost.units
        );

        // Submit job to pool and get result key
        let key = self.pool.submit_async(job, meta.into()).await?;

        info!(
            "🎫 WORKER_POOL: Job queued - request_id={}, key={:?}",
            request_id, key
        );

        // Retrieve result with timeout
        let timeout = Duration::from_secs(self.config.timeout_secs);
        let result = self.pool.retrieve_async(&key, timeout).await?;

        info!(
            "✅ WORKER_POOL: Job result retrieved - request_id={}, type={:?}",
            request_id,
            match &result {
                InferenceResult::ChatCompletion { .. } => "ChatCompletion",
                InferenceResult::Completion { .. } => "Completion",
                InferenceResult::Streaming { .. } => "Streaming",
                InferenceResult::Error { .. } => "Error",
            }
        );

        // Convert to serializable form
        let serializable = match result {
            InferenceResult::ChatCompletion { response } => {
                info!(
                    "📊 WORKER_POOL: ChatCompletion result - request_id={}, tokens={}",
                    request_id, response.usage.total_tokens
                );
                SerializableInferenceResult::chat_completion(response)
            }
            InferenceResult::Completion { response } => {
                info!(
                    "📊 WORKER_POOL: Completion result - request_id={}",
                    request_id
                );
                SerializableInferenceResult::completion(response)
            }
            InferenceResult::Streaming {
                request_id: req_id,
                token_rx,
            } => {
                let channel_key = uuid::Uuid::new_v4().to_string();
                info!(
                    "📡 WORKER_POOL: Streaming result, registering channel - request_id={}, key={}",
                    request_id, channel_key
                );
                self.streaming_registry
                    .register(channel_key.clone(), req_id.clone(), token_rx);
                SerializableInferenceResult::streaming_channel(req_id, channel_key)
            }
            InferenceResult::Error { message } => {
                error!(
                    "❌ WORKER_POOL: Inference error - request_id={}, error={}",
                    request_id, message
                );
                SerializableInferenceResult::error(message)
            }
        };

        Ok(serializable)
    }

    /// Get pool statistics.
    ///
    /// Returns current pool state including active tasks, queue depth,
    /// resource usage, etc.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let prometheus_stats = self.pool.stats();

        PoolStats {
            worker_threads: prometheus_stats.worker_count,
            active_tasks: prometheus_stats.active_tasks as usize,
            queued_tasks: prometheus_stats.queued_tasks as usize,
            used_units: prometheus_stats.used_units as usize,
            total_units: self.config.max_units as usize,
            completed_tasks: prometheus_stats.completed_tasks,
            failed_tasks: prometheus_stats.failed_tasks,
        }
    }

    /// Get number of available execution slots.
    #[must_use]
    pub fn available_permits(&self) -> usize {
        let stats = self.pool.stats();
        self.config
            .worker_count
            .saturating_sub(stats.active_tasks as usize)
    }

    /// Get current queue depth.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.pool.stats().queued_tasks as usize
    }

    /// Get the streaming registry.
    #[must_use]
    pub fn streaming_registry(&self) -> &Arc<StreamingRegistry> {
        &self.streaming_registry
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    /// Build an `AdmissionPermit` the same way `admit` does, against a bare
    /// semaphore, so we can unit-test the accounting without a real pipeline.
    async fn acquire(sem: &Arc<Semaphore>, in_flight: &Arc<AtomicUsize>) -> AdmissionPermit {
        let permit = Arc::clone(sem).acquire_owned().await.unwrap();
        in_flight.fetch_add(1, Ordering::SeqCst);
        AdmissionPermit {
            _permit: permit,
            in_flight: Arc::clone(in_flight),
        }
    }

    #[tokio::test]
    async fn permit_drop_decrements_in_flight() {
        let sem = Arc::new(Semaphore::new(2));
        let in_flight = Arc::new(AtomicUsize::new(0));

        let p1 = acquire(&sem, &in_flight).await;
        let p2 = acquire(&sem, &in_flight).await;
        assert_eq!(in_flight.load(Ordering::SeqCst), 2);

        drop(p1);
        assert_eq!(in_flight.load(Ordering::SeqCst), 1);
        drop(p2);
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn waiting_guard_decrements_on_drop_even_under_cancellation() {
        // Models the `admit` reservation: increment `waiting`, then a guard that
        // decrements on drop. Dropping the guard without reaching the "success"
        // decrement (the cancellation case) must still restore the counter.
        let waiting = Arc::new(AtomicUsize::new(0));

        waiting.fetch_add(1, Ordering::AcqRel);
        {
            let _guard = WaitingGuard(Arc::clone(&waiting));
            assert_eq!(waiting.load(Ordering::Acquire), 1);
            // Simulate cancellation: guard dropped here without any explicit
            // success path running.
        }
        assert_eq!(
            waiting.load(Ordering::Acquire),
            0,
            "WaitingGuard must restore the counter on drop"
        );
    }

    #[tokio::test]
    async fn semaphore_caps_concurrency_and_drop_frees_a_slot() {
        // One slot: a second acquire must wait until the first is dropped.
        let sem = Arc::new(Semaphore::new(1));
        let in_flight = Arc::new(AtomicUsize::new(0));

        let p1 = acquire(&sem, &in_flight).await;
        assert_eq!(sem.available_permits(), 0);

        // Second acquire cannot proceed while p1 is held.
        let pending = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire(&sem, &in_flight),
        )
        .await;
        assert!(pending.is_err(), "acquire should block while the only slot is taken");

        // Releasing the first frees the slot for the next waiter.
        drop(p1);
        let p2 = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire(&sem, &in_flight),
        )
        .await
        .expect("acquire should succeed after a slot frees");
        assert_eq!(in_flight.load(Ordering::SeqCst), 1);
        drop(p2);
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);
    }
}
