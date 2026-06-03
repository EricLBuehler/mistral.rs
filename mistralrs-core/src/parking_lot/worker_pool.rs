//! Admission gate for the parking-lot scheduler.
//!
//! [`InferenceWorkerPool`] is a concurrency/admission gate placed *in front of*
//! the engine — it does not execute inference. Token generation is owned by the
//! engine's single batched scheduler loop (KV cache, prefix caching, paged
//! attention, one `Pipeline` behind a mutex), so there is nothing for a pool of
//! worker threads to parallelize. Instead the gate bounds the number of
//! concurrent in-flight requests and applies backpressure, while the engine
//! performs the actual work and streams responses on the caller's own channel.
//!
//! See `engine/add_request.rs` for the admission flow.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::info;

use super::ParkingLotSchedulerConfig;

/// RAII admission permit held for the lifetime of one in-flight inference.
///
/// [`InferenceWorkerPool::admit`] hands out one of these to bound concurrency
/// and account for in-flight work. The permit is released — freeing a slot for
/// a queued request — when it is dropped, which the caller arranges to coincide
/// with the request reaching a terminal response.
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

/// Configuration for the inference admission gate.
#[derive(Debug, Clone)]
pub struct InferenceWorkerPoolConfig {
    /// Maximum number of concurrently executing requests.
    pub worker_count: usize,

    /// Maximum number of requests allowed to wait for a slot before new
    /// arrivals are rejected with [`AdmissionError::QueueFull`].
    pub max_queue_depth: usize,
}

impl Default for InferenceWorkerPoolConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get(),
            max_queue_depth: 1000,
        }
    }
}

impl InferenceWorkerPoolConfig {
    /// Create a new config with explicit values.
    #[must_use]
    pub fn new(worker_count: usize, max_queue_depth: usize) -> Self {
        Self {
            worker_count,
            max_queue_depth,
        }
    }

    /// Create from YAML scheduler configuration, applying defaults for any
    /// unspecified values.
    #[must_use]
    pub fn from_scheduler_config(config: ParkingLotSchedulerConfig) -> Self {
        Self {
            worker_count: config.pool.worker_threads.unwrap_or_else(num_cpus::get),
            max_queue_depth: config.limits.max_queue_depth.unwrap_or(1000),
        }
    }
}

/// Concurrency/admission gate for LLM inference.
///
/// Bounds concurrent in-flight requests to `worker_count` via a semaphore and
/// rejects arrivals once `max_queue_depth` requests are already waiting.
pub struct InferenceWorkerPool {
    /// Configuration.
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
    /// Create a new admission gate.
    #[must_use]
    pub fn new(config: InferenceWorkerPoolConfig) -> Self {
        info!(
            "🚦 Inference admission gate ready — max_concurrent={}, max_queue={}",
            config.worker_count, config.max_queue_depth
        );
        let admission = Arc::new(Semaphore::new(config.worker_count.max(1)));
        Self {
            config,
            admission,
            in_flight: Arc::new(AtomicUsize::new(0)),
            waiting: Arc::new(AtomicUsize::new(0)),
        }
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

    /// Maximum number of concurrently executing requests.
    #[must_use]
    pub fn max_concurrent(&self) -> usize {
        self.config.worker_count
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    /// Build an `AdmissionPermit` the same way `admit` does, against a bare
    /// semaphore, so we can unit-test the accounting without a real pipeline.
    async fn acquire(sem: &Arc<Semaphore>, in_flight: &Arc<AtomicUsize>) -> AdmissionPermit {
        let permit = Arc::clone(sem).acquire_owned().await.unwrap();
        in_flight.fetch_add(1, Ordering::AcqRel);
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
        assert_eq!(in_flight.load(Ordering::Acquire), 2);

        drop(p1);
        assert_eq!(in_flight.load(Ordering::Acquire), 1);
        drop(p2);
        assert_eq!(in_flight.load(Ordering::Acquire), 0);
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
        assert!(
            pending.is_err(),
            "acquire should block while the only slot is taken"
        );

        // Releasing the first frees the slot for the next waiter.
        drop(p1);
        let p2 = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire(&sem, &in_flight),
        )
        .await
        .expect("acquire should succeed after a slot frees");
        assert_eq!(in_flight.load(Ordering::Acquire), 1);
        drop(p2);
        assert_eq!(in_flight.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn admit_rejects_when_queue_full() {
        // worker_count=1, max_queue_depth=1: one executing, one waiting allowed,
        // the third arrival is rejected.
        let pool = InferenceWorkerPool::new(InferenceWorkerPoolConfig::new(1, 1));

        let p1 = pool.admit().await.expect("first admit fills the only slot");
        assert_eq!(pool.in_flight(), 1);

        // Second admit blocks (slot taken) and occupies the single queue slot.
        let pool_ref = &pool;
        let queued = async move { pool_ref.admit().await };
        tokio::pin!(queued);
        // Let the queued admit register as waiting.
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut queued)
                .await
                .is_err(),
            "second admit should block on the full slot"
        );
        assert_eq!(pool.waiting(), 1);

        // Third admit: queue is full, reject immediately.
        let rejected = pool.admit().await;
        assert!(matches!(rejected, Err(AdmissionError::QueueFull { .. })));

        // Free the slot; the queued admit now resolves.
        drop(p1);
        let p2 = tokio::time::timeout(std::time::Duration::from_millis(50), &mut queued)
            .await
            .expect("queued admit resolves after slot frees")
            .expect("admit succeeds");
        assert_eq!(pool.in_flight(), 1);
        drop(p2);
    }
}
