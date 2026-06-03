//! Admission gate for the parking-lot scheduler.
//!
//! When the `parking-lot-scheduler` feature is enabled, an [`InferenceWorkerPool`]
//! sits in front of the engine as a concurrency/admission gate. It does NOT
//! execute inference — token generation is owned by the engine's batched
//! scheduler loop. The gate bounds the number of concurrently in-flight
//! requests and applies backpressure (rejecting excess load once the wait queue
//! is saturated), while responses stream back on the caller's own channel.
//!
//! See `engine/add_request.rs` for the admission flow and `worker_pool` for the
//! gate itself. Runtime configuration is parsed from YAML/CLI into
//! [`ParkingLotSchedulerConfig`].

pub mod config;
pub mod worker_pool;

#[cfg(test)]
mod tests;

// Re-export config types
pub use config::{LimitsConfig, ParkingLotSchedulerConfig, PoolConfig};

// Re-export the admission gate
pub use worker_pool::{
    AdmissionError, AdmissionPermit, InferenceWorkerPool, InferenceWorkerPoolConfig,
};
