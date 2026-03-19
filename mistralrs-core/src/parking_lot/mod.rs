//! Prometheus parking-lot scheduler integration for mistral.rs.
//!
//! This module provides integration with the prometheus_parking_lot crate
//! for resource-constrained scheduling of LLM inference requests.
//!
//! # Architecture
//!
//! The parking-lot scheduler manages GPU resources (primarily KV-cache blocks)
//! and queues excess work when capacity is exhausted:
//!
//! - `InferenceJob`: The task payload describing an inference request
//! - `InferenceResult`: The result type (completion or streaming tokens)
//! - `LlmExecutor`: The task executor that processes inference jobs
//! - `ResourceAdapter`: Maps KV-cache blocks to generic resource units
//! - `WorkerPool`: Manages dedicated worker threads for CPU/GPU-bound work
//!
//! # Design Notes
//!
//! We use prometheus_parking_lot primitives (TaskId, Priority, ResourceCost, etc.)
//! for type compatibility and resource tracking. The executor integrates with
//! mistral.rs's existing Pipeline trait.

pub mod config;
pub mod executor;
pub mod job;
pub mod resource_adapter;
pub mod streaming_registry;
pub mod types;
pub mod worker_pool;

#[cfg(test)]
mod tests;

// Re-export config types
pub use config::{LimitsConfig, ParkingLotSchedulerConfig, PoolConfig};

// Re-export executor
pub use executor::LlmExecutor;

// Re-export job types
pub use job::{
    InferenceJob, InferenceResult, SerializableInferenceResult, StreamingTokenResult,
};

// Re-export streaming registry
pub use streaming_registry::StreamingRegistry;

// Re-export worker pool
pub use worker_pool::{InferenceWorkerPool, InferenceWorkerPoolConfig, PoolStats};

// Re-export resource adapter
pub use resource_adapter::{calculate_resource_cost, ResourceAdapter, DEFAULT_BLOCK_SIZE};

// Re-export types
pub use types::{
    now_ms,
    // prometheus_parking_lot primitives
    InMemoryMailbox,
    InMemoryQueue,
    Mailbox,
    MailboxKey,
    PoolLimits,
    Priority,
    ResourceCost,
    ResourceCostExt,
    ResourceKind,
    ScheduledTask,
    Spawn,
    // Local types
    TaskExecutor,
    TaskId,
    TaskMetadata,
    TaskQueue,
    TaskStatus,
    TenantId,
    TokioSpawner,
    WakeState,
};
