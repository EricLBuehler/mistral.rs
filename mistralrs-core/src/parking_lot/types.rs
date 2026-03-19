//! Core types for the parking-lot scheduler.
//!
//! This module re-exports types from prometheus_parking_lot and provides
//! LLM-specific extensions for mistral.rs.

use async_trait::async_trait;

// Re-export core types from prometheus_parking_lot
pub use prometheus_parking_lot::core::{
    Mailbox, PoolLimits, ScheduledTask, Spawn, TaskMetadata as ParkingLotTaskMetadata, TaskQueue,
    TaskStatus, WakeState, WorkerExecutor as PrometheusWorkerExecutor,
};
pub use prometheus_parking_lot::infra::mailbox::memory::InMemoryMailbox;
pub use prometheus_parking_lot::infra::queue::memory::InMemoryQueue;
pub use prometheus_parking_lot::runtime::tokio_spawner::TokioSpawner;
pub use prometheus_parking_lot::util::clock::now_ms;
pub use prometheus_parking_lot::util::serde::{
    MailboxKey, Priority, ResourceCost, ResourceKind, TaskId,
};

/// Tenant identifier for multi-tenant scenarios.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TenantId(pub String);

impl From<String> for TenantId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for TenantId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// Task metadata for LLM inference.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskMetadata {
    /// Unique task identifier
    pub id: TaskId,
    /// Task priority
    pub priority: Priority,
    /// Resource cost of this task
    pub cost: ResourceCost,
    /// Creation timestamp in milliseconds
    pub created_at_ms: u128,
    /// Optional deadline in milliseconds
    pub deadline_ms: Option<u128>,
    /// Optional mailbox key for result delivery
    pub mailbox: Option<MailboxKey>,
}

impl TaskMetadata {
    /// Create new task metadata.
    #[must_use]
    pub fn new(id: impl Into<TaskId>, cost: ResourceCost) -> Self {
        Self {
            id: id.into(),
            priority: Priority::Normal,
            cost,
            created_at_ms: now_ms(),
            deadline_ms: None,
            mailbox: None,
        }
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the deadline.
    #[must_use]
    pub fn with_deadline_ms(mut self, deadline_ms: u128) -> Self {
        self.deadline_ms = Some(deadline_ms);
        self
    }

    /// Set the mailbox key.
    #[must_use]
    pub fn with_mailbox(mut self, key: MailboxKey) -> Self {
        self.mailbox = Some(key);
        self
    }
}

/// Convert to prometheus_parking_lot's TaskMetadata.
impl From<TaskMetadata> for ParkingLotTaskMetadata {
    fn from(meta: TaskMetadata) -> Self {
        ParkingLotTaskMetadata {
            id: meta.id,
            mailbox: meta.mailbox,
            priority: meta.priority,
            cost: meta.cost,
            deadline_ms: meta.deadline_ms,
            created_at_ms: meta.created_at_ms,
        }
    }
}

/// Task executor trait for LLM inference.
///
/// This is our own trait that doesn't require Serialize/Deserialize on the result,
/// allowing us to return channels for streaming responses.
#[async_trait]
pub trait TaskExecutor<P, T>: Send + Sync
where
    P: Send + 'static,
    T: Send + 'static,
{
    /// Execute a task payload and return the result.
    async fn execute(&self, payload: P, meta: TaskMetadata) -> T;
}

/// Extension methods for ResourceCost used in LLM contexts.
pub trait ResourceCostExt {
    /// Create a GPU VRAM cost.
    fn gpu_vram(units: u32) -> ResourceCost;

    /// Create a CPU slots cost.
    fn cpu_slots(units: u32) -> ResourceCost;
}

impl ResourceCostExt for ResourceCost {
    fn gpu_vram(units: u32) -> ResourceCost {
        ResourceCost {
            kind: ResourceKind::GpuVram,
            units,
        }
    }

    fn cpu_slots(units: u32) -> ResourceCost {
        ResourceCost {
            kind: ResourceKind::Cpu,
            units,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_id() {
        let tenant = TenantId::from("test-tenant");
        assert_eq!(tenant.0, "test-tenant");
    }

    #[test]
    fn test_resource_cost() {
        let cost = ResourceCost::gpu_vram(100);
        assert_eq!(cost.kind, ResourceKind::GpuVram);
        assert_eq!(cost.units, 100);
    }

    #[test]
    fn test_task_metadata() {
        let meta = TaskMetadata::new(42u64, ResourceCost::gpu_vram(10))
            .with_priority(Priority::High)
            .with_deadline_ms(999999);

        assert_eq!(meta.id, 42);
        assert_eq!(meta.priority, Priority::High);
        assert_eq!(meta.cost.units, 10);
        assert_eq!(meta.deadline_ms, Some(999999));
    }
}
