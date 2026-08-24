use std::time::{Duration, Instant};

use candle_core::Device;

use crate::{
    utils::{debug::DeviceRepr, memory_usage::CudaAllocatorSnapshot},
    MemoryUsage,
};

const BYTES_PER_MIB: usize = 1024 * 1024;
const BASE_FREE_DIVISOR: usize = 128;
const BASE_FREE_MIN_BYTES: usize = 512 * BYTES_PER_MIB;
const BASE_FREE_MAX_BYTES: usize = 2 * 1024 * BYTES_PER_MIB;
const WARM_CACHE_DIVISOR: usize = 128;
const WARM_CACHE_MIN_BYTES: usize = 256 * BYTES_PER_MIB;
const WARM_CACHE_MAX_BYTES: usize = 1024 * BYTES_PER_MIB;
const IDLE_RECLAIM_MULTIPLIER: usize = 2;
const TRIM_COOLDOWN: Duration = Duration::from_secs(1);
pub(super) const GRAPH_RECLAIM_BATCH_SIZE: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PromptBatchMemoryAction {
    Proceed,
    Retain(usize),
    Reject,
}

pub(super) fn prompt_batch_memory_action(
    current: usize,
    transient_pressure: bool,
) -> PromptBatchMemoryAction {
    if !transient_pressure {
        PromptBatchMemoryAction::Proceed
    } else if current > 1 {
        PromptBatchMemoryAction::Retain(current.div_ceil(2))
    } else {
        PromptBatchMemoryAction::Reject
    }
}

pub(super) fn record_prompt_batch_reduction(previous: usize, retained: usize) {
    metrics::counter!("mistralrs_cuda_prompt_batch_reductions_total").increment(1);
    metrics::counter!("mistralrs_cuda_prompt_sequences_deferred_total")
        .increment(u64::try_from(previous.saturating_sub(retained)).unwrap_or(u64::MAX));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaintenancePoint {
    Idle,
    PromptPreflight,
    PromptBoundary,
    GraphReclaimed,
}

impl MaintenancePoint {
    fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::PromptPreflight => "prompt_preflight",
            Self::PromptBoundary => "prompt_boundary",
            Self::GraphReclaimed => "graph_reclaimed",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MaintenanceOutcome {
    graph_pressure: bool,
    transient_pressure: bool,
    insufficient_total_capacity: bool,
    maintenance_failed: bool,
    capture_active: bool,
    reclaim_deferred: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PromptMemoryStatus {
    pub(super) graph_pressure: bool,
    pub(super) transient_pressure: bool,
    pub(super) insufficient_total_capacity: bool,
    pub(super) maintenance_failed: bool,
}

struct MaintainedDevice {
    device: Device,
    last_trim: Option<Instant>,
    capture_deferred: bool,
}

pub(super) struct CudaMemoryPoolMaintenance {
    devices: Vec<MaintainedDevice>,
}

impl CudaMemoryPoolMaintenance {
    pub(super) fn new(devices: Vec<Device>) -> Self {
        let devices = devices
            .into_iter()
            .map(|device| {
                record_pending(&device, false);
                MaintainedDevice {
                    device,
                    last_trim: None,
                    capture_deferred: false,
                }
            })
            .collect();
        Self { devices }
    }

    pub(super) fn after_prompt_step(&mut self) -> bool {
        self.maintain(MaintenancePoint::PromptBoundary, 0)
            .graph_pressure
    }

    pub(super) fn before_prompt_step(&mut self, transient_bytes: usize) -> PromptMemoryStatus {
        self.maintain(MaintenancePoint::PromptPreflight, transient_bytes)
            .into()
    }

    pub(super) fn after_graph_reclaim(&mut self, transient_bytes: usize) -> PromptMemoryStatus {
        self.maintain(MaintenancePoint::GraphReclaimed, transient_bytes)
            .into()
    }

    pub(super) fn when_idle(&mut self) -> bool {
        self.maintain(MaintenancePoint::Idle, 0).graph_pressure
    }

    fn maintain(&mut self, point: MaintenancePoint, transient_bytes: usize) -> MaintenanceOutcome {
        let outcomes = self.devices.iter_mut().map(|maintained| {
            match maintain_device(maintained, point, transient_bytes) {
                Ok(outcome) => outcome,
                Err(err) => {
                    let device = maintained.device.device_pretty_repr();
                    metrics::counter!(
                        "mistralrs_cuda_memory_maintenance_total",
                        "device" => device.clone(),
                        "reason" => point.label(),
                        "action" => "maintain",
                        "outcome" => "error"
                    )
                    .increment(1);
                    tracing::warn!("CUDA memory maintenance failed on {device}: {err}");
                    maintenance_failure_outcome()
                }
            }
        });
        aggregate_outcomes(outcomes)
    }
}

fn aggregate_outcomes(
    outcomes: impl IntoIterator<Item = MaintenanceOutcome>,
) -> MaintenanceOutcome {
    let mut aggregate = MaintenanceOutcome::default();
    for outcome in outcomes {
        aggregate.graph_pressure |= outcome.graph_pressure;
        aggregate.transient_pressure |= outcome.transient_pressure;
        aggregate.insufficient_total_capacity |= outcome.insufficient_total_capacity;
        aggregate.maintenance_failed |= outcome.maintenance_failed;
        aggregate.capture_active |= outcome.capture_active;
        aggregate.reclaim_deferred |= outcome.reclaim_deferred;
    }
    aggregate.reclaim_deferred |= aggregate.capture_active;
    if aggregate.reclaim_deferred {
        aggregate.graph_pressure = false;
    }
    aggregate
}

fn maintenance_failure_outcome() -> MaintenanceOutcome {
    MaintenanceOutcome {
        transient_pressure: true,
        maintenance_failed: true,
        reclaim_deferred: true,
        ..MaintenanceOutcome::default()
    }
}

impl From<MaintenanceOutcome> for PromptMemoryStatus {
    fn from(value: MaintenanceOutcome) -> Self {
        Self {
            graph_pressure: value.graph_pressure,
            transient_pressure: value.transient_pressure,
            insufficient_total_capacity: value.insufficient_total_capacity,
            maintenance_failed: value.maintenance_failed,
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn maintain_device(
    maintained: &mut MaintainedDevice,
    point: MaintenancePoint,
    transient_bytes: usize,
) -> candle_core::Result<MaintenanceOutcome> {
    let Some(mut snapshot) = MemoryUsage.query_cuda_allocator(&maintained.device)? else {
        return Ok(maintenance_failure_outcome());
    };
    record_allocator_metrics(&maintained.device, snapshot);

    let thresholds = PressureThresholds::from_snapshot(snapshot);
    let exceeds_physical_capacity = transient_bytes > snapshot.total;
    let cached = snapshot
        .async_pool
        .map(|pool| pool.current.cached())
        .unwrap_or(0);
    let hard_pressure = snapshot.available < thresholds.hard_free;
    let required_free = thresholds.required_available(transient_bytes, cached);
    let transient_required = thresholds.transient_required_available(transient_bytes, cached);
    metrics::gauge!(
        "mistralrs_cuda_memory_required_available_bytes",
        "device" => maintained.device.device_pretty_repr(),
        "reason" => point.label()
    )
    .set(required_free as f64);
    let preflight_pressure = point == MaintenancePoint::PromptPreflight
        && transient_required > 0
        && snapshot.available < transient_required;
    let capture_active =
        crate::pipeline::cuda_graph::cuda_graph_memory_pool_scope_active(&maintained.device)?;
    if capture_active {
        maintained.capture_deferred = true;
        record_pending(&maintained.device, true);
        record_maintenance(&maintained.device, point, "trim", "deferred");
        return Ok(MaintenanceOutcome {
            transient_pressure: preflight_pressure,
            insufficient_total_capacity: exceeds_physical_capacity,
            capture_active: true,
            reclaim_deferred: true,
            ..MaintenanceOutcome::default()
        });
    }

    let capture_deferred = std::mem::take(&mut maintained.capture_deferred);
    if capture_deferred {
        record_pending(&maintained.device, false);
    }
    let force_barrier = hard_pressure
        || preflight_pressure
        || point == MaintenancePoint::GraphReclaimed
        || (capture_deferred && point == MaintenancePoint::Idle);
    if force_barrier {
        if hard_pressure {
            record_pressure(&maintained.device, "hard");
        } else if preflight_pressure {
            record_pressure(&maintained.device, "transient");
        }
        synchronize_context(&maintained.device)?;
        record_maintenance(&maintained.device, point, "barrier", "success");
        snapshot = MemoryUsage
            .query_cuda_allocator(&maintained.device)?
            .expect("CUDA device disappeared during memory maintenance");
        record_allocator_metrics(&maintained.device, snapshot);
    }

    let thresholds = PressureThresholds::from_snapshot(snapshot);
    let cached = snapshot
        .async_pool
        .map(|pool| pool.current.cached())
        .unwrap_or(0);
    let cooldown_elapsed = maintained
        .last_trim
        .is_none_or(|last| last.elapsed() >= TRIM_COOLDOWN);
    let should_trim = if force_barrier || point == MaintenancePoint::GraphReclaimed {
        cached > thresholds.warm_cache
    } else {
        match point {
            MaintenancePoint::Idle => {
                cached
                    > thresholds
                        .warm_cache
                        .saturating_mul(IDLE_RECLAIM_MULTIPLIER)
            }
            MaintenancePoint::PromptBoundary => {
                snapshot.available < thresholds.base_free
                    && cached > thresholds.warm_cache
                    && cooldown_elapsed
            }
            MaintenancePoint::PromptPreflight => {
                snapshot.available < thresholds.required_available(transient_bytes, cached)
                    && cached > thresholds.warm_cache
            }
            MaintenancePoint::GraphReclaimed => unreachable!(),
        }
    };

    if should_trim {
        let pool = snapshot
            .async_pool
            .expect("reclaimable CUDA pool bytes require an async pool");
        let target = pool.current.used.saturating_add(thresholds.warm_cache);
        let before = pool.current.reserved;
        MemoryUsage.trim_cuda_memory_pool(&maintained.device, target)?;
        let after = MemoryUsage
            .query_cuda_allocator(&maintained.device)?
            .expect("CUDA device disappeared during memory maintenance");
        record_allocator_metrics(&maintained.device, after);
        record_reclaimed(&maintained.device, point, before, after);
        maintained.last_trim = Some(Instant::now());
        snapshot = after;
    } else {
        record_maintenance(&maintained.device, point, "observe", "success");
    }

    let thresholds = PressureThresholds::from_snapshot(snapshot);
    let cached = snapshot
        .async_pool
        .map(|pool| pool.current.cached())
        .unwrap_or(0);
    let required_free = thresholds.required_available(transient_bytes, cached);
    let graph_pressure = snapshot.available < required_free;
    let transient_required = thresholds.transient_required_available(transient_bytes, cached);
    let transient_pressure = transient_required > 0 && snapshot.available < transient_required;
    let insufficient_total_capacity = exceeds_physical_capacity;
    if graph_pressure && !hard_pressure && !preflight_pressure {
        record_pressure(
            &maintained.device,
            if snapshot.available < thresholds.hard_free {
                "hard"
            } else if transient_pressure {
                "transient"
            } else {
                "reserve"
            },
        );
    }
    Ok(MaintenanceOutcome {
        graph_pressure,
        transient_pressure,
        insufficient_total_capacity,
        maintenance_failed: false,
        capture_active: false,
        reclaim_deferred: false,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PressureThresholds {
    base_free: usize,
    hard_free: usize,
    warm_cache: usize,
}

impl PressureThresholds {
    fn from_snapshot(snapshot: CudaAllocatorSnapshot) -> Self {
        let base_free =
            (snapshot.total / BASE_FREE_DIVISOR).clamp(BASE_FREE_MIN_BYTES, BASE_FREE_MAX_BYTES);
        Self {
            base_free,
            hard_free: base_free / 2,
            warm_cache: (snapshot.total / WARM_CACHE_DIVISOR)
                .clamp(WARM_CACHE_MIN_BYTES, WARM_CACHE_MAX_BYTES),
        }
    }

    fn required_available(self, transient_bytes: usize, cached_bytes: usize) -> usize {
        self.base_free
            .max(self.transient_required_available(transient_bytes, cached_bytes))
    }

    fn transient_required_available(self, transient_bytes: usize, cached_bytes: usize) -> usize {
        if transient_bytes == 0 {
            0
        } else {
            transient_bytes
                .saturating_add(self.warm_cache)
                .saturating_sub(cached_bytes)
        }
    }
}

fn synchronize_context(device: &Device) -> candle_core::Result<()> {
    MemoryUsage.synchronize_cuda_context(device)?;
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn record_allocator_metrics(device: &Device, snapshot: CudaAllocatorSnapshot) {
    let device = device.device_pretty_repr();
    for (state, value) in [("total", snapshot.total), ("available", snapshot.available)] {
        metrics::gauge!(
            "mistralrs_cuda_device_memory_bytes",
            "device" => device.clone(),
            "state" => state
        )
        .set(value as f64);
    }
    if let Some(pool) = snapshot.async_pool {
        for (state, watermark, value) in [
            ("reserved", "current", pool.current.reserved),
            ("used", "current", pool.current.used),
            ("reclaimable", "current", pool.current.cached()),
            ("reserved", "high", pool.reserved_high),
            ("used", "high", pool.used_high),
        ] {
            metrics::gauge!(
                "mistralrs_cuda_memory_pool_bytes",
                "device" => device.clone(),
                "pool" => "async",
                "state" => state,
                "watermark" => watermark
            )
            .set(value as f64);
        }
        metrics::gauge!(
            "mistralrs_cuda_memory_pool_release_threshold_bytes",
            "device" => device.clone(),
            "pool" => "async"
        )
        .set(pool.release_threshold as f64);
    }
    if let Some(pool) = snapshot.graph_pool {
        for (state, watermark, value) in [
            ("reserved", "current", pool.reserved),
            ("used", "current", pool.used),
            ("reserved", "high", pool.reserved_high),
            ("used", "high", pool.used_high),
        ] {
            metrics::gauge!(
                "mistralrs_cuda_memory_pool_bytes",
                "device" => device.clone(),
                "pool" => "graph",
                "state" => state,
                "watermark" => watermark
            )
            .set(value as f64);
        }
    }
}

fn record_reclaimed(
    device: &Device,
    point: MaintenancePoint,
    reserved_before: usize,
    after: CudaAllocatorSnapshot,
) {
    let reserved_after = after
        .async_pool
        .map(|pool| pool.current.reserved)
        .unwrap_or(reserved_before);
    let reclaimed = reserved_before.saturating_sub(reserved_after);
    let outcome = if reclaimed == 0 {
        "unchanged"
    } else {
        metrics::counter!(
            "mistralrs_cuda_memory_reclaimed_bytes_total",
            "device" => device.device_pretty_repr(),
            "pool" => "async",
            "reason" => point.label()
        )
        .increment(u64::try_from(reclaimed).unwrap_or(u64::MAX));
        "success"
    };
    record_maintenance(device, point, "trim", outcome);
}

fn record_pressure(device: &Device, level: &'static str) {
    metrics::counter!(
        "mistralrs_cuda_memory_pressure_total",
        "device" => device.device_pretty_repr(),
        "level" => level
    )
    .increment(1);
}

fn record_pending(device: &Device, pending: bool) {
    metrics::gauge!(
        "mistralrs_cuda_memory_maintenance_pending",
        "device" => device.device_pretty_repr()
    )
    .set(if pending { 1.0 } else { 0.0 });
}

fn record_maintenance(
    device: &Device,
    point: MaintenancePoint,
    action: &'static str,
    outcome: &'static str,
) {
    metrics::counter!(
        "mistralrs_cuda_memory_maintenance_total",
        "device" => device.device_pretty_repr(),
        "reason" => point.label(),
        "action" => action,
        "outcome" => outcome
    )
    .increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::memory_usage::{
        CudaGraphMemoryUsage, CudaMemoryPoolSnapshot, CudaMemoryPoolUsage,
    };

    const GIB: usize = 1024 * 1024 * 1024;

    fn snapshot(available: usize, reserved: usize, used: usize) -> CudaAllocatorSnapshot {
        CudaAllocatorSnapshot {
            total: 96 * GIB,
            available,
            async_pool: Some(CudaMemoryPoolSnapshot {
                current: CudaMemoryPoolUsage { reserved, used },
                reserved_high: reserved,
                used_high: used,
                release_threshold: 0,
            }),
            graph_pool: Some(CudaGraphMemoryUsage {
                reserved: 0,
                used: 0,
                reserved_high: 0,
                used_high: 0,
            }),
        }
    }

    #[test]
    fn pressure_thresholds_use_cuda_capacity() {
        let thresholds = PressureThresholds::from_snapshot(snapshot(2 * GIB, 95 * GIB, 90 * GIB));
        assert_eq!(thresholds.base_free, 768 * BYTES_PER_MIB);
        assert_eq!(thresholds.hard_free, 384 * BYTES_PER_MIB);
        assert_eq!(thresholds.warm_cache, 768 * BYTES_PER_MIB);
    }

    #[test]
    fn prompt_preflight_credits_reusable_async_pool_memory() {
        let snapshot = snapshot(8 * GIB, 6 * GIB, 2 * GIB);
        let thresholds = PressureThresholds::from_snapshot(snapshot);
        let cached = snapshot.async_pool.unwrap().current.cached();
        let transient = 11 * GIB;

        assert_eq!(cached, 4 * GIB);
        assert!(snapshot.available >= thresholds.required_available(transient, cached));
        assert!(snapshot.available < transient.saturating_add(thresholds.warm_cache));
    }

    #[test]
    fn prompt_preflight_preserves_only_one_warm_reserve() {
        let snapshot = snapshot(8 * GIB, 3 * GIB, 2 * GIB);
        let thresholds = PressureThresholds::from_snapshot(snapshot);
        let cached = snapshot.async_pool.unwrap().current.cached();

        assert_eq!(
            thresholds.transient_required_available(8 * GIB, cached),
            8 * GIB + thresholds.warm_cache - cached
        );
    }

    #[test]
    fn active_capture_suppresses_cross_device_reclamation() {
        let aggregate = aggregate_outcomes([
            MaintenanceOutcome {
                graph_pressure: true,
                transient_pressure: true,
                capture_active: false,
                ..MaintenanceOutcome::default()
            },
            MaintenanceOutcome {
                capture_active: true,
                ..MaintenanceOutcome::default()
            },
        ]);
        assert!(aggregate.capture_active);
        assert!(!aggregate.graph_pressure);
        assert!(aggregate.transient_pressure);
        assert!(aggregate.reclaim_deferred);
    }

    #[test]
    fn failed_accounting_blocks_transient_work_without_reclaiming_graphs() {
        let aggregate = aggregate_outcomes([
            MaintenanceOutcome {
                graph_pressure: true,
                ..MaintenanceOutcome::default()
            },
            maintenance_failure_outcome(),
        ]);
        assert!(!aggregate.graph_pressure);
        assert!(aggregate.transient_pressure);
        assert!(aggregate.maintenance_failed);
        assert!(aggregate.reclaim_deferred);
    }

    #[test]
    fn permanent_capacity_on_any_required_device_rejects_the_prompt() {
        let insufficient = MaintenanceOutcome {
            insufficient_total_capacity: true,
            ..MaintenanceOutcome::default()
        };
        assert!(aggregate_outcomes([insufficient, insufficient]).insufficient_total_capacity);
        assert!(
            aggregate_outcomes([insufficient, MaintenanceOutcome::default()])
                .insufficient_total_capacity
        );
    }

    #[test]
    fn threshold_clamps_cover_small_and_large_devices() {
        let mut small = snapshot(1, 1, 1);
        small.total = 4 * GIB;
        assert_eq!(
            PressureThresholds::from_snapshot(small).base_free,
            BASE_FREE_MIN_BYTES
        );
        let mut large = snapshot(1, 1, 1);
        large.total = 512 * GIB;
        assert_eq!(
            PressureThresholds::from_snapshot(large).base_free,
            BASE_FREE_MAX_BYTES
        );
        assert_eq!(
            PressureThresholds::from_snapshot(large).warm_cache,
            WARM_CACHE_MAX_BYTES
        );
    }

    #[test]
    fn prompt_batch_pressure_reduces_then_rejects_before_empty_submission() {
        assert_eq!(
            prompt_batch_memory_action(16, true),
            PromptBatchMemoryAction::Retain(8)
        );
        assert_eq!(
            prompt_batch_memory_action(3, true),
            PromptBatchMemoryAction::Retain(2)
        );
        assert_eq!(
            prompt_batch_memory_action(2, true),
            PromptBatchMemoryAction::Retain(1)
        );
        assert_eq!(
            prompt_batch_memory_action(1, true),
            PromptBatchMemoryAction::Reject
        );
        assert_eq!(
            prompt_batch_memory_action(1, false),
            PromptBatchMemoryAction::Proceed
        );
    }
}
