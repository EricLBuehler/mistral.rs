use super::*;

pub(crate) struct CudaPhaseTimer {
    pub(super) start: CudaEvent,
    pub(super) stream: Arc<CudaStream>,
}

impl CudaPhaseTimer {
    pub(crate) fn start(stream: &Arc<CudaStream>) -> candle_core::Result<Option<Self>> {
        let enabled = *CUDA_PHASE_TIMINGS_ENABLED.get_or_init(|| {
            std::env::var(CUDA_PHASE_TIMINGS_ENV)
                .ok()
                .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        });
        if !enabled {
            return Ok(None);
        }
        let capture_status = stream.capture_status().map_err(candle_core::Error::wrap)?;
        if capture_status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Ok(None);
        }
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        Ok(Some(Self {
            start,
            stream: stream.clone(),
        }))
    }

    pub(crate) fn finish(
        self,
        component: &'static str,
        batch: usize,
        rows: usize,
    ) -> candle_core::Result<()> {
        let end = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        end.synchronize().map_err(candle_core::Error::wrap)?;
        let latency_ms = self
            .start
            .elapsed_ms(&end)
            .map_err(candle_core::Error::wrap)?;
        tracing::info!(component, batch, rows, latency_ms, "CUDA phase timing");
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphComponent {
    Target,
    #[cfg(any(test, all(feature = "flash-attn", target_family = "unix")))]
    DFlash,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphEvictionReason {
    Capacity,
    MemoryPressure,
    SpecStateBudget,
}

impl CudaGraphEvictionReason {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Capacity => "capacity",
            Self::MemoryPressure => "memory_pressure",
            Self::SpecStateBudget => "spec_state_budget",
        }
    }
}

impl CudaGraphComponent {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Target => "target",
            #[cfg(any(test, all(feature = "flash-attn", target_family = "unix")))]
            Self::DFlash => "dflash",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphEvent {
    Capture,
    Replay,
    EagerFallback,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphDispatchMode {
    Replay,
    Eager,
    Skipped,
}

impl CudaGraphDispatchMode {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Replay => "replay",
            Self::Eager => "eager",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphDispatchReason {
    CacheHit,
    Disabled,
    ModelUnsupported,
    SpeculativeConflict,
    PagedAttentionUnavailable,
    Prefill,
    IncompatibleShape,
    BatchUnsupported,
    CacheConfigUnavailable,
    RuntimeDisabled,
    PaddingUnavailable,
    CachePopulation,
    Fallback,
    MemoryPressure,
}

impl CudaGraphDispatchReason {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::CacheHit => "cache_hit",
            Self::Disabled => "disabled",
            Self::ModelUnsupported => "model_unsupported",
            Self::SpeculativeConflict => "speculative_conflict",
            Self::PagedAttentionUnavailable => "paged_attention_unavailable",
            Self::Prefill => "prefill",
            Self::IncompatibleShape => "incompatible_shape",
            Self::BatchUnsupported => "batch_unsupported",
            Self::CacheConfigUnavailable => "cache_config_unavailable",
            Self::RuntimeDisabled => "runtime_disabled",
            Self::PaddingUnavailable => "padding_unavailable",
            Self::CachePopulation => "cache_population",
            Self::Fallback => "fallback",
            Self::MemoryPressure => "memory_pressure",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct CudaGraphDispatchLabels {
    pub(super) component: &'static str,
    pub(super) mode: &'static str,
    pub(super) reason: &'static str,
}

pub(super) const fn cuda_graph_dispatch_labels(
    component: CudaGraphComponent,
    mode: CudaGraphDispatchMode,
    reason: CudaGraphDispatchReason,
) -> CudaGraphDispatchLabels {
    CudaGraphDispatchLabels {
        component: component.label(),
        mode: mode.label(),
        reason: reason.label(),
    }
}

pub(crate) fn record_cuda_graph_dispatch(
    component: CudaGraphComponent,
    mode: CudaGraphDispatchMode,
    reason: CudaGraphDispatchReason,
) {
    let labels = cuda_graph_dispatch_labels(component, mode, reason);
    metrics::counter!(
        CUDA_GRAPH_DISPATCH_METRIC,
        "component" => labels.component,
        "mode" => labels.mode,
        "reason" => labels.reason,
    )
    .increment(1);
}

impl CudaGraphEvent {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Capture => "capture",
            Self::Replay => "replay",
            Self::EagerFallback => "eager_fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum CudaGraphOutcome {
    Success,
    Failure,
}

impl CudaGraphOutcome {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct CudaGraphEventLabels {
    pub(super) component: &'static str,
    pub(super) event: &'static str,
    pub(super) outcome: &'static str,
}

pub(super) const fn cuda_graph_event_labels(
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
) -> CudaGraphEventLabels {
    CudaGraphEventLabels {
        component: component.label(),
        event: event.label(),
        outcome: outcome.label(),
    }
}

pub(super) fn record_cuda_graph_event(
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
) {
    let labels = cuda_graph_event_labels(component, event, outcome);
    metrics::counter!(
        CUDA_GRAPH_EVENTS_METRIC,
        "component" => labels.component,
        "event" => labels.event,
        "outcome" => labels.outcome,
    )
    .increment(1);
}

pub(crate) fn record_cuda_graph_evictions(
    component: CudaGraphComponent,
    reason: CudaGraphEvictionReason,
    count: usize,
) {
    metrics::counter!(
        CUDA_GRAPH_EVICTIONS_METRIC,
        "component" => component.label(),
        "reason" => reason.label()
    )
    .increment(u64::try_from(count).unwrap_or(u64::MAX));
}

pub(crate) fn record_cuda_graph_resident_entries(component: CudaGraphComponent, count: usize) {
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    metrics::gauge!(
        CUDA_GRAPH_RESIDENT_ENTRIES_METRIC,
        "component" => component.label()
    )
    .set(f64::from(count));
}

pub(crate) fn take_cuda_graph_capacity_eviction<T>(
    entries: &mut Vec<T>,
    capacity: usize,
) -> Option<T> {
    assert!(capacity > 0, "CUDA graph cache capacity must be nonzero");
    (entries.len() >= capacity).then(|| entries.remove(0))
}

pub(crate) fn reclaim_cuda_graph_entries(
    max_entries: usize,
    reclaim_target: impl FnOnce(usize) -> usize,
    reclaim_speculative: impl FnOnce(usize) -> usize,
) -> usize {
    if max_entries == 0 {
        return 0;
    }
    let target = reclaim_target(max_entries);
    debug_assert!(target <= max_entries);
    let remaining = max_entries - target;
    if remaining == 0 {
        return target;
    }
    let speculative = reclaim_speculative(remaining);
    debug_assert!(speculative <= remaining);
    target + speculative
}

#[must_use]
pub(crate) struct CudaGraphEventGuard {
    pub(super) component: CudaGraphComponent,
    pub(super) event: CudaGraphEvent,
    pub(super) outcome: CudaGraphOutcome,
}

impl CudaGraphEventGuard {
    pub(crate) const fn new(component: CudaGraphComponent, event: CudaGraphEvent) -> Self {
        Self {
            component,
            event,
            outcome: CudaGraphOutcome::Failure,
        }
    }

    pub(crate) fn success(mut self) {
        self.outcome = CudaGraphOutcome::Success;
    }
}

impl Drop for CudaGraphEventGuard {
    fn drop(&mut self) {
        record_cuda_graph_event(self.component, self.event, self.outcome);
        if self.outcome == CudaGraphOutcome::Success && self.event == CudaGraphEvent::EagerFallback
        {
            record_cuda_graph_dispatch(
                self.component,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::Fallback,
            );
        }
    }
}
