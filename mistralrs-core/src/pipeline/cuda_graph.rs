use std::{
    collections::HashMap,
    fmt,
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use candle_core::cuda_backend::cudarc::driver::{
    sys, CudaEvent, CudaStream, DevicePtr, PinnedHostSlice,
};
use candle_core::{DType, Device, DeviceLocation, Storage, Tensor, Var};

#[cfg(target_family = "unix")]
use crate::paged_attention::plan::DecodePlan;
use crate::{
    flashinfer::{
        make_fa3_decode_state, Fa3DecodeState, FlashInferMetadata, FlashInferPagedAttentionView,
        FlashInferPagedAttentionViews, FlashInferPagedKv, FlashInferTilePlan,
    },
    paged_attention::{AttentionBackendKind, ModelConfigLike},
};

use crate::device_map::DeviceMapper;
use crate::kv_cache::HybridCache;
use crate::paged_attention::_PAD_SLOT_ID;
use crate::pipeline::{
    decode_positions_tensor,
    text_models_inputs_processor::{
        make_flash_params, DecodePagedRows, DecodePagedRowsGraphKey, FlashParams,
        PagedAttentionInputMetadata, PagedDecodeMetadataRequirements,
    },
    DecodeGraphPrecaptureCtx, RecurrentBatchKind,
};
use crate::speculative::SpeculativeGraphState;

const CUDA_GRAPH_INSTANTIATE_FLAGS: u64 =
    sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64;
// Matches the standard CUDA paged-attention V2 partition size.
const PAGED_ATTENTION_PARTITION_SIZE: usize = 512;
const TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY: usize = 64;
// Batches up to this size get their own graph; larger ones pad up to the next power of two.
pub(crate) const CUDA_GRAPH_EXACT_BATCH_BUCKETS: usize = 8;
pub(crate) const CUDA_GRAPH_MAX_BATCH_BUCKET: usize = 64;
pub(crate) const CUDA_GRAPH_PRECAPTURE_MAX_BATCH: usize = 16;
const CUDA_GRAPH_SPEC_STATE_BUDGET_PERCENT: usize = 4;
const CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES_ENV: &str = "MISTRALRS_CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES";
const CUDA_GRAPH_EVENTS_METRIC: &str = "mistralrs_cuda_graph_events_total";
const CUDA_GRAPH_DISPATCH_METRIC: &str = "mistralrs_cuda_graph_dispatch_total";
const CUDA_GRAPH_EVICTIONS_METRIC: &str = "mistralrs_cuda_graph_evictions_total";
const CUDA_GRAPH_RESIDENT_ENTRIES_METRIC: &str = "mistralrs_cuda_graph_resident_entries";
static NEXT_CUDA_DECODE_GRAPH_GENERATION: AtomicU64 = AtomicU64::new(1);
static CUDA_GRAPH_MEMORY_POOL_SCOPES: OnceLock<Mutex<HashMap<usize, MemoryPoolScopeState>>> =
    OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphComponent {
    Target,
    DFlash,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphEvictionReason {
    Capacity,
    MemoryPressure,
    SpecStateBudget,
}

impl CudaGraphEvictionReason {
    const fn label(self) -> &'static str {
        match self {
            Self::Capacity => "capacity",
            Self::MemoryPressure => "memory_pressure",
            Self::SpecStateBudget => "spec_state_budget",
        }
    }
}

impl CudaGraphComponent {
    const fn label(self) -> &'static str {
        match self {
            Self::Target => "target",
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
    const fn label(self) -> &'static str {
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
}

impl CudaGraphDispatchReason {
    const fn label(self) -> &'static str {
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
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CudaGraphDispatchLabels {
    component: &'static str,
    mode: &'static str,
    reason: &'static str,
}

const fn cuda_graph_dispatch_labels(
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
    const fn label(self) -> &'static str {
        match self {
            Self::Capture => "capture",
            Self::Replay => "replay",
            Self::EagerFallback => "eager_fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CudaGraphOutcome {
    Success,
    Failure,
}

impl CudaGraphOutcome {
    const fn label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CudaGraphEventLabels {
    component: &'static str,
    event: &'static str,
    outcome: &'static str,
}

const fn cuda_graph_event_labels(
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

fn record_cuda_graph_event(
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
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
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

struct MemoryPoolScopeState {
    guards: usize,
    release_threshold: u64,
}

/// Graph batch bucket a decode batch pads up to, or None when it is too large to graph.
pub(crate) fn cuda_graph_batch_bucket(batch: usize) -> Option<usize> {
    if batch == 0 {
        None
    } else if batch <= CUDA_GRAPH_EXACT_BATCH_BUCKETS {
        Some(batch)
    } else {
        let bucket = batch.next_power_of_two();
        (bucket <= CUDA_GRAPH_MAX_BATCH_BUCKET).then_some(bucket)
    }
}

/// The batch buckets captured ahead of time at load.
pub(crate) fn cuda_graph_precapture_batches() -> impl Iterator<Item = usize> {
    (1..=CUDA_GRAPH_EXACT_BATCH_BUCKETS).chain(std::iter::once(CUDA_GRAPH_PRECAPTURE_MAX_BATCH))
}

pub(crate) fn cuda_graph_startup_capture_allowed(q_len: usize) -> bool {
    q_len > 0
}

pub(crate) fn prepare_fa3_decode_schedules(
    metadata: &PagedAttentionInputMetadata,
) -> candle_core::Result<()> {
    let Some(flashinfer) = metadata.flashinfer.as_ref() else {
        return Ok(());
    };
    flashinfer.for_each_fa3_decode_schedule(|prepare| {
        mistralrs_paged_attn::fa3_prepare_decode_metadata(
            mistralrs_paged_attn::Fa3DecodeMetadata {
                paged_kv_indptr: prepare.paged_kv_indptr,
                paged_kv_indices: prepare.paged_kv_indices,
                paged_kv_last_page_len: prepare.paged_kv_last_page_len,
                page_table: &prepare.buffers.page_table,
                seqused_k: &prepare.buffers.seqused_k,
                cu_seqlens_q: &prepare.buffers.cu_seqlens_q,
                scheduler_metadata: &prepare.buffers.scheduler_metadata,
            },
            prepare.buffers.schedule(prepare.key)?,
        )
    })
}

/// One decode step, padded up to its graph batch bucket. Pad rows alias row 0 for reads and skip
/// their KV and recurrent writes, so the model can run them and drop their outputs.
#[derive(Clone)]
pub(crate) struct CudaGraphDecodeStep {
    pub(crate) input_ids: Tensor,
    pub(crate) seqlen_offsets: Vec<usize>,
    pub(crate) context_lens: Vec<(usize, usize)>,
    pub(crate) position_ids: Vec<usize>,
    pub(crate) metadata: PagedAttentionInputMetadata,
    pub(crate) state_indices: Option<Vec<u32>>,
    pub(crate) real_batch: usize,
}

pub(crate) struct CudaGraphDecodeStepInputs<'a> {
    pub(crate) input_ids: &'a Tensor,
    pub(crate) seqlen_offsets: &'a [usize],
    pub(crate) context_lens: &'a [(usize, usize)],
    pub(crate) position_ids: &'a [usize],
    pub(crate) metadata: &'a PagedAttentionInputMetadata,
    pub(crate) state_indices: Option<&'a [u32]>,
    pub(crate) pad_slot: Option<u32>,
}

impl CudaGraphDecodeStep {
    /// Returns None when the step can't be padded (no host rows to rebuild the metadata from, or a
    /// hybrid batch without a pad slot).
    pub(crate) fn padded(
        inputs: CudaGraphDecodeStepInputs<'_>,
        batch: usize,
    ) -> candle_core::Result<Option<Self>> {
        let CudaGraphDecodeStepInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices,
            pad_slot,
        } = inputs;
        let real_batch = input_ids.dim(0)?;
        if real_batch == batch {
            return Ok(Some(Self {
                input_ids: input_ids.clone(),
                seqlen_offsets: seqlen_offsets.to_vec(),
                context_lens: context_lens.to_vec(),
                position_ids: position_ids.to_vec(),
                metadata: metadata.clone(),
                state_indices: state_indices.map(<[u32]>::to_vec),
                real_batch,
            }));
        }
        let Some(rows) = metadata.decode_rows.as_ref() else {
            return Ok(None);
        };
        let state_indices = match (state_indices, pad_slot) {
            (Some(slots), Some(pad_slot)) => {
                let mut padded = slots.to_vec();
                padded.resize(batch, pad_slot);
                Some(padded)
            }
            (Some(_), None) => return Ok(None),
            (None, _) => None,
        };
        let pad = batch - real_batch;
        let (_, q_len) = input_ids.dims2()?;
        let pad_ids = input_ids.narrow(0, 0, 1)?.repeat((pad, 1))?;
        let input_ids = Tensor::cat(&[input_ids, &pad_ids], 0)?;
        let mut seqlen_offsets = seqlen_offsets.to_vec();
        seqlen_offsets.resize(batch, seqlen_offsets[0]);
        let mut context_lens = context_lens.to_vec();
        context_lens.resize(batch, context_lens[0]);
        let mut position_ids = position_ids.to_vec();
        position_ids.resize(batch, position_ids[0]);
        let rows = Arc::new(rows.padded(batch));
        if rows.query_len != q_len {
            candle_core::bail!(
                "CUDA graph decode rows cover {} query tokens but the input has {q_len}",
                rows.query_len
            );
        }
        let metadata = rows.build().map_err(candle_core::Error::msg)?;
        Ok(Some(Self {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices,
            real_batch,
        }))
    }

    pub(crate) fn batch(&self) -> usize {
        self.seqlen_offsets.len()
    }

    /// Drops the pad rows from a `[batch, ...]` or `[batch * q, ...]` output.
    pub(crate) fn narrow_rows(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        let batch = self.batch();
        if batch == self.real_batch {
            return Ok(tensor.clone());
        }
        let rows = tensor.dim(0)? / batch * self.real_batch;
        tensor.narrow(0, 0, rows)
    }

    fn one_token_continuation(&self, input_ids: Tensor) -> candle_core::Result<Option<Self>> {
        let (batch, q_len) = input_ids.dims2()?;
        let Some(rows) = self.metadata.decode_rows.as_ref() else {
            return Ok(None);
        };
        if q_len != 1
            || rows.query_len != 1
            || rows.decode_window != 1
            || batch != self.batch()
            || rows.batch_size() != batch
            || self.real_batch == 0
            || self.real_batch > batch
            || self.seqlen_offsets.len() != batch
            || self.context_lens.len() != batch
            || self.position_ids.len() != batch
        {
            return Ok(None);
        }

        let mut slot_mappings = Vec::with_capacity(self.real_batch);
        let mut block_tables = Vec::with_capacity(self.real_batch);
        let mut context_lens = Vec::with_capacity(self.real_batch);
        let mut full_block_tables = Vec::with_capacity(self.real_batch);
        let mut full_context_lens = Vec::with_capacity(self.real_batch);
        for row in 0..self.real_batch {
            let Some(&current_slot) = rows.slot_mappings[row].first() else {
                return Ok(None);
            };
            let Ok(current_slot) = usize::try_from(current_slot) else {
                return Ok(None);
            };
            let full_table = &rows.full_block_tables[row];
            let current_block = current_slot / rows.block_size;
            let Some(current_block_idx) =
                full_table.iter().position(|&block| block == current_block)
            else {
                return Ok(None);
            };
            let current_block_offset = current_slot % rows.block_size;
            let (next_block_idx, next_slot) = if current_block_offset + 1 < rows.block_size {
                (current_block_idx, current_slot + 1)
            } else {
                let next_block_idx = current_block_idx + 1;
                let Some(&next_block) = full_table.get(next_block_idx) else {
                    return Ok(None);
                };
                let Some(next_slot) = next_block.checked_mul(rows.block_size) else {
                    return Ok(None);
                };
                (next_block_idx, next_slot)
            };
            let Ok(next_slot) = i64::try_from(next_slot) else {
                return Ok(None);
            };
            let Some(next_full_context_len) = rows.full_context_lens[row].checked_add(1) else {
                return Ok(None);
            };

            let (paged_table, paged_context_len) = match rows.sliding_window {
                Some(window) => {
                    let window_start = next_full_context_len.saturating_sub(window);
                    let block_aligned_start = window_start / rows.block_size * rows.block_size;
                    let paged_context_len = next_full_context_len - block_aligned_start;
                    let needed_blocks = paged_context_len.div_ceil(rows.block_size);
                    let table_end = next_block_idx + 1;
                    let table_start = table_end.saturating_sub(needed_blocks);
                    (
                        full_table[table_start..table_end].to_vec(),
                        paged_context_len,
                    )
                }
                None => (full_table.clone(), next_full_context_len),
            };
            slot_mappings.push(vec![next_slot]);
            block_tables.push(paged_table);
            context_lens.push(paged_context_len);
            full_block_tables.push(full_table.clone());
            full_context_lens.push(next_full_context_len);
        }

        let rows = Arc::new(
            DecodePagedRows {
                slot_mappings,
                block_tables,
                context_lens,
                full_block_tables,
                full_context_lens,
                query_len: 1,
                block_size: rows.block_size,
                use_standard_metadata: rows.use_standard_metadata,
                max_paged_context_len: rows.max_paged_context_len,
                sliding_window: rows.sliding_window,
                decode_window: rows.decode_window,
                devices: rows.devices.clone(),
                num_kv_heads: rows.num_kv_heads,
            }
            .padded(batch),
        );
        let metadata = rows.build_graph_staged().map_err(candle_core::Error::msg)?;
        let Some(mut seqlen_offsets) = self.seqlen_offsets[..self.real_batch]
            .iter()
            .map(|offset| offset.checked_add(1))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        seqlen_offsets.resize(batch, seqlen_offsets[0]);
        let mut context_lens = self.context_lens[..self.real_batch].to_vec();
        context_lens.resize(batch, context_lens[0]);
        let Some(mut position_ids) = self.position_ids[..self.real_batch]
            .iter()
            .map(|position| position.checked_add(1))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        position_ids.resize(batch, position_ids[0]);
        Ok(Some(Self {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices: self.state_indices.clone(),
            real_batch: self.real_batch,
        }))
    }
}

/// A fabricated batch-1 decode step (token 0 at position 0 over one block, no KV writes) that the
/// precapture pads up to every bucket.
pub(crate) struct CudaGraphPrecaptureInputs {
    pub(crate) input_ids: Tensor,
    pub(crate) seqlen_offsets: Vec<usize>,
    pub(crate) context_lens: Vec<(usize, usize)>,
    pub(crate) position_ids: Vec<usize>,
    pub(crate) metadata: PagedAttentionInputMetadata,
    pub(crate) flash_meta: FlashParams,
}

impl CudaGraphPrecaptureInputs {
    pub(crate) fn new(
        ctx: &DecodeGraphPrecaptureCtx,
        q_len: usize,
        device: &Device,
        mapper: Option<&dyn DeviceMapper>,
    ) -> candle_core::Result<Self> {
        let devices = mapper
            .map(|mapper| mapper.get_unique_devices())
            .unwrap_or_else(|| vec![device.clone()]);
        let rows = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![_PAD_SLOT_ID; q_len]],
            block_tables: vec![vec![0]; q_len],
            context_lens: vec![1; q_len],
            full_block_tables: vec![vec![0]; q_len],
            full_context_lens: vec![1; q_len],
            query_len: q_len,
            block_size: ctx.block_size,
            use_standard_metadata: ctx.attention_backend == AttentionBackendKind::Standard,
            max_paged_context_len: ctx.max_paged_context_len,
            sliding_window: ctx.sliding_window,
            decode_window: 1,
            devices,
            num_kv_heads: ctx.num_kv_heads,
        });
        let metadata = rows.build_materialized().map_err(candle_core::Error::msg)?;
        let q_len_u32 = u32::try_from(q_len).map_err(candle_core::Error::wrap)?;
        let flash_meta = if crate::using_flash_attn() {
            make_flash_params(
                device,
                mapper,
                &[0, q_len_u32],
                &[0, q_len_u32],
                ctx.sliding_window,
                true,
                false,
            )
            .map_err(candle_core::Error::msg)?
        } else {
            FlashParams::empty(true)
        };
        Ok(Self {
            input_ids: Tensor::zeros((1, q_len), DType::U32, device)?,
            seqlen_offsets: vec![0],
            context_lens: vec![(0, q_len)],
            position_ids: vec![q_len],
            metadata,
            flash_meta,
        })
    }

    pub(crate) fn step_inputs<'a>(
        &'a self,
        state_indices: Option<&'a [u32]>,
        pad_slot: Option<u32>,
    ) -> CudaGraphDecodeStepInputs<'a> {
        CudaGraphDecodeStepInputs {
            input_ids: &self.input_ids,
            seqlen_offsets: &self.seqlen_offsets,
            context_lens: &self.context_lens,
            position_ids: &self.position_ids,
            metadata: &self.metadata,
            state_indices,
            pad_slot,
        }
    }
}

pub(crate) struct HybridGraphSlots {
    pub(crate) real: Vec<u32>,
    pub(crate) storage_generation: u64,
}

/// The batch's live recurrent slots after reserving graph capacity.
pub(crate) fn hybrid_graph_slots(
    cache: &mut HybridCache,
) -> candle_core::Result<Option<HybridGraphSlots>> {
    let Some(real) = cache.state_indices_host().map(<[u32]>::to_vec) else {
        return Ok(None);
    };
    cache.graph_pad_slot()?;
    Ok(Some(HybridGraphSlots {
        real,
        storage_generation: cache.recurrent_storage_generation(),
    }))
}

/// Points the hybrid cache's state indices at fresh `Var` buffers holding `host`, one per recurrent
/// device, so a captured forward reads slots the replay can overwrite.
pub(crate) fn install_hybrid_graph_state_indices(
    cache: &mut HybridCache,
    host: &[u32],
) -> candle_core::Result<CudaGraphVarMap> {
    let mut vars = CudaGraphVarMap::new();
    let mut tensors = Vec::new();
    for device in cache.recurrent_devices() {
        let var = Var::from_tensor(&Tensor::from_vec(host.to_vec(), (host.len(),), &device)?)?;
        tensors.push((device.clone(), var.as_detached_tensor()));
        vars.insert(device.location(), var);
    }
    cache.set_state_indices_tensors(host.to_vec(), tensors);
    Ok(vars)
}

fn copy_state_indices(
    dst: &CudaGraphVarMap,
    host: &[u32],
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    for (location, var) in dst {
        host_staging.copy_from_u32_slice("state_indices", *location, host, var)?;
    }
    Ok(())
}

pub(crate) struct CudaGraphHandle {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

unsafe impl Send for CudaGraphHandle {}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        let _ = self.stream.context().bind_to_thread();
        if !self.exec.is_null() {
            let _ = unsafe { sys::cuGraphExecDestroy(self.exec) };
            self.exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            let _ = unsafe { sys::cuGraphDestroy(self.graph) };
            self.graph = std::ptr::null_mut();
        }
    }
}

impl CudaGraphHandle {
    pub(crate) fn end_capture(stream: &Arc<CudaStream>) -> candle_core::Result<Option<Self>> {
        let mut graph = std::ptr::null_mut();
        let result = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph stream end capture failed"));
        }
        if graph.is_null() {
            return Ok(None);
        }

        let mut exec = std::ptr::null_mut();
        let result = unsafe {
            sys::cuGraphInstantiateWithFlags(&mut exec, graph, CUDA_GRAPH_INSTANTIATE_FLAGS)
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            let _ = unsafe { sys::cuGraphDestroy(graph) };
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph instantiate failed"));
        }

        Ok(Some(Self {
            graph,
            exec,
            stream: stream.clone(),
        }))
    }

    pub(crate) fn upload(&self) -> candle_core::Result<()> {
        let result = unsafe { sys::cuGraphUpload(self.exec, self.stream.cu_stream()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(
                candle_core::Error::msg(format!("{result:?}")).context("CUDA graph upload failed")
            );
        }
        let _ = self.stream.context().check_err();
        Ok(())
    }

    pub(crate) fn launch(&self) -> candle_core::Result<()> {
        let result = unsafe { sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(
                candle_core::Error::msg(format!("{result:?}")).context("CUDA graph launch failed")
            );
        }
        let _ = self.stream.context().check_err();
        Ok(())
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CudaDecodeGraphKey {
    device: DeviceLocation,
    input_shape: Vec<usize>,
    input_dtype: DType,
    recurrent_batch_kind: RecurrentBatchKind,
    max_context_len: Option<usize>,
    full_max_context_len: Option<usize>,
    tensors: Vec<CudaGraphTensorKey>,
    decode_rows: Option<DecodePagedRowsGraphKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CudaGraphTensorKey {
    name: &'static str,
    location: DeviceLocation,
    shape: Vec<usize>,
    dtype: DType,
}

type CudaGraphVarMap = HashMap<DeviceLocation, Var>;
type FlashInferDecodeScratchMaps = (
    Option<HashMap<DeviceLocation, Tensor>>,
    Option<HashMap<DeviceLocation, Tensor>>,
);

struct CudaGraphPinnedAllocation<T> {
    allocation: PinnedHostSlice<T>,
    ptr: NonNull<T>,
}

// The allocation owns the pointer and all accesses require an exclusive borrow.
unsafe impl<T: Send> Send for CudaGraphPinnedAllocation<T> {}

impl<T> CudaGraphPinnedAllocation<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.allocation.len()) }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

enum CudaGraphPinnedData {
    U8(CudaGraphPinnedAllocation<u8>),
    U32(CudaGraphPinnedAllocation<u32>),
    I32(CudaGraphPinnedAllocation<i32>),
    I64(CudaGraphPinnedAllocation<i64>),
    F32(CudaGraphPinnedAllocation<f32>),
}

struct CudaGraphPinnedBuffer {
    data: CudaGraphPinnedData,
    initialized: bool,
}

struct CudaGraphCopyCompletion {
    event: CudaEvent,
    stream: Arc<CudaStream>,
    pending: bool,
    active: bool,
    ordered_after_graph: bool,
}

pub(crate) struct CudaGraphHostStaging {
    buffers: HashMap<(&'static str, DeviceLocation), CudaGraphPinnedBuffer>,
    completions: HashMap<DeviceLocation, CudaGraphCopyCompletion>,
    graph_complete: CudaEvent,
    graph_stream: Arc<CudaStream>,
    graph_pending: bool,
}

fn same_cuda_stream(left: &CudaStream, right: &CudaStream) -> bool {
    Arc::ptr_eq(left.context(), right.context()) && left.cu_stream() == right.cu_stream()
}

impl Drop for CudaGraphHostStaging {
    fn drop(&mut self) {
        for completion in self.completions.values() {
            if completion.pending {
                let _ = completion.event.synchronize();
            } else if completion.active {
                let _ = completion.stream.synchronize();
            }
        }
    }
}

pub(crate) struct CudaDecodeGraphCaptureCtx<'a> {
    pub(crate) key: CudaDecodeGraphKey,
    pub(crate) input_ids: &'a Tensor,
    pub(crate) seqlen_offsets: &'a [usize],
    pub(crate) position_ids: &'a [usize],
    pub(crate) block_size: usize,
    pub(crate) kv_cache: &'a [(Tensor, Tensor)],
    pub(crate) metadata: &'a PagedAttentionInputMetadata,
    pub(crate) model_metadata: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
    pub(crate) activation_dtype: DType,
    pub(crate) warmup_logits: &'a Tensor,
    pub(crate) state_indices: Option<CudaGraphVarMap>,
    pub(crate) real_batch: usize,
}

pub(crate) struct CudaDecodeGraphMetadataBuffers {
    requirements: PagedDecodeMetadataRequirements,
    flashinfer_views_alias: bool,
    slot_mappings: CudaGraphVarMap,
    block_tables: Option<CudaGraphVarMap>,
    context_lens: Option<CudaGraphVarMap>,
    full_block_tables: Option<CudaGraphVarMap>,
    full_context_lens: Option<CudaGraphVarMap>,
    paged_kv_indptr: Option<CudaGraphVarMap>,
    paged_kv_indices: Option<CudaGraphVarMap>,
    paged_kv_last_page_len: Option<CudaGraphVarMap>,
    full_paged_kv_indptr: Option<CudaGraphVarMap>,
    full_paged_kv_indices: Option<CudaGraphVarMap>,
    full_paged_kv_last_page_len: Option<CudaGraphVarMap>,
    paged_kv_request_indices: Option<CudaGraphVarMap>,
    paged_kv_tile_indices: Option<CudaGraphVarMap>,
    paged_kv_o_indptr: Option<CudaGraphVarMap>,
    paged_kv_chunk_size: Option<CudaGraphVarMap>,
    paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    full_paged_kv_request_indices: Option<CudaGraphVarMap>,
    full_paged_kv_tile_indices: Option<CudaGraphVarMap>,
    full_paged_kv_o_indptr: Option<CudaGraphVarMap>,
    full_paged_kv_chunk_size: Option<CudaGraphVarMap>,
    full_paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    flashinfer_decode_tmp_v: Option<HashMap<DeviceLocation, Tensor>>,
    flashinfer_decode_tmp_s: Option<HashMap<DeviceLocation, Tensor>>,
    fa3_decode: Option<Fa3DecodeState>,
    rope_positions: CudaGraphVarMap,
}

impl CudaDecodeGraphKey {
    pub(crate) fn new(
        input_ids: &Tensor,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
        recurrent_batch_kind: RecurrentBatchKind,
    ) -> candle_core::Result<Self> {
        let decode_rows = metadata.decode_rows.as_ref().map(|rows| rows.graph_key());
        let mut tensors = Vec::new();
        if decode_rows.is_none() {
            push_graph_tensor_keys("slot_mappings", Some(&metadata.slot_mappings), &mut tensors);
            push_graph_tensor_keys("block_tables", metadata.block_tables.as_ref(), &mut tensors);
            push_graph_tensor_keys("context_lens", metadata.context_lens.as_ref(), &mut tensors);
            push_graph_tensor_keys(
                "full_block_tables",
                metadata.full_block_tables.as_ref(),
                &mut tensors,
            );
            push_graph_tensor_keys(
                "full_context_lens",
                metadata.full_context_lens.as_ref(),
                &mut tensors,
            );
            push_flashinfer_graph_tensor_keys(metadata, &mut tensors);
            if flashinfer_views_alias(metadata) {
                tensors.retain(|tensor| !tensor.name.starts_with("full_"));
            }
        }
        tensors.sort_by(|a, b| {
            a.name.cmp(b.name).then_with(|| {
                device_location_sort_key(&a.location).cmp(&device_location_sort_key(&b.location))
            })
        });

        Ok(Self {
            device: input_ids.device().location(),
            input_shape: input_ids.dims().to_vec(),
            input_dtype: input_ids.dtype(),
            recurrent_batch_kind,
            max_context_len: decode_rows
                .is_none()
                .then(|| {
                    graph_context_len(
                        metadata.max_context_len,
                        bucket_context_len(metadata.block_tables.as_ref(), block_size),
                    )
                })
                .flatten(),
            full_max_context_len: decode_rows
                .is_none()
                .then(|| {
                    graph_context_len(
                        metadata.full_max_context_len,
                        bucket_context_len(metadata.full_block_tables.as_ref(), block_size),
                    )
                })
                .flatten(),
            tensors,
            decode_rows,
        })
    }
}

impl CudaDecodeGraphMetadataBuffers {
    pub(crate) fn new(
        metadata: &PagedAttentionInputMetadata,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        seq_len: usize,
        block_size: usize,
        kv_cache: &[(Tensor, Tensor)],
        model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
        activation_dtype: DType,
    ) -> candle_core::Result<(Self, PagedAttentionInputMetadata)> {
        let slot_mappings = var_map_from_tensor_map(&metadata.slot_mappings)?;
        if seqlen_offsets.len() != position_ids.len() {
            candle_core::bail!(
                "CUDA graph decode has {} KV offsets but {} position ends",
                seqlen_offsets.len(),
                position_ids.len()
            );
        }
        let rope_positions =
            rope_positions_var_map(&metadata.slot_mappings, position_ids, seq_len)?;
        let (flashinfer_decode_tmp_v, flashinfer_decode_tmp_s) = flashinfer_decode_scratch_maps(
            metadata,
            seqlen_offsets.len(),
            kv_cache,
            model_metadata,
            activation_dtype,
        )?;
        let fa3_decode = metadata
            .flashinfer
            .as_ref()
            .map(|flashinfer| {
                make_fa3_decode_state(
                    flashinfer,
                    seqlen_offsets.len(),
                    seq_len,
                    kv_cache,
                    model_metadata,
                    activation_dtype,
                )
            })
            .transpose()?
            .flatten();
        let flashinfer_views_alias = flashinfer_views_alias(metadata);
        let requirements = PagedDecodeMetadataRequirements::graph(
            metadata.block_tables.is_some(),
            metadata.context_lens.is_some(),
            metadata.flashinfer.is_some(),
            metadata.flashinfer.is_some(),
        );
        let mut buffers = Self {
            requirements,
            flashinfer_views_alias,
            slot_mappings,
            block_tables: option_var_map_from_tensor_map(metadata.block_tables.as_ref())?,
            context_lens: option_var_map_from_tensor_map(metadata.context_lens.as_ref())?,
            full_block_tables: option_var_map_from_tensor_map_if_distinct(
                metadata.full_block_tables.as_ref(),
                flashinfer_views_alias,
            )?,
            full_context_lens: option_var_map_from_tensor_map_if_distinct(
                metadata.full_context_lens.as_ref(),
                flashinfer_views_alias,
            )?,
            paged_kv_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
            )?,
            paged_kv_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
            )?,
            paged_kv_last_page_len: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
            )?,
            full_paged_kv_indptr: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                flashinfer_views_alias,
            )?,
            full_paged_kv_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_last_page_len: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
                flashinfer_views_alias,
            )?,
            paged_kv_request_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
            )?,
            paged_kv_tile_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
            )?,
            paged_kv_o_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.o_indptr),
            )?,
            paged_kv_chunk_size: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
            )?,
            paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
            )?,
            full_paged_kv_request_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_tile_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_o_indptr: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.o_indptr),
                flashinfer_views_alias,
            )?,
            full_paged_kv_chunk_size: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
                flashinfer_views_alias,
            )?,
            full_paged_kv_block_valid_mask: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
                flashinfer_views_alias,
            )?,
            flashinfer_decode_tmp_v,
            flashinfer_decode_tmp_s,
            fa3_decode,
            rope_positions,
        };
        if flashinfer_views_alias {
            buffers.full_block_tables = buffers.block_tables.clone();
            buffers.full_context_lens = buffers.context_lens.clone();
            buffers.full_paged_kv_indptr = buffers.paged_kv_indptr.clone();
            buffers.full_paged_kv_indices = buffers.paged_kv_indices.clone();
            buffers.full_paged_kv_last_page_len = buffers.paged_kv_last_page_len.clone();
            buffers.full_paged_kv_request_indices = buffers.paged_kv_request_indices.clone();
            buffers.full_paged_kv_tile_indices = buffers.paged_kv_tile_indices.clone();
            buffers.full_paged_kv_o_indptr = buffers.paged_kv_o_indptr.clone();
            buffers.full_paged_kv_chunk_size = buffers.paged_kv_chunk_size.clone();
            buffers.full_paged_kv_block_valid_mask = buffers.paged_kv_block_valid_mask.clone();
        }
        let metadata = buffers.metadata_from(metadata, block_size);
        Ok((buffers, metadata))
    }

    fn finish_capture(&mut self, metadata: &PagedAttentionInputMetadata) {
        let tile_plan_used = metadata
            .flashinfer
            .as_ref()
            .is_some_and(FlashInferMetadata::decode_tile_plan_was_used);
        self.requirements = PagedDecodeMetadataRequirements::graph(
            self.block_tables.is_some(),
            self.context_lens.is_some(),
            self.fa3_decode.is_some() || tile_plan_used,
            tile_plan_used,
        );
    }

    fn copy_from(
        &mut self,
        metadata: &PagedAttentionInputMetadata,
        position_ids: &[usize],
        seq_len: usize,
        host_staging: &mut CudaGraphHostStaging,
    ) -> candle_core::Result<()> {
        let graph_update = if metadata.has_host_staged_decode_tensors() {
            Some(
                metadata
                    .decode_rows
                    .as_ref()
                    .expect("host-staged decode metadata requires source rows")
                    .build_graph_update(self.requirements)
                    .map_err(candle_core::Error::msg)?,
            )
        } else {
            None
        };
        let metadata = graph_update.as_ref().unwrap_or(metadata);
        copy_var_map(
            &self.slot_mappings,
            &metadata.slot_mappings,
            "slot_mappings",
            host_staging,
        )?;
        if self.requirements.context_lens {
            copy_option_var_map(
                &self.context_lens,
                metadata.context_lens.as_ref(),
                "context_lens",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_context_lens,
                    metadata.full_context_lens.as_ref(),
                    "full_context_lens",
                    host_staging,
                )?;
            }
        }
        if self.requirements.block_tables {
            copy_option_var_map(
                &self.block_tables,
                metadata.block_tables.as_ref(),
                "block_tables",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_block_tables,
                    metadata.full_block_tables.as_ref(),
                    "full_block_tables",
                    host_staging,
                )?;
            }
        }
        if self.requirements.flashinfer_paged_kv {
            copy_option_var_map(
                &self.paged_kv_last_page_len,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
                "paged_kv_last_page_len",
                host_staging,
            )?;
            copy_option_var_map(
                &self.paged_kv_indptr,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
                "paged_kv_indptr",
                host_staging,
            )?;
            copy_option_var_map(
                &self.paged_kv_indices,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
                "paged_kv_indices",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_paged_kv_last_page_len,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
                    "full_paged_kv_last_page_len",
                    host_staging,
                )?;
                copy_option_var_map(
                    &self.full_paged_kv_indptr,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                    "full_paged_kv_indptr",
                    host_staging,
                )?;
                copy_option_var_map(
                    &self.full_paged_kv_indices,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                    "full_paged_kv_indices",
                    host_staging,
                )?;
            }
        }
        if self.requirements.flashinfer_tile_plan {
            copy_flashinfer_tile_plan(
                metadata,
                false,
                FlashInferTilePlanVars {
                    request_indices: &self.paged_kv_request_indices,
                    kv_tile_indices: &self.paged_kv_tile_indices,
                    o_indptr: &self.paged_kv_o_indptr,
                    kv_chunk_size: &self.paged_kv_chunk_size,
                    block_valid_mask: &self.paged_kv_block_valid_mask,
                },
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_flashinfer_tile_plan(
                    metadata,
                    true,
                    FlashInferTilePlanVars {
                        request_indices: &self.full_paged_kv_request_indices,
                        kv_tile_indices: &self.full_paged_kv_tile_indices,
                        o_indptr: &self.full_paged_kv_o_indptr,
                        kv_chunk_size: &self.full_paged_kv_chunk_size,
                        block_valid_mask: &self.full_paged_kv_block_valid_mask,
                    },
                    host_staging,
                )?;
            }
        }
        copy_rope_positions(&self.rope_positions, position_ids, seq_len, host_staging)?;
        Ok(())
    }

    fn flashinfer_metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> Option<FlashInferMetadata> {
        let original = metadata.flashinfer.as_ref()?;
        let logical = FlashInferPagedAttentionView {
            block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            max_context_len: original
                .views
                .logical
                .max_context_len
                .or_else(|| bucket_context_len_from_vars(&self.full_block_tables, block_size)),
            paged_kv: flashinfer_paged_kv_from_vars(
                &self.full_paged_kv_indptr,
                &self.full_paged_kv_indices,
                &self.full_paged_kv_last_page_len,
            )?,
            tile_plan: flashinfer_tile_plan_from_vars(
                &self.full_paged_kv_request_indices,
                &self.full_paged_kv_tile_indices,
                &self.full_paged_kv_o_indptr,
                &self.full_paged_kv_chunk_size,
                &self.full_paged_kv_block_valid_mask,
            )?,
        };
        let sliding = if let Some(view) = original.views.sliding.as_ref() {
            Some(FlashInferPagedAttentionView {
                block_tables: option_tensor_map_from_var_map(&self.block_tables),
                context_lens: option_tensor_map_from_var_map(&self.context_lens),
                max_context_len: view
                    .max_context_len
                    .or_else(|| bucket_context_len_from_vars(&self.block_tables, block_size)),
                paged_kv: flashinfer_paged_kv_from_vars(
                    &self.paged_kv_indptr,
                    &self.paged_kv_indices,
                    &self.paged_kv_last_page_len,
                )?,
                tile_plan: flashinfer_tile_plan_from_vars(
                    &self.paged_kv_request_indices,
                    &self.paged_kv_tile_indices,
                    &self.paged_kv_o_indptr,
                    &self.paged_kv_chunk_size,
                    &self.paged_kv_block_valid_mask,
                )?,
            })
        } else {
            None
        };

        Some(
            FlashInferMetadata {
                views: FlashInferPagedAttentionViews { logical, sliding },
                decode_tmp_v: self.flashinfer_decode_tmp_v.clone(),
                decode_tmp_s: self.flashinfer_decode_tmp_s.clone(),
                fa3_decode: self.fa3_decode.clone(),
                decode_tile_plan_used: None,
            }
            .track_decode_tile_plan(),
        )
    }

    fn metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> PagedAttentionInputMetadata {
        PagedAttentionInputMetadata {
            block_tables: option_tensor_map_from_var_map(&self.block_tables),
            context_lens: option_tensor_map_from_var_map(&self.context_lens),
            block_size: metadata.block_size,
            paged_context_lens_cpu: metadata.paged_context_lens_cpu.clone(),
            full_paged_context_lens_cpu: metadata.full_paged_context_lens_cpu.clone(),
            slot_mappings: tensor_map_from_var_map(&self.slot_mappings),
            max_context_len: graph_context_len(
                metadata.max_context_len,
                bucket_context_len_from_vars(&self.block_tables, block_size),
            ),
            full_block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            full_context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            full_max_context_len: graph_context_len(
                metadata.full_max_context_len,
                bucket_context_len_from_vars(&self.full_block_tables, block_size),
            ),
            is_first_prompt_chunk: metadata.is_first_prompt_chunk,
            is_final_prompt_chunk: metadata.is_final_prompt_chunk,
            prompt_chunk_attention_policy: metadata.prompt_chunk_attention_policy,
            has_noncausal_mm_context: metadata.has_noncausal_mm_context,
            prefix_gather_workspace_limit: metadata.prefix_gather_workspace_limit,
            mm_prefix_ranges: metadata.mm_prefix_ranges.clone(),
            full_mm_prefix_ranges: metadata.full_mm_prefix_ranges.clone(),
            prefill_attention_heads: metadata.prefill_attention_heads,
            prefill_key_value_heads: metadata.prefill_key_value_heads,
            prefill_head_dim: metadata.prefill_head_dim,
            flashinfer: self.flashinfer_metadata_from(metadata, block_size),
            rope_positions: Some(tensor_map_from_var_map(&self.rope_positions)),
            num_cached_tokens: metadata.num_cached_tokens.clone(),
            query_lens: metadata.query_lens.clone(),
            cu_seqlens_q: metadata.cu_seqlens_q.clone(),
            cu_seqlens_kv: metadata.cu_seqlens_kv.clone(),
            decode_rows: metadata.decode_rows.clone(),
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct CudaGraphSpecStateUsage {
    bytes: HashMap<DeviceLocation, usize>,
    device_totals: HashMap<DeviceLocation, usize>,
}

impl CudaGraphSpecStateUsage {
    fn from_state(state: &dyn SpeculativeGraphState) -> candle_core::Result<Self> {
        let mut usage = Self::default();
        for tensor in state.tensors() {
            let location = tensor.device().location();
            let bytes = tensor
                .elem_count()
                .saturating_mul(tensor.dtype().size_in_bytes());
            usage
                .bytes
                .entry(location)
                .and_modify(|total| *total = total.saturating_add(bytes))
                .or_insert(bytes);
            if let std::collections::hash_map::Entry::Vacant(entry) =
                usage.device_totals.entry(location)
            {
                let Device::Cuda(device) = tensor.device() else {
                    candle_core::bail!("CUDA graph speculative state expected CUDA tensors");
                };
                let (_, total) = device
                    .cuda_stream()
                    .context()
                    .mem_get_info()
                    .map_err(candle_core::Error::wrap)?;
                entry.insert(total);
            }
        }
        Ok(usage)
    }

    fn total_bytes(&self) -> usize {
        self.bytes
            .values()
            .fold(0usize, |total, bytes| total.saturating_add(*bytes))
    }
}

fn default_spec_state_budget(total: usize) -> usize {
    total.saturating_mul(CUDA_GRAPH_SPEC_STATE_BUDGET_PERCENT) / 100
}

fn configured_spec_state_budget(total: usize) -> usize {
    std::env::var(CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| default_spec_state_budget(total))
}

fn spec_state_eviction_plan(
    existing: &[CudaGraphSpecStateUsage],
    incoming: &CudaGraphSpecStateUsage,
    budgets: &HashMap<DeviceLocation, usize>,
) -> Vec<usize> {
    let mut totals = incoming.bytes.clone();
    for usage in existing {
        for (location, bytes) in &usage.bytes {
            totals
                .entry(*location)
                .and_modify(|total| *total = total.saturating_add(*bytes))
                .or_insert(*bytes);
        }
    }

    let mut retained = vec![true; existing.len()];
    let mut evictions = Vec::new();
    loop {
        let over_budget = totals
            .iter()
            .filter_map(|(location, bytes)| {
                (*bytes > budgets.get(location).copied().unwrap_or(usize::MAX)).then_some(*location)
            })
            .collect::<Vec<_>>();
        if over_budget.is_empty() {
            break;
        }
        let Some((idx, usage)) = existing.iter().enumerate().find(|(idx, usage)| {
            retained[*idx]
                && over_budget
                    .iter()
                    .any(|location| usage.bytes.get(location).is_some_and(|bytes| *bytes > 0))
        }) else {
            break;
        };
        retained[idx] = false;
        evictions.push(idx);
        for (location, bytes) in &usage.bytes {
            totals
                .entry(*location)
                .and_modify(|total| *total = total.saturating_sub(*bytes));
        }
    }
    evictions
}

pub(crate) struct CudaDecodeGraphEntry {
    generation: u64,
    replay_epoch: u64,
    key: CudaDecodeGraphKey,
    host_staging: CudaGraphHostStaging,
    input_ids: Var,
    metadata_buffers: CudaDecodeGraphMetadataBuffers,
    state_indices: Option<CudaGraphVarMap>,
    _metadata: PagedAttentionInputMetadata,
    logits: Tensor,
    // Proposer-facing outputs living in persistent buffers the replay refreshes
    spec_state: Option<Arc<dyn SpeculativeGraphState>>,
    spec_state_usage: CudaGraphSpecStateUsage,
    // Must stay last so graph-backed tensors enqueue their frees before the graph exec is destroyed.
    graph: CudaGraphHandle,
}

pub struct CudaDecodeGraphLaunch {
    generation: u64,
    replay_epoch: u64,
    key: CudaDecodeGraphKey,
    input_ids: Tensor,
    graph_stream: Arc<CudaStream>,
    real_batch: usize,
    source: CudaGraphDecodeStep,
}

impl CudaDecodeGraphLaunch {
    pub(crate) fn resident_input(&self) -> &Tensor {
        &self.input_ids
    }

    pub(crate) fn graph_stream(&self) -> &Arc<CudaStream> {
        &self.graph_stream
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn real_batch(&self) -> usize {
        self.real_batch
    }

    fn one_token_continuation(&self) -> candle_core::Result<Option<CudaGraphDecodeStep>> {
        let Some(continuation) = self.source.one_token_continuation(self.input_ids.clone())? else {
            return Ok(None);
        };
        let key = CudaDecodeGraphKey::new(
            &continuation.input_ids,
            &continuation.metadata,
            continuation
                .metadata
                .decode_rows
                .as_ref()
                .expect("continuation must retain decode rows")
                .block_size,
            self.key.recurrent_batch_kind,
        )?;
        Ok((key == self.key).then_some(continuation))
    }

    fn matches(&self, entry: &CudaDecodeGraphEntry) -> bool {
        cuda_graph_replay_version_matches(
            entry.generation,
            entry.replay_epoch,
            self.generation,
            self.replay_epoch,
        ) && self.key == entry.key
    }
}

fn cuda_graph_replay_version_matches(
    entry_generation: u64,
    entry_replay_epoch: u64,
    launch_generation: u64,
    launch_replay_epoch: u64,
) -> bool {
    entry_generation == launch_generation && entry_replay_epoch == launch_replay_epoch
}

impl fmt::Debug for CudaDecodeGraphLaunch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDecodeGraphLaunch")
            .field("generation", &self.generation)
            .field("replay_epoch", &self.replay_epoch)
            .field("input_shape", &self.input_ids.shape())
            .field("input_device", self.input_ids.device())
            .field("real_batch", &self.real_batch)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CudaDecodeGraphReplayInput<'a> {
    Host,
    Resident(&'a CudaDecodeGraphLaunch),
}

impl CudaDecodeGraphEntry {
    pub(crate) fn with_spec_state(
        mut self,
        spec_state: Option<Box<dyn SpeculativeGraphState>>,
        usage: Option<CudaGraphSpecStateUsage>,
    ) -> Self {
        assert_eq!(spec_state.is_some(), usage.is_some());
        self.spec_state = spec_state.map(Arc::from);
        self.spec_state_usage = usage.unwrap_or_default();
        self
    }

    fn launch(
        &self,
        step: &CudaGraphDecodeStep,
        replay_epoch: u64,
    ) -> candle_core::Result<Option<CudaDecodeGraphLaunch>> {
        let input_ids = self.input_ids.as_detached_tensor();
        let (batch, q_len) = input_ids.dims2()?;
        if input_ids.dtype() != DType::U32
            || q_len != 1
            || !input_ids.is_contiguous()
            || self.spec_state.is_some()
        {
            return Ok(None);
        }
        if step.real_batch > batch {
            candle_core::bail!(
                "CUDA graph resident input has batch capacity {batch}, smaller than {} live rows",
                step.real_batch
            );
        }
        Ok(Some(CudaDecodeGraphLaunch {
            generation: self.generation,
            replay_epoch,
            key: self.key.clone(),
            input_ids,
            graph_stream: self.graph.stream.clone(),
            real_batch: step.real_batch,
            source: step.clone(),
        }))
    }

    fn release(self) -> (Arc<CudaStream>, candle_core::Result<()>) {
        let Self {
            generation: _,
            replay_epoch,
            key: _,
            host_staging,
            input_ids,
            metadata_buffers,
            state_indices,
            _metadata,
            logits,
            spec_state,
            spec_state_usage: _,
            graph,
        } = self;
        let stream = graph.stream.clone();
        let mut release_result = stream
            .synchronize()
            .map_err(candle_core::Error::wrap)
            .map_err(|err| err.context("CUDA graph entry release wait failed"));
        drop_cuda_graph_entry_resource(host_staging, &stream, "host staging", &mut release_result);
        drop_cuda_graph_entry_resource(
            spec_state,
            &stream,
            "speculative state",
            &mut release_result,
        );
        drop_cuda_graph_entry_logits(logits, &stream, replay_epoch, &mut release_result);
        drop_cuda_graph_entry_resource(
            _metadata,
            &stream,
            "paged-attention metadata",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(
            state_indices,
            &stream,
            "state indices",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(
            metadata_buffers,
            &stream,
            "metadata buffers",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(input_ids, &stream, "input ids", &mut release_result);
        let storage_result = stream
            .synchronize()
            .map_err(candle_core::Error::wrap)
            .map_err(|err| err.context("CUDA graph entry storage release failed"));
        if release_result.is_ok() {
            release_result = storage_result;
        }
        drop(graph);
        (stream, release_result)
    }
}

fn drop_cuda_graph_entry_logits(
    logits: Tensor,
    stream: &Arc<CudaStream>,
    replay_epoch: u64,
    release_result: &mut candle_core::Result<()>,
) {
    drop(logits);
    if let Err(err) = stream.context().check_err() {
        // Alloc nodes have no materialized allocation before first launch, but CudaSlice::drop still frees them.
        if replay_epoch == 0 && err.0 == sys::CUresult::CUDA_ERROR_INVALID_VALUE {
            return;
        }
        if release_result.is_ok() {
            *release_result = Err(
                candle_core::Error::wrap(err).context("CUDA graph entry logits release failed")
            );
        }
    }
}

fn drop_cuda_graph_entry_resource<T>(
    resource: T,
    stream: &Arc<CudaStream>,
    name: &'static str,
    release_result: &mut candle_core::Result<()>,
) {
    drop(resource);
    if let Err(err) = stream.context().check_err() {
        if release_result.is_ok() {
            *release_result = Err(candle_core::Error::wrap(err)
                .context(format!("CUDA graph entry {name} release failed")));
        }
    }
}

pub(crate) struct CudaDecodeGraphReplay {
    pub(crate) logits: Tensor,
    pub(crate) spec_state: Option<Arc<dyn SpeculativeGraphState>>,
    pub(crate) launch: Option<CudaDecodeGraphLaunch>,
}

#[derive(Default)]
pub(crate) struct CudaDecodeGraphState {
    entries: Vec<CudaDecodeGraphEntry>,
    spec_state_budgets: HashMap<DeviceLocation, usize>,
    disabled: bool,
    suspended: bool,
    eager_retry_blocked: bool,
    recurrent_storage_generation: Option<u64>,
}

impl Drop for CudaDecodeGraphState {
    fn drop(&mut self) {
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, 0);
    }
}

impl CudaDecodeGraphState {
    pub(crate) fn disabled(&self) -> bool {
        self.disabled || self.suspended
    }

    pub(crate) fn disable(&mut self) {
        self.disabled = true;
        self.clear();
    }

    pub(crate) fn take_eager_retry_allowed(&mut self) -> bool {
        !std::mem::take(&mut self.eager_retry_blocked)
    }

    pub(crate) fn block_eager_retry(&mut self) {
        self.eager_retry_blocked = true;
    }

    pub(crate) fn clear(&mut self) {
        self.eager_retry_blocked = false;
        let entries = std::mem::take(&mut self.entries);
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, 0);
        release_cuda_graph_entries(entries);
    }

    pub(crate) fn evict_lru_for_memory_pressure(&mut self, max_entries: usize) -> usize {
        let entries = drain_lru_entries(&mut self.entries, max_entries);
        let evicted = entries.len();
        if evicted == 0 {
            return 0;
        }
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
        release_cuda_graph_entries(entries);
        record_cuda_graph_evictions(
            CudaGraphComponent::Target,
            CudaGraphEvictionReason::MemoryPressure,
            evicted,
        );
        evicted
    }

    pub(crate) fn observe_recurrent_storage_generation(&mut self, generation: u64) {
        let previous = self.recurrent_storage_generation.replace(generation);
        if previous.is_some_and(|previous| previous != generation) {
            self.clear();
        }
    }

    pub(crate) fn suspend(&mut self) {
        self.suspended = true;
        self.clear();
    }

    pub(crate) fn resume(&mut self) {
        self.suspended = false;
        self.clear();
    }

    pub(crate) fn contains(&self, key: &CudaDecodeGraphKey) -> bool {
        self.entries.iter().any(|entry| entry.key == *key)
    }

    pub(crate) fn replay(
        &mut self,
        key: &CudaDecodeGraphKey,
        step: &CudaGraphDecodeStep,
        input: CudaDecodeGraphReplayInput<'_>,
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        let Some(pos) = self.entries.iter().position(|entry| entry.key == *key) else {
            return Ok(None);
        };
        let mut entry = self.entries.remove(pos);
        if let CudaDecodeGraphReplayInput::Resident(launch) = input {
            if !launch.matches(&entry) || launch.real_batch != step.real_batch {
                self.entries.push(entry);
                return Ok(None);
            }
        }
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::Target, CudaGraphEvent::Replay);
        let prelaunch = (|| -> candle_core::Result<_> {
            match input {
                CudaDecodeGraphReplayInput::Host => {
                    entry.input_ids.set(&step.input_ids).map_err(|err| {
                        err.context(format!(
                            "CUDA graph input update failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                            entry.generation, entry.replay_epoch, step.real_batch, entry.key
                        ))
                    })?
                }
                CudaDecodeGraphReplayInput::Resident(_) => {}
            }
            let replay_epoch = entry
                .replay_epoch
                .checked_add(1)
                .expect("CUDA decode graph replay epoch overflow");
            let spec_state = entry
                .spec_state
                .as_deref()
                .map(|state| state.for_real_batch(step.real_batch))
                .transpose()?
                .map(Arc::from);
            let replay = CudaDecodeGraphReplay {
                logits: step.narrow_rows(&entry.logits)?,
                spec_state,
                launch: entry.launch(step, replay_epoch)?,
            };
            let (_, seq_len) = step.input_ids.dims2()?;
            let metadata_buffers = &mut entry.metadata_buffers;
            let state_indices = &entry.state_indices;
            entry
                .host_staging
                .update(|host_staging| {
                    metadata_buffers.copy_from(
                        &step.metadata,
                        &step.position_ids,
                        seq_len,
                        host_staging,
                    )?;
                    match (state_indices, &step.state_indices) {
                        (Some(dst), Some(host)) => copy_state_indices(dst, host, host_staging),
                        (None, None) => Ok(()),
                        _ => candle_core::bail!(
                            "hybrid state indices changed optional state during CUDA graph replay"
                        ),
                    }
                })
                .map_err(|err| {
                    err.context(format!(
                        "CUDA graph metadata update failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                        entry.generation, entry.replay_epoch, step.real_batch, entry.key
                    ))
                })?;
            entry.host_staging.order_before_graph().map_err(|err| {
                err.context(format!(
                    "CUDA graph metadata ordering failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                    entry.generation, entry.replay_epoch, step.real_batch, entry.key
                ))
            })?;
            Ok(Some((replay_epoch, replay)))
        })();
        let (replay_epoch, mut replay) = match prelaunch {
            Ok(Some(prelaunch)) => prelaunch,
            Ok(None) => {
                self.entries.push(entry);
                return Ok(None);
            }
            Err(err) => {
                self.entries.push(entry);
                return Err(err);
            }
        };
        if let Err(err) = entry.graph.launch().map_err(|err| {
            err.context(format!(
                "CUDA graph replay launch failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                entry.generation, entry.replay_epoch, step.real_batch, entry.key
            ))
        }) {
            self.disabled = true;
            self.block_eager_retry();
            self.entries.push(entry);
            return Err(err);
        }
        if let Err(record_err) = entry.host_staging.record_graph_complete().map_err(|err| {
            err.context(format!(
                "CUDA graph completion recording failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                entry.generation, entry.replay_epoch, step.real_batch, entry.key
            ))
        }) {
            let synchronize_result = entry
                .graph
                .stream()
                .synchronize()
                .map_err(candle_core::Error::wrap)
                .map_err(|err| err.context("CUDA graph replay recovery synchronization failed"));
            entry.replay_epoch = replay_epoch;
            replay.launch = None;
            self.disabled = true;
            self.block_eager_retry();
            self.entries.push(entry);
            return match synchronize_result {
                Ok(()) => {
                    tracing::warn!(
                        "CUDA decode graphs retired after completion recording error: {record_err:?}"
                    );
                    record_cuda_graph_dispatch(
                        CudaGraphComponent::Target,
                        CudaGraphDispatchMode::Replay,
                        CudaGraphDispatchReason::CacheHit,
                    );
                    Ok(Some(replay))
                }
                Err(synchronize_err) => {
                    tracing::warn!(
                        "CUDA decode graph completion recording and recovery synchronization failed: {record_err:?}; {synchronize_err:?}"
                    );
                    Err(candle_core::Error::msg(format!(
                        "{record_err}; CUDA graph state may have advanced and recovery failed: {synchronize_err}"
                    )))
                }
            };
        }
        entry.replay_epoch = replay_epoch;
        self.entries.push(entry);
        graph_event.success();
        record_cuda_graph_dispatch(
            CudaGraphComponent::Target,
            CudaGraphDispatchMode::Replay,
            CudaGraphDispatchReason::CacheHit,
        );
        Ok(Some(replay))
    }

    pub(crate) fn replay_one_token(
        &mut self,
        launch: CudaDecodeGraphLaunch,
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        let Some(step) = launch.one_token_continuation()? else {
            return Ok(None);
        };
        self.replay(
            &launch.key,
            &step,
            CudaDecodeGraphReplayInput::Resident(&launch),
        )
    }

    pub(crate) fn prepare_spec_state_admission(
        &mut self,
        spec_state: &dyn SpeculativeGraphState,
    ) -> candle_core::Result<CudaGraphSpecStateUsage> {
        let usage = CudaGraphSpecStateUsage::from_state(spec_state)?;
        self.evict_for_spec_state(&usage);
        Ok(usage)
    }

    pub(crate) fn insert(&mut self, mut entry: CudaDecodeGraphEntry) {
        self.evict_for_spec_state(&entry.spec_state_usage);
        if let Some(evicted) = take_cuda_graph_capacity_eviction(
            &mut self.entries,
            TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY,
        ) {
            record_cuda_graph_evictions(
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::Capacity,
                1,
            );
            release_cuda_graph_entries(vec![evicted]);
        }
        entry.generation = self.allocate_generation();
        self.entries.push(entry);
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
    }

    fn evict_for_spec_state(&mut self, incoming: &CudaGraphSpecStateUsage) {
        for (location, total) in &incoming.device_totals {
            self.spec_state_budgets
                .entry(*location)
                .or_insert_with(|| configured_spec_state_budget(*total));
        }
        let usages = self
            .entries
            .iter()
            .map(|entry| entry.spec_state_usage.clone())
            .collect::<Vec<_>>();
        let mut evictions = spec_state_eviction_plan(&usages, incoming, &self.spec_state_budgets);
        if !evictions.is_empty() {
            let evicted = evictions.len();
            tracing::debug!(
                entries = evicted,
                incoming_bytes = incoming.total_bytes(),
                "Evicting CUDA graphs to stay within the speculative state budget"
            );
            evictions.sort_unstable();
            let mut entries = Vec::with_capacity(evictions.len());
            for idx in evictions.into_iter().rev() {
                entries.push(self.entries.remove(idx));
            }
            record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
            release_cuda_graph_entries(entries);
            record_cuda_graph_evictions(
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::SpecStateBudget,
                evicted,
            );
        }
    }

    fn allocate_generation(&mut self) -> u64 {
        NEXT_CUDA_DECODE_GRAPH_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .expect("CUDA decode graph generation overflow")
    }
}

fn drain_lru_entries<T>(entries: &mut Vec<T>, max_entries: usize) -> Vec<T> {
    let count = max_entries.min(entries.len());
    entries.drain(..count).collect()
}

fn release_cuda_graph_entries(entries: Vec<CudaDecodeGraphEntry>) {
    let mut streams = Vec::new();
    for entry in entries {
        let (stream, release_result) = entry.release();
        if let Err(err) = release_result {
            tracing::warn!("Failed to release CUDA graph entry storage: {err:?}");
        }
        if !streams.iter().any(|known: &Arc<CudaStream>| {
            known.context().cu_device() == stream.context().cu_device()
        }) {
            streams.push(stream);
        }
    }
    for stream in streams {
        if let Err(err) = trim_cuda_graph_memory(&stream) {
            tracing::warn!("Failed to trim released CUDA graph memory: {err:?}");
        }
    }
}

pub(crate) fn capture_cuda_decode_graph<F>(
    ctx: CudaDecodeGraphCaptureCtx<'_>,
    forward: F,
) -> candle_core::Result<CudaDecodeGraphEntry>
where
    F: FnOnce(&Tensor, &PagedAttentionInputMetadata) -> candle_core::Result<Tensor>,
{
    let CudaDecodeGraphCaptureCtx {
        key,
        input_ids,
        seqlen_offsets,
        position_ids,
        block_size,
        kv_cache,
        metadata,
        model_metadata,
        activation_dtype,
        warmup_logits,
        state_indices,
        real_batch,
    } = ctx;
    let materialized_metadata = metadata
        .materialize_decode_tensors()
        .map_err(candle_core::Error::msg)?;
    let metadata = &materialized_metadata;
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids = Var::from_tensor(input_ids)?;
    let (mut metadata_buffers, metadata) = CudaDecodeGraphMetadataBuffers::new(
        metadata,
        seqlen_offsets,
        position_ids,
        seq_len,
        block_size,
        kv_cache,
        model_metadata,
        activation_dtype,
    )?;
    let graph_input_ids = input_ids.as_detached_tensor();
    let Device::Cuda(cuda_device) = graph_input_ids.device() else {
        candle_core::bail!("CUDA graph decode expected CUDA input ids");
    };
    graph_input_ids.device().synchronize()?;
    let stream = cuda_device.cuda_stream();
    let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();

    if let Err(err) = stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
    {
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(
            candle_core::Error::msg(err.to_string()).context("CUDA graph begin capture failed")
        );
    }

    if let Err(err) = prepare_fa3_decode_schedules(&metadata) {
        end_cuda_capture_discard(&stream);
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(err.context("FA3 decode preparation capture failed"));
    }

    let logits = match forward(&graph_input_ids, &metadata) {
        Ok(logits) => logits,
        Err(err) => {
            end_cuda_capture_discard(&stream);
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(err.context("CUDA graph captured forward failed"));
        }
    };
    if logits.shape() != warmup_logits.shape()
        || logits.dtype() != warmup_logits.dtype()
        || logits.device().location() != warmup_logits.device().location()
        || !logits.is_contiguous()
    {
        end_cuda_capture_discard(&stream);
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(candle_core::Error::msg(
            "captured CUDA graph logits do not match the contiguous warmup output",
        ));
    }

    let graph = match CudaGraphHandle::end_capture(&stream) {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(candle_core::Error::msg(
                "CUDA graph capture returned no graph",
            ));
        }
        Err(err) => {
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(err);
        }
    };
    restore_event_tracking_after_capture(&stream, restore_event_tracking);

    graph.upload()?;
    metadata_buffers.finish_capture(&metadata);
    let host_staging = CudaGraphHostStaging::new(graph.stream.clone())?;
    tracing::debug!(
        "Captured CUDA decode graph: batch bucket {batch} ({real_batch} live rows), {seq_len} query tokens"
    );

    Ok(CudaDecodeGraphEntry {
        generation: 0,
        replay_epoch: 0,
        key,
        host_staging,
        input_ids,
        metadata_buffers,
        state_indices,
        _metadata: metadata,
        logits,
        spec_state: None,
        spec_state_usage: CudaGraphSpecStateUsage::default(),
        graph,
    })
}

pub(crate) fn cuda_decode_graphs_enabled() -> bool {
    crate::perf_flags::cuda_graphs_enabled()
}

pub(crate) fn cuda_decode_graph_batch_kind_supported(kind: RecurrentBatchKind) -> bool {
    matches!(
        kind,
        RecurrentBatchKind::Decode | RecurrentBatchKind::SpeculativeDecode
    )
}

pub(crate) fn cuda_decode_graph_supported_for_model(
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
) -> bool {
    let Some(metadata) = model_metadata else {
        return false;
    };
    #[cfg(target_family = "unix")]
    {
        (0..metadata.num_layers()).all(|layer_idx| {
            !DecodePlan::requires_host_context_lengths(
                metadata.attention_backend_kind_for_layer(layer_idx),
                metadata.k_head_dim_for_layer(layer_idx),
            )
        })
    }
    #[cfg(not(target_family = "unix"))]
    {
        (0..metadata.num_layers()).all(|layer_idx| {
            !matches!(
                metadata.attention_backend_kind_for_layer(layer_idx),
                AttentionBackendKind::FlashInfer
            )
        })
    }
}

#[must_use]
pub(crate) struct CudaGraphMemoryPoolGuard {
    stream: Arc<CudaStream>,
    pool: Option<usize>,
}

impl Drop for CudaGraphMemoryPoolGuard {
    fn drop(&mut self) {
        let Some(pool) = self.pool else { return };
        if let Err(err) = self.stream.context().bind_to_thread() {
            tracing::warn!("Failed to bind CUDA context while restoring graph memory pool: {err}");
        }
        let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
        let mut scopes = scopes
            .lock()
            .expect("CUDA graph memory pool scopes poisoned");
        let Some(scope) = scopes.get_mut(&pool) else {
            tracing::warn!("CUDA graph memory pool scope disappeared before restoration");
            return;
        };
        scope.guards = scope
            .guards
            .checked_sub(1)
            .expect("CUDA graph memory pool guard underflow");
        if scope.guards != 0 {
            return;
        }
        let release_threshold = scope.release_threshold;
        let pool_ptr = pool as sys::CUmemoryPool;
        if let Err(err) = set_memory_pool_release_threshold(pool_ptr, release_threshold) {
            tracing::warn!("Failed to restore CUDA graph memory pool threshold: {err:?}");
        }
        let _ = self.stream.synchronize();
        if let Err(err) = trim_cuda_graph_memory_bound(&self.stream) {
            tracing::warn!("Failed to trim CUDA graph memory after capture: {err:?}");
        }
        scopes.remove(&pool);
    }
}

fn memory_pool_release_threshold(pool: sys::CUmemoryPool) -> candle_core::Result<u64> {
    let mut value = 0u64;
    let result = unsafe {
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut value as *mut u64).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool release threshold lookup failed"));
    }
    Ok(value)
}

fn set_memory_pool_release_threshold(
    pool: sys::CUmemoryPool,
    mut value: u64,
) -> candle_core::Result<()> {
    let result = unsafe {
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut value as *mut u64).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool release threshold setup failed"));
    }
    Ok(())
}

fn cuda_memory_pool(stream: &Arc<CudaStream>) -> candle_core::Result<sys::CUmemoryPool> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let mut pool = std::ptr::null_mut();
    let result = unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool lookup failed"));
    }
    Ok(pool)
}

pub(crate) fn cuda_graph_memory_pool_scope_active(device: &Device) -> candle_core::Result<bool> {
    let Device::Cuda(device) = device else {
        return Ok(false);
    };
    let stream = device.cuda_stream();
    if !stream.context().has_async_alloc() {
        return Ok(false);
    }
    let pool = cuda_memory_pool(&stream)? as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    Ok(scopes.contains_key(&pool))
}

fn trim_cuda_graph_memory_bound(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
    let result = unsafe { sys::cuDeviceGraphMemTrim(stream.context().cu_device()) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(
            candle_core::Error::msg(format!("{result:?}")).context("CUDA graph memory trim failed")
        );
    }
    Ok(())
}

#[cfg(test)]
fn cuda_graph_memory_attribute(
    stream: &Arc<CudaStream>,
    attribute: sys::CUgraphMem_attribute,
) -> candle_core::Result<usize> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let mut value = 0usize;
    let result = unsafe {
        sys::cuDeviceGetGraphMemAttribute(
            stream.context().cu_device(),
            attribute,
            (&mut value as *mut usize).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph memory attribute lookup failed"));
    }
    Ok(value)
}

pub(crate) fn trim_cuda_graph_memory(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("CUDA graph memory trim context bind failed"))?;
    let pool = cuda_memory_pool(stream)? as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    if scopes.contains_key(&pool) {
        return Ok(());
    }
    stream
        .synchronize()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("CUDA graph memory trim synchronization failed"))?;
    trim_cuda_graph_memory_bound(stream)
}

pub(crate) fn prepare_cuda_graph_memory_pool(
    stream: &Arc<CudaStream>,
) -> candle_core::Result<CudaGraphMemoryPoolGuard> {
    if !stream.context().has_async_alloc() {
        return Ok(CudaGraphMemoryPoolGuard {
            stream: stream.clone(),
            pool: None,
        });
    }

    let pool = cuda_memory_pool(stream)?;
    let pool_key = pool as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let mut scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    if let Some(scope) = scopes.get_mut(&pool_key) {
        scope.guards = scope
            .guards
            .checked_add(1)
            .expect("CUDA graph memory pool guard overflow");
    } else {
        let release_threshold = memory_pool_release_threshold(pool)?;
        set_memory_pool_release_threshold(pool, u64::MAX)?;
        scopes.insert(
            pool_key,
            MemoryPoolScopeState {
                guards: 1,
                release_threshold,
            },
        );
    }
    drop(scopes);

    let guard = CudaGraphMemoryPoolGuard {
        stream: stream.clone(),
        pool: Some(pool_key),
    };
    for attr in [
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
    ] {
        let mut enabled = 1i32;
        let result =
            unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut enabled as *mut i32).cast()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph mempool reuse setup failed"));
        }
    }

    Ok(guard)
}

fn flashinfer_decode_scratch_maps(
    metadata: &PagedAttentionInputMetadata,
    batch: usize,
    kv_cache: &[(Tensor, Tensor)],
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
    activation_dtype: DType,
) -> candle_core::Result<FlashInferDecodeScratchMaps> {
    let Some(model_metadata) = model_metadata else {
        return Ok((None, None));
    };
    let split_rows = flashinfer_split_rows(metadata, batch)?;
    if split_rows.is_empty() {
        return Ok((None, None));
    }

    let mut specs: HashMap<DeviceLocation, (Device, DType, usize, usize)> = HashMap::new();
    let layer_count = model_metadata.num_layers().min(kv_cache.len());
    for (layer_idx, (key_cache, value_cache)) in kv_cache.iter().enumerate().take(layer_count) {
        if model_metadata.attention_backend_kind_for_layer(layer_idx)
            != AttentionBackendKind::FlashInfer
        {
            continue;
        }
        let location = key_cache.device().location();
        if !split_rows.contains_key(&location) {
            continue;
        }
        if key_cache.dtype() != value_cache.dtype() {
            candle_core::bail!("FlashInfer graph scratch expects matching KV cache dtypes");
        }
        let (_, _, _, head_dim) = key_cache.dims4()?;
        let num_qo_heads = model_metadata.num_attn_heads_for_layer(layer_idx);
        let entry = specs.entry(location).or_insert((
            key_cache.device().clone(),
            activation_dtype,
            num_qo_heads,
            head_dim,
        ));
        if entry.1 != activation_dtype {
            candle_core::bail!("FlashInfer graph scratch expects one activation dtype per device");
        }
        entry.2 = entry.2.max(num_qo_heads);
        entry.3 = entry.3.max(head_dim);
    }

    let mut tmp_v = HashMap::new();
    let mut tmp_s = HashMap::new();
    for (location, rows) in split_rows {
        let Some((device, dtype, num_qo_heads, head_dim)) = specs.get(&location) else {
            continue;
        };
        tmp_v.insert(location, unsafe {
            Tensor::empty((rows, *num_qo_heads, *head_dim), *dtype, device)?
        });
        tmp_s.insert(location, unsafe {
            Tensor::empty((rows, *num_qo_heads), DType::F32, device)?
        });
    }

    if tmp_v.is_empty() {
        Ok((None, None))
    } else {
        Ok((Some(tmp_v), Some(tmp_s)))
    }
}

fn flashinfer_split_rows(
    metadata: &PagedAttentionInputMetadata,
    batch: usize,
) -> candle_core::Result<HashMap<DeviceLocation, usize>> {
    let mut rows = HashMap::new();
    collect_flashinfer_split_rows(
        flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
        batch,
        &mut rows,
    )?;
    collect_flashinfer_split_rows(
        flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
        batch,
        &mut rows,
    )?;
    Ok(rows)
}

pub(crate) fn disable_event_tracking_for_capture(stream: &Arc<CudaStream>) -> bool {
    let restore = stream.context().is_event_tracking();
    if restore {
        unsafe { stream.context().disable_event_tracking() };
    }
    restore
}

pub(crate) fn restore_event_tracking_after_capture(stream: &Arc<CudaStream>, restore: bool) {
    if restore {
        unsafe { stream.context().enable_event_tracking() };
    }
}

pub(crate) fn end_cuda_capture_discard(stream: &Arc<CudaStream>) {
    if matches!(
        stream.capture_status(),
        Ok(status) if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    ) {
        let mut graph = std::ptr::null_mut();
        let result = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        if result == sys::CUresult::CUDA_SUCCESS && !graph.is_null() {
            let _ = unsafe { sys::cuGraphDestroy(graph) };
        }
    }
}

fn device_location_sort_key(location: &DeviceLocation) -> (u8, usize) {
    match location {
        DeviceLocation::Cpu => (0, 0),
        DeviceLocation::Cuda { gpu_id } => (1, *gpu_id),
        DeviceLocation::Metal { gpu_id } => (2, *gpu_id),
    }
}

fn push_graph_tensor_keys(
    name: &'static str,
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    keys: &mut Vec<CudaGraphTensorKey>,
) {
    if let Some(map) = map {
        keys.extend(map.iter().map(|(location, tensor)| CudaGraphTensorKey {
            name,
            location: *location,
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
        }));
    }
}

fn push_flashinfer_graph_tensor_keys(
    metadata: &PagedAttentionInputMetadata,
    keys: &mut Vec<CudaGraphTensorKey>,
) {
    let paged = flashinfer_paged_view(metadata);
    let full = flashinfer_full_view(metadata);
    push_graph_tensor_keys(
        "paged_kv_indptr",
        paged.map(|view| &view.paged_kv.indptr),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_indices",
        paged.map(|view| &view.paged_kv.indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_last_page_len",
        paged.map(|view| &view.paged_kv.last_page_len),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_indptr",
        full.map(|view| &view.paged_kv.indptr),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_indices",
        full.map(|view| &view.paged_kv.indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_last_page_len",
        full.map(|view| &view.paged_kv.last_page_len),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_request_indices",
        paged.map(|view| &view.tile_plan.request_indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_tile_indices",
        paged.map(|view| &view.tile_plan.kv_tile_indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_o_indptr",
        paged.map(|view| &view.tile_plan.o_indptr),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_chunk_size",
        paged.map(|view| &view.tile_plan.kv_chunk_size),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_block_valid_mask",
        paged.map(|view| &view.tile_plan.block_valid_mask),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_request_indices",
        full.map(|view| &view.tile_plan.request_indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_tile_indices",
        full.map(|view| &view.tile_plan.kv_tile_indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_o_indptr",
        full.map(|view| &view.tile_plan.o_indptr),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_chunk_size",
        full.map(|view| &view.tile_plan.kv_chunk_size),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_block_valid_mask",
        full.map(|view| &view.tile_plan.block_valid_mask),
        keys,
    );
}

fn flashinfer_paged_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    let views = &metadata.flashinfer.as_ref()?.views;
    Some(views.sliding.as_ref().unwrap_or(&views.logical))
}

fn flashinfer_full_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    Some(&metadata.flashinfer.as_ref()?.views.logical)
}

fn flashinfer_views_alias(metadata: &PagedAttentionInputMetadata) -> bool {
    metadata
        .flashinfer
        .as_ref()
        .is_some_and(|flashinfer| flashinfer.views.sliding.is_none())
}

fn flashinfer_paged_kv_from_vars(
    indptr: &Option<CudaGraphVarMap>,
    indices: &Option<CudaGraphVarMap>,
    last_page_len: &Option<CudaGraphVarMap>,
) -> Option<FlashInferPagedKv> {
    Some(FlashInferPagedKv {
        indptr: option_tensor_map_from_var_map(indptr)?,
        indices: option_tensor_map_from_var_map(indices)?,
        last_page_len: option_tensor_map_from_var_map(last_page_len)?,
    })
}

fn flashinfer_tile_plan_from_vars(
    request_indices: &Option<CudaGraphVarMap>,
    kv_tile_indices: &Option<CudaGraphVarMap>,
    o_indptr: &Option<CudaGraphVarMap>,
    kv_chunk_size: &Option<CudaGraphVarMap>,
    block_valid_mask: &Option<CudaGraphVarMap>,
) -> Option<FlashInferTilePlan> {
    Some(FlashInferTilePlan {
        request_indices: option_tensor_map_from_var_map(request_indices)?,
        kv_tile_indices: option_tensor_map_from_var_map(kv_tile_indices)?,
        o_indptr: option_tensor_map_from_var_map(o_indptr)?,
        kv_chunk_size: option_tensor_map_from_var_map(kv_chunk_size)?,
        block_valid_mask: option_tensor_map_from_var_map(block_valid_mask)?,
    })
}

fn collect_flashinfer_split_rows(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    batch: usize,
    split_rows: &mut HashMap<DeviceLocation, usize>,
) -> candle_core::Result<()> {
    let Some(map) = map else {
        return Ok(());
    };
    for (location, tensor) in map {
        let rows = tensor.dims1()?;
        if rows > batch {
            split_rows
                .entry(*location)
                .and_modify(|current| *current = (*current).max(rows))
                .or_insert(rows);
        }
    }
    Ok(())
}

fn bucket_context_len_from_vars(map: &Option<CudaGraphVarMap>, block_size: usize) -> Option<usize> {
    map.as_ref()
        .and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

fn bucket_context_len(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    block_size: usize,
) -> Option<usize> {
    map.and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

fn graph_context_len(actual: Option<usize>, capacity: Option<usize>) -> Option<usize> {
    match (actual, capacity) {
        (Some(actual), Some(capacity)) => Some(
            actual
                .div_ceil(PAGED_ATTENTION_PARTITION_SIZE)
                .max(1)
                .saturating_mul(PAGED_ATTENTION_PARTITION_SIZE)
                .min(capacity),
        ),
        (Some(actual), None) => Some(actual),
        (None, capacity) => capacity,
    }
}

fn var_map_from_tensor_map(
    map: &HashMap<DeviceLocation, Tensor>,
) -> candle_core::Result<CudaGraphVarMap> {
    map.iter()
        .map(|(location, tensor)| Ok((*location, Var::from_tensor(tensor)?)))
        .collect()
}

fn option_var_map_from_tensor_map(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
) -> candle_core::Result<Option<CudaGraphVarMap>> {
    map.map(var_map_from_tensor_map).transpose()
}

fn option_var_map_from_tensor_map_if_distinct(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    aliases_existing: bool,
) -> candle_core::Result<Option<CudaGraphVarMap>> {
    if aliases_existing {
        Ok(None)
    } else {
        option_var_map_from_tensor_map(map)
    }
}

fn tensor_map_from_var_map(map: &CudaGraphVarMap) -> HashMap<DeviceLocation, Tensor> {
    map.iter()
        .map(|(location, var)| (*location, var.as_detached_tensor()))
        .collect()
}

fn option_tensor_map_from_var_map(
    map: &Option<CudaGraphVarMap>,
) -> Option<HashMap<DeviceLocation, Tensor>> {
    map.as_ref().map(tensor_map_from_var_map)
}

impl CudaGraphPinnedBuffer {
    fn new(dst: &Var) -> candle_core::Result<Self> {
        let Device::Cuda(device) = dst.device() else {
            candle_core::bail!("CUDA graph host staging requires a CUDA destination");
        };
        let stream = device.cuda_stream();
        let context = stream.context();
        let len = dst.elem_count();
        macro_rules! allocate {
            ($variant:ident, $ty:ty) => {{
                let mut allocation = unsafe {
                    context
                        .alloc_pinned::<$ty>(len)
                        .map_err(candle_core::Error::wrap)?
                };
                let ptr = NonNull::new(allocation.as_mut_ptr().map_err(candle_core::Error::wrap)?)
                    .ok_or_else(|| {
                        candle_core::Error::msg("CUDA returned a null pinned pointer")
                    })?;
                CudaGraphPinnedData::$variant(CudaGraphPinnedAllocation { allocation, ptr })
            }};
        }
        let data = match dst.dtype() {
            DType::U8 => allocate!(U8, u8),
            DType::U32 => allocate!(U32, u32),
            DType::I32 => allocate!(I32, i32),
            DType::I64 => allocate!(I64, i64),
            DType::F32 => allocate!(F32, f32),
            dtype => candle_core::bail!(
                "CUDA graph host staging does not support metadata dtype {dtype:?}"
            ),
        };
        Ok(Self {
            data,
            initialized: false,
        })
    }

    fn copy_from(
        &mut self,
        src: &Tensor,
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if src.shape() != dst.shape() || src.dtype() != dst.dtype() {
            candle_core::bail!("CUDA graph host staging expected matching tensors");
        }
        let (src_storage, src_layout) = src.storage_and_layout();
        let Storage::Cpu(src_storage) = &*src_storage else {
            candle_core::bail!("CUDA graph host staging expected CPU source metadata");
        };
        if !src_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous source metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA destination metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous destination metadata");
        }
        let len = src.elem_count();
        let src_offset = src_layout.start_offset();
        let dst_offset = dst_layout.start_offset();

        macro_rules! stage_and_copy {
            ($variant:ident, $ty:ty) => {{
                let CudaGraphPinnedData::$variant(host) = &mut self.data else {
                    candle_core::bail!("CUDA graph host staging dtype changed");
                };
                let src = src_storage.as_slice::<$ty>()?;
                let src = &src[src_offset..src_offset + len];
                let host_slice = host.as_mut_slice();
                if self.initialized && host_slice == src {
                    return Ok(());
                }
                host_slice.copy_from_slice(src);
                self.initialized = true;
                let dst = dst_storage.as_cuda_slice::<$ty>()?;
                let dst = dst.slice(dst_offset..dst_offset + len);
                let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
                let result = unsafe {
                    sys::cuMemcpyHtoDAsync_v2(
                        dst_ptr,
                        host.as_ptr().cast(),
                        len * std::mem::size_of::<$ty>(),
                        stream.cu_stream(),
                    )
                };
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(candle_core::Error::msg(format!("{result:?}"))
                        .context("CUDA graph metadata H2D copy failed"));
                }
            }};
        }

        match src.dtype() {
            DType::U8 => stage_and_copy!(U8, u8),
            DType::U32 => stage_and_copy!(U32, u32),
            DType::I32 => stage_and_copy!(I32, i32),
            DType::I64 => stage_and_copy!(I64, i64),
            DType::F32 => stage_and_copy!(F32, f32),
            dtype => candle_core::bail!(
                "CUDA graph host staging does not support metadata dtype {dtype:?}"
            ),
        }
        Ok(())
    }

    fn copy_from_u32_slice(
        &mut self,
        src: &[u32],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::U32 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching u32 state indices");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA state indices");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous state indices");
        }
        let CudaGraphPinnedData::U32(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging state index dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<u32>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph state index H2D copy failed"));
        }
        Ok(())
    }

    fn copy_from_f32_slice(
        &mut self,
        src: &[f32],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::F32 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching f32 metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA f32 metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous f32 metadata");
        }
        let CudaGraphPinnedData::F32(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging f32 metadata dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<f32>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph f32 metadata H2D copy failed"));
        }
        Ok(())
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    fn copy_from_i64_slice(
        &mut self,
        src: &[i64],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::I64 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching i64 metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA i64 metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous i64 metadata");
        }
        let CudaGraphPinnedData::I64(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging i64 metadata dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<i64>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph i64 metadata H2D copy failed"));
        }
        Ok(())
    }
}

impl CudaGraphHostStaging {
    pub(crate) fn new(graph_stream: Arc<CudaStream>) -> candle_core::Result<Self> {
        let graph_complete = graph_stream
            .context()
            .new_event(None)
            .map_err(candle_core::Error::wrap)?;
        Ok(Self {
            buffers: HashMap::new(),
            completions: HashMap::new(),
            graph_complete,
            graph_stream,
            graph_pending: false,
        })
    }

    pub(crate) fn update(
        &mut self,
        copy: impl FnOnce(&mut Self) -> candle_core::Result<()>,
    ) -> candle_core::Result<()> {
        self.begin_update()?;
        let copy_result = copy(self);
        let finish_result = self.finish_update();
        copy_result.and(finish_result)
    }

    fn begin_update(&mut self) -> candle_core::Result<()> {
        for completion in self.completions.values_mut() {
            completion.ordered_after_graph = false;
            if completion.active {
                completion
                    .stream
                    .synchronize()
                    .map_err(candle_core::Error::wrap)?;
                completion.active = false;
            }
            if completion.pending {
                completion
                    .event
                    .synchronize()
                    .map_err(candle_core::Error::wrap)?;
                completion.pending = false;
            }
        }
        Ok(())
    }

    fn finish_update(&mut self) -> candle_core::Result<()> {
        let mut result = Ok(());
        for completion in self.completions.values_mut() {
            if !completion.active {
                continue;
            }
            match completion.event.record(&completion.stream) {
                Ok(()) => {
                    completion.pending = true;
                    completion.active = false;
                }
                Err(err) => {
                    completion.pending = false;
                    if completion.stream.synchronize().is_ok() {
                        completion.active = false;
                    }
                    if result.is_ok() {
                        result = Err(candle_core::Error::wrap(err));
                    }
                }
            }
        }
        result
    }

    pub(crate) fn order_before_graph(&self) -> candle_core::Result<()> {
        for completion in self.completions.values() {
            if completion.pending && !same_cuda_stream(&completion.stream, &self.graph_stream) {
                self.graph_stream
                    .wait(&completion.event)
                    .map_err(candle_core::Error::wrap)?;
            }
        }
        Ok(())
    }

    pub(crate) fn record_graph_complete(&mut self) -> candle_core::Result<()> {
        self.graph_complete
            .record(&self.graph_stream)
            .map_err(candle_core::Error::wrap)?;
        self.graph_pending = true;
        Ok(())
    }

    fn prepare_copy(
        &mut self,
        location: DeviceLocation,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        let completion = match self.completions.entry(location) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let event = stream
                    .context()
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                    .map_err(candle_core::Error::wrap)?;
                entry.insert(CudaGraphCopyCompletion {
                    event,
                    stream: stream.clone(),
                    pending: false,
                    active: false,
                    ordered_after_graph: false,
                })
            }
        };
        if !same_cuda_stream(&completion.stream, stream) {
            candle_core::bail!("CUDA graph metadata stream changed during replay");
        }
        if self.graph_pending && !completion.ordered_after_graph {
            if !same_cuda_stream(&completion.stream, &self.graph_stream) {
                completion
                    .stream
                    .wait(&self.graph_complete)
                    .map_err(candle_core::Error::wrap)?;
            }
            completion.ordered_after_graph = true;
        }
        completion.active = true;
        Ok(())
    }

    fn copy_from(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &Tensor,
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from(src, dst, &stream)
    }

    pub(crate) fn copy_from_u32_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[u32],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_u32_slice(src, dst, &stream)
    }

    pub(crate) fn copy_from_f32_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[f32],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_f32_slice(src, dst, &stream)
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn copy_from_i64_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[i64],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_i64_slice(src, dst, &stream)
    }
}

fn copy_var_map(
    dst: &CudaGraphVarMap,
    src: &HashMap<DeviceLocation, Tensor>,
    name: &'static str,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    if dst.len() != src.len() {
        candle_core::bail!("{name} device count changed during CUDA graph replay");
    }
    for (location, dst) in dst {
        let src = src
            .get(location)
            .ok_or_else(|| candle_core::Error::msg(format!("{name} missing {location:?}")))?;
        if src.device().is_cpu() && dst.device().is_cuda() {
            host_staging.copy_from(name, *location, src, dst)?;
        } else {
            dst.set(src)?;
        }
    }
    Ok(())
}

fn copy_option_var_map(
    dst: &Option<CudaGraphVarMap>,
    src: Option<&HashMap<DeviceLocation, Tensor>>,
    name: &'static str,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    match (dst, src) {
        (Some(dst), Some(src)) => copy_var_map(dst, src, name, host_staging),
        (None, None) => Ok(()),
        _ => candle_core::bail!("{name} changed optional state during CUDA graph replay"),
    }
}

struct FlashInferTilePlanVars<'a> {
    request_indices: &'a Option<CudaGraphVarMap>,
    kv_tile_indices: &'a Option<CudaGraphVarMap>,
    o_indptr: &'a Option<CudaGraphVarMap>,
    kv_chunk_size: &'a Option<CudaGraphVarMap>,
    block_valid_mask: &'a Option<CudaGraphVarMap>,
}

fn copy_flashinfer_tile_plan(
    metadata: &PagedAttentionInputMetadata,
    full: bool,
    vars: FlashInferTilePlanVars<'_>,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    let view = if full {
        flashinfer_full_view(metadata)
    } else {
        flashinfer_paged_view(metadata)
    };
    copy_option_var_map(
        vars.request_indices,
        view.map(|view| &view.tile_plan.request_indices),
        if full {
            "full_paged_kv_request_indices"
        } else {
            "paged_kv_request_indices"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.kv_tile_indices,
        view.map(|view| &view.tile_plan.kv_tile_indices),
        if full {
            "full_paged_kv_tile_indices"
        } else {
            "paged_kv_tile_indices"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.o_indptr,
        view.map(|view| &view.tile_plan.o_indptr),
        if full {
            "full_paged_kv_o_indptr"
        } else {
            "paged_kv_o_indptr"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.kv_chunk_size,
        view.map(|view| &view.tile_plan.kv_chunk_size),
        if full {
            "full_paged_kv_chunk_size"
        } else {
            "paged_kv_chunk_size"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.block_valid_mask,
        view.map(|view| &view.tile_plan.block_valid_mask),
        if full {
            "full_paged_kv_block_valid_mask"
        } else {
            "paged_kv_block_valid_mask"
        },
        host_staging,
    )
}

fn rope_positions_var_map(
    slot_mappings: &HashMap<DeviceLocation, Tensor>,
    position_ids: &[usize],
    seq_len: usize,
) -> candle_core::Result<CudaGraphVarMap> {
    slot_mappings
        .iter()
        .map(|(location, tensor)| {
            let positions = decode_positions_tensor(position_ids, seq_len, tensor.device())?;
            Ok((*location, Var::from_tensor(&positions)?))
        })
        .collect()
}

fn copy_rope_positions(
    dst: &CudaGraphVarMap,
    position_ids: &[usize],
    seq_len: usize,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    let positions = decode_positions_tensor(position_ids, seq_len, &Device::Cpu)?;
    for (location, dst) in dst {
        if dst.device().is_cuda() {
            host_staging.copy_from("rope_positions", *location, &positions, dst)?;
        } else {
            dst.set(&positions)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_usage(bytes: &[(usize, usize)]) -> CudaGraphSpecStateUsage {
        CudaGraphSpecStateUsage {
            bytes: bytes
                .iter()
                .map(|(gpu_id, bytes)| (DeviceLocation::Cuda { gpu_id: *gpu_id }, *bytes))
                .collect(),
            device_totals: HashMap::new(),
        }
    }

    #[test]
    fn graph_event_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let mut labels = HashSet::new();
        for component in [CudaGraphComponent::Target, CudaGraphComponent::DFlash] {
            for event in [
                CudaGraphEvent::Capture,
                CudaGraphEvent::Replay,
                CudaGraphEvent::EagerFallback,
            ] {
                for outcome in [CudaGraphOutcome::Success, CudaGraphOutcome::Failure] {
                    assert!(labels.insert(cuda_graph_event_labels(component, event, outcome)));
                }
            }
        }

        assert_eq!(
            CUDA_GRAPH_EVENTS_METRIC,
            "mistralrs_cuda_graph_events_total"
        );
        assert_eq!(labels.len(), 12);
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.component)
                .collect::<HashSet<_>>(),
            HashSet::from(["target", "dflash"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.event)
                .collect::<HashSet<_>>(),
            HashSet::from(["capture", "replay", "eager_fallback"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.outcome)
                .collect::<HashSet<_>>(),
            HashSet::from(["success", "failure"])
        );
    }

    #[test]
    fn graph_dispatch_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let reasons = [
            CudaGraphDispatchReason::CacheHit,
            CudaGraphDispatchReason::Disabled,
            CudaGraphDispatchReason::ModelUnsupported,
            CudaGraphDispatchReason::SpeculativeConflict,
            CudaGraphDispatchReason::PagedAttentionUnavailable,
            CudaGraphDispatchReason::Prefill,
            CudaGraphDispatchReason::IncompatibleShape,
            CudaGraphDispatchReason::BatchUnsupported,
            CudaGraphDispatchReason::CacheConfigUnavailable,
            CudaGraphDispatchReason::RuntimeDisabled,
            CudaGraphDispatchReason::PaddingUnavailable,
            CudaGraphDispatchReason::CachePopulation,
            CudaGraphDispatchReason::Fallback,
        ];
        let mut labels = HashSet::new();
        for component in [CudaGraphComponent::Target, CudaGraphComponent::DFlash] {
            for mode in [
                CudaGraphDispatchMode::Replay,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchMode::Skipped,
            ] {
                for reason in reasons {
                    assert!(labels.insert(cuda_graph_dispatch_labels(component, mode, reason)));
                }
            }
        }

        assert_eq!(
            CUDA_GRAPH_DISPATCH_METRIC,
            "mistralrs_cuda_graph_dispatch_total"
        );
        assert_eq!(labels.len(), 78);
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.mode)
                .collect::<HashSet<_>>(),
            HashSet::from(["replay", "eager", "skipped"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.reason)
                .collect::<HashSet<_>>()
                .len(),
            reasons.len()
        );
    }

    #[test]
    fn graph_eviction_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let labels = [
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::Capacity,
            ),
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::MemoryPressure,
            ),
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::SpecStateBudget,
            ),
            (
                CudaGraphComponent::DFlash,
                CudaGraphEvictionReason::Capacity,
            ),
            (
                CudaGraphComponent::DFlash,
                CudaGraphEvictionReason::MemoryPressure,
            ),
        ]
        .into_iter()
        .map(|(component, reason)| (component.label(), reason.label()))
        .collect::<HashSet<_>>();

        assert_eq!(
            CUDA_GRAPH_EVICTIONS_METRIC,
            "mistralrs_cuda_graph_evictions_total"
        );
        assert_eq!(
            CUDA_GRAPH_RESIDENT_ENTRIES_METRIC,
            "mistralrs_cuda_graph_resident_entries"
        );
        assert_eq!(labels.len(), 5);
        assert!(labels.contains(&("target", "capacity")));
        assert!(labels.contains(&("dflash", "capacity")));
        assert!(labels.contains(&("target", "memory_pressure")));
        assert!(labels.contains(&("dflash", "memory_pressure")));
        assert!(labels.contains(&("target", "spec_state_budget")));
    }

    #[test]
    fn capacity_eviction_removes_one_lru_entry_and_preserves_the_bound() {
        let mut entries = vec![10, 20, 30];
        assert_eq!(take_cuda_graph_capacity_eviction(&mut entries, 4), None);
        entries.push(40);
        assert_eq!(take_cuda_graph_capacity_eviction(&mut entries, 4), Some(10));
        entries.push(50);
        assert_eq!(entries, vec![20, 30, 40, 50]);
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn target_cache_retains_64_entries_before_lru_eviction() {
        let capacity = TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY;
        assert_eq!(capacity, 64);

        let mut entries = (0..capacity - 1).collect::<Vec<_>>();
        assert_eq!(
            take_cuda_graph_capacity_eviction(&mut entries, capacity),
            None
        );
        entries.push(capacity - 1);
        assert_eq!(entries.len(), capacity);

        assert_eq!(
            take_cuda_graph_capacity_eviction(&mut entries, capacity),
            Some(0),
        );
        entries.push(capacity);

        assert_eq!(entries.len(), capacity);
        assert_eq!(entries.first(), Some(&1));
        assert_eq!(entries.last(), Some(&capacity));
    }

    #[test]
    fn batch_buckets_are_exact_then_power_of_two() {
        assert_eq!(cuda_graph_batch_bucket(0), None);
        for batch in 1..=CUDA_GRAPH_EXACT_BATCH_BUCKETS {
            assert_eq!(cuda_graph_batch_bucket(batch), Some(batch));
        }
        assert_eq!(cuda_graph_batch_bucket(9), Some(16));
        assert_eq!(cuda_graph_batch_bucket(16), Some(16));
        assert_eq!(cuda_graph_batch_bucket(17), Some(32));
        assert_eq!(
            cuda_graph_batch_bucket(CUDA_GRAPH_MAX_BATCH_BUCKET),
            Some(CUDA_GRAPH_MAX_BATCH_BUCKET)
        );
        assert_eq!(
            cuda_graph_batch_bucket(CUDA_GRAPH_MAX_BATCH_BUCKET + 1),
            None
        );
    }

    #[test]
    fn precapture_includes_the_first_power_of_two_bucket() {
        assert_eq!(
            cuda_graph_precapture_batches().collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 16]
        );
    }

    #[test]
    fn startup_precapture_accepts_decode_and_verification_widths() {
        assert!(!cuda_graph_startup_capture_allowed(0));
        assert!(cuda_graph_startup_capture_allowed(1));
        assert!(cuda_graph_startup_capture_allowed(4));
        assert!(cuda_graph_startup_capture_allowed(8));
    }

    #[test]
    fn decode_graph_batch_kind_rejects_every_prefill_chunk() {
        assert!(!cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::Prefill
        ));
        assert!(cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::Decode
        ));
        assert!(cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::SpeculativeDecode
        ));
    }

    #[test]
    fn graph_context_len_tracks_paged_attention_partitions() {
        assert_eq!(graph_context_len(Some(1), Some(2048)), Some(512));
        assert_eq!(graph_context_len(Some(512), Some(2048)), Some(512));
        assert_eq!(graph_context_len(Some(513), Some(2048)), Some(1024));
        assert_eq!(graph_context_len(Some(1537), Some(2048)), Some(2048));
    }

    #[test]
    fn graph_context_len_preserves_nonstandard_metadata() {
        assert_eq!(graph_context_len(Some(513), None), Some(513));
        assert_eq!(graph_context_len(None, Some(2048)), Some(2048));
        assert_eq!(graph_context_len(None, None), None);
    }

    #[test]
    fn decode_row_graph_key_is_independent_of_materialization() {
        let table = vec![1, 2, 3, 4];
        let rows = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![127]],
            block_tables: vec![table.clone()],
            context_lens: vec![128],
            full_block_tables: vec![table],
            full_context_lens: vec![128],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 1_604_288,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 4,
        });
        let staged = rows.build_graph_staged().unwrap();
        let materialized = rows.build_materialized().unwrap();
        let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
        let staged_key =
            CudaDecodeGraphKey::new(&input_ids, &staged, 32, RecurrentBatchKind::Decode).unwrap();
        let materialized_key =
            CudaDecodeGraphKey::new(&input_ids, &materialized, 32, RecurrentBatchKind::Decode)
                .unwrap();
        assert_eq!(staged_key, materialized_key);
        assert!(staged_key.tensors.is_empty());
        assert!(staged_key.decode_rows.is_some());
        let speculative_key = CudaDecodeGraphKey::new(
            &input_ids,
            &materialized,
            32,
            RecurrentBatchKind::SpeculativeDecode,
        )
        .unwrap();
        assert_ne!(staged_key, speculative_key);

        let mut next_bucket_rows = (*rows).clone();
        next_bucket_rows.block_tables = vec![(1..=17).collect()];
        next_bucket_rows.full_block_tables = next_bucket_rows.block_tables.clone();
        next_bucket_rows.context_lens = vec![513];
        next_bucket_rows.full_context_lens = vec![513];
        let next_bucket = Arc::new(next_bucket_rows).build_graph_staged().unwrap();
        let next_bucket_key =
            CudaDecodeGraphKey::new(&input_ids, &next_bucket, 32, RecurrentBatchKind::Decode)
                .unwrap();
        assert_ne!(staged_key, next_bucket_key);
    }

    #[test]
    fn graph_buffers_share_unsliding_flashinfer_metadata() {
        let table = vec![1, 2, 3, 4];
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![127]],
            block_tables: vec![table.clone()],
            context_lens: vec![128],
            full_block_tables: vec![table],
            full_context_lens: vec![128],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 1_604_288,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 4,
        })
        .build()
        .unwrap();
        let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
        let key =
            CudaDecodeGraphKey::new(&input_ids, &metadata, 32, RecurrentBatchKind::Decode).unwrap();
        assert!(key
            .tensors
            .iter()
            .all(|tensor| !tensor.name.starts_with("full_")));

        let (buffers, _) = CudaDecodeGraphMetadataBuffers::new(
            &metadata,
            &[127],
            &[128],
            1,
            32,
            &[],
            None,
            DType::F32,
        )
        .unwrap();
        assert!(buffers.flashinfer_views_alias);
        let location = Device::Cpu.location();
        assert_eq!(
            buffers.paged_kv_indices.as_ref().unwrap()[&location].id(),
            buffers.full_paged_kv_indices.as_ref().unwrap()[&location].id()
        );
        assert_eq!(
            buffers.paged_kv_request_indices.as_ref().unwrap()[&location].id(),
            buffers.full_paged_kv_request_indices.as_ref().unwrap()[&location].id()
        );
    }

    #[test]
    fn graph_rope_positions_expand_adjusted_ends_for_verification() {
        let table = vec![1, 2];
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![37, 38, 39], vec![37, 38, 39]],
            block_tables: vec![table.clone(); 6],
            context_lens: vec![38, 39, 40, 38, 39, 40],
            full_block_tables: vec![table; 6],
            full_context_lens: vec![38, 39, 40, 38, 39, 40],
            query_len: 3,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 128,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 4,
        })
        .build_materialized()
        .unwrap();

        let (buffers, _) = CudaDecodeGraphMetadataBuffers::new(
            &metadata,
            &[97, 97],
            &[100, 52],
            3,
            32,
            &[],
            None,
            DType::F32,
        )
        .unwrap();
        let positions = buffers.rope_positions[&Device::Cpu.location()]
            .as_detached_tensor()
            .to_vec1::<u32>()
            .unwrap();

        assert_eq!(positions, vec![97, 98, 99, 49, 50, 51]);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_metadata_replay_updates_persistent_cuda_buffers() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let rows = |table: Vec<usize>, context_len: usize| -> anyhow::Result<_> {
            Ok(Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![i64::try_from(context_len - 1)?]],
                block_tables: vec![table.clone()],
                context_lens: vec![context_len],
                full_block_tables: vec![table],
                full_context_lens: vec![context_len],
                query_len: 1,
                block_size: 32,
                use_standard_metadata: false,
                max_paged_context_len: 1_604_288,
                sliding_window: None,
                decode_window: 1,
                devices: vec![device.clone()],
                num_kv_heads: 4,
            }))
        };
        let initial = rows(vec![1, 2, 3, 4], 128)?.build_materialized()?;
        let updated = rows(vec![5, 6, 7, 8, 9, 10, 11, 12], 256)?.build()?;
        assert!(updated.has_host_staged_decode_tensors());

        let (mut buffers, _) = CudaDecodeGraphMetadataBuffers::new(
            &initial,
            &[127],
            &[128],
            1,
            32,
            &[],
            None,
            DType::F32,
        )?;
        let location = device.location();
        let state_indices = Var::from_tensor(&Tensor::zeros((3,), DType::U32, &device)?)?;
        let mut state_indices_map = HashMap::from([(location, state_indices)]);
        let mut host_staging = CudaGraphHostStaging::new(device.as_cuda_device()?.cuda_stream())?;
        host_staging.update(|host_staging| {
            buffers.copy_from(&updated, &[256], 1, host_staging)?;
            copy_state_indices(&state_indices_map, &[3, 5, 7], host_staging)
        })?;
        host_staging.update(|host_staging| {
            buffers.copy_from(&updated, &[256], 1, host_staging)?;
            copy_state_indices(&state_indices_map, &[11, 13, 17], host_staging)
        })?;
        device.synchronize()?;

        let indices = buffers.paged_kv_indices.as_ref().unwrap()[&location]
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<i32>()?;
        assert_eq!(&indices[..8], &[5, 6, 7, 8, 9, 10, 11, 12]);
        let positions = buffers.rope_positions[&location]
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        assert_eq!(positions, vec![255]);
        assert!(host_staging.buffers.len() > 1);
        assert_eq!(host_staging.completions.len(), 1);

        let state_indices = state_indices_map
            .remove(&location)
            .unwrap()
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        assert_eq!(state_indices, vec![11, 13, 17]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_staging_orders_secondary_streams_both_directions() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let graph_stream = device.as_cuda_device()?.cuda_stream();
        let copy_stream = graph_stream.fork()?;
        assert!(!same_cuda_stream(&graph_stream, &copy_stream));
        let location = device.location();
        let mut staging = CudaGraphHostStaging::new(graph_stream.clone())?;

        staging.record_graph_complete()?;
        staging.update(|staging| staging.prepare_copy(location, &copy_stream))?;
        assert!(staging.completions[&location].ordered_after_graph);
        staging.order_before_graph()?;

        staging.record_graph_complete()?;
        staging.update(|staging| staging.prepare_copy(location, &copy_stream))?;
        assert!(staging.completions[&location].ordered_after_graph);
        staging.order_before_graph()?;
        graph_stream.synchronize()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_memory_pool_scope_restores_release_threshold() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let pool = cuda_memory_pool(&stream)?;
        let original = memory_pool_release_threshold(pool)?;

        let first = prepare_cuda_graph_memory_pool(&stream)?;
        let second = prepare_cuda_graph_memory_pool(&stream)?;
        assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
        drop(first);
        assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
        drop(second);
        assert_eq!(memory_pool_release_threshold(pool)?, original);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_memory_cleanup_returns_allocator_to_baseline() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let used_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
        let reserved_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT;
        let used_before = cuda_graph_memory_attribute(&stream, used_attribute)?;
        let reserved_before = cuda_graph_memory_attribute(&stream, reserved_attribute)?;

        let guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let output = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;
        graph.launch()?;
        stream.synchronize()?;
        drop(output);
        stream.synchronize()?;
        drop(graph);
        drop(guard);

        trim_cuda_graph_memory(&stream)?;
        assert_eq!(
            cuda_graph_memory_attribute(&stream, used_attribute)?,
            used_before
        );
        assert_eq!(
            cuda_graph_memory_attribute(&stream, reserved_attribute)?,
            reserved_before
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn unlaunched_graph_cleanup_returns_allocator_to_baseline() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let used_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
        let reserved_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT;
        let used_before = cuda_graph_memory_attribute(&stream, used_attribute)?;
        let reserved_before = cuda_graph_memory_attribute(&stream, reserved_attribute)?;

        let guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let logits = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        let mut release_result = Ok(());
        drop_cuda_graph_entry_logits(logits, &stream, 0, &mut release_result);
        release_result?;
        drop(graph);
        drop(guard);
        trim_cuda_graph_memory(&stream)?;

        assert_eq!(
            cuda_graph_memory_attribute(&stream, used_attribute)?,
            used_before
        );
        assert_eq!(
            cuda_graph_memory_attribute(&stream, reserved_attribute)?,
            reserved_before
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_replay_retains_captured_output_storage() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let cuda_device = device.as_cuda_device()?;
        let stream = cuda_device.cuda_stream();
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;

        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let warmup = input.as_detached_tensor().affine(2.0, 1.0)?;
        device.synchronize()?;
        drop(warmup);

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let output = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        for (values, expected) in [
            (vec![3f32, 5.0], vec![7f32, 11.0]),
            (vec![8f32, 13.0], vec![17f32, 27.0]),
        ] {
            input.set(&Tensor::from_vec(values, 2, &device)?)?;
            graph.launch()?;
            stream.synchronize()?;
            assert_eq!(output.to_vec1::<f32>()?, expected);
        }

        drop(graph);
        drop(output);
        device.synchronize()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_copy_supports_dense_row_source() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let cuda_device = device.as_cuda_device()?;
        let stream = cuda_device.cuda_stream();
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;

        let input = Var::from_tensor(&Tensor::from_vec(
            (1u16..=16).map(f32::from).collect::<Vec<_>>(),
            (2, 2, 4),
            &device,
        )?)?;
        let output = Var::from_tensor(&Tensor::zeros((2, 2, 2), DType::F32, &device)?)?;
        let source = input.as_detached_tensor().narrow(2, 1, 2)?;
        assert!(!source.is_contiguous());

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        crate::cuda::graph::copy_tensor(&source, &output.as_detached_tensor())?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        graph.launch()?;
        stream.synchronize()?;
        assert_eq!(
            output.as_detached_tensor().to_vec3::<f32>()?,
            vec![
                vec![vec![2.0, 3.0], vec![6.0, 7.0]],
                vec![vec![10.0, 11.0], vec![14.0, 15.0]],
            ]
        );

        input.set(&Tensor::from_vec(
            (1u16..=16).rev().map(f32::from).collect::<Vec<_>>(),
            (2, 2, 4),
            &device,
        )?)?;
        graph.launch()?;
        stream.synchronize()?;
        assert_eq!(
            output.as_detached_tensor().to_vec3::<f32>()?,
            vec![
                vec![vec![15.0, 14.0], vec![11.0, 10.0]],
                vec![vec![7.0, 6.0], vec![3.0, 2.0]],
            ]
        );
        Ok(())
    }

    #[test]
    fn graph_suspension_does_not_clear_permanent_disable() {
        let mut state = CudaDecodeGraphState::default();
        state.suspend();
        assert!(state.disabled());
        state.resume();
        assert!(!state.disabled());
        state.disable();
        state.suspend();
        state.resume();
        assert!(state.disabled());
    }

    #[test]
    fn eager_retry_block_applies_to_one_failed_step() {
        let mut state = CudaDecodeGraphState::default();
        assert!(state.take_eager_retry_allowed());
        state.block_eager_retry();
        assert!(!state.take_eager_retry_allowed());
        assert!(state.take_eager_retry_allowed());
        state.block_eager_retry();
        state.clear();
        assert!(state.take_eager_retry_allowed());
    }

    #[test]
    fn graph_generations_are_not_reused_after_cleanup() {
        let mut state = CudaDecodeGraphState::default();
        let first = state.allocate_generation();
        state.clear();
        let second = state.allocate_generation();
        state.suspend();
        state.resume();
        let third = state.allocate_generation();
        assert!(first < second && second < third);
        assert!(!cuda_graph_replay_version_matches(second, 0, first, 1));
        assert!(!cuda_graph_replay_version_matches(third, 0, second, 1));
        assert!(cuda_graph_replay_version_matches(third, 7, third, 7));
    }

    #[test]
    fn memory_pressure_eviction_drains_lru_entries_in_one_batch() {
        let mut entries = vec![10, 20, 30, 40];
        assert!(drain_lru_entries(&mut entries, 0).is_empty());
        assert_eq!(drain_lru_entries(&mut entries, 2), vec![10, 20]);
        assert_eq!(entries, vec![30, 40]);
        assert_eq!(drain_lru_entries(&mut entries, usize::MAX), vec![30, 40]);
        assert!(entries.is_empty());
    }

    #[test]
    fn graph_reclaim_quota_is_shared_between_target_and_dflash() {
        let mut target_entries = 2usize;
        let mut dflash_entries = 4usize;
        let reclaimed = reclaim_cuda_graph_entries(
            3,
            |limit| {
                let reclaimed = target_entries.min(limit);
                target_entries -= reclaimed;
                reclaimed
            },
            |limit| {
                let reclaimed = dflash_entries.min(limit);
                dflash_entries -= reclaimed;
                reclaimed
            },
        );
        assert_eq!(reclaimed, 3);
        assert_eq!(target_entries, 0);
        assert_eq!(dflash_entries, 3);

        let mut dflash_called = false;
        assert_eq!(
            reclaim_cuda_graph_entries(
                2,
                |_| 2,
                |_| {
                    dflash_called = true;
                    0
                }
            ),
            2
        );
        assert!(!dflash_called);

        let mut target_called = false;
        assert_eq!(
            reclaim_cuda_graph_entries(
                0,
                |_| {
                    target_called = true;
                    0
                },
                |_| 0,
            ),
            0
        );
        assert!(!target_called);
    }

    #[test]
    fn speculative_state_budget_is_four_percent_by_default() {
        assert_eq!(default_spec_state_budget(100), 4);
        assert_eq!(default_spec_state_budget(98_000), 3_920);
    }

    #[test]
    fn speculative_state_budget_evicts_lru_entries_per_device() {
        let gpu0 = DeviceLocation::Cuda { gpu_id: 0 };
        let gpu1 = DeviceLocation::Cuda { gpu_id: 1 };
        let budgets = HashMap::from([(gpu0, 100), (gpu1, 100)]);
        let existing = vec![
            spec_usage(&[]),
            spec_usage(&[(0, 40)]),
            spec_usage(&[(1, 60)]),
        ];

        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(1, 50)]), &budgets),
            vec![2]
        );
        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(0, 120)]), &budgets),
            vec![1]
        );
        let existing = vec![
            spec_usage(&[(0, 30)]),
            spec_usage(&[(0, 50)]),
            spec_usage(&[(0, 10)]),
        ];
        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(0, 60)]), &budgets),
            vec![0, 1]
        );
    }

    #[test]
    fn one_token_continuation_advances_host_decode_state() -> anyhow::Result<()> {
        let rows = Arc::new(
            DecodePagedRows {
                slot_mappings: vec![vec![39]],
                block_tables: vec![vec![9]],
                context_lens: vec![4],
                full_block_tables: vec![vec![9, 17]],
                full_context_lens: vec![4],
                query_len: 1,
                block_size: 4,
                use_standard_metadata: false,
                max_paged_context_len: 64,
                sliding_window: Some(4),
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 1,
            }
            .padded(2),
        );
        let step = CudaGraphDecodeStep {
            input_ids: Tensor::zeros((2, 1), DType::U32, &Device::Cpu)?,
            seqlen_offsets: vec![100, 100],
            context_lens: vec![(0, 1), (0, 1)],
            position_ids: vec![43, 43],
            metadata: rows.build_graph_staged()?,
            state_indices: Some(vec![2, 5]),
            real_batch: 1,
        };
        let continuation = step
            .one_token_continuation(step.input_ids.clone())?
            .expect("next allocated block should permit one token");
        let rows = continuation.metadata.decode_rows.as_ref().unwrap();
        assert_eq!(rows.slot_mappings, vec![vec![68], vec![_PAD_SLOT_ID]]);
        assert_eq!(rows.block_tables, vec![vec![9, 17], vec![9, 17]]);
        assert_eq!(rows.context_lens, vec![5, 5]);
        assert_eq!(rows.full_context_lens, vec![5, 5]);
        assert_eq!(continuation.seqlen_offsets, vec![101, 101]);
        assert_eq!(continuation.context_lens, vec![(0, 1), (0, 1)]);
        assert_eq!(continuation.position_ids, vec![44, 44]);
        assert_eq!(continuation.state_indices, Some(vec![2, 5]));
        Ok(())
    }

    #[test]
    fn one_token_continuation_requires_an_allocated_boundary_slot() -> anyhow::Result<()> {
        let rows = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![39]],
            block_tables: vec![vec![9]],
            context_lens: vec![4],
            full_block_tables: vec![vec![9]],
            full_context_lens: vec![4],
            query_len: 1,
            block_size: 4,
            use_standard_metadata: false,
            max_paged_context_len: 64,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 1,
        });
        let step = CudaGraphDecodeStep {
            input_ids: Tensor::zeros((1, 1), DType::U32, &Device::Cpu)?,
            seqlen_offsets: vec![3],
            context_lens: vec![(0, 1)],
            position_ids: vec![4],
            metadata: rows.build_graph_staged()?,
            state_indices: None,
            real_batch: 1,
        };
        assert!(step
            .one_token_continuation(step.input_ids.clone())?
            .is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn resident_replay_skips_the_host_input_update() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![0]],
            block_tables: vec![vec![0]],
            context_lens: vec![1],
            full_block_tables: vec![vec![0]],
            full_context_lens: vec![1],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 32,
            sliding_window: None,
            decode_window: 1,
            devices: vec![device.clone()],
            num_kv_heads: 1,
        })
        .build_materialized()?;
        let initial_ids = Tensor::from_vec(vec![1u32], (1, 1), &device)?;
        let key = CudaDecodeGraphKey::new(&initial_ids, &metadata, 32, RecurrentBatchKind::Decode)?;
        let warmup_logits = initial_ids.to_dtype(DType::F32)?;
        let entry = capture_cuda_decode_graph(
            CudaDecodeGraphCaptureCtx {
                key: key.clone(),
                input_ids: &initial_ids,
                seqlen_offsets: &[0],
                position_ids: &[1],
                block_size: 32,
                kv_cache: &[],
                metadata: &metadata,
                model_metadata: None,
                activation_dtype: DType::F32,
                warmup_logits: &warmup_logits,
                state_indices: None,
                real_batch: 1,
            },
            |input_ids, _| input_ids.to_dtype(DType::F32),
        )?;
        let mut state = CudaDecodeGraphState::default();
        state.insert(entry);

        let step = |token: u32| -> candle_core::Result<CudaGraphDecodeStep> {
            Ok(CudaGraphDecodeStep {
                input_ids: Tensor::from_vec(vec![token], (1, 1), &device)?,
                seqlen_offsets: vec![0],
                context_lens: vec![(0, 1)],
                position_ids: vec![0],
                metadata: metadata.clone(),
                state_indices: None,
                real_batch: 1,
            })
        };
        let host_step = step(7)?;
        let host_replay = state
            .replay(&key, &host_step, CudaDecodeGraphReplayInput::Host)?
            .expect("captured graph missing");
        let launch = host_replay.launch.expect("qlen=1 launch missing");
        launch.graph_stream().synchronize()?;
        assert_eq!(host_replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);

        let resident_step = step(13)?;
        let resident_replay = state
            .replay(
                &key,
                &resident_step,
                CudaDecodeGraphReplayInput::Resident(&launch),
            )?
            .expect("resident graph entry changed");
        resident_replay
            .launch
            .as_ref()
            .unwrap()
            .graph_stream()
            .synchronize()?;
        assert_eq!(resident_replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
        let resident_launch = resident_replay.launch.unwrap();
        assert_eq!(resident_launch.generation(), launch.generation());
        assert!(resident_launch.one_token_continuation()?.is_some());
        assert!(state.replay_one_token(launch)?.is_none());
        let lookahead = state
            .replay_one_token(resident_launch)?
            .expect("one-token continuation should retain the graph key");
        lookahead
            .launch
            .as_ref()
            .unwrap()
            .graph_stream()
            .synchronize()?;
        assert_eq!(lookahead.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
        Ok(())
    }
}
