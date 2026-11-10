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

use crate::paged_attention::plan::DecodePlan;
use crate::{
    flashinfer::{
        make_fa3_decode_state, Fa3DecodeState, FlashInferMetadata, FlashInferPagedAttentionView,
        FlashInferPagedAttentionViews, FlashInferPagedKv, FlashInferTilePlan,
    },
    paged_attention::{
        block_table_rows::BlockTableSnapshot, AttentionBackendKind, ModelConfigLike,
    },
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

mod buckets;
mod events;
mod memory;
mod metadata;
mod staging;

pub(crate) use buckets::*;
pub(crate) use events::*;
pub(crate) use memory::*;
pub(crate) use metadata::*;
pub(crate) use staging::*;

const CUDA_GRAPH_INSTANTIATE_FLAGS: u64 =
    sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH.0 as u64;
// Matches the standard CUDA paged-attention V2 partition size.
const PAGED_ATTENTION_PARTITION_SIZE: usize = 512;
const CAPTURE_ATTENTION_OUTPUT_BUFFERS: usize = 2;
const TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY: usize = 64;
const TARGET_CUDA_DECODE_GRAPH_CACHE_MAX_CAPACITY: usize = 96;
const TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM: usize = 16;
const CUDA_GRAPH_BATCH_BUCKET_GRANULARITY: usize = 8;
const CUDA_GRAPH_COARSE_EXACT_BATCH_BUCKETS: usize = 8;
const TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS: usize = 16;
pub(crate) const CUDA_GRAPH_MAX_BATCH_BUCKET: usize = 128;
pub(crate) const CUDA_GRAPH_PRECAPTURE_MAX_BATCH: usize = CUDA_GRAPH_MAX_BATCH_BUCKET;
const CUDA_GRAPH_SPEC_STATE_BUDGET_FLOOR_PERCENT: usize = 4;
const CUDA_GRAPH_SPEC_STATE_BUDGET_CEILING_PERCENT: usize = 8;
// C128 buckets plus one extra max-batch context graph require just under ten largest entries.
const CUDA_GRAPH_SPEC_STATE_WORKING_SET_MULTIPLIER: usize = 10;
const CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES_ENV: &str = "MISTRALRS_CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES";
const CUDA_PHASE_TIMINGS_ENV: &str = "MISTRALRS_CUDA_PHASE_TIMINGS";
const CUDA_GRAPH_EVENTS_METRIC: &str = "mistralrs_cuda_graph_events_total";
const CUDA_GRAPH_DISPATCH_METRIC: &str = "mistralrs_cuda_graph_dispatch_total";
const CUDA_GRAPH_EVICTIONS_METRIC: &str = "mistralrs_cuda_graph_evictions_total";
const CUDA_GRAPH_RESIDENT_ENTRIES_METRIC: &str = "mistralrs_cuda_graph_resident_entries";
static NEXT_CUDA_DECODE_GRAPH_GENERATION: AtomicU64 = AtomicU64::new(1);
static CUDA_GRAPH_MEMORY_POOL_SCOPES: OnceLock<Mutex<HashMap<usize, MemoryPoolScopeState>>> =
    OnceLock::new();
static CUDA_PHASE_TIMINGS_ENABLED: OnceLock<bool> = OnceLock::new();
const CAPTURE_RECLAIM_BATCH_SIZE: usize = 1;

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
        let input_ids = if input_ids.dtype() == DType::U32 && input_ids.device().is_cuda() {
            crate::cuda::input_packing::pad_decode_input(input_ids, batch)?
        } else {
            let pad_ids = input_ids.narrow(0, 0, 1)?.repeat((pad, 1))?;
            Tensor::cat(&[input_ids, &pad_ids], 0)?
        };
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
        let mut full_context_lens = Vec::with_capacity(self.real_batch);
        for row in 0..self.real_batch {
            let Some(&current_slot) = rows.slot_mappings[row].first() else {
                return Ok(None);
            };
            let Ok(current_slot) = usize::try_from(current_slot) else {
                return Ok(None);
            };
            let full_table = rows.full_block_table(row);
            let current_block = current_slot / rows.block_size;
            let Some(current_block_idx) =
                full_table.iter().position(|&block| block == current_block)
            else {
                return Ok(None);
            };
            let current_block_offset = current_slot % rows.block_size;
            let next_slot = if current_block_offset + 1 < rows.block_size {
                current_slot + 1
            } else {
                let next_block_idx = current_block_idx + 1;
                let Some(&next_block) = full_table.get(next_block_idx) else {
                    return Ok(None);
                };
                let Some(next_slot) = next_block.checked_mul(rows.block_size) else {
                    return Ok(None);
                };
                next_slot
            };
            let Ok(next_slot) = i64::try_from(next_slot) else {
                return Ok(None);
            };
            let Some(next_full_context_len) = rows.full_context_lens[row].checked_add(1) else {
                return Ok(None);
            };

            let paged_context_len = match rows.sliding_window {
                Some(window) => {
                    let window_start = next_full_context_len.saturating_sub(window);
                    let block_aligned_start = window_start / rows.block_size * rows.block_size;
                    next_full_context_len - block_aligned_start
                }
                None => next_full_context_len,
            };
            slot_mappings.push(vec![next_slot]);
            block_tables.push(
                rows.block_tables
                    .table_arc(rows.block_tables.row_table_index(row)),
            );
            context_lens.push(paged_context_len);
            full_context_lens.push(next_full_context_len);
        }

        let rows = Arc::new(
            DecodePagedRows {
                slot_mappings,
                block_tables: BlockTableSnapshot::from_sequence_tables(block_tables, 1),
                context_lens,
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
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![0]], q_len),
            context_lens: vec![1; q_len],
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

impl CudaDecodeGraphKey {
    fn has_same_spec_state_shape(&self, other: &Self) -> bool {
        self.device == other.device
            && self.input_shape == other.input_shape
            && self.input_dtype == other.input_dtype
            && self.recurrent_batch_kind == other.recurrent_batch_kind
    }

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

        // Row keys bucket table capacity more coarsely than the captured V2 partition count.
        let (max_context_len, full_max_context_len) = match metadata.decode_rows.as_ref() {
            Some(rows) if rows.use_standard_metadata => (
                metadata
                    .max_context_len
                    .map(paged_attention_partition_context_len),
                metadata
                    .full_max_context_len
                    .map(paged_attention_partition_context_len),
            ),
            Some(_) => (None, None),
            None => (
                graph_context_len(
                    metadata.max_context_len,
                    bucket_context_len(metadata.block_tables.as_ref(), block_size),
                ),
                graph_context_len(
                    metadata.full_max_context_len,
                    bucket_context_len(metadata.full_block_tables.as_ref(), block_size),
                ),
            ),
        };

        Ok(Self {
            device: input_ids.device().location(),
            input_shape: input_ids.dims().to_vec(),
            input_dtype: input_ids.dtype(),
            recurrent_batch_kind,
            max_context_len,
            full_max_context_len,
            tensors,
            decode_rows,
        })
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

fn default_spec_state_budget(total: usize, largest_entry: usize) -> usize {
    let floor = total.saturating_mul(CUDA_GRAPH_SPEC_STATE_BUDGET_FLOOR_PERCENT) / 100;
    let ceiling = total.saturating_mul(CUDA_GRAPH_SPEC_STATE_BUDGET_CEILING_PERCENT) / 100;
    let working_set = largest_entry.saturating_mul(CUDA_GRAPH_SPEC_STATE_WORKING_SET_MULTIPLIER);
    floor.max(working_set.min(ceiling)).max(largest_entry)
}

fn configured_spec_state_budget(total: usize, largest_entry: usize) -> usize {
    std::env::var(CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| default_spec_state_budget(total, largest_entry))
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
    // Proposer-facing graph outputs refreshed by replay.
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
        drop_cuda_graph_entry_output(
            spec_state,
            &stream,
            replay_epoch,
            "speculative state",
            &mut release_result,
        );
        drop_cuda_graph_entry_output(logits, &stream, replay_epoch, "logits", &mut release_result);
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

fn unmaterialized_graph_output_drop_error(replay_epoch: u64, error: sys::CUresult) -> bool {
    replay_epoch == 0 && error == sys::CUresult::CUDA_ERROR_INVALID_VALUE
}

fn drop_cuda_graph_entry_output<T>(
    output: T,
    stream: &Arc<CudaStream>,
    replay_epoch: u64,
    name: &'static str,
    release_result: &mut candle_core::Result<()>,
) {
    drop(output);
    if let Err(err) = stream.context().check_err() {
        if unmaterialized_graph_output_drop_error(replay_epoch, err.0) {
            return;
        }
        if release_result.is_ok() {
            *release_result = Err(candle_core::Error::wrap(err)
                .context(format!("CUDA graph entry {name} release failed")));
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

pub(crate) struct CudaDecodeGraphState {
    entries: Vec<CudaDecodeGraphEntry>,
    spec_state_budgets: HashMap<DeviceLocation, usize>,
    capacity: usize,
    disabled: bool,
    suspended: bool,
    eager_retry_blocked: bool,
    recurrent_storage_generation: Option<u64>,
}

impl Default for CudaDecodeGraphState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            spec_state_budgets: HashMap::new(),
            capacity: TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY,
            disabled: false,
            suspended: false,
            eager_retry_blocked: false,
            recurrent_storage_generation: None,
        }
    }
}

impl Drop for CudaDecodeGraphState {
    fn drop(&mut self) {
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, 0);
    }
}

impl CudaDecodeGraphState {
    pub(crate) fn ensure_capacity(&mut self, capacity: usize) {
        self.capacity = self.capacity.max(capacity);
    }

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
            let spec_state = match &entry.spec_state {
                Some(state) if step.real_batch == step.input_ids.dim(0)? => Some(state.clone()),
                Some(state) => Some(Arc::from(state.for_real_batch(step.real_batch)?)),
                None => None,
            };
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
        let (_, timing_rows) = step.input_ids.dims2()?;
        let phase_timer = CudaPhaseTimer::start(entry.graph.stream())?;
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
        if let Some(timer) = phase_timer {
            timer.finish("target_graph", step.real_batch, timing_rows)?;
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

    pub(crate) fn prepare_spec_state_admission_for_key(&mut self, key: &CudaDecodeGraphKey) {
        let usage = self
            .entries
            .iter()
            .rev()
            .find(|entry| entry.key.has_same_spec_state_shape(key))
            .map(|entry| entry.spec_state_usage.clone());
        if let Some(usage) = usage {
            self.evict_for_spec_state(&usage);
        }
    }

    pub(crate) fn insert(&mut self, mut entry: CudaDecodeGraphEntry) {
        self.evict_for_spec_state(&entry.spec_state_usage);
        if let Some(evicted) = take_cuda_graph_capacity_eviction(&mut self.entries, self.capacity) {
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
            let incoming_bytes = incoming.bytes.get(location).copied().unwrap_or(0);
            let configured = configured_spec_state_budget(*total, incoming_bytes);
            self.spec_state_budgets
                .entry(*location)
                .and_modify(|budget| *budget = (*budget).max(configured))
                .or_insert(configured);
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
        let mut generation = NEXT_CUDA_DECODE_GRAPH_GENERATION.load(Ordering::Relaxed);
        loop {
            let next = generation
                .checked_add(1)
                .expect("CUDA decode graph generation overflow");
            match NEXT_CUDA_DECODE_GRAPH_GENERATION.compare_exchange_weak(
                generation,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return generation,
                Err(current) => generation = current,
            }
        }
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
    let (mut metadata_buffers, metadata) =
        CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata,
            seqlen_offsets,
            position_ids,
            seq_len,
            block_size,
            kv_cache,
            model_metadata,
            activation_dtype,
        })?;
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
    let graph_mem = |attribute| cuda_graph_memory_attribute(&stream, attribute).unwrap_or(0) >> 20;
    tracing::debug!(
        reserved_mib = graph_mem(sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT),
        used_mib = graph_mem(sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT),
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
    (0..metadata.num_layers()).all(|layer_idx| {
        !DecodePlan::requires_host_context_lengths(
            metadata.attention_backend_kind_for_layer(layer_idx),
            metadata.k_head_dim_for_layer(layer_idx),
            metadata.sliding_window_for_layer(layer_idx).is_some()
                || metadata
                    .prefix_prefill_attention_features(layer_idx)
                    .is_some_and(|features| features.has_sliding_window),
        )
    })
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

#[cfg(test)]
mod tests;
