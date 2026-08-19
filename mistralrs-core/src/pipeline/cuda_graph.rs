use std::{collections::HashMap, sync::Arc};

use candle_core::cuda_backend::cudarc::driver::{sys, CudaStream};
use candle_core::{DType, Device, DeviceLocation, Tensor, Var};

#[cfg(target_family = "unix")]
use crate::paged_attention::plan::DecodePlan;
use crate::{
    flashinfer::{
        FlashInferMetadata, FlashInferPagedAttentionView, FlashInferPagedAttentionViews,
        FlashInferPagedKv, FlashInferTilePlan,
    },
    paged_attention::{AttentionBackendKind, ModelConfigLike},
};

use crate::device_map::DeviceMapper;
use crate::kv_cache::HybridCache;
use crate::paged_attention::_PAD_SLOT_ID;
use crate::pipeline::{
    text_models_inputs_processor::{
        make_flash_params, DecodePagedRows, FlashParams, PagedAttentionInputMetadata,
    },
    text_positions_tensor, DecodeGraphPrecaptureCtx,
};
use crate::speculative::SpeculativeGraphState;

const CUDA_GRAPH_INSTANTIATE_FLAGS: u64 =
    sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64;
// Matches the standard CUDA paged-attention V2 partition size.
const PAGED_ATTENTION_PARTITION_SIZE: usize = 512;
pub(crate) const CUDA_DECODE_GRAPH_CACHE_CAPACITY: usize = 32;
// Batches up to this size get their own graph; larger ones pad up to the next power of two.
pub(crate) const CUDA_GRAPH_EXACT_BATCH_BUCKETS: usize = 8;
pub(crate) const CUDA_GRAPH_MAX_BATCH_BUCKET: usize = 64;

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
pub(crate) fn cuda_graph_precapture_batches() -> std::ops::RangeInclusive<usize> {
    1..=CUDA_GRAPH_EXACT_BATCH_BUCKETS
}

/// One decode step, padded up to its graph batch bucket. Pad rows alias row 0 for every read, skip
/// their KV writes and point at the hybrid pad slot, so the model can run them and their outputs be
/// dropped.
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
        let metadata = rows.build().map_err(candle_core::Error::msg)?;
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
    pub(crate) pad_slot: u32,
    // Allocating the pad slot reallocated the pools, invalidating every captured graph
    pub(crate) grew: bool,
}

/// The batch's live recurrent slots plus the scratch slot pad rows write into.
pub(crate) fn hybrid_graph_slots(cache: &mut HybridCache) -> Option<HybridGraphSlots> {
    let real = cache.state_indices_host()?.to_vec();
    let capacity = cache.recurrent_capacity();
    let pad_slot = cache.graph_pad_slot()?;
    Some(HybridGraphSlots {
        real,
        pad_slot: u32::try_from(pad_slot).ok()?,
        grew: cache.recurrent_capacity() != capacity,
    })
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

fn copy_state_indices(dst: &CudaGraphVarMap, host: &[u32]) -> candle_core::Result<()> {
    for var in dst.values() {
        var.set(&Tensor::from_vec(
            host.to_vec(),
            (host.len(),),
            var.device(),
        )?)?;
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CudaDecodeGraphKey {
    device: DeviceLocation,
    input_shape: Vec<usize>,
    input_dtype: DType,
    max_context_len: Option<usize>,
    full_max_context_len: Option<usize>,
    tensors: Vec<CudaGraphTensorKey>,
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
    pub(crate) block_size: usize,
    pub(crate) kv_cache: &'a [(Tensor, Tensor)],
    pub(crate) metadata: &'a PagedAttentionInputMetadata,
    pub(crate) model_metadata: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
    pub(crate) warmup_logits: &'a Tensor,
    pub(crate) state_indices: Option<CudaGraphVarMap>,
    pub(crate) real_batch: usize,
}

pub(crate) struct CudaDecodeGraphMetadataBuffers {
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
    rope_positions: CudaGraphVarMap,
}

impl CudaDecodeGraphKey {
    pub(crate) fn new(
        input_ids: &Tensor,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> candle_core::Result<Self> {
        let mut tensors = Vec::new();
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
        push_graph_tensor_keys(
            "paged_kv_indptr",
            flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_indices",
            flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_last_page_len",
            flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_indptr",
            flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_indices",
            flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_last_page_len",
            flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_request_indices",
            flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_tile_indices",
            flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_o_indptr",
            flashinfer_paged_view(metadata).map(|view| &view.tile_plan.o_indptr),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_chunk_size",
            flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_block_valid_mask",
            flashinfer_paged_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_request_indices",
            flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_tile_indices",
            flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_o_indptr",
            flashinfer_full_view(metadata).map(|view| &view.tile_plan.o_indptr),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_chunk_size",
            flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_block_valid_mask",
            flashinfer_full_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
            &mut tensors,
        );
        tensors.sort_by(|a, b| {
            a.name.cmp(b.name).then_with(|| {
                device_location_sort_key(&a.location).cmp(&device_location_sort_key(&b.location))
            })
        });

        Ok(Self {
            device: input_ids.device().location(),
            input_shape: input_ids.dims().to_vec(),
            input_dtype: input_ids.dtype(),
            max_context_len: graph_context_len(
                metadata.max_context_len,
                bucket_context_len(metadata.block_tables.as_ref(), block_size),
            ),
            full_max_context_len: graph_context_len(
                metadata.full_max_context_len,
                bucket_context_len(metadata.full_block_tables.as_ref(), block_size),
            ),
            tensors,
        })
    }
}

impl CudaDecodeGraphMetadataBuffers {
    pub(crate) fn new(
        metadata: &PagedAttentionInputMetadata,
        seqlen_offsets: &[usize],
        seq_len: usize,
        block_size: usize,
        kv_cache: &[(Tensor, Tensor)],
        model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
    ) -> candle_core::Result<(Self, PagedAttentionInputMetadata)> {
        let slot_mappings = var_map_from_tensor_map(&metadata.slot_mappings)?;
        let rope_positions =
            rope_positions_var_map(&metadata.slot_mappings, seqlen_offsets, seq_len)?;
        let (flashinfer_decode_tmp_v, flashinfer_decode_tmp_s) = flashinfer_decode_scratch_maps(
            metadata,
            seqlen_offsets.len(),
            kv_cache,
            model_metadata,
        )?;
        let buffers = Self {
            slot_mappings,
            block_tables: option_var_map_from_tensor_map(metadata.block_tables.as_ref())?,
            context_lens: option_var_map_from_tensor_map(metadata.context_lens.as_ref())?,
            full_block_tables: option_var_map_from_tensor_map(metadata.full_block_tables.as_ref())?,
            full_context_lens: option_var_map_from_tensor_map(metadata.full_context_lens.as_ref())?,
            paged_kv_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
            )?,
            paged_kv_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
            )?,
            paged_kv_last_page_len: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
            )?,
            full_paged_kv_indptr: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
            )?,
            full_paged_kv_indices: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
            )?,
            full_paged_kv_last_page_len: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
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
            full_paged_kv_request_indices: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
            )?,
            full_paged_kv_tile_indices: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
            )?,
            full_paged_kv_o_indptr: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.o_indptr),
            )?,
            full_paged_kv_chunk_size: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
            )?,
            full_paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
            )?,
            flashinfer_decode_tmp_v,
            flashinfer_decode_tmp_s,
            rope_positions,
        };
        let metadata = buffers.metadata_from(metadata, block_size);
        Ok((buffers, metadata))
    }

    pub(crate) fn copy_from(
        &mut self,
        metadata: &PagedAttentionInputMetadata,
        seqlen_offsets: &[usize],
        seq_len: usize,
    ) -> candle_core::Result<()> {
        copy_var_map(
            &self.slot_mappings,
            &metadata.slot_mappings,
            "slot_mappings",
        )?;
        copy_option_var_map(
            &self.context_lens,
            metadata.context_lens.as_ref(),
            "context_lens",
        )?;
        copy_option_var_map(
            &self.full_context_lens,
            metadata.full_context_lens.as_ref(),
            "full_context_lens",
        )?;
        copy_option_var_map(
            &self.paged_kv_last_page_len,
            flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
            "paged_kv_last_page_len",
        )?;
        copy_option_var_map(
            &self.full_paged_kv_last_page_len,
            flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
            "full_paged_kv_last_page_len",
        )?;
        {
            copy_option_var_map(
                &self.block_tables,
                metadata.block_tables.as_ref(),
                "block_tables",
            )?;
            copy_option_var_map(
                &self.paged_kv_indptr,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
                "paged_kv_indptr",
            )?;
            copy_option_var_map(
                &self.paged_kv_indices,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
                "paged_kv_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_request_indices,
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
                "paged_kv_request_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_tile_indices,
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
                "paged_kv_tile_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_o_indptr,
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.o_indptr),
                "paged_kv_o_indptr",
            )?;
            copy_option_var_map(
                &self.paged_kv_chunk_size,
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
                "paged_kv_chunk_size",
            )?;
            copy_option_var_map(
                &self.paged_kv_block_valid_mask,
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
                "paged_kv_block_valid_mask",
            )?;
        }
        {
            copy_option_var_map(
                &self.full_block_tables,
                metadata.full_block_tables.as_ref(),
                "full_block_tables",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_indptr,
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                "full_paged_kv_indptr",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_indices,
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                "full_paged_kv_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_request_indices,
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
                "full_paged_kv_request_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_tile_indices,
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
                "full_paged_kv_tile_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_o_indptr,
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.o_indptr),
                "full_paged_kv_o_indptr",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_chunk_size,
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
                "full_paged_kv_chunk_size",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_block_valid_mask,
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
                "full_paged_kv_block_valid_mask",
            )?;
        }
        copy_rope_positions(&self.rope_positions, seqlen_offsets, seq_len)?;
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

        Some(FlashInferMetadata {
            views: FlashInferPagedAttentionViews { logical, sliding },
            decode_tmp_v: self.flashinfer_decode_tmp_v.clone(),
            decode_tmp_s: self.flashinfer_decode_tmp_s.clone(),
        })
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

pub(crate) struct CudaDecodeGraphEntry {
    key: CudaDecodeGraphKey,
    graph: CudaGraphHandle,
    input_ids: Var,
    metadata_buffers: CudaDecodeGraphMetadataBuffers,
    state_indices: Option<CudaGraphVarMap>,
    _metadata: PagedAttentionInputMetadata,
    logits: Tensor,
    // Proposer-facing outputs living in persistent buffers the replay refreshes
    spec_state: Option<Arc<dyn SpeculativeGraphState>>,
}

impl CudaDecodeGraphEntry {
    pub(crate) fn with_spec_state(
        mut self,
        spec_state: Option<Box<dyn SpeculativeGraphState>>,
    ) -> Self {
        self.spec_state = spec_state.map(Arc::from);
        self
    }
}

pub(crate) struct CudaDecodeGraphReplay {
    pub(crate) logits: Tensor,
    pub(crate) spec_state: Option<Arc<dyn SpeculativeGraphState>>,
}

#[derive(Default)]
pub(crate) struct CudaDecodeGraphState {
    entries: Vec<CudaDecodeGraphEntry>,
    disabled: bool,
    suspended: bool,
}

impl CudaDecodeGraphState {
    pub(crate) fn disabled(&self) -> bool {
        self.disabled || self.suspended
    }

    pub(crate) fn disable(&mut self) {
        self.disabled = true;
        self.clear();
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
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
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        let Some(pos) = self.entries.iter().position(|entry| entry.key == *key) else {
            return Ok(None);
        };
        let mut entry = self.entries.remove(pos);
        entry.input_ids.set(&step.input_ids)?;
        let (_, seq_len) = step.input_ids.dims2()?;
        entry
            .metadata_buffers
            .copy_from(&step.metadata, &step.seqlen_offsets, seq_len)?;
        match (&entry.state_indices, &step.state_indices) {
            (Some(dst), Some(host)) => copy_state_indices(dst, host)?,
            (None, None) => {}
            _ => candle_core::bail!(
                "hybrid state indices changed optional state during CUDA graph replay"
            ),
        }
        entry
            .graph
            .launch()
            .map_err(|err| err.context("CUDA graph replay launch failed"))?;
        let replay = CudaDecodeGraphReplay {
            logits: step.narrow_rows(&entry.logits)?,
            spec_state: entry.spec_state.clone(),
        };
        self.entries.push(entry);
        Ok(Some(replay))
    }

    pub(crate) fn insert(&mut self, entry: CudaDecodeGraphEntry) {
        if self.entries.len() >= CUDA_DECODE_GRAPH_CACHE_CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push(entry);
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
        block_size,
        kv_cache,
        metadata,
        model_metadata,
        warmup_logits,
        state_indices,
        real_batch,
    } = ctx;
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids = Var::from_tensor(input_ids)?;
    let (metadata_buffers, metadata) = CudaDecodeGraphMetadataBuffers::new(
        metadata,
        seqlen_offsets,
        seq_len,
        block_size,
        kv_cache,
        model_metadata,
    )?;
    let graph_input_ids = input_ids.as_detached_tensor();
    let graph_logits = unsafe {
        Tensor::empty(
            warmup_logits.shape().clone(),
            warmup_logits.dtype(),
            warmup_logits.device(),
        )?
    };
    let Device::Cuda(cuda_device) = graph_input_ids.device() else {
        candle_core::bail!("CUDA graph decode expected CUDA input ids");
    };
    graph_input_ids.device().synchronize()?;
    let stream = cuda_device.cuda_stream();
    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();

    if let Err(err) = stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
    {
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(
            candle_core::Error::msg(err.to_string()).context("CUDA graph begin capture failed")
        );
    }

    let logits = match forward(&graph_input_ids, &metadata) {
        Ok(logits) => logits,
        Err(err) => {
            end_cuda_capture_discard(&stream);
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(err.context("CUDA graph captured forward failed"));
        }
    };
    if let Err(err) = crate::cuda::graph::copy_tensor(&logits, &graph_logits) {
        end_cuda_capture_discard(&stream);
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(err.context("CUDA graph output copy capture failed"));
    }
    drop(logits);

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
    tracing::debug!(
        "Captured CUDA decode graph: batch bucket {batch} ({real_batch} live rows), {seq_len} query tokens"
    );

    Ok(CudaDecodeGraphEntry {
        key,
        graph,
        input_ids,
        metadata_buffers,
        state_indices,
        _metadata: metadata,
        logits: graph_logits,
        spec_state: None,
    })
}

pub(crate) fn cuda_decode_graphs_enabled() -> bool {
    crate::perf_flags::cuda_graphs_enabled()
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

pub(crate) fn prepare_cuda_graph_memory_pool(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
    if !stream.context().has_async_alloc() {
        return Ok(());
    }

    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let dev = stream.context().cu_device();
    let mut pool = std::ptr::null_mut();
    let result = unsafe { sys::cuDeviceGetMemPool(&mut pool, dev) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool lookup failed"));
    }

    let mut release_threshold = u64::MAX;
    let result = unsafe {
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut release_threshold as *mut u64).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool release threshold setup failed"));
    }

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

    Ok(())
}

fn flashinfer_decode_scratch_maps(
    metadata: &PagedAttentionInputMetadata,
    batch: usize,
    kv_cache: &[(Tensor, Tensor)],
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
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
            key_cache.dtype(),
            num_qo_heads,
            head_dim,
        ));
        if entry.1 != key_cache.dtype() {
            candle_core::bail!("FlashInfer graph scratch expects one dtype per device");
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

fn copy_var_map(
    dst: &CudaGraphVarMap,
    src: &HashMap<DeviceLocation, Tensor>,
    name: &str,
) -> candle_core::Result<()> {
    if dst.len() != src.len() {
        candle_core::bail!("{name} device count changed during CUDA graph replay");
    }
    for (location, dst) in dst {
        let src = src
            .get(location)
            .ok_or_else(|| candle_core::Error::msg(format!("{name} missing {location:?}")))?;
        dst.set(src)?;
    }
    Ok(())
}

fn copy_option_var_map(
    dst: &Option<CudaGraphVarMap>,
    src: Option<&HashMap<DeviceLocation, Tensor>>,
    name: &str,
) -> candle_core::Result<()> {
    match (dst, src) {
        (Some(dst), Some(src)) => copy_var_map(dst, src, name),
        (None, None) => Ok(()),
        _ => candle_core::bail!("{name} changed optional state during CUDA graph replay"),
    }
}

fn rope_positions_var_map(
    slot_mappings: &HashMap<DeviceLocation, Tensor>,
    seqlen_offsets: &[usize],
    seq_len: usize,
) -> candle_core::Result<CudaGraphVarMap> {
    slot_mappings
        .iter()
        .map(|(location, tensor)| {
            let positions = text_positions_tensor(seqlen_offsets, seq_len, tensor.device())?;
            Ok((*location, Var::from_tensor(&positions)?))
        })
        .collect()
}

fn copy_rope_positions(
    dst: &CudaGraphVarMap,
    seqlen_offsets: &[usize],
    seq_len: usize,
) -> candle_core::Result<()> {
    for dst in dst.values() {
        let positions = text_positions_tensor(seqlen_offsets, seq_len, dst.device())?;
        dst.set(&positions)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
