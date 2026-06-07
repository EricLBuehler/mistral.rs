use std::{collections::HashMap, sync::Arc};

use candle_core::cuda_backend::cudarc::driver::{sys, CudaStream};
use candle_core::{DType, Device, DeviceLocation, Tensor, Var};

use crate::{
    flashinfer::{
        FlashInferMetadata, FlashInferPagedAttentionView, FlashInferPagedAttentionViews,
        FlashInferPagedKv, FlashInferTilePlan,
    },
    paged_attention::{AttentionBackendKind, ModelConfigLike},
};

use crate::pipeline::{
    text_models_inputs_processor::PagedAttentionInputMetadata, text_positions_tensor,
};

const CUDA_GRAPH_INSTANTIATE_FLAGS: u64 =
    sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64;
pub(crate) const CUDA_DECODE_GRAPH_CACHE_CAPACITY: usize = 32;

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
    tensors: Vec<CudaGraphTensorKey>,
    state_key: Option<Vec<u32>>,
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
    pub(crate) retained_tensors: Vec<Tensor>,
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
    paged_kv_q_indptr: Option<CudaGraphVarMap>,
    paged_kv_qo_tile_indices: Option<CudaGraphVarMap>,
    paged_kv_request_indices: Option<CudaGraphVarMap>,
    paged_kv_tile_indices: Option<CudaGraphVarMap>,
    paged_kv_o_indptr: Option<CudaGraphVarMap>,
    paged_kv_chunk_size: Option<CudaGraphVarMap>,
    paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    full_paged_kv_q_indptr: Option<CudaGraphVarMap>,
    full_paged_kv_qo_tile_indices: Option<CudaGraphVarMap>,
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
        _block_size: usize,
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
            "paged_kv_q_indptr",
            flashinfer_paged_view(metadata).map(|view| &view.tile_plan.q_indptr),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_qo_tile_indices",
            flashinfer_paged_view(metadata).map(|view| &view.tile_plan.qo_tile_indices),
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
            "full_paged_kv_q_indptr",
            flashinfer_full_view(metadata).map(|view| &view.tile_plan.q_indptr),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_qo_tile_indices",
            flashinfer_full_view(metadata).map(|view| &view.tile_plan.qo_tile_indices),
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
            tensors,
            state_key: None,
        })
    }

    pub(crate) fn with_state_key(mut self, state_key: Option<Vec<u32>>) -> Self {
        self.state_key = state_key;
        self
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
            paged_kv_q_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.q_indptr),
            )?,
            paged_kv_qo_tile_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.qo_tile_indices),
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
            full_paged_kv_q_indptr: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.q_indptr),
            )?,
            full_paged_kv_qo_tile_indices: option_var_map_from_tensor_map(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.qo_tile_indices),
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
                &self.paged_kv_q_indptr,
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.q_indptr),
                "paged_kv_q_indptr",
            )?;
            copy_option_var_map(
                &self.paged_kv_qo_tile_indices,
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.qo_tile_indices),
                "paged_kv_qo_tile_indices",
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
                &self.full_paged_kv_q_indptr,
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.q_indptr),
                "full_paged_kv_q_indptr",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_qo_tile_indices,
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.qo_tile_indices),
                "full_paged_kv_qo_tile_indices",
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
                &self.full_paged_kv_q_indptr,
                &self.full_paged_kv_qo_tile_indices,
                &self.full_paged_kv_request_indices,
                &self.full_paged_kv_tile_indices,
                &self.full_paged_kv_o_indptr,
                &self.full_paged_kv_chunk_size,
                &self.full_paged_kv_block_valid_mask,
            )?,
            prefill_tile_plan: original.views.logical.prefill_tile_plan.clone(),
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
                    &self.paged_kv_q_indptr,
                    &self.paged_kv_qo_tile_indices,
                    &self.paged_kv_request_indices,
                    &self.paged_kv_tile_indices,
                    &self.paged_kv_o_indptr,
                    &self.paged_kv_chunk_size,
                    &self.paged_kv_block_valid_mask,
                )?,
                prefill_tile_plan: view.prefill_tile_plan.clone(),
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
            max_context_len: bucket_context_len_from_vars(&self.block_tables, block_size),
            full_block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            full_context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            full_max_context_len: bucket_context_len_from_vars(&self.full_block_tables, block_size),
            is_first_prompt_chunk: metadata.is_first_prompt_chunk,
            prompt_chunk_attention_policy: metadata.prompt_chunk_attention_policy,
            has_noncausal_mm_context: metadata.has_noncausal_mm_context,
            disable_cuda_graphs: metadata.disable_cuda_graphs,
            prefill_attention_heads: metadata.prefill_attention_heads,
            prefill_key_value_heads: metadata.prefill_key_value_heads,
            prefill_head_dim: metadata.prefill_head_dim,
            flashinfer: self.flashinfer_metadata_from(metadata, block_size),
            rope_positions: Some(tensor_map_from_var_map(&self.rope_positions)),
            num_cached_tokens: metadata.num_cached_tokens.clone(),
            query_lens: metadata.query_lens.clone(),
            cu_seqlens_q: metadata.cu_seqlens_q.clone(),
            cu_seqlens_kv: metadata.cu_seqlens_kv.clone(),
        }
    }
}

pub(crate) struct CudaDecodeGraphEntry {
    key: CudaDecodeGraphKey,
    graph: CudaGraphHandle,
    input_ids: Var,
    metadata_buffers: CudaDecodeGraphMetadataBuffers,
    _metadata: PagedAttentionInputMetadata,
    _retained_tensors: Vec<Tensor>,
    logits: Tensor,
}

#[derive(Default)]
pub(crate) struct CudaDecodeGraphState {
    entries: Vec<CudaDecodeGraphEntry>,
    disabled: bool,
}

impl CudaDecodeGraphState {
    pub(crate) fn disabled(&self) -> bool {
        self.disabled
    }

    pub(crate) fn disable(&mut self) {
        self.disabled = true;
        self.clear();
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    pub(crate) fn replay(
        &mut self,
        key: &CudaDecodeGraphKey,
        input_ids: &Tensor,
        metadata: &PagedAttentionInputMetadata,
        seqlen_offsets: &[usize],
    ) -> candle_core::Result<Option<Tensor>> {
        let Some(pos) = self.entries.iter().position(|entry| entry.key == *key) else {
            return Ok(None);
        };
        let mut entry = self.entries.remove(pos);
        entry.input_ids.set(input_ids)?;
        let (_, seq_len) = input_ids.dims2()?;
        entry
            .metadata_buffers
            .copy_from(metadata, seqlen_offsets, seq_len)?;
        entry
            .graph
            .launch()
            .map_err(|err| err.context("CUDA graph replay launch failed"))?;
        let logits = entry.logits.clone();
        self.entries.push(entry);
        Ok(Some(logits))
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
        retained_tensors,
    } = ctx;
    let (_, seq_len) = input_ids.dims2()?;
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

    Ok(CudaDecodeGraphEntry {
        key,
        graph,
        input_ids,
        metadata_buffers,
        _metadata: metadata,
        _retained_tensors: retained_tensors,
        logits: graph_logits,
    })
}

pub(crate) fn cuda_decode_graphs_enabled() -> bool {
    crate::perf_flags::cuda_graphs_enabled()
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
    q_indptr: &Option<CudaGraphVarMap>,
    qo_tile_indices: &Option<CudaGraphVarMap>,
    request_indices: &Option<CudaGraphVarMap>,
    kv_tile_indices: &Option<CudaGraphVarMap>,
    o_indptr: &Option<CudaGraphVarMap>,
    kv_chunk_size: &Option<CudaGraphVarMap>,
    block_valid_mask: &Option<CudaGraphVarMap>,
) -> Option<FlashInferTilePlan> {
    Some(FlashInferTilePlan {
        q_indptr: option_tensor_map_from_var_map(q_indptr)?,
        qo_tile_indices: option_tensor_map_from_var_map(qo_tile_indices)?,
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
