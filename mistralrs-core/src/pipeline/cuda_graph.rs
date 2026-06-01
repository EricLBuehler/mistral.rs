use std::{collections::HashMap, sync::Arc};

use candle_core::cuda_backend::cudarc::driver::{sys, CudaStream};
use candle_core::{DType, Device, DeviceLocation, Tensor, Var};

use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;

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
    rope_positions: CudaGraphVarMap,
    block_table_signature: Option<Vec<u64>>,
    full_block_table_signature: Option<Vec<u64>>,
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
            metadata.paged_kv_indptr.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_indices",
            metadata.paged_kv_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_last_page_len",
            metadata.paged_kv_last_page_len.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_indptr",
            metadata.full_paged_kv_indptr.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_indices",
            metadata.full_paged_kv_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_last_page_len",
            metadata.full_paged_kv_last_page_len.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_request_indices",
            metadata.paged_kv_request_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_tile_indices",
            metadata.paged_kv_tile_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_o_indptr",
            metadata.paged_kv_o_indptr.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_chunk_size",
            metadata.paged_kv_chunk_size.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "paged_kv_block_valid_mask",
            metadata.paged_kv_block_valid_mask.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_request_indices",
            metadata.full_paged_kv_request_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_tile_indices",
            metadata.full_paged_kv_tile_indices.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_o_indptr",
            metadata.full_paged_kv_o_indptr.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_chunk_size",
            metadata.full_paged_kv_chunk_size.as_ref(),
            &mut tensors,
        );
        push_graph_tensor_keys(
            "full_paged_kv_block_valid_mask",
            metadata.full_paged_kv_block_valid_mask.as_ref(),
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
            max_context_len: bucket_context_len(metadata.block_tables.as_ref(), block_size),
            full_max_context_len: bucket_context_len(
                metadata.full_block_tables.as_ref(),
                block_size,
            ),
            tensors,
        })
    }
}

impl CudaDecodeGraphMetadataBuffers {
    pub(crate) fn new(
        metadata: &PagedAttentionInputMetadata,
        seqlen_offsets: &[usize],
        block_size: usize,
    ) -> candle_core::Result<(Self, PagedAttentionInputMetadata)> {
        let slot_mappings = var_map_from_tensor_map(&metadata.slot_mappings)?;
        let rope_positions = rope_positions_var_map(&metadata.slot_mappings, seqlen_offsets)?;
        let buffers = Self {
            slot_mappings,
            block_tables: option_var_map_from_tensor_map(metadata.block_tables.as_ref())?,
            context_lens: option_var_map_from_tensor_map(metadata.context_lens.as_ref())?,
            full_block_tables: option_var_map_from_tensor_map(metadata.full_block_tables.as_ref())?,
            full_context_lens: option_var_map_from_tensor_map(metadata.full_context_lens.as_ref())?,
            paged_kv_indptr: option_var_map_from_tensor_map(metadata.paged_kv_indptr.as_ref())?,
            paged_kv_indices: option_var_map_from_tensor_map(metadata.paged_kv_indices.as_ref())?,
            paged_kv_last_page_len: option_var_map_from_tensor_map(
                metadata.paged_kv_last_page_len.as_ref(),
            )?,
            full_paged_kv_indptr: option_var_map_from_tensor_map(
                metadata.full_paged_kv_indptr.as_ref(),
            )?,
            full_paged_kv_indices: option_var_map_from_tensor_map(
                metadata.full_paged_kv_indices.as_ref(),
            )?,
            full_paged_kv_last_page_len: option_var_map_from_tensor_map(
                metadata.full_paged_kv_last_page_len.as_ref(),
            )?,
            paged_kv_request_indices: option_var_map_from_tensor_map(
                metadata.paged_kv_request_indices.as_ref(),
            )?,
            paged_kv_tile_indices: option_var_map_from_tensor_map(
                metadata.paged_kv_tile_indices.as_ref(),
            )?,
            paged_kv_o_indptr: option_var_map_from_tensor_map(metadata.paged_kv_o_indptr.as_ref())?,
            paged_kv_chunk_size: option_var_map_from_tensor_map(
                metadata.paged_kv_chunk_size.as_ref(),
            )?,
            paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                metadata.paged_kv_block_valid_mask.as_ref(),
            )?,
            full_paged_kv_request_indices: option_var_map_from_tensor_map(
                metadata.full_paged_kv_request_indices.as_ref(),
            )?,
            full_paged_kv_tile_indices: option_var_map_from_tensor_map(
                metadata.full_paged_kv_tile_indices.as_ref(),
            )?,
            full_paged_kv_o_indptr: option_var_map_from_tensor_map(
                metadata.full_paged_kv_o_indptr.as_ref(),
            )?,
            full_paged_kv_chunk_size: option_var_map_from_tensor_map(
                metadata.full_paged_kv_chunk_size.as_ref(),
            )?,
            full_paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                metadata.full_paged_kv_block_valid_mask.as_ref(),
            )?,
            rope_positions,
            block_table_signature: metadata.block_table_signature.clone(),
            full_block_table_signature: metadata.full_block_table_signature.clone(),
        };
        let metadata = buffers.metadata_from(metadata, block_size);
        Ok((buffers, metadata))
    }

    pub(crate) fn copy_from(
        &mut self,
        metadata: &PagedAttentionInputMetadata,
        seqlen_offsets: &[usize],
    ) -> candle_core::Result<()> {
        let block_tables_changed =
            signature_changed(&self.block_table_signature, &metadata.block_table_signature);
        let full_block_tables_changed = signature_changed(
            &self.full_block_table_signature,
            &metadata.full_block_table_signature,
        );

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
            metadata.paged_kv_last_page_len.as_ref(),
            "paged_kv_last_page_len",
        )?;
        copy_option_var_map(
            &self.full_paged_kv_last_page_len,
            metadata.full_paged_kv_last_page_len.as_ref(),
            "full_paged_kv_last_page_len",
        )?;
        if block_tables_changed {
            copy_option_var_map(
                &self.block_tables,
                metadata.block_tables.as_ref(),
                "block_tables",
            )?;
            copy_option_var_map(
                &self.paged_kv_indptr,
                metadata.paged_kv_indptr.as_ref(),
                "paged_kv_indptr",
            )?;
            copy_option_var_map(
                &self.paged_kv_indices,
                metadata.paged_kv_indices.as_ref(),
                "paged_kv_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_request_indices,
                metadata.paged_kv_request_indices.as_ref(),
                "paged_kv_request_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_tile_indices,
                metadata.paged_kv_tile_indices.as_ref(),
                "paged_kv_tile_indices",
            )?;
            copy_option_var_map(
                &self.paged_kv_o_indptr,
                metadata.paged_kv_o_indptr.as_ref(),
                "paged_kv_o_indptr",
            )?;
            copy_option_var_map(
                &self.paged_kv_chunk_size,
                metadata.paged_kv_chunk_size.as_ref(),
                "paged_kv_chunk_size",
            )?;
            copy_option_var_map(
                &self.paged_kv_block_valid_mask,
                metadata.paged_kv_block_valid_mask.as_ref(),
                "paged_kv_block_valid_mask",
            )?;
            self.block_table_signature = metadata.block_table_signature.clone();
        }
        if full_block_tables_changed {
            copy_option_var_map(
                &self.full_block_tables,
                metadata.full_block_tables.as_ref(),
                "full_block_tables",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_indptr,
                metadata.full_paged_kv_indptr.as_ref(),
                "full_paged_kv_indptr",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_indices,
                metadata.full_paged_kv_indices.as_ref(),
                "full_paged_kv_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_request_indices,
                metadata.full_paged_kv_request_indices.as_ref(),
                "full_paged_kv_request_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_tile_indices,
                metadata.full_paged_kv_tile_indices.as_ref(),
                "full_paged_kv_tile_indices",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_o_indptr,
                metadata.full_paged_kv_o_indptr.as_ref(),
                "full_paged_kv_o_indptr",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_chunk_size,
                metadata.full_paged_kv_chunk_size.as_ref(),
                "full_paged_kv_chunk_size",
            )?;
            copy_option_var_map(
                &self.full_paged_kv_block_valid_mask,
                metadata.full_paged_kv_block_valid_mask.as_ref(),
                "full_paged_kv_block_valid_mask",
            )?;
            self.full_block_table_signature = metadata.full_block_table_signature.clone();
        }
        copy_rope_positions(&self.rope_positions, seqlen_offsets)?;
        Ok(())
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
            disable_cuda_graphs: metadata.disable_cuda_graphs,
            paged_kv_indptr: option_tensor_map_from_var_map(&self.paged_kv_indptr),
            paged_kv_indices: option_tensor_map_from_var_map(&self.paged_kv_indices),
            paged_kv_last_page_len: option_tensor_map_from_var_map(&self.paged_kv_last_page_len),
            full_paged_kv_indptr: option_tensor_map_from_var_map(&self.full_paged_kv_indptr),
            full_paged_kv_indices: option_tensor_map_from_var_map(&self.full_paged_kv_indices),
            full_paged_kv_last_page_len: option_tensor_map_from_var_map(
                &self.full_paged_kv_last_page_len,
            ),
            paged_kv_q_indptr: metadata.paged_kv_q_indptr.clone(),
            paged_kv_qo_tile_indices: metadata.paged_kv_qo_tile_indices.clone(),
            paged_kv_request_indices: option_tensor_map_from_var_map(
                &self.paged_kv_request_indices,
            ),
            paged_kv_tile_indices: option_tensor_map_from_var_map(&self.paged_kv_tile_indices),
            paged_kv_o_indptr: option_tensor_map_from_var_map(&self.paged_kv_o_indptr),
            paged_kv_chunk_size: option_tensor_map_from_var_map(&self.paged_kv_chunk_size),
            paged_kv_block_valid_mask: option_tensor_map_from_var_map(
                &self.paged_kv_block_valid_mask,
            ),
            block_table_signature: self.block_table_signature.clone(),
            full_paged_kv_q_indptr: metadata.full_paged_kv_q_indptr.clone(),
            full_paged_kv_qo_tile_indices: metadata.full_paged_kv_qo_tile_indices.clone(),
            full_paged_kv_request_indices: option_tensor_map_from_var_map(
                &self.full_paged_kv_request_indices,
            ),
            full_paged_kv_tile_indices: option_tensor_map_from_var_map(
                &self.full_paged_kv_tile_indices,
            ),
            full_paged_kv_o_indptr: option_tensor_map_from_var_map(&self.full_paged_kv_o_indptr),
            full_paged_kv_chunk_size: option_tensor_map_from_var_map(
                &self.full_paged_kv_chunk_size,
            ),
            full_paged_kv_block_valid_mask: option_tensor_map_from_var_map(
                &self.full_paged_kv_block_valid_mask,
            ),
            full_block_table_signature: self.full_block_table_signature.clone(),
            rope_positions: Some(tensor_map_from_var_map(&self.rope_positions)),
            num_cached_tokens: metadata.num_cached_tokens.clone(),
            query_lens: metadata.query_lens.clone(),
            cu_seqlens_q: metadata.cu_seqlens_q.clone(),
            cu_seqlens_kv: metadata.cu_seqlens_kv.clone(),
        }
    }
}

pub(crate) fn cuda_decode_graphs_enabled() -> bool {
    crate::perf_flags::cuda_graphs_enabled()
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
        let _ = CudaGraphHandle::end_capture(stream);
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

fn bucket_context_len(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    block_size: usize,
) -> Option<usize> {
    map.and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

fn bucket_context_len_from_vars(map: &Option<CudaGraphVarMap>, block_size: usize) -> Option<usize> {
    map.as_ref()
        .and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

fn signature_changed(previous: &Option<Vec<u64>>, current: &Option<Vec<u64>>) -> bool {
    match (previous, current) {
        (Some(previous), Some(current)) => previous != current,
        _ => true,
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

fn rope_positions_tensor(seqlen_offsets: &[usize], device: &Device) -> candle_core::Result<Tensor> {
    let mut positions = Vec::with_capacity(seqlen_offsets.len());
    for offset in seqlen_offsets {
        positions.push(u32::try_from(*offset).map_err(candle_core::Error::wrap)?);
    }
    Tensor::from_vec(positions, (seqlen_offsets.len(),), device)
}

fn rope_positions_var_map(
    slot_mappings: &HashMap<DeviceLocation, Tensor>,
    seqlen_offsets: &[usize],
) -> candle_core::Result<CudaGraphVarMap> {
    slot_mappings
        .iter()
        .map(|(location, tensor)| {
            let positions = rope_positions_tensor(seqlen_offsets, tensor.device())?;
            Ok((*location, Var::from_tensor(&positions)?))
        })
        .collect()
}

fn copy_rope_positions(dst: &CudaGraphVarMap, seqlen_offsets: &[usize]) -> candle_core::Result<()> {
    for dst in dst.values() {
        let positions = rope_positions_tensor(seqlen_offsets, dst.device())?;
        dst.set(&positions)?;
    }
    Ok(())
}
