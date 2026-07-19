use std::collections::HashMap;

#[cfg(all(feature = "cuda", target_family = "unix"))]
use candle_core::Result;
use candle_core::{DeviceLocation, Tensor};

use crate::paged_attention::attention_backend::{
    AttentionBackend, AttentionBackendKind, AttentionLayerSpec,
};

mod metadata;
pub(crate) use metadata::{
    decode_split_pages, flashinfer_metadata, flashinfer_paged_kv, flashinfer_tile_plan,
    flashinfer_view, make_paged_kv_decode_tensors, make_paged_kv_decode_tensors_from_lens,
    make_paged_kv_tensors,
};

// Metadata is copied per CUDA device; graph replay may substitute graph-owned tensors.
pub type DeviceTensorMap = HashMap<DeviceLocation, Tensor>;

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE: usize = 512;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_DECODE_MAX_HEAD_SIZE: usize = 512;

#[derive(Clone, Debug)]
pub struct FlashInferPagedKv {
    // CSR-style page table: indptr selects each request's range in flattened page indices.
    pub indptr: DeviceTensorMap,
    pub indices: DeviceTensorMap,
    pub last_page_len: DeviceTensorMap,
}

#[derive(Clone, Debug)]
pub struct FlashInferTilePlan {
    // Split-KV decode work queue metadata.
    pub request_indices: DeviceTensorMap,
    pub kv_tile_indices: DeviceTensorMap,
    pub o_indptr: DeviceTensorMap,
    pub kv_chunk_size: DeviceTensorMap,
    pub block_valid_mask: DeviceTensorMap,
}

#[derive(Clone, Debug)]
pub struct FlashInferPagedAttentionView {
    // One KV view: logical full-context metadata, or a decode-only sliding-window view.
    pub block_tables: Option<DeviceTensorMap>,
    pub context_lens: Option<DeviceTensorMap>,
    pub max_context_len: Option<usize>,
    pub paged_kv: FlashInferPagedKv,
    pub tile_plan: FlashInferTilePlan,
}

#[derive(Clone, Debug)]
pub struct FlashInferPagedAttentionViews {
    // Decode selects sliding when the active layer is windowed.
    pub logical: FlashInferPagedAttentionView,
    pub sliding: Option<FlashInferPagedAttentionView>,
}

#[derive(Clone, Debug)]
pub struct FlashInferMetadata {
    pub views: FlashInferPagedAttentionViews,
    pub decode_tmp_v: Option<DeviceTensorMap>,
    pub decode_tmp_s: Option<DeviceTensorMap>,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) struct FlashInferDecodeMetadata<'a> {
    pub paged_kv_indptr: &'a Tensor,
    pub paged_kv_indices: &'a Tensor,
    pub paged_kv_last_page_len: &'a Tensor,
    pub request_indices: &'a Tensor,
    pub kv_tile_indices: &'a Tensor,
    pub o_indptr: &'a Tensor,
    pub kv_chunk_size: &'a Tensor,
    pub block_valid_mask: &'a Tensor,
    pub tmp_v: Option<&'a Tensor>,
    pub tmp_s: Option<&'a Tensor>,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlashInferDecodePlan;

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl FlashInferDecodePlan {
    pub fn head_size_limit(kind: AttentionBackendKind) -> usize {
        match kind {
            AttentionBackendKind::FlashInfer => FLASHINFER_DECODE_MAX_HEAD_SIZE,
            AttentionBackendKind::Standard => STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE,
        }
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) struct FlashInferDecodePlanInput {
    pub head_size: usize,
    pub has_alibi: bool,
    pub has_sinks: bool,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn decode_plan(input: FlashInferDecodePlanInput) -> Result<FlashInferDecodePlan> {
    // Decode can fall back for size limits, but unsupported attention features are hard errors.
    if input.has_alibi || input.has_sinks {
        candle_core::bail!("FlashInfer paged attention does not support alibi/sinks");
    }
    if input.head_size > FLASHINFER_DECODE_MAX_HEAD_SIZE {
        candle_core::bail!(
            "FlashInfer decode does not support head_size={}",
            input.head_size
        );
    }
    Ok(FlashInferDecodePlan)
}

pub struct FlashInferAttentionBackend;

impl AttentionBackend for FlashInferAttentionBackend {
    fn kind(&self) -> AttentionBackendKind {
        AttentionBackendKind::FlashInfer
    }

    fn supports_layer(&self, spec: AttentionLayerSpec) -> bool {
        if !cfg!(feature = "cuda") || !crate::perf_flags::flashinfer_decode_enabled() {
            return false;
        }
        spec.k_head_dim == spec.v_head_dim
            && matches!(spec.k_head_dim, 64 | 128 | 256 | 512)
            && supports_flashinfer_group_size(spec.q_heads, spec.kv_heads)
    }
}

fn supports_flashinfer_group_size(q_heads: usize, kv_heads: usize) -> bool {
    kv_heads != 0 && q_heads.is_multiple_of(kv_heads) && q_heads / kv_heads <= 8
}

impl FlashInferPagedAttentionViews {
    pub fn select(&self, sliding_window: Option<usize>) -> &FlashInferPagedAttentionView {
        if sliding_window.is_some() {
            self.sliding.as_ref().unwrap_or(&self.logical)
        } else {
            &self.logical
        }
    }
}

impl FlashInferMetadata {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub(crate) fn decode_metadata(
        &self,
        device: &DeviceLocation,
        sliding_window: Option<usize>,
    ) -> Result<FlashInferDecodeMetadata<'_>> {
        let view = self.views.select(sliding_window);
        Ok(FlashInferDecodeMetadata {
            paged_kv_indptr: metadata_tensor(&view.paged_kv.indptr, device, "paged_kv_indptr")?,
            paged_kv_indices: metadata_tensor(&view.paged_kv.indices, device, "paged_kv_indices")?,
            paged_kv_last_page_len: metadata_tensor(
                &view.paged_kv.last_page_len,
                device,
                "paged_kv_last_page_len",
            )?,
            request_indices: metadata_tensor(
                &view.tile_plan.request_indices,
                device,
                "paged_kv_request_indices",
            )?,
            kv_tile_indices: metadata_tensor(
                &view.tile_plan.kv_tile_indices,
                device,
                "paged_kv_tile_indices",
            )?,
            o_indptr: metadata_tensor(&view.tile_plan.o_indptr, device, "paged_kv_o_indptr")?,
            kv_chunk_size: metadata_tensor(
                &view.tile_plan.kv_chunk_size,
                device,
                "paged_kv_chunk_size",
            )?,
            block_valid_mask: metadata_tensor(
                &view.tile_plan.block_valid_mask,
                device,
                "paged_kv_block_valid_mask",
            )?,
            tmp_v: self
                .decode_tmp_v
                .as_ref()
                .and_then(|tensors| tensors.get(device)),
            tmp_s: self
                .decode_tmp_s
                .as_ref()
                .and_then(|tensors| tensors.get(device)),
        })
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn metadata_tensor<'a>(
    map: &'a DeviceTensorMap,
    device: &DeviceLocation,
    name: &'static str,
) -> Result<&'a Tensor> {
    map.get(device)
        .ok_or_else(|| candle_core::Error::msg(format!("{name} missing")))
}
