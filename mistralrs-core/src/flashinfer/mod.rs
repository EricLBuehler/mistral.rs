use std::collections::HashMap;

#[cfg(all(feature = "cuda", target_family = "unix"))]
use candle_core::{DType, Result};
use candle_core::{DeviceLocation, Tensor};

use crate::paged_attention::attention_backend::{
    AttentionBackend, AttentionBackendKind, AttentionLayerSpec,
};

mod metadata;
mod tiling;
pub(crate) use metadata::{
    decode_split_pages, flashinfer_metadata, flashinfer_paged_kv, flashinfer_tile_plan,
    flashinfer_view, make_decode_q_tensors, make_paged_kv_decode_tensors,
    make_paged_kv_decode_tensors_from_lens, make_paged_kv_prefill_tensors, make_paged_kv_tensors,
};
pub(crate) use tiling::FlashInferPrefillTiling;

// Metadata is copied per CUDA device; graph replay may substitute graph-owned tensors.
pub type DeviceTensorMap = HashMap<DeviceLocation, Tensor>;

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE: usize = 512;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_PREFILL_MAX_HEAD_SIZE: usize = 256;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_DECODE_MAX_HEAD_SIZE: usize = 512;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_TENSOR_CORE_DECODE_ENABLED: bool = false;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE: usize = 256;

#[derive(Clone, Debug)]
pub struct FlashInferPagedKv {
    // CSR-style page table: indptr selects each request's range in flattened page indices.
    pub indptr: DeviceTensorMap,
    pub indices: DeviceTensorMap,
    pub last_page_len: DeviceTensorMap,
}

#[derive(Clone, Debug)]
pub struct FlashInferTilePlan {
    // Work queue metadata: request id plus QO/KV tile coordinates for each FlashInfer tile.
    pub q_indptr: DeviceTensorMap,
    pub qo_tile_indices: DeviceTensorMap,
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
    pub prefill_tile_plan: FlashInferTilePlan,
}

#[derive(Clone, Debug)]
pub struct FlashInferPagedAttentionViews {
    // Prefill always uses logical; decode selects sliding when the active layer is windowed.
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
    pub q_indptr: Option<&'a Tensor>,
    pub qo_tile_indices: Option<&'a Tensor>,
    pub request_indices: &'a Tensor,
    pub kv_tile_indices: &'a Tensor,
    pub o_indptr: &'a Tensor,
    pub kv_chunk_size: &'a Tensor,
    pub block_valid_mask: &'a Tensor,
    pub tmp_v: Option<&'a Tensor>,
    pub tmp_s: Option<&'a Tensor>,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) struct FlashInferPrefillMetadata<'a> {
    pub paged_kv_indptr: &'a Tensor,
    pub paged_kv_indices: &'a Tensor,
    pub paged_kv_last_page_len: &'a Tensor,
    pub q_indptr: &'a Tensor,
    pub request_indices: &'a Tensor,
    pub qo_tile_indices: &'a Tensor,
    pub kv_tile_indices: &'a Tensor,
    pub o_indptr: &'a Tensor,
    pub kv_chunk_size: &'a Tensor,
    pub block_valid_mask: &'a Tensor,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlashInferPrefillPlan {
    causal: bool,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl FlashInferPrefillPlan {
    pub fn causal(self) -> bool {
        self.causal
    }

    fn supports_head_dim(head_size: usize) -> bool {
        head_size <= FLASHINFER_PREFILL_MAX_HEAD_SIZE
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) struct FlashInferPrefillPlanInput {
    pub device_is_cuda: bool,
    pub dtype: DType,
    pub has_sinks: bool,
    pub causal: bool,
    pub causality_known: bool,
    pub head_size: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub query_lens_match_seq_len: bool,
    pub attention_backend: AttentionBackendKind,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn prefill_plan(input: FlashInferPrefillPlanInput) -> Option<FlashInferPrefillPlan> {
    // FlashInfer prefill writes directly into paged KV, so only fully causal batches are eligible.
    (input.device_is_cuda
        && input.dtype != DType::F32
        && !input.has_sinks
        && input.causality_known
        && input.causal
        && FlashInferPrefillPlan::supports_head_dim(input.head_size)
        && FlashInferPrefillTiling::supports_group_size(
            input.attention_heads,
            input.key_value_heads,
        )
        && input.query_lens_match_seq_len
        && input.attention_backend == AttentionBackendKind::FlashInfer)
        .then_some(FlashInferPrefillPlan {
            causal: input.causal,
        })
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlashInferDecodePlan {
    use_tensor_cores: bool,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl FlashInferDecodePlan {
    pub fn use_tensor_cores(self) -> bool {
        self.use_tensor_cores
    }

    pub fn head_size_limit(kind: AttentionBackendKind) -> usize {
        match kind {
            AttentionBackendKind::FlashInfer => FLASHINFER_DECODE_MAX_HEAD_SIZE,
            AttentionBackendKind::Standard => STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE,
        }
    }

    fn use_tensor_core_decode(head_size: usize, dtype: DType) -> bool {
        FLASHINFER_TENSOR_CORE_DECODE_ENABLED
            && head_size <= FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE
            && dtype != DType::F32
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) struct FlashInferDecodePlanInput {
    pub dtype: DType,
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
    Ok(FlashInferDecodePlan {
        use_tensor_cores: FlashInferDecodePlan::use_tensor_core_decode(
            input.head_size,
            input.dtype,
        ),
    })
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
            && FlashInferPrefillTiling::supports_group_size(spec.q_heads, spec.kv_heads)
    }
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
        use_tensor_cores: bool,
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
            q_indptr: if use_tensor_cores {
                Some(metadata_tensor(
                    &view.tile_plan.q_indptr,
                    device,
                    "paged_kv_q_indptr",
                )?)
            } else {
                None
            },
            qo_tile_indices: if use_tensor_cores {
                Some(metadata_tensor(
                    &view.tile_plan.qo_tile_indices,
                    device,
                    "paged_kv_qo_tile_indices",
                )?)
            } else {
                None
            },
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

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub(crate) fn prefill_metadata(
        &self,
        device: &DeviceLocation,
    ) -> Result<FlashInferPrefillMetadata<'_>> {
        // Prefill keeps absolute token coordinates; window_left applies sliding masks inside FlashInfer.
        let view = &self.views.logical;
        Ok(FlashInferPrefillMetadata {
            paged_kv_indptr: metadata_tensor(&view.paged_kv.indptr, device, "paged_kv_indptr")?,
            paged_kv_indices: metadata_tensor(&view.paged_kv.indices, device, "paged_kv_indices")?,
            paged_kv_last_page_len: metadata_tensor(
                &view.paged_kv.last_page_len,
                device,
                "paged_kv_last_page_len",
            )?,
            q_indptr: metadata_tensor(
                &view.prefill_tile_plan.q_indptr,
                device,
                "paged_kv_q_indptr",
            )?,
            request_indices: metadata_tensor(
                &view.prefill_tile_plan.request_indices,
                device,
                "paged_kv_request_indices",
            )?,
            qo_tile_indices: metadata_tensor(
                &view.prefill_tile_plan.qo_tile_indices,
                device,
                "paged_kv_qo_tile_indices",
            )?,
            kv_tile_indices: metadata_tensor(
                &view.prefill_tile_plan.kv_tile_indices,
                device,
                "paged_kv_tile_indices",
            )?,
            o_indptr: metadata_tensor(
                &view.prefill_tile_plan.o_indptr,
                device,
                "paged_kv_o_indptr",
            )?,
            kv_chunk_size: metadata_tensor(
                &view.prefill_tile_plan.kv_chunk_size,
                device,
                "paged_kv_chunk_size",
            )?,
            block_valid_mask: metadata_tensor(
                &view.prefill_tile_plan.block_valid_mask,
                device,
                "paged_kv_block_valid_mask",
            )?,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn prefill_input(causal: bool) -> FlashInferPrefillPlanInput {
        FlashInferPrefillPlanInput {
            device_is_cuda: true,
            dtype: DType::BF16,
            has_sinks: false,
            causality_known: true,
            causal,
            head_size: 256,
            attention_heads: 16,
            key_value_heads: 8,
            query_lens_match_seq_len: true,
            attention_backend: AttentionBackendKind::FlashInfer,
        }
    }

    #[test]
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn flashinfer_prefill_declines_noncausal_plans() {
        assert!(prefill_plan(prefill_input(true)).is_some());
        assert!(prefill_plan(prefill_input(false)).is_none());
    }

    #[test]
    fn flashinfer_prefill_tile_q_matches_ampere_selector() {
        assert_eq!(FlashInferPrefillTiling::tile_q_for(16, 256), 16);
        assert_eq!(FlashInferPrefillTiling::tile_q_for(17, 256), 64);
        assert_eq!(FlashInferPrefillTiling::tile_q_for(65, 128), 128);
        assert_eq!(FlashInferPrefillTiling::tile_q_for(65, 256), 64);
    }
}
