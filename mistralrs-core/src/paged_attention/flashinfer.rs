use std::collections::HashMap;

use candle_core::{DType, DeviceLocation, Result, Tensor};

use super::attention_backend::{AttentionBackend, AttentionBackendKind, AttentionLayerSpec};

pub type DeviceTensorMap = HashMap<DeviceLocation, Tensor>;

pub const FLASHINFER_PREFILL_TILE_Q: usize = 64;
pub const FLASHINFER_PREFILL_MAX_GROUP_SIZE: usize = 8;

pub const STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE: usize = 512;
pub const FLASHINFER_PREFILL_MAX_HEAD_SIZE: usize = 256;
pub const FLASHINFER_DECODE_MAX_HEAD_SIZE: usize = 512;
pub const FLASHINFER_TENSOR_CORE_DECODE_ENABLED: bool = false;
pub const FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE: usize = 256;

#[derive(Clone, Debug)]
pub struct FlashInferPagedKv {
    pub indptr: DeviceTensorMap,
    pub indices: DeviceTensorMap,
    pub last_page_len: DeviceTensorMap,
}

#[derive(Clone, Debug)]
pub struct FlashInferTilePlan {
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
    pub block_tables: Option<DeviceTensorMap>,
    pub context_lens: Option<DeviceTensorMap>,
    pub max_context_len: Option<usize>,
    pub paged_kv: FlashInferPagedKv,
    pub tile_plan: FlashInferTilePlan,
    pub prefill_tile_plan: FlashInferTilePlan,
    pub block_table_signature: Option<Vec<u64>>,
}

#[derive(Clone, Debug)]
pub struct FlashInferPagedAttentionViews {
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

#[derive(Clone, Copy, Debug)]
pub(crate) struct FlashInferPrefillPlan {
    causal: bool,
}

impl FlashInferPrefillPlan {
    pub fn causal(self) -> bool {
        self.causal
    }
}

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
    pub supports_ported_tile_q: bool,
    pub attention_backend: AttentionBackendKind,
}

pub(crate) fn prefill_plan(input: FlashInferPrefillPlanInput) -> Option<FlashInferPrefillPlan> {
    (crate::perf_flags::flashinfer_prefill_enabled()
        && input.device_is_cuda
        && input.dtype != DType::F32
        && !input.has_sinks
        && input.causality_known
        && input.causal
        && supports_prefill_head_dim(input.head_size)
        && supports_prefill_group_size(input.attention_heads, input.key_value_heads)
        && input.query_lens_match_seq_len
        && input.supports_ported_tile_q
        && input.attention_backend == AttentionBackendKind::FlashInfer)
        .then_some(FlashInferPrefillPlan {
            causal: input.causal,
        })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FlashInferDecodePlan {
    use_tensor_cores: bool,
}

impl FlashInferDecodePlan {
    pub fn use_tensor_cores(self) -> bool {
        self.use_tensor_cores
    }
}

pub(crate) struct FlashInferDecodePlanInput {
    pub dtype: DType,
    pub head_size: usize,
    pub has_alibi: bool,
    pub has_sinks: bool,
}

pub(crate) fn decode_plan(input: FlashInferDecodePlanInput) -> Result<FlashInferDecodePlan> {
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
        use_tensor_cores: use_tensor_core_decode(input.head_size, input.dtype),
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
        if spec.kv_heads == 0 || !spec.q_heads.is_multiple_of(spec.kv_heads) {
            return false;
        }
        let q_group = spec.q_heads / spec.kv_heads;
        spec.k_head_dim == spec.v_head_dim
            && matches!(spec.k_head_dim, 64 | 128 | 256 | 512)
            && q_group <= FLASHINFER_PREFILL_MAX_GROUP_SIZE
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

pub fn decode_head_size_limit(kind: AttentionBackendKind) -> usize {
    match kind {
        AttentionBackendKind::FlashInfer => FLASHINFER_DECODE_MAX_HEAD_SIZE,
        AttentionBackendKind::Standard => STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE,
    }
}

pub fn supports_prefill_head_dim(head_size: usize) -> bool {
    head_size <= FLASHINFER_PREFILL_MAX_HEAD_SIZE
}

pub fn supports_prefill_group_size(q_heads: usize, kv_heads: usize) -> bool {
    kv_heads != 0
        && q_heads.is_multiple_of(kv_heads)
        && q_heads / kv_heads <= FLASHINFER_PREFILL_MAX_GROUP_SIZE
}

pub fn determine_prefill_tile_q(avg_packed_qo_len: usize, head_dim: usize) -> usize {
    if avg_packed_qo_len > 64 && head_dim < 256 {
        128
    } else if avg_packed_qo_len > 16 {
        64
    } else {
        16
    }
}

pub fn supports_ported_prefill_tile_q(
    query_lens: &[usize],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> bool {
    if query_lens.is_empty() || !supports_prefill_group_size(q_heads, kv_heads) {
        return false;
    }
    let group_size = q_heads / kv_heads;
    let sum_packed_qo_len = query_lens
        .iter()
        .copied()
        .map(|len| len.saturating_mul(group_size))
        .sum::<usize>();
    let avg_packed_qo_len = sum_packed_qo_len / query_lens.len();
    determine_prefill_tile_q(avg_packed_qo_len, head_dim) == FLASHINFER_PREFILL_TILE_Q
}

pub fn use_tensor_core_decode(head_size: usize, dtype: DType) -> bool {
    FLASHINFER_TENSOR_CORE_DECODE_ENABLED
        && head_size <= FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE
        && dtype != DType::F32
}

#[cfg(test)]
mod tests {
    use super::*;

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
            supports_ported_tile_q: true,
            attention_backend: AttentionBackendKind::FlashInfer,
        }
    }

    #[test]
    fn flashinfer_prefill_declines_noncausal_plans() {
        assert!(prefill_plan(prefill_input(true)).is_some());
        assert!(prefill_plan(prefill_input(false)).is_none());
    }
}
