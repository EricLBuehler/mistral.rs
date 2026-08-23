use std::collections::HashMap;

#[cfg(feature = "cuda")]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[cfg(all(feature = "cuda", target_family = "unix"))]
use candle_core::Result;
use candle_core::{DeviceLocation, Tensor};

use crate::paged_attention::attention_backend::{
    AttentionBackend, AttentionBackendKind, AttentionLayerSpec,
};

mod metadata;
#[cfg(feature = "cuda")]
pub(crate) use metadata::make_fa3_decode_state;
pub(crate) use metadata::{
    decode_split_capacity_pages, decode_split_pages, flashinfer_metadata, flashinfer_paged_kv,
    flashinfer_tile_plan, flashinfer_view, make_paged_kv_decode_tensors,
    make_paged_kv_decode_tensors_from_lens, make_paged_kv_tensors,
};
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) use metadata::{
    fa3_device_num_sm, fa3_prefill_cache_num_sm, register_fa3_prefill_caches,
    with_fa3_prefill_workspace, Fa3PrefillWorkspaceRegistration,
};

// Metadata is copied per CUDA device; graph replay may substitute graph-owned tensors.
pub type DeviceTensorMap = HashMap<DeviceLocation, Tensor>;

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) const FA3_DECODE_NUM_SPLITS: usize = 32;
#[cfg(all(feature = "cuda", target_family = "unix"))]
const FA3_PAGED_MIN_SPLITS: usize = 2;
#[cfg(all(feature = "cuda", target_family = "unix"))]
const FA3_DECODE_HEAD_DIM: usize = 256;
#[cfg(all(feature = "cuda", target_family = "unix"))]
const FA3_SCHEDULER_BATCH_ALIGNMENT: usize = 4;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) const FA3_DECODE_MAX_QUERY_LEN: usize = mistralrs_paged_attn::FA3_DECODE_MAX_QUERY_LEN;

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Fa3DecodeView {
    Logical,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Fa3DecodeScheduleKey {
    pub device: DeviceLocation,
    pub view: Fa3DecodeView,
    pub batch: usize,
    pub query_len: usize,
    pub causal: bool,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub num_splits: usize,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fa3PagedScheduleShape {
    pub device: DeviceLocation,
    pub view: Fa3DecodeView,
    pub batch: usize,
    pub query_len: usize,
    pub causal: bool,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Fa3PagedScheduleShape {
    pub(crate) fn prefill_schedule_key(self, num_sm: usize) -> Option<Fa3DecodeScheduleKey> {
        let num_splits = fa3_prefill_num_splits(
            self.batch,
            self.query_len,
            self.q_heads,
            self.kv_heads,
            num_sm,
        )?;
        self.schedule_key(num_splits)
    }

    pub(crate) fn decode_schedule_key(self) -> Option<Fa3DecodeScheduleKey> {
        self.schedule_key(FA3_DECODE_NUM_SPLITS)
    }

    fn schedule_key(self, num_splits: usize) -> Option<Fa3DecodeScheduleKey> {
        let key = Fa3DecodeScheduleKey {
            device: self.device,
            view: self.view,
            batch: self.batch,
            query_len: self.query_len,
            causal: self.causal,
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            page_size: self.page_size,
            num_splits,
        };
        key.supported().then_some(key)
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Fa3DecodeScheduleKey {
    pub(crate) fn total_q(&self) -> Option<usize> {
        self.batch.checked_mul(self.query_len)
    }

    pub(crate) fn supported(&self) -> bool {
        self.batch > 0
            && self.query_len > 0
            && self.query_len <= FA3_DECODE_MAX_QUERY_LEN
            && self.total_q().is_some()
            && self.q_heads > 0
            && self.kv_heads > 0
            && self.q_heads.is_multiple_of(self.kv_heads)
            && self.head_dim == FA3_DECODE_HEAD_DIM
            && self.page_size > 0
            && (FA3_PAGED_MIN_SPLITS..=FA3_DECODE_NUM_SPLITS).contains(&self.num_splits)
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn fa3_prefill_num_splits(
    batch: usize,
    query_len: usize,
    q_heads: usize,
    kv_heads: usize,
    num_sm: usize,
) -> Option<usize> {
    if batch == 0
        || query_len == 0
        || query_len > FA3_DECODE_MAX_QUERY_LEN
        || q_heads == 0
        || kv_heads == 0
        || num_sm == 0
        || !supports_flashinfer_group_size(q_heads, kv_heads)
    {
        return None;
    }
    let group_size = q_heads / kv_heads;
    let query_tiles = query_len.checked_mul(group_size)?.div_ceil(128).max(1);
    let resident_tiles = batch.checked_mul(kv_heads)?.checked_mul(query_tiles)?;
    Some(
        num_sm
            .div_ceil(resident_tiles)
            .clamp(FA3_PAGED_MIN_SPLITS, FA3_DECODE_NUM_SPLITS),
    )
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Fa3PrefillPoolBytes {
    quantized_query: usize,
    scheduler_metadata: usize,
    output_accum: usize,
    lse_accum: usize,
    output_lse: usize,
    page_table: usize,
    cu_seqlens_q: usize,
    seqused_k: usize,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Fa3PrefillPoolBytes {
    pub(crate) fn component_max(self, other: Self) -> Self {
        Self {
            quantized_query: self.quantized_query.max(other.quantized_query),
            scheduler_metadata: self.scheduler_metadata.max(other.scheduler_metadata),
            output_accum: self.output_accum.max(other.output_accum),
            lse_accum: self.lse_accum.max(other.lse_accum),
            output_lse: self.output_lse.max(other.output_lse),
            page_table: self.page_table.max(other.page_table),
            cu_seqlens_q: self.cu_seqlens_q.max(other.cu_seqlens_q),
            seqused_k: self.seqused_k.max(other.seqused_k),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn bytes(self) -> candle_core::Result<usize> {
        checked_workspace_sum(&[
            self.quantized_query,
            self.scheduler_metadata,
            self.output_accum,
            self.lse_accum,
            self.output_lse,
            self.page_table,
            self.cu_seqlens_q,
            self.seqused_k,
        ])
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Fa3PrefillWorkspaceBytes {
    pool: Fa3PrefillPoolBytes,
    transient: usize,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Fa3PrefillWorkspaceBytes {
    pub(crate) fn pool(self) -> Fa3PrefillPoolBytes {
        self.pool
    }

    pub(crate) fn transient_bytes(self) -> usize {
        self.transient
    }

    pub(crate) fn bytes(self) -> candle_core::Result<usize> {
        self.pool
            .bytes()?
            .checked_add(self.transient)
            .ok_or_else(|| candle_core::Error::msg("FA3 prefill workspace size overflow"))
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn fa3_prefill_workspace_components(
    batch: usize,
    query_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_pages_per_sequence: usize,
    num_sm: usize,
) -> candle_core::Result<Fa3PrefillWorkspaceBytes> {
    if head_dim != FA3_DECODE_HEAD_DIM || max_pages_per_sequence == 0 {
        candle_core::bail!("invalid FA3 prefill workspace shape");
    }
    let num_splits = fa3_prefill_num_splits(batch, query_len, q_heads, kv_heads, num_sm)
        .ok_or_else(|| candle_core::Error::msg("invalid FA3 prefill workspace shape"))?;
    let total_q = batch
        .checked_mul(query_len)
        .ok_or_else(|| candle_core::Error::msg("FA3 prefill query count overflow"))?;
    let rounded_batch = batch
        .div_ceil(FA3_SCHEDULER_BATCH_ALIGNMENT)
        .checked_mul(FA3_SCHEDULER_BATCH_ALIGNMENT)
        .ok_or_else(|| candle_core::Error::msg("FA3 scheduler row count overflow"))?;
    let scheduler_len = (2 + usize::from(query_len > 1))
        .checked_mul(rounded_batch)
        .and_then(|len| len.checked_add(1))
        .ok_or_else(|| candle_core::Error::msg("FA3 scheduler row count overflow"))?;
    let cu_seqlens_len = batch
        .checked_add(1)
        .ok_or_else(|| candle_core::Error::msg("FA3 cumulative query length overflow"))?;
    let query_bytes = checked_workspace_bytes(
        &[total_q, q_heads, head_dim],
        candle_core::DType::BF16.size_in_bytes(),
    )?;
    Ok(Fa3PrefillWorkspaceBytes {
        pool: Fa3PrefillPoolBytes {
            quantized_query: checked_workspace_bytes(
                &[total_q, q_heads, head_dim],
                candle_core::DType::F8E4M3.size_in_bytes(),
            )?,
            output_accum: checked_workspace_bytes(
                &[num_splits, q_heads, total_q, head_dim],
                candle_core::DType::F32.size_in_bytes(),
            )?,
            lse_accum: checked_workspace_bytes(
                &[num_splits, q_heads, total_q],
                candle_core::DType::F32.size_in_bytes(),
            )?,
            output_lse: checked_workspace_bytes(
                &[q_heads, total_q],
                candle_core::DType::F32.size_in_bytes(),
            )?,
            page_table: checked_workspace_bytes(
                &[batch, max_pages_per_sequence],
                candle_core::DType::I32.size_in_bytes(),
            )?,
            scheduler_metadata: checked_workspace_bytes(
                &[scheduler_len],
                candle_core::DType::I32.size_in_bytes(),
            )?,
            cu_seqlens_q: checked_workspace_bytes(
                &[cu_seqlens_len],
                candle_core::DType::I32.size_in_bytes(),
            )?,
            seqused_k: checked_workspace_bytes(&[batch], candle_core::DType::I32.size_in_bytes())?,
        },
        transient: query_bytes
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::msg("FA3 prefill transient size overflow"))?,
    })
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[allow(dead_code)]
pub(crate) fn fa3_prefill_workspace_bytes(
    batch: usize,
    query_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_pages_per_sequence: usize,
    num_sm: usize,
) -> candle_core::Result<usize> {
    fa3_prefill_workspace_components(
        batch,
        query_len,
        q_heads,
        kv_heads,
        head_dim,
        max_pages_per_sequence,
        num_sm,
    )?
    .bytes()
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn checked_workspace_sum(bytes: &[usize]) -> candle_core::Result<usize> {
    bytes
        .iter()
        .try_fold(0usize, |total, bytes| total.checked_add(*bytes))
        .ok_or_else(|| candle_core::Error::msg("FA3 prefill workspace size overflow"))
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn checked_workspace_bytes(parts: &[usize], element_size: usize) -> candle_core::Result<usize> {
    parts
        .iter()
        .try_fold(element_size, |bytes, part| bytes.checked_mul(*part))
        .ok_or_else(|| candle_core::Error::msg("FA3 prefill workspace size overflow"))
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Debug)]
pub(crate) struct Fa3DecodeBuffers {
    pub query: Tensor,
    pub scheduler_metadata: Tensor,
    pub output_accum: Tensor,
    pub lse_accum: Tensor,
    pub output_lse: Tensor,
    pub cu_seqlens_q: Tensor,
    pub page_table: Tensor,
    pub seqused_k: Tensor,
    pub max_pages_per_sequence: usize,
    pub num_sm: usize,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Fa3DecodeBuffers {
    pub(crate) fn schedule(
        &self,
        key: Fa3DecodeScheduleKey,
    ) -> Result<mistralrs_paged_attn::Fa3DecodeSchedule> {
        let DeviceLocation::Cuda { gpu_id } = key.device else {
            candle_core::bail!("FA3 decode state must be on CUDA");
        };
        let max_seqlen_k = self
            .max_pages_per_sequence
            .checked_mul(key.page_size)
            .ok_or_else(|| candle_core::Error::msg("FA3 maximum KV length overflow"))?;
        Ok(mistralrs_paged_attn::Fa3DecodeSchedule {
            batch_size: key.batch,
            query_len: key.query_len,
            total_q: key
                .total_q()
                .ok_or_else(|| candle_core::Error::msg("FA3 query count overflow"))?,
            causal: key.causal,
            q_heads: key.q_heads,
            kv_heads: key.kv_heads,
            head_dim: key.head_dim,
            page_size: key.page_size,
            max_seqlen_k,
            num_splits: key.num_splits,
            num_sm: self.num_sm,
            device_id: gpu_id,
        })
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Debug, Default)]
pub(crate) struct Fa3DecodeState {
    schedules: HashMap<Fa3DecodeScheduleKey, Fa3DecodeBuffers>,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Fa3DecodeState {
    pub(crate) fn insert(&mut self, key: Fa3DecodeScheduleKey, buffers: Fa3DecodeBuffers) {
        self.schedules.insert(key, buffers);
    }

    pub(crate) fn get(&self, key: &Fa3DecodeScheduleKey) -> Option<&Fa3DecodeBuffers> {
        self.schedules.get(key)
    }

    pub(crate) fn schedules(
        &self,
    ) -> impl Iterator<Item = (&Fa3DecodeScheduleKey, &Fa3DecodeBuffers)> {
        self.schedules.iter()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.schedules.is_empty()
    }
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
#[derive(Clone, Debug, Default)]
pub(crate) struct Fa3DecodeState;

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) struct Fa3DecodePrepare<'a> {
    pub key: Fa3DecodeScheduleKey,
    pub paged_kv_indptr: &'a Tensor,
    pub paged_kv_indices: &'a Tensor,
    pub paged_kv_last_page_len: &'a Tensor,
    pub buffers: &'a Fa3DecodeBuffers,
}

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
    #[cfg_attr(not(all(feature = "cuda", target_family = "unix")), allow(dead_code))]
    pub(crate) fa3_decode: Option<Fa3DecodeState>,
    #[cfg(feature = "cuda")]
    pub(crate) decode_tile_plan_used: Option<Arc<AtomicBool>>,
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
    if kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
        return false;
    }
    // Must match DISPATCH_GQA_GROUP_SIZE in FlashInfer's utils.cuh.
    matches!(q_heads / kv_heads, 1 | 2 | 3 | 4 | 6 | 8 | 16)
}

impl FlashInferPagedAttentionViews {
    pub fn select(&self, sliding_window: Option<usize>) -> &FlashInferPagedAttentionView {
        if sliding_window.is_some() {
            self.sliding.as_ref().unwrap_or(&self.logical)
        } else {
            &self.logical
        }
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub(crate) fn fa3_view(&self, view: Fa3DecodeView) -> &FlashInferPagedAttentionView {
        match view {
            Fa3DecodeView::Logical => &self.logical,
        }
    }
}

impl FlashInferMetadata {
    #[cfg(feature = "cuda")]
    pub(crate) fn track_decode_tile_plan(mut self) -> Self {
        self.decode_tile_plan_used = Some(Arc::new(AtomicBool::new(false)));
        self
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn decode_tile_plan_was_used(&self) -> bool {
        self.decode_tile_plan_used
            .as_ref()
            .is_some_and(|used| used.load(Ordering::Relaxed))
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub(crate) fn fa3_decode_buffers(
        &self,
        key: &Fa3DecodeScheduleKey,
    ) -> Option<&Fa3DecodeBuffers> {
        self.fa3_decode.as_ref()?.get(key)
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub(crate) fn for_each_fa3_decode_schedule(
        &self,
        mut f: impl FnMut(Fa3DecodePrepare<'_>) -> Result<()>,
    ) -> Result<()> {
        let Some(state) = self.fa3_decode.as_ref() else {
            return Ok(());
        };
        for (&key, buffers) in state.schedules() {
            let view = self.views.fa3_view(key.view);
            f(Fa3DecodePrepare {
                key,
                paged_kv_indptr: metadata_tensor(
                    &view.paged_kv.indptr,
                    &key.device,
                    "fa3_paged_kv_indptr",
                )?,
                paged_kv_indices: metadata_tensor(
                    &view.paged_kv.indices,
                    &key.device,
                    "fa3_paged_kv_indices",
                )?,
                paged_kv_last_page_len: metadata_tensor(
                    &view.paged_kv.last_page_len,
                    &key.device,
                    "fa3_paged_kv_last_page_len",
                )?,
                buffers,
            })?;
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub(crate) fn decode_metadata(
        &self,
        device: &DeviceLocation,
        sliding_window: Option<usize>,
    ) -> Result<FlashInferDecodeMetadata<'_>> {
        if let Some(used) = self.decode_tile_plan_used.as_ref() {
            used.store(true, Ordering::Relaxed);
        }
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

#[cfg(test)]
mod tests {
    use super::supports_flashinfer_group_size;
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    use super::{
        fa3_prefill_num_splits, fa3_prefill_workspace_bytes, fa3_prefill_workspace_components,
        Fa3DecodeScheduleKey, Fa3DecodeView, Fa3PagedScheduleShape, Fa3PrefillPoolBytes,
        Fa3PrefillWorkspaceBytes, FA3_DECODE_MAX_QUERY_LEN, FA3_DECODE_NUM_SPLITS,
    };
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    use candle_core::DeviceLocation;

    #[test]
    fn flashinfer_group_size_matches_kernel_instantiations() {
        for group_size in [1, 2, 3, 4, 6, 8, 16] {
            assert!(supports_flashinfer_group_size(group_size * 2, 2));
        }

        for group_size in [0, 5, 7, 9, 15, 17] {
            assert!(!supports_flashinfer_group_size(group_size * 2, 2));
        }
        assert!(!supports_flashinfer_group_size(14, 0));
        assert!(!supports_flashinfer_group_size(15, 2));
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_schedule_capability_is_shape_based() {
        let key = Fa3DecodeScheduleKey {
            device: DeviceLocation::Cuda { gpu_id: 0 },
            view: Fa3DecodeView::Logical,
            batch: 16,
            query_len: 1,
            causal: false,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 256,
            page_size: 32,
            num_splits: FA3_DECODE_NUM_SPLITS,
        };
        assert!(key.supported());
        assert_eq!(key.total_q(), Some(16));
        assert!(!Fa3DecodeScheduleKey {
            head_dim: 128,
            ..key
        }
        .supported());
        assert!(!Fa3DecodeScheduleKey { q_heads: 30, ..key }.supported());
        assert!(Fa3DecodeScheduleKey {
            query_len: 8,
            causal: true,
            ..key
        }
        .supported());
        assert!(!Fa3DecodeScheduleKey {
            query_len: FA3_DECODE_MAX_QUERY_LEN + 1,
            ..key
        }
        .supported());
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_split_cap_tracks_query_occupancy() {
        assert_eq!(fa3_prefill_num_splits(1, 8, 24, 4, 132), Some(32));
        assert_eq!(fa3_prefill_num_splits(3, 8, 24, 4, 132), Some(11));
        assert_eq!(fa3_prefill_num_splits(8, 8, 24, 4, 132), Some(5));
        assert_eq!(fa3_prefill_num_splits(16, 8, 24, 4, 132), Some(3));
        assert_eq!(fa3_prefill_num_splits(1, 128, 24, 4, 132), Some(6));
        assert_eq!(fa3_prefill_num_splits(8, 128, 24, 4, 132), Some(2));
        assert_eq!(fa3_prefill_num_splits(16, 128, 24, 4, 132), Some(2));
        assert_eq!(fa3_prefill_num_splits(1, 128, 24, 0, 132), None);
        assert_eq!(fa3_prefill_num_splits(1, 128, 20, 4, 132), None);
        assert_eq!(
            fa3_prefill_num_splits(1, FA3_DECODE_MAX_QUERY_LEN + 1, 24, 4, 132),
            None
        );
        assert_eq!(fa3_prefill_num_splits(usize::MAX, 128, 24, 4, 132), None);
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_decode_schedule_preserves_long_row_split_capacity() {
        for (batch, prefill_splits) in [(3, 11), (16, 3)] {
            let shape = Fa3PagedScheduleShape {
                device: DeviceLocation::Cuda { gpu_id: 0 },
                view: Fa3DecodeView::Logical,
                batch,
                query_len: 8,
                causal: true,
                q_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                page_size: 32,
            };
            let prefill = shape
                .prefill_schedule_key(132)
                .expect("supported FA3 prefill schedule");
            let decode = shape
                .decode_schedule_key()
                .expect("supported FA3 decode schedule");
            assert_eq!(prefill.num_splits, prefill_splits);
            assert_eq!(decode.num_splits, FA3_DECODE_NUM_SPLITS);
            assert!(prefill.supported());
            assert!(decode.supported());
        }
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_pool_uses_component_wise_maxima() {
        let query_heavy = Fa3PrefillPoolBytes {
            quantized_query: 10,
            output_accum: 3,
            ..Default::default()
        };
        let metadata_heavy = Fa3PrefillPoolBytes {
            scheduler_metadata: 20,
            output_accum: 2,
            ..Default::default()
        };
        let combined = query_heavy.component_max(metadata_heavy);
        assert_eq!(
            combined,
            Fa3PrefillPoolBytes {
                quantized_query: 10,
                scheduler_metadata: 20,
                output_accum: 3,
                ..Default::default()
            }
        );
        assert_eq!(combined.bytes().unwrap(), 33);
        assert!(combined.bytes().unwrap() > query_heavy.bytes().unwrap());
        assert!(combined.bytes().unwrap() > metadata_heavy.bytes().unwrap());
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_workspace_adds_pool_and_transient_bytes() {
        let components = fa3_prefill_workspace_components(2, 8, 24, 4, 256, 17, 132).unwrap();
        assert_eq!(
            components.bytes().unwrap(),
            components
                .pool()
                .bytes()
                .unwrap()
                .checked_add(components.transient_bytes())
                .unwrap()
        );
        assert_eq!(
            fa3_prefill_workspace_bytes(2, 8, 24, 4, 256, 17, 132).unwrap(),
            components.bytes().unwrap()
        );
        assert!(Fa3PrefillWorkspaceBytes {
            pool: Fa3PrefillPoolBytes {
                quantized_query: usize::MAX,
                ..Default::default()
            },
            transient: 1,
        }
        .bytes()
        .is_err());
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_workspace_covers_every_transient_allocation() {
        assert_eq!(
            fa3_prefill_workspace_bytes(1, 1, 24, 4, 256, 3_125, 132).unwrap(),
            832_868
        );
        assert_eq!(
            fa3_prefill_workspace_bytes(16, 128, 24, 4, 256, 3_125, 132).unwrap(),
            164_368_008
        );
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_workspace_rejects_invalid_or_overflowing_shapes() {
        assert!(fa3_prefill_workspace_bytes(0, 1, 24, 4, 256, 1, 132).is_err());
        assert!(fa3_prefill_workspace_bytes(1, 1, 24, 5, 256, 1, 132).is_err());
        assert!(fa3_prefill_workspace_bytes(1, 1, 20, 4, 256, 1, 132).is_err());
        assert!(fa3_prefill_workspace_bytes(1, 1, 24, 4, 128, 1, 132).is_err());
        assert!(
            fa3_prefill_workspace_bytes(1, FA3_DECODE_MAX_QUERY_LEN + 1, 24, 4, 256, 1, 132)
                .is_err()
        );
        assert!(fa3_prefill_workspace_bytes(1, 1, 24, 4, 256, 0, 132).is_err());
        assert!(fa3_prefill_workspace_bytes(usize::MAX, 128, 24, 4, 256, usize::MAX, 132).is_err());
    }
}
