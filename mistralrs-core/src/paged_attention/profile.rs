use std::sync::Arc;

use candle_core::{Device, Tensor};
use tokio::sync::Mutex;

use super::{CacheConfig, ModelConfigLike};
use crate::device_map::DeviceMapper;
use crate::paged_attention::block_hash::MultimodalAttentionPolicy;
use crate::paged_attention::KVCacheManager;
use crate::pipeline::text_models_inputs_processor::{
    make_prompt_chunk, PagedAttentionInputMetadata, PagedAttentionMeta,
};
use crate::pipeline::EitherCache;
use crate::pipeline::{ModelForwardContext, RecurrentBatchKind, RecurrentMetadata};
use crate::{get_mut_arcmutex, MemoryUsage};

/// Request id used for the synthetic sequence; the manager is private to this profile.
const PROFILE_REQUEST_ID: usize = 0;

/// Context length of the second probe, clamped to what the cache can hold. Large enough that any
/// per-context growth would clear measurement noise.
pub(crate) const ACTIVATION_PROBE_CONTEXT: usize = 65536;

/// Activation cost as a base plus a per-context-token slope.
///
/// Attention reads the whole KV cache, so a chunk costs more the further into a sequence it runs.
/// Expressing the growth per context token lets it be budgeted next to the KV cost per token.
pub(crate) struct ActivationScaling {
    pub base_bytes: usize,
    pub bytes_per_context_token: f64,
}

#[allow(clippy::cast_precision_loss)]
impl ActivationScaling {
    pub fn from_probes(base_bytes: usize, far_bytes: usize, far_context: usize) -> Self {
        let growth = far_bytes.saturating_sub(base_bytes) as f64;
        Self {
            base_bytes,
            bytes_per_context_token: growth / far_context.max(1) as f64,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn bytes_at(&self, context: usize) -> usize {
        self.base_bytes + (self.bytes_per_context_token * context as f64) as usize
    }
}

pub(crate) struct PrefillProfileCtx<'a> {
    pub device: &'a Device,
    pub cache_config: &'a CacheConfig,
    pub model_config: &'a dyn ModelConfigLike,
    pub kv_cache: &'a [(Tensor, Tensor)],
    pub tokens: usize,
    /// Tokens already resident in the KV cache when the chunk runs. Attention reads the whole
    /// context, so the activation peak depends on this, not just on the chunk size.
    pub context_offset: usize,
    pub sliding_window: Option<usize>,
    pub mapper: Option<&'a dyn DeviceMapper>,
    pub cache: &'a EitherCache,
}

/// Run one synthetic prefill and return the bytes of device memory it left resident.
///
/// The CUDA mempool release threshold is set to never return memory to the driver, so free memory
/// after the forward is the high-water mark and no separate peak counter is needed.
pub(crate) fn measure_prefill_activation_bytes<F>(
    ctx: PrefillProfileCtx<'_>,
    forward: F,
) -> anyhow::Result<usize>
where
    F: FnOnce(&Tensor, &mut ModelForwardContext<'_>) -> candle_core::Result<Tensor>,
{
    // Diagnostics must never stop a model from loading, and the input builders assert rather than
    // return on malformed metadata.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        measure_prefill_activation_bytes_inner(ctx, forward)
    }))
    .unwrap_or_else(|_| Err(anyhow::anyhow!("profiling forward pass panicked")))
}

fn measure_prefill_activation_bytes_inner<F>(
    ctx: PrefillProfileCtx<'_>,
    forward: F,
) -> anyhow::Result<usize>
where
    F: FnOnce(&Tensor, &mut ModelForwardContext<'_>) -> candle_core::Result<Tensor>,
{
    if !ctx.device.is_cuda() {
        anyhow::bail!("prefill profiling requires a CUDA device");
    }
    let block_size = ctx.cache_config.block_size;
    let tokens = ctx.tokens.max(block_size);
    let context_offset = ctx.context_offset / block_size * block_size;
    let total_tokens = context_offset + tokens;

    let manager = KVCacheManager::new(
        ctx.cache_config.num_gpu_blocks,
        block_size,
        false,
        ctx.model_config.kv_cache_group_ids(),
    );
    let manager = Arc::new(Mutex::new(manager));
    {
        let mut guard = get_mut_arcmutex!(manager);
        guard
            .allocate_slots(PROFILE_REQUEST_ID, total_tokens, &[])
            .ok_or_else(|| {
                anyhow::anyhow!("KV cache too small to profile {total_tokens} tokens")
            })?;
    }

    let mut meta = PagedAttentionMeta {
        sliding_window: ctx.sliding_window,
        block_size,
        max_paged_context_len: total_tokens,
        attention_backend: ctx.model_config.attention_backend_kind(),
        has_flashinfer_decode_layers: false,
        prefill_attention_heads: ctx.model_config.num_attn_heads(),
        prefill_key_value_heads: ctx.model_config.num_kv_heads(),
        prefill_head_dim: ctx.model_config.k_head_dim(),
        kv_cache_manager: manager.clone(),
        prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
        has_noncausal_mm_context: false,
        mm_prefix_ranges_by_seq_id: Default::default(),
        full_mm_prefix_ranges_by_seq_id: Default::default(),
        enable_packed_prefill: false,
        is_final_prompt_chunk: true,
    };

    let toks = vec![0u32; total_tokens];
    let prefix_lens = [context_offset];
    // `chunk_offset_toks` adds to the prefix length rather than replacing it, so the offset is
    // expressed purely through `prefix_lens`, matching how chunked prefill drives this.
    let inputs = make_prompt_chunk(
        0,
        vec![toks.as_slice()],
        &[PROFILE_REQUEST_ID],
        ctx.device,
        None,
        false,
        Some(&mut meta),
        ctx.mapper,
        Some(&prefix_lens),
        ctx.sliding_window,
        false,
    )?;

    let paged_meta: PagedAttentionInputMetadata = inputs
        .paged_attn_meta
        .ok_or_else(|| anyhow::anyhow!("profile inputs carried no paged attention metadata"))?;

    // Linear-attention layers refuse to run without a recurrent slot, so claim one for the profile.
    let recurrent_slot = if ctx.cache.is_hybrid() {
        let mut hybrid = ctx.cache.hybrid();
        let slot = hybrid
            .allocate_seq()
            .ok_or_else(|| anyhow::anyhow!("no free recurrent state slot to profile with"))?;
        let host = vec![u32::try_from(slot).expect("recurrent slot exceeds u32")];
        let indices = Tensor::from_vec(host.clone(), (1,), ctx.device)?;
        hybrid.set_state_indices_with_host(Some(indices), Some(host));
        Some(slot)
    } else {
        None
    };

    let recurrent_meta = ctx.cache.is_hybrid().then(|| {
        let hybrid = ctx.cache.hybrid();
        let host = hybrid.state_indices_host().map(ToOwned::to_owned);
        hybrid
            .state_indices()
            .cloned()
            .map(|indices| RecurrentMetadata::new(RecurrentBatchKind::Prefill, indices, host))
    });

    ctx.device.synchronize()?;
    let before = MemoryUsage.query(ctx.device)?.available();

    let mut forward_ctx = ModelForwardContext::new(
        &inputs.positions,
        &inputs.context_lens,
        &inputs.position_ids,
        Some((ctx.kv_cache, &paged_meta)),
        &inputs.flash_meta,
    );
    if let Some(meta) = recurrent_meta.flatten() {
        forward_ctx = forward_ctx
            .with_recurrent_batch_kind(RecurrentBatchKind::Prefill)
            .with_recurrent_metadata(Some(meta));
    }
    let result = forward(&inputs.input, &mut forward_ctx);

    ctx.device.synchronize()?;
    let after = MemoryUsage.query(ctx.device)?.available();

    if let Some(slot) = recurrent_slot {
        let mut hybrid = ctx.cache.hybrid();
        hybrid.set_state_indices(None);
        hybrid.free_seq(slot);
    }
    result?;

    {
        let mut guard = get_mut_arcmutex!(manager);
        guard.free(PROFILE_REQUEST_ID);
    }

    Ok(before.saturating_sub(after))
}
