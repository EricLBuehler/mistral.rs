use std::{collections::HashMap, sync::Once};

use candle_core::{DType, Device, DeviceLocation, Result, Tensor};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use mistralrs_paged_attn::{
    fa3_fp8_decode, flashinfer_decode, gather_kv_cache_flashinfer, reshape_and_cache_flashinfer,
    Fa3DecodeParams, FlashInferDecodeScratch, KvCacheScales as FlashInferKvCacheScales,
    DEFAULT_FP8_KV_CACHE_SCALES,
};
use mistralrs_paged_attn::{paged_attention, reshape_and_cache};

#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::attention::sliding_window_left;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::flashinfer::{
    fa3_device_num_sm, fa3_prefill_cache_num_sm, with_fa3_prefill_workspace, Fa3DecodeScheduleKey,
    Fa3DecodeView, Fa3PagedScheduleShape,
};
use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::Sdpa,
    paged_attention::{
        block_aligned_sliding_window_start,
        plan::{
            gather_prefill_workspace_for_lengths, DecodePlan, DecodePlanInput,
            GatherPrefillWorkspaceRequest, PrefixPrefillPlan, PrefixPrefillPlanInput,
        },
        AttentionBackendKind, Fp8AttentionScales, _PAD_SLOT_ID,
    },
    pipeline::text_models_inputs_processor::{
        FlashKMeta, FlashParams, PagedAttentionInputMetadata,
    },
};

static UNCALIBRATED_FP8_ATTENTION_WARNING: Once = Once::new();

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy)]
struct Fa3DecodeCandidate {
    key: Fa3DecodeScheduleKey,
    query_dtype: DType,
    query_contiguous: bool,
    key_cache_dtype: DType,
    value_cache_dtype: DType,
    mask_is_none: bool,
    shapes_match: bool,
    has_alibi: bool,
    has_sinks: bool,
    has_softcap: bool,
    has_sliding_window: bool,
    has_noncausal_mm_context: bool,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Fa3DecodeCandidate {
    fn schedule_key(self) -> Option<Fa3DecodeScheduleKey> {
        (self.query_dtype == DType::BF16
            && self.query_contiguous
            && self.key_cache_dtype == DType::F8E4M3
            && self.value_cache_dtype == DType::F8E4M3
            && self.mask_is_none
            && self.shapes_match
            && !self.has_alibi
            && !self.has_sinks
            && !self.has_softcap
            && !self.has_sliding_window
            && !self.has_noncausal_mm_context
            && self.key.supported())
        .then_some(self.key)
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy)]
struct FlashInferDecodeCall<'call, 'ctx> {
    ctx: &'call PagedForwardCtx<'ctx>,
    query: &'call Tensor,
    key_cache: &'call Tensor,
    value_cache: &'call Tensor,
    dev: &'call DeviceLocation,
    attention_mask: &'call AttentionMask,
}

#[derive(Clone, Copy)]
struct CacheScales<'a> {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    attention: Fp8AttentionScales,
    k: Option<&'a Tensor>,
    v: Option<&'a Tensor>,
}

impl CacheScales<'_> {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn flashinfer(self, key_cache: &Tensor) -> FlashInferKvCacheScales {
        if key_cache.dtype() == DType::F8E4M3 {
            FlashInferKvCacheScales {
                k: self.attention.k,
                v: self.attention.v,
            }
        } else {
            DEFAULT_FP8_KV_CACHE_SCALES
        }
    }
}

fn resolve_tensor_for_device(
    tensors: &HashMap<candle_core::DeviceLocation, Tensor>,
    device: &Device,
    what: &str,
) -> Result<Tensor> {
    if let Some(tensor) = tensors.get(&device.location()) {
        return Ok(tensor.clone());
    }
    if let Some(tensor) = tensors.values().next() {
        return tensor.to_device(device);
    }
    candle_core::bail!("Missing {what} tensor for {:?}", device.location())
}

fn checked_sequence_token_count(lengths: &[usize]) -> Result<usize> {
    let total = lengths.iter().try_fold(0usize, |total, &len| {
        total
            .checked_add(len)
            .ok_or_else(|| candle_core::Error::msg("sequence token count overflow"))
    })?;
    i32::try_from(total)
        .map_err(|_| candle_core::Error::msg("sequence token count exceeds kernel i32 limit"))?;
    Ok(total)
}

fn cumulative_seqlens_from_lengths(lengths: &[usize], device: &Device) -> Result<Tensor> {
    let expected_total = checked_sequence_token_count(lengths)?;
    let mut cumulative = Vec::with_capacity(lengths.len() + 1);
    let mut total = 0usize;
    cumulative.push(0u32);
    for &len in lengths {
        total += len;
        cumulative.push(u32::try_from(total).map_err(candle_core::Error::wrap)?);
    }
    debug_assert_eq!(total, expected_total);
    Tensor::new(&cumulative[..], &Device::Cpu)?.to_device(device)
}

fn block_aligned_window_len_for_query(
    full_len: usize,
    query_len: usize,
    window: usize,
    block_size: usize,
) -> usize {
    full_len - block_aligned_sliding_window_start(full_len, query_len, window, block_size)
}

fn cache_block_size(key_cache: &Tensor, value_cache: &Tensor) -> Result<usize> {
    match AttentionBackendKind::from_cache(key_cache, value_cache) {
        AttentionBackendKind::FlashInfer => Ok(key_cache.dims4()?.2),
        AttentionBackendKind::Standard => Ok(key_cache.dims5()?.3),
    }
}

fn cache_kv_shape(key_cache: &Tensor, value_cache: &Tensor) -> Result<(usize, usize)> {
    match AttentionBackendKind::from_cache(key_cache, value_cache) {
        AttentionBackendKind::FlashInfer => {
            let (_, num_kv_heads, _, head_size) = key_cache.dims4()?;
            Ok((num_kv_heads, head_size))
        }
        AttentionBackendKind::Standard => {
            let (_, num_kv_heads, head_size_blocks, _, x) = key_cache.dims5()?;
            Ok((num_kv_heads, head_size_blocks * x))
        }
    }
}

fn cache_input_shape(tensor: &Tensor) -> Result<(usize, usize, usize)> {
    match *tensor.dims() {
        [tokens, heads, head_size] => Ok((tokens, heads, head_size)),
        [batch, seq_len, heads, head_size] => Ok((
            batch
                .checked_mul(seq_len)
                .ok_or_else(|| candle_core::Error::msg("cache input token count overflow"))?,
            heads,
            head_size,
        )),
        _ => candle_core::bail!(
            "cache input must have shape [tokens, heads, head_size] or [batch, seq_len, heads, head_size], got {:?}",
            tensor.shape()
        ),
    }
}

fn cache_input_can_write_directly(tensor: &Tensor) -> Result<bool> {
    let (_, heads, head_size) = cache_input_shape(tensor)?;
    let dims = tensor.dims();
    let stride = tensor.stride();
    let row_stride = match *dims {
        [_, _, _] => stride[0],
        [batch, seq_len, _, _] => {
            if !cfg!(all(feature = "cuda", target_family = "unix")) {
                return Ok(false);
            }
            let row_stride = if seq_len == 1 { stride[0] } else { stride[1] };
            if batch > 1 && seq_len > 1 && stride[0] != seq_len.saturating_mul(row_stride) {
                return Ok(false);
            }
            row_stride
        }
        _ => unreachable!(),
    };
    Ok(stride[stride.len() - 1] == 1
        && stride[stride.len() - 2] == head_size
        && row_stride >= heads.saturating_mul(head_size))
}

fn write_kv_cache(
    key: &Tensor,
    value: &Tensor,
    scales: CacheScales<'_>,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let key_packed;
    let key = if cache_input_can_write_directly(key)? {
        key
    } else {
        key_packed = key.force_contiguous()?.reshape(cache_input_shape(key)?)?;
        &key_packed
    };
    let value_packed;
    let value = if cache_input_can_write_directly(value)? {
        value
    } else {
        value_packed = value
            .force_contiguous()?
            .reshape(cache_input_shape(value)?)?;
        &value_packed
    };
    match AttentionBackendKind::from_cache(key_cache, value_cache) {
        AttentionBackendKind::FlashInfer => {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            {
                reshape_and_cache_flashinfer(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    scales.flashinfer(key_cache),
                )
            }
            #[cfg(not(all(feature = "cuda", target_family = "unix")))]
            {
                unreachable!("FlashInfer cache is only available with CUDA")
            }
        }
        AttentionBackendKind::Standard => reshape_and_cache(
            key,
            value,
            scales.k,
            scales.v,
            key_cache,
            value_cache,
            slot_mapping,
        ),
    }
}

fn gather_kv_cache_for_layout(
    key_cache: &Tensor,
    value_cache: &Tensor,
    scales: CacheScales<'_>,
    block_tables: &Tensor,
    cu_kv: &Tensor,
    num_tokens: usize, // Must equal cu_kv[-1].
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    match AttentionBackendKind::from_cache(key_cache, value_cache) {
        AttentionBackendKind::FlashInfer => {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            {
                gather_kv_cache_flashinfer(
                    key_cache,
                    value_cache,
                    block_tables,
                    cu_kv,
                    num_tokens,
                    dtype,
                    scales.flashinfer(key_cache),
                )
            }
            #[cfg(not(all(feature = "cuda", target_family = "unix")))]
            {
                unreachable!("FlashInfer cache is only available with CUDA")
            }
        }
        AttentionBackendKind::Standard => mistralrs_paged_attn::gather_kv_cache(
            key_cache,
            value_cache,
            scales.k,
            scales.v,
            block_tables,
            cu_kv,
            num_tokens,
            dtype,
        ),
    }
}

fn new_token_lens_from_slot_mapping(
    slot_mapping: &Tensor,
    batch_size: usize,
    seq_len: usize,
) -> Result<Vec<usize>> {
    let slot_mapping_cpu = slot_mapping.to_device(&Device::Cpu)?;
    let slot_mapping_cpu = if slot_mapping_cpu.dims().len() == 2 {
        slot_mapping_cpu
    } else {
        slot_mapping_cpu.reshape((batch_size, seq_len))?
    };
    Ok(slot_mapping_cpu
        .to_vec2::<i64>()?
        .into_iter()
        .map(|row| row.into_iter().filter(|&slot| slot != _PAD_SLOT_ID).count())
        .collect())
}

fn query_layout_is_dense(query_lens: &[usize], batch_size: usize, seq_len: usize) -> bool {
    query_lens.iter().sum::<usize>() == batch_size * seq_len
}

fn validate_varlen_segment_partition(
    query_lens: &[usize],
    segment_lens: &[usize],
) -> Result<usize> {
    if query_lens.is_empty()
        || query_lens.contains(&0)
        || segment_lens.is_empty()
        || segment_lens.contains(&0)
    {
        candle_core::bail!("packed varlen segments contain an empty sequence");
    }
    let mut segment_index = 0usize;
    for &query_len in query_lens {
        let mut remaining = query_len;
        while remaining > 0 {
            let segment = segment_lens.get(segment_index).copied().ok_or_else(|| {
                candle_core::Error::msg("packed varlen segments do not cover every query")
            })?;
            if segment > remaining {
                candle_core::bail!("packed varlen segment crosses a logical query boundary");
            }
            remaining -= segment;
            segment_index += 1;
        }
    }
    if segment_index != segment_lens.len() {
        candle_core::bail!("packed varlen segments contain trailing queries");
    }
    Ok(segment_lens.iter().copied().max().unwrap_or(0))
}

fn decode_query_rows(query: &Tensor, kv_lens: &[usize]) -> Result<usize> {
    let query_rows = query.dim(0)?;
    if query_rows != kv_lens.len() {
        candle_core::bail!(
            "decode gather has {query_rows} query rows for {} KV rows",
            kv_lens.len()
        );
    }
    Ok(query_rows)
}

fn pad_packed_query(query: &Tensor, query_lens: &[usize]) -> Result<Tensor> {
    let (batch, heads, total_tokens, head_size) = query.dims4()?;
    if batch != 1 || query_lens.is_empty() || query_lens.contains(&0) {
        candle_core::bail!("packed sinks query has invalid logical dimensions");
    }
    let logical_tokens = query_lens.iter().try_fold(0usize, |total, &len| {
        total
            .checked_add(len)
            .ok_or_else(|| candle_core::Error::msg("packed sinks query length overflow"))
    })?;
    if logical_tokens != total_tokens {
        candle_core::bail!(
            "packed sinks query has {total_tokens} tokens for {logical_tokens} logical tokens"
        );
    }

    let max_query_len = query_lens.iter().copied().max().unwrap_or(0);
    let mut offset = 0usize;
    let mut rows = Vec::with_capacity(query_lens.len());
    for &query_len in query_lens {
        let row = query.narrow(2, offset, query_len)?;
        offset += query_len;
        if query_len == max_query_len {
            rows.push(row);
        } else {
            let padding = Tensor::zeros(
                (1, heads, max_query_len - query_len, head_size),
                query.dtype(),
                query.device(),
            )?;
            rows.push(Tensor::cat(&[&row, &padding], 2)?);
        }
    }
    Tensor::cat(&rows, 0)
}

fn repack_padded_query(output: &Tensor, query_lens: &[usize]) -> Result<Tensor> {
    let (batch, _, max_query_len, _) = output.dims4()?;
    if batch != query_lens.len()
        || query_lens.is_empty()
        || query_lens.contains(&0)
        || query_lens.iter().any(|&len| len > max_query_len)
    {
        candle_core::bail!("padded sinks output has invalid logical dimensions");
    }

    let mut rows = Vec::with_capacity(query_lens.len());
    for (batch_idx, &query_len) in query_lens.iter().enumerate() {
        rows.push(output.narrow(0, batch_idx, 1)?.narrow(2, 0, query_len)?);
    }
    Tensor::cat(&rows, 2)
}

fn should_use_gather_path(
    has_block_tables: bool,
    has_cached_prefix: bool,
    has_noncausal_mm_context: bool,
    mask_is_prefill: bool,
    single_token_first_prompt: bool,
    write_cache: bool,
) -> bool {
    if write_cache {
        (has_cached_prefix || has_noncausal_mm_context || mask_is_prefill) && has_block_tables
    } else {
        (has_cached_prefix
            || has_noncausal_mm_context
            || mask_is_prefill
            || single_token_first_prompt)
            && has_block_tables
    }
}

fn select_optional_view<'a, T>(
    use_full: bool,
    full: Option<&'a T>,
    regular: Option<&'a T>,
) -> Option<&'a T> {
    if use_full {
        full
    } else {
        regular
    }
}

fn noncausal_mm_view_is_valid(
    has_noncausal_mm_context: bool,
    has_block_tables: bool,
    has_mm_prefix_ranges: bool,
) -> bool {
    !has_noncausal_mm_context || has_block_tables && has_mm_prefix_ranges
}

fn unpack_gathered_kv(
    packed: &Tensor,
    kv_lens: &[usize],
    num_kv_heads: usize,
    head_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
    let mut start = 0;
    let mut unpacked = Vec::with_capacity(kv_lens.len());

    for &kv_len in kv_lens {
        let mut seq = packed
            .narrow(0, start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        if kv_len < max_kv {
            let pad = Tensor::zeros(
                (1, num_kv_heads, max_kv - kv_len, head_size),
                packed.dtype(),
                device,
            )?;
            seq = Tensor::cat(&[&seq, &pad], 2)?;
        }
        unpacked.push(seq);
        start += kv_len;
    }

    Tensor::cat(&unpacked, 0)
}

fn adjust_kv_mask(mask: &Tensor, kv_seq_len: usize) -> Result<Tensor> {
    let mask_dims = mask.dims();
    match mask.rank() {
        2 if mask_dims[1] > kv_seq_len => mask.narrow(1, mask_dims[1] - kv_seq_len, kv_seq_len),
        3 if mask_dims[2] > kv_seq_len => mask.narrow(2, mask_dims[2] - kv_seq_len, kv_seq_len),
        4 if mask_dims[3] > kv_seq_len => mask.narrow(3, mask_dims[3] - kv_seq_len, kv_seq_len),
        _ => Ok(mask.clone()),
    }
}

fn prefix_attention_output_layout(
    output: Tensor,
    attention_mask: &AttentionMask,
) -> Result<Tensor> {
    if matches!(attention_mask, AttentionMask::None) {
        output.transpose(1, 2)?.contiguous()
    } else {
        Ok(output)
    }
}

fn should_reconstruct_prefix_mask(
    attention_mask: &AttentionMask,
    prefix_causal: bool,
    has_noncausal_mm_context: bool,
    has_padding_or_window: bool,
) -> bool {
    matches!(attention_mask, AttentionMask::None)
        && (prefix_causal || has_noncausal_mm_context || has_padding_or_window)
}

fn should_use_packed_prefix_mask(
    packed: bool,
    query_layout_is_dense: bool,
    attention_mask: &AttentionMask,
    dense_flash_causal: bool,
) -> bool {
    packed
        && query_layout_is_dense
        && matches!(attention_mask, AttentionMask::CausalFlash)
        && !dense_flash_causal
}

fn prefix_prefill_is_causal(
    has_multi_token_query: bool,
    declared_causal: bool,
    has_mm_prefix_ranges: bool,
) -> bool {
    has_multi_token_query && (declared_causal || has_mm_prefix_ranges)
}

type SeqMmPrefixRanges = Vec<Vec<(usize, usize)>>;

#[allow(clippy::too_many_arguments)]
fn prefix_gather_causal_mask(
    query_lens: &[usize],
    kv_lens: &[usize],
    mm_prefix_ranges: Option<&[Vec<(usize, usize)>]>,
    q_max: usize,
    kv_max: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<AttentionMask> {
    let batch = query_lens.len();
    let mut mask = Vec::with_capacity(batch * q_max * kv_max);
    for (batch_idx, (&q_len, &kv_len)) in query_lens.iter().zip(kv_lens.iter()).enumerate() {
        let prefix_len = kv_len.saturating_sub(q_len);
        for q_idx in 0..q_max {
            for kv_idx in 0..kv_max {
                let masked = if q_idx >= q_len {
                    kv_idx != 0
                } else if kv_idx >= kv_len {
                    true
                } else {
                    let q_pos = prefix_len + q_idx;
                    let future = kv_idx > q_pos;
                    let too_old = sliding_window
                        .is_some_and(|window| q_pos >= window && kv_idx <= q_pos - window);
                    let mm_prefix = mm_prefix_ranges
                        .and_then(|ranges| ranges.get(batch_idx))
                        .is_some_and(|ranges| {
                            ranges.iter().any(|&(start, end)| {
                                q_pos >= start && q_pos < end && kv_idx >= start && kv_idx < end
                            })
                        });
                    (future || too_old) && !mm_prefix
                };
                mask.push(if masked { f32::NEG_INFINITY } else { 0.0 });
            }
        }
    }
    Ok(AttentionMask::Custom(
        Tensor::from_vec(mask, (batch, 1, q_max, kv_max), device)?.to_dtype(dtype)?,
    ))
}

fn packed_prefix_gather_causal_mask(
    query_lens: &[usize],
    kv_lens: &[usize],
    mm_prefix_ranges: Option<&[Vec<(usize, usize)>]>,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total_q = query_lens.iter().sum::<usize>();
    let total_kv = kv_lens.iter().sum::<usize>();
    let mut mask = vec![f32::NEG_INFINITY; total_q * total_kv];
    let mut q_offset = 0usize;
    let mut kv_offset = 0usize;
    for (batch_idx, (&q_len, &kv_len)) in query_lens.iter().zip(kv_lens).enumerate() {
        let prefix_len = kv_len.saturating_sub(q_len);
        for q_idx in 0..q_len {
            let q_pos = prefix_len + q_idx;
            for kv_idx in 0..kv_len {
                let future = kv_idx > q_pos;
                let too_old = sliding_window
                    .is_some_and(|window| q_pos >= window && kv_idx <= q_pos - window);
                let mm_prefix = mm_prefix_ranges
                    .and_then(|ranges| ranges.get(batch_idx))
                    .is_some_and(|ranges| {
                        ranges.iter().any(|&(start, end)| {
                            q_pos >= start && q_pos < end && kv_idx >= start && kv_idx < end
                        })
                    });
                if !(future || too_old) || mm_prefix {
                    mask[(q_offset + q_idx) * total_kv + kv_offset + kv_idx] = 0.0;
                }
            }
        }
        q_offset += q_len;
        kv_offset += kv_len;
    }
    Tensor::from_vec(mask, (1usize, 1usize, total_q, total_kv), device)?.to_dtype(dtype)
}

fn mm_prefix_ranges_from_tensor(tensor: Option<&Tensor>) -> Result<Option<SeqMmPrefixRanges>> {
    let Some(tensor) = tensor else {
        return Ok(None);
    };
    let ranges = tensor
        .to_device(&Device::Cpu)?
        .to_vec3::<i32>()?
        .into_iter()
        .map(|seq_ranges| {
            seq_ranges
                .into_iter()
                .filter_map(|range| {
                    let start = *range.first()?;
                    let end = *range.get(1)?;
                    (start < end).then_some((start as usize, end as usize))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(Some(ranges))
}

fn packed_varlen_flash_is_usable(
    device_is_cuda: bool,
    flash_attn_enabled: bool,
    dtype: DType,
    head_size: usize,
    has_softcap: bool,
    has_sliding_window: bool,
) -> bool {
    device_is_cuda
        && flash_attn_enabled
        && matches!(dtype, DType::F16 | DType::BF16)
        && crate::attention::flash_backend_supports_sdpa(head_size, has_softcap, has_sliding_window)
}

fn supports_packed_varlen_sdpa(query: &Tensor, head_size: usize, sdpa_params: &SdpaParams) -> bool {
    packed_varlen_flash_is_usable(
        query.device().is_cuda(),
        crate::using_flash_attn(),
        query.dtype(),
        head_size,
        sdpa_params.softcap.is_some(),
        sdpa_params.sliding_window.is_some(),
    )
}

fn supports_sinks_varlen_sdpa(query: &Tensor, head_size: usize) -> bool {
    crate::attention::sinks_backend_is_available(query, head_size)
}

fn should_use_decode_gather_varlen(
    supports_general_varlen: bool,
    has_sinks: bool,
    supports_sinks_varlen: bool,
    query_rows: usize,
) -> bool {
    if has_sinks {
        supports_sinks_varlen && query_rows > 1
    } else {
        supports_general_varlen
    }
}

#[derive(Clone, Copy)]
struct PagedForwardDims {
    batch_size: usize,
    attention_heads: usize,
    seq_len: usize,
    head_size: usize,
    key_value_heads: usize,
}

#[derive(Clone, Copy)]
struct PagedForwardTensors<'a> {
    query: &'a Tensor,
    key: &'a Tensor,
    value: &'a Tensor,
    attention_mask: &'a AttentionMask,
}

struct PagedForwardSetup<'a> {
    tensors: PagedForwardTensors<'a>,
    donor_cache_shape: Option<(usize, usize)>,
    input_metadata: &'a PagedAttentionInputMetadata,
    sdpa_params: &'a SdpaParams,
    flash_params: Option<&'a FlashParams>,
    write_cache: bool,
}

struct PagedForwardCtx<'a> {
    input_metadata: &'a PagedAttentionInputMetadata,
    sdpa_params: &'a SdpaParams,
    flash_params: Option<&'a FlashParams>,
    slot_mapping_full: &'a Tensor,
    slot_mapping: Tensor,
    dims: PagedForwardDims,
    use_full: bool,
    alibi_slopes: Option<Tensor>,
}

impl PagedForwardCtx<'_> {
    fn block_tables(&self, dev: &DeviceLocation) -> Option<&Tensor> {
        if self.use_full {
            self.input_metadata.full_block_tables.as_ref()?.get(dev)
        } else {
            self.input_metadata.block_tables.as_ref()?.get(dev)
        }
    }

    fn context_lens(&self, dev: &DeviceLocation) -> Option<&Tensor> {
        if self.use_full {
            self.input_metadata.full_context_lens.as_ref()?.get(dev)
        } else {
            self.input_metadata.context_lens.as_ref()?.get(dev)
        }
    }

    fn context_lens_cpu(&self) -> Option<&[usize]> {
        if self.use_full {
            self.input_metadata.full_paged_context_lens_cpu.as_deref()
        } else {
            self.input_metadata.paged_context_lens_cpu.as_deref()
        }
    }

    fn mm_prefix_ranges(&self, dev: &DeviceLocation) -> Option<&Tensor> {
        select_optional_view(
            self.use_full,
            self.input_metadata.full_mm_prefix_ranges.as_ref(),
            self.input_metadata.mm_prefix_ranges.as_ref(),
        )?
        .get(dev)
    }

    fn has_noncausal_mm_context(&self) -> bool {
        select_optional_view(
            self.use_full,
            self.input_metadata.full_mm_prefix_ranges.as_ref(),
            self.input_metadata.mm_prefix_ranges.as_ref(),
        )
        .is_some()
    }

    fn use_gather_path(&self, attention_mask: &AttentionMask, write_cache: bool) -> bool {
        let has_block_tables = self.input_metadata.block_tables.is_some();
        let has_cached_prefix = self.input_metadata.num_cached_tokens.is_some();
        let mask_is_prefill = !matches!(attention_mask, AttentionMask::None)
            && (!write_cache
                || self.input_metadata.query_lens.is_some()
                    && !self.input_metadata.is_first_prompt_chunk);
        let single_token_first_prompt =
            self.input_metadata.is_first_prompt_chunk && self.dims.seq_len == 1;
        should_use_gather_path(
            has_block_tables,
            has_cached_prefix,
            self.has_noncausal_mm_context(),
            mask_is_prefill,
            single_token_first_prompt,
            write_cache,
        )
    }
}

pub struct PagedAttention {
    alibi_slopes: Option<Tensor>,
    fp8_attention_scales: Fp8AttentionScales,
    fp8_attention_scales_calibrated: bool,
    // read only in the cuda FA3 path
    #[allow(dead_code)]
    fp8_q_scale: Tensor,
    fp8_k_scale: Tensor,
    fp8_v_scale: Tensor,
}

#[allow(dead_code)]
impl PagedAttention {
    pub fn fp8_attention_scales(&self) -> Fp8AttentionScales {
        self.fp8_attention_scales
    }

    pub fn has_calibrated_fp8_attention_scales(&self) -> bool {
        self.fp8_attention_scales_calibrated
    }
}

impl PagedAttention {
    pub fn new(head_dim: usize, device: &Device, alibi_slopes: Option<Vec<f32>>) -> Result<Self> {
        Self::new_with_fp8_attention_scales(head_dim, device, alibi_slopes, None)
    }

    pub fn new_with_fp8_attention_scales(
        head_dim: usize,
        device: &Device,
        alibi_slopes: Option<Vec<f32>>,
        fp8_attention_scales: Option<Fp8AttentionScales>,
    ) -> Result<Self> {
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            assert_eq!(alibi_slopes.len(), head_dim);
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        let fp8_attention_scales_calibrated = fp8_attention_scales.is_some();
        let fp8_attention_scales = fp8_attention_scales.unwrap_or_default().validate()?;
        Ok(Self {
            alibi_slopes,
            fp8_attention_scales,
            fp8_attention_scales_calibrated,
            fp8_q_scale: Tensor::new(fp8_attention_scales.q, device)?,
            fp8_k_scale: Tensor::new(fp8_attention_scales.k, device)?,
            fp8_v_scale: Tensor::new(fp8_attention_scales.v, device)?,
        })
    }

    #[cfg(test)]
    fn fp8_scale_tensors(&self) -> [&Tensor; 3] {
        [&self.fp8_q_scale, &self.fp8_k_scale, &self.fp8_v_scale]
    }

    fn cache_scales<'a>(&'a self, key_cache: &Tensor) -> CacheScales<'a> {
        let (k, v) = if key_cache.dtype() == DType::F8E4M3 {
            if !self.fp8_attention_scales_calibrated {
                UNCALIBRATED_FP8_ATTENTION_WARNING.call_once(|| {
                    tracing::warn!(
                        "FP8 KV cache is using uncalibrated unit Q/K/V scales; load offline q_scale, k_scale, and v_scale for calibrated serving"
                    );
                });
            }
            (Some(&self.fp8_k_scale), Some(&self.fp8_v_scale))
        } else {
            (None, None)
        };
        CacheScales {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            attention: self.fp8_attention_scales,
            k,
            v,
        }
    }

    /// Gather a batch of sequences' full KV from the paged cache into one contiguous
    /// `[num_seqs, kv_heads, kv_len, head_size]` pair. All sequences share the same kv_len
    /// (the block-diffusion scheduler buckets by context length). Block-diffusion canvas
    /// passes snapshot the frozen encoder cache once per block and reuse it across all
    /// denoising steps. Metadata block-table rows are per query row; one row per sequence
    /// is selected by stride.
    pub fn gather_canvas_kv(
        &self,
        key_cache: &Tensor,
        value_cache: &Tensor,
        input_metadata: &PagedAttentionInputMetadata,
        num_seqs: usize,
        kv_len: usize,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let device = key_cache.device();
        let loc = device.location();
        let block_tables = input_metadata
            .full_block_tables
            .as_ref()
            .and_then(|m| m.get(&loc))
            .or_else(|| {
                input_metadata
                    .block_tables
                    .as_ref()
                    .and_then(|m| m.get(&loc))
            })
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "canvas KV gather requires block tables (full: {:?}, windowed: {:?}, want {:?})",
                    input_metadata
                        .full_block_tables
                        .as_ref()
                        .map(|m| m.keys().collect::<Vec<_>>()),
                    input_metadata
                        .block_tables
                        .as_ref()
                        .map(|m| m.keys().collect::<Vec<_>>()),
                    loc,
                ))
            })?;
        let total_rows = block_tables.dim(0)?;
        let rows_per_seq = total_rows / num_seqs;
        let table = if rows_per_seq <= 1 {
            block_tables.narrow(0, 0, num_seqs)?
        } else {
            let row_idx: Vec<u32> = (0..num_seqs)
                .map(|i| u32::try_from(i * rows_per_seq).map_err(candle_core::Error::wrap))
                .collect::<candle_core::Result<_>>()?;
            block_tables.index_select(&Tensor::from_vec(row_idx, (num_seqs,), device)?, 0)?
        };
        let kv_lens = vec![kv_len; num_seqs];
        let num_tokens = checked_sequence_token_count(&kv_lens)?;
        let cu_kv = cumulative_seqlens_from_lengths(&kv_lens, device)?;
        let scales = self.cache_scales(key_cache);
        let (k, v) = gather_kv_cache_for_layout(
            key_cache,
            value_cache,
            scales,
            &table,
            &cu_kv,
            num_tokens,
            dtype,
        )?;
        // Packed [num_seqs * kv_len, kv_heads, head_size] -> [num_seqs, kv_heads, kv_len, hd].
        let unpack = |t: Tensor| -> Result<Tensor> {
            let (_, kv_heads, head_size) = t.dims3()?;
            t.reshape((num_seqs, kv_len, kv_heads, head_size))?
                .transpose(1, 2)?
                .contiguous()
        };
        Ok((unpack(k)?, unpack(v)?))
    }

    fn build_forward_ctx<'a>(&self, setup: PagedForwardSetup<'a>) -> Result<PagedForwardCtx<'a>> {
        let slot_mapping_full = setup
            .input_metadata
            .slot_mappings
            .get(&setup.tensors.query.device().location())
            .unwrap();
        let dims = slot_mapping_full.dims();
        let slot_mapping = if dims.len() > 1 {
            slot_mapping_full.flatten(0, dims.len() - 1)?
        } else {
            slot_mapping_full.clone()
        };

        let (batch_size, attention_heads, seq_len, head_size) =
            setup.tensors.query.shape().dims4()?;
        let (key_value_heads, kv_head_size) = if !setup.write_cache {
            setup.donor_cache_shape.expect("missing donor cache shape")
        } else {
            let (_, key_value_heads, _, kv_head_size) = setup.tensors.key.shape().dims4()?;
            (key_value_heads, kv_head_size)
        };
        if kv_head_size != head_size {
            candle_core::bail!(
                "paged attention query/cache head dim mismatch: query={head_size}, kv={kv_head_size}"
            );
        }

        let has_flashinfer_sliding_view = setup
            .input_metadata
            .flashinfer
            .as_ref()
            .is_some_and(|metadata| metadata.views.sliding.is_some());
        let use_full = setup.sdpa_params.sliding_window.is_none()
            && (setup.input_metadata.full_block_tables.is_some() || has_flashinfer_sliding_view);
        let alibi_slopes = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            Some(alibi_slopes.to_device(setup.tensors.query.device())?)
        } else {
            None
        };

        Ok(PagedForwardCtx {
            input_metadata: setup.input_metadata,
            sdpa_params: setup.sdpa_params,
            flash_params: setup.flash_params,
            slot_mapping_full,
            slot_mapping,
            dims: PagedForwardDims {
                batch_size,
                attention_heads,
                seq_len,
                head_size,
                key_value_heads,
            },
            use_full,
            alibi_slopes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn try_prefix_gather_prefill(
        &self,
        ctx: &PagedForwardCtx<'_>,
        tensors: PagedForwardTensors<'_>,
        key_cache: &mut Option<Tensor>,
        value_cache: &mut Option<Tensor>,
        write_cache: bool,
    ) -> Result<Option<Tensor>> {
        let dev = tensors.query.device().location();
        let block_tables = ctx.block_tables(&dev);
        let mm_prefix_ranges = ctx.mm_prefix_ranges(&dev);
        if !noncausal_mm_view_is_valid(
            ctx.has_noncausal_mm_context(),
            block_tables.is_some(),
            mm_prefix_ranges.is_some(),
        ) {
            let view = if ctx.use_full { "full" } else { "sliding" };
            candle_core::bail!(
                "noncausal multimodal prefix attention is missing {view} cache metadata for {dev:?}"
            );
        }
        if !ctx.use_gather_path(tensors.attention_mask, write_cache) {
            return Ok(None);
        }

        let block_tables = block_tables.ok_or_else(|| {
            candle_core::Error::msg(format!(
                "paged prefix attention is missing block tables for {dev:?}"
            ))
        })?;
        if write_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let k_flat = tensors.key.transpose(1, 2)?;
            let v_flat = tensors.value.transpose(1, 2)?;
            let key_cache = key_cache.as_mut().unwrap();
            let value_cache = value_cache.as_mut().unwrap();
            let scales = self.cache_scales(key_cache);
            write_kv_cache(
                &k_flat,
                &v_flat,
                scales,
                key_cache,
                value_cache,
                &ctx.slot_mapping,
            )?;
        }

        assert!(
            ctx.alibi_slopes.is_none(),
            "alibi slopes not supported in prefix cache path"
        );

        let device = tensors.query.device();
        let query_lens = match ctx.input_metadata.query_lens.clone() {
            Some(query_lens) => query_lens,
            // Fallback costs a GPU->CPU sync per layer; the inputs processor normally fills it.
            None => new_token_lens_from_slot_mapping(
                ctx.slot_mapping_full,
                ctx.dims.batch_size,
                ctx.dims.seq_len,
            )?,
        };
        let full_kv_lens = if let Some(lens) = ctx
            .input_metadata
            .full_paged_context_lens_cpu
            .as_deref()
            .filter(|lens| lens.len() == query_lens.len())
        {
            lens.to_vec()
        } else if let Some(num_cached_tokens) = ctx.input_metadata.num_cached_tokens.as_ref() {
            num_cached_tokens
                .iter()
                .zip(query_lens.iter())
                .map(|(&cached, &query_len)| cached + query_len)
                .collect::<Vec<_>>()
        } else {
            query_lens.clone()
        };
        let block_size =
            cache_block_size(key_cache.as_ref().unwrap(), value_cache.as_ref().unwrap())?;
        let kv_lens = if let Some(lens) = ctx
            .context_lens_cpu()
            .filter(|lens| lens.len() == query_lens.len())
        {
            lens.to_vec()
        } else if let Some(window) = ctx.sdpa_params.sliding_window {
            if !ctx.use_full {
                full_kv_lens
                    .iter()
                    .zip(query_lens.iter())
                    .map(|(&len, &query_len)| {
                        block_aligned_window_len_for_query(len, query_len, window, block_size)
                    })
                    .collect::<Vec<_>>()
            } else {
                full_kv_lens
            }
        } else {
            full_kv_lens
        };
        let cu_kv = if ctx.sdpa_params.sliding_window.is_none() {
            if let Some(map) = ctx.input_metadata.cu_seqlens_kv.as_ref() {
                resolve_tensor_for_device(map, device, "cu_seqlens_kv")?
            } else {
                cumulative_seqlens_from_lengths(&kv_lens, device)?
            }
        } else {
            cumulative_seqlens_from_lengths(&kv_lens, device)?
        };
        let num_kv_tokens = checked_sequence_token_count(&kv_lens)?;
        let query_layout_is_dense =
            query_layout_is_dense(&query_lens, ctx.dims.batch_size, ctx.dims.seq_len);
        let declared_causal = ctx.flash_params.map_or(
            ctx.input_metadata.prompt_chunk_attention_policy
                == crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
            |params| params.causal,
        );
        let prefix_causal = prefix_prefill_is_causal(
            query_lens.iter().any(|&len| len > 1),
            declared_causal,
            mm_prefix_ranges.is_some(),
        );
        let causality_known = !tensors.attention_mask.is_custom() || ctx.flash_params.is_some();
        let attention_backend = AttentionBackendKind::from_cache(
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
        );
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        let fa3_supported = fa3_prefill_cache_num_sm(
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            ctx.dims.attention_heads,
            ctx.dims.key_value_heads,
            ctx.dims.head_size,
            block_size,
        )?
        .is_some();
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        let fa3_supported = false;
        let prefill_plan_input = PrefixPrefillPlanInput {
            device_is_cuda: tensors.query.device().is_cuda(),
            dtype: tensors.query.dtype(),
            cache_dtype: key_cache.as_ref().unwrap().dtype(),
            has_alibi: ctx.alibi_slopes.is_some(),
            has_sinks: ctx.sdpa_params.sinks.is_some(),
            has_custom_mask: tensors.attention_mask.is_custom(),
            causality_known,
            head_size: ctx.dims.head_size,
            has_softcap: ctx.sdpa_params.softcap.is_some(),
            has_sliding_window: ctx.sdpa_params.sliding_window.is_some(),
            query_layout_is_dense,
            query_len: ctx.dims.seq_len,
            q_heads: ctx.dims.attention_heads,
            kv_heads: ctx.dims.key_value_heads,
            writes_cache: write_cache,
            is_causal: prefix_causal,
            has_noncausal_mm_context: mm_prefix_ranges.is_some(),
            fa3_supported,
            block_size,
            attention_backend,
        };
        let prefill_plan = PrefixPrefillPlan::choose(prefill_plan_input);
        if matches!(prefill_plan, PrefixPrefillPlan::GatherSdpa) {
            if let Some(limit) = ctx.input_metadata.prefix_gather_workspace_limit {
                let v_head_dim =
                    tensors.value.dims().last().copied().ok_or_else(|| {
                        candle_core::Error::msg("value tensor has no head dimension")
                    })?;
                let required =
                    gather_prefill_workspace_for_lengths(GatherPrefillWorkspaceRequest {
                        query_lens: &query_lens,
                        kv_lens: &kv_lens,
                        q_heads: ctx.dims.attention_heads,
                        kv_heads: ctx.dims.key_value_heads,
                        k_head_dim: ctx.dims.head_size,
                        v_head_dim,
                        dtype: tensors.query.dtype(),
                        plan_input: prefill_plan_input,
                    })?;
                if required > limit {
                    candle_core::bail!(
                        "prompt KV gather requires {required} bytes, exceeding its preflight workspace limit of {limit} bytes"
                    );
                }
            }
        }
        match prefill_plan {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            PrefixPrefillPlan::Fa3Fp8Paged => {
                let output = self.run_fa3_paged_prefill(
                    ctx,
                    tensors.query,
                    key_cache.as_ref().unwrap(),
                    value_cache.as_ref().unwrap(),
                    prefix_causal,
                )?;
                return prefix_attention_output_layout(output, tensors.attention_mask).map(Some);
            }
            #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
            PrefixPrefillPlan::FlashAttentionPaged => {
                let output = self.run_flash_attention_paged_prefill(
                    ctx,
                    tensors.query,
                    key_cache.as_ref().unwrap(),
                    value_cache.as_ref().unwrap(),
                    block_tables,
                    &query_lens,
                    &kv_lens,
                    &cu_kv,
                    block_size,
                    prefix_causal,
                    mm_prefix_ranges,
                )?;
                return prefix_attention_output_layout(output, tensors.attention_mask).map(Some);
            }
            PrefixPrefillPlan::GatherSdpa => {}
        }
        let simple_full_causal = matches!(tensors.attention_mask, AttentionMask::CausalFlash)
            && ctx.sdpa_params.sliding_window.is_none()
            && mm_prefix_ranges.is_none()
            && query_lens.len() == 1;

        let key_cache_ref = key_cache.as_ref().unwrap();
        let scales = self.cache_scales(key_cache_ref);
        let (k_gathered, v_gathered) = gather_kv_cache_for_layout(
            key_cache_ref,
            value_cache.as_ref().unwrap(),
            scales,
            block_tables,
            &cu_kv,
            num_kv_tokens,
            tensors.query.dtype(),
        )?;
        let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
        // Pure-causal prefix prefills run flash varlen over the gathered KV: fa2 aligns causal
        // bottom-right when kv_len > q_len, so no O(q*kv) mask or score materialization is needed.
        let pure_causal_varlen = prefix_causal
            && causality_known
            && mm_prefix_ranges.is_none()
            && ctx.sdpa_params.sliding_window.is_none()
            && !tensors.attention_mask.is_custom()
            && query_layout_is_dense
            && supports_packed_varlen_sdpa(tensors.query, ctx.dims.head_size, ctx.sdpa_params);
        if pure_causal_varlen {
            let cu_q = if let Some(fp) = ctx.flash_params {
                if !fp.cumulative_seqlens_q.is_empty() {
                    resolve_tensor_for_device(
                        &fp.cumulative_seqlens_q,
                        device,
                        "cumulative_seqlens_q",
                    )?
                } else {
                    cumulative_seqlens_from_lengths(&query_lens, device)?
                }
            } else {
                cumulative_seqlens_from_lengths(&query_lens, device)?
            };
            let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let mut cu_q_map = HashMap::new();
            cu_q_map.insert(device.location(), cu_q);
            let mut cu_kv_map = HashMap::new();
            cu_kv_map.insert(device.location(), cu_kv);
            let prefix_flash_params = FlashParams {
                max_q: u32::try_from(query_lens.iter().copied().max().unwrap_or(0))
                    .map_err(candle_core::Error::wrap)?,
                cumulative_seqlens_q: cu_q_map,
                logical_k: FlashKMeta {
                    max: u32::try_from(kv_lens.iter().copied().max().unwrap_or(0))
                        .map_err(candle_core::Error::wrap)?,
                    cumulative_seqlens: cu_kv_map,
                },
                sliding_k: None,
                causal: true,
                packed: ctx.flash_params.is_some_and(|params| params.packed),
                varlen_segment_lens: None,
            };
            let output = Sdpa.run_attention(
                tensors.query,
                &k_4d,
                &v_4d,
                &AttentionMask::CausalFlash,
                Some(&prefix_flash_params),
                ctx.sdpa_params,
            )?;
            return prefix_attention_output_layout(output, tensors.attention_mask).map(Some);
        }
        let mm_prefix_ranges_cpu = mm_prefix_ranges_from_tensor(mm_prefix_ranges)?;
        let has_padding_or_window = query_lens.iter().any(|&len| len != ctx.dims.seq_len)
            || kv_lens.iter().any(|&len| len != max_kv)
            || ctx.sdpa_params.sliding_window.is_some();
        let dense_flash_causal = query_layout_is_dense
            && mm_prefix_ranges.is_none()
            && tensors.query.device().is_cuda()
            && supports_packed_varlen_sdpa(tensors.query, ctx.dims.head_size, ctx.sdpa_params);
        if should_use_packed_prefix_mask(
            ctx.flash_params.is_some_and(|params| params.packed),
            query_layout_is_dense,
            tensors.attention_mask,
            dense_flash_causal,
        ) {
            let mask = packed_prefix_gather_causal_mask(
                &query_lens,
                &kv_lens,
                mm_prefix_ranges_cpu.as_deref(),
                ctx.sdpa_params.sliding_window,
                tensors.query.dtype(),
                device,
            )?;
            let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let output = Sdpa.run_attention_noflash(
                tensors.query,
                &k_4d,
                &v_4d,
                Some(&mask),
                ctx.sdpa_params,
                false,
            )?;
            return prefix_attention_output_layout(output, tensors.attention_mask).map(Some);
        }
        let adjusted_mask = match tensors.attention_mask {
            AttentionMask::Custom(t) => AttentionMask::Custom(adjust_kv_mask(t, max_kv)?),
            AttentionMask::CausalFlash if simple_full_causal => AttentionMask::None,
            AttentionMask::CausalFlash if dense_flash_causal => AttentionMask::CausalFlash,
            AttentionMask::CausalFlash => prefix_gather_causal_mask(
                &query_lens,
                &kv_lens,
                mm_prefix_ranges_cpu.as_deref(),
                ctx.dims.seq_len,
                max_kv,
                ctx.sdpa_params.sliding_window,
                tensors.query.dtype(),
                device,
            )?,
            AttentionMask::None
                if should_reconstruct_prefix_mask(
                    tensors.attention_mask,
                    prefix_causal,
                    ctx.has_noncausal_mm_context(),
                    has_padding_or_window,
                ) =>
            {
                prefix_gather_causal_mask(
                    &query_lens,
                    &kv_lens,
                    mm_prefix_ranges_cpu.as_deref(),
                    ctx.dims.seq_len,
                    max_kv,
                    ctx.sdpa_params.sliding_window,
                    tensors.query.dtype(),
                    device,
                )?
            }
            other => other.clone(),
        };

        if query_layout_is_dense
            && !adjusted_mask.is_custom()
            && supports_packed_varlen_sdpa(tensors.query, ctx.dims.head_size, ctx.sdpa_params)
        {
            let cu_q = if let Some(fp) = ctx.flash_params {
                if !fp.cumulative_seqlens_q.is_empty() {
                    resolve_tensor_for_device(
                        &fp.cumulative_seqlens_q,
                        device,
                        "cumulative_seqlens_q",
                    )?
                } else {
                    cumulative_seqlens_from_lengths(&query_lens, device)?
                }
            } else {
                cumulative_seqlens_from_lengths(&query_lens, device)?
            };
            let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let mut cu_q_map = HashMap::new();
            cu_q_map.insert(device.location(), cu_q);
            let mut cu_kv_map = HashMap::new();
            cu_kv_map.insert(device.location(), cu_kv);
            let prefix_flash_params = FlashParams {
                max_q: u32::try_from(query_lens.iter().copied().max().unwrap_or(0))
                    .map_err(candle_core::Error::wrap)?,
                cumulative_seqlens_q: cu_q_map,
                logical_k: FlashKMeta {
                    max: u32::try_from(kv_lens.iter().copied().max().unwrap_or(0))
                        .map_err(candle_core::Error::wrap)?,
                    cumulative_seqlens: cu_kv_map,
                },
                sliding_k: None,
                causal: prefix_causal,
                packed: ctx.flash_params.is_some_and(|params| params.packed),
                varlen_segment_lens: None,
            };
            if simple_full_causal {
                let output = Sdpa.run_attention_noflash(
                    tensors.query,
                    &k_4d,
                    &v_4d,
                    None,
                    ctx.sdpa_params,
                    prefix_flash_params.causal,
                )?;
                return prefix_attention_output_layout(output, tensors.attention_mask).map(Some);
            }
            let output = Sdpa.run_attention(
                tensors.query,
                &k_4d,
                &v_4d,
                &adjusted_mask,
                Some(&prefix_flash_params),
                ctx.sdpa_params,
            )?;
            return prefix_attention_output_layout(output, tensors.attention_mask).map(Some);
        }

        let k_batched = unpack_gathered_kv(
            &k_gathered,
            &kv_lens,
            ctx.dims.key_value_heads,
            ctx.dims.head_size,
            device,
        )?;
        let v_batched = unpack_gathered_kv(
            &v_gathered,
            &kv_lens,
            ctx.dims.key_value_heads,
            ctx.dims.head_size,
            device,
        )?;
        let output = Sdpa.run_attention(
            tensors.query,
            &k_batched,
            &v_batched,
            &adjusted_mask,
            None,
            ctx.sdpa_params,
        )?;
        prefix_attention_output_layout(output, tensors.attention_mask).map(Some)
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn run_fa3_paged_prefill(
        &self,
        ctx: &PagedForwardCtx<'_>,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        causal: bool,
    ) -> Result<Tensor> {
        let metadata = ctx
            .input_metadata
            .flashinfer
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("FA3 prefill metadata is missing"))?;
        let num_sm = fa3_device_num_sm(query.device())
            .ok_or_else(|| candle_core::Error::msg("FA3 prefill requires an SM90 CUDA device"))?;
        let (num_pages, kv_heads, page_size, head_dim) = key_cache.dims4()?;
        if num_pages == 0
            || value_cache.dims4()? != key_cache.dims4()?
            || query.dims4()?
                != (
                    ctx.dims.batch_size,
                    ctx.dims.attention_heads,
                    ctx.dims.seq_len,
                    head_dim,
                )
        {
            candle_core::bail!("FA3 prefill cache/query shape invariant failed");
        }
        let key = (Fa3PagedScheduleShape {
            device: query.device().location(),
            view: Fa3DecodeView::Logical,
            batch: ctx.dims.batch_size,
            query_len: ctx.dims.seq_len,
            causal,
            q_heads: ctx.dims.attention_heads,
            kv_heads,
            head_dim,
            page_size,
        })
        .prefill_schedule_key(num_sm)
        .ok_or_else(|| candle_core::Error::msg("FA3 prefill schedule invariant failed"))?;
        let query = query
            .transpose(1, 2)?
            .reshape((
                ctx.dims.batch_size.saturating_mul(ctx.dims.seq_len),
                ctx.dims.attention_heads,
                ctx.dims.head_size,
            ))?
            .contiguous()?;
        let output =
            with_fa3_prefill_workspace(metadata, key, key_cache, query.device(), |buffers| {
                fa3_fp8_decode(Fa3DecodeParams {
                    query: &query,
                    quantized_query: &buffers.query,
                    key_cache,
                    value_cache,
                    page_table: &buffers.page_table,
                    seqused_k: &buffers.seqused_k,
                    cu_seqlens_q: &buffers.cu_seqlens_q,
                    scheduler_metadata: &buffers.scheduler_metadata,
                    output_accum: &buffers.output_accum,
                    lse_accum: &buffers.lse_accum,
                    output_lse: &buffers.output_lse,
                    q_descale: &self.fp8_q_scale,
                    k_descale: &self.fp8_k_scale,
                    v_descale: &self.fp8_v_scale,
                    schedule: buffers.schedule(key)?,
                    softmax_scale: ctx.sdpa_params.softmax_scale,
                })
            })?;
        output
            .reshape((
                ctx.dims.batch_size,
                ctx.dims.seq_len,
                ctx.dims.attention_heads,
                ctx.dims.head_size,
            ))?
            .transpose(1, 2)
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    #[allow(clippy::too_many_arguments)]
    fn run_flash_attention_paged_prefill(
        &self,
        ctx: &PagedForwardCtx<'_>,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_tables: &Tensor,
        query_lens: &[usize],
        kv_lens: &[usize],
        cu_kv: &Tensor,
        block_size: usize,
        causal: bool,
        mm_prefix_ranges: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = query.device();
        let cu_q = if let Some(fp) = ctx.flash_params {
            if !fp.cumulative_seqlens_q.is_empty() {
                resolve_tensor_for_device(&fp.cumulative_seqlens_q, device, "cumulative_seqlens_q")?
            } else {
                cumulative_seqlens_from_lengths(query_lens, device)?
            }
        } else {
            cumulative_seqlens_from_lengths(query_lens, device)?
        };
        let q_flat =
            query
                .transpose(1, 2)?
                .reshape(((), ctx.dims.attention_heads, ctx.dims.head_size))?;
        let k_paged = key_cache.transpose(1, 2)?;
        let v_paged = value_cache.transpose(1, 2)?;
        let window_size_right = causal.then_some(0);
        let out = mistralrs_flash_attn::flash_attn_varlen_paged_windowed(
            &q_flat,
            &k_paged,
            &v_paged,
            &cu_q,
            cu_kv,
            block_tables,
            mm_prefix_ranges,
            query_lens.iter().copied().max().unwrap_or(0),
            kv_lens.iter().copied().max().unwrap_or(0),
            ctx.sdpa_params.softmax_scale,
            sliding_window_left(ctx.sdpa_params.sliding_window),
            window_size_right,
            block_size,
            ctx.sdpa_params.softcap,
        )?;
        out.reshape((
            ctx.dims.batch_size,
            ctx.dims.seq_len,
            ctx.dims.attention_heads,
            ctx.dims.head_size,
        ))?
        .transpose(1, 2)
    }

    fn try_regular_prompt(
        &self,
        ctx: &PagedForwardCtx<'_>,
        tensors: PagedForwardTensors<'_>,
        key_cache: &mut Option<Tensor>,
        value_cache: &mut Option<Tensor>,
        write_cache: bool,
    ) -> Result<Option<Tensor>> {
        let single_token_first_prompt =
            write_cache && ctx.input_metadata.is_first_prompt_chunk && ctx.dims.seq_len == 1;
        let custom_decode = tensors.attention_mask.is_custom()
            && ctx.input_metadata.query_lens.is_none()
            && !ctx.input_metadata.is_first_prompt_chunk;
        if custom_decode
            || matches!(tensors.attention_mask, AttentionMask::None) && !single_token_first_prompt
        {
            return Ok(None);
        }

        let att = if ctx.flash_params.is_some_and(|params| params.packed)
            && ctx.sdpa_params.sinks.is_some()
        {
            let query_lens = ctx.input_metadata.query_lens.as_deref().ok_or_else(|| {
                candle_core::Error::msg("packed sinks prefill is missing logical query lengths")
            })?;
            let padded_query = pad_packed_query(tensors.query, query_lens)?;
            let padded_output = Sdpa.run_attention(
                &padded_query,
                tensors.key,
                tensors.value,
                tensors.attention_mask,
                ctx.flash_params,
                ctx.sdpa_params,
            )?;
            repack_padded_query(&padded_output, query_lens)?
        } else {
            Sdpa.run_attention(
                tensors.query,
                tensors.key,
                tensors.value,
                tensors.attention_mask,
                ctx.flash_params,
                ctx.sdpa_params,
            )?
        };

        if write_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let key = tensors.key.transpose(1, 2)?;
            let value = tensors.value.transpose(1, 2)?;
            let key_cache = key_cache.as_mut().unwrap();
            let value_cache = value_cache.as_mut().unwrap();
            let scales = self.cache_scales(key_cache);
            write_kv_cache(
                &key,
                &value,
                scales,
                key_cache,
                value_cache,
                &ctx.slot_mapping,
            )?;
        }
        Ok(Some(att))
    }

    fn run_decode(
        &self,
        ctx: &PagedForwardCtx<'_>,
        tensors: PagedForwardTensors<'_>,
        key_cache: &mut Option<Tensor>,
        value_cache: &mut Option<Tensor>,
        write_cache: bool,
    ) -> Result<Tensor> {
        let query = if ctx.dims.seq_len > 1 {
            tensors.query.transpose(1, 2)?.reshape((
                (),
                ctx.dims.attention_heads,
                ctx.dims.head_size,
            ))?
        } else {
            tensors
                .query
                .reshape(((), ctx.dims.attention_heads, ctx.dims.head_size))?
        };
        let (key, value) = if write_cache {
            (
                Some(tensors.key.transpose(1, 2)?),
                Some(tensors.value.transpose(1, 2)?),
            )
        } else {
            (None, None)
        };

        if write_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let key_cache = key_cache.as_mut().unwrap();
            let value_cache = value_cache.as_mut().unwrap();
            let scales = self.cache_scales(key_cache);
            write_kv_cache(
                key.as_ref().unwrap(),
                value.as_ref().unwrap(),
                scales,
                key_cache,
                value_cache,
                &ctx.slot_mapping,
            )?;
        }

        let dev = query.device().location();
        let key_cache_ref = key_cache.as_ref().unwrap();
        let value_cache_ref = value_cache.as_ref().unwrap();
        if tensors.attention_mask.is_custom() {
            return self.run_decode_gather_sdpa(
                ctx,
                &query,
                key_cache_ref,
                value_cache_ref,
                &dev,
                tensors.attention_mask,
            );
        }
        let attention_backend = AttentionBackendKind::from_cache(key_cache_ref, value_cache_ref);
        match DecodePlan::choose(DecodePlanInput {
            attention_backend,
            head_size: ctx.dims.head_size,
            has_alibi: ctx.alibi_slopes.is_some(),
            has_sinks: ctx.sdpa_params.sinks.is_some(),
            has_sliding_window: ctx.sdpa_params.sliding_window.is_some(),
        })? {
            DecodePlan::GatherSdpa => self.run_decode_gather_sdpa(
                ctx,
                &query,
                key_cache_ref,
                value_cache_ref,
                &dev,
                tensors.attention_mask,
            ),
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            DecodePlan::FlashInfer(_) => self.run_flashinfer_decode(FlashInferDecodeCall {
                ctx,
                query: &query,
                key_cache: key_cache_ref,
                value_cache: value_cache_ref,
                dev: &dev,
                attention_mask: tensors.attention_mask,
            }),
            DecodePlan::PagedAttention => {
                self.run_standard_paged_decode(ctx, &query, key_cache_ref, value_cache_ref, &dev)
            }
        }
    }

    fn run_decode_gather_sdpa(
        &self,
        ctx: &PagedForwardCtx<'_>,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        dev: &DeviceLocation,
        attention_mask: &AttentionMask,
    ) -> Result<Tensor> {
        let block_tables = ctx.block_tables(dev).unwrap();
        let kv_lens: Vec<usize> = match ctx.context_lens_cpu() {
            Some(lens) => lens.to_vec(),
            // Fallback costs a GPU->CPU sync per layer per decode step.
            None => {
                let context_lens_t = ctx.context_lens(dev).unwrap();
                match context_lens_t.dtype() {
                    DType::U32 => context_lens_t
                        .to_vec1::<u32>()?
                        .into_iter()
                        .map(|len| len as usize)
                        .collect(),
                    DType::I32 => context_lens_t
                        .to_vec1::<i32>()?
                        .into_iter()
                        .map(|len| len as usize)
                        .collect(),
                    other => candle_core::bail!("unexpected context_lens dtype {other:?}"),
                }
            }
        };
        let query_rows = decode_query_rows(query, &kv_lens)?;
        let num_kv_tokens = checked_sequence_token_count(&kv_lens)?;
        let cu_kv = cumulative_seqlens_from_lengths(&kv_lens, query.device())?;
        let scales = self.cache_scales(key_cache);
        let (k_gathered, v_gathered) = gather_kv_cache_for_layout(
            key_cache,
            value_cache,
            scales,
            block_tables,
            &cu_kv,
            num_kv_tokens,
            query.dtype(),
        )?;
        let q_4d = query.reshape((query_rows, ctx.dims.attention_heads, 1, ctx.dims.head_size))?;

        if !attention_mask.is_custom()
            && should_use_decode_gather_varlen(
                supports_packed_varlen_sdpa(query, ctx.dims.head_size, ctx.sdpa_params),
                ctx.sdpa_params.sinks.is_some(),
                supports_sinks_varlen_sdpa(query, ctx.dims.head_size),
                query_rows,
            )
        {
            let cu_q = cumulative_seqlens_from_lengths(&vec![1usize; query_rows], query.device())?;
            let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let mut cu_q_map = HashMap::new();
            cu_q_map.insert(*dev, cu_q);
            let mut cu_kv_map = HashMap::new();
            cu_kv_map.insert(*dev, cu_kv);
            let decode_flash_params = FlashParams {
                max_q: 1,
                cumulative_seqlens_q: cu_q_map,
                logical_k: FlashKMeta {
                    max: u32::try_from(kv_lens.iter().copied().max().unwrap_or(0))
                        .map_err(candle_core::Error::wrap)?,
                    cumulative_seqlens: cu_kv_map,
                },
                sliding_k: None,
                causal: false,
                packed: false,
                varlen_segment_lens: None,
            };
            return Sdpa.run_attention(
                &q_4d,
                &k_4d,
                &v_4d,
                &AttentionMask::None,
                Some(&decode_flash_params),
                ctx.sdpa_params,
            );
        }

        let k_batched = unpack_gathered_kv(
            &k_gathered,
            &kv_lens,
            ctx.dims.key_value_heads,
            ctx.dims.head_size,
            query.device(),
        )?;
        let v_batched = unpack_gathered_kv(
            &v_gathered,
            &kv_lens,
            ctx.dims.key_value_heads,
            ctx.dims.head_size,
            query.device(),
        )?;
        let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
        let decode_mask = match attention_mask {
            AttentionMask::Custom(mask) => AttentionMask::Custom(adjust_kv_mask(mask, max_kv)?),
            _ if ctx.sdpa_params.sliding_window.is_some()
                || kv_lens.iter().any(|&kv_len| kv_len != max_kv) =>
            {
                prefix_gather_causal_mask(
                    &vec![1; query_rows],
                    &kv_lens,
                    None,
                    1,
                    max_kv,
                    ctx.sdpa_params.sliding_window,
                    query.dtype(),
                    query.device(),
                )?
            }
            _ => AttentionMask::None,
        };
        Sdpa.run_attention(
            &q_4d,
            &k_batched,
            &v_batched,
            &decode_mask,
            None,
            ctx.sdpa_params,
        )
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn try_run_fa3_decode(&self, call: FlashInferDecodeCall<'_, '_>) -> Result<Option<Tensor>> {
        let FlashInferDecodeCall {
            ctx,
            query,
            key_cache,
            value_cache,
            dev,
            attention_mask,
        } = call;
        let (_, kv_heads, page_size, head_dim) = key_cache.dims4()?;
        let Some(key) = (Fa3PagedScheduleShape {
            device: *dev,
            view: Fa3DecodeView::Logical,
            batch: ctx.dims.batch_size,
            query_len: ctx.dims.seq_len,
            causal: ctx.dims.seq_len > 1,
            q_heads: ctx.dims.attention_heads,
            kv_heads,
            head_dim,
            page_size,
        })
        .decode_schedule_key() else {
            return Ok(None);
        };
        let Some(key) = (Fa3DecodeCandidate {
            key,
            query_dtype: query.dtype(),
            query_contiguous: query.is_contiguous(),
            key_cache_dtype: key_cache.dtype(),
            value_cache_dtype: value_cache.dtype(),
            mask_is_none: attention_mask.is_none(),
            shapes_match: value_cache.dims4()? == key_cache.dims4()?
                && query.dims3()?
                    == (
                        ctx.dims.batch_size.saturating_mul(ctx.dims.seq_len),
                        ctx.dims.attention_heads,
                        head_dim,
                    ),
            has_alibi: ctx.alibi_slopes.is_some(),
            has_sinks: ctx.sdpa_params.sinks.is_some(),
            has_softcap: ctx.sdpa_params.softcap.is_some(),
            has_sliding_window: ctx.sdpa_params.sliding_window.is_some(),
            has_noncausal_mm_context: ctx.has_noncausal_mm_context(),
        })
        .schedule_key() else {
            return Ok(None);
        };
        let Some(buffers) = ctx
            .input_metadata
            .flashinfer
            .as_ref()
            .and_then(|metadata| metadata.fa3_decode_buffers(&key))
        else {
            return Ok(None);
        };
        let output = fa3_fp8_decode(Fa3DecodeParams {
            query,
            quantized_query: &buffers.query,
            key_cache,
            value_cache,
            page_table: &buffers.page_table,
            seqused_k: &buffers.seqused_k,
            cu_seqlens_q: &buffers.cu_seqlens_q,
            scheduler_metadata: &buffers.scheduler_metadata,
            output_accum: &buffers.output_accum,
            lse_accum: &buffers.lse_accum,
            output_lse: &buffers.output_lse,
            q_descale: &self.fp8_q_scale,
            k_descale: &self.fp8_k_scale,
            v_descale: &self.fp8_v_scale,
            schedule: buffers.schedule(key)?,
            softmax_scale: ctx.sdpa_params.softmax_scale,
        })
        .map_err(|err| {
            err.context(format!(
                "FA3 FP8 decode failed: batch={} query_len={} qo_heads={} kv_heads={} head_size={} page_size={}",
                ctx.dims.batch_size,
                ctx.dims.seq_len,
                ctx.dims.attention_heads,
                kv_heads,
                head_dim,
                page_size,
            ))
        })?;
        Ok(Some(output))
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn run_flashinfer_decode(&self, call: FlashInferDecodeCall<'_, '_>) -> Result<Tensor> {
        if let Some(output) = self.try_run_fa3_decode(call)? {
            return Ok(output);
        }
        let FlashInferDecodeCall {
            ctx,
            query,
            key_cache,
            value_cache,
            dev,
            ..
        } = call;
        let fi_meta = ctx
            .input_metadata
            .flashinfer
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("FlashInfer metadata missing"))?
            .decode_metadata(dev, ctx.sdpa_params.sliding_window)?;
        let (_, num_kv_heads, _, _) = key_cache.dims4()?;
        flashinfer_decode(
            query,
            key_cache,
            value_cache,
            self.cache_scales(key_cache).flashinfer(key_cache),
            fi_meta.paged_kv_indptr,
            fi_meta.paged_kv_indices,
            fi_meta.paged_kv_last_page_len,
            fi_meta.request_indices,
            fi_meta.kv_tile_indices,
            fi_meta.o_indptr,
            fi_meta.kv_chunk_size,
            fi_meta.block_valid_mask,
            ctx.sdpa_params.softmax_scale,
            sliding_window_left(ctx.sdpa_params.sliding_window),
            ctx.sdpa_params.softcap,
            fi_meta
                .tmp_v
                .zip(fi_meta.tmp_s)
                .map(|(tmp_v, tmp_s)| FlashInferDecodeScratch { tmp_v, tmp_s }),
        )
        .map_err(|err| {
            err.context(format!(
                "FlashInfer decode failed: batch={} padded_batch={} qo_heads={} kv_heads={} head_size={}",
                ctx.dims.batch_size,
                fi_meta.request_indices.dims1().unwrap_or(ctx.dims.batch_size),
                ctx.dims.attention_heads,
                num_kv_heads,
                ctx.dims.head_size,
            ))
        })
    }

    fn run_standard_paged_decode(
        &self,
        ctx: &PagedForwardCtx<'_>,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        dev: &DeviceLocation,
    ) -> Result<Tensor> {
        let scales = self.cache_scales(key_cache);
        paged_attention(
            query,
            scales.k,
            scales.v,
            key_cache,
            value_cache,
            ctx.block_tables(dev).unwrap(),
            ctx.context_lens(dev).unwrap(),
            ctx.alibi_slopes.as_ref(),
            if ctx.use_full {
                ctx.input_metadata.full_max_context_len.unwrap()
            } else {
                ctx.input_metadata.max_context_len.unwrap()
            },
            ctx.sdpa_params.softmax_scale,
            ctx.sdpa_params.softcap.unwrap_or(1.0f32),
            ctx.sdpa_params.sinks.as_ref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
        mut key_cache: Option<Tensor>,
        mut value_cache: Option<Tensor>,
        input_metadata: &PagedAttentionInputMetadata,
        sdpa_params: &SdpaParams,
        flash_params: Option<&FlashParams>,
        write_cache: bool,
    ) -> Result<Tensor> {
        let tensors = PagedForwardTensors {
            query,
            key,
            value,
            attention_mask,
        };
        let donor_cache_shape = if write_cache {
            None
        } else {
            Some(cache_kv_shape(
                key_cache.as_ref().expect("missing donor key cache"),
                value_cache.as_ref().expect("missing donor value cache"),
            )?)
        };
        let ctx = self.build_forward_ctx(PagedForwardSetup {
            tensors,
            donor_cache_shape,
            input_metadata,
            sdpa_params,
            flash_params,
            write_cache,
        })?;

        if let Some((flash_params, segment_lens)) = flash_params.and_then(|params| {
            params
                .varlen_segment_lens
                .as_deref()
                .map(|segment_lens| (params, segment_lens))
        }) {
            let query_lens = input_metadata.query_lens.as_deref().ok_or_else(|| {
                candle_core::Error::msg("packed varlen segments are missing logical query lengths")
            })?;
            let max_segment = validate_varlen_segment_partition(query_lens, segment_lens)?;
            let token_count = query_lens.iter().sum::<usize>();
            let location = query.device().location();
            let cu_q = flash_params
                .cumulative_seqlens_q
                .get(&location)
                .ok_or_else(|| {
                    candle_core::Error::msg(
                        "packed varlen segments are missing query offsets for the layer device",
                    )
                })?;
            let cu_k = flash_params
                .logical_k
                .cumulative_seqlens
                .get(&location)
                .ok_or_else(|| {
                    candle_core::Error::msg(
                        "packed varlen segments are missing key offsets for the layer device",
                    )
                })?;
            if !write_cache
                || input_metadata.num_cached_tokens.is_some()
                || input_metadata.has_noncausal_mm_context
                || !flash_params.packed
                || !flash_params.causal
                || flash_params.sliding_k.is_some()
                || sdpa_params.sliding_window.is_some()
                || !matches!(attention_mask, AttentionMask::CausalFlash)
                || !query.device().is_cuda()
                || !crate::using_flash_attn()
                || query.dtype() == DType::F32
                || ctx.dims.batch_size != 1
                || ctx.dims.seq_len != token_count
                || key.dim(0)? != 1
                || key.dim(2)? != token_count
                || value.dim(0)? != 1
                || value.dim(2)? != token_count
                || ctx.slot_mapping.dim(0)? != token_count
                || flash_params.max_q as usize != max_segment
                || flash_params.logical_k.max as usize != max_segment
                || cu_q.dims1()? != segment_lens.len() + 1
                || cu_k.dims1()? != segment_lens.len() + 1
            {
                candle_core::bail!("packed varlen segment metadata is not safe for direct prefill");
            }
        }

        if let Some(out) = self.try_prefix_gather_prefill(
            &ctx,
            tensors,
            &mut key_cache,
            &mut value_cache,
            write_cache,
        )? {
            return Ok(out);
        }
        if let Some(out) =
            self.try_regular_prompt(&ctx, tensors, &mut key_cache, &mut value_cache, write_cache)?
        {
            return Ok(out);
        }
        self.run_decode(&ctx, tensors, &mut key_cache, &mut value_cache, write_cache)
    }

    /// Standard paged attention forward: writes key/value to cache, then
    /// runs attention (Sdpa for prompt, paged kernel for decode).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
        key_cache: Option<Tensor>,
        value_cache: Option<Tensor>,
        input_metadata: &PagedAttentionInputMetadata,
        sdpa_params: &SdpaParams,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        self.forward_impl(
            query,
            key,
            value,
            attention_mask,
            key_cache,
            value_cache,
            input_metadata,
            sdpa_params,
            flash_params,
            true,
        )
    }

    /// Read-only paged attention against a donor layer's cache. Identical to
    /// [`forward`] but never calls `reshape_and_cache`, the donor layer has
    /// already written its K,V.  On prompt the donor's cached K,V are
    /// gathered; on decode the paged-attention kernel reads them directly.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_donor_cache(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        attention_mask: &AttentionMask,
        input_metadata: &PagedAttentionInputMetadata,
        sdpa_params: &SdpaParams,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        // key/value are unused (donor's cache already has them), but
        // forward_impl needs tensors for shape queries. Reuse query as
        // a placeholder, reshape_and_cache is skipped so they're never read.
        self.forward_impl(
            query,
            query,
            query,
            attention_mask,
            Some(key_cache.clone()),
            Some(value_cache.clone()),
            input_metadata,
            sdpa_params,
            flash_params,
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;

    #[test]
    fn cumulative_seqlens_match_checked_host_token_count() -> Result<()> {
        let lengths = [3usize, 0, 5];
        let num_tokens = checked_sequence_token_count(&lengths)?;
        let cu_seqlens = cumulative_seqlens_from_lengths(&lengths, &Device::Cpu)?;

        assert_eq!(num_tokens, 8);
        assert_eq!(cu_seqlens.to_vec1::<u32>()?, vec![0, 3, 3, 8]);
        Ok(())
    }

    #[test]
    fn checked_sequence_token_count_rejects_invalid_kernel_sizes() {
        let kernel_limit = usize::try_from(i32::MAX).unwrap();
        assert!(checked_sequence_token_count(&[kernel_limit, 1]).is_err());
        assert!(checked_sequence_token_count(&[usize::MAX, 1]).is_err());
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn supported_fa3_decode_candidate() -> Fa3DecodeCandidate {
        Fa3DecodeCandidate {
            key: Fa3PagedScheduleShape {
                device: DeviceLocation::Cuda { gpu_id: 0 },
                view: Fa3DecodeView::Logical,
                batch: 8,
                query_len: 1,
                causal: false,
                q_heads: 16,
                kv_heads: 4,
                head_dim: 256,
                page_size: 32,
            }
            .decode_schedule_key()
            .expect("supported FA3 decode schedule"),
            query_dtype: DType::BF16,
            query_contiguous: true,
            key_cache_dtype: DType::F8E4M3,
            value_cache_dtype: DType::F8E4M3,
            mask_is_none: true,
            shapes_match: true,
            has_alibi: false,
            has_sinks: false,
            has_softcap: false,
            has_sliding_window: false,
            has_noncausal_mm_context: false,
        }
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_decode_candidate_requires_supported_full_decode() {
        let supported = supported_fa3_decode_candidate();
        assert_eq!(supported.schedule_key(), Some(supported.key));

        macro_rules! reject {
            ($field:ident, $value:expr) => {{
                let mut candidate = supported;
                candidate.$field = $value;
                assert!(candidate.schedule_key().is_none());
            }};
        }

        reject!(query_dtype, DType::F16);
        reject!(query_contiguous, false);
        reject!(key_cache_dtype, DType::BF16);
        reject!(value_cache_dtype, DType::BF16);
        reject!(mask_is_none, false);
        reject!(shapes_match, false);
        reject!(has_alibi, true);
        reject!(has_sinks, true);
        reject!(has_softcap, true);
        reject!(has_sliding_window, true);
        reject!(has_noncausal_mm_context, true);

        let mut unsupported_shape = supported;
        unsupported_shape.key.head_dim = 128;
        assert!(unsupported_shape.schedule_key().is_none());

        let mut speculative = supported;
        speculative.key.query_len = 8;
        speculative.key.causal = true;
        assert_eq!(speculative.schedule_key(), Some(speculative.key));
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_decode_schedule_lookup_is_exact() {
        let candidate = supported_fa3_decode_candidate();
        let schedules = HashMap::from([(candidate.key, ())]);
        assert!(schedules.contains_key(&candidate.schedule_key().unwrap()));

        let mut different_batch = candidate;
        different_batch.key.batch /= 2;
        assert!(!schedules.contains_key(&different_batch.schedule_key().unwrap()));

        let mut different_heads = candidate;
        different_heads.key.q_heads /= 2;
        assert!(!schedules.contains_key(&different_heads.schedule_key().unwrap()));

        let mut different_query = candidate;
        different_query.key.query_len = 8;
        different_query.key.causal = true;
        assert!(!schedules.contains_key(&different_query.schedule_key().unwrap()));
    }

    #[test]
    fn owns_validated_fp8_attention_scales() -> Result<()> {
        let scales = Fp8AttentionScales {
            q: 0.25,
            k: 0.5,
            v: 0.75,
        };
        let attention =
            PagedAttention::new_with_fp8_attention_scales(128, &Device::Cpu, None, Some(scales))?;
        assert_eq!(attention.fp8_attention_scales(), scales);
        assert!(attention.has_calibrated_fp8_attention_scales());
        let [q_scale, k_scale, v_scale] = attention.fp8_scale_tensors();
        assert_eq!(q_scale.to_scalar::<f32>()?, scales.q);
        assert_eq!(k_scale.to_scalar::<f32>()?, scales.k);
        assert_eq!(v_scale.to_scalar::<f32>()?, scales.v);
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        {
            let fp8_cache = Tensor::zeros((1,), DType::F8E4M3, &Device::Cpu)?;
            let flashinfer_scales = attention.cache_scales(&fp8_cache).flashinfer(&fp8_cache);
            assert_eq!(flashinfer_scales.k, scales.k);
            assert_eq!(flashinfer_scales.v, scales.v);
        }

        let default = PagedAttention::new(128, &Device::Cpu, None)?;
        assert_eq!(default.fp8_attention_scales(), Fp8AttentionScales::UNIT);
        assert!(!default.has_calibrated_fp8_attention_scales());

        assert!(PagedAttention::new_with_fp8_attention_scales(
            128,
            &Device::Cpu,
            None,
            Some(Fp8AttentionScales {
                q: 1.0,
                k: 0.0,
                v: 1.0,
            }),
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn cache_write_accepts_row_strided_dense_heads() -> Result<()> {
        let packed = Tensor::zeros((2, 3, 16), DType::F32, &Device::Cpu)?;
        let row_strided = packed.narrow(D::Minus1, 4, 8)?.unfold(D::Minus1, 4, 4)?;
        assert_eq!(row_strided.dims(), &[2, 3, 2, 4]);
        assert_eq!(row_strided.stride(), &[48, 16, 4, 1]);
        assert_eq!(cache_input_shape(&row_strided)?, (6, 2, 4));
        assert_eq!(
            cache_input_can_write_directly(&row_strided)?,
            cfg!(all(feature = "cuda", target_family = "unix"))
        );

        let row_strided = packed
            .narrow(1, 0, 1)?
            .squeeze(1)?
            .narrow(D::Minus1, 4, 8)?
            .unfold(D::Minus1, 4, 4)?;
        assert_eq!(row_strided.stride(), &[48, 4, 1]);
        assert!(cache_input_can_write_directly(&row_strided)?);
        Ok(())
    }

    #[test]
    fn cache_write_packs_singleton_token_stride() -> Result<()> {
        let source = Tensor::zeros((8, 1, 128), DType::F32, &Device::Cpu)?;
        let singleton = source.transpose(0, 1)?;
        assert_eq!(singleton.dims(), &[1, 8, 128]);
        assert!(!cache_input_can_write_directly(&singleton)?);

        let packed = singleton.force_contiguous()?;
        assert_eq!(packed.stride(), &[1024, 128, 1]);
        assert!(cache_input_can_write_directly(&packed)?);
        Ok(())
    }

    #[test]
    fn varlen_segments_partition_each_logical_query() {
        assert_eq!(
            validate_varlen_segment_partition(&[3, 2], &[2, 1, 2]).unwrap(),
            2
        );
    }

    #[test]
    fn varlen_segments_reject_boundary_crossing_and_trailing_segments() {
        assert!(validate_varlen_segment_partition(&[3, 2], &[2, 2, 1]).is_err());
        assert!(validate_varlen_segment_partition(&[3, 2], &[2, 1, 2, 1]).is_err());
        assert!(validate_varlen_segment_partition(&[3, 2], &[2, 1]).is_err());
        assert!(validate_varlen_segment_partition(&[3, 0], &[2, 1]).is_err());
    }

    #[test]
    fn ragged_decode_mask_excludes_padding() {
        let mask =
            prefix_gather_causal_mask(&[1, 1], &[2, 4], None, 1, 4, None, DType::F32, &Device::Cpu)
                .unwrap();
        let AttentionMask::Custom(mask) = mask else {
            panic!("expected custom mask");
        };
        let mask = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(
            mask,
            vec![
                0.0,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                0.0
            ]
        );
    }

    #[test]
    fn ragged_prompt_mask_keeps_padding_rows_finite() {
        let mask =
            prefix_gather_causal_mask(&[2], &[2], None, 4, 2, None, DType::F32, &Device::Cpu)
                .unwrap();
        let AttentionMask::Custom(mask) = mask else {
            panic!("expected custom mask");
        };
        let mask = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(
            mask,
            vec![
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY
            ]
        );
    }

    #[test]
    fn sliding_mask_uses_window_as_token_capacity() {
        let mask =
            prefix_gather_causal_mask(&[1], &[6], None, 1, 6, Some(4), DType::F32, &Device::Cpu)
                .unwrap();
        let AttentionMask::Custom(mask) = mask else {
            panic!("expected custom mask");
        };

        assert_eq!(
            mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn sliding_mask_moves_with_each_chunked_query_row() {
        let mask =
            prefix_gather_causal_mask(&[3], &[8], None, 3, 8, Some(4), DType::F32, &Device::Cpu)
                .unwrap();
        let AttentionMask::Custom(mask) = mask else {
            panic!("expected custom mask");
        };
        let mask = mask.squeeze(0).unwrap().squeeze(0).unwrap();

        for (row, visible_start) in [2, 3, 4].into_iter().enumerate() {
            let values = mask.get(row).unwrap().to_vec1::<f32>().unwrap();
            for (column, value) in values.into_iter().enumerate() {
                let visible = column >= visible_start && column <= row + 5;
                assert_eq!(value == 0.0, visible);
            }
        }
    }

    #[test]
    fn sliding_mask_handles_unit_window_and_query_longer_than_window() {
        let unit =
            prefix_gather_causal_mask(&[1], &[3], None, 1, 3, Some(1), DType::F32, &Device::Cpu)
                .unwrap();
        let AttentionMask::Custom(unit) = unit else {
            panic!("expected custom mask");
        };
        assert_eq!(
            unit.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
        );

        let chunk =
            prefix_gather_causal_mask(&[5], &[7], None, 5, 7, Some(2), DType::F32, &Device::Cpu)
                .unwrap();
        let AttentionMask::Custom(chunk) = chunk else {
            panic!("expected custom mask");
        };
        let chunk = chunk.squeeze(0).unwrap().squeeze(0).unwrap();
        for row in 0..5 {
            let values = chunk.get(row).unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(values.iter().filter(|&&value| value == 0.0).count(), 2);
        }
    }

    #[test]
    fn noncausal_media_range_overrides_both_sliding_window_edges() {
        let mask = prefix_gather_causal_mask(
            &[2],
            &[8],
            Some(&[vec![(1, 8)]]),
            2,
            8,
            Some(3),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let AttentionMask::Custom(mask) = mask else {
            panic!("expected custom mask");
        };
        let mask = mask.squeeze(0).unwrap().squeeze(0).unwrap();

        assert_eq!(
            mask.get(0).unwrap().to_vec1::<f32>().unwrap(),
            vec![f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            mask.get(1).unwrap().to_vec1::<f32>().unwrap(),
            vec![f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn packed_query_layout_is_dense_without_rectangular_lengths() {
        assert!(query_layout_is_dense(&[3, 5], 1, 8));
        assert!(query_layout_is_dense(&[5, 5], 2, 5));
        assert!(!query_layout_is_dense(&[3, 5], 2, 5));
    }

    #[test]
    fn packed_prefix_selects_block_diagonal_mask_before_padded_mask() {
        assert!(should_use_packed_prefix_mask(
            true,
            true,
            &AttentionMask::CausalFlash,
            false
        ));
        assert!(!should_use_packed_prefix_mask(
            true,
            true,
            &AttentionMask::CausalFlash,
            true
        ));
        assert!(!should_use_packed_prefix_mask(
            false,
            true,
            &AttentionMask::CausalFlash,
            false
        ));
        assert!(!should_use_packed_prefix_mask(
            true,
            true,
            &AttentionMask::None,
            false
        ));
    }

    #[test]
    fn cached_suffix_reconstructs_mask_and_decode_layout() {
        assert!(should_reconstruct_prefix_mask(
            &AttentionMask::None,
            true,
            false,
            false
        ));
        assert!(should_reconstruct_prefix_mask(
            &AttentionMask::None,
            false,
            true,
            false
        ));
        assert!(should_reconstruct_prefix_mask(
            &AttentionMask::None,
            false,
            false,
            true
        ));
        assert!(!should_reconstruct_prefix_mask(
            &AttentionMask::CausalFlash,
            true,
            false,
            true
        ));

        let output =
            Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], (1, 2, 3, 1), &Device::Cpu).unwrap();
        let output = prefix_attention_output_layout(output, &AttentionMask::None).unwrap();
        assert_eq!(output.dims(), &[1, 3, 2, 1]);
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![0, 3, 1, 4, 2, 5]
        );
    }

    #[test]
    fn single_token_suffix_mask_excludes_padding_and_sliding_history() {
        let mask = prefix_gather_causal_mask(
            &[1, 1],
            &[3, 5],
            None,
            1,
            5,
            Some(3),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let AttentionMask::Custom(mask) = mask else {
            panic!("expected custom mask");
        };
        assert_eq!(
            mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                0.0,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ]
        );
    }

    #[test]
    fn gathered_decode_counts_each_query_token_as_a_row() {
        let query = Tensor::zeros((6, 4, 8), DType::F32, &Device::Cpu).unwrap();

        assert_eq!(
            decode_query_rows(&query, &[7, 8, 9, 10, 11, 12]).unwrap(),
            6
        );
        assert!(decode_query_rows(&query, &[7, 8]).is_err());
    }

    #[test]
    fn packed_sinks_query_round_trips_through_padded_varlen_layout() {
        let query =
            Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], (1, 1, 6, 1), &Device::Cpu).unwrap();
        let padded = pad_packed_query(&query, &[2, 4]).unwrap();

        assert_eq!(padded.dims(), &[2, 1, 4, 1]);
        assert_eq!(
            padded.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![0, 1, 0, 0, 2, 3, 4, 5]
        );
        let repacked = repack_padded_query(&padded, &[2, 4]).unwrap();
        assert_eq!(
            repacked.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            query.flatten_all().unwrap().to_vec1::<u32>().unwrap()
        );
    }

    #[test]
    fn packed_sinks_query_rejects_inconsistent_lengths() {
        let query = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();

        assert!(pad_packed_query(&query, &[2, 2]).is_err());
        assert!(pad_packed_query(&query, &[5, 0]).is_err());
    }

    #[test]
    fn sinks_decode_uses_varlen_without_general_flash_support() {
        assert!(should_use_decode_gather_varlen(false, true, true, 2));
        assert!(!should_use_decode_gather_varlen(false, true, true, 1));
        assert!(!should_use_decode_gather_varlen(false, true, false, 2));
        assert!(!should_use_decode_gather_varlen(true, true, false, 2));
        assert!(!should_use_decode_gather_varlen(false, false, true, 2));
        assert!(should_use_decode_gather_varlen(true, false, false, 1));
    }

    #[test]
    fn packed_varlen_requires_a_compatible_flash_backend() {
        assert!(!packed_varlen_flash_is_usable(
            false,
            true,
            DType::F16,
            128,
            false,
            false
        ));
        assert!(!packed_varlen_flash_is_usable(
            true,
            false,
            DType::F16,
            128,
            false,
            false
        ));
        assert!(!packed_varlen_flash_is_usable(
            true,
            true,
            DType::F32,
            128,
            false,
            false
        ));
        assert!(!packed_varlen_flash_is_usable(
            true,
            true,
            DType::F16,
            640,
            false,
            false
        ));
        assert!(!packed_varlen_flash_is_usable(
            true,
            true,
            DType::F16,
            512,
            true,
            false
        ));
        assert!(!packed_varlen_flash_is_usable(
            true,
            true,
            DType::F16,
            320,
            false,
            true
        ));
        assert_eq!(
            packed_varlen_flash_is_usable(true, true, DType::F16, 128, false, false),
            cfg!(any(feature = "flash-attn", feature = "flash-attn-v3"))
        );
    }

    #[test]
    fn uncached_noncausal_prompt_uses_gather_path() {
        assert!(should_use_gather_path(true, false, true, true, false, true));
        assert!(!should_use_gather_path(
            true, false, false, false, false, true
        ));
    }

    #[test]
    fn uncached_later_prompt_mask_uses_gather_path() {
        assert!(should_use_gather_path(
            true, false, false, true, false, true
        ));
        assert!(!should_use_gather_path(
            false, false, false, true, false, true
        ));
    }

    #[test]
    fn exact_mm_ranges_keep_the_direct_prefix_path_causal() {
        assert!(prefix_prefill_is_causal(true, false, true));
        assert!(prefix_prefill_is_causal(true, true, false));
        assert!(!prefix_prefill_is_causal(true, false, false));
        assert!(!prefix_prefill_is_causal(false, true, true));
    }

    #[test]
    fn full_mm_range_view_does_not_reuse_sliding_ranges() {
        let regular = 7;
        let selected = select_optional_view(true, None, Some(&regular));

        assert_eq!(selected, None);
        assert_eq!(
            select_optional_view(false, Some(&11), Some(&regular)),
            Some(&regular)
        );
    }

    #[test]
    fn noncausal_mm_context_requires_the_selected_cache_view() {
        assert!(noncausal_mm_view_is_valid(false, false, false));
        assert!(noncausal_mm_view_is_valid(true, true, true));
        assert!(!noncausal_mm_view_is_valid(true, false, true));
        assert!(!noncausal_mm_view_is_valid(true, true, false));
    }

    #[test]
    fn packed_custom_mask_preserves_boundaries_and_mm_prefixes() {
        let mask = packed_prefix_gather_causal_mask(
            &[2, 3],
            &[2, 3],
            Some(&[vec![(0, 2)], vec![]]),
            Some(2),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

        assert_eq!(
            mask[0],
            vec![
                0.0,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY
            ]
        );
        assert_eq!(
            mask[2],
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY
            ]
        );
        assert_eq!(
            mask[4],
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0
            ]
        );
    }

    #[test]
    fn overlapping_mm_ranges_do_not_merge_attention_groups() {
        let mask = packed_prefix_gather_causal_mask(
            &[6],
            &[6],
            Some(&[vec![(0, 4), (2, 6)]]),
            None,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

        assert_eq!(
            mask[0],
            vec![0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
        assert_eq!(mask[3], vec![0.0; 6]);
    }
}
