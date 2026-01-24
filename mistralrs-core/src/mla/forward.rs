//! MLA forward pass functions for decode and cache operations.

use candle_core::{Device, Result, Tensor};

use crate::{
    attention::SdpaParams,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};

use super::MlaWeights;

#[cfg(all(feature = "cuda", target_family = "unix"))]
use candle_core::{DType, D};

#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::layers::Sdpa;

#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::ops::SplitOp;

/// Environment variable to disable MLA optimization.
#[cfg(all(feature = "cuda", target_family = "unix"))]
const MISTRALRS_NO_MLA: &str = "MISTRALRS_NO_MLA";

/// Check if MLA is disabled via environment variable.
#[cfg(all(feature = "cuda", target_family = "unix"))]
fn is_mla_disabled() -> bool {
    std::env::var(MISTRALRS_NO_MLA).is_ok_and(|x| x == "1")
}

/// Check if MLA decode should be used for single-token generation.
///
/// MLA decode is used when:
/// - `MISTRALRS_NO_MLA` is not set to "1"
/// - No attention mask (single-token decode)
/// - Sequence length is 1
/// - Paged attention is enabled
/// - Running on CUDA
/// - Paged KV indptr metadata is available
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub fn should_use_mla_decode(
    attention_mask: Option<&Tensor>,
    seq_len: usize,
    paged_attn_enabled: bool,
    device: &Device,
    metadata: &Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
) -> bool {
    !is_mla_disabled()
        && attention_mask.is_none()
        && seq_len == 1
        && paged_attn_enabled
        && matches!(device, Device::Cuda(_))
        && metadata
            .as_ref()
            .and_then(|(_, meta)| meta.paged_kv_indptr.as_ref())
            .is_some()
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
pub fn should_use_mla_decode(
    _attention_mask: Option<&Tensor>,
    _seq_len: usize,
    _paged_attn_enabled: bool,
    _device: &Device,
    _metadata: &Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
) -> bool {
    false
}

/// Check if MLA cache forward should be used for prefill with prefix caching.
///
/// MLA cache is used when:
/// - `MISTRALRS_NO_MLA` is not set to "1"
/// - Paged attention is enabled
/// - Running on CUDA
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub fn should_use_mla_cache(paged_attn_enabled: bool, device: &Device) -> bool {
    !is_mla_disabled() && paged_attn_enabled && matches!(device, Device::Cuda(_))
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
pub fn should_use_mla_cache(_paged_attn_enabled: bool, _device: &Device) -> bool {
    false
}

/// MLA decode forward pass for single-token generation.
///
/// Uses FlashInfer MLA kernels for efficient decode with latent attention.
///
/// # Arguments
/// * `q_nope` - Non-positional query tensor [bs, num_heads, seq_len, qk_nope_head_dim]
/// * `q_pe` - Positional query tensor [bs, num_heads, seq_len, qk_rope_head_dim]
/// * `ckv` - Compressed KV tensor [bs, seq_len, kv_lora_rank]
/// * `k_pe` - Positional key tensor [bs, 1, seq_len, qk_rope_head_dim]
/// * `metadata` - PagedAttention cache and metadata
/// * `mla_weights` - Cached MLA weight matrices
/// * `kv_b_proj` - KV B projection layer for weight computation
/// * `sdpa_params` - SDPA parameters including softmax scale
/// * `num_attention_heads` - Number of attention heads
/// * `kv_lora_rank` - KV latent dimension
/// * `qk_rope_head_dim` - Rope head dimension
/// * `qk_nope_head_dim` - Non-positional head dimension
/// * `v_head_dim` - Value head dimension
/// * `bs` - Batch size
/// * `seq_len` - Sequence length (should be 1)
#[cfg(all(feature = "cuda", target_family = "unix"))]
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_forward(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv: &Tensor,
    k_pe: &Tensor,
    metadata: &Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    mla_weights: &MlaWeights,
    kv_b_proj: &dyn mistralrs_quant::QuantMethod,
    sdpa_params: &SdpaParams,
    num_attention_heads: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    bs: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let ((key_cache, value_cache), input_metadata) = metadata
        .as_ref()
        .ok_or_else(|| candle_core::Error::msg("paged attention metadata missing"))?;
    let device_location = q_nope.device().location();
    let slot_mapping = input_metadata
        .slot_mappings
        .get(&device_location)
        .ok_or_else(|| candle_core::Error::msg("slot mapping missing"))?;
    let slot_mapping = if slot_mapping.dims().len() > 1 {
        slot_mapping.flatten(0, slot_mapping.dims().len())?
    } else {
        slot_mapping.clone()
    };
    let paged_kv_indptr = input_metadata
        .paged_kv_indptr
        .as_ref()
        .and_then(|m| m.get(&device_location))
        .ok_or_else(|| candle_core::Error::msg("paged_kv_indptr missing"))?;
    let paged_kv_indices = input_metadata
        .paged_kv_indices
        .as_ref()
        .and_then(|m| m.get(&device_location))
        .ok_or_else(|| candle_core::Error::msg("paged_kv_indices missing"))?;
    let paged_kv_last_page_len = input_metadata
        .paged_kv_last_page_len
        .as_ref()
        .and_then(|m| m.get(&device_location))
        .ok_or_else(|| candle_core::Error::msg("paged_kv_last_page_len missing"))?;
    let paged_kv_request_indices = input_metadata
        .paged_kv_request_indices
        .as_ref()
        .and_then(|m| m.get(&device_location))
        .ok_or_else(|| candle_core::Error::msg("paged_kv_request_indices missing"))?;
    let paged_kv_tile_indices = input_metadata
        .paged_kv_tile_indices
        .as_ref()
        .and_then(|m| m.get(&device_location))
        .ok_or_else(|| candle_core::Error::msg("paged_kv_tile_indices missing"))?;
    let paged_kv_o_indptr = input_metadata
        .paged_kv_o_indptr
        .as_ref()
        .and_then(|m| m.get(&device_location))
        .ok_or_else(|| candle_core::Error::msg("paged_kv_o_indptr missing"))?;
    let paged_kv_chunk_size = input_metadata
        .paged_kv_chunk_size
        .as_ref()
        .and_then(|m| m.get(&device_location))
        .ok_or_else(|| candle_core::Error::msg("paged_kv_chunk_size missing"))?;

    let ckv_flat = ckv.contiguous()?.reshape((bs * seq_len, kv_lora_rank))?;
    let k_pe_flat = k_pe
        .squeeze(1)?
        .contiguous()?
        .reshape((bs * seq_len, qk_rope_head_dim))?;

    mistralrs_paged_attn::concat_and_cache_mla(
        &ckv_flat,
        &k_pe_flat,
        key_cache,
        value_cache,
        &slot_mapping,
    )?;

    let q_nope = q_nope.squeeze(2)?.contiguous()?;
    let q_pe = q_pe.squeeze(2)?.contiguous()?;
    let (w_uk, w_uv_t) = mla_weights.get_or_compute(
        kv_b_proj,
        q_nope.device(),
        num_attention_heads,
        kv_lora_rank,
        qk_nope_head_dim,
        v_head_dim,
    )?;
    let ql_nope = q_nope
        .unsqueeze(D::Minus2)?
        .broadcast_matmul(&w_uk.unsqueeze(0)?)?
        .squeeze(D::Minus2)?
        .contiguous()?;

    let attn_latent = mistralrs_paged_attn::flashinfer_mla_decode(
        &ql_nope,
        &q_pe,
        key_cache,
        value_cache,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        paged_kv_request_indices,
        paged_kv_tile_indices,
        paged_kv_o_indptr,
        paged_kv_chunk_size,
        sdpa_params.softmax_scale,
    )?;

    attn_latent
        .unsqueeze(D::Minus2)?
        .broadcast_matmul(&w_uv_t.unsqueeze(0)?)?
        .squeeze(D::Minus2)
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_forward(
    _q_nope: &Tensor,
    _q_pe: &Tensor,
    _ckv: &Tensor,
    _k_pe: &Tensor,
    _metadata: &Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    _mla_weights: &MlaWeights,
    _kv_b_proj: &dyn mistralrs_quant::QuantMethod,
    _sdpa_params: &SdpaParams,
    _num_attention_heads: usize,
    _kv_lora_rank: usize,
    _qk_rope_head_dim: usize,
    _qk_nope_head_dim: usize,
    _v_head_dim: usize,
    _bs: usize,
    _seq_len: usize,
) -> Result<Tensor> {
    candle_core::bail!("MLA decode requires CUDA support")
}

/// MLA cache forward pass for prefill with prefix caching support.
///
/// Handles both fresh prefill and prefix-cached scenarios.
///
/// # Arguments
/// * `q` - Query tensor [bs, num_heads, seq_len, head_dim]
/// * `k` - Key tensor [bs, num_heads, seq_len, head_dim]
/// * `v` - Value tensor [bs, num_heads, seq_len, v_head_dim]
/// * `ckv` - Compressed KV tensor [bs, seq_len, kv_lora_rank]
/// * `k_pe` - Positional key tensor [bs, 1, seq_len, qk_rope_head_dim]
/// * `attention_mask` - Optional attention mask
/// * `seqlen_offsets` - Prefix lengths for each sequence
/// * `metadata` - PagedAttention cache and metadata
/// * `flash_params` - Flash attention parameters
/// * `kv_b_proj` - KV B projection layer
/// * `sdpa_params` - SDPA parameters
/// * `num_attention_heads` - Number of attention heads
/// * `kv_lora_rank` - KV latent dimension
/// * `qk_rope_head_dim` - Rope head dimension
/// * `qk_nope_head_dim` - Non-positional head dimension
/// * `v_head_dim` - Value head dimension
/// * `bs` - Batch size
/// * `seq_len` - Sequence length
#[cfg(all(feature = "cuda", target_family = "unix"))]
#[allow(clippy::too_many_arguments)]
pub fn mla_cache_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ckv: &Tensor,
    k_pe: &Tensor,
    attention_mask: Option<&Tensor>,
    seqlen_offsets: &[usize],
    metadata: &Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    flash_params: &FlashParams,
    kv_b_proj: &dyn mistralrs_quant::QuantMethod,
    sdpa_params: &SdpaParams,
    num_attention_heads: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    bs: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let mut key_cache = None;
    let mut value_cache = None;
    let mut input_metadata = None;
    if let Some(((key_cache_tensor, value_cache_tensor), meta)) = metadata.as_ref() {
        key_cache = Some(key_cache_tensor);
        value_cache = Some(value_cache_tensor);
        input_metadata = Some(*meta);

        let device_location = q.device().location();
        let slot_mapping = meta
            .slot_mappings
            .get(&device_location)
            .ok_or_else(|| candle_core::Error::msg("slot mapping missing"))?;
        let slot_mapping = if slot_mapping.dims().len() > 1 {
            slot_mapping.flatten(0, slot_mapping.dims().len())?
        } else {
            slot_mapping.clone()
        };
        let ckv_flat = ckv.contiguous()?.reshape((bs * seq_len, kv_lora_rank))?;
        let k_pe_flat = k_pe
            .squeeze(1)?
            .contiguous()?
            .reshape((bs * seq_len, qk_rope_head_dim))?;
        mistralrs_paged_attn::concat_and_cache_mla(
            &ckv_flat,
            &k_pe_flat,
            key_cache_tensor,
            value_cache_tensor,
            &slot_mapping,
        )?;
    }

    let prefix_lens = seqlen_offsets;
    let needs_prefix = prefix_lens.iter().any(|&len| len > 0);
    if !needs_prefix && attention_mask.is_some() {
        Sdpa.run_attention(q, k, v, attention_mask, Some(flash_params), sdpa_params)
    } else {
        let ((key_cache, value_cache), input_metadata) =
            match (key_cache, value_cache, input_metadata) {
                (Some(k), Some(v), Some(m)) => ((k, v), m),
                _ => {
                    return Sdpa.run_attention(
                        q,
                        k,
                        v,
                        attention_mask,
                        Some(flash_params),
                        sdpa_params,
                    );
                }
            };

        let device_location = q.device().location();
        let slot_mapping = input_metadata
            .slot_mappings
            .get(&device_location)
            .ok_or_else(|| candle_core::Error::msg("slot mapping missing"))?;
        let slot_mapping_cpu = slot_mapping.to_device(&Device::Cpu)?;
        let slot_mapping_cpu = if slot_mapping_cpu.dims().len() == 2 {
            slot_mapping_cpu
        } else {
            slot_mapping_cpu.reshape((bs, seq_len))?
        };
        let slot_mapping_vec = slot_mapping_cpu.to_vec2::<i64>()?;
        let cur_lens: Vec<usize> = slot_mapping_vec
            .iter()
            .map(|row| {
                row.iter()
                    .filter(|&&v| v != crate::paged_attention::_PAD_SLOT_ID)
                    .count()
            })
            .collect();

        let block_tables = input_metadata
            .block_tables
            .as_ref()
            .and_then(|m| m.get(&device_location))
            .ok_or_else(|| candle_core::Error::msg("block tables missing"))?;
        let block_tables_cpu = block_tables.to_device(&Device::Cpu)?;
        let (block_rows, block_stride) = block_tables_cpu.dims2()?;
        let block_tables_vec = block_tables_cpu.to_vec2::<u32>()?;
        let expected_rows: usize = cur_lens.iter().sum();
        let mut block_tables_seq: Vec<i32> = Vec::with_capacity(bs * block_stride);
        if block_rows == bs {
            for row in block_tables_vec.iter().take(bs) {
                block_tables_seq.extend(row.iter().map(|&v| v as i32));
            }
        } else if block_rows == expected_rows {
            let mut offset = 0usize;
            for len in cur_lens.iter() {
                if offset < block_tables_vec.len() {
                    block_tables_seq.extend(block_tables_vec[offset].iter().map(|&v| v as i32));
                } else {
                    block_tables_seq.extend(std::iter::repeat_n(0, block_stride));
                }
                offset = offset.saturating_add(*len);
            }
        } else {
            candle_core::bail!(
                "unexpected block_tables rows: got {block_rows}, expected {bs} or {expected_rows}"
            );
        }

        let block_tables_seq =
            Tensor::from_vec(block_tables_seq, (bs, block_stride), &Device::Cpu)?
                .to_device(q.device())?;

        let total_prefix_tokens: usize = prefix_lens.iter().sum();
        let mut k_prefix = None;
        let mut v_prefix = None;
        let mut prefix_offsets = Vec::with_capacity(bs + 1);
        prefix_offsets.push(0usize);
        for len in prefix_lens {
            let next = *prefix_offsets.last().unwrap() + *len;
            prefix_offsets.push(next);
        }

        if total_prefix_tokens > 0 {
            let mut cu_seq_lens = Vec::with_capacity(bs + 1);
            cu_seq_lens.push(0i32);
            let mut token_to_seq = Vec::with_capacity(total_prefix_tokens);
            #[allow(clippy::cast_possible_truncation)]
            for (seq_idx, len) in prefix_lens.iter().enumerate() {
                let next = *cu_seq_lens.last().unwrap() + *len as i32;
                cu_seq_lens.push(next);
                token_to_seq.extend(std::iter::repeat_n(seq_idx as i32, *len));
            }

            let cu_seq_lens =
                Tensor::from_vec(cu_seq_lens, (bs + 1,), &Device::Cpu)?.to_device(q.device())?;
            let token_to_seq =
                Tensor::from_vec(token_to_seq, (total_prefix_tokens,), &Device::Cpu)?
                    .to_device(q.device())?;

            let (ckv_prefix, kpe_prefix) = mistralrs_paged_attn::gather_mla_cache(
                key_cache,
                value_cache,
                &block_tables_seq,
                &cu_seq_lens,
                &token_to_seq,
            )?;

            let mut kv_prefix = kv_b_proj.forward_autocast(&ckv_prefix)?;
            kv_prefix = kv_prefix.reshape((
                total_prefix_tokens,
                num_attention_heads,
                qk_nope_head_dim + v_head_dim,
            ))?;
            let kv_prefix_split = kv_prefix.split(&[qk_nope_head_dim, v_head_dim], D::Minus1)?;
            let k_nope_prefix = kv_prefix_split[0].clone();
            let v_prefix_full = kv_prefix_split[1].clone();
            let kpe_prefix = kpe_prefix.unsqueeze(1)?.expand((
                total_prefix_tokens,
                num_attention_heads,
                qk_rope_head_dim,
            ))?;
            let k_prefix_full = Tensor::cat(&[&k_nope_prefix, &kpe_prefix], D::Minus1)?;
            k_prefix = Some(k_prefix_full);
            v_prefix = Some(v_prefix_full);
        }

        let mut outputs = Vec::with_capacity(bs);
        for (seq_idx, cur_len) in cur_lens.iter().enumerate() {
            let cur_len = (*cur_len).min(seq_len);
            if cur_len == 0 {
                outputs.push(Tensor::zeros(
                    (seq_len, num_attention_heads, v_head_dim),
                    q.dtype(),
                    q.device(),
                )?);
                continue;
            }
            let mut q_i = q.narrow(0, seq_idx, 1)?;
            let mut k_i = k.narrow(0, seq_idx, 1)?;
            let mut v_i = v.narrow(0, seq_idx, 1)?;
            if cur_len < seq_len {
                q_i = q_i.narrow(D::Minus2, 0, cur_len)?;
                k_i = k_i.narrow(D::Minus2, 0, cur_len)?;
                v_i = v_i.narrow(D::Minus2, 0, cur_len)?;
            }

            let prefix_len = prefix_lens[seq_idx];
            let k_full = if let Some(k_prefix_full) = &k_prefix {
                let start = prefix_offsets[seq_idx];
                let k_prefix_i = if prefix_len > 0 {
                    Some(k_prefix_full.narrow(0, start, prefix_len)?)
                } else {
                    None
                };
                let k_i = k_i.squeeze(0)?.contiguous()?;
                if let Some(k_prefix_i) = k_prefix_i {
                    let k_prefix_i = k_prefix_i.transpose(0, 1)?.contiguous()?;
                    Tensor::cat(&[&k_prefix_i, &k_i], 1)?
                } else {
                    k_i
                }
            } else {
                k_i.squeeze(0)?.contiguous()?
            };

            let v_full = if let Some(v_prefix_full) = &v_prefix {
                let start = prefix_offsets[seq_idx];
                let v_prefix_i = if prefix_len > 0 {
                    Some(v_prefix_full.narrow(0, start, prefix_len)?)
                } else {
                    None
                };
                let v_i = v_i.squeeze(0)?.contiguous()?;
                if let Some(v_prefix_i) = v_prefix_i {
                    let v_prefix_i = v_prefix_i.transpose(0, 1)?.contiguous()?;
                    Tensor::cat(&[&v_prefix_i, &v_i], 1)?
                } else {
                    v_i
                }
            } else {
                v_i.squeeze(0)?.contiguous()?
            };

            let mask = if cur_len > 1 {
                let offset = cur_len + prefix_len;
                let mask: Vec<_> = (0..cur_len)
                    .flat_map(|i| (0..offset).map(move |j| u8::from(j + cur_len > i + offset)))
                    .collect();
                let mask = Tensor::from_slice(&mask, (cur_len, offset), q.device())?
                    .to_dtype(DType::U8)?;
                let zero = Tensor::new(0.0f32, q.device())?;
                let mask = crate::layers_masker::masked_fill(
                    &zero.to_dtype(q.dtype())?.broadcast_as(mask.shape())?,
                    &mask,
                    f32::NEG_INFINITY,
                )?;
                Some(mask)
            } else {
                None
            };

            let attn_out_i = Sdpa.run_attention_noflash(
                &q_i,
                &k_full.unsqueeze(0)?,
                &v_full.unsqueeze(0)?,
                mask.as_ref(),
                sdpa_params,
            )?;
            let mut attn_out_i = attn_out_i.squeeze(0)?.transpose(0, 1)?;
            if cur_len < seq_len {
                attn_out_i = attn_out_i.pad_with_zeros(D::Minus(3), 0, seq_len - cur_len)?;
            }
            outputs.push(attn_out_i);
        }
        Tensor::cat(&outputs, 0)
    }
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
#[allow(clippy::too_many_arguments)]
pub fn mla_cache_forward(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _ckv: &Tensor,
    _k_pe: &Tensor,
    _attention_mask: Option<&Tensor>,
    _seqlen_offsets: &[usize],
    _metadata: &Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    _flash_params: &FlashParams,
    _kv_b_proj: &dyn mistralrs_quant::QuantMethod,
    _sdpa_params: &SdpaParams,
    _num_attention_heads: usize,
    _kv_lora_rank: usize,
    _qk_rope_head_dim: usize,
    _qk_nope_head_dim: usize,
    _v_head_dim: usize,
    _bs: usize,
    _seq_len: usize,
) -> Result<Tensor> {
    candle_core::bail!("MLA cache requires CUDA support")
}
