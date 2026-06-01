use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use mistralrs_paged_attn::{
    flashinfer_decode, flashinfer_prefill, gather_kv_cache_flashinfer, reshape_and_cache_flashinfer,
};
use mistralrs_paged_attn::{kv_scale_update, paged_attention, reshape_and_cache};

const KV_SCALE_UPDATE_ITERATION: i32 = 128;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::Sdpa,
    paged_attention::{
        AttentionBackendKind, _PAD_SLOT_ID, FLASHINFER_DECODE_MAX_HEAD_SIZE,
        STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE,
    },
    pipeline::text_models_inputs_processor::{
        FlashKMeta, FlashParams, PagedAttentionInputMetadata,
    },
};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::{
    paged_attention::{
        FLASHINFER_PREFILL_MAX_HEAD_SIZE, FLASHINFER_TENSOR_CORE_DECODE_ENABLED,
        FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE,
    },
    pipeline::text_models_inputs_processor::FLASHINFER_PREFILL_MAX_GROUP_SIZE,
};

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

fn cumulative_seqlens_from_lengths(lengths: &[usize], device: &Device) -> Result<Tensor> {
    let mut cumulative = Vec::with_capacity(lengths.len() + 1);
    cumulative.push(0u32);
    for &len in lengths {
        #[allow(clippy::cast_possible_truncation)]
        cumulative.push(cumulative.last().copied().unwrap_or(0) + len as u32);
    }
    Tensor::new(&cumulative[..], &Device::Cpu)?.to_device(device)
}

fn block_aligned_window_start(full_len: usize, window: usize, block_size: usize) -> usize {
    let window_start = full_len.saturating_sub(window);
    (window_start / block_size) * block_size
}

fn block_aligned_window_len_for_query(
    full_len: usize,
    query_len: usize,
    window: usize,
    block_size: usize,
) -> usize {
    let block_start = block_aligned_window_start(full_len, window, block_size);
    let query_start = full_len.saturating_sub(query_len);
    if block_start <= query_start {
        full_len - block_start
    } else {
        full_len
    }
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

fn cache_input_is_packed(tensor: &Tensor) -> Result<bool> {
    let (_, heads, head_size) = tensor.dims3()?;
    let stride = tensor.stride();
    Ok(stride[2] == 1 && stride[1] == head_size && stride[0] == heads * head_size)
}

fn write_kv_cache(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let key_packed;
    let key = if cache_input_is_packed(key)? {
        key
    } else {
        key_packed = key.contiguous()?;
        &key_packed
    };
    let value_packed;
    let value = if cache_input_is_packed(value)? {
        value
    } else {
        value_packed = value.contiguous()?;
        &value_packed
    };
    match AttentionBackendKind::from_cache(key_cache, value_cache) {
        AttentionBackendKind::FlashInfer => {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            {
                reshape_and_cache_flashinfer(key, value, key_cache, value_cache, slot_mapping)
            }
            #[cfg(not(all(feature = "cuda", target_family = "unix")))]
            {
                unreachable!("FlashInfer cache is only available with CUDA")
            }
        }
        AttentionBackendKind::Standard => reshape_and_cache(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
        ),
    }
}

fn gather_kv_cache_for_layout(
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    block_tables: &Tensor,
    cu_kv: &Tensor,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    match AttentionBackendKind::from_cache(key_cache, value_cache) {
        AttentionBackendKind::FlashInfer => {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            {
                gather_kv_cache_flashinfer(key_cache, value_cache, block_tables, cu_kv, dtype)
            }
            #[cfg(not(all(feature = "cuda", target_family = "unix")))]
            {
                unreachable!("FlashInfer cache is only available with CUDA")
            }
        }
        AttentionBackendKind::Standard => mistralrs_paged_attn::gather_kv_cache(
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            block_tables,
            cu_kv,
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

fn supports_packed_varlen_sdpa(query: &Tensor) -> bool {
    query.device().is_cpu()
        || (query.device().is_cuda() && crate::using_flash_attn() && query.dtype() != DType::F32)
}

pub struct PagedAttention {
    alibi_slopes: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    kv_updated_times: AtomicI32,
}

impl PagedAttention {
    pub fn new(head_dim: usize, device: &Device, alibi_slopes: Option<Vec<f32>>) -> Result<Self> {
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            assert_eq!(alibi_slopes.len(), head_dim);
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        Ok(Self {
            alibi_slopes,
            k_scale: Some(Tensor::new(1f32, device)?),
            v_scale: Some(Tensor::new(1f32, device)?),
            kv_updated_times: AtomicI32::new(0),
        })
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        unused_variables
    )]
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
        if write_cache {
            if let (Some(k_scale), Some(v_scale), Some(key_cache)) =
                (&self.k_scale, &self.v_scale, &key_cache)
            {
                if self.kv_updated_times.load(Ordering::Relaxed) < KV_SCALE_UPDATE_ITERATION
                    && key_cache.dtype() == DType::F8E4M3
                {
                    kv_scale_update(key, value, k_scale, v_scale)?;
                    self.kv_updated_times.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        let slot_mapping_full = input_metadata
            .slot_mappings
            .get(&query.device().location())
            .unwrap();
        let dims = slot_mapping_full.dims();
        let slot_mapping = if dims.len() > 1 {
            &slot_mapping_full.flatten(0, dims.len() - 1)?
        } else {
            slot_mapping_full
        };

        let (batch_size, attention_heads, seq_len, head_size) = query.shape().dims4()?;
        let (key_value_heads, kv_head_size) = if !write_cache {
            cache_kv_shape(
                key_cache.as_ref().expect("missing donor key cache"),
                value_cache.as_ref().expect("missing donor value cache"),
            )?
        } else {
            let (_, key_value_heads, _, kv_head_size) = key.shape().dims4()?;
            (key_value_heads, kv_head_size)
        };
        if kv_head_size != head_size {
            candle_core::bail!(
                "paged attention query/cache head dim mismatch: query={head_size}, kv={kv_head_size}"
            );
        }

        // For models with per-layer sliding windows (GPT-OSS, Gemma2):
        // - Full-attention layers (sliding_window == None) use the full block tables.
        // - Sliding-window layers (sliding_window == Some) use the windowed block tables.
        // If full_block_tables is not populated, fall back to the regular block_tables.
        let use_full = sdpa_params.sliding_window.is_none()
            && (input_metadata.full_block_tables.is_some()
                || input_metadata.full_paged_kv_indptr.is_some());

        let resolve_block_tables = |dev: &candle_core::DeviceLocation| -> Option<&Tensor> {
            if use_full {
                input_metadata.full_block_tables.as_ref()?.get(dev)
            } else {
                input_metadata.block_tables.as_ref()?.get(dev)
            }
        };
        let resolve_context_lens = |dev: &candle_core::DeviceLocation| -> Option<&Tensor> {
            if use_full {
                input_metadata.full_context_lens.as_ref()?.get(dev)
            } else {
                input_metadata.context_lens.as_ref()?.get(dev)
            }
        };

        let alibi_slopes = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            Some(alibi_slopes.to_device(query.device())?)
        } else {
            None
        };

        // === Prefix cache / donor-gather prompt path ===
        // Entered when:
        //  - write_cache=true  AND num_cached_tokens is set (prefix cache hit)
        //  - write_cache=false AND attention_mask is set  (donor cache prompt)
        // The gather path needs block_tables. During calibration forwards
        // there is no paged cache, so block_tables is None — skip to the
        // regular prompt path.
        let has_block_tables = input_metadata.block_tables.is_some();
        let has_cached_prefix = input_metadata.num_cached_tokens.is_some();
        let mask_is_prefill = !matches!(attention_mask, AttentionMask::None);
        let single_token_first_prompt = input_metadata.is_first_prompt_chunk && seq_len == 1;
        let use_gather_path = if write_cache {
            has_cached_prefix && has_block_tables
        } else {
            (mask_is_prefill || single_token_first_prompt) && has_block_tables
        };

        if use_gather_path {
            let block_tables = resolve_block_tables(&query.device().location()).unwrap();
            let context_lens = resolve_context_lens(&query.device().location()).unwrap();
            // Write new tokens to cache (skipped for donor/shared layers)
            if write_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
                let k_flat = key
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                let v_flat = value
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                let key_cache = key_cache.as_mut().unwrap();
                let value_cache = value_cache.as_mut().unwrap();
                write_kv_cache(
                    &k_flat,
                    &v_flat,
                    self.k_scale.as_ref(),
                    self.v_scale.as_ref(),
                    key_cache,
                    value_cache,
                    slot_mapping,
                )?;
            }

            assert!(
                alibi_slopes.is_none(),
                "alibi slopes not supported in prefix cache path"
            );

            let device = query.device();

            let new_token_lens =
                new_token_lens_from_slot_mapping(slot_mapping_full, batch_size, seq_len)?;
            let query_lens = input_metadata
                .query_lens
                .clone()
                .unwrap_or_else(|| new_token_lens.clone());
            let full_kv_lens =
                if let Some(num_cached_tokens) = input_metadata.num_cached_tokens.as_ref() {
                    num_cached_tokens
                        .iter()
                        .zip(query_lens.iter())
                        .map(|(&cached, &query_len)| cached + query_len)
                        .collect::<Vec<_>>()
                } else {
                    new_token_lens.clone()
                };
            let kv_lens = if let Some(window) = sdpa_params.sliding_window {
                if !use_full {
                    let block_size = cache_block_size(
                        key_cache.as_ref().unwrap(),
                        value_cache.as_ref().unwrap(),
                    )?;
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

            // Resolve cu_seqlens_kv: scheduler-provided for prefix cache hits,
            // or synthesize it from the actual slot-mapping lengths on first prompt.
            let cu_kv = if sdpa_params.sliding_window.is_none() {
                if let Some(map) = input_metadata.cu_seqlens_kv.as_ref() {
                    resolve_tensor_for_device(map, device, "cu_seqlens_kv")?
                } else {
                    cumulative_seqlens_from_lengths(&kv_lens, device)?
                }
            } else {
                cumulative_seqlens_from_lengths(&kv_lens, device)?
            };

            #[cfg(all(feature = "cuda", target_family = "unix"))]
            if query.device().is_cuda()
                && query.dtype() != DType::F32
                && !has_cached_prefix
                && sdpa_params.sinks.is_none()
                && head_size <= FLASHINFER_PREFILL_MAX_HEAD_SIZE
                && attention_heads / key_value_heads <= FLASHINFER_PREFILL_MAX_GROUP_SIZE
                && query_lens.iter().all(|&len| len == seq_len)
                && AttentionBackendKind::from_cache(
                    key_cache.as_ref().unwrap(),
                    value_cache.as_ref().unwrap(),
                ) == AttentionBackendKind::FlashInfer
            {
                if let Ok(fi_meta) =
                    input_metadata.flashinfer_prefill_metadata(&device.location(), use_full)
                {
                    let q_flat = if seq_len > 1 {
                        query
                            .transpose(1, 2)?
                            .reshape(((), attention_heads, head_size))?
                    } else {
                        query.reshape(((), attention_heads, head_size))?
                    };
                    if let Ok(out) = flashinfer_prefill(
                        &q_flat,
                        key_cache.as_ref().unwrap(),
                        value_cache.as_ref().unwrap(),
                        fi_meta.paged_kv_indptr,
                        fi_meta.paged_kv_indices,
                        fi_meta.paged_kv_last_page_len,
                        fi_meta.q_indptr,
                        fi_meta.request_indices,
                        fi_meta.qo_tile_indices,
                        fi_meta.kv_tile_indices,
                        fi_meta.o_indptr,
                        fi_meta.kv_chunk_size,
                        fi_meta.block_valid_mask,
                        batch_size,
                        sdpa_params.softmax_scale,
                        sdpa_params.sliding_window,
                        sdpa_params.softcap,
                    ) {
                        return out
                            .reshape((batch_size, seq_len, attention_heads, head_size))?
                            .transpose(1, 2);
                    }
                }
            }

            let (k_gathered, v_gathered) = gather_kv_cache_for_layout(
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                block_tables,
                &cu_kv,
                query.dtype(),
            )?;

            if supports_packed_varlen_sdpa(query) {
                let cu_q = if let Some(fp) = flash_params {
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

                // gathered: (total_kv, kv_heads, dim) -> (1, kv_heads, total_kv, dim)
                let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
                let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;

                let mut cu_q_map = HashMap::new();
                cu_q_map.insert(device.location(), cu_q);
                let mut cu_kv_map = HashMap::new();
                cu_kv_map.insert(device.location(), cu_kv);
                let prefix_flash_params = FlashParams {
                    max_q: query_lens.iter().copied().max().unwrap_or(0) as u32,
                    cumulative_seqlens_q: cu_q_map,
                    logical_k: FlashKMeta {
                        max: kv_lens.iter().copied().max().unwrap_or(0) as u32,
                        cumulative_seqlens: cu_kv_map,
                    },
                    sliding_k: None,
                    causal: query_lens.iter().any(|&len| len > 1)
                        && flash_params.map_or(mask_is_prefill, |fp| fp.causal),
                };

                return Sdpa.run_attention(
                    query,
                    &k_4d,
                    &v_4d,
                    attention_mask,
                    Some(&prefix_flash_params),
                    sdpa_params,
                );
            }

            let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
            let k_batched =
                unpack_gathered_kv(&k_gathered, &kv_lens, key_value_heads, head_size, device)?;
            let v_batched =
                unpack_gathered_kv(&v_gathered, &kv_lens, key_value_heads, head_size, device)?;
            let adjusted_mask = match attention_mask {
                AttentionMask::Custom(t) => AttentionMask::Custom(adjust_kv_mask(t, max_kv)?),
                other => other.clone(),
            };

            return Sdpa.run_attention(
                query,
                &k_batched,
                &v_batched,
                &adjusted_mask,
                None,
                sdpa_params,
            );
        }

        // === Regular prompt path (no prefix cache, write_cache=true only) ===
        #[allow(clippy::cast_possible_truncation)]
        let single_token_first_prompt =
            write_cache && input_metadata.is_first_prompt_chunk && seq_len == 1;
        let att = if matches!(attention_mask, AttentionMask::None) && !single_token_first_prompt {
            None
        } else {
            Some(Sdpa.run_attention(
                query,
                key,
                value,
                attention_mask,
                flash_params,
                sdpa_params,
            )?)
        };

        if let Some(att) = att {
            if write_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
                let (key, value) = if seq_len > 1 {
                    let k = key
                        .transpose(1, 2)?
                        .reshape(((), key_value_heads, head_size))?;
                    let v = value
                        .transpose(1, 2)?
                        .reshape(((), key_value_heads, head_size))?;
                    (k, v)
                } else {
                    (
                        key.reshape(((), key_value_heads, head_size))?,
                        value.reshape(((), key_value_heads, head_size))?,
                    )
                };
                let key_cache = key_cache.as_mut().unwrap();
                let value_cache = value_cache.as_mut().unwrap();
                write_kv_cache(
                    &key,
                    &value,
                    self.k_scale.as_ref(),
                    self.v_scale.as_ref(),
                    key_cache,
                    value_cache,
                    slot_mapping,
                )?;
            }
            // Return result in prefill or first prefix chunk
            return Ok(att);
        }

        // === Decode path ===
        // paged-attn expects [batch_size, num_tokens, num_heads, head_size]
        let query = if seq_len > 1 {
            query
                .transpose(1, 2)?
                .reshape(((), attention_heads, head_size))?
        } else {
            query.reshape(((), attention_heads, head_size))?
        };
        let (key, value) = if write_cache {
            if seq_len > 1 {
                (
                    Some(
                        key.transpose(1, 2)?
                            .reshape(((), key_value_heads, head_size))?,
                    ),
                    Some(
                        value
                            .transpose(1, 2)?
                            .reshape(((), key_value_heads, head_size))?,
                    ),
                )
            } else {
                (
                    Some(key.reshape(((), key_value_heads, head_size))?),
                    Some(value.reshape(((), key_value_heads, head_size))?),
                )
            }
        } else {
            (None, None)
        };

        if write_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let key_cache = key_cache.as_mut().unwrap();
            let value_cache = value_cache.as_mut().unwrap();
            write_kv_cache(
                key.as_ref().unwrap(),
                value.as_ref().unwrap(),
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                key_cache,
                value_cache,
                slot_mapping,
            )?;
        }

        #[allow(clippy::cast_possible_truncation)]
        let dev = query.device().location();

        let attention_backend = AttentionBackendKind::from_cache(
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
        );
        let decode_head_size_limit = match attention_backend {
            AttentionBackendKind::FlashInfer => FLASHINFER_DECODE_MAX_HEAD_SIZE,
            AttentionBackendKind::Standard => STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE,
        };

        if head_size > decode_head_size_limit {
            let block_tables = resolve_block_tables(&dev).unwrap();
            let context_lens_t = resolve_context_lens(&dev).unwrap();
            let kv_lens: Vec<usize> = match context_lens_t.dtype() {
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
            };
            let cu_kv = cumulative_seqlens_from_lengths(&kv_lens, query.device())?;
            let (k_gathered, v_gathered) = gather_kv_cache_for_layout(
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                block_tables,
                &cu_kv,
                query.dtype(),
            )?;
            let q_4d = query.reshape((batch_size, attention_heads, 1, head_size))?;

            if supports_packed_varlen_sdpa(&query) {
                let cu_q =
                    cumulative_seqlens_from_lengths(&vec![1usize; batch_size], query.device())?;
                let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
                let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;
                let mut cu_q_map = HashMap::new();
                cu_q_map.insert(dev, cu_q);
                let mut cu_kv_map = HashMap::new();
                cu_kv_map.insert(dev, cu_kv);
                let decode_flash_params = FlashParams {
                    max_q: 1,
                    cumulative_seqlens_q: cu_q_map,
                    logical_k: FlashKMeta {
                        max: kv_lens.iter().copied().max().unwrap_or(0) as u32,
                        cumulative_seqlens: cu_kv_map,
                    },
                    sliding_k: None,
                    // decode: the single new token attends all cached keys; no causal masking.
                    causal: false,
                };
                return Sdpa.run_attention(
                    &q_4d,
                    &k_4d,
                    &v_4d,
                    &AttentionMask::None,
                    Some(&decode_flash_params),
                    sdpa_params,
                );
            }

            let k_batched = unpack_gathered_kv(
                &k_gathered,
                &kv_lens,
                key_value_heads,
                head_size,
                query.device(),
            )?;
            let v_batched = unpack_gathered_kv(
                &v_gathered,
                &kv_lens,
                key_value_heads,
                head_size,
                query.device(),
            )?;
            return Sdpa.run_attention(
                &q_4d,
                &k_batched,
                &v_batched,
                &AttentionMask::None,
                None,
                sdpa_params,
            );
        }

        #[cfg(all(feature = "cuda", target_family = "unix"))]
        if attention_backend == AttentionBackendKind::FlashInfer {
            if alibi_slopes.is_some() || sdpa_params.sinks.is_some() {
                candle_core::bail!("FlashInfer paged attention does not support alibi/sinks");
            }
            let use_tensor_cores = FLASHINFER_TENSOR_CORE_DECODE_ENABLED
                && head_size <= FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE
                && query.dtype() != DType::F32;
            let fi_meta =
                input_metadata.flashinfer_decode_metadata(&dev, use_full, use_tensor_cores)?;

            return flashinfer_decode(
                &query,
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                fi_meta.paged_kv_indptr,
                fi_meta.paged_kv_indices,
                fi_meta.paged_kv_last_page_len,
                fi_meta.request_indices,
                fi_meta.kv_tile_indices,
                fi_meta.o_indptr,
                fi_meta.kv_chunk_size,
                fi_meta.block_valid_mask,
                sdpa_params.softmax_scale,
                sdpa_params.sliding_window,
                sdpa_params.softcap,
                use_tensor_cores,
            );
        }

        let res = paged_attention(
            &query,
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            resolve_block_tables(&dev).unwrap(),
            resolve_context_lens(&dev).unwrap(),
            alibi_slopes.as_ref(),
            if use_full {
                input_metadata.full_max_context_len.unwrap()
            } else {
                input_metadata.max_context_len.unwrap()
            },
            sdpa_params.softmax_scale,
            sdpa_params.softcap.unwrap_or(1.0f32),
            sdpa_params.sinks.as_ref(),
        )?;

        Ok(res)
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
