use std::collections::HashMap;

use candle_core::{DType, Device, DeviceLocation, Result, Tensor};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use mistralrs_paged_attn::{
    flashinfer_decode, flashinfer_prefill, gather_kv_cache_flashinfer,
    reshape_and_cache_flashinfer, FlashInferDecodeScratch,
};
use mistralrs_paged_attn::{kv_scale_update, paged_attention, reshape_and_cache};

const KV_SCALE_UPDATE_ITERATION: i32 = 128;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::Sdpa,
    paged_attention::{
        plan::{DecodePlan, DecodePlanInput, PrefixPrefillPlan, PrefixPrefillPlanInput},
        AttentionBackendKind, _PAD_SLOT_ID,
    },
    pipeline::text_models_inputs_processor::{
        FlashKMeta, FlashParams, PagedAttentionInputMetadata,
    },
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

fn prefix_gather_causal_mask(
    query_lens: &[usize],
    kv_lens: &[usize],
    q_max: usize,
    kv_max: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<AttentionMask> {
    let batch = query_lens.len();
    let mut mask = Vec::with_capacity(batch * q_max * kv_max);
    for (&q_len, &kv_len) in query_lens.iter().zip(kv_lens.iter()) {
        let prefix_len = kv_len.saturating_sub(q_len);
        for q_idx in 0..q_max {
            for kv_idx in 0..kv_max {
                let masked = if q_idx >= q_len || kv_idx >= kv_len {
                    q_idx >= q_len || kv_idx != 0
                } else {
                    let q_pos = prefix_len + q_idx;
                    let future = kv_idx > q_pos;
                    let too_old = sliding_window
                        .is_some_and(|window| q_pos >= window && kv_idx <= q_pos - window);
                    future || too_old
                };
                mask.push(if masked { f32::NEG_INFINITY } else { 0.0 });
            }
        }
    }
    Ok(AttentionMask::Custom(
        Tensor::from_vec(mask, (batch, 1, q_max, kv_max), device)?.to_dtype(dtype)?,
    ))
}

fn supports_packed_varlen_sdpa(query: &Tensor) -> bool {
    query.device().is_cpu()
        || (query.device().is_cuda() && crate::using_flash_attn() && query.dtype() != DType::F32)
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

    fn use_gather_path(&self, attention_mask: &AttentionMask, write_cache: bool) -> bool {
        let has_block_tables = self.input_metadata.block_tables.is_some();
        let has_cached_prefix = self.input_metadata.num_cached_tokens.is_some();
        let mask_is_prefill = !matches!(attention_mask, AttentionMask::None);
        let single_token_first_prompt =
            self.input_metadata.is_first_prompt_chunk && self.dims.seq_len == 1;
        if write_cache {
            has_cached_prefix && has_block_tables
        } else {
            (has_cached_prefix || mask_is_prefill || single_token_first_prompt) && has_block_tables
        }
    }
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

    fn update_kv_scale_if_needed(
        &self,
        tensors: PagedForwardTensors<'_>,
        key_cache: Option<&Tensor>,
        write_cache: bool,
    ) -> Result<()> {
        if write_cache {
            if let (Some(k_scale), Some(v_scale), Some(key_cache)) =
                (&self.k_scale, &self.v_scale, key_cache)
            {
                if self.kv_updated_times.load(Ordering::Relaxed) < KV_SCALE_UPDATE_ITERATION
                    && key_cache.dtype() == DType::F8E4M3
                {
                    kv_scale_update(tensors.key, tensors.value, k_scale, v_scale)?;
                    self.kv_updated_times.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        Ok(())
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
        if !ctx.use_gather_path(tensors.attention_mask, write_cache) {
            return Ok(None);
        }

        let dev = tensors.query.device().location();
        let block_tables = ctx.block_tables(&dev).unwrap();
        if write_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let k_flat = tensors.key.transpose(1, 2)?.reshape((
                (),
                ctx.dims.key_value_heads,
                ctx.dims.head_size,
            ))?;
            let v_flat = tensors.value.transpose(1, 2)?.reshape((
                (),
                ctx.dims.key_value_heads,
                ctx.dims.head_size,
            ))?;
            let key_cache = key_cache.as_mut().unwrap();
            let value_cache = value_cache.as_mut().unwrap();
            write_kv_cache(
                &k_flat,
                &v_flat,
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
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
        let new_token_lens = new_token_lens_from_slot_mapping(
            ctx.slot_mapping_full,
            ctx.dims.batch_size,
            ctx.dims.seq_len,
        )?;
        let query_lens = ctx
            .input_metadata
            .query_lens
            .clone()
            .unwrap_or_else(|| new_token_lens.clone());
        let full_kv_lens =
            if let Some(num_cached_tokens) = ctx.input_metadata.num_cached_tokens.as_ref() {
                num_cached_tokens
                    .iter()
                    .zip(query_lens.iter())
                    .map(|(&cached, &query_len)| cached + query_len)
                    .collect::<Vec<_>>()
            } else {
                new_token_lens.clone()
            };
        let kv_lens = if let Some(window) = ctx.sdpa_params.sliding_window {
            if !ctx.use_full {
                let block_size =
                    cache_block_size(key_cache.as_ref().unwrap(), value_cache.as_ref().unwrap())?;
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
        let query_lens_match_seq_len = query_lens.iter().all(|&len| len == ctx.dims.seq_len);
        let mask_is_prefill = !matches!(tensors.attention_mask, AttentionMask::None);
        let prefill_causal = query_lens.iter().any(|&len| len > 1)
            && ctx.flash_params.map_or(mask_is_prefill, |fp| fp.causal);
        let causality_known = !tensors.attention_mask.is_custom() || ctx.flash_params.is_some();
        let attention_backend = AttentionBackendKind::from_cache(
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
        );
        let prefill_plan = PrefixPrefillPlan::choose(PrefixPrefillPlanInput {
            device_is_cuda: tensors.query.device().is_cuda(),
            dtype: tensors.query.dtype(),
            has_sinks: ctx.sdpa_params.sinks.is_some(),
            causal: prefill_causal,
            causality_known,
            head_size: ctx.dims.head_size,
            attention_heads: ctx.dims.attention_heads,
            key_value_heads: ctx.dims.key_value_heads,
            query_lens_match_seq_len,
            attention_backend,
        });
        match prefill_plan {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            PrefixPrefillPlan::FlashInfer(plan) => {
                return self
                    .run_flashinfer_prefill(
                        ctx,
                        tensors.query,
                        key_cache.as_ref().unwrap(),
                        value_cache.as_ref().unwrap(),
                        plan,
                    )
                    .map(Some);
            }
            PrefixPrefillPlan::GatherSdpa => {}
        }

        let cu_kv = if ctx.sdpa_params.sliding_window.is_none() {
            if let Some(map) = ctx.input_metadata.cu_seqlens_kv.as_ref() {
                resolve_tensor_for_device(map, device, "cu_seqlens_kv")?
            } else {
                cumulative_seqlens_from_lengths(&kv_lens, device)?
            }
        } else {
            cumulative_seqlens_from_lengths(&kv_lens, device)?
        };

        let (k_gathered, v_gathered) = gather_kv_cache_for_layout(
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            block_tables,
            &cu_kv,
            tensors.query.dtype(),
        )?;
        let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
        let adjusted_mask = match tensors.attention_mask {
            AttentionMask::Custom(t) => AttentionMask::Custom(adjust_kv_mask(t, max_kv)?),
            AttentionMask::CausalFlash => prefix_gather_causal_mask(
                &query_lens,
                &kv_lens,
                ctx.dims.seq_len,
                max_kv,
                ctx.sdpa_params.sliding_window,
                tensors.query.dtype(),
                device,
            )?,
            other => other.clone(),
        };

        if supports_packed_varlen_sdpa(tensors.query) {
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
            let mask_is_prefill = !matches!(tensors.attention_mask, AttentionMask::None);
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
                causal: query_lens.iter().any(|&len| len > 1)
                    && ctx.flash_params.map_or(mask_is_prefill, |fp| fp.causal),
            };
            return Sdpa
                .run_attention(
                    tensors.query,
                    &k_4d,
                    &v_4d,
                    &adjusted_mask,
                    Some(&prefix_flash_params),
                    ctx.sdpa_params,
                )
                .map(Some);
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
        Sdpa.run_attention(
            tensors.query,
            &k_batched,
            &v_batched,
            &adjusted_mask,
            None,
            ctx.sdpa_params,
        )
        .map(Some)
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn run_flashinfer_prefill(
        &self,
        ctx: &PagedForwardCtx<'_>,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        plan: crate::flashinfer::FlashInferPrefillPlan,
    ) -> Result<Tensor> {
        let flashinfer = ctx
            .input_metadata
            .flashinfer
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("FlashInfer metadata missing"))?;
        let fi_meta = flashinfer.prefill_metadata(&query.device().location())?;
        let q_flat = if ctx.dims.seq_len > 1 {
            query
                .transpose(1, 2)?
                .reshape(((), ctx.dims.attention_heads, ctx.dims.head_size))?
        } else {
            query.reshape(((), ctx.dims.attention_heads, ctx.dims.head_size))?
        };
        let out = flashinfer_prefill(
            &q_flat,
            key_cache,
            value_cache,
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
            ctx.dims.batch_size,
            plan.causal(),
            ctx.sdpa_params.softmax_scale,
            ctx.sdpa_params.sliding_window,
            ctx.sdpa_params.softcap,
        )
        .map_err(|err| {
            err.context(format!(
                "FlashInfer prefill failed: batch={} qo_heads={} kv_heads={} head_size={}",
                ctx.dims.batch_size,
                ctx.dims.attention_heads,
                ctx.dims.key_value_heads,
                ctx.dims.head_size,
            ))
        })?;
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
        let Some(att) = (if matches!(tensors.attention_mask, AttentionMask::None)
            && !single_token_first_prompt
        {
            None
        } else {
            Some(Sdpa.run_attention(
                tensors.query,
                tensors.key,
                tensors.value,
                tensors.attention_mask,
                ctx.flash_params,
                ctx.sdpa_params,
            )?)
        }) else {
            return Ok(None);
        };

        if write_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let (key, value) = if ctx.dims.seq_len > 1 {
                let k = tensors.key.transpose(1, 2)?.reshape((
                    (),
                    ctx.dims.key_value_heads,
                    ctx.dims.head_size,
                ))?;
                let v = tensors.value.transpose(1, 2)?.reshape((
                    (),
                    ctx.dims.key_value_heads,
                    ctx.dims.head_size,
                ))?;
                (k, v)
            } else {
                (
                    tensors
                        .key
                        .reshape(((), ctx.dims.key_value_heads, ctx.dims.head_size))?,
                    tensors
                        .value
                        .reshape(((), ctx.dims.key_value_heads, ctx.dims.head_size))?,
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
            if ctx.dims.seq_len > 1 {
                (
                    Some(tensors.key.transpose(1, 2)?.reshape((
                        (),
                        ctx.dims.key_value_heads,
                        ctx.dims.head_size,
                    ))?),
                    Some(tensors.value.transpose(1, 2)?.reshape((
                        (),
                        ctx.dims.key_value_heads,
                        ctx.dims.head_size,
                    ))?),
                )
            } else {
                (
                    Some(tensors.key.reshape((
                        (),
                        ctx.dims.key_value_heads,
                        ctx.dims.head_size,
                    ))?),
                    Some(tensors.value.reshape((
                        (),
                        ctx.dims.key_value_heads,
                        ctx.dims.head_size,
                    ))?),
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
                &ctx.slot_mapping,
            )?;
        }

        let dev = query.device().location();
        let key_cache_ref = key_cache.as_ref().unwrap();
        let value_cache_ref = value_cache.as_ref().unwrap();
        let attention_backend = AttentionBackendKind::from_cache(key_cache_ref, value_cache_ref);
        match DecodePlan::choose(DecodePlanInput {
            attention_backend,
            dtype: query.dtype(),
            head_size: ctx.dims.head_size,
            has_alibi: ctx.alibi_slopes.is_some(),
            has_sinks: ctx.sdpa_params.sinks.is_some(),
        })? {
            DecodePlan::GatherSdpa => {
                self.run_decode_gather_sdpa(ctx, &query, key_cache_ref, value_cache_ref, &dev)
            }
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            DecodePlan::FlashInfer(plan) => {
                self.run_flashinfer_decode(ctx, &query, key_cache_ref, value_cache_ref, &dev, plan)
            }
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
    ) -> Result<Tensor> {
        let block_tables = ctx.block_tables(dev).unwrap();
        let context_lens_t = ctx.context_lens(dev).unwrap();
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
            key_cache,
            value_cache,
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            block_tables,
            &cu_kv,
            query.dtype(),
        )?;
        let q_4d = query.reshape((
            ctx.dims.batch_size,
            ctx.dims.attention_heads,
            1,
            ctx.dims.head_size,
        ))?;

        if supports_packed_varlen_sdpa(query) {
            let cu_q = cumulative_seqlens_from_lengths(
                &vec![1usize; ctx.dims.batch_size],
                query.device(),
            )?;
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
        Sdpa.run_attention(
            &q_4d,
            &k_batched,
            &v_batched,
            &AttentionMask::None,
            None,
            ctx.sdpa_params,
        )
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn run_flashinfer_decode(
        &self,
        ctx: &PagedForwardCtx<'_>,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        dev: &DeviceLocation,
        flashinfer_plan: crate::flashinfer::FlashInferDecodePlan,
    ) -> Result<Tensor> {
        let use_tensor_cores = flashinfer_plan.use_tensor_cores();
        let fi_meta = ctx
            .input_metadata
            .flashinfer
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("FlashInfer metadata missing"))?
            .decode_metadata(dev, ctx.sdpa_params.sliding_window, use_tensor_cores)?;
        let use_tensor_cores = use_tensor_cores && fi_meta.tmp_v.is_none();
        let (_, num_kv_heads, _, _) = key_cache.dims4()?;
        flashinfer_decode(
            query,
            key_cache,
            value_cache,
            fi_meta.paged_kv_indptr,
            fi_meta.paged_kv_indices,
            fi_meta.paged_kv_last_page_len,
            fi_meta.q_indptr,
            fi_meta.qo_tile_indices,
            fi_meta.request_indices,
            fi_meta.kv_tile_indices,
            fi_meta.o_indptr,
            fi_meta.kv_chunk_size,
            fi_meta.block_valid_mask,
            ctx.sdpa_params.softmax_scale,
            ctx.sdpa_params.sliding_window,
            ctx.sdpa_params.softcap,
            use_tensor_cores,
            fi_meta
                .tmp_v
                .zip(fi_meta.tmp_s)
                .map(|(tmp_v, tmp_s)| FlashInferDecodeScratch { tmp_v, tmp_s }),
        )
        .map_err(|err| {
            err.context(format!(
                "FlashInfer decode failed: batch={} padded_batch={} qo_heads={} kv_heads={} head_size={} use_tensor_cores={use_tensor_cores}",
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
        paged_attention(
            query,
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
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
        self.update_kv_scale_if_needed(tensors, key_cache.as_ref(), write_cache)?;
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
