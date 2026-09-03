#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use mistralrs_quant::{
    linear_no_bias, ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder,
};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding_with_legacy_tied_uqff, CausalMasker, Llama3RotaryEmbedding, RmsNorm, Sdpa},
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashKMeta, FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::TextConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedChunkMaskRow {
    query_start: usize,
    query_len: usize,
    key_start: usize,
    key_len: usize,
}

fn causal_mask_values(
    rows: &[FixedChunkMaskRow],
    query_width: usize,
    key_width: usize,
    chunk_size: Option<usize>,
) -> Result<Vec<f32>> {
    if rows.is_empty()
        || query_width == 0
        || key_width == 0
        || chunk_size.is_some_and(|size| size == 0)
    {
        candle_core::bail!("Llama4 attention has invalid mask dimensions");
    }

    let mut values = Vec::with_capacity(rows.len() * query_width * key_width);
    for row in rows {
        if row.query_len == 0
            || row.query_len > query_width
            || row.key_len == 0
            || row.key_len > key_width
        {
            candle_core::bail!("Llama4 attention has invalid mask row lengths");
        }
        let query_end = row
            .query_start
            .checked_add(row.query_len)
            .ok_or_else(|| candle_core::Error::msg("Llama4 query position overflow"))?;
        let key_end = row
            .key_start
            .checked_add(row.key_len)
            .ok_or_else(|| candle_core::Error::msg("Llama4 key position overflow"))?;
        if row.query_start < row.key_start || query_end > key_end {
            candle_core::bail!("Llama4 attention has inconsistent mask positions");
        }

        for query_idx in 0..query_width {
            if query_idx >= row.query_len {
                values.push(0.0);
                values.extend(std::iter::repeat_n(f32::NEG_INFINITY, key_width - 1));
                continue;
            }

            let query_position = row.query_start + query_idx;
            let chunk_start = chunk_size.map_or(0, |size| query_position / size * size);
            for key_idx in 0..key_width {
                let visible = key_idx < row.key_len
                    && row.key_start + key_idx >= chunk_start
                    && row.key_start + key_idx <= query_position;
                values.push(if visible { 0.0 } else { f32::NEG_INFINITY });
            }
        }
    }
    Ok(values)
}

#[cfg(test)]
fn fixed_chunk_mask_values(
    rows: &[FixedChunkMaskRow],
    query_width: usize,
    key_width: usize,
    chunk_size: usize,
) -> Result<Vec<f32>> {
    causal_mask_values(rows, query_width, key_width, Some(chunk_size))
}

fn fixed_chunk_mask_layout(
    batch_size: usize,
    query_width: usize,
    seqlen_offsets: &[usize],
    metadata: Option<&PagedAttentionInputMetadata>,
) -> Result<(Vec<FixedChunkMaskRow>, usize)> {
    if batch_size == 0 || query_width == 0 {
        candle_core::bail!("Llama4 fixed-chunk attention has an empty query");
    }

    let Some(metadata) = metadata else {
        if seqlen_offsets.len() != batch_size {
            candle_core::bail!(
                "Llama4 fixed-chunk attention has {} offsets for batch size {batch_size}",
                seqlen_offsets.len()
            );
        }
        let rows = seqlen_offsets
            .iter()
            .map(|&query_start| {
                let key_len = query_start
                    .checked_add(query_width)
                    .ok_or_else(|| candle_core::Error::msg("Llama4 key length overflow"))?;
                Ok(FixedChunkMaskRow {
                    query_start,
                    query_len: query_width,
                    key_start: 0,
                    key_len,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let key_width = rows.iter().map(|row| row.key_len).max().unwrap_or(0);
        return Ok((rows, key_width));
    };

    let selected_lens = metadata
        .paged_context_lens_cpu
        .as_deref()
        .ok_or_else(|| candle_core::Error::msg("Llama4 attention is missing paged KV lengths"))?;
    let full_lens = metadata
        .full_paged_context_lens_cpu
        .as_deref()
        .unwrap_or(selected_lens);
    if selected_lens.len() != full_lens.len() {
        candle_core::bail!("Llama4 attention has inconsistent paged KV lengths");
    }

    let rows = if let Some(query_lens) = metadata.query_lens.as_deref() {
        if query_lens.len() != batch_size
            || selected_lens.len() != batch_size
            || query_lens.iter().any(|&len| len == 0 || len > query_width)
        {
            candle_core::bail!("Llama4 attention has invalid prompt row lengths");
        }
        query_lens
            .iter()
            .zip(selected_lens)
            .zip(full_lens)
            .map(|((&query_len, &key_len), &full_len)| {
                let query_start = full_len.checked_sub(query_len).ok_or_else(|| {
                    candle_core::Error::msg("Llama4 query starts before its KV context")
                })?;
                let key_start = full_len.checked_sub(key_len).ok_or_else(|| {
                    candle_core::Error::msg("Llama4 paged KV window exceeds its full context")
                })?;
                Ok(FixedChunkMaskRow {
                    query_start,
                    query_len,
                    key_start,
                    key_len,
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let query_rows = batch_size
            .checked_mul(query_width)
            .ok_or_else(|| candle_core::Error::msg("Llama4 query row count overflow"))?;
        if selected_lens.len() != query_rows {
            candle_core::bail!(
                "Llama4 decode has {} KV rows for {query_rows} query rows",
                selected_lens.len()
            );
        }
        selected_lens
            .iter()
            .zip(full_lens)
            .map(|(&key_len, &full_len)| {
                let query_start = full_len.checked_sub(1).ok_or_else(|| {
                    candle_core::Error::msg("Llama4 decode has an empty KV context")
                })?;
                let key_start = full_len.checked_sub(key_len).ok_or_else(|| {
                    candle_core::Error::msg("Llama4 paged KV window exceeds its full context")
                })?;
                Ok(FixedChunkMaskRow {
                    query_start,
                    query_len: 1,
                    key_start,
                    key_len,
                })
            })
            .collect::<Result<Vec<_>>>()?
    };
    let key_width = rows.iter().map(|row| row.key_len).max().unwrap_or(0);
    Ok((rows, key_width))
}

fn absolute_causal_attention_mask(
    input_ids: &Tensor,
    seqlen_offsets: &[usize],
    metadata: Option<&PagedAttentionInputMetadata>,
    chunk_size: Option<usize>,
    dtype: DType,
) -> Result<AttentionMask> {
    let (batch_size, query_width) = input_ids.dims2()?;
    let (rows, key_width) =
        fixed_chunk_mask_layout(batch_size, query_width, seqlen_offsets, metadata)?;
    let row_count = rows.len();
    let mask_query_width = if metadata.is_some_and(|metadata| metadata.query_lens.is_none()) {
        1
    } else {
        query_width
    };
    let values = causal_mask_values(&rows, mask_query_width, key_width, chunk_size)?;
    let mask = Tensor::from_vec(
        values,
        (row_count, 1, mask_query_width, key_width),
        input_ids.device(),
    )?
    .to_dtype(dtype)?;
    Ok(AttentionMask::Custom(mask))
}

fn fixed_chunk_attention_mask(
    input_ids: &Tensor,
    seqlen_offsets: &[usize],
    metadata: Option<&PagedAttentionInputMetadata>,
    chunk_size: usize,
    dtype: DType,
) -> Result<AttentionMask> {
    absolute_causal_attention_mask(input_ids, seqlen_offsets, metadata, Some(chunk_size), dtype)
}

fn chunked_flash_segment_lens(query_lens: &[usize], chunk_size: usize) -> Result<Vec<usize>> {
    if chunk_size == 0 || query_lens.is_empty() || query_lens.contains(&0) {
        candle_core::bail!("Llama4 packed chunked attention has invalid sequence lengths");
    }
    let mut segments = Vec::new();
    for &query_len in query_lens {
        let mut remaining = query_len;
        while remaining > 0 {
            let segment = remaining.min(chunk_size);
            segments.push(segment);
            remaining -= segment;
        }
    }
    Ok(segments)
}

fn chunked_flash_params(
    query_lens: &[usize],
    chunk_size: usize,
    devices: &[Device],
) -> Result<FlashParams> {
    let segment_lens = chunked_flash_segment_lens(query_lens, chunk_size)?;
    let mut cumulative = Vec::with_capacity(segment_lens.len() + 1);
    cumulative.push(0u32);
    for &segment in &segment_lens {
        let segment = u32::try_from(segment).map_err(candle_core::Error::wrap)?;
        let next = cumulative
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(segment)
            .ok_or_else(|| candle_core::Error::msg("Llama4 packed token count overflow"))?;
        cumulative.push(next);
    }
    let mut cumulative_seqlens_q = HashMap::new();
    let mut cumulative_seqlens_k = HashMap::new();
    for device in devices {
        let tensor = Tensor::new(cumulative.as_slice(), device)?;
        cumulative_seqlens_q.insert(device.location(), tensor.clone());
        cumulative_seqlens_k.insert(device.location(), tensor);
    }
    let max_segment = u32::try_from(segment_lens.iter().copied().max().unwrap_or(0))
        .map_err(candle_core::Error::wrap)?;
    Ok(FlashParams {
        max_q: max_segment,
        cumulative_seqlens_q,
        logical_k: FlashKMeta {
            max: max_segment,
            cumulative_seqlens: cumulative_seqlens_k,
        },
        sliding_k: None,
        causal: true,
        packed: true,
        varlen_segment_lens: Some(segment_lens),
    })
}

struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Llama3RotaryEmbedding>,
    max_seq_len: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    norm: Option<RmsNorm>,
    use_rope: bool,
    floor_scale: Option<f32>,
    attn_scale: Option<f32>,
    attn_temperature_tuning: Option<f32>,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        layer_idx: usize,
        loading_isq: bool,
        mapper: &dyn DeviceMapper,
        rope: Arc<Llama3RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = ColumnParallelLayer::new(
            size_in,
            size_q,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        )?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            size_q,
            size_in,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let use_rope = !(layer_idx + 1).is_multiple_of(4);
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let norm = if cfg.use_qk_norm && use_rope {
            let vb = mapper.set_device(layer_idx, vb, false);
            Some(RmsNorm::from_w(
                Tensor::ones(head_dim, vb.dtype(), vb.device())?,
                1e-6,
            )?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb: rope,
            max_seq_len: cfg.max_position_embeddings,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: use_rope.then_some(cfg.attention_chunk_size),
                sinks: None,
            },
            norm,
            use_rope,
            floor_scale: cfg.floor_scale,
            attn_scale: cfg.attn_scale,
            attn_temperature_tuning: cfg.attn_temperature_tuning,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
        attention_mask: &AttentionMask,
        flash_params_override: Option<&FlashParams>,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;

        q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        if self.use_rope {
            let rope_positions = ctx
                .text_positions(q.device(), q.dim(2)?)?
                .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
            (q, k) = self.rotary_emb.forward(&q, &k, rope_positions)?;
        }

        if let Some(qk_norm) = &self.norm {
            q = qk_norm.forward(&q)?;
            k = qk_norm.forward(&k)?;
        }

        if self.attn_temperature_tuning.is_some() && !self.use_rope {
            let floor_scale = self.floor_scale.unwrap() as f64;
            let attn_scale = self.attn_scale.unwrap() as f64;
            let floor = ((position_ids.to_dtype(DType::F32)? + 1.)? / floor_scale)?.floor()?;
            let attn_scales = (((floor + 1.0)?.log()? * attn_scale)? + 1.0)?;

            q = q
                .to_dtype(DType::F32)?
                .broadcast_mul(&attn_scales.reshape((b_sz, 1, seq_len, 1))?)?
                .to_dtype(q.dtype())?;
        }

        let metadata = ctx.paged_layer(layer_idx);
        let flash_params = flash_params_override.unwrap_or_else(|| ctx.flash_params());
        let packed_sdpa_params = flash_params_override.map(|_| SdpaParams {
            n_kv_groups: self.sdpa_params.n_kv_groups,
            softcap: self.sdpa_params.softcap,
            softmax_scale: self.sdpa_params.softmax_scale,
            sliding_window: None,
            sinks: self.sdpa_params.sinks.clone(),
        });
        let sdpa_params = packed_sdpa_params.as_ref().unwrap_or(&self.sdpa_params);
        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                    // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    // Sanity check.
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    attention_mask,
                    Some(flash_params),
                    sdpa_params,
                )?
            }
        };

        y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.o_proj.forward(&y)
    }
}
/// MoE layer for Llama4 using the unified MoEExperts
struct TextMoe {
    experts: MoEExperts,
    shared_expert: crate::layers::Mlp,
    router: Arc<dyn QuantMethod>,
    topk: usize,
}

impl TextMoe {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        quantization_config: &Option<QuantizedConfig>,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        layer_device: Device,
    ) -> Result<Self> {
        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_local_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.intermediate_size,
            expert_proj_names: crate::moe::ExpertProjNames::DEFAULT,
        };

        let experts = MoEExperts::new(
            &moe_cfg,
            vb.clone(),
            layer_device.clone(),
            comm,
            loading_isq,
            quantization_config,
            cfg.hidden_act,
        )?;

        let router = linear_no_bias(
            cfg.hidden_size,
            cfg.num_local_experts,
            quantization_config,
            vb.pp("router").set_device(layer_device),
        )?;

        let shared_expert = crate::layers::Mlp::new(
            vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.intermediate_size,
            quantization_config,
            cfg.hidden_act,
            comm,
        )?;

        Ok(Self {
            experts,
            shared_expert,
            router,
            topk: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let router_logits = self.router.forward(&xs_flat)?;

        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.topk,
                score_function: crate::ops::MoeRouterScoreFunction::Raw,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Sigmoid,
                renormalize: false,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;

        let routed_out = self
            .experts
            .forward(xs, topk.values, &topk.indices)?
            .reshape((bs, seq_len, hidden_dim))?;

        let out = self.shared_expert.forward(xs)?;

        out + routed_out
    }
}

enum MoeOrMlp {
    Mlp(crate::layers::Mlp),
    Moe(TextMoe),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(l) => l.forward(xs),
            Self::Moe(l) => l.forward(xs),
        }
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    ff: MoeOrMlp,
    use_chunked_attention: bool,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Arc<Llama3RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let use_chunked_attention = !(layer_idx + 1).is_multiple_of(4);
        let attn = CausalSelfAttention::new(
            vb.pp("self_attn"),
            cfg,
            layer_idx,
            loading_isq,
            mapper,
            rope,
            paged_attn,
            comm,
        )?;
        let is_moe_layer = cfg.moe_layers().contains(&layer_idx);
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);
        let ff = if is_moe_layer {
            let moe = TextMoe::new(
                mapper.set_device(layer_idx, vb.pp("feed_forward"), loading_isq),
                cfg,
                &cfg.quantization_config,
                comm,
                loading_isq,
                layer_device,
            )?;
            MoeOrMlp::Moe(moe)
        } else {
            let mlp = crate::layers::Mlp::new(
                mapper.set_device(layer_idx, vb.pp("feed_forward"), loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size_mlp,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?;
            MoeOrMlp::Mlp(mlp)
        };
        let rms_1 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            ff,
            use_chunked_attention,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
        attention_mask: &AttentionMask,
        chunked_mask: &AttentionMask,
        chunked_flash_params: Option<&FlashParams>,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let mask = if self.use_chunked_attention {
            if chunked_flash_params.is_some() {
                AttentionMask::CausalFlash
            } else {
                chunked_mask.clone()
            }
        } else {
            attention_mask.clone()
        };
        let flash_params_override = if self.use_chunked_attention {
            chunked_flash_params
        } else {
            None
        };
        let x = (self.attn.forward(
            &x,
            position_ids,
            &mask,
            flash_params_override,
            kv_cache,
            ctx,
            layer_idx,
        )? + residual)?;
        let residual = &x;
        let x = (self.ff.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

pub struct TextModel {
    wte: Arc<dyn QuantMethod>,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    kv_cache: crate::pipeline::EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    attention_chunk_size: usize,
}

impl TextModel {
    pub fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &TextConfig,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();

        let wte = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb_lm_head.clone(), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            wte.clone()
        };
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut ropes = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(Llama3RotaryEmbedding::new_llama4(
                    vb_m.dtype(),
                    cfg,
                    device,
                    is_gptx,
                )?),
            );
        }
        let blocks = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading text repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|i| {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(i)?;
            Block::new(
                vb_m.pp(format!("layers.{i}")),
                cfg,
                &*mapper,
                i,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                paged_attn,
                &comm,
                normal_loading_metadata.real_device.clone(),
            )
        })?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            dtype,
            kv_cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: Some(cfg.attention_chunk_size),
                k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::StandardNoFlashInfer,
            },
            mapper,
            attention_chunk_size: cfg.attention_chunk_size,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.wte.embedding_forward(input_ids, self.dtype)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut x = input_embeds;
        let cache = &mut self.kv_cache.normal().0;
        let position_ids = ctx
            .text_positions(input_ids.device(), input_ids.dim(1)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
            .to_dtype(DType::I32)?;
        let mask_cache = ctx.mask_cache(cache);

        let mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            x.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let mask = if mask.is_custom()
            && !ctx.is_first_prompt_chunk()
            && ctx
                .paged_input_metadata()
                .is_some_and(|metadata| metadata.query_lens.is_some())
        {
            absolute_causal_attention_mask(
                input_ids,
                ctx.seqlen_offsets(),
                ctx.paged_input_metadata(),
                None,
                x.dtype(),
            )?
        } else {
            mask
        };
        let chunked_flash_params = if ctx.flash_params().packed {
            if ctx.seqlen_offsets().iter().any(|&offset| offset != 0) {
                candle_core::bail!("Llama4 packed chunked attention does not support cached keys");
            }
            if !ctx.is_first_prompt_chunk() {
                candle_core::bail!(
                    "Llama4 packed chunked attention requires the first prompt chunk"
                );
            }
            let query_lens = ctx
                .paged_input_metadata()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    candle_core::Error::msg(
                        "Llama4 packed chunked attention is missing logical query lengths",
                    )
                })?;
            let expected_tokens = query_lens.iter().sum::<usize>();
            if expected_tokens != input_ids.dim(1)? {
                candle_core::bail!(
                    "Llama4 packed chunked attention has {expected_tokens} logical tokens but {} physical tokens",
                    input_ids.dim(1)?
                );
            }
            Some(chunked_flash_params(
                query_lens,
                self.attention_chunk_size,
                &self.mapper.get_unique_devices(),
            )?)
        } else {
            None
        };
        let chunked_mask = if chunked_flash_params.is_none() {
            fixed_chunk_attention_mask(
                input_ids,
                ctx.seqlen_offsets(),
                ctx.paged_input_metadata(),
                self.attention_chunk_size,
                x.dtype(),
            )?
        } else {
            AttentionMask::None
        };
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        let chunked_mask = DeviceMappedMask::new(chunked_mask, &*self.mapper)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, block_idx)?;
            let mask_for_layer = mask.get(x.device());
            let chunked_mask_for_layer = chunked_mask.get(x.device());
            x = block.forward(
                &x,
                &position_ids.to_device(x.device())?,
                &mask_for_layer,
                &chunked_mask_for_layer,
                chunked_flash_params.as_ref(),
                &mut cache[block_idx],
                ctx,
                block_idx,
            )?;
        }
        let x = x.to_device(&self.device)?;
        let x = self.ln_f.forward(&x)?;
        let x = ctx.logits(&x)?;
        ctx.lm_head(&*self.lm_head, &x)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.wte);
        uvb_m.pp("norm").add(&self.ln_f);

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.rms_1);
            uvb_l.pp("post_attention_layernorm").add(&layer.rms_2);
        }

        uvb_m.to_safetensors()
    }
}

impl IsqModel for TextModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.residual_tensors_m(uvb.pp("model"))
    }
}

impl crate::speculative::SpeculativeTargetMixin for TextModel {}

impl NormalModel for TextModel {
    fn forward(&self, _input_ids: &Tensor, _ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        unreachable!()
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &crate::pipeline::EitherCache {
        &self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.blocks[0].attn.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for TextModel {}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible_keys(
        values: &[f32],
        row: usize,
        query: usize,
        query_width: usize,
        key_width: usize,
    ) -> Vec<usize> {
        let start = (row * query_width + query) * key_width;
        values[start..start + key_width]
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| (*value == 0.0).then_some(idx))
            .collect()
    }

    #[test]
    fn fixed_chunk_mask_resets_at_chunk_edges() {
        let rows = [FixedChunkMaskRow {
            query_start: 0,
            query_len: 6,
            key_start: 0,
            key_len: 6,
        }];
        let values = fixed_chunk_mask_values(&rows, 6, 6, 4).unwrap();

        assert_eq!(visible_keys(&values, 0, 3, 6, 6), vec![0, 1, 2, 3]);
        assert_eq!(visible_keys(&values, 0, 4, 6, 6), vec![4]);
        assert_eq!(visible_keys(&values, 0, 5, 6, 6), vec![4, 5]);
    }

    #[test]
    fn fixed_chunk_mask_handles_later_prompt_chunks() {
        let rows = [FixedChunkMaskRow {
            query_start: 7,
            query_len: 3,
            key_start: 0,
            key_len: 10,
        }];
        let values = fixed_chunk_mask_values(&rows, 3, 10, 4).unwrap();

        assert_eq!(visible_keys(&values, 0, 0, 3, 10), vec![4, 5, 6, 7]);
        assert_eq!(visible_keys(&values, 0, 1, 3, 10), vec![8]);
        assert_eq!(visible_keys(&values, 0, 2, 3, 10), vec![8, 9]);
    }

    #[test]
    fn full_causal_mask_keeps_the_later_prompt_prefix() {
        let rows = [FixedChunkMaskRow {
            query_start: 7,
            query_len: 3,
            key_start: 0,
            key_len: 10,
        }];
        let values = causal_mask_values(&rows, 3, 10, None).unwrap();

        assert_eq!(
            visible_keys(&values, 0, 1, 3, 10),
            (0..=8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn fixed_chunk_mask_uses_absolute_positions_for_ragged_rows() {
        let rows = [
            FixedChunkMaskRow {
                query_start: 3,
                query_len: 1,
                key_start: 0,
                key_len: 4,
            },
            FixedChunkMaskRow {
                query_start: 8,
                query_len: 1,
                key_start: 4,
                key_len: 5,
            },
        ];
        let values = fixed_chunk_mask_values(&rows, 1, 5, 4).unwrap();

        assert_eq!(visible_keys(&values, 0, 0, 1, 5), vec![0, 1, 2, 3]);
        assert_eq!(visible_keys(&values, 1, 0, 1, 5), vec![4]);
    }

    #[test]
    fn fixed_chunk_layout_uses_paged_prompt_and_decode_rows() {
        let mut prompt = PagedAttentionInputMetadata::dummy(&Device::Cpu).unwrap();
        prompt.query_lens = Some(vec![3, 1]);
        prompt.paged_context_lens_cpu = Some(vec![7, 3]);
        prompt.full_paged_context_lens_cpu = Some(vec![10, 9]);
        let (rows, key_width) = fixed_chunk_mask_layout(2, 3, &[7, 8], Some(&prompt)).unwrap();
        assert_eq!(
            rows,
            vec![
                FixedChunkMaskRow {
                    query_start: 7,
                    query_len: 3,
                    key_start: 3,
                    key_len: 7,
                },
                FixedChunkMaskRow {
                    query_start: 8,
                    query_len: 1,
                    key_start: 6,
                    key_len: 3,
                },
            ]
        );
        assert_eq!(key_width, 7);

        let mut decode = PagedAttentionInputMetadata::dummy(&Device::Cpu).unwrap();
        decode.paged_context_lens_cpu = Some(vec![4, 5, 3, 4]);
        decode.full_paged_context_lens_cpu = Some(vec![4, 9, 12, 13]);
        let (rows, key_width) = fixed_chunk_mask_layout(2, 2, &[2, 11], Some(&decode)).unwrap();
        assert_eq!(
            rows,
            vec![
                FixedChunkMaskRow {
                    query_start: 3,
                    query_len: 1,
                    key_start: 0,
                    key_len: 4,
                },
                FixedChunkMaskRow {
                    query_start: 8,
                    query_len: 1,
                    key_start: 4,
                    key_len: 5,
                },
                FixedChunkMaskRow {
                    query_start: 11,
                    query_len: 1,
                    key_start: 9,
                    key_len: 3,
                },
                FixedChunkMaskRow {
                    query_start: 12,
                    query_len: 1,
                    key_start: 9,
                    key_len: 4,
                },
            ]
        );
        assert_eq!(key_width, 5);
    }

    #[test]
    fn fixed_chunk_mask_rejects_invalid_metadata() {
        assert!(fixed_chunk_mask_values(
            &[FixedChunkMaskRow {
                query_start: 4,
                query_len: 1,
                key_start: 5,
                key_len: 1,
            }],
            1,
            1,
            4,
        )
        .is_err());

        let mut metadata = PagedAttentionInputMetadata::dummy(&Device::Cpu).unwrap();
        metadata.query_lens = Some(vec![3]);
        metadata.paged_context_lens_cpu = Some(vec![2]);
        metadata.full_paged_context_lens_cpu = Some(vec![2]);
        assert!(fixed_chunk_mask_layout(1, 3, &[0], Some(&metadata)).is_err());
    }

    #[test]
    fn packed_chunked_segments_restart_at_logical_row_boundaries() {
        assert_eq!(
            chunked_flash_segment_lens(&[3, 2], 2).unwrap(),
            vec![2, 1, 2]
        );
        let params = chunked_flash_params(&[3, 2], 2, &[Device::Cpu]).unwrap();
        assert_eq!(params.varlen_segment_lens, Some(vec![2, 1, 2]));
        assert_eq!(params.max_q, 2);
        assert_eq!(
            params.cumulative_seqlens_q[&Device::Cpu.location()]
                .to_vec1::<u32>()
                .unwrap(),
            vec![0, 2, 3, 5]
        );
    }

    #[test]
    fn packed_chunked_segments_reject_invalid_metadata() {
        assert!(chunked_flash_segment_lens(&[2, 0], 2).is_err());
        assert!(chunked_flash_segment_lens(&[2], 0).is_err());
    }
}
