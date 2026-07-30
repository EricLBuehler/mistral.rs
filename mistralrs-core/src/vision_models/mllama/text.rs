#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use std::{collections::HashMap, ops::Range, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use crate::{
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding_with_legacy_tied_uqff, CausalMasker, Llama3RotaryEmbedding, RmsNorm, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::PagedAttentionInputMetadata, EitherCache, IsqModel, KvCache,
        ModelForwardContext, NormalCache, NormalLoadingMetadata,
    },
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::MLlamaTextConfig;

struct MLlamaTextMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act: Activation,
}

impl MLlamaTextMlp {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down_proj: RowParallelLayer::new(
                cfg.intermediate_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let res = self.down_proj.forward(
            &self
                .act
                .forward(&self.gate_proj.forward(xs)?)?
                .broadcast_mul(&self.up_proj.forward(xs)?)?,
        )?;
        Ok(res)
    }
}

struct MLlamaTextSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    sdpa_params: SdpaParams,
    rope: Arc<Llama3RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    paged_attn: Option<PagedAttention>,
}

impl MLlamaTextSelfAttention {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        rope: Arc<Llama3RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("q_proj"),
            )?,
            k_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("k_proj"),
            )?,
            v_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("v_proj"),
            )?,
            o_proj: RowParallelLayer::new(
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("o_proj"),
            )?,
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            rope,
            num_heads: cfg.num_attention_heads / comm.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim,
            paged_attn,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &AttentionMask,
        ctx: &mut ModelForwardContext<'_>,
        kv_cache: &mut KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (bs, q_len, _) = hidden_states.dims3()?;

        let (q, k, v) = crate::ops::qkv_projections(
            hidden_states,
            &*self.q_proj,
            &*self.k_proj,
            &*self.v_proj,
        )?;
        let (q, k, mut v) = if q_len != 1 {
            let q = q
                .reshape((bs, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((bs, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((bs, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((bs, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
            .clone();
        let (q, mut k) = self.rope.forward(&q, &k, &positions)?;

        let metadata = ctx.paged_layer(layer_idx);
        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                )?,
                None => {
                    if matches!(attention_mask, AttentionMask::None) {
                        candle_core::bail!("Mllama paged self-attention is missing cache metadata");
                    }
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )?
                }
            },
            None => {
                (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    attention_mask,
                    Some(ctx.flash_params()),
                    &self.sdpa_params,
                )?
            }
        };
        attn_output = if matches!(attention_mask, AttentionMask::None) {
            attn_output.reshape((bs, q_len, ()))?
        } else {
            attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((bs, q_len, ()))?
        }
        .to_dtype(q.dtype())?;

        let res = self.o_proj.forward(&attn_output)?;
        Ok(res)
    }
}

struct MLlamaSelfAttentionDecoderLayer {
    attn: MLlamaTextSelfAttention,
    mlp: MLlamaTextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

struct MLlamaDecoderLayerLoad<'a> {
    cfg: &'a MLlamaTextConfig,
    mapper: &'a dyn DeviceMapper,
    layer_idx: usize,
    loading_isq: bool,
    comm: &'a Arc<mistralrs_quant::Comm>,
}

impl MLlamaSelfAttentionDecoderLayer {
    fn new(
        vb: ShardedVarBuilder,
        rope: Arc<Llama3RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        args: MLlamaDecoderLayerLoad<'_>,
    ) -> Result<Self> {
        let MLlamaDecoderLayerLoad {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = args;
        let mlp = MLlamaTextMlp::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            comm,
        )?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let attn = MLlamaTextSelfAttention::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            rope,
            paged_attn,
            comm,
        )?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &AttentionMask,
        ctx: &mut ModelForwardContext<'_>,
        kv_cache: &mut KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        let mut hidden_states = self.input_layernorm.forward(hidden_states)?;

        hidden_states =
            self.attn
                .forward(&hidden_states, attention_mask, ctx, kv_cache, layer_idx)?;
        hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let mut hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        hidden_states = self.mlp.forward(&hidden_states)?;

        residual + hidden_states
    }
}

struct MLlamaTextCrossAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

fn validate_packed_query_lens(
    query_lens: &[usize],
    logical_batch: usize,
    total_tokens: usize,
) -> Result<()> {
    if query_lens.len() != logical_batch || query_lens.is_empty() || query_lens.contains(&0) {
        candle_core::bail!("Mllama packed query lengths do not match the logical batch");
    }
    let query_tokens = query_lens.iter().try_fold(0usize, |total, &query_len| {
        total
            .checked_add(query_len)
            .ok_or_else(|| candle_core::Error::msg("Mllama packed query length overflow"))
    })?;
    if query_tokens != total_tokens {
        candle_core::bail!("Mllama packed query lengths do not cover the physical tokens");
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PackedCrossAttentionShape {
    physical_batch: usize,
    total_tokens: usize,
    state_batch: usize,
    state_tokens: usize,
    mask_shape: Option<(usize, usize, usize)>,
}

fn packed_cross_attention_ranges(
    shape: PackedCrossAttentionShape,
    query_lens: &[usize],
) -> Result<Vec<Range<usize>>> {
    if shape.physical_batch != 1 {
        candle_core::bail!("Mllama packed cross-attention requires physical batch size 1");
    }
    validate_packed_query_lens(query_lens, shape.state_batch, shape.total_tokens)?;
    if let Some((mask_batch, max_query_len, mask_tokens)) = shape.mask_shape {
        if mask_batch != shape.state_batch
            || mask_tokens != shape.state_tokens
            || query_lens.iter().any(|&len| len > max_query_len)
        {
            candle_core::bail!("Mllama packed cross-attention mask is inconsistent");
        }
    }

    let mut offset = 0usize;
    Ok(query_lens
        .iter()
        .map(|&query_len| {
            let range = offset..offset + query_len;
            offset += query_len;
            range
        })
        .collect())
}

fn pack_full_text_row_mask(mask: &Tensor, query_lens: &[usize]) -> Result<Tensor> {
    let (logical_batch, singleton, max_query_len, tail) = mask.dims4()?;
    if singleton != 1 || tail != 1 {
        candle_core::bail!("Mllama full-row mask has invalid dimensions");
    }
    validate_packed_query_lens(query_lens, logical_batch, query_lens.iter().sum::<usize>())?;
    if query_lens.iter().any(|&len| len > max_query_len) {
        candle_core::bail!("Mllama full-row mask is shorter than a logical query");
    }
    let mut rows = Vec::with_capacity(logical_batch);
    for (batch_idx, &query_len) in query_lens.iter().enumerate() {
        rows.push(
            mask.narrow(0, batch_idx, 1)?
                .narrow(2, 0, query_len)?
                .squeeze(1)?,
        );
    }
    Tensor::cat(&rows, 1)
}

impl MLlamaTextCrossAttention {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("q_proj"),
            )?,
            k_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("k_proj"),
            )?,
            v_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("v_proj"),
            )?,
            o_proj: RowParallelLayer::new(
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::new(
                cfg.head_dim(),
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("q_norm"), false),
            )?,
            k_norm: RmsNorm::new(
                cfg.head_dim(),
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("k_norm"), false),
            )?,
            num_heads: cfg.num_attention_heads / comm.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim: cfg.head_dim(),
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                softcap: None,
                softmax_scale: 1.0 / (cfg.head_dim() as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn project_cross_states(
        &self,
        cross_attn_states: &Tensor,
        query_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, state_tokens, _) = cross_attn_states.dims3()?;
        let (k, v) = if self.k_proj.is_dynamic_lora_active() || self.v_proj.is_dynamic_lora_active()
        {
            let mut keys = Vec::with_capacity(batch);
            let mut values = Vec::with_capacity(batch);
            for batch_idx in 0..batch {
                let route_row = batch_idx.checked_mul(query_len).ok_or_else(|| {
                    candle_core::Error::msg("Mllama cross-attention route row overflow")
                })?;
                let states = cross_attn_states.narrow(0, batch_idx, 1)?;
                let (k, v) = mistralrs_quant::with_lora_execution_repeated_row(
                    route_row,
                    state_tokens,
                    || Ok((self.k_proj.forward(&states)?, self.v_proj.forward(&states)?)),
                )?;
                keys.push(k);
                values.push(v);
            }
            (Tensor::cat(&keys, 0)?, Tensor::cat(&values, 0)?)
        } else {
            (
                self.k_proj.forward(cross_attn_states)?,
                self.v_proj.forward(cross_attn_states)?,
            )
        };

        let k = self.k_norm.forward(
            &k.reshape((batch, (), self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?,
        )?;
        let v = v
            .reshape((batch, (), self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        Ok((k, v))
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: Option<&Tensor>,
        attention_mask: &AttentionMask,
    ) -> Result<Tensor> {
        let (bs, q_len, _) = hidden_states.dims3()?;

        let mut q = self.q_proj.forward(hidden_states)?;
        q = q
            .reshape((bs, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        q = self.q_norm.forward(&q)?;

        let (k, v) = if let Some(cross_attn_states) = cross_attn_states {
            if cross_attn_states.dim(0)? != bs {
                candle_core::bail!(
                    "Mllama cross-attention state batch does not match the query batch"
                );
            }
            self.project_cross_states(cross_attn_states, q_len)?
        } else {
            candle_core::bail!("Cross attn cannot find k,v cache or cross attn hidden states!")
        };

        let repeated_mask = match attention_mask {
            AttentionMask::Custom(m) => {
                AttentionMask::Custom(m.repeat((1, self.num_heads, 1, 1))?)
            }
            other => other.clone(),
        };
        let attn_output = Sdpa
            .run_attention(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                &repeated_mask,
                None,
                &self.sdpa_params,
            )?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs, q_len, ()))?
            .to_dtype(q.dtype())?;

        let res = self.o_proj.forward(&attn_output)?;
        Ok(res)
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: &Tensor,
        attention_mask: &AttentionMask,
        query_lens: &[usize],
    ) -> Result<Tensor> {
        let (physical_batch, total_tokens, _) = hidden_states.dims3()?;
        let (logical_batch, state_tokens, _) = cross_attn_states.dims3()?;
        let mask_shape = match attention_mask {
            AttentionMask::Custom(mask) => Some(mask.dims3()?),
            _ => None,
        };
        let ranges = packed_cross_attention_ranges(
            PackedCrossAttentionShape {
                physical_batch,
                total_tokens,
                state_batch: logical_batch,
                state_tokens,
                mask_shape,
            },
            query_lens,
        )?;

        let mut outputs = Vec::with_capacity(logical_batch);
        for (batch_idx, range) in ranges.into_iter().enumerate() {
            let hidden = hidden_states.narrow(1, range.start, range.len())?;
            let states = cross_attn_states.narrow(0, batch_idx, 1)?;
            let mask = match attention_mask {
                AttentionMask::Custom(mask) => AttentionMask::Custom(
                    mask.narrow(0, batch_idx, 1)?.narrow(1, 0, range.len())?,
                ),
                other => other.clone(),
            };
            outputs.push(mistralrs_quant::with_lora_execution_row_range(
                range.clone(),
                || self.forward(&hidden, Some(&states), &mask),
            )?);
        }
        Tensor::cat(&outputs, 1)
    }
}

struct MLlamaCrossAttentionDecoderLayer {
    attn: MLlamaTextCrossAttention,
    attn_gate: Tensor,
    mlp: MLlamaTextMlp,
    mlp_gate: Tensor,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MLlamaCrossAttentionDecoderLayer {
    fn new(vb: ShardedVarBuilder, args: MLlamaDecoderLayerLoad<'_>) -> Result<Self> {
        let MLlamaDecoderLayerLoad {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = args;
        let mlp = MLlamaTextMlp::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            comm,
        )?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let attn = MLlamaTextCrossAttention::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("cross_attn"), loading_isq),
            mapper,
            layer_idx,
            comm,
        )?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            attn_gate: mapper
                .set_device(layer_idx, vb.clone(), false)
                .get((1,), "cross_attn_attn_gate")?,
            mlp_gate: mapper
                .set_device(layer_idx, vb.clone(), false)
                .get((1,), "cross_attn_mlp_gate")?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: Option<&Tensor>,
        attention_mask: &AttentionMask,
        full_text_row_masked_out_mask: Option<&Tensor>,
        packed_query_lens: Option<&[usize]>,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        let mut hidden_states = self.input_layernorm.forward(hidden_states)?;

        hidden_states = if let Some(query_lens) = packed_query_lens {
            self.attn.forward_packed(
                &hidden_states,
                cross_attn_states.ok_or_else(|| {
                    candle_core::Error::msg(
                        "Mllama packed cross-attention is missing encoder states",
                    )
                })?,
                attention_mask,
                query_lens,
            )?
        } else {
            self.attn
                .forward(&hidden_states, cross_attn_states, attention_mask)?
        };
        hidden_states = (residual + hidden_states.broadcast_mul(&self.attn_gate.tanh()?)?)?;

        let residual = &hidden_states;
        let mut hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        hidden_states = self.mlp.forward(&hidden_states)?;
        if let Some(full_text_row_masked_out_mask) = full_text_row_masked_out_mask {
            let full_text_row_masked_out_mask = match packed_query_lens {
                Some(query_lens) => {
                    pack_full_text_row_mask(full_text_row_masked_out_mask, query_lens)?
                }
                None => full_text_row_masked_out_mask.i((.., 0))?,
            };
            hidden_states = full_text_row_masked_out_mask
                .to_dtype(hidden_states.dtype())?
                .broadcast_mul(&hidden_states)?;
        }

        residual + hidden_states.broadcast_mul(&self.mlp_gate.tanh()?)?
    }
}

enum MLlamaDecoderLayer {
    CrossAttn(MLlamaCrossAttentionDecoderLayer),
    SelfAttn(MLlamaSelfAttentionDecoderLayer),
}

pub(super) struct MLlamaTextModel {
    embed_tokens: Arc<dyn QuantMethod>,
    lm_head: Arc<dyn QuantMethod>,
    norm: RmsNorm,
    layers: Vec<MLlamaDecoderLayer>,
    dtype: DType,
    pub(crate) cfg: ModelConfigMetadata,
    pub(crate) cache: EitherCache,
    pub(crate) device: Device,
    pub(crate) max_position_embeddings: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl MLlamaTextModel {
    pub(super) fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb.dtype();

        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size + 8,
            cfg.hidden_size,
            mapper.set_nm_device(
                vb.pp("model.embed_tokens"),
                normal_loading_metadata.loading_isq,
            ),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };

        let vb = vb.pp("model");

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb.pp("norm"), false),
        )?;

        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(Llama3RotaryEmbedding::new_mllama3(
                    vb.dtype(),
                    cfg,
                    device,
                    is_gptx,
                )?),
            );
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let comm = mapper.get_comm_for(i)?;
            if cfg.cross_attention_layers.contains(&i) {
                layers.push(MLlamaDecoderLayer::CrossAttn(
                    MLlamaCrossAttentionDecoderLayer::new(
                        vb.pp(format!("layers.{i}")),
                        MLlamaDecoderLayerLoad {
                            cfg,
                            mapper: &*mapper,
                            layer_idx: i,
                            loading_isq: false,
                            comm: &comm,
                        },
                    )?,
                ))
            } else {
                let device = mapper
                    .device_for(i, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                let paged_attn = match &attention_mechanism {
                    AttentionImplementation::Eager => None,
                    AttentionImplementation::PagedAttention => {
                        Some(PagedAttention::new(cfg.head_dim(), device, None)?)
                    }
                };
                layers.push(MLlamaDecoderLayer::SelfAttn(
                    MLlamaSelfAttentionDecoderLayer::new(
                        vb.pp(format!("layers.{i}")),
                        ropes
                            .get(&device.location())
                            .expect("No RoPE for device location!")
                            .clone(),
                        paged_attn,
                        MLlamaDecoderLayerLoad {
                            cfg,
                            mapper: &*mapper,
                            layer_idx: i,
                            loading_isq: normal_loading_metadata.loading_isq,
                            comm: &comm,
                        },
                    )?,
                ))
            }
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            device: normal_loading_metadata.real_device,
            max_position_embeddings: cfg.max_position_embeddings,
            mapper,
        })
    }

    pub(super) fn forward(
        &self,
        input_ids: &Tensor,
        cross_attn_states: Option<&Tensor>,
        cross_attention_mask: &AttentionMask,
        full_text_row_masked_out_mask: Option<&Tensor>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
        let packed_query_lens = if ctx.flash_params().packed {
            let (physical_batch, physical_tokens) = input_ids.dims2()?;
            if physical_batch != 1 {
                candle_core::bail!("packed Mllama forward requires physical batch size 1");
            }
            let query_lens = ctx
                .paged_input_metadata()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    candle_core::Error::msg(
                        "packed Mllama forward is missing logical query lengths",
                    )
                })?;
            validate_packed_query_lens(query_lens, query_lens.len(), physical_tokens)?;
            Some(query_lens.to_vec())
        } else {
            None
        };

        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let self_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache as &dyn PastKvLenCache,
            hidden_states.dtype(),
            &CausalMaskConfig::default(),
        )?;

        let self_mask = DeviceMappedMask::new(self_mask, &*self.mapper)?;
        let cross_attention_mask =
            DeviceMappedMask::new(cross_attention_mask.clone(), &*self.mapper)?;
        let full_text_row_masked_out_mask_mapped = DeviceMappedMask::new(
            match full_text_row_masked_out_mask {
                Some(t) => AttentionMask::Custom(t.clone()),
                None => AttentionMask::None,
            },
            &*self.mapper,
        )?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = self.mapper.map(hidden_states, i)?;
            match layer {
                MLlamaDecoderLayer::SelfAttn(attn) => {
                    hidden_states = attn.forward(
                        &hidden_states,
                        &self_mask.get(hidden_states.device()),
                        ctx,
                        &mut cache[i],
                        i,
                    )?;
                }
                MLlamaDecoderLayer::CrossAttn(attn) => {
                    // For text-only path we should skip cross attention layers.
                    // Let's check if the layer is cross attention layer and if we have cross attention states
                    // or cached cross attention states.
                    if cross_attn_states.is_none() {
                        continue;
                    }
                    let cross_mask = cross_attention_mask.get(hidden_states.device());
                    let ftrmom = full_text_row_masked_out_mask_mapped.get(hidden_states.device());
                    let cross_attn_states = cross_attn_states
                        .map(|states| states.to_device(hidden_states.device()))
                        .transpose()?;
                    hidden_states = attn.forward(
                        &hidden_states,
                        cross_attn_states.as_ref(),
                        &cross_mask,
                        ftrmom.as_option_tensor(),
                        packed_query_lens.as_deref(),
                    )?;
                }
            }
        }

        hidden_states = hidden_states.to_device(&self.device)?;
        hidden_states = self.norm.forward(&hidden_states)?;

        hidden_states = self.lm_head.forward(&ctx.logits(&hidden_states)?)?;

        Ok(hidden_states)
    }
}

impl IsqModel for MLlamaTextModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("model.embed_tokens").add(&self.embed_tokens);
        uvb.pp("lm_head").add(&self.lm_head);

        let uvb = uvb.pp("model");

        uvb.pp("norm").add(&self.norm);

        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("layers").pp(i);
            match layer {
                MLlamaDecoderLayer::CrossAttn(crossattn) => {
                    // Cross attention layers are not quantized
                    uvb_l
                        .pp("post_attention_layernorm")
                        .add(&crossattn.post_attention_layernorm);
                    uvb_l.pp("input_layernorm").add(&crossattn.input_layernorm);
                    uvb_l.add_tensor("cross_attn_attn_gate", crossattn.attn_gate.clone());
                    uvb_l.add_tensor("cross_attn_mlp_gate", crossattn.mlp_gate.clone());

                    let uvb_attn = uvb_l.pp("cross_attn");
                    uvb_attn.pp("q_proj").add(&crossattn.attn.q_proj);
                    uvb_attn.pp("k_proj").add(&crossattn.attn.k_proj);
                    uvb_attn.pp("v_proj").add(&crossattn.attn.v_proj);
                    uvb_attn.pp("o_proj").add(&crossattn.attn.o_proj);
                    uvb_attn.pp("q_norm").add(&crossattn.attn.q_norm);
                    uvb_attn.pp("k_norm").add(&crossattn.attn.k_norm);

                    let uvb_mlp = uvb_l.pp("mlp");
                    uvb_mlp.pp("gate_proj").add(&crossattn.mlp.gate_proj);
                    uvb_mlp.pp("up_proj").add(&crossattn.mlp.up_proj);
                    uvb_mlp.pp("down_proj").add(&crossattn.mlp.down_proj);
                }
                MLlamaDecoderLayer::SelfAttn(selfattn) => {
                    uvb_l
                        .pp("post_attention_layernorm")
                        .add(&selfattn.post_attention_layernorm);
                    uvb_l.pp("input_layernorm").add(&selfattn.input_layernorm);
                }
            }
        }

        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pack_full_text_row_mask, packed_cross_attention_ranges, validate_packed_query_lens,
        PackedCrossAttentionShape,
    };
    use candle_core::{Device, Tensor};

    #[test]
    fn packed_query_lengths_require_an_exact_partition() {
        validate_packed_query_lens(&[2, 3], 2, 5).unwrap();
        assert!(validate_packed_query_lens(&[2, 3], 1, 5).is_err());
        assert!(validate_packed_query_lens(&[2, 0], 2, 2).is_err());
        assert!(validate_packed_query_lens(&[2, 2], 2, 5).is_err());
        assert!(validate_packed_query_lens(&[usize::MAX, 1], 2, usize::MAX).is_err());
    }

    #[test]
    fn packed_cross_attention_maps_unequal_queries_to_encoder_rows() {
        let ranges = packed_cross_attention_ranges(
            PackedCrossAttentionShape {
                physical_batch: 1,
                total_tokens: 5,
                state_batch: 2,
                state_tokens: 8,
                mask_shape: Some((2, 3, 8)),
            },
            &[2, 3],
        )
        .unwrap();

        assert_eq!(ranges, vec![0..2, 2..5]);
    }

    #[test]
    fn packed_cross_attention_rejects_state_and_mask_cardinality_mismatches() {
        let shape = PackedCrossAttentionShape {
            physical_batch: 1,
            total_tokens: 5,
            state_batch: 2,
            state_tokens: 8,
            mask_shape: Some((2, 3, 8)),
        };
        let mut wrong_state_batch = shape;
        wrong_state_batch.state_batch = 1;
        assert!(packed_cross_attention_ranges(wrong_state_batch, &[2, 3]).is_err());

        let mut wrong_mask_batch = shape;
        wrong_mask_batch.mask_shape = Some((1, 3, 8));
        assert!(packed_cross_attention_ranges(wrong_mask_batch, &[2, 3]).is_err());

        let mut wrong_mask_tokens = shape;
        wrong_mask_tokens.mask_shape = Some((2, 3, 7));
        assert!(packed_cross_attention_ranges(wrong_mask_tokens, &[2, 3]).is_err());
    }

    #[test]
    fn full_row_mask_packs_only_live_query_rows() {
        let mask = Tensor::from_vec(vec![1u32, 1, 0, 0, 1, 1], (2, 1, 3, 1), &Device::Cpu).unwrap();
        let packed = pack_full_text_row_mask(&mask, &[2, 3]).unwrap();

        assert_eq!(packed.dims(), &[1, 5, 1]);
        assert_eq!(
            packed.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![1, 1, 0, 1, 1]
        );
    }
}
