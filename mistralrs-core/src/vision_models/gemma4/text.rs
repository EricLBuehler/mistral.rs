#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Embedding;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantMethodConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder, UnquantLinear,
};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        embedding, Activation, CausalMasker, GemmaRmsNorm, MatMul, Mlp, RmsNorm, RotaryEmbedding,
        ScaledEmbedding, Sdpa,
    },
    layers_masker::PastKvLenCache,
    moe::{MoEExperts, MoEExpertsConfig},
    ops::TopKLastDimOp,
    paged_attention::{
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata, PagedAttention,
    },
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalCacheType, NormalLoadingMetadata,
        VisionModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::Gemma4TextConfig;

macro_rules! is_sliding {
    ($layer_idx:expr, $cfg:expr) => {{
        let is_last = $layer_idx == $cfg.num_hidden_layers - 1;
        !is_last && ($layer_idx + 1) % $cfg.sliding_window_pattern != 0
    }};
}

fn first_kv_shared_layer_idx(cfg: &Gemma4TextConfig) -> usize {
    cfg.num_hidden_layers
        .saturating_sub(cfg.num_kv_shared_layers)
}

fn kv_shared_layer_index(cfg: &Gemma4TextConfig, layer_idx: usize) -> Result<Option<usize>> {
    let first_kv_shared_layer_idx = first_kv_shared_layer_idx(cfg);
    if first_kv_shared_layer_idx == 0 || layer_idx < first_kv_shared_layer_idx {
        return Ok(None);
    }

    let attention_type = &cfg.layer_types[layer_idx];
    cfg.layer_types[..first_kv_shared_layer_idx]
        .iter()
        .rposition(|ty| ty == attention_type)
        .map(Some)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Gemma4 layer {layer_idx} is configured to share KV without a prior `{attention_type}` donor layer."
            ))
        })
}

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

/// Proportional RoPE for Gemma4 full-attention layers.
///
/// Unlike standard RotaryEmbedding, this computes inv_freq for only the first
/// `rope_angles` dimensions but uses `head_dim` as the denominator, then pads
/// with zeros. The result: cos=1, sin=0 for non-rotated positions, so the
/// standard rotary formula `x*cos + rotate_half(x)*sin` acts as identity
/// for those dims.
struct ProportionalRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    is_gpt_neox: bool,
}

impl ProportionalRotaryEmbedding {
    fn new(
        base: f32,
        head_dim: usize,
        partial_rotary_factor: f64,
        max_position_embeddings: usize,
        device: &Device,
        is_gpt_neox: bool,
        dtype: DType,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        // Compute inv_freq for rotated dimensions using head_dim as denominator
        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / base.powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions
        for _ in rope_angles..half_dim {
            inv_freq_vec.push(0f32);
        }

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        // freqs shape: [max_pos, half_dim]
        // candle's rope function expects cos/sin of shape [seq, half_dim]
        // (it handles the duplication internally)
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            is_gpt_neox,
        })
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _qh, seq_len, _n_embd) = q.dims4()?;

        let rope = if self.is_gpt_neox {
            candle_nn::rotary_emb::rope
        } else {
            candle_nn::rotary_emb::rope_i
        };

        if seqlen_offsets.len() == 1 {
            let cos = self.cos.narrow(0, seqlen_offsets[0], seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offsets[0], seq_len)?;
            let q_embed = rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (seq_idx, offset) in seqlen_offsets.iter().enumerate() {
                let cos = self.cos.narrow(0, *offset, seq_len)?;
                let sin = self.sin.narrow(0, *offset, seq_len)?;
                let q_s = q.i(seq_idx)?;
                let k_s = k.i(seq_idx)?;
                let q_embed = rope(&q_s.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                let k_embed = rope(&k_s.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Router
// ────────────────────────────────────────────────────────────────────────────

struct Gemma4Router {
    scale: Tensor,
    proj: candle_nn::Linear,
    hidden_size: usize,
    top_k: usize,
    eps: f64,
}

impl Gemma4Router {
    fn new(
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        eps: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let scale = vb.get(hidden_size, "scale")?;
        let proj_w = vb.pp("proj").get((num_experts, hidden_size), "weight")?;
        let proj = candle_nn::Linear::new(proj_w, None);
        Ok(Self {
            scale,
            proj,
            hidden_size,
            top_k,
            eps,
        })
    }

    /// Returns (topk_weights, topk_ids) both of shape [num_tokens, top_k].
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let rms = (xs_f32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        let normed = xs_f32.broadcast_div(&rms)?;

        let root_size = (self.hidden_size as f64).powf(-0.5);
        let scaled = (normed * root_size)?;
        let scale_f32 = self
            .scale
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let scaled = scaled.broadcast_mul(&scale_f32)?;

        let logits = scaled
            .to_dtype(self.proj.weight().dtype())?
            .apply(&self.proj)?;
        let logits_f32 = logits.to_dtype(DType::F32)?;
        // Immediate ISQ can occasionally produce NaN router rows for the 26b
        // MoE path. Sanitize them before top-k so we never emit invalid expert
        // ids such as u32::MAX from the CUDA top-k kernel.
        let finite_mask = logits_f32.eq(&logits_f32)?;
        let logits_f32 = finite_mask.where_cond(&logits_f32, &Tensor::zeros_like(&logits_f32)?)?;
        let logits_f32 = logits_f32.clamp(-1e4, 1e4)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?;

        // Select top-k experts by SCORE (logits), not by probability
        // This matches HF which does topk(expert_scores), then masks probs
        let topk = logits_f32.topk(self.top_k)?;
        let topk_indices = topk.indices; // [batch, seq, top_k]

        // Gather the softmax probabilities for the selected experts
        let topk_probs = probs.gather(&topk_indices, D::Minus1)?;

        // Renormalize: divide by sum of selected probs
        let renorm = topk_probs.sum_keepdim(D::Minus1)?;
        let renorm = renorm.clamp(1e-6, f64::INFINITY)?;
        let topk_weights = topk_probs.broadcast_div(&renorm)?;

        Ok((topk_weights, topk_indices))
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Attention
// ────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Option<Arc<dyn QuantMethod>>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    is_sliding: bool,
    is_k_eq_v: bool,
    partial_rotary_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    kv_shared_layer_index: Option<usize>,
    rms_norm_eps: f64,
    layer_idx: usize,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        mapper: &dyn DeviceMapper,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
        kv_shared_layer_index: Option<usize>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let sliding = is_sliding!(layer_idx, cfg);

        let (head_dim, num_kv_heads) = if sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = cfg
                .num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads);
            (cfg.global_head_dim, global_kv)
        };

        let partial_rotary_dim = if sliding {
            head_dim
        } else {
            (cfg.global_head_dim as f64 * cfg.partial_rotary_factor()) as usize
        };

        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm);
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;

        let is_k_eq_v = !sliding && cfg.attention_k_eq_v;
        let v_proj = if is_k_eq_v {
            None
        } else {
            Some(ColumnParallelLayer::new_with_shard(
                hidden_sz,
                num_kv_heads * head_dim,
                &cfg.quantization_config,
                bias,
                comm,
                kv_shard,
                vb.pp("v_proj"),
            )?)
        };

        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("o_proj"),
        )?;

        let sliding_window = if sliding {
            Some(cfg.effective_sliding_window())
        } else {
            None
        };

        let q_norm = GemmaRmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = GemmaRmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("k_norm"), false),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb_global,
            rotary_emb_local,
            is_sliding: sliding,
            is_k_eq_v,
            partial_rotary_dim,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    num_kv_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0,
                sliding_window,
                sinks: None,
            },
            q_norm,
            k_norm,
            kv_shared_layer_index,
            rms_norm_eps: cfg.rms_norm_eps,
            layer_idx,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_caches: &mut [KvCache],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs_proj = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs_proj = xs_proj.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs_proj, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&xs_proj, &*self.k_proj)?;
        let mut v = if let Some(ref v_proj) = self.v_proj {
            MatMul.qmethod_matmul(&xs_proj, &**v_proj)?
        } else {
            // K=V: clone k before norms/RoPE
            k.clone()
        };
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        // Apply Q/K norms
        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;
        // V norm (RMS without learnable weight)
        v = v_norm(&v, self.rms_norm_eps)?;

        // Apply RoPE
        if self.is_sliding {
            let (q_rot, k_rot) = self.rotary_emb_local.forward(&q, &k, seqlen_offsets)?;
            q = q_rot;
            k = k_rot;
        } else {
            // ProportionalRotaryEmbedding handles the full head_dim with zero-padded
            // inv_freq, cos=1, sin=0 for non-rotated positions, so identity.
            let (q_rot, k_rot) = self.rotary_emb_global.forward(&q, &k, seqlen_offsets)?;
            q = q_rot;
            k = k_rot;
        }

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => {
                // Paged attention path, do NOT use normal kv_caches.
                let mask = if self.is_sliding {
                    sliding_attention_mask
                } else {
                    attention_mask
                };
                let paged_mask = if flash_params.is_some() {
                    attention_mask
                } else {
                    mask
                };

                let is_shared = self.kv_shared_layer_index.is_some();

                match metadata {
                    Some(((key_cache, value_cache), input_metadata)) => {
                        if is_shared {
                            // Shared layer: key_cache/value_cache point to the DONOR's paged cache (set up in forward_embeds).
                            // Read from donor's cache without writing.
                            paged_attn.forward_donor_cache(
                                &q,
                                &key_cache,
                                &value_cache,
                                paged_mask,
                                input_metadata,
                                &self.sdpa_params,
                                flash_params,
                            )?
                        } else {
                            // Non-shared: standard paged attention with raw k,v.
                            paged_attn.forward(
                                &q,
                                &k,
                                &v,
                                paged_mask,
                                Some(key_cache),
                                Some(value_cache),
                                input_metadata,
                                &self.sdpa_params,
                                flash_params,
                            )?
                        }
                    }
                    None => {
                        let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                        assert!(paged_mask.is_some());
                        paged_attn.forward(
                            &q,
                            &k,
                            &v,
                            paged_mask,
                            None,
                            None,
                            &input_metadata,
                            &self.sdpa_params,
                            flash_params,
                        )?
                    }
                }
            }
            None => {
                // Eager attention path, use normal kv_caches.
                let (k, v, is_shared_kv) = if let Some(donor_idx) = self.kv_shared_layer_index {
                    let donor_cache = &kv_caches[donor_idx];
                    let dk = donor_cache.k()?.unwrap().to_device(q.device())?;
                    let dv = donor_cache.v()?.unwrap().to_device(q.device())?;
                    (dk, dv, true)
                } else {
                    let (k, v) = kv_caches[self.layer_idx].append(&k, &v)?;
                    (k, v, false)
                };

                let mask = if self.is_sliding {
                    sliding_attention_mask
                } else {
                    attention_mask
                };

                // Adjust mask dimensions if using shared KV cache (donor may
                // have different seq len)
                let mask = if is_shared_kv {
                    let adjust_mask = |m: Option<&Tensor>| -> Result<Option<Tensor>> {
                        if let Some(mask) = m {
                            let kv_seq_len = k.dims()[2];
                            let mask_dims = mask.dims();
                            match mask.rank() {
                                2 if mask_dims[1] > kv_seq_len => {
                                    Ok(Some(mask.narrow(1, 0, kv_seq_len)?))
                                }
                                3 if mask_dims[2] > kv_seq_len => {
                                    Ok(Some(mask.narrow(2, 0, kv_seq_len)?))
                                }
                                4 if mask_dims[3] > kv_seq_len => {
                                    Ok(Some(mask.narrow(3, 0, kv_seq_len)?))
                                }
                                _ => Ok(Some(mask.clone())),
                            }
                        } else {
                            Ok(None)
                        }
                    };
                    adjust_mask(mask)?
                } else {
                    mask.cloned()
                };

                Sdpa.run_attention(&q, &k, &v, mask.as_ref(), flash_params, &self.sdpa_params)?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        let has_mask = attention_mask.is_some() || sliding_attention_mask.is_some();
        attn_output = if has_mask {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Decoder layer
// ────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn crate::amoe::MlpLayer>,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
    // MoE
    moe_block: Option<MoEExperts>,
    per_expert_scale: Option<Tensor>,
    router: Option<Gemma4Router>,
    pre_feedforward_layernorm_2: Option<GemmaRmsNorm>,
    post_feedforward_layernorm_1: Option<GemmaRmsNorm>,
    post_feedforward_layernorm_2: Option<GemmaRmsNorm>,
    // PLE
    per_layer_input_gate: Option<Arc<dyn QuantMethod>>,
    per_layer_projection: Option<Arc<dyn QuantMethod>>,
    post_per_layer_input_norm: Option<GemmaRmsNorm>,
    // Layer scalar
    layer_scalar: Option<Tensor>,
    act: Activation,
    layer_idx: usize,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
        kv_shared_layer_index: Option<usize>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            mapper,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
            kv_shared_layer_index,
        )?;

        let mlp_intermediate = if cfg.use_double_wide_mlp && kv_shared_layer_index.is_some() {
            cfg.intermediate_size * 2
        } else {
            cfg.intermediate_size
        };
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            mlp_intermediate,
            &cfg.quantization_config,
            cfg.hidden_activation,
            comm,
        )?;

        let input_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let pre_feedforward_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm"), false),
        )?;
        let post_feedforward_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_feedforward_layernorm"), false),
        )?;

        // MoE components
        let (
            moe_block,
            per_expert_scale,
            router,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
        ) = if cfg.enable_moe_block {
            let num_experts = cfg.num_experts.unwrap_or(128);
            let top_k = cfg.top_k_experts.unwrap_or(2);
            let expert_inter = cfg
                .expert_intermediate_size
                .unwrap_or(cfg.intermediate_size);

            let moe_vb = mapper.set_device(layer_idx, vb.pp("moe"), false);
            let moe_cfg = MoEExpertsConfig {
                num_experts,
                num_experts_per_tok: top_k,
                hidden_size: cfg.hidden_size,
                moe_intermediate_size: expert_inter,
            };
            let moe = MoEExperts::new_direct(
                &moe_cfg,
                moe_vb.clone(),
                comm,
                loading_isq,
                &cfg.quantization_config,
                cfg.hidden_activation,
            )?;
            let scale = moe_vb.get(num_experts, "per_expert_scale")?;
            let rtr = Gemma4Router::new(
                cfg.hidden_size,
                num_experts,
                top_k,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("router"), false),
            )?;
            let pre_ff_2 = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm_2"), false),
            )?;
            let post_ff_1 = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("post_feedforward_layernorm_1"), false),
            )?;
            let post_ff_2 = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("post_feedforward_layernorm_2"), false),
            )?;
            (
                Some(moe),
                Some(scale),
                Some(rtr),
                Some(pre_ff_2),
                Some(post_ff_1),
                Some(post_ff_2),
            )
        } else {
            (None, None, None, None, None, None)
        };

        // PLE per-layer components
        let ple_dim = cfg.hidden_size_per_layer_input.unwrap_or(0);
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) = if ple_dim > 0
        {
            let gate = mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                ple_dim,
                &None,
                mapper.set_device(layer_idx, vb.pp("per_layer_input_gate"), loading_isq),
            )?;
            let proj = mistralrs_quant::linear_no_bias(
                ple_dim,
                cfg.hidden_size,
                &None,
                mapper.set_device(layer_idx, vb.pp("per_layer_projection"), loading_isq),
            )?;
            let norm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("post_per_layer_input_norm"), false),
            )?;
            (Some(gate), Some(proj), Some(norm))
        } else {
            (None, None, None)
        };

        let layer_scalar = vb.get((1,), "layer_scalar").ok();

        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            moe_block,
            per_expert_scale,
            router,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar,
            act: cfg.hidden_activation,
            layer_idx,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_caches: &mut [KvCache],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();

        let residual = xs.clone();
        let normed = self.input_layernorm.forward(&xs)?;
        let attn_out = self
            .self_attn
            .forward(
                &normed,
                attention_mask,
                sliding_attention_mask,
                seqlen_offsets,
                kv_caches,
                metadata,
                flash_params,
            )?
            .apply(&self.post_attention_layernorm)?;
        xs = (attn_out + &residual)?;

        // Feedforward
        let residual = xs.clone();

        if let (Some(ref moe), Some(ref per_expert_scale), Some(ref router)) =
            (&self.moe_block, &self.per_expert_scale, &self.router)
        {
            // MoE path: parallel MLP + MoE with separate norms
            let post_ff_1 = self
                .post_feedforward_layernorm_1
                .as_ref()
                .expect("post_feedforward_layernorm_1 required for MoE");
            let post_ff_2 = self
                .post_feedforward_layernorm_2
                .as_ref()
                .expect("post_feedforward_layernorm_2 required for MoE");
            let pre_ff_2 = self
                .pre_feedforward_layernorm_2
                .as_ref()
                .expect("pre_feedforward_layernorm_2 required for MoE");

            // Branch 1: MLP with pre_feedforward_layernorm → post_feedforward_layernorm_1
            let mlp_out = self
                .mlp
                .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?;
            let mlp_normed = mlp_out.apply(post_ff_1)?;

            // Branch 2: MoE with pre_feedforward_layernorm_2 → post_feedforward_layernorm_2
            let (topk_weights, topk_ids) = router.forward(&xs)?;

            // Fold per_expert_scale into routing weights
            let scales = per_expert_scale
                .to_dtype(DType::F32)?
                .index_select(&topk_ids.flatten_all()?.to_dtype(DType::U32)?, 0)?
                .reshape(topk_ids.shape())?;
            let topk_weights = (topk_weights.to_dtype(DType::F32)? * scales)?;

            let moe_input = xs.apply(pre_ff_2)?;

            let (b, s, _) = moe_input.dims3()?;
            let topk_weights = topk_weights.reshape((b * s, ()))?;
            let topk_ids = topk_ids.reshape((b * s, ()))?.to_dtype(DType::U32)?;

            let moe_result = moe.forward(&moe_input, topk_weights, &topk_ids)?;
            let moe_normed = moe_result.apply(post_ff_2)?;

            // Combine branches, then apply post_feedforward_layernorm
            let combined = (mlp_normed + moe_normed)?;
            let combined = combined.apply(&self.post_feedforward_layernorm)?;
            xs = (&residual + combined)?;
        } else {
            // Dense path: MLP only
            let mlp_out = self
                .mlp
                .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?
                .apply(&self.post_feedforward_layernorm)?;
            xs = (&residual + mlp_out)?;
        };

        // PLE: per-layer embedding injection (after feedforward, before layer scalar)
        if let (Some(ref gate), Some(ref proj), Some(ref norm)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
        ) {
            if let Some(pli) = per_layer_input {
                let residual_ple = xs.clone();
                let original_dtype = xs.dtype();
                // gate: Linear(hidden_size -> ple_dim)
                let mut gate_in = xs;
                if let Some(t) = gate.quantized_act_type() {
                    gate_in = gate_in.to_dtype(t)?;
                }
                let mut gated = MatMul.qmethod_matmul(&gate_in, &**gate)?;
                if gate.quantized_act_type().is_some() {
                    gated = gated.to_dtype(original_dtype)?;
                }
                // activation + elementwise multiply with per_layer_input
                gated = gated.apply(&self.act)?;
                gated = (gated * pli)?;
                // projection: Linear(ple_dim -> hidden_size)
                if let Some(t) = proj.quantized_act_type() {
                    gated = gated.to_dtype(t)?;
                }
                let mut projected = MatMul.qmethod_matmul(&gated, &**proj)?;
                if proj.quantized_act_type().is_some() {
                    projected = projected.to_dtype(original_dtype)?;
                }
                // post-norm + residual
                let normed = norm.forward(&projected)?;
                xs = (residual_ple + normed)?;
            }
        }

        // Apply layer scalar
        if let Some(ref scalar) = self.layer_scalar {
            xs = xs.broadcast_mul(scalar)?;
        }

        Ok(xs)
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  TextModel
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Gemma4ModelConfigLike {
    base: ModelConfigMetadata,
    per_layer_num_kv_heads: Vec<usize>,
    per_layer_k_head_dim: Vec<usize>,
    per_layer_v_head_dim: Vec<usize>,
    per_layer_uses_own_kv_cache: Vec<bool>,
}

impl ModelConfigLike for Gemma4ModelConfigLike {
    fn max_seq_len(&self) -> usize {
        self.base.max_seq_len
    }

    fn num_layers(&self) -> usize {
        self.base.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.base.hidden_size
    }

    fn num_kv_heads(&self) -> usize {
        self.base.num_kv_heads
    }

    fn num_attn_heads(&self) -> usize {
        self.base.num_attn_heads
    }

    fn k_head_dim(&self) -> usize {
        self.base.k_head_dim
    }

    fn v_head_dim(&self) -> usize {
        self.base.v_head_dim
    }

    fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        self.per_layer_num_kv_heads
            .get(layer_idx)
            .copied()
            .unwrap_or(self.base.num_kv_heads)
    }

    fn k_head_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.per_layer_k_head_dim
            .get(layer_idx)
            .copied()
            .unwrap_or(self.base.k_head_dim)
    }

    fn v_head_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.per_layer_v_head_dim
            .get(layer_idx)
            .copied()
            .unwrap_or(self.base.v_head_dim)
    }

    fn uses_own_kv_cache_for_layer(&self, layer_idx: usize) -> bool {
        self.per_layer_uses_own_kv_cache
            .get(layer_idx)
            .copied()
            .unwrap_or(true)
    }

    fn kv_cache_layout(&self) -> crate::paged_attention::KvCacheLayout {
        self.base.kv_cache_layout
    }

    fn kv_cache_elements_per_token(&self) -> usize {
        let num_layers = self.base.num_layers;
        let total: usize = (0..num_layers)
            .map(|i| {
                let kv_heads = self.num_kv_heads_for_layer(i);
                let k_dim = self.k_head_dim_for_layer(i);
                let v_dim = self.v_head_dim_for_layer(i);
                kv_heads * (k_dim + v_dim)
            })
            .sum();
        total / num_layers
    }
}

#[allow(dead_code)]
pub struct TextModel {
    embed_tokens: ScaledEmbedding,
    layers: Vec<DecoderLayer>,
    norm: GemmaRmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    lm_head_is_tied: bool,
    // PLE global
    embed_tokens_per_layer: Option<Embedding>,
    per_layer_model_projection: Option<Arc<dyn QuantMethod>>,
    per_layer_projection_norm: Option<RmsNorm>,
    hidden_size_per_layer_input: usize,
    num_hidden_layers: usize,
    vocab_size_per_layer_input: usize,
    per_layer_input_scale: f64,
    per_layer_projection_scalar: f64,
    // Standard
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: usize,
    final_logit_softcapping: Option<f64>,
    image_token_id: Option<usize>,
    use_bidirectional_vision_attention: bool,
    cfg: ModelConfigMetadata,
    model_config: Arc<dyn ModelConfigLike + Send + Sync>,
}

impl TextModel {
    pub fn new(
        cfg: &Gemma4TextConfig,
        image_token_id: Option<usize>,
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

        let vb_m = vb;
        let embed_tokens = ScaledEmbedding::new(
            (cfg.hidden_size as f64).sqrt(),
            embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
                &cfg.quantization_config,
            )?,
        );

        // Build RoPE instances per device
        let _partial_rotary_dim =
            (cfg.global_head_dim as f64 * cfg.partial_rotary_factor()) as usize;

        let mut global_ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            global_ropes.entry(device.location()).or_insert_with(|| {
                Arc::new(
                    ProportionalRotaryEmbedding::new(
                        cfg.rope_theta as f32,
                        cfg.global_head_dim,
                        cfg.partial_rotary_factor(),
                        cfg.max_position_embeddings,
                        device,
                        is_gptx,
                        vb_m.dtype(),
                    )
                    .expect("Failed to create global ProportionalRotaryEmbedding"),
                )
            });
        }

        let mut local_ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            local_ropes.entry(device.location()).or_insert_with(|| {
                Arc::new(
                    RotaryEmbedding::new(
                        cfg.rope_local_base_freq() as f32,
                        cfg.head_dim,
                        cfg.max_position_embeddings,
                        device,
                        is_gptx,
                        vb_m.dtype(),
                    )
                    .expect("Failed to create local RotaryEmbedding"),
                )
            });
        }
        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb_global = global_ropes
                .get(&device.location())
                .expect("No global RoPE for device location!")
                .clone();
            let rotary_emb_local = local_ropes
                .get(&device.location())
                .expect("No local RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    let hd = if is_sliding!(layer_idx, cfg) {
                        cfg.head_dim
                    } else {
                        cfg.global_head_dim
                    };
                    Some(PagedAttention::new(hd, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            let kv_shared_layer_index = kv_shared_layer_index(cfg, layer_idx)?;

            DecoderLayer::new(
                rotary_emb_global,
                rotary_emb_local,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
                kv_shared_layer_index,
            )
        })?;

        let norm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_m.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            // Keep Gemma 4's tied output projection in BF16. Quantizing the
            // enormous tied lm_head destabilizes low-bit decoding first, which is
            // especially visible around rare control tokens.
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                candle_nn::Linear::new(
                    mapper.cast_nm_device(
                        embed_tokens.embeddings(),
                        normal_loading_metadata.loading_isq,
                    )?,
                    None,
                ),
            ))?)
        };

        // PLE global components
        let ple_dim = cfg.hidden_size_per_layer_input.unwrap_or(0);
        let ple_vocab = cfg.vocab_size_per_layer_input.unwrap_or(cfg.vocab_size);
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if ple_dim > 0 {
                let ple_emb_vb = mapper.set_nm_device(vb_m.pp("embed_tokens_per_layer"), false);
                let ple_emb_weight =
                    ple_emb_vb.get((ple_vocab, cfg.num_hidden_layers * ple_dim), "weight")?;
                let ple_emb = Embedding::new(ple_emb_weight, cfg.num_hidden_layers * ple_dim);

                let ple_proj = mistralrs_quant::linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_hidden_layers * ple_dim,
                    &None,
                    mapper.set_nm_device(
                        vb_m.pp("per_layer_model_projection"),
                        normal_loading_metadata.loading_isq,
                    ),
                )?;

                let ple_norm = RmsNorm::new(
                    ple_dim,
                    cfg.rms_norm_eps,
                    mapper.set_nm_device(vb_m.pp("per_layer_projection_norm"), false),
                )?;

                (Some(ple_emb), Some(ple_proj), Some(ple_norm))
            } else {
                (None, None, None)
            };

        let mut per_layer_num_kv_heads = Vec::with_capacity(cfg.num_hidden_layers);
        let mut per_layer_k_head_dim = Vec::with_capacity(cfg.num_hidden_layers);
        let mut per_layer_v_head_dim = Vec::with_capacity(cfg.num_hidden_layers);
        let mut per_layer_uses_own_kv_cache = Vec::with_capacity(cfg.num_hidden_layers);
        let cache_types = (0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                let world_size = mapper.get_comm_for(layer_idx)?.world_size();
                let is_sliding = is_sliding!(layer_idx, cfg);
                let head_dim = if is_sliding {
                    cfg.head_dim
                } else {
                    cfg.global_head_dim
                };
                let num_kv_heads = if is_sliding {
                    cfg.num_key_value_heads
                } else {
                    cfg.num_global_key_value_heads
                        .unwrap_or(cfg.num_key_value_heads)
                };

                per_layer_num_kv_heads.push((num_kv_heads / world_size).max(1));
                per_layer_k_head_dim.push(head_dim);
                per_layer_v_head_dim.push(head_dim);

                if let Some(owner) = kv_shared_layer_index(cfg, layer_idx)? {
                    per_layer_uses_own_kv_cache.push(false);
                    Ok(NormalCacheType::Shared { owner })
                } else if is_sliding {
                    per_layer_uses_own_kv_cache.push(true);
                    Ok(NormalCacheType::SlidingWindow {
                        window: cfg.effective_sliding_window(),
                    })
                } else {
                    per_layer_uses_own_kv_cache.push(true);
                    Ok(NormalCacheType::Normal {
                        max_seq_len: cfg.max_position_embeddings,
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let cfg_metadata = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size()).max(1),
            sliding_window: Some(cfg.effective_sliding_window()),
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        let model_config: Arc<dyn ModelConfigLike + Send + Sync> =
            Arc::new(Gemma4ModelConfigLike {
                base: cfg_metadata.clone(),
                per_layer_num_kv_heads,
                per_layer_k_head_dim,
                per_layer_v_head_dim,
                per_layer_uses_own_kv_cache,
            });

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_is_tied: cfg.tie_word_embeddings,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            hidden_size_per_layer_input: ple_dim,
            num_hidden_layers: cfg.num_hidden_layers,
            vocab_size_per_layer_input: ple_vocab,
            per_layer_input_scale: 2f64.powf(-0.5),
            per_layer_projection_scalar: (cfg.hidden_size as f64).powf(-0.5),
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: cfg.max_position_embeddings,
            sliding_window: cfg.effective_sliding_window(),
            final_logit_softcapping: cfg.final_logit_softcapping,
            image_token_id,
            use_bidirectional_vision_attention: matches!(
                cfg.use_bidirectional_attention.as_deref(),
                Some("vision")
            ),
            cfg: cfg_metadata,
            model_config,
            mapper,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn model_config_like(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.model_config.clone()
    }

    /// Compute PLE per-layer inputs from both token embeddings and hidden state projection.
    /// Returns a Vec of per-layer tensors of shape [batch, seq, ple_dim].
    fn compute_ple(
        &self,
        ple_input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Option<Vec<Tensor>>> {
        if self.hidden_size_per_layer_input == 0 {
            return Ok(None);
        }

        let ple_emb = self
            .embed_tokens_per_layer
            .as_ref()
            .expect("PLE embedding missing");
        let ple_proj = self
            .per_layer_model_projection
            .as_ref()
            .expect("PLE projection missing");
        let ple_norm = self
            .per_layer_projection_norm
            .as_ref()
            .expect("PLE norm missing");

        let ple_dim = self.hidden_size_per_layer_input;
        let (b, seq, _) = inputs_embeds.dims3()?;

        // 1. Token-level per-layer embeddings: [b, seq, num_layers * ple_dim]
        let embedded = ple_emb.forward(ple_input_ids)?;
        // Scale by sqrt(ple_dim)
        let embedded = (embedded * (ple_dim as f64).sqrt())?;
        // Reshape to [b, seq, num_layers, ple_dim]
        let embedded = embedded.reshape((b, seq, self.num_hidden_layers, ple_dim))?;

        // 2. Project input embeddings: Linear(hidden_size -> num_layers * ple_dim)
        let original_dtype = inputs_embeds.dtype();
        let mut proj_input = inputs_embeds.clone();
        if let Some(t) = ple_proj.quantized_act_type() {
            proj_input = proj_input.to_dtype(t)?;
        }
        let mut projected = MatMul.qmethod_matmul(&proj_input, &**ple_proj)?;
        if ple_proj.quantized_act_type().is_some() {
            projected = projected.to_dtype(original_dtype)?;
        }
        // Apply scalar: hidden_size^-0.5
        let projected = (projected * self.per_layer_projection_scalar)?;
        // Reshape to [b, seq, num_layers, ple_dim]
        let projected = projected.reshape((b, seq, self.num_hidden_layers, ple_dim))?;

        // 3. Normalize the projection
        let projected = ple_norm.forward(&projected)?;

        // 4. Combine: (projection + embedding) * 2^-0.5
        let combined = ((projected + embedded)? * self.per_layer_input_scale)?;

        // 5. Split into per-layer tensors via single transpose + contiguous + narrow slices
        // combined: [b, seq, num_layers, ple_dim] → transpose to [b, num_layers, seq, ple_dim]
        let combined = combined.transpose(1, 2)?.contiguous()?;
        let mut per_layer_inputs = Vec::with_capacity(self.num_hidden_layers);
        for i in 0..self.num_hidden_layers {
            // narrow on dim 1 is zero-copy since combined is contiguous
            let chunk = combined.narrow(1, i, 1)?.squeeze(1)?;
            per_layer_inputs.push(chunk);
        }

        Ok(Some(per_layer_inputs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        ple_input_ids: &Tensor,
        mut xs: Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;

        // Compute PLE per-layer inputs
        let per_layer_inputs = self.compute_ple(ple_input_ids, &xs)?;

        // Larger Gemma 4 variants use a mixed causal/bidirectional mask for
        // image soft tokens during prefill. Flash attention cannot consume that
        // per-token override, so we materialize real masks and bypass flash only
        // for this path.
        let has_bidirectional = self.use_bidirectional_vision_attention
            && self.image_token_id.is_some()
            && input_ids.dim(1)? > 1;
        let mask_cache: &dyn PastKvLenCache = metadata
            .as_ref()
            .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
            .unwrap_or(cache as &dyn PastKvLenCache);

        let (attention_mask, sliding_attention_mask, layer_flash_params) = if has_bidirectional {
            let attention_mask =
                CausalMasker.make_causal_mask_as_attn_bias(input_ids, mask_cache, xs.dtype())?;
            let attention_mask = attention_mask.map(|m| m.to_device(&Device::Cpu).unwrap());

            let sliding_attention_mask = CausalMasker
                .make_sliding_window_causal_mask_as_attn_bias(
                    input_ids,
                    mask_cache,
                    Some(self.sliding_window),
                    xs.dtype(),
                )?;
            let sliding_attention_mask = sliding_attention_mask
                .map(|m| {
                    Self::apply_image_bidirectional_mask(
                        &m,
                        input_ids,
                        self.image_token_id.expect("missing image token id"),
                    )
                })
                .transpose()?;
            let sliding_attention_mask =
                sliding_attention_mask.map(|m| m.to_device(&Device::Cpu).unwrap());

            (attention_mask, sliding_attention_mask, None)
        } else {
            let attention_mask = CausalMasker.make_causal_mask_matrix(
                input_ids,
                mask_cache,
                xs.dtype(),
                self.cfg.num_attn_heads,
            )?;
            let attention_mask = attention_mask.map(|m| m.to_device(&Device::Cpu).unwrap());
            let attention_mask = attention_mask.filter(|_| {
                metadata
                    .as_ref()
                    .map(|(_, meta)| meta.is_first_prompt_chunk)
                    .unwrap_or(true)
            });
            let sliding_attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
                input_ids,
                mask_cache,
                Some(self.sliding_window),
                xs.dtype(),
                self.cfg.num_attn_heads,
            )?;
            let sliding_attention_mask =
                sliding_attention_mask.map(|m| m.to_device(&Device::Cpu).unwrap());
            let sliding_attention_mask = sliding_attention_mask.filter(|_| {
                metadata
                    .as_ref()
                    .map(|(_, meta)| meta.is_first_prompt_chunk)
                    .unwrap_or(true)
            });

            (attention_mask, sliding_attention_mask, Some(flash_params))
        };

        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        let sliding_attention_mask = DeviceMappedMask::new(sliding_attention_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let per_layer_input = per_layer_inputs
                .as_ref()
                .map(|pli| self.mapper.map(pli[i].clone(), i))
                .transpose()?;
            xs = layer.forward(
                &xs,
                per_layer_input.as_ref(),
                attention_mask.as_ref().map(|m| m.get(xs.device())),
                sliding_attention_mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                cache,
                metadata.as_ref().map(|(kv_cache, metadata)| {
                    let cache_idx = layer.self_attn.kv_shared_layer_index.unwrap_or(i);
                    (kv_cache[cache_idx].clone(), *metadata)
                }),
                layer_flash_params,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let mut xs = extract_logits(&xs, context_lens)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }

        let mut xs = MatMul.qmethod_matmul(&xs, &*self.lm_head)?;

        if let Some(final_logit_softcapping) = self.final_logit_softcapping {
            xs = (xs / final_logit_softcapping)?;
            xs = xs.tanh()?;
            xs = (xs * final_logit_softcapping)?;
        }

        Ok(xs)
    }

    fn apply_image_bidirectional_mask(
        causal_mask: &Tensor,
        input_ids: &Tensor,
        image_token_id: usize,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let total_len = causal_mask.dim(1)?;
        let past_kv_len = total_len - seq_len;

        let input_ids_1d = input_ids.squeeze(0)?;
        let is_image = input_ids_1d
            .eq(image_token_id as f64)?
            .to_dtype(candle_core::DType::U32)?;

        let is_image_vec: Vec<u32> = is_image.to_vec1()?;
        let mut group_ids = vec![-1i64; seq_len];
        let mut current_group: i64 = -1;
        for i in 0..seq_len {
            if is_image_vec[i] == 1 {
                if i == 0 || is_image_vec[i - 1] == 0 {
                    current_group += 1;
                }
                group_ids[i] = current_group;
            }
        }

        let device = causal_mask.device();
        let dtype = causal_mask.dtype();

        let mut override_vals = vec![0f32; seq_len * total_len];
        for qi in 0..seq_len {
            if group_ids[qi] < 0 {
                continue;
            }
            for ki in 0..seq_len {
                if group_ids[ki] >= 0 && group_ids[qi] == group_ids[ki] {
                    let col = ki + past_kv_len;
                    override_vals[qi * total_len + col] = 1.0;
                }
            }
        }

        let override_mask = Tensor::from_vec(override_vals, (seq_len, total_len), device)?;
        let zero = Tensor::zeros((seq_len, total_len), dtype, device)?;
        let override_bool = override_mask.to_dtype(candle_core::DType::U8)?;
        override_bool.where_cond(&zero, causal_mask)
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  IsqModel
// ────────────────────────────────────────────────────────────────────────────

impl IsqModel for TextModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        // Keep Gemma 4's tied output projection in BF16. Quantizing the
        // enormous tied lm_head destabilizes low-bit decoding first, which is
        // especially visible around rare control tokens.
        if !self.lm_head_is_tied {
            tensors.push((&mut self.lm_head, None));
        }
        if let Some(ref mut proj) = self.per_layer_model_projection {
            tensors.push((proj, None));
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            if let Some(ref mut v) = layer.self_attn.v_proj {
                tensors.push((v, Some(i)));
            }
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.extend(
                layer
                    .mlp
                    .get_isq_layers()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
            if let Some(ref mut moe) = layer.moe_block {
                tensors.extend(
                    moe.get_isq_layers()
                        .into_iter()
                        .map(|m| (m, Some(i)))
                        .collect::<Vec<_>>(),
                );
            }
            if let Some(ref mut gate) = layer.per_layer_input_gate {
                tensors.push((gate, Some(i)));
            }
            if let Some(ref mut proj) = layer.per_layer_projection {
                tensors.push((proj, Some(i)));
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb;
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        if let Some(ref emb) = self.embed_tokens_per_layer {
            uvb_m
                .pp("embed_tokens_per_layer")
                .add_tensor("weight", emb.embeddings().clone());
        }
        if let Some(ref norm) = self.per_layer_projection_norm {
            uvb_m.pp("per_layer_projection_norm").add(norm);
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l
                .pp("self_attn")
                .pp("q_norm")
                .add(&layer.self_attn.q_norm);
            uvb_l
                .pp("self_attn")
                .pp("k_norm")
                .add(&layer.self_attn.k_norm);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("pre_feedforward_layernorm")
                .add(&layer.pre_feedforward_layernorm);
            uvb_l
                .pp("post_feedforward_layernorm")
                .add(&layer.post_feedforward_layernorm);
            if let Some(ref norm) = layer.pre_feedforward_layernorm_2 {
                uvb_l.pp("pre_feedforward_layernorm_2").add(norm);
            }
            if let Some(ref norm) = layer.post_feedforward_layernorm_1 {
                uvb_l.pp("post_feedforward_layernorm_1").add(norm);
            }
            if let Some(ref norm) = layer.post_feedforward_layernorm_2 {
                uvb_l.pp("post_feedforward_layernorm_2").add(norm);
            }
            if let Some(ref norm) = layer.post_per_layer_input_norm {
                uvb_l.pp("post_per_layer_input_norm").add(norm);
            }
            if let Some(ref scalar) = layer.layer_scalar {
                uvb_l.add_tensor("layer_scalar", scalar.clone());
            }

            // per_expert_scale is not quantizable, store as residual
            if let Some(ref scale) = layer.per_expert_scale {
                uvb_l
                    .pp("moe")
                    .add_tensor("per_expert_scale", scale.clone());
            }
            if let Some(ref router) = layer.router {
                let uvb_router = uvb_l.pp("router");
                uvb_router.add_tensor("scale", router.scale.clone());
                uvb_router
                    .pp("proj")
                    .add_tensor("weight", router.proj.weight().clone());
            }
        }

        uvb_m.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        let mut names = Vec::new();
        // lm_head
        if !self.lm_head_is_tied {
            names.push(None);
        }
        // per_layer_model_projection
        if self.per_layer_model_projection.is_some() {
            names.push(None);
        }
        for i in 0..self.layers.len() {
            names.push(Some(format!("blk.{i}.attn_q.weight")));
            names.push(Some(format!("blk.{i}.attn_k.weight")));
            if self.layers[i].self_attn.v_proj.is_some() {
                names.push(Some(format!("blk.{i}.attn_v.weight")));
            }
            names.push(Some(format!("blk.{i}.attn_output.weight")));
            names.push(Some(format!("blk.{i}.ffn_gate.weight")));
            names.push(Some(format!("blk.{i}.ffn_up.weight")));
            names.push(Some(format!("blk.{i}.ffn_down.weight")));
            if self.layers[i].per_layer_input_gate.is_some() {
                names.push(None);
            }
            if self.layers[i].per_layer_projection.is_some() {
                names.push(None);
            }
        }
        Ok(names)
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  VisionModel
// ────────────────────────────────────────────────────────────────────────────

impl VisionModel for TextModel {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        _seqlen_offsets: &[usize],
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _model_specific_args: Box<dyn std::any::Any>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        unreachable!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.model_config_like()
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  AnyMoeBaseModelMixin (empty)
// ────────────────────────────────────────────────────────────────────────────

impl AnyMoeBaseModelMixin for TextModel {}
