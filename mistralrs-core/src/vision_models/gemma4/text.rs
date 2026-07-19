#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{
    softcap, ColumnParallelLayer, QuantMethod, QuantMethodConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder, UnquantLinear,
};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        embedding, Activation, CausalMasker, Mlp, RmsNorm, RotaryEmbedding, ScaledEmbedding, Sdpa,
    },
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{
        block_hash::MultimodalAttentionPolicy, AttentionBackendKind, AttentionImplementation,
        KvCacheLayout, KvCacheTopology, ModelConfigLike, ModelConfigMetadata, PagedAttention,
    },
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, MultimodalModel, NormalCache,
        NormalCacheType, NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::Gemma4TextConfig;

macro_rules! is_sliding {
    ($layer_idx:expr, $cfg:expr) => {
        $cfg.layer_types[$layer_idx] == "sliding_attention"
    };
}

pub(super) fn first_kv_shared_layer_idx(cfg: &Gemma4TextConfig) -> usize {
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

/// Proportional RoPE for Gemma4 full-attention layers.
///
/// Unlike standard RotaryEmbedding, this computes inv_freq for only the first
/// `rope_angles` dimensions but uses `head_dim` as the denominator, then pads
/// with zeros. The result: cos=1, sin=0 for non-rotated positions, so the
/// standard rotary formula `x*cos + rotate_half(x)*sin` acts as identity
/// for those dims.
pub(super) struct ProportionalRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    is_gpt_neox: bool,
}

impl ProportionalRotaryEmbedding {
    pub(super) fn new(
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
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

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

    #[allow(clippy::too_many_arguments)]
    fn forward_qkv_norm(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        v_weight: &Tensor,
        q_eps: f64,
        k_eps: f64,
        v_eps: f64,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        crate::layers::qkv_rms_norm_rope(
            q,
            k,
            v,
            q_weight,
            k_weight,
            v_weight,
            q_eps,
            k_eps,
            v_eps,
            &self.cos,
            &self.sin,
            self.is_gpt_neox,
            positions,
        )
    }

    fn forward_q_norm(
        &self,
        q: &Tensor,
        q_weight: &Tensor,
        q_eps: f64,
        positions: &Tensor,
    ) -> Result<Tensor> {
        crate::layers::q_rms_norm_rope(
            q,
            q_weight,
            q_eps,
            &self.cos,
            &self.sin,
            self.is_gpt_neox,
            positions,
        )
    }

    pub(super) fn forward_q(&self, q: &Tensor, positions: &Tensor) -> Result<Tensor> {
        crate::layers::apply_rotary_q(q, &self.cos, &self.sin, positions, self.is_gpt_neox)
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Router
// ────────────────────────────────────────────────────────────────────────────

struct Gemma4Router {
    norm: RmsNorm,
    scale: Tensor,
    proj: candle_nn::Linear,
    top_k: usize,
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
        let proj = candle_nn::Linear::new(proj_w.to_dtype(vb.dtype())?, None);
        // Pre-combine: weight = scale * hidden_size^(-0.5)
        let root_size = (hidden_size as f64).powf(-0.5);
        let combined_weight = (&scale * root_size)?;
        let norm = RmsNorm::from_w(combined_weight, eps)?;
        Ok(Self {
            norm,
            scale,
            proj,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor, per_expert_scale: &Tensor) -> Result<(Tensor, Tensor)> {
        let normed = xs.apply(&self.norm)?;

        let logits = normed
            .to_dtype(self.proj.weight().dtype())?
            .apply(&self.proj)?;

        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: Some((-1e4, 1e4)),
            },
            None,
            Some(per_expert_scale),
        )?;
        Ok((topk.values, topk.indices))
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Attention
// ────────────────────────────────────────────────────────────────────────────

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Option<Arc<dyn QuantMethod>>,
    v_proj: Option<Arc<dyn QuantMethod>>,
    merged_qkv_proj: Option<crate::ops::MergedDenseProjection>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    is_sliding: bool,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    v_norm_rms: Option<RmsNorm>,
    kv_shared_layer_index: Option<usize>,
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

        let use_alternative_attention = cfg.attention_k_eq_v && !sliding;
        let (head_dim, num_kv_heads) = if sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = if use_alternative_attention {
                cfg.num_global_key_value_heads
                    .unwrap_or(cfg.num_key_value_heads)
            } else {
                cfg.num_key_value_heads
            };
            (cfg.global_head_dim, global_kv)
        };

        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("q_proj"),
        )?;

        let is_shared = kv_shared_layer_index.is_some();
        let (k_proj, v_proj, merged_qkv_proj, k_norm, v_norm_rms) = if is_shared {
            (None, None, None, None, None)
        } else {
            let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
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
            let merged_qkv_proj = if let Some(v_proj) = v_proj.as_ref() {
                crate::ops::MergedDenseProjection::new(&[
                    q_proj.as_ref(),
                    k_proj.as_ref(),
                    v_proj.as_ref(),
                ])?
            } else {
                crate::ops::MergedDenseProjection::new(&[q_proj.as_ref(), k_proj.as_ref()])?
            };
            let k_norm = RmsNorm::new(
                head_dim,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("k_norm"), false),
            )?;
            let v_dev = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&candle_core::Device::Cpu);
            let v_norm_weight = Tensor::ones(head_dim, vb.dtype(), v_dev)?;
            let v_norm_rms = RmsNorm::from_w(v_norm_weight, cfg.rms_norm_eps)?;
            (
                Some(k_proj),
                v_proj,
                merged_qkv_proj,
                Some(k_norm),
                Some(v_norm_rms),
            )
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

        let q_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let num_heads = num_heads / comm.world_size();
        let num_kv_heads = (num_kv_heads / comm.world_size()).max(1);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            merged_qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb_global,
            rotary_emb_local,
            is_sliding: sliding,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    num_kv_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0,
                sliding_window,
                sinks: None,
            },
            q_norm,
            k_norm,
            v_norm_rms,
            kv_shared_layer_index,
            layer_idx,
        })
    }

    fn force_eager_prefill(&self) -> bool {
        !self.is_sliding && self.head_dim > 512
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        sliding_attention_mask: &AttentionMask,
        rope_positions: &Tensor,
        kv_caches: &mut [KvCache],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let is_shared = self.kv_shared_layer_index.is_some();

        let qkv = if !is_shared {
            if let Some(merged_qkv_proj) = &self.merged_qkv_proj {
                let mut parts = merged_qkv_proj.forward(xs)?.into_iter();
                let q = parts.next().unwrap();
                let k = parts.next().unwrap();
                let v = parts.next().unwrap_or_else(|| k.clone());
                Some((q, k, v))
            } else {
                match (self.k_proj.as_ref(), self.v_proj.as_ref()) {
                    (Some(k_proj), Some(v_proj)) => Some(crate::ops::qkv_projections(
                        xs,
                        self.q_proj.as_ref(),
                        k_proj.as_ref(),
                        v_proj.as_ref(),
                    )?),
                    _ => None,
                }
            }
        } else {
            None
        };

        // Q projection (always needed)
        let mut q = if let Some((q, _, _)) = qkv.as_ref() {
            q.clone()
        } else {
            self.q_proj.forward(xs)?
        };
        q = if q_len != 1 {
            q.reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
        } else {
            q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?
        };

        // K/V projection, reshape, norms (skip for shared layers that reuse donor KV)
        let (mut k, mut v) = if !is_shared {
            let (k, v) = if let Some((_, k, v)) = qkv {
                (k, v)
            } else {
                let k_proj = self
                    .k_proj
                    .as_ref()
                    .expect("Gemma4 non-shared attention missing k_proj");
                let k = k_proj.forward(xs)?;
                let v = if let Some(ref v_proj) = self.v_proj {
                    v_proj.forward(xs)?
                } else {
                    k.clone()
                };
                (k, v)
            };
            let (k, v) = if q_len != 1 {
                (
                    k.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                        .transpose(1, 2)?,
                    v.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                        .transpose(1, 2)?,
                )
            } else {
                (
                    k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?,
                    v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?,
                )
            };
            (Some(k), Some(v))
        } else {
            (None, None)
        };

        if self.is_sliding {
            if let (Some(k_val), Some(v_val)) = (k.take(), v.take()) {
                let k_norm = self
                    .k_norm
                    .as_ref()
                    .expect("Gemma4 non-shared attention missing k_norm");
                let v_norm_rms = self
                    .v_norm_rms
                    .as_ref()
                    .expect("Gemma4 non-shared attention missing v_norm");
                let (q_rot, k_rot, v_norm) = self.rotary_emb_local.forward_qkv_norm(
                    &q,
                    &k_val,
                    &v_val,
                    self.q_norm.weight(),
                    k_norm.weight(),
                    v_norm_rms.weight(),
                    self.q_norm.eps(),
                    k_norm.eps(),
                    v_norm_rms.eps(),
                    rope_positions,
                )?;
                q = q_rot;
                k = Some(k_rot);
                v = Some(v_norm);
            } else {
                q = self.rotary_emb_local.forward_q_norm(
                    &q,
                    self.q_norm.weight(),
                    self.q_norm.eps(),
                    rope_positions,
                )?;
            }
        } else {
            if let (Some(k_val), Some(v_val)) = (k.take(), v.take()) {
                let k_norm = self
                    .k_norm
                    .as_ref()
                    .expect("Gemma4 non-shared attention missing k_norm");
                let v_norm_rms = self
                    .v_norm_rms
                    .as_ref()
                    .expect("Gemma4 non-shared attention missing v_norm");
                let (q_rot, k_rot, v_norm) = self.rotary_emb_global.forward_qkv_norm(
                    &q,
                    &k_val,
                    &v_val,
                    self.q_norm.weight(),
                    k_norm.weight(),
                    v_norm_rms.weight(),
                    self.q_norm.eps(),
                    k_norm.eps(),
                    v_norm_rms.eps(),
                    rope_positions,
                )?;
                q = q_rot;
                k = Some(k_rot);
                v = Some(v_norm);
            } else {
                q = self.rotary_emb_global.forward_q_norm(
                    &q,
                    self.q_norm.weight(),
                    self.q_norm.eps(),
                    rope_positions,
                )?;
            }
        }

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => {
                // Paged attention path, do NOT use normal kv_caches.
                // Select the correct per-layer mask: sliding window for
                // sliding layers, full causal for full-attention layers.
                // This must be used even when paged attention is active
                // because `Custom` masks route to eager attention (not
                // flash), so flash attn's `window_size_left` never applies.
                let mask = if self.is_sliding {
                    sliding_attention_mask
                } else {
                    attention_mask
                };
                let paged_mask = mask;

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
                                k.as_ref().unwrap(),
                                v.as_ref().unwrap(),
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
                        assert!(!matches!(paged_mask, AttentionMask::None));
                        paged_attn.forward(
                            &q,
                            k.as_ref().unwrap(),
                            v.as_ref().unwrap(),
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
                let (mut k, mut v) = if let Some(donor_idx) = self.kv_shared_layer_index {
                    let donor_cache = &kv_caches[donor_idx];
                    // Use appended_k/v to get the full K/V from the donor's last
                    // append (retained + new during prefill), not the truncated
                    // sliding-window buffer that current_data()/k()/v() returns.
                    let dk = donor_cache.appended_k()?.unwrap().to_device(q.device())?;
                    let dv = donor_cache.appended_v()?.unwrap().to_device(q.device())?;
                    (dk, dv)
                } else {
                    kv_caches[self.layer_idx].append(k.as_ref().unwrap(), v.as_ref().unwrap())?
                };

                // Some sliding layers intentionally use a full normal cache
                // because they donate KV to later shared layers. Decode often
                // has no explicit mask, so every sliding layer must still clamp
                // its effective KV span to the sliding window here. Rotating
                // caches are already at most `window`, so this is a no-op for
                // the usual non-donor sliding layers.
                if let Some((start, len)) = sliding_decode_kv_window(
                    self.is_sliding,
                    q_len,
                    self.sdpa_params.sliding_window,
                    k.dim(2)?,
                ) {
                    k = k.narrow(2, start, len)?;
                    v = v.narrow(2, start, len)?;
                }

                let mask = if self.is_sliding {
                    sliding_attention_mask
                } else {
                    attention_mask
                };

                // Adjust mask dimensions to match actual KV cache length.
                // This is needed when:
                //   - shared KV cache (donor may have different seq len)
                //   - sliding/rotating KV cache (cache trimmed to window size
                //     but mask was created for the full sequence length)
                let adjust_mask = |m: Option<&Tensor>| -> Result<Option<Tensor>> {
                    if let Some(mask) = m {
                        let kv_seq_len = k.dims()[2];
                        let mask_dims = mask.dims();
                        match mask.rank() {
                            2 if mask_dims[1] > kv_seq_len => Ok(Some(mask.narrow(
                                1,
                                mask_dims[1] - kv_seq_len,
                                kv_seq_len,
                            )?)),
                            3 if mask_dims[2] > kv_seq_len => Ok(Some(mask.narrow(
                                2,
                                mask_dims[2] - kv_seq_len,
                                kv_seq_len,
                            )?)),
                            4 if mask_dims[3] > kv_seq_len => Ok(Some(mask.narrow(
                                3,
                                mask_dims[3] - kv_seq_len,
                                kv_seq_len,
                            )?)),
                            _ => Ok(Some(mask.clone())),
                        }
                    } else {
                        Ok(None)
                    }
                };
                let mask = match adjust_mask(mask.as_option_tensor())? {
                    Some(t) => AttentionMask::Custom(t),
                    None => AttentionMask::None,
                };

                Sdpa.run_attention(&q, &k, &v, &mask, flash_params, &self.sdpa_params)?
            }
        };

        let has_mask = !matches!(attention_mask, AttentionMask::None)
            || !matches!(sliding_attention_mask, AttentionMask::None);
        attn_output = if has_mask {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let res = self.o_proj.forward(&attn_output)?;
        Ok(res)
    }
}

impl Attention {
    /// Bidirectional canvas attention for block-diffusion decoding: the batch holds N
    /// sequences with EQUAL context length (scheduler buckets guarantee it), so queries are
    /// [N, heads, q_len, hd] and `cached_kv` is one batched [N, kv_heads, ctx, hd] snapshot.
    /// One flash call, causal=false; the cache is read but never written.
    pub(in crate::vision_models) fn forward_canvas(
        &self,
        xs: &Tensor,
        rope_positions: &Tensor,
        cached_kv: Option<&(Tensor, Tensor)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let (q, k, v) = if let Some(merged_qkv_proj) = &self.merged_qkv_proj {
            let mut parts = merged_qkv_proj.forward(xs)?.into_iter();
            let q = parts.next().unwrap();
            let k = parts.next().unwrap();
            let v = parts.next().unwrap_or_else(|| k.clone());
            (q, k, v)
        } else {
            let k_proj = self
                .k_proj
                .as_ref()
                .expect("Gemma4 canvas attention missing k_proj");
            let k = k_proj.forward(xs)?;
            let v = if let Some(ref v_proj) = self.v_proj {
                v_proj.forward(xs)?
            } else {
                k.clone()
            };
            (self.q_proj.forward(xs)?, k, v)
        };
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, mut k, mut v) = if self.is_sliding {
            let k_norm = self
                .k_norm
                .as_ref()
                .expect("Gemma4 canvas attention missing k_norm");
            let v_norm_rms = self
                .v_norm_rms
                .as_ref()
                .expect("Gemma4 canvas attention missing v_norm");
            self.rotary_emb_local.forward_qkv_norm(
                &q,
                &k,
                &v,
                self.q_norm.weight(),
                k_norm.weight(),
                v_norm_rms.weight(),
                self.q_norm.eps(),
                k_norm.eps(),
                v_norm_rms.eps(),
                rope_positions,
            )?
        } else {
            let k_norm = self
                .k_norm
                .as_ref()
                .expect("Gemma4 canvas attention missing k_norm");
            let v_norm_rms = self
                .v_norm_rms
                .as_ref()
                .expect("Gemma4 canvas attention missing v_norm");
            self.rotary_emb_global.forward_qkv_norm(
                &q,
                &k,
                &v,
                self.q_norm.weight(),
                k_norm.weight(),
                v_norm_rms.weight(),
                self.q_norm.eps(),
                k_norm.eps(),
                v_norm_rms.eps(),
                rope_positions,
            )?
        };

        if let Some((ck, cv)) = cached_kv {
            let (mut ck, mut cv) = (ck.clone(), cv.clone());
            // Sliding layers see an anchored window: the last `sliding_window` cached tokens,
            // shared by every canvas query. Rotating caches are already at most the window.
            if let (true, Some(window)) = (self.is_sliding, self.sdpa_params.sliding_window) {
                let cache_len = ck.dim(2)?;
                if cache_len > window {
                    ck = ck.narrow(2, cache_len - window, window)?;
                    cv = cv.narrow(2, cache_len - window, window)?;
                }
            }
            k = Tensor::cat(&[&ck.to_device(k.device())?, &k], 2)?;
            v = Tensor::cat(&[&cv.to_device(v.device())?, &v], 2)?;
        }

        // The anchored cache window above already implements the sliding rule; per-query
        // windows inside the canvas must NOT apply (the canvas is fully bidirectional).
        let sdpa_params = SdpaParams {
            sliding_window: None,
            n_kv_groups: self.sdpa_params.n_kv_groups,
            softcap: self.sdpa_params.softcap,
            softmax_scale: self.sdpa_params.softmax_scale,
            sinks: self.sdpa_params.sinks.clone(),
        };
        let attn_output = Sdpa
            .run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::None,
                Some(flash_params),
                &sdpa_params,
            )?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?;
        self.o_proj.forward(&attn_output)
    }
}

fn sliding_decode_kv_window(
    is_sliding: bool,
    q_len: usize,
    sliding_window: Option<usize>,
    kv_len: usize,
) -> Option<(usize, usize)> {
    let window = sliding_window?;
    if !is_sliding || q_len != 1 || kv_len <= window {
        return None;
    }
    Some((kv_len - window, window))
}

// ────────────────────────────────────────────────────────────────────────────
//  Decoder layer
// ────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn crate::amoe::MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    // MoE
    moe_block: Option<MoEExperts>,
    per_expert_scale: Option<Tensor>,
    router: Option<Gemma4Router>,
    pre_feedforward_layernorm_2: Option<RmsNorm>,
    post_feedforward_layernorm_1: Option<RmsNorm>,
    post_feedforward_layernorm_2: Option<RmsNorm>,
    // PLE
    per_layer_input_gate: Option<Arc<dyn QuantMethod>>,
    per_layer_projection: Option<Arc<dyn QuantMethod>>,
    post_per_layer_input_norm: Option<RmsNorm>,
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
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm"), false),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
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
                .expert_intermediate_size()
                .unwrap_or(cfg.intermediate_size);

            // Support both old ("moe") and new ("experts") weight paths
            let moe_prefix = if vb.pp("moe").contains_tensor("gate_up_proj") {
                "moe"
            } else {
                "experts"
            };
            let moe_vb = mapper.set_device(layer_idx, vb.pp(moe_prefix), false);
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
            // per_expert_scale may live under "moe", "experts", or "router"
            // UQFF residuals always store it under "moe"
            let router_vb = mapper.set_device(layer_idx, vb.pp("router"), false);
            let moe_explicit_vb = mapper.set_device(layer_idx, vb.pp("moe"), false);
            let scale = moe_vb
                .get(num_experts, "per_expert_scale")
                .or_else(|_| moe_explicit_vb.get(num_experts, "per_expert_scale"))
                .or_else(|_| router_vb.get(num_experts, "per_expert_scale"))?
                .to_dtype(DType::F32)?;
            let rtr = Gemma4Router::new(
                cfg.hidden_size,
                num_experts,
                top_k,
                cfg.rms_norm_eps,
                router_vb,
            )?;
            let pre_ff_2 = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm_2"), false),
            )?;
            let post_ff_1 = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("post_feedforward_layernorm_1"), false),
            )?;
            let post_ff_2 = RmsNorm::new(
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
            let norm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("post_per_layer_input_norm"), false),
            )?;
            (Some(gate), Some(proj), Some(norm))
        } else {
            (None, None, None)
        };

        let layer_scalar = mapper
            .set_device(layer_idx, vb, false)
            .get((1,), "layer_scalar")
            .ok();

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
        input_normed: Option<&Tensor>,
        next_input_layernorm: Option<&RmsNorm>,
        per_layer_input: Option<&Tensor>,
        attention_mask: &AttentionMask,
        sliding_attention_mask: &AttentionMask,
        rope_positions: &Tensor,
        kv_caches: &mut [KvCache],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
        layer_scalar_override: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let residual = xs.clone();
        let normed = if let Some(input_normed) = input_normed {
            input_normed.clone()
        } else {
            self.input_layernorm.forward(xs)?
        };
        let attn_out = self.self_attn.forward(
            &normed,
            attention_mask,
            sliding_attention_mask,
            rope_positions,
            kv_caches,
            metadata,
            flash_params,
        )?;

        let (post_attn, pre_ff_normed) = self
            .post_attention_layernorm
            .forward_residual_then_rms_norm(
                &attn_out,
                &residual,
                &self.pre_feedforward_layernorm,
            )?;

        self.forward_post_attn(
            post_attn,
            pre_ff_normed,
            next_input_layernorm,
            per_layer_input,
            layer_scalar_override,
        )
    }

    /// Block-diffusion canvas pass: bidirectional attention reading the KV cache without
    /// writing it, then the standard FFN/MoE flow.
    pub(in crate::vision_models) fn forward_canvas(
        &self,
        xs: &Tensor,
        rope_positions: &Tensor,
        cached_kv: Option<&(Tensor, Tensor)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out =
            self.self_attn
                .forward_canvas(&normed, rope_positions, cached_kv, flash_params)?;
        let (post_attn, pre_ff_normed) = self
            .post_attention_layernorm
            .forward_residual_then_rms_norm(&attn_out, xs, &self.pre_feedforward_layernorm)?;
        let (out, _) = self.forward_post_attn(post_attn, pre_ff_normed, None, None, None)?;
        Ok(out)
    }

    fn forward_post_attn(
        &self,
        xs: Tensor,
        pre_ff_normed: Tensor,
        next_input_layernorm: Option<&RmsNorm>,
        per_layer_input: Option<&Tensor>,
        layer_scalar_override: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let mut xs = xs;
        let layer_scalar = layer_scalar_override.or(self.layer_scalar.as_ref());

        // Feedforward
        let residual = xs.clone();
        let mut next_normed = None;
        let mut layer_scalar_applied = false;

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
            let mlp_in = pre_ff_normed.clone();
            let mlp_out = self.mlp.forward(&mlp_in)?;
            let mlp_normed = mlp_out.apply(post_ff_1)?;

            let (topk_weights, topk_ids) = router.forward(&xs, per_expert_scale)?;

            let moe_input = xs.apply(pre_ff_2)?;
            let (b, s, _) = moe_input.dims3()?;

            let topk_ids_u32 = topk_ids.reshape((b * s, ()))?.to_dtype(DType::U32)?;
            let topk_weights = topk_weights.reshape((b * s, ()))?.to_dtype(DType::F32)?;

            let moe_result = moe.forward(&moe_input, topk_weights, &topk_ids_u32)?;
            let moe_normed = moe_result.apply(post_ff_2)?;

            // Combine branches, then apply post_feedforward_layernorm (matches HF line 1694)
            let combined = (mlp_normed + moe_normed)?;
            let combined = combined.apply(&self.post_feedforward_layernorm)?;
            xs = (&residual + combined)?;
        } else {
            // Dense path: MLP only
            let mlp_out = self.mlp.forward(&pre_ff_normed)?;
            if self.per_layer_input_gate.is_none() {
                if let (Some(next_norm), Some(scalar)) = (next_input_layernorm, layer_scalar) {
                    layer_scalar_applied = true;
                    let (out, normed) = self
                        .post_feedforward_layernorm
                        .forward_residual_scaled_then_rms_norm(
                            &mlp_out, &residual, scalar, next_norm,
                        )?;
                    xs = out;
                    next_normed = Some(normed);
                } else if let Some(next_norm) = next_input_layernorm {
                    let (out, normed) = self
                        .post_feedforward_layernorm
                        .forward_residual_then_rms_norm(&mlp_out, &residual, next_norm)?;
                    xs = out;
                    next_normed = Some(normed);
                } else {
                    xs = self
                        .post_feedforward_layernorm
                        .forward_residual(&mlp_out, &residual)?;
                }
            } else {
                xs = self
                    .post_feedforward_layernorm
                    .forward_residual(&mlp_out, &residual)?;
            }
        };

        // PLE: per-layer embedding injection (after feedforward, before layer scalar)
        if let (Some(ref gate), Some(ref proj), Some(ref norm)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
        ) {
            if let Some(pli) = per_layer_input {
                let residual_ple = xs.clone();
                // gate: Linear(hidden_size -> ple_dim)
                let gate_in = xs;
                let gated = gate.forward(&gate_in)?;
                // activation + elementwise multiply with per_layer_input
                let gated = crate::ops::mul_and_act(&gated, pli, self.act)?;
                // projection: Linear(ple_dim -> hidden_size)
                let projected = proj.forward(&gated)?;
                // post-norm + residual
                xs = if let Some(scalar) = layer_scalar {
                    layer_scalar_applied = true;
                    if let Some(next_norm) = next_input_layernorm {
                        let (out, normed) = norm.forward_residual_scaled_then_rms_norm(
                            &projected,
                            &residual_ple,
                            scalar,
                            next_norm,
                        )?;
                        next_normed = Some(normed);
                        out
                    } else {
                        norm.forward_residual_scaled(&projected, &residual_ple, scalar)?
                    }
                } else if let Some(next_norm) = next_input_layernorm {
                    let (out, normed) =
                        norm.forward_residual_then_rms_norm(&projected, &residual_ple, next_norm)?;
                    next_normed = Some(normed);
                    out
                } else {
                    norm.forward_residual(&projected, &residual_ple)?
                };
            }
        }

        // Apply layer scalar
        if !layer_scalar_applied {
            if let Some(scalar) = layer_scalar {
                xs = xs.broadcast_mul(scalar)?;
            }
        }

        Ok((xs, next_normed))
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
    kv_cache_topology: KvCacheTopology,
}

fn gemma4_attention_backend_for_layer(
    config: &Gemma4ModelConfigLike,
    layer_idx: usize,
) -> AttentionBackendKind {
    if !cfg!(feature = "cuda") || !crate::perf_flags::flashinfer_decode_enabled() {
        return AttentionBackendKind::Standard;
    }
    let q_heads = config.num_attn_heads();
    let kv_heads = config.num_kv_heads_for_layer(layer_idx);
    let head_dim = config.k_head_dim_for_layer(layer_idx);
    if kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
        return AttentionBackendKind::Standard;
    }
    if config.v_head_dim_for_layer(layer_idx) == head_dim
        && matches!(head_dim, 64 | 128 | 256 | 512)
    {
        AttentionBackendKind::FlashInfer
    } else {
        AttentionBackendKind::Standard
    }
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

    fn has_kv_cache_sharing(&self) -> bool {
        self.kv_cache_topology.has_shared_layers()
    }

    fn kv_cache_topology(&self) -> KvCacheTopology {
        self.kv_cache_topology.clone()
    }

    fn attention_backend_kind(&self) -> AttentionBackendKind {
        if (0..self.num_layers()).any(|layer_idx| {
            self.attention_backend_kind_for_layer(layer_idx) == AttentionBackendKind::Standard
        }) {
            AttentionBackendKind::Standard
        } else {
            AttentionBackendKind::FlashInfer
        }
    }

    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        gemma4_attention_backend_for_layer(self, layer_idx)
    }

    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        match self.attention_backend_kind_for_layer(layer_idx) {
            AttentionBackendKind::FlashInfer => KvCacheLayout::FlashInferHnd,
            AttentionBackendKind::Standard => KvCacheLayout::Standard,
        }
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
    norm: RmsNorm,
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
    last_spec_hidden: Mutex<Option<Tensor>>,
    store_spec_hidden: AtomicBool,
    image_token_id: Option<usize>,
    video_token_id: Option<usize>,
    use_bidirectional_vision_attention: bool,
    cfg: ModelConfigMetadata,
    model_config: Arc<dyn ModelConfigLike + Send + Sync>,
}

#[derive(Clone)]
struct PrefillQuerySelection {
    source_context_lens: Vec<(usize, usize)>,
    reduced_context_lens: Vec<(usize, usize)>,
    seqlen_offsets: Vec<usize>,
    num_cached_tokens: Vec<usize>,
    query_lens: Vec<usize>,
}

impl PrefillQuerySelection {
    fn from_logits_context(
        q_len: usize,
        context_lens: &[(usize, usize)],
        seqlen_offsets: &[usize],
    ) -> Option<Self> {
        if context_lens.is_empty() || context_lens.len() != seqlen_offsets.len() {
            return None;
        }

        let mut reduced_context_lens = Vec::with_capacity(context_lens.len());
        let mut tail_offsets = Vec::with_capacity(context_lens.len());
        let mut num_cached_tokens = Vec::with_capacity(context_lens.len());
        let mut query_lens = Vec::with_capacity(context_lens.len());

        for ((start, len), offset) in context_lens.iter().zip(seqlen_offsets.iter()) {
            if *len != 1 || start.checked_add(*len)? != q_len {
                return None;
            }
            reduced_context_lens.push((0, *len));
            tail_offsets.push(offset + start);
            num_cached_tokens.push(offset + start);
            query_lens.push(*len);
        }

        Some(Self {
            source_context_lens: context_lens.to_vec(),
            reduced_context_lens,
            seqlen_offsets: tail_offsets,
            num_cached_tokens,
            query_lens,
        })
    }

    fn reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        extract_logits(tensor, self.source_context_lens.clone())
    }
}

struct KvSharingFastPrefillPlan {
    first_shared_layer: usize,
    query_selection: PrefillQuerySelection,
    paged_metadata: Option<PagedAttentionInputMetadata>,
}

impl TextModel {
    pub fn new(
        cfg: &Gemma4TextConfig,
        image_token_id: Option<usize>,
        video_token_id: Option<usize>,
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
        if cfg.enable_moe_block {
            crate::moe::prelog_moe_backend(
                normal_loading_metadata.real_device.clone(),
                vb_m.dtype(),
                normal_loading_metadata.loading_isq,
                &cfg.quantization_config,
                cfg.hidden_activation,
            );
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

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let lm_head: Arc<dyn QuantMethod> = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_m.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            let embed_weight = mapper.cast_nm_device(
                embed_tokens.embeddings(),
                normal_loading_metadata.loading_isq,
            )?;
            let lin = candle_nn::Linear::new(embed_weight, None);
            if cfg.keep_tied_lm_head_unquantized {
                Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(lin))?)
                    as Arc<dyn QuantMethod>
            } else {
                ReplicatedLayer::from_linear(
                    lin,
                    mapper.set_nm_device(vb_m.pp("lm_head"), normal_loading_metadata.loading_isq),
                )?
            }
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

        // Pre-compute which non-shared layers serve as KV donors for shared layers.
        // Donor layers must use a full (non-rotating) cache so that shared consumers
        // see the complete sequence, even when the donor itself is a sliding-window layer.
        let first_shared = first_kv_shared_layer_idx(cfg);
        let mut donor_layers = std::collections::HashSet::<usize>::new();
        if first_shared < cfg.num_hidden_layers {
            for shared_idx in first_shared..cfg.num_hidden_layers {
                let attention_type = &cfg.layer_types[shared_idx];
                if let Some(donor_idx) = cfg.layer_types[..first_shared]
                    .iter()
                    .rposition(|ty| ty == attention_type)
                {
                    donor_layers.insert(donor_idx);
                }
            }
        }

        let mut per_layer_num_kv_heads = Vec::with_capacity(cfg.num_hidden_layers);
        let mut per_layer_k_head_dim = Vec::with_capacity(cfg.num_hidden_layers);
        let mut per_layer_v_head_dim = Vec::with_capacity(cfg.num_hidden_layers);
        let mut kv_cache_layer_owners = Vec::with_capacity(cfg.num_hidden_layers);
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
                    kv_cache_layer_owners.push(owner);
                    Ok(NormalCacheType::Shared { owner })
                } else if is_sliding {
                    kv_cache_layer_owners.push(layer_idx);
                    if donor_layers.contains(&layer_idx) {
                        // Donor for shared layers: full cache so consumers see
                        // the entire sequence. SWA masking still applied via
                        // attention_mask in the forward pass.
                        Ok(NormalCacheType::Normal {
                            max_seq_len: cfg.max_position_embeddings,
                        })
                    } else {
                        Ok(NormalCacheType::SlidingWindow {
                            window: cfg.effective_sliding_window(),
                        })
                    }
                } else {
                    kv_cache_layer_owners.push(layer_idx);
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
                kv_cache_topology: KvCacheTopology::from_layer_owners(kv_cache_layer_owners),
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
            last_spec_hidden: Mutex::new(None),
            store_spec_hidden: AtomicBool::new(false),
            image_token_id,
            video_token_id,
            use_bidirectional_vision_attention: matches!(
                cfg.use_bidirectional_attention.as_deref(),
                Some("vision")
            ),
            cfg: cfg_metadata,
            model_config,
            mapper,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn supports_cuda_decode_graphs(&self) -> bool {
        true
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn last_spec_hidden(&self) -> Option<Tensor> {
        self.last_spec_hidden.lock().ok().and_then(|h| h.clone())
    }

    pub fn set_store_spec_hidden(&self, store: bool) {
        self.store_spec_hidden.store(store, Ordering::Relaxed);
        if !store {
            if let Ok(mut hidden) = self.last_spec_hidden.lock() {
                *hidden = None;
            }
        }
    }

    pub fn model_config_like(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.model_config.clone()
    }

    pub fn device_mapper(&self) -> &dyn DeviceMapper {
        &*self.mapper
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
        let projected = ple_proj.forward(inputs_embeds)?;
        // Apply scalar: hidden_size^-0.5
        let projected = (projected * self.per_layer_projection_scalar)?;
        // Reshape to [b, seq, num_layers, ple_dim]
        let projected = projected.reshape((b, seq, self.num_hidden_layers, ple_dim))?;

        // 3. Normalize the projection
        let projected = ple_norm.forward(&projected)?;

        // 4. Combine: (projection + embedding) * 2^-0.5
        let combined = ((projected + embedded)? * self.per_layer_input_scale)?;

        // 5. Split into per-layer tensors without materializing a transposed copy.
        let mut per_layer_inputs = Vec::with_capacity(self.num_hidden_layers);
        for i in 0..self.num_hidden_layers {
            let chunk = combined.narrow(2, i, 1)?.squeeze(2)?;
            per_layer_inputs.push(chunk);
        }

        Ok(Some(per_layer_inputs))
    }

    fn kv_sharing_fast_prefill_plan(
        &self,
        input_ids: &Tensor,
        context_lens: &[(usize, usize)],
        seqlen_offsets: &[usize],
        metadata: Option<&PagedAttentionInputMetadata>,
        has_bidirectional: bool,
        requires_full_prefill_queries: bool,
    ) -> Result<Option<KvSharingFastPrefillPlan>> {
        if requires_full_prefill_queries
            || has_bidirectional
            || metadata.is_some_and(|metadata| metadata.has_noncausal_mm_context)
        {
            return Ok(None);
        }
        let (b_sz, q_len) = input_ids.dims2()?;
        if q_len <= 1 || context_lens.len() != b_sz || seqlen_offsets.len() != b_sz {
            return Ok(None);
        }
        let first_shared = self.first_kv_shared_layer_idx();
        if first_shared == 0 || first_shared >= self.layers.len() {
            return Ok(None);
        }
        if !self.layers[first_shared..]
            .iter()
            .all(|layer| layer.self_attn.kv_shared_layer_index.is_some())
        {
            return Ok(None);
        }

        let Some(query_selection) =
            PrefillQuerySelection::from_logits_context(q_len, context_lens, seqlen_offsets)
        else {
            return Ok(None);
        };

        let paged_metadata = if let Some(metadata) = metadata {
            if metadata.block_tables.is_none() {
                return Ok(None);
            }
            Some(
                metadata
                    .for_reduced_prefill_queries(
                        &self.mapper.get_unique_devices(),
                        &query_selection.num_cached_tokens,
                        &query_selection.query_lens,
                    )
                    .map_err(|err| candle_core::Error::Msg(err.to_string()))?,
            )
        } else {
            None
        };

        Ok(Some(KvSharingFastPrefillPlan {
            first_shared_layer: first_shared,
            query_selection,
            paged_metadata,
        }))
    }

    fn first_kv_shared_layer_idx(&self) -> usize {
        self.layers
            .iter()
            .position(|layer| layer.self_attn.kv_shared_layer_index.is_some())
            .unwrap_or(self.layers.len())
    }

    fn contains_vision_tokens(&self, input_ids: &Tensor) -> Result<bool> {
        let Some(image_token_id) = self.image_token_id.map(|id| id as u32) else {
            return Ok(false);
        };
        let video_token_id = self.video_token_id.map(|id| id as u32);
        let ids = input_ids
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        Ok(ids
            .iter()
            .any(|&id| id == image_token_id || video_token_id == Some(id)))
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        ple_input_ids: &Tensor,
        xs: Tensor,
        ctx: &mut ModelForwardContext<'_>,
        has_images: bool,
    ) -> Result<Tensor> {
        self.forward_embeds_scaled(input_ids, ple_input_ids, xs, ctx, has_images, None)
    }

    /// `forward_embeds` with per-layer scalar overrides, used by DiffusionGemma's encoder
    /// mode (the shared backbone holds the decoder's scalars; the encoder has its own).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vision_models) fn forward_embeds_scaled(
        &self,
        input_ids: &Tensor,
        ple_input_ids: &Tensor,
        mut xs: Tensor,
        ctx: &mut ModelForwardContext<'_>,
        has_images: bool,
        layer_scalar_overrides: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let mut context_lens = ctx.context_lens().to_vec();
        let flash_params = ctx.flash_params().clone();

        // Compute PLE per-layer inputs
        let per_layer_inputs = self.compute_ple(ple_input_ids, &xs)?;

        let q_len = input_ids.dim(1)?;
        // Paged metadata already knows whether noncausal mm tokens reach this chunk's queries;
        // scanning input_ids would cost a GPU->CPU sync on every prefill chunk.
        let has_vision_tokens = q_len > 1
            && match ctx.paged_input_metadata() {
                Some(metadata) => metadata.has_noncausal_mm_context,
                None => self.contains_vision_tokens(input_ids)?,
            };
        let is_non_causal_media_chunk = ctx.prompt_chunk_attention_policy()
            == MultimodalAttentionPolicy::NonCausal
            && q_len > 1;
        let has_bidirectional = self.use_bidirectional_vision_attention
            && (has_images || is_non_causal_media_chunk || has_vision_tokens)
            && q_len > 1
            && self.image_token_id.is_some();
        let use_paged_mm_prefix_path = has_bidirectional
            && ctx.is_paged()
            && xs.device().is_cuda()
            && crate::using_flash_attn();
        let mask_cache = ctx.mask_cache(cache);

        let bidir_flash = FlashParams::empty(false);
        let force_eager_full_attention = self
            .layers
            .iter()
            .any(|layer| layer.self_attn.force_eager_prefill());
        let is_paged_decode = ctx.is_paged() && q_len == 1 && !ctx.is_first_prompt_chunk();
        let is_paged_prefill_chunk = ctx.is_paged() && q_len > 1 && !ctx.is_first_prompt_chunk();

        let (attention_mask, sliding_attention_mask, layer_flash_params) = if has_bidirectional
            && !use_paged_mm_prefix_path
        {
            let attention_mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    force_custom: true,
                    ..Default::default()
                },
            )?;
            let attention_mask = match attention_mask {
                AttentionMask::Custom(m) => {
                    AttentionMask::Custom(Self::apply_image_bidirectional_mask(
                        &m,
                        input_ids,
                        self.image_token_id.expect("missing image token id"),
                        self.video_token_id,
                    )?)
                }
                other => other,
            };

            let sliding_attention_mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    sliding_window: Some(self.sliding_window),
                    force_custom: true,
                },
            )?;
            let sliding_attention_mask = match sliding_attention_mask {
                AttentionMask::Custom(m) => {
                    AttentionMask::Custom(Self::apply_image_bidirectional_mask(
                        &m,
                        input_ids,
                        self.image_token_id.expect("missing image token id"),
                        self.video_token_id,
                    )?)
                }
                other => other,
            };

            (attention_mask, sliding_attention_mask, Some(&bidir_flash))
        } else if is_paged_decode {
            (
                AttentionMask::None,
                AttentionMask::None,
                Some(&flash_params),
            )
        } else {
            // Keep full-attention layers on flash-attn when their head dim is
            // supported. PagedAttention still needs a non-None prompt mask
            // (CausalFlash is enough) to route prompt chunks through SDPA
            // before writing to the paged cache.
            let attention_mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    force_custom: force_eager_full_attention,
                    ..Default::default()
                },
            )?;
            let is_first = ctx.is_first_prompt_chunk();
            let attention_mask = if is_first || is_paged_prefill_chunk {
                match attention_mask {
                    AttentionMask::Custom(m) => AttentionMask::Custom(m.to_device(&Device::Cpu)?),
                    other => other,
                }
            } else {
                AttentionMask::None
            };
            let sliding_attention_mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    sliding_window: Some(self.sliding_window),
                    force_custom: false,
                },
            )?;
            let sliding_attention_mask = if is_first || is_paged_prefill_chunk {
                match sliding_attention_mask {
                    AttentionMask::Custom(m) => AttentionMask::Custom(m.to_device(&Device::Cpu)?),
                    other => other,
                }
            } else {
                AttentionMask::None
            };

            (attention_mask, sliding_attention_mask, Some(&flash_params))
        };

        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        let sliding_attention_mask = DeviceMappedMask::new(sliding_attention_mask, &*self.mapper)?;

        let fast_prefill_tail = self.kv_sharing_fast_prefill_plan(
            input_ids,
            &context_lens,
            ctx.seqlen_offsets(),
            ctx.paged_input_metadata(),
            has_bidirectional,
            ctx.requires_full_prefill_queries(),
        )?;
        let mut reduced_to_logits = false;
        let no_attention_mask = AttentionMask::None;
        let mut input_normed = None;

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(plan) = fast_prefill_tail
                .as_ref()
                .filter(|plan| i == plan.first_shared_layer)
            {
                xs = plan.query_selection.reduce(&xs)?;
                input_normed = input_normed
                    .map(|normed| plan.query_selection.reduce(&normed))
                    .transpose()?;
                context_lens = plan.query_selection.reduced_context_lens.clone();
                reduced_to_logits = true;
            }

            xs = self.mapper.map(xs, i)?;
            let layer_input_normed = input_normed
                .take()
                .map(|normed| self.mapper.map(normed, i))
                .transpose()?;
            let per_layer_input = per_layer_inputs
                .as_ref()
                .map(|pli| {
                    let pli = if reduced_to_logits {
                        fast_prefill_tail
                            .as_ref()
                            .expect("missing active fast prefill plan")
                            .query_selection
                            .reduce(&pli[i])?
                    } else {
                        pli[i].clone()
                    };
                    self.mapper.map(pli, i)
                })
                .transpose()?;
            let this_layer_flash = if reduced_to_logits {
                None
            } else {
                layer_flash_params
            };
            let rope_positions = if reduced_to_logits {
                let plan = fast_prefill_tail
                    .as_ref()
                    .expect("missing active fast prefill plan");
                if let Some(metadata) = plan.paged_metadata.as_ref() {
                    crate::pipeline::metadata_rope_positions(metadata, xs.device())
                        .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
                        .clone()
                } else {
                    ctx.text_positions_from_offsets(
                        plan.query_selection.seqlen_offsets.as_slice(),
                        xs.dim(1)?,
                        xs.device(),
                    )?
                }
            } else {
                ctx.text_positions(xs.device(), xs.dim(1)?)?
                    .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
                    .clone()
            };
            let (layer_attention_mask, layer_sliding_attention_mask) = if reduced_to_logits {
                (&no_attention_mask, &no_attention_mask)
            } else {
                (
                    &attention_mask.get(xs.device()),
                    &sliding_attention_mask.get(xs.device()),
                )
            };
            let candidate_next_norm = if let Some(next_layer) = self.layers.get(i + 1) {
                Some(&next_layer.input_layernorm)
            } else {
                Some(&self.norm)
            };
            let next_input_layernorm =
                candidate_next_norm.filter(|&norm| norm.weight().device().same_device(xs.device()));
            let (layer_out, next_normed) = layer.forward(
                &xs,
                layer_input_normed.as_ref(),
                next_input_layernorm,
                per_layer_input.as_ref(),
                layer_attention_mask,
                layer_sliding_attention_mask,
                &rope_positions,
                cache,
                {
                    let cache_idx = layer.self_attn.kv_shared_layer_index.unwrap_or(i);
                    ctx.paged_layer(cache_idx).map(|(kv_cache, metadata)| {
                        let metadata = if reduced_to_logits {
                            fast_prefill_tail
                                .as_ref()
                                .and_then(|plan| plan.paged_metadata.as_ref())
                                .unwrap_or(metadata)
                        } else {
                            metadata
                        };
                        (kv_cache, metadata)
                    })
                },
                this_layer_flash,
                layer_scalar_overrides.map(|scalars| &scalars[i]),
            )?;
            xs = layer_out;
            input_normed = next_normed;
        }
        let xs = if let Some(normed) = input_normed {
            normed.to_device(&self.device)?
        } else {
            xs.to_device(&self.device)?.apply(&self.norm)?
        };
        let xs = extract_logits(&xs, context_lens)?;
        if self.store_spec_hidden.load(Ordering::Relaxed) {
            if let Ok(mut hidden) = self.last_spec_hidden.lock() {
                *hidden = Some(xs.clone());
            }
        }
        let mut xs = self.lm_head.forward(&xs)?;
        if let Some(final_logit_softcapping) = self.final_logit_softcapping {
            xs = softcap(&xs, final_logit_softcapping as f32)?;
        }

        Ok(xs)
    }

    fn apply_image_bidirectional_mask(
        causal_mask: &Tensor,
        input_ids: &Tensor,
        image_token_id: usize,
        video_token_id: Option<usize>,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let total_len = causal_mask.dim(1)?;
        let past_kv_len = total_len - seq_len;

        let input_ids_1d = input_ids.squeeze(0)?;
        // Both image and video tokens get bidirectional attention within their
        // respective groups.
        let is_image = input_ids_1d.eq(image_token_id as f64)?;
        let is_vision = if let Some(vid_id) = video_token_id {
            is_image.add(&input_ids_1d.eq(vid_id as f64)?)?
        } else {
            is_image
        };
        let is_vision_vec: Vec<u32> = is_vision.to_dtype(candle_core::DType::U32)?.to_vec1()?;
        let mut group_ids = vec![-1i64; seq_len];
        let mut current_group: i64 = -1;
        for i in 0..seq_len {
            if is_vision_vec[i] == 1 {
                if i == 0 || is_vision_vec[i - 1] == 0 {
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

    /// Block-diffusion canvas pass over already-embedded (and self-conditioned) canvas
    /// inputs. Bidirectional attention over [cached context + canvas], cache read-only.
    /// Returns softcapped logits for every canvas position.
    pub(in crate::vision_models) fn forward_canvas_embeds(
        &self,
        mut xs: Tensor,
        rope_positions: &Tensor,
        canvas_kv: &[(Tensor, Tensor)],
    ) -> Result<Tensor> {
        let flash_params = FlashParams::empty(false);
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let positions = rope_positions.to_device(xs.device())?;
            xs = layer.forward_canvas(&xs, &positions, canvas_kv.get(i), &flash_params)?;
        }
        let xs = xs.to_device(&self.device)?.apply(&self.norm)?;
        let mut logits = self.lm_head.forward(&xs)?;
        if let Some(cap) = self.final_logit_softcapping {
            logits = softcap(&logits, cap as f32)?;
        }
        Ok(logits)
    }

    pub(in crate::vision_models) fn embedding(&self) -> &ScaledEmbedding {
        &self.embed_tokens
    }

    /// Snapshot the frozen encoder cache for a batch of sequences with EQUAL context
    /// length: one contiguous [N, kv_heads, kv_len, head_size] pair per layer. Paged caches
    /// gather all sequences in a single kernel call per layer.
    pub(in crate::vision_models) fn gather_canvas_kv(
        &self,
        ctx: &mut ModelForwardContext<'_>,
        num_seqs: usize,
        kv_len: usize,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let mut snapshot = Vec::with_capacity(self.layers.len());
        if ctx.is_paged() {
            for (i, layer) in self.layers.iter().enumerate() {
                let ((key_cache, value_cache), metadata) = ctx.paged_layer(i).ok_or_else(|| {
                    candle_core::Error::Msg("missing paged layer cache for canvas".to_string())
                })?;
                let paged = layer
                    .self_attn
                    .paged_attn
                    .as_ref()
                    .expect("paged metadata implies paged attention layers");
                let dtype = layer.self_attn.q_norm.weight().dtype();
                snapshot.push(paged.gather_canvas_kv(
                    &key_cache,
                    &value_cache,
                    metadata,
                    num_seqs,
                    kv_len,
                    dtype,
                )?);
            }
        } else {
            let cache = &self.cache.normal().0;
            for kv in cache.iter() {
                let (Some(k), Some(v)) = (kv.k()?, kv.v()?) else {
                    candle_core::bail!("empty KV cache during canvas generation");
                };
                let (k, v) = if k.dim(2)? > kv_len && !kv.is_rotating() {
                    (k.narrow(2, 0, kv_len)?, v.narrow(2, 0, kv_len)?)
                } else {
                    (k, v)
                };
                snapshot.push((k, v));
            }
        }
        Ok(snapshot)
    }
}

impl IsqModel for TextModel {
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
            if let Some(ref k_norm) = layer.self_attn.k_norm {
                uvb_l.pp("self_attn").pp("k_norm").add(k_norm);
            }
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
}

// ────────────────────────────────────────────────────────────────────────────
//  MultimodalModel
// ────────────────────────────────────────────────────────────────────────────

impl crate::speculative::SpeculativeTargetMixin for TextModel {}

impl crate::block_diffusion::BlockDiffusionMixin for TextModel {}

impl MultimodalModel for TextModel {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        _model_specific_args: Box<dyn std::any::Any>,
        _ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        unreachable!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
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

#[cfg(test)]
mod tests {
    use super::sliding_decode_kv_window;

    #[test]
    fn sliding_decode_kv_window_clamps_only_single_token_sliding_decode() {
        assert_eq!(
            sliding_decode_kv_window(true, 1, Some(512), 7354),
            Some((6842, 512))
        );
        assert_eq!(sliding_decode_kv_window(true, 1, Some(512), 512), None);
        assert_eq!(sliding_decode_kv_window(true, 1, Some(512), 128), None);
        assert_eq!(sliding_decode_kv_window(true, 7, Some(512), 7354), None);
        assert_eq!(sliding_decode_kv_window(false, 1, Some(512), 7354), None);
        assert_eq!(sliding_decode_kv_window(true, 1, None, 7354), None);
    }
}
