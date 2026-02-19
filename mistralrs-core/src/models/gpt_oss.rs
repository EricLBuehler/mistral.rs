#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    ColumnParallelLayer, MXFP4Layer, QuantMethod, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        self, embedding, CausalMasker, GptOssRotaryEmbedding, MatMul, RmsNorm, RotaryEmbedding,
        Sdpa,
    },
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalCacheType, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::progress::NiceProgressBar,
};

serde_default_fn!(bool, default_tie_word_embeddings, false);
serde_default_fn!(f32, default_alpha, 1.702);
serde_default_fn!(f32, default_swiglu_limit, 7.0);
serde_default_fn!(f64, default_beta_fast, 32.0);
serde_default_fn!(f64, default_beta_slow, 1.0);

/// YARN rope scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(default = "default_beta_fast")]
    pub beta_fast: f64,
    #[serde(default = "default_beta_slow")]
    pub beta_slow: f64,
    #[serde(default)]
    pub truncate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub layer_types: Vec<LayerType>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default)]
    pub attention_bias: bool,
    pub rope_scaling: Option<RopeScaling>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// Wrapper enum for both standard and GPT-OSS YARN rotary embeddings
#[derive(Clone)]
pub enum GptOssRotaryEmbeddingVariant {
    Standard(Arc<RotaryEmbedding>),
    Yarn(Arc<GptOssRotaryEmbedding>),
}

impl GptOssRotaryEmbeddingVariant {
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Standard(rope) => rope.forward(q, k, seqlen_offsets),
            Self::Yarn(rope) => rope.forward(q, k, seqlen_offsets),
        }
    }
}

/// Custom SwiGLU activation: (up + 1) * gate * sigmoid(gate * alpha)
/// With clamping: gate max=limit, up [-limit, limit]
#[allow(dead_code)]
fn gptoss_swiglu(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() {
        return mistralrs_quant::gptoss_swiglu_fused(gate, up, alpha, limit);
    }

    let dtype = gate.dtype();
    let limit_d = limit as f64;

    let gate_clamped =
        gate.minimum(&Tensor::full(limit_d as f32, gate.shape(), gate.device())?.to_dtype(dtype)?)?;
    let up_clamped = up.clamp(-limit_d, limit_d)?;

    let gate_scaled = (&gate_clamped * alpha as f64)?;
    let sigmoid_val = candle_nn::ops::sigmoid(&gate_scaled)?;
    let glu = (&gate_clamped * &sigmoid_val)?;

    let up_plus_one = (&up_clamped + 1.0)?;
    up_plus_one.mul(&glu)
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: GptOssRotaryEmbeddingVariant,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    #[allow(dead_code)]
    is_sliding: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: GptOssRotaryEmbeddingVariant,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &None,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &None,
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &None,
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &None,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        let sinks = mapper
            .set_device(layer_idx, vb.clone(), false)
            .get((num_heads,), "sinks")?;

        let is_sliding = matches!(
            cfg.layer_types.get(layer_idx),
            Some(LayerType::SlidingAttention)
        );
        let sliding_window = if is_sliding { cfg.sliding_window } else { None };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window,
                sinks: Some(sinks),
            },
            is_sliding,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        _layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&xs, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&xs, &*self.v_proj)?;
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

        (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;
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
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                // For sliding window layers with a wrapped RotatingCache, the
                // returned K/V are in circular buffer order (not temporal).
                // Reorder to temporal order so the mask columns align correctly:
                // [past_tokens | new_tokens].
                let (k, v) = if self.is_sliding && q_len > 1 {
                    match &*kv_cache {
                        KvCache::Rotating { k: kc, .. }
                            if kc.current_seq_len >= kc.max_seq_len && kc.offset > 0 =>
                        {
                            let dim = 2; // (batch, heads, seq, head_dim)
                            let offset = kc.offset;
                            let max_len = kc.max_seq_len;
                            let p1_k = k.narrow(dim, offset, max_len - offset)?;
                            let p2_k = k.narrow(dim, 0, offset)?;
                            let p1_v = v.narrow(dim, offset, max_len - offset)?;
                            let p2_v = v.narrow(dim, 0, offset)?;
                            (
                                Tensor::cat(&[&p1_k, &p2_k], dim)?,
                                Tensor::cat(&[&p1_v, &p2_v], dim)?,
                            )
                        }
                        _ => (k, v),
                    }
                } else {
                    (k, v)
                };

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
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

struct GptOssMoE {
    gate: Linear,
    gate_up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    #[allow(dead_code)]
    num_experts: usize,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    alpha: f32,
    limit: f32,
}

impl GptOssMoE {
    fn new(cfg: &Config, vb: ShardedVarBuilder, layer_device: Device) -> Result<Self> {
        let gate = layers::linear(
            cfg.hidden_size,
            cfg.num_local_experts,
            vb.pp("router").set_device(layer_device.clone()),
        )?;

        let experts_vb = vb.pp("experts").set_device(layer_device);

        let gate_up_proj = MXFP4Layer::packed_gptoss_linear(
            cfg.num_local_experts,
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            true,
            "gate_up_proj",
            experts_vb.clone(),
        )?;

        let down_proj = MXFP4Layer::packed_gptoss_linear(
            cfg.num_local_experts,
            cfg.intermediate_size,
            cfg.hidden_size,
            true,
            "down_proj",
            experts_vb,
        )?;

        Ok(Self {
            gate,
            gate_up_proj,
            down_proj,
            num_experts: cfg.num_local_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            intermediate_size: cfg.intermediate_size,
            alpha: cfg.alpha,
            limit: cfg.swiglu_limit,
        })
    }

    fn forward(&self, xs: &Tensor, _layer_idx: usize) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;

        #[cfg(feature = "cuda")]
        let (topk_weights, topk_ids) = {
            let result = crate::ops::cuda_topk_softmax(&router_logits, self.num_experts_per_tok)?;
            (result.values, result.indices)
        };
        #[cfg(not(feature = "cuda"))]
        let (topk_weights, topk_ids) = {
            use crate::ops::TopKLastDimOp;
            use candle_core::DType;
            let router_f32 = router_logits.to_dtype(DType::F32)?;
            let topk_result = router_f32.topk(self.num_experts_per_tok)?;
            let topk_weights = candle_nn::ops::softmax_last_dim(&topk_result.values)?;
            (topk_weights, topk_result.indices)
        };

        let gate_up = self.gate_up_proj.gather_forward(&xs_flat, &topk_ids)?;
        let (num_tokens, topk_dim, _) = gate_up.dims3()?;

        #[cfg(feature = "cuda")]
        let activated = {
            let gate_up_for_kernel =
                gate_up.reshape((num_tokens * topk_dim, self.intermediate_size, 2))?;
            let result = mistralrs_quant::gptoss_swiglu_interleaved(
                &gate_up_for_kernel,
                self.intermediate_size,
                self.alpha,
                self.limit,
            )?;
            result.reshape((num_tokens, topk_dim, self.intermediate_size))?
        };

        #[cfg(not(feature = "cuda"))]
        let activated = {
            let gate_up_reshaped =
                gate_up.reshape((num_tokens, topk_dim, self.intermediate_size, 2))?;
            let gate = gate_up_reshaped
                .narrow(D::Minus1, 0, 1)?
                .squeeze(D::Minus1)?;
            let up = gate_up_reshaped
                .narrow(D::Minus1, 1, 1)?
                .squeeze(D::Minus1)?;
            gptoss_swiglu(&gate, &up, self.alpha, self.limit)?
        };

        let expert_out = self.down_proj.gather_forward(&activated, &topk_ids)?;

        let topk_weights = topk_weights
            .to_dtype(expert_out.dtype())?
            .unsqueeze(D::Minus1)?;
        let weighted = expert_out.broadcast_mul(&topk_weights)?;
        let output = weighted.sum(D::Minus2)?;

        output.reshape((b_size, seq_len, hidden_dim))
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: GptOssMoE,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: GptOssRotaryEmbeddingVariant,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        real_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;

        let mlp = GptOssMoE::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mlp"), false),
            real_device,
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

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
            layer_idx,
        )?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs, layer_idx)?;

        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    #[allow(dead_code)]
    cfg: Config,
    cfg_metadata: ModelConfigMetadata,
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &None,
        )?;

        let mut ropes: HashMap<_, GptOssRotaryEmbeddingVariant> = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);

            let rope = if let Some(rope_scaling) = &cfg.rope_scaling {
                if rope_scaling.rope_type == "yarn" {
                    GptOssRotaryEmbeddingVariant::Yarn(Arc::new(GptOssRotaryEmbedding::new(
                        cfg.rope_theta,
                        cfg.head_dim(),
                        cfg.max_position_embeddings,
                        rope_scaling.factor,
                        rope_scaling.original_max_position_embeddings,
                        rope_scaling.beta_fast,
                        rope_scaling.beta_slow,
                        rope_scaling.truncate,
                        device,
                        vb_m.dtype(),
                    )?))
                } else {
                    GptOssRotaryEmbeddingVariant::Standard(Arc::new(RotaryEmbedding::new(
                        cfg.rope_theta as f32,
                        cfg.head_dim(),
                        cfg.max_position_embeddings,
                        device,
                        is_gptx,
                        vb_m.dtype(),
                    )?))
                }
            } else {
                GptOssRotaryEmbeddingVariant::Standard(Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    cfg.head_dim(),
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?))
            };
            ropes.insert(device.location(), rope);
        }

        let vb_l = vb_m.pp("layers");

        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .into_iter()
        .map(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device)
                .clone();

            let rotary_emb = ropes
                .get(&device.location())
                .cloned()
                .expect("No RoPE for device");

            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim(), &device, None)?)
                }
            };

            let comm = mapper.get_comm_for(layer_idx)?;

            DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                device,
                &comm,
            )
        })
        .collect::<Result<Vec<_>>>()?;

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &None,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };

        let head_dim = cfg.head_dim();
        let cfg_metadata = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        let cache_types: Vec<NormalCacheType> = (0..cfg.num_hidden_layers)
            .map(|layer_idx| match cfg.layer_types.get(layer_idx) {
                Some(LayerType::SlidingAttention) => NormalCacheType::SlidingWindow {
                    window: cfg.sliding_window.unwrap_or(cfg.max_position_embeddings),
                },
                _ => NormalCacheType::Normal {
                    max_seq_len: cfg.max_position_embeddings,
                },
            })
            .collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: cfg.max_position_embeddings,
            mapper,
            cfg: cfg.clone(),
            cfg_metadata,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;

        let sliding_window = self.cfg.sliding_window;

        // Use the `_as_attn_bias` variants which always construct real masks.
        // The standard `make_causal_mask_matrix` returns a dummy (1,1) tensor when
        // flash-attn is enabled on CUDA, but the CPU sinks fallback needs a real mask.
        let mask_cache: &dyn PastKvLenCache = metadata
            .as_ref()
            .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
            .unwrap_or(cache as &dyn PastKvLenCache);
        let causal_mask =
            CausalMasker.make_causal_mask_as_attn_bias(input_ids, mask_cache, xs.dtype())?;

        let sliding_mask = CausalMasker.make_sliding_window_causal_mask_as_attn_bias(
            input_ids,
            mask_cache,
            sliding_window,
            xs.dtype(),
        )?;

        let should_use_mask = metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true);
        let causal_mask = if should_use_mask { causal_mask } else { None };
        let sliding_mask = if should_use_mask { sliding_mask } else { None };
        let causal_mask = DeviceMappedMask::new(causal_mask, &*self.mapper)?;
        let sliding_mask = DeviceMappedMask::new(sliding_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;

            let layer_mask = if layer.self_attn.is_sliding {
                sliding_mask.as_ref().or(causal_mask.as_ref())
            } else {
                causal_mask.as_ref()
            };

            xs = layer.forward(
                &xs,
                layer_mask.map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache[i],
                metadata.as_ref().map(|(kv, m)| (kv[i].clone(), *m)),
                flash_params,
                i,
            )?;
        }

        xs = xs.to_device(&self.device)?;
        xs = self.norm.forward(&xs)?;
        let mut xs = extract_logits(&xs, context_lens)?;

        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        MatMul.qmethod_matmul(&xs, &*self.lm_head)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut layers = Vec::new();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layers.push((&mut layer.self_attn.q_proj, Some(i)));
            layers.push((&mut layer.self_attn.k_proj, Some(i)));
            layers.push((&mut layer.self_attn.v_proj, Some(i)));
            layers.push((&mut layer.self_attn.o_proj, Some(i)));
        }
        layers.push((&mut self.lm_head, None));
        (layers, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.inner_forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
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
        candle_core::bail!("GPT-OSS does not support X-LoRA")
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

    fn is_xlora(&self) -> bool {
        false
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg_metadata
    }
}

impl AnyMoeBaseModelMixin for Model {}
