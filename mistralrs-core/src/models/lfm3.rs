#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Module, Result, Tensor, D};

use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::{self, embedding_with_legacy_tied_uqff, Activation, CausalMasker, RmsNorm, Sdpa},
    layers_masker::{CausalMaskConfig, PastKvLenCache},
    paged_attention::{
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata, PagedAttention,
        PrefixPrefillAttentionFeatures,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, ForwardMaskCache, IsqModel, KvCache, ModelForwardContext,
        NormalLoadingMetadata, NormalModel, RecurrentBatchKind,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

const DEFAULT_VOCAB_SIZE: usize = 128_000;
const DEFAULT_HIDDEN_SIZE: usize = 2048;
const DEFAULT_INTERMEDIATE_SIZE: usize = 7680;
const DEFAULT_NUM_LAYERS: usize = 24;
const DEFAULT_MAX_POSITIONS: usize = 128_000;
const DEFAULT_CONV_WIDTH: usize = 7;
const DEFAULT_SLIDING_WINDOW: usize = 256;
const DEFAULT_FULL_ROPE_THETA: f64 = 1_000_000.0;
const DEFAULT_SLIDING_ROPE_THETA: f64 = 10_000.0;

serde_default_fn!(bool, default_true, true);
serde_default_fn!(usize, default_vocab_size, DEFAULT_VOCAB_SIZE);
serde_default_fn!(usize, default_hidden_size, DEFAULT_HIDDEN_SIZE);
serde_default_fn!(usize, default_intermediate_size, DEFAULT_INTERMEDIATE_SIZE);
serde_default_fn!(usize, default_num_hidden_layers, DEFAULT_NUM_LAYERS);
serde_default_fn!(usize, default_num_attention_heads, 16);
serde_default_fn!(usize, default_num_key_value_heads, 4);
serde_default_fn!(
    usize,
    default_max_position_embeddings,
    DEFAULT_MAX_POSITIONS
);
serde_default_fn!(usize, default_conv_l_cache, DEFAULT_CONV_WIDTH);
serde_default_fn!(usize, default_sliding_window, DEFAULT_SLIDING_WINDOW);
serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(f64, default_partial_rotary_factor, 1.0);
serde_default_fn!(f64, default_rope_theta, DEFAULT_FULL_ROPE_THETA);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: String,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LayerRopeParameters {
    pub full_attention: RopeParameters,
    pub sliding_attention: RopeParameters,
}

impl Default for LayerRopeParameters {
    fn default() -> Self {
        Self {
            full_attention: RopeParameters {
                rope_theta: DEFAULT_FULL_ROPE_THETA,
                rope_type: "default".to_string(),
                partial_rotary_factor: 0.5,
            },
            sliding_attention: RopeParameters {
                rope_theta: DEFAULT_SLIDING_ROPE_THETA,
                rope_type: "default".to_string(),
                partial_rotary_factor: 1.0,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub rope_parameters: LayerRopeParameters,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default = "default_conv_l_cache", rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_true")]
    pub use_sandwich_norm: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    pub quantization_config: Option<QuantizedConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum LayerType {
    #[serde(rename = "full_attention")]
    Attention,
    #[serde(rename = "sliding_attention")]
    SlidingAttention,
    #[serde(rename = "conv")]
    Conv,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        if self.layer_types.is_empty() {
            (0..self.num_hidden_layers)
                .map(|i| {
                    if i % 4 == 2 {
                        LayerType::Attention
                    } else if i % 4 == 0 && i > 0 {
                        LayerType::SlidingAttention
                    } else {
                        LayerType::Conv
                    }
                })
                .collect()
        } else {
            self.layer_types.clone()
        }
    }

    fn rope(&self, layer_type: LayerType) -> &RopeParameters {
        match layer_type {
            LayerType::Attention => &self.rope_parameters.full_attention,
            LayerType::SlidingAttention => &self.rope_parameters.sliding_attention,
            LayerType::Conv => unreachable!(),
        }
    }
}

pub(crate) struct ModelConfig {
    base: ModelConfigMetadata,
    layer_types: Vec<LayerType>,
    sliding_window: usize,
}

impl ModelConfig {
    pub(crate) fn new(
        base: ModelConfigMetadata,
        layer_types: Vec<LayerType>,
        sliding_window: usize,
    ) -> Self {
        Self {
            base,
            layer_types,
            sliding_window,
        }
    }
}

impl ModelConfigLike for ModelConfig {
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
    fn layer_has_paged_kv_cache(&self, layer_idx: usize) -> bool {
        self.layer_types[layer_idx] != LayerType::Conv
    }
    fn sliding_window_for_layer(&self, layer_idx: usize) -> Option<usize> {
        (self.layer_types[layer_idx] == LayerType::SlidingAttention).then_some(self.sliding_window)
    }
    fn prefix_prefill_attention_features(
        &self,
        layer_idx: usize,
    ) -> Option<PrefixPrefillAttentionFeatures> {
        self.layer_has_paged_kv_cache(layer_idx)
            .then(|| PrefixPrefillAttentionFeatures {
                has_sliding_window: self.sliding_window_for_layer(layer_idx).is_some(),
                ..Default::default()
            })
    }
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    is_gptx: bool,
}

impl RotaryEmbedding {
    fn new(
        cfg: &Config,
        layer_type: LayerType,
        device: &Device,
        dtype: DType,
        is_gptx: bool,
    ) -> Result<Self> {
        let params = cfg.rope(layer_type);
        if params.rope_type != "default" && !params.rope_type.is_empty() {
            candle_core::bail!("LFM3 RoPE type `{}` is not supported", params.rope_type);
        }
        let head_dim = cfg.head_dim();
        let rotated_pairs = (params.partial_rotary_factor * head_dim as f64) as usize / 2;
        let inv_freq = (0..head_dim / 2)
            .map(|i| {
                if i < rotated_pairs {
                    1.0f32 / (params.rope_theta as f32).powf((2 * i) as f32 / head_dim as f32)
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let frequencies = positions.matmul(&inv_freq)?;
        Ok(Self {
            cos: frequencies.cos()?.to_dtype(dtype)?,
            sin: frequencies.sin()?.to_dtype(dtype)?,
            is_gptx,
        })
    }

    fn forward_qk_norm(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_norm: &RmsNorm,
        k_norm: &RmsNorm,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        layers::qk_rms_norm_rope(
            q,
            k,
            q_norm.weight(),
            k_norm.weight(),
            q_norm.eps(),
            k_norm.eps(),
            &self.cos,
            &self.sin,
            self.is_gptx,
            positions,
        )
    }
}

struct LayerLoadContext<'a> {
    cfg: &'a Config,
    mapper: &'a dyn DeviceMapper,
    layer_idx: usize,
    loading_isq: bool,
    comm: &'a Arc<mistralrs_quant::Comm>,
}

struct Mlp {
    w1: Arc<dyn QuantMethod>,
    w2: Arc<dyn QuantMethod>,
    w3: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn new(vb: ShardedVarBuilder, ctx: &LayerLoadContext<'_>) -> Result<Self> {
        let LayerLoadContext {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = *ctx;
        let intermediate_size = cfg.intermediate_size;
        let vb = mapper.set_device(layer_idx, vb, loading_isq);
        let w1 = ColumnParallelLayer::new(
            cfg.hidden_size,
            intermediate_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w1"),
        )?;
        let w2 = RowParallelLayer::new(
            intermediate_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w2"),
        )?;
        let w3 = ColumnParallelLayer::new(
            cfg.hidden_size,
            intermediate_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w3"),
        )?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::ops::quantized_ffn(x, &*self.w1, &*self.w3, &*self.w2, Activation::Silu)
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
    gate_proj: Arc<dyn QuantMethod>,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        vb: ShardedVarBuilder,
        ctx: &LayerLoadContext<'_>,
        rotary_emb: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let LayerLoadContext {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = *ctx;
        let vb_attn = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
        let head_dim = cfg.head_dim();
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            vb_attn.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_attn.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_attn.pp("v_proj"),
        )?;
        let out_proj = RowParallelLayer::new(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_attn.pp("o_proj"),
        )?;
        let gate_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            cfg.num_attention_heads,
            &cfg.quantization_config,
            false,
            comm,
            vb_attn.pp("gate_proj"),
        )?;
        let vb_norms = mapper.set_device(layer_idx, vb.pp("self_attn"), false);
        let q_layernorm = RmsNorm::new(head_dim, cfg.norm_eps, vb_norms.pp("q_norm"))?;
        let k_layernorm = RmsNorm::new(head_dim, cfg.norm_eps, vb_norms.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            gate_proj,
            q_layernorm,
            k_layernorm,
            num_heads: cfg.num_attention_heads / comm.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: (cfg.layer_types()[layer_idx] == LayerType::SlidingAttention)
                    .then_some(cfg.sliding_window),
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let rope_positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
        (q, k) = self.rotary_emb.forward_qk_norm(
            &q,
            &k,
            &self.q_layernorm,
            &self.k_layernorm,
            rope_positions,
        )?;

        let metadata = ctx.paged_layer(layer_idx);
        let y = match &self.paged_attn {
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
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(ctx.flash_params()),
                    &self.sdpa_params,
                )?
            }
        };

        let y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let gate = candle_nn::ops::sigmoid(&self.gate_proj.forward(x)?)?.unsqueeze(D::Minus1)?;
        let y = mistralrs_quant::elementwise::broadcast_mul(
            &y.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?,
            &gate,
        )?
        .reshape((b_sz, seq_len, ()))?;
        self.out_proj.forward(&y)
    }
}

struct ShortConv {
    weight: Tensor,
    bias: Option<Tensor>,
    in_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
}

impl ShortConv {
    fn new(vb: ShardedVarBuilder, ctx: &LayerLoadContext<'_>) -> Result<Self> {
        let LayerLoadContext {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            ..
        } = *ctx;
        let vb_proj = mapper.set_device(layer_idx, vb.pp("conv"), loading_isq);
        let in_proj = mistralrs_quant::linear_b(
            cfg.hidden_size,
            3 * cfg.hidden_size,
            cfg.conv_bias,
            &cfg.quantization_config,
            vb_proj.pp("in_proj"),
        )?;
        let out_proj = mistralrs_quant::linear_b(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.conv_bias,
            &cfg.quantization_config,
            vb_proj.pp("out_proj"),
        )?;
        let vb_conv = mapper.set_device(layer_idx, vb.pp("conv").pp("conv"), false);
        let weight = vb_conv.get((cfg.hidden_size, 1, cfg.conv_l_cache), "weight")?;
        let bias = cfg
            .conv_bias
            .then(|| vb_conv.get(cfg.hidden_size, "bias"))
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            in_proj,
            out_proj,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        conv_state: &mut Tensor,
        use_existing_state: bool,
        query_lens: Option<&[usize]>,
    ) -> Result<Tensor> {
        let projected = self.in_proj.forward(x)?;
        let output = if let Some(query_lens) = query_lens {
            let mut offset = 0;
            let mut outputs = Vec::with_capacity(query_lens.len());
            let mut states = Vec::with_capacity(query_lens.len());
            for (row, &len) in query_lens.iter().enumerate() {
                let segment = projected.narrow(1, offset, len)?;
                let state = conv_state.narrow(0, row, 1)?;
                let (output, state) = mistralrs_quant::lfm3_causal_conv1d(
                    &segment,
                    &self.weight,
                    self.bias.as_ref(),
                    &state,
                    use_existing_state,
                )?;
                outputs.push(output);
                states.push(state);
                offset += len;
            }
            *conv_state = Tensor::cat(&states, 0)?;
            Tensor::cat(&outputs, 1)?
        } else {
            let (output, next_state) = mistralrs_quant::lfm3_causal_conv1d(
                &projected,
                &self.weight,
                self.bias.as_ref(),
                conv_state,
                use_existing_state,
            )?;
            *conv_state = next_state;
            output
        };
        self.out_proj.forward(&output)
    }
}

enum LayerImpl {
    Attention(Attention),
    Conv(ShortConv),
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    feed_forward: Mlp,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_operator_norm: Option<RmsNorm>,
    post_ffn_norm: Option<RmsNorm>,
}

impl DecoderLayer {
    fn new(
        layer_type: LayerType,
        vb: ShardedVarBuilder,
        ctx: &LayerLoadContext<'_>,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let LayerLoadContext {
            cfg,
            mapper,
            layer_idx,
            ..
        } = *ctx;
        let layer_impl = match layer_type {
            LayerType::Attention | LayerType::SlidingAttention => {
                LayerImpl::Attention(Attention::new(
                    vb.clone(),
                    ctx,
                    rotary_emb.expect("attention layers require RoPE"),
                    paged_attn,
                )?)
            }
            LayerType::Conv => LayerImpl::Conv(ShortConv::new(vb.clone(), ctx)?),
        };
        let feed_forward = Mlp::new(vb.pp("feed_forward"), ctx)?;
        let operator_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.norm_eps,
            mapper.set_device(layer_idx, vb.pp("operator_norm"), false),
        )?;
        let ffn_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.norm_eps,
            mapper.set_device(layer_idx, vb.pp("ffn_norm"), false),
        )?;
        let post_operator_norm = cfg
            .use_sandwich_norm
            .then(|| {
                RmsNorm::new(
                    cfg.hidden_size,
                    cfg.norm_eps,
                    mapper.set_device(layer_idx, vb.pp("post_operator_norm"), false),
                )
            })
            .transpose()?;
        let post_ffn_norm = cfg
            .use_sandwich_norm
            .then(|| {
                RmsNorm::new(
                    cfg.hidden_size,
                    cfg.norm_eps,
                    mapper.set_device(layer_idx, vb.pp("post_ffn_norm"), false),
                )
            })
            .transpose()?;
        Ok(Self {
            post_operator_norm,
            post_ffn_norm,
            layer_impl,
            feed_forward,
            operator_norm,
            ffn_norm,
        })
    }

    fn forward_attention(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let LayerImpl::Attention(attn) = &self.layer_impl else {
            candle_core::bail!("expected attention layer")
        };
        let residual = x;
        let attn_out = attn.forward(
            &self.operator_norm.forward(x)?,
            attention_mask,
            kv_cache,
            ctx,
            layer_idx,
        )?;
        self.finish(&attn_out, residual)
    }

    fn forward_conv(
        &self,
        x: &Tensor,
        conv_state: &mut Tensor,
        use_existing_state: bool,
        packed_query_lens: Option<&[usize]>,
    ) -> Result<Tensor> {
        let LayerImpl::Conv(conv) = &self.layer_impl else {
            candle_core::bail!("expected conv layer")
        };
        let residual = x;
        let normalized = self.operator_norm.forward(x)?;
        let conv_out = conv.forward(
            &normalized,
            conv_state,
            use_existing_state,
            packed_query_lens,
        )?;
        self.finish(&conv_out, residual)
    }

    fn finish(&self, operator: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let (x, normed) = match &self.post_operator_norm {
            Some(norm) => {
                norm.forward_residual_then_rms_norm(operator, residual, &self.ffn_norm)?
            }
            None => self.ffn_norm.forward_add_rms_norm(operator, residual)?,
        };
        let ffn_out = self.feed_forward.forward(&normed)?;
        match &self.post_ffn_norm {
            Some(norm) => norm.forward_residual(&ffn_out, &x),
            None => mistralrs_quant::elementwise::broadcast_add(&ffn_out, &x),
        }
    }
}

pub struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    embedding_norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    sliding_window: usize,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        Self::new_inner(
            cfg,
            vb.pp("model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let layer_types = cfg.layer_types();
        if layer_types.len() != cfg.num_hidden_layers {
            candle_core::bail!("LFM3 layer_types must match num_hidden_layers");
        }
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }

        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();
        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings().then(|| {
                mapper.set_nm_device(vb_lm_head.clone(), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;
        let embedding_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.norm_eps,
            mapper.set_nm_device(vb_m.pp("embedding_norm"), false),
        )?;
        let lm_head = if cfg.tie_word_embeddings() {
            embed_tokens.clone()
        } else {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        };

        let head_dim = cfg.head_dim();
        let mut ropes = HashMap::new();
        for (layer_idx, layer_type) in layer_types.iter().enumerate() {
            if matches!(layer_type, LayerType::Conv) {
                continue;
            }
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            if let std::collections::hash_map::Entry::Vacant(entry) =
                ropes.entry((device.location(), *layer_type))
            {
                entry.insert(Arc::new(RotaryEmbedding::new(
                    cfg,
                    *layer_type,
                    device,
                    vb_m.dtype(),
                    is_gptx,
                )?));
            }
        }

        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let comm = mapper.get_comm_for(layer_idx)?;
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&(device.location(), layer_types[layer_idx]))
                .cloned();
            let paged_attn = match (&attention_mechanism, layer_types[layer_idx]) {
                (
                    AttentionImplementation::PagedAttention,
                    LayerType::Attention | LayerType::SlidingAttention,
                ) => Some(PagedAttention::new(head_dim, device, None)?),
                _ => None,
            };
            DecoderLayer::new(
                layer_types[layer_idx],
                vb_l.pp(layer_idx),
                &LayerLoadContext {
                    cfg,
                    mapper: &*mapper,
                    layer_idx,
                    loading_isq: normal_loading_metadata.loading_isq,
                    comm: &comm,
                },
                rotary_emb,
                paged_attn,
            )
        })?;

        let hybrid_layer_types = layer_types
            .iter()
            .map(|layer_type| match layer_type {
                LayerType::Attention | LayerType::SlidingAttention => HybridLayerType::Attention,
                LayerType::Conv => HybridLayerType::Recurrent,
            })
            .collect::<Vec<_>>();
        let layer_devices = (0..hybrid_layer_types.len())
            .map(|layer_idx| {
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone()
            })
            .collect::<Vec<_>>();
        let mut cache = HybridCache::new(
            HybridCacheConfig {
                layer_types: hybrid_layer_types,
                max_seq_len: cfg.max_position_embeddings,
                recurrent: RecurrentLayerConfig {
                    conv_dim: cfg.hidden_size,
                    conv_width: cfg.conv_l_cache,
                    state: crate::kv_cache::RecurrentStateSpec::Opaque { dims: Vec::new() },
                    recurrent_dtype: None,
                },
            },
            vb_m.dtype(),
            &layer_devices,
        )?;
        for (layer_idx, layer_type) in layer_types.iter().enumerate() {
            if *layer_type == LayerType::SlidingAttention {
                cache.caches[layer_idx] = HybridLayerCache::Attention(KvCache::new_rotating(
                    2,
                    cfg.sliding_window,
                    cfg.sliding_window,
                ));
            }
        }
        let cache = Arc::new(Mutex::new(cache));

        Ok(Self {
            embed_tokens,
            layers,
            layer_types,
            embedding_norm,
            lm_head,
            dtype,
            cache: EitherCache::Hybrid(cache),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: None,
                k_head_dim: head_dim,
                v_head_dim: head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            sliding_window: cfg.sliding_window,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward_embeds(input_ids, self.embed(input_ids)?, ctx)
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut x = input_embeds;
        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let has_conv_layers = self.layer_types.contains(&LayerType::Conv);
        let packed_query_lens = if ctx.flash_params().packed {
            Some(
                ctx.paged_input_metadata()
                    .and_then(|metadata| metadata.query_lens.clone())
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "LFM3 packed ShortConv requires logical query lengths",
                        )
                    })?,
            )
        } else {
            None
        };
        if has_conv_layers && recurrent_metadata.is_none() {
            candle_core::bail!("Hybrid recurrent metadata is required for LFM3 conv layers");
        }
        if has_conv_layers {
            if let Some(query_lens) = packed_query_lens.as_deref() {
                let recurrent_metadata = recurrent_metadata
                    .as_ref()
                    .expect("checked above: LFM3 conv layers require recurrent metadata");
                if recurrent_metadata.batch_kind() != RecurrentBatchKind::Prefill {
                    candle_core::bail!("LFM3 packed ShortConv cannot run a decode batch");
                }
                let (physical_batch, physical_tokens, _) = x.dims3()?;
                if physical_batch != 1
                    || query_lens.iter().sum::<usize>() != physical_tokens
                    || query_lens.contains(&0)
                {
                    candle_core::bail!("invalid packed LFM3 query lengths");
                }
                let index_count = recurrent_metadata.state_indices().dims1()?;
                if index_count != query_lens.len() {
                    candle_core::bail!(
                        "LFM3 packed ShortConv has {index_count} recurrent state indices but {} logical sequences",
                        query_lens.len()
                    );
                }
                if let Some(host_indices) = recurrent_metadata.state_indices_host() {
                    if host_indices.len() != query_lens.len() {
                        candle_core::bail!(
                            "LFM3 packed ShortConv has {} host state indices but {} logical sequences",
                            host_indices.len(),
                            query_lens.len()
                        );
                    }
                }
            }
        }
        let mut hybrid_cache = self.cache.hybrid();

        let past_len = if ctx.is_paged() {
            ForwardMaskCache::Paged(ctx.seqlen_offsets()).get_past_kv_len()?
        } else {
            hybrid_cache.get_past_kv_len()?
        };
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            &ForwardMaskCache::Paged(&[past_len]),
            x.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let sliding_past_len = if ctx.is_paged() {
            past_len
        } else {
            past_len.min(self.sliding_window)
        };
        let sliding_mask = CausalMasker.make_causal_mask(
            input_ids,
            &ForwardMaskCache::Paged(&[sliding_past_len]),
            x.dtype(),
            &CausalMaskConfig {
                sliding_window: Some(self.sliding_window),
                force_custom: false,
            },
        )?;
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        let sliding_mask = DeviceMappedMask::new(sliding_mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;
            match &layer.layer_impl {
                LayerImpl::Attention(_) => {
                    let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(layer_idx)
                    else {
                        candle_core::bail!(
                            "Hybrid cache layer {layer_idx} is not attention for LFM3"
                        );
                    };
                    x = layer.forward_attention(
                        &x,
                        &if self.layer_types[layer_idx] == LayerType::SlidingAttention {
                            sliding_mask.get(x.device())
                        } else {
                            mask.get(x.device())
                        },
                        kv_cache,
                        ctx,
                        layer_idx,
                    )?;
                }
                LayerImpl::Conv(_) => {
                    let recurrent_metadata = recurrent_metadata
                        .as_ref()
                        .expect("checked above: LFM3 conv layers require recurrent metadata");
                    let indices = hybrid_cache
                        .state_indices_for_layer(layer_idx)?
                        .ok_or_else(|| {
                            candle_core::Error::msg(format!(
                                "Hybrid cache layer {layer_idx} is missing recurrent state indices"
                            ))
                        })?;
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer_idx)
                    else {
                        candle_core::bail!(
                            "Hybrid cache layer {layer_idx} is not recurrent for LFM3"
                        );
                    };
                    let mut conv_state = pool.gather_conv_state(&indices)?;
                    let use_existing_state = recurrent_metadata.batch_kind()
                        == RecurrentBatchKind::Decode
                        || !ctx.is_first_prompt_chunk()
                        || ctx.seqlen_offsets().iter().any(|offset| *offset > 0);
                    x = layer.forward_conv(
                        &x,
                        &mut conv_state,
                        use_existing_state,
                        packed_query_lens.as_deref(),
                    )?;
                    pool.scatter_conv_state_with_host_indices(
                        &indices,
                        recurrent_metadata.state_indices_host(),
                        &conv_state,
                    )?;
                }
            }
        }

        let x = x.to_device(&self.device)?;
        let x = self.embedding_norm.forward(&x)?;
        let x = ctx.logits(&x)?;
        ctx.lm_head(&*self.lm_head, &x)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("embedding_norm").add(&self.embedding_norm);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("operator_norm").add(&layer.operator_norm);
            uvb_l.pp("ffn_norm").add(&layer.ffn_norm);
            match &layer.layer_impl {
                LayerImpl::Attention(attn) => {
                    uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_layernorm);
                    uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_layernorm);
                }
                LayerImpl::Conv(conv) => {
                    let uvb_conv = uvb_l.pp("conv").pp("conv");
                    uvb_conv.add_tensor("weight", conv.weight.clone());
                    if let Some(bias) = &conv.bias {
                        uvb_conv.add_tensor("bias", bias.clone());
                    }
                }
            }
            if let Some(norm) = &layer.post_operator_norm {
                uvb_l.pp("post_operator_norm").add(norm);
            }
            if let Some(norm) = &layer.post_ffn_norm {
                uvb_l.pp("post_ffn_norm").add(norm);
            }
        }
        uvb_m.to_safetensors()
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.residual_tensors_m(UnVarBuilder::new().pp("model"))
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
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
        candle_core::bail!("LFM3 does not support X-LoRA forward")
    }

    fn is_xlora(&self) -> bool {
        false
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }

    fn max_seq_len(&self) -> usize {
        self.cfg.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }

    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(ModelConfig::new(
            self.cfg.clone(),
            self.layer_types.clone(),
            self.sliding_window,
        ))
    }

    fn supports_packed_prefill(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for Model {}
#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 2e-5;
    const ROTARY_TOLERANCE: f32 = 1e-6;
    const HIDDEN_SIZE: usize = 16;
    const CONV_WIDTH: usize = 7;
    const SEQ_LEN: usize = 8;
    const FIRST_SEQ_LEN: usize = 3;

    #[test]
    fn partial_rope_preserves_full_head_frequency_spacing() -> Result<()> {
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "hidden_size": HIDDEN_SIZE,
            "num_attention_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 32,
        }))
        .map_err(candle_core::Error::wrap)?;
        let rope =
            RotaryEmbedding::new(&cfg, LayerType::Attention, &Device::Cpu, DType::F32, true)?;
        let sin = rope.sin.to_vec2::<f32>()?;
        let cos = rope.cos.to_vec2::<f32>()?;
        let position = 11;
        let frequency = (DEFAULT_FULL_ROPE_THETA as f32).powf(-2.0 / cfg.head_dim() as f32);
        assert!((sin[position][1] - (position as f32 * frequency).sin()).abs() < TOLERANCE);
        for (&sin, &cos) in sin[position][2..].iter().zip(&cos[position][2..]) {
            assert!(sin.abs() < ROTARY_TOLERANCE);
            assert!((cos - 1.0).abs() < ROTARY_TOLERANCE);
        }
        Ok(())
    }

    #[test]
    fn packed_short_conv_preserves_independent_sequence_states() -> Result<()> {
        let device = Device::Cpu;
        let identity = Tensor::eye(HIDDEN_SIZE, DType::F32, &device)?;
        let weights = HashMap::from([
            (
                "in_proj.weight".to_string(),
                Tensor::cat(&[&identity, &identity, &identity], 0)?,
            ),
            ("out_proj.weight".to_string(), identity),
        ]);
        let vb = mistralrs_quant::ShardedSafeTensors::wrap(weights, DType::F32, device.clone());
        let conv = ShortConv {
            weight: Tensor::ones((HIDDEN_SIZE, 1, CONV_WIDTH), DType::F32, &device)?,
            bias: None,
            in_proj: mistralrs_quant::linear_no_bias(
                HIDDEN_SIZE,
                3 * HIDDEN_SIZE,
                &None,
                vb.pp("in_proj"),
            )?,
            out_proj: mistralrs_quant::linear_no_bias(
                HIDDEN_SIZE,
                HIDDEN_SIZE,
                &None,
                vb.pp("out_proj"),
            )?,
        };
        let input = Tensor::arange(0f32, (SEQ_LEN * HIDDEN_SIZE) as f32, &device)?
            .reshape((1, SEQ_LEN, HIDDEN_SIZE))?
            .affine(0.01, -0.5)?;
        let mut packed_state = Tensor::zeros((2, HIDDEN_SIZE, CONV_WIDTH), DType::F32, &device)?;
        let mut first_state = Tensor::zeros((1, HIDDEN_SIZE, CONV_WIDTH), DType::F32, &device)?;
        let mut second_state = first_state.clone();
        for use_existing in [false, true] {
            let packed = conv.forward(
                &input,
                &mut packed_state,
                use_existing,
                Some(&[FIRST_SEQ_LEN, SEQ_LEN - FIRST_SEQ_LEN]),
            )?;
            let first = conv.forward(
                &input.narrow(1, 0, FIRST_SEQ_LEN)?,
                &mut first_state,
                use_existing,
                None,
            )?;
            let second = conv.forward(
                &input.narrow(1, FIRST_SEQ_LEN, SEQ_LEN - FIRST_SEQ_LEN)?,
                &mut second_state,
                use_existing,
                None,
            )?;
            assert_close(&packed, &Tensor::cat(&[first, second], 1)?)?;
            assert_close(
                &packed_state,
                &Tensor::cat(&[first_state.clone(), second_state.clone()], 0)?,
            )?;
        }
        Ok(())
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        let error = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            error < TOLERANCE,
            "maximum error {error} exceeds {TOLERANCE}"
        );
        Ok(())
    }
}
