#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! HYV3 causal language model support.
//!
//! The public checkpoints use Q/K RMSNorm, sigmoid MoE routing with expert
//! bias, one shared expert, and an fp32 output head option.

use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::moe::{MoEExperts, MoEExpertsConfig};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{self, embedding, Activation, CausalMasker, RmsNorm, RotaryEmbedding, Sdpa},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(bool, default_false, false);
serde_default_fn!(f64, default_router_scaling_factor, 1.0);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RopeParameters {
    pub(crate) rope_theta: f64,
    #[serde(default)]
    pub(crate) rope_type: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub(crate) architectures: Vec<String>,
    #[serde(default)]
    pub(crate) model_type: Option<String>,
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    #[serde(default)]
    pub(crate) rope_theta: Option<f64>,
    #[serde(default)]
    pub(crate) rope_parameters: Option<RopeParameters>,
    pub(crate) head_dim: Option<usize>,
    #[serde(default)]
    pub(crate) quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) moe_intermediate_size: usize,
    #[serde(default)]
    pub(crate) expert_hidden_dim: Option<usize>,
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) num_shared_experts: usize,
    pub(crate) first_k_dense_replace: usize,
    #[serde(default)]
    pub(crate) num_nextn_predict_layers: usize,
    #[serde(default = "default_true")]
    pub(crate) qk_norm: bool,
    #[serde(default = "default_true")]
    pub(crate) route_norm: bool,
    #[serde(default = "default_false")]
    pub(crate) moe_router_use_sigmoid: bool,
    #[serde(default = "default_false")]
    pub(crate) moe_router_enable_expert_bias: bool,
    #[serde(default = "default_router_scaling_factor")]
    pub(crate) router_scaling_factor: f64,
    #[serde(default)]
    pub(crate) output_router_logits: bool,
    #[serde(default)]
    pub(crate) enable_attention_fp32_softmax: bool,
    #[serde(default)]
    pub(crate) enable_lm_head_fp32: bool,
    #[serde(default)]
    pub(crate) enable_moe_fp32_combine: bool,
    #[serde(default)]
    pub(crate) use_grouped_mm: bool,
}

impl Config {
    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn rope_theta(&self) -> Result<f64> {
        self.rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .map(|params| params.rope_theta)
            })
            .ok_or_else(|| candle_core::Error::msg("HyV3 config is missing rope theta"))
    }

    fn validate(&self) -> Result<()> {
        if self.model_type.as_deref() != Some("hy_v3")
            || !self
                .architectures
                .iter()
                .any(|arch| arch == "HYV3ForCausalLM")
        {
            candle_core::bail!("HyV3 loader expects HYV3ForCausalLM / model_type=hy_v3");
        }
        if !self.qk_norm {
            candle_core::bail!("HyV3 checkpoint support currently requires qk_norm=true");
        }
        if !self.moe_router_use_sigmoid {
            candle_core::bail!("HyV3 checkpoint support currently requires sigmoid routing");
        }
        if !self.moe_router_enable_expert_bias {
            candle_core::bail!("HyV3 checkpoint support currently requires expert_bias routing");
        }
        if self.use_grouped_mm {
            candle_core::bail!("HyV3 grouped-mm checkpoint path is not implemented");
        }
        if self
            .rope_parameters
            .as_ref()
            .and_then(|params| params.rope_type.as_deref())
            .is_some_and(|rope_type| rope_type != "default")
        {
            candle_core::bail!("HyV3 loader supports only default RoPE");
        }
        if self
            .expert_hidden_dim
            .is_some_and(|dim| dim != self.moe_intermediate_size)
        {
            candle_core::bail!("HyV3 expert_hidden_dim must match moe_intermediate_size");
        }
        Ok(())
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
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
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let q_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("k_norm"), false),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
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
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
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

        let rope_positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
        (q, k) = self.rotary_emb.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            rope_positions,
        )?;
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
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(attention_mask, AttentionMask::None));
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

        attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn_output)
    }
}

#[derive(Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        i_size: usize,
    ) -> Result<Self> {
        let gate_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            i_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_out = self.gate_proj.forward(xs)?;
        let up_out = self.up_proj.forward(xs)?;
        let current_hidden_states = crate::ops::mul_and_act(&gate_out, &up_out, self.act_fn)?;
        self.down_proj.forward(&current_hidden_states)
    }
}

struct MoeMlp {
    gate: Linear,
    expert_bias: Tensor,
    shared_mlp: Mlp,
    experts: MoEExperts,
    num_experts_per_tok: usize,
    route_norm: bool,
    router_scaling_factor: f32,
}

impl MoeMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.num_experts,
            vb.pp("router").pp("gate").set_device(layer_device.clone()),
        )?;
        let expert_bias = vb
            .clone()
            .set_device(layer_device.clone())
            .get_with_hints_dtype(
                (cfg.num_experts,),
                "expert_bias",
                Default::default(),
                DType::F32,
            )?;
        let shared_mlp = Mlp::new(
            cfg,
            vb.pp("shared_mlp"),
            comm,
            cfg.moe_intermediate_size * cfg.num_shared_experts,
        )?;
        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
        };
        let experts = MoEExperts::new(
            &moe_cfg,
            vb,
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        Ok(Self {
            gate,
            expert_bias,
            shared_mlp,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            route_norm: cfg.route_norm,
            router_scaling_factor: cfg.router_scaling_factor as f32,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        // HYV3 applies expert bias for top-k selection only. The routed weights
        // are gathered from the original sigmoid scores and normalized below.
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Sigmoid,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: self.route_norm,
                norm_min: 1e-20,
                output_scale: self.router_scaling_factor,
                logit_clip: None,
            },
            Some(&self.expert_bias),
            None,
        )?;

        let routed = self.experts.forward(xs, topk.values, &topk.indices)?;
        let shared = self.shared_mlp.forward(xs)?;
        (routed + shared)?.reshape((b_size, seq_len, hidden_dim))
    }

    fn gate(&self) -> &Linear {
        &self.gate
    }
}

enum MoeOrMlp {
    Moe(MoeMlp),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::Moe(m) => m.forward(xs),
        }
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
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

        let mlp_vb = mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq);
        let mlp = if layer_idx < cfg.first_k_dense_replace {
            MoeOrMlp::Mlp(Mlp::new(cfg, mlp_vb, comm, cfg.intermediate_size)?)
        } else {
            let layer_device = mapper
                .device_for(layer_idx, false)
                .cloned()
                .unwrap_or(real_device);
            MoeOrMlp::Moe(MoeMlp::new(cfg, mlp_vb, layer_device, comm, loading_isq)?)
        };
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

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    lm_head_fp32: bool,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
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
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let head_dim = cfg.head_dim();
        let rope_theta = cfg.rope_theta()?;
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?),
            );
        }

        let load_in_parallel =
            !(normal_loading_metadata.real_device.is_metal() && cfg.quantization_config.is_none());
        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .run(load_in_parallel, |layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(head_dim, device, None)
                        .expect("PagedAttention creation failed"),
                ),
            };
            let comm = mapper
                .get_comm_for(layer_idx)
                .expect("Failed to get comm for layer");
            DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                normal_loading_metadata.real_device.clone(),
                &comm,
            )
        })?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        // Some HYV3 checkpoints store the output projection in fp32.
        let lm_head_vb = if cfg.enable_lm_head_fp32 {
            vb_lm_head.set_dtype(DType::F32)
        } else {
            vb_lm_head
        };
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(lm_head_vb, normal_loading_metadata.loading_isq),
            )?
        } else {
            let embeddings = mapper.cast_nm_device(
                embed_tokens.embeddings(),
                normal_loading_metadata.loading_isq,
            )?;
            let embeddings = if cfg.enable_lm_head_fp32 {
                embeddings.to_dtype(DType::F32)?
            } else {
                embeddings
            };
            ReplicatedLayer::from_linear(
                candle_nn::Linear::new(embeddings, None),
                mapper.set_nm_device(lm_head_vb, normal_loading_metadata.loading_isq),
            )?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_fp32: cfg.enable_lm_head_fp32,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: None,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward_embeds(input_ids, self.embed_tokens.forward(input_ids)?, ctx)
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut xs = input_embeds;
        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(&xs, &attention_mask.get(xs.device()), &mut cache[i], ctx, i)?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        let xs = if self.lm_head_fp32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs
        };
        self.lm_head.forward(&xs)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("self_attn")
                .pp("q_norm")
                .add(&layer.self_attn.q_norm);
            uvb_l
                .pp("self_attn")
                .pp("k_norm")
                .add(&layer.self_attn.k_norm);
            if let MoeOrMlp::Moe(moe) = &layer.mlp {
                uvb_l.pp("mlp").pp("router").pp("gate").add(moe.gate());
                uvb_l
                    .pp("mlp")
                    .add_tensor("expert_bias", moe.expert_bias.clone());
            }
        }

        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
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
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
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
        &self.cfg
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_rope_parameters_format() {
        let cfg: Config = serde_json::from_str(
            r#"{
                "architectures": ["HYV3ForCausalLM"],
                "model_type": "hy_v3",
                "vocab_size": 120832,
                "hidden_size": 4096,
                "intermediate_size": 13312,
                "num_hidden_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "hidden_act": "silu",
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-5,
                "rope_parameters": {
                    "rope_theta": 11158840.0,
                    "rope_type": "default"
                },
                "head_dim": 128,
                "tie_word_embeddings": false,
                "moe_intermediate_size": 1536,
                "expert_hidden_dim": 1536,
                "num_experts": 192,
                "num_experts_per_tok": 8,
                "num_shared_experts": 1,
                "first_k_dense_replace": 1,
                "num_nextn_predict_layers": 1,
                "qk_norm": true,
                "route_norm": true,
                "moe_router_use_sigmoid": true,
                "moe_router_enable_expert_bias": true,
                "router_scaling_factor": 2.826,
                "output_router_logits": true,
                "enable_lm_head_fp32": true,
                "use_grouped_mm": false
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.num_hidden_layers, 80);
        assert_eq!(cfg.num_nextn_predict_layers, 1);
        assert_eq!(cfg.rope_theta().unwrap(), 11158840.0);
        cfg.validate().unwrap();
    }
}
