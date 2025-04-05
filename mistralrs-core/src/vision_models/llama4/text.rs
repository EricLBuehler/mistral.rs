#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::MlpLayer,
    attention::SdpaParams,
    device_map::DeviceMapper,
    layers::{
        embedding, linear_no_bias, Activation, CausalMasker, Llama3RopeConfig,
        Llama3RotaryEmbedding, MatMul, RmsNorm, Sdpa,
    },
    layers_masker::PastKvLenCache,
    ops::{BitWiseOp, TopKLastDimOp, TopKOutput},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, KvCache, NormalCache, NormalLoadingMetadata,
    },
    serde_default_fn,
    utils::progress::NiceProgressBar,
};

serde_default_fn!(bool, word_emb_default, false);
serde_default_fn!(bool, use_flash_attn_default, false);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct TextConfig {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "use_flash_attn_default")]
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    pub floor_scale: Option<f32>,
    pub attn_scale: Option<f32>,
    pub attn_temperature_tuning: Option<f32>,
    pub use_qk_norm: bool,
    pub moe_layers: Option<Vec<usize>>,
    pub interleave_moe_layer_step: usize,
    pub intermediate_size_mlp: usize,
    pub num_local_experts: usize,
    pub expert_dim: usize,
    pub num_experts_per_tok: usize,
    pub attention_chunk_size: usize,
}

impl TextConfig {
    fn moe_layers(&self) -> Vec<usize> {
        self.moe_layers.clone().unwrap_or(
            (self.interleave_moe_layer_step - 1..self.num_hidden_layers)
                .step_by(self.interleave_moe_layer_step)
                .collect(),
        )
    }
}

struct Llama4L2Norm {
    eps: f64,
}

impl Llama4L2Norm {
    fn new(eps: f64) -> Self {
        Self { eps }
    }

    fn norm(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(
            &(x.sqr()?.mean_keepdim(D::Minus1)? + self.eps as f64)?
                .sqrt()?
                .recip()?,
        )
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.norm(&x.to_dtype(DType::F32)?)?.to_dtype(x.dtype())
    }
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
    norm: Option<Llama4L2Norm>,
    use_rope: bool,
    floor_scale: Option<f32>,
    attn_scale: Option<f32>,
    attn_temperature_tuning: Option<f32>,
}

impl CausalSelfAttention {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        layer_idx: usize,
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
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            size_q,
            size_in,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_proj"),
        )?;
        let use_rope = (layer_idx + 1) % 4 != 0;
        let norm = if cfg.use_qk_norm && use_rope {
            Some(Llama4L2Norm::new(1e-6))
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
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            rotary_emb: rope,
            max_seq_len: cfg.max_position_embeddings,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                use_flash_attn: cfg.use_flash_attn,
                softcap: None,
                softmax_scale: 1.0 / ((cfg.hidden_size / cfg.num_attention_heads) as f32).sqrt(),
                sliding_window: None,
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
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&x, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&x, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&x, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        q = q.reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?;
        k = k.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?;
        v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        (q, k) = if self.use_rope {
            self.rotary_emb.forward(&q, &k, seqlen_offsets)?
        } else {
            (q, k)
        };

        if let Some(qk_norm) = &self.norm {
            q = qk_norm.forward(&q)?;
            k = qk_norm.forward(&k)?;
        }

        if self.attn_temperature_tuning.is_some() && self.use_rope {
            let attn_scales = (((((position_ids.to_dtype(DType::F32)? + 1.)?
                / self.floor_scale.unwrap() as f64)?
                .floor()?
                + 1.0)?
                * self.attn_scale.unwrap() as f64)?
                + 1.0)?;

            let attn_scales = attn_scales.reshape((b_sz, seq_len, 1, 1))?;
            q = (q * attn_scales)?;
        }

        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;

        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                    // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    // Sanity check.
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask.clone().as_ref(),
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

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            y = y.to_dtype(t)?;
        }
        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&y, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Mlp {
    fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quantization_config: &Option<QuantizedConfig>,
        hidden_act: Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down: RowParallelLayer::new(
                intermediate_size,
                hidden_size,
                quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act: hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = self.gate.forward(&xs)?;
        let rhs = self.up.forward(&xs)?;
        let mut res = self.down.forward(&candle_nn::ops::mul_and_act(
            &lhs,
            &rhs,
            self.act.try_into()?,
        )?)?;
        if self.gate.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct TextExperts {
    gate_up_proj: Tensor,
    down_proj: Tensor,
    act: Activation,
    num_experts: usize,
    hidden_size: usize,
    expert_dim: usize,
}

impl TextExperts {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        quantization_config: &Option<QuantizedConfig>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate_up_proj: vb.get(
                (
                    cfg.num_local_experts,
                    cfg.hidden_size,
                    cfg.intermediate_size * 2,
                ),
                "gate_up_proj",
            )?,
            down_proj: vb.get(
                (
                    cfg.num_local_experts,
                    cfg.intermediate_size,
                    cfg.hidden_size,
                ),
                "down_proj",
            )?,
            act: cfg.hidden_act,
            num_experts: cfg.num_local_experts,
            hidden_size: cfg.hidden_size,
            expert_dim: cfg.intermediate_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.reshape((self.num_experts, (), self.hidden_size))?;
        let gate_up = xs.broadcast_matmul(&self.gate_up_proj)?;
        let gate = gate_up.narrow(D::Minus1, 0, self.expert_dim)?;
        let up = gate_up.narrow(D::Minus1, self.expert_dim, self.expert_dim)?;
        let next_states = (up * gate.apply(&self.act)?)?.broadcast_matmul(&self.down_proj)?;
        next_states.reshape(((), self.hidden_size))
    }
}

struct TextMoe {
    experts: TextExperts,
    shared_expert: Mlp,
    router: Linear,
    topk: usize,
    hidden_dim: usize,
    num_experts: usize,
}

impl TextMoe {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        quantization_config: &Option<QuantizedConfig>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let experts = TextExperts::new(vb.pp("experts"), cfg, quantization_config, comm)?;
        let router = linear_no_bias(cfg.hidden_size, cfg.num_local_experts, vb.pp("router"))?;
        let shared_expert = Mlp::new(
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
            hidden_dim: cfg.hidden_size,
            num_experts: cfg.num_local_experts,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let router_logits = self.router.forward(&xs)?.transpose(0, 1)?;
        let tokens_per_expert = bs * seq_len;

        let TopKOutput {
            values: router_top_value,
            indices: router_indices,
        } = router_logits.transpose(0, 1)?.topk(self.topk)?;
        let mut router_scores = (router_logits.transpose(0, 1)?.ones_like()? * f64::NEG_INFINITY)?
            .scatter_add(&router_indices, &router_top_value, 1)?
            .transpose(0, 1)?;

        // We do this to make sure we have -inf for non topK tokens before going through the !
        // Here we are just creating a tensor to index each and every single one of the hidden states.
        // This is an arange.
        let router_indices =
            (Tensor::ones(tokens_per_expert, DType::U32, xs.device())?.cumsum(0)? - 1.)?
                .reshape((router_scores.dim(0)?, ()))?;
        router_scores = candle_nn::ops::sigmoid(&router_scores.to_dtype(DType::F32)?)?
            .to_dtype(router_scores.dtype())?;

        let router_indices = router_indices.reshape(((), 1))?.repeat((1, hidden_dim))?;
        let mut routed_in = xs.gather(&router_indices, 0)?;
        // we gather inputs corresponding to each expert based on the router indices
        routed_in = (routed_in * router_scores.reshape(((), 1))?)?;
        let routed_out = self.experts.forward(&routed_in)?;
        let mut out = self.shared_expert.forward(&routed_out)?;

        // TODO TODO TODO DO EP!!!
        // now that we finished expert computation -> we scatter add because we gathered previously
        // we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
        // this scales a lot better if you do EP!
        out = out.scatter_add(&router_indices, &routed_out.reshape(((), hidden_dim))?, 0)?;
        Ok(out)
    }
}

enum MoeOrMlp {
    Mlp(Mlp),
    Moe(TextMoe),
}

impl MoeOrMlp {
    fn foward(&self, xs: &Tensor) -> Result<Tensor> {
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
    ) -> Result<Self> {
        let use_chunked_attention = (layer_idx + 1) % 4 != 0;
        let attn = CausalSelfAttention::new(
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            cfg,
            layer_idx,
            rope,
            paged_attn,
            comm,
        )?;
        let is_moe_layer = cfg.moe_layers().contains(&layer_idx);
        let ff = if is_moe_layer {
            let moe = TextMoe::new(vb.pp("feed_forward"), cfg, &cfg.quantization_config, comm)?;
            MoeOrMlp::Moe(moe)
        } else {
            let mlp = Mlp::new(
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
        attention_mask: &Option<Tensor>,
        chunked_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let mask = if self.use_chunked_attention {
            chunked_mask
        } else {
            attention_mask
        };
        let x = (self.attn.forward(
            &x,
            position_ids,
            mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )? + residual)?;
        let residual = &x;
        let x = (self.ff.foward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    kv_cache: crate::pipeline::EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    attention_chunk_size: usize,
}

impl Llama {
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

        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &None,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(wte.embeddings(), normal_loading_metadata.loading_isq)?,
                None,
            ))?
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
        let blocks: Vec<_> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .into_iter()
        .map(|i| {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(head_dim, device, None)
                        .expect("Failed to create PagedAttention"),
                ),
            };
            let comm = mapper.get_comm_for(i).unwrap();
            Block::new(
                vb_m.pp(format!("layers.{i}")),
                cfg,
                &*mapper,
                i,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                paged_attn,
                &comm,
            )
            .expect("Failed to load block.")
        })
        .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
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
                sliding_window: None,
                k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            },
            mapper,
            attention_chunk_size: cfg.attention_chunk_size,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.wte.forward(input_ids)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward_embeds(
            input_ids,
            self.wte.forward(input_ids)?,
            position_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        position_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut x = input_embeds;
        let cache = &mut self.kv_cache.normal().0;
        let cache_for_mask = metadata
            .as_ref()
            .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
            .unwrap_or(cache as &dyn PastKvLenCache);
        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            cache_for_mask,
            x.dtype(),
            self.blocks[0].attn.num_attention_heads,
        )?;
        // https://github.com/huggingface/transformers/blob/25b7f272347a93d6fb73cad126f6f6dc88e8ce89/src/transformers/models/llama4/modeling_llama4.py#L801
        let chunked_mask = if input_ids.dim(1)? > self.attention_chunk_size && mask.is_some() {
            let start = cache_for_mask.get_past_kv_len()?;
            let key_length = if start >= self.attention_chunk_size {
                self.attention_chunk_size + input_ids.dim(1)? - 1
            } else if start < self.attention_chunk_size
                && start + input_ids.dim(1)? > self.attention_chunk_size
            {
                start + input_ids.dim(1)?
            } else {
                self.attention_chunk_size
            };
            let end = start + key_length;

            let arange = Tensor::arange(start as i32, end as i32, input_ids.device())?;
            let block_pos = Tensor::abs(
                &(arange.unsqueeze(0)? / self.attention_chunk_size as f64)?
                    .broadcast_sub(&(arange.unsqueeze(1)? / self.attention_chunk_size as f64)?)?,
            )?;
            let token_pos = arange.unsqueeze(0)?.broadcast_sub(&arange.unsqueeze(1)?)?;
            let chunked_mask = block_pos.eq(0.)?.bitwise_and(&token_pos.le(0.)?)?;

            Some(chunked_mask.bitwise_and(&mask.as_ref().unwrap())?)
        } else {
            None
        };
        // PagedAttention prompt chunking
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, block_idx)?;
            x = block.forward(
                &x,
                position_ids,
                &mask.clone().map(|m| m.to_device(x.device()).unwrap()),
                &chunked_mask
                    .clone()
                    .map(|m| m.to_device(x.device()).unwrap()),
                seqlen_offsets,
                &mut cache[block_idx],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[block_idx].clone(), *metadata)),
                flash_params,
            )?;
        }
        let x = x.to_device(&self.device)?;
        let mut x = self.ln_f.forward(&x)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let xs = MatMul.qmethod_matmul(&x, &*self.lm_head)?;
        extract_logits(&xs, context_lens)
    }
}
