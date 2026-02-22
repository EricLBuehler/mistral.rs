#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::LayerNorm;
use mistralrs_quant::{
    ColumnParallelLayer, NonZeroOp, QuantMethod, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder,
};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        self, layer_norm, Activation, CausalMasker, MatMul, PhiRopeConfig, PhiRopeScalingConfig,
        PhiRotaryEmbedding, Sdpa,
    },
    layers_masker::{masked_fill, PastKvLenCache},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, word_emb_default, false);

// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) rope_scaling: Option<PhiRopeScalingConfig>,
    pub(crate) max_position_embeddings: usize,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) original_max_position_embeddings: usize,

    pub(crate) quantization_config: Option<QuantizedConfig>,
    pub(crate) lm_head_bias: bool,
    pub(crate) attention_bias: bool,
    pub(crate) num_local_experts: usize,
    pub(crate) router_jitter_noise: f64,
    #[serde(default = "word_emb_default")]
    pub(crate) tie_word_embeddings: bool,
}

impl From<Config> for PhiRopeConfig {
    fn from(val: Config) -> Self {
        PhiRopeConfig {
            rope_scaling: val.rope_scaling,
            max_position_embeddings: val.max_position_embeddings,
            original_max_position_embeddings: val.original_max_position_embeddings,
            rope_theta: val.rope_theta,
            head_dim: val.hidden_size / val.num_attention_heads,
            partial_rotary_factor: None,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<PhiRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
                sinks: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
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

        let (q, k, v) = if q_len != 1 {
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

        let (q, k) = self
            .rotary_emb
            .forward(&q, &k, seqlen_offsets, position_ids)?;

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
                    // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                    // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    // Sanity check.
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
            _ => {
                let (k, v) = kv_cache.append(&k, &v)?;

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

#[derive(Clone)]
struct Mlp {
    w1: Arc<dyn QuantMethod>,
    w2: Arc<dyn QuantMethod>,
    w3: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: ShardedVarBuilder, comm: &Arc<mistralrs_quant::Comm>) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;

        let w1 = ColumnParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w1"),
        )?;
        let w2 = RowParallelLayer::new(
            i_size,
            hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w2"),
        )?;
        let w3 = ColumnParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w3"),
        )?;

        Ok(Self {
            w1,
            w2,
            w3,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.w1.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let w1_out = MatMul.qmethod_matmul(&xs, &*self.w1)?;
        let w3_out = MatMul.qmethod_matmul(&xs, &*self.w3)?;
        let current_hidden_states = crate::ops::mul_and_act(&w1_out, &w3_out, self.act_fn)?;
        let mut res = MatMul.qmethod_matmul(&current_hidden_states, &*self.w2)?;
        if self.w1.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct MoeMlp {
    gate: candle_nn::Linear,
    experts: Vec<Mlp>,
    router_jitter_noise: f64,
    num_experts: usize,
}

impl MoeMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_experts = cfg.num_local_experts;
        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate").set_device(layer_device),
        )?;

        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(Mlp::new(cfg, experts_vb.pp(i), comm)?);
        }

        Ok(Self {
            gate,
            experts,
            router_jitter_noise: cfg.router_jitter_noise,
            num_experts,
        })
    }

    fn sparsemixer(&self, scores: &Tensor, jitter_eps: f64) -> Result<(Tensor, Tensor)> {
        // Compute mask for sparsity
        let selected_experts = scores.argmax_keepdim(D::Minus1)?;
        let mask_logits_threshold = scores.gather(&selected_experts, D::Minus1)?;
        let factor = scores.abs()?.broadcast_minimum(&mask_logits_threshold)?;
        let mask_logits_threshold = mask_logits_threshold
            .broadcast_sub(scores)?
            .broadcast_div(&factor)?
            .gt(2. * jitter_eps)?;

        // Apply mask
        let masked_gates = masked_fill(scores, &mask_logits_threshold, f64::NEG_INFINITY)?;

        // Compute scores
        let masked_gates = candle_nn::ops::softmax_last_dim(&masked_gates)?;
        let multiplier = masked_gates.gather(&selected_experts, D::Minus1)?;

        // Mask out first expert
        let masked_scores = scores.scatter_add(
            &selected_experts
                .broadcast_as(scores.shape())?
                .contiguous()?,
            &(scores.ones_like()? * f64::NEG_INFINITY)?,
            D::Minus1,
        )?;

        // Compute mask for sparsity
        let selected_experts_top2 = masked_scores.argmax_keepdim(D::Minus1)?;
        let mask_logits_threshold = masked_scores.gather(&selected_experts_top2, D::Minus1)?;
        let factor = scores.abs()?.broadcast_minimum(&mask_logits_threshold)?;
        let mask_logits_threshold = mask_logits_threshold
            .broadcast_sub(scores)?
            .broadcast_div(&factor)?
            .gt(2. * jitter_eps)?;

        // Apply mask
        let masked_gates_top2 =
            masked_fill(&masked_scores, &mask_logits_threshold, f64::NEG_INFINITY)?;
        let masked_gates_top2 = candle_nn::ops::softmax_last_dim(&masked_gates_top2)?;
        let multiplier_top2 = masked_gates_top2.gather(&selected_experts_top2, D::Minus1)?;

        let multiplier = Tensor::cat(&[multiplier, multiplier_top2], D::Minus1)?;
        let selected_experts = Tensor::cat(&[selected_experts, selected_experts_top2], D::Minus1)?;

        Ok((multiplier, selected_experts))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq, hidden) = xs.dims3()?;
        let xs = xs.reshape(((), hidden))?;
        let xs_dev = xs.device();
        let xs = xs.to_device(&Device::Cpu)?;

        // Sparse MoE block accumulates hidden states on CPU, but MLP and gate weights are untouched (maybe on GPU)

        let router_logits = self
            .gate
            .forward(&xs.to_device(xs_dev)?)?
            .to_device(&Device::Cpu)?;
        let (routing_weights, selected_experts) = self.sparsemixer(
            &router_logits.to_device(&Device::Cpu)?,
            self.router_jitter_noise,
        )?;

        let mut final_hidden_states = Tensor::zeros((bs * seq, hidden), xs.dtype(), xs_dev)?;

        // One hot encode the selected experts to create an expert mask
        // this will be used to easily index which expert to activate
        let experts_mask =
            candle_nn::encoding::one_hot(selected_experts, self.num_experts, 1u8, 0u8)?
                .permute((2, 1, 0))?;

        // Loop over all avail experts in the model and perform the computation on each expert
        for expert_idx in 0..self.num_experts {
            let expert = &self.experts[expert_idx];
            let expert_mask = experts_mask.i(expert_idx)?;
            assert_eq!(expert_mask.rank(), 2);
            let nonzero_mask = expert_mask.contiguous()?.nonzero()?;
            let idx = nonzero_mask.i((.., 0))?;
            let top_x = nonzero_mask.i((.., 1))?;

            if top_x.dim(0)? == 0 {
                continue;
            }

            // Index the correct hidden staters and compute the expert hidden state
            // for the current expert, we need to make sure to multiply the output hidden
            // states by `routing_weights` on the corresponding tokens (top-1, top-2)
            let current_state = xs.index_select(&top_x, 0)?.reshape((1, (), hidden))?;
            let current_routing_weights = routing_weights
                .index_select(&top_x, 0)?
                .gather(&idx.unsqueeze(1)?.contiguous()?, 1)?;
            let exp_out = expert
                .forward(&current_state.to_device(xs_dev)?)?
                .to_device(&Device::Cpu)?;

            let current_hidden_states = exp_out.broadcast_mul(&current_routing_weights)?;

            final_hidden_states = final_hidden_states.index_add(
                &top_x.contiguous()?.to_device(xs_dev)?,
                &current_hidden_states
                    .squeeze(0)?
                    .to_dtype(xs.dtype())?
                    .to_device(xs_dev)?,
                0,
            )?;
        }

        final_hidden_states
            .reshape((bs, seq, hidden))?
            .to_device(xs_dev)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeMlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
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
            paged_attn,
            comm,
        )?;
        let mlp = MoeMlp::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("block_sparse_moe"), loading_isq),
            mapper
                .device_for(layer_idx, false)
                .cloned()
                .unwrap_or(real_device),
            comm,
        )?;
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = layer_norm(
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
        position_ids: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            position_ids,
            kv_cache,
            metadata,
            flash_params,
        )?;
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
    norm: LayerNorm,
    lm_head: Arc<dyn QuantMethod>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: Option<usize>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
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
        let vb_m = vb.pp("model");

        let embed_tokens = layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(PhiRotaryEmbedding::new(vb.dtype(), cfg.clone(), device)?),
            );
        }
        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim(), device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb.clone(),
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
        let norm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                cfg.lm_head_bias,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            unreachable!()
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::new_sliding(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
                cfg.sliding_window,
            )),
            max_seq_len: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: cfg.sliding_window,
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
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.sliding_window,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        // PagedAttention prompt chunking
        let attention_mask = attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                position_ids,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                flash_params,
            )?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
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
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            for expert in &mut layer.mlp.experts {
                tensors.push((&mut expert.w1, Some(i)));
                tensors.push((&mut expert.w2, Some(i)));
                tensors.push((&mut expert.w3, Some(i)));
            }
        }
        (tensors, &*self.mapper)
    }
    fn get_layers_moe_experts_only(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            for expert in &mut layer.mlp.experts {
                tensors.push((&mut expert.w1, Some(i)));
                tensors.push((&mut expert.w2, Some(i)));
                tensors.push((&mut expert.w3, Some(i)));
            }
        }
        (tensors, &*self.mapper)
    }

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
        }

        uvb.to_safetensors()
    }

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
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

            let uvb_attn = uvb_l.pp("self_attn");
            uvb_attn.pp("q_proj").add(&layer.self_attn.q_proj);
            uvb_attn.pp("k_proj").add(&layer.self_attn.k_proj);
            uvb_attn.pp("v_proj").add(&layer.self_attn.v_proj);
            uvb_attn.pp("o_proj").add(&layer.self_attn.o_proj);
        }

        Some(uvb.to_safetensors())
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            &position_ids,
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
        unimplemented!()
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
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {}
