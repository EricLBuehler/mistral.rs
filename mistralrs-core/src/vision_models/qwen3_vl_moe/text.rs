#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module};
use mistralrs_quant::{
    ColumnParallelLayer, MatMul, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use super::config::TextConfig;
use crate::{
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{self, Activation, Qwen3VLRotaryEmbedding, RmsNorm, Sdpa},
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

// Dense MLP for non-MoE layers
#[derive(Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        i_size: usize,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;

        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = ColumnParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            i_size,
            hidden_size,
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
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = MatMul
            .qmethod_matmul(&xs, &*self.gate_proj)?
            .apply(&self.act_fn)?;
        let rhs = MatMul.qmethod_matmul(&xs, &*self.up_proj)?;
        let mut res = MatMul.qmethod_matmul(&(lhs * rhs)?, &*self.down_proj)?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

/// MoE MLP layer for Qwen3 VL MoE
struct MoeMlp {
    gate: Linear,
    experts: MoEExperts,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoeMlp {
    fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        // Load gate
        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.num_experts,
            vb.pp("gate").set_device(layer_device.clone()),
        )?;

        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
        };

        // Load experts with automatic backend selection
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
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        // Compute routing weights
        let router_logits = self.gate.forward(&xs_flat)?;
        let routing_weights =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        // Get top-k experts
        let topk_ids = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        // Forward through experts (is_prefill determined internally based on seq_len)
        let ys = self.experts.forward(xs, topk_weights, &topk_ids)?;

        ys.reshape((b_size, seq_len, hidden_dim))
    }

    fn gate(&self) -> &Linear {
        &self.gate
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        self.experts.get_isq_layers()
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
    rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        cfg: &TextConfig,
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
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * cfg.head_dim,
            &cfg.quantization_config,
            false,
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
            num_kv_heads * cfg.head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * cfg.head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * cfg.head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let q_norm = RmsNorm::new(
            cfg.head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = RmsNorm::new(
            cfg.head_dim,
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
            head_dim: cfg.head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        cos_sin: &(Tensor, Tensor),
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

        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        self.rotary_emb.forward(cos_sin, &mut q, &mut k)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

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
                let (cache_k, cache_v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &cache_k.contiguous()?,
                    &cache_v.contiguous()?,
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
        let mut res = self.o_proj.forward(&attn_output)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;

        // Check if this layer should be MoE or dense MLP
        let is_moe = !cfg.mlp_only_layers.contains(&layer_idx)
            && (cfg.num_experts > 0 && (layer_idx + 1).is_multiple_of(cfg.decoder_sparse_step));

        let mlp = if is_moe {
            let vb = mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq);
            let layer_device = mapper
                .device_for(layer_idx, false)
                .cloned()
                .unwrap_or(real_device.clone());

            MoeOrMlp::Moe(MoeMlp::new(cfg, vb, layer_device, comm, loading_isq)?)
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                cfg,
                mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
                comm,
                cfg.intermediate_size,
            )?)
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

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            cos_sin,
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

pub struct Qwen3VLMoETextModel {
    embed_tokens: Embedding,
    pub(super) norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    lm_head: Arc<dyn QuantMethod>,
    pub(super) cache: EitherCache,
    pub(super) cfg: ModelConfigMetadata,
    pub(super) device: Device,
    pub(super) dtype: DType,
    pub(super) max_seq_len: usize,
}

impl Qwen3VLMoETextModel {
    pub fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        tie: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        // Support both HuggingFace naming (model.language_model.*) and MLX naming (language_model.model.*)
        let vb_m = if vb.contains_tensor("language_model.model.embed_tokens.weight") {
            vb.pp("language_model").pp("model")
        } else {
            vb.pp("model").pp("language_model")
        };

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
                Arc::new(Qwen3VLRotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    cfg.head_dim,
                    device,
                    cfg.rope_scaling.mrope_section.clone(),
                )?),
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
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim, device, None)?)
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
                &comm,
                normal_loading_metadata.real_device.clone(),
            )
        })?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !tie {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
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
        Ok(Self {
            embed_tokens,
            norm,
            layers,
            lm_head,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: normal_loading_metadata.real_device.clone(),
            dtype: vb.dtype(),
            mapper,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let cos_sin = self.layers[0]
            .self_attn
            .rotary_emb
            .compute_cos_sin(position_ids, xs.dtype())?;

        let attention_mask = DeviceMappedMask::new(attention_mask.cloned(), &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask.as_ref().map(|m| m.get(xs.device())),
                &cos_sin,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta)),
                flash_params,
            )?;

            // Integrate DeepStack visual features when provided.
            if let (Some(visual_pos_masks), Some(deepstack)) =
                (visual_pos_masks, deepstack_visual_embeds)
            {
                if i < deepstack.len() {
                    xs = self.deepstack_process(xs, visual_pos_masks, &deepstack[i])?;
                }
            }
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let mut xs = extract_logits(&xs, context_lens)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        self.lm_head.forward(&xs)
    }

    /// Matches transformers `_deepstack_process`:
    ///   hidden_states = hidden_states.clone()
    ///   hidden_states[visual_pos_masks, :] += visual_embeds
    fn deepstack_process(
        &self,
        hidden_states: Tensor,
        visual_pos_masks: &Tensor,
        visual_embeds: &Tensor,
    ) -> Result<Tensor> {
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let visual_embeds = visual_embeds.to_device(device)?.to_dtype(dtype)?;

        let (batch, seq, hidden) = hidden_states.dims3()?;
        let total = batch * seq;
        let hidden_flat = hidden_states.reshape((total, hidden))?;

        // Get flat boolean mask and find nonzero positions
        let mask_flat: Vec<f32> = visual_pos_masks
            .to_device(device)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let indices: Vec<u32> = mask_flat
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, _)| i as u32)
            .collect();

        if indices.is_empty() {
            return Ok(hidden_states);
        }
        if indices.len() != visual_embeds.dim(0)? {
            candle_core::bail!(
                "Mismatch between DeepStack visual embeds ({}) and mask positions ({})",
                visual_embeds.dim(0)?,
                indices.len()
            );
        }

        let idx = Tensor::from_vec(indices, (visual_embeds.dim(0)?,), device)?;
        let idx_expanded = idx.unsqueeze(1)?.repeat((1, hidden))?;
        let result = hidden_flat.scatter_add(&idx_expanded, &visual_embeds, 0)?;
        result.reshape((batch, seq, hidden))
    }
}

impl IsqModel for Qwen3VLMoETextModel {
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
            match &mut layer.mlp {
                MoeOrMlp::Mlp(mlp) => {
                    tensors.push((&mut mlp.gate_proj, Some(i)));
                    tensors.push((&mut mlp.up_proj, Some(i)));
                    tensors.push((&mut mlp.down_proj, Some(i)));
                }
                MoeOrMlp::Moe(moe) => {
                    for layer in moe.get_isq_layers() {
                        tensors.push((layer, Some(i)));
                    }
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_lm = uvb.pp("model").pp("language_model");
        uvb_lm.pp("embed_tokens").add(&self.embed_tokens);
        uvb_lm.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_lm.pp("layers").pp(layer_idx);
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
                uvb_l.pp("mlp").pp("gate").add(moe.gate());
            }
        }

        uvb.to_safetensors()
    }
}
