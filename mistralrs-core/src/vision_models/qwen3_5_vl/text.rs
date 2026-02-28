#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Qwen3.5 MoE text backbone (hybrid attention: FullAttention + GatedDeltaNet, MoE feed-forward).
//! Used as the language model inside the multimodal Qwen3.5 MoE vision model.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{HybridCache, HybridCacheConfig, HybridLayerType},
    layers::{embedding, GemmaRmsNorm, RotaryEmbedding},
    layers_masker::PastKvLenCache,
    models::{
        deltanet::{DeltaNetConfig, GatedDeltaNet, GdnLayerCache, GdnProjection},
        qwen3_next::{FullAttention, Mlp, SparseMoeBlock},
    },
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::TextConfig;

// ====================== DeltaNetConfig impl ======================

struct TextConfigAdapter<'a>(&'a TextConfig);

impl<'a> DeltaNetConfig for TextConfigAdapter<'a> {
    fn hidden_size(&self) -> usize {
        self.0.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.0.rms_norm_eps
    }
    fn linear_num_key_heads(&self) -> usize {
        self.0.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.0.linear_num_value_heads
    }
    fn linear_key_head_dim(&self) -> usize {
        self.0.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.0.linear_value_head_dim
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.0.linear_conv_kernel_dim
    }
    fn quantization_config(&self) -> &Option<mistralrs_quant::QuantizedConfig> {
        &self.0.quantization_config
    }
}

// ====================== Layer types ======================

#[derive(Debug, Clone)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

impl TextConfig {
    pub fn layer_types_parsed(&self) -> Vec<LayerType> {
        self.layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" | "attention" => LayerType::FullAttention,
                _ => LayerType::LinearAttention,
            })
            .collect()
    }
}

// ====================== Feed-forward ======================

enum FeedForward {
    Dense(Mlp),
    MoE(SparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::MoE(moe) => moe.forward(x),
        }
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match self {
            FeedForward::Dense(mlp) => mlp.get_isq_layers(),
            FeedForward::MoE(moe) => moe.get_isq_layers(),
        }
    }
}

// ====================== Decoder Layer ======================

enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    ffn: FeedForward,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: Option<&mut KvCache>,
        gdn_cache: Option<&mut GdnLayerCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;

        let attn_out = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => {
                let kv_cache = kv_cache.expect("FullAttention needs kv_cache");
                attn.forward(
                    &x,
                    attention_mask,
                    seqlen_offsets,
                    kv_cache,
                    metadata,
                    flash_params,
                )?
            }
            LayerImpl::LinearAttention(gdn) => {
                let gdn_cache = gdn_cache.expect("LinearAttention needs gdn_cache");
                gdn.forward(&x, gdn_cache)?
            }
        };

        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        ffn_out + residual
    }
}

// ====================== Local hybrid cache ======================

enum LocalLayerCache {
    Attention(KvCache),
    LinearAttention(GdnLayerCache),
}

struct LocalHybridCache {
    caches: Vec<LocalLayerCache>,
}

impl LocalHybridCache {
    fn new(
        layer_types: &[LayerType],
        cfg: &TextConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let adapter = TextConfigAdapter(cfg);
        let mut caches = Vec::with_capacity(layer_types.len());
        for lt in layer_types {
            match lt {
                LayerType::FullAttention => {
                    caches.push(LocalLayerCache::Attention(KvCache::new_normal(
                        2,
                        cfg.max_position_embeddings,
                        HybridCache::CACHE_GROW_SIZE,
                    )));
                }
                LayerType::LinearAttention => {
                    caches.push(LocalLayerCache::LinearAttention(GdnLayerCache::new(
                        &adapter, dtype, device,
                    )?));
                }
            }
        }
        Ok(Self { caches })
    }

    fn seqlen(&self) -> usize {
        for cache in &self.caches {
            if let LocalLayerCache::Attention(kv) = cache {
                return kv.current_seq_len();
            }
        }
        0
    }
}

impl PastKvLenCache for LocalHybridCache {
    fn get_past_kv_len(&self) -> Result<usize> {
        Ok(self.seqlen())
    }
}

// ====================== Text Model ======================

pub struct Qwen3_5VLTextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: GemmaRmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    local_cache: Arc<Mutex<LocalHybridCache>>,
    pub(super) cache: EitherCache,
    pub(super) device: Device,
    pub(super) dtype: DType,
    pub(super) max_seq_len: usize,
    pub(super) cfg: ModelConfigMetadata,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub(super) sliding_window: Option<usize>,
}

impl Qwen3_5VLTextModel {
    pub fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        tie: bool,
        is_moe: bool,
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

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
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

        let norm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let layer_types = cfg.layer_types_parsed();
        let adapter = TextConfigAdapter(cfg);
        let partial_rotary_factor = cfg.rope_parameters.partial_rotary_factor;
        let rot_dim = (cfg.head_dim as f64 * partial_rotary_factor) as usize;

        // Build RotaryEmbedding for attention layers
        let mut ropes = HashMap::new();
        for (i, layer_type) in layer_types.iter().enumerate().take(cfg.num_hidden_layers) {
            if matches!(layer_type, LayerType::FullAttention) {
                let device = mapper
                    .device_for(i, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(device.location())
                {
                    let rope = RotaryEmbedding::new_partial(
                        cfg.rope_theta() as f32,
                        rot_dim,
                        cfg.max_position_embeddings,
                        device,
                        false,
                        vb.dtype(),
                    )?;
                    e.insert(Arc::new(rope));
                }
            }
        }

        let num_full = layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::FullAttention))
            .count();
        let num_linear = layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::LinearAttention))
            .count();
        tracing::info!(
            "Qwen3.5-MoE: {} full attention layers, {} linear attention (GDN) layers",
            num_full,
            num_linear
        );

        // Build layers
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        ) {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let comm = mapper.get_comm_for(i)?;
            let vb_layer = vb_l.pp(i);

            let dummy_moe_intermediate_size = if is_moe { cfg.moe_intermediate_size } else { 0 };
            let dummy_shared_expert_intermediate_size = if is_moe {
                cfg.shared_expert_intermediate_size
            } else {
                0
            };
            let dummy_num_experts = if is_moe { cfg.num_experts } else { 0 };
            let dummy_num_experts_per_tok = if is_moe { cfg.num_experts_per_tok } else { 0 };

            let dummy_cfg = crate::models::qwen3_next::Config {
                vocab_size: cfg.vocab_size,
                hidden_size: cfg.hidden_size,
                intermediate_size: cfg.intermediate_size.unwrap_or(0),
                num_hidden_layers: cfg.num_hidden_layers,
                num_attention_heads: cfg.num_attention_heads,
                num_key_value_heads: cfg.num_key_value_heads,
                hidden_act: cfg.activation(),
                max_position_embeddings: cfg.max_position_embeddings,
                rms_norm_eps: cfg.rms_norm_eps,
                rope_theta: cfg.rope_theta(),
                head_dim: cfg.head_dim,
                partial_rotary_factor: cfg.rope_parameters.partial_rotary_factor,
                linear_conv_kernel_dim: cfg.linear_conv_kernel_dim,
                linear_key_head_dim: cfg.linear_key_head_dim,
                linear_value_head_dim: cfg.linear_value_head_dim,
                linear_num_key_heads: cfg.linear_num_key_heads,
                linear_num_value_heads: cfg.linear_num_value_heads,
                decoder_sparse_step: 1, // unused
                moe_intermediate_size: dummy_moe_intermediate_size,
                shared_expert_intermediate_size: dummy_shared_expert_intermediate_size,
                num_experts_per_tok: dummy_num_experts_per_tok,
                num_experts: dummy_num_experts,
                norm_topk_prob: cfg.norm_topk_prob,
                mlp_only_layers: cfg.mlp_only_layers.clone(),
                full_attention_interval: 4, // unused
                tie_word_embeddings: false,
                quantization_config: cfg.quantization_config.clone(),
            };

            let layer_impl = match &layer_types[i] {
                LayerType::FullAttention => {
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
                    LayerImpl::FullAttention(FullAttention::load(
                        vb_layer.clone(),
                        &dummy_cfg,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        rotary_emb,
                        paged_attn,
                        &comm,
                    )?)
                }
                LayerType::LinearAttention => {
                    LayerImpl::LinearAttention(GatedDeltaNet::load_qwen3_5(
                        vb_layer.clone(),
                        &adapter,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        &comm,
                    )?)
                }
            };

            let input_layernorm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(i, vb_layer.pp("input_layernorm"), false),
            )?;
            let post_attention_layernorm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(i, vb_layer.pp("post_attention_layernorm"), false),
            )?;

            let is_moe_layer = is_moe && !cfg.mlp_only_layers.contains(&i);
            let ffn = if !is_moe_layer {
                FeedForward::Dense(Mlp::new(
                    mapper.set_device(i, vb_layer.pp("mlp"), normal_loading_metadata.loading_isq),
                    cfg.hidden_size,
                    cfg.dense_intermediate_size(),
                    &cfg.quantization_config,
                    cfg.activation(),
                    &comm,
                )?)
            } else {
                FeedForward::MoE(SparseMoeBlock::new(
                    &dummy_cfg,
                    mapper.set_device(i, vb_layer.pp("mlp"), normal_loading_metadata.loading_isq),
                    &*mapper,
                    i,
                    normal_loading_metadata.loading_isq,
                    &comm,
                    normal_loading_metadata.real_device.clone(),
                )?)
            };

            layers.push(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                ffn,
            });
        }

        // Create local hybrid cache
        let local_cache = Arc::new(Mutex::new(LocalHybridCache::new(
            &layer_types,
            cfg,
            &normal_loading_metadata.real_device,
            vb_m.dtype(),
        )?));

        // Create pipeline hybrid cache config
        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Mamba,
            })
            .collect();

        let conv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim;

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: cfg.max_position_embeddings,
            max_num_seqs: 1,
            mamba_conv_dim: conv_dim,
            mamba_d_conv: cfg.linear_conv_kernel_dim,
            mamba_n_heads: cfg.linear_num_value_heads,
            mamba_head_dim: cfg.linear_key_head_dim,
            mamba_d_state: cfg.linear_value_head_dim,
        };

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(
                hybrid_cache_config,
                vb_m.dtype(),
                &normal_loading_metadata.real_device,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("Failed to create hybrid cache: {}", e))
            })?,
        ));

        let num_attention_heads = cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            local_cache,
            cache: EitherCache::Hybrid(pipeline_cache),
            device: normal_loading_metadata.real_device.clone(),
            dtype: vb.dtype(),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: num_attention_heads,
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            sliding_window: cfg.sliding_window,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let mut local_cache = self.local_cache.lock().unwrap();

        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;

            let (kv_cache, gdn_cache) = match &mut local_cache.caches[i] {
                LocalLayerCache::Attention(kv) => (Some(kv), None),
                LocalLayerCache::LinearAttention(gdn) => {
                    // Reset GDN cache on first chunk
                    if metadata
                        .as_ref()
                        .map(|(_, meta)| meta.is_first_prompt_chunk)
                        .unwrap_or(true)
                    {
                        // Check if seqlen_offset is 0 via cache
                        if gdn.seqlen_offset == 0 || xs.dim(1)? > 1 {
                            gdn.reset()?;
                        }
                    }
                    (None, Some(gdn))
                }
            };

            xs = layer.forward(
                &xs,
                &attention_mask.as_ref().map(|m| m.get(xs.device()).clone()),
                seqlen_offsets,
                kv_cache,
                gdn_cache,
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

    /// DeepStack: hidden_states[visual_mask] += visual_embeds
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

impl IsqModel for Qwen3_5VLTextModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match &mut layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    tensors.push((&mut attn.q_proj, Some(i)));
                    tensors.push((&mut attn.k_proj, Some(i)));
                    tensors.push((&mut attn.v_proj, Some(i)));
                    tensors.push((&mut attn.o_proj, Some(i)));
                }
                LayerImpl::LinearAttention(gdn) => {
                    tensors.push((&mut gdn.out_proj, Some(i)));
                }
            }
            for m in layer.ffn.get_isq_layers() {
                tensors.push((m, Some(i)));
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

            match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
                    uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
                }
                LayerImpl::LinearAttention(gdn) => {
                    match &gdn.projection {
                        GdnProjection::SplitQkvZa {
                            in_proj_qkv,
                            in_proj_z,
                            in_proj_b,
                            in_proj_a,
                        } => {
                            uvb_l
                                .pp("linear_attn")
                                .pp("in_proj_qkv")
                                .add_tensor("weight", in_proj_qkv.weight().clone());
                            uvb_l
                                .pp("linear_attn")
                                .pp("in_proj_z")
                                .add_tensor("weight", in_proj_z.weight().clone());
                            uvb_l
                                .pp("linear_attn")
                                .pp("in_proj_b")
                                .add_tensor("weight", in_proj_b.weight().clone());
                            uvb_l
                                .pp("linear_attn")
                                .pp("in_proj_a")
                                .add_tensor("weight", in_proj_a.weight().clone());
                        }
                        GdnProjection::FusedQkvzBa {
                            in_proj_qkvz,
                            in_proj_ba,
                        } => {
                            uvb_l
                                .pp("linear_attn")
                                .pp("in_proj_qkvz")
                                .add_tensor("weight", in_proj_qkvz.weight().clone());
                            uvb_l
                                .pp("linear_attn")
                                .pp("in_proj_ba")
                                .add_tensor("weight", in_proj_ba.weight().clone());
                        }
                    }
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("conv1d.weight", gdn.conv1d_weight.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("dt_bias", gdn.dt_bias.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("A_log", gdn.a_log.clone());
                    uvb_l
                        .pp("linear_attn")
                        .pp("norm")
                        .add_tensor("weight", gdn.norm.weight.clone());
                }
            }

            match &layer.ffn {
                FeedForward::MoE(moe) => {
                    uvb_l
                        .pp("mlp")
                        .pp("gate")
                        .add_tensor("weight", moe.gate.weight().clone());
                    uvb_l
                        .pp("mlp")
                        .pp("shared_expert_gate")
                        .add_tensor("weight", moe.shared_expert_gate.weight().clone());
                    uvb_l
                        .pp("mlp")
                        .pp("shared_expert")
                        .pp("gate_proj")
                        .add(&moe.shared_expert.gate_proj);
                    uvb_l
                        .pp("mlp")
                        .pp("shared_expert")
                        .pp("up_proj")
                        .add(&moe.shared_expert.up_proj);
                    uvb_l
                        .pp("mlp")
                        .pp("shared_expert")
                        .pp("down_proj")
                        .add(&moe.shared_expert.down_proj);
                }
                FeedForward::Dense(mlp) => {
                    uvb_l.pp("mlp").pp("gate_proj").add(&mlp.gate_proj);
                    uvb_l.pp("mlp").pp("up_proj").add(&mlp.up_proj);
                    uvb_l.pp("mlp").pp("down_proj").add(&mlp.down_proj);
                }
            }
        }

        uvb.to_safetensors()
    }
}
