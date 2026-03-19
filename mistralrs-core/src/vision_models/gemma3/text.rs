use std::{collections::HashMap, sync::Arc};

use candle_core::{Device, Module, Result, Tensor};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer, MoeMlp},
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    get_delta_from_lora_ab,
    layers::{
        embedding, CausalMasker, Gemma3RotaryEmbedding, GemmaRmsNorm, MatMul, Mlp, RotaryEmbedding,
        ScaledEmbedding, Sdpa,
    },
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalCacheType, NormalLoadingMetadata,
        VisionModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::Gemma3TextConfig;

macro_rules! is_sliding {
    ($layer_idx:expr, $cfg:expr) => {
        ($layer_idx + 1) % $cfg.sliding_window_pattern != 0
    };
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb_global: Arc<Gemma3RotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    use_sliding_window: bool,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<Gemma3RotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma3TextConfig,
        layer_idx: usize,
        mapper: &dyn DeviceMapper,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("o_proj"),
        )?;
        let sliding_window = if is_sliding!(layer_idx, cfg) {
            Some(cfg.sliding_window)
        } else {
            None
        };

        let q_norm = GemmaRmsNorm::new(
            cfg.head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = GemmaRmsNorm::new(
            cfg.head_dim,
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
            use_sliding_window: sliding_window.is_some(),
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: cfg.attn_logit_softcapping.map(|x| x as f32),
                softmax_scale: 1.0 / (cfg.query_pre_attn_scalar as f32).sqrt(),
                sliding_window,
                sinks: None,
            },
            q_norm,
            k_norm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
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

        (q, k) = match self.use_sliding_window {
            true => self.rotary_emb_local.forward(&q, &k, seqlen_offsets)?,
            false => self.rotary_emb_global.forward(&q, &k, seqlen_offsets)?,
        };

        let mask = if self.use_sliding_window {
            sliding_attention_mask
        } else {
            attention_mask
        };

        // With flash (Some): pass attention_mask (global causal; flash kernel handles sliding window)
        // Without flash (None): pass mask (per-layer mask with sliding window baked in)
        let paged_mask = if flash_params.is_some() {
            attention_mask
        } else {
            mask
        };

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    paged_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    flash_params,
                )?,
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
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                match flash_params {
                    Some(fp) => {
                        Sdpa.run_attention(&q, &k, &v, mask, Some(fp), &self.sdpa_params)?
                    }
                    None => Sdpa.run_attention_noflash(&q, &k, &v, mask, &self.sdpa_params)?,
                }
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        // Transpose needed whenever SDPA was used (i.e., any mask was present)
        attn_output = if paged_mask.is_some() || mask.is_some() {
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

struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<Gemma3RotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma3TextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
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
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
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
        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(
                &xs,
                attention_mask,
                sliding_attention_mask,
                seqlen_offsets,
                kv_cache,
                metadata,
                flash_params,
            )?
            .apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?
            .apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

pub struct TextModel {
    embed_tokens: ScaledEmbedding,
    layers: Vec<DecoderLayer>,
    norm: GemmaRmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: usize,
    final_logit_softcapping: Option<f64>,
    cfg: ModelConfigMetadata,
    image_token_index: Option<usize>,
}

impl TextModel {
    pub fn new(
        cfg: &Gemma3TextConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
        image_token_index: Option<usize>,
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
        let embed_tokens = ScaledEmbedding::new(
            (cfg.hidden_size as f64).sqrt(),
            embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
                &cfg.quantization_config,
            )?,
        );

        let mut global_ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            global_ropes.insert(
                device.location(),
                Arc::new(Gemma3RotaryEmbedding::new(
                    is_gptx,
                    vb.dtype(),
                    cfg,
                    device,
                )?),
            );
        }

        let mut local_ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            local_ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_local_base_freq as f32,
                    cfg.head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
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
            let rotary_emb_global = global_ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let rotary_emb_local = local_ropes
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
                rotary_emb_global,
                rotary_emb_local,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
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
        let cache_types = (0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                is_sliding!(layer_idx, cfg)
                    .then(|| NormalCacheType::SlidingWindow {
                        window: cfg.sliding_window,
                    })
                    .unwrap_or(NormalCacheType::Normal {
                        max_seq_len: cfg.max_position_embeddings,
                    })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
            final_logit_softcapping: cfg.final_logit_softcapping,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            image_token_index,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        mut xs: Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        has_images: bool,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;

        // When images are present, we need bidirectional attention for image tokens.
        // Flash attention doesn't support per-token mixed causal/bidirectional masking,
        // so we construct real masks and bypass flash attention during image prefill
        // by passing flash_params=None to layers.
        // See: https://github.com/vllm-project/vllm/blob/5819ca8944af4f7dcbac3c6b73179f760e05910d/vllm/config/model.py#L1116-L1125
        let has_bidirectional =
            has_images && self.image_token_index.is_some() && input_ids.dim(1)? > 1;

        let (attention_mask, sliding_attention_mask, layer_flash_params) = if has_bidirectional {
            // Build real masks (not flash-attn dummies) with bidirectional regions for image tokens
            let image_token_index = self.image_token_index.unwrap();
            let mask_cache: &dyn PastKvLenCache = metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache);
            let causal_mask =
                CausalMasker.make_causal_mask_as_attn_bias(input_ids, mask_cache, xs.dtype())?;
            let sliding_mask = CausalMasker.make_sliding_window_causal_mask_as_attn_bias(
                input_ids,
                mask_cache,
                Some(self.sliding_window),
                xs.dtype(),
            )?;

            // Apply bidirectional override for image tokens
            let attention_mask = causal_mask
                .map(|m| Self::apply_image_bidirectional_mask(&m, input_ids, image_token_index))
                .transpose()?;
            let sliding_attention_mask = sliding_mask
                .map(|m| Self::apply_image_bidirectional_mask(&m, input_ids, image_token_index))
                .transpose()?;

            // Move to CPU (same optimization as normal path)
            let attention_mask = attention_mask.map(|m| m.to_device(&Device::Cpu).unwrap());
            let sliding_attention_mask =
                sliding_attention_mask.map(|m| m.to_device(&Device::Cpu).unwrap());

            // PagedAttention prompt chunking filter
            let attention_mask = attention_mask.filter(|_| {
                metadata
                    .as_ref()
                    .map(|(_, meta)| meta.is_first_prompt_chunk)
                    .unwrap_or(true)
            });
            let sliding_attention_mask = sliding_attention_mask.filter(|_| {
                metadata
                    .as_ref()
                    .map(|(_, meta)| meta.is_first_prompt_chunk)
                    .unwrap_or(true)
            });

            // None = bypass flash, use the mask directly
            (attention_mask, sliding_attention_mask, None)
        } else {
            // Standard path: use CausalMasker (returns dummy (1,1) when flash-attn on CUDA)
            let attention_mask = CausalMasker.make_causal_mask_matrix(
                input_ids,
                metadata
                    .as_ref()
                    .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                    .unwrap_or(cache as &dyn PastKvLenCache),
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
                metadata
                    .as_ref()
                    .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                    .unwrap_or(cache as &dyn PastKvLenCache),
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
            xs = layer.forward(
                &xs,
                attention_mask.as_ref().map(|m| m.get(xs.device())),
                sliding_attention_mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
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

    /// Apply bidirectional attention override for image tokens within the same image group.
    /// Where both query and key positions are image tokens in the same contiguous group,
    /// the mask value is set to 0.0 (attend) instead of -inf (mask).
    fn apply_image_bidirectional_mask(
        causal_mask: &Tensor,
        input_ids: &Tensor,
        image_token_index: usize,
    ) -> Result<Tensor> {
        // input_ids: (1, seq_len), causal_mask: (seq_len, total_len) where total_len = seq_len + past_kv_len
        let (_, seq_len) = input_ids.dims2()?;
        let total_len = causal_mask.dim(1)?;
        let past_kv_len = total_len - seq_len;

        // Flatten input_ids to 1D: (seq_len,)
        let input_ids_1d = input_ids.squeeze(0)?;

        // is_image: (seq_len,) boolean - true where token is an image token
        let is_image = input_ids_1d
            .eq(image_token_index as f64)?
            .to_dtype(candle_core::DType::U32)?;

        // Compute image group IDs via contiguous block detection
        // is_prev_image: shift right by 1, pad left with 0
        let is_image_vec: Vec<u32> = is_image.to_vec1()?;
        let mut group_ids = vec![-1i64; seq_len];
        let mut current_group: i64 = -1;
        for i in 0..seq_len {
            if is_image_vec[i] == 1 {
                // Start new group if previous token is not an image token
                if i == 0 || is_image_vec[i - 1] == 0 {
                    current_group += 1;
                }
                group_ids[i] = current_group;
            }
        }

        // Build the bidirectional override mask on CPU as f32
        // For efficiency, we compute this as a Vec and create the tensor once
        let device = causal_mask.device();
        let dtype = causal_mask.dtype();

        // The mask covers (seq_len, total_len). Positions 0..past_kv_len are past KV cache
        // tokens (no image tokens there during image prefill since past_kv_len=0 typically).
        // Positions past_kv_len..total_len correspond to current input_ids.
        let mut override_vals = vec![0f32; seq_len * total_len];
        for qi in 0..seq_len {
            if group_ids[qi] < 0 {
                continue; // Not an image token query
            }
            for ki in 0..seq_len {
                if group_ids[ki] >= 0 && group_ids[qi] == group_ids[ki] {
                    // Both are image tokens in the same group: mark for bidirectional override
                    let col = ki + past_kv_len;
                    override_vals[qi * total_len + col] = 1.0;
                }
            }
        }

        let override_mask = Tensor::from_vec(override_vals, (seq_len, total_len), device)?;

        // Where override is 1, set mask to 0.0 (attend); otherwise keep original causal mask.
        // We use where_cond instead of multiplication to avoid NaN from -inf * 0.
        let zero = Tensor::zeros((seq_len, total_len), dtype, device)?;
        let override_bool = override_mask.to_dtype(candle_core::DType::U8)?;
        override_bool.where_cond(&zero, causal_mask)
    }
}

impl IsqModel for TextModel {
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
            tensors.extend(
                layer
                    .mlp
                    .get_isq_layers()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
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
        }

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        // NOTE: dependant on the exact implementation in get_layers!
        let mut names = Vec::new();
        // lm_head
        names.push(None);
        for i in 0..self.layers.len() {
            names.push(Some(format!("blk.{i}.attn_q.weight")));
            names.push(Some(format!("blk.{i}.attn_k.weight")));
            names.push(Some(format!("blk.{i}.attn_v.weight")));
            names.push(Some(format!("blk.{i}.attn_output.weight")));
            names.push(Some(format!("blk.{i}.ffn_gate.weight")));
            names.push(Some(format!("blk.{i}.ffn_up.weight")));
            names.push(Some(format!("blk.{i}.ffn_down.weight")));
        }
        Ok(names)
    }
}

impl VisionModel for TextModel {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        _seqlen_offsets: &[usize],
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _model_specific_args: Box<dyn std::any::Any>, // pixel attention mask, or image sizes, or anything else
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
}

impl AnyMoeBaseModelMixin for TextModel {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.layers {
            mlps.push(&*layer.mlp);
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.layers {
            mlps.push(&mut layer.mlp);
        }
        mlps
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        let mut experts: Vec<Vec<Box<dyn MlpLayer>>> = Vec::new();
        if layers.is_empty() {
            layers = (0..self.layers.len()).collect::<Vec<_>>();
        }
        for _ in 0..layers.len() {
            experts.push(Vec::new());
        }
        for vb in additional_vbs {
            let vb = vb.pp(&prefix);
            for (layer, row) in experts.iter_mut().enumerate() {
                if !layers.contains(&layer) {
                    continue;
                }

                let intermediate_size = self.layers[layer].mlp.get_params()[1];
                let hidden_size = self.layers[layer].mlp.get_params()[0];
                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = self.layers[layer].mlp.dtype_device();
                        row.push(Box::new(Mlp::replicate(
                            self.layers[layer].mlp.get_params(),
                            vb.pp(layer).pp(&mlp).set_dtype(dtype).set_device(device),
                            self.layers[layer].mlp.hidden_act(),
                            &self.mapper.get_comm_for(layer)?,
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer).pp(&mlp);

                        let gate_proj_delta = if target_modules.contains(&"gate_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "gate_proj"
                            ))
                        } else {
                            None
                        };
                        let up_proj_delta = if target_modules.contains(&"up_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "up_proj"
                            ))
                        } else {
                            None
                        };
                        let down_proj_delta = if target_modules.contains(&"down_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (intermediate_size, hidden_size),
                                "down_proj"
                            ))
                        } else {
                            None
                        };

                        row.push(self.layers[layer].mlp.new_added_delta(vec![
                            gate_proj_delta,
                            up_proj_delta,
                            down_proj_delta,
                        ])?);
                    }
                }
            }
        }
        for (layer, expert) in layers.into_iter().zip(experts) {
            let mut experts_all = vec![self.layers[layer].mlp.clone()];
            experts_all.extend(expert);
            let (dtype, device) = self.layers[layer].mlp.dtype_device();
            self.layers[layer].mlp = Box::new(MoeMlp::new(
                experts_all,
                config.clone(),
                dtype,
                &device,
                layer,
                gate_vb.as_ref(),
            )?);
        }
        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}
