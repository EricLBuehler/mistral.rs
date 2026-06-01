#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{Device, Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};
use mm_embedding::{InputMode, Phi4MMImageAudioEmbedding};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{self, Activation, CausalMasker, Phi4MMRotaryEmbedding, RmsNorm, Sdpa},
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
        PagedAttention,
    },
    pipeline::{
        text_models_inputs_processor::PagedAttentionInputMetadata, EitherCache, IsqModel, KvCache,
        ModelForwardContext, MultimodalModel, NormalCache, NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

mod audio_embedding;
mod config;
mod image_embedding;
pub(crate) mod inputs_processor;
mod mm_embedding;

pub(crate) use config::Phi4MMConfig;
pub(crate) use image_embedding::PHI4_MM_VISION_CFG;

struct Attention {
    qkv_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Phi4MMRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<Phi4MMRotaryEmbedding>,
        cfg: &Phi4MMConfig,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;

        // No TP here.
        let qkv_proj = mistralrs_quant::linear_no_bias_static_lora(
            cfg.hidden_size,
            op_size,
            cfg.loras(),
            vb.pp("qkv_proj"),
        )?;

        let o_proj = mistralrs_quant::linear_no_bias_static_lora(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.loras(),
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
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

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

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

        let position_ids = ctx.position_ids_vec();
        let positions = ctx
            .rope_positions(q.device())?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
        let (q, k) = self
            .rotary_emb
            .forward_positions(&q, &k, positions, &position_ids)?;

        let metadata = ctx.paged_layer(layer_idx);
        let flash_params = ctx.flash_params();
        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k.contiguous()?,
                    &v.contiguous()?,
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
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k.contiguous()?,
                        &v.contiguous()?,
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

        attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let res = self.o_proj.forward(&attn_output)?;
        Ok(res)
    }
}

#[derive(Clone)]
struct Mlp {
    gate_up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Phi4MMConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;

        // No TP here.
        let gate_up_proj = mistralrs_quant::linear_no_bias_static_lora(
            hidden_size,
            2 * i_size,
            cfg.loras(),
            vb.pp("gate_up_proj"),
        )?;

        let down_proj = mistralrs_quant::linear_no_bias_static_lora(
            i_size,
            hidden_size,
            cfg.loras(),
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = self.gate_up_proj.forward(xs)?;
        let up_states = crate::ops::split_mul_and_act(&up_states, self.i_size, self.act_fn)?;
        let res = self.down_proj.forward(&up_states)?;
        Ok(res)
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    self_attn: Attention,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<Phi4MMRotaryEmbedding>,
        cfg: &Phi4MMConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
        )?;
        let mlp = Mlp::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
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
            input_layernorm,
            post_attention_layernorm,
            mlp,
            self_attn,
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

pub struct Phi4MMModel {
    embed_tokens: candle_nn::Embedding,
    embed_tokens_extend: Phi4MMImageAudioEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: Option<usize>,
    cfg: ModelConfigMetadata,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Phi4MMModel {
    pub fn new(
        cfg: &Phi4MMConfig,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");

        let embed_tokens = layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let vb_l = vb_m.pp("layers");
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(Phi4MMRotaryEmbedding::new(vb.dtype(), cfg, device)?),
            );
        }
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
                    Some(PagedAttention::new(cfg.head_dim(), device, None)?)
                }
            };
            DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
            )
        })?;
        let norm = RmsNorm::new(
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

        let embed_tokens_extend = Phi4MMImageAudioEmbedding::new(
            cfg,
            embed_tokens.clone(),
            mapper.set_nm_device(vb_m.pp("embed_tokens_extend"), false),
        )?;

        Ok(Self {
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
            embed_tokens,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads(),
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            embed_tokens_extend,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_image_embeds: Option<Tensor>,
        image_attention_mask: Option<Tensor>,
        image_sizes: Option<Vec<(u32, u32)>>,
        input_audio_embeds: Option<Tensor>,
        audio_embed_sizes: Option<Vec<usize>>,
        audio_attention_mask: Option<Tensor>,
        ctx: &mut ModelForwardContext<'_>,
        image_hashes: &[u64],
    ) -> Result<Tensor> {
        let mut xs = if input_image_embeds.is_some() || input_audio_embeds.is_some() {
            let projection_mode = match (&input_image_embeds, &input_audio_embeds) {
                (Some(_), Some(_)) | (Some(_), None) => InputMode::Vision,
                (None, Some(_)) => InputMode::Speech,
                _ => unreachable!("already know either are some"),
            };

            self.embed_tokens_extend.forward(
                input_ids,
                input_image_embeds.as_ref(),
                &match image_attention_mask.as_ref() {
                    Some(t) => AttentionMask::Custom((*t).clone()),
                    None => AttentionMask::None,
                },
                image_sizes,
                input_audio_embeds.as_ref(),
                audio_embed_sizes,
                &match audio_attention_mask.as_ref() {
                    Some(t) => AttentionMask::Custom((*t).clone()),
                    None => AttentionMask::None,
                },
                projection_mode,
                image_hashes,
                &self.encoder_cache,
            )?
        } else {
            self.embed_tokens.forward(input_ids)?
        };
        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window: self.sliding_window,
                ..Default::default()
            },
        )?;
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(&xs, &attention_mask.get(xs.device()), &mut cache[i], ctx, i)?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
    }
}

#[derive(Default)]
pub(crate) struct Phi4MMVisionSpecificArgs {
    pub image_sizes: Option<Vec<(u32, u32)>>,
    pub input_image_embeds: Option<Tensor>,
    pub image_attention_mask: Option<Tensor>,
    pub input_audio_embeds: Option<Tensor>,
    pub audio_embed_sizes: Option<Vec<usize>>,
    pub audio_attention_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
}

impl crate::speculative::SpeculativeTargetMixin for Phi4MMModel {}

impl MultimodalModel for Phi4MMModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let Phi4MMVisionSpecificArgs {
            input_image_embeds,
            image_attention_mask,
            image_sizes,
            input_audio_embeds,
            audio_attention_mask,
            audio_embed_sizes,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Phi4MMVisionSpecificArgs`");

        self.forward(
            input_ids,
            input_image_embeds,
            image_attention_mask,
            image_sizes,
            input_audio_embeds,
            audio_embed_sizes,
            audio_attention_mask,
            ctx,
            &image_hashes,
        )
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
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Phi4MMVisionSpecificArgs::default())
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        Some(
            self.encoder_cache
                .lock()
                .expect("encoder cache poisoned")
                .counters(),
        )
    }
}

impl IsqModel for Phi4MMModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.qkv_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.push((&mut layer.mlp.gate_up_proj, Some(i)));
            tensors.push((&mut layer.mlp.down_proj, Some(i)));
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);
        uvb_m
            .pp("embed_tokens_extend")
            .extend(self.embed_tokens_extend.residual_tensors());

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
        }

        uvb.to_safetensors()
    }
}

impl AnyMoeBaseModelMixin for Phi4MMModel {}
