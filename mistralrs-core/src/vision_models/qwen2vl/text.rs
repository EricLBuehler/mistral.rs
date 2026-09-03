use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use crate::{
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{self, Activation, F32RmsNorm, Qwen2VLRotaryEmbedding, Sdpa},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalCacheType,
        NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::Config;

fn cache_types(
    layer_sliding_windows: &[Option<usize>],
    max_position_embeddings: usize,
) -> Vec<NormalCacheType> {
    layer_sliding_windows
        .iter()
        .map(|sliding_window| match sliding_window {
            Some(window) => NormalCacheType::SlidingWindow { window: *window },
            None => NormalCacheType::Normal {
                max_seq_len: max_position_embeddings,
            },
        })
        .collect()
}

struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: ShardedVarBuilder, comm: &Arc<mistralrs_quant::Comm>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = ColumnParallelLayer::new(
            hidden_sz,
            intermediate_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = ColumnParallelLayer::new(
            hidden_sz,
            intermediate_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            intermediate_sz,
            hidden_sz,
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
        let xs = xs.clone();
        let lhs = self.gate_proj.forward(&xs)?;
        let rhs = self.up_proj.forward(&xs)?;
        self.down_proj
            .forward(&crate::ops::mul_and_act(&lhs, &rhs, self.act_fn)?)?
            .to_dtype(original_dtype)
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
    rotary_emb: Arc<Qwen2VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

struct AttentionForward<'a> {
    attention_mask: &'a AttentionMask,
    sliding_attention_mask: &'a AttentionMask,
    cos_sin: &'a (Tensor, Tensor),
    metadata: Option<((Tensor, Tensor), &'a PagedAttentionInputMetadata)>,
    flash_params: &'a FlashParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<Qwen2VLRotaryEmbedding>,
        cfg: &Config,
        sliding_window: Option<usize>,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            true,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        )?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            true,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            true,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_proj"),
        )?;
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
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        kv_cache: &mut KvCache,
        args: AttentionForward<'_>,
    ) -> Result<Tensor> {
        let AttentionForward {
            attention_mask,
            sliding_attention_mask,
            cos_sin,
            metadata,
            flash_params,
        } = args;
        let (b_sz, q_len, _) = xs.dims3()?;

        let (q, k, v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        let (mut q, mut k, v) = if q_len != 1 {
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

        let cos_sin = &(
            cos_sin.0.to_device(q.device())?,
            cos_sin.1.to_device(q.device())?,
        );
        self.rotary_emb.forward(cos_sin, &mut q, &mut k)?;
        let attention_mask = if self.sdpa_params.sliding_window.is_some() {
            sliding_attention_mask
        } else {
            attention_mask
        };

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
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                let f32_mask = match attention_mask {
                    AttentionMask::Custom(mask) => {
                        AttentionMask::Custom(mask.to_dtype(DType::F32)?)
                    }
                    other => other.clone(),
                };
                Sdpa.run_attention(
                    &q.to_dtype(DType::F32)?,
                    &k.contiguous()?.to_dtype(DType::F32)?,
                    &v.contiguous()?.to_dtype(DType::F32)?,
                    &f32_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
                .to_dtype(q.dtype())?
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

pub struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: F32RmsNorm,
    post_attention_layernorm: F32RmsNorm,
}

struct DecoderLayerLoad<'a> {
    cfg: &'a Config,
    mapper: &'a dyn DeviceMapper,
    layer_idx: usize,
    loading_isq: bool,
    paged_attn: Option<PagedAttention>,
    sliding_window: Option<usize>,
    comm: &'a Arc<mistralrs_quant::Comm>,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<Qwen2VLRotaryEmbedding>,
        vb: ShardedVarBuilder,
        args: DecoderLayerLoad<'_>,
    ) -> Result<Self> {
        let DecoderLayerLoad {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            sliding_window,
            comm,
        } = args;
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            sliding_window,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            comm,
        )?;
        let input_layernorm = F32RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = F32RmsNorm::new(
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
        kv_cache: &mut KvCache,
        args: AttentionForward<'_>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, kv_cache, args)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Qwen2VLTextModel {
    embed_tokens: Arc<dyn QuantMethod>,
    pub(super) norm: F32RmsNorm,
    layers: Vec<DecoderLayer>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    lm_head: Arc<dyn QuantMethod>,
    pub(super) cache: EitherCache,
    pub(super) cfg: ModelConfigMetadata,
    pub(super) device: Device,
    pub(super) dtype: DType,
    pub(super) max_seq_len: usize,
    pub(super) sliding_window: Option<usize>,
}

impl Qwen2VLTextModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        // Support both HuggingFace naming (model.*) and MLX naming (language_model.model.*)
        let vb_m =
            if layers::contains_tensor_or_uqff(&vb, "language_model.model.embed_tokens.weight") {
                vb.pp("language_model").pp("model")
            } else {
                vb.pp("model")
            };

        let embed_tokens = layers::embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(Qwen2VLRotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    device,
                    cfg.rope_scaling.mrope_section.clone(),
                )?),
            );
        }

        let vb_l = vb_m.pp("layers");
        let layer_sliding_windows = cfg.layer_sliding_windows()?;
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
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb.clone(),
                vb_l.pp(layer_idx),
                DecoderLayerLoad {
                    cfg,
                    mapper: &*mapper,
                    layer_idx,
                    loading_isq: normal_loading_metadata.loading_isq,
                    paged_attn,
                    sliding_window: layer_sliding_windows[layer_idx],
                    comm: &comm,
                },
            )
        })?;
        let norm = F32RmsNorm::new(
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
            embed_tokens.clone()
        };
        let sliding_window = layer_sliding_windows.iter().flatten().next().copied();
        Ok(Self {
            embed_tokens,
            norm,
            layers,
            lm_head,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types(
                &layer_sliding_windows,
                cfg.max_position_embeddings,
            ))),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window,
                k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: normal_loading_metadata.real_device.clone(),
            dtype: vb.dtype(),
            mapper,
            sliding_window,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)
    }

    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: &AttentionMask,
        sliding_attention_mask: &AttentionMask,
        position_ids: &Tensor,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let cos_sin = self.layers[0]
            .self_attn
            .rotary_emb
            .compute_cos_sin(position_ids, xs.dtype())?;

        let attention_mask = DeviceMappedMask::new(attention_mask.clone(), &*self.mapper)?;
        let sliding_attention_mask =
            DeviceMappedMask::new(sliding_attention_mask.clone(), &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                &mut cache[i],
                AttentionForward {
                    attention_mask: &attention_mask.get(xs.device()),
                    sliding_attention_mask: &sliding_attention_mask.get(xs.device()),
                    cos_sin: &cos_sin,
                    metadata: ctx.paged_layer(i),
                    flash_params: ctx.flash_params(),
                },
            )?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        ctx.lm_head(&*self.lm_head, &xs)
    }
}

impl IsqModel for Qwen2VLTextModel {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eager_cache_layout_matches_layer_windows() {
        let types = cache_types(&[None, Some(128), None], 4096);

        assert!(matches!(
            &types[0],
            NormalCacheType::Normal { max_seq_len: 4096 }
        ));
        assert!(matches!(
            &types[1],
            NormalCacheType::SlidingWindow { window: 128 }
        ));
        assert!(matches!(
            &types[2],
            NormalCacheType::Normal { max_seq_len: 4096 }
        ));
    }
}
