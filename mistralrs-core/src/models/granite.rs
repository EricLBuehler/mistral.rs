#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer, MoeMlp},
    attention::SdpaParams,
    device_map::DeviceMapper,
    layers::{embedding, CausalMasker, MatMul, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, word_emb_default, true);
serde_default_fn!(f32, default_one, 1.0);
serde_default_fn!(f32, default_rope_theta, 10_000.0);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GraniteLayerType {
    #[default]
    Attention,
    Mamba,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GraniteRopeType {
    #[default]
    Default,
    Granite,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GraniteRopeConfig {
    pub factor: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rope_type: GraniteRopeType,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<GraniteRopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Vec<GraniteLayerType>,
    #[serde(default = "default_one")]
    pub attention_multiplier: f32,
    #[serde(default = "default_one")]
    pub embedding_multiplier: f32,
    #[serde(default = "default_one")]
    pub residual_multiplier: f32,
    #[serde(default = "default_one")]
    pub logits_scaling: f32,
}

impl Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn shared_intermediate_size(&self) -> usize {
        self.shared_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn layer_types(&self) -> Vec<GraniteLayerType> {
        if self.layer_types.is_empty() {
            vec![GraniteLayerType::Attention; self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        }
    }
}

/// GraniteMLP uses a fused gate-up projection followed by output projection
/// Input: shared_mlp.input_linear (hidden -> shared_intermediate * 2)
/// Output: shared_mlp.output_linear (shared_intermediate -> hidden)
#[derive(Clone)]
pub struct GraniteMlp {
    input_linear: Arc<dyn QuantMethod>,
    output_linear: Arc<dyn QuantMethod>,
    params: Vec<usize>,
}

impl GraniteMlp {
    pub fn new(
        vb: ShardedVarBuilder,
        cfg: &Config,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let shared_intermediate_size = cfg.shared_intermediate_size();
        let input_linear = ColumnParallelLayer::new(
            cfg.hidden_size,
            shared_intermediate_size * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("shared_mlp").pp("input_linear"),
        )?;
        let output_linear = RowParallelLayer::new(
            shared_intermediate_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("shared_mlp").pp("output_linear"),
        )?;
        Ok(Self {
            input_linear,
            output_linear,
            params: vec![cfg.hidden_size, shared_intermediate_size],
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.input_linear.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let projected = MatMul.qmethod_matmul(&x, &*self.input_linear)?;
        let chunks = projected.chunk(2, candle_core::D::Minus1)?;
        let gated = (candle_nn::ops::silu(&chunks[0])? * &chunks[1])?;
        let mut res = MatMul.qmethod_matmul(&gated, &*self.output_linear)?;
        if self.input_linear.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

impl MlpLayer for GraniteMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.input_linear, &mut self.output_linear]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Self {
            input_linear: self.input_linear.clone(),
            output_linear: self.output_linear.clone(),
            params: self.params.clone(),
        })
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    fn hidden_act(&self) -> crate::layers::Activation {
        crate::layers::Activation::Silu
    }
    fn new_added_delta(&self, _deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        candle_core::bail!("LoRA adapter not supported for GraniteMlp")
    }
    fn dtype_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.input_linear.dtype_and_device()
    }
}

impl crate::amoe::AnyMoeTrainableLayer for GraniteMlp {}

struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    max_seq_len: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
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

        (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

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
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
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

    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        rope: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = cfg.head_dim() * cfg.num_attention_heads;
        let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
        let q_proj = ColumnParallelLayer::new(
            size_in,
            size_q,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads(),
            cfg.head_dim(),
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
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: (cfg.num_key_value_heads() / comm.world_size()).max(1),
            head_dim: cfg.head_dim(),
            rotary_emb: rope,
            max_seq_len: cfg.max_position_embeddings,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads(),
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                // GraniteMoeHybrid uses attention_multiplier instead of 1/sqrt(d)
                softmax_scale: cfg.attention_multiplier,
                sliding_window: None,
            },
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Box<dyn MlpLayer>,
    residual_multiplier: f32,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let attn_out = self.attn.forward(
            &x,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        // Scale residual connection
        let attn_out = scale_tensor(attn_out, self.residual_multiplier)?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let mlp_out = self.mlp.forward(&self.rms_2.forward(&x)?)?;
        // Scale residual connection
        let mlp_out = scale_tensor(mlp_out, self.residual_multiplier)?;
        let x = (mlp_out + residual)?;
        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::load(
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            cfg,
            rope,
            paged_attn,
            comm,
        )?;
        let mlp = GraniteMlp::new(
            mapper.set_device(layer_idx, vb.clone(), loading_isq),
            cfg,
            comm,
        )?;
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
            mlp: Box::new(mlp),
            residual_multiplier: cfg.residual_multiplier,
        })
    }
}

fn scale_tensor(tensor: Tensor, scale: f32) -> Result<Tensor> {
    if (scale - 1.0).abs() < f32::EPSILON {
        Ok(tensor)
    } else {
        tensor.affine(scale as f64, 0.)
    }
}

pub struct GraniteMoeHybrid {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    kv_cache: crate::pipeline::EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    embedding_multiplier: f32,
    logits_scaling: f32,
}

impl GraniteMoeHybrid {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
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
                &cfg.quantization_config,
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

        let head_dim = cfg.head_dim();

        // Build RoPE embeddings per device
        // Note: granite rope_type scaling is not yet supported, using default rope
        if let Some(GraniteRopeConfig {
            rope_type: GraniteRopeType::Granite,
            ..
        }) = &cfg.rope_scaling
        {
            tracing::warn!(
                "Granite-style rope scaling is not yet fully supported. Using default rope scaling."
            );
        }

        let mut ropes = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            if !ropes.contains_key(&device.location()) {
                let rope = RotaryEmbedding::new(
                    cfg.rope_theta,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    true, // is_gpt_neox style
                    vb_m.dtype(),
                )?;
                ropes.insert(device.location(), Arc::new(rope));
            }
        }

        // Validate layer types - we only support attention layers for now
        let layer_types = cfg.layer_types();
        for (idx, layer_type) in layer_types.iter().enumerate() {
            if matches!(layer_type, GraniteLayerType::Mamba) {
                candle_core::bail!(
                    "Mamba layers are not yet supported in GraniteMoeHybrid inference (layer {})",
                    idx
                );
            }
        }

        let blocks: Vec<_> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|i| {
            let device = mapper
                .device_for(i, false)
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
            let comm = mapper.get_comm_for(i)?;
            Block::load(
                vb_m.pp(format!("layers.{i}")),
                cfg,
                &*mapper,
                i,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                paged_attn,
                &comm,
            )
        })?;

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
                num_kv_heads: (cfg.num_key_value_heads() / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: None,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
            },
            mapper,
            embedding_multiplier: cfg.embedding_multiplier,
            logits_scaling: if cfg.logits_scaling == 0.0 {
                1.0
            } else {
                1.0 / cfg.logits_scaling
            },
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut x = self.wte.forward(input_ids)?;
        // Scale embeddings
        x = scale_tensor(x, self.embedding_multiplier)?;

        let cache = &mut self.kv_cache.normal().0;
        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            x.dtype(),
            self.blocks[0].attn.num_attention_heads,
        )?;
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
                &mask.clone().map(|m| m.to_device(x.device()).unwrap()),
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
        let mut logits = MatMul.qmethod_matmul(&x, &*self.lm_head)?;

        // Scale logits
        logits = scale_tensor(logits, self.logits_scaling)?;

        extract_logits(&logits, context_lens)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.wte);
        uvb_m.pp("norm").add(&self.ln_f);

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.rms_1);
            uvb_l.pp("post_attention_layernorm").add(&layer.rms_2);
        }

        uvb_m.to_safetensors()
    }
}

impl IsqModel for GraniteMoeHybrid {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.blocks.iter_mut().enumerate() {
            tensors.push((&mut layer.attn.q_proj, Some(i)));
            tensors.push((&mut layer.attn.k_proj, Some(i)));
            tensors.push((&mut layer.attn.v_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));
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
        self.residual_tensors_m(uvb.pp("model"))
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        let mut names = Vec::new();
        // lm_head
        names.push(None);
        for i in 0..self.blocks.len() {
            names.push(Some(format!("blk.{i}.attn_q.weight")));
            names.push(Some(format!("blk.{i}.attn_k.weight")));
            names.push(Some(format!("blk.{i}.attn_v.weight")));
            names.push(Some(format!("blk.{i}.attn_output.weight")));
            // GraniteMlp has input_linear and output_linear
            names.push(Some(format!("blk.{i}.ffn_input.weight")));
            names.push(Some(format!("blk.{i}.ffn_output.weight")));
        }
        Ok(names)
    }
}

impl NormalModel for GraniteMoeHybrid {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
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
        unimplemented!()
    }
    fn cache(&self) -> &crate::pipeline::EitherCache {
        &self.kv_cache
    }
    fn cache_mut(&mut self) -> &mut crate::pipeline::EitherCache {
        &mut self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.blocks[0].attn.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for GraniteMoeHybrid {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.blocks {
            mlps.push(&*layer.mlp);
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.blocks {
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
            layers = (0..self.blocks.len()).collect::<Vec<_>>();
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

                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = self.blocks[layer].mlp.dtype_device();
                        // For GraniteMlp, we need custom handling
                        let cfg_for_layer = Config {
                            hidden_size: self.blocks[layer].mlp.get_params()[0],
                            shared_intermediate_size: Some(self.blocks[layer].mlp.get_params()[1]),
                            intermediate_size: self.blocks[layer].mlp.get_params()[1],
                            vocab_size: 0,
                            num_hidden_layers: 0,
                            num_attention_heads: 0,
                            num_key_value_heads: None,
                            rms_norm_eps: 0.0,
                            rope_theta: 0.0,
                            max_position_embeddings: 0,
                            rope_scaling: None,
                            quantization_config: None,
                            tie_word_embeddings: false,
                            layer_types: vec![],
                            attention_multiplier: 1.0,
                            embedding_multiplier: 1.0,
                            residual_multiplier: 1.0,
                            logits_scaling: 1.0,
                        };
                        row.push(Box::new(GraniteMlp::new(
                            vb.pp(layer).pp(&mlp).set_dtype(dtype).set_device(device),
                            &cfg_for_layer,
                            &self.mapper.get_comm_for(layer)?,
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter { .. } => {
                        candle_core::bail!("LoRA adapters not supported for GraniteMoeHybrid MLP")
                    }
                }
            }
        }
        for (layer, expert) in layers.into_iter().zip(experts) {
            let mut experts_all = vec![self.blocks[layer].mlp.clone()];
            experts_all.extend(expert);
            let (dtype, device) = self.blocks[layer].mlp.dtype_device();
            self.blocks[layer].mlp = Box::new(MoeMlp::new(
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
