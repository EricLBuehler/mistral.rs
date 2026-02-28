#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Qwen3.5 dense model (hybrid attention: FullAttention + GatedDeltaNet, dense MLP).

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Embedding;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{HybridCache, HybridCacheConfig, HybridLayerType},
    layers::{embedding, CausalMasker, GemmaRmsNorm, MatMul, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::deltanet::{DeltaNetConfig, GatedDeltaNet, GdnLayerCache, GdnProjection};

serde_default_fn!(bool, default_tie, true);
serde_default_fn!(f64, default_rope_theta, 10_000.0);
serde_default_fn!(f64, default_rms_norm_eps, 1e-6);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(f64, default_partial_rotary_factor, 0.25);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: crate::layers::Activation,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub head_dim: usize,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    // GDN config
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    // Hybrid layer types
    pub layers_block_type: Vec<String>,
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

#[derive(Debug, Clone)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

impl Config {
    pub fn layer_types(&self) -> Vec<LayerType> {
        self.layers_block_type
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" | "attention" => LayerType::FullAttention,
                _ => LayerType::LinearAttention,
            })
            .collect()
    }
}

impl DeltaNetConfig for Config {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }
    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }
    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.quantization_config
    }
}

// ====================== Full Attention (same as qwen3_next) ======================

#[allow(dead_code)]
struct FullAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    rot_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl FullAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rotary_emb: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb_sa = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm);
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("o_proj"),
        )?;

        let vb_sa_norms = mapper.set_device(layer_idx, vb.pp("self_attn"), false);
        let q_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("q_norm"))?;
        let k_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("k_norm"))?;

        let rot_dim = (head_dim as f64 * cfg.partial_rotary_factor) as usize;

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
            rot_dim,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(num_kv_heads, num_heads, comm),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

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
        let mut q_gate = MatMul.qmethod_matmul(&x, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&x, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&x, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q_gate = q_gate.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
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

        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        if self.rot_dim < self.head_dim {
            let q_rot = q.narrow(D::Minus1, 0, self.rot_dim)?;
            let q_pass = q.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, self.rot_dim)?;
            let k_pass = k.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;

            let (q_rot, k_rot) = self.rotary_emb.forward(&q_rot, &k_rot, seqlen_offsets)?;
            q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;
            k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
        } else {
            let (q_new, k_new) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;
            q = q_new;
            k = k_new;
        }

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

        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        y = y.broadcast_mul(&gate)?;

        let mut res = MatMul.qmethod_matmul(&y, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

// ====================== Dense MLP ======================

#[derive(Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: crate::layers::Activation,
}

impl Mlp {
    fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quant_config: &Option<QuantizedConfig>,
        act_fn: crate::layers::Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            intermediate_size,
            hidden_size,
            quant_config,
            false,
            comm,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let gate = MatMul.qmethod_matmul(&xs, &*self.gate_proj)?;
        let up = MatMul.qmethod_matmul(&xs, &*self.up_proj)?;
        let activated = crate::ops::mul_and_act(&gate, &up, self.act_fn)?;
        let mut res = MatMul.qmethod_matmul(&activated, &*self.down_proj)?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj]
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
    mlp: Mlp,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward_attention(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attn = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => attn,
            _ => candle_core::bail!("Expected full attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let attn_out = attn.forward(
            &x,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(&normed)?;
        ffn_out + residual
    }

    fn forward_linear(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let gdn_out = gdn.forward(&x, cache)?;
        let x = (gdn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(&normed)?;
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
    fn new(layer_types: &[LayerType], cfg: &Config, device: &Device, dtype: DType) -> Result<Self> {
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
                        cfg, dtype, device,
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

// ====================== Top-level Model ======================

#[allow(dead_code)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    norm: GemmaRmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    local_cache: Arc<Mutex<LocalHybridCache>>,
    kv_cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    num_attention_heads: usize,
    max_seq_len: usize,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");

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

        let layer_types = cfg.layer_types();

        // Build RoPE for attention layers (partial rotary)
        let rot_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
        let mut ropes = HashMap::new();
        for (i, layer_type) in layer_types.iter().enumerate().take(cfg.num_hidden_layers) {
            if matches!(layer_type, LayerType::FullAttention) {
                let device = mapper
                    .device_for(i, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(device.location())
                {
                    let rope = RotaryEmbedding::new_partial(
                        cfg.rope_theta as f32,
                        rot_dim,
                        cfg.max_position_embeddings,
                        device,
                        true,
                        vb_m.dtype(),
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
            "Qwen3.5: {} full attention layers, {} linear attention (GDN) layers",
            num_full,
            num_linear
        );

        // Build layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        ) {
            let comm = mapper.get_comm_for(i)?;
            let vb_layer = vb_m.pp(format!("layers.{i}"));

            let layer_impl = match &layer_types[i] {
                LayerType::FullAttention => {
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
                            Some(PagedAttention::new(cfg.head_dim, device, None)?)
                        }
                    };
                    LayerImpl::FullAttention(FullAttention::load(
                        vb_layer.clone(),
                        cfg,
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
                        cfg,
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

            let mlp = Mlp::new(
                mapper.set_device(i, vb_layer.pp("mlp"), normal_loading_metadata.loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                &comm,
            )?;

            layers.push(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                mlp,
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
            layer_types,
            norm,
            lm_head,
            local_cache,
            kv_cache: EitherCache::Hybrid(pipeline_cache),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: num_attention_heads,
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            num_attention_heads,
            max_seq_len: cfg.max_position_embeddings,
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
        let mut x = self.embed_tokens.forward(input_ids)?;

        let mut local_cache = self.local_cache.lock().unwrap();

        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*local_cache as &dyn PastKvLenCache),
            x.dtype(),
            self.num_attention_heads,
        )?;
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;

            match &layer.layer_impl {
                LayerImpl::FullAttention(_) => {
                    if let LocalLayerCache::Attention(kv_cache) = &mut local_cache.caches[layer_idx]
                    {
                        let mask_for_layer = mask.as_ref().map(|m| m.get(x.device()).clone());
                        x = layer.forward_attention(
                            &x,
                            &mask_for_layer,
                            seqlen_offsets,
                            kv_cache,
                            metadata.as_ref().map(|(kv_cache, metadata)| {
                                (kv_cache[layer_idx].clone(), *metadata)
                            }),
                            flash_params,
                        )?;
                    }
                }
                LayerImpl::LinearAttention(_) => {
                    if let LocalLayerCache::LinearAttention(gdn_cache) =
                        &mut local_cache.caches[layer_idx]
                    {
                        if seqlen_offsets[0] == 0 {
                            gdn_cache.reset()?;
                        }
                        x = layer.forward_linear(&x, gdn_cache)?;
                    }
                }
            }
        }

        let x = x.to_device(&self.device)?;
        let x = self.norm.forward(&x)?;

        let mut x = extract_logits(&x, context_lens)?;

        if let Some(t) = self.lm_head.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let logits = MatMul.qmethod_matmul(&x, &*self.lm_head)?;

        Ok(logits)
    }
}

// ====================== Trait Implementations ======================

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
            for m in layer.mlp.get_isq_layers() {
                tensors.push((m, Some(i)));
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

            match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
                    uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
                }
                LayerImpl::LinearAttention(gdn) => {
                    // Save split projection weights
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
        }

        uvb.to_safetensors()
    }
}

impl NormalModel for Model {
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
    fn cache(&self) -> &EitherCache {
        &self.kv_cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.kv_cache
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
