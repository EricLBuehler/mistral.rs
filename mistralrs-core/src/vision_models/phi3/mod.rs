#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) mod phi3_inputs_processor;

// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
use candle_core::{
    shape::ShapeWithOneHole, DType, Device, IndexOp, Module, Result, Shape, Tensor, D,
};
use either::Either;
use mistralrs_quant::{
    BitWiseOp, NonZeroOp, QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder,
};
use std::{
    any::Any,
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeTrainableLayer, MlpLayer, MoeMlp},
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    get_delta_from_lora_ab,
    layers::{
        self, Activation, CausalMasker, MatMul, PhiRopeConfig, PhiRopeScalingConfig,
        PhiRotaryEmbedding, RmsNorm, Sdpa,
    },
    layers_masker::PastKvLenCache,
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
        PagedAttention,
    },
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, VisionModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
    vision_models::clip::{ClipConfig, ClipVisionTransformer},
    AnyMoeConfig, AnyMoeExpertType,
};

use super::clip;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct EmbedLayerConfig {
    pub hd_transform_order: Option<String>,
    pub projection_cls: Option<String>,
    pub use_hd_transform: Option<bool>,
    pub with_learnable_separator: Option<bool>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct ImageProcessorConfig {
    pub image_dim_out: usize,
    pub name: String,
    pub num_img_tokens: usize,
    pub layer_idx: Option<isize>,
    pub type_feature: Option<String>,
}

serde_default_fn!(bool, word_emb_default, false);

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: usize,
    pub embd_layer: EmbedLayerConfig,
    pub img_processor: ImageProcessorConfig,
    #[serde(alias = "quantization")]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
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

trait ModuleWithMetadata: Module + Debug + Send + Sync {
    fn device(&self) -> Device;
    fn dtype(&self) -> DType;
}

#[derive(Debug)]
struct QuantMethodWrapper(Arc<dyn QuantMethod>);

impl Module for QuantMethodWrapper {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

impl ModuleWithMetadata for QuantMethodWrapper {
    fn device(&self) -> Device {
        self.0.unquant_weight_bias().unwrap().0.device().clone()
    }
    fn dtype(&self) -> DType {
        self.0.unquant_weight_bias().unwrap().0.dtype()
    }
}

impl ModuleWithMetadata for candle_nn::Activation {
    fn device(&self) -> Device {
        unreachable!()
    }
    fn dtype(&self) -> DType {
        unreachable!()
    }
}

#[derive(Debug)]
struct BigShapeWithOneHole((usize, usize, usize, usize, usize, ()));

fn hole_size(el_count: usize, prod_d: usize, s: &dyn std::fmt::Debug) -> Result<usize> {
    if prod_d == 0 {
        candle_core::bail!("cannot reshape tensor of {el_count} elements to {s:?}")
    }
    if !el_count.is_multiple_of(prod_d) {
        candle_core::bail!("cannot reshape tensor with {el_count} elements to {s:?}")
    }
    Ok(el_count / prod_d)
}

impl ShapeWithOneHole for BigShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, d4, d5, ()) = self.0;
        let d = hole_size(el_count, d1 * d2 * d3 * d4 * d5, &self)?;
        Ok((d1, d2, d3, d4, d5, d).into())
    }
}

// =================== BASE LAYERS ===================

struct Attention {
    qkv_proj: Arc<dyn QuantMethod>,
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
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;

        // No TP here.
        let qkv_proj = mistralrs_quant::linear_no_bias(
            cfg.hidden_size,
            op_size,
            &cfg.quantization_config,
            vb.pp("qkv_proj"),
        )?;

        let o_proj = mistralrs_quant::linear_no_bias(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
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
        if let Some(t) = self.qkv_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut qkv = MatMul.qmethod_matmul(&xs, &*self.qkv_proj)?;
        if self.qkv_proj.quantized_act_type().is_some() {
            qkv = qkv.to_dtype(original_dtype)?;
        }
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

        let (q, k) = self
            .rotary_emb
            .forward(&q, &k, seqlen_offsets, position_ids)?;

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
                    assert!(attention_mask.is_some());
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

        if let Some(t) = self.qkv_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.qkv_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Clone)]
struct Mlp {
    gate_up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
    i_size: usize,
    params: Vec<usize>,
}

impl Mlp {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;

        // No TP here.
        let gate_up_proj = mistralrs_quant::linear_no_bias(
            hidden_size,
            2 * i_size,
            &cfg.quantization_config,
            vb.pp("gate_up_proj"),
        )?;

        let down_proj = mistralrs_quant::linear_no_bias(
            i_size,
            hidden_size,
            &cfg.quantization_config,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
            params: vec![hidden_size, i_size],
        })
    }
}

impl AnyMoeTrainableLayer for Mlp {}

impl MlpLayer for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_up_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let up_states = MatMul.qmethod_matmul(&xs, &*self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        let mut res = MatMul.qmethod_matmul(&up_states, &*self.down_proj)?;
        if self.gate_up_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.gate_up_proj, &mut self.down_proj]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Clone::clone(self))
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    fn hidden_act(&self) -> Activation {
        self.act_fn
    }
    // gate_up, down
    fn new_added_delta(&self, deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        let new_gate_up = if let Some(ref delta) = deltas[0] {
            self.gate_up_proj.add_delta_w(delta)?
        } else {
            self.gate_up_proj.clone()
        };
        let new_down = if let Some(ref delta) = deltas[1] {
            self.down_proj.add_delta_w(delta)?
        } else {
            self.down_proj.clone()
        };

        Ok(Box::new(Self {
            gate_up_proj: new_gate_up,
            down_proj: new_down,
            act_fn: self.act_fn,
            i_size: self.i_size,
            params: self.params.clone(),
        }))
    }

    fn dtype_device(&self) -> (DType, Device) {
        self.gate_up_proj.dtype_and_device()
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
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
            self_attn,
            mlp: Box::new(mlp),
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
        let xs = self
            .self_attn
            .forward(
                &xs,
                attention_mask,
                seqlen_offsets,
                position_ids,
                kv_cache,
                metadata,
                flash_params,
            )
            .unwrap();
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

// =================== ============= ===================

// =================== VISION LAYERS ===================

const MAX_INPUT_ID: f64 = 1e9;

#[derive(Debug)]
struct EmbeddingLayers(Vec<Box<dyn ModuleWithMetadata>>);

impl Module for EmbeddingLayers {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.0 {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
pub struct ImageEmbedding {
    wte: candle_nn::Embedding,
    image_dim_out: usize,
    num_img_tokens: usize,
    glb_gn: Option<Tensor>,
    sub_gn: Option<Tensor>,
    layers: EmbeddingLayers,
    type_feature: String,
    layer_idx: isize,
    image_processor: ClipVisionTransformer,
    hd_transform_order: String,
    use_hd_transform: bool,
    vocab_size: usize,
    tensors: Vec<(String, Tensor)>,
}

pub(crate) const PHI3V_CLIP_CONFIG: ClipConfig = ClipConfig {
    hidden_act: clip::Activation::QuickGelu,
    hidden_size: 1024,
    image_size: 336,
    intermediate_size: 4096,
    num_attention_heads: 16,
    num_channels: 3,
    num_hidden_layers: 24,
    patch_size: 14,
};

impl ImageEmbedding {
    fn new(
        config: &Config,
        wte: candle_nn::Embedding,
        embed_config: &EmbedLayerConfig,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        if config.img_processor.name != "clip_vision_model" {
            candle_core::bail!(
                "img_processor=`{}` nor supported.",
                config.img_processor.name
            );
        }
        let image_dim_out = config.img_processor.image_dim_out;
        let num_img_tokens = config.img_processor.num_img_tokens;

        // CLIP image processor here...
        let image_processor =
            ClipVisionTransformer::new(vb.pp("img_processor.vision_model"), &PHI3V_CLIP_CONFIG)?;

        // High dim transform
        let use_hd_transform = embed_config.use_hd_transform.unwrap_or(false);
        let with_learnable_separator = embed_config.with_learnable_separator.unwrap_or(false);
        let hd_transform_order = embed_config
            .hd_transform_order
            .clone()
            .unwrap_or("glb_sub".to_string());
        assert_eq!(use_hd_transform, with_learnable_separator);
        let (glb_gn, sub_gn) = if with_learnable_separator {
            let glb_gn = vb.get((1, 1, image_dim_out * 4), "glb_GN")?;
            let sub_gn = vb.get((1, 1, 1, image_dim_out * 4), "sub_GN")?;
            (Some(glb_gn), Some(sub_gn))
        } else {
            (None, None)
        };

        // Inner projection
        let projection_cls = embed_config
            .projection_cls
            .clone()
            .unwrap_or("linear".to_string());

        let mut tensors = Vec::new();
        let layers: Vec<Box<dyn ModuleWithMetadata>> =
            match (projection_cls.as_str(), use_hd_transform) {
                ("linear", _) => {
                    let a = mistralrs_quant::linear_b(
                        image_dim_out,
                        hidden_size,
                        true,
                        &None,
                        vb.pp("img_projection"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.bias".to_string(), b));
                    }
                    vec![Box::new(QuantMethodWrapper(a))]
                }
                ("mlp", true) => {
                    let dim_proj = hidden_size;
                    let a = mistralrs_quant::linear_b(
                        image_dim_out * 4,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.0"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.0.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.0.bias".to_string(), b));
                    }
                    let b = mistralrs_quant::linear_b(
                        dim_proj,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.2"),
                    )?;
                    let (b_w, b_b) = b.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.2.weight".to_string(), b_w));
                    if let Some(b) = b_b {
                        tensors.push(("img_projection.2.bias".to_string(), b));
                    }
                    vec![
                        Box::new(QuantMethodWrapper(a)),
                        Box::new(candle_nn::Activation::Gelu),
                        Box::new(QuantMethodWrapper(b)),
                    ]
                }
                ("mlp", false) => {
                    let dim_proj = hidden_size;
                    let a = mistralrs_quant::linear_b(
                        image_dim_out,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.0"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.0.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.0.bias".to_string(), b));
                    }
                    let b = mistralrs_quant::linear_b(
                        dim_proj,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.2"),
                    )?;
                    let (b_w, b_b) = b.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.2.weight".to_string(), b_w));
                    if let Some(b) = b_b {
                        tensors.push(("img_projection.2.bias".to_string(), b));
                    }
                    vec![
                        Box::new(QuantMethodWrapper(a)),
                        Box::new(candle_nn::Activation::Gelu),
                        Box::new(QuantMethodWrapper(b)),
                    ]
                }
                _ => {
                    candle_core::bail!("projection_cls=`{projection_cls}` not implemented.");
                }
            };

        let layer_idx = config.img_processor.layer_idx.unwrap_or(-2);
        let type_feature = config
            .img_processor
            .type_feature
            .clone()
            .unwrap_or("patch".to_string());

        Ok(Self {
            wte,
            image_dim_out,
            num_img_tokens,
            glb_gn,
            sub_gn,
            layer_idx,
            type_feature,
            image_processor,
            layers: EmbeddingLayers(layers),
            hd_transform_order,
            use_hd_transform,
            vocab_size: config.vocab_size,
            tensors,
        })
    }

    fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = self
            .image_processor
            .forward_get_hidden_states(&pixel_values.to_dtype(self.wte.embeddings().dtype())?)?;
        let img_feature =
            hidden_states[(hidden_states.len() as isize + self.layer_idx) as usize].clone();
        if self.type_feature == "patch" {
            img_feature.i((.., 1..))
        } else if self.type_feature == "cls_patch" {
            Ok(img_feature)
        } else {
            candle_core::bail!("Unsupported image feature type {}", self.type_feature)
        }
    }

    #[allow(non_snake_case)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_sizes: Option<Vec<(usize, usize)>>,
        image_hashes: &[u64],
        encoder_cache: &Mutex<EncoderCacheManager>,
    ) -> Result<Tensor> {
        let input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;

        let input_ids_lt = input_ids.lt(0.0f64)?;
        let input_ids_gt = input_ids.gt(-MAX_INPUT_ID)?;
        // positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=False)
        let positions = input_ids_lt.bitwise_and(&input_ids_gt)?.nonzero()?;
        let target_dev = self.layers.0[0].device();
        let target_dtype = self.layers.0[0].dtype();

        let mut select = false;
        // If some, use hd transform case and it contains num_img_toks
        let mut hd_transform = None;
        let mut image_set_tensor = None;
        let n_hashes = image_hashes.len();
        if positions.dim(0)? > 0 {
            select = true;
            // input_ids[positions[:, 0], positions[:, 1]]
            if self.use_hd_transform {
                if let Some(image_sizes_ref) = image_sizes.as_ref() {
                    assert_eq!(pixel_values.dims().len(), 5);
                    let bs = pixel_values.dim(0)?;

                    // Check cache for each image
                    let mut per_image_cached: Vec<Option<Tensor>> = vec![None; bs];
                    let mut miss_indices = Vec::new();
                    if n_hashes > 0 && n_hashes == bs {
                        let mut guard = encoder_cache.lock().expect("encoder cache lock poisoned");
                        for (i, &hash) in image_hashes.iter().enumerate() {
                            if let Some(cached) = guard.get(hash) {
                                per_image_cached[i] = Some(cached[0].clone());
                            } else {
                                miss_indices.push(i);
                            }
                        }
                    } else {
                        miss_indices = (0..bs).collect();
                    }

                    // Only run CLIP on miss images
                    let mut img_features_per_image: Vec<Option<Tensor>> = vec![None; bs];
                    if !miss_indices.is_empty() {
                        // We need CLIP features for all miss images
                        let miss_pv: Vec<Tensor> = miss_indices
                            .iter()
                            .map(|&i| pixel_values.get(i))
                            .collect::<Result<Vec<_>>>()?;
                        let miss_pv = Tensor::stack(&miss_pv, 0)?;
                        let miss_features = self.get_image_features(&miss_pv.flatten(0, 1)?)?;
                        let base_feat_dim = (miss_features.dims()[1] as f32).sqrt() as usize;
                        assert_eq!(base_feat_dim, 24);
                        let miss_bs = miss_indices.len();
                        let miss_features = miss_features.reshape((
                            miss_bs,
                            (),
                            base_feat_dim.pow(2),
                            self.image_dim_out,
                        ))?;
                        for (batch_idx, &orig_idx) in miss_indices.iter().enumerate() {
                            img_features_per_image[orig_idx] = Some(miss_features.get(batch_idx)?);
                        }
                    }

                    let C = self.image_dim_out;
                    let H = 24usize; // base_feat_dim

                    let mut image_set_tensor_inner = Vec::new();
                    let mut output_len = Vec::new();
                    for (bs_, &(h, w)) in image_sizes_ref.iter().enumerate().take(bs) {
                        // Check if we have a cache hit for this image
                        if let Some(ref cached_tensor) = per_image_cached[bs_] {
                            let cnt = cached_tensor.dim(1)?;
                            output_len.push(cnt);
                            image_set_tensor_inner.push(cached_tensor.clone());
                            continue;
                        }

                        let img_feats = img_features_per_image[bs_]
                            .as_ref()
                            .expect("miss image should have features");

                        let h = h / 336;
                        let w = w / 336;
                        let B_ = h * w;

                        // 1 x (24x24) x 1024
                        let global_img_feature = img_feats.i(..1)?;

                        // 1 x 12 x 12 x 4096
                        let glb_img = global_img_feature
                            .reshape((1, H, H, C))?
                            .reshape((1, H / 2, 2, H / 2, 2, C))?
                            .contiguous()?
                            .permute((0, 1, 3, 2, 4, 5))?
                            .reshape((1, H / 2, H / 2, 4 * C))?
                            .contiguous()?;
                        let temp_glbl_gn = self
                            .sub_gn
                            .as_ref()
                            .expect("Need `sub_gn` if `use_hd_transform`")
                            .repeat((1, H / 2, 1, 1))?;

                        // 1 x 156 x 4096
                        let glb_img =
                            Tensor::cat(&[glb_img, temp_glbl_gn], 2)?.reshape((1, (), 4 * C))?;

                        // (max_num_crops-1) x (12x12) x C
                        let sub_img = img_feats.i(1..)?;

                        // Get rid of padding sub_img
                        let sub_img = sub_img.i(..B_)?;

                        // (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                        let sub_img = sub_img
                            .reshape((B_, H, H, C))?
                            .reshape((B_, H / 2, 2, H / 2, 2, C))?
                            .contiguous()?
                            .permute((0, 1, 3, 2, 4, 5))?
                            .reshape((B_, (), 4 * C))?
                            .contiguous()?;
                        let sub_img = sub_img
                            .reshape(BigShapeWithOneHole((1usize, h, w, 12usize, 12usize, ())))?
                            .permute((0, 1, 3, 2, 4, 5))?
                            .reshape((1, h * 12, w * 12, 4 * C))?;
                        let temp_sub_gn = self
                            .sub_gn
                            .as_ref()
                            .expect("Need `sub_gn` if `use_hd_transform`")
                            .repeat((1, h * 12, 1, 1))?;

                        let sub_img =
                            Tensor::cat(&[sub_img, temp_sub_gn], 2)?.reshape((1, (), 4 * C))?;

                        // (1, num_img_tokens, 1024*4)

                        let img = match self.hd_transform_order.as_str() {
                            "glb_sub" => Tensor::cat(
                                &[
                                    glb_img,
                                    self.glb_gn
                                        .as_ref()
                                        .expect("Need `glb_gn` if `use_hd_transform`")
                                        .clone(),
                                    sub_img,
                                ],
                                1,
                            )?,
                            "sub_glb" => Tensor::cat(
                                &[
                                    sub_img,
                                    self.glb_gn
                                        .as_ref()
                                        .expect("Need `glb_gn` if `use_hd_transform`")
                                        .clone(),
                                    glb_img,
                                ],
                                1,
                            )?,
                            other => {
                                candle_core::bail!("Invalid hd_transform_order=`{other}`");
                            }
                        };

                        let temp_len = (h * w + 1) * 144 + 1 + (h + 1) * 12;
                        assert_eq!(temp_len, img.dims()[1]);
                        output_len.push(temp_len);

                        let layerout = self
                            .layers
                            .forward(&img.to_device(&target_dev)?.to_dtype(target_dtype)?)?;

                        // Cache the projected features for this image
                        if n_hashes > 0 && bs_ < n_hashes {
                            let mut guard =
                                encoder_cache.lock().expect("encoder cache lock poisoned");
                            guard.insert(image_hashes[bs_], vec![layerout.clone()]);
                        }

                        image_set_tensor_inner.push(layerout);
                    }

                    hd_transform = Some(output_len);
                    image_set_tensor = Some(Either::Left(image_set_tensor_inner));
                }
            } else if pixel_values.dims().len() == 4 {
                let n_imgs = pixel_values.dim(0)?;
                if n_hashes > 0 && n_hashes == n_imgs {
                    // Per-image caching for non-HD path
                    let mut per_image_features: Vec<Option<Tensor>> = vec![None; n_imgs];
                    let mut miss_indices = Vec::new();
                    {
                        let mut guard = encoder_cache.lock().expect("encoder cache lock poisoned");
                        for (i, &hash) in image_hashes.iter().enumerate() {
                            if let Some(cached) = guard.get(hash) {
                                per_image_features[i] = Some(cached[0].clone());
                            } else {
                                miss_indices.push(i);
                            }
                        }
                    }
                    if !miss_indices.is_empty() {
                        for &idx in &miss_indices {
                            let single_pv = pixel_values.get(idx)?.unsqueeze(0)?;
                            let tt = self
                                .get_image_features(&single_pv)?
                                .to_device(&target_dev)?
                                .to_dtype(target_dtype)?
                                .reshape(((), self.image_dim_out))?;
                            let feats = self.layers.forward(&tt)?;
                            {
                                let mut guard =
                                    encoder_cache.lock().expect("encoder cache lock poisoned");
                                guard.insert(image_hashes[idx], vec![feats.clone()]);
                            }
                            per_image_features[idx] = Some(feats);
                        }
                    }
                    let all_feats: Vec<Tensor> = per_image_features
                        .into_iter()
                        .map(|f| f.expect("all images should be resolved"))
                        .collect();
                    let image_set_tensor_inner = Tensor::cat(&all_feats, 0)?;
                    image_set_tensor = Some(Either::Right(image_set_tensor_inner));
                } else {
                    let tt = self
                        .get_image_features(pixel_values)?
                        .to_device(&target_dev)?
                        .to_dtype(target_dtype)?
                        .reshape(((), self.image_dim_out))?;
                    let image_set_tensor_inner = self.layers.forward(&tt)?;
                    image_set_tensor = Some(Either::Right(image_set_tensor_inner));
                }
            } else if pixel_values.dims().len() == 3 {
                let tt = pixel_values
                    .to_device(&target_dev)?
                    .to_dtype(target_dtype)?
                    .reshape(((), self.image_dim_out))?;
                let image_set_tensor_inner = self.layers.forward(&tt)?;
                image_set_tensor = Some(Either::Right(image_set_tensor_inner));
            } else {
                unreachable!()
            }
        }

        let input_ids = input_ids.clamp(0.0, self.vocab_size as f64)?;
        let mut hidden_states = self.wte.forward(&input_ids)?;
        if select {
            match (hd_transform, image_set_tensor) {
                (Some(output_lens), Some(Either::Left(image_set_tensors))) => {
                    let mut idx = 0;
                    for (i, cnt) in output_lens.into_iter().enumerate() {
                        let img_set_tensor = image_set_tensors[i]
                            .to_device(&target_dev)?
                            .to_dtype(target_dtype)?;
                        // hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = ...
                        let p_0 = positions.i((idx, 0))?.to_scalar::<u32>()? as usize;
                        let p_1 = positions.i((idx, 1))?.to_scalar::<u32>()? as usize;
                        hidden_states = hidden_states.slice_assign(
                            &[p_0..p_0 + 1, p_1..p_1 + cnt, 0..img_set_tensor.dims()[2]],
                            &img_set_tensor,
                        )?;
                        idx += cnt;
                    }
                }
                (None, Some(Either::Right(image_set_tensor))) => {
                    let mut idx = 0;
                    // Know len(img_embeds) == pixel_values.dim(0) == len(selected_g_values)
                    // https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/dbcdaaacf52c8e40cf8de6d6ffa6ff6860e5f256/image_embedding_phi3_v.py#L259
                    for i in 0..pixel_values.dim(0)? {
                        let cnt = self.num_img_tokens;
                        let img_set_tensor = image_set_tensor
                            .i(i * cnt..(i + 1) * cnt)?
                            .to_device(&target_dev)?
                            .to_dtype(target_dtype)?;
                        let p_0 = positions.i((idx, 0))?.to_scalar::<u32>()? as usize;
                        let p_1 = positions.i((idx, 1))?.to_scalar::<u32>()? as usize;
                        // hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = ...
                        hidden_states = hidden_states.slice_assign(
                            &[p_0..p_0 + 1, p_1..p_1 + cnt, 0..img_set_tensor.dims()[2]],
                            &img_set_tensor,
                        )?;
                        idx += cnt;
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(hidden_states)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        if let Some(glb_gn) = self.glb_gn.clone() {
            uvb.add_tensor("glb_GN", glb_gn);
        }
        if let Some(sub_gn) = self.sub_gn.clone() {
            uvb.add_tensor("sub_GN", sub_gn);
        }
        uvb.extend(self.tensors.clone());
        uvb.pp("img_processor.vision_model")
            .extend(self.image_processor.residual_tensors());

        uvb.to_safetensors()
    }
}

// =================== ============= ===================

pub struct Model {
    vision_embed_tokens: ImageEmbedding,
    embed_tokens: candle_nn::Embedding,
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

impl Model {
    pub fn new(
        cfg: &Config,
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
        let vision_embed_tokens = ImageEmbedding::new(
            cfg,
            embed_tokens.clone(),
            &cfg.embd_layer,
            mapper.set_nm_device(vb_m.pp("vision_embed_tokens"), false),
        )?;
        let vb_l = vb_m.pp("layers");
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

        Ok(Self {
            vision_embed_tokens,
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
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        context_lens: Vec<(usize, usize)>,
        image_sizes: Option<Vec<(usize, usize)>>,
        image_hashes: &[u64],
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = if let Some(ref pixel_values) = pixel_values {
            self.vision_embed_tokens.forward(
                input_ids,
                pixel_values,
                image_sizes,
                image_hashes,
                &self.encoder_cache,
            )?
        } else {
            self.embed_tokens.forward(input_ids)?
        };
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*cache as &dyn PastKvLenCache),
            self.sliding_window,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
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
            tensors.push((&mut layer.self_attn.qkv_proj, Some(i)));
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
        uvb_m
            .pp("vision_embed_tokens")
            .extend(self.vision_embed_tokens.residual_tensors());

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

#[derive(Default)]
pub(crate) struct Phi3VisionSpecificArgs {
    pub image_sizes: Option<Vec<(usize, usize)>>,
    pub image_hashes: Vec<u64>,
}

impl VisionModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let Phi3VisionSpecificArgs {
            image_sizes,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Phi3VisionSpecificArgs`");
        self.forward(
            input_ids,
            pixel_values,
            seqlen_offsets,
            &position_ids,
            context_lens,
            image_sizes,
            &image_hashes,
            metadata,
            flash_params,
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
        Box::new(Phi3VisionSpecificArgs::default())
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

impl AnyMoeBaseModelMixin for Model {
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
                        row.push(Box::new(Mlp::new(
                            &Config {
                                intermediate_size: self.layers[layer].mlp.get_params()[1],
                                hidden_size: self.layers[layer].mlp.get_params()[0],
                                ..Default::default()
                            },
                            vb.pp(layer).pp(&mlp),
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer).pp(&mlp);

                        let gate_up_proj_delta =
                            if target_modules.contains(&"gate_up_proj".to_string()) {
                                Some(get_delta_from_lora_ab!(
                                    vb_mlp,
                                    rank,
                                    alpha,
                                    (hidden_size, 2 * intermediate_size),
                                    "gate_up_proj"
                                ))
                            } else {
                                None
                            };
                        let down_proj_delta = if target_modules.contains(&"down_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "down_proj"
                            ))
                        } else {
                            None
                        };

                        row.push(
                            self.layers[layer]
                                .mlp
                                .new_added_delta(vec![gate_up_proj_delta, down_proj_delta])?,
                        );
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
