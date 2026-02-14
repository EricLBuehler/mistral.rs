#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) mod idefics2_input_processor;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Module};
use mistralrs_quant::{Convolution, ShardedVarBuilder};
use serde::Deserialize;
use std::{
    any::Any,
    ops::Mul,
    sync::{Arc, Mutex},
};

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    device_map::DeviceMapper,
    layers::{
        conv2d, embedding, layer_norm, linear, linear_no_bias, repeat_kv, Activation, CausalMasker,
        MatMul, QLinear, RmsNorm,
    },
    models::mistral::Model as Mistral,
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    AnyMoeConfig, AnyMoeExpertType,
};

use crate::models::mistral;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py

fn default_32000() -> usize {
    32000
}
fn default_32001() -> usize {
    32001
}
fn default_4096() -> usize {
    4096
}
fn default_14336() -> usize {
    14336
}
fn default_32() -> usize {
    32
}
fn default_8() -> usize {
    8
}
fn default_act() -> Activation {
    Activation::Silu
}
fn default_131072() -> usize {
    131072
}
fn default_eps() -> f64 {
    1e-6
}
fn default_rope() -> f64 {
    10000.0
}
fn default_false() -> bool {
    false
}
fn default_sliding() -> Option<usize> {
    Some(4096)
}
fn default_gelu() -> Activation {
    Activation::GeluPytorchTanh
}
fn default_64() -> usize {
    64
}
fn default_3() -> usize {
    3
}
fn default_16() -> usize {
    16
}
fn default_96() -> usize {
    96
}
fn default_4() -> usize {
    4
}
fn default_0_0() -> f32 {
    0.0
}
fn default_0_02() -> f32 {
    0.02
}
fn default_768() -> usize {
    768
}
fn default_3072() -> usize {
    3072
}
fn default_12() -> usize {
    12
}
fn default_224() -> usize {
    224
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PerceiverConfig {
    #[serde(default = "default_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_64")]
    pub resampler_n_latents: usize,
    #[serde(default = "default_3")]
    pub resampler_depth: usize,
    #[serde(default = "default_16")]
    pub resampler_n_heads: usize,
    #[serde(default = "default_96")]
    pub resampler_head_dim: usize,
    #[serde(default = "default_4")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_0_0")]
    pub attention_dropout: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_768")]
    pub hidden_size: usize,
    #[serde(default = "default_3072")]
    pub intermediate_size: usize,
    #[serde(default = "default_12")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_12")]
    pub num_attention_heads: usize,
    #[serde(default = "default_3")]
    pub num_channels: usize,
    #[serde(default = "default_224")]
    pub image_size: usize,
    #[serde(default = "default_32")]
    pub patch_size: usize,
    #[serde(default = "default_gelu")]
    pub hidden_act: Activation,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_0_0")]
    pub attn_dropout: f32,
    #[serde(default = "default_0_02")]
    pub initializer_range: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct TextConfig {
    #[serde(default = "default_32000")]
    pub(crate) vocab_size: usize,
    #[serde(default = "default_4096")]
    pub(crate) hidden_size: usize,
    #[serde(default = "default_14336")]
    pub(crate) intermediate_size: usize,
    #[serde(default = "default_32")]
    pub(crate) num_hidden_layers: usize,
    #[serde(default = "default_32")]
    pub(crate) num_attention_heads: usize,
    #[serde(default = "default_8")]
    pub(crate) num_key_value_heads: usize,
    #[serde(default = "default_act")]
    pub(crate) hidden_act: Activation,
    #[serde(default = "default_131072")]
    pub(crate) max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub(crate) rope_theta: f64,
    #[serde(default = "default_sliding")]
    pub(crate) sliding_window: Option<usize>,

    model_type: String, // Must be mistral for now
}

impl From<TextConfig> for mistral::Config {
    fn from(val: TextConfig) -> Self {
        mistral::Config {
            vocab_size: val.vocab_size,
            hidden_act: val.hidden_act,
            hidden_size: val.hidden_size,
            intermediate_size: val.intermediate_size,
            num_hidden_layers: val.num_hidden_layers,
            num_attention_heads: val.num_attention_heads,
            num_key_value_heads: val.num_key_value_heads,
            max_position_embeddings: val.max_position_embeddings,
            rms_norm_eps: val.rms_norm_eps,
            rope_theta: val.rope_theta,
            rope_parameters: None,
            sliding_window: val.sliding_window,
            head_dim: None,
            quantization_config: None,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct Config {
    pub perceiver_config: PerceiverConfig,
    pub vision_config: VisionConfig,
    pub(crate) text_config: TextConfig,
    #[serde(default = "default_32001")]
    pub image_token_id: usize,
    #[serde(default = "default_false")]
    pub tie_word_embeddings: bool,
}

// == START VISION MODEL ==

struct VisionEmbeddings {
    patch_size: usize,
    patch_embedding: Conv2d,
    num_patches_per_side: usize,
    position_embedding: Embedding,
}

/// torch.bucketize with right=True
/// Returns a 1d tensor of shape (xs.len(),) on the CPU
fn bucketize_right(xs: &[f32], boundaries: &[f32], device: &Device) -> Result<Tensor> {
    use std::cmp::Ordering;

    let mut result = Vec::with_capacity(xs.len());

    for &x in xs {
        // binary_search_by returns:
        //   Ok(i)   if boundaries[i] == x
        //   Err(i)  if x would be inserted at i
        //
        // The returned i is the "insertion point" for x to keep
        // boundaries sorted. That i is the smallest position
        // where boundaries[i] >= x (i.e. bisect_left).

        let idx = match boundaries.binary_search_by(|&val| {
            // Use partial_cmp here; assume no NaNs.
            // For robust handling of NaNs, you might need a custom comparison.
            val.partial_cmp(&x).unwrap_or(Ordering::Less)
        }) {
            Ok(i) => i + 1,
            Err(i) => i,
        };

        result.push(idx as u32);
    }

    Tensor::from_vec(result, (xs.len(),), device)
}

impl VisionEmbeddings {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_config = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            conv_config,
            vb.pp("patch_embedding"),
        )?;
        let num_patches_per_side = config.image_size / config.patch_size;
        let num_patches = num_patches_per_side.pow(2);
        Ok(Self {
            patch_size: config.patch_size,
            patch_embedding,
            num_patches_per_side,
            position_embedding: embedding(
                num_patches,
                config.hidden_size,
                vb.pp("position_embedding"),
                &None,
            )?,
        })
    }

    fn forward(&self, pixel_values: &Tensor, patch_attention_mask: &Tensor) -> Result<Tensor> {
        let (bs, _, max_im_h, max_im_w) = pixel_values.dims4()?;

        let patch_embeds = Convolution.forward_2d(&self.patch_embedding, pixel_values)?;

        let embeddings = patch_embeds.flatten(2, D::Minus1)?.transpose(1, 2)?;

        let (max_nb_patches_h, max_nb_patches_w) =
            (max_im_h / self.patch_size, max_im_w / self.patch_size);
        let boundaries = Tensor::arange_step(
            1.0 / self.num_patches_per_side as f32,
            1.0,
            1.0 / self.num_patches_per_side as f32,
            pixel_values.device(),
        )?
        .to_vec1::<f32>()?;
        let position_ids = Tensor::full(
            0u32,
            (bs, max_nb_patches_h * max_nb_patches_w),
            pixel_values.device(),
        )?;

        let mut new_position_ids = Vec::new();
        for (b_idx, p_attn_mask) in patch_attention_mask.chunk(bs, 0)?.iter().enumerate() {
            let p_attn_mask = p_attn_mask.squeeze(0)?;
            let nb_patches_h = p_attn_mask.i((.., 0))?.sum_all()?;
            let nb_patches_w = p_attn_mask.i((0,))?.sum_all()?;

            let fractional_coords_h = Tensor::arange_step(
                0.0,
                1.0 - 1e-6,
                1.0 / nb_patches_h.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                pixel_values.device(),
            )?
            .to_vec1::<f32>()?;
            let fractional_coords_w = Tensor::arange_step(
                0.0,
                1.0 - 1e-6,
                1.0 / nb_patches_w.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                pixel_values.device(),
            )?
            .to_vec1::<f32>()?;

            let bucket_coords_h =
                bucketize_right(&fractional_coords_h, &boundaries, pixel_values.device())?;
            let bucket_coords_w =
                bucketize_right(&fractional_coords_w, &boundaries, pixel_values.device())?;

            let pos_ids = bucket_coords_h
                .unsqueeze(D::Minus1)?
                .mul(self.num_patches_per_side as f64)?
                .broadcast_add(&bucket_coords_w)?
                .flatten_all()?
                .to_vec1::<u32>()?;

            let true_indices = p_attn_mask
                .flatten_all()?
                .to_vec1::<u8>()?
                .iter()
                .enumerate()
                .filter_map(|(i, x)| if *x != 0 { Some(i) } else { None })
                .collect::<Vec<_>>();
            let position_ids_b = position_ids.i(b_idx)?;

            let mut new_position_ids_b = position_ids_b.to_vec1::<u32>()?;
            let new_position_ids_b_len = new_position_ids_b.len();
            for (i, true_idx) in true_indices.into_iter().enumerate() {
                new_position_ids_b[true_idx] = pos_ids[i];
            }

            new_position_ids.push(Tensor::from_vec(
                new_position_ids_b,
                new_position_ids_b_len,
                pixel_values.device(),
            )?);
        }
        let position_ids = Tensor::stack(&new_position_ids, 0)?;
        let position_ids = position_ids.to_device(self.position_embedding.embeddings().device())?;
        embeddings.broadcast_add(&self.position_embedding.forward(&position_ids)?)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("patch_embedding").add(&self.patch_embedding);
        uvb.pp("position_embedding").add(&self.position_embedding);

        uvb.to_safetensors()
    }
}

struct Attention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    neg_inf: Tensor,
}

impl Attention {
    fn new(config: VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let o_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            scale,
            q_proj: QLinear::from_linear(q_proj),
            k_proj: QLinear::from_linear(k_proj),
            v_proj: QLinear::from_linear(v_proj),
            o_proj: QLinear::from_linear(o_proj),
            neg_inf: Tensor::new(f32::NEG_INFINITY, vb.device())?.to_dtype(vb.dtype())?,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if self.q_proj.is_quant() {
            xs = xs.to_dtype(DType::F32)?;
        }
        let mut q = self.q_proj.forward(&xs)?;
        let mut k = self.k_proj.forward(&xs)?;
        let mut v = self.v_proj.forward(&xs)?;
        if self.q_proj.is_quant() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let attn_weights =
            (MatMul.matmul(&q.contiguous()?, &k.transpose(2, 3)?.contiguous()?)? * self.scale)?;

        let attn_weights = CausalMasker.apply_mask_one_and_zero(
            &attention_mask.map(|x| x.to_dtype(DType::U8).unwrap()),
            attn_weights,
            &self.neg_inf,
        )?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let mut attn_output = MatMul.matmul(&attn_weights, &v.contiguous()?)?;

        if self.q_proj.is_quant() {
            attn_output = attn_output.to_dtype(DType::F32)?;
        }
        let mut res = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.embed_dim))?
            .apply(&self.o_proj)?;
        if self.q_proj.is_quant() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("out_proj").add(&self.o_proj);

        uvb.to_safetensors()
    }
}

struct VisionMLP {
    activation: Activation,
    fc1: QLinear,
    fc2: QLinear,
}

impl VisionMLP {
    fn new(config: VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let fc1 = linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            activation: config.hidden_act,
            fc1: QLinear::from_linear(fc1),
            fc2: QLinear::from_linear(fc2),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        let original_dtype = x.dtype();
        if self.fc1.is_quant() {
            x = x.to_dtype(DType::F32)?;
        }
        let x = self.fc1.forward(&x)?;
        let x = self.activation.forward(&x)?;
        let mut res = self.fc2.forward(&x)?;
        if self.fc1.is_quant() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("fc1").add(&self.fc1);
        uvb.pp("fc2").add(&self.fc2);

        uvb.to_safetensors()
    }
}

struct EncoderLayer {
    mlp: VisionMLP,
    attn: Attention,
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
}

impl EncoderLayer {
    fn new(config: VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mlp = VisionMLP::new(config.clone(), vb.pp("mlp"))?;
        let attn = Attention::new(config.clone(), vb.pp("self_attn"))?;
        let layer_norm_1 = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm1"),
        )?;
        let layer_norm_2 = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm2"),
        )?;
        Ok(Self {
            mlp,
            attn,
            layer_norm_1,
            layer_norm_2,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();

        let hidden_states = self.layer_norm_1.forward(xs)?;
        let hidden_states = self.attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = (hidden_states + residual)?;

        let residual = &hidden_states;
        let hidden_states = self.layer_norm_2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        hidden_states + residual
    }
}

struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let vb_l = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(EncoderLayer::new(config.clone(), vb_l.pp(i))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = xs.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

struct VisionTransformer {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
    config: VisionConfig,
}

impl VisionTransformer {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(config, vb.pp("embeddings"))?;
        let post_layernorm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("post_layernorm"),
        )?;
        let encoder = Encoder::new(config, vb.pp("encoder"))?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
            config: config.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let bs = pixel_values.dim(0)?;
        let patch_attention_mask = if let Some(attn_mask) = attention_mask {
            attn_mask.clone()
        } else {
            let patch_size = self.config.patch_size;
            Tensor::ones(
                (
                    bs,
                    pixel_values.dim(2)? / patch_size,
                    pixel_values.dim(3)? / patch_size,
                ),
                DType::U8,
                pixel_values.device(),
            )?
        };

        let hidden_states = self
            .embeddings
            .forward(pixel_values, &patch_attention_mask)?;

        let attention_mask = if attention_mask.is_none() {
            None
        } else {
            let mask = patch_attention_mask
                .reshape((patch_attention_mask.dim(0)?, ()))?
                .to_dtype(hidden_states.dtype())?;
            Some(CausalMasker.expand_mask(&mask, hidden_states.dtype(), None)?)
        };
        let hidden_states = self
            .encoder
            .forward(&hidden_states, attention_mask.as_ref())?;
        hidden_states.apply(&self.post_layernorm)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("post_layernorm").add(&self.post_layernorm);
        uvb.pp("embeddings")
            .extend(self.embeddings.residual_tensors());

        let uvb_enc = uvb.pp("encoder");
        for (i, layer) in self.encoder.layers.iter().enumerate() {
            let uvb_l = uvb_enc.pp("layers").pp(i);

            uvb_l.pp("layer_norm1").add(&layer.layer_norm_1);
            uvb_l.pp("layer_norm2").add(&layer.layer_norm_2);
            uvb_l.pp("mlp").extend(layer.mlp.residual_tensors());
            uvb_l.pp("self_attn").extend(layer.attn.residual_tensors());
        }

        uvb.to_safetensors()
    }
}

// == END VISION MODEL ==

// == START CONNECTOR ==
struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    activation: Activation,
}

impl Mlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        output_size: usize,
        activation: Activation,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, output_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj: QLinear::from_linear(gate_proj),
            up_proj: QLinear::from_linear(up_proj),
            down_proj: QLinear::from_linear(down_proj),
            activation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        let original_dtype = x.dtype();
        if self.gate_proj.is_quant() {
            x = x.to_dtype(DType::F32)?;
        }
        let mut res = self.down_proj.forward(
            &(self.activation.forward(&self.gate_proj.forward(&x)?)?
                * self.up_proj.forward(&x)?)?,
        )?;
        if self.gate_proj.is_quant() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("gate_proj").add(&self.gate_proj);
        uvb.pp("up_proj").add(&self.up_proj);
        uvb.pp("down_proj").add(&self.down_proj);

        uvb.to_safetensors()
    }
}

struct PerceiverAttention {
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    neg_inf: Tensor,
}

impl PerceiverAttention {
    fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_size = config.text_config.hidden_size;
        let num_heads = config.perceiver_config.resampler_n_heads;
        let head_dim = config.perceiver_config.resampler_head_dim;
        let num_key_value_heads = config.perceiver_config.num_key_value_heads;
        let num_key_value_groups = num_heads / num_key_value_heads;

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            num_heads,
            head_dim,
            q_proj: QLinear::from_linear(q_proj),
            k_proj: QLinear::from_linear(k_proj),
            v_proj: QLinear::from_linear(v_proj),
            o_proj: QLinear::from_linear(o_proj),
            neg_inf: Tensor::new(f32::NEG_INFINITY, vb.device())?.to_dtype(vb.dtype())?,
            num_kv_heads: num_key_value_heads,
            num_kv_groups: num_key_value_groups,
        })
    }

    fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = latents.dims3()?;
        let kv_seq_len = q_len + context.dims()[1];

        let mut hidden_states = Tensor::cat(&[context, latents], D::Minus2)?;

        let original_dtype = latents.dtype();
        let mut latents = latents.clone();
        if self.q_proj.is_quant() {
            latents = latents.to_dtype(DType::F32)?;
            hidden_states = hidden_states.to_dtype(DType::F32)?;
        }
        let mut q = self.q_proj.forward(&latents)?;
        let mut k = self.k_proj.forward(&hidden_states)?;
        let mut v = self.v_proj.forward(&hidden_states)?;
        if self.q_proj.is_quant() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, kv_seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, kv_seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let attn_weights = (MatMul.matmul(&q.contiguous()?, &k.transpose(2, 3)?.contiguous()?)?
            / (self.head_dim as f64).sqrt())?;

        let attn_weights = CausalMasker.apply_mask_one_and_zero(
            &Some(attention_mask.to_dtype(DType::U8)?),
            attn_weights,
            &self.neg_inf,
        )?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let mut attn_output = MatMul.matmul(&attn_weights, &v.contiguous()?)?;

        if self.q_proj.is_quant() {
            attn_output = attn_output.to_dtype(DType::F32)?;
        }
        let mut res = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)?;
        if self.q_proj.is_quant() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("o_proj").add(&self.o_proj);

        uvb.to_safetensors()
    }
}

struct PerceiverLayer {
    input_latents_norm: RmsNorm,
    input_context_norm: RmsNorm,
    self_attn: PerceiverAttention,
    post_attn_norm: RmsNorm,
    mlp: Mlp,
}

impl PerceiverLayer {
    fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_size = config.text_config.hidden_size;
        let mlp_act = config.perceiver_config.hidden_act;
        let rms_eps = config.text_config.rms_norm_eps;

        Ok(Self {
            input_latents_norm: RmsNorm::new(hidden_size, rms_eps, vb.pp("input_latents_norm"))?,
            input_context_norm: RmsNorm::new(hidden_size, rms_eps, vb.pp("input_context_norm"))?,
            self_attn: PerceiverAttention::new(config, vb.pp("self_attn"))?,
            post_attn_norm: RmsNorm::new(hidden_size, rms_eps, vb.pp("post_attention_layernorm"))?,
            mlp: Mlp::new(
                hidden_size,
                hidden_size * 4,
                hidden_size,
                mlp_act,
                vb.pp("mlp"),
            )?,
        })
    }

    fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = latents;

        let latents = self.input_latents_norm.forward(latents)?;
        let context = self.input_context_norm.forward(context)?;

        let latents = self.self_attn.forward(&latents, &context, attention_mask)?;
        let latents = (residual + latents)?;
        let residual = &latents;

        let latents = self.post_attn_norm.forward(&latents)?;
        let latents = self.mlp.forward(&latents)?;
        residual + latents
    }
}

struct PerceiverResampler {
    latents: Tensor,
    layers: Vec<PerceiverLayer>,
    norm: RmsNorm,
    n_latents: usize,
}

impl PerceiverResampler {
    fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let n_latents = config.perceiver_config.resampler_n_latents;
        let hidden_size = config.text_config.hidden_size;
        let depth = config.perceiver_config.resampler_depth;

        let latents = vb.get((n_latents, hidden_size), "latents")?;
        let mut layers = Vec::new();
        let vb_l = vb.pp("layers");
        for i in 0..depth {
            layers.push(PerceiverLayer::new(config, vb_l.pp(i))?);
        }
        let norm = RmsNorm::new(hidden_size, config.text_config.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            latents,
            layers,
            norm,
            n_latents,
        })
    }

    fn forward(&self, context: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut s = vec![context.dim(0)?];
        s.extend(self.latents.dims());
        let latents = self.latents.unsqueeze(0)?.expand(s)?;

        let latent_attention_mask = Tensor::ones(
            (attention_mask.dim(0)?, latents.dim(1)?),
            attention_mask.dtype(),
            attention_mask.device(),
        )?;
        let attention_mask = Tensor::cat(&[attention_mask, &latent_attention_mask], D::Minus1)?;
        let attention_mask =
            CausalMasker.expand_mask(&attention_mask, latents.dtype(), Some(self.n_latents))?;

        let mut compressed_context = latents;
        for perceiver_layer in &self.layers {
            compressed_context =
                perceiver_layer.forward(&compressed_context, context, &attention_mask)?;
        }
        self.norm.forward(&compressed_context)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("norm").add(&self.norm);
        uvb.add_tensor("latents", self.latents.clone());

        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("layers").pp(i);

            uvb_l
                .pp("input_latents_norm")
                .add(&layer.input_latents_norm);
            uvb_l
                .pp("input_context_norm")
                .add(&layer.input_context_norm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attn_norm);
            uvb_l.pp("mlp").extend(layer.mlp.residual_tensors());
            uvb_l
                .pp("self_attn")
                .extend(layer.self_attn.residual_tensors());
        }

        uvb.to_safetensors()
    }
}

struct Connector {
    modality_projection: Mlp,
    perceiver_resampler: PerceiverResampler,
}

impl Connector {
    fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let modality_projection = Mlp::new(
            config.vision_config.hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
            config.text_config.hidden_act,
            vb.pp("modality_projection"),
        )?;
        let perceiver_resampler = PerceiverResampler::new(config, vb.pp("perceiver_resampler"))?;
        Ok(Self {
            modality_projection,
            perceiver_resampler,
        })
    }

    fn forward(&self, image_hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let image_hidden_states = self.modality_projection.forward(image_hidden_states)?;
        self.perceiver_resampler
            .forward(&image_hidden_states, attention_mask)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("modality_projection")
            .extend(self.modality_projection.residual_tensors());
        uvb.pp("perceiver_resampler")
            .extend(self.perceiver_resampler.residual_tensors());

        uvb.to_safetensors()
    }
}

// == END CONNECTOR ==

// == START MODEL ==

pub(crate) struct Idefics2SpecificArgs {
    pub pixel_attention_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
}

pub struct Idefics2 {
    vision_model: VisionTransformer,
    connector: Connector,
    text_model: Mistral,
    dtype: DType,
    config: Config,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Idefics2 {
    pub fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let text_model = Mistral::new_inner(
            &config.text_config.clone().into(),
            vb_m.pp("text_model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let vision_model = VisionTransformer::new(
            &config.vision_config,
            vb_m.pp("vision_model")
                .set_device(text_model.device().clone()),
        )?;
        let connector = Connector::new(
            config,
            vb_m.pp("connector").set_device(text_model.device().clone()),
        )?;
        Ok(Self {
            vision_model,
            connector,
            text_model,
            dtype: vb.dtype(),
            config: config.clone(),
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn inputs_merger(
        &self,
        input_ids: &Tensor,
        input_embeds: &Tensor,
        image_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        // Docs copied from Transformers impl
        /*
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder (and potentially the perceiver), and that hidden state is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        */
        let (_, _, vision_hidden_size) = image_hidden_states.dims3()?;
        let bs = input_ids.dim(0)?;
        let special_image_token_mask = input_ids.eq(self.config.image_token_id as f64)?;
        let mut new_inputs_embeds = input_embeds.clone();
        let reshaped_image_hidden_states =
            image_hidden_states.reshape((bs, (), vision_hidden_size))?;
        assert_eq!(input_embeds.dim(0)?, 1);
        assert_eq!(reshaped_image_hidden_states.dim(0)?, 1);
        let special_image_token_mask = special_image_token_mask.i(0)?.to_vec1::<u8>()?;
        let mut image_hidden_state_i = 0;
        for (i, v) in special_image_token_mask.iter().enumerate() {
            if *v != 0 {
                new_inputs_embeds = new_inputs_embeds.slice_assign(
                    &[
                        0..new_inputs_embeds.dim(0)?,
                        i..i + 1,
                        0..new_inputs_embeds.dim(2)?,
                    ],
                    &reshaped_image_hidden_states
                        .i((.., image_hidden_state_i, ..))?
                        .unsqueeze(1)?,
                )?;
                image_hidden_state_i += 1;
            }
        }
        Ok(new_inputs_embeds)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        pixel_attention_mask: Option<Tensor>,
        image_hashes: &[u64],
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let input_embeds = if let Some(pixel_values) = pixel_values {
            // == START VISUAL INPUTS INTEGRATION ==
            let (batch_size, num_images, _, _, _) = pixel_values.dims5()?;
            let mut s = vec![batch_size * num_images];
            s.extend(pixel_values.dims()[2..].to_vec());
            let pixel_values = pixel_values.reshape(s)?;

            // Remove padding images which are full of 0s
            let nb_values_per_image = pixel_values.dims()[1..].iter().product::<usize>();
            let real_images_inds = pixel_values
                .eq(0.0f64)?
                .sum(vec![
                    pixel_values.dims().len() - 1,
                    pixel_values.dims().len() - 2,
                    pixel_values.dims().len() - 3,
                ])?
                .ne(nb_values_per_image as f64)?;
            let mut batches = Vec::new();
            for (batch, use_it) in pixel_values
                .chunk(pixel_values.dim(0)?, 0)?
                .iter()
                .zip(real_images_inds.chunk(real_images_inds.dim(0)?, 0)?)
            {
                let use_it = use_it.squeeze(0)?.to_scalar::<u8>()? != 0;
                if use_it {
                    batches.push(batch.clone());
                }
            }
            let pixel_values = Tensor::cat(&batches, 0)?;

            // Vision attention mask
            let pixel_attention_mask = if let Some(pixel_attention_mask) = pixel_attention_mask {
                let pixel_attention_mask = pixel_attention_mask.reshape((
                    batch_size * num_images,
                    pixel_attention_mask.dims()[2],
                    pixel_attention_mask.dims()[3],
                ))?;
                let mut batches = Vec::new();
                for (batch, use_it) in pixel_attention_mask
                    .chunk(pixel_attention_mask.dim(0)?, 0)?
                    .iter()
                    .zip(real_images_inds.chunk(real_images_inds.dim(0)?, 0)?)
                {
                    let use_it = use_it.squeeze(0)?.to_scalar::<u8>()? != 0;
                    if use_it {
                        batches.push(batch.clone());
                    }
                }
                Tensor::cat(&batches, 0)?
            } else {
                Tensor::ones(
                    (
                        pixel_values.dims()[0],
                        pixel_values.dims()[2],
                        pixel_values.dims()[3],
                    ),
                    DType::U8,
                    pixel_values.device(),
                )?
            };

            let patch_size = self.config.vision_config.patch_size;
            let patches_subgrid = pixel_attention_mask.unfold(1, patch_size, patch_size)?;
            let patches_subgrid = patches_subgrid.unfold(2, patch_size, patch_size)?;

            let patch_attention_mask = patches_subgrid
                .sum((D::Minus1, D::Minus2))?
                .eq((patch_size * patch_size) as f64)?
                .to_dtype(DType::U8)?;

            let pixel_values = pixel_values.to_dtype(self.dtype)?;

            // Get seq from vision encoder + connector, with per-image caching
            let image_hidden_states = if !image_hashes.is_empty() {
                let n = pixel_values.dim(0)?;
                let mut per_image: Vec<Option<Tensor>> = vec![None; n];
                let mut miss_indices: Vec<usize> = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in image_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(hash) {
                            per_image[i] = Some(cached[0].clone());
                        } else {
                            miss_indices.push(i);
                        }
                    }
                }
                if !miss_indices.is_empty() {
                    for &i in &miss_indices {
                        let pv = pixel_values.get(i)?.unsqueeze(0)?;
                        let mask = patch_attention_mask.get(i)?.unsqueeze(0)?;
                        let hidden = self.vision_model.forward(&pv, Some(&mask))?;
                        let hidden = self
                            .connector
                            .forward(&hidden, &mask.reshape((1_usize, ()))?)?;
                        let result = hidden.squeeze(0)?;
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            guard.insert(image_hashes[i], vec![result.clone()]);
                        }
                        per_image[i] = Some(result);
                    }
                }
                let slices: Vec<Tensor> = per_image.into_iter().map(|t| t.unwrap()).collect();
                Tensor::stack(&slices, 0)?
            } else {
                // No caching: original path
                let image_hidden_states = self
                    .vision_model
                    .forward(&pixel_values, Some(&patch_attention_mask))?;
                self.connector.forward(
                    &image_hidden_states,
                    &patch_attention_mask.reshape((pixel_values.dim(0)?, ()))?,
                )?
            };

            self.inputs_merger(
                input_ids,
                &self.text_model.get_input_embeddings(input_ids)?,
                &image_hidden_states,
            )?
        } else {
            self.text_model.get_input_embeddings(input_ids)?
        };

        self.text_model.forward_embeds(
            input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

impl IsqModel for Idefics2 {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(
            &mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>,
            Option<usize>,
        )>,
        &dyn DeviceMapper,
    ) {
        self.text_model.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m
            .pp("text_model")
            .extend(self.text_model.residual_tensors());
        uvb_m
            .pp("vision_model")
            .extend(self.vision_model.residual_tensors());
        uvb_m
            .pp("connector")
            .extend(self.connector.residual_tensors());

        uvb.to_safetensors()
    }
}

// AnyMoE is forwarded to the base model
impl AnyMoeBaseModelMixin for Idefics2 {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.text_model.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.text_model.get_mlps_mut()
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        self.text_model.create_anymoe_layers(
            additional_vbs,
            config,
            (prefix, mlp),
            layers,
            expert_type,
            gate_vb,
        )
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}

impl VisionModel for Idefics2 {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _: Vec<usize>, // Ignore, it is for phi3
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let Idefics2SpecificArgs {
            pixel_attention_mask,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Idefics2SpecificArgs`");
        self.forward_inner(
            input_ids,
            pixel_values,
            seqlen_offsets,
            context_lens,
            pixel_attention_mask,
            &image_hashes,
            metadata,
            flash_params,
        )
    }
    fn cache(&self) -> &EitherCache {
        self.text_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.text_model.cache_mut()
    }
    fn device(&self) -> &Device {
        self.text_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.text_model.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.text_model.config()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Idefics2SpecificArgs {
            pixel_attention_mask: None,
            image_hashes: vec![],
        })
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
