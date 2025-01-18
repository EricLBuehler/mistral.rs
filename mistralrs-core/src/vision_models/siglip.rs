
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    conv2d, embedding, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, Embedding,
    LayerNorm, Module, VarBuilder,
};
use serde::Deserialize;
use std::{any::Any, ops::Mul};

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer}, device_map::DeviceMapper, layers::{repeat_kv, Activation, CausalMasker, MatMul, QLinear, RmsNorm}, models::mistral::Model as Mistral, paged_attention::{AttentionImplementation, ModelConfigMetadata}, pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    }, serde_default_fn, utils::unvarbuilder::UnVarBuilder, AnyMoeConfig, AnyMoeExpertType
};

serde_default_fn!(usize, hidden_size, 768);
serde_default_fn!(usize, intermediate_size, 3072);
serde_default_fn!(usize, num_hidden_layers, 12);
serde_default_fn!(usize, num_attention_heads, 12);
serde_default_fn!(usize, num_channels, 3);
serde_default_fn!(usize, image_size, 224);
serde_default_fn!(usize, patch_size, 16);
serde_default_fn!(Activation, hidden_act, Activation::GeluPytorchTanh);
serde_default_fn!(f64, layer_norm_eps, 1e-6);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SiglipVisionConfig {
    #[serde(default = "hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "num_channels")]
    pub num_channels: usize,
    #[serde(default = "image_size")]
    pub image_size: usize,
    #[serde(default = "patch_size")]
    pub patch_size: usize,
    #[serde(default = "hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "layer_norm_eps")]
    pub layer_norm_eps: f64,
}

struct VisionEmbeddings {
    patch_size: usize,
    patch_embedding: Conv2d,
    num_patches_per_side: usize,
    position_embedding: Embedding,
}

/// torch.bucketize with right=True
/// Returns a 1d tensor of shape (xs.len(),) on the CPU
fn bucketize_right(xs: &[f32], boundaries: &[f32], device: &Device) -> Result<Tensor> {
    // Initialize a vector to store the bucket indices
    let mut indices = vec![0; xs.len()];

    // Iterate over each element in `xs`
    for (i, &x) in xs.iter().enumerate() {
        // Find the index of the bucket for the current element
        let mut index = 0;
        for (j, &boundary) in boundaries.iter().enumerate() {
            if x < boundary {
                index = j;
                break;
            }
        }
        // If the value is greater than or equal to all boundaries, set the index to the length of boundaries
        if index == 0 && x >= boundaries[boundaries.len() - 1] {
            index = boundaries.len();
        }
        indices[i] = index as u32;
    }

    Tensor::from_vec(indices, (xs.len(),), device)
}

impl VisionEmbeddings {
    fn new(config: &SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
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
            )?,
        })
    }

    fn forward(&self, pixel_values: &Tensor, patch_attention_mask: &Tensor, tgt_sizes: Option<&Tensor>) -> Result<Tensor> {
        let (bs, _, max_im_h, max_im_w) = pixel_values.dims4()?;

        let patch_embeds = self.patch_embedding.forward(pixel_values)?;

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
                .flatten_all()?;

            let position_ids_b = position_ids.i(b_idx)?;
            new_position_ids.push(
                p_attn_mask
                    .flatten_all()?
                    .where_cond(&pos_ids, &position_ids_b)?,
            );
        }
        let position_ids = Tensor::stack(&new_position_ids, 0)?;
        let position_ids = position_ids.to_device(self.position_embedding.embeddings().device())?;
        embeddings.broadcast_add(&self.position_embedding.forward(&position_ids)?)
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
    fn new(config: SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
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
}

struct VisionMLP {
    activation: Activation,
    fc1: QLinear,
    fc2: QLinear,
}

impl VisionMLP {
    fn new(config: SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
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
}

struct EncoderLayer {
    mlp: VisionMLP,
    attn: Attention,
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
}

impl EncoderLayer {
    fn new(config: SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
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
    fn new(config: &SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
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
    config: SiglipVisionConfig,
}

impl VisionTransformer {
    fn new(config: &SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
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

    fn forward(&self, pixel_values: &Tensor, attention_mask: Option<&Tensor>, tgt_sizes: Option<&Tensor>) -> Result<Tensor> {
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
            .forward(pixel_values, &patch_attention_mask, tgt_sizes)?;

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
}