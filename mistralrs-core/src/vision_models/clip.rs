#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

// Sourced from https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/clip/vision_model.rs
use candle_core::{IndexOp, Result, Shape, Tensor, D};
use candle_nn::{Conv2dConfig, Module};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    layers::{self, MatMul},
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
};

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub enum Activation {
    QuickGelu,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::QuickGelu => xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?),
        }
    }
}

serde_default_fn!(usize, d_hidden_size, 768);
serde_default_fn!(usize, d_intermediate_size, 3072);
serde_default_fn!(usize, d_num_hidden_layers, 12);
serde_default_fn!(usize, d_num_attention_heads, 12);
serde_default_fn!(usize, d_num_channels, 3);
serde_default_fn!(usize, d_image_size, 224);
serde_default_fn!(usize, d_patch_size, 32);
serde_default_fn!(Activation, d_act, Activation::QuickGelu);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClipConfig {
    #[serde(default = "d_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "d_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "d_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_num_channels")]
    pub num_channels: usize,
    #[serde(default = "d_image_size")]
    pub image_size: usize,
    #[serde(default = "d_patch_size")]
    pub patch_size: usize,
    #[serde(default = "d_act")]
    pub hidden_act: Activation,
}

// https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L112
#[derive(Clone, Debug)]
struct ClipVisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    position_ids: Tensor,
    class_embedding: Tensor,
    position_embedding: candle_nn::Embedding,
}

impl ClipVisionEmbeddings {
    fn new(vs: ShardedVarBuilder, c: &ClipConfig) -> Result<Self> {
        // originally nn.Parameter
        let class_embedding = if vs.contains_tensor("class_embedding") {
            vs.get(c.hidden_size, "class_embedding")?
        } else {
            Tensor::randn(0f32, 1f32, c.hidden_size, vs.device())?
        };

        let num_patches = (c.image_size / c.patch_size).pow(2);
        let num_positions = num_patches + 1;
        let position_ids = Tensor::arange(0, num_positions as i64, vs.device())?;

        let conv2dconfig = Conv2dConfig {
            stride: c.patch_size,
            ..Default::default()
        };
        let position_embedding = layers::embedding(
            num_positions,
            c.hidden_size,
            vs.pp("position_embedding"),
            &None,
        )?;
        let patch_embedding = layers::conv2d_no_bias(
            c.num_channels,
            c.hidden_size,
            c.patch_size,
            conv2dconfig,
            vs.pp("patch_embedding"),
        )?;
        Ok(Self {
            patch_embedding,
            position_ids,
            class_embedding,
            position_embedding,
        })
    }
}

impl Module for ClipVisionEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.shape().dims();
        let patch_embeds = self
            .patch_embedding
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let shape = Shape::from((batch_size[0], 1, self.class_embedding.dim(D::Minus1)?));
        let class_embeds = self.class_embedding.expand(shape)?;
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        embeddings.broadcast_add(&position_embedding)
    }
}

#[derive(Clone, Debug)]
struct ClipAttention {
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    q_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl ClipAttention {
    fn new(vs: ShardedVarBuilder, c: &ClipConfig) -> Result<Self> {
        let hidden_size = c.hidden_size;
        let num_attention_heads = c.num_attention_heads;
        let k_proj = mistralrs_quant::linear(hidden_size, hidden_size, &None, vs.pp("k_proj"))?;
        let v_proj = mistralrs_quant::linear(hidden_size, hidden_size, &None, vs.pp("v_proj"))?;
        let q_proj = mistralrs_quant::linear(hidden_size, hidden_size, &None, vs.pp("q_proj"))?;
        let out_proj = mistralrs_quant::linear(hidden_size, hidden_size, &None, vs.pp("out_proj"))?;
        let head_dim = hidden_size / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(ClipAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (bsz, seq_len, hidden_size) = xs.dims3()?;

        let query_states = (self.q_proj.forward(xs)? * self.scale)?;
        let proj_shape = (bsz * self.num_attention_heads, seq_len, self.head_dim);
        let query_states = self
            .shape(&query_states, seq_len, bsz)?
            .reshape(proj_shape)?;
        let key_states = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?;
        let value_states = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?;
        let attn_weights = MatMul.matmul(&query_states, &key_states.transpose(1, 2)?)?;

        let src_len = key_states.dim(1)?;

        let attn_weights = if let Some(causal_attention_mask) = causal_attention_mask {
            attn_weights
                .reshape((bsz, self.num_attention_heads, seq_len, src_len))?
                .broadcast_add(causal_attention_mask)?
                .reshape((bsz * self.num_attention_heads, seq_len, src_len))?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let attn_output = MatMul.matmul(&attn_weights, &value_states)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_attention_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, hidden_size))?;
        self.out_proj.forward(&attn_output)
    }
}

#[derive(Clone, Debug)]
struct ClipMlp {
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
    activation: Activation,
}

impl ClipMlp {
    fn new(vs: ShardedVarBuilder, c: &ClipConfig) -> Result<Self> {
        let fc1 = mistralrs_quant::linear(c.hidden_size, c.intermediate_size, &None, vs.pp("fc1"))?;
        let fc2 = mistralrs_quant::linear(c.intermediate_size, c.hidden_size, &None, vs.pp("fc2"))?;

        Ok(ClipMlp {
            fc1,
            fc2,
            activation: c.hidden_act,
        })
    }
}

impl ClipMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        self.fc2.forward(&self.activation.forward(&xs)?)
    }
}

#[derive(Clone, Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: candle_nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: candle_nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: ShardedVarBuilder, c: &ClipConfig) -> Result<Self> {
        let self_attn = ClipAttention::new(vs.pp("self_attn"), c)?;
        let layer_norm1 = layers::layer_norm(c.hidden_size, 1e-5, vs.pp("layer_norm1"))?;
        let mlp = ClipMlp::new(vs.pp("mlp"), c)?;
        let layer_norm2 = layers::layer_norm(c.hidden_size, 1e-5, vs.pp("layer_norm2"))?;

        Ok(ClipEncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, causal_attention_mask)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

#[derive(Clone, Debug)]
pub struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    pub fn new(vs: ShardedVarBuilder, c: &ClipConfig) -> Result<Self> {
        let vs = vs.pp("layers");
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();
        for index in 0..c.num_hidden_layers {
            let layer = ClipEncoderLayer::new(vs.pp(index.to_string()), c)?;
            layers.push(layer)
        }
        Ok(ClipEncoder { layers })
    }

    pub fn forward_get_hidden_states(
        &self,
        xs: &Tensor,
        causal_attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let mut xs = xs.clone();
        let mut hidden_states = Vec::new();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
            hidden_states.push(xs.clone());
        }
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L743
#[derive(Clone, Debug)]
pub struct ClipVisionTransformer {
    embeddings: ClipVisionEmbeddings,
    encoder: ClipEncoder,
    pre_layer_norm: candle_nn::LayerNorm,
    final_layer_norm: candle_nn::LayerNorm,
}

impl ClipVisionTransformer {
    /// Create a CLIP vision transformer model. Expects the vb to point to the root (not model)
    /// where (for example) `.pp("embeddings")` is valid.
    pub fn new(vb: ShardedVarBuilder, c: &ClipConfig) -> Result<Self> {
        let embeddings = ClipVisionEmbeddings::new(vb.pp("embeddings"), c)?;
        let pre_layer_norm = layers::layer_norm(c.hidden_size, 1e-5, vb.pp("pre_layrnorm"))?;
        let encoder = ClipEncoder::new(vb.pp("encoder"), c)?;
        let final_layer_norm = layers::layer_norm(c.hidden_size, 1e-5, vb.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            pre_layer_norm,
        })
    }

    pub fn forward_get_hidden_states(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let hidden_states = pixel_values
            .apply(&self.embeddings)?
            .apply(&self.pre_layer_norm)?;
        let mut result = self
            .encoder
            .forward_get_hidden_states(&hidden_states, None)?;
        let encoder_outputs = result.last().unwrap();
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        result.push(self.final_layer_norm.forward(&pooled_output)?.clone());
        Ok(result)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("pre_layrnorm").add(&self.pre_layer_norm);
        uvb.pp("post_layernorm").add(&self.final_layer_norm);

        // vision embeddings
        {
            let uvb_emb = uvb.pp("embeddings");

            uvb_emb.add_tensor("class_embedding", self.embeddings.class_embedding.clone());
            uvb_emb
                .pp("position_embedding")
                .add(&self.embeddings.position_embedding);
            uvb_emb
                .pp("patch_embedding")
                .add(&self.embeddings.patch_embedding);
        }

        // encoder
        {
            let uvb_enc = uvb.pp("encoder");

            for (i, layer) in self.encoder.layers.iter().enumerate() {
                let uvb_l = uvb_enc.pp("layers").pp(i);

                uvb_l.pp("layer_norm1").add(&layer.layer_norm1);
                uvb_l.pp("layer_norm2").add(&layer.layer_norm2);

                let uvb_mlp = uvb_l.pp("mlp");
                uvb_mlp.pp("fc1").add(&layer.mlp.fc1);
                uvb_mlp.pp("fc2").add(&layer.mlp.fc2);

                let uvb_attn = uvb_l.pp("self_attn");
                uvb_attn.pp("q_proj").add(&layer.self_attn.q_proj);
                uvb_attn.pp("k_proj").add(&layer.self_attn.k_proj);
                uvb_attn.pp("v_proj").add(&layer.self_attn.v_proj);
                uvb_attn.pp("out_proj").add(&layer.self_attn.out_proj);
            }
        }

        uvb.to_safetensors()
    }
}
