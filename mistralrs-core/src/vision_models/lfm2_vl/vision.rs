use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{dense_embedding, layer_norm, linear, Activation, CausalMasker, Sdpa},
    pipeline::text_models_inputs_processor::FlashParams,
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::VisionConfig;

struct VisionEmbeddings {
    patch_embedding: Linear,
    position_embedding: Embedding,
    position_embedding_size: usize,
    hidden_size: usize,
}

struct ResizeWeights {
    indices: Vec<usize>,
    weights: Vec<f32>,
}

impl VisionEmbeddings {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let in_features = config.num_channels * config.patch_size * config.patch_size;
        let position_embedding_size = (config.num_patches as f64).sqrt() as usize;
        Ok(Self {
            patch_embedding: linear(in_features, config.hidden_size, vb.pp("patch_embedding"))?,
            position_embedding: dense_embedding(
                config.num_patches,
                config.hidden_size,
                vb.pp("position_embedding"),
                &None,
            )?,
            position_embedding_size,
            hidden_size: config.hidden_size,
        })
    }

    fn resized_positional_embeddings(
        &self,
        spatial_shapes: &Tensor,
        max_len: usize,
    ) -> Result<Tensor> {
        let shapes = spatial_shapes
            .to_dtype(DType::U32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<u32>()?;
        let positional_embeddings = self
            .position_embedding
            .embeddings()
            .reshape((
                self.position_embedding_size,
                self.position_embedding_size,
                self.hidden_size,
            ))?
            .permute((2, 0, 1))?
            .unsqueeze(0)?;

        let mut resized = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let height = shape[0] as usize;
            let width = shape[1] as usize;
            if height == 0 || width == 0 || height * width > max_len {
                candle_core::bail!(
                    "LFM2-VL spatial shape ({height}, {width}) is incompatible with max patches {max_len}"
                );
            }
            let embedding = self
                .resize_positional_embedding(&positional_embeddings, height, width)?
                .squeeze(0)?
                .reshape((self.hidden_size, height * width))?
                .transpose(0, 1)?;
            let embedding = if height * width < max_len {
                let pad = embedding
                    .narrow(0, 0, 1)?
                    .repeat((max_len - height * width, 1))?;
                Tensor::cat(&[embedding, pad], 0)?
            } else {
                embedding
            };
            resized.push(embedding);
        }
        Tensor::stack(&resized, 0)
    }

    fn resize_weights(input_size: usize, output_size: usize) -> Vec<ResizeWeights> {
        let scale = input_size as f64 / output_size as f64;
        let support = scale.max(1.0);
        (0..output_size)
            .map(|output_idx| {
                let center = scale * (output_idx as f64 + 0.5);
                let start = ((center - support + 0.5) as isize).max(0) as usize;
                let end = ((center + support + 0.5) as usize).min(input_size);
                let inv_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
                let mut indices = Vec::with_capacity(end.saturating_sub(start));
                let mut weights = Vec::with_capacity(end.saturating_sub(start));
                let mut total = 0.0f64;
                for input_idx in start..end {
                    let x = (input_idx as f64 - center + 0.5) * inv_scale;
                    let weight = (1.0 - x.abs()).max(0.0);
                    indices.push(input_idx);
                    weights.push(weight as f32);
                    total += weight;
                }
                if total != 0.0 {
                    for weight in &mut weights {
                        *weight /= total as f32;
                    }
                }
                ResizeWeights { indices, weights }
            })
            .collect()
    }

    fn resize_positional_embedding(
        &self,
        positional_embeddings: &Tensor,
        target_h: usize,
        target_w: usize,
    ) -> Result<Tensor> {
        let dtype = positional_embeddings.dtype();
        let device = positional_embeddings.device().clone();
        let (_, channels, input_h, input_w) = positional_embeddings.dims4()?;
        let values = positional_embeddings
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let height_weights = Self::resize_weights(input_h, target_h);
        let width_weights = Self::resize_weights(input_w, target_w);
        let mut output = vec![0f32; channels * target_h * target_w];

        for channel in 0..channels {
            for (out_y, y_weights) in height_weights.iter().enumerate() {
                for (out_x, x_weights) in width_weights.iter().enumerate() {
                    let mut value = 0.0f32;
                    for (&input_y, &weight_y) in y_weights.indices.iter().zip(&y_weights.weights) {
                        for (&input_x, &weight_x) in
                            x_weights.indices.iter().zip(&x_weights.weights)
                        {
                            let input_idx = (channel * input_h + input_y) * input_w + input_x;
                            value += values[input_idx] * weight_y * weight_x;
                        }
                    }
                    output[(channel * target_h + out_y) * target_w + out_x] = value;
                }
            }
        }

        Tensor::from_vec(output, (1, channels, target_h, target_w), &device)?.to_dtype(dtype)
    }

    fn forward(&self, pixel_values: &Tensor, spatial_shapes: &Tensor) -> Result<Tensor> {
        let patch_embeds = self
            .patch_embedding
            .forward(&pixel_values.to_dtype(self.patch_embedding.weight().dtype())?)?;
        let pos_embeds = self
            .resized_positional_embeddings(spatial_shapes, pixel_values.dim(1)?)?
            .to_device(patch_embeds.device())?;
        patch_embeds.broadcast_add(&pos_embeds)
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
    scale: f32,
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
}

impl Attention {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            q_proj: mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("q_proj"))?,
            k_proj: mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("k_proj"))?,
            v_proj: mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("v_proj"))?,
            o_proj: mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("out_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &AttentionMask) -> Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let y = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attention_mask,
            Some(&FlashParams::empty(false)),
            &SdpaParams {
                n_kv_groups: 1,
                sliding_window: None,
                softcap: None,
                softmax_scale: self.scale,
                sinks: None,
            },
        )?;
        self.o_proj.forward(
            &y.transpose(1, 2)?
                .reshape((batch, seq_len, self.embed_dim))?,
        )
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

struct Mlp {
    activation: Activation,
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            activation: config.hidden_act,
            fc1: mistralrs_quant::linear(
                config.hidden_size,
                config.intermediate_size,
                &None,
                vb.pp("fc1"),
            )?,
            fc2: mistralrs_quant::linear(
                config.intermediate_size,
                config.hidden_size,
                &None,
                vb.pp("fc2"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2
            .forward(&self.activation.forward(&self.fc1.forward(xs)?)?)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("fc1").add(&self.fc1);
        uvb.pp("fc2").add(&self.fc2);
        uvb.to_safetensors()
    }
}

struct EncoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            mlp: Mlp::new(config, vb.pp("mlp"))?,
            layer_norm1: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("layer_norm1"),
            )?,
            layer_norm2: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("layer_norm2"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &AttentionMask) -> Result<Tensor> {
        let residual = xs;
        let xs = self
            .self_attn
            .forward(&self.layer_norm1.forward(xs)?, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.mlp.forward(&self.layer_norm2.forward(&xs)?)?;
        xs + residual
    }
}

struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(EncoderLayer::new(config, vb_l.pp(layer_idx))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &AttentionMask) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, attention_mask)?;
        }
        Ok(xs)
    }
}

pub struct VisionModel {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
}

impl VisionModel {
    pub fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: VisionEmbeddings::new(config, vb.pp("embeddings"))?,
            encoder: Encoder::new(config, vb.pp("encoder"))?,
            post_layernorm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("post_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_attention_mask: &Tensor,
        spatial_shapes: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(pixel_values, spatial_shapes)?;
        let mask = CausalMasker.expand_mask(
            &pixel_attention_mask.to_dtype(hidden_states.dtype())?,
            hidden_states.dtype(),
            None,
        )?;
        let hidden_states = self
            .encoder
            .forward(&hidden_states, &AttentionMask::Custom(mask))?;
        self.post_layernorm.forward(&hidden_states)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("embeddings")
            .extend(self.embeddings.residual_tensors());
        let uvb_enc = uvb.pp("encoder");
        for (idx, layer) in self.encoder.layers.iter().enumerate() {
            let uvb_l = uvb_enc.pp("layers").pp(idx);
            uvb_l.pp("layer_norm1").add(&layer.layer_norm1);
            uvb_l.pp("layer_norm2").add(&layer.layer_norm2);
            uvb_l
                .pp("self_attn")
                .extend(layer.self_attn.residual_tensors());
            uvb_l.pp("mlp").extend(layer.mlp.residual_tensors());
        }
        uvb.pp("post_layernorm").add(&self.post_layernorm);
        uvb.to_safetensors()
    }
}
