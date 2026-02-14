#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Module};
use mistralrs_quant::{Convolution, QuantMethod, ShardedVarBuilder};
use std::{collections::HashMap, ops::Mul, sync::Arc};

use crate::{
    attention::SdpaParams,
    layers::{conv2d, embedding, layer_norm, Activation, CausalMasker, Sdpa},
    pipeline::text_models_inputs_processor::FlashParams,
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
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

impl Default for SiglipVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 16,
            hidden_act: Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }
}

pub(super) struct VisionEmbeddings {
    patch_size: usize,
    patch_embedding: Conv2d,
    num_patches_per_side: usize,
    pub(super) position_embedding: Embedding,
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
            Ok(i) => i + 1, // right=True: insert after equal elements
            Err(i) => i,
        };

        result.push(idx as u32);
    }

    Tensor::from_vec(result, (xs.len(),), device)
}

impl VisionEmbeddings {
    fn new(config: &SiglipVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
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

    fn forward(
        &self,
        pixel_values: &Tensor,
        patch_attention_mask: &Tensor,
        tgt_sizes: Option<&Tensor>,
    ) -> Result<Tensor> {
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
            let (nb_patches_h, nb_patches_w) = if let Some(tgt_sizes) = tgt_sizes {
                (tgt_sizes.i((b_idx, 0))?, tgt_sizes.i((b_idx, 1))?)
            } else {
                let nb_patches_h = p_attn_mask.i((.., 0))?.sum_all()?;
                let nb_patches_w = p_attn_mask.i((0,))?.sum_all()?;
                (nb_patches_h, nb_patches_w)
            };

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

        let pos_emb = self.position_embedding.forward(&position_ids)?;
        let combined = embeddings.broadcast_add(&pos_emb)?;
        Ok(combined)
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
    fn new(config: SiglipVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_proj = mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("q_proj"))?;
        let k_proj = mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("k_proj"))?;
        let v_proj = mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("v_proj"))?;
        let o_proj = mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("out_proj"))?;

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let mut v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Vision encoders use bidirectional (non-causal) self-attention.
        // Use flash attention with causal=false for both correctness and numerical
        // compatibility with HuggingFace's PyTorch implementation.
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: self.scale,
            sinks: None,
        };

        // Build FlashParams with causal=false for bidirectional vision attention.
        // Empty cumulative_seqlens: flash backend uses the non-varlen path with causal=false.
        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: false,
        };

        let attn_weights = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attention_mask,
            Some(&flash_params),
            &sdpa_params,
        )?;

        self.o_proj.forward(&attn_weights.transpose(1, 2)?.reshape((
            b_sz,
            q_len,
            self.embed_dim,
        ))?)
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
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
}

impl VisionMLP {
    fn new(config: SiglipVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let fc1 = mistralrs_quant::linear(
            config.hidden_size,
            config.intermediate_size,
            &None,
            vb.pp("fc1"),
        )?;
        let fc2 = mistralrs_quant::linear(
            config.intermediate_size,
            config.hidden_size,
            &None,
            vb.pp("fc2"),
        )?;
        Ok(Self {
            activation: config.hidden_act,
            fc1,
            fc2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.fc1.forward(x)?;
        x = self.activation.forward(&x)?;
        self.fc2.forward(&x)
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
    fn new(config: SiglipVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
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
    fn new(config: &SiglipVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let vb_l = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(EncoderLayer::new(config.clone(), vb_l.pp(i))?);
        }
        Ok(Self { layers })
    }

    fn forward_get_hidden_states(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        hidden_states_index: isize,
    ) -> Result<Tensor> {
        let mut hidden_states = xs.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
            if (self.layers.len() as isize + hidden_states_index) as usize == layer_idx {
                return Ok(hidden_states);
            }
        }
        Ok(hidden_states)
    }
}

pub struct SiglipVisionTransformer {
    pub(super) embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
    config: SiglipVisionConfig,
}

impl SiglipVisionTransformer {
    pub fn new(config: &SiglipVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
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

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        attention_mask: Option<&Tensor>,
        tgt_sizes: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_get_hidden_states(pixel_values, attention_mask, tgt_sizes, -1)
    }

    pub fn forward_get_hidden_states(
        &self,
        pixel_values: &Tensor,
        attention_mask: Option<&Tensor>,
        tgt_sizes: Option<&Tensor>,
        hidden_states_index: isize,
    ) -> Result<Tensor> {
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

        let hidden_states =
            self.embeddings
                .forward(pixel_values, &patch_attention_mask, tgt_sizes)?;

        let attention_mask = if attention_mask.is_none() {
            None
        } else {
            let mask = patch_attention_mask
                .reshape((patch_attention_mask.dim(0)?, ()))?
                .to_dtype(hidden_states.dtype())?;
            Some(CausalMasker.expand_mask(&mask, hidden_states.dtype(), None)?)
        };
        let hidden_states = self.encoder.forward_get_hidden_states(
            &hidden_states,
            attention_mask.as_ref(),
            hidden_states_index + 1,
        )?;
        hidden_states.apply(&self.post_layernorm)
    }

    pub fn dtype(&self) -> DType {
        self.embeddings.patch_embedding.weight().dtype()
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
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
