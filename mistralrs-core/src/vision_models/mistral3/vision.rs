use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use mistralrs_quant::{linear_b, Convolution, QuantMethod, ShardedVarBuilder};

use crate::{
    layers::{self, GetFloatInfo, RmsNorm},
    pipeline::NormalLoadingMetadata,
    utils::unvarbuilder::UnVarBuilder,
};

fn default_act() -> candle_nn::Activation {
    candle_nn::Activation::Silu
}

fn default_hidden_size() -> usize {
    1024
}

fn default_intermediate_size() -> usize {
    4096
}

fn default_num_channels() -> usize {
    3
}

fn default_num_hidden_layers() -> usize {
    24
}

fn default_num_attention_heads() -> usize {
    16
}

fn default_rope_theta() -> f64 {
    10000.0
}

/// RoPE parameters for Mistral3 vision model
#[derive(serde::Deserialize, Debug, Clone)]
pub struct Mistral3VisionRopeParameters {
    pub rope_theta: f64,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Mistral3VisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    // Support both flat rope_theta and nested rope_parameters
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_parameters: Option<Mistral3VisionRopeParameters>,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    pub head_dim: Option<usize>,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_act")]
    pub hidden_act: candle_nn::Activation,
}

impl Mistral3VisionConfig {
    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get rope_theta from either flat field or rope_parameters
    pub fn get_rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.rope_theta)
            .unwrap_or(self.rope_theta)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    scale: f64,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Mistral3VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();
        let q_proj = linear_b(h, h, false, &None, vb.pp("q_proj"))?;
        let k_proj = linear_b(h, h, false, &None, vb.pp("k_proj"))?;
        let v_proj = linear_b(h, h, false, &None, vb.pp("v_proj"))?;
        let o_proj = linear_b(h, h, false, &None, vb.pp("o_proj"))?;
        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            scale,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        emb: &RotaryEmbedding,
        subsampled_positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, patches, _) = xs.dims3()?;
        let query_states = self.q_proj.forward_autocast(xs)?;
        let key_states = self.k_proj.forward_autocast(xs)?;
        let value_states = self.v_proj.forward_autocast(xs)?;

        let shape = (b, patches, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let (query_states, key_states) =
            emb.apply_rotary_emb_qkv(&query_states, &key_states, subsampled_positions)?;
        let attn_weights = (query_states.matmul(&key_states.t()?)? * self.scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        self.o_proj.forward_autocast(
            &attn_weights
                .matmul(&value_states)?
                .transpose(1, 2)?
                .reshape((b, patches, ()))?,
        )
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Mistral3VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate_proj = linear_b(h, i, false, &None, vb.pp("gate_proj"))?;
        let up_proj = linear_b(h, i, false, &None, vb.pp("up_proj"))?;
        let down_proj = linear_b(i, h, false, &None, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down_proj.forward_autocast(
            &(self.gate_proj.forward_autocast(xs)?.apply(&self.act_fn)?
                * self.up_proj.forward_autocast(xs)?)?,
        )
    }
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    attention_norm: RmsNorm,
    feed_forward: Mlp,
    attention: Attention,
    ffn_norm: RmsNorm,
}

impl AttentionLayer {
    fn new(
        cfg: &Mistral3VisionConfig,
        vb: ShardedVarBuilder,
        normal_loading_metadata: &NormalLoadingMetadata,
    ) -> Result<Self> {
        let attention_norm = RmsNorm::new(
            cfg.hidden_size,
            1e-5,
            vb.pp("attention_norm")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let feed_forward = Mlp::new(cfg, vb.pp("feed_forward"))?;
        let attention = Attention::new(cfg, vb.pp("attention"))?;
        let ffn_norm = RmsNorm::new(
            cfg.hidden_size,
            1e-5,
            vb.pp("ffn_norm")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        Ok(Self {
            attention_norm,
            feed_forward,
            attention,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        emb: &RotaryEmbedding,
        subsampled_positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention.forward(
            &xs.apply(&self.attention_norm)?,
            emb,
            subsampled_positions,
            attention_mask,
        )?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.ffn_norm)?.apply(&self.feed_forward)?;
        xs + residual
    }
}

#[derive(Debug, Clone)]
struct Transformer {
    layers: Vec<AttentionLayer>,
}

impl Transformer {
    fn new(
        cfg: &Mistral3VisionConfig,
        vb: ShardedVarBuilder,
        normal_loading_metadata: &NormalLoadingMetadata,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = AttentionLayer::new(cfg, vb.pp(layer_idx), normal_loading_metadata)?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        xs: &Tensor,
        emb: &RotaryEmbedding,
        subsampled_positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, emb, subsampled_positions, attention_mask)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Mistral3VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device();
        let dim = cfg.head_dim();
        let rope_theta = cfg.get_rope_theta() as f32;
        let max_patches_per_side = cfg.image_size / cfg.patch_size;
        let freqs: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let freqs_h = freqs.iter().step_by(2).copied().collect::<Vec<_>>();
        let freqs_h = Tensor::new(freqs_h, dev)?;
        let freqs_w = freqs.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();
        let freqs_w = Tensor::new(freqs_w, dev)?;
        let h = Tensor::arange(0u32, max_patches_per_side as u32, dev)?.to_dtype(DType::F32)?;
        let w = Tensor::arange(0u32, max_patches_per_side as u32, dev)?.to_dtype(DType::F32)?;
        let freqs_h = h.unsqueeze(1)?.matmul(&freqs_h.unsqueeze(0)?)?;
        let freqs_w = w.unsqueeze(1)?.matmul(&freqs_w.unsqueeze(0)?)?;
        let inv_freq = Tensor::cat(
            &[
                freqs_h.unsqueeze(1)?.repeat((1, max_patches_per_side, 1))?,
                freqs_w.unsqueeze(0)?.repeat((max_patches_per_side, 1, 1))?,
            ],
            D::Minus1,
        )?
        .reshape(((), dim / 2))?;
        let cos = inv_freq.cos()?.to_dtype(dtype)?;
        let sin = inv_freq.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        subsampled_positions: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, _seq_len, _n_embd) = q.dims4()?;
        let (cos, sin) = match subsampled_positions {
            None => (&self.cos, &self.sin),
            Some(pos) => (
                &self.cos.index_select(pos, 0)?,
                &self.sin.index_select(pos, 0)?,
            ),
        };
        let q_embed = candle_nn::rotary_emb::rope(q, cos, sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, cos, sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub struct Mistral3VisionModel {
    patch_conv: candle_nn::Conv2d,
    ln_pre: RmsNorm,
    transformer: Transformer,
    patch_positional_embedding: RotaryEmbedding,
    max_image_width: u32,
    patch_size: usize,
    dtype: DType,
}

impl Mistral3VisionModel {
    pub fn new(
        cfg: &Mistral3VisionConfig,
        vb: ShardedVarBuilder,
        normal_loading_metadata: &NormalLoadingMetadata,
    ) -> Result<Self> {
        let conv2d_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_conv = layers::conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv2d_cfg,
            vb.pp("patch_conv")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let ln_pre = RmsNorm::new(
            cfg.hidden_size,
            1e-5,
            vb.pp("ln_pre")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let transformer = Transformer::new(cfg, vb.pp("transformer"), normal_loading_metadata)?;
        let patch_positional_embedding = RotaryEmbedding::new(
            cfg,
            vb.pp("patch_positional_embedding")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let max_image_width = (cfg.image_size / cfg.patch_size) as u32;
        Ok(Self {
            patch_conv,
            ln_pre,
            transformer,
            patch_positional_embedding,
            max_image_width,
            patch_size: cfg.patch_size,
            dtype: vb.dtype(),
        })
    }

    fn position_ids_in_meshgrid(
        &self,
        patch_embeds_list: &Vec<Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let mut positions = Vec::new();
        for patch in patch_embeds_list {
            let (height, width) = (patch.dim(D::Minus2)?, patch.dim(D::Minus1)?);
            let idx = Tensor::arange(0, height as u32, device)?;
            let idy = Tensor::arange(0, width as u32, device)?;
            let mesh = Tensor::meshgrid(&[idx, idy], false)?;
            let ids = (&mesh[0] * (self.max_image_width as f64) + &mesh[1])?.flatten_all()?;
            positions.push(ids);
        }
        Tensor::cat(&positions, 0)
    }

    fn generate_block_attention_mask(
        &self,
        patch_embeds_list: Vec<usize>,
        patch_embeds: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = patch_embeds.dim(1)?;
        let mut causal_mask = (Tensor::ones(
            (seq_len, seq_len),
            patch_embeds.dtype(),
            patch_embeds.device(),
        )? * patch_embeds.dtype().finfo()?.min)?;

        let block_end_idx: Vec<usize> = patch_embeds_list.iter().fold(Vec::new(), |mut acc, &x| {
            let new_sum = x + acc.last().copied().unwrap_or(0);
            acc.push(new_sum);
            acc
        });
        let block_start_idx: Vec<usize> = {
            let mut extended = vec![0];
            extended.extend_from_slice(&patch_embeds_list[..patch_embeds_list.len() - 1]);
            extended.into_iter().fold(Vec::new(), |mut acc, x| {
                let new_sum = x + acc.last().copied().unwrap_or(0);
                acc.push(new_sum);
                acc
            })
        };
        for (start, end) in block_start_idx.into_iter().zip(block_end_idx) {
            causal_mask = causal_mask.slice_assign(
                &[start..end, start..end],
                &Tensor::zeros(
                    (end - start, end - start),
                    causal_mask.dtype(),
                    causal_mask.device(),
                )?,
            )?;
        }

        causal_mask
            .reshape((1, 1, causal_mask.dim(0)?, causal_mask.dim(1)?))?
            .repeat((patch_embeds.dim(0)?, 1, 1, 1))
    }

    pub fn forward(&self, xs: &Tensor, image_sizes: Vec<(u32, u32)>) -> Result<Tensor> {
        let patch_embeds = Convolution.forward_2d(&self.patch_conv, xs)?;
        let patch_embeds_list = image_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                patch_embeds
                    .i(i)?
                    .narrow(D::Minus2, 0, size.0 as usize / self.patch_size)?
                    .narrow(D::Minus1, 0, size.1 as usize / self.patch_size)
            })
            .collect::<Result<Vec<Tensor>>>()?;
        let patch_embeds = Tensor::cat(
            &patch_embeds_list
                .iter()
                .map(|p| p.flatten_from(1)?.t())
                .collect::<Result<Vec<Tensor>>>()?,
            0,
        )?
        .unsqueeze(0)?;
        let patch_embeds = patch_embeds.apply(&self.ln_pre)?;

        let subsampled_positions =
            Some(self.position_ids_in_meshgrid(&patch_embeds_list, patch_embeds.device())?);

        let attention_mask = self.generate_block_attention_mask(
            patch_embeds_list
                .iter()
                .map(|p| Ok(p.dim(D::Minus2)? * p.dim(D::Minus1)?))
                .collect::<Result<Vec<usize>>>()?,
            &patch_embeds,
        )?;

        self.transformer.forward(
            &patch_embeds,
            &self.patch_positional_embedding,
            subsampled_positions.as_ref(),
            Some(&attention_mask),
        )
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn get_layers(&mut self) -> Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)> {
        let mut tensors = Vec::new();
        for layer in &mut self.transformer.layers {
            tensors.push((&mut layer.attention.q_proj, None));
            tensors.push((&mut layer.attention.k_proj, None));
            tensors.push((&mut layer.attention.v_proj, None));
            tensors.push((&mut layer.attention.o_proj, None));

            tensors.push((&mut layer.feed_forward.gate_proj, None));
            tensors.push((&mut layer.feed_forward.up_proj, None));
            tensors.push((&mut layer.feed_forward.down_proj, None));
        }
        tensors
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("patch_conv").add(&self.patch_conv);
        uvb.pp("ln_pre").add(&self.ln_pre);

        {
            let uvb_pos = uvb.pp("patch_positional_embedding");
            uvb_pos.add_tensor("cos", self.patch_positional_embedding.cos.clone());
            uvb_pos.add_tensor("sin", self.patch_positional_embedding.sin.clone());
        }

        for (layer_idx, layer) in self.transformer.layers.iter().enumerate() {
            let uvb_l = uvb.pp("transformer").pp("layers").pp(layer_idx);
            uvb_l.pp("attention_norm").add(&layer.attention_norm);
            uvb_l.pp("ffn_norm").add(&layer.ffn_norm);
        }

        uvb.to_safetensors()
    }
}
