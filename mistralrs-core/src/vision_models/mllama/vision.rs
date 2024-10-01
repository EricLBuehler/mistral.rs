#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::ops::Mul;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{
    conv2d_no_bias, embedding, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, Embedding,
    LayerNorm, LayerNormConfig, Linear, Module, VarBuilder,
};

use crate::{
    attention::SdpaParams,
    layers::{FusedBiasLinear, Sdpa},
};

use super::{MLlamaVisionConfig, VisionActivation};

struct MLlamaPrecomputedPositionEmbedding {
    gate: Tensor,
    embedding: Tensor,
    tile_embedding: Embedding,
    num_patches: usize,
    hidden_size: usize,
    max_num_tiles: usize,
}

impl MLlamaPrecomputedPositionEmbedding {
    fn new(cfg: &MLlamaVisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
        Ok(Self {
            // NOTE: Preapply the tanh
            gate: vb.get((1,), "gate")?.tanh()?,
            embedding: vb.get((num_patches, cfg.hidden_size), "embedding")?,
            tile_embedding: embedding(
                cfg.max_aspect_ratio_id() + 1,
                cfg.max_num_tiles * num_patches * cfg.hidden_size,
                vb.pp("tile_embedding"),
            )?,
            num_patches,
            hidden_size: cfg.hidden_size,
            max_num_tiles: cfg.max_num_tiles,
        })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L197
    fn forward(&self, hidden_state: &Tensor, aspect_ratio_ids: &Tensor) -> Result<Tensor> {
        // position embeddings
        let mut gated_pos_embed = (1. - &self.gate)?.broadcast_mul(&self.embedding)?;
        let hidden_state = hidden_state.broadcast_add(&gated_pos_embed.reshape((
            1,
            1,
            self.num_patches,
            self.hidden_size,
        ))?)?;

        // precomputed tile position embeddings
        let mut tile_position_embedding = self.tile_embedding.forward(aspect_ratio_ids)?;
        let bs = hidden_state.dim(0)?;
        tile_position_embedding = tile_position_embedding.reshape((
            bs,
            self.max_num_tiles,
            self.num_patches,
            self.hidden_size,
        ))?;
        gated_pos_embed = self.gate.broadcast_mul(&tile_position_embedding)?;

        hidden_state.broadcast_add(&gated_pos_embed)
    }
}

struct MLlamaPrecomputedAspectRatioEmbedding {
    embedding: Embedding,
    gate: Option<Tensor>,
    max_num_tiles: usize,
    hidden_size: usize,
}

impl MLlamaPrecomputedAspectRatioEmbedding {
    fn new<const GATED: bool>(cfg: &MLlamaVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embedding: embedding(
                cfg.max_aspect_ratio_id() + 1,
                cfg.max_num_tiles * cfg.hidden_size,
                vb.pp("embedding"),
            )?,
            gate: if GATED {
                // NOTE: Preapply the tanh
                Some(vb.get((1,), "gate")?.tanh()?)
            } else {
                None
            },
            max_num_tiles: cfg.max_num_tiles,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, hidden_state: &Tensor, aspect_ratio_ids: &Tensor) -> Result<Tensor> {
        let mut embeddings = self.embedding.forward(aspect_ratio_ids)?;
        embeddings = embeddings.reshape(((), self.max_num_tiles, 1, self.hidden_size))?;

        if let Some(gate) = &self.gate {
            embeddings = embeddings.broadcast_mul(gate)?;
        }

        hidden_state.broadcast_add(&embeddings)
    }
}

struct MLlamaVisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    sdpa_params: SdpaParams,
    num_heads: usize,
    head_dim: usize,
}

impl MLlamaVisionAttention {
    fn new(cfg: &MLlamaVisionConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                vb.pp("o_proj"),
            )?,
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
                use_flash_attn: false,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
            },
            num_heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L243
    fn forward(&self, hidden_state: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut q = self.q_proj.forward(hidden_state)?;
        let mut k = self.k_proj.forward(hidden_state)?;
        let mut v = self.v_proj.forward(hidden_state)?;

        // Should be same, no caching...
        let (bs, q_sq, _) = q.dims3()?;
        let (_, k_sq, _) = k.dims3()?;

        q = q
            .reshape((bs, q_sq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((bs, k_sq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((bs, k_sq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let attn_output = Sdpa
            .run_attention(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                attention_mask,
                None,
                &self.sdpa_params,
            )?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs, q_sq, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

struct MLlamaMlp {
    act: VisionActivation,
    fc1: FusedBiasLinear,
    fc2: FusedBiasLinear,
}

impl MLlamaMlp {
    fn new(cfg: &MLlamaVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act: cfg.hidden_act,
            fc1: FusedBiasLinear::try_from(linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("fc1"),
            )?)?,
            fc2: FusedBiasLinear::try_from(linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("fc2"),
            )?)?,
        })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L223
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.fc2
            .forward(&self.act.forward(&self.fc1.forward(hidden_states)?)?)
    }
}

struct MLlamaVisionEncoderLayer {
    self_attn: MLlamaVisionAttention,
    mlp: MLlamaMlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    gate_attn: Option<Tensor>,
    gate_ffn: Option<Tensor>,
}

impl MLlamaVisionEncoderLayer {
    fn new<const GATED: bool>(cfg: &MLlamaVisionConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = MLlamaVisionAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = MLlamaMlp::new(cfg, vb.pp("mlp"))?;

        let input_layernorm = layer_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        if GATED {
            Ok(Self {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                // NOTE: Preapply the tanh
                gate_attn: Some(vb.get((1,), "gate_attn")?.tanh()?),
                // NOTE: Preapply the tanh
                gate_ffn: Some(vb.get((1,), "gate_ffn")?.tanh()?),
            })
        } else {
            Ok(Self {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                gate_attn: None,
                gate_ffn: None,
            })
        }
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L348
    fn forward(&self, hidden_state: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Self attn
        let residual = hidden_state;
        let mut hidden_state = self.input_layernorm.forward(hidden_state)?;

        hidden_state = self.self_attn.forward(&hidden_state, attention_mask)?;

        if let Some(gate) = &self.gate_attn {
            hidden_state = gate.broadcast_mul(&hidden_state)?;
        }
        hidden_state = (residual + hidden_state)?;

        // FF
        let residual = hidden_state.clone();
        hidden_state = self.post_attention_layernorm.forward(&hidden_state)?;

        hidden_state = self.mlp.forward(&hidden_state)?;

        if let Some(gate) = &self.gate_ffn {
            hidden_state = gate.broadcast_mul(&hidden_state)?;
        }
        residual + hidden_state
    }
}

struct MLlamaVisionEncoder {
    layers: Vec<MLlamaVisionEncoderLayer>,
}

impl MLlamaVisionEncoder {
    fn new<const GATED: bool>(
        cfg: &MLlamaVisionConfig,
        num_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..num_layers {
            layers.push(MLlamaVisionEncoderLayer::new::<GATED>(
                cfg,
                layers_vb.pp(i),
            )?);
        }
        Ok(Self { layers })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L394
    /// Also return hidden states, all hidden states for this later
    fn forward_with_states(
        &self,
        hidden_state: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut hidden_state = hidden_state.clone();
        let mut hidden_states = Vec::new();
        for layer in &self.layers {
            hidden_states.push(hidden_state.clone());
            hidden_state = layer.forward(&hidden_state, attention_mask)?;
        }
        hidden_states.push(hidden_state.clone());
        Ok((hidden_state, hidden_states))
    }
}

fn _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: &Tensor,
    num_patches: usize,
    target_length: usize,
    dtype: DType,
    num_attn_heads: usize,
) -> Result<Tensor> {
    let (bs, max_num_tiles) = aspect_ratio_mask.dims2()?;
    let mut attention_mask = aspect_ratio_mask
        .reshape((bs, max_num_tiles, 1, 1))?
        .repeat((1, 1, target_length, 1))?;

    // Mask padding patches
    let pad_patches = target_length - num_patches;
    let (bs, d1, d2, d3) = attention_mask.dims4()?;
    attention_mask = attention_mask.slice_assign(
        &[&.., &.., &(d2 - pad_patches..), &..],
        &Tensor::zeros(
            (bs, d1, pad_patches, d3),
            attention_mask.dtype(),
            attention_mask.device(),
        )?,
    )?;

    // Invert the mask
    attention_mask = (1. - attention_mask.to_dtype(DType::F32)?.to_dtype(dtype)?)?;

    // Reshape to 2d and create 4d attn mask
    // (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape((bs, max_num_tiles * target_length, 1))?;
    attention_mask =
        attention_mask.matmul(&attention_mask.transpose(D::Minus1, D::Minus2)?.mul(-1e15)?)?;
    attention_mask
        .unsqueeze(1)?
        .contiguous()?
        .repeat((1, num_attn_heads, 1, 1))
}

pub(super) struct MLlamaVisionModel {
    patch_embedding: Conv2d,
    class_embedding: Tensor,
    gated_positional_embedding: MLlamaPrecomputedPositionEmbedding,
    pre_tile_positional_embedding: MLlamaPrecomputedAspectRatioEmbedding,
    post_tile_positional_embedding: MLlamaPrecomputedAspectRatioEmbedding,
    layernorm_pre: LayerNorm,
    layernorm_post: LayerNorm,
    transformer: MLlamaVisionEncoder,
    global_transformer: MLlamaVisionEncoder,
    pub(super) num_patches: usize,
    intermediate_layers_indices: Tensor,
    num_attn_heads: usize,
}

impl MLlamaVisionModel {
    pub(super) fn new(cfg: &MLlamaVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedding = conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;

        let class_embedding = vb.get((cfg.hidden_size,), "class_embedding")?;
        let gated_positional_embedding =
            MLlamaPrecomputedPositionEmbedding::new(cfg, vb.pp("gated_positional_embedding"))?;

        let pre_tile_positional_embedding = MLlamaPrecomputedAspectRatioEmbedding::new::<true>(
            cfg,
            vb.pp("pre_tile_positional_embedding"),
        )?;
        let post_tile_positional_embedding = MLlamaPrecomputedAspectRatioEmbedding::new::<true>(
            cfg,
            vb.pp("post_tile_positional_embedding"),
        )?;

        // layer norms
        let layernorm_pre = layer_norm(
            cfg.hidden_size,
            LayerNormConfig::default(),
            vb.pp("layernorm_pre"),
        )?;
        let layernorm_post = layer_norm(
            cfg.hidden_size,
            LayerNormConfig::default(),
            vb.pp("layernorm_post"),
        )?;

        // encoders
        let transformer =
            MLlamaVisionEncoder::new::<false>(cfg, cfg.num_hidden_layers, vb.pp("transformer"))?;
        let global_transformer = MLlamaVisionEncoder::new::<true>(
            cfg,
            cfg.num_global_layers,
            vb.pp("global_transformer"),
        )?;

        Ok(Self {
            patch_embedding,
            class_embedding,
            gated_positional_embedding,
            pre_tile_positional_embedding,
            post_tile_positional_embedding,
            layernorm_post,
            layernorm_pre,
            transformer,
            global_transformer,
            num_patches: (cfg.image_size / cfg.patch_size).pow(2) + 1,
            intermediate_layers_indices: Tensor::new(
                cfg.intermediate_layers_indices
                    .iter()
                    .map(|i| *i as u32)
                    .collect::<Vec<_>>(),
                vb.device(),
            )?,
            num_attn_heads: cfg.num_attention_heads,
        })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L1425
    pub(super) fn forward(
        &self,
        pixel_values: &Tensor,
        aspect_ratio_ids: &Tensor,
        aspect_ratio_mask: &Tensor,
    ) -> Result<Tensor> {
        let pixel_values = pixel_values.to_dtype(self.class_embedding.dtype())?;

        let bs = pixel_values.dim(0)?;
        let num_concurrent_media = pixel_values.dim(1)?;
        let num_tiles = pixel_values.dim(2)?;
        let num_channels = pixel_values.dim(3)?;
        let height = pixel_values.dim(4)?;
        let width = pixel_values.dim(5)?;

        let pixel_values = pixel_values.reshape((
            bs * num_concurrent_media * num_tiles,
            num_channels,
            height,
            width,
        ))?;
        let aspect_ratio_ids = aspect_ratio_ids.reshape((bs * num_concurrent_media, ()))?;

        // Patch embedding
        let patch_embeds = self.patch_embedding.forward(&pixel_values)?;
        let mut hidden_state = patch_embeds.flatten_from(2)?.transpose(1, 2)?;

        // Tile embeddings
        let (_, mut num_patches, dim) = hidden_state.dims3()?;
        hidden_state = hidden_state.reshape((bs * num_concurrent_media, num_tiles, (), dim))?;
        hidden_state = self
            .pre_tile_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids)?;

        // Add cls token
        hidden_state =
            hidden_state.reshape((bs * num_concurrent_media * num_tiles, num_patches, dim))?;
        hidden_state = self.apply_class_embedding(&hidden_state)?;
        num_patches += 1;

        // Position embeddings
        hidden_state =
            hidden_state.reshape((bs * num_concurrent_media, num_tiles, num_patches, dim))?;
        hidden_state = self
            .gated_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids)?;

        hidden_state = self.layernorm_pre.forward(&hidden_state)?;

        // Compute the number of tokens to pad
        let num_padding_patches = (8 - (hidden_state.dim(D::Minus2)? as isize % 8)) % 8;
        // Compute padding tuple for pad function
        // (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        let _padding = (0usize, 0usize, 0usize, num_padding_patches);
        if num_padding_patches >= 0 {
            hidden_state =
                hidden_state.pad_with_zeros(D::Minus2, 0, num_padding_patches as usize)?;
        } else {
            hidden_state = hidden_state.narrow(
                D::Minus2,
                0,
                (hidden_state.dim(2)? as isize + num_padding_patches) as usize,
            )?;
        }

        // Prepare attention mask
        let mut attention_mask = aspect_ratio_mask.reshape((bs * num_concurrent_media, ()))?;
        attention_mask = _prepare_aspect_ratio_attention_mask(
            &attention_mask,
            self.num_patches,
            hidden_state.dim(2)?,
            hidden_state.dtype(),
            self.num_attn_heads,
        )?;

        // Apply encoder
        hidden_state = hidden_state.reshape((bs * num_concurrent_media, (), dim))?;
        let (mut hidden_state, all_intermediate_hidden_states) = self
            .transformer
            .forward_with_states(&hidden_state, Some(&attention_mask))?;

        hidden_state = self.layernorm_post.forward(&hidden_state)?;

        // Apply global encoder
        hidden_state = hidden_state.reshape((
            bs * num_concurrent_media,
            num_tiles,
            (num_patches as isize + num_padding_patches) as usize,
            dim,
        ))?;
        hidden_state = self
            .post_tile_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids)?;
        hidden_state = hidden_state.reshape((
            bs * num_concurrent_media,
            num_tiles * (num_patches as isize + num_padding_patches) as usize,
            dim,
        ))?;
        (hidden_state, _) = self
            .global_transformer
            .forward_with_states(&hidden_state, Some(&attention_mask))?;

        // Remove padding from hidden state
        hidden_state = hidden_state.reshape((
            bs * num_concurrent_media,
            num_tiles,
            (num_patches as isize + num_padding_patches) as usize,
            dim,
        ))?;
        hidden_state = hidden_state.narrow(
            2,
            0,
            (hidden_state.dims()[2] as isize - num_padding_patches) as usize,
        )?;
        hidden_state =
            hidden_state.reshape((bs, num_concurrent_media, num_tiles, num_patches, dim))?;

        // Collect intermediate layer outputs from encoder output
        let mut intermediate_hidden_states =
            Tensor::stack(&all_intermediate_hidden_states, D::Minus1)?;
        drop(all_intermediate_hidden_states);
        intermediate_hidden_states = intermediate_hidden_states
            .index_select(&self.intermediate_layers_indices, D::Minus1)?;

        // Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape((
            bs * num_concurrent_media,
            num_tiles,
            (num_patches as isize + num_padding_patches) as usize,
            (),
        ))?;
        intermediate_hidden_states = intermediate_hidden_states.narrow(
            2,
            0,
            (intermediate_hidden_states.dims()[2] as isize - num_padding_patches) as usize,
        )?;
        intermediate_hidden_states = intermediate_hidden_states.reshape((
            bs,
            num_concurrent_media,
            num_tiles,
            num_patches,
            (),
        ))?;

        // Concatenate final hidden state and intermediate hidden states
        Tensor::cat(&[hidden_state, intermediate_hidden_states], D::Minus1)
    }

    fn apply_class_embedding(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let (bs, _, hidden_size) = hidden_state.dims3()?;
        let class_embedding = self.class_embedding.expand((bs, 1, hidden_size))?;
        Tensor::cat(&[class_embedding, hidden_state.clone()], 1)
    }
}
