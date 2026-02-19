#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, LayerNormConfig, Module};
use mistralrs_quant::{
    ColumnParallelLayer, Convolution, QuantMethod, RowParallelLayer, ShardedVarBuilder,
};

use crate::{
    attention::SdpaParams,
    layers::{conv2d_no_bias, embedding, layer_norm, GetFloatInfo, Sdpa},
    pipeline::{text_models_inputs_processor::FlashParams, IsqModel},
    utils::unvarbuilder::UnVarBuilder,
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
    fn new(cfg: &MLlamaVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
        Ok(Self {
            gate: vb.get((1,), "gate")?,
            embedding: vb.get((num_patches, cfg.hidden_size), "embedding")?,
            tile_embedding: embedding(
                cfg.max_aspect_ratio_id() + 1,
                cfg.max_num_tiles * num_patches * cfg.hidden_size,
                vb.pp("tile_embedding"),
                &None,
            )?,
            num_patches,
            hidden_size: cfg.hidden_size,
            max_num_tiles: cfg.max_num_tiles,
        })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L197
    fn forward(&self, hidden_state: &Tensor, aspect_ratio_ids: &Tensor) -> Result<Tensor> {
        // position embeddings
        let mut gated_pos_embed = (1. - &self.gate.tanh()?)?.broadcast_mul(&self.embedding)?;
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
        gated_pos_embed = self.gate.tanh()?.broadcast_mul(&tile_position_embedding)?;

        hidden_state.broadcast_add(&gated_pos_embed)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb_gpe = UnVarBuilder::new();

        uvb_gpe.add_tensor("gate", self.gate.clone());
        uvb_gpe.add_tensor("embedding", self.embedding.clone());
        uvb_gpe.pp("tile_embedding").add(&self.tile_embedding);

        uvb_gpe.to_safetensors()
    }
}

struct MLlamaPrecomputedAspectRatioEmbedding {
    embedding: Embedding,
    gate: Option<Tensor>,
    max_num_tiles: usize,
    hidden_size: usize,
}

impl MLlamaPrecomputedAspectRatioEmbedding {
    fn new<const GATED: bool>(cfg: &MLlamaVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            embedding: embedding(
                cfg.max_aspect_ratio_id() + 1,
                cfg.max_num_tiles * cfg.hidden_size,
                vb.pp("embedding"),
                &None,
            )?,
            gate: if GATED {
                Some(vb.get((1,), "gate")?)
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
            embeddings = embeddings.broadcast_mul(&gate.tanh()?)?;
        }

        hidden_state.broadcast_add(&embeddings)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb_ptpe = UnVarBuilder::new();

        if let Some(gate) = self.gate.clone() {
            uvb_ptpe.add_tensor("gate", gate);
        }
        uvb_ptpe.pp("embedding").add(&self.embedding);

        uvb_ptpe.to_safetensors()
    }
}

struct MLlamaVisionAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    sdpa_params: SdpaParams,
    num_heads: usize,
    head_dim: usize,
}

impl MLlamaVisionAttention {
    fn new(
        cfg: &MLlamaVisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &None,
                false,
                comm,
                vb.pp("q_proj"),
            )?,
            k_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &None,
                false,
                comm,
                vb.pp("k_proj"),
            )?,
            v_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &None,
                false,
                comm,
                vb.pp("v_proj"),
            )?,
            o_proj: RowParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &None,
                false,
                comm,
                vb.pp("o_proj"),
            )?,
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            num_heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L243
    fn forward(&self, hidden_state: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_state = hidden_state.clone();
        let original_dtype = hidden_state.dtype();
        if let Some(t) = self.q_proj.quantized_act_type() {
            hidden_state = hidden_state.to_dtype(t)?;
        }
        let mut q = self.q_proj.forward(&hidden_state)?;
        let mut k = self.k_proj.forward(&hidden_state)?;
        let mut v = self.v_proj.forward(&hidden_state)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

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

        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: false,
        };

        let mut attn_output = Sdpa
            .run_attention(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                attention_mask,
                Some(&flash_params),
                &self.sdpa_params,
            )?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs, q_sq, ()))?
            .to_dtype(q.dtype())?;

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        let mut res = self.o_proj.forward(&attn_output)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct MLlamaMlp {
    act: VisionActivation,
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
}

impl MLlamaMlp {
    fn new(
        cfg: &MLlamaVisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            act: cfg.hidden_act,
            fc1: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                &None,
                true,
                comm,
                vb.pp("fc1"),
            )?,
            fc2: RowParallelLayer::new(
                cfg.intermediate_size,
                cfg.hidden_size,
                &None,
                true,
                comm,
                vb.pp("fc2"),
            )?,
        })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L223
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let original_dtype = hidden_states.dtype();
        let mut hidden_states = hidden_states.clone();
        if let Some(t) = self.fc1.quantized_act_type() {
            hidden_states = hidden_states.to_dtype(t)?;
        }
        hidden_states = self
            .fc2
            .forward(&self.act.forward(&self.fc1.forward(&hidden_states)?)?)?;
        if self.fc1.quantized_act_type().is_some() {
            hidden_states = hidden_states.to_dtype(original_dtype)?;
        }
        Ok(hidden_states)
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
    fn new<const GATED: bool>(
        cfg: &MLlamaVisionConfig,
        vb: ShardedVarBuilder,
        real_dev: &Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = MLlamaVisionAttention::new(cfg, vb.pp("self_attn"), comm)?;
        let mlp = MLlamaMlp::new(cfg, vb.pp("mlp"), comm)?;

        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            vb.pp("input_layernorm").set_device(real_dev.clone()),
        )?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            vb.pp("post_attention_layernorm")
                .set_device(real_dev.clone()),
        )?;

        if GATED {
            Ok(Self {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                gate_attn: Some(vb.get((1,), "gate_attn")?.to_device(real_dev)?),
                gate_ffn: Some(vb.get((1,), "gate_ffn")?.to_device(real_dev)?),
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
            hidden_state = gate.broadcast_mul(&hidden_state.tanh()?)?;
        }
        hidden_state = (residual + hidden_state)?;

        // FF
        let residual = hidden_state.clone();
        hidden_state = self.post_attention_layernorm.forward(&hidden_state)?;

        hidden_state = self.mlp.forward(&hidden_state)?;

        if let Some(gate) = &self.gate_ffn {
            hidden_state = gate.broadcast_mul(&hidden_state.tanh()?)?;
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
        vb: ShardedVarBuilder,
        real_dev: &Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..num_layers {
            layers.push(MLlamaVisionEncoderLayer::new::<GATED>(
                cfg,
                layers_vb.pp(i),
                real_dev,
                comm,
            )?);
        }
        Ok(Self { layers })
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L394
    /// Also (optionally) return hidden states at some indices
    fn forward_with_states(
        &self,
        hidden_state: &Tensor,
        attention_mask: Option<&Tensor>,
        intermediate_layers_indices: Option<&[usize]>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut hidden_state = hidden_state.clone();
        let mut hidden_states = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if intermediate_layers_indices.is_some_and(|indices: &[usize]| indices.contains(&i)) {
                hidden_states.push(hidden_state.clone());
            }
            hidden_state = layer.forward(&hidden_state, attention_mask)?;
        }
        Ok((hidden_state, hidden_states))
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb_t = UnVarBuilder::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_t.pp("layers").pp(i);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            if let Some(gate) = layer.gate_attn.clone() {
                uvb_l.add_tensor("gate_attn", gate);
            }
            if let Some(gate) = layer.gate_ffn.clone() {
                uvb_l.add_tensor("gate_ffn", gate);
            }
        }

        uvb_t.to_safetensors()
    }
}

fn _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: &Tensor,
    num_patches: usize,
    target_length: usize,
    dtype: DType,
    _num_attn_heads: usize,
) -> Result<Tensor> {
    let (bs, max_num_tiles) = aspect_ratio_mask.dims2()?;
    let mut attention_mask = aspect_ratio_mask
        .reshape((bs, max_num_tiles, 1, 1))?
        .repeat((1, 1, target_length, 1))?;

    // Mask padding patches
    let pad_patches = target_length - num_patches;
    let (bs, d1, d2, d3) = attention_mask.dims4()?;
    attention_mask = attention_mask.slice_assign(
        &[0..bs, 0..d1, (d2 - pad_patches)..d2, 0..d3],
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
    let neg_inf_value = dtype.finfo()?.min;
    attention_mask = attention_mask.reshape((bs, max_num_tiles * target_length, 1))?;
    attention_mask.matmul(
        &attention_mask
            .transpose(D::Minus1, D::Minus2)?
            .mul(neg_inf_value)?,
    )
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
    intermediate_layers_indices: Vec<usize>,
    num_attn_heads: usize,
}

impl MLlamaVisionModel {
    pub(super) fn new(
        cfg: &MLlamaVisionConfig,
        vb: ShardedVarBuilder,
        real_dev: &Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let patch_embedding = conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding").set_device(real_dev.clone()),
        )?;

        let class_embedding = vb
            .get((cfg.hidden_size,), "class_embedding")?
            .to_device(real_dev)?;
        let gated_positional_embedding = MLlamaPrecomputedPositionEmbedding::new(
            cfg,
            vb.pp("gated_positional_embedding")
                .set_device(real_dev.clone()),
        )?;

        let pre_tile_positional_embedding = MLlamaPrecomputedAspectRatioEmbedding::new::<true>(
            cfg,
            vb.pp("pre_tile_positional_embedding")
                .set_device(real_dev.clone()),
        )?;
        let post_tile_positional_embedding = MLlamaPrecomputedAspectRatioEmbedding::new::<true>(
            cfg,
            vb.pp("post_tile_positional_embedding")
                .set_device(real_dev.clone()),
        )?;

        // layer norms
        let layernorm_pre = layer_norm(
            cfg.hidden_size,
            LayerNormConfig::default(),
            vb.pp("layernorm_pre").set_device(real_dev.clone()),
        )?;
        let layernorm_post = layer_norm(
            cfg.hidden_size,
            LayerNormConfig::default(),
            vb.pp("layernorm_post").set_device(real_dev.clone()),
        )?;

        // encoders
        let transformer = MLlamaVisionEncoder::new::<false>(
            cfg,
            cfg.num_hidden_layers,
            vb.pp("transformer"),
            real_dev,
            comm,
        )?;
        let global_transformer = MLlamaVisionEncoder::new::<true>(
            cfg,
            cfg.num_global_layers,
            vb.pp("global_transformer"),
            real_dev,
            comm,
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
            intermediate_layers_indices: cfg.intermediate_layers_indices.clone(),
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
        let patch_embeds = Convolution.forward_2d(&self.patch_embedding, &pixel_values)?;
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
        if attention_mask.dim(0)? != 1 {
            attention_mask = attention_mask.unsqueeze(1)?;
        }

        // Apply encoder
        hidden_state = hidden_state.reshape((bs * num_concurrent_media, (), dim))?;
        let (mut hidden_state, all_intermediate_hidden_states) =
            self.transformer.forward_with_states(
                &hidden_state,
                Some(&attention_mask),
                Some(&self.intermediate_layers_indices),
            )?;

        // Collect intermediate layer outputs from encoder output
        let mut intermediate_hidden_states =
            Tensor::stack(&all_intermediate_hidden_states, D::Minus1)?;
        drop(all_intermediate_hidden_states);

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
        (hidden_state, _) = self.global_transformer.forward_with_states(
            &hidden_state,
            Some(&attention_mask),
            None,
        )?;

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

    pub fn get_isq_layers(&mut self) -> Vec<&mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>> {
        let mut layers = Vec::new();
        for layer in &mut self.global_transformer.layers {
            layers.push(&mut layer.self_attn.q_proj);
            layers.push(&mut layer.self_attn.k_proj);
            layers.push(&mut layer.self_attn.v_proj);
            layers.push(&mut layer.self_attn.o_proj);

            layers.push(&mut layer.mlp.fc1);
            layers.push(&mut layer.mlp.fc2);
        }
        for layer in &mut self.transformer.layers {
            layers.push(&mut layer.self_attn.q_proj);
            layers.push(&mut layer.self_attn.k_proj);
            layers.push(&mut layer.self_attn.v_proj);
            layers.push(&mut layer.self_attn.o_proj);

            layers.push(&mut layer.mlp.fc1);
            layers.push(&mut layer.mlp.fc2);
        }
        layers
    }
}

impl IsqModel for MLlamaVisionModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(
            &mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>,
            Option<usize>,
        )>,
        &dyn crate::device_map::DeviceMapper,
    ) {
        unreachable!("MLlamaVision model cannot be quantized.");
    }
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("patch_embedding").add(&self.patch_embedding);
        uvb.add_tensor("class_embedding", self.class_embedding.clone());

        // gated_positional_embedding
        uvb.pp("gated_positional_embedding")
            .extend(self.gated_positional_embedding.residual_tensors());

        // pre_tile_positional_embedding
        uvb.pp("pre_tile_positional_embedding")
            .extend(self.pre_tile_positional_embedding.residual_tensors());

        // post_tile_positional_embedding
        uvb.pp("post_tile_positional_embedding")
            .extend(self.post_tile_positional_embedding.residual_tensors());

        uvb.pp("layernorm_pre").add(&self.layernorm_pre);
        uvb.pp("layernorm_post").add(&self.layernorm_post);

        // transformer
        uvb.pp("transformer")
            .extend(self.transformer.residual_tensors());

        // global_transformer
        uvb.pp("global_transformer")
            .extend(self.global_transformer.residual_tensors());

        uvb.to_safetensors()
    }
}
