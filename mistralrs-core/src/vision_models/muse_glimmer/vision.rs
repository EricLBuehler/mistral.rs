use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, LayerNormConfig, Linear};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use super::config::{Config, VisionAttentionType, VisionConfig};
use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{self, Activation, Sdpa},
    pipeline::text_models_inputs_processor::FlashParams,
    utils::unvarbuilder::UnVarBuilder,
};

fn scaleless_rms_norm_f32(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?;
    let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
    let inv_rms = (&variance + eps)?.recip()?.sqrt()?;
    xs.broadcast_mul(&inv_rms)?.to_dtype(dtype)
}

fn interpolation_taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let source = (index as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let floor = source.floor() as isize;
    [(floor, 0isize), (floor + 1, 1isize)].map(|(raw, offset)| {
        let distance = (source - floor as f32 - offset as f32).abs();
        let in_bounds = if raw >= 0 && raw < side as isize {
            1.0
        } else {
            0.0
        };
        let weight = (1.0 - distance).max(0.0) * in_bounds;
        (raw.clamp(0, side as isize - 1) as usize, weight)
    })
}

fn interpolation_indices_and_weights(
    grids: &[[usize; 3]],
    pos_height: usize,
    pos_width: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    for &[frames, height, width] in grids {
        for _ in 0..frames {
            for row in 0..height {
                let row_taps = interpolation_taps(row, height, pos_height);
                for col in 0..width {
                    let col_taps = interpolation_taps(col, width, pos_width);
                    for (row_index, row_weight) in row_taps {
                        for (col_index, col_weight) in col_taps {
                            indices.push((row_index * pos_width + col_index) as u32);
                            weights.push(row_weight * col_weight);
                        }
                    }
                }
            }
        }
    }
    (indices, weights)
}

fn sequence_layout(
    grids: &[[usize; 3]],
    window_side: usize,
) -> (Vec<u32>, Vec<u32>, Vec<usize>, Vec<usize>) {
    let total = grids
        .iter()
        .map(|[frames, height, width]| frames * height * width)
        .sum::<usize>();
    let mut permutation = Vec::with_capacity(total);
    let mut full_cu = vec![0];
    let mut window_cu = vec![0];
    let mut offset = 0usize;
    for &[frames, height, width] in grids {
        let area = height * width;
        for frame in 0..frames {
            let frame_start = offset + frame * area;
            for window_row in (0..height).step_by(window_side) {
                for window_col in (0..width).step_by(window_side) {
                    let end_row = (window_row + window_side).min(height);
                    let end_col = (window_col + window_side).min(width);
                    for row in window_row..end_row {
                        for col in window_col..end_col {
                            permutation.push((frame_start + row * width + col) as u32);
                        }
                    }
                    window_cu.push(permutation.len());
                }
            }
            full_cu.push(frame_start + area);
        }
        offset += frames * area;
    }
    let mut inverse = vec![0u32; total];
    for (new_index, &old_index) in permutation.iter().enumerate() {
        inverse[old_index as usize] = new_index as u32;
    }
    (permutation, inverse, full_cu, window_cu)
}

fn pixel_shuffle_indices(grids: &[[usize; 3]], merge_size: usize) -> Vec<u32> {
    let mut indices = Vec::new();
    let mut offset = 0usize;
    for &[frames, height, width] in grids {
        let area = height * width;
        for frame in 0..frames {
            let frame_start = offset + frame * area;
            for block_row in 0..height / merge_size {
                for block_col in 0..width / merge_size {
                    for inner_row in 0..merge_size {
                        for inner_col in 0..merge_size {
                            let row = block_row * merge_size + inner_row;
                            let col = block_col * merge_size + inner_col;
                            indices.push((frame_start + row * width + col) as u32);
                        }
                    }
                }
            }
        }
        offset += frames * area;
    }
    indices
}

fn vision_rope_cos_sin(
    grids: &[[usize; 3]],
    permutation: &[u32],
    head_dim: usize,
    theta: f64,
    half_split: bool,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let frequency_count = head_dim / 4;
    let inv_freq = (0..frequency_count)
        .map(|index| 1.0 / (theta as f32).powf((2 * index) as f32 / (head_dim / 2) as f32))
        .collect::<Vec<_>>();
    let mut positions = Vec::with_capacity(permutation.len());
    for &[frames, height, width] in grids {
        for _ in 0..frames {
            for row in 0..height {
                for col in 0..width {
                    positions.push(((col + 1) as f32, (row + 1) as f32));
                }
            }
        }
    }
    let mut frequencies = Vec::with_capacity(permutation.len() * head_dim);
    for &index in permutation {
        let (width, height) = positions[index as usize];
        if half_split {
            for position in [width, height, width, height] {
                frequencies.extend(inv_freq.iter().map(|frequency| position * frequency));
            }
        } else {
            for position in [width, height] {
                for frequency in &inv_freq {
                    let value = position * frequency;
                    frequencies.extend([value, value]);
                }
            }
        }
    }
    let frequencies = Tensor::from_vec(frequencies, (permutation.len(), head_dim), device)?;
    Ok((frequencies.cos()?, frequencies.sin()?))
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let dim = xs.dim(D::Minus1)?;
    let first = xs.narrow(D::Minus1, 0, dim / 2)?;
    let second = xs.narrow(D::Minus1, dim / 2, dim / 2)?;
    Tensor::cat(&[&second.neg()?, &first], D::Minus1)
}

fn rotate_adjacent(xs: &Tensor) -> Result<Tensor> {
    let (seq_len, num_heads, head_dim) = xs.dims3()?;
    let xs = xs.reshape((seq_len, num_heads, head_dim / 2, 2))?;
    let even = xs.narrow(D::Minus1, 0, 1)?;
    let odd = xs.narrow(D::Minus1, 1, 1)?;
    Tensor::cat(&[&odd.neg()?, &even], D::Minus1)?.reshape((seq_len, num_heads, head_dim))
}

fn apply_vision_rope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    half_split: bool,
) -> Result<(Tensor, Tensor)> {
    let q_dtype = q.dtype();
    let k_dtype = k.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let cos = cos.unsqueeze(1)?.to_dtype(DType::F32)?;
    let sin = sin.unsqueeze(1)?.to_dtype(DType::F32)?;
    let rotate = if half_split {
        rotate_half
    } else {
        rotate_adjacent
    };
    let q = (q.broadcast_mul(&cos)? + rotate(&q)?.broadcast_mul(&sin)?)?;
    let k = (k.broadcast_mul(&cos)? + rotate(&k)?.broadcast_mul(&sin)?)?;
    Ok((q.to_dtype(q_dtype)?, k.to_dtype(k_dtype)?))
}

struct PatchEmbedder {
    projection: Linear,
    full_input_size: usize,
    collapsed_temporal: bool,
}

fn patch_projection_input(
    xs: &Tensor,
    full_input_size: usize,
    collapsed_temporal: bool,
) -> Result<Tensor> {
    if collapsed_temporal && xs.dim(1)? == full_input_size {
        xs.narrow(1, 0, full_input_size / 2)?.contiguous()
    } else {
        Ok(xs.clone())
    }
}

impl PatchEmbedder {
    fn new(cfg: &VisionConfig, collapsed_temporal: bool, vb: ShardedVarBuilder) -> Result<Self> {
        let frame_size = 3 * cfg.patch_size * cfg.patch_size;
        let input_size = if collapsed_temporal {
            frame_size
        } else {
            cfg.patch_temporal * frame_size
        };
        Ok(Self {
            projection: layers::linear_no_bias(input_size, cfg.hidden_size, vb)?,
            full_input_size: cfg.patch_temporal * frame_size,
            collapsed_temporal,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = patch_projection_input(xs, self.full_input_size, self.collapsed_temporal)?;
        self.projection.forward(&xs)
    }
}

struct VisionAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    head_dim: usize,
    rope_half_split: bool,
}

impl VisionAttention {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        rope_half_split: bool,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.hidden_size,
                &None,
                true,
                comm,
                vb.pp("q_proj"),
            )?,
            k_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.hidden_size,
                &None,
                true,
                comm,
                vb.pp("k_proj"),
            )?,
            v_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.hidden_size,
                &None,
                true,
                comm,
                vb.pp("v_proj"),
            )?,
            proj: RowParallelLayer::new(
                cfg.hidden_size,
                cfg.hidden_size,
                &None,
                true,
                comm,
                vb.pp("proj"),
            )?,
            num_heads: cfg.num_attention_heads / comm.world_size(),
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            rope_half_split,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let (q, k, v) = crate::ops::qkv_projections(
            &xs.unsqueeze(0)?,
            &*self.q_proj,
            &*self.k_proj,
            &*self.v_proj,
        )?;
        let q = q
            .squeeze(0)?
            .reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = k
            .squeeze(0)?
            .reshape((seq_len, self.num_heads, self.head_dim))?;
        let v = v
            .squeeze(0)?
            .reshape((seq_len, self.num_heads, self.head_dim))?;
        let (q, k) = apply_vision_rope(&q, &k, cos, sin, self.rope_half_split)?;
        let flash_params = FlashParams::empty(false);
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };
        let mut outputs = Vec::with_capacity(cu_seqlens.len().saturating_sub(1));
        for bounds in cu_seqlens.windows(2) {
            let start = bounds[0];
            let len = bounds[1] - start;
            if len == 0 {
                continue;
            }
            let q = q
                .narrow(0, start, len)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let k = k
                .narrow(0, start, len)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let v = v
                .narrow(0, start, len)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let output = Sdpa
                .run_attention(
                    &q,
                    &k,
                    &v,
                    &AttentionMask::None,
                    Some(&flash_params),
                    &sdpa_params,
                )?
                .squeeze(0)?
                .transpose(0, 1)?
                .reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(output.to_dtype(xs.dtype())?);
        }
        let output = Tensor::cat(&outputs, 0)?.unsqueeze(0)?;
        self.proj.forward(&output)?.squeeze(0)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("proj").add(&self.proj);
        uvb.to_safetensors()
    }
}

struct VisionMlp {
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
    activation: Activation,
}

impl VisionMlp {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
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
            activation: cfg.hidden_act,
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

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attention: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        rope_half_split: bool,
    ) -> Result<Self> {
        let norm_config = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        Ok(Self {
            norm1: layers::layer_norm(cfg.hidden_size, norm_config, vb.pp("norm1"))?,
            norm2: layers::layer_norm(cfg.hidden_size, norm_config, vb.pp("norm2"))?,
            attention: VisionAttention::new(cfg, vb.pp("attn"), comm, rope_half_split)?,
            mlp: VisionMlp::new(cfg, vb.pp("mlp"), comm)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let xs = (xs
            + self
                .attention
                .forward(&self.norm1.forward(xs)?, cu_seqlens, cos, sin)?)?;
        &xs + self.mlp.forward(&self.norm2.forward(&xs)?)?
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("norm1").add(&self.norm1);
        uvb.pp("norm2").add(&self.norm2);
        uvb.pp("attn").extend(self.attention.residual_tensors());
        uvb.pp("mlp").extend(self.mlp.residual_tensors());
        uvb.to_safetensors()
    }
}

struct VisionAdapter {
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
    activation: Activation,
}

impl VisionAdapter {
    fn new(cfg: &Config, vb: ShardedVarBuilder, comm: &Arc<mistralrs_quant::Comm>) -> Result<Self> {
        Ok(Self {
            fc1: ColumnParallelLayer::new(
                cfg.out_hidden_size,
                cfg.projector_hidden_size,
                &None,
                false,
                comm,
                vb.pp("fc1"),
            )?,
            fc2: RowParallelLayer::new(
                cfg.projector_hidden_size,
                cfg.projector_hidden_size,
                &None,
                false,
                comm,
                vb.pp("fc2"),
            )?,
            activation: cfg.projector_hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.activation.forward(&self.fc1.forward(xs)?)?;
        self.activation.forward(&self.fc2.forward(&xs)?)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("fc1").add(&self.fc1);
        uvb.pp("fc2").add(&self.fc2);
        uvb.to_safetensors()
    }
}

pub(super) struct VisionModel {
    patch_embedder: PatchEmbedder,
    position_embedding: Embedding,
    ln_pre: LayerNorm,
    ln_post: LayerNorm,
    blocks: Vec<VisionBlock>,
    layer_types: Vec<VisionAttentionType>,
    adapter: VisionAdapter,
    projection: Arc<dyn QuantMethod>,
    head_dim: usize,
    window_side: usize,
    merge_size: usize,
    pos_emb_height: usize,
    pos_emb_width: usize,
    rope_theta: f64,
    perception_norm_eps: f64,
    collapsed_temporal: bool,
    rope_half_split: bool,
}

impl VisionModel {
    pub(super) fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        rope_half_split: bool,
    ) -> Result<Self> {
        cfg.validate()?;
        let tower = vb.pp("model").pp("vision_tower");
        let patch_embedder = PatchEmbedder::new(
            &cfg.vision_config,
            cfg.gguf_collapsed_temporal,
            tower.pp("patch_embedder").pp("patch_embedding"),
        )?;
        let position_embedding_vb = tower.pp("patch_embedder").pp("position_embedding_table");
        let position_embedding = Embedding::new(
            position_embedding_vb.get(
                (
                    cfg.vision_config.pos_emb_height * cfg.vision_config.pos_emb_width,
                    cfg.vision_config.hidden_size,
                ),
                "weight",
            )?,
            cfg.vision_config.hidden_size,
        );
        let norm_config = LayerNormConfig {
            eps: cfg.vision_config.layer_norm_eps,
            ..Default::default()
        };
        let ln_pre = layers::layer_norm(
            cfg.vision_config.hidden_size,
            norm_config,
            tower.pp("ln_pre"),
        )?;
        let ln_post = layers::layer_norm(
            cfg.vision_config.hidden_size,
            norm_config,
            tower.pp("ln_post"),
        )?;
        let mut blocks = Vec::with_capacity(cfg.vision_config.num_hidden_layers);
        for layer_idx in 0..cfg.vision_config.num_hidden_layers {
            blocks.push(VisionBlock::new(
                &cfg.vision_config,
                tower.pp("layers").pp(layer_idx),
                comm,
                rope_half_split,
            )?);
        }
        let adapter = VisionAdapter::new(cfg, vb.pp("model").pp("vision_adapter"), comm)?;
        let projection = ReplicatedLayer::new(
            cfg.projector_hidden_size,
            cfg.text_config.hidden_size,
            &None,
            false,
            vb.pp("model").pp("vision_projection"),
        )?;
        Ok(Self {
            patch_embedder,
            position_embedding,
            ln_pre,
            ln_post,
            blocks,
            layer_types: cfg.vision_config.layer_types()?,
            adapter,
            projection,
            head_dim: cfg.vision_config.hidden_size / cfg.vision_config.num_attention_heads,
            window_side: cfg.vision_config.pos_emb_height,
            merge_size: cfg.vision_config.merge_size,
            pos_emb_height: cfg.vision_config.pos_emb_height,
            pos_emb_width: cfg.vision_config.pos_emb_width,
            rope_theta: cfg.vision_config.rope_parameters.rope_theta,
            perception_norm_eps: cfg.text_config.rms_norm_eps,
            collapsed_temporal: cfg.gguf_collapsed_temporal,
            rope_half_split,
        })
    }

    pub(super) fn collapsed_temporal(&self) -> bool {
        self.collapsed_temporal
    }

    pub(super) fn forward(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let grids = grid_thw
            .to_vec2::<u32>()?
            .into_iter()
            .map(|grid| [grid[0] as usize, grid[1] as usize, grid[2] as usize])
            .collect::<Vec<_>>();
        if grids
            .iter()
            .any(|[_, height, width]| height % self.merge_size != 0 || width % self.merge_size != 0)
        {
            candle_core::bail!("Muse-Glimmer vision grids must be divisible by merge_size");
        }
        let expected_patches = grids
            .iter()
            .map(|[frames, height, width]| frames * height * width)
            .sum::<usize>();
        if pixel_values.dim(0)? != expected_patches {
            candle_core::bail!(
                "Muse-Glimmer received {} patches for a grid requiring {expected_patches}",
                pixel_values.dim(0)?
            );
        }

        let dtype = self.position_embedding.embeddings().dtype();
        let mut hidden_states = self
            .patch_embedder
            .forward(&pixel_values.to_dtype(dtype)?)?;
        let (indices, weights) =
            interpolation_indices_and_weights(&grids, self.pos_emb_height, self.pos_emb_width);
        let indices = Tensor::from_vec(indices, (expected_patches, 4), hidden_states.device())?;
        let weights = Tensor::from_vec(weights, (expected_patches, 4, 1), hidden_states.device())?;
        let position_embeddings = self
            .position_embedding
            .embeddings()
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((expected_patches, 4, ()))?
            .to_dtype(DType::F32)?
            .broadcast_mul(&weights)?
            .sum(1)?
            .to_dtype(hidden_states.dtype())?;
        hidden_states = (hidden_states + position_embeddings)?;
        hidden_states = hidden_states.apply(&self.ln_pre)?;

        let (permutation, inverse, full_cu, window_cu) = sequence_layout(&grids, self.window_side);
        let permutation_tensor = Tensor::from_vec(
            permutation.clone(),
            permutation.len(),
            hidden_states.device(),
        )?;
        hidden_states = hidden_states.index_select(&permutation_tensor, 0)?;
        let (cos, sin) = vision_rope_cos_sin(
            &grids,
            &permutation,
            self.head_dim,
            self.rope_theta,
            self.rope_half_split,
            hidden_states.device(),
        )?;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let cu_seqlens = match self.layer_types[layer_idx] {
                VisionAttentionType::FullAttention => &full_cu,
                VisionAttentionType::WindowAttention => &window_cu,
            };
            hidden_states = block.forward(&hidden_states, cu_seqlens, &cos, &sin)?;
        }
        let inverse = Tensor::from_vec(inverse, expected_patches, hidden_states.device())?;
        hidden_states = hidden_states
            .index_select(&inverse, 0)?
            .apply(&self.ln_post)?;

        let shuffle = pixel_shuffle_indices(&grids, self.merge_size);
        let output_tokens = shuffle.len() / self.merge_size.pow(2);
        let shuffle = Tensor::from_vec(shuffle, expected_patches, hidden_states.device())?;
        hidden_states = hidden_states
            .index_select(&shuffle, 0)?
            .reshape((output_tokens, self.merge_size.pow(2), ()))?
            .permute((0, 2, 1))?
            .contiguous()?
            .reshape((output_tokens, ()))?;
        hidden_states = self.adapter.forward(&hidden_states)?;
        hidden_states = self.projection.forward(&hidden_states)?;
        let hidden_states = scaleless_rms_norm_f32(&hidden_states, self.perception_norm_eps)?;
        Ok(hidden_states)
    }

    pub(super) fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let tower = uvb.pp("model").pp("vision_tower");
        tower
            .pp("patch_embedder")
            .pp("patch_embedding")
            .add(&self.patch_embedder.projection);
        tower
            .pp("patch_embedder")
            .pp("position_embedding_table")
            .add(&self.position_embedding);
        tower.pp("ln_pre").add(&self.ln_pre);
        tower.pp("ln_post").add(&self.ln_post);
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            tower
                .pp("layers")
                .pp(layer_idx)
                .extend(block.residual_tensors());
        }
        uvb.pp("model")
            .pp("vision_adapter")
            .extend(self.adapter.residual_tensors());
        uvb.pp("model")
            .pp("vision_projection")
            .add(&self.projection);
        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write, sync::Arc};

    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding, GgufWeightSource};
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn gguf_f32_patch_projection_is_a_bf16_residual() -> Result<()> {
        let source_weight = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((4, 3, 1, 1))?;
        let qtensor = QTensor::quantize(&source_weight, GgmlDType::F32)?;
        let mut file = NamedTempFile::new()?;
        gguf_file::write(
            file.as_file_mut(),
            &[],
            &[("v.patch_embd.weight", &qtensor)],
        )?;
        file.as_file_mut().flush()?;

        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        let native_name = "model.vision_tower.patch_embedder.patch_embedding.weight";
        let bindings = GgufBindingMap::new().with_binding(
            native_name,
            GgufTensorBinding::tensor("v.patch_embd.weight").reshape(vec![4, 3]),
        );
        let source = Arc::new(GgufWeightSource::new(archive, &bindings, DType::BF16)?);
        let vb = source
            .sharded_var_builder(Device::Cpu)
            .pp("model.vision_tower.patch_embedder.patch_embedding");
        let cfg: VisionConfig =
            serde_json::from_str(r#"{"hidden_size":4,"patch_size":1,"patch_temporal":2}"#).unwrap();
        let patch_embedder = PatchEmbedder::new(&cfg, true, vb)?;
        let uvb = UnVarBuilder::new();
        uvb.pp("model")
            .pp("vision_tower")
            .pp("patch_embedder")
            .pp("patch_embedding")
            .add(&patch_embedder.projection);
        let residuals = uvb.to_safetensors();
        let (_, residual) = residuals
            .iter()
            .find(|(name, _)| name == native_name)
            .unwrap();

        assert_eq!(residual.dtype(), DType::BF16);
        assert_eq!(residual.dims(), &[4, 3]);
        Ok(())
    }

    #[test]
    fn collapsed_temporal_projection_input_is_contiguous() -> Result<()> {
        let pixels = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 12))?;
        let collapsed = patch_projection_input(&pixels, 12, true)?;
        assert_eq!(collapsed.dims(), &[2, 6]);
        assert!(collapsed.is_contiguous());

        let weights = Tensor::ones((6, 3), DType::F32, &Device::Cpu)?;
        assert_eq!(collapsed.matmul(&weights)?.dims(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn window_layout_matches_transformers_raster_windows() {
        let (permutation, inverse, full, window) = sequence_layout(&[[1, 3, 3]], 2);
        assert_eq!(permutation, vec![0, 1, 3, 4, 2, 5, 6, 7, 8]);
        assert_eq!(inverse, vec![0, 1, 4, 2, 3, 5, 6, 7, 8]);
        assert_eq!(full, vec![0, 9]);
        assert_eq!(window, vec![0, 4, 6, 8, 9]);
    }

    #[test]
    fn full_attention_keeps_video_frames_segmented() {
        let (permutation, inverse, full, window) = sequence_layout(&[[2, 2, 2]], 32);
        assert_eq!(permutation, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(inverse, permutation);
        assert_eq!(full, vec![0, 4, 8]);
        assert_eq!(window, vec![0, 4, 8]);
    }

    #[test]
    fn pixel_shuffle_is_channel_major_after_spatial_gather() {
        assert_eq!(
            pixel_shuffle_indices(&[[1, 4, 4]], 2),
            vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
    }

    #[test]
    fn interpolation_uses_half_pixel_centers_and_zero_padding() {
        let (indices, weights) = interpolation_indices_and_weights(&[[1, 3, 3]], 2, 2);
        assert_eq!(&indices[..4], &[0, 0, 0, 0]);
        let corner_weight = weights[..4].iter().sum::<f32>();
        assert!((corner_weight - 25.0 / 36.0).abs() < 1e-6);
        let center = 4 * 4;
        assert_eq!(&indices[center..center + 4], &[0, 1, 2, 3]);
        assert!(weights[center..center + 4]
            .iter()
            .all(|weight| (*weight - 0.25).abs() < 1e-6));
    }

    #[test]
    fn adjacent_rope_matches_half_split_after_gguf_weight_unpermute() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[[0.25f32, -0.5, 1.0]], &device)?;
        let weight = Tensor::from_vec(
            (0..24).map(|value| value as f32 / 11.0 - 0.7).collect(),
            (8, 3),
            &device,
        )?;
        let to_adjacent = Tensor::new(&[0u32, 4, 1, 5, 2, 6, 3, 7], &device)?;
        let to_half_split = Tensor::new(&[0u32, 2, 4, 6, 1, 3, 5, 7], &device)?;
        let weight_adjacent = weight.index_select(&to_adjacent, 0)?;
        let q_half_split = input.matmul(&weight.t()?)?.reshape((1, 1, 8))?;
        let q_adjacent = input.matmul(&weight_adjacent.t()?)?.reshape((1, 1, 8))?;

        let (cos_half_split, sin_half_split) =
            vision_rope_cos_sin(&[[1, 1, 1]], &[0], 8, 10_000.0, true, &device)?;
        let (cos_adjacent, sin_adjacent) =
            vision_rope_cos_sin(&[[1, 1, 1]], &[0], 8, 10_000.0, false, &device)?;
        let (rotated_half_split, _) = apply_vision_rope(
            &q_half_split,
            &q_half_split,
            &cos_half_split,
            &sin_half_split,
            true,
        )?;
        let (rotated_adjacent, _) = apply_vision_rope(
            &q_adjacent,
            &q_adjacent,
            &cos_adjacent,
            &sin_adjacent,
            false,
        )?;
        let rotated_adjacent = rotated_adjacent.index_select(&to_half_split, 2)?;
        let expected = rotated_half_split.flatten_all()?.to_vec1::<f32>()?;
        let actual = rotated_adjacent.flatten_all()?.to_vec1::<f32>()?;
        for (expected, actual) in expected.iter().zip(actual) {
            assert!((expected - actual).abs() < 1e-6);
        }
        Ok(())
    }
}
