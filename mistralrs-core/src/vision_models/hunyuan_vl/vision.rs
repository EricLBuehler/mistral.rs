use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, LayerNormConfig, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    attention::{AttentionMask, Sdpa, SdpaParams},
    layers::{conv2d, layer_norm, linear, F32RmsNorm},
    pipeline::text_models_inputs_processor::FlashParams,
    utils::unvarbuilder::UnVarBuilder,
    vision_models::hunyuan_vl::config::VisionConfig,
};

/// The learned position table reserves its first row for a cls token that is never sampled.
const POSITION_TABLE_EXTRA_TOKENS: usize = 1;

// Match HF's FP32 bilinear interpolation with explicit output size and align_corners=False.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn interpolate_patch_positions(
    table: &Tensor,
    grid: &[Vec<u32>],
    position_edge: usize,
    position_offset: usize,
) -> Result<Tensor> {
    let device = table.device();
    let dtype = table.dtype();
    let source_coordinates = |out_len: usize| {
        let scale = position_edge as f32 / out_len as f32;
        (0..out_len)
            .map(|i| ((i as f32 + 0.5) * scale - 0.5).max(0.0))
            .collect::<Vec<_>>()
    };

    let mut idx_lists: [Vec<i64>; 4] = Default::default();
    let mut weight_lists: [Vec<f32>; 4] = Default::default();

    for g in grid {
        let t = g[0] as usize;
        let h_vals = source_coordinates(g[1] as usize);
        let w_vals = source_coordinates(g[2] as usize);
        for _ in 0..t {
            for h_val in &h_vals {
                let hf = h_val.floor() as usize;
                let hc = (hf + 1).min(position_edge - 1);
                let dh = h_val - hf as f32;
                for w_val in &w_vals {
                    let wf = w_val.floor() as usize;
                    let wc = (wf + 1).min(position_edge - 1);
                    let dw = w_val - wf as f32;

                    idx_lists[0].push((position_offset + hf * position_edge + wf) as i64);
                    idx_lists[1].push((position_offset + hf * position_edge + wc) as i64);
                    idx_lists[2].push((position_offset + hc * position_edge + wf) as i64);
                    idx_lists[3].push((position_offset + hc * position_edge + wc) as i64);
                    weight_lists[0].push((1.0 - dh) * (1.0 - dw));
                    weight_lists[1].push((1.0 - dh) * dw);
                    weight_lists[2].push(dh * (1.0 - dw));
                    weight_lists[3].push(dh * dw);
                }
            }
        }
    }

    let mut merged: Option<Tensor> = None;
    for (idxs, weights) in idx_lists.iter().zip(weight_lists.iter()) {
        let idx_tensor = Tensor::from_vec(idxs.clone(), (idxs.len(),), device)?;
        let values = table.index_select(&idx_tensor, 0)?.to_dtype(DType::F32)?;
        let weights =
            Tensor::from_vec(weights.clone(), (weights.len(),), device)?.unsqueeze(D::Minus1)?;
        let term = values.broadcast_mul(&weights)?;
        merged = Some(match merged {
            Some(accumulated) => (accumulated + term)?,
            None => term,
        });
    }
    let merged = match merged {
        Some(merged) => merged,
        None => Tensor::zeros((0, table.dim(D::Minus1)?), DType::F32, device)?,
    };
    merged.to_dtype(dtype)
}

struct PatchEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    num_channels: usize,
    patch_size: usize,
    hidden_size: usize,
    position_edge: usize,
    position_offset: usize,
}

impl PatchEmbeddings {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let patch_embedding = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;
        let position_edge = cfg.max_image_size / cfg.patch_size;
        let num_positions = position_edge * position_edge + POSITION_TABLE_EXTRA_TOKENS;
        let position_embedding = Embedding::new(
            vb.pp("position_embedding")
                .get((num_positions, cfg.hidden_size), "weight")?,
            cfg.hidden_size,
        );
        Ok(Self {
            patch_embedding,
            position_embedding,
            num_channels: cfg.num_channels,
            patch_size: cfg.patch_size,
            hidden_size: cfg.hidden_size,
            position_edge,
            position_offset: POSITION_TABLE_EXTRA_TOKENS,
        })
    }

    fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        interpolate_patch_positions(
            self.position_embedding.embeddings(),
            &grid_thw.to_vec2::<u32>()?,
            self.position_edge,
            self.position_offset,
        )
    }

    fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let dtype = self.position_embedding.embeddings().dtype();
        let xs = xs.to_dtype(dtype)?.reshape((
            (),
            self.num_channels,
            self.patch_size,
            self.patch_size,
        ))?;
        let xs = self
            .patch_embedding
            .forward(&xs)?
            .reshape(((), self.hidden_size))?;
        xs.broadcast_add(&self.fast_pos_embed_interpolate(grid_thw)?)
    }
}

struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    #[allow(clippy::cast_precision_loss)]
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let q_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?;
        let o_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let ys = Sdpa.run_attention(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            &AttentionMask::None,
            Some(&FlashParams::empty(false)),
            &SdpaParams {
                n_kv_groups: 1,
                softcap: None,
                softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        )?;
        self.o_proj
            .forward(&ys.transpose(1, 2)?.reshape((b, s, ()))?)
    }
}

struct VisionMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    act_fn: crate::layers::Activation,
}

impl VisionMlp {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            dense_h_to_4h: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("dense_h_to_4h"),
            )?,
            dense_4h_to_h: linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("dense_4h_to_h"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.dense_4h_to_h
            .forward(&self.act_fn.forward(&self.dense_h_to_4h.forward(xs)?)?)
    }
}

struct VisionLayer {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    self_attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionLayer {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: cfg.rms_norm_eps,
            ..Default::default()
        };
        Ok(Self {
            input_layernorm: layer_norm(cfg.hidden_size, ln_cfg, vb.pp("input_layernorm"))?,
            post_attention_layernorm: layer_norm(
                cfg.hidden_size,
                ln_cfg,
                vb.pp("post_attention_layernorm"),
            )?,
            self_attn: VisionAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: VisionMlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = (self.self_attn.forward(&self.input_layernorm.forward(xs)?)? + residual)?;
        let residual = &xs;
        self.mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?
            .add(residual)
    }
}

struct PatchMerger {
    spatial_merge_size: usize,
    before_rms: F32RmsNorm,
    after_rms: F32RmsNorm,
    proj0: Conv2d,
    proj2: Conv2d,
    mlp: Linear,
    act_fn: crate::layers::Activation,
    image_begin: Tensor,
    image_end: Tensor,
    image_newline: Tensor,
    #[allow(dead_code)]
    image_sep: Tensor,
}

impl PatchMerger {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let merged_hidden = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        Ok(Self {
            spatial_merge_size: cfg.spatial_merge_size,
            before_rms: F32RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("before_rms"))?,
            after_rms: F32RmsNorm::new(cfg.out_hidden_size, cfg.rms_norm_eps, vb.pp("after_rms"))?,
            proj0: conv2d(
                cfg.hidden_size,
                cfg.hidden_size * 2,
                cfg.spatial_merge_size,
                Conv2dConfig {
                    stride: cfg.spatial_merge_size,
                    ..Default::default()
                },
                vb.pp("proj").pp(0),
            )?,
            proj2: conv2d(
                cfg.hidden_size * 2,
                merged_hidden,
                1,
                Default::default(),
                vb.pp("proj").pp(2),
            )?,
            mlp: linear(merged_hidden, cfg.out_hidden_size, vb.pp("mlp"))?,
            act_fn: cfg.hidden_act,
            image_begin: vb.get((cfg.out_hidden_size,), "image_begin")?,
            image_end: vb.get((cfg.out_hidden_size,), "image_end")?,
            image_newline: vb.get((merged_hidden,), "image_newline")?,
            image_sep: vb.get((cfg.out_hidden_size,), "image_sep")?,
        })
    }

    fn forward_one(&self, xs: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        if !h.is_multiple_of(self.spatial_merge_size) || !w.is_multiple_of(self.spatial_merge_size)
        {
            candle_core::bail!("HunyuanVL image grid must be divisible by spatial_merge_size");
        }
        let h_merged = h / self.spatial_merge_size;
        let w_merged = w / self.spatial_merge_size;
        let hidden = xs.dim(D::Minus1)?;
        let xs = self
            .before_rms
            .forward(xs)?
            .reshape((1, h, w, hidden))?
            .permute((0, 3, 1, 2))?;
        let xs = self.act_fn.forward(&self.proj0.forward(&xs)?)?;
        let xs = self.proj2.forward(&xs)?;
        let c = xs.dim(1)?;
        let newline = self
            .image_newline
            .reshape((1, c, 1, 1))?
            .repeat((1, 1, h_merged, 1))?;
        let xs = Tensor::cat(&[xs, newline], D::Minus1)?;
        let xs = xs
            .permute((0, 2, 3, 1))?
            .reshape((h_merged * (w_merged + 1), c))?;
        let xs = self.mlp.forward(&xs)?;
        let ys = Tensor::cat(
            &[
                self.image_begin.unsqueeze(0)?,
                xs,
                self.image_end.unsqueeze(0)?,
            ],
            0,
        )?;
        self.after_rms.forward(&ys)
    }
}

pub(crate) struct HunyuanVLVisionModel {
    embeddings: PatchEmbeddings,
    layers: Vec<VisionLayer>,
    perceive: PatchMerger,
}

impl HunyuanVLVisionModel {
    pub fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let embeddings = PatchEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| VisionLayer::new(cfg, vb.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let perceive = PatchMerger::new(cfg, vb.pp("perceive"))?;
        Ok(Self {
            embeddings,
            layers,
            perceive,
        })
    }

    pub fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(xs, grid_thw)?;

        let grid = grid_thw.to_vec2::<u32>()?;
        let mut offset = 0usize;
        let mut images = Vec::with_capacity(grid.len());
        for g in grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let len = t * h * w;
            if t != 1 {
                candle_core::bail!("HunyuanVL image path currently expects grid_t=1, got {t}");
            }
            let mut item = hidden_states.narrow(0, offset, len)?.unsqueeze(0)?;
            offset += len;
            for layer in &self.layers {
                item = layer.forward(&item)?;
            }
            images.push(self.perceive.forward_one(&item.squeeze(0)?, h, w)?);
        }
        Tensor::cat(&images, 0)
    }

    pub(crate) fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_embeddings = uvb.pp("embeddings");
        uvb_embeddings
            .pp("patch_embedding")
            .add(&self.embeddings.patch_embedding);
        uvb_embeddings
            .pp("position_embedding")
            .add(&self.embeddings.position_embedding);

        let uvb_layers = uvb.pp("layers");
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_layer = uvb_layers.pp(layer_idx);
            uvb_layer.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_layer
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_layer
                .pp("self_attn")
                .extend(layer.self_attn.residual_tensors());
            uvb_layer.pp("mlp").extend(layer.mlp.residual_tensors());
        }

        uvb.pp("perceive").extend(self.perceive.residual_tensors());

        uvb.to_safetensors()
    }
}

impl VisionAttention {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("o_proj").add(&self.o_proj);
        uvb.to_safetensors()
    }
}

impl VisionMlp {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("dense_h_to_4h").add(&self.dense_h_to_4h);
        uvb.pp("dense_4h_to_h").add(&self.dense_4h_to_h);
        uvb.to_safetensors()
    }
}

impl PatchMerger {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("before_rms").add(&self.before_rms);
        uvb.pp("after_rms").add(&self.after_rms);
        uvb.pp("proj").pp(0).add(&self.proj0);
        uvb.pp("proj").pp(2).add(&self.proj2);
        uvb.pp("mlp").add(&self.mlp);
        uvb.add_tensor("image_begin", self.image_begin.clone());
        uvb.add_tensor("image_end", self.image_end.clone());
        uvb.add_tensor("image_newline", self.image_newline.clone());
        uvb.add_tensor("image_sep", self.image_sep.clone());
        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn assert_bidirectional_attention(device: &Device, dtype: DType) -> Result<()> {
        let size = 8;
        let zero = Tensor::zeros((size, size), dtype, device)?;
        let identity = Tensor::eye(size, dtype, device)?;
        let attention = VisionAttention {
            q_proj: Linear::new(zero.clone(), None),
            k_proj: Linear::new(zero, None),
            v_proj: Linear::new(identity.clone(), None),
            o_proj: Linear::new(identity, None),
            num_heads: 1,
            head_dim: size,
        };
        let mut values = vec![0f32; 3 * size];
        values[2 * size..].fill(3.0);
        let input = Tensor::from_vec(values, (1, 3, size), device)?.to_dtype(dtype)?;
        let output = attention
            .forward(&input)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        for row in &output[0] {
            for value in row {
                assert!(
                    (value - 1.0).abs() < 0.01,
                    "each patch must see the final patch: {value}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn vision_attention_reads_later_patches() -> Result<()> {
        assert_bidirectional_attention(&Device::Cpu, DType::F32)
    }

    #[test]
    fn patch_positions_match_size_based_bilinear_interpolation() -> Result<()> {
        let device = Device::Cpu;
        let edge = 4usize;
        // Row 0 is the cls token; the 4x4 grid below holds value = row * edge + col.
        let mut values = vec![1000f32, 0.0];
        for row in 0..edge {
            for col in 0..edge {
                values.push(f32::from(u16::try_from(row * edge + col).unwrap()));
                values.push(2.0 * f32::from(u16::try_from(row * edge + col).unwrap()));
            }
        }
        let table = Tensor::from_vec(values, (edge * edge + 1, 2), &device)?;
        let positions =
            interpolate_patch_positions(&table, &[vec![1, 6, 5]], edge, 1)?.to_vec2::<f32>()?;
        assert_eq!(positions.len(), 30);

        let expected = [
            [0.0, 0.7, 1.5, 2.3, 3.0],
            [2.0, 2.7, 3.5, 4.3, 5.0],
            [4.666_667, 5.366_667, 6.166_667, 6.966_667, 7.666_667],
            [7.333_333, 8.033_334, 8.833_333, 9.633_333, 10.333_33],
            [10.0, 10.7, 11.5, 12.3, 13.0],
            [12.0, 12.7, 13.5, 14.3, 15.0],
        ];
        for (row, expected_row) in positions.chunks(5).zip(expected) {
            for (value, expected) in row.iter().zip(expected_row) {
                assert!(
                    (value[0] - expected).abs() < 1e-4,
                    "{value:?} != {expected}"
                );
                assert!((value[1] - 2.0 * expected).abs() < 1e-4, "{value:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn patch_positions_accumulate_in_f32_before_casting_back() -> Result<()> {
        let device = Device::Cpu;
        let edge = 4usize;
        let mut values = vec![-1f32];
        values.extend((0..edge * edge).map(|idx| f32::from(u16::try_from(idx).unwrap())));
        let table =
            Tensor::from_vec(values, (edge * edge + 1, 1), &device)?.to_dtype(DType::BF16)?;
        let positions = interpolate_patch_positions(&table, &[vec![1, 6, 5]], edge, 1)?;
        assert_eq!(positions.dtype(), DType::BF16);
        let positions = positions.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        // Rounding each partial sum to BF16 would land on 7.6875 instead.
        assert_eq!(positions[2 * 5 + 4][0], 7.656_25);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_bf16_vision_attention_reads_later_patches() -> Result<()> {
        assert_bidirectional_attention(&Device::new_cuda(0)?, DType::BF16)
    }
}
