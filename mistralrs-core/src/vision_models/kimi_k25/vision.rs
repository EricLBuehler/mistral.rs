use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    attention::SdpaParams,
    layers::{self, Sdpa},
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::VisionConfig;

// ── Patch Embedding ──

struct PatchEmbed {
    proj: Conv2d,
    pos_emb_weight: Tensor,
    init_height: usize,
    init_width: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let proj = layers::conv2d(
            3,
            cfg.vt_hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;

        // Learnable 2D positional embedding: (init_height, init_width, hidden_size)
        let pos_emb_weight = vb.get(
            (cfg.init_pos_emb_height, cfg.init_pos_emb_width, cfg.vt_hidden_size),
            "pos_emb.weight",
        )?;

        Ok(Self {
            proj,
            pos_emb_weight,
            init_height: cfg.init_pos_emb_height,
            init_width: cfg.init_pos_emb_width,
            hidden_size: cfg.vt_hidden_size,
        })
    }

    fn interpolate_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        if h == self.init_height && w == self.init_width {
            // No interpolation needed
            return self.pos_emb_weight.reshape((h * w, self.hidden_size));
        }

        // Bilinear interpolation of learned pos embed from (init_h, init_w, dim) to (h, w, dim)
        // Reshape to (1, dim, init_h, init_w) for upsample_nearest2d or manual interp
        let pos = self
            .pos_emb_weight
            .permute((2, 0, 1))?
            .unsqueeze(0)?;

        let interpolated =
            pos.upsample_nearest2d(h, w)?;

        // (1, dim, h, w) -> (h, w, dim)
        let interpolated = interpolated
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?;

        interpolated.reshape((h * w, self.hidden_size))
    }

    fn forward(&self, xs: &Tensor, grid_thws: &[(usize, usize, usize)]) -> Result<Tensor> {
        // xs: (L, 3, patch_size, patch_size) where L = total patches across all images
        let projected = self.proj.forward(xs)?;
        // (L, hidden_size, 1, 1) -> (L, hidden_size)
        let projected = projected.reshape((xs.dim(0)?, self.hidden_size))?;

        // Add positional embeddings per image
        let mut pos_embs = Vec::new();
        for &(t, h, w) in grid_thws {
            let pos_emb_2d = self.interpolate_pos_embed(h, w)?;
            if t == 1 {
                pos_embs.push(pos_emb_2d);
            } else {
                // For video: repeat spatial pos embed for each frame
                // (h*w, dim) -> (t, h*w, dim) -> (t*h*w, dim)
                let repeated = pos_emb_2d
                    .unsqueeze(0)?
                    .expand((t, h * w, self.hidden_size))?
                    .reshape((t * h * w, self.hidden_size))?;
                pos_embs.push(repeated);
            }
        }

        let all_pos_embs = Tensor::cat(&pos_embs, 0)?;
        projected.add(&all_pos_embs)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("proj").add(&self.proj);
        uvb.pp("pos_emb").add_tensor("weight", self.pos_emb_weight.clone());
        uvb.to_safetensors()
    }
}

// ── 2D Rotary Position Embedding ──

fn precompute_rope_2d(
    head_dim: usize,
    max_height: usize,
    max_width: usize,
    theta_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // head_dim must be divisible by 4 for 2D RoPE
    assert!(head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE");
    let half_dim = head_dim / 2; // = 36 for head_dim=72
    let quarter_dim = head_dim / 4; // = 18 for head_dim=72

    // Compute inverse frequencies: 1 / (theta ^ (4i / dim)) for i in 0..quarter_dim
    let inv_freq: Vec<f32> = (0..quarter_dim)
        .map(|i| 1.0 / theta_base.powf((4 * i) as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (quarter_dim,), device)?;

    // Generate position grids
    let n = max_height * max_width;
    let x_data: Vec<f32> = (0..n)
        .map(|i| (i % max_width) as f32)
        .collect();
    let y_data: Vec<f32> = (0..n)
        .map(|i| (i / max_width) as f32)
        .collect();
    let x_pos = Tensor::from_vec(x_data, (n,), device)?;
    let y_pos = Tensor::from_vec(y_data, (n,), device)?;

    // x_freqs, y_freqs: (N, quarter_dim)
    let x_freqs = x_pos.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    let y_freqs = y_pos.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    // Interleave x and y: [x0, y0, x1, y1, ...] → (N, half_dim)
    let x_freqs_exp = x_freqs.unsqueeze(2)?; // (N, quarter_dim, 1)
    let y_freqs_exp = y_freqs.unsqueeze(2)?; // (N, quarter_dim, 1)
    let interleaved = Tensor::cat(&[x_freqs_exp, y_freqs_exp], 2)?; // (N, quarter_dim, 2)
    let interleaved = interleaved.reshape((n, half_dim))?; // (N, half_dim)

    // Reshape to (max_height, max_width, half_dim)
    let interleaved = interleaved.reshape((max_height, max_width, half_dim))?;

    let cos = interleaved.cos()?;
    let sin = interleaved.sin()?;

    Ok((cos, sin))
}

fn get_rope_2d(
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    grid_thws: &[(usize, usize, usize)],
) -> Result<(Tensor, Tensor)> {
    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();

    for &(t, h, w) in grid_thws {
        // Extract the (h, w) region from precomputed cache
        let cos_hw = cos_cache.i((..h, ..w, ..))?.reshape((h * w, ()))?;
        let sin_hw = sin_cache.i((..h, ..w, ..))?.reshape((h * w, ()))?;

        if t > 1 {
            // Repeat for each temporal frame
            let cos_hw = cos_hw.unsqueeze(0)?.repeat((t, 1, 1))?.reshape((t * h * w, ()))?;
            let sin_hw = sin_hw.unsqueeze(0)?.repeat((t, 1, 1))?.reshape((t * h * w, ()))?;
            cos_parts.push(cos_hw);
            sin_parts.push(sin_hw);
        } else {
            cos_parts.push(cos_hw);
            sin_parts.push(sin_hw);
        }
    }

    let cos = Tensor::cat(&cos_parts, 0)?;
    let sin = Tensor::cat(&sin_parts, 0)?;
    Ok((cos, sin))
}

/// Apply 2D RoPE with interleaved x/y frequencies using complex multiplication.
/// q, k: (seq_len, num_heads, head_dim)
/// cos, sin: (seq_len, head_dim/2)
fn apply_rope_interleaved(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let seq = q.dim(0)?;
    let heads = q.dim(1)?;
    let hd = q.dim(2)?;
    let half_hd = hd / 2;

    // Reshape to pairs: (seq, heads, half_hd, 2)
    let q_pairs = q.reshape((seq, heads, half_hd, 2))?;
    let k_pairs = k.reshape((seq, heads, half_hd, 2))?;

    // cos, sin: (seq, half_hd) -> (seq, 1, half_hd, 1) for broadcasting
    let cos = cos.reshape((seq, 1, half_hd, 1))?;
    let sin = sin.reshape((seq, 1, half_hd, 1))?;

    let q_real = q_pairs.narrow(D::Minus1, 0, 1)?;
    let q_imag = q_pairs.narrow(D::Minus1, 1, 1)?;
    let q_out_real = q_real.broadcast_mul(&cos)?.sub(&q_imag.broadcast_mul(&sin)?)?;
    let q_out_imag = q_real.broadcast_mul(&sin)?.add(&q_imag.broadcast_mul(&cos)?)?;
    let q_out = Tensor::cat(&[q_out_real, q_out_imag], D::Minus1)?.reshape((seq, heads, hd))?;

    let k_real = k_pairs.narrow(D::Minus1, 0, 1)?;
    let k_imag = k_pairs.narrow(D::Minus1, 1, 1)?;
    let k_out_real = k_real.broadcast_mul(&cos)?.sub(&k_imag.broadcast_mul(&sin)?)?;
    let k_out_imag = k_real.broadcast_mul(&sin)?.add(&k_imag.broadcast_mul(&cos)?)?;
    let k_out = Tensor::cat(&[k_out_real, k_out_imag], D::Minus1)?.reshape((seq, heads, hd))?;

    Ok((q_out, k_out))
}

// ── Vision MLP ──

struct VisionMlp {
    fc0: Linear,
    fc1: Linear,
}

impl VisionMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let fc0 = layers::linear(hidden_size, intermediate_size, vb.pp("fc0"))?;
        let fc1 = layers::linear(intermediate_size, hidden_size, vb.pp("fc1"))?;
        Ok(Self { fc0, fc1 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc1.forward(&self.fc0.forward(xs)?.gelu_erf()?)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("fc0").add(&self.fc0);
        uvb.pp("fc1").add(&self.fc1);
        uvb.to_safetensors()
    }
}

// ── Vision Attention ──

struct VisionAttention {
    wqkv: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(hidden_size: usize, num_heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let wqkv = layers::linear(hidden_size, hidden_size * 3, vb.pp("wqkv"))?;
        let wo = layers::linear(hidden_size, hidden_size, vb.pp("wo"))?;
        Ok(Self {
            wqkv,
            wo,
            num_heads,
            head_dim,
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
        let hidden_states = self.wqkv.forward(xs)?;

        // (seq_len, 3 * num_heads * head_dim) -> (seq_len, 3, num_heads, head_dim)
        let qkv = hidden_states.reshape((seq_len, 3, self.num_heads, self.head_dim))?;
        let mut q = qkv.i((.., 0, .., ..))?.squeeze(1)?; // (seq_len, num_heads, head_dim)
        let mut k = qkv.i((.., 1, .., ..))?.squeeze(1)?;
        let mut v = qkv.i((.., 2, .., ..))?.squeeze(1)?;

        // Apply RoPE in F32
        let orig_dtype = q.dtype();
        q = q.to_dtype(DType::F32)?;
        k = k.to_dtype(DType::F32)?;
        v = v.to_dtype(DType::F32)?;

        let cos_f32 = cos.to_dtype(DType::F32)?;
        let sin_f32 = sin.to_dtype(DType::F32)?;
        let (q_rot, k_rot) = apply_rope_interleaved(&q, &k, &cos_f32, &sin_f32)?;

        // Windowed attention: process each sequence segment independently
        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;

            // (len, num_heads, head_dim) -> (num_heads, len, head_dim) -> (1, num_heads, len, head_dim)
            let q_chunk = q_rot.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k_rot.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;

            let mut chunk_out = Sdpa
                .run_attention(
                    &q_chunk.unsqueeze(0)?,
                    &k_chunk.unsqueeze(0)?,
                    &v_chunk.unsqueeze(0)?,
                    None,
                    None,
                    &SdpaParams {
                        n_kv_groups: 1,
                        sliding_window: None,
                        softcap: None,
                        softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                    },
                )?
                .squeeze(0)?
                .transpose(0, 1)?;
            chunk_out.device().synchronize()?;
            chunk_out = chunk_out.reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(chunk_out.to_dtype(orig_dtype)?);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.wo.forward(&attn_output)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("wqkv").add(&self.wqkv);
        uvb.pp("wo").add(&self.wo);
        uvb.to_safetensors()
    }
}

// ── Vision Block ──

struct VisionBlock {
    norm0: LayerNorm,
    norm1: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let norm0 = layers::layer_norm(cfg.vt_hidden_size, 1e-6, vb.pp("norm0"))?;
        let norm1 = layers::layer_norm(cfg.vt_hidden_size, 1e-6, vb.pp("norm1"))?;
        // wqkv/wo are at block level, not under an "attn" prefix
        let attn =
            VisionAttention::new(cfg.vt_hidden_size, cfg.vt_num_attention_heads, vb.clone())?;
        let mlp = VisionMlp::new(cfg.vt_hidden_size, cfg.vt_intermediate_size, vb.pp("mlp"))?;
        Ok(Self {
            norm0,
            norm1,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let hidden = self.norm0.forward(xs)?;
        let hidden = self.attn.forward(&hidden, cu_seqlens, cos, sin)?;
        let xs = (residual + hidden)?;

        let residual = xs.clone();
        let hidden = self.norm1.forward(&xs)?;
        let hidden = self.mlp.forward(&hidden)?;
        residual + hidden
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("norm0").add(&self.norm0);
        uvb.pp("norm1").add(&self.norm1);
        let attn_tensors = self.attn.residual_tensors();
        for (name, tensor) in attn_tensors {
            uvb.add_tensor(&name, tensor);
        }
        let mlp_tensors = self.mlp.residual_tensors();
        for (name, tensor) in mlp_tensors {
            uvb.pp("mlp").add_tensor(&name, tensor);
        }
        uvb.to_safetensors()
    }
}

// ── Temporal Pooling + Spatial Patch Merger ──

/// Temporal mean pooling + spatial 2×2 patch merging.
/// Returns a list of tensors, one per image, each of shape (num_merged_patches, merge_area, hidden_dim).
fn tpool_patch_merger(
    hidden_states: &Tensor,
    grid_thws: &[(usize, usize, usize)],
    merge_kernel_size: &[usize],
) -> Result<Vec<Tensor>> {
    let d_model = hidden_states.dim(D::Minus1)?;
    let kh = merge_kernel_size[0];
    let kw = merge_kernel_size[1];

    let mut outputs = Vec::new();
    let mut offset = 0;
    for &(t, h, w) in grid_thws {
        let seq_len = t * h * w;
        let seq = hidden_states.narrow(0, offset, seq_len)?;

        let new_h = h / kh;
        let new_w = w / kw;

        // Reshape: (t, h, w, d) -> (t, new_h, kh, new_w, kw, d)
        let reshaped = seq.reshape((t, h, w, d_model))?;
        let reshaped = reshaped.reshape((t, new_h, kh, new_w, kw, d_model))?;
        // Permute to (t, new_h, new_w, kh, kw, d) then mean over temporal dim (0)
        let permuted = reshaped.permute((0, 1, 3, 2, 4, 5))?;
        let pooled = permuted.mean(0)?; // (new_h, new_w, kh, kw, d)

        // Reshape to (new_h * new_w, kh * kw, d)
        let merged = pooled.reshape((new_h * new_w, kh * kw, d_model))?;
        outputs.push(merged);

        offset += seq_len;
    }

    Ok(outputs)
}

// ── Main Vision Model ──

pub struct MoonViT3D {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    final_layernorm: LayerNorm,
    cos_cache: Tensor,
    sin_cache: Tensor,
    merge_kernel_size: Vec<usize>,
}

impl MoonViT3D {
    pub fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(cfg, vb.pp("patch_embed"))?;

        let blocks_vb = vb.pp("encoder").pp("blocks");
        let mut blocks = Vec::new();
        for i in 0..cfg.vt_num_hidden_layers {
            blocks.push(VisionBlock::new(cfg, blocks_vb.pp(i))?);
        }

        let final_layernorm = layers::layer_norm(
            cfg.vt_hidden_size,
            1e-6,
            vb.pp("encoder").pp("final_layernorm"),
        )?;

        // Precompute 2D RoPE for max grid size (512×512)
        let max_size = 512;
        let (cos_cache, sin_cache) = precompute_rope_2d(
            cfg.head_dim(),
            max_size,
            max_size,
            10000.0,
            vb.device(),
        )?;

        Ok(Self {
            patch_embed,
            blocks,
            final_layernorm,
            cos_cache,
            sin_cache,
            merge_kernel_size: cfg.merge_kernel_size.clone(),
        })
    }

    /// Forward pass through the vision tower.
    ///
    /// # Arguments
    /// - `pixel_values`: (L, 3, patch_size, patch_size) - all patches flattened
    /// - `grid_thws`: list of (temporal, height, width) for each image
    ///
    /// # Returns
    /// List of tensors, one per image, each (num_merged_patches, merge_area, hidden_dim)
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thws: &[(usize, usize, usize)],
    ) -> Result<Vec<Tensor>> {
        let mut hidden_states = self.patch_embed.forward(pixel_values, grid_thws)?;

        // Build cumulative sequence lengths for windowed attention
        let mut cu_seqlens = vec![0usize];
        for &(t, h, w) in grid_thws {
            let last = *cu_seqlens.last().unwrap();
            cu_seqlens.push(last + t * h * w);
        }

        // Get 2D RoPE for the current grid dimensions
        let (cos, sin) = get_rope_2d(&self.cos_cache, &self.sin_cache, grid_thws)?;

        // Process through transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
        }

        hidden_states = self.final_layernorm.forward(&hidden_states)?;

        // Apply temporal pooling + spatial patch merging
        tpool_patch_merger(&hidden_states, grid_thws, &self.merge_kernel_size)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let pe_tensors = self.patch_embed.residual_tensors();
        for (name, tensor) in pe_tensors {
            uvb.pp("patch_embed").add_tensor(&name, tensor);
        }
        for (i, block) in self.blocks.iter().enumerate() {
            let block_tensors = block.residual_tensors();
            for (name, tensor) in block_tensors {
                uvb.pp("encoder")
                    .pp("blocks")
                    .pp(i)
                    .add_tensor(&name, tensor);
            }
        }
        uvb.pp("encoder").pp("final_layernorm").add(&self.final_layernorm);
        uvb.to_safetensors()
    }
}
