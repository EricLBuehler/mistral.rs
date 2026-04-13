#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::attention::AttentionMask;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::{
    attention::{Sdpa, SdpaParams},
    layers::{Activation, RmsNorm},
    pipeline::text_models_inputs_processor::FlashParams,
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::Gemma4VisionConfig;

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

// ── Clippable Linear (Gemma4ClippableLinear equivalent) ─────────────────────

/// Linear layer with optional input/output clamping, matching HF's Gemma4ClippableLinear.
struct ClippableLinear {
    inner: Arc<dyn QuantMethod>,
    input_min: Option<f64>,
    input_max: Option<f64>,
    output_min: Option<f64>,
    output_max: Option<f64>,
    has_linear_prefix: bool,
}

impl ClippableLinear {
    fn new(in_features: usize, out_features: usize, vb: ShardedVarBuilder) -> Result<Self> {
        // Some checkpoints nest the weight under a `.linear.` sub-module,
        // others store it directly. Probe to pick the right path.
        let has_linear_prefix = vb.pp("linear").contains_tensor("weight");
        let linear_vb = if has_linear_prefix {
            vb.pp("linear")
        } else {
            vb.clone()
        };
        let inner = mistralrs_quant::linear_no_bias(in_features, out_features, &None, linear_vb)?;

        // Load clipping buffers if present
        let input_min = vb
            .get((), "input_min")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_scalar::<f32>().ok())
            .map(|v| v as f64);
        let input_max = vb
            .get((), "input_max")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_scalar::<f32>().ok())
            .map(|v| v as f64);
        let output_min = vb
            .get((), "output_min")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_scalar::<f32>().ok())
            .map(|v| v as f64);
        let output_max = vb
            .get((), "output_max")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_scalar::<f32>().ok())
            .map(|v| v as f64);

        Ok(Self {
            inner,
            input_min,
            input_max,
            output_min,
            output_max,
            has_linear_prefix,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        if let (Some(lo), Some(hi)) = (self.input_min, self.input_max) {
            x = x.clamp(lo, hi)?;
        }
        let mut out = self.inner.forward(&x)?;
        if let (Some(lo), Some(hi)) = (self.output_min, self.output_max) {
            out = out.clamp(lo, hi)?;
        }
        Ok(out)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        if self.has_linear_prefix {
            uvb.pp("linear").add(&self.inner);
        } else {
            uvb.add(&self.inner);
        }
        if let Some(v) = self.input_min {
            uvb.add_tensor(
                "input_min",
                Tensor::new(v as f32, &candle_core::Device::Cpu).unwrap(),
            );
        }
        if let Some(v) = self.input_max {
            uvb.add_tensor(
                "input_max",
                Tensor::new(v as f32, &candle_core::Device::Cpu).unwrap(),
            );
        }
        if let Some(v) = self.output_min {
            uvb.add_tensor(
                "output_min",
                Tensor::new(v as f32, &candle_core::Device::Cpu).unwrap(),
            );
        }
        if let Some(v) = self.output_max {
            uvb.add_tensor(
                "output_max",
                Tensor::new(v as f32, &candle_core::Device::Cpu).unwrap(),
            );
        }
        uvb.to_safetensors()
    }
}

// ── 2D Vision Rotary Embedding ──────────────────────────────────────────────

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
    ndim: usize,
}

impl VisionRotaryEmbedding {
    fn new(head_dim: usize, theta: f64, ndim: usize, device: &Device) -> Result<Self> {
        let dim_per_dim = head_dim / ndim;
        let half = dim_per_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta.powf(2.0 * i as f64 / dim_per_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, 1, half), device)?;
        Ok(Self { inv_freq, ndim })
    }

    fn forward(&self, _x: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        // positions: [b, num_patches, 2]
        let inv_freq = self.inv_freq.to_dtype(DType::F32)?;

        let mut emb_parts = Vec::with_capacity(self.ndim);
        for d in 0..self.ndim {
            let pos_d = positions.i((.., .., d))?.to_dtype(DType::F32)?;
            // pos_d: [b, num_patches] -> [b, num_patches, 1]
            let pos_d = pos_d.unsqueeze(D::Minus1)?;
            // freqs_d: [b, num_patches, dim_per_dim/2]
            let freqs_d = pos_d.broadcast_mul(&inv_freq)?;
            // emb_d: [b, num_patches, dim_per_dim]
            let emb_d = Tensor::cat(&[&freqs_d, &freqs_d], D::Minus1)?;
            emb_parts.push(emb_d);
        }
        // full_emb: [b, num_patches, head_dim]
        let full_emb = Tensor::cat(&emb_parts, D::Minus1)?;
        let cos = full_emb.cos()?;
        let sin = full_emb.sin()?;
        Ok((cos, sin))
    }
}

/// Apply standard rotary embedding (GPT-NeoX style: split in half).
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_2d_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, ndim: usize) -> Result<Tensor> {
    // x: [b, heads, seq, head_dim]
    // cos/sin: [b, seq, head_dim]
    let head_dim = x.dim(D::Minus1)?;
    let dim_per_dim = head_dim / ndim;

    // Expand cos/sin to [b, 1, seq, head_dim] for broadcasting
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    let mut parts = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let x_part = x.narrow(D::Minus1, d * dim_per_dim, dim_per_dim)?;
        let cos_part = cos.narrow(D::Minus1, d * dim_per_dim, dim_per_dim)?;
        let sin_part = sin.narrow(D::Minus1, d * dim_per_dim, dim_per_dim)?;
        let rotated =
            (x_part.broadcast_mul(&cos_part)? + rotate_half(&x_part)?.broadcast_mul(&sin_part)?)?;
        parts.push(rotated);
    }
    Tensor::cat(&parts, D::Minus1)
}

// ── PatchEmbedder ───────────────────────────────────────────────────────────

struct PatchEmbedder {
    input_proj: ClippableLinear,
    position_embedding_table: Tensor,
    patch_size: usize,
    position_embedding_size: usize,
}

impl PatchEmbedder {
    fn new(cfg: &Gemma4VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let ps = cfg.patch_size;
        let input_proj = ClippableLinear::new(ps * ps * 3, cfg.hidden_size, vb.pp("input_proj"))?;
        let position_embedding_table = vb.get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;
        Ok(Self {
            input_proj,
            position_embedding_table,
            patch_size: ps,
            position_embedding_size: cfg.position_embedding_size,
        })
    }

    fn forward(
        &self,
        pixel_values: &Tensor,
        patch_positions: &Tensor,
        padding_mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, c, h, w) = pixel_values.dims4()?;
        let ps = self.patch_size;
        let ph = h / ps;
        let pw = w / ps;

        // Patchify: (b, c, ph, ps, pw, ps) -> permute(0,2,4,3,5,1) -> (b, ph*pw, ps*ps*c)
        let patches = pixel_values
            .reshape((b, c, ph, ps, pw, ps))?
            .permute((0, 2, 4, 3, 5, 1))?
            .reshape((b, ph * pw, ps * ps * c))?
            .contiguous()?;

        // Scale to [-1, 1]
        let patches = ((patches - 0.5)? * 2.0)?;

        // Linear projection (with optional clipping)
        let patches = self.input_proj.forward(&patches)?;

        // Position embeddings via index_select (replaces one_hot + matmul)
        // Clamp -1 positions to 0 (padding), we'll zero them out with the mask
        let clamped_pos = patch_positions.clamp(0i64, self.position_embedding_size as i64 - 1)?;
        let n = clamped_pos.dim(1)?;
        let table_dtype = self.position_embedding_table.dtype();
        let hidden = self.position_embedding_table.dim(2)?;

        let pos_emb_0 = {
            let pos_d = clamped_pos.i((.., .., 0usize))?; // [b, n]
            let table_d = self.position_embedding_table.i(0)?; // [pos_emb_size, hidden]
            let flat_idx = pos_d.flatten_all()?.to_dtype(DType::U32)?;
            table_d
                .index_select(&flat_idx, 0)?
                .reshape((b, n, hidden))?
                .to_dtype(table_dtype)?
        };

        let pos_emb_1 = {
            let pos_d = clamped_pos.i((.., .., 1usize))?;
            let table_d = self.position_embedding_table.i(1)?;
            let flat_idx = pos_d.flatten_all()?.to_dtype(DType::U32)?;
            table_d
                .index_select(&flat_idx, 0)?
                .reshape((b, n, hidden))?
                .to_dtype(table_dtype)?
        };

        let pos_emb = (pos_emb_0 + pos_emb_1)?;

        // Zero out padding positions: padding_mask is [b, num_patches], true = padding
        let mask = padding_mask
            .unsqueeze(2)?
            .broadcast_as(pos_emb.shape())?
            .to_dtype(DType::U8)?;
        let zeros = Tensor::zeros_like(&pos_emb)?;
        let pos_emb = mask.where_cond(&zeros, &pos_emb)?;
        patches + pos_emb
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("input_proj")
            .extend(self.input_proj.residual_tensors());
        uvb.add_tensor(
            "position_embedding_table",
            self.position_embedding_table.clone(),
        );
        uvb.to_safetensors()
    }
}

// ── VisionAttention ─────────────────────────────────────────────────────────

struct VisionAttention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    o_proj: ClippableLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rms_norm_eps: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

impl VisionAttention {
    fn new(cfg: &Gemma4VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = ClippableLinear::new(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = ClippableLinear::new(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = ClippableLinear::new(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = ClippableLinear::new(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rms_norm_eps: cfg.rms_norm_eps,
            num_heads,
            num_kv_heads,
            head_dim,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                sliding_window: None,
                softcap: None,
                softmax_scale: 1.0,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: &AttentionMask,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to (b, seq, heads, head_dim)
        let mut q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK norms and V norm (RMS without learned weight)
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        let v = v_norm(&v, self.rms_norm_eps)?.transpose(1, 2)?;

        // Transpose to (b, heads, seq, head_dim) for RoPE
        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;

        // Apply 2D RoPE
        q = apply_2d_rope(&q, cos, sin, 2)?;
        k = apply_2d_rope(&k, cos, sin, 2)?;
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let attn_output = if !matches!(attention_mask, AttentionMask::None) {
            // CUDA flash-attn in the shared backend does not consume arbitrary
            // dense masks, so padded vision batches must stay on the masked path.
            Sdpa.run_attention_noflash(
                &q,
                &k,
                &v,
                attention_mask.as_option_tensor(),
                &self.sdpa_params,
            )?
        } else {
            Sdpa.run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::None,
                Some(flash_params),
                &self.sdpa_params,
            )?
        };

        // Reshape back: (b, heads, seq, head_dim) -> (b, seq, hidden)
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.o_proj.forward(&attn_output)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("q_proj").extend(self.q_proj.residual_tensors());
        uvb.pp("k_proj").extend(self.k_proj.residual_tensors());
        uvb.pp("v_proj").extend(self.v_proj.residual_tensors());
        uvb.pp("o_proj").extend(self.o_proj.residual_tensors());
        uvb.pp("q_norm").add(&self.q_norm);
        uvb.pp("k_norm").add(&self.k_norm);
        uvb.to_safetensors()
    }
}

// ── VisionMlp ───────────────────────────────────────────────────────────────

struct VisionMlp {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
    act: Activation,
}

impl VisionMlp {
    fn new(cfg: &Gemma4VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let gate_proj =
            ClippableLinear::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            ClippableLinear::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            ClippableLinear::new(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act: cfg.hidden_activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.act.forward(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("gate_proj")
            .extend(self.gate_proj.residual_tensors());
        uvb.pp("up_proj").extend(self.up_proj.residual_tensors());
        uvb.pp("down_proj")
            .extend(self.down_proj.residual_tensors());
        uvb.to_safetensors()
    }
}

// ── VisionEncoderLayer ──────────────────────────────────────────────────────

struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl VisionEncoderLayer {
    fn new(cfg: &Gemma4VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attn = VisionAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = VisionMlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: &AttentionMask,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        // Pre-norm attention with post-norm
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, cos, sin, attention_mask, flash_params)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (residual + xs)?;

        // Pre-norm MLP with post-norm
        let residual = xs.clone();
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        residual + xs
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("self_attn")
            .extend(self.self_attn.residual_tensors());
        uvb.pp("mlp").extend(self.mlp.residual_tensors());
        uvb.pp("input_layernorm").add(&self.input_layernorm);
        uvb.pp("post_attention_layernorm")
            .add(&self.post_attention_layernorm);
        uvb.pp("pre_feedforward_layernorm")
            .add(&self.pre_feedforward_layernorm);
        uvb.pp("post_feedforward_layernorm")
            .add(&self.post_feedforward_layernorm);
        uvb.to_safetensors()
    }
}

// ── VisionPooler ────────────────────────────────────────────────────────────

struct VisionPooler {
    hidden_size: usize,
    default_output_length: usize,
    pooling_kernel_size: usize,
}

impl VisionPooler {
    fn new(cfg: &Gemma4VisionConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            default_output_length: cfg.default_output_length,
            pooling_kernel_size: cfg.pooling_kernel_size,
        }
    }

    fn pooling_k(&self) -> usize {
        self.pooling_kernel_size
    }

    fn avg_pool_by_positions(
        &self,
        x: &Tensor,
        patch_positions: &Tensor,
        output_length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b, num_patches, dim) = x.dims3()?;
        let k = ((num_patches as f64 / output_length as f64).sqrt()) as i64;
        let k_sq = k * k;
        let device = x.device();

        // Clamp padding positions (-1) to 0
        let clamped = patch_positions.clamp(0i64, i64::MAX)?;
        let pos_x = clamped.i((.., .., 0usize))?.to_dtype(DType::F32)?;
        let pos_y = clamped.i((.., .., 1usize))?.to_dtype(DType::F32)?;

        // max_x per batch: [b, 1]
        let max_x = (pos_x.max_keepdim(D::Minus1)? + 1.0)?;

        // kernel indices: kx + (max_x / k).floor() * ky  →  [b, num_patches]
        let kf = k as f64;
        let kx = (pos_x / kf)?.floor()?;
        let ky = (pos_y / kf)?.floor()?;
        let stride = (max_x / kf)?.floor()?;
        let kernel_idxs = (kx + stride.broadcast_mul(&ky)?)?.to_dtype(DType::U32)?;

        // Scatter-add pooling: accumulate x / k² into output bins
        let original_dtype = x.dtype();
        let x_scaled = (x.to_dtype(DType::F32)? / k_sq as f64)?;
        let idx_expanded = kernel_idxs
            .unsqueeze(2)?
            .broadcast_as(&[b, num_patches, dim])?
            .contiguous()?;
        let output = Tensor::zeros((b, output_length, dim), DType::F32, device)?
            .scatter_add(&idx_expanded, &x_scaled, 1)?
            .to_dtype(original_dtype)?;

        // Mask: which output positions received any contributions
        let ones = Tensor::ones((b, num_patches), DType::F32, device)?;
        let weight_sum = Tensor::zeros((b, output_length), DType::F32, device)?.scatter_add(
            &kernel_idxs,
            &ones,
            1,
        )?;
        let mask = weight_sum.gt(0.0)?;

        Ok((output, mask))
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        patch_positions: &Tensor,
        padding_positions: &Tensor,
        output_length: Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let output_length = output_length.unwrap_or(self.default_output_length);
        let (pooled, mask) = if hidden_states.dim(1)? == output_length {
            let mask = padding_positions
                .to_dtype(DType::F32)?
                .eq(&Tensor::zeros_like(
                    &padding_positions.to_dtype(DType::F32)?,
                )?)?;
            (hidden_states.clone(), mask)
        } else {
            self.avg_pool_by_positions(hidden_states, patch_positions, output_length)?
        };
        let pooled = (pooled * (self.hidden_size as f64).sqrt())?;
        Ok((pooled, mask))
    }
}

// ── VisionTower ─────────────────────────────────────────────────────────────

pub struct VisionTower {
    patch_embedder: PatchEmbedder,
    encoder_layers: Vec<VisionEncoderLayer>,
    pooler: VisionPooler,
    rotary_emb: VisionRotaryEmbedding,
    std_bias: Option<Tensor>,
    std_scale: Option<Tensor>,
    patch_size: usize,
}

impl VisionTower {
    pub fn new(cfg: &Gemma4VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let patch_embedder = PatchEmbedder::new(cfg, vb.pp("patch_embedder"))?;

        let mut encoder_layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_enc = vb.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            encoder_layers.push(VisionEncoderLayer::new(cfg, vb_enc.pp(i))?);
        }

        let pooler = VisionPooler::new(cfg);

        let rotary_emb =
            VisionRotaryEmbedding::new(cfg.head_dim, cfg.rope_theta(), 2, vb.device())?;

        let (std_bias, std_scale) = if cfg.standardize {
            (
                Some(vb.get(cfg.hidden_size, "std_bias")?),
                Some(vb.get(cfg.hidden_size, "std_scale")?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            patch_embedder,
            encoder_layers,
            pooler,
            rotary_emb,
            std_bias,
            std_scale,
            patch_size: cfg.patch_size,
        })
    }

    /// Encode a single image: embed → encoder (flash attention, no padding) → pool.
    fn encode_single(&self, pv: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        let (_, _, h, w) = pv.dims4()?;
        let ph = h / self.patch_size;
        let pw = w / self.patch_size;
        let num_patches = ph * pw;

        // Position IDs for real patches only (no padding needed)
        let mut pos_data = Vec::with_capacity(num_patches * 2);
        for row in 0..ph {
            for col in 0..pw {
                pos_data.push(col as i64);
                pos_data.push(row as i64);
            }
        }
        let positions =
            Tensor::from_vec(pos_data, (1, num_patches, 2), &Device::Cpu)?.to_device(device)?;
        let no_padding = Tensor::zeros((1, num_patches), DType::U8, device)?;

        // Patch embed (no padding → no mask needed)
        let embeds = self.patch_embedder.forward(pv, &positions, &no_padding)?;

        // 2D RoPE
        let (cos, sin) = self.rotary_emb.forward(&embeds, &positions)?;
        let cos = cos.to_dtype(dtype)?;
        let sin = sin.to_dtype(dtype)?;

        // Encoder layers — no padding mask, so flash attention is used
        let flash_params = FlashParams::empty(false);
        let mut hidden_states = embeds;
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                &AttentionMask::None,
                &flash_params,
            )?;
        }

        // Pool: output_length = num_patches / k² (computed from actual patches)
        let k = self.pooler.pooling_k();
        let output_length = num_patches / (k * k);
        let (pooled, pool_mask) =
            self.pooler
                .forward(&hidden_states, &positions, &no_padding, Some(output_length))?;

        // Strip any zero-weight output positions (shouldn't happen with
        // well-aligned patches, but handle gracefully)
        let mask_u8 = pool_mask.i(0)?.to_dtype(DType::U8)?;
        let indices = mask_u8.nonzero()?.squeeze(1)?;
        if indices.dim(0)? < output_length {
            pooled.i(0)?.index_select(&indices, 0)
        } else {
            pooled.squeeze(0)
        }
    }

    pub fn forward(&self, pixel_values_list: &[Tensor]) -> Result<Tensor> {
        let device = pixel_values_list[0].device().clone();
        let dtype = pixel_values_list[0].dtype();

        // Encode each image separately at its natural patch count.
        // This avoids padding to max_patches, which would disable flash attention
        // (the dense padding mask forces the noflash path).
        let mut all_real_tokens = Vec::with_capacity(pixel_values_list.len());
        for pv in pixel_values_list {
            let tokens = self.encode_single(pv, &device, dtype)?;
            all_real_tokens.push(tokens);
        }
        let mut hidden_states = Tensor::cat(&all_real_tokens, 0)?;
        if let (Some(std_bias), Some(std_scale)) = (&self.std_bias, &self.std_scale) {
            let std_bias = std_bias
                .to_device(hidden_states.device())?
                .to_dtype(hidden_states.dtype())?;
            let std_scale = std_scale
                .to_device(hidden_states.device())?
                .to_dtype(hidden_states.dtype())?;
            hidden_states = (hidden_states.broadcast_sub(&std_bias)?).broadcast_mul(&std_scale)?;
        }
        hidden_states.unsqueeze(0)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("patch_embedder")
            .extend(self.patch_embedder.residual_tensors());

        let uvb_enc = uvb.pp("encoder");
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            uvb_enc.pp("layers").pp(i).extend(layer.residual_tensors());
        }

        if let Some(ref std_bias) = self.std_bias {
            uvb.add_tensor("std_bias", std_bias.clone());
        }
        if let Some(ref std_scale) = self.std_scale {
            uvb.add_tensor("std_scale", std_scale.clone());
        }

        uvb.to_safetensors()
    }
}
