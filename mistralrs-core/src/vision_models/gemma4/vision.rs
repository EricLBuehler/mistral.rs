#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::{
    attention::{Sdpa, SdpaParams},
    layers::{Activation, GemmaRmsNorm},
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
        let inner =
            mistralrs_quant::linear_no_bias(in_features, out_features, &None, linear_vb)?;

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
        let mut out = self.inner.forward_autocast(&x)?;
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

        // Position embeddings via one-hot matmul
        // Clamp -1 positions to 0 (padding), we'll zero them out with the mask
        let clamped_pos = patch_positions.clamp(0i64, self.position_embedding_size as i64 - 1)?;

        let pos_emb_0 = {
            let pos_d = clamped_pos.i((.., .., 0usize))?;
            let one_hot_d =
                candle_nn::encoding::one_hot(pos_d, self.position_embedding_size, 1f32, 0f32)?
                    .to_dtype(self.position_embedding_table.dtype())?;
            let emb_table_d = self.position_embedding_table.i(0)?.unsqueeze(0)?;
            one_hot_d.matmul(&emb_table_d)?
        };

        let pos_emb_1 = {
            let pos_d = clamped_pos.i((.., .., 1usize))?;
            let one_hot_d =
                candle_nn::encoding::one_hot(pos_d, self.position_embedding_size, 1f32, 0f32)?
                    .to_dtype(self.position_embedding_table.dtype())?;
            let emb_table_d = self.position_embedding_table.i(1)?.unsqueeze(0)?;
            one_hot_d.matmul(&emb_table_d)?
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
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
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

        let q_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

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
        attention_mask: Option<&Tensor>,
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

        let attn_output = if attention_mask.is_some() {
            // CUDA flash-attn in the shared backend does not consume arbitrary
            // dense masks, so padded vision batches must stay on the masked path.
            Sdpa.run_attention_noflash(&q, &k, &v, attention_mask, &self.sdpa_params)?
        } else {
            Sdpa.run_attention(&q, &k, &v, None, Some(flash_params), &self.sdpa_params)?
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
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl VisionEncoderLayer {
    fn new(cfg: &Gemma4VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attn = VisionAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = VisionMlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            GemmaRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = GemmaRmsNorm::new(
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
        attention_mask: Option<&Tensor>,
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
}

impl VisionPooler {
    fn new(cfg: &Gemma4VisionConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            default_output_length: cfg.default_output_length,
        }
    }

    fn avg_pool_by_positions(
        &self,
        x: &Tensor,
        patch_positions: &Tensor,
        output_length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b, num_patches, _dim) = x.dims3()?;
        let k = ((num_patches as f64 / output_length as f64).sqrt()) as i64;
        let k_sq = k * k;

        // Clamp padding positions (-1) to 0
        let clamped = patch_positions.clamp(0i64, i64::MAX)?;
        let pos_x = clamped.i((.., .., 0usize))?.to_dtype(DType::F32)?;
        let pos_y = clamped.i((.., .., 1usize))?.to_dtype(DType::F32)?;

        // max_x per batch: [b, 1]
        let max_x = (pos_x.max_keepdim(D::Minus1)? + 1.0)?;

        // kernel indices: kx + (max_x / k).floor() * ky
        let kf = k as f64;
        let kx = (pos_x / kf)?.floor()?;
        let ky = (pos_y / kf)?.floor()?;
        let stride = (max_x / kf)?.floor()?;
        let kernel_idxs = (kx + stride.broadcast_mul(&ky)?)?;

        // Build one-hot weights: [b, num_patches, output_length]
        let kernel_idxs = kernel_idxs.to_dtype(DType::I64)?;
        let weights = candle_nn::encoding::one_hot(kernel_idxs, output_length, 1f32, 0f32)?
            .to_dtype(DType::F32)?;
        let weights = (weights / k_sq as f64)?;

        // output = weights^T @ x: [b, output_length, dim]
        let output = weights.transpose(1, 2)?.to_dtype(x.dtype())?.matmul(x)?;

        // mask: True where any weight is nonzero for that output position
        let weight_sum = weights.sum(1)?;
        let zeros = Tensor::zeros_like(&weight_sum)?;
        let mask = weight_sum.ne(&zeros)?;

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
    max_patches: usize,
    patch_size: usize,
    hidden_size: usize,
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

        let max_patches =
            cfg.default_output_length * cfg.pooling_kernel_size * cfg.pooling_kernel_size;

        Ok(Self {
            patch_embedder,
            encoder_layers,
            pooler,
            rotary_emb,
            max_patches,
            patch_size: cfg.patch_size,
            hidden_size: cfg.hidden_size,
        })
    }

    fn patch_positions(&self, pixel_values: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
        let (b, _, h, w) = pixel_values.dims4()?;
        let ph = h / self.patch_size;
        let pw = w / self.patch_size;
        let num_patches = ph * pw;
        let num_padding = self.max_patches - num_patches;

        // Create grid positions on CPU then move to device
        let mut pos_data = Vec::with_capacity(b * num_patches * 2);
        for _batch in 0..b {
            for row in 0..ph {
                for col in 0..pw {
                    pos_data.push(col as i64);
                    pos_data.push(row as i64);
                }
            }
        }
        let real_positions =
            Tensor::from_vec(pos_data, (b, num_patches, 2), &Device::Cpu)?.to_device(device)?;

        let positions = if num_padding > 0 {
            let pad_positions = Tensor::full(-1i64, (b, num_padding, 2), device)?;
            Tensor::cat(&[&real_positions, &pad_positions], 1)?
        } else {
            real_positions
        };

        // Padding mask: true (1) for padding positions, false (0) for valid
        let mut padding_data = vec![0u8; b * self.max_patches];
        for batch_idx in 0..b {
            for i in num_patches..self.max_patches {
                padding_data[batch_idx * self.max_patches + i] = 1;
            }
        }
        let padding = Tensor::from_vec(padding_data, (b, self.max_patches), &Device::Cpu)?
            .to_device(device)?;

        Ok((positions, padding))
    }

    fn zero_padded_hidden_states(
        &self,
        hidden_states: &Tensor,
        padding_positions: &Tensor,
    ) -> Result<Tensor> {
        let mask = padding_positions
            .unsqueeze(2)?
            .broadcast_as(hidden_states.shape())?
            .to_dtype(DType::U8)?;
        let zeros = Tensor::zeros_like(hidden_states)?;
        mask.where_cond(&zeros, hidden_states)
    }

    pub fn forward(&self, pixel_values_list: &[Tensor]) -> Result<Tensor> {
        let device = pixel_values_list[0].device().clone();
        let dtype = pixel_values_list[0].dtype();

        let mut all_embeds = Vec::with_capacity(pixel_values_list.len());
        let mut all_positions = Vec::with_capacity(pixel_values_list.len());
        let mut all_padding = Vec::with_capacity(pixel_values_list.len());

        for pv in pixel_values_list {
            let (_, _, h, w) = pv.dims4()?;
            let ph = h / self.patch_size;
            let pw = w / self.patch_size;
            let num_patches = ph * pw;
            let Some(num_padding) = self.max_patches.checked_sub(num_patches) else {
                candle_core::bail!(
                    "Gemma4 vision input exceeds max patches: {num_patches} > {} (h={h}, w={w}, patch_size={})",
                    self.max_patches,
                    self.patch_size
                );
            };

            let (positions, padding) = self.patch_positions(pv, &device)?;

            // Embed real patches only (no padding through patch embedder)
            let real_positions = positions.narrow(1, 0, num_patches)?;
            let real_padding = padding.narrow(1, 0, num_patches)?;
            let embeds = self
                .patch_embedder
                .forward(pv, &real_positions, &real_padding)?;

            // Pad embeddings to max_patches with zeros (matching HF behavior).
            // The pooler math requires input_seq_len == max_patches.
            let embeds = if num_padding > 0 {
                let pad_embeds =
                    Tensor::zeros((1, num_padding, self.hidden_size), embeds.dtype(), &device)?;
                Tensor::cat(&[&embeds, &pad_embeds], 1)?
            } else {
                embeds
            };

            // Use FULL positions and padding (including -1 padding positions)
            all_embeds.push(embeds);
            all_positions.push(positions);
            all_padding.push(padding);
        }

        let inputs_embeds = Tensor::cat(&all_embeds, 0)?;
        let patch_positions = Tensor::cat(&all_positions, 0)?;
        let padding_positions = Tensor::cat(&all_padding, 0)?;

        let has_padding = all_padding.iter().any(|padding| {
            padding
                .sum_all()
                .and_then(|t| t.to_scalar::<u8>())
                .map(|sum| sum != 0)
                .unwrap_or(false)
        });

        // Build attention mask only when padding is present. The shared CUDA flash-attn
        // path cannot consume this dense padding mask, so masked batches run noflash.
        let attention_mask = if has_padding {
            let valid_mask = padding_positions
                .to_dtype(DType::F32)?
                .eq(0.0)?
                .to_dtype(DType::F32)?; // [b, max_patches]: 1=valid, 0=padding
            let attn_mask_2d = valid_mask.unsqueeze(2)?.matmul(&valid_mask.unsqueeze(1)?)?; // [b, mp, mp]: 1=attend, 0=don't
            let attend = attn_mask_2d.gt(0.0)?.to_dtype(DType::U8)?;
            let zeros = Tensor::zeros(attn_mask_2d.shape(), dtype, &device)?;
            let large_neg = Tensor::try_from(-1e9f32)?
                .to_dtype(dtype)?
                .to_device(&device)?
                .broadcast_as(attn_mask_2d.shape())?;
            Some(attend.where_cond(&zeros, &large_neg)?.unsqueeze(1)?)
        } else {
            None
        };

        // Compute 2D RoPE for all positions (padding positions at -1 are fine
        // since their hidden states are zero, so RoPE values don't matter)
        let (cos, sin) = self.rotary_emb.forward(&inputs_embeds, &patch_positions)?;
        let cos = cos.to_dtype(dtype)?;
        let sin = sin.to_dtype(dtype)?;

        // Bidirectional flash params
        let flash_params = FlashParams::empty(false);

        // Encoder layers
        let mut hidden_states = inputs_embeds;
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                attention_mask.as_ref(),
                &flash_params,
            )?;
            if has_padding {
                hidden_states =
                    self.zero_padded_hidden_states(&hidden_states, &padding_positions)?;
            }
        }

        // Pool with full max_patches (k = sqrt(max_patches / output_length) = 3)
        let (pooled, pool_mask) =
            self.pooler
                .forward(&hidden_states, &patch_positions, &padding_positions, None)?;

        // Strip padding tokens, keep only valid (non-padding) tokens
        let batch = pooled.dim(0)?;
        let mut all_real_tokens = Vec::with_capacity(batch);
        for b in 0..batch {
            let hs = pooled.i(b)?;
            let mask = pool_mask.i(b)?;
            let indices = mask.to_dtype(DType::U8)?.nonzero()?.squeeze(1)?;
            let real_tokens = hs.index_select(&indices, 0)?;
            all_real_tokens.push(real_tokens);
        }
        Ok(Tensor::cat(&all_real_tokens, 0)?.unsqueeze(0)?)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("patch_embedder")
            .extend(self.patch_embedder.residual_tensors());

        let uvb_enc = uvb.pp("encoder");
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            uvb_enc.pp("layers").pp(i).extend(layer.residual_tensors());
        }

        uvb.to_safetensors()
    }
}
