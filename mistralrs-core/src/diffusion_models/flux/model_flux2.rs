//! FLUX.2 Transformer implementation (Flux2Transformer2DModel)
//!
//! Weight paths match the BFL-format `flux-2-klein-9b.safetensors`:
//! - img_in, txt_in, time_in.in_layer/out_layer
//! - double_blocks.*.img_attn.qkv, img_attn.proj, img_attn.norm.query_norm.scale, key_norm.scale
//! - double_blocks.*.img_mlp.0, img_mlp.2
//! - double_blocks.*.txt_attn.qkv, txt_attn.proj, txt_attn.norm.query_norm.scale, key_norm.scale
//! - double_blocks.*.txt_mlp.0, txt_mlp.2
//! - single_blocks.*.linear1, linear2, norm.query_norm.scale, key_norm.scale
//! - double_stream_modulation_img.lin, double_stream_modulation_txt.lin
//! - single_stream_modulation.lin
//! - final_layer.adaLN_modulation.1, final_layer.linear

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm};
use mistralrs_quant::ShardedVarBuilder;

use crate::attention::{Sdpa, SdpaParams};
use crate::layers;

use super::common;

// Re-export Config from model.rs since it's compatible
pub use super::model::Config;

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    layers::linear_no_bias(in_dim, out_dim, vb)
}

/// Creates a LayerNorm without learnable parameters (elementwise_affine=False).
fn layer_norm_no_affine(dim: usize, dtype: DType, device: &Device) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, dtype, device)?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let half = n_embd / 2;
    let x = x.contiguous()?.reshape((b_sz, n_head, seq_len, half, 2))?;
    let freq_dims = freq_cis.dims();
    if freq_dims.len() < 3 {
        candle_core::bail!("unexpected rope rank {} for freq_cis", freq_dims.len())
    }
    let rope_half = freq_dims[freq_dims.len() - 3];
    if rope_half > half {
        candle_core::bail!("rope dim {rope_half} exceeds head dim {half} (seq_len={seq_len})")
    }

    // Apply RoPE to the first rope_half dims, passthrough the remainder.
    let rot = x.narrow(D::Minus2, 0, rope_half)?;
    let pass = if rope_half < half {
        Some(x.narrow(D::Minus2, rope_half, half - rope_half)?)
    } else {
        None
    };

    let x0 = rot.narrow(D::Minus1, 0, 1)?;
    let x1 = rot.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    let rot = (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?;
    let x = if let Some(pass) = pass {
        Tensor::cat(&[&rot, &pass], D::Minus2)?
    } else {
        rot
    };
    x.reshape(dims.to_vec())
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let v = v.contiguous()?;

    let head_dim = q.dim(D::Minus1)?;
    let sdpa_params = SdpaParams {
        n_kv_groups: 1,
        softcap: None,
        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        sliding_window: None,
        sinks: None,
    };
    let x = Sdpa.run_attention(
        &q,
        &k,
        &v,
        None,
        Some(&common::DIFFUSION_FLASH_PARAMS),
        &sdpa_params,
    )?;
    x.transpose(1, 2)?.flatten_from(2)
}

#[derive(Debug, Clone)]
pub struct EmbedNd {
    inv_freqs: Vec<Tensor>,
}

impl EmbedNd {
    fn new(theta: usize, axes_dim: Vec<usize>, device: &Device) -> Result<Self> {
        let inv_freqs = axes_dim
            .iter()
            .map(|&dim| common::precompute_inv_freq(dim, theta, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { inv_freqs })
    }
}

impl candle_core::Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            let r =
                common::rope_with_inv_freq(&ids.get_on_dim(D::Minus1, idx)?, &self.inv_freqs[idx])?;
            emb.push(r)
        }
        let emb = Tensor::cat(&emb, 2)?;
        emb.unsqueeze(1)
    }
}

/// MLP Embedder: time_in.in_layer, time_in.out_layer (BFL format)
#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let in_layer = linear_no_bias(in_sz, h_sz, vb.pp("in_layer"))?;
        let out_layer = linear_no_bias(h_sz, h_sz, vb.pp("out_layer"))?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }
}

impl candle_core::Module for MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

/// Shared modulation: double_stream_modulation_img.lin, etc. (BFL uses "lin")
#[derive(Debug, Clone)]
pub struct Flux2Modulation {
    lin: Linear,
}

impl Flux2Modulation {
    fn new(dim: usize, mod_param_sets: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let lin = linear_no_bias(dim, dim * 3 * mod_param_sets, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, temb: &Tensor) -> Result<Tensor> {
        let mod_out = temb.silu()?.apply(&self.lin)?;
        if mod_out.rank() == 2 {
            mod_out.unsqueeze(1)
        } else {
            Ok(mod_out)
        }
    }
}

/// QK Norm: norm.query_norm.scale, norm.key_norm.scale
#[derive(Debug, Clone)]
pub struct QkNorm {
    query_norm: RmsNorm,
    key_norm: RmsNorm,
}

impl QkNorm {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let query_norm = vb.get(dim, "query_norm.scale")?;
        let query_norm = RmsNorm::new(query_norm, 1e-6);
        let key_norm = vb.get(dim, "key_norm.scale")?;
        let key_norm = RmsNorm::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
        })
    }
}

/// SwiGLU MLP: img_mlp.0, img_mlp.2 (BFL format uses indices)
#[derive(Debug, Clone)]
struct Flux2Mlp {
    gate_up: Linear, // Index 0: combined gate and up projection
    down: Linear,    // Index 2: down projection
    hidden_dim: usize,
}

impl Flux2Mlp {
    fn new(in_sz: usize, hidden_sz: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let gate_up = linear_no_bias(in_sz, hidden_sz * 2, vb.pp("0"))?;
        let down = linear_no_bias(hidden_sz, in_sz, vb.pp("2"))?;
        Ok(Self {
            gate_up,
            down,
            hidden_dim: hidden_sz,
        })
    }
}

impl Flux2Mlp {
    fn cast_to(&mut self, device: &Device) -> Result<()> {
        self.gate_up = Linear::new(
            self.gate_up.weight().to_device(device)?,
            self.gate_up.bias().map(|x| x.to_device(device).unwrap()),
        );
        self.down = Linear::new(
            self.down.weight().to_device(device)?,
            self.down.bias().map(|x| x.to_device(device).unwrap()),
        );
        Ok(())
    }
}

impl candle_core::Module for Flux2Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = xs.apply(&self.gate_up)?;
        let gate = gate_up.narrow(D::Minus1, 0, self.hidden_dim)?;
        let up = gate_up.narrow(D::Minus1, self.hidden_dim, self.hidden_dim)?;
        let hidden = (gate.silu()? * up)?;
        hidden.apply(&self.down)
    }
}

/// Self-attention with fused QKV: img_attn.qkv, img_attn.proj, img_attn.norm.*
#[derive(Debug, Clone)]
pub struct Flux2SelfAttention {
    qkv: Linear,
    norm: QkNorm,
    proj: Linear,
    num_attention_heads: usize,
}

impl Flux2SelfAttention {
    fn new(dim: usize, num_attention_heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let head_dim = dim / num_attention_heads;
        let qkv = linear_no_bias(dim, dim * 3, vb.pp("qkv"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = linear_no_bias(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            norm,
            proj,
            num_attention_heads,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_attention_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }

    fn cast_to(&mut self, device: &Device) -> Result<()> {
        self.qkv = Linear::new(
            self.qkv.weight().to_device(device)?,
            self.qkv.bias().map(|x| x.to_device(device).unwrap()),
        );
        self.proj = Linear::new(
            self.proj.weight().to_device(device)?,
            self.proj.bias().map(|x| x.to_device(device).unwrap()),
        );
        self.norm = QkNorm {
            query_norm: RmsNorm::new(
                self.norm
                    .query_norm
                    .clone()
                    .into_inner()
                    .weight()
                    .to_device(device)?,
                1e-6,
            ),
            key_norm: RmsNorm::new(
                self.norm
                    .key_norm
                    .clone()
                    .into_inner()
                    .weight()
                    .to_device(device)?,
                1e-6,
            ),
        };
        Ok(())
    }
}

/// FLUX.2 Double Stream Block (double_blocks.*)
#[derive(Debug, Clone)]
pub struct Flux2DoubleStreamBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm1_context: LayerNorm,
    norm2_context: LayerNorm,
    img_attn: Flux2SelfAttention,
    img_mlp: Flux2Mlp,
    txt_attn: Flux2SelfAttention,
    txt_mlp: Flux2Mlp,
    h_sz: usize,
}

impl Flux2DoubleStreamBlock {
    fn new(cfg: &Config, vb: ShardedVarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let h_sz = cfg.hidden_size();
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;

        let norm1 = layer_norm_no_affine(h_sz, dtype, device)?;
        let norm2 = layer_norm_no_affine(h_sz, dtype, device)?;
        let norm1_context = layer_norm_no_affine(h_sz, dtype, device)?;
        let norm2_context = layer_norm_no_affine(h_sz, dtype, device)?;

        let img_attn = Flux2SelfAttention::new(h_sz, cfg.num_attention_heads, vb.pp("img_attn"))?;
        let img_mlp = Flux2Mlp::new(h_sz, mlp_sz, vb.pp("img_mlp"))?;
        let txt_attn = Flux2SelfAttention::new(h_sz, cfg.num_attention_heads, vb.pp("txt_attn"))?;
        let txt_mlp = Flux2Mlp::new(h_sz, mlp_sz, vb.pp("txt_mlp"))?;

        Ok(Self {
            norm1,
            norm2,
            norm1_context,
            norm2_context,
            img_attn,
            img_mlp,
            txt_attn,
            txt_mlp,
            h_sz,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        pe: &Tensor,
        mod_img: &Tensor,
        mod_txt: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let h_sz = self.h_sz;

        // Parse modulation: (shift, scale, gate) x 2 for MSA and MLP
        let img_shift_msa = mod_img.narrow(D::Minus1, 0, h_sz)?;
        let img_scale_msa = mod_img.narrow(D::Minus1, h_sz, h_sz)?;
        let img_gate_msa = mod_img.narrow(D::Minus1, h_sz * 2, h_sz)?;
        let img_shift_mlp = mod_img.narrow(D::Minus1, h_sz * 3, h_sz)?;
        let img_scale_mlp = mod_img.narrow(D::Minus1, h_sz * 4, h_sz)?;
        let img_gate_mlp = mod_img.narrow(D::Minus1, h_sz * 5, h_sz)?;

        let txt_shift_msa = mod_txt.narrow(D::Minus1, 0, h_sz)?;
        let txt_scale_msa = mod_txt.narrow(D::Minus1, h_sz, h_sz)?;
        let txt_gate_msa = mod_txt.narrow(D::Minus1, h_sz * 2, h_sz)?;
        let txt_shift_mlp = mod_txt.narrow(D::Minus1, h_sz * 3, h_sz)?;
        let txt_scale_mlp = mod_txt.narrow(D::Minus1, h_sz * 4, h_sz)?;
        let txt_gate_mlp = mod_txt.narrow(D::Minus1, h_sz * 5, h_sz)?;

        // Norm + modulate image
        let img_normed = img.apply(&self.norm1)?;
        let img_mod = img_normed
            .broadcast_mul(&(img_scale_msa + 1.0)?)?
            .broadcast_add(&img_shift_msa)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_mod)?;

        // Norm + modulate text
        let txt_normed = txt.apply(&self.norm1_context)?;
        let txt_mod = txt_normed
            .broadcast_mul(&(txt_scale_msa + 1.0)?)?
            .broadcast_add(&txt_shift_msa)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_mod)?;

        // Joint attention
        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn_out = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn_out = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        // Residual + gate for image attention
        let img_attn_proj = img_attn_out.apply(&self.img_attn.proj)?;
        let img = (img + img_gate_msa.broadcast_mul(&img_attn_proj)?)?;

        // Image MLP
        let img_normed2 = img.apply(&self.norm2)?;
        let img_mod2 = img_normed2
            .broadcast_mul(&(img_scale_mlp + 1.0)?)?
            .broadcast_add(&img_shift_mlp)?;
        let img = (img + img_gate_mlp.broadcast_mul(&self.img_mlp.forward(&img_mod2)?)?)?;

        // Residual + gate for text attention
        let txt_attn_proj = txt_attn_out.apply(&self.txt_attn.proj)?;
        let txt = (txt + txt_gate_msa.broadcast_mul(&txt_attn_proj)?)?;

        // Text MLP
        let txt_normed2 = txt.apply(&self.norm2_context)?;
        let txt_mod2 = txt_normed2
            .broadcast_mul(&(txt_scale_mlp + 1.0)?)?
            .broadcast_add(&txt_shift_mlp)?;
        let txt = (txt + txt_gate_mlp.broadcast_mul(&self.txt_mlp.forward(&txt_mod2)?)?)?;

        Ok((img, txt))
    }

    fn cast_to(&mut self, device: &Device) -> Result<()> {
        self.norm1 = LayerNorm::new_no_bias(self.norm1.weight().to_device(device)?, 1e-6);
        self.norm2 = LayerNorm::new_no_bias(self.norm2.weight().to_device(device)?, 1e-6);
        self.norm1_context =
            LayerNorm::new_no_bias(self.norm1_context.weight().to_device(device)?, 1e-6);
        self.norm2_context =
            LayerNorm::new_no_bias(self.norm2_context.weight().to_device(device)?, 1e-6);
        self.img_attn.cast_to(device)?;
        self.img_mlp.cast_to(device)?;
        self.txt_attn.cast_to(device)?;
        self.txt_mlp.cast_to(device)?;
        Ok(())
    }
}

/// FLUX.2 Single Stream Block (single_blocks.*)
#[derive(Debug, Clone)]
pub struct Flux2SingleStreamBlock {
    pre_norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    norm: QkNorm,
    h_sz: usize,
    mlp_sz: usize,
    num_attention_heads: usize,
}

impl Flux2SingleStreamBlock {
    fn new(cfg: &Config, vb: ShardedVarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let h_sz = cfg.hidden_size();
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_attention_heads;

        let pre_norm = layer_norm_no_affine(h_sz, dtype, device)?;

        // linear1: projects to QKV (3*h_sz) + gate_up for MLP (2*mlp_sz)
        let linear1 = linear_no_bias(h_sz, h_sz * 3 + mlp_sz * 2, vb.pp("linear1"))?;
        // linear2: projects attn out (h_sz) + mlp out (mlp_sz) back to h_sz
        let linear2 = linear_no_bias(h_sz + mlp_sz, h_sz, vb.pp("linear2"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;

        Ok(Self {
            pre_norm,
            linear1,
            linear2,
            norm,
            h_sz,
            mlp_sz,
            num_attention_heads: cfg.num_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor, pe: &Tensor, mod_params: &Tensor) -> Result<Tensor> {
        let h_sz = self.h_sz;
        let (b, seq_len, _) = xs.dims3()?;

        // Parse modulation
        let shift = mod_params.narrow(D::Minus1, 0, h_sz)?;
        let scale = mod_params.narrow(D::Minus1, h_sz, h_sz)?;
        let gate = mod_params.narrow(D::Minus1, h_sz * 2, h_sz)?;

        // Norm + modulate
        let normed = xs.apply(&self.pre_norm)?;
        let modulated = normed
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;

        let proj = modulated.apply(&self.linear1)?;

        // Split into QKV and MLP
        let qkv = proj.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let mlp_in = proj.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz * 2)?;

        // Process QKV
        let qkv = qkv.reshape((b, seq_len, 3, self.num_attention_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        let attn = attention(&q, &k, &v, pe)?;

        // SwiGLU MLP
        let mlp_gate = mlp_in.narrow(D::Minus1, 0, self.mlp_sz)?;
        let mlp_up = mlp_in.narrow(D::Minus1, self.mlp_sz, self.mlp_sz)?;
        let mlp_out = (mlp_gate.silu()? * mlp_up)?;

        // Concatenate and project
        let combined = Tensor::cat(&[attn, mlp_out], D::Minus1)?;
        let output = combined.apply(&self.linear2)?;

        // Residual + gate
        xs + gate.broadcast_mul(&output)?
    }

    fn cast_to(&mut self, device: &Device) -> Result<()> {
        self.pre_norm = LayerNorm::new_no_bias(self.pre_norm.weight().to_device(device)?, 1e-6);
        self.linear1 = Linear::new(
            self.linear1.weight().to_device(device)?,
            self.linear1.bias().map(|x| x.to_device(device).unwrap()),
        );
        self.linear2 = Linear::new(
            self.linear2.weight().to_device(device)?,
            self.linear2.bias().map(|x| x.to_device(device).unwrap()),
        );
        self.norm = QkNorm {
            query_norm: RmsNorm::new(
                self.norm
                    .query_norm
                    .clone()
                    .into_inner()
                    .weight()
                    .to_device(device)?,
                1e-6,
            ),
            key_norm: RmsNorm::new(
                self.norm
                    .key_norm
                    .clone()
                    .into_inner()
                    .weight()
                    .to_device(device)?,
                1e-6,
            ),
        };
        Ok(())
    }
}

/// Final layer: final_layer.adaLN_modulation.1, final_layer.linear
#[derive(Debug, Clone)]
pub struct Flux2LastLayer {
    norm_final: LayerNorm,
    ada_ln_modulation: Linear,
    linear: Linear,
}

impl Flux2LastLayer {
    fn new(
        h_sz: usize,
        out_c: usize,
        vb: ShardedVarBuilder,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let norm_final = layer_norm_no_affine(h_sz, dtype, device)?;
        // adaLN_modulation.1 - produces shift and scale (index 1 because SiLU is at 0)
        let ada_ln_modulation = linear_no_bias(h_sz, 2 * h_sz, vb.pp("adaLN_modulation").pp("1"))?;
        let linear = linear_no_bias(h_sz, out_c, vb.pp("linear"))?;
        Ok(Self {
            norm_final,
            ada_ln_modulation,
            linear,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec
            .silu()?
            .apply(&self.ada_ln_modulation)?
            .chunk(2, D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);

        let xs = xs.apply(&self.norm_final)?;
        let xs = xs
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        xs.apply(&self.linear)
    }
}

/// FLUX.2 Transformer Model (BFL format weight paths)
#[derive(Debug, Clone)]
pub struct Flux2 {
    img_in: Linear,
    txt_in: Linear,
    time_in: MlpEmbedder,
    pe_embedder: EmbedNd,
    timestep_freqs: common::TimestepFreqs,
    double_stream_modulation_img: Flux2Modulation,
    double_stream_modulation_txt: Flux2Modulation,
    single_stream_modulation: Flux2Modulation,
    double_blocks: Vec<Flux2DoubleStreamBlock>,
    single_blocks: Vec<Flux2SingleStreamBlock>,
    final_layer: Flux2LastLayer,
    device: Device,
    offloaded: bool,
}

impl Flux2 {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        device: Device,
        dtype: DType,
        offloaded: bool,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size();

        let img_in = linear_no_bias(
            cfg.in_channels,
            hidden_size,
            vb.pp("img_in").set_device(device.clone()),
        )?;
        let txt_in = linear_no_bias(
            cfg.joint_attention_dim,
            hidden_size,
            vb.pp("txt_in").set_device(device.clone()),
        )?;

        let time_in = MlpEmbedder::new(
            256,
            hidden_size,
            vb.pp("time_in").set_device(device.clone()),
        )?;

        let double_stream_modulation_img = Flux2Modulation::new(
            hidden_size,
            2,
            vb.pp("double_stream_modulation_img")
                .set_device(device.clone()),
        )?;
        let double_stream_modulation_txt = Flux2Modulation::new(
            hidden_size,
            2,
            vb.pp("double_stream_modulation_txt")
                .set_device(device.clone()),
        )?;
        let single_stream_modulation = Flux2Modulation::new(
            hidden_size,
            1,
            vb.pp("single_stream_modulation").set_device(device.clone()),
        )?;

        let mut double_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_d = vb.pp("double_blocks");
        for idx in 0..cfg.num_layers {
            let db = Flux2DoubleStreamBlock::new(cfg, vb_d.pp(idx), dtype, &device)?;
            double_blocks.push(db)
        }

        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        let vb_s = vb.pp("single_blocks");
        for idx in 0..cfg.num_single_layers {
            let sb = Flux2SingleStreamBlock::new(cfg, vb_s.pp(idx), dtype, &device)?;
            single_blocks.push(sb)
        }

        let final_layer = Flux2LastLayer::new(
            hidden_size,
            cfg.in_channels,
            vb.pp("final_layer").set_device(device.clone()),
            dtype,
            &device,
        )?;

        let pe_embedder = EmbedNd::new(cfg.rope_theta, cfg.axes_dims_rope.clone(), &device)?;
        let timestep_freqs = common::TimestepFreqs::new(256, &device)?;

        Ok(Self {
            img_in,
            txt_in,
            time_in,
            pe_embedder,
            timestep_freqs,
            double_stream_modulation_img,
            double_stream_modulation_txt,
            single_stream_modulation,
            double_blocks,
            single_blocks,
            final_layer,
            device,
            offloaded,
        })
    }

    /// Precompute positional embeddings from txt_ids and img_ids.
    /// Call once before the denoising loop; pass the result to each forward() call.
    pub fn compute_pe(&self, txt_ids: &Tensor, img_ids: &Tensor) -> Result<Tensor> {
        let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
        ids.apply(&self.pe_embedder)
    }

    pub fn forward(
        &mut self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        pe: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 {
            candle_core::bail!("unexpected shape for txt {:?}", txt.shape())
        }
        if img.rank() != 3 {
            candle_core::bail!("unexpected shape for img {:?}", img.shape())
        }

        let dtype = img.dtype();
        let pe = match pe {
            Some(pe) => pe.clone(),
            None => self.compute_pe(txt_ids, img_ids)?,
        };

        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;

        let vec_ = self
            .timestep_freqs
            .embed(timesteps, dtype)?
            .apply(&self.time_in)?;

        let double_mod_img = self.double_stream_modulation_img.forward(&vec_)?;
        let double_mod_txt = self.double_stream_modulation_txt.forward(&vec_)?;
        let single_mod = self.single_stream_modulation.forward(&vec_)?;

        for block in self.double_blocks.iter_mut() {
            if self.offloaded {
                block.cast_to(&self.device)?;
            }
            (img, txt) = block.forward(&img, &txt, &pe, &double_mod_img, &double_mod_txt)?;
            if self.offloaded {
                block.cast_to(&Device::Cpu)?;
            }
        }

        let mut combined = Tensor::cat(&[&txt, &img], 1)?;
        for block in self.single_blocks.iter_mut() {
            if self.offloaded {
                block.cast_to(&self.device)?;
            }
            combined = block.forward(&combined, &pe, &single_mod)?;
            if self.offloaded {
                block.cast_to(&Device::Cpu)?;
            }
        }

        let img = combined.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }
}
