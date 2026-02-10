#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm};
use mistralrs_quant::ShardedVarBuilder;
use serde::Deserialize;

use crate::layers::{self, MatMul};

// FLUX.1 defaults
const DEFAULT_MLP_RATIO: f64 = 4.;
const DEFAULT_HIDDEN_SIZE: usize = 3072;
const DEFAULT_AXES_DIM: &[usize] = &[16, 56, 56];
const DEFAULT_THETA: usize = 10000;
const DEFAULT_POOLED_PROJECTION_DIM: usize = 768;

fn default_mlp_ratio() -> f64 {
    DEFAULT_MLP_RATIO
}

fn default_pooled_projection_dim() -> usize {
    DEFAULT_POOLED_PROJECTION_DIM
}

fn default_rope_theta() -> usize {
    DEFAULT_THETA
}

fn default_axes_dims_rope() -> Vec<usize> {
    DEFAULT_AXES_DIM.to_vec()
}

fn default_attention_head_dim() -> Option<usize> {
    None
}

fn default_use_bias() -> bool {
    true // FLUX.1 uses biases, FLUX.2 doesn't
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub in_channels: usize,
    /// Pooled projection dimension for CLIP embeddings. FLUX.2 doesn't use this.
    #[serde(default = "default_pooled_projection_dim")]
    pub pooled_projection_dim: usize,
    pub joint_attention_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub guidance_embeds: bool,
    /// MLP expansion ratio. FLUX.1 uses 4.0, FLUX.2 uses 3.0.
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    /// RoPE theta value. FLUX.1 uses 10000, FLUX.2 uses 2000.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: usize,
    /// Axis dimensions for RoPE. FLUX.1 uses [16, 56, 56], FLUX.2 uses [32, 32, 32, 32].
    #[serde(default = "default_axes_dims_rope")]
    pub axes_dims_rope: Vec<usize>,
    /// Attention head dimension. If set, hidden_size = attention_head_dim * num_attention_heads.
    /// Otherwise, falls back to default hidden size.
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: Option<usize>,
    /// Whether to use biases in linear layers. FLUX.1 uses biases (true), FLUX.2 doesn't (false).
    #[serde(default = "default_use_bias")]
    pub use_bias: bool,
}

impl Config {
    /// Get the hidden size for this model configuration.
    pub fn hidden_size(&self) -> usize {
        self.attention_head_dim
            .map(|dim| dim * self.num_attention_heads)
            .unwrap_or(DEFAULT_HIDDEN_SIZE)
    }

    /// Check if this is a FLUX.2 style model (detected by attention_head_dim being set)
    pub fn is_flux2(&self) -> bool {
        self.attention_head_dim.is_some()
    }

    /// Get whether to use biases in linear layers.
    /// FLUX.1 uses biases, FLUX.2 doesn't.
    /// Auto-detects based on attention_head_dim presence if use_bias is default.
    pub fn should_use_bias(&self) -> bool {
        // If attention_head_dim is set, this is FLUX.2 and shouldn't use bias
        if self.is_flux2() {
            false
        } else {
            self.use_bias
        }
    }
}

fn layer_norm(dim: usize, vb: ShardedVarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

/// Create a linear layer with or without bias based on config
fn linear_maybe_bias(
    in_dim: usize,
    out_dim: usize,
    use_bias: bool,
    vb: ShardedVarBuilder,
) -> Result<Linear> {
    if use_bias {
        layers::linear(in_dim, out_dim, vb)
    } else {
        layers::linear_no_bias(in_dim, out_dim, vb)
    }
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;
    let attn_weights = (MatMul.matmul(&q, &k.t()?)? * scale_factor)?;
    let attn_scores = MatMul.matmul(&candle_nn::ops::softmax_last_dim(&attn_weights)?, &v)?;
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    attn_scores.reshape(batch_dims)
}

fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        candle_core::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    let theta = theta as f64;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), dev)?;
    let inv_freq = inv_freq.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention(&q, &k, v)?;
    x.transpose(1, 2)?.flatten_from(2)
}

fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        candle_core::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(candle_core::DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(candle_core::DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

#[derive(Debug, Clone)]
pub struct EmbedNd {
    #[allow(unused)]
    dim: usize,
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedNd {
    fn new(dim: usize, theta: usize, axes_dim: Vec<usize>) -> Self {
        Self {
            dim,
            theta,
            axes_dim,
        }
    }
}

impl candle_core::Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            let r = rope(
                &ids.get_on_dim(D::Minus1, idx)?,
                self.axes_dim[idx],
                self.theta,
            )?;
            emb.push(r)
        }
        let emb = Tensor::cat(&emb, 2)?;
        emb.unsqueeze(1)
    }
}

#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let in_layer = layers::linear(in_sz, h_sz, vb.pp("in_layer"))?;
        let out_layer = layers::linear(h_sz, h_sz, vb.pp("out_layer"))?;
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

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct Modulation1 {
    lin: Linear,
}

impl Modulation1 {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let lin = layers::linear(dim, 3 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct Modulation2 {
    lin: Linear,
}

impl Modulation2 {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let lin = layers::linear(dim, 6 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

#[derive(Debug, Clone)]
pub struct SelfAttention {
    qkv: Linear,
    norm: QkNorm,
    proj: Linear,
    num_attention_heads: usize,
}

impl SelfAttention {
    fn new(
        dim: usize,
        num_attention_heads: usize,
        qkv_bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_attention_heads;
        let qkv = layers::linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = layers::linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            norm,
            proj,
            num_attention_heads,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_attention_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }

    #[allow(unused)]
    fn forward(&self, xs: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (q, k, v) = self.qkv(xs)?;
        attention(&q, &k, &v, pe)?.apply(&self.proj)
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

#[derive(Debug, Clone)]
struct Mlp {
    lin1: Linear,
    lin2: Linear,
}

impl Mlp {
    fn new(in_sz: usize, mlp_sz: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let lin1 = layers::linear(in_sz, mlp_sz, vb.pp("0"))?;
        let lin2 = layers::linear(mlp_sz, in_sz, vb.pp("2"))?;
        Ok(Self { lin1, lin2 })
    }

    fn cast_to(&mut self, device: &Device) -> Result<()> {
        self.lin1 = Linear::new(
            self.lin1.weight().to_device(device)?,
            self.lin1.bias().map(|x| x.to_device(device).unwrap()),
        );
        self.lin2 = Linear::new(
            self.lin2.weight().to_device(device)?,
            self.lin2.bias().map(|x| x.to_device(device).unwrap()),
        );
        Ok(())
    }
}

impl candle_core::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)
    }
}

#[derive(Debug, Clone)]
pub struct DoubleStreamBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleStreamBlock {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size();
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let img_mod = Modulation2::new(h_sz, vb.pp("img_mod"))?;
        let img_norm1 = layer_norm(h_sz, vb.pp("img_norm1"))?;
        let img_attn = SelfAttention::new(h_sz, cfg.num_attention_heads, true, vb.pp("img_attn"))?;
        let img_norm2 = layer_norm(h_sz, vb.pp("img_norm2"))?;
        let img_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("img_mlp"))?;
        let txt_mod = Modulation2::new(h_sz, vb.pp("txt_mod"))?;
        let txt_norm1 = layer_norm(h_sz, vb.pp("txt_norm1"))?;
        let txt_attn = SelfAttention::new(h_sz, cfg.num_attention_heads, true, vb.pp("txt_attn"))?;
        let txt_norm2 = layer_norm(h_sz, vb.pp("txt_norm2"))?;
        let txt_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("txt_mlp"))?;
        Ok(Self {
            img_mod,
            img_norm1,
            img_attn,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_attn,
            txt_norm2,
            txt_mlp,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?; // shift, scale, gate
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?; // shift, scale, gate
        let img_modulated = img.apply(&self.img_norm1)?;
        let img_modulated = img_mod1.scale_shift(&img_modulated)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;

        let txt_modulated = txt.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_modulated)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        let img = (img + img_mod1.gate(&img_attn.apply(&self.img_attn.proj)?))?;
        let img = (&img
            + img_mod2.gate(
                &img_mod2
                    .scale_shift(&img.apply(&self.img_norm2)?)?
                    .apply(&self.img_mlp)?,
            )?)?;

        let txt = (txt + txt_mod1.gate(&txt_attn.apply(&self.txt_attn.proj)?))?;
        let txt = (&txt
            + txt_mod2.gate(
                &txt_mod2
                    .scale_shift(&txt.apply(&self.txt_norm2)?)?
                    .apply(&self.txt_mlp)?,
            )?)?;

        Ok((img, txt))
    }

    fn cast_to(&mut self, device: &Device) -> Result<()> {
        self.img_mod.lin = Linear::new(
            self.img_mod.lin.weight().to_device(device)?,
            self.img_mod
                .lin
                .bias()
                .map(|x| x.to_device(device).unwrap()),
        );
        self.img_norm1 = LayerNorm::new_no_bias(self.img_norm1.weight().to_device(device)?, 1e-6);
        self.img_attn.cast_to(device)?;
        self.img_norm2 = LayerNorm::new_no_bias(self.img_norm2.weight().to_device(device)?, 1e-6);
        self.img_mlp.cast_to(device)?;

        self.txt_mod.lin = Linear::new(
            self.txt_mod.lin.weight().to_device(device)?,
            self.txt_mod
                .lin
                .bias()
                .map(|x| x.to_device(device).unwrap()),
        );
        self.txt_norm1 = LayerNorm::new_no_bias(self.txt_norm1.weight().to_device(device)?, 1e-6);
        self.txt_attn.cast_to(device)?;
        self.txt_norm2 = LayerNorm::new_no_bias(self.txt_norm2.weight().to_device(device)?, 1e-6);
        self.txt_mlp.cast_to(device)?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    linear1: Linear,
    linear2: Linear,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_attention_heads: usize,
}

impl SingleStreamBlock {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size();
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_attention_heads;
        let linear1 = layers::linear(h_sz, h_sz * 3 + mlp_sz, vb.pp("linear1"))?;
        let linear2 = layers::linear(h_sz + mlp_sz, h_sz, vb.pp("linear2"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let pre_norm = layer_norm(h_sz, vb.pp("pre_norm"))?;
        let modulation = Modulation1::new(h_sz, vb.pp("modulation"))?;
        Ok(Self {
            linear1,
            linear2,
            norm,
            pre_norm,
            modulation,
            h_sz,
            mlp_sz,
            num_attention_heads: cfg.num_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let mod_ = self.modulation.forward(vec_)?;
        let x_mod = mod_.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let x_mod = x_mod.apply(&self.linear1)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_attention_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output = Tensor::cat(&[attn, mlp.gelu()?], 2)?.apply(&self.linear2)?;
        xs + mod_.gate(&output)
    }

    fn cast_to(&mut self, device: &Device) -> Result<()> {
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
        self.pre_norm = LayerNorm::new_no_bias(self.pre_norm.weight().to_device(device)?, 1e-6);
        self.modulation.lin = Linear::new(
            self.modulation.lin.weight().to_device(device)?,
            self.modulation
                .lin
                .bias()
                .map(|x| x.to_device(device).unwrap()),
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LastLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl LastLayer {
    fn new(h_sz: usize, p_sz: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let norm_final = layer_norm(h_sz, vb.pp("norm_final"))?;
        let linear = layers::linear(h_sz, p_sz * p_sz * out_c, vb.pp("linear"))?;
        let ada_ln_modulation = layers::linear(h_sz, 2 * h_sz, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        xs.apply(&self.linear)
    }
}

#[derive(Debug, Clone)]
pub struct Flux {
    img_in: Linear,
    txt_in: Linear,
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
    device: Device,
    offloaded: bool,
}

impl Flux {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        device: Device,
        offloaded: bool,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let use_bias = cfg.should_use_bias();
        let img_in = linear_maybe_bias(
            cfg.in_channels,
            hidden_size,
            use_bias,
            vb.pp("img_in").set_device(device.clone()),
        )?;
        let txt_in = linear_maybe_bias(
            cfg.joint_attention_dim,
            hidden_size,
            use_bias,
            vb.pp("txt_in").set_device(device.clone()),
        )?;
        let mut double_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_d = vb.pp("double_blocks");
        for idx in 0..cfg.num_layers {
            let db = DoubleStreamBlock::new(cfg, vb_d.pp(idx))?;
            double_blocks.push(db)
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        let vb_s = vb.pp("single_blocks");
        for idx in 0..cfg.num_single_layers {
            let sb = SingleStreamBlock::new(cfg, vb_s.pp(idx))?;
            single_blocks.push(sb)
        }
        let time_in = MlpEmbedder::new(
            256,
            hidden_size,
            vb.pp("time_in").set_device(device.clone()),
        )?;
        let vector_in = MlpEmbedder::new(
            cfg.pooled_projection_dim,
            hidden_size,
            vb.pp("vector_in").set_device(device.clone()),
        )?;
        let guidance_in = if cfg.guidance_embeds {
            let mlp = MlpEmbedder::new(
                256,
                hidden_size,
                vb.pp("guidance_in").set_device(device.clone()),
            )?;
            Some(mlp)
        } else {
            None
        };
        let final_layer = LastLayer::new(
            hidden_size,
            1,
            cfg.in_channels,
            vb.pp("final_layer").set_device(device.clone()),
        )?;
        let pe_dim = hidden_size / cfg.num_attention_heads;
        let pe_embedder = EmbedNd::new(pe_dim, cfg.rope_theta, cfg.axes_dims_rope.clone());
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_blocks,
            single_blocks,
            final_layer,
            device: device.clone(),
            offloaded,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 {
            candle_core::bail!("unexpected shape for txt {:?}", txt.shape())
        }
        if img.rank() != 3 {
            candle_core::bail!("unexpected shape for img {:?}", img.shape())
        }
        let dtype = img.dtype();
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;
        let vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        // Double blocks
        for block in self.double_blocks.iter_mut() {
            if self.offloaded {
                block.cast_to(&self.device)?;
            }
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?;
            if self.offloaded {
                block.cast_to(&Device::Cpu)?;
            }
        }
        // Single blocks
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in self.single_blocks.iter_mut() {
            if self.offloaded {
                block.cast_to(&self.device)?;
            }
            img = block.forward(&img, &vec_, &pe)?;
            if self.offloaded {
                block.cast_to(&Device::Cpu)?;
            }
        }
        let img = img.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }
}
