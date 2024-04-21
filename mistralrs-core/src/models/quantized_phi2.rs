#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm};

use super::repeat_kv;
use super::Cache;

pub const MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone)]
struct QLinear {
    inner: candle_core::quantized::QMatMul,
    bias: Tensor,
}

impl QLinear {
    fn new<R: std::io::Read + std::io::Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let w = ct.tensor(r, &format!("{name}.weight"), device)?;
        let b = ct.tensor(r, &format!("{name}.bias"), device)?;
        let inner = candle_core::quantized::QMatMul::from_qtensor(w)?;
        let bias = b.dequantize(device)?;
        Ok(Self { inner, bias })
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)?.broadcast_add(&self.bias)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    ffn_up: QLinear,
    ffn_down: QLinear,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.ffn_up)?.gelu()?.apply(&self.ffn_down)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_qkv: QLinear,
    attn_output: QLinear,
    attn_norm: LayerNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    rope_dim: usize,
    neg_inf: Tensor,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, xs: &Tensor, start_offsets: &[usize]) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, _n_embd) = xs.dims4()?;
        let xs_rot = xs.i((.., .., .., ..self.rope_dim))?;
        let xs_pass = xs.i((.., .., .., self.rope_dim..))?;
        let mut chunks = Vec::new();
        for (b, offset) in (0..xs.dim(0)?).zip(start_offsets) {
            let cos = self.cos.narrow(0, *offset, seq_len)?;
            let sin = self.sin.narrow(0, *offset, seq_len)?;
            let xs_rot =
                candle_nn::rotary_emb::rope(&xs_rot.i(b)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            chunks.push(Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?);
        }
        Tensor::cat(&chunks, 0)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let qkv =
            self.attn_qkv
                .forward(x)?
                .reshape((b_sz, seq_len, 3, self.n_head, self.head_dim))?;

        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        // This call to contiguous ensures that the fast kernel can be called below. It's
        // actually a no-op except when processing the initial prompt so has no significant
        // impact on performance.
        let v = v.contiguous()?;

        let q = self.apply_rotary_emb(&q, seqlen_offsets)?.contiguous()?;
        let k = self.apply_rotary_emb(&k, seqlen_offsets)?;

        let (k, v) = match &*kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attn_output.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: LayerNorm,
    output: QLinear,
    masks: HashMap<usize, Tensor>,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

fn layer_norm(w: QTensor, b: QTensor, eps: f64) -> Result<LayerNorm> {
    let w = w.dequantize(&w.device())?;
    let b = b.dequantize(&b.device())?;
    let ln = LayerNorm::new(w, b, eps);
    Ok(ln)
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let head_count = md_get("phi2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("phi2.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("phi2.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("phi2.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("phi2.rope.dimension_count")?.to_u32()? as usize;
        let ln_eps = md_get("phi2.attention.layer_norm_epsilon")?.to_f32()? as f64;
        let (cos, sin) = precomput_freqs_cis(rope_dim, 10_000., device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let max_seq_len = md_get("phi2.context_length")
            .and_then(|m| m.to_u64())
            .unwrap_or(MAX_SEQ_LEN as u64) as usize;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let output_norm = layer_norm(
            ct.tensor(reader, "output_norm.weight", device)?,
            ct.tensor(reader, "output_norm.bias", device)?,
            ln_eps,
        )?;
        let output = QLinear::new(&ct, reader, "output", device)?;
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let ffn_up = QLinear::new(&ct, reader, &format!("{prefix}.ffn_up"), device)?;
            let ffn_down = QLinear::new(&ct, reader, &format!("{prefix}.ffn_down"), device)?;
            let mlp = Mlp { ffn_up, ffn_down };
            let attn_norm = layer_norm(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                ct.tensor(reader, &format!("{prefix}.attn_norm.bias"), device)?,
                ln_eps,
            )?;
            layers.push(LayerWeights {
                attn_qkv: QLinear::new(&ct, reader, &format!("{prefix}.attn_qkv"), device)?,
                attn_output: QLinear::new(&ct, reader, &format!("{prefix}.attn_output"), device)?,
                attn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                rope_dim,
                neg_inf: neg_inf.clone(),
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            masks: HashMap::new(),
            device: device.clone(),
            cache: Cache::new(block_count, false),
            max_seq_len,
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        _context_lens: Vec<usize>,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = xs.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, xs.device())?)
        };
        let mut xs = self.tok_embeddings.forward(xs)?;
        let mut cache = self.cache.lock();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let residual = &xs;
            let xs_norm = xs.apply(&layer.attn_norm)?;
            let attn_outputs = layer.forward_attn(
                &xs_norm,
                mask.as_ref(),
                seqlen_offsets,
                cache.get_mut(i).unwrap(),
            )?;
            let feed_forward_hidden_states = layer.mlp.forward(&xs_norm)?;
            xs = (attn_outputs + feed_forward_hidden_states + residual)?
        }
        let xs = xs.apply(&self.output_norm)?.i((.., seq_len - 1, ..))?;
        self.output.forward(&xs)
    }
}
