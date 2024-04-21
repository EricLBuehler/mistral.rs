use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::RotaryEmbedding;
use candle_nn::{Embedding, LayerNorm};

use crate::pipeline::extract_logits;
use crate::pipeline::PHI2_IS_GPTX;

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
    neg_inf: Tensor,
    rotary: Arc<RotaryEmbedding>,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
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

        let mut q = q.reshape((b_sz * seq_len, self.n_head, self.head_dim))?;
        let mut k = k.reshape((b_sz * seq_len, self.n_kv_head, self.head_dim))?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        self.rotary
            .forward(start_offsets, &start_offsets_kernel, &mut q, &mut k, b_sz)?;

        if q.rank() == 3 {
            q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
        }

        let (k, v) = match &*kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                let k = candle_nn::ops::kvconcat(k_cache, &k, 2)?.contiguous()?;
                let v = candle_nn::ops::kvconcat(v_cache, &v, 2)?.contiguous()?;
                (k, v)
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?.contiguous()?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?.contiguous()?;

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
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let head_dim = embedding_length / head_count;
        let max_seq_len = md_get("phi2.context_length")
            .and_then(|m| m.to_u64())
            .unwrap_or(MAX_SEQ_LEN as u64) as usize;
        let rotary = Arc::new(RotaryEmbedding::new_partial(
            10_000.,
            head_dim,
            rope_dim,
            max_seq_len,
            device,
            PHI2_IS_GPTX,
            DType::F32,
        )?);

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
                head_dim,
                neg_inf: neg_inf.clone(),
                rotary: rotary.clone(),
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
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<usize>,
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
                start_offsets,
                start_offsets_kernel.clone(),
                cache.get_mut(i).unwrap(),
            )?;
            let feed_forward_hidden_states = layer.mlp.forward(&xs_norm)?;
            xs = (attn_outputs + feed_forward_hidden_states + residual)?
        }
        let xs = xs.apply(&self.output_norm)?.i((.., seq_len - 1, ..))?;
        extract_logits(&self.output.forward(&xs.contiguous()?)?, context_lens)
    }
}
