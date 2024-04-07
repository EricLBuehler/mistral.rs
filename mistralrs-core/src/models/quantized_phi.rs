#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;

use candle_core::quantized::gguf_file;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Embedding, LayerNorm, Module, RotaryEmbedding};

use crate::pipeline::PHI2_IS_GPTX;

use super::Cache;

pub const MAX_SEQ_LEN: u32 = 2048;

#[derive(Clone, Debug)]
struct QLinear {
    weight: QMatMul,
    bias: Tensor,
}

impl QLinear {
    fn new(weight: QTensor, bias: QTensor) -> Result<Self> {
        let b = bias.dequantize(&weight.device())?;
        Ok(Self {
            weight: QMatMul::from_qtensor(weight)?,
            bias: b,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.weight.forward(xs)?.broadcast_add(&self.bias)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_qkv: QLinear,
    attn_norm: LayerNorm,
    ffn_down: QLinear,
    ffn_up: QLinear,
    output: QLinear,
    n_head: usize,
    n_kv_head: usize,
    rotary: RotaryEmbedding,
    hidden_size: usize,
    head_dim: usize,
    softmax_scale: f64,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: &Option<Tensor>,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _n_embd) = x.dims3()?;

        let qkv = self.attn_qkv.forward(&x)?;
        let q = qkv.i((.., .., 0..self.hidden_size))?;
        let k = qkv.i((.., .., self.hidden_size..self.hidden_size * 2))?;
        let v = qkv.i((.., .., self.hidden_size * 2..))?;

        //let q = self.attn_norm.forward(&q)?;
        //let k = self.attn_norm.forward(&k)?;

        let mut q = q.reshape((b_size * seq_len, self.n_head, self.head_dim))?;
        let mut k = k.reshape((b_size * seq_len, self.n_kv_head, self.head_dim))?;
        let v = v
            .reshape((b_size, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        self.rotary
            .forward(start_offsets, &start_offsets_kernel, &mut q, &mut k, b_size)?;

        if q.rank() == 3 {
            q = q
                .reshape((b_size, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_size, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        }

        let (k, v) = match &*kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = candle_nn::ops::kvconcat(prev_k, &k, 2)?;
                let v = candle_nn::ops::kvconcat(prev_v, &v, 2)?;
                (k, v)
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        // Repeat kv.
        let k = self.repeat_kv(k)?.contiguous()?;
        let v = self.repeat_kv(v)?.contiguous()?;

        let attn_weights = (q
            .to_dtype(DType::F32)?
            .contiguous()?
            .matmul(&k.to_dtype(DType::F32)?.t()?)?
            * self.softmax_scale)?;
        let attn_weights = match mask {
            None => attn_weights,
            Some(mask) => masked_fill(
                &attn_weights,
                &mask.broadcast_left((b_size, self.n_head))?,
                f32::NEG_INFINITY,
            )?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?.to_dtype(v.dtype())?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_size, seq_len, ()))?;
        self.output.forward(&attn_output)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.n_head / self.n_kv_head;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: LayerNorm,
    output: QLinear,
    masks: HashMap<usize, Tensor>,
    pub device: Device,
    pub cache: Cache,
    act: Activation,
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

        let embedding_length = md_get("phi2.embedding_length")?.to_u32()? as usize;
        let block_count = md_get("phi2.block_count")?.to_u32()? as usize;
        let head_count = md_get("phi2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("phi2.attention.head_count_kv")?.to_u32()? as usize;
        let attn_layer_norm_eps = md_get("phi2.attention.layer_norm_epsilon")?.to_f32()?;
        let rope_dim = md_get("phi2.rope.dimension_count")?.to_u32()? as usize;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let head_dim = embedding_length / head_count;
        let rotary = RotaryEmbedding::new_partial(
            rope_freq_base,
            head_dim,
            rope_dim,
            MAX_SEQ_LEN as usize,
            device,
            PHI2_IS_GPTX,
            DType::F32,
        )?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let embedding = Embedding::new(tok_embeddings, embedding_length);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attn_norm = LayerNorm::new(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?
                    .dequantize(device)?,
                ct.tensor(reader, &format!("{prefix}.attn_norm.bias"), device)?
                    .dequantize(device)?,
                attn_layer_norm_eps as f64,
            );
            let output = QLinear::new(
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?,
                ct.tensor(reader, &format!("{prefix}.attn_output.bias"), device)?,
            )?;
            let attn_qkv = QLinear::new(
                ct.tensor(reader, &format!("{prefix}.attn_qkv.weight"), device)?,
                ct.tensor(reader, &format!("{prefix}.attn_qkv.bias"), device)?,
            )?;
            let ffn_down = QLinear::new(
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?,
                ct.tensor(reader, &format!("{prefix}.ffn_down.bias"), device)?,
            )?;
            let ffn_up = QLinear::new(
                ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?,
                ct.tensor(reader, &format!("{prefix}.ffn_up.bias"), device)?,
            )?;
            layers.push(LayerWeights {
                attn_qkv,
                attn_norm,
                ffn_down,
                ffn_up,
                output,
                n_head: head_count,
                n_kv_head: head_count_kv,
                rotary: rotary.clone(),
                hidden_size: embedding_length,
                head_dim,
                softmax_scale: 1f64 / (head_dim as f64).sqrt(),
            })
        }
        let output_norm = {
            let w = ct.tensor(reader, "output_norm.weight", device)?;
            LayerNorm::new(
                w.dequantize(&w.device())?,
                ct.tensor(reader, "output_norm.bias", device)?
                    .dequantize(&w.device())?,
                attn_layer_norm_eps as f64,
            )
        };

        let output = QLinear::new(
            ct.tensor(reader, "output.weight", device)?,
            ct.tensor(reader, "output.bias", device)?,
        )?;

        Ok(Self {
            tok_embeddings: embedding,
            layers,
            norm: output_norm,
            output,
            masks: HashMap::new(),
            device: device.clone(),
            cache: Cache::new(block_count, false),
            act: Activation::NewGelu,
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
        x: &Tensor,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, x.device())?)
        };
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let mut cache = self.cache.lock();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let residual = layer_in.clone();
            let x = layer.attn_norm.forward(&residual)?;
            let attn_outputs = layer.forward_attn(
                &x,
                &mask,
                start_offsets,
                start_offsets_kernel.clone(),
                cache.get_mut(i).unwrap(),
            )?;
            dbg!(attn_outputs.mean_all());

            // MLP
            let feed_forward_hidden_states = layer.ffn_up.forward(&x)?;
            let feed_forward_hidden_states = self.act.forward(&feed_forward_hidden_states)?;
            let feed_forward_hidden_states = layer.ffn_down.forward(&feed_forward_hidden_states)?;
            dbg!(feed_forward_hidden_states.mean_all());
            dbg!(residual.mean_all());
            println!();
            //layer_in = attn_outputs;
            //continue;
            layer_in = (attn_outputs + feed_forward_hidden_states + residual)?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x.contiguous()?)
    }
}
