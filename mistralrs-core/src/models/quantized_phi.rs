#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;

use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Module, RotaryEmbedding};

use super::{Cache, QRmsNorm};

pub const MAX_SEQ_LEN: u32 = 4096;

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
        self.weight.forward(xs)?.broadcast_add(xs)
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
        dbg!(x);
        let qkv = self.attn_qkv.forward(x)?;
        dbg!(&qkv);
        todo!()
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

        let context_length = md_get("phi2.context_length")?.to_u32()? as usize;
        let embedding_length = md_get("phi2.embedding_length")?.to_u32()? as usize;
        let feed_forward_length = md_get("phi2.feed_forward_length")?.to_u32()? as usize;
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
            false,
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
                    .dequantize(&device)?,
                ct.tensor(reader, &format!("{prefix}.attn_norm.bias"), device)?
                    .dequantize(&device)?,
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
            let residual = x;
            dbg!(x);
            let x = layer.attn_norm.forward(&x)?;
            dbg!(x);
            let attn_outputs = layer.forward_attn(
                &x,
                &mask,
                start_offsets,
                start_offsets_kernel.clone(),
                cache.get_mut(i).unwrap(),
            )?;

            // MLP
            let feed_forward_hidden_states = layer.ffn_up.forward(&x)?;
            let feed_forward_hidden_states = layer.ffn_down.forward(&feed_forward_hidden_states)?;
            layer_in = (attn_outputs + feed_forward_hidden_states + residual)?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x.contiguous()?)
    }
}
