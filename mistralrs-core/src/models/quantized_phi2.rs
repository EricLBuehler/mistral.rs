#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm};

use crate::device_map::DeviceMapper;
use crate::layers::ScaledDotProductAttention;
use crate::layers::{repeat_kv, CausalMasker, QLinear};
use crate::pipeline::{extract_logits, Cache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::DeviceMapMetadata;

pub const MAX_SEQ_LEN: usize = 4096;

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
}

impl LayerWeights {
    fn forward(&self, xs: &Tensor, start_offsets: &[usize]) -> Result<Tensor> {
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
        Tensor::cat(&chunks, 0)?.contiguous()
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

        let q = self.forward(&q, seqlen_offsets)?.contiguous()?;
        let k = self.forward(&k, seqlen_offsets)?;

        let (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let y = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.n_head,
            self.head_dim,
            mask,
            false,
            b_sz,
            seq_len,
        )?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attn_output.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: LayerNorm,
    output: QLinear,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
    max_seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
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

// phi2 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
struct PropsGGUF {
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    embedding_length: usize,
    rope_dim: usize,
    ln_eps: f64,
    max_seq_len: usize,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("phi2")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "rope.dimension_count",
            "attention.layer_norm_rms_epsilon",
            "context_length",
        ];
        c.has_required_keys(&required)?;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec
        let props = Self {
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            ln_eps: c.get_value::<f32>("attention.layer_norm_rms_epsilon")? as f64,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(MAX_SEQ_LEN as u64) as usize,
        };

        Ok(props)
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mapper: DeviceMapMetadata,
    ) -> Result<Self> {
        // Parameter extraction from metadata.
        let metadata = ContentMetadata {
            path_prefix: "phi2",
            metadata: &ct.metadata,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rope_dim,
            ln_eps,
            max_seq_len,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let (cos, sin) = precomput_freqs_cis(rope_dim, 10_000., device, max_seq_len)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let output_norm = layer_norm(
            ct.tensor(reader, "output_norm.weight", device)?,
            ct.tensor(reader, "output_norm.bias", device)?,
            ln_eps,
        )?;
        let output = QLinear::new(&ct, reader, "output", device)?;
        let mut layers = Vec::with_capacity(block_count);
        let mapper = mapper.into_mapper(block_count, device)?;
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);

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
                cos: cos.clone().to_device(device)?,
                sin: sin.clone().to_device(device)?,
                rope_dim,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            device: device.clone(),
            cache: Cache::new(block_count, false),
            max_seq_len,
            mapper,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut xs = self.tok_embeddings.forward(input_ids)?;
        let mut cache = self.cache.lock();
        let mask = CausalMasker.make_causal_mask_as_attn_bias(
            input_ids,
            &cache,
            xs.dtype(),
            self.layers[0].n_head,
        )?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let residual = &xs;
            let xs_norm = xs.apply(&layer.attn_norm)?;
            let attn_outputs = layer.forward_attn(
                &xs_norm,
                mask.as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                cache.get_mut(i).unwrap(),
            )?;
            let feed_forward_hidden_states = layer.mlp.forward(&xs_norm)?;
            xs = (attn_outputs + feed_forward_hidden_states + residual)?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = extract_logits(&xs.apply(&self.output_norm)?, context_lens)?;
        self.output.forward(&xs)
    }
}
