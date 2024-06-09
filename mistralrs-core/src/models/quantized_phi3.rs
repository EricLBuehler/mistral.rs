#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::device_map::DeviceMapper;
use crate::layers::{repeat_kv, CausalMasker, MatMul, RmsNorm, ScaledDotProductAttention};
use crate::pipeline::Cache;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::DeviceMapMetadata;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QMatMul;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Embedding;

#[derive(Debug, Clone)]
struct Mlp {
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    i_size: usize,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = MatMul.qmatmul(xs, &self.ffn_up)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.silu()?)?;
        MatMul.qmatmul(&up_states, &self.ffn_down)
    }
}

fn rms_norm(w: QTensor, eps: f64) -> Result<RmsNorm> {
    let w = w.dequantize(&w.device())?;
    let rms = RmsNorm::from_w(w, eps)?;
    Ok(rms)
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_qkv: QMatMul,
    attn_output: QMatMul,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    sliding_window: usize,
}

impl LayerWeights {
    fn apply_rotary_emb(&self, xs: &Tensor, seqlen_offsets: &[usize]) -> Result<Tensor> {
        let (_b_sz, _h, seq_len, _n_embd) = xs.dims4()?;
        let mut outputs = Vec::new();
        for (i, offset) in seqlen_offsets.iter().enumerate() {
            let cos = self.cos.narrow(0, *offset, seq_len)?;
            let sin = self.sin.narrow(0, *offset, seq_len)?;
            outputs.push(candle_nn::rotary_emb::rope(
                &xs.i(i)?.unsqueeze(0)?.contiguous()?,
                &cos,
                &sin,
            )?);
        }
        Tensor::cat(&outputs, 0)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let qkv = MatMul.qmatmul(x, &self.attn_qkv)?;
        let query_pos = self.n_head * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.n_kv_head * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.n_kv_head * self.head_dim,
            self.n_kv_head * self.head_dim,
        )?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, seqlen_offsets)?.contiguous()?;
        let k = self.apply_rotary_emb(&k, seqlen_offsets)?;

        let (k, v, attn_mask) = Cache::update_kv_cache_sliding_window(
            kv_cache,
            k,
            v,
            mask,
            Some(self.sliding_window),
            true,
        )?;

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let y = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.n_head,
            self.head_dim,
            attn_mask.as_ref(),
            false,
            b_sz,
            seq_len,
        )?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = MatMul.qmatmul(&y, &self.attn_output)?;
        Ok(y)
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: RmsNorm,
    output: QMatMul,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
    context_window: usize,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_window as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_window, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

// phi3 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub i_size: usize,
    pub rope_dim: usize,
    pub rms_eps: f64,
    pub context_window: usize,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("phi3")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "feed_forward_length",
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
            i_size: c.get_value::<u32>("feed_forward_length")? as usize,
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            rms_eps: c.get_value::<f32>("attention.layer_norm_rms_epsilon")? as f64,
            context_window: c.get_value::<u32>("context_length")? as usize,
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
            path_prefix: "phi3",
            metadata: &ct.metadata,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            i_size,
            rope_dim,
            rms_eps,
            context_window,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let (cos, sin) = precomput_freqs_cis(rope_dim, 10_000., device, context_window)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let output_norm = rms_norm(ct.tensor(reader, "output_norm.weight", device)?, rms_eps)?;
        let output = QMatMul::from_qtensor(ct.tensor(reader, "output.weight", device)?)?;
        let mut layers = Vec::with_capacity(block_count);
        let mapper = mapper.into_mapper(block_count, device)?;
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let ffn_up = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{prefix}.ffn_up.weight"),
                device,
            )?)?;
            let ffn_down = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{prefix}.ffn_down.weight"),
                device,
            )?)?;
            let mlp = Mlp {
                ffn_up,
                ffn_down,
                i_size,
            };
            let attn_norm = rms_norm(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_eps,
            )?;
            let ffn_norm = rms_norm(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_eps,
            )?;
            layers.push(LayerWeights {
                attn_qkv: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.attn_qkv.weight"),
                    device,
                )?)?,
                attn_output: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.attn_output.weight"),
                    device,
                )?)?,
                attn_norm,
                ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.to_device(device)?,
                sin: sin.to_device(device)?,
                sliding_window: context_window,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            mapper: Some(mapper),
            device: device.clone(),
            cache: Cache::new(block_count, false),
            max_seq_len: context_window,
        })
    }
}

impl ModelWeights {
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offsets: &[usize]) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.tok_embeddings.forward(input_ids)?;
        let mut cache = self.cache.lock();
        let mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
            input_ids,
            &cache,
            Some(self.max_seq_len),
            xs.dtype(),
            self.layers[0].n_head,
        )?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }
            let residual = &xs;
            let ys = xs.apply(&layer.attn_norm)?;
            let ys = layer.forward_attn(
                &ys,
                mask.as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[i],
            )?;
            let ys = (ys + residual)?;
            let residual = &ys;
            let ys = ys.apply(&layer.ffn_norm)?;
            let ys = layer.mlp.forward(&ys)?;
            xs = (ys + residual)?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.output_norm)?.i((.., seq_len - 1, ..))?;
        MatMul.qmatmul(&xs, &self.output)
    }
}
