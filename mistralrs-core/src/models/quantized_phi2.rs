#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm};
use mistralrs_quant::GgufMatMul;
use mistralrs_quant::QuantMethod;
use mistralrs_quant::QuantMethodConfig;

use crate::attention::SdpaParams;
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::MatMul;
use crate::layers::Sdpa;
use crate::layers::{CausalMasker, QLinear};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::AttentionImplementation;
use crate::paged_attention::PagedAttention;
use crate::pipeline::extract_logits;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::EitherCache;
use crate::pipeline::KvCache;
use crate::pipeline::NormalCache;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Clone)]
struct Mlp {
    ffn_up: Arc<dyn QuantMethod>,
    ffn_down: Arc<dyn QuantMethod>,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        MatMul.qmethod_matmul(&MatMul.qmethod_matmul(xs, &*self.ffn_up)?, &*self.ffn_down)
    }
}

struct LayerWeights {
    attn_qkv: Arc<dyn QuantMethod>,
    attn_output: Arc<dyn QuantMethod>,
    attn_norm: LayerNorm,
    mlp: Mlp,
    n_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    rope_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
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
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let qkv = self
            .attn_qkv
            .forward(x)?
            .reshape((b_sz, seq_len, 3, self.n_head, self.head_dim))?
            .to_dtype(self.dtype)?;

        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        // This call to contiguous ensures that the fast kernel can be called below. It's
        // actually a no-op except when processing the initial prompt so has no significant
        // impact on performance.
        let v = v.contiguous()?;

        let q = self.forward(&q, seqlen_offsets)?.contiguous()?;
        let k = self.forward(&k, seqlen_offsets)?;

        let y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        let y = if mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let y = self.attn_output.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: LayerNorm,
    output: QLinear,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    dtype: DType,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
    max_seq_len: usize,
    dtype: DType,
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
    let cos = idx_theta.cos()?.to_dtype(dtype)?;
    let sin = idx_theta.sin()?.to_dtype(dtype)?;
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
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
        };

        Ok(props)
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        // Parameter extraction from metadata.
        let metadata = ContentMetadata {
            path_prefix: "phi2",
            metadata: ct.get_metadata(),
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

        let (cos, sin) = precomput_freqs_cis(rope_dim, 10_000., device, max_seq_len, dtype)?;

        let tok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let output_norm = layer_norm(
            ct.tensor("output_norm.weight", device)?,
            ct.tensor("output_norm.bias", device)?,
            ln_eps,
        )?;
        let output = QLinear::new(&mut ct, "output", device)?;
        let mut layers = Vec::with_capacity(block_count);
        let head_dim = embedding_length / head_count;

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);

            let ffn_up = QLinear::new(&mut ct, &format!("{prefix}.ffn_up"), device)?;
            let ffn_down = QLinear::new(&mut ct, &format!("{prefix}.ffn_down"), device)?;
            let QMatMul::QTensor(ffn_up_w) = ffn_up.inner_ref().clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(ffn_down_w) = ffn_down.inner_ref().clone() else {
                unreachable!()
            };
            let mlp = Mlp {
                ffn_up: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: ffn_up_w,
                    b: ffn_up.bias().cloned(),
                })?),
                ffn_down: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: ffn_down_w,
                    b: ffn_down.bias().cloned(),
                })?),
            };
            let attn_norm = layer_norm(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                ct.tensor(&format!("{prefix}.attn_norm.bias"), device)?,
                ln_eps,
            )?;
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let qkv = QLinear::new(&mut ct, &format!("{prefix}.attn_qkv"), device)?;
            let out = QLinear::new(&mut ct, &format!("{prefix}.attn_output"), device)?;
            let QMatMul::QTensor(qkv_w) = qkv.inner_ref().clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(out_w) = out.inner_ref().clone() else {
                unreachable!()
            };
            layers.push(LayerWeights {
                attn_qkv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: qkv_w,
                    b: qkv.bias().cloned(),
                })?),
                attn_output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: out_w,
                    b: out.bias().cloned(),
                })?),
                attn_norm,
                mlp,
                n_head: head_count,
                head_dim,
                cos: cos.clone().to_device(device)?,
                sin: sin.clone().to_device(device)?,
                rope_dim,
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                    sinks: None,
                },
                dtype,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper,
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut xs = self.tok_embeddings.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            self.layers[0].n_head,
        )?;
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let residual = &xs;
            let xs_norm = xs.apply(&layer.attn_norm)?;
            let attn_outputs = layer.forward_attn(
                &xs_norm,
                mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let feed_forward_hidden_states = layer.mlp.forward(&xs_norm)?;
            xs = (attn_outputs + feed_forward_hidden_states + residual)?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = extract_logits(&xs.apply(&self.output_norm)?, context_lens)?;
        self.output.forward(&xs)
    }
}
