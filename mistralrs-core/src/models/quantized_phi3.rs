#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use crate::attention::SdpaParams;
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMasker, MatMul, RmsNorm, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use candle_core::quantized::QMatMul;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Embedding;
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

#[derive(Clone)]
struct Mlp {
    ffn_up: Arc<dyn QuantMethod>,
    ffn_down: Arc<dyn QuantMethod>,
    i_size: usize,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = MatMul.qmethod_matmul(xs, &*self.ffn_up)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states =
            crate::ops::mul_and_act(&gate, &up_states, crate::layers::Activation::Silu)?;
        MatMul.qmethod_matmul(&up_states, &*self.ffn_down)
    }
}

fn rms_norm(w: QTensor, eps: f64) -> Result<RmsNorm> {
    let w = w.dequantize(&w.device())?;
    let rms = RmsNorm::from_w(w, eps)?;
    Ok(rms)
}

struct LayerWeights {
    attn_qkv: Arc<dyn QuantMethod>,
    attn_output: Arc<dyn QuantMethod>,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
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
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let qkv = MatMul
            .qmethod_matmul(x, &*self.attn_qkv)?
            .to_dtype(self.dtype)?;
        let query_pos = self.n_head * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.n_kv_head * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.n_kv_head * self.head_dim,
            self.n_kv_head * self.head_dim,
        )?;

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v.contiguous()?)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let q = self.apply_rotary_emb(&q, seqlen_offsets)?;
        let k = self.apply_rotary_emb(&k, seqlen_offsets)?;
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
        let y = MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attn_output)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: RmsNorm,
    output: QMatMul,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    dtype: DType,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
    context_window: usize,
    dtype: DType,
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
    let cos = idx_theta.cos()?.to_dtype(dtype)?;
    let sin = idx_theta.sin()?.to_dtype(dtype)?;
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
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        // Parameter extraction from metadata.
        let metadata = ContentMetadata {
            path_prefix: "phi3",
            metadata: ct.get_metadata(),
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

        let (cos, sin) = precomput_freqs_cis(rope_dim, 10_000., device, context_window, dtype)?;

        let tok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let output_norm = rms_norm(ct.tensor("output_norm.weight", device)?, rms_eps)?;
        let output = QMatMul::from_qtensor(ct.tensor("output.weight", device)?)?;
        let mut layers = Vec::with_capacity(block_count);
        let head_dim = embedding_length / head_count;

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let ffn_up =
                QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?;
            let ffn_down =
                QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?;
            let QMatMul::QTensor(ffn_up_w) = ffn_up else {
                unreachable!()
            };
            let QMatMul::QTensor(ffn_down_w) = ffn_down else {
                unreachable!()
            };
            let mlp = Mlp {
                ffn_up: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: ffn_up_w,
                    b: None,
                })?),
                ffn_down: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: ffn_down_w,
                    b: None,
                })?),
                i_size,
            };
            let attn_norm = rms_norm(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                rms_eps,
            )?;
            let ffn_norm = rms_norm(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?,
                rms_eps,
            )?;
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let qkv =
                QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.attn_qkv.weight"), device)?)?;
            let out =
                QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?;
            let QMatMul::QTensor(qkv_w) = qkv.clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(out_w) = out.clone() else {
                unreachable!()
            };
            layers.push(LayerWeights {
                attn_qkv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: qkv_w,
                    b: None,
                })?),
                attn_output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: out_w,
                    b: None,
                })?),
                attn_norm,
                ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.to_device(device)?,
                sin: sin.to_device(device)?,
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: Some(context_window),
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
            mapper: Some(mapper),
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new_sliding(
                block_count,
                context_window,
                Some(context_window),
            )),
            max_seq_len: context_window,
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.tok_embeddings.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            Some(self.max_seq_len),
            self.dtype,
            self.layers[0].n_head,
        )?;
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = DeviceMappedMask::new(mask, &**self.mapper.as_ref().unwrap())?;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }
            let residual = &xs;
            let ys = xs.apply(&layer.attn_norm)?;
            let ys = layer.forward_attn(
                &ys,
                mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let ys = (ys + residual)?;
            let residual = &ys;
            let ys = ys.apply(&layer.ffn_norm)?;
            let ys = layer.mlp.forward(&ys)?;
            xs = (ys + residual)?
        }
        let xs = xs
            .apply(&self.output_norm)?
            .i((.., seq_len - 1, ..))?
            .contiguous()?;
        MatMul.qmatmul(&xs, &self.output)
    }
}
