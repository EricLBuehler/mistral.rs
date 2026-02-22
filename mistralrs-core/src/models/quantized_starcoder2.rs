#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::SdpaParams;
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMasker, MatMul, QLinear, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use candle_core::quantized::QMatMul;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

#[derive(Clone)]
struct Mlp {
    ffn_up: Arc<dyn QuantMethod>,
    ffn_down: Arc<dyn QuantMethod>,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        MatMul.qmethod_matmul(
            &MatMul
                .qmethod_matmul(xs, &*self.ffn_up)?
                .apply(&candle_nn::Activation::GeluPytorchTanh)?,
            &*self.ffn_down,
        )
    }
}

fn layer_norm(w: QTensor, b: QTensor, eps: f64) -> Result<LayerNorm> {
    let w = w.dequantize(&w.device())?;
    let b = b.dequantize(&b.device())?;
    let ln = LayerNorm::new(w, b, eps);
    Ok(ln)
}

struct LayerWeights {
    attn_q: Arc<dyn QuantMethod>,
    attn_k: Arc<dyn QuantMethod>,
    attn_v: Arc<dyn QuantMethod>,
    attn_output: Arc<dyn QuantMethod>,
    attn_norm: LayerNorm,
    ffn_norm: LayerNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, hidden_size) = x.dims3()?;

        let q = MatMul
            .qmethod_matmul(x, &*self.attn_q)?
            .to_dtype(self.dtype)?;
        let k = MatMul
            .qmethod_matmul(x, &*self.attn_k)?
            .to_dtype(self.dtype)?;
        let v = MatMul
            .qmethod_matmul(x, &*self.attn_v)?
            .to_dtype(self.dtype)?;

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, q_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

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
            y.transpose(1, 2)?.reshape(&[b_sz, q_len, hidden_size])?
        } else {
            y.reshape(&[b_sz, q_len, hidden_size])?
        };

        MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attn_output)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: LayerNorm,
    output: QMatMul,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    dtype: DType,
}

// starcoder2 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub layer_norm_epsilon: f64,
    pub context_window: usize,
    pub rope_freq_base: f32,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("starcoder2")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_epsilon",
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
            layer_norm_epsilon: c.get_value::<f32>("attention.layer_norm_epsilon")? as f64,
            context_window: c.get_value::<u32>("context_length")? as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(100_000_f32),
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
            path_prefix: "starcoder2",
            metadata: ct.get_metadata(),
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            layer_norm_epsilon,
            context_window,
            rope_freq_base,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let tok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let head_dim = embedding_length / head_count;
        let output_norm = layer_norm(
            ct.tensor("output_norm.weight", device)?,
            ct.tensor("output_norm.bias", device)?,
            layer_norm_epsilon,
        )?;
        let output = QMatMul::from_qtensor(ct.tensor("output.weight", device)?)?;
        let mut layers = Vec::with_capacity(block_count);

        let mut ropes = HashMap::new();
        for layer_idx in 0..block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    context_window,
                    device,
                    true,
                    dtype,
                )?),
            );
        }

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

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
                layer_norm_epsilon,
            )?;
            let ffn_norm = layer_norm(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?,
                ct.tensor(&format!("{prefix}.ffn_norm.bias"), device)?,
                layer_norm_epsilon,
            )?;
            let attn_q = QLinear::new(&mut ct, &format!("{prefix}.attn_q"), device)?;
            let attn_k = QLinear::new(&mut ct, &format!("{prefix}.attn_k"), device)?;
            let attn_v = QLinear::new(&mut ct, &format!("{prefix}.attn_v"), device)?;
            let attn_output = QLinear::new(&mut ct, &format!("{prefix}.attn_output"), device)?;
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let QMatMul::QTensor(q_w) = attn_q.inner_ref().clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(k_w) = attn_k.inner_ref().clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(v_w) = attn_v.inner_ref().clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(o_w) = attn_output.inner_ref().clone() else {
                unreachable!()
            };
            layers.push(LayerWeights {
                attn_q: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: q_w,
                    b: attn_q.bias().cloned(),
                })?),
                attn_k: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: k_w,
                    b: attn_k.bias().cloned(),
                })?),
                attn_v: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: v_w,
                    b: attn_v.bias().cloned(),
                })?),
                attn_output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: o_w,
                    b: attn_output.bias().cloned(),
                })?),
                attn_norm,
                ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary_emb: rotary,
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
            mapper: Some(mapper),
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, context_window)),
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
