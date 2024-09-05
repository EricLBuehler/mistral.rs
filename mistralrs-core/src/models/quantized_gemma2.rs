#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use crate::attention::SdpaParams;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{CausalMasker, MatMul, RmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, Cache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::NiceProgressBar;
use crate::{DeviceMapMetadata, Topology};
use candle_core::quantized::QMatMul;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

#[derive(Clone)]
struct Mlp {
    ffn_gate: Arc<dyn QuantMethod>,
    ffn_up: Arc<dyn QuantMethod>,
    ffn_down: Arc<dyn QuantMethod>,
    act_fn: candle_nn::Activation,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = MatMul
            .qmethod_matmul(xs, &*self.ffn_gate)?
            .apply(&self.act_fn)?;
        let rhs = MatMul.qmethod_matmul(xs, &*self.ffn_up)?;
        MatMul.qmethod_matmul(&(lhs * rhs)?, &*self.ffn_down)
    }
}

fn rms_norm(w: QTensor, eps: f64) -> Result<RmsNorm> {
    let w = w.dequantize(&w.device())?;
    let rms = RmsNorm::from_w(w, eps)?;
    Ok(rms)
}

struct LayerWeights {
    attn_q: Arc<dyn QuantMethod>,
    attn_k: Arc<dyn QuantMethod>,
    attn_v: Arc<dyn QuantMethod>,
    attn_output: Arc<dyn QuantMethod>,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    rotary: RotaryEmbedding,
    attn_logit_softcapping: f64,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        sliding_mask: Option<&Tensor>,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let q = MatMul.qmethod_matmul(x, &*self.attn_q)?;
        let k = MatMul.qmethod_matmul(x, &*self.attn_k)?;
        let v = MatMul.qmethod_matmul(x, &*self.attn_v)?;

        let mut q = q.reshape((b_sz * seq_len, self.n_head, self.head_dim))?;
        let mut k = k.reshape((b_sz * seq_len, self.n_kv_head, self.head_dim))?;
        let v = if seq_len != 1 {
            v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?
        } else {
            // Optimization for seqlen = 1, avoid transpose and just modify reshape dims
            v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?
        };

        self.rotary
            .forward(start_offsets, &start_offsets_kernel, &mut q, &mut k, b_sz)?;

        if q.rank() == 3 && seq_len != 1 {
            q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        } else if q.rank() == 3 {
            // Optimization for seqlen = 1, avoid transpose and just modify reshape dims
            q = q
                .reshape((b_sz, self.n_head, seq_len, self.head_dim))?
                .contiguous()?;
            k = k
                .reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?
                .contiguous()?;
        }

        let mask = if self.sliding_window.is_some() {
            sliding_mask
        } else {
            mask
        };

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
                    Some(self.attn_logit_softcapping),
                )?
            }
            None => {
                let (k, v, attn_mask) = Cache::update_kv_cache_sliding_window(
                    kv_cache,
                    k,
                    v,
                    mask,
                    self.sliding_window,
                    true,
                )?;

                Sdpa.run_attention(&q, &k, &v, attn_mask.as_ref(), None, &self.sdpa_params)?
            }
        };

        let y = if mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let y = MatMul.qmethod_matmul(&y, &*self.attn_output)?;
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
    pub cache: Cache,
    pub max_seq_len: usize,
    final_logit_softcapping: f64,
}

// gemma2 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_eps: f64,
    pub context_window: usize,
    pub key_length: usize,
    pub value_length: usize,
    pub attn_logit_softcapping: f64,
    pub final_logit_softcapping: f64,
    pub query_pre_attn_scalar: f64,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("gemma2")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "attention.key_length",
            "attention.value_length",
            "context_length",
            "attn_logit_softcapping",
            "final_logit_softcapping",
        ];
        c.has_required_keys(&required)?;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec
        let props = Self {
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
            rms_eps: c.get_value::<f32>("attention.layer_norm_rms_epsilon")? as f64,
            context_window: c.get_value::<u32>("context_length")? as usize,
            key_length: c.get_value::<u32>("attention.key_length")? as usize,
            value_length: c.get_value::<u32>("attention.value_length")? as usize,
            attn_logit_softcapping: c.get_value::<f32>("attn_logit_softcapping")? as f64,
            final_logit_softcapping: c.get_value::<f32>("final_logit_softcapping")? as f64,
            query_pre_attn_scalar: 256., // TODO: this may not be correct.
        };

        Ok(props)
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: DeviceMapMetadata,
        topology: Option<&'_ Topology>,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // Parameter extraction from metadata.
        let metadata = ContentMetadata {
            path_prefix: "gemma2",
            metadata: ct.get_metadata(),
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_eps,
            context_window,
            key_length,
            value_length,
            attn_logit_softcapping,
            final_logit_softcapping,
            query_pre_attn_scalar,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let tok_embeddings_q = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let output_norm = rms_norm(ct.tensor("output_norm.weight", device)?, rms_eps)?;
        let output = QMatMul::from_qtensor(tok_embeddings_q)?;
        let mut layers = Vec::with_capacity(block_count);

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        let rotary =
            RotaryEmbedding::new(10000., head_dim, context_window, device, true, DType::F32)?;

        let mapper = mapper.into_mapper(block_count, device, topology)?;

        for layer_idx in NiceProgressBar::<_, 'b'>(0..block_count, "Loading repeating layers") {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let ffn_up =
                QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?;
            let ffn_down =
                QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?;
            let ffn_gate =
                QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?;
            let QMatMul::QTensor(ffn_up_w) = ffn_up else {
                unreachable!()
            };
            let QMatMul::QTensor(ffn_down_w) = ffn_down else {
                unreachable!()
            };
            let QMatMul::QTensor(ffn_gate_w) = ffn_gate else {
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
                ffn_gate: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: ffn_gate_w,
                    b: None,
                })?),
                act_fn: candle_nn::Activation::GeluPytorchTanh,
            };
            let attn_norm = rms_norm(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                rms_eps,
            )?;
            let ffn_norm = rms_norm(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?,
                rms_eps,
            )?;
            let post_ffn_norm = rms_norm(
                ct.tensor(&format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_eps,
            )?;
            let post_attn_norm = rms_norm(
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?,
                rms_eps,
            )?;
            let sliding_window = if layer_idx % 2 == 0 {
                // ^ Order is SWA, global, SWA
                Some(context_window)
            } else {
                None
            };
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(PagedAttention::new(
                    head_count,
                    head_dim,
                    (1.0 / query_pre_attn_scalar.sqrt()) as f32,
                    Some(head_count_kv),
                    sliding_window,
                    device,
                    None,
                )?),
            };
            let q = QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?)?;
            let k = QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.attn_k.weight"), device)?)?;
            let v = QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.attn_v.weight"), device)?)?;
            let out =
                QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?;
            let QMatMul::QTensor(q_w) = q.clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(k_w) = k.clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(v_w) = v.clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(out_w) = out.clone() else {
                unreachable!()
            };
            layers.push(LayerWeights {
                attn_q: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: q_w,
                    b: None,
                })?),
                attn_k: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: k_w,
                    b: None,
                })?),
                attn_v: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: v_w,
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
                rotary: rotary.clone(),
                sliding_window,
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    use_flash_attn: false,
                    softcap: Some(attn_logit_softcapping as f32),
                    softmax_scale: 1.0 / (query_pre_attn_scalar as f32).sqrt(),
                    sliding_window,
                },
                post_ffn_norm,
                post_attn_norm,
                attn_logit_softcapping,
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
            final_logit_softcapping,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        mut metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut xs = self.tok_embeddings.forward(input_ids)?;
        let mut cache = self.cache.lock();
        let mask = CausalMasker.make_causal_mask_as_attn_bias(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*cache as &dyn PastKvLenCache),
            DType::F32,
            self.layers[0].n_head,
        )?;
        let sliding_mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*cache as &dyn PastKvLenCache),
            Some(self.max_seq_len),
            DType::F32,
            self.layers[0].n_head,
        )?;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }
            let residual = &xs;
            let ys = xs.apply(&layer.attn_norm)?;
            let ys = layer
                .forward_attn(
                    &ys,
                    mask.as_ref()
                        .map(|m| m.to_device(xs.device()).unwrap())
                        .as_ref(),
                    sliding_mask
                        .as_ref()
                        .map(|m| m.to_device(xs.device()).unwrap())
                        .as_ref(),
                    start_offsets,
                    start_offsets_kernel.clone(),
                    &mut cache[i],
                    metadata
                        .as_mut()
                        .map(|(kv_cache, metadata)| (kv_cache[i].clone(), &mut **metadata)),
                )?
                .apply(&layer.post_attn_norm)?;
            let ys = (ys + residual)?;
            let residual = &ys;
            let ys = ys.apply(&layer.ffn_norm)?;
            let ys = layer.mlp.forward(&ys)?.apply(&layer.post_ffn_norm)?;
            xs = (ys + residual)?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.output_norm)?;
        let mut xs = MatMul.qmatmul(&xs, &self.output)?;

        xs = (xs / self.final_logit_softcapping)?;
        xs = xs.tanh()?;
        xs = (xs * self.final_logit_softcapping)?;

        extract_logits(&xs, context_lens)
    }
}
