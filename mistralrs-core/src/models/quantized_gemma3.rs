#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{
    Activation, CausalMaskConfig, CausalMasker, RmsNorm, RotaryEmbedding, ScaledEmbedding, Sdpa,
};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

// Defaults fall back to upstream Gemma 3 config values when the GGUF
// metadata omits an optional key. Mirrors `vision_models/gemma3/config.rs`
// and the per-arch reads in `ggml-org/llama.cpp:src/models/gemma3.cpp:3-18`.
const DEFAULT_MAX_SEQ_LEN: u32 = 131072;
const DEFAULT_ROPE_FREQ_BASE: f32 = 1_000_000.0;
const DEFAULT_ROPE_FREQ_BASE_SWA: f32 = 10_000.0;
const DEFAULT_RMS_NORM_EPS: f32 = 1e-6;
const DEFAULT_HEAD_DIM: usize = 256;
const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6;
const HIDDEN_ACTIVATION: Activation = Activation::GeluPytorchTanh;

// `n_layer == 62` is the Gemma 3 27B variant; the attention softmax scale uses
// `1 / sqrt(n_embd / n_head)` there instead of `1 / sqrt(head_dim)`. See
// `ggml-org/llama.cpp:src/models/gemma3.cpp:20-33`.
const GEMMA3_27B_LAYER_COUNT: usize = 62;

struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        let y = crate::ops::mul_and_act(&w1, &w3, self.act)?;
        self.feed_forward_w2.forward(&y)
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    is_sliding: bool,
    sliding_window: Option<usize>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
    layer_idx: usize,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        sliding_attention_mask: &AttentionMask,
        seqlen_offsets: &[usize],
        kv_caches: &mut [KvCache],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let (q, k, v) = if seq_len != 1 {
            (
                q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                    .transpose(1, 2)?,
                k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?,
                v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?,
            )
        } else {
            (
                q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?,
                k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?,
                v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?,
            )
        };

        let (q, k) = self.rotary.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            seqlen_offsets,
        )?;

        let q = q.to_dtype(self.dtype)?;
        let k = k.to_dtype(self.dtype)?;
        let v = v.to_dtype(self.dtype)?;

        let mut y = match &self.paged_attn {
            Some(paged_attn) => {
                let mask = if self.is_sliding {
                    sliding_attention_mask
                } else {
                    attention_mask
                };
                match metadata {
                    Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        mask,
                        Some(key_cache),
                        Some(value_cache),
                        input_metadata,
                        &self.sdpa_params,
                        None,
                    )?,
                    None => {
                        let dummy = PagedAttentionInputMetadata::dummy(q.device())?;
                        paged_attn.forward(
                            &q,
                            &k,
                            &v,
                            mask,
                            None,
                            None,
                            &dummy,
                            &self.sdpa_params,
                            None,
                        )?
                    }
                }
            }
            None => {
                let (k_full, v_full) = kv_caches[self.layer_idx].append(&k, &v)?;
                let (k_used, v_used) = if let Some((start, len)) = sliding_decode_kv_window(
                    self.is_sliding,
                    seq_len,
                    self.sliding_window,
                    k_full.dim(2)?,
                ) {
                    (k_full.narrow(2, start, len)?, v_full.narrow(2, start, len)?)
                } else {
                    (k_full, v_full)
                };

                let mask = if self.is_sliding {
                    sliding_attention_mask
                } else {
                    attention_mask
                };
                Sdpa.run_attention(&q, &k_used, &v_used, mask, None, &self.sdpa_params)?
            }
        };

        let has_mask = !matches!(attention_mask, AttentionMask::None)
            || !matches!(sliding_attention_mask, AttentionMask::None);
        y = if has_mask {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.attention_wo.forward(&y.to_dtype(x.dtype())?)
    }
}

fn sliding_decode_kv_window(
    is_sliding: bool,
    q_len: usize,
    sliding_window: Option<usize>,
    kv_len: usize,
) -> Option<(usize, usize)> {
    let window = sliding_window?;
    if !is_sliding || q_len != 1 || kv_len <= window {
        return None;
    }
    Some((kv_len - window, window))
}

pub struct ModelWeights {
    embed_tokens: ScaledEmbedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    final_logit_softcapping: Option<f64>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    sliding_window: Option<usize>,
}

// `gemma3.*` metadata. Keys verified against
// `ggml-org/llama.cpp:src/models/gemma3.cpp:3-18` and the per-arch
// tensor list at `gguf-py/gguf/constants.py:2397-2414`.
pub(crate) struct PropsGGUF {
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    embedding_length: usize,
    rms_norm_eps: f32,
    max_seq_len: usize,
    rope_freq_base: f32,
    rope_freq_base_swa: f32,
    head_dim: usize,
    sliding_window: usize,
    sliding_window_pattern: usize,
    final_logit_softcapping: Option<f64>,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("gemma3")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let embedding_length = c.get_value::<u32>("embedding_length")? as usize;
        // `attention.key_length` is the per-head K dimension. Gemma 3 keeps K
        // and V at the same dimension on every layer, unlike Gemma 4 which
        // splits sliding vs full.
        let head_dim = c
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(DEFAULT_HEAD_DIM);
        let value_length = c
            .get_value::<u32>("attention.value_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(head_dim);
        if value_length != head_dim {
            anyhow::bail!(
                "Gemma 3 GGUF requires attention.key_length == attention.value_length, got {head_dim} != {value_length}"
            );
        }

        Ok(Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length,
            rms_norm_eps: c
                .get_value("attention.layer_norm_rms_epsilon")
                .unwrap_or(DEFAULT_RMS_NORM_EPS),
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c
                .get_value("rope.freq_base")
                .ok()
                .unwrap_or(DEFAULT_ROPE_FREQ_BASE),
            rope_freq_base_swa: c
                .get_value("rope.freq_base_swa")
                .ok()
                .unwrap_or(DEFAULT_ROPE_FREQ_BASE_SWA),
            head_dim,
            sliding_window: c
                .get_value::<u32>("attention.sliding_window")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            sliding_window_pattern: c
                .get_value::<u32>("attention.sliding_window_pattern")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(DEFAULT_SLIDING_WINDOW_PATTERN),
            final_logit_softcapping: c
                .get_option_value::<f32>("final_logit_softcapping")?
                .map(f64::from),
        })
    }
}

// Per-layer SWA flag reconstructed from the metadata-stored pattern. llama.cpp
// uses `set_swa_pattern(swa_period)` with the default `dense_first = false`,
// which marks layer `il` as SWA when `il % n_pattern < n_pattern - 1` (see
// `ggml-org/llama.cpp:src/llama-hparams.cpp:8-18` and
// `ggml-org/llama.cpp:src/models/gemma3.cpp:8`).
fn is_sliding_layer(layer_idx: usize, sliding_window_pattern: usize) -> bool {
    sliding_window_pattern == 0
        || (layer_idx % sliding_window_pattern) < sliding_window_pattern.saturating_sub(1)
}

fn gemma_norm_from_qtensor(
    qt: candle_core::quantized::QTensor,
    eps: f32,
    device: &Device,
) -> Result<RmsNorm> {
    // Gemma stores raw RmsNorm weights; the model adds 1.0 at runtime, matching
    // `Gemma3Model.norm_shift` in `ggml-org/llama.cpp:conversion/gemma.py:124`.
    let w = qt.dequantize(device)?;
    let w = (w + 1.0)?;
    RmsNorm::from_w(w, eps as f64)
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        let meta = ct.get_metadata();
        let metadata = ContentMetadata {
            path_prefix: "gemma3",
            metadata: meta,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            rope_freq_base_swa,
            head_dim,
            sliding_window,
            sliding_window_pattern,
            final_logit_softcapping,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let output_norm = gemma_norm_from_qtensor(
            ct.tensor("output_norm.weight", device)?,
            rms_norm_eps,
            device,
        )?;
        let output_qt = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        // Build one RoPE per (device, sliding-vs-global). Both use the same
        // `head_dim`; only the base frequency differs. Mirrors the safetensors
        // text core at `vision_models/gemma3/text.rs:428-460`.
        let mut sliding_ropes: HashMap<_, Arc<RotaryEmbedding>> = HashMap::new();
        let mut global_ropes: HashMap<_, Arc<RotaryEmbedding>> = HashMap::new();
        for layer_idx in 0..block_count {
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            sliding_ropes.entry(dev.location()).or_insert_with(|| {
                Arc::new(
                    RotaryEmbedding::new(
                        rope_freq_base_swa,
                        head_dim,
                        max_seq_len,
                        dev,
                        true,
                        DType::F32,
                    )
                    .expect("failed to build sliding RoPE"),
                )
            });
            global_ropes.entry(dev.location()).or_insert_with(|| {
                Arc::new(
                    RotaryEmbedding::new(
                        rope_freq_base,
                        head_dim,
                        max_seq_len,
                        dev,
                        true,
                        DType::F32,
                    )
                    .expect("failed to build global RoPE"),
                )
            });
        }

        let sliding_window_opt = if sliding_window == 0 {
            None
        } else {
            Some(sliding_window)
        };

        // Match the 27B special case in `ggml-org/llama.cpp:src/models/gemma3.cpp:31-33`:
        // 27B uses `1 / sqrt(n_embd / n_head)`, other sizes use `1 / sqrt(head_dim)`.
        let softmax_scale = if block_count == GEMMA3_27B_LAYER_COUNT {
            1.0 / ((embedding_length as f32) / (head_count as f32)).sqrt()
        } else {
            1.0 / (head_dim as f32).sqrt()
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            let is_sliding = is_sliding_layer(layer_idx, sliding_window_pattern);

            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), dev)?;
            let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), dev)?;
            let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), dev)?;
            let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), dev)?;

            let feed_forward_w1 = ct.tensor(&format!("{prefix}.ffn_gate.weight"), dev)?;
            let feed_forward_w2 = ct.tensor(&format!("{prefix}.ffn_down.weight"), dev)?;
            let feed_forward_w3 = ct.tensor(&format!("{prefix}.ffn_up.weight"), dev)?;

            let q_norm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;
            let k_norm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.attn_k_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;

            let input_layernorm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;
            let post_attention_layernorm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;
            let pre_feedforward_layernorm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;
            let post_feedforward_layernorm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.post_ffw_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;

            let rotary = if is_sliding {
                sliding_ropes
                    .get(&dev.location())
                    .expect("missing sliding RoPE for device")
                    .clone()
            } else {
                global_ropes
                    .get(&dev.location())
                    .expect("missing global RoPE for device")
                    .clone()
            };

            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, dev, None)?)
                }
            };

            layers.push(LayerWeights {
                attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wq),
                    b: None,
                })?),
                attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wk),
                    b: None,
                })?),
                attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wv),
                    b: None,
                })?),
                attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wo),
                    b: None,
                })?),
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                q_norm,
                k_norm,
                mlp: Mlp {
                    feed_forward_w1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w1),
                        b: None,
                    })?),
                    feed_forward_w2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w2),
                        b: None,
                    })?),
                    feed_forward_w3: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w3),
                        b: None,
                    })?),
                    act: HIDDEN_ACTIVATION,
                },
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary,
                is_sliding,
                sliding_window: if is_sliding { sliding_window_opt } else { None },
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale,
                    sliding_window: if is_sliding { sliding_window_opt } else { None },
                    sinks: None,
                },
                dtype,
                layer_idx,
            });
        }

        Ok(Self {
            embed_tokens: ScaledEmbedding::new(
                (embedding_length as f64).sqrt(),
                Embedding::new(tok_embeddings, embedding_length),
            ),
            layers,
            norm: output_norm,
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output_qt),
                b: None,
            })?),
            final_logit_softcapping,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
            sliding_window: sliding_window_opt,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(x)?;
        let cache = &mut self.cache.normal().0;

        let attention_mask = CausalMasker.make_causal_mask(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let attention_mask = if metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true)
        {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(attention_mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(attention_mask)
        };

        let sliding_mask_owned = if let Some(window) = self.sliding_window {
            let m = CausalMasker.make_causal_mask(
                x,
                metadata
                    .as_ref()
                    .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                    .unwrap_or(cache as &dyn PastKvLenCache),
                self.dtype,
                &CausalMaskConfig {
                    sliding_window: Some(window),
                    force_custom: false,
                },
            )?;
            let m = if metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
            {
                m
            } else {
                AttentionMask::None
            };
            let dm = if let Some(ref mapper) = self.mapper {
                DeviceMappedMask::new(m, &**mapper)?
            } else {
                DeviceMappedMask::from_single(m)
            };
            Some(dm)
        } else {
            None
        };
        let sliding_mask_ref: &DeviceMappedMask =
            sliding_mask_owned.as_ref().unwrap_or(&attention_mask);

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }

            let residual = xs.clone();
            let normed = layer.input_layernorm.forward(&xs)?;
            let attn_out = layer.forward_attn(
                &normed,
                &attention_mask.get(xs.device()),
                &sliding_mask_ref.get(xs.device()),
                seqlen_offsets,
                cache.as_mut_slice(),
                metadata
                    .as_ref()
                    .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta)),
            )?;
            xs = layer
                .post_attention_layernorm
                .forward_residual(&attn_out, &residual)?;

            let residual = xs.clone();
            let normed = xs.apply(&layer.pre_feedforward_layernorm)?;
            let mlp_out = layer.mlp.forward(&normed)?;
            xs = layer
                .post_feedforward_layernorm
                .forward_residual(&mlp_out, &residual)?;
        }

        let xs = self.norm.forward(&xs)?;
        let xs = extract_logits(&xs, context_lens)?;
        let mut logits = self.output.forward(&xs.contiguous()?)?;
        if let Some(softcap) = self.final_logit_softcapping {
            let scale = softcap as f32;
            logits = ((logits / scale as f64)?.tanh()? * scale as f64)?;
        }
        Ok(logits)
    }
}
