#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::SdpaParams;
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMasker, MatMul, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::extract_logits;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::EitherCache;
use crate::pipeline::KvCache;
use crate::pipeline::NormalCache;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
// Default fallback for models that don't specify context_length
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w1)?;
        let w3 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w3)?;
        let y = crate::ops::mul_and_act(&w1, &w3, crate::layers::Activation::Silu)?;
        MatMul.qmethod_matmul(&y, &*self.feed_forward_w2)
    }
}

enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: Arc<dyn QuantMethod>,
        experts: Vec<Mlp>,
    },
}

impl MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = MatMul.qmethod_matmul(&xs, &**feed_forward_gate_inp)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    attention_post_norm: QRmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: QRmsNorm,
    ffn_post_norm: QRmsNorm,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary_global: Arc<RotaryEmbedding>,
    rotary_local: Arc<RotaryEmbedding>,
    use_sliding_window: bool,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = MatMul
            .qmethod_matmul(x, &*self.attention_wq)?
            .to_dtype(self.dtype)?;
        let k = MatMul
            .qmethod_matmul(x, &*self.attention_wk)?
            .to_dtype(self.dtype)?;
        let v = MatMul
            .qmethod_matmul(x, &*self.attention_wv)?
            .to_dtype(self.dtype)?;

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
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        let k = k_flat.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;

        let (q, k) = if self.use_sliding_window {
            self.rotary_local.forward(&q, &k, start_offsets)?
        } else {
            self.rotary_global.forward(&q, &k, start_offsets)?
        };
        let mask = if self.use_sliding_window {
            sliding_attention_mask
        } else {
            attention_mask
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

        let y = MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attention_wo)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

// gemma3 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
pub(crate) struct PropsGGUF {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rope_dim: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub rope_local_freq_base: f32,
    pub sliding_window: usize,
    pub sliding_window_pattern: Vec<bool>,
    pub sliding_window_pattern_interval: usize,
    pub causal: bool,
    pub query_pre_attn_scalar: Option<f32>,
    pub key_length: usize,
    pub value_length: usize,
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
            "rope.dimension_count",
            "attention.layer_norm_rms_epsilon",
            "attention.sliding_window",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec
        let sliding_window_pattern = c
            .get_value::<Vec<bool>>("attention.sliding_window_pattern")
            .ok()
            .unwrap_or_default();
        let sliding_window_pattern_interval = c
            .get_value::<u32>("attention.sliding_window_pattern")
            .ok()
            .unwrap_or(6) as usize;

        let props = Self {
            n_expert: c.get_value::<u32>("expert_count").ok().unwrap_or(0) as usize,
            n_expert_used: c.get_value::<u32>("expert_used_count").ok().unwrap_or(0) as usize,
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            rope_local_freq_base: c
                .get_value("rope.local.freq_base")
                .ok()
                .unwrap_or(10_000_f32),
            sliding_window: c.get_value::<u32>("attention.sliding_window")? as usize,
            sliding_window_pattern,
            sliding_window_pattern_interval,
            causal: c.get_value::<bool>("attention.causal").ok().unwrap_or(true),
            query_pre_attn_scalar: c.get_value("attention.query_pre_attn_scalar").ok(),
            key_length: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
        };

        Ok(props)
    }
}

fn gemma3_use_sliding_window(
    layer_idx: usize,
    sliding_window_pattern: &[bool],
    sliding_window_pattern_interval: usize,
) -> bool {
    if !sliding_window_pattern.is_empty() {
        return sliding_window_pattern[layer_idx % sliding_window_pattern.len()];
    }
    !(layer_idx + 1).is_multiple_of(sliding_window_pattern_interval)
}

fn ensure_gemma3_causal_mode(causal: bool) -> Result<()> {
    if !causal {
        candle_core::bail!(
            "Unsupported Gemma 3 GGUF variant: `attention.causal=false` (embedding/non-causal). Only causal text-generation Gemma 3 GGUF is supported."
        );
    }
    Ok(())
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        // Choose GGUF path prefix based on architecture so tensor names resolve.
        let path_prefix = "gemma3";

        let metadata = ContentMetadata {
            path_prefix,
            metadata: ct.get_metadata(),
        };
        let PropsGGUF {
            n_expert,
            n_expert_used,
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rope_dim,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            rope_local_freq_base,
            sliding_window,
            sliding_window_pattern,
            sliding_window_pattern_interval,
            causal,
            query_pre_attn_scalar,
            key_length,
            value_length,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;
        ensure_gemma3_causal_mode(causal)?;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if !ct.has_tensor("output.weight") {
            ct.tensor("token_embd.weight", device)?
        } else {
            ct.tensor("output.weight", device)?
        };
        let mut layers = Vec::with_capacity(block_count);

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        let mut global_ropes = HashMap::new();
        let mut local_ropes = HashMap::new();
        for layer_idx in 0..block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            global_ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    rope_dim,
                    max_seq_len,
                    device,
                    false,
                    dtype,
                )?),
            );
            local_ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_local_freq_base,
                    rope_dim,
                    max_seq_len,
                    device,
                    false,
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
            let rotary_global = global_ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let rotary_local = local_ropes
                .get(&device.location())
                .expect("No local RoPE for device location!")
                .clone();

            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;
            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
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
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                match ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device) {
                    Ok(feed_forward_gate_exps) => {
                        let feed_forward_down_exps =
                            ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
                        let feed_forward_up_exps =
                            ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;

                        let dequant_ffn_gate = feed_forward_gate_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;
                        let dequant_ffn_down = feed_forward_down_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;
                        let dequant_ffn_up = feed_forward_up_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;

                        assert_eq!(dequant_ffn_up.len(), dequant_ffn_down.len());
                        assert_eq!(dequant_ffn_gate.len(), dequant_ffn_down.len());
                        assert_eq!(dequant_ffn_gate.len(), n_expert);

                        let gate_type = feed_forward_gate_exps.dtype();
                        let down_type = feed_forward_down_exps.dtype();
                        let up_type = feed_forward_up_exps.dtype();

                        for (ff_w1, (ff_w2, ff_w3)) in dequant_ffn_gate
                            .into_iter()
                            .zip(dequant_ffn_down.into_iter().zip(dequant_ffn_up))
                        {
                            experts.push(Mlp {
                                feed_forward_w1: Arc::new(GgufMatMul::new(
                                    QuantMethodConfig::Gguf {
                                        q_weight: Arc::new(QTensor::quantize(&ff_w1, gate_type)?),
                                        b: None,
                                    },
                                )?),
                                feed_forward_w2: Arc::new(GgufMatMul::new(
                                    QuantMethodConfig::Gguf {
                                        q_weight: Arc::new(QTensor::quantize(&ff_w2, down_type)?),
                                        b: None,
                                    },
                                )?),
                                feed_forward_w3: Arc::new(GgufMatMul::new(
                                    QuantMethodConfig::Gguf {
                                        q_weight: Arc::new(QTensor::quantize(&ff_w3, up_type)?),
                                        b: None,
                                    },
                                )?),
                            })
                        }
                    }
                    Err(_) => {
                        for i in 0..n_expert {
                            let feed_forward_w1 =
                                ct.tensor(&format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                            let feed_forward_w2 =
                                ct.tensor(&format!("{prefix}.ffn_down.{i}.weight"), device)?;
                            let feed_forward_w3 =
                                ct.tensor(&format!("{prefix}.ffn_up.{i}.weight"), device)?;
                            experts.push(Mlp {
                                feed_forward_w1: Arc::new(GgufMatMul::new(
                                    QuantMethodConfig::Gguf {
                                        q_weight: Arc::new(feed_forward_w1),
                                        b: None,
                                    },
                                )?),
                                feed_forward_w2: Arc::new(GgufMatMul::new(
                                    QuantMethodConfig::Gguf {
                                        q_weight: Arc::new(feed_forward_w2),
                                        b: None,
                                    },
                                )?),
                                feed_forward_w3: Arc::new(GgufMatMul::new(
                                    QuantMethodConfig::Gguf {
                                        q_weight: Arc::new(feed_forward_w3),
                                        b: None,
                                    },
                                )?),
                            })
                        }
                    }
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_gate_inp),
                        b: None,
                    })?),
                    experts,
                }
            };
            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let attention_post_norm =
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
            let ffn_post_norm = ct.tensor(&format!("{prefix}.post_ffw_norm.weight"), device)?;
            let q_norm = ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?;
            let k_norm = ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?;
            let use_sliding_window = gemma3_use_sliding_window(
                layer_idx,
                &sliding_window_pattern,
                sliding_window_pattern_interval,
            );
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
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
                attention_norm: QRmsNorm::new(attention_norm, rms_norm_eps)?,
                attention_post_norm: QRmsNorm::new(attention_post_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: QRmsNorm::new(ffn_norm, rms_norm_eps)?,
                ffn_post_norm: QRmsNorm::new(ffn_post_norm, rms_norm_eps)?,
                q_norm: QRmsNorm::new(q_norm, rms_norm_eps)?,
                k_norm: QRmsNorm::new(k_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary_global: rotary_global.clone(),
                rotary_local: rotary_local.clone(),
                use_sliding_window,
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / query_pre_attn_scalar.unwrap_or(head_dim as f32).sqrt(),
                    sliding_window: use_sliding_window.then_some(sliding_window),
                    sinks: None,
                },
                dtype,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output),
                b: None,
            })?),
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask_matrix(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            self.layers[0].n_head,
        )?;
        // PagedAttention prompt chunking
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        let sliding_window = self
            .layers
            .iter()
            .find_map(|layer| layer.sdpa_params.sliding_window);
        let sliding_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            sliding_window,
            self.dtype,
            self.layers[0].n_head,
        )?;
        let sliding_mask = sliding_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let sliding_mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(sliding_mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(sliding_mask)
        };
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &x,
                mask.as_ref().map(|m| m.get(x.device())),
                sliding_mask.as_ref().map(|m| m.get(x.device())),
                start_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let x = layer.attention_post_norm.forward(&attn)?;
            let x = (x + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = layer.ffn_post_norm.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
        }
        let layer_in = layer_in.to_device(&self.device)?;
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::{ensure_gemma3_causal_mode, gemma3_use_sliding_window};

    #[test]
    fn sliding_window_pattern_bool_array_is_honored() {
        let pattern = vec![true, true, true, true, true, false];
        let got: Vec<bool> = (0..12)
            .map(|idx| gemma3_use_sliding_window(idx, &pattern, 99))
            .collect();
        let expected = vec![
            true, true, true, true, true, false, true, true, true, true, true, false,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn sliding_window_pattern_interval_fallback_is_used_when_bool_array_missing() {
        let got: Vec<bool> = (0..6)
            .map(|idx| gemma3_use_sliding_window(idx, &[], 6))
            .collect();
        let expected = vec![true, true, true, true, true, false];
        assert_eq!(got, expected);
    }

    #[test]
    fn non_causal_gemma3_is_rejected() {
        let err = ensure_gemma3_causal_mode(false).expect_err("non-causal must be rejected");
        assert!(err.to_string().contains("attention.causal=false"));
    }
}
