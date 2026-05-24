#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF loader for Qwen3.5 (Qwen3-Next) hybrid MoE models.
//!
//! Qwen3.5 uses a hybrid architecture:
//! - 75% of layers use Gated Delta Networks (linear attention, O(n))
//! - 25% of layers use standard multi-head attention with output gating
//! - All layers use Sparse MoE with shared experts
//!
//! This loader reads GGUF files with architecture "qwen35moe" and constructs
//! the model using the shared `deltanet::GatedDeltaNet` module for GDN layers.

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::SdpaParams;
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::models::deltanet::{GatedDeltaNet, GdnLayerCache, GdnProjection, RmsNormGated};
use crate::ops::{TopKLastDimOp, TopKOutput};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

const DEFAULT_MAX_SEQ_LEN: u32 = 4096;
const FULL_ATTENTION_INTERVAL: usize = 4;

fn is_full_attention_layer(layer_idx: usize) -> bool {
    (layer_idx + 1) % FULL_ATTENTION_INTERVAL == 0
}

// ====================== MoE ======================

struct FusedMoe {
    gate: QMatMul,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl FusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs.dtype();
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let TopKOutput {
            values: mut scores,
            indices,
        } = routing_weights.topk(self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            let xs = xs.to_dtype(DType::F32)?.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self.gate_experts.indexed_moe_forward(&xs, &indices)?;
            let up = self.up_experts.indexed_moe_forward(&xs, &indices)?;
            let activated = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
            self.down_experts
                .indexed_moe_forward(&activated, &indices)?
        };
        ys.broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((batch, seq_len, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

struct SharedExpert {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    shared_gate: candle_nn::Linear,
}

impl SharedExpert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward_autocast(xs)?;
        let up = self.up_proj.forward_autocast(xs)?;
        let mlp_out = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        let mlp_out = self.down_proj.forward_autocast(&mlp_out)?;
        // Sigmoid gating — shared_gate is [1, hidden] linear, outputs scalar per token
        let gate_val = candle_nn::ops::sigmoid(&self.shared_gate.forward(&xs.to_dtype(self.shared_gate.weight().dtype())?)?)?;
        gate_val.to_dtype(mlp_out.dtype())?.broadcast_mul(&mlp_out)
    }
}

struct MoeBlock {
    sparse_moe: FusedMoe,
    shared_expert: SharedExpert,
}

impl MoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let sparse_out = self.sparse_moe.forward(xs)?;
        let shared_out = self.shared_expert.forward(xs)?;
        sparse_out + shared_out
    }
}

/// Dense SwiGLU MLP (used by non-MoE Qwen3.5 variants like the 9B dense model).
struct DenseMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
}

impl DenseMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward_autocast(xs)?;
        let up = self.up_proj.forward_autocast(xs)?;
        let activated = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        self.down_proj.forward_autocast(&activated)
    }
}

/// Feed-forward block — either MoE (sparse + shared expert) or dense MLP.
enum FfnBlock {
    Moe(MoeBlock),
    Dense(DenseMlp),
}

impl FfnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            FfnBlock::Moe(moe) => moe.forward(xs),
            FfnBlock::Dense(mlp) => mlp.forward(xs),
        }
    }
}

// ====================== Full attention layer (with output gate) ======================

struct FullAttentionLayer {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    q_norm: Option<QRmsNorm>,
    k_norm: Option<QRmsNorm>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Qwen3.5 Q projection outputs 2*head_dim; second half is sigmoid gate for attention output.
    has_output_gate: bool,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl FullAttentionLayer {
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q_full = self.attention_wq.forward_autocast(x)?;
        let k = self.attention_wk.forward_autocast(x)?;
        let v = self.attention_wv.forward_autocast(x)?;

        // If Q projects to 2*head_dim per head, split into query and output gate.
        // Gate values are interleaved per head: reshape to (b, seq, n_head, 2*head_dim)
        // then narrow dim 3 to split each head's [q | gate].
        let (q, output_gate) = if self.has_output_gate {
            let q_gate = q_full.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;
            let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?
                .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
            let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?
                .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
            (q, Some(gate))
        } else {
            (q_full, None)
        };

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

        // Per-head RMSNorm (Qwen3.5 specific)
        let (q, k) = if let (Some(ref q_norm), Some(ref k_norm)) = (&self.q_norm, &self.k_norm) {
            let q_flat = q.flatten(0, 2)?;
            let k_flat = k.flatten(0, 2)?;
            let q_flat = q_norm.forward(&q_flat)?;
            let k_flat = k_norm.forward(&k_flat)?;
            let q = q_flat.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k_flat.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        let (q, k) = self.rotary.forward(&q, &k, start_offsets)?;

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let mut y = match &self.paged_attn {
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

        y = if mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        // Apply output gate: y = y * sigmoid(gate)
        if let Some(ref gate) = output_gate {
            y = (y * candle_nn::ops::sigmoid(gate)?)?;
        }

        let y = self.attention_wo.forward_autocast(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

// ====================== Decoder layer (hybrid) ======================

enum LayerImpl {
    FullAttention(FullAttentionLayer),
    LinearAttention(GatedDeltaNet),
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    attn_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    ffn: FfnBlock,
}

// ====================== Per-layer cache ======================

enum LocalLayerCache {
    Attention(KvCache),
    LinearAttention(GdnLayerCache),
}

struct LocalHybridCache {
    caches: Vec<LocalLayerCache>,
}

impl LocalHybridCache {
    fn seqlen(&self) -> usize {
        for cache in &self.caches {
            if let LocalLayerCache::Attention(kv) = cache {
                return kv.current_seq_len();
            }
        }
        0
    }
}

impl PastKvLenCache for LocalHybridCache {
    fn get_past_kv_len(&self) -> Result<usize> {
        Ok(self.seqlen())
    }
}

// ====================== Top-level model ======================

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    local_cache: std::sync::Mutex<LocalHybridCache>,
}

// ====================== GGUF metadata ======================

struct PropsGGUF {
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    embedding_length: usize,
    rms_norm_eps: f32,
    max_seq_len: usize,
    rope_freq_base: f32,
    head_dim: usize,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    // GDN-specific
    linear_num_k_heads: usize,
    linear_num_v_heads: usize,
    linear_head_k_dim: usize,
    linear_head_v_dim: usize,
    linear_conv_kernel_size: usize,
}

fn verify_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    if actual_arch != "qwen35moe" && actual_arch != "qwen35" {
        candle_core::bail!("Expected `qwen35moe` or `qwen35` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let head_count_kv = c.get_value::<u32>("attention.head_count_kv")? as usize;
        let head_dim = c
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(embed_len / head_count);

        // GDN linear attention head configuration.
        // ssm.state_size = per-head dim for both K and V in GDN (default 128)
        let ssm_state_size = c
            .get_value::<u32>("ssm.state_size")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(128);
        // ssm.group_count = num_k_heads (default 16, same as attention head_count)
        let linear_num_k_heads = c
            .get_value::<u32>("ssm.group_count")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(head_count);
        // num_v_heads = 2 * num_k_heads for Qwen3.5 (kv_group_size=2)
        let linear_num_v_heads = c
            .get_value::<u32>("attention.linear_head_count")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(linear_num_k_heads * 2);
        let linear_head_k_dim = c
            .get_value::<u32>("attention.linear_key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(ssm_state_size);
        let linear_head_v_dim = c
            .get_value::<u32>("attention.linear_value_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(ssm_state_size);
        let linear_conv_kernel_size = c
            .get_value::<u32>("ssm.conv_kernel")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(4);

        Ok(Self {
            head_count,
            head_count_kv,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_000_f32),
            head_dim,
            num_experts: c.get_value::<u32>("expert_count").ok().map(|x| x as usize),
            num_experts_per_tok: c.get_value::<u32>("expert_used_count").ok().map(|x| x as usize),
            linear_num_k_heads,
            linear_num_v_heads,
            linear_head_k_dim,
            linear_head_v_dim,
            linear_conv_kernel_size,
        })
    }
}

// ====================== GGUF loading ======================

fn gguf_matmul(tensor: candle_core::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(tensor),
        b: None,
    })?))
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
        let actual_arch = verify_arch(meta)?;

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
            metadata: meta,
        };
        let props = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output = if !ct.has_tensor("output.weight") {
            ct.tensor("token_embd.weight", device)?
        } else {
            ct.tensor("output.weight", device)?
        };

        let mut ropes = HashMap::new();
        for layer_idx in 0..props.block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    props.rope_freq_base,
                    props.head_dim,
                    props.max_seq_len,
                    device,
                    true,
                    dtype,
                )?),
            );
        }

        let key_dim = props.linear_num_k_heads * props.linear_head_k_dim;
        let value_dim = props.linear_num_v_heads * props.linear_head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let mut layers = Vec::with_capacity(props.block_count);
        let mut local_caches = Vec::with_capacity(props.block_count);

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..props.block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            // --- Layer implementation (attention or GDN) ---
            let layer_impl = if is_full_attention_layer(layer_idx) {
                // Full attention with output gate
                let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
                let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
                let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
                let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;

                // Check if Q projects to 2*head_dim (output gate)
                let q_out_features = attention_wq.shape().dims()[0];
                let expected_q_dim = props.head_count * props.head_dim;
                let has_output_gate = q_out_features == expected_q_dim * 2;

                let q_norm = if ct.has_tensor(&format!("{prefix}.attn_q_norm.weight")) {
                    Some(QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                        props.rms_norm_eps,
                    )?)
                } else {
                    None
                };
                let k_norm = if ct.has_tensor(&format!("{prefix}.attn_k_norm.weight")) {
                    Some(QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                        props.rms_norm_eps,
                    )?)
                } else {
                    None
                };

                let paged_attn = match &attention_mechanism {
                    AttentionImplementation::Eager => None,
                    AttentionImplementation::PagedAttention => {
                        Some(PagedAttention::new(props.head_dim, device, None)?)
                    }
                };

                local_caches.push(LocalLayerCache::Attention(KvCache::new_normal(
                    2,
                    props.max_seq_len,
                    64, // cache grow size
                )));

                LayerImpl::FullAttention(FullAttentionLayer {
                    attention_wq: gguf_matmul(attention_wq)?,
                    attention_wk: gguf_matmul(attention_wk)?,
                    attention_wv: gguf_matmul(attention_wv)?,
                    attention_wo: gguf_matmul(attention_wo)?,
                    q_norm,
                    k_norm,
                    n_head: props.head_count,
                    n_kv_head: props.head_count_kv,
                    head_dim: props.head_dim,
                    has_output_gate,
                    rotary,
                    paged_attn,
                    sdpa_params: SdpaParams {
                        n_kv_groups: props.head_count / props.head_count_kv,
                        softcap: None,
                        softmax_scale: 1.0 / (props.head_dim as f32).sqrt(),
                        sliding_window: None,
                        sinks: None,
                    },
                    dtype,
                })
            } else {
                // GDN linear attention layer
                let in_proj_qkv = ct.tensor(&format!("{prefix}.attn_qkv.weight"), device)?;
                let in_proj_z = ct.tensor(&format!("{prefix}.attn_gate.weight"), device)?;
                let ssm_alpha = ct.tensor(&format!("{prefix}.ssm_alpha.weight"), device)?;
                let ssm_beta = ct.tensor(&format!("{prefix}.ssm_beta.weight"), device)?;
                let conv1d_weight = ct.tensor(&format!("{prefix}.ssm_conv1d.weight"), device)?;
                let ssm_a = ct.tensor(&format!("{prefix}.ssm_a"), device)?;
                let dt_bias_t = ct.tensor(&format!("{prefix}.ssm_dt.bias"), device)?;
                let ssm_norm = ct.tensor(&format!("{prefix}.ssm_norm.weight"), device)?;
                let out_proj = ct.tensor(&format!("{prefix}.ssm_out.weight"), device)?;

                // Convert ssm_a: GGUF stores -exp(A_log), recover A_log = log(-ssm_a)
                let ssm_a_deq = ssm_a.dequantize(device)?;
                let a_log = ssm_a_deq.neg()?.log()?.to_dtype(dtype)?;
                let dt_bias = dt_bias_t.dequantize(device)?.to_dtype(dtype)?;
                let conv_w = conv1d_weight.dequantize(device)?.to_dtype(dtype)?;
                let norm_w = ssm_norm.dequantize(device)?.to_dtype(dtype)?;

                // GGUF stores V-heads in tiled layout for both MoE and dense Qwen3.5.
                let gdn = GatedDeltaNet {
                    projection: GdnProjection::SplitQkvZa {
                        in_proj_qkv: gguf_matmul(in_proj_qkv)?,
                        in_proj_z: gguf_matmul(in_proj_z)?,
                        in_proj_b: gguf_matmul(ssm_beta)?,
                        in_proj_a: gguf_matmul(ssm_alpha)?,
                    },
                    conv1d_weight: conv_w,
                    dt_bias,
                    a_log,
                    norm: RmsNormGated {
                        weight: norm_w,
                        eps: props.rms_norm_eps as f64,
                    },
                    out_proj: gguf_matmul(out_proj)?,
                    num_k_heads: props.linear_num_k_heads,
                    num_v_heads: props.linear_num_v_heads,
                    head_k_dim: props.linear_head_k_dim,
                    head_v_dim: props.linear_head_v_dim,
                    conv_kernel_size: props.linear_conv_kernel_size,
                    key_dim,
                    value_dim,
                    tiled_v_heads: true,
                };

                let cache = GdnLayerCache {
                    conv_state: Tensor::zeros(
                        (1, conv_dim, props.linear_conv_kernel_size),
                        dtype,
                        device,
                    )?,
                    recurrent_state: Tensor::zeros(
                        (1, props.linear_num_v_heads, props.linear_head_k_dim, props.linear_head_v_dim),
                        dtype,
                        device,
                    )?,
                    seqlen_offset: 0,
                };
                local_caches.push(LocalLayerCache::LinearAttention(cache));

                LayerImpl::LinearAttention(gdn)
            };

            // --- Norms ---
            let attn_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                props.rms_norm_eps,
            )?;
            let ffn_norm_tensor_name = if ct.has_tensor(&format!("{prefix}.ffn_norm.weight")) {
                format!("{prefix}.ffn_norm.weight")
            } else {
                format!("{prefix}.post_attention_norm.weight")
            };
            let ffn_norm = QRmsNorm::new(
                ct.tensor(&ffn_norm_tensor_name, device)?,
                props.rms_norm_eps,
            )?;

            // --- FFN: MoE or dense MLP ---
            let ffn = if ct.has_tensor(&format!("{prefix}.ffn_gate_inp.weight")) {
                // MoE variant (Qwen3.5 MoE models)
                let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let gate_experts = ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?;
                let up_experts = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;
                let down_experts = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;

                let sparse_moe = FusedMoe {
                    gate: QMatMul::from_qtensor(gate)?,
                    gate_experts: QMatMul::from_qtensor(gate_experts)?,
                    up_experts: QMatMul::from_qtensor(up_experts)?,
                    down_experts: QMatMul::from_qtensor(down_experts)?,
                    norm_topk_prob: true,
                    num_experts_per_tok: props.num_experts_per_tok.unwrap_or(4),
                };

                // Shared expert
                let shared_gate_proj = ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), device)?;
                let shared_up_proj = ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), device)?;
                let shared_down_proj = ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), device)?;
                let shared_gate_qt = ct.tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"), device)?;
                let shared_gate_w = shared_gate_qt.dequantize(device)?
                    .reshape((1, props.embedding_length))?;
                let shared_gate = candle_nn::Linear::new(shared_gate_w, None);

                FfnBlock::Moe(MoeBlock {
                    sparse_moe,
                    shared_expert: SharedExpert {
                        gate_proj: gguf_matmul(shared_gate_proj)?,
                        up_proj: gguf_matmul(shared_up_proj)?,
                        down_proj: gguf_matmul(shared_down_proj)?,
                        shared_gate,
                    },
                })
            } else {
                // Dense MLP variant (Qwen3.5 dense models like 9B)
                let gate_proj = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
                let up_proj = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                let down_proj = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;

                FfnBlock::Dense(DenseMlp {
                    gate_proj: gguf_matmul(gate_proj)?,
                    up_proj: gguf_matmul(up_proj)?,
                    down_proj: gguf_matmul(down_proj)?,
                })
            };

            layers.push(DecoderLayer {
                layer_impl,
                attn_norm,
                ffn_norm,
                ffn,
            });
        }

        let local_cache = LocalHybridCache {
            caches: local_caches,
        };

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, props.embedding_length),
            layers,
            norm,
            output: gguf_matmul(output)?,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(props.block_count, props.max_seq_len)),
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
            local_cache: std::sync::Mutex::new(local_cache),
        })
    }
}

// ====================== Cache management ======================

impl ModelWeights {
    /// Clear the local hybrid cache (GDN recurrent state + attention KV).
    /// Called by the pipeline between requests to free GPU memory.
    pub fn clear_local_cache(&self) {
        let mut local_cache = self.local_cache.lock().unwrap();
        // Free attention KV caches FIRST (these are large, grow with seq_len)
        for cache in &mut local_cache.caches {
            if let LocalLayerCache::Attention(kv_cache) = cache {
                kv_cache.reset();
            }
        }
        // Then reset GDN caches (small fixed-size, zeros_like allocates before freeing)
        for cache in &mut local_cache.caches {
            if let LocalLayerCache::LinearAttention(gdn_cache) = cache {
                let _ = gdn_cache.reset();
            }
        }
    }
}

// ====================== Forward pass ======================

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?.to_dtype(self.dtype)?;
        let mut local_cache = self.local_cache.lock().unwrap();

        // Reset ALL caches on new sequence (both GDN recurrent state and attention KV).
        // Free attention KV first (large), then GDN (small but zeros_like allocates).
        if start_offsets[0] == 0 {
            for cache in &mut local_cache.caches {
                if let LocalLayerCache::Attention(kv_cache) = cache {
                    kv_cache.reset();
                }
            }
            for cache in &mut local_cache.caches {
                if let LocalLayerCache::LinearAttention(gdn_cache) = cache {
                    gdn_cache.reset()?;
                }
            }
        }

        // Build causal mask for full attention layers
        let mask = CausalMasker.make_causal_mask_matrix(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*local_cache as &dyn PastKvLenCache),
            self.dtype,
            self.layers[0]
                .layer_impl
                .n_head_for_mask()
                .unwrap_or(1),
        )?;
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

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let residual = &layer_in;
            let x = layer.attn_norm.forward(&layer_in)?;

            let attn_out = match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    if let LocalLayerCache::Attention(ref mut kv_cache) = local_cache.caches[i] {
                        attn.forward(
                            &x,
                            mask.as_ref().map(|m| m.get(x.device())),
                            start_offsets,
                            kv_cache,
                            metadata
                                .as_ref()
                                .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                        ).map_err(|e| candle_core::Error::Msg(format!("layer {i} full_attn: {e}")))?
                    } else {
                        candle_core::bail!("Expected KV cache for full attention layer {i}");
                    }
                }
                LayerImpl::LinearAttention(gdn) => {
                    if let LocalLayerCache::LinearAttention(ref mut gdn_cache) = local_cache.caches[i]
                    {
                        gdn.forward(&x, gdn_cache)
                            .map_err(|e| candle_core::Error::Msg(format!("layer {i} gdn: {e}")))?
                    } else {
                        candle_core::bail!("Expected GDN cache for linear attention layer {i}");
                    }
                }
            };

            let x = (attn_out + residual)
                .map_err(|e| candle_core::Error::Msg(format!("layer {i} attn_residual: {e}")))?;
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.ffn.forward(&x)
                .map_err(|e| candle_core::Error::Msg(format!("layer {i} ffn: {e}")))?;
            layer_in = (x + residual)
                .map_err(|e| candle_core::Error::Msg(format!("layer {i} moe_residual: {e}")))?;
        }

        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        let result = self.output.forward_autocast(&x.contiguous()?)?;
        Ok(result)
    }
}

impl LayerImpl {
    fn n_head_for_mask(&self) -> Option<usize> {
        match self {
            LayerImpl::FullAttention(attn) => Some(attn.n_head),
            LayerImpl::LinearAttention(_) => None,
        }
    }
}
