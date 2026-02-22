#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer, MoeMlp},
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType},
    layers::{embedding, CausalMasker, MatMul, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, word_emb_default, true);
serde_default_fn!(f32, default_one, 1.0);
serde_default_fn!(f32, default_rope_theta, 10_000.0);
serde_default_fn!(usize, default_mamba_d_conv, 4);
serde_default_fn!(usize, default_mamba_d_state, 256);
serde_default_fn!(usize, default_mamba_expand, 2);
serde_default_fn!(usize, default_mamba_n_groups, 1);
serde_default_fn!(usize, default_mamba_chunk_size, 256);
serde_default_fn!(bool, default_mamba_conv_bias, true);
serde_default_fn!(bool, default_mamba_proj_bias, false);
serde_default_fn!(usize, default_num_local_experts, 0);
serde_default_fn!(usize, default_num_experts_per_tok, 2);
serde_default_fn!(String, default_position_embedding_type, "rope".to_string());

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GraniteLayerType {
    #[default]
    Attention,
    Mamba,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GraniteRopeType {
    #[default]
    Default,
    Granite,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GraniteRopeConfig {
    pub factor: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rope_type: GraniteRopeType,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<GraniteRopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Vec<GraniteLayerType>,
    #[serde(default = "default_one")]
    pub attention_multiplier: f32,
    #[serde(default = "default_one")]
    pub embedding_multiplier: f32,
    #[serde(default = "default_one")]
    pub residual_multiplier: f32,
    #[serde(default = "default_one")]
    pub logits_scaling: f32,
    // Mamba configuration
    pub mamba_n_heads: Option<usize>,
    #[serde(default = "default_mamba_n_groups")]
    pub mamba_n_groups: usize,
    #[serde(default = "default_mamba_d_state")]
    pub mamba_d_state: usize,
    pub mamba_d_head: Option<usize>,
    #[serde(default = "default_mamba_d_conv")]
    pub mamba_d_conv: usize,
    #[serde(default = "default_mamba_expand")]
    pub mamba_expand: usize,
    #[serde(default = "default_mamba_chunk_size")]
    pub mamba_chunk_size: usize,
    #[serde(default = "default_mamba_conv_bias")]
    pub mamba_conv_bias: bool,
    #[serde(default = "default_mamba_proj_bias")]
    pub mamba_proj_bias: bool,
    // MoE configuration
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    // Position embedding type: "rope" or "nope" (no position embedding)
    #[serde(default = "default_position_embedding_type")]
    pub position_embedding_type: String,
}

impl Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn shared_intermediate_size(&self) -> usize {
        self.shared_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn layer_types(&self) -> Vec<GraniteLayerType> {
        if self.layer_types.is_empty() {
            vec![GraniteLayerType::Attention; self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        }
    }

    // Mamba helper methods
    pub fn mamba_intermediate_size(&self) -> usize {
        self.mamba_expand * self.hidden_size
    }

    pub fn mamba_n_heads(&self) -> usize {
        self.mamba_n_heads.unwrap_or(128)
    }

    pub fn mamba_d_head(&self) -> usize {
        self.mamba_d_head
            .unwrap_or(self.mamba_intermediate_size() / self.mamba_n_heads())
    }

    pub fn mamba_conv_dim(&self) -> usize {
        self.mamba_intermediate_size() + 2 * self.mamba_n_groups * self.mamba_d_state
    }
}

/// GraniteMLP uses a fused gate-up projection followed by output projection
/// Input: shared_mlp.input_linear (hidden -> shared_intermediate * 2)
/// Output: shared_mlp.output_linear (shared_intermediate -> hidden)
#[derive(Clone)]
pub struct GraniteMlp {
    input_linear: Arc<dyn QuantMethod>,
    output_linear: Arc<dyn QuantMethod>,
    params: Vec<usize>,
}

impl GraniteMlp {
    pub fn new(
        vb: ShardedVarBuilder,
        cfg: &Config,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let shared_intermediate_size = cfg.shared_intermediate_size();
        let input_linear = ColumnParallelLayer::new(
            cfg.hidden_size,
            shared_intermediate_size * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("shared_mlp").pp("input_linear"),
        )?;
        let output_linear = RowParallelLayer::new(
            shared_intermediate_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("shared_mlp").pp("output_linear"),
        )?;
        Ok(Self {
            input_linear,
            output_linear,
            params: vec![cfg.hidden_size, shared_intermediate_size],
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.input_linear.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let projected = MatMul.qmethod_matmul(&x, &*self.input_linear)?;
        let chunks = projected.chunk(2, candle_core::D::Minus1)?;
        let gated =
            crate::ops::mul_and_act(&chunks[0], &chunks[1], crate::layers::Activation::Silu)?;
        let mut res = MatMul.qmethod_matmul(&gated, &*self.output_linear)?;
        if self.input_linear.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

impl MlpLayer for GraniteMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.input_linear, &mut self.output_linear]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Self {
            input_linear: self.input_linear.clone(),
            output_linear: self.output_linear.clone(),
            params: self.params.clone(),
        })
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    fn hidden_act(&self) -> crate::layers::Activation {
        crate::layers::Activation::Silu
    }
    fn new_added_delta(&self, _deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        candle_core::bail!("LoRA adapter not supported for GraniteMlp")
    }
    fn dtype_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.input_linear.dtype_and_device()
    }
}

impl crate::amoe::AnyMoeTrainableLayer for GraniteMlp {}

// ====================== MoE (Mixture of Experts) Implementation ======================

/// Top-K gating router for sparse MoE
struct GraniteTopKGating {
    layer: candle_nn::Linear,
    num_experts: usize,
    top_k: usize,
}

impl GraniteTopKGating {
    fn new(
        input_size: usize,
        num_experts: usize,
        top_k: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let weight = vb.pp("layer").get((num_experts, input_size), "weight")?;
        let layer = candle_nn::Linear::new(weight, None);
        Ok(Self {
            layer,
            num_experts,
            top_k,
        })
    }

    /// Routes tokens to experts
    /// Returns (batch_index, batch_gates, expert_size)
    /// - batch_index: indices of tokens assigned to each expert (sorted by expert)
    /// - batch_gates: routing weights for each token-expert pair (sorted by expert)
    /// - expert_size: number of tokens assigned to each expert
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let (num_tokens, _) = x.dims2()?;
        let device = x.device();
        let dtype = x.dtype();

        // Compute routing logits: (num_tokens, num_experts)
        let logits = self.layer.forward(x)?;

        // Softmax over experts
        let gates = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;

        // Get top-k expert indices and gates per token
        let gates_vec: Vec<f32> = gates
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        // Collect (expert_idx, token_idx, gate) tuples
        let mut expert_token_gates: Vec<(usize, usize, f32)> = Vec::new();
        let mut expert_counts = vec![0usize; self.num_experts];

        for token_idx in 0..num_tokens {
            // Get gates for this token
            let start = token_idx * self.num_experts;
            let end = start + self.num_experts;
            let token_gates: Vec<(usize, f32)> = gates_vec[start..end]
                .iter()
                .enumerate()
                .map(|(i, &g)| (i, g))
                .collect();

            // Sort by gate value and take top-k
            let mut sorted: Vec<(usize, f32)> = token_gates;
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let selected: Vec<(usize, f32)> = sorted.into_iter().take(self.top_k).collect();

            // Normalize selected gates
            let sum: f32 = selected.iter().map(|(_, g)| g).sum();
            let normalized: Vec<(usize, f32)> = selected
                .iter()
                .map(|(e, g)| (*e, if sum > 0.0 { g / sum } else { 0.0 }))
                .collect();

            for (expert_idx, gate) in normalized {
                expert_token_gates.push((expert_idx, token_idx, gate));
                expert_counts[expert_idx] += 1;
            }
        }

        // Sort by expert index so tokens are grouped by expert
        expert_token_gates.sort_by_key(|(expert_idx, _, _)| *expert_idx);

        // Extract sorted batch_index and batch_gates
        let all_batch_indices: Vec<u32> = expert_token_gates
            .iter()
            .map(|(_, token_idx, _)| *token_idx as u32)
            .collect();
        let all_batch_gates: Vec<f32> = expert_token_gates
            .iter()
            .map(|(_, _, gate)| *gate)
            .collect();

        let indices_len = all_batch_indices.len();
        let gates_len = all_batch_gates.len();
        let batch_index = Tensor::from_vec(all_batch_indices, (indices_len,), device)?;
        let batch_gates =
            Tensor::from_vec(all_batch_gates, (gates_len,), device)?.to_dtype(dtype)?;

        Ok((batch_index, batch_gates, expert_counts))
    }
}

/// Parallel experts layer - processes all experts in a batched manner
struct GraniteParallelExperts {
    weights: Vec<Tensor>,
    output_size: usize,
}

impl GraniteParallelExperts {
    fn new(
        num_experts: usize,
        input_size: usize,
        output_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let all_weights = vb.get((num_experts, output_size, input_size), "weight")?;
        let weights = (0..num_experts)
            .map(|i| all_weights.i(i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights,
            output_size,
        })
    }

    fn forward(&self, x: &Tensor, expert_size: &[usize]) -> Result<Tensor> {
        let dtype = x.dtype();
        let device = x.device();

        let mut outputs = Vec::new();
        let mut offset = 0;

        for (expert_idx, &size) in expert_size.iter().enumerate() {
            if size == 0 {
                continue;
            }
            let expert_input = x.narrow(0, offset, size)?;
            let expert_output = expert_input.matmul(&self.weights[expert_idx].t()?)?;
            outputs.push(expert_output);
            offset += size;
        }

        if outputs.is_empty() {
            Tensor::zeros((0, self.output_size), dtype, device)
        } else {
            Tensor::cat(&outputs, 0)
        }
    }
}

/// Sparse Mixture of Experts layer
struct GraniteMoE {
    input_linear: GraniteParallelExperts,
    output_linear: GraniteParallelExperts,
    router: GraniteTopKGating,
    input_size: usize,
}

impl GraniteMoE {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let input_size = cfg.hidden_size;
        let hidden_size = cfg.intermediate_size;
        let num_experts = cfg.num_local_experts;
        let top_k = cfg.num_experts_per_tok;

        Ok(Self {
            input_linear: GraniteParallelExperts::new(
                num_experts,
                input_size,
                hidden_size * 2, // Gated
                vb.pp("input_linear"),
            )?,
            output_linear: GraniteParallelExperts::new(
                num_experts,
                hidden_size,
                input_size,
                vb.pp("output_linear"),
            )?,
            router: GraniteTopKGating::new(input_size, num_experts, top_k, vb.pp("router"))?,
            input_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, emb_size) = x.dims3()?;
        let dtype = x.dtype();
        let device = x.device();
        let num_tokens = batch_size * seq_len;

        let x_flat = x.reshape((num_tokens, emb_size))?;
        let (batch_index, batch_gates, expert_size) = self.router.forward(&x_flat)?;

        if batch_index.dim(0)? == 0 {
            return Tensor::zeros((batch_size, seq_len, self.input_size), dtype, device);
        }

        // Route tokens through experts
        let expert_inputs = x_flat.index_select(&batch_index, 0)?;
        let hidden = self.input_linear.forward(&expert_inputs, &expert_size)?;

        // Gated activation: silu(first_half) * second_half
        let chunks = hidden.chunk(2, candle_core::D::Minus1)?;
        let hidden =
            crate::ops::mul_and_act(&chunks[0], &chunks[1], crate::layers::Activation::Silu)?;

        let expert_outputs = self.output_linear.forward(&hidden, &expert_size)?;
        let expert_outputs = expert_outputs.broadcast_mul(&batch_gates.unsqueeze(1)?)?;

        // Scatter-add outputs back to token positions
        let batch_index_vec: Vec<i64> = batch_index.to_dtype(candle_core::DType::I64)?.to_vec1()?;
        let expert_outputs_f32 = expert_outputs.to_dtype(candle_core::DType::F32)?;
        let num_outputs = expert_outputs_f32.dim(0)?;

        let expert_outputs_vec: Vec<Vec<f32>> = (0..num_outputs)
            .map(|i| expert_outputs_f32.i(i)?.to_vec1())
            .collect::<Result<Vec<_>>>()?;

        let mut output_vec = vec![vec![0.0f32; self.input_size]; batch_size * seq_len];
        for (i, &token_idx) in batch_index_vec.iter().enumerate() {
            let token_idx = token_idx as usize;
            for (j, &val) in expert_outputs_vec[i].iter().enumerate() {
                output_vec[token_idx][j] += val;
            }
        }

        let flat_output: Vec<f32> = output_vec.into_iter().flatten().collect();
        Tensor::from_vec(flat_output, (num_tokens, self.input_size), device)?
            .to_dtype(dtype)?
            .reshape((batch_size, seq_len, self.input_size))
    }
}

// ====================== Mamba Implementation ======================

/// Per-layer Mamba state cache (local to granite model).
/// Stores conv state and SSM state for recurrent processing.
#[derive(Debug)]
struct MambaLayerCache {
    /// Convolution state: (batch, conv_dim, d_conv)
    pub conv_state: Tensor,
    /// SSM state: (batch, n_heads, head_dim, d_state)
    pub ssm_state: Tensor,
    /// Current sequence length offset for this layer
    pub seqlen_offset: usize,
}

impl MambaLayerCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_size: usize,
        conv_dim: usize,
        d_conv: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        dtype: candle_core::DType,
        device: &Device,
    ) -> Result<Self> {
        let conv_state = Tensor::zeros((batch_size, conv_dim, d_conv), dtype, device)?;
        let ssm_state = Tensor::zeros((batch_size, n_heads, head_dim, d_state), dtype, device)?;
        Ok(Self {
            conv_state,
            ssm_state,
            seqlen_offset: 0,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.ssm_state = self.ssm_state.zeros_like()?;
        self.seqlen_offset = 0;
        Ok(())
    }
}

impl Clone for MambaLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
            seqlen_offset: self.seqlen_offset,
        }
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

fn create_mamba_cache(
    batch_size: usize,
    cfg: &Config,
    dtype: candle_core::DType,
    device: &Device,
) -> Result<MambaLayerCache> {
    let conv_dim = cfg.mamba_conv_dim();
    MambaLayerCache::new(
        batch_size,
        conv_dim,
        cfg.mamba_d_conv,
        cfg.mamba_n_heads(),
        cfg.mamba_d_head(),
        cfg.mamba_d_state,
        dtype,
        device,
    )
}

/// RMSNorm with optional gating (for Mamba output)
struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    fn new(
        hidden_size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let mut weight = vb.get((hidden_size,), "weight")?;
        // Move weight to target device for ISQ compatibility
        if let Some(target_dev) = isq_target_device {
            weight = weight.to_device(target_dev)?;
        }
        Ok(Self { weight, eps })
    }

    fn forward(&self, hidden_states: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let mut hidden_states = hidden_states.to_dtype(candle_core::DType::F32)?;

        // Apply gating if provided
        if let Some(gate) = gate {
            let gate = candle_nn::ops::silu(&gate.to_dtype(candle_core::DType::F32)?)?;
            hidden_states = hidden_states.broadcast_mul(&gate)?;
        }

        // RMS normalization
        let variance = hidden_states.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let hidden_states = hidden_states.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        // Apply weight and convert back to original dtype
        hidden_states
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight.to_dtype(dtype)?)
    }
}

/// Mamba2-style mixer layer
struct MambaLayer {
    in_proj: candle_nn::Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Option<Tensor>,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    norm: RmsNormGated,
    out_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    conv_kernel_size: usize,
    n_groups: usize,
    time_step_min: f64,
    time_step_max: f64,
}

impl MambaLayer {
    /// Load Mamba layer weights.
    /// When `isq_target_device` is Some, all weights are moved to the specified device.
    /// This is used during ISQ to ensure Mamba weights (which are not quantized) stay on GPU.
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let intermediate_size = cfg.mamba_intermediate_size();
        let conv_dim = cfg.mamba_conv_dim();
        let num_heads = cfg.mamba_n_heads();
        let head_dim = cfg.mamba_d_head();
        let ssm_state_size = cfg.mamba_d_state;
        let conv_kernel_size = cfg.mamba_d_conv;
        let n_groups = cfg.mamba_n_groups;

        let projection_size = intermediate_size + conv_dim + num_heads;
        let in_proj_vb = vb.pp("in_proj");
        let mut in_proj_weight = in_proj_vb.get((projection_size, cfg.hidden_size), "weight")?;
        let mut in_proj_bias = if cfg.mamba_proj_bias {
            Some(in_proj_vb.get(projection_size, "bias")?)
        } else {
            None
        };

        let mut conv1d_weight = vb
            .pp("conv1d")
            .get((conv_dim, 1, conv_kernel_size), "weight")?;
        let mut conv1d_bias = if cfg.mamba_conv_bias {
            Some(vb.pp("conv1d").get(conv_dim, "bias")?)
        } else {
            None
        };

        let mut dt_bias = vb.get(num_heads, "dt_bias")?;
        let mut a_log = vb.get(num_heads, "A_log")?;
        let mut d = vb.get(num_heads, "D")?;
        let norm = RmsNormGated::new(
            intermediate_size,
            cfg.rms_norm_eps,
            vb.pp("norm"),
            isq_target_device,
        )?;

        let out_proj_vb = vb.pp("out_proj");
        let mut out_proj_weight =
            out_proj_vb.get((cfg.hidden_size, intermediate_size), "weight")?;
        let mut out_proj_bias = if cfg.mamba_proj_bias {
            Some(out_proj_vb.get(cfg.hidden_size, "bias")?)
        } else {
            None
        };

        // When ISQ is enabled, move all Mamba weights to target GPU device
        // This prevents device mismatch issues since Mamba layers use candle_nn::Linear
        // (not QuantMethod) and their weights don't get quantized/moved by ISQ pipeline
        if let Some(target_dev) = isq_target_device {
            tracing::debug!(
                "Moving Mamba weights to {:?} for ISQ compatibility",
                target_dev
            );
            in_proj_weight = in_proj_weight.to_device(target_dev)?;
            if let Some(ref bias) = in_proj_bias {
                in_proj_bias = Some(bias.to_device(target_dev)?);
            }
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            if let Some(ref bias) = conv1d_bias {
                conv1d_bias = Some(bias.to_device(target_dev)?);
            }
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
            d = d.to_device(target_dev)?;
            out_proj_weight = out_proj_weight.to_device(target_dev)?;
            if let Some(ref bias) = out_proj_bias {
                out_proj_bias = Some(bias.to_device(target_dev)?);
            }
        }

        let in_proj = candle_nn::Linear::new(in_proj_weight, in_proj_bias);
        let out_proj = candle_nn::Linear::new(out_proj_weight, out_proj_bias);

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            a_log,
            d,
            norm,
            out_proj,
            num_heads,
            head_dim,
            intermediate_size,
            ssm_state_size,
            conv_kernel_size,
            n_groups,
            time_step_min: 0.0,
            time_step_max: f64::MAX,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut MambaLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let groups_time_state_size = self.n_groups * self.ssm_state_size;

        // 1. Input projection
        let projected = self.in_proj.forward(x)?;
        let gate = projected.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?;
        let hidden_states_b_c = projected.narrow(
            candle_core::D::Minus1,
            self.intermediate_size,
            self.intermediate_size + 2 * groups_time_state_size,
        )?;
        let dt = projected.narrow(
            candle_core::D::Minus1,
            self.intermediate_size + self.intermediate_size + 2 * groups_time_state_size,
            self.num_heads,
        )?;

        // Check if we're in cached single-token mode
        let use_cache = cache.seqlen_offset > 0 && seq_len == 1;

        let y = if use_cache {
            // Cached single-token forward
            self.forward_cached(
                &hidden_states_b_c.squeeze(1)?,
                &dt.squeeze(1)?,
                cache,
                batch_size,
            )?
            .unsqueeze(1)?
        } else {
            // Full sequence forward (no fast path, pure torch implementation)
            self.forward_full(&hidden_states_b_c, &dt, cache, batch_size, seq_len)?
        };

        // Apply gated normalization
        let y = self.norm.forward(&y, Some(&gate))?;

        // Output projection
        self.out_proj.forward(&y.to_dtype(dtype)?)
    }

    fn forward_cached(
        &self,
        hidden_states_b_c: &Tensor, // (batch, conv_dim)
        dt: &Tensor,                // (batch, num_heads)
        cache: &mut MambaLayerCache,
        batch_size: usize,
    ) -> Result<Tensor> {
        let groups_time_state_size = self.n_groups * self.ssm_state_size;

        // Update conv state: roll and insert new values
        // conv_state: (batch, conv_dim, d_conv)
        let conv_state = cache.conv_state.narrow(2, 1, self.conv_kernel_size - 1)?;
        let new_col = hidden_states_b_c.unsqueeze(2)?;
        cache.conv_state = Tensor::cat(&[conv_state, new_col], 2)?;

        // Apply convolution: sum(conv_state * weight)
        // weight: (conv_dim, 1, kernel_size) -> squeeze to (conv_dim, kernel_size)
        let weight = self.conv1d_weight.squeeze(1)?;
        let mut hidden_states_b_c =
            (cache.conv_state.clone() * weight.unsqueeze(0)?)?.sum(candle_core::D::Minus1)?;

        if let Some(ref bias) = self.conv1d_bias {
            hidden_states_b_c = hidden_states_b_c.broadcast_add(bias)?;
        }
        let hidden_states_b_c = candle_nn::ops::silu(&hidden_states_b_c)?;

        // Split into hidden_states, B, C
        let hidden_states =
            hidden_states_b_c.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?;
        let b = hidden_states_b_c.narrow(
            candle_core::D::Minus1,
            self.intermediate_size,
            groups_time_state_size,
        )?;
        let c = hidden_states_b_c.narrow(
            candle_core::D::Minus1,
            self.intermediate_size + groups_time_state_size,
            groups_time_state_size,
        )?;

        // SSM computation for single token
        // A = -exp(A_log)
        let a = self.a_log.to_dtype(candle_core::DType::F32)?.exp()?.neg()?;

        // dt with bias and softplus
        let dt_dtype = dt.dtype();
        let dt_bias = self
            .dt_bias
            .to_dtype(dt_dtype)?
            .unsqueeze(0)?
            .expand((batch_size, self.num_heads))?;
        let dt = dt.broadcast_add(&dt_bias)?;
        let dt = softplus(&dt.to_dtype(candle_core::DType::F32)?)?;
        // Clamp dt
        let dt = dt.clamp(self.time_step_min, self.time_step_max)?;

        // Expand dimensions for broadcasting
        // dt: (batch, num_heads) -> (batch, num_heads, head_dim)
        let dt = dt
            .unsqueeze(2)?
            .expand((batch_size, self.num_heads, self.head_dim))?;

        // a: (num_heads) -> (num_heads, head_dim, state_size)
        let a = a
            .unsqueeze(1)?
            .unsqueeze(2)?
            .expand((self.num_heads, self.head_dim, self.ssm_state_size))?
            .to_dtype(candle_core::DType::F32)?;

        // dA = exp(dt * A): (batch, num_heads, head_dim, state_size)
        let da = dt
            .unsqueeze(3)?
            .to_dtype(candle_core::DType::F32)?
            .broadcast_mul(&a.unsqueeze(0)?)?
            .exp()?;

        // Reshape B: (batch, n_groups * state_size) -> (batch, num_heads, state_size)
        let b = b
            .reshape((batch_size, self.n_groups, self.ssm_state_size))?
            .to_dtype(candle_core::DType::F32)?;
        let b = b
            .unsqueeze(2)?
            .expand((
                batch_size,
                self.n_groups,
                self.num_heads / self.n_groups,
                self.ssm_state_size,
            ))?
            .reshape((batch_size, self.num_heads, self.ssm_state_size))?;

        // dB = dt * B: (batch, num_heads, head_dim, state_size)
        let dt_f32 = dt.to_dtype(candle_core::DType::F32)?;
        let db = dt_f32.unsqueeze(3)?.broadcast_mul(&b.unsqueeze(2)?)?;

        // hidden_states: (batch, intermediate_size) -> (batch, num_heads, head_dim)
        let hidden_states = hidden_states
            .reshape((batch_size, self.num_heads, self.head_dim))?
            .to_dtype(candle_core::DType::F32)?;

        // dBx = dB * x: (batch, num_heads, head_dim, state_size)
        let dbx = db.broadcast_mul(&hidden_states.unsqueeze(3)?)?;

        // Update SSM state: state = state * dA + dBx
        let ssm_state = cache
            .ssm_state
            .to_dtype(candle_core::DType::F32)?
            .broadcast_mul(&da)?
            .broadcast_add(&dbx)?;
        cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;

        // Reshape C: (batch, n_groups * state_size) -> (batch, num_heads, state_size)
        let c = c
            .reshape((batch_size, self.n_groups, self.ssm_state_size))?
            .to_dtype(candle_core::DType::F32)?;
        let c = c
            .unsqueeze(2)?
            .expand((
                batch_size,
                self.n_groups,
                self.num_heads / self.n_groups,
                self.ssm_state_size,
            ))?
            .reshape((batch_size, self.num_heads, self.ssm_state_size))?;

        // y = (state @ C^T): (batch, num_heads, head_dim)
        // state: (batch, num_heads, head_dim, state_size)
        // C: (batch, num_heads, state_size)
        let y = cache
            .ssm_state
            .to_dtype(candle_core::DType::F32)?
            .matmul(&c.unsqueeze(3)?)?
            .squeeze(3)?;

        // D skip connection: y = y + x * D
        let d = self
            .d
            .to_dtype(candle_core::DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .expand((batch_size, self.num_heads, self.head_dim))?;
        let y = y.broadcast_add(&hidden_states.broadcast_mul(&d)?)?;

        // Reshape output: (batch, num_heads, head_dim) -> (batch, intermediate_size)
        let y = y.reshape((batch_size, self.intermediate_size))?;

        cache.seqlen_offset += 1;
        Ok(y)
    }

    fn forward_full(
        &self,
        hidden_states_b_c: &Tensor, // (batch, seq_len, conv_dim)
        dt: &Tensor,                // (batch, seq_len, num_heads)
        cache: &mut MambaLayerCache,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let groups_time_state_size = self.n_groups * self.ssm_state_size;

        // Store conv state for future use
        let hidden_states_b_c_t = hidden_states_b_c.transpose(1, 2)?; // (batch, conv_dim, seq_len)
        let pad_width = self.conv_kernel_size.saturating_sub(seq_len);
        let conv_state = if pad_width > 0 {
            let zeros = Tensor::zeros(
                (batch_size, hidden_states_b_c_t.dim(1)?, pad_width),
                hidden_states_b_c_t.dtype(),
                hidden_states_b_c_t.device(),
            )?;
            Tensor::cat(&[zeros, hidden_states_b_c_t.clone()], 2)?
        } else {
            hidden_states_b_c_t.narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)?
        };
        cache.conv_state = conv_state;

        // Apply conv1d
        // Pad input for causal conv
        let padded = Tensor::cat(
            &[
                Tensor::zeros(
                    (
                        batch_size,
                        self.conv_kernel_size - 1,
                        hidden_states_b_c.dim(2)?,
                    ),
                    hidden_states_b_c.dtype(),
                    hidden_states_b_c.device(),
                )?,
                hidden_states_b_c.clone(),
            ],
            1,
        )?;

        // Manual grouped conv1d
        let padded_t = padded.transpose(1, 2)?; // (batch, conv_dim, seq_len + pad)
        let weight = self.conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;

        // For each output position, compute the convolution
        let mut conv_outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let window = padded_t.narrow(2, i, self.conv_kernel_size)?;
            let out = (window * weight.unsqueeze(0)?)?.sum(candle_core::D::Minus1)?;
            conv_outputs.push(out);
        }
        let mut hidden_states_b_c = Tensor::stack(&conv_outputs, 1)?; // (batch, seq_len, conv_dim)

        if let Some(ref bias) = self.conv1d_bias {
            let bias = bias.to_dtype(hidden_states_b_c.dtype())?;
            hidden_states_b_c =
                hidden_states_b_c.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        let hidden_states_b_c = candle_nn::ops::silu(&hidden_states_b_c)?;

        // Split into hidden_states, B, C
        let hidden_states =
            hidden_states_b_c.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?;
        let b = hidden_states_b_c.narrow(
            candle_core::D::Minus1,
            self.intermediate_size,
            groups_time_state_size,
        )?;
        let c = hidden_states_b_c.narrow(
            candle_core::D::Minus1,
            self.intermediate_size + groups_time_state_size,
            groups_time_state_size,
        )?;

        // Reshape for SSM first
        let hidden_states = hidden_states
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .to_dtype(candle_core::DType::F32)?;
        let b = b
            .reshape((batch_size, seq_len, self.n_groups, self.ssm_state_size))?
            .to_dtype(candle_core::DType::F32)?;
        let c = c
            .reshape((batch_size, seq_len, self.n_groups, self.ssm_state_size))?
            .to_dtype(candle_core::DType::F32)?;

        // SSM computation with chunking
        let a = self.a_log.to_dtype(candle_core::DType::F32)?.exp()?.neg()?;

        // dt processing
        let dt_dtype = dt.dtype();
        let dt_bias = self
            .dt_bias
            .to_dtype(dt_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .expand((batch_size, seq_len, self.num_heads))?;

        // b, c are already on the correct device (derived from hidden_states)
        // ssm_state uses cache which is initialized on the correct device
        let mut ssm_state = cache.ssm_state.to_dtype(candle_core::DType::F32)?;

        // SSM computation with chunking

        // dt transform
        let dt = dt.broadcast_add(&dt_bias)?;
        let dt = softplus(&dt.to_dtype(candle_core::DType::F32)?)?;
        let dt = dt.clamp(self.time_step_min, self.time_step_max)?;

        // D coefficient (weights on correct device from ISQ loading)
        let d_coeff = self.d.to_dtype(candle_core::DType::F32)?;

        // Expand B and C to num_heads
        let b = b
            .unsqueeze(3)?
            .expand((
                batch_size,
                seq_len,
                self.n_groups,
                self.num_heads / self.n_groups,
                self.ssm_state_size,
            ))?
            .reshape((batch_size, seq_len, self.num_heads, self.ssm_state_size))?;
        let c = c
            .unsqueeze(3)?
            .expand((
                batch_size,
                seq_len,
                self.n_groups,
                self.num_heads / self.n_groups,
                self.ssm_state_size,
            ))?
            .reshape((batch_size, seq_len, self.num_heads, self.ssm_state_size))?;
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let dt_t = dt.i((.., t, ..))?.unsqueeze(2)?.expand((
                batch_size,
                self.num_heads,
                self.head_dim,
            ))?;
            let x_t = hidden_states.i((.., t, .., ..))?;
            let b_t = b.i((.., t, .., ..))?;
            let c_t = c.i((.., t, .., ..))?;

            // dA = exp(dt * A)
            let a_expanded = a.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?.expand((
                batch_size,
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
            ))?;
            let da = dt_t.unsqueeze(3)?.broadcast_mul(&a_expanded)?.exp()?;

            // dB = dt * B
            let db = dt_t.unsqueeze(3)?.broadcast_mul(&b_t.unsqueeze(2)?)?;

            // dBx = dB * x
            let dbx = db.broadcast_mul(&x_t.unsqueeze(3)?)?;

            // Update state: state = state * dA + dBx
            ssm_state = ssm_state.broadcast_mul(&da)?.broadcast_add(&dbx)?;

            // Output: y = state @ C^T
            let y_t = ssm_state.matmul(&c_t.unsqueeze(3)?)?.squeeze(3)?;

            // D skip connection
            let d_expanded = d_coeff.unsqueeze(0)?.unsqueeze(2)?.expand((
                batch_size,
                self.num_heads,
                self.head_dim,
            ))?;
            let y_t = y_t.broadcast_add(&x_t.broadcast_mul(&d_expanded)?)?;

            outputs.push(y_t);
        }

        // Store final state
        cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;
        cache.seqlen_offset = seq_len;

        // Stack outputs: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, intermediate_size)
        let y = Tensor::stack(&outputs, 1)?;
        y.reshape((batch_size, seq_len, self.intermediate_size))
    }
}

/// Mamba decoder block (replaces attention block for Mamba layers)
struct MambaBlock {
    rms_1: RmsNorm,
    mamba: MambaLayer,
    rms_2: RmsNorm,
    mlp: Box<dyn MlpLayer>,
    block_sparse_moe: Option<GraniteMoE>,
    residual_multiplier: f32,
}

impl MambaBlock {
    fn forward(&self, x: &Tensor, cache: &mut MambaLayerCache) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let mamba_out = self.mamba.forward(&x, cache)?;
        let mamba_out = scale_tensor(mamba_out, self.residual_multiplier)?;
        let x = (mamba_out + residual)?;
        let residual = &x;
        let normed = self.rms_2.forward(&x)?;

        // Combine MoE and shared MLP outputs (if MoE present)
        let ffn_out = if let Some(ref moe) = self.block_sparse_moe {
            let moe_out = moe.forward(&normed)?;
            let mlp_out = self.mlp.forward(&normed)?;
            (moe_out + mlp_out)?
        } else {
            self.mlp.forward(&normed)?
        };

        let ffn_out = scale_tensor(ffn_out, self.residual_multiplier)?;
        ffn_out + residual
    }

    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        // When ISQ is enabled, get the target device to move Mamba weights to GPU
        // This prevents device mismatch since Mamba uses candle_nn::Linear (not QuantMethod)
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false)
        } else {
            None
        };

        let mamba = MambaLayer::load(
            mapper.set_device(layer_idx, vb.pp("mamba"), loading_isq),
            cfg,
            isq_target_device,
        )?;
        let mlp = GraniteMlp::new(
            mapper.set_device(layer_idx, vb.clone(), loading_isq),
            cfg,
            comm,
        )?;
        // Load MoE if num_local_experts > 0
        let block_sparse_moe = if cfg.num_local_experts > 0 {
            Some(GraniteMoE::new(
                cfg,
                mapper.set_device(layer_idx, vb.pp("block_sparse_moe"), loading_isq),
            )?)
        } else {
            None
        };
        let rms_1 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            rms_1,
            mamba,
            rms_2,
            mlp: Box::new(mlp),
            block_sparse_moe,
            residual_multiplier: cfg.residual_multiplier,
        })
    }
}

/// Enum to represent either an attention or mamba layer
enum DecoderLayer {
    Attention(Block),
    Mamba(MambaBlock),
}

// Use HybridLayerCache from kv_cache instead of a local type alias

// ====================== End Mamba Implementation ======================

#[allow(dead_code)]
struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Option<Arc<RotaryEmbedding>>, // Optional - None when position_embedding_type == "nope"
    max_seq_len: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&x, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&x, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&x, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        // Apply rotary embeddings only if position_embedding_type is not "nope"
        (q, k) = if let Some(ref rotary_emb) = self.rotary_emb {
            rotary_emb.forward(&q, &k, seqlen_offsets)?
        } else {
            (q, k)
        };

        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask.clone().as_ref(),
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            y = y.to_dtype(t)?;
        }
        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&y, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        rope: Option<Arc<RotaryEmbedding>>, // Optional - None when position_embedding_type == "nope"
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = cfg.head_dim() * cfg.num_attention_heads;
        let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
        let q_proj = ColumnParallelLayer::new(
            size_in,
            size_q,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard =
            mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads(), cfg.head_dim(), comm);
        let k_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            size_q,
            size_in,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: (cfg.num_key_value_heads() / comm.world_size()).max(1),
            head_dim: cfg.head_dim(),
            rotary_emb: rope, // Now optional
            max_seq_len: cfg.max_position_embeddings,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads(),
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                // GraniteMoeHybrid uses attention_multiplier instead of 1/sqrt(d)
                softmax_scale: cfg.attention_multiplier,
                sliding_window: None,
                sinks: None,
            },
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Box<dyn MlpLayer>,
    block_sparse_moe: Option<GraniteMoE>,
    residual_multiplier: f32,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let attn_out = self.attn.forward(
            &x,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        // Scale residual connection
        let attn_out = scale_tensor(attn_out, self.residual_multiplier)?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.rms_2.forward(&x)?;

        // Combine MoE and shared MLP outputs (if MoE present)
        let ffn_out = if let Some(ref moe) = self.block_sparse_moe {
            let moe_out = moe.forward(&normed)?;
            let mlp_out = self.mlp.forward(&normed)?;
            (moe_out + mlp_out)?
        } else {
            self.mlp.forward(&normed)?
        };

        // Scale residual connection
        let ffn_out = scale_tensor(ffn_out, self.residual_multiplier)?;
        let x = (ffn_out + residual)?;
        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Option<Arc<RotaryEmbedding>>, // Optional - None when position_embedding_type == "nope"
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::load(
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            cfg,
            rope, // Pass the optional RoPE
            paged_attn,
            comm,
        )?;
        let mlp = GraniteMlp::new(
            mapper.set_device(layer_idx, vb.clone(), loading_isq),
            cfg,
            comm,
        )?;
        // Load MoE if num_local_experts > 0
        let block_sparse_moe = if cfg.num_local_experts > 0 {
            Some(GraniteMoE::new(
                cfg,
                mapper.set_device(layer_idx, vb.pp("block_sparse_moe"), loading_isq),
            )?)
        } else {
            None
        };
        let rms_1 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp: Box::new(mlp),
            block_sparse_moe,
            residual_multiplier: cfg.residual_multiplier,
        })
    }
}

fn scale_tensor(tensor: Tensor, scale: f32) -> Result<Tensor> {
    if (scale - 1.0).abs() < f32::EPSILON {
        Ok(tensor)
    } else {
        tensor.affine(scale as f64, 0.)
    }
}

/// Local enum to represent either a KV cache or Mamba cache for a layer
/// (used internally by GraniteMoeHybrid - not exported)
enum GraniteLayerCache {
    Attention(KvCache),
    Mamba(MambaLayerCache),
}

/// Hybrid cache that can store either KV cache or Mamba cache per layer
/// (local to granite model - wraps kv_cache::HybridCache for pipeline integration)
#[allow(dead_code)]
struct GraniteHybridCache {
    pub caches: Vec<GraniteLayerCache>,
    max_seq_len: usize,
}

impl GraniteHybridCache {
    pub fn new(
        layer_types: &[GraniteLayerType],
        cfg: &Config,
        device: &Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let mut caches = Vec::with_capacity(layer_types.len());
        for layer_type in layer_types {
            match layer_type {
                GraniteLayerType::Attention => {
                    caches.push(GraniteLayerCache::Attention(KvCache::new_normal(
                        2,
                        cfg.max_position_embeddings,
                        HybridCache::CACHE_GROW_SIZE,
                    )));
                }
                GraniteLayerType::Mamba => {
                    caches.push(GraniteLayerCache::Mamba(create_mamba_cache(
                        1, cfg, dtype, device,
                    )?));
                }
            }
        }
        Ok(Self {
            caches,
            max_seq_len: cfg.max_position_embeddings,
        })
    }

    pub fn seqlen(&self) -> usize {
        // Return the seqlen from the first attention layer
        for cache in &self.caches {
            if let GraniteLayerCache::Attention(kv) = cache {
                return kv.current_seq_len();
            }
        }
        // If no attention layers, return 0
        0
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            match cache {
                GraniteLayerCache::Attention(kv) => kv.reset(),
                GraniteLayerCache::Mamba(mamba) => {
                    let _ = mamba.reset();
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }
}

impl PastKvLenCache for GraniteHybridCache {
    fn get_past_kv_len(&self) -> Result<usize> {
        Ok(self.seqlen())
    }
}

#[allow(dead_code)]
pub struct GraniteMoeHybrid {
    wte: Embedding,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<GraniteLayerType>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    hybrid_cache: Arc<Mutex<GraniteHybridCache>>,
    // EitherCache for pipeline integration
    kv_cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    embedding_multiplier: f32,
    logits_scaling: f32,
    num_attention_heads: usize,
    max_seq_len: usize,
}

impl GraniteMoeHybrid {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;

        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(wte.embeddings(), normal_loading_metadata.loading_isq)?,
                None,
            ))?
        };
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let head_dim = cfg.head_dim();

        // Check if position embeddings should be used
        let use_position_embeddings = cfg.position_embedding_type != "nope";

        if !use_position_embeddings {
            tracing::info!("GraniteMoeHybrid: position_embedding_type is 'nope', skipping RoPE");
        }

        // Build RoPE embeddings per device (only if position embeddings are used)
        // Note: granite rope_type scaling is not yet supported, using default rope
        if use_position_embeddings {
            if let Some(GraniteRopeConfig {
                rope_type: GraniteRopeType::Granite,
                ..
            }) = &cfg.rope_scaling
            {
                tracing::warn!(
                    "Granite-style rope scaling is not yet fully supported. Using default rope scaling."
                );
            }
        }

        let mut ropes = HashMap::new();
        if use_position_embeddings {
            for i in 0..cfg.num_hidden_layers {
                let device = mapper
                    .device_for(i, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(device.location())
                {
                    let rope = RotaryEmbedding::new(
                        cfg.rope_theta,
                        head_dim,
                        cfg.max_position_embeddings,
                        device,
                        true, // is_gpt_neox style
                        vb_m.dtype(),
                    )?;
                    e.insert(Arc::new(rope));
                }
            }
        }

        let layer_types = cfg.layer_types();

        // Log layer configuration
        let num_mamba = layer_types
            .iter()
            .filter(|t| matches!(t, GraniteLayerType::Mamba))
            .count();
        let num_attn = layer_types
            .iter()
            .filter(|t| matches!(t, GraniteLayerType::Attention))
            .count();
        tracing::info!(
            "GraniteMoeHybrid: {} attention layers, {} mamba layers",
            num_attn,
            num_mamba
        );

        // Build layers based on layer_types
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        ) {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let comm = mapper.get_comm_for(i)?;
            let vb_layer = vb_m.pp(format!("layers.{i}"));

            let layer = match &layer_types[i] {
                GraniteLayerType::Attention => {
                    // Get optional RoPE - None when position_embedding_type == "nope"
                    let rotary_emb = if use_position_embeddings {
                        Some(
                            ropes
                                .get(&device.location())
                                .expect("No RoPE for device location!")
                                .clone(),
                        )
                    } else {
                        None
                    };
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(head_dim, device, None)?)
                        }
                    };
                    DecoderLayer::Attention(Block::load(
                        vb_layer,
                        cfg,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        rotary_emb, // Now optional
                        paged_attn,
                        &comm,
                    )?)
                }
                GraniteLayerType::Mamba => DecoderLayer::Mamba(MambaBlock::load(
                    vb_layer,
                    cfg,
                    &*mapper,
                    i,
                    normal_loading_metadata.loading_isq,
                    &comm,
                )?),
            };
            layers.push(layer);
        }

        // Create hybrid cache for internal use
        let hybrid_cache = Arc::new(Mutex::new(GraniteHybridCache::new(
            &layer_types,
            cfg,
            &normal_loading_metadata.real_device,
            vb_m.dtype(),
        )?));

        // Create pipeline-compatible hybrid cache config
        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                GraniteLayerType::Attention => HybridLayerType::Attention,
                GraniteLayerType::Mamba => HybridLayerType::Mamba,
            })
            .collect();

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: cfg.max_position_embeddings,
            // batch_size=1 is enforced for hybrid models, so only 1 slot needed
            max_num_seqs: 1,
            mamba_conv_dim: cfg.mamba_conv_dim(),
            mamba_d_conv: cfg.mamba_d_conv,
            mamba_n_heads: cfg.mamba_n_heads(),
            mamba_head_dim: cfg.mamba_d_head(),
            mamba_d_state: cfg.mamba_d_state,
        };

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(
                hybrid_cache_config,
                vb_m.dtype(),
                &normal_loading_metadata.real_device,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("Failed to create hybrid cache: {}", e))
            })?,
        ));

        let num_attention_heads = cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size();

        Ok(Self {
            wte,
            layers,
            layer_types,
            ln_f,
            lm_head,
            hybrid_cache,
            kv_cache: EitherCache::Hybrid(pipeline_cache),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads() / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: num_attention_heads,
                sliding_window: None,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            embedding_multiplier: cfg.embedding_multiplier,
            logits_scaling: if cfg.logits_scaling == 0.0 {
                1.0
            } else {
                1.0 / cfg.logits_scaling
            },
            num_attention_heads,
            max_seq_len: cfg.max_position_embeddings,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (_batch_size, _seq_len) = input_ids.dims2()?;
        let mut x = self.wte.forward(input_ids)?;
        // Scale embeddings
        x = scale_tensor(x, self.embedding_multiplier)?;

        // Get both internal cache and pipeline cache
        let mut internal_cache = self.hybrid_cache.lock().unwrap();
        let mut pipeline_cache = self.kv_cache.hybrid();

        // Get state_indices for Mamba layers from pipeline cache
        let state_indices = pipeline_cache.state_indices().cloned();

        // Build attention mask - use seqlen_offsets for attention layers
        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*internal_cache as &dyn PastKvLenCache),
            x.dtype(),
            self.num_attention_heads,
        )?;
        // PagedAttention prompt chunking
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;

            match layer {
                DecoderLayer::Attention(block) => {
                    // Use internal cache for attention layers
                    if let GraniteLayerCache::Attention(kv_cache) =
                        &mut internal_cache.caches[layer_idx]
                    {
                        let mask_for_layer = mask.as_ref().map(|m| m.get(x.device()).clone());
                        x = block.forward(
                            &x,
                            &mask_for_layer,
                            seqlen_offsets,
                            kv_cache,
                            metadata.as_ref().map(|(kv_cache, metadata)| {
                                (kv_cache[layer_idx].clone(), *metadata)
                            }),
                            flash_params,
                        )?;
                    }
                }
                DecoderLayer::Mamba(block) => {
                    // For batch_size=1, use internal cache (faster, no gather/scatter overhead)
                    // For batch_size>1, use pool-based approach with gather/scatter
                    let batch_size = x.dim(0)?;

                    if batch_size == 1 {
                        // Single sequence: use internal cache directly (no pool overhead)
                        if let GraniteLayerCache::Mamba(mamba_cache) =
                            &mut internal_cache.caches[layer_idx]
                        {
                            // Reset state at start of new sequence (prompt phase)
                            if seqlen_offsets[0] == 0 {
                                mamba_cache.reset()?;
                            }
                            x = block.forward(&x, mamba_cache)?;
                        }
                    } else if let (Some(ref indices), Some(HybridLayerCache::Mamba(pool))) =
                        (&state_indices, pipeline_cache.get_mut(layer_idx))
                    {
                        // Multiple sequences: use pool with gather/scatter
                        let conv_state = pool.gather_conv_state(indices)?;
                        let ssm_state = pool.gather_ssm_state(indices)?;

                        // Get seqlen_offset from first sequence (assumes all same phase)
                        let first_idx: u32 = indices.i(0)?.to_scalar()?;
                        let seqlen_offset = pool.get_seqlen_offset(first_idx as usize);

                        // Create temporary cache with gathered states
                        let mut temp_cache = MambaLayerCache {
                            conv_state,
                            ssm_state,
                            seqlen_offset,
                        };

                        // Run Mamba forward
                        x = block.forward(&x, &mut temp_cache)?;

                        // Scatter updated states back to pool
                        pool.scatter_conv_state(indices, &temp_cache.conv_state)?;
                        pool.scatter_ssm_state(indices, &temp_cache.ssm_state)?;

                        // Update seqlen_offsets in pool for each sequence
                        let indices_vec: Vec<u32> = indices.to_vec1()?;
                        for &idx in &indices_vec {
                            pool.set_seqlen_offset(idx as usize, temp_cache.seqlen_offset);
                        }
                    } else {
                        // Fallback: use internal cache
                        if let GraniteLayerCache::Mamba(mamba_cache) =
                            &mut internal_cache.caches[layer_idx]
                        {
                            // Reset state at start of new sequence (prompt phase)
                            if seqlen_offsets[0] == 0 {
                                mamba_cache.reset()?;
                            }
                            x = block.forward(&x, mamba_cache)?;
                        }
                    }
                }
            }
        }

        let x = x.to_device(&self.device)?;
        let x = self.ln_f.forward(&x)?;
        let mut x = extract_logits(&x, context_lens)?;

        if let Some(t) = self.lm_head.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let mut logits = MatMul.qmethod_matmul(&x, &*self.lm_head)?;

        // Scale logits
        logits = scale_tensor(logits, self.logits_scaling)?;

        Ok(logits)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.wte);
        uvb_m.pp("norm").add(&self.ln_f);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            match layer {
                DecoderLayer::Attention(block) => {
                    uvb_l.pp("input_layernorm").add(&block.rms_1);
                    uvb_l.pp("post_attention_layernorm").add(&block.rms_2);
                }
                DecoderLayer::Mamba(block) => {
                    uvb_l.pp("input_layernorm").add(&block.rms_1);
                    uvb_l.pp("post_attention_layernorm").add(&block.rms_2);
                }
            }
        }

        uvb_m.to_safetensors()
    }
}

impl IsqModel for GraniteMoeHybrid {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                DecoderLayer::Attention(block) => {
                    tensors.push((&mut block.attn.q_proj, Some(i)));
                    tensors.push((&mut block.attn.k_proj, Some(i)));
                    tensors.push((&mut block.attn.v_proj, Some(i)));
                    tensors.push((&mut block.attn.o_proj, Some(i)));
                    tensors.extend(
                        block
                            .mlp
                            .get_isq_layers()
                            .into_iter()
                            .map(|m| (m, Some(i)))
                            .collect::<Vec<_>>(),
                    );
                }
                DecoderLayer::Mamba(block) => {
                    // Mamba layers have MLP but no attention projections to quantize
                    // The mamba in_proj/out_proj are candle_nn::Linear, not QuantMethod
                    tensors.extend(
                        block
                            .mlp
                            .get_isq_layers()
                            .into_iter()
                            .map(|m| (m, Some(i)))
                            .collect::<Vec<_>>(),
                    );
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.residual_tensors_m(uvb.pp("model"))
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        let mut names = Vec::new();
        // lm_head
        names.push(None);
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                DecoderLayer::Attention(_) => {
                    names.push(Some(format!("blk.{i}.attn_q.weight")));
                    names.push(Some(format!("blk.{i}.attn_k.weight")));
                    names.push(Some(format!("blk.{i}.attn_v.weight")));
                    names.push(Some(format!("blk.{i}.attn_output.weight")));
                    // GraniteMlp has input_linear and output_linear
                    names.push(Some(format!("blk.{i}.ffn_input.weight")));
                    names.push(Some(format!("blk.{i}.ffn_output.weight")));
                }
                DecoderLayer::Mamba(_) => {
                    // Mamba layers only have MLP for ISQ
                    names.push(Some(format!("blk.{i}.ffn_input.weight")));
                    names.push(Some(format!("blk.{i}.ffn_output.weight")));
                }
            }
        }
        Ok(names)
    }
}

impl NormalModel for GraniteMoeHybrid {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &crate::pipeline::EitherCache {
        &self.kv_cache
    }
    fn cache_mut(&mut self) -> &mut crate::pipeline::EitherCache {
        &mut self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for GraniteMoeHybrid {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.layers {
            match layer {
                DecoderLayer::Attention(block) => mlps.push(&*block.mlp),
                DecoderLayer::Mamba(block) => mlps.push(&*block.mlp),
            }
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.layers {
            match layer {
                DecoderLayer::Attention(block) => mlps.push(&mut block.mlp),
                DecoderLayer::Mamba(block) => mlps.push(&mut block.mlp),
            }
        }
        mlps
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, _mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        let mut experts: Vec<Vec<Box<dyn MlpLayer>>> = Vec::new();
        if layers.is_empty() {
            layers = (0..self.layers.len()).collect::<Vec<_>>();
        }
        for _ in 0..layers.len() {
            experts.push(Vec::new());
        }

        // Helper to get MLP from a layer
        fn get_mlp(layer: &DecoderLayer) -> &dyn MlpLayer {
            match layer {
                DecoderLayer::Attention(block) => &*block.mlp,
                DecoderLayer::Mamba(block) => &*block.mlp,
            }
        }

        for vb in additional_vbs {
            let vb = vb.pp(&prefix);
            for (layer_idx, row) in experts.iter_mut().enumerate() {
                if !layers.contains(&layer_idx) {
                    continue;
                }

                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let layer_mlp = get_mlp(&self.layers[layer_idx]);
                        let (dtype, device) = layer_mlp.dtype_device();
                        // For GraniteMlp, we need custom handling
                        let cfg_for_layer = Config {
                            hidden_size: layer_mlp.get_params()[0],
                            shared_intermediate_size: Some(layer_mlp.get_params()[1]),
                            intermediate_size: layer_mlp.get_params()[1],
                            vocab_size: 0,
                            num_hidden_layers: 0,
                            num_attention_heads: 0,
                            num_key_value_heads: None,
                            rms_norm_eps: 0.0,
                            rope_theta: 0.0,
                            max_position_embeddings: 0,
                            rope_scaling: None,
                            quantization_config: None,
                            tie_word_embeddings: false,
                            layer_types: vec![],
                            attention_multiplier: 1.0,
                            embedding_multiplier: 1.0,
                            residual_multiplier: 1.0,
                            logits_scaling: 1.0,
                            // Mamba fields (not used for MLP but needed for struct)
                            mamba_n_heads: None,
                            mamba_n_groups: 1,
                            mamba_d_state: 256,
                            mamba_d_head: None,
                            mamba_d_conv: 4,
                            mamba_expand: 2,
                            mamba_chunk_size: 256,
                            mamba_conv_bias: true,
                            mamba_proj_bias: false,
                            // MoE fields (not used for MLP but needed for struct)
                            num_local_experts: 0,
                            num_experts_per_tok: 2,
                            // Position embedding type (not used for MLP)
                            position_embedding_type: "rope".to_string(),
                        };
                        row.push(Box::new(GraniteMlp::new(
                            vb.pp(layer_idx)
                                .pp("mlp")
                                .set_dtype(dtype)
                                .set_device(device),
                            &cfg_for_layer,
                            &self.mapper.get_comm_for(layer_idx)?,
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter { .. } => {
                        candle_core::bail!("LoRA adapters not supported for GraniteMoeHybrid MLP")
                    }
                }
            }
        }
        for (layer_idx, expert) in layers.into_iter().zip(experts) {
            let mlp_box = match &mut self.layers[layer_idx] {
                DecoderLayer::Attention(block) => &mut block.mlp,
                DecoderLayer::Mamba(block) => &mut block.mlp,
            };
            let mut experts_all = vec![mlp_box.clone()];
            experts_all.extend(expert);
            let (dtype, device) = mlp_box.dtype_device();
            *mlp_box = Box::new(MoeMlp::new(
                experts_all,
                config.clone(),
                dtype,
                &device,
                layer_idx,
                gate_vb.as_ref(),
            )?);
        }
        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}
