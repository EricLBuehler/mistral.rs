#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::Module;
use mistralrs_quant::{
    apply_immediate_isq, should_apply_immediate_isq, ColumnParallelLayer, QuantMethod,
    QuantizedConfig, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    ops::Range,
    sync::{Arc, Mutex},
};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer, MoeMlp},
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::{embedding_with_legacy_tied_uqff, CausalMasker, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, ForwardMaskCache, IsqModel, KvCache, ModelForwardContext,
        NormalLoadingMetadata, NormalModel, RecurrentBatchKind,
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
        let projected = self.input_linear.forward(x)?;
        let chunks = projected.chunk(2, candle_core::D::Minus1)?;
        let gated =
            crate::ops::mul_and_act(&chunks[0], &chunks[1], crate::layers::Activation::Silu)?;
        let res = self.output_linear.forward(&gated)?;
        Ok(res)
    }
}

impl MlpLayer for GraniteMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
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
    layer: Arc<dyn QuantMethod>,
    num_experts: usize,
    top_k: usize,
}

impl GraniteTopKGating {
    fn new(
        input_size: usize,
        num_experts: usize,
        top_k: usize,
        quantization_config: &Option<QuantizedConfig>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let layer = ReplicatedLayer::new(
            input_size,
            num_experts,
            quantization_config,
            false,
            vb.pp("layer"),
        )?;
        Ok(Self {
            layer,
            num_experts,
            top_k,
        })
    }

    fn topk(&self, x: &Tensor) -> Result<crate::ops::TopKOutput> {
        let logits = self.layer.forward(x)?;
        crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )
    }

    fn grouped_routes(
        &self,
        topk: &crate::ops::TopKOutput,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let selected_experts = topk.indices.to_vec2::<u32>()?;
        let routing_weights = topk
            .values
            .to_dtype(candle_core::DType::F32)?
            .to_vec2::<f32>()?;

        // Collect (expert_idx, token_idx, gate) tuples
        let mut expert_token_gates: Vec<(usize, usize, f32)> = Vec::new();
        let mut expert_counts = vec![0usize; self.num_experts];

        for (token_idx, (experts, weights)) in selected_experts
            .iter()
            .zip(routing_weights.iter())
            .enumerate()
        {
            for (&expert_idx, &gate) in experts.iter().zip(weights.iter()) {
                let expert_idx = expert_idx as usize;
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
enum GraniteParallelExpertWeights {
    Dense(Vec<Tensor>),
    Quantized(Arc<dyn QuantMethod>),
}

struct GraniteParallelExperts {
    weights: GraniteParallelExpertWeights,
    output_size: usize,
}

impl GraniteParallelExperts {
    fn new(
        num_experts: usize,
        input_size: usize,
        output_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        if let Some(source) = vb.weight_source() {
            let load_device = mistralrs_quant::weight_source_load_device(&vb);
            if let Some(weight) = source.load_linear(
                &vb.prefix(),
                &load_device,
                mistralrs_quant::Shard::default(),
            )? {
                let weight = apply_immediate_isq(weight, vb)?;
                return Ok(Self {
                    weights: GraniteParallelExpertWeights::Quantized(weight),
                    output_size,
                });
            }
        }
        let all_weights = vb.get((num_experts, output_size, input_size), "weight")?;
        if should_apply_immediate_isq(&vb) {
            let layer = Arc::new(mistralrs_quant::UnquantLinear::new(
                mistralrs_quant::QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                    all_weights,
                    None,
                )),
            )?);
            return Ok(Self {
                weights: GraniteParallelExpertWeights::Quantized(apply_immediate_isq(layer, vb)?),
                output_size,
            });
        }
        let weights = (0..num_experts)
            .map(|i| all_weights.i(i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights: GraniteParallelExpertWeights::Dense(weights),
            output_size,
        })
    }

    fn quantized(&self) -> Option<&Arc<dyn QuantMethod>> {
        match &self.weights {
            GraniteParallelExpertWeights::Dense(_) => None,
            GraniteParallelExpertWeights::Quantized(weight) => Some(weight),
        }
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        match &self.weights {
            GraniteParallelExpertWeights::Dense(weights) => {
                let weight = Tensor::stack(weights, 0)
                    .expect("Granite expert weight reconstruction should succeed");
                uvb.add_tensor("weight", weight);
            }
            GraniteParallelExpertWeights::Quantized(weight) => uvb.add(weight),
        }
        uvb.to_safetensors()
    }

    fn forward(&self, x: &Tensor, expert_size: &[usize]) -> Result<Tensor> {
        let dtype = x.dtype();
        let device = x.device();

        if let GraniteParallelExpertWeights::Quantized(weight) = &self.weights {
            let expert_ids = expert_size
                .iter()
                .enumerate()
                .flat_map(|(expert, count)| std::iter::repeat_n(expert as u32, *count))
                .collect::<Vec<_>>();
            let rows = expert_ids.len();
            let expert_ids = Tensor::from_vec(expert_ids, (rows, 1), device)?;
            let routed_x = x.unsqueeze(1)?;
            weight.process_routed_stats(&routed_x, &expert_ids)?;
            return weight.gather_forward(&routed_x, &expert_ids)?.squeeze(1);
        }
        let GraniteParallelExpertWeights::Dense(weights) = &self.weights else {
            unreachable!()
        };
        let mut outputs = Vec::new();
        let mut offset = 0;

        for (expert_idx, &size) in expert_size.iter().enumerate() {
            if size == 0 {
                continue;
            }
            let expert_input = x.narrow(0, offset, size)?;
            let expert_output = expert_input.matmul(&weights[expert_idx].t()?)?;
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
            router: GraniteTopKGating::new(
                input_size,
                num_experts,
                top_k,
                &cfg.quantization_config,
                vb.pp("router"),
            )?,
            input_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, emb_size) = x.dims3()?;
        let dtype = x.dtype();
        let device = x.device();
        let num_tokens = batch_size * seq_len;

        let x_flat = x.reshape((num_tokens, emb_size))?;
        let topk = self.router.topk(&x_flat)?;
        if let (Some(input_linear), Some(output_linear)) = (
            self.input_linear.quantized(),
            self.output_linear.quantized(),
        ) {
            input_linear.process_routed_stats(&x_flat, &topk.indices)?;
            let hidden = input_linear.gather_forward(&x_flat.unsqueeze(1)?, &topk.indices)?;
            let chunks = hidden.chunk(2, candle_core::D::Minus1)?;
            let hidden =
                crate::ops::mul_and_act(&chunks[0], &chunks[1], crate::layers::Activation::Silu)?;
            output_linear.process_routed_stats(&hidden, &topk.indices)?;
            let expert_outputs = output_linear.gather_forward(&hidden, &topk.indices)?;
            return expert_outputs
                .broadcast_mul(&topk.values.to_dtype(dtype)?.unsqueeze(2)?)?
                .sum(1)?
                .reshape((batch_size, seq_len, self.input_size));
        }
        let (batch_index, batch_gates, expert_size) =
            self.router.grouped_routes(&topk, dtype, device)?;

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

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("input_linear")
            .extend(self.input_linear.residual_tensors());
        uvb.pp("output_linear")
            .extend(self.output_linear.residual_tensors());
        uvb.pp("router").pp("layer").add(&self.router.layer);
        uvb.to_safetensors()
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
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.ssm_state = self.ssm_state.zeros_like()?;
        Ok(())
    }
}

impl Clone for MambaLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct PackedMambaShape {
    physical_batch: usize,
    physical_tokens: usize,
    conv_state_batch: usize,
    conv_dim: usize,
    conv_width: usize,
    ssm_state_batch: usize,
    ssm_heads: usize,
    ssm_head_dim: usize,
    ssm_state_width: usize,
    expected_conv_dim: usize,
    expected_conv_width: usize,
    expected_ssm_heads: usize,
    expected_ssm_head_dim: usize,
    expected_ssm_state_width: usize,
}

fn packed_mamba_query_ranges(
    physical_batch: usize,
    physical_tokens: usize,
    query_lens: &[usize],
) -> Result<Vec<Range<usize>>> {
    if physical_batch != 1 {
        candle_core::bail!(
            "Granite packed Mamba requires physical batch size 1, got {physical_batch}"
        );
    }
    if query_lens.is_empty() {
        candle_core::bail!("Granite packed Mamba requires at least one logical sequence");
    }

    let mut offset = 0usize;
    let mut ranges = Vec::with_capacity(query_lens.len());
    for (sequence_index, &query_len) in query_lens.iter().enumerate() {
        if query_len == 0 {
            candle_core::bail!(
                "Granite packed Mamba logical sequence {sequence_index} has zero tokens"
            );
        }
        let end = offset
            .checked_add(query_len)
            .ok_or_else(|| candle_core::Error::msg("Granite packed Mamba query length overflow"))?;
        ranges.push(offset..end);
        offset = end;
    }
    if offset != physical_tokens {
        candle_core::bail!(
            "Granite packed Mamba has {offset} logical tokens but {physical_tokens} physical tokens"
        );
    }
    Ok(ranges)
}

fn packed_mamba_ranges(shape: PackedMambaShape, query_lens: &[usize]) -> Result<Vec<Range<usize>>> {
    if shape.conv_state_batch != query_lens.len() {
        candle_core::bail!(
            "Granite packed Mamba has {} convolution state rows but {} logical sequences",
            shape.conv_state_batch,
            query_lens.len()
        );
    }
    if shape.ssm_state_batch != query_lens.len() {
        candle_core::bail!(
            "Granite packed Mamba has {} SSM state rows but {} logical sequences",
            shape.ssm_state_batch,
            query_lens.len()
        );
    }
    if shape.conv_dim != shape.expected_conv_dim || shape.conv_width != shape.expected_conv_width {
        candle_core::bail!(
            "Granite packed Mamba convolution state shape mismatch: expected ({}, {}), got ({}, {})",
            shape.expected_conv_dim,
            shape.expected_conv_width,
            shape.conv_dim,
            shape.conv_width
        );
    }
    if shape.ssm_heads != shape.expected_ssm_heads
        || shape.ssm_head_dim != shape.expected_ssm_head_dim
        || shape.ssm_state_width != shape.expected_ssm_state_width
    {
        candle_core::bail!(
            "Granite packed Mamba SSM state shape mismatch: expected ({}, {}, {}), got ({}, {}, {})",
            shape.expected_ssm_heads,
            shape.expected_ssm_head_dim,
            shape.expected_ssm_state_width,
            shape.ssm_heads,
            shape.ssm_head_dim,
            shape.ssm_state_width
        );
    }
    packed_mamba_query_ranges(shape.physical_batch, shape.physical_tokens, query_lens)
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
    in_proj: Arc<dyn QuantMethod>,
    conv1d_weight: Tensor,
    conv1d_bias: Option<Tensor>,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    norm: RmsNormGated,
    out_proj: Arc<dyn QuantMethod>,
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
        let projection_vb = |name: &str| {
            let vb = vb.pp(name);
            if let Some(device) = isq_target_device {
                vb.set_device(device.clone())
            } else {
                vb
            }
        };
        let in_proj = ReplicatedLayer::new(
            cfg.hidden_size,
            projection_size,
            &cfg.quantization_config,
            cfg.mamba_proj_bias,
            projection_vb("in_proj"),
        )?;

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

        let out_proj = ReplicatedLayer::new(
            intermediate_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.mamba_proj_bias,
            projection_vb("out_proj"),
        )?;

        if let Some(target_dev) = isq_target_device {
            tracing::debug!(
                "Moving Mamba weights to {:?} for ISQ compatibility",
                target_dev
            );
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            if let Some(ref bias) = conv1d_bias {
                conv1d_bias = Some(bias.to_device(target_dev)?);
            }
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
            d = d.to_device(target_dev)?;
        }

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

    fn projected_parts(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let groups_time_state_size = self.n_groups * self.ssm_state_size;
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
        Ok((gate, hidden_states_b_c, dt))
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("in_proj").add(&self.in_proj);
        let uvb_conv = uvb.pp("conv1d");
        uvb_conv.add_tensor("weight", self.conv1d_weight.clone());
        if let Some(bias) = &self.conv1d_bias {
            uvb_conv.add_tensor("bias", bias.clone());
        }
        uvb.add_tensor("dt_bias", self.dt_bias.clone());
        uvb.add_tensor("A_log", self.a_log.clone());
        uvb.add_tensor("D", self.d.clone());
        uvb.pp("norm")
            .add_tensor("weight", self.norm.weight.clone());
        uvb.pp("out_proj").add(&self.out_proj);
        uvb.to_safetensors()
    }

    fn forward(
        &self,
        x: &Tensor,
        cache: &mut MambaLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let (gate, hidden_states_b_c, dt) = self.projected_parts(x)?;

        let y = if matches!(batch_kind, RecurrentBatchKind::Decode) {
            if seq_len != 1 {
                candle_core::bail!("Mamba decode expects a single-token query.");
            }
            self.forward_cached(
                &hidden_states_b_c.squeeze(1)?,
                &dt.squeeze(1)?,
                cache,
                batch_size,
            )?
            .unsqueeze(1)?
        } else {
            self.forward_full(&hidden_states_b_c, &dt, cache, batch_size, seq_len)?
        };

        // Apply gated normalization
        let y = self.norm.forward(&y, Some(&gate))?;

        // Output projection
        self.out_proj.forward(&y.to_dtype(dtype)?)
    }

    fn forward_packed_prefill(
        &self,
        x: &Tensor,
        cache: &mut MambaLayerCache,
        query_lens: &[usize],
    ) -> Result<Tensor> {
        let (physical_batch, physical_tokens, _) = x.dims3()?;
        let (conv_state_batch, conv_dim, conv_width) = cache.conv_state.dims3()?;
        let (ssm_state_batch, ssm_heads, ssm_head_dim, ssm_state_width) =
            cache.ssm_state.dims4()?;
        let ranges = packed_mamba_ranges(
            PackedMambaShape {
                physical_batch,
                physical_tokens,
                conv_state_batch,
                conv_dim,
                conv_width,
                ssm_state_batch,
                ssm_heads,
                ssm_head_dim,
                ssm_state_width,
                expected_conv_dim: self.intermediate_size + 2 * self.n_groups * self.ssm_state_size,
                expected_conv_width: self.conv_kernel_size,
                expected_ssm_heads: self.num_heads,
                expected_ssm_head_dim: self.head_dim,
                expected_ssm_state_width: self.ssm_state_size,
            },
            query_lens,
        )?;

        let dtype = x.dtype();
        let (gate, hidden_states_b_c, dt) = self.projected_parts(x)?;
        let mut outputs = Vec::with_capacity(ranges.len());
        let mut conv_states = Vec::with_capacity(ranges.len());
        let mut ssm_states = Vec::with_capacity(ranges.len());
        for (state_index, range) in ranges.into_iter().enumerate() {
            let mut segment_cache = MambaLayerCache {
                conv_state: cache.conv_state.narrow(0, state_index, 1)?,
                ssm_state: cache.ssm_state.narrow(0, state_index, 1)?,
            };
            outputs.push(self.forward_full(
                &hidden_states_b_c.narrow(1, range.start, range.len())?,
                &dt.narrow(1, range.start, range.len())?,
                &mut segment_cache,
                1,
                range.len(),
            )?);
            conv_states.push(segment_cache.conv_state);
            ssm_states.push(segment_cache.ssm_state);
        }

        cache.conv_state = Tensor::cat(&conv_states, 0)?;
        cache.ssm_state = Tensor::cat(&ssm_states, 0)?;
        let y = Tensor::cat(&outputs, 1)?;
        let y = self.norm.forward(&y, Some(&gate))?;
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

        let hidden_states_b_c_t = hidden_states_b_c.transpose(1, 2)?;
        let state_width = cache.conv_state.dim(2)?;
        if state_width != self.conv_kernel_size {
            candle_core::bail!(
                "Mamba convolution state width is {state_width}, expected {}",
                self.conv_kernel_size
            );
        }
        let prior_state = cache.conv_state.clone();
        let state_and_input = Tensor::cat(&[prior_state.clone(), hidden_states_b_c_t.clone()], 2)?;
        cache.conv_state = state_and_input.narrow(
            2,
            state_and_input.dim(2)? - self.conv_kernel_size,
            self.conv_kernel_size,
        )?;
        let padded_t = Tensor::cat(
            &[
                prior_state.narrow(2, 1, self.conv_kernel_size - 1)?,
                hidden_states_b_c_t,
            ],
            2,
        )?;
        let weight = self.conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;

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

        // SSM computation
        let a = self.a_log.to_dtype(candle_core::DType::F32)?.exp()?.neg()?;

        // Expand B and C from groups to num_heads
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

        #[cfg(feature = "cuda")]
        let use_cuda = matches!(hidden_states.device(), Device::Cuda(_));
        #[cfg(not(feature = "cuda"))]
        let use_cuda = false;

        #[cfg(feature = "metal")]
        let use_metal = hidden_states.device().is_metal();
        #[cfg(not(feature = "metal"))]
        let use_metal = false;

        if use_cuda {
            // CUDA kernel handles dt_bias + softplus + clamp internally
            let dt_f32 = dt.to_dtype(candle_core::DType::F32)?;
            let dt_bias_f32 = self.dt_bias.to_dtype(candle_core::DType::F32)?;
            let d_f32 = self.d.to_dtype(candle_core::DType::F32)?;
            let mut ssm_state = cache.ssm_state.to_dtype(candle_core::DType::F32)?;

            let y = crate::cuda::ssm::selective_scan_cuda(
                &hidden_states,
                &dt_f32,
                &a,
                &b,
                &c,
                &d_f32,
                &dt_bias_f32,
                &mut ssm_state,
                self.time_step_min as f32,
                self.time_step_max as f32,
            )?;

            cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;
            y.reshape((batch_size, seq_len, self.intermediate_size))
        } else if use_metal {
            // Metal kernel handles dt_bias + softplus + clamp internally
            let dt_f32 = dt.to_dtype(candle_core::DType::F32)?;
            let dt_bias_f32 = self.dt_bias.to_dtype(candle_core::DType::F32)?;
            let d_f32 = self.d.to_dtype(candle_core::DType::F32)?;
            let mut ssm_state = cache.ssm_state.to_dtype(candle_core::DType::F32)?;

            let y = crate::metal::ssm::selective_scan_metal(
                &hidden_states,
                &dt_f32,
                &a,
                &b,
                &c,
                &d_f32,
                &dt_bias_f32,
                &mut ssm_state,
                self.time_step_min as f32,
                self.time_step_max as f32,
            )?;

            cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;
            y.reshape((batch_size, seq_len, self.intermediate_size))
        } else {
            // CPU fallback: per-timestep Rust loop
            let dt_dtype = dt.dtype();
            let dt_bias = self
                .dt_bias
                .to_dtype(dt_dtype)?
                .unsqueeze(0)?
                .unsqueeze(0)?
                .expand((batch_size, seq_len, self.num_heads))?;
            let mut ssm_state = cache.ssm_state.to_dtype(candle_core::DType::F32)?;

            let dt = dt.broadcast_add(&dt_bias)?;
            let dt = softplus(&dt.to_dtype(candle_core::DType::F32)?)?;
            let dt = dt.clamp(self.time_step_min, self.time_step_max)?;

            let d_coeff = self.d.to_dtype(candle_core::DType::F32)?;

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

            cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;

            let y = Tensor::stack(&outputs, 1)?;
            y.reshape((batch_size, seq_len, self.intermediate_size))
        }
    }
}

/// Mamba decoder block (replaces attention block for Mamba layers)
struct MambaBlock {
    rms_1: RmsNorm,
    mamba: MambaLayer,
    rms_2: RmsNorm,
    mlp: Option<Box<dyn MlpLayer>>,
    block_sparse_moe: Option<GraniteMoE>,
    residual_multiplier: f32,
}

impl MambaBlock {
    fn forward(
        &self,
        x: &Tensor,
        cache: &mut MambaLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let mamba_out = self.mamba.forward(&x, cache, batch_kind)?;
        let mamba_out = scale_tensor(mamba_out, self.residual_multiplier)?;
        let x = (mamba_out + residual)?;
        let residual = &x;
        let normed = self.rms_2.forward(&x)?;

        let ffn_out =
            granite_ffn_forward(self.mlp.as_deref(), self.block_sparse_moe.as_ref(), &normed)?;

        let ffn_out = scale_tensor(ffn_out, self.residual_multiplier)?;
        ffn_out + residual
    }

    fn forward_packed_prefill(
        &self,
        x: &Tensor,
        cache: &mut MambaLayerCache,
        query_lens: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let mamba_out = self.mamba.forward_packed_prefill(&x, cache, query_lens)?;
        let mamba_out = scale_tensor(mamba_out, self.residual_multiplier)?;
        let x = (mamba_out + residual)?;
        let residual = &x;
        let normed = self.rms_2.forward(&x)?;

        let ffn_out =
            granite_ffn_forward(self.mlp.as_deref(), self.block_sparse_moe.as_ref(), &normed)?;

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
        let mlp_vb = mapper.set_device(layer_idx, vb.clone(), loading_isq);
        let shared_input_vb = mlp_vb.pp("shared_mlp").pp("input_linear");
        let mlp = if crate::layers::contains_tensor_or_weight_source(&shared_input_vb, "weight") {
            Some(Box::new(GraniteMlp::new(mlp_vb, cfg, comm)?) as Box<dyn MlpLayer>)
        } else {
            None
        };
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
            mlp,
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
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
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

        (q, k) = if let Some(ref rotary_emb) = self.rotary_emb {
            let positions = ctx
                .text_positions(q.device(), q.dim(2)?)?
                .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
            rotary_emb.forward(&q, &k, positions)?
        } else {
            (q, k)
        };

        let metadata = ctx.paged_layer(layer_idx);
        let flash_params = ctx.flash_params();
        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
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
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let res = self.o_proj.forward(&y)?;
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
            mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads(), cfg.head_dim(), comm)?;
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
                )?,
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
    mlp: Option<Box<dyn MlpLayer>>,
    block_sparse_moe: Option<GraniteMoE>,
    residual_multiplier: f32,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let attn_out = self
            .attn
            .forward(&x, attention_mask, kv_cache, ctx, layer_idx)?;
        // Scale residual connection
        let attn_out = scale_tensor(attn_out, self.residual_multiplier)?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.rms_2.forward(&x)?;

        let ffn_out =
            granite_ffn_forward(self.mlp.as_deref(), self.block_sparse_moe.as_ref(), &normed)?;

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
        let mlp_vb = mapper.set_device(layer_idx, vb.clone(), loading_isq);
        let shared_input_vb = mlp_vb.pp("shared_mlp").pp("input_linear");
        let mlp = if crate::layers::contains_tensor_or_weight_source(&shared_input_vb, "weight") {
            Some(Box::new(GraniteMlp::new(mlp_vb, cfg, comm)?) as Box<dyn MlpLayer>)
        } else {
            None
        };
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
            mlp,
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

fn granite_ffn_forward(
    mlp: Option<&dyn MlpLayer>,
    moe: Option<&GraniteMoE>,
    xs: &Tensor,
) -> Result<Tensor> {
    match (mlp, moe) {
        (Some(mlp), Some(moe)) => mlp.forward(xs)? + moe.forward(xs)?,
        (Some(mlp), None) => mlp.forward(xs),
        (None, Some(moe)) => moe.forward(xs),
        (None, None) => {
            candle_core::bail!("Granite layer has neither dense nor expert FFN weights")
        }
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
        layer_devices: &[Device],
        dtype: candle_core::DType,
    ) -> Result<Self> {
        if layer_devices.len() != layer_types.len() {
            candle_core::bail!(
                "Granite hybrid cache has {} layers but {} layer devices",
                layer_types.len(),
                layer_devices.len()
            );
        }
        let mut caches = Vec::with_capacity(layer_types.len());
        for (layer_type, device) in layer_types.iter().zip(layer_devices) {
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
    wte: Arc<dyn QuantMethod>,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<GraniteLayerType>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
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
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
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
        let dtype = vb_m.dtype();

        let wte = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb_lm_head.clone(), normal_loading_metadata.loading_isq)
            }),
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
            wte.clone()
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
                        is_gptx,
                        vb_m.dtype(),
                    )?;
                    e.insert(Arc::new(rope));
                }
            }
        }

        let layer_types = cfg.layer_types();
        let layer_devices = (0..layer_types.len())
            .map(|layer_idx| {
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone()
            })
            .collect::<Vec<_>>();

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
            &layer_devices,
            vb_m.dtype(),
        )?));

        // Create pipeline-compatible hybrid cache config
        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                GraniteLayerType::Attention => HybridLayerType::Attention,
                GraniteLayerType::Mamba => HybridLayerType::Recurrent,
            })
            .collect();

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: cfg.max_position_embeddings,
            recurrent: RecurrentLayerConfig {
                conv_dim: cfg.mamba_conv_dim(),
                conv_width: cfg.mamba_d_conv,
                state: crate::kv_cache::RecurrentStateSpec::Opaque {
                    dims: vec![cfg.mamba_n_heads(), cfg.mamba_d_head(), cfg.mamba_d_state],
                },
                recurrent_dtype: None,
            },
        };

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(hybrid_cache_config, vb_m.dtype(), &layer_devices).map_err(|e| {
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
            dtype,
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

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let (_batch_size, _seq_len) = input_ids.dims2()?;
        let mut x = self.wte.embedding_forward(input_ids, self.dtype)?;
        // Scale embeddings
        x = scale_tensor(x, self.embedding_multiplier)?;

        // Get both internal cache and pipeline cache
        let mut internal_cache = self.hybrid_cache.lock().unwrap();
        let mut pipeline_cache = self.kv_cache.hybrid();

        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let has_mamba_layers = self
            .layer_types
            .iter()
            .any(|layer_type| matches!(layer_type, GraniteLayerType::Mamba));
        let packed_query_lens = if ctx.flash_params().packed {
            Some(
                ctx.paged_input_metadata()
                    .and_then(|metadata| metadata.query_lens.clone())
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "Granite packed prefill requires logical query lengths",
                        )
                    })?,
            )
        } else {
            None
        };
        if has_mamba_layers {
            if let Some(query_lens) = packed_query_lens.as_deref() {
                let metadata = recurrent_metadata.as_ref().ok_or_else(|| {
                    candle_core::Error::msg(
                        "Granite packed Mamba requires hybrid recurrent metadata",
                    )
                })?;
                if metadata.batch_kind() != RecurrentBatchKind::Prefill {
                    candle_core::bail!("Granite packed Mamba cannot run a decode batch");
                }
                let (physical_batch, physical_tokens, _) = x.dims3()?;
                packed_mamba_query_ranges(physical_batch, physical_tokens, query_lens)?;
                let index_count = metadata.state_indices().dims1()?;
                if index_count != query_lens.len() {
                    candle_core::bail!(
                        "Granite packed Mamba has {index_count} recurrent state indices but {} logical sequences",
                        query_lens.len()
                    );
                }
                if let Some(host_indices) = metadata.state_indices_host() {
                    if host_indices.len() != query_lens.len() {
                        candle_core::bail!(
                            "Granite packed Mamba has {} host state indices but {} logical sequences",
                            host_indices.len(),
                            query_lens.len()
                        );
                    }
                }
            }
        }
        let recurrent_batch_kind = recurrent_metadata
            .as_ref()
            .map(|metadata| metadata.batch_kind())
            .or_else(|| ctx.recurrent_batch_kind())
            .unwrap_or(RecurrentBatchKind::Prefill);

        let paged_mask_cache = ForwardMaskCache::Paged(ctx.seqlen_offsets());
        let mask_cache = if ctx.is_paged() {
            &paged_mask_cache as &dyn PastKvLenCache
        } else {
            &*pipeline_cache as &dyn PastKvLenCache
        };
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            mask_cache,
            x.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let mask = if ctx.is_first_prompt_chunk() {
            mask
        } else {
            AttentionMask::None
        };
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;

            match layer {
                DecoderLayer::Attention(block) => {
                    if let Some(HybridLayerCache::Attention(kv_cache)) =
                        pipeline_cache.get_mut(layer_idx)
                    {
                        let mask_for_layer = &mask.get(x.device());
                        x = block.forward(&x, mask_for_layer, kv_cache, ctx, layer_idx)?;
                    } else if let GraniteLayerCache::Attention(kv_cache) =
                        &mut internal_cache.caches[layer_idx]
                    {
                        let mask_for_layer = &mask.get(x.device());
                        x = block.forward(&x, mask_for_layer, kv_cache, ctx, layer_idx)?;
                    }
                }
                DecoderLayer::Mamba(block) => {
                    if let Some(metadata) = recurrent_metadata.as_ref() {
                        let indices = pipeline_cache
                            .state_indices_for_layer(layer_idx)?
                            .ok_or_else(|| {
                                candle_core::Error::msg(format!(
                                    "Hybrid cache layer {layer_idx} is missing recurrent state indices"
                                ))
                            })?;
                        let Some(HybridLayerCache::Recurrent(pool)) =
                            pipeline_cache.get_mut(layer_idx)
                        else {
                            candle_core::bail!(
                                "Hybrid cache layer {layer_idx} is not recurrent for Granite"
                            );
                        };
                        let conv_state = pool.gather_conv_state(&indices)?;
                        let ssm_state = pool.gather_recurrent_state(&indices)?;

                        let mut temp_cache = MambaLayerCache {
                            conv_state,
                            ssm_state,
                        };

                        x = if let Some(query_lens) = packed_query_lens.as_deref() {
                            block.forward_packed_prefill(&x, &mut temp_cache, query_lens)?
                        } else {
                            block.forward(&x, &mut temp_cache, metadata.batch_kind())?
                        };

                        pool.scatter_conv_state_with_host_indices(
                            &indices,
                            metadata.state_indices_host(),
                            &temp_cache.conv_state,
                        )?;
                        pool.scatter_recurrent_state_with_host_indices(
                            &indices,
                            metadata.state_indices_host(),
                            &temp_cache.ssm_state,
                        )?;
                    } else {
                        if let GraniteLayerCache::Mamba(mamba_cache) =
                            &mut internal_cache.caches[layer_idx]
                        {
                            if recurrent_batch_kind == RecurrentBatchKind::Prefill {
                                mamba_cache.reset()?;
                            }
                            x = block.forward(&x, mamba_cache, recurrent_batch_kind)?;
                        }
                    }
                }
            }
        }

        let x = x.to_device(&self.device)?;
        let x = self.ln_f.forward(&x)?;
        let x = ctx.logits(&x)?;

        let mut logits = self.lm_head.forward(&x)?;

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
                    if let Some(moe) = &block.block_sparse_moe {
                        uvb_l.pp("block_sparse_moe").extend(moe.residual_tensors());
                    }
                }
                DecoderLayer::Mamba(block) => {
                    uvb_l.pp("input_layernorm").add(&block.rms_1);
                    uvb_l.pp("post_attention_layernorm").add(&block.rms_2);
                    uvb_l.pp("mamba").extend(block.mamba.residual_tensors());
                    if let Some(moe) = &block.block_sparse_moe {
                        uvb_l.pp("block_sparse_moe").extend(moe.residual_tensors());
                    }
                }
            }
        }

        uvb_m.to_safetensors()
    }
}

impl IsqModel for GraniteMoeHybrid {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.residual_tensors_m(uvb.pp("model"))
    }
}

impl crate::speculative::SpeculativeTargetMixin for GraniteMoeHybrid {}

impl NormalModel for GraniteMoeHybrid {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
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
        candle_core::bail!("GraniteMoeHybrid does not support X-LoRA forward")
    }
    fn cache(&self) -> &crate::pipeline::EitherCache {
        &self.kv_cache
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
    fn supports_packed_prefill(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for GraniteMoeHybrid {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.layers {
            match layer {
                DecoderLayer::Attention(block) => {
                    if let Some(mlp) = block.mlp.as_deref() {
                        mlps.push(mlp);
                    }
                }
                DecoderLayer::Mamba(block) => {
                    if let Some(mlp) = block.mlp.as_deref() {
                        mlps.push(mlp);
                    }
                }
            }
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.layers {
            match layer {
                DecoderLayer::Attention(block) => {
                    if let Some(mlp) = block.mlp.as_mut() {
                        mlps.push(mlp);
                    }
                }
                DecoderLayer::Mamba(block) => {
                    if let Some(mlp) = block.mlp.as_mut() {
                        mlps.push(mlp);
                    }
                }
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
        fn get_mlp(layer: &DecoderLayer) -> Result<&dyn MlpLayer> {
            match layer {
                DecoderLayer::Attention(block) => block
                    .mlp
                    .as_deref()
                    .ok_or_else(|| candle_core::Error::msg("Granite layer has no shared MLP")),
                DecoderLayer::Mamba(block) => block
                    .mlp
                    .as_deref()
                    .ok_or_else(|| candle_core::Error::msg("Granite layer has no shared MLP")),
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
                        let layer_mlp = get_mlp(&self.layers[layer_idx])?;
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
                DecoderLayer::Attention(block) => block.mlp.as_mut(),
                DecoderLayer::Mamba(block) => block.mlp.as_mut(),
            }
            .ok_or_else(|| candle_core::Error::msg("Granite layer has no shared MLP"))?;
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
        self.layers.iter().all(|layer| match layer {
            DecoderLayer::Attention(block) => block.mlp.is_some(),
            DecoderLayer::Mamba(block) => block.mlp.is_some(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::IsqModelLoader;

    const HIDDEN_SIZE: usize = 4;
    const INTERMEDIATE_SIZE: usize = 4;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 2;
    const STATE_SIZE: usize = 2;
    const NUM_GROUPS: usize = 1;
    const CONV_WIDTH: usize = 3;
    const CONV_DIM: usize = INTERMEDIATE_SIZE + 2 * NUM_GROUPS * STATE_SIZE;
    const PROJECTION_SIZE: usize = INTERMEDIATE_SIZE + CONV_DIM + NUM_HEADS;
    const ASSERT_EPS: f32 = 1e-4;

    fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let value = ((index.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as f32;
                ((value / 128.0) - 1.0) * scale + offset
            })
            .collect()
    }

    fn unquant_linear(weight: Tensor, bias: Option<Tensor>) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(mistralrs_quant::UnquantLinear::new(
            mistralrs_quant::QuantMethodConfig::Unquantized(candle_nn::Linear::new(weight, bias)),
        )?))
    }

    fn mamba_layer(device: &Device) -> Result<MambaLayer> {
        let in_proj_weight = Tensor::from_vec(
            patterned(PROJECTION_SIZE * HIDDEN_SIZE, 1, 0.08, 0.01),
            (PROJECTION_SIZE, HIDDEN_SIZE),
            device,
        )?;
        let in_proj_bias = Tensor::from_vec(
            patterned(PROJECTION_SIZE, 2, 0.02, 0.0),
            (PROJECTION_SIZE,),
            device,
        )?;
        let out_proj_weight = Tensor::from_vec(
            patterned(HIDDEN_SIZE * INTERMEDIATE_SIZE, 3, 0.1, -0.01),
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            device,
        )?;
        let out_proj_bias =
            Tensor::from_vec(patterned(HIDDEN_SIZE, 4, 0.02, 0.0), (HIDDEN_SIZE,), device)?;
        Ok(MambaLayer {
            in_proj: unquant_linear(in_proj_weight, Some(in_proj_bias))?,
            conv1d_weight: Tensor::from_vec(
                patterned(CONV_DIM * CONV_WIDTH, 5, 0.08, 0.01),
                (CONV_DIM, 1, CONV_WIDTH),
                device,
            )?,
            conv1d_bias: Some(Tensor::from_vec(
                patterned(CONV_DIM, 6, 0.03, 0.0),
                (CONV_DIM,),
                device,
            )?),
            dt_bias: Tensor::from_vec(patterned(NUM_HEADS, 7, 0.1, 0.2), (NUM_HEADS,), device)?,
            a_log: Tensor::from_vec(patterned(NUM_HEADS, 8, 0.1, -0.5), (NUM_HEADS,), device)?,
            d: Tensor::from_vec(patterned(NUM_HEADS, 9, 0.1, 0.3), (NUM_HEADS,), device)?,
            norm: RmsNormGated {
                weight: Tensor::from_vec(
                    patterned(INTERMEDIATE_SIZE, 10, 0.1, 1.0),
                    (INTERMEDIATE_SIZE,),
                    device,
                )?,
                eps: 1e-5,
            },
            out_proj: unquant_linear(out_proj_weight, Some(out_proj_bias))?,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            ssm_state_size: STATE_SIZE,
            conv_kernel_size: CONV_WIDTH,
            n_groups: NUM_GROUPS,
            time_step_min: 0.0,
            time_step_max: f64::MAX,
        })
    }

    fn mamba_cache(device: &Device) -> Result<MambaLayerCache> {
        Ok(MambaLayerCache {
            conv_state: Tensor::from_vec(
                patterned(2 * CONV_DIM * CONV_WIDTH, 11, 0.03, 0.0),
                (2, CONV_DIM, CONV_WIDTH),
                device,
            )?,
            ssm_state: Tensor::from_vec(
                patterned(2 * NUM_HEADS * HEAD_DIM * STATE_SIZE, 12, 0.03, 0.0),
                (2, NUM_HEADS, HEAD_DIM, STATE_SIZE),
                device,
            )?,
        })
    }

    #[test]
    fn mamba_residual_contains_every_checkpoint_tensor() -> Result<()> {
        let names = mamba_layer(&Device::Cpu)?
            .residual_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<std::collections::HashSet<_>>();

        for name in [
            "in_proj.weight",
            "in_proj.bias",
            "conv1d.weight",
            "conv1d.bias",
            "dt_bias",
            "A_log",
            "D",
            "norm.weight",
            "out_proj.weight",
            "out_proj.bias",
        ] {
            assert!(names.contains(name), "missing Mamba residual tensor {name}");
        }
        Ok(())
    }

    #[test]
    fn safetensors_granite_experts_participate_in_immediate_isq() -> anyhow::Result<()> {
        const PREFIX: &str = "model.layers.0.block_sparse_moe.input_linear";
        let loader = crate::pipeline::GraniteMoeHybridLoader;
        for predicates in [
            loader.immediate_isq_predicates("")?,
            loader.immediate_isq_predicates_moqe("")?,
        ] {
            let weight = Tensor::ones((2, 4, 3), DType::F32, &Device::Cpu)?;
            let vb = mistralrs_quant::ShardedSafeTensors::wrap(
                HashMap::from([(format!("{PREFIX}.weight"), weight)]),
                DType::F32,
                Device::Cpu,
            );
            let tracker = vb.tracker().clone();
            mistralrs_quant::set_immediate_isq(
                Some(mistralrs_quant::IsqType::Q8_0),
                predicates,
                mistralrs_quant::IsqCaptureMode::CaptureMatches,
            );
            let experts = GraniteParallelExperts::new(2, 3, 4, vb.pp(PREFIX));
            mistralrs_quant::clear_immediate_isq();

            assert!(experts?.quantized().is_some());
            assert_eq!(tracker.get().len(), 1);
            assert_eq!(tracker.get()[0].key, PREFIX);
        }
        Ok(())
    }

    #[test]
    fn dense_granite_expert_residual_reconstructs_stacked_weight() -> Result<()> {
        let experts = GraniteParallelExperts {
            weights: GraniteParallelExpertWeights::Dense(vec![
                Tensor::zeros((4, 3), DType::F32, &Device::Cpu)?,
                Tensor::ones((4, 3), DType::F32, &Device::Cpu)?,
            ]),
            output_size: 4,
        };
        let residual = experts.residual_tensors();

        assert_eq!(residual.len(), 1);
        assert_eq!(residual[0].0, "weight");
        assert_eq!(residual[0].1.dims(), &[2, 4, 3]);
        Ok(())
    }

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(lhs.len(), rhs.len());
        for (index, (&lhs, &rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= ASSERT_EPS,
                "index={index} lhs={lhs} rhs={rhs} diff={diff}"
            );
        }
        Ok(())
    }

    fn packed_shape() -> PackedMambaShape {
        PackedMambaShape {
            physical_batch: 1,
            physical_tokens: 7,
            conv_state_batch: 3,
            conv_dim: CONV_DIM,
            conv_width: CONV_WIDTH,
            ssm_state_batch: 3,
            ssm_heads: NUM_HEADS,
            ssm_head_dim: HEAD_DIM,
            ssm_state_width: STATE_SIZE,
            expected_conv_dim: CONV_DIM,
            expected_conv_width: CONV_WIDTH,
            expected_ssm_heads: NUM_HEADS,
            expected_ssm_head_dim: HEAD_DIM,
            expected_ssm_state_width: STATE_SIZE,
        }
    }

    #[test]
    fn packed_mamba_matches_independent_unequal_prefills() -> Result<()> {
        let device = Device::Cpu;
        let layer = mamba_layer(&device)?;
        let x = Tensor::from_vec(
            patterned(5 * HIDDEN_SIZE, 13, 0.2, 0.01),
            (1, 5, HIDDEN_SIZE),
            &device,
        )?;
        let initial_cache = mamba_cache(&device)?;
        let mut packed_cache = initial_cache.clone();
        let packed = layer.forward_packed_prefill(&x, &mut packed_cache, &[2, 3])?;

        let mut reference_outputs = Vec::new();
        let mut reference_conv_states = Vec::new();
        let mut reference_ssm_states = Vec::new();
        let mut offset = 0;
        for (state_index, segment_len) in [2, 3].into_iter().enumerate() {
            let mut segment_cache = MambaLayerCache {
                conv_state: initial_cache.conv_state.narrow(0, state_index, 1)?,
                ssm_state: initial_cache.ssm_state.narrow(0, state_index, 1)?,
            };
            reference_outputs.push(layer.forward(
                &x.narrow(1, offset, segment_len)?,
                &mut segment_cache,
                RecurrentBatchKind::Prefill,
            )?);
            reference_conv_states.push(segment_cache.conv_state);
            reference_ssm_states.push(segment_cache.ssm_state);
            offset += segment_len;
        }
        let reference = Tensor::cat(&reference_outputs, 1)?;
        let reference_conv_state = Tensor::cat(&reference_conv_states, 0)?;
        let reference_ssm_state = Tensor::cat(&reference_ssm_states, 0)?;

        assert_close(&packed, &reference)?;
        assert_close(&packed_cache.conv_state, &reference_conv_state)?;
        assert_close(&packed_cache.ssm_state, &reference_ssm_state)
    }

    #[test]
    fn packed_expert_forward_matches_dense_grouping() -> Result<()> {
        const EXPERTS: usize = 3;
        const TOP_K: usize = 2;
        const EXPERT_INTERMEDIATE: usize = 3;

        let device = Device::Cpu;
        let input_weights = Tensor::from_vec(
            patterned(
                EXPERTS * EXPERT_INTERMEDIATE * 2 * HIDDEN_SIZE,
                20,
                0.08,
                0.01,
            ),
            (EXPERTS, EXPERT_INTERMEDIATE * 2, HIDDEN_SIZE),
            &device,
        )?;
        let output_weights = Tensor::from_vec(
            patterned(EXPERTS * HIDDEN_SIZE * EXPERT_INTERMEDIATE, 21, 0.09, -0.01),
            (EXPERTS, HIDDEN_SIZE, EXPERT_INTERMEDIATE),
            &device,
        )?;
        let router = unquant_linear(
            Tensor::from_vec(
                patterned(EXPERTS * HIDDEN_SIZE, 22, 0.1, 0.0),
                (EXPERTS, HIDDEN_SIZE),
                &device,
            )?,
            None,
        )?;
        let dense = GraniteMoE {
            input_linear: GraniteParallelExperts {
                weights: GraniteParallelExpertWeights::Dense(
                    (0..EXPERTS)
                        .map(|expert| input_weights.i(expert))
                        .collect::<Result<Vec<_>>>()?,
                ),
                output_size: EXPERT_INTERMEDIATE * 2,
            },
            output_linear: GraniteParallelExperts {
                weights: GraniteParallelExpertWeights::Dense(
                    (0..EXPERTS)
                        .map(|expert| output_weights.i(expert))
                        .collect::<Result<Vec<_>>>()?,
                ),
                output_size: HIDDEN_SIZE,
            },
            router: GraniteTopKGating {
                layer: router.clone(),
                num_experts: EXPERTS,
                top_k: TOP_K,
            },
            input_size: HIDDEN_SIZE,
        };
        let packed_input = unquant_linear(input_weights, None)?;
        let packed_output = unquant_linear(output_weights, None)?;
        let packed = GraniteMoE {
            input_linear: GraniteParallelExperts {
                weights: GraniteParallelExpertWeights::Quantized(packed_input.clone()),
                output_size: EXPERT_INTERMEDIATE * 2,
            },
            output_linear: GraniteParallelExperts {
                weights: GraniteParallelExpertWeights::Quantized(packed_output.clone()),
                output_size: HIDDEN_SIZE,
            },
            router: GraniteTopKGating {
                layer: router,
                num_experts: EXPERTS,
                top_k: TOP_K,
            },
            input_size: HIDDEN_SIZE,
        };
        let input = Tensor::from_vec(
            patterned(2 * 3 * HIDDEN_SIZE, 23, 0.2, 0.01),
            (2, 3, HIDDEN_SIZE),
            &device,
        )?;
        packed_input.begin_track_stats()?;
        packed_output.begin_track_stats()?;
        let packed_output_tensor = packed.forward(&input)?;
        assert_eq!(packed_input.stats_snapshot(), Some((1, 12)));
        assert_eq!(packed_output.stats_snapshot(), Some((1, 12)));
        assert_close(&packed_output_tensor, &dense.forward(&input)?)
    }

    #[test]
    fn packed_expert_forward_records_routed_stats() -> Result<()> {
        let device = Device::Cpu;
        let weight = unquant_linear(Tensor::ones((2, 2, 3), DType::F32, &device)?, None)?;
        let experts = GraniteParallelExperts {
            weights: GraniteParallelExpertWeights::Quantized(weight.clone()),
            output_size: 2,
        };
        let input = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]], &device)?;

        weight.begin_track_stats()?;
        experts.forward(&input, &[1, 2])?;
        assert_eq!(weight.stats_snapshot(), Some((1, 3)));
        assert_eq!(
            weight.end_track_stats()?.to_vec2::<f32>()?,
            vec![
                vec![1., 4., 9.],
                vec![(16. + 49.) / 2., (25. + 64.) / 2., (36. + 81.) / 2.]
            ]
        );
        Ok(())
    }

    #[test]
    fn mamba_prefill_continuation_matches_one_shot() -> Result<()> {
        let device = Device::Cpu;
        let layer = mamba_layer(&device)?;
        let seq_len = 6;
        let split = 2;
        let x = Tensor::from_vec(
            patterned(seq_len * HIDDEN_SIZE, 14, 0.2, 0.01),
            (1, seq_len, HIDDEN_SIZE),
            &device,
        )?;
        let initial = mamba_cache(&device)?;
        let initial = MambaLayerCache {
            conv_state: initial.conv_state.narrow(0, 0, 1)?,
            ssm_state: initial.ssm_state.narrow(0, 0, 1)?,
        };
        let mut one_shot_cache = initial.clone();
        let mut chunked_cache = initial;

        let one_shot = layer.forward(&x, &mut one_shot_cache, RecurrentBatchKind::Prefill)?;
        let first = layer.forward(
            &x.narrow(1, 0, split)?,
            &mut chunked_cache,
            RecurrentBatchKind::Prefill,
        )?;
        let second = layer.forward(
            &x.narrow(1, split, seq_len - split)?,
            &mut chunked_cache,
            RecurrentBatchKind::Prefill,
        )?;

        assert_close(&one_shot, &Tensor::cat(&[first, second], 1)?)?;
        assert_close(&one_shot_cache.conv_state, &chunked_cache.conv_state)?;
        assert_close(&one_shot_cache.ssm_state, &chunked_cache.ssm_state)
    }

    #[test]
    fn packed_mamba_maps_tokens_to_state_rows() {
        assert_eq!(
            packed_mamba_ranges(packed_shape(), &[2, 1, 4]).unwrap(),
            vec![0..2, 2..3, 3..7]
        );
    }

    #[test]
    fn packed_mamba_rejects_cardinality_mismatches() {
        let mut wrong_conv_batch = packed_shape();
        wrong_conv_batch.conv_state_batch = 2;
        assert!(packed_mamba_ranges(wrong_conv_batch, &[2, 1, 4]).is_err());

        let mut wrong_ssm_batch = packed_shape();
        wrong_ssm_batch.ssm_state_batch = 2;
        assert!(packed_mamba_ranges(wrong_ssm_batch, &[2, 1, 4]).is_err());

        assert!(packed_mamba_ranges(packed_shape(), &[2, 4]).is_err());
        assert!(packed_mamba_ranges(packed_shape(), &[2, 0, 5]).is_err());
    }

    #[test]
    fn packed_mamba_rejects_incompatible_shapes() {
        let mut non_packed_batch = packed_shape();
        non_packed_batch.physical_batch = 3;
        assert!(packed_mamba_ranges(non_packed_batch, &[2, 1, 4]).is_err());

        let mut wrong_conv_width = packed_shape();
        wrong_conv_width.conv_width = CONV_WIDTH - 1;
        assert!(packed_mamba_ranges(wrong_conv_width, &[2, 1, 4]).is_err());

        let mut wrong_ssm_width = packed_shape();
        wrong_ssm_width.ssm_state_width = STATE_SIZE + 1;
        assert!(packed_mamba_ranges(wrong_ssm_width, &[2, 1, 4]).is_err());
    }
}
