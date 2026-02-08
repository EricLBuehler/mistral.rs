//! Unified MoE experts layer supporting multiple backends and weight formats.
//!
//! This module provides `MoEExperts`, a flexible experts layer that:
//! - Does NOT carry the gate (router) - gate is external
//! - Supports both per-expert and stacked weight formats
//! - Handles backend selection (fused/fast/slow)
//! - Manages tensor parallelism with all-reduce

use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{
    FusedExperts, MatMul, PackedExperts, QuantMethod, QuantizedConfig, ShardedVarBuilder,
    SumAllReduce,
};
use std::sync::Arc;

use crate::cuda::moe;
use crate::layers::Activation;
use crate::moe::shard;

/// Configuration for MoEExperts
pub struct MoEExpertsConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
}

/// Backend selection for MoE experts
pub enum MoEExpertsBackend {
    /// Use fused CUDA kernels with raw tensors (fastest for CUDA unquantized)
    Fused,
    /// Use gather-based implementation (good for Metal, ISQ)
    Fast,
    /// Use loop-based implementation (fallback for quantized)
    Slow,
}

impl MoEExpertsBackend {
    /// Determine the best backend based on device and quantization settings
    pub fn select(
        device: &Device,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
    ) -> Self {
        let has_immediate_isq = mistralrs_quant::get_immediate_isq().is_some();
        let use_fast = device.is_metal()
            || (device.is_cuda()
                && (loading_isq || quantization_config.is_some() || has_immediate_isq));

        if use_fast {
            Self::Fast
        } else if quantization_config.is_none()
            && !loading_isq
            && !has_immediate_isq
            && device.is_cuda()
        {
            Self::Fused
        } else {
            Self::Slow
        }
    }
}

/// Internal representation of fused expert weights for CUDA kernels
struct FusedExpertsWeights {
    /// gate_up weights: [E, N, K] for standard, [E, K, N] for stacked
    gate_up_w: Tensor,
    /// down weights: [E, N, K] for standard, [E, K, N] for stacked
    down_w: Tensor,
    /// Size of intermediate dimension (after sharding)
    w_size_n: usize,
    /// Whether weights are in stacked format [E, K, N]
    stacked_format: bool,
}

/// Internal representation for gather-based experts (Metal/ISQ)
struct FastExpertsWeights {
    fused_gate_proj: Arc<dyn QuantMethod>,
    fused_up_proj: Arc<dyn QuantMethod>,
    fused_down_proj: Arc<dyn QuantMethod>,
}

/// Internal representation for loop-based experts (quantized fallback)
struct SlowExpertsWeights {
    experts: PackedExperts,
}

/// MoE experts layer without gate
///
/// This struct encapsulates the expert weights and forward logic,
/// but does NOT include the routing gate. The caller is responsible
/// for computing routing weights and topk indices.
pub struct MoEExperts {
    backend: MoEExpertsBackendImpl,
    act: Activation,
    num_experts_per_tok: usize,
    all_reduce: SumAllReduce,
    world_size: usize,
}

enum MoEExpertsBackendImpl {
    Fused(FusedExpertsWeights),
    Fast(FastExpertsWeights),
    Slow(SlowExpertsWeights),
}

impl MoEExperts {
    /// Create MoEExperts with automatic backend selection
    ///
    /// Automatically detects weight format (stacked vs per-expert) and
    /// selects the appropriate backend based on device and quantization.
    pub fn new(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        let backend = MoEExpertsBackend::select(&layer_device, loading_isq, quantization_config);
        Self::new_with_backend(
            cfg,
            vb,
            layer_device,
            comm,
            backend,
            quantization_config,
            act,
        )
    }

    /// Create MoEExperts with explicit backend selection
    pub fn new_with_backend(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        backend: MoEExpertsBackend,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        let experts_vb = vb.pp("experts").set_device(layer_device.clone());

        // Detect format: stacked has "gate_up_proj", per-expert has "0.gate_proj"
        let is_stacked = experts_vb.contains_tensor("gate_up_proj");

        let backend_impl = match backend {
            MoEExpertsBackend::Fused => {
                if is_stacked {
                    MoEExpertsBackendImpl::Fused(Self::load_fused_stacked(cfg, experts_vb, comm)?)
                } else {
                    MoEExpertsBackendImpl::Fused(Self::load_fused_standard(cfg, experts_vb, comm)?)
                }
            }
            MoEExpertsBackend::Fast => {
                if is_stacked {
                    MoEExpertsBackendImpl::Fast(Self::load_fast_stacked(
                        cfg,
                        vb,
                        quantization_config,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fast(Self::load_fast_standard(
                        cfg,
                        vb,
                        quantization_config,
                    )?)
                }
            }
            MoEExpertsBackend::Slow => MoEExpertsBackendImpl::Slow(Self::load_slow(
                cfg,
                experts_vb,
                comm,
                quantization_config,
            )?),
        };

        Ok(Self {
            backend: backend_impl,
            act,
            num_experts_per_tok: cfg.num_experts_per_tok,
            all_reduce: SumAllReduce::new(comm),
            world_size: comm.world_size(),
        })
    }

    /// Load fused weights in standard per-expert format
    fn load_fused_standard(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<FusedExpertsWeights> {
        let num_experts = cfg.num_experts;
        let mut gate_up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            let expert_vb = experts_vb.pp(i.to_string());
            // n x k format
            let gate_expert = expert_vb.pp("gate_proj").get_with_hints(
                (cfg.moe_intermediate_size, cfg.hidden_size),
                "weight",
                shard(0, comm.rank(), comm.world_size()),
            )?;
            let up_expert = expert_vb.pp("up_proj").get_with_hints(
                (cfg.moe_intermediate_size, cfg.hidden_size),
                "weight",
                shard(0, comm.rank(), comm.world_size()),
            )?;
            let down_expert = expert_vb.pp("down_proj").get_with_hints(
                (cfg.hidden_size, cfg.moe_intermediate_size),
                "weight",
                shard(1, comm.rank(), comm.world_size()),
            )?;
            // Pack gate_proj and up_proj
            let gate_up_expert = Tensor::cat(&[&gate_expert, &up_expert], 0)?;

            gate_up_experts.push(gate_up_expert);
            down_experts.push(down_expert);
        }

        let gate_up_w = Tensor::stack(&gate_up_experts, 0)?;
        let down_w = Tensor::stack(&down_experts, 0)?;
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(FusedExpertsWeights {
            gate_up_w,
            down_w,
            w_size_n,
            stacked_format: false,
        })
    }

    /// Load fused weights in stacked format
    fn load_fused_stacked(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<FusedExpertsWeights> {
        let num_experts = cfg.num_experts;

        // Load stacked gate_up_proj: [num_experts, hidden, inter*2]
        let gate_up_w = experts_vb.get_with_hints(
            (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
            "gate_up_proj",
            shard(2, comm.rank(), comm.world_size()),
        )?;

        // Load stacked down_proj: [num_experts, inter, hidden]
        let down_w = experts_vb.get_with_hints(
            (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
            "down_proj",
            shard(1, comm.rank(), comm.world_size()),
        )?;

        let w_size_n = gate_up_w.dim(2)? / 2;

        Ok(FusedExpertsWeights {
            gate_up_w,
            down_w,
            w_size_n,
            stacked_format: true,
        })
    }

    /// Load fast (gather-based) weights in standard per-expert format
    fn load_fast_standard(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        quantization_config: &Option<QuantizedConfig>,
    ) -> Result<FastExpertsWeights> {
        let FusedExperts {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        } = FusedExperts::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts,
            quantization_config,
            vb,
        )?;

        Ok(FastExpertsWeights {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        })
    }

    /// Load fast (gather-based) weights in stacked format
    fn load_fast_stacked(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        quantization_config: &Option<QuantizedConfig>,
    ) -> Result<FastExpertsWeights> {
        // FusedExperts auto-detects stacked format
        let FusedExperts {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        } = FusedExperts::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts,
            quantization_config,
            vb,
        )?;

        Ok(FastExpertsWeights {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        })
    }

    /// Load slow (loop-based) weights using PackedExperts
    fn load_slow(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        quantization_config: &Option<QuantizedConfig>,
    ) -> Result<SlowExpertsWeights> {
        let experts = PackedExperts::new(
            cfg.num_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            quantization_config,
            false,
            comm,
            experts_vb,
        )?;

        Ok(SlowExpertsWeights { experts })
    }

    /// Forward pass through experts
    ///
    /// # Arguments
    /// * `xs` - Input tensor of shape [batch, seq_len, hidden_dim]
    /// * `topk_weights` - Top-k routing weights of shape [num_tokens, num_experts_per_tok]
    /// * `topk_ids` - Top-k expert indices of shape [num_tokens, num_experts_per_tok]
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_dim]
    pub fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        // Prefill = processing multiple tokens; Decode = single token generation
        let is_prefill = seq_len > 1;

        let mut ys = match &self.backend {
            MoEExpertsBackendImpl::Fused(weights) => {
                self.forward_fused(xs, &topk_weights, topk_ids, weights, is_prefill)?
            }
            MoEExpertsBackendImpl::Fast(weights) => {
                self.forward_fast(xs, &topk_weights, topk_ids, weights)?
            }
            MoEExpertsBackendImpl::Slow(weights) => {
                self.forward_slow(xs, &topk_weights, topk_ids, weights)?
            }
        };

        // Apply all-reduce for tensor parallelism
        if self.world_size > 1 {
            ys = self.all_reduce.sum_all_reduce(&ys)?;
        }

        ys.reshape((b_size, seq_len, hidden_dim))
    }

    /// Fused CUDA kernel forward pass
    fn forward_fused(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        weights: &FusedExpertsWeights,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (_b_size, _seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let (num_tokens, _) = xs.dims2()?;

        // Sort tokens by expert for efficient processing
        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use crate::ops::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        // First GEMM: gate_up projection
        let gate_up = if weights.stacked_format {
            moe::moe_gemm_transposed(
                &xs,
                &weights.gate_up_w,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        } else {
            moe::moe_gemm(
                &xs,
                &weights.gate_up_w,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        };

        // Split and apply activation
        let gate = gate_up
            .narrow(D::Minus1, 0, weights.w_size_n)?
            .contiguous()?;
        let up = gate_up
            .narrow(D::Minus1, weights.w_size_n, weights.w_size_n)?
            .contiguous()?;

        let down_inputs = (up * gate.apply(&self.act)?)?.reshape(((), weights.w_size_n))?;

        // Second GEMM: down projection with weight aggregation
        let ys = if weights.stacked_format {
            moe::moe_gemm_transposed(
                &down_inputs,
                &weights.down_w,
                &Some(topk_weights.clone()),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        } else {
            moe::moe_gemm(
                &down_inputs,
                &weights.down_w,
                &Some(topk_weights.clone()),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        };

        ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)
    }

    /// Gather-based forward pass (Metal/ISQ)
    fn forward_fast(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        weights: &FastExpertsWeights,
    ) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let num_tokens = b_size * seq_len;

        let xs_flat = xs.reshape((num_tokens, hidden_dim))?;

        let ys = if xs.device().is_cuda() {
            // CUDA path: use indexed_moe_forward compatible shapes
            let xs = xs_flat.reshape((num_tokens, 1, hidden_dim))?;
            let gate = weights
                .fused_gate_proj
                .gather_forward_autocast(&xs, topk_ids)?;
            let up = weights
                .fused_up_proj
                .gather_forward_autocast(&xs, topk_ids)?;
            weights
                .fused_down_proj
                .gather_forward_autocast(&(up * gate.apply(&self.act)?)?, topk_ids)?
        } else {
            // Metal path: use broadcast gather shapes
            let xs = xs.reshape((b_size, seq_len, 1, 1, hidden_dim))?;
            let indices = topk_ids.reshape((b_size, seq_len, self.num_experts_per_tok))?;
            let gate = weights
                .fused_gate_proj
                .gather_forward_autocast(&xs, &indices)?;
            let up = weights
                .fused_up_proj
                .gather_forward_autocast(&xs, &indices)?;
            let xs = weights
                .fused_down_proj
                .gather_forward_autocast(&(up * gate.apply(&self.act)?)?, &indices)?;
            xs.squeeze(D::Minus2)?
                .reshape((num_tokens, self.num_experts_per_tok, hidden_dim))?
        };

        ys.to_dtype(DType::F32)?
            .broadcast_mul(&topk_weights.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(original_dtype)
    }

    /// Loop-based forward pass (quantized fallback)
    fn forward_slow(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        weights: &SlowExpertsWeights,
    ) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;

        let routing_weights = topk_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let experts_per_tok = topk_ids.to_vec2::<u32>()?;
        let num_experts = weights.experts.gate_proj.len();

        let mut top_x = vec![vec![]; num_experts];
        let mut selected_experts = vec![vec![]; num_experts];

        for (row_idx, (rw, expert_idxs)) in routing_weights
            .iter()
            .zip(experts_per_tok.iter())
            .enumerate()
        {
            for (&rw, &expert_idx) in rw.iter().zip(expert_idxs.iter()) {
                #[allow(clippy::cast_possible_truncation)]
                top_x[expert_idx as usize].push(row_idx as u32);
                selected_experts[expert_idx as usize].push(rw)
            }
        }

        let mut ys = xs.zeros_like()?;
        for expert_idx in 0..num_experts {
            let top_x_expert = &top_x[expert_idx];
            if top_x_expert.is_empty() {
                continue;
            }
            let top_x_tensor = Tensor::new(top_x_expert.as_slice(), xs.device())?;
            let selected_experts_tensor =
                Tensor::new(selected_experts[expert_idx].as_slice(), xs.device())?
                    .reshape(((), 1))?
                    .to_dtype(xs.dtype())?;
            let current_state = xs
                .index_select(&top_x_tensor, 0)?
                .reshape(((), hidden_dim))?;

            // Forward through expert MLP
            let original_dtype = current_state.dtype();
            let mut expert_input = current_state.clone();
            if let Some(t) = weights.experts.gate_proj[expert_idx].quantized_act_type() {
                expert_input = expert_input.to_dtype(t)?;
            }
            let gate_out = MatMul
                .qmethod_matmul(&expert_input, &*weights.experts.gate_proj[expert_idx])?
                .apply(&self.act)?;
            let up_out =
                MatMul.qmethod_matmul(&expert_input, &*weights.experts.up_proj[expert_idx])?;
            let mut current_hidden_states = MatMul.qmethod_matmul(
                &(gate_out * up_out)?,
                &*weights.experts.down_proj[expert_idx],
            )?;
            if weights.experts.gate_proj[expert_idx]
                .quantized_act_type()
                .is_some()
            {
                current_hidden_states = current_hidden_states.to_dtype(original_dtype)?;
            }

            let current_hidden_states =
                current_hidden_states.broadcast_mul(&selected_experts_tensor)?;
            ys = ys.index_add(&top_x_tensor, &current_hidden_states, 0)?;
        }

        ys.reshape((b_size * seq_len, hidden_dim))
    }

    /// Get mutable references to quantizable layers for ISQ
    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match &mut self.backend {
            MoEExpertsBackendImpl::Fused(_) => vec![],
            MoEExpertsBackendImpl::Fast(weights) => {
                vec![
                    &mut weights.fused_gate_proj,
                    &mut weights.fused_up_proj,
                    &mut weights.fused_down_proj,
                ]
            }
            MoEExpertsBackendImpl::Slow(weights) => {
                let mut layers = Vec::new();
                for (gate, (up, down)) in weights.experts.gate_proj.iter_mut().zip(
                    weights
                        .experts
                        .up_proj
                        .iter_mut()
                        .zip(weights.experts.down_proj.iter_mut()),
                ) {
                    layers.push(gate);
                    layers.push(up);
                    layers.push(down);
                }
                layers
            }
        }
    }
}
