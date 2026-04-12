//! Unified MoE experts layer supporting multiple backends and weight formats.
//!
//! This module provides `MoEExperts`, a flexible experts layer that:
//! - Does NOT carry the gate (router) - gate is external
//! - Supports both per-expert and stacked weight formats
//! - Handles backend selection (fused/fast/slow)
//! - Manages tensor parallelism with all-reduce

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    apply_immediate_isq, should_apply_immediate_isq, DummyLayer, FusedExperts,
    PackedExperts, QuantMethod, QuantMethodConfig, QuantizedConfig, ShardedVarBuilder,
    SumAllReduce, UnquantLinear,
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
    num_experts: usize,
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
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            all_reduce: SumAllReduce::new(comm),
            world_size: comm.world_size(),
        })
    }

    /// Create MoEExperts from a VarBuilder already at the experts level.
    ///
    /// Unlike `new` which does `vb.pp("experts")` internally, this takes the VB
    /// already pointing at the experts-level path. Use this when the model's weight
    /// structure doesn't have an "experts" sublevel (e.g., Gemma 4 uses `moe.*` directly).
    ///
    /// Supports two weight formats:
    /// - Combined stacked: `gate_up_proj` [E, hidden, 2*inter]
    /// - Per-expert: `{i}/gate_proj/weight` [inter, hidden]
    pub fn new_direct(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        let layer_device = experts_vb.device().clone();
        let backend = MoEExpertsBackend::select(&layer_device, loading_isq, quantization_config);

        let is_stacked_combined = experts_vb.contains_tensor("gate_up_proj");

        let backend_impl = match backend {
            MoEExpertsBackend::Fused => {
                if is_stacked_combined {
                    MoEExpertsBackendImpl::Fused(Self::load_fused_stacked(cfg, experts_vb, comm)?)
                } else {
                    MoEExpertsBackendImpl::Fused(Self::load_fused_standard(cfg, experts_vb, comm)?)
                }
            }
            MoEExpertsBackend::Fast => {
                if is_stacked_combined && quantization_config.is_none() {
                    MoEExpertsBackendImpl::Fast(Self::load_fast_combined_stacked(cfg, experts_vb)?)
                } else if is_stacked_combined {
                    MoEExpertsBackendImpl::Slow(Self::load_slow_from_combined_stacked(
                        cfg, experts_vb,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fast(Self::load_fast_direct_standard(cfg, experts_vb)?)
                }
            }
            MoEExpertsBackend::Slow => {
                if is_stacked_combined {
                    MoEExpertsBackendImpl::Slow(Self::load_slow_from_combined_stacked(
                        cfg, experts_vb,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Slow(Self::load_slow(
                        cfg,
                        experts_vb,
                        comm,
                        quantization_config,
                    )?)
                }
            }
        };

        Ok(Self {
            backend: backend_impl,
            act,
            num_experts: cfg.num_experts,
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

        // Stacked format has two conventions:
        // Convention A: [num_experts, hidden, inter*2] (CUDA kernel format)
        // Convention B (nn.Linear): [num_experts, inter*2, hidden]
        // Try A first, fall back to B with transpose.
        let gate_up_w = experts_vb
            .get_with_hints(
                (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                "gate_up_proj",
                shard(2, comm.rank(), comm.world_size()),
            )
            .or_else(|_| {
                experts_vb
                    .get_with_hints(
                        (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                        "gate_up_proj",
                        shard(1, comm.rank(), comm.world_size()),
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })?;

        let down_w = experts_vb
            .get_with_hints(
                (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                "down_proj",
                shard(1, comm.rank(), comm.world_size()),
            )
            .or_else(|_| {
                experts_vb
                    .get_with_hints(
                        (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                        "down_proj",
                        shard(2, comm.rank(), comm.world_size()),
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })?;

        let w_size_n = gate_up_w.dim(2)? / 2;

        Ok(FusedExpertsWeights {
            gate_up_w,
            down_w,
            w_size_n,
            stacked_format: true,
        })
    }

    /// Load fast (gather-based) weights from a combined stacked `gate_up_proj`.
    ///
    /// Supports:
    /// - `gate_up_proj`: [E, hidden, 2*inter] or [E, 2*inter, hidden]
    /// - `down_proj`: [E, inter, hidden] or [E, hidden, inter]
    fn load_fast_combined_stacked(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
    ) -> Result<FastExpertsWeights> {
        let num_experts = cfg.num_experts;

        let isq_gate_up = should_apply_immediate_isq(&experts_vb.pp("gate_up_proj"));
        let isq_down = should_apply_immediate_isq(&experts_vb.pp("down_proj"));

        // When immediate ISQ is active, load directly on CPU to avoid creating
        // large GPU buffers that will be immediately copied to CPU for quantization.
        // On unified memory systems (Metal), this prevents doubling memory usage.
        let load_vb = if (isq_gate_up || isq_down) && !experts_vb.device().is_cpu() {
            experts_vb.clone().set_device(Device::Cpu)
        } else {
            experts_vb.clone()
        };

        let gate_up_proj = load_vb
            .get(
                (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                "gate_up_proj",
            )
            .or_else(|_| {
                load_vb
                    .get(
                        (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                        "gate_up_proj",
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })?;
        let down_proj_packed = load_vb
            .get(
                (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                "down_proj",
            )
            .or_else(|_| {
                load_vb
                    .get(
                        (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                        "down_proj",
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })?;

        let gate_proj = gate_up_proj
            .narrow(2, 0, cfg.moe_intermediate_size)?
            .transpose(1, 2)?
            .contiguous()?;
        let up_proj = gate_up_proj
            .narrow(2, cfg.moe_intermediate_size, cfg.moe_intermediate_size)?
            .transpose(1, 2)?
            .contiguous()?;
        // Drop gate_up_proj early to free memory before creating more tensors
        drop(gate_up_proj);
        let down_proj = down_proj_packed.transpose(1, 2)?.contiguous()?;
        drop(down_proj_packed);

        let mut fused_gate_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(gate_proj, None)),
        )?);
        let mut fused_up_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(up_proj, None)),
        )?);
        let mut fused_down_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(down_proj, None)),
        )?);

        // Pass the original-device VB (not CPU) so apply_immediate_isq targets
        // the correct device for the quantized weights.
        let vb_gate_up = experts_vb.pp("gate_up_proj");
        let vb_down = experts_vb.pp("down_proj");
        fused_gate_proj = apply_immediate_isq(fused_gate_proj, vb_gate_up.clone())?;
        fused_up_proj = apply_immediate_isq(fused_up_proj, vb_gate_up)?;
        fused_down_proj = apply_immediate_isq(fused_down_proj, vb_down)?;

        Ok(FastExpertsWeights {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        })
    }

    /// Load fast (gather-based) weights in per-expert format from a VB already
    /// at the experts level (no `.pp("experts")` applied).
    ///
    /// Handles both real per-expert weights and UQFF dummy layers.
    fn load_fast_direct_standard(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
    ) -> Result<FastExpertsWeights> {
        let num_experts = cfg.num_experts;

        // UQFF loading: experts have no real tensors yet, create dummy layers
        // that will be replaced during deserialization.
        if !experts_vb.pp("0").contains_tensor("gate_proj.weight") {
            let fused_gate_proj: Arc<dyn QuantMethod> =
                Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?);
            let fused_up_proj: Arc<dyn QuantMethod> =
                Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?);
            let fused_down_proj: Arc<dyn QuantMethod> =
                Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?);
            return Ok(FastExpertsWeights {
                fused_gate_proj,
                fused_up_proj,
                fused_down_proj,
            });
        }

        // Real per-expert weights: load, stack, and optionally ISQ
        let load_experts_vb =
            if mistralrs_quant::get_immediate_isq().is_some() && !experts_vb.device().is_cpu() {
                experts_vb.clone().set_device(Device::Cpu)
            } else {
                experts_vb.clone()
            };

        let mut gate_proj_vec = Vec::with_capacity(num_experts);
        let mut up_proj_vec = Vec::with_capacity(num_experts);
        let mut down_proj_vec = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            let expert_vb = load_experts_vb.pp(i.to_string());
            gate_proj_vec.push(expert_vb.get(
                (cfg.moe_intermediate_size, cfg.hidden_size),
                "gate_proj.weight",
            )?);
            up_proj_vec.push(expert_vb.get(
                (cfg.moe_intermediate_size, cfg.hidden_size),
                "up_proj.weight",
            )?);
            down_proj_vec.push(expert_vb.get(
                (cfg.hidden_size, cfg.moe_intermediate_size),
                "down_proj.weight",
            )?);
        }

        let mut fused_gate_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(Tensor::stack(&gate_proj_vec, 0)?, None)),
        )?);
        let mut fused_up_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(Tensor::stack(&up_proj_vec, 0)?, None)),
        )?);
        let mut fused_down_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(Tensor::stack(&down_proj_vec, 0)?, None)),
        )?);

        let expert0_vb = experts_vb.pp("0");
        fused_gate_proj = apply_immediate_isq(fused_gate_proj, expert0_vb.pp("gate_proj"))?;
        fused_up_proj = apply_immediate_isq(fused_up_proj, expert0_vb.pp("up_proj"))?;
        fused_down_proj = apply_immediate_isq(fused_down_proj, expert0_vb.pp("down_proj"))?;

        Ok(FastExpertsWeights {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        })
    }

    /// Load slow (loop-based) weights from a combined stacked `gate_up_proj`.
    ///
    /// Supports both direct stacked conventions used by Gemma4 checkpoints:
    /// - `gate_up_proj`: [E, hidden, 2*inter] or [E, 2*inter, hidden]
    /// - `down_proj`: [E, inter, hidden] or [E, hidden, inter]
    fn load_slow_from_combined_stacked(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
    ) -> Result<SlowExpertsWeights> {
        let num_experts = cfg.num_experts;

        let isq_gate_up = should_apply_immediate_isq(&experts_vb.pp("gate_up_proj"));
        let isq_down = should_apply_immediate_isq(&experts_vb.pp("down_proj"));

        // When immediate ISQ is active, load directly on CPU to avoid creating
        // large GPU buffers that will be immediately copied to CPU for quantization.
        let load_vb = if (isq_gate_up || isq_down) && !experts_vb.device().is_cpu() {
            experts_vb.clone().set_device(Device::Cpu)
        } else {
            experts_vb.clone()
        };

        let gate_up_proj = load_vb
            .get(
                (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                "gate_up_proj",
            )
            .or_else(|_| {
                load_vb
                    .get(
                        (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                        "gate_up_proj",
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })?;
        let down_proj_packed = load_vb
            .get(
                (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                "down_proj",
            )
            .or_else(|_| {
                load_vb
                    .get(
                        (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                        "down_proj",
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })?;

        // Pass the original-device VB (not CPU) so apply_immediate_isq targets
        // the correct device for the quantized weights.
        let vb_gate_up = experts_vb.pp("gate_up_proj");
        let vb_down = experts_vb.pp("down_proj");

        let mut gate_proj = Vec::with_capacity(num_experts);
        let mut up_proj = Vec::with_capacity(num_experts);
        let mut down_proj = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            let gate_up_expert = gate_up_proj.i(i)?;
            let gate = gate_up_expert
                .narrow(1, 0, cfg.moe_intermediate_size)?
                .transpose(0, 1)?
                .contiguous()?;
            let up = gate_up_expert
                .narrow(1, cfg.moe_intermediate_size, cfg.moe_intermediate_size)?
                .transpose(0, 1)?
                .contiguous()?;
            let down = down_proj_packed.i(i)?.transpose(0, 1)?.contiguous()?;

            let mut gate_layer: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(candle_nn::Linear::new(gate, None)),
            )?);
            let mut up_layer: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(candle_nn::Linear::new(up, None)),
            )?);
            let mut down_layer: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(candle_nn::Linear::new(down, None)),
            )?);

            gate_layer = apply_immediate_isq(gate_layer, vb_gate_up.clone())?;
            up_layer = apply_immediate_isq(up_layer, vb_gate_up.clone())?;
            down_layer = apply_immediate_isq(down_layer, vb_down.clone())?;

            gate_proj.push(gate_layer);
            up_proj.push(up_layer);
            down_proj.push(down_layer);
        }

        Ok(SlowExpertsWeights {
            experts: PackedExperts {
                gate_proj,
                up_proj,
                down_proj,
            },
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

        #[cfg(feature = "cuda")]
        if xs.device().is_cuda() {
            let is_prefill = seq_len > 1;
            // Try fused decode path for single-token decode (most impactful)
            if !is_prefill {
                if let Some(result) = self.forward_fast_decode(
                    &xs_flat,
                    topk_weights,
                    topk_ids,
                    weights,
                    num_tokens,
                    original_dtype,
                )? {
                    return Ok(result);
                }
            }

            // Try grouped MoE path for CUDA prefill (much faster for many tokens)
            // Only use for large prefills where the overhead is worthwhile
            if is_prefill && num_tokens >= 32 {
                if let Some(result) = self.forward_fast_grouped(
                    &xs_flat,
                    topk_weights,
                    topk_ids,
                    weights,
                    num_tokens,
                    original_dtype,
                )? {
                    return Ok(result);
                }
            }
        }

        let ys = if xs.device().is_cuda() {
            // CUDA path: use indexed_moe_forward compatible shapes
            let xs = xs_flat.reshape((num_tokens, 1, hidden_dim))?;
            let gate = weights
                .fused_gate_proj
                .gather_forward(&xs, topk_ids)?;
            let up = weights
                .fused_up_proj
                .gather_forward(&xs, topk_ids)?;
            weights
                .fused_down_proj
                .gather_forward(&(up * gate.apply(&self.act)?)?, topk_ids)?
        } else {
            // Metal path: use broadcast gather shapes
            let xs = xs.reshape((b_size, seq_len, 1, 1, hidden_dim))?;
            let indices = topk_ids.reshape((b_size, seq_len, self.num_experts_per_tok))?;
            let gate = weights
                .fused_gate_proj
                .gather_forward(&xs, &indices)?;
            let up = weights
                .fused_up_proj
                .gather_forward(&xs, &indices)?;
            let xs = weights
                .fused_down_proj
                .gather_forward(&(up * gate.apply(&self.act)?)?, &indices)?;
            xs.squeeze(D::Minus2)?
                .reshape((num_tokens, self.num_experts_per_tok, hidden_dim))?
        };

        ys.to_dtype(DType::F32)?
            .broadcast_mul(&topk_weights.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(original_dtype)
    }

    /// Fused MoE decode path for CUDA.
    ///
    /// Reduces kernel launches from ~20 to 4 per MoE layer by:
    /// 1. Quantizing input to Q8_1 once (shared between gate+up)
    /// 2. Fusing gate+up projections with activation+multiply in one kernel
    /// 3. Fusing down projection with topk_weights and cross-expert aggregation
    ///
    /// Returns Ok(Some(result)) if fused path succeeded, Ok(None) to fall back.
    #[cfg(feature = "cuda")]
    fn forward_fast_decode(
        &self,
        xs_flat: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        weights: &FastExpertsWeights,
        num_tokens: usize,
        original_dtype: DType,
    ) -> Result<Option<Tensor>> {
        use candle_core::cuda::cudarc::driver::DevicePtr;

        let dev = xs_flat.device().as_cuda_device()?;

        // Get QTensors - bail to fallback if not quantized
        let gate_qt = match weights.fused_gate_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let up_qt = match weights.fused_up_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let down_qt = match weights.fused_down_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };

        // Get topk_ids as contiguous u32 CudaSlice
        let topk_ids_flat = topk_ids.flatten_all()?.contiguous()?;
        let (ti_storage, ti_layout) = topk_ids_flat.storage_and_layout();
        let ti_cuda = match &*ti_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let ti_u32_slice = ti_cuda.as_cuda_slice::<u32>()?;
        assert!(ti_layout.start_offset() == 0, "expected contiguous tensor");

        // Get topk_weights as contiguous f32 CudaSlice
        let tw_f32 = topk_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (tw_storage, tw_layout) = tw_f32.storage_and_layout();
        let tw_cuda = match &*tw_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let tw_slice = tw_cuda.as_cuda_slice::<f32>()?;
        let tw_ptr = tw_slice
            .slice(tw_layout.start_offset()..)
            .device_ptr(tw_slice.stream())
            .0 as *const f32;

        // Map activation to CUDA kernel act_type
        let act_type = match self.act {
            Activation::GeluPytorchTanh => mistralrs_quant::ACT_GELU_PYTORCH_TANH,
            Activation::Silu | Activation::Swish => mistralrs_quant::ACT_SILU,
            _ => return Ok(None), // Fall back for unsupported activations
        };

        let result = mistralrs_quant::indexed_moe_fused_decode(
            gate_qt,
            up_qt,
            down_qt,
            xs_flat,
            ti_u32_slice,
            tw_ptr,
            num_tokens,
            self.num_experts_per_tok,
            act_type,
            dev,
        )?;

        Ok(Some(result.to_dtype(original_dtype)?))
    }

    /// Grouped MoE forward for CUDA prefill.
    ///
    /// Returns Ok(Some(result)) if grouped path succeeded, Ok(None) to fall back.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_fast_grouped(
        &self,
        xs_flat: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        weights: &FastExpertsWeights,
        num_tokens: usize,
        original_dtype: DType,
    ) -> Result<Option<Tensor>> {
        let topk = self.num_experts_per_tok;
        let num_experts = self.num_experts;
        let total_assignments = num_tokens * topk;

        let dev = xs_flat.device().as_cuda_device()?;

        // Get topk_ids as contiguous u32 CudaSlice
        let topk_ids_flat = topk_ids.flatten_all()?.contiguous()?;
        let (ti_storage, ti_layout) = topk_ids_flat.storage_and_layout();
        let ti_cuda = match &*ti_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let ti_u32_slice = ti_cuda.as_cuda_slice::<u32>()?;
        assert!(ti_layout.start_offset() == 0, "expected contiguous tensor");

        // Build dispatch tables on GPU (no CPU-GPU sync)
        // moe_dispatch_build takes u32 and casts to i32 internally for the CUDA kernel
        let (expert_bounds, sorted_token_ids) =
            mistralrs_quant::moe_dispatch_build(ti_u32_slice, total_assignments, num_experts, dev)?;

        // Use the pre-quantized Q8_0 grouped kernel path
        let gate_qt = match weights.fused_gate_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let up_qt = match weights.fused_up_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let down_qt = match weights.fused_down_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };

        // Quantize input to Q8_1 ONCE, shared between gate and up.
        // quantize_input_q8_1 accepts BF16/F16/F32 directly (no conversion needed).
        let (input_q8, k, k_padded) = mistralrs_quant::quantize_input_q8_1(xs_flat, dev)?;

        // Gate projection using pre-quantized input
        let gate = mistralrs_quant::grouped_moe_gemm_prequantized(
            gate_qt,
            &input_q8,
            k,
            k_padded,
            &expert_bounds,
            &sorted_token_ids,
            None,
            total_assignments,
            topk,
            num_experts,
            1,
            dev,
        )?;

        // Up projection reusing same pre-quantized input
        let up = mistralrs_quant::grouped_moe_gemm_prequantized(
            up_qt,
            &input_q8,
            k,
            k_padded,
            &expert_bounds,
            &sorted_token_ids,
            None,
            total_assignments,
            topk,
            num_experts,
            1,
            dev,
        )?;

        drop(input_q8);

        // Apply activation
        let activated = (up * gate.apply(&self.act)?)?.contiguous()?;

        // Quantize down projection input
        let (down_input_q8, down_k, down_k_padded) =
            mistralrs_quant::quantize_input_q8_1(&activated, dev)?;

        // Get topk_weights pointer
        use candle_core::cuda::cudarc::driver::DevicePtr;
        let tw_f32 = topk_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (tw_storage, tw_layout) = tw_f32.storage_and_layout();
        let tw_cuda = match &*tw_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let tw_slice = tw_cuda.as_cuda_slice::<f32>()?;
        let tw_ptr = tw_slice
            .slice(tw_layout.start_offset()..)
            .device_ptr(tw_slice.stream())
            .0 as *const f32;

        // Down projection with topk_weights + atomicAdd
        let down = mistralrs_quant::grouped_moe_gemm_prequantized(
            down_qt,
            &down_input_q8,
            down_k,
            down_k_padded,
            &expert_bounds,
            &sorted_token_ids,
            Some((tw_ptr, 0)),
            total_assignments,
            topk,
            num_experts,
            0,
            dev,
        )?;

        Ok(Some(down.to_dtype(original_dtype)?))
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
                let expert_idx = expert_idx as usize;
                #[allow(clippy::cast_possible_truncation)]
                top_x[expert_idx].push(row_idx as u32);
                selected_experts[expert_idx].push(rw)
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
            let expert_input = current_state.clone();
            let gate_out = weights.experts.gate_proj[expert_idx].forward(&expert_input)?
                .apply(&self.act)?;
            let up_out =
                weights.experts.up_proj[expert_idx].forward(&expert_input)?;
            let current_hidden_states =
                weights.experts.down_proj[expert_idx].forward(&(gate_out * up_out)?)?;

            let current_hidden_states =
                current_hidden_states.broadcast_mul(&selected_experts_tensor)?;
            ys = ys.index_add(&top_x_tensor, &current_hidden_states, 0)?;
        }

        ys.reshape((b_size * seq_len, hidden_dim))
    }

    /// Get mutable references to quantizable layers for ISQ
    /// Returns mutable references to all ISQ-quantizable layers.
    /// The count must match `num_isq_layers`.
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

    /// Returns the number of ISQ-quantizable layers.
    /// Must match the length of `get_isq_layers`.
    pub fn num_isq_layers(&self) -> usize {
        match &self.backend {
            MoEExpertsBackendImpl::Fused(_) => 0,
            MoEExpertsBackendImpl::Fast(_) => 3,
            MoEExpertsBackendImpl::Slow(weights) => weights.experts.gate_proj.len() * 3,
        }
    }
}
