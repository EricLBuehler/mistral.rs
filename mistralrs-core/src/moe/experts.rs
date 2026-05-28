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
    apply_immediate_isq, should_apply_immediate_isq, DummyLayer, FusedExperts, PackedExperts,
    QuantMethod, QuantMethodConfig, QuantizedConfig, ShardedVarBuilder, SumAllReduce,
    UnquantLinear,
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
#[derive(Clone, Copy)]
pub enum MoEExpertsBackend {
    /// Use fused CUDA kernels with raw tensors (fastest for CUDA unquantized)
    Fused,
    #[cfg(feature = "cutile")]
    Cutile,
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
        if let Ok(force) = std::env::var("MISTRALRS_MOE_BACKEND") {
            match force.as_str() {
                "fused" | "native" | "legacy" | "wmma" => return Self::Fused,
                "fast" => return Self::Fast,
                "slow" => return Self::Slow,
                #[cfg(feature = "cutile")]
                "cutile" => return Self::Cutile,
                _ => {}
            }
        }

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

#[cfg(feature = "cutile")]
struct CutileExpertsWeights {
    gate_up_w: Tensor,
    down_w: Tensor,
    w_size_n: usize,
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
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    num_experts: usize,
    num_experts_per_tok: usize,
    all_reduce: SumAllReduce,
    world_size: usize,
}

enum MoEExpertsBackendImpl {
    Fused(FusedExpertsWeights),
    #[cfg(feature = "cutile")]
    Cutile(CutileExpertsWeights),
    Fast(FastExpertsWeights),
    Slow(SlowExpertsWeights),
}

trait MoEBackendWeights {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor>;

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>>;

    fn num_isq_layers(&self) -> usize;
}

impl MoEExpertsBackendImpl {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        match self {
            Self::Fused(weights) => MoEBackendWeights::forward(weights, forward, config),
            #[cfg(feature = "cutile")]
            Self::Cutile(weights) => MoEBackendWeights::forward(weights, forward, config),
            Self::Fast(weights) => MoEBackendWeights::forward(weights, forward, config),
            Self::Slow(weights) => MoEBackendWeights::forward(weights, forward, config),
        }
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match self {
            Self::Fused(weights) => MoEBackendWeights::get_isq_layers(weights),
            #[cfg(feature = "cutile")]
            Self::Cutile(weights) => MoEBackendWeights::get_isq_layers(weights),
            Self::Fast(weights) => MoEBackendWeights::get_isq_layers(weights),
            Self::Slow(weights) => MoEBackendWeights::get_isq_layers(weights),
        }
    }

    fn num_isq_layers(&self) -> usize {
        match self {
            Self::Fused(weights) => MoEBackendWeights::num_isq_layers(weights),
            #[cfg(feature = "cutile")]
            Self::Cutile(weights) => MoEBackendWeights::num_isq_layers(weights),
            Self::Fast(weights) => MoEBackendWeights::num_isq_layers(weights),
            Self::Slow(weights) => MoEBackendWeights::num_isq_layers(weights),
        }
    }
}

#[derive(Clone, Copy)]
enum MoEForwardPhase {
    Prefill,
    Decode,
}

impl MoEForwardPhase {
    fn from_shape(seq_len: usize) -> Self {
        if seq_len > 1 {
            Self::Prefill
        } else {
            Self::Decode
        }
    }

    fn is_prefill(self) -> bool {
        matches!(self, Self::Prefill)
    }
}

#[derive(Clone, Copy)]
struct MoEForwardShape {
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_tokens: usize,
    phase: MoEForwardPhase,
}

impl MoEForwardShape {
    fn new(batch_size: usize, seq_len: usize, hidden_dim: usize) -> Self {
        Self {
            batch_size,
            seq_len,
            hidden_dim,
            num_tokens: batch_size * seq_len,
            phase: MoEForwardPhase::from_shape(seq_len),
        }
    }

    fn flat(self) -> (usize, usize) {
        (self.num_tokens, self.hidden_dim)
    }

    fn output(self) -> (usize, usize, usize) {
        (self.batch_size, self.seq_len, self.hidden_dim)
    }
}

struct MoEForward<'a> {
    xs: &'a Tensor,
    xs_flat: &'a Tensor,
    topk_weights: &'a Tensor,
    topk_ids: &'a Tensor,
    original_dtype: DType,
    shape: MoEForwardShape,
}

#[derive(Clone, Copy)]
struct MoEForwardConfig {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    num_experts: usize,
    num_experts_per_tok: usize,
    act: Activation,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum MoECudaFastPath {
    Decode,
    GroupedPrefill,
}

impl MoEExperts {
    #[cfg(feature = "cutile")]
    fn should_default_cutile(
        backend: MoEExpertsBackend,
        device: &Device,
        dtype: DType,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> bool {
        matches!(backend, MoEExpertsBackend::Fused)
            && std::env::var("MISTRALRS_MOE_BACKEND").is_err()
            && device.is_cuda()
            && dtype == DType::BF16
            && !loading_isq
            && quantization_config.is_none()
            && mistralrs_quant::get_immediate_isq().is_none()
            && matches!(act, Activation::NewGelu | Activation::GeluPytorchTanh)
    }

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
        #[cfg(feature = "cutile")]
        let backend = if Self::should_default_cutile(
            backend,
            &layer_device,
            vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        ) {
            MoEExpertsBackend::Cutile
        } else {
            backend
        };
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
                    MoEExpertsBackendImpl::Fused(FusedExpertsWeights::load_stacked(
                        cfg, experts_vb, comm,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fused(FusedExpertsWeights::load_standard(
                        cfg, experts_vb, comm,
                    )?)
                }
            }
            #[cfg(feature = "cutile")]
            MoEExpertsBackend::Cutile => {
                if is_stacked {
                    MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::load_stacked(
                        cfg, experts_vb, comm,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::load_standard(
                        cfg, experts_vb, comm,
                    )?)
                }
            }
            MoEExpertsBackend::Fast => {
                if is_stacked {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_stacked(
                        cfg,
                        vb,
                        quantization_config,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_standard(
                        cfg,
                        vb,
                        quantization_config,
                    )?)
                }
            }
            MoEExpertsBackend::Slow => MoEExpertsBackendImpl::Slow(SlowExpertsWeights::load(
                cfg,
                experts_vb,
                comm,
                quantization_config,
            )?),
        };

        Ok(Self::from_backend(backend_impl, cfg, comm, act))
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
        #[cfg(feature = "cutile")]
        let backend = if Self::should_default_cutile(
            backend,
            &layer_device,
            experts_vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        ) {
            MoEExpertsBackend::Cutile
        } else {
            backend
        };

        let is_stacked_combined = experts_vb.contains_tensor("gate_up_proj");

        let backend_impl = match backend {
            MoEExpertsBackend::Fused => {
                if is_stacked_combined {
                    MoEExpertsBackendImpl::Fused(FusedExpertsWeights::load_stacked(
                        cfg, experts_vb, comm,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fused(FusedExpertsWeights::load_standard(
                        cfg, experts_vb, comm,
                    )?)
                }
            }
            #[cfg(feature = "cutile")]
            MoEExpertsBackend::Cutile => {
                if is_stacked_combined {
                    MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::load_stacked(
                        cfg, experts_vb, comm,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::load_standard(
                        cfg, experts_vb, comm,
                    )?)
                }
            }
            MoEExpertsBackend::Fast => {
                if is_stacked_combined && quantization_config.is_none() {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_combined_stacked(
                        cfg, experts_vb,
                    )?)
                } else if is_stacked_combined {
                    MoEExpertsBackendImpl::Slow(SlowExpertsWeights::load_combined_stacked(
                        cfg, experts_vb,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_direct_standard(
                        cfg, experts_vb,
                    )?)
                }
            }
            MoEExpertsBackend::Slow => {
                if is_stacked_combined {
                    MoEExpertsBackendImpl::Slow(SlowExpertsWeights::load_combined_stacked(
                        cfg, experts_vb,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Slow(SlowExpertsWeights::load(
                        cfg,
                        experts_vb,
                        comm,
                        quantization_config,
                    )?)
                }
            }
        };

        Ok(Self::from_backend(backend_impl, cfg, comm, act))
    }

    fn from_backend(
        backend: MoEExpertsBackendImpl,
        cfg: &MoEExpertsConfig,
        comm: &Arc<mistralrs_quant::Comm>,
        act: Activation,
    ) -> Self {
        Self {
            backend,
            act,
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            all_reduce: SumAllReduce::new(comm),
            world_size: comm.world_size(),
        }
    }
}

impl FusedExpertsWeights {
    fn load_standard(
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

    fn load_stacked(
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
}

#[cfg(feature = "cutile")]
impl CutileExpertsWeights {
    fn load_standard(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<CutileExpertsWeights> {
        let num_experts = cfg.num_experts;
        let mut gate_up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            let expert_vb = experts_vb.pp(i.to_string());
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
            let gate_up_expert = Tensor::cat(&[&gate_expert, &up_expert], 0)?;

            gate_up_experts.push(gate_up_expert);
            down_experts.push(down_expert);
        }

        let gate_up_w = Tensor::stack(&gate_up_experts, 0)?
            .transpose(1, 2)?
            .contiguous()?;
        let down_w = Tensor::stack(&down_experts, 0)?
            .transpose(1, 2)?
            .contiguous()?;
        let w_size_n = gate_up_w.dim(2)? / 2;

        Ok(CutileExpertsWeights {
            gate_up_w,
            down_w,
            w_size_n,
        })
    }

    fn load_stacked(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<CutileExpertsWeights> {
        let num_experts = cfg.num_experts;

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

        Ok(CutileExpertsWeights {
            gate_up_w,
            down_w,
            w_size_n,
        })
    }
}

impl FastExpertsWeights {
    fn load_combined_stacked(
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

    fn load_direct_standard(
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
}

impl SlowExpertsWeights {
    fn load_combined_stacked(
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
}

impl FastExpertsWeights {
    fn load_standard(
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

    fn load_stacked(
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
}

impl SlowExpertsWeights {
    fn load(
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
}

impl MoEExperts {
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
        let (batch_size, seq_len, hidden_dim) = xs.dims3()?;
        let shape = MoEForwardShape::new(batch_size, seq_len, hidden_dim);
        let xs_flat = xs.reshape(shape.flat())?;
        let forward = MoEForward {
            xs,
            xs_flat: &xs_flat,
            topk_weights: &topk_weights,
            topk_ids,
            original_dtype: xs.dtype(),
            shape,
        };

        let config = self.forward_config();
        let mut ys = self
            .backend
            .forward(&forward, config)
            .map_err(|err| err.context("moe experts forward"))?;

        // Apply all-reduce for tensor parallelism
        if self.world_size > 1 {
            ys = self.all_reduce.sum_all_reduce(&ys)?;
        }

        ys.reshape(forward.shape.output())
    }

    fn forward_config(&self) -> MoEForwardConfig {
        MoEForwardConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            act: self.act,
        }
    }
}

impl FusedExpertsWeights {
    fn forward_impl(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let is_prefill = forward.shape.phase.is_prefill();
        let (expert_ids, sorted_token_ids) = if forward.shape.phase.is_prefill() {
            #[cfg(feature = "cuda")]
            {
                use crate::ops::ArgSortOp;
                forward.topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            forward.topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            forward.topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        let gate_up = if self.stacked_format {
            moe::moe_gemm_transposed(
                forward.xs_flat,
                &self.gate_up_w,
                &None,
                &sorted_token_ids,
                &expert_ids,
                config.num_experts_per_tok,
                is_prefill,
            )?
        } else {
            moe::moe_gemm(
                forward.xs_flat,
                &self.gate_up_w,
                &None,
                &sorted_token_ids,
                &expert_ids,
                config.num_experts_per_tok,
                is_prefill,
            )?
        };

        let gate = gate_up.narrow(D::Minus1, 0, self.w_size_n)?.contiguous()?;
        let up = gate_up
            .narrow(D::Minus1, self.w_size_n, self.w_size_n)?
            .contiguous()?;

        let down_inputs = (up * gate.apply(&config.act)?)?.reshape(((), self.w_size_n))?;

        let ys = if self.stacked_format {
            moe::moe_gemm_transposed(
                &down_inputs,
                &self.down_w,
                &Some(forward.topk_weights.clone()),
                &sorted_token_ids,
                &expert_ids,
                config.num_experts_per_tok,
                is_prefill,
            )?
        } else {
            moe::moe_gemm(
                &down_inputs,
                &self.down_w,
                &Some(forward.topk_weights.clone()),
                &sorted_token_ids,
                &expert_ids,
                config.num_experts_per_tok,
                is_prefill,
            )?
        };

        ys.reshape((forward.shape.num_tokens, (), forward.shape.hidden_dim))?
            .sum(D::Minus2)
    }
}

#[cfg(feature = "cutile")]
impl CutileExpertsWeights {
    fn forward_impl(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        use candle_core::Storage;

        let dev = forward.xs_flat.device().as_cuda_device()?;
        let num_tokens = forward.shape.num_tokens;
        let topk = config.num_experts_per_tok;
        let num_experts = config.num_experts;
        let inter = self.w_size_n;
        let num_valid = num_tokens * topk;

        let cfg = mistralrs_quant::cutile::get_default_config(num_tokens, num_experts);

        let ti_flat = forward.topk_ids.flatten_all()?.contiguous()?;
        let (ti_storage, ti_layout) = ti_flat.storage_and_layout();
        let ti_slice = match &*ti_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle_core::bail!("topk_ids must be a cuda tensor"),
        };
        assert_eq!(ti_layout.start_offset(), 0, "expected contiguous topk_ids");

        let (sids, eids, ntpp, em) = mistralrs_quant::cutile::moe_align(
            ti_slice,
            num_tokens,
            num_experts,
            topk,
            cfg.bm,
            dev,
        )?;

        let ic1 = mistralrs_quant::cutile::cutile_grouped_gemm(
            forward.xs_flat,
            &self.gate_up_w,
            &sids,
            &eids,
            &ntpp,
            None,
            em,
            num_valid,
            topk,
            false,
            cfg,
            dev,
        )?;

        let ic2 = mistralrs_quant::cutile::gelu_tanh_and_mul(&ic1, inter, dev)?;

        let tw_flat = forward
            .topk_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (tw_storage, tw_layout) = tw_flat.storage_and_layout();
        let tw_slice = match &*tw_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("topk_weights must be a cuda tensor"),
        };
        assert_eq!(
            tw_layout.start_offset(),
            0,
            "expected contiguous topk_weights"
        );

        let ic3 = mistralrs_quant::cutile::cutile_grouped_gemm(
            &ic2,
            &self.down_w,
            &sids,
            &eids,
            &ntpp,
            Some(tw_slice),
            em,
            num_valid,
            1,
            true,
            cfg,
            dev,
        )?;

        mistralrs_quant::cutile::moe_sum_bf16(&ic3, num_tokens, topk, dev)
    }
}

impl MoEBackendWeights for FusedExpertsWeights {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        self.forward_impl(forward, config)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![]
    }

    fn num_isq_layers(&self) -> usize {
        0
    }
}

#[cfg(feature = "cutile")]
impl MoEBackendWeights for CutileExpertsWeights {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        self.forward_impl(forward, config)
            .map_err(|err| err.context("moe experts cutile"))
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![]
    }

    fn num_isq_layers(&self) -> usize {
        0
    }
}

impl FastExpertsWeights {
    fn forward_impl(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(result) = self.forward_cuda(forward, config)? {
            return Ok(result);
        }

        self.forward_gather(forward, config)
    }

    #[cfg(feature = "cuda")]
    fn select_cuda_fast_path(forward: &MoEForward) -> Option<MoECudaFastPath> {
        if !forward.xs.device().is_cuda() {
            return None;
        }

        if forward.shape.phase.is_prefill() {
            (forward.shape.num_tokens >= 32).then_some(MoECudaFastPath::GroupedPrefill)
        } else {
            Some(MoECudaFastPath::Decode)
        }
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Option<Tensor>> {
        match Self::select_cuda_fast_path(forward) {
            Some(MoECudaFastPath::Decode) => self
                .forward_decode(forward, config)
                .map_err(|err| err.context("moe experts fast decode")),
            Some(MoECudaFastPath::GroupedPrefill) => self
                .forward_grouped(forward, config)
                .map_err(|err| err.context("moe experts fast grouped")),
            None => Ok(None),
        }
    }

    fn forward_gather(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let ys = if forward.xs.device().is_cuda() {
            let xs =
                forward
                    .xs_flat
                    .reshape((forward.shape.num_tokens, 1, forward.shape.hidden_dim))?;
            let gate = self.fused_gate_proj.gather_forward(&xs, forward.topk_ids)?;
            let up = self.fused_up_proj.gather_forward(&xs, forward.topk_ids)?;
            self.fused_down_proj
                .gather_forward(&(up * gate.apply(&config.act)?)?, forward.topk_ids)?
        } else {
            let xs = forward.xs.reshape((
                forward.shape.batch_size,
                forward.shape.seq_len,
                1,
                1,
                forward.shape.hidden_dim,
            ))?;
            let indices = forward.topk_ids.reshape((
                forward.shape.batch_size,
                forward.shape.seq_len,
                config.num_experts_per_tok,
            ))?;
            let gate = self.fused_gate_proj.gather_forward(&xs, &indices)?;
            let up = self.fused_up_proj.gather_forward(&xs, &indices)?;
            let xs = self
                .fused_down_proj
                .gather_forward(&(up * gate.apply(&config.act)?)?, &indices)?;
            xs.squeeze(D::Minus2)?.reshape((
                forward.shape.num_tokens,
                config.num_experts_per_tok,
                forward.shape.hidden_dim,
            ))?
        };

        ys.to_dtype(DType::F32)?
            .broadcast_mul(&forward.topk_weights.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(forward.original_dtype)
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
    fn forward_decode(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Option<Tensor>> {
        use candle_core::cuda::cudarc::driver::DevicePtr;

        let dev = forward.xs_flat.device().as_cuda_device()?;

        // Get QTensors - bail to fallback if not quantized
        let gate_qt = match self.fused_gate_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let up_qt = match self.fused_up_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let down_qt = match self.fused_down_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };

        // Get topk_ids as contiguous u32 CudaSlice
        let topk_ids_flat = forward.topk_ids.flatten_all()?.contiguous()?;
        let (ti_storage, ti_layout) = topk_ids_flat.storage_and_layout();
        let ti_cuda = match &*ti_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let ti_u32_slice = ti_cuda.as_cuda_slice::<u32>()?;
        assert!(ti_layout.start_offset() == 0, "expected contiguous tensor");

        // Get topk_weights as contiguous f32 CudaSlice
        let tw_f32 = forward
            .topk_weights
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
        let act_type = match config.act {
            Activation::GeluPytorchTanh => mistralrs_quant::ACT_GELU_PYTORCH_TANH,
            Activation::Silu | Activation::Swish => mistralrs_quant::ACT_SILU,
            _ => return Ok(None), // Fall back for unsupported activations
        };

        // SAFETY: tw_ptr is a valid device pointer obtained from the topk_weights tensor above.
        let result = unsafe {
            mistralrs_quant::indexed_moe_fused_decode(
                gate_qt,
                up_qt,
                down_qt,
                forward.xs_flat,
                ti_u32_slice,
                tw_ptr,
                forward.shape.num_tokens,
                config.num_experts_per_tok,
                act_type,
                dev,
            )?
        };

        Ok(Some(result.to_dtype(forward.original_dtype)?))
    }

    /// Grouped MoE forward for CUDA prefill.
    ///
    /// Returns Ok(Some(result)) if grouped path succeeded, Ok(None) to fall back.
    #[cfg(feature = "cuda")]
    fn forward_grouped(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Option<Tensor>> {
        let topk = config.num_experts_per_tok;
        let num_experts = config.num_experts;
        let total_assignments = forward.shape.num_tokens * topk;

        let dev = forward.xs_flat.device().as_cuda_device()?;

        // Get topk_ids as contiguous u32 CudaSlice
        let topk_ids_flat = forward.topk_ids.flatten_all()?.contiguous()?;
        let (ti_storage, ti_layout) = topk_ids_flat.storage_and_layout();
        let ti_cuda = match &*ti_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let ti_u32_slice = ti_cuda.as_cuda_slice::<u32>()?;
        assert!(ti_layout.start_offset() == 0, "expected contiguous tensor");

        // Build dispatch tables on GPU (no CPU-GPU sync)
        // moe_dispatch_build takes u32 and casts to i32 internally for the CUDA kernel
        let (expert_bounds, sorted_token_ids, sorted_source_ids) =
            mistralrs_quant::moe_dispatch_build(
                ti_u32_slice,
                total_assignments,
                num_experts,
                topk,
                dev,
            )?;

        // Use the pre-quantized Q8_0 grouped kernel path
        let gate_qt = match self.fused_gate_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let up_qt = match self.fused_up_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let down_qt = match self.fused_down_proj.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };

        let use_mmq_gate_up =
            gate_qt.dtype() == up_qt.dtype() && mistralrs_quant::supports_mmq(gate_qt.dtype());

        let (gate, up, down_input_dim1) = if use_mmq_gate_up {
            let (gate, up) = mistralrs_quant::grouped_moe_mmq_pair(
                gate_qt,
                up_qt,
                forward.xs_flat,
                &sorted_source_ids,
                &sorted_token_ids,
                &expert_bounds,
                total_assignments,
                topk,
                num_experts,
                dev,
            )?;
            (gate, up, 2)
        } else {
            // Quantize input to Q8_1 ONCE, shared between gate and up.
            // quantize_input_q8_1 accepts BF16/F16/F32 directly (no conversion needed).
            let (input_q8, k, k_padded) =
                mistralrs_quant::quantize_input_q8_1(forward.xs_flat, dev)?;

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
            (gate, up, 0)
        };

        // Get topk_weights pointer
        use candle_core::cuda::cudarc::driver::DevicePtr;
        let tw_f32 = forward
            .topk_weights
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

        let glu_activation = match config.act {
            Activation::Silu | Activation::Swish => Some(mistralrs_quant::GluActivationType::Silu),
            Activation::NewGelu | Activation::GeluPytorchTanh => {
                Some(mistralrs_quant::GluActivationType::Gelu)
            }
            Activation::Gelu => Some(mistralrs_quant::GluActivationType::GeluErf),
            Activation::Relu => Some(mistralrs_quant::GluActivationType::Relu),
            _ => None,
        };

        let down = if let (true, Some(glu_activation)) = (
            mistralrs_quant::supports_mmq(down_qt.dtype()),
            glu_activation,
        ) {
            let down_assignments = mistralrs_quant::grouped_moe_mmq_from_glu_pair(
                down_qt,
                &gate,
                &up,
                &sorted_token_ids,
                &sorted_token_ids,
                &expert_bounds,
                total_assignments,
                forward.shape.num_tokens,
                num_experts,
                glu_activation as i32,
                dev,
            )?;
            if forward.original_dtype == DType::BF16 {
                unsafe {
                    mistralrs_quant::moe_weighted_reduce_flat_bf16(
                        &down_assignments,
                        tw_ptr,
                        forward.shape.num_tokens,
                        topk,
                        dev,
                    )?
                }
            } else {
                unsafe {
                    mistralrs_quant::moe_weighted_reduce_flat(
                        &down_assignments,
                        tw_ptr,
                        forward.shape.num_tokens,
                        topk,
                        dev,
                    )?
                }
            }
        } else {
            // Apply activation
            let activated = crate::ops::mul_and_act(&gate, &up, config.act)?;

            // Quantize down projection input
            let (down_input_q8, down_k, down_k_padded) =
                mistralrs_quant::quantize_input_q8_1(&activated, dev)?;

            // Down projection with topk_weights + atomicAdd
            mistralrs_quant::grouped_moe_gemm_prequantized(
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
                down_input_dim1,
                dev,
            )?
        };

        if down.dtype() == forward.original_dtype {
            Ok(Some(down))
        } else {
            Ok(Some(down.to_dtype(forward.original_dtype)?))
        }
    }
}

impl MoEBackendWeights for FastExpertsWeights {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        self.forward_impl(forward, config)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![
            &mut self.fused_gate_proj,
            &mut self.fused_up_proj,
            &mut self.fused_down_proj,
        ]
    }

    fn num_isq_layers(&self) -> usize {
        3
    }
}

impl SlowExpertsWeights {
    fn forward_impl(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let routing_weights = forward
            .topk_weights
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let experts_per_tok = forward.topk_ids.to_vec2::<u32>()?;
        let num_experts = self.experts.gate_proj.len();

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

        let mut ys = forward.xs_flat.zeros_like()?;
        for expert_idx in 0..num_experts {
            let top_x_expert = &top_x[expert_idx];
            if top_x_expert.is_empty() {
                continue;
            }
            let top_x_tensor = Tensor::new(top_x_expert.as_slice(), forward.xs.device())?;
            let selected_experts_tensor =
                Tensor::new(selected_experts[expert_idx].as_slice(), forward.xs.device())?
                    .reshape(((), 1))?
                    .to_dtype(forward.xs.dtype())?;
            let current_state = forward
                .xs_flat
                .index_select(&top_x_tensor, 0)?
                .reshape(((), forward.shape.hidden_dim))?;

            // Forward through expert MLP
            let expert_input = current_state.clone();
            let gate_out = self.experts.gate_proj[expert_idx]
                .forward(&expert_input)?
                .apply(&config.act)?;
            let up_out = self.experts.up_proj[expert_idx].forward(&expert_input)?;
            let current_hidden_states =
                self.experts.down_proj[expert_idx].forward(&(gate_out * up_out)?)?;

            let current_hidden_states =
                current_hidden_states.broadcast_mul(&selected_experts_tensor)?;
            ys = ys.index_add(&top_x_tensor, &current_hidden_states, 0)?;
        }

        ys.reshape(forward.shape.flat())
    }
}

impl MoEBackendWeights for SlowExpertsWeights {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        self.forward_impl(forward, config)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = Vec::new();
        for (gate, (up, down)) in self.experts.gate_proj.iter_mut().zip(
            self.experts
                .up_proj
                .iter_mut()
                .zip(self.experts.down_proj.iter_mut()),
        ) {
            layers.push(gate);
            layers.push(up);
            layers.push(down);
        }
        layers
    }

    fn num_isq_layers(&self) -> usize {
        self.experts.gate_proj.len() * 3
    }
}

impl MoEExperts {
    /// Get mutable references to quantizable layers for ISQ
    /// Returns mutable references to all ISQ-quantizable layers.
    /// The count must match `num_isq_layers`.
    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        self.backend.get_isq_layers()
    }

    /// Returns the number of ISQ-quantizable layers.
    /// Must match the length of `get_isq_layers`.
    pub fn num_isq_layers(&self) -> usize {
        self.backend.num_isq_layers()
    }
}
