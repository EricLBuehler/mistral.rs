use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    apply_immediate_isq, should_apply_immediate_isq, DummyLayer, FusedExperts, PackedExperts,
    QuantMethod, QuantMethodConfig, QuantizedConfig, ShardedVarBuilder, UnquantLinear,
};
use std::sync::Arc;

use crate::cuda::moe;
#[cfg(feature = "cuda")]
use crate::layers::Activation;

use super::checkpoint::ExpertCheckpoint;
use super::config::MoEExpertsConfig;
#[cfg(feature = "cuda")]
use super::forward::MoECudaFastPath;
use super::forward::{MoEForward, MoEForwardConfig};

/// Canonical stacked expert weights, ENK [E, N, K] = [E, out, in]. The raw backends (Fused,
/// Cutile) hold exactly this; nothing else stores a layout.
pub(super) struct StackedExpertWeights {
    pub(super) gate_up: Tensor,
    pub(super) down: Tensor,
    /// Intermediate size (after sharding) = gate_up.dim(1) / 2.
    pub(super) w_size_n: usize,
}

pub(super) struct FusedExpertsWeights {
    pub(super) w: StackedExpertWeights,
}

#[cfg(feature = "cutile")]
pub(super) struct CutileExpertsWeights {
    pub(super) w: StackedExpertWeights,
}

/// Gather-based experts (Metal / ISQ / pre-quantized).
pub(super) struct FastExpertsWeights {
    pub(super) fused_gate_proj: Arc<dyn QuantMethod>,
    pub(super) fused_up_proj: Arc<dyn QuantMethod>,
    pub(super) fused_down_proj: Arc<dyn QuantMethod>,
}

/// Loop-based experts (quantized fallback).
pub(super) struct SlowExpertsWeights {
    pub(super) experts: PackedExperts,
}

impl StackedExpertWeights {
    pub(super) fn from_checkpoint(ckpt: &ExpertCheckpoint) -> Result<StackedExpertWeights> {
        let (gate_up, down) = ckpt.stacked_enk()?;
        let w_size_n = gate_up.dim(1)? / 2;
        Ok(StackedExpertWeights {
            gate_up,
            down,
            w_size_n,
        })
    }
}

#[cfg(feature = "cutile")]
impl CutileExpertsWeights {
    pub(super) fn from_checkpoint(ckpt: &ExpertCheckpoint) -> Result<CutileExpertsWeights> {
        let w = StackedExpertWeights::from_checkpoint(ckpt)?;
        mistralrs_quant::cutile::register_moe_shape(
            w.gate_up.clone(),
            w.down.clone(),
            ckpt.cfg.num_experts,
            ckpt.cfg.num_experts_per_tok,
        );
        Ok(CutileExpertsWeights { w })
    }
}

impl FastExpertsWeights {
    pub(super) fn load_combined_stacked(
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

    pub(super) fn load_direct_standard(
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
    pub(super) fn load_combined_stacked(
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
    /// Standard tree (`experts.*`). `FusedExperts::new` auto-detects per-expert vs combined and
    /// handles pre-quantized weights.
    pub(super) fn load_fused(
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
}

impl SlowExpertsWeights {
    pub(super) fn load(
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

impl FusedExpertsWeights {
    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
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

        let gate_up = moe::moe_gemm(
            forward.xs_flat,
            &self.w.gate_up,
            &None,
            &sorted_token_ids,
            &expert_ids,
            config.num_experts_per_tok,
            is_prefill,
        )?;

        let gate = gate_up
            .narrow(D::Minus1, 0, self.w.w_size_n)?
            .contiguous()?;
        let up = gate_up
            .narrow(D::Minus1, self.w.w_size_n, self.w.w_size_n)?
            .contiguous()?;

        let down_inputs = (up * gate.apply(&config.act)?)?.reshape(((), self.w.w_size_n))?;

        let ys = moe::moe_gemm(
            &down_inputs,
            &self.w.down,
            &Some(forward.topk_weights.clone()),
            &sorted_token_ids,
            &expert_ids,
            config.num_experts_per_tok,
            is_prefill,
        )?;

        ys.reshape((forward.shape.num_tokens, (), forward.shape.hidden_dim))?
            .sum(D::Minus2)
    }
}

#[cfg(feature = "cutile")]
impl CutileExpertsWeights {
    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        use candle_core::Storage;

        let dev = forward.xs_flat.device().as_cuda_device()?;
        let num_tokens = forward.shape.num_tokens;
        let topk = config.num_experts_per_tok;
        let num_experts = config.num_experts;
        let inter = self.w.w_size_n;
        let num_valid = num_tokens * topk;

        let cfg = mistralrs_quant::cutile::get_default_config(num_tokens, num_experts);

        let ti_flat = forward.topk_ids.flatten_all()?.contiguous()?;
        let (ti_storage, ti_layout) = ti_flat.storage_and_layout();
        let ti_slice = match &*ti_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle_core::bail!("topk_ids must be a cuda tensor"),
        };
        assert_eq!(ti_layout.start_offset(), 0, "expected contiguous topk_ids");

        let (sids, eids, ntpp, em) = mistralrs_quant::moe::cuda::moe_align(
            ti_slice,
            num_tokens,
            num_experts,
            topk,
            cfg.bm,
            dev,
        )?;

        let ic1 = mistralrs_quant::cutile::cutile_grouped_gemm(
            forward.xs_flat,
            &self.w.gate_up,
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

        let ic2 = mistralrs_quant::moe::cuda::gelu_tanh_and_mul(&ic1, inter, dev)?;

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
            &self.w.down,
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

        mistralrs_quant::moe::cuda::moe_sum_bf16(&ic3, num_tokens, topk, dev)
    }
}

impl FastExpertsWeights {
    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(result) = self.forward_cuda(forward, config)? {
            return Ok(result);
        }

        self.forward_gather(forward, config)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn select_cuda_fast_path(forward: &MoEForward) -> Option<MoECudaFastPath> {
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
    pub(super) fn forward_cuda(
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

    pub(super) fn forward_gather(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
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

    /// Fused MoE decode path for CUDA. Returns Ok(Some) on success, Ok(None) to fall back.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_decode(
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

    /// Grouped MoE forward for CUDA prefill. Returns Ok(Some) on success, Ok(None) to fall back.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_grouped(
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

impl SlowExpertsWeights {
    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
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
