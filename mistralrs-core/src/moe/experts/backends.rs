use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    apply_immediate_isq_with_key, should_apply_immediate_isq, DummyLayer, PreQuantizedExperts,
    QuantMethod, QuantMethodConfig, QuantizedConfig, Shard, ShardedVarBuilder, UnquantLinear,
    UqffExpertKeys,
};
use std::sync::Arc;

use crate::cuda::moe;
#[cfg(feature = "cuda")]
use crate::layers::Activation;

use super::checkpoint::ExpertCheckpoint;
use super::config::{ExpertProj, MoEExpertsConfig};
#[cfg(feature = "cuda")]
use super::forward::MoECudaFastPath;
use super::forward::{MoEForward, MoEForwardConfig};

#[cfg(feature = "cuda")]
const GROUPED_PREFILL_MIN_TOKENS: usize = 32;

/// Canonical stacked expert weights, ENK [E, N, K] = [E, out, in]. The raw backends (Fused,
/// Cutile) hold exactly this; nothing else stores a layout.
#[derive(Clone)]
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

#[cfg(feature = "cuda")]
pub(super) struct CutlassExpertsWeights {
    pub(super) w: StackedExpertWeights,
}

/// Below this batch size the grouped-GEMM setup cost exceeds the fused decode kernels.
#[cfg(feature = "cuda")]
pub(super) const CUTLASS_MOE_MIN_TOKENS: usize = 64;

#[cfg(feature = "cuda")]
fn gated_act(act: Activation) -> mistralrs_quant::moe::cuda::GatedAct {
    match act {
        Activation::Silu => mistralrs_quant::moe::cuda::GatedAct::Silu,
        _ => mistralrs_quant::moe::cuda::GatedAct::GeluTanh,
    }
}

/// Gather-based experts (Metal / CPU / ISQ / pre-quantized).
pub(super) struct FastExpertsWeights {
    pub(super) fused_gate_proj: Arc<dyn QuantMethod>,
    pub(super) fused_up_proj: Arc<dyn QuantMethod>,
    pub(super) fused_down_proj: Arc<dyn QuantMethod>,
    /// Sharded across ranks (partial sums needing an all-reduce) vs replicated.
    pub(super) sharded: bool,
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

#[cfg(feature = "cuda")]
impl CutlassExpertsWeights {
    pub(super) fn from_checkpoint(ckpt: &ExpertCheckpoint) -> Result<CutlassExpertsWeights> {
        Ok(CutlassExpertsWeights {
            w: StackedExpertWeights::from_checkpoint(ckpt)?,
        })
    }

    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        // Grouped GEMM launch overhead dominates tiny batches; decode goes through the fused
        // kernels instead (same weights, decode-optimized path).
        if forward.shape.num_tokens < CUTLASS_MOE_MIN_TOKENS {
            let fused = FusedExpertsWeights { w: self.w.clone() };
            return fused.forward_impl(forward, config);
        }
        let dev = forward.xs_flat.device().as_cuda_device()?;
        let act = gated_act(config.act);
        mistralrs_quant::moe::cutlass_fused_moe(
            forward.xs_flat,
            &self.w.gate_up,
            &self.w.down,
            forward.topk_ids,
            forward.topk_weights,
            config.num_experts,
            act,
            dev,
        )
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

/// Whether the checkpoint actually ships quantized expert tensors. FP8 configs sometimes ship
/// unquantized weights; the scale tensors are the tell.
pub(super) fn experts_are_prequantized(
    config: &Option<QuantizedConfig>,
    experts_vb: &ShardedVarBuilder,
) -> bool {
    match config {
        None => false,
        Some(QuantizedConfig::Fp8 { .. }) => {
            experts_vb.contains_tensor("gate_up_proj.weight_scale_inv")
                || crate::moe::ExpertProjNames::KNOWN.iter().any(|names| {
                    experts_vb
                        .pp("0")
                        .contains_tensor(&format!("{}.weight_scale_inv", names.gate))
                })
        }
        Some(_) => true,
    }
}

impl FastExpertsWeights {
    /// Load the three stacked expert layers from a UQFF artifact under their canonical names.
    /// Shards across ranks when the quantization geometry allows; replicates otherwise.
    pub(super) fn from_uqff(
        cfg: &MoEExpertsConfig,
        experts_vb: &ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Option<FastExpertsWeights>> {
        let Some(reader) = experts_vb.uqff_reader() else {
            return Ok(None);
        };
        let keys = UqffExpertKeys::new(&experts_vb.prefix());
        if !reader.contains(&format!("{}.weight", keys.gate)) {
            return Ok(None);
        }

        let rank = comm.rank();
        let world_size = comm.world_size();
        let inter = cfg.moe_intermediate_size;
        // down shards its packed (input) dim, so the per-rank slice must be block-aligned.
        let sharded = world_size > 1
            && inter.is_multiple_of(world_size)
            && reader
                .shard_alignment(&keys.down)
                .is_ok_and(|align| (inter / world_size).is_multiple_of(align));
        if world_size > 1 && !sharded {
            mistralrs_quant::log::once_log_warn(
                "UQFF expert quantization geometry does not allow sharding; replicating experts per rank.",
            );
        }

        let (gate_shard, down_shard) = if sharded {
            (
                Shard::Simple {
                    dim: 1,
                    rank,
                    world_size,
                },
                Shard::Simple {
                    dim: 2,
                    rank,
                    world_size,
                },
            )
        } else {
            (Shard::default(), Shard::default())
        };

        let device = experts_vb.device();
        let load = |key: &str, shard: Shard| -> Result<Arc<dyn QuantMethod>> {
            reader.load_linear(key, device, shard)?.ok_or_else(|| {
                candle_core::Error::Msg(format!("Missing UQFF expert tensor `{key}`."))
            })
        };
        Ok(Some(FastExpertsWeights {
            fused_gate_proj: load(&keys.gate, gate_shard)?,
            fused_up_proj: load(&keys.up, gate_shard)?,
            fused_down_proj: load(&keys.down, down_shard)?,
            sharded,
        }))
    }

    /// Load unquantized experts in any on-disk layout via [`ExpertCheckpoint`], then ISQ-wrap.
    pub(super) fn load_unquantized(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<FastExpertsWeights> {
        let shape_of = |rel: &str| experts_vb.tensor_shape(rel).map(|s| s.to_vec());
        let Some(layout) =
            super::checkpoint::ExpertSourceLayout::detect(&shape_of, crate::moe::ExpertProj::Gate)
        else {
            let dummy = || -> Result<Arc<dyn QuantMethod>> {
                Ok(Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?))
            };
            return Ok(FastExpertsWeights {
                fused_gate_proj: dummy()?,
                fused_up_proj: dummy()?,
                fused_down_proj: dummy()?,
                sharded: false,
            });
        };

        // ISQ predicates match source names, so the predicate VBs follow the on-disk layout.
        let (vb_gate, vb_up, vb_down) = match &layout {
            super::checkpoint::ExpertSourceLayout::Fused { .. } => {
                let gate_up = experts_vb.pp("gate_up_proj");
                (gate_up.clone(), gate_up, experts_vb.pp("down_proj"))
            }
            super::checkpoint::ExpertSourceLayout::PerExpert { names, .. } => {
                let expert0 = experts_vb.pp("0");
                (
                    expert0.pp(names.gate),
                    expert0.pp(names.up),
                    expert0.pp(names.down),
                )
            }
        };

        // When immediate ISQ is active, read on CPU to avoid creating large GPU buffers that
        // would be immediately copied back for quantization (critical on unified memory).
        let isq = should_apply_immediate_isq(&vb_gate) || should_apply_immediate_isq(&vb_down);
        let read_vb = if isq && !experts_vb.device().is_cpu() {
            experts_vb.clone().set_device(Device::Cpu)
        } else {
            experts_vb.clone()
        };

        let wrap = |w: Tensor| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(w, None)),
            )?))
        };
        let keys = UqffExpertKeys::new(&experts_vb.prefix());
        let shard = (comm.world_size() == 1).then(mistralrs_quant::Shard::default);
        let checkpoint = ExpertCheckpoint::new(cfg, read_vb, comm)?;
        let load = |proj: ExpertProj,
                    vb: ShardedVarBuilder,
                    key: String|
         -> Result<Arc<dyn QuantMethod>> {
            apply_immediate_isq_with_key(
                wrap(checkpoint.stacked_proj(proj)?)?,
                vb,
                Some(key),
                shard,
            )
        };

        let fused_gate_proj = load(ExpertProj::Gate, vb_gate, keys.gate)?;
        let fused_up_proj = load(ExpertProj::Up, vb_up, keys.up)?;
        let fused_down_proj = load(ExpertProj::Down, vb_down, keys.down)?;
        Ok(FastExpertsWeights {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
            sharded: comm.world_size() > 1,
        })
    }
}

impl FastExpertsWeights {
    /// Pre-quantized standard tree (`experts.*` / `switch_mlp.*`).
    pub(super) fn load_prequantized(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        quantization_config: &Option<QuantizedConfig>,
    ) -> Result<FastExpertsWeights> {
        let PreQuantizedExperts {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        } = PreQuantizedExperts::new(
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
            sharded: false,
        })
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

        let ic2 = mistralrs_quant::moe::cuda::act_and_mul(&ic1, inter, gated_act(config.act), dev)?;

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
        // while collecting, force the gather path; fused kernels never materialize the routed inputs
        #[cfg(feature = "cuda")]
        if self.fused_gate_proj.stats_snapshot().is_none() {
            if let Some(result) = self.forward_cuda(forward, config)? {
                return Ok(result);
            }
        }

        self.forward_gather(forward, config)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn select_cuda_fast_path(forward: &MoEForward) -> Option<MoECudaFastPath> {
        if !forward.xs.device().is_cuda() {
            return None;
        }

        if !forward.shape.phase.is_prefill()
            || forward.shape.num_tokens < GROUPED_PREFILL_MIN_TOKENS
        {
            Some(MoECudaFastPath::Decode)
        } else {
            Some(MoECudaFastPath::GroupedPrefill)
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
        // Routed stats are fed here: only the block knows the token-to-expert pairing.
        let ids = forward.topk_ids;
        self.fused_gate_proj
            .process_routed_stats(forward.xs_flat, ids)?;
        self.fused_up_proj
            .process_routed_stats(forward.xs_flat, ids)?;
        let ys = if forward.xs.device().is_cuda() {
            let xs =
                forward
                    .xs_flat
                    .reshape((forward.shape.num_tokens, 1, forward.shape.hidden_dim))?;
            let gate = self.fused_gate_proj.gather_forward(&xs, forward.topk_ids)?;
            let up = self.fused_up_proj.gather_forward(&xs, forward.topk_ids)?;
            let down_in = (up * gate.apply(&config.act)?)?;
            self.fused_down_proj.process_routed_stats(&down_in, ids)?;
            self.fused_down_proj
                .gather_forward(&down_in, forward.topk_ids)?
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
            let down_in = (up * gate.apply(&config.act)?)?;
            let inter = down_in.dim(D::Minus1)?;
            self.fused_down_proj.process_routed_stats(
                &down_in.reshape((forward.shape.num_tokens, config.num_experts_per_tok, inter))?,
                ids,
            )?;
            let xs = self.fused_down_proj.gather_forward(&down_in, &indices)?;
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
            _ => return Ok(None),
        };

        // SAFETY: tw_ptr is a valid device pointer obtained from the topk_weights tensor above.
        let result = unsafe {
            mistralrs_quant::indexed_moe_fused_decode(
                &gate_qt,
                &up_qt,
                &down_qt,
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
                &gate_qt,
                &up_qt,
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
                &gate_qt,
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
                &up_qt,
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
                &down_qt,
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
                &down_qt,
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
