#[cfg(any(feature = "cutile", test))]
use candle_core::DType;
use candle_core::{Device, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    apply_immediate_isq_with_key, should_apply_immediate_isq, DummyLayer, LoraExpertInputMode,
    LoraExpertProjection, PreQuantizedExperts, QuantMethod, QuantMethodConfig, QuantizedConfig,
    QuantizedExpertKeys, Shard, ShardedVarBuilder, UnquantLinear,
};
use std::sync::Arc;

use crate::cuda::moe;
#[cfg(feature = "cutile")]
use crate::layers::Activation;

use super::checkpoint::ExpertCheckpoint;
#[cfg(feature = "cuda")]
use super::config::gated_act;
use super::config::{ExpertProj, MoEExpertsConfig};
use super::forward::{MoEForward, MoEForwardConfig};

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

/// Gather-based experts (Metal / CPU / ISQ / pre-quantized).
pub(super) struct FastExpertsWeights {
    pub(super) fused_gate_proj: Arc<dyn QuantMethod>,
    pub(super) fused_up_proj: Arc<dyn QuantMethod>,
    pub(super) fused_down_proj: Arc<dyn QuantMethod>,
    /// Sharded across ranks (partial sums needing an all-reduce) vs replicated.
    pub(super) sharded: bool,
}

// Per-linear expert formats load without the router config, so their warmup reads the model's routing here.
#[cfg(feature = "cutile")]
fn register_cutile_moe_routing(cfg: &MoEExpertsConfig, device: &Device) -> Result<()> {
    mistralrs_quant::cutile::register_moe_routing(device, cfg.num_experts, cfg.num_experts_per_tok)
}

fn should_stage_prequantized_on_cpu(experts_vb: &ShardedVarBuilder) -> bool {
    ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .all(|name| should_apply_immediate_isq(&experts_vb.pp(name)))
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
        if forward.lora.is_some() {
            return FusedExpertsWeights { w: self.w.clone() }.forward_impl(forward, config);
        }
        // Grouped GEMM launch overhead dominates tiny batches; decode goes through the fused
        // kernels instead (same weights, decode-optimized path).
        if forward.shape.num_tokens < CUTLASS_MOE_MIN_TOKENS {
            let fused = FusedExpertsWeights { w: self.w.clone() };
            return fused.forward_impl(forward, config);
        }
        let dev = forward.xs_flat.device().as_cuda_device()?;
        let act = gated_act(config.act)?;
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

/// Blockwise FP8 experts on the cuTile fused MoE.
#[cfg(feature = "cutile")]
pub(super) struct CutileFp8ExpertsWeights {
    w: mistralrs_quant::cutile::CutileFp8MoeWeights,
}

#[cfg(feature = "cutile")]
impl CutileFp8ExpertsWeights {
    /// `None` when the checkpoint, device, or configuration is not one this backend serves.
    pub(super) fn try_from_checkpoint(
        cfg: &MoEExpertsConfig,
        experts_vb: &ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Option<Self>> {
        use mistralrs_quant::cutile::{fp8_moe_supported, FP8_MOE_GROUP};
        let Some(QuantizedConfig::Fp8 {
            weight_block_size: Some(block),
            activation_scheme,
            ..
        }) = quantization_config
        else {
            return Ok(None);
        };
        if block.as_slice() != [FP8_MOE_GROUP, FP8_MOE_GROUP]
            || *activation_scheme != Some(mistralrs_quant::Fp8ActivationScheme::Dynamic)
            || comm.world_size() > 1
            || super::config::gated_act(act).is_err()
            || !matches!(
                super::config::MoEExpertsBackend::from_env(),
                None | Some(super::config::MoEExpertsBackend::Cutile)
            )
        {
            return Ok(None);
        }
        let Device::Cuda(dev) = experts_vb.device() else {
            return Ok(None);
        };
        if !mistralrs_quant::cutile::device_supported(dev)
            || !fp8_moe_supported(dev, cfg.hidden_size, cfg.moe_intermediate_size)
        {
            return Ok(None);
        }
        let ckpt = ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?;
        let Some((gate_up, gate_up_scales, down, down_scales)) =
            ckpt.stacked_enk_fp8(FP8_MOE_GROUP)?
        else {
            return Ok(None);
        };
        let w = mistralrs_quant::cutile::CutileFp8MoeWeights::new(
            gate_up,
            gate_up_scales,
            down,
            down_scales,
            cfg.num_experts,
            cfg.num_experts_per_tok,
        )?;
        mistralrs_quant::cutile::register_moe_fp8_shape(&w);
        mistralrs_quant::log::once_log_info(
            "MoE experts backend: cuTile blockwise FP8 grouped GEMM",
        );
        Ok(Some(Self { w }))
    }

    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        if forward.lora.is_some() {
            candle_core::bail!("dynamic LoRA is not supported on cuTile FP8 experts");
        }
        let activation = match gated_act(config.act)? {
            mistralrs_quant::moe::cuda::GatedAct::Silu => mistralrs_quant::GluActivationType::Silu,
            mistralrs_quant::moe::cuda::GatedAct::GeluTanh => {
                mistralrs_quant::GluActivationType::Gelu
            }
        };
        let dev = forward.xs_flat.device().as_cuda_device()?;
        let xs = forward.xs_flat.to_dtype(DType::BF16)?;
        mistralrs_quant::cutile::cutile_fused_moe_fp8(
            &xs,
            &self.w,
            forward.topk_ids,
            forward.topk_weights,
            activation,
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
    pub(super) fn from_weight_source(
        cfg: &MoEExpertsConfig,
        experts_vb: &ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Option<FastExpertsWeights>> {
        let Some(source) = experts_vb.weight_source() else {
            return Ok(None);
        };
        let keys = QuantizedExpertKeys::new(&experts_vb.prefix());
        if !source.contains(&format!("{}.weight", keys.gate)) {
            return Ok(None);
        }

        let rank = comm.rank();
        let world_size = comm.world_size();
        let inter = cfg.moe_intermediate_size;
        let down_has_bias = source.contains(&format!("{}.bias", keys.down));
        // down shards its packed (input) dim, so the per-rank slice must be block-aligned.
        let sharded = world_size > 1
            && !down_has_bias
            && inter.is_multiple_of(world_size)
            && source
                .shard_alignment(&keys.down)
                .is_ok_and(|align| (inter / world_size).is_multiple_of(align));
        if world_size > 1 && !sharded {
            if down_has_bias {
                mistralrs_quant::log::once_log_warn(
                    "Biased expert down projections require replication across ranks.",
                );
            } else {
                mistralrs_quant::log::once_log_warn(
                    "Expert quantization geometry does not allow sharding; replicating experts per rank.",
                );
            }
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

        let load = |key: &str,
                    predicate_vb: ShardedVarBuilder,
                    shard: Shard|
         -> Result<Arc<dyn QuantMethod>> {
            let load_device = mistralrs_quant::weight_source_load_device(&predicate_vb);
            let layer = source
                .load_linear(key, &load_device, shard)?
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("Missing expert weight `{key}`."))
                })?;
            apply_immediate_isq_with_key(layer, predicate_vb, Some(key.to_string()), Some(shard))
        };
        Ok(Some(FastExpertsWeights {
            fused_gate_proj: load(&keys.gate, experts_vb.pp("gate_proj"), gate_shard)?,
            fused_up_proj: load(&keys.up, experts_vb.pp("up_proj"), gate_shard)?,
            fused_down_proj: load(&keys.down, experts_vb.pp("down_proj"), down_shard)?,
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
        let keys = QuantizedExpertKeys::new(&experts_vb.prefix());
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

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::layers::Activation;
    use crate::moe::experts::forward::{MoEForwardPhase, MoEForwardShape};
    use candle_core::quantized::{GgmlDType, QTensor};
    use mistralrs_quant::{
        with_lora_execution, GgufMatMul, LoraExecution, LoraExpertExecution,
        LoraExpertProjectionNames, LoraExpertProjectionWeights, LoraExpertSiteSpec,
        LoraExpertWeights, LoraLayerRegistry, LoraSiteKey,
    };

    fn values(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let index = u16::try_from(index % 4096).expect("index is bounded");
                (f32::from(index) * 0.017 + phase).sin() * 0.08
            })
            .collect()
    }

    fn tensor(shape: impl Into<candle_core::Shape>, phase: f32) -> Result<Tensor> {
        let shape = shape.into();
        Tensor::from_vec(values(shape.elem_count(), phase), shape, &Device::Cpu)
    }

    fn quant_method(weight: Tensor, device: &Device) -> Result<Arc<dyn QuantMethod>> {
        let weight = QTensor::quantize_onto(&weight, GgmlDType::Q4_0, device)?;
        Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(weight),
            b: None,
        })?))
    }

    fn lora_projection(
        experts: usize,
        rank: usize,
        input: usize,
        output: usize,
        phase: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<LoraExpertProjectionWeights> {
        LoraExpertProjectionWeights::new(
            tensor((experts, rank, input), phase)?
                .to_dtype(dtype)?
                .to_device(device)?,
            tensor((experts, output, rank), phase + 0.4)?
                .to_dtype(dtype)?
                .to_device(device)?,
            Tensor::from_vec(vec![0.7f32; experts], experts, device)?,
        )
    }

    #[test]
    fn quantized_fast_decode_lora_matches_gather_pipeline() -> Result<()> {
        const EXPERTS: usize = 3;
        const HIDDEN: usize = 64;
        const INTERMEDIATE: usize = 96;
        const RANK: usize = 8;
        const TOKENS: usize = 2;
        const TOPK: usize = 2;

        let device = Device::new_cuda(0)?;
        let fast = FastExpertsWeights {
            fused_gate_proj: quant_method(tensor((EXPERTS, INTERMEDIATE, HIDDEN), 0.1)?, &device)?,
            fused_up_proj: quant_method(tensor((EXPERTS, INTERMEDIATE, HIDDEN), 0.7)?, &device)?,
            fused_down_proj: quant_method(tensor((EXPERTS, HIDDEN, INTERMEDIATE), 1.3)?, &device)?,
            sharded: false,
        };

        let registry = LoraLayerRegistry::new();
        let site = registry.register_expert(
            LoraSiteKey::new("test.layers.0.experts"),
            LoraExpertSiteSpec::new(
                EXPERTS,
                HIDDEN,
                INTERMEDIATE,
                LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
                Shard::default(),
                Shard::default(),
            )?,
            DType::F32,
            device.clone(),
        )?;
        registry.finalize()?;
        let adapter = LoraExpertWeights::new(
            &site,
            Some(lora_projection(
                EXPERTS,
                RANK,
                HIDDEN,
                INTERMEDIATE,
                1.9,
                DType::F32,
                &device,
            )?),
            Some(lora_projection(
                EXPERTS,
                RANK,
                HIDDEN,
                INTERMEDIATE,
                2.3,
                DType::F32,
                &device,
            )?),
            Some(lora_projection(
                EXPERTS,
                RANK,
                INTERMEDIATE,
                HIDDEN,
                2.7,
                DType::F32,
                &device,
            )?),
        )?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(1); TOKENS]);
        execution.insert_expert(&site, 1, adapter)?;
        let execution = Arc::new(execution);
        let lora = with_lora_execution(Some(execution), || LoraExpertExecution::current(&site))?
            .expect("expert LoRA is active");

        let xs = tensor((TOKENS, 1, HIDDEN), 3.1)?.to_device(&device)?;
        let xs_flat = xs.reshape((TOKENS, HIDDEN))?;
        let topk_ids = Tensor::from_slice(&[0u32, 2, 1, 0], (TOKENS, TOPK), &device)?;
        let topk_weights = Tensor::from_slice(&[0.65f32, 0.35, 0.4, 0.6], (TOKENS, TOPK), &device)?;
        let forward = MoEForward {
            xs_flat: &xs_flat,
            topk_weights: &topk_weights,
            topk_ids: &topk_ids,
            original_dtype: DType::F32,
            shape: MoEForwardShape {
                batch_size: TOKENS,
                seq_len: 1,
                hidden_dim: HIDDEN,
                num_tokens: TOKENS,
                phase: MoEForwardPhase::Decode,
            },
            lora: Some(lora),
        };
        let config = MoEForwardConfig {
            num_experts: EXPERTS,
            num_experts_per_tok: TOPK,
            act: Activation::Silu,
        };

        let actual = fast.forward_impl(&forward, config)?;
        let expected = fast.forward_reference(&forward, config)?;
        let error = (&actual - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error <= 0.08, "fast decode max error {error}");
        Ok(())
    }

    fn run_quantized_grouped_prefill_lora(dtype: DType, tolerance: f32) -> Result<()> {
        const EXPERTS: usize = 3;
        const HIDDEN: usize = 64;
        const INTERMEDIATE: usize = 96;
        const RANK: usize = 8;
        const TOKENS: usize = 32;
        const TOPK: usize = 2;

        let device = Device::new_cuda(0)?;
        let fast = FastExpertsWeights {
            fused_gate_proj: quant_method(
                (tensor((EXPERTS, INTERMEDIATE, HIDDEN), 0.1)? * 4.0)?,
                &device,
            )?,
            fused_up_proj: quant_method(
                (tensor((EXPERTS, INTERMEDIATE, HIDDEN), 0.7)? * 4.0)?,
                &device,
            )?,
            fused_down_proj: quant_method(
                (tensor((EXPERTS, HIDDEN, INTERMEDIATE), 1.3)? * 4.0)?,
                &device,
            )?,
            sharded: false,
        };

        let registry = LoraLayerRegistry::new();
        let site = registry.register_expert(
            LoraSiteKey::new("test.layers.0.experts"),
            LoraExpertSiteSpec::new(
                EXPERTS,
                HIDDEN,
                INTERMEDIATE,
                LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
                Shard::default(),
                Shard::default(),
            )?,
            dtype,
            device.clone(),
        )?;
        registry.finalize()?;
        let adapter = LoraExpertWeights::new(
            &site,
            Some(lora_projection(
                EXPERTS,
                RANK,
                HIDDEN,
                INTERMEDIATE,
                1.9,
                dtype,
                &device,
            )?),
            Some(lora_projection(
                EXPERTS,
                RANK,
                HIDDEN,
                INTERMEDIATE,
                2.3,
                dtype,
                &device,
            )?),
            Some(lora_projection(
                EXPERTS,
                RANK,
                INTERMEDIATE,
                HIDDEN,
                2.7,
                dtype,
                &device,
            )?),
        )?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(1); TOKENS]);
        execution.insert_expert(&site, 1, adapter)?;
        let execution = Arc::new(execution);
        let lora = with_lora_execution(Some(execution), || LoraExpertExecution::current(&site))?
            .expect("expert LoRA is active");

        let xs = (tensor((1, TOKENS, HIDDEN), 3.1)? * 4.0)?
            .to_dtype(dtype)?
            .to_device(&device)?;
        let xs_flat = xs.reshape((TOKENS, HIDDEN))?;
        let topk_ids = (0..TOKENS)
            .flat_map(|token| {
                [
                    u32::try_from(token % EXPERTS).expect("expert index is bounded"),
                    u32::try_from((token + 1) % EXPERTS).expect("expert index is bounded"),
                ]
            })
            .collect::<Vec<_>>();
        let topk_ids = Tensor::from_vec(topk_ids, (TOKENS, TOPK), &device)?;
        let topk_weights = (0..TOKENS)
            .flat_map(|token| {
                let cycle = u16::try_from(token % 5).expect("cycle is bounded");
                let first = 0.25 + 0.5 * f32::from(cycle) / 4.0;
                [first, 1.0 - first]
            })
            .collect::<Vec<_>>();
        let topk_weights = Tensor::from_vec(topk_weights, (TOKENS, TOPK), &device)?;
        let forward = MoEForward {
            xs_flat: &xs_flat,
            topk_weights: &topk_weights,
            topk_ids: &topk_ids,
            original_dtype: dtype,
            shape: MoEForwardShape {
                batch_size: 1,
                seq_len: TOKENS,
                hidden_dim: HIDDEN,
                num_tokens: TOKENS,
                phase: MoEForwardPhase::Prefill,
            },
            lora: Some(lora),
        };
        let config = MoEForwardConfig {
            num_experts: EXPERTS,
            num_experts_per_tok: TOPK,
            act: Activation::Silu,
        };

        let actual = fast.forward_impl(&forward, config)?;
        let expected = fast.forward_reference(&forward, config)?;
        assert_eq!(actual.dtype(), dtype);
        assert_eq!(expected.dtype(), dtype);
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        let error = (&actual - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let scale = expected.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            error <= tolerance * (1.0 + scale),
            "grouped prefill {dtype:?} max error {error} at reference scale {scale}"
        );
        Ok(())
    }

    #[test]
    fn quantized_grouped_prefill_lora_f32_matches_gather_pipeline() -> Result<()> {
        run_quantized_grouped_prefill_lora(DType::F32, 0.08)
    }

    #[test]
    fn quantized_grouped_prefill_lora_f16_matches_gather_pipeline() -> Result<()> {
        run_quantized_grouped_prefill_lora(DType::F16, 0.1)
    }

    #[test]
    fn quantized_grouped_prefill_lora_bf16_matches_gather_pipeline() -> Result<()> {
        run_quantized_grouped_prefill_lora(DType::BF16, 0.12)
    }
}

impl FastExpertsWeights {
    fn apply_immediate_isq(self, experts_vb: ShardedVarBuilder) -> Result<Self> {
        let keys = QuantizedExpertKeys::new(&experts_vb.prefix());
        let load = |layer: Arc<dyn QuantMethod>, name: &str, key: String| {
            apply_immediate_isq_with_key(
                layer,
                experts_vb.pp(name),
                Some(key),
                Some(Shard::default()),
            )
        };
        Ok(Self {
            fused_gate_proj: load(self.fused_gate_proj, "gate_proj", keys.gate)?,
            fused_up_proj: load(self.fused_up_proj, "up_proj", keys.up)?,
            fused_down_proj: load(self.fused_down_proj, "down_proj", keys.down)?,
            sharded: self.sharded,
        })
    }

    /// Pre-quantized standard tree (`experts.*` / `switch_mlp.*`).
    pub(super) fn load_prequantized(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        quantization_config: &Option<QuantizedConfig>,
    ) -> Result<FastExpertsWeights> {
        #[cfg(feature = "cutile")]
        register_cutile_moe_routing(cfg, vb.device())?;
        let experts_vb = vb.pp("experts");
        let read_vb = if should_stage_prequantized_on_cpu(&experts_vb) && !vb.device().is_cpu() {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };
        let PreQuantizedExperts {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        } = PreQuantizedExperts::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts,
            quantization_config,
            read_vb,
        )?;
        FastExpertsWeights {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
            sharded: false,
        }
        .apply_immediate_isq(experts_vb)
    }
}

impl FusedExpertsWeights {
    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        if forward.lora.is_some() {
            return self.forward_lora(forward, config);
        }
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

    fn forward_lora(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let is_prefill = forward.shape.phase.is_prefill();
        let (expert_ids, sorted_token_ids) = if is_prefill {
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
        let top_k = config.num_experts_per_tok;
        let num_tokens = forward.shape.num_tokens;
        let inter = self.w.w_size_n;

        let gate_up = moe::moe_gemm(
            forward.xs_flat,
            &self.w.gate_up,
            &None,
            &sorted_token_ids,
            &expert_ids,
            top_k,
            is_prefill,
        )?;
        let lora = forward.lora.as_ref().expect("LoRA path requires execution");
        let gate_up =
            lora.add_gate_up_delta_combined_owned(forward.xs_flat, gate_up, forward.topk_ids)?;
        let down_inputs = crate::ops::split_mul_and_act(&gate_up, inter, config.act)?;
        let down = moe::moe_gemm(
            &down_inputs.reshape((num_tokens * top_k, inter))?,
            &self.w.down,
            &Some(forward.topk_weights.clone()),
            &sorted_token_ids,
            &expert_ids,
            top_k,
            is_prefill,
        )?
        .reshape((num_tokens, top_k, forward.shape.hidden_dim))?;
        let down = lora.add_delta_owned(
            LoraExpertProjection::Down,
            &down_inputs,
            down,
            forward.topk_ids,
            Some(forward.topk_weights),
            LoraExpertInputMode::RoutedRows,
        )?;
        down.sum(D::Minus2)?.to_dtype(forward.original_dtype)
    }
}

#[cfg(feature = "cutile")]
impl CutileExpertsWeights {
    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        if forward.lora.is_some() {
            return self.forward_lora(forward, config);
        }
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
        let (sids, eids, ntpp, em) = mistralrs_quant::moe::cuda::moe_align(
            &ti_slice.slice(ti_layout.start_offset()..ti_layout.start_offset() + num_valid),
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

        let ic2 =
            mistralrs_quant::moe::cuda::act_and_mul(&ic1, inter, gated_act(config.act)?, dev)?;

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
        let tw_view =
            tw_slice.slice(tw_layout.start_offset()..tw_layout.start_offset() + num_valid);

        let ic3 = mistralrs_quant::cutile::cutile_grouped_gemm(
            &ic2,
            &self.w.down,
            &sids,
            &eids,
            &ntpp,
            Some(&tw_view),
            em,
            num_valid,
            1,
            true,
            cfg,
            dev,
        )?;

        mistralrs_quant::moe::cuda::moe_sum_bf16(&ic3, num_tokens, topk, dev)
    }

    fn forward_lora(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
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
        let (sids, eids, ntpp, em) = mistralrs_quant::moe::cuda::moe_align(
            &ti_slice.slice(ti_layout.start_offset()..ti_layout.start_offset() + num_valid),
            num_tokens,
            num_experts,
            topk,
            cfg.bm,
            dev,
        )?;
        let gate_up = mistralrs_quant::cutile::cutile_grouped_gemm(
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
        let lora = forward.lora.as_ref().expect("LoRA path requires execution");
        let gate_up =
            lora.add_gate_up_delta_combined_owned(forward.xs_flat, gate_up, forward.topk_ids)?;
        let down_inputs = crate::ops::split_mul_and_act(&gate_up, inter, config.act)?;

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
        let tw_view =
            tw_slice.slice(tw_layout.start_offset()..tw_layout.start_offset() + num_valid);
        let down = mistralrs_quant::cutile::cutile_grouped_gemm(
            &down_inputs.reshape((num_valid, inter))?,
            &self.w.down,
            &sids,
            &eids,
            &ntpp,
            Some(&tw_view),
            em,
            num_valid,
            1,
            true,
            cfg,
            dev,
        )?
        .reshape((num_tokens, topk, forward.shape.hidden_dim))?;
        let down = lora.add_delta_owned(
            LoraExpertProjection::Down,
            &down_inputs,
            down,
            forward.topk_ids,
            Some(forward.topk_weights),
            LoraExpertInputMode::RoutedRows,
        )?;
        mistralrs_quant::moe::cuda::moe_sum_bf16(
            &down.reshape((num_valid, forward.shape.hidden_dim))?,
            num_tokens,
            topk,
            dev,
        )?
        .to_dtype(forward.original_dtype)
    }
}

impl FastExpertsWeights {
    fn routed_experts(
        &self,
        config: MoEForwardConfig,
    ) -> mistralrs_quant::routed_moe::RoutedExperts<'_> {
        mistralrs_quant::routed_moe::RoutedExperts {
            gate: self.fused_gate_proj.as_ref(),
            up: self.fused_up_proj.as_ref(),
            down: self.fused_down_proj.as_ref(),
            num_experts: config.num_experts,
        }
    }

    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        self.routed_experts(config).forward(
            forward.routed_input(),
            mistralrs_quant::routed_moe::RoutedActivation {
                fused: crate::ops::glu_activation_type(config.act),
                fallback: &config.act,
            },
        )
    }

    #[cfg(test)]
    fn forward_reference(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        self.routed_experts(config).gather(
            forward.routed_input(),
            mistralrs_quant::routed_moe::RoutedActivation {
                fused: crate::ops::glu_activation_type(config.act),
                fallback: &config.act,
            },
        )
    }
}

#[cfg(test)]
mod immediate_isq_tests {
    use super::*;
    use candle_core::Device;
    use regex::Regex;
    use std::collections::HashMap;

    struct ClearImmediateIsq;

    impl Drop for ClearImmediateIsq {
        fn drop(&mut self) {
            mistralrs_quant::clear_immediate_isq();
        }
    }

    fn dummy() -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?))
    }

    fn experts_vb() -> ShardedVarBuilder {
        mistralrs_quant::ShardedSafeTensors::wrap(HashMap::new(), DType::F32, Device::Cpu)
            .pp("model.layers.0.mlp.experts")
    }

    #[test]
    fn partial_prequantized_selection_keeps_original_load_device() {
        mistralrs_quant::set_immediate_isq(
            Some(mistralrs_quant::IsqType::Q8_0),
            vec![
                Regex::new(r"^model\.layers\.0\.mlp\.experts\.gate_proj\.weight$")
                    .expect("valid regex"),
            ],
            mistralrs_quant::IsqCaptureMode::CaptureMatches,
        );
        let _clear = ClearImmediateIsq;

        assert!(!should_stage_prequantized_on_cpu(&experts_vb()));
    }

    #[test]
    fn prequantized_experts_register_canonical_immediate_isq_keys() -> Result<()> {
        let experts_vb = experts_vb();
        let tracker = experts_vb.tracker().clone();
        mistralrs_quant::set_immediate_isq(
            Some(mistralrs_quant::IsqType::Q8_0),
            vec![Regex::new(
                r"^model\.layers\.0\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )
            .expect("valid regex")],
            mistralrs_quant::IsqCaptureMode::CaptureMatches,
        );
        let _clear = ClearImmediateIsq;
        assert!(should_stage_prequantized_on_cpu(&experts_vb));

        FastExpertsWeights {
            fused_gate_proj: dummy()?,
            fused_up_proj: dummy()?,
            fused_down_proj: dummy()?,
            sharded: false,
        }
        .apply_immediate_isq(experts_vb)?;

        let modules = tracker.get().clone();
        assert_eq!(modules.len(), 3);
        assert_eq!(
            modules
                .iter()
                .map(|module| module.key.as_str())
                .collect::<Vec<_>>(),
            [
                "model.layers.0.mlp.experts.gate_proj",
                "model.layers.0.mlp.experts.up_proj",
                "model.layers.0.mlp.experts.down_proj",
            ]
        );
        for module in modules {
            module.ct.resolve()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod nvfp4_tests {
    use super::*;
    use crate::layers::Activation;
    use crate::moe::experts::forward::MoEForwardShape;
    use mistralrs_quant::{Nvfp4ActivationMode, Nvfp4Layer, Nvfp4LayerParts};

    const FEATURES: usize = 16;
    const EXPERTS: usize = 2;
    const TOKENS: usize = 2;
    const TOPK: usize = 2;

    fn projection(values: [f32; EXPERTS], quantized: bool) -> Result<Arc<dyn QuantMethod>> {
        let device = &Device::Cpu;
        let globals = Tensor::from_vec(
            values
                .into_iter()
                .flat_map(|value| std::iter::repeat_n(value, FEATURES))
                .collect::<Vec<_>>(),
            (EXPERTS, FEATURES),
            device,
        )?;
        if quantized {
            Ok(Arc::new(Nvfp4Layer::from_parts(Nvfp4LayerParts {
                weight: Tensor::full(0x22u8, (EXPERTS, FEATURES, FEATURES / 2), device)?,
                scales: Tensor::ones((EXPERTS, FEATURES, 1), DType::F8E4M3, device)?,
                global_scales: globals,
                input_scale: None,
                activation: Nvfp4ActivationMode::None,
                bias: None,
                dtype: DType::F32,
            })?))
        } else {
            Ok(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(
                    globals
                        .unsqueeze(2)?
                        .broadcast_as((EXPERTS, FEATURES, FEATURES))?
                        .contiguous()?,
                    None,
                )),
            )?))
        }
    }

    #[test]
    fn cpu_gather_combines_nvfp4_and_unquantized_projections() -> Result<()> {
        let device = &Device::Cpu;
        let mut input = vec![0f32; TOKENS * FEATURES];
        input[0] = 1.0;
        input[FEATURES] = 2.0;
        let xs = Tensor::from_vec(input, (1, TOKENS, FEATURES), device)?;
        let xs_flat = xs.reshape((TOKENS, FEATURES))?;
        let ids = Tensor::new(&[[0u32, 1], [1, 0]], device)?;
        let weights = Tensor::new(&[[0.25f32, 0.75], [0.75, 0.25]], device)?;
        let forward = MoEForward {
            xs_flat: &xs_flat,
            topk_weights: &weights,
            topk_ids: &ids,
            original_dtype: DType::F32,
            shape: MoEForwardShape::new(1, TOKENS, FEATURES),
            lora: None,
        };
        let config = MoEForwardConfig {
            num_experts: EXPERTS,
            num_experts_per_tok: TOPK,
            act: Activation::Relu,
        };
        for quantized_projection in 0..3 {
            let experts = FastExpertsWeights {
                fused_gate_proj: projection([1.0, 2.0], quantized_projection == 0)?,
                fused_up_proj: projection([3.0, 4.0], quantized_projection == 1)?,
                fused_down_proj: projection([1.0, 2.0], quantized_projection == 2)?,
                sharded: false,
            };
            let output = experts
                .forward_reference(&forward, config)?
                .to_vec2::<f32>()?;
            assert_eq!(output, vec![vec![204.0; FEATURES], vec![816.0; FEATURES]]);
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;
    use crate::layers::Activation;
    use crate::moe::experts::forward::{MoEForwardPhase, MoEForwardShape};

    fn tensor(shape: impl Into<candle_core::Shape>, phase: f32) -> Result<Tensor> {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|index| {
                let index = u16::try_from(index % 4096).expect("index is bounded");
                (f32::from(index) * 0.017 + phase).sin() * 0.08
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu)
    }

    #[test]
    fn metal_sorted_afq_moe_matches_gather_pipeline() -> Result<()> {
        // 128-wide experts exercise the gather route that previously picked a missing bs_qmv_quad kernel.
        for (hidden, intermediate) in [(512, 512), (128, 256)] {
            sorted_matches_gather(hidden, intermediate)?;
        }
        Ok(())
    }

    fn sorted_matches_gather(hidden: usize, intermediate: usize) -> Result<()> {
        use mistralrs_quant::{AfqBits, AfqGroupSize, AfqLayer};

        const EXPERTS: usize = 4;
        const TOKENS: usize = 40;
        const TOPK: usize = 2;

        let device = Device::new_metal(0)?;
        let afq = |shape: (usize, usize, usize), phase: f32| -> Result<Arc<dyn QuantMethod>> {
            let weight = tensor(shape, phase)?
                .to_dtype(DType::BF16)?
                .to_device(&device)?;
            Ok(Arc::new(AfqLayer::new(QuantMethodConfig::Afq {
                weight,
                bias: None,
                bits: AfqBits::Four,
                group_size: AfqGroupSize::Med,
            })?))
        };
        let fast = FastExpertsWeights {
            fused_gate_proj: afq((EXPERTS, intermediate, hidden), 0.1)?,
            fused_up_proj: afq((EXPERTS, intermediate, hidden), 0.7)?,
            fused_down_proj: afq((EXPERTS, hidden, intermediate), 1.3)?,
            sharded: false,
        };

        let xs = (tensor((TOKENS, 1, hidden), 3.1)? * 10.0)?
            .to_dtype(DType::BF16)?
            .to_device(&device)?;
        let xs_flat = xs.reshape((TOKENS, hidden))?;
        let ids = (0..TOKENS * TOPK)
            .map(|i| u32::try_from((i * 7 + i / TOPK) % EXPERTS).expect("small"))
            .collect::<Vec<_>>();
        let topk_ids = Tensor::from_slice(&ids, (TOKENS, TOPK), &device)?;
        let topk_weights = Tensor::from_vec(
            (0..TOKENS * TOPK)
                .map(|i| if i % 2 == 0 { 0.6f32 } else { 0.4 })
                .collect::<Vec<_>>(),
            (TOKENS, TOPK),
            &device,
        )?;
        let forward = MoEForward {
            xs_flat: &xs_flat,
            topk_weights: &topk_weights,
            topk_ids: &topk_ids,
            original_dtype: DType::BF16,
            shape: MoEForwardShape {
                batch_size: TOKENS,
                seq_len: 1,
                hidden_dim: hidden,
                num_tokens: TOKENS,
                phase: MoEForwardPhase::Prefill,
            },
            lora: None,
        };
        for act in [Activation::Silu, Activation::GeluPytorchTanh] {
            let config = MoEForwardConfig {
                num_experts: EXPERTS,
                num_experts_per_tok: TOPK,
                act,
            };
            let actual = fast.forward_impl(&forward, config)?;
            let expected = fast.forward_reference(&forward, config)?;
            let expected_f32 = expected.to_dtype(DType::F32)?;
            let error = ((actual.to_dtype(DType::F32)? - &expected_f32)?.abs()?
                / expected_f32.abs()?.maximum(0.05)?)?
            .max_all()?
            .to_scalar::<f32>()?;
            assert!(error < 3e-2, "{act:?}: relative error {error}");
        }
        Ok(())
    }
}
