//! Unified MoE experts layer supporting multiple backends and weight formats.
//!
//! `MoEExperts` does NOT carry the gate (router); the caller computes routing. Three orthogonal
//! axes, each in one place: [`config`] (which backend to run), [`checkpoint`] (what's on disk ->
//! canonical ENK), and [`backends`] (materialize + forward). [`forward`] holds the per-call shapes.

mod backends;
mod checkpoint;
mod config;
mod forward;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::{IsqType, QuantizedConfig, ShardedVarBuilder, SumAllReduce};
use std::sync::Arc;

use crate::layers::Activation;

pub use config::{ExpertProjNames, MoEExpertsConfig};

#[cfg(feature = "cutile")]
use backends::CutileExpertsWeights;
#[cfg(feature = "cuda")]
use backends::CutlassExpertsWeights;
use backends::{
    experts_are_prequantized, FastExpertsWeights, FusedExpertsWeights, StackedExpertWeights,
};
use checkpoint::ExpertCheckpoint;
use config::{BackendChoice, MoEExpertsBackend};
use forward::{MoEForward, MoEForwardConfig, MoEForwardShape};

/// MoE experts layer without the gate; the caller computes routing weights + topk indices.
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
    #[cfg(feature = "cuda")]
    Cutlass(CutlassExpertsWeights),
    Fast(FastExpertsWeights),
}

impl MoEExpertsBackendImpl {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        match self {
            Self::Fused(w) => w.forward_impl(forward, config),
            #[cfg(feature = "cutile")]
            Self::Cutile(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutile")),
            #[cfg(feature = "cuda")]
            Self::Cutlass(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutlass")),
            Self::Fast(w) => w.forward_impl(forward, config),
        }
    }
}

// The gather-based forward requires `gather_forward` support from the quantized layer.
fn check_isq_gather_support() -> Result<()> {
    let Some(params) = mistralrs_quant::get_immediate_isq() else {
        return Ok(());
    };
    if let Some(ty) = params.ty {
        if matches!(
            ty,
            IsqType::HQQ4 | IsqType::HQQ8 | IsqType::F8E4M3 | IsqType::F8Q8
        ) {
            candle_core::bail!("ISQ type {ty} is not supported for MoE experts.");
        }
    }
    Ok(())
}

impl MoEExperts {
    /// Create MoEExperts for a standard model: experts live under `vb.pp("experts")`.
    pub fn new(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        let experts_vb = vb.pp("experts").set_device(layer_device.clone());
        if let Some(fast) = FastExpertsWeights::from_uqff(cfg, &experts_vb, comm)? {
            return Ok(Self::from_backend(
                MoEExpertsBackendImpl::Fast(fast),
                cfg,
                comm,
                act,
            ));
        }
        check_isq_gather_support()?;
        let choice = BackendChoice::new(
            layer_device,
            experts_vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        );
        let ckpt = ExpertCheckpoint::new(cfg, experts_vb.clone(), comm);

        let backend_impl = match MoEExpertsBackend::resolve(&choice) {
            MoEExpertsBackend::Fused => MoEExpertsBackendImpl::Fused(FusedExpertsWeights {
                w: StackedExpertWeights::from_checkpoint(&ckpt)?,
            }),
            #[cfg(feature = "cutile")]
            MoEExpertsBackend::Cutile => {
                MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::from_checkpoint(&ckpt)?)
            }
            #[cfg(feature = "cuda")]
            MoEExpertsBackend::Cutlass => {
                MoEExpertsBackendImpl::Cutlass(CutlassExpertsWeights::from_checkpoint(&ckpt)?)
            }
            MoEExpertsBackend::Fast => {
                if experts_are_prequantized(cfg, quantization_config, &experts_vb) {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_prequantized(
                        cfg,
                        vb,
                        quantization_config,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_unquantized(
                        cfg, experts_vb, comm,
                    )?)
                }
            }
        };

        Ok(Self::from_backend(backend_impl, cfg, comm, act))
    }

    /// Create MoEExperts from a VB already at the experts level (gemma4's flat `moe.*`, no
    /// `experts.*` sublevel).
    pub fn new_direct(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        if let Some(fast) = FastExpertsWeights::from_uqff(cfg, &experts_vb, comm)? {
            return Ok(Self::from_backend(
                MoEExpertsBackendImpl::Fast(fast),
                cfg,
                comm,
                act,
            ));
        }
        check_isq_gather_support()?;
        let choice = BackendChoice::new(
            experts_vb.device().clone(),
            experts_vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        );
        let ckpt = ExpertCheckpoint::new(cfg, experts_vb.clone(), comm);

        let backend_impl = match MoEExpertsBackend::resolve(&choice) {
            MoEExpertsBackend::Fused => MoEExpertsBackendImpl::Fused(FusedExpertsWeights {
                w: StackedExpertWeights::from_checkpoint(&ckpt)?,
            }),
            #[cfg(feature = "cutile")]
            MoEExpertsBackend::Cutile => {
                MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::from_checkpoint(&ckpt)?)
            }
            #[cfg(feature = "cuda")]
            MoEExpertsBackend::Cutlass => {
                MoEExpertsBackendImpl::Cutlass(CutlassExpertsWeights::from_checkpoint(&ckpt)?)
            }
            MoEExpertsBackend::Fast => {
                if experts_are_prequantized(cfg, quantization_config, &experts_vb) {
                    candle_core::bail!(
                        "Pre-quantized experts are not supported for flat expert trees."
                    );
                }
                MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_unquantized(
                    cfg, experts_vb, comm,
                )?)
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

        // Sharded experts produce partial sums; replicated ones are already complete.
        let sharded = match &self.backend {
            MoEExpertsBackendImpl::Fast(w) => w.sharded,
            _ => true,
        };
        if self.world_size > 1 && sharded {
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
