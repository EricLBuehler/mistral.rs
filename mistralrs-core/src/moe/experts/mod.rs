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
use mistralrs_quant::{QuantMethod, QuantizedConfig, ShardedVarBuilder, SumAllReduce};
use std::sync::Arc;

use crate::layers::Activation;

pub use config::MoEExpertsConfig;

#[cfg(feature = "cutile")]
use backends::CutileExpertsWeights;
use backends::{FastExpertsWeights, FusedExpertsWeights, SlowExpertsWeights, StackedExpertWeights};
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
    Fast(FastExpertsWeights),
    Slow(SlowExpertsWeights),
}

impl MoEExpertsBackendImpl {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        match self {
            Self::Fused(w) => w.forward_impl(forward, config),
            #[cfg(feature = "cutile")]
            Self::Cutile(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutile")),
            Self::Fast(w) => w.forward_impl(forward, config),
            Self::Slow(w) => w.forward_impl(forward, config),
        }
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match self {
            Self::Fused(_) => vec![],
            #[cfg(feature = "cutile")]
            Self::Cutile(_) => vec![],
            Self::Fast(w) => vec![
                &mut w.fused_gate_proj,
                &mut w.fused_up_proj,
                &mut w.fused_down_proj,
            ],
            Self::Slow(w) => {
                let e = &mut w.experts;
                let mut layers = Vec::with_capacity(e.gate_proj.len() * 3);
                for ((gate, up), down) in e
                    .gate_proj
                    .iter_mut()
                    .zip(e.up_proj.iter_mut())
                    .zip(e.down_proj.iter_mut())
                {
                    layers.push(gate);
                    layers.push(up);
                    layers.push(down);
                }
                layers
            }
        }
    }

    fn num_isq_layers(&self) -> usize {
        match self {
            Self::Fused(_) => 0,
            #[cfg(feature = "cutile")]
            Self::Cutile(_) => 0,
            Self::Fast(_) => 3,
            Self::Slow(w) => w.experts.gate_proj.len() * 3,
        }
    }
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
        let choice = BackendChoice::new(
            layer_device,
            experts_vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        );
        let ckpt = ExpertCheckpoint::new(cfg, experts_vb.clone(), comm);

        let backend_impl =
            match MoEExpertsBackend::resolve(&choice) {
                MoEExpertsBackend::Fused => MoEExpertsBackendImpl::Fused(FusedExpertsWeights {
                    w: StackedExpertWeights::from_checkpoint(&ckpt)?,
                }),
                #[cfg(feature = "cutile")]
                MoEExpertsBackend::Cutile => {
                    MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::from_checkpoint(&ckpt)?)
                }
                MoEExpertsBackend::Fast => MoEExpertsBackendImpl::Fast(
                    FastExpertsWeights::load_fused(cfg, vb, quantization_config)?,
                ),
                MoEExpertsBackend::Slow => MoEExpertsBackendImpl::Slow(SlowExpertsWeights::load(
                    cfg,
                    experts_vb,
                    comm,
                    quantization_config,
                )?),
            };

        Ok(Self::from_backend(backend_impl, cfg, comm, act))
    }

    /// Create MoEExperts from a VB already at the experts level (gemma4's flat `moe.*`, no
    /// `experts.*` sublevel). The raw backends share `new`'s path; the quant backends use the
    /// flat-tree loaders.
    pub fn new_direct(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
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
            MoEExpertsBackend::Fast => {
                if ckpt.combined && quantization_config.is_none() {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_combined_stacked(
                        cfg, experts_vb,
                    )?)
                } else if ckpt.combined {
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
                if ckpt.combined {
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
