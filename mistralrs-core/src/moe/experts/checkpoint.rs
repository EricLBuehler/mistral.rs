use candle_core::{Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;
use std::sync::Arc;

use crate::moe::shard;

use super::config::MoEExpertsConfig;

/// Reads the experts checkpoint in any on-disk layout (combined `gate_up_proj` vs per-expert, detected once) and yields canonical ENK weights; the caller picks the root prefix.
pub(super) struct ExpertCheckpoint<'a> {
    pub(super) cfg: &'a MoEExpertsConfig,
    vb: ShardedVarBuilder,
    comm: &'a Arc<mistralrs_quant::Comm>,
    pub(super) combined: bool,
}

impl<'a> ExpertCheckpoint<'a> {
    pub(super) fn new(
        cfg: &'a MoEExpertsConfig,
        vb: ShardedVarBuilder,
        comm: &'a Arc<mistralrs_quant::Comm>,
    ) -> Self {
        let combined = vb.contains_tensor("gate_up_proj");
        Self {
            cfg,
            vb,
            comm,
            combined,
        }
    }

    /// Canonical ENK: gate_up [E, 2*inter, hidden], down [E, hidden, inter].
    pub(super) fn stacked_enk(&self) -> Result<(Tensor, Tensor)> {
        let cfg = self.cfg;
        let num_experts = cfg.num_experts;
        if self.combined {
            let gate_up = self.read_proj(
                (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                "gate_up_proj",
                1,
                2,
            )?;
            let down = self.read_proj(
                (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                "down_proj",
                2,
                1,
            )?;
            Ok((gate_up, down))
        } else {
            // Per-expert nn.Linear weights [out, in]; stacking gives natural ENK directly.
            let mut gate_up_experts = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);
            for i in 0..num_experts {
                let expert_vb = self.vb.pp(i.to_string());
                let gate = expert_vb.pp("gate_proj").get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, self.comm.rank(), self.comm.world_size()),
                )?;
                let up = expert_vb.pp("up_proj").get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, self.comm.rank(), self.comm.world_size()),
                )?;
                let down = expert_vb.pp("down_proj").get_with_hints(
                    (cfg.hidden_size, cfg.moe_intermediate_size),
                    "weight",
                    shard(1, self.comm.rank(), self.comm.world_size()),
                )?;
                gate_up_experts.push(Tensor::cat(&[&gate, &up], 0)?);
                down_experts.push(down);
            }
            Ok((
                Tensor::stack(&gate_up_experts, 0)?,
                Tensor::stack(&down_experts, 0)?,
            ))
        }
    }

    /// Read a combined `[E, out, in]` projection: canonical ENK first, transposing a conv-A
    /// `[E, in, out]` checkpoint into ENK as the fallback.
    fn read_proj(
        &self,
        canonical: (usize, usize, usize),
        transposed: (usize, usize, usize),
        name: &str,
        canonical_shard: usize,
        transposed_shard: usize,
    ) -> Result<Tensor> {
        self.vb
            .get_with_hints(
                canonical,
                name,
                shard(canonical_shard, self.comm.rank(), self.comm.world_size()),
            )
            .or_else(|_| {
                self.vb
                    .get_with_hints(
                        transposed,
                        name,
                        shard(transposed_shard, self.comm.rank(), self.comm.world_size()),
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })
    }
}
