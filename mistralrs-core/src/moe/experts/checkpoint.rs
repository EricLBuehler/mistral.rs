use candle_core::{Result, Tensor};
use mistralrs_quant::{Shard, ShardedVarBuilder};
use std::sync::Arc;

use crate::moe::shard;

use super::config::MoEExpertsConfig;

/// Reads the experts checkpoint in any on-disk layout (combined `gate_up_proj` vs per-expert, detected once) and yields canonical ENK weights; the caller picks the root prefix.
pub(super) struct ExpertCheckpoint<'a> {
    pub(super) cfg: &'a MoEExpertsConfig,
    vb: ShardedVarBuilder,
    rank: usize,
    world_size: usize,
    pub(super) combined: bool,
}

impl<'a> ExpertCheckpoint<'a> {
    pub(super) fn new(
        cfg: &'a MoEExpertsConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Self {
        let combined = vb.contains_tensor("gate_up_proj");
        Self {
            cfg,
            vb,
            rank: comm.rank(),
            world_size: comm.world_size(),
            combined,
        }
    }

    /// Canonical ENK: gate_up [E, 2*inter, hidden], down [E, hidden, inter].
    pub(super) fn stacked_enk(&self) -> Result<(Tensor, Tensor)> {
        let cfg = self.cfg;
        let num_experts = cfg.num_experts;
        if self.combined {
            // gate_up concatenates gate and up along the inter dim, so a naive shard of that dim
            // would hand whole gate/up halves to different ranks. Shard each half separately.
            let gate_up = if self.world_size > 1 {
                let inter_shard = cfg.moe_intermediate_size / self.world_size;
                if !cfg.moe_intermediate_size.is_multiple_of(self.world_size) {
                    candle_core::bail!(
                        "Intermediate size {} is not divisible by world size {}.",
                        cfg.moe_intermediate_size,
                        self.world_size
                    );
                }
                let gate = self.read_proj_offset(
                    (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                    "gate_up_proj",
                    1,
                    2,
                    self.rank * inter_shard,
                    inter_shard,
                )?;
                let up = self.read_proj_offset(
                    (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                    "gate_up_proj",
                    1,
                    2,
                    cfg.moe_intermediate_size + self.rank * inter_shard,
                    inter_shard,
                )?;
                Tensor::cat(&[&gate, &up], 1)?
            } else {
                self.read_proj(
                    (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                    "gate_up_proj",
                    1,
                    2,
                )?
            };
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
            let names = cfg.proj_names;
            for i in 0..num_experts {
                let expert_vb = self.vb.pp(i.to_string());
                let gate = expert_vb.pp(names.gate).get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, self.rank, self.world_size),
                )?;
                let up = expert_vb.pp(names.up).get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, self.rank, self.world_size),
                )?;
                let down = expert_vb.pp(names.down).get_with_hints(
                    (cfg.hidden_size, cfg.moe_intermediate_size),
                    "weight",
                    shard(1, self.rank, self.world_size),
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
                shard(canonical_shard, self.rank, self.world_size),
            )
            .or_else(|_| {
                self.vb
                    .get_with_hints(
                        transposed,
                        name,
                        shard(transposed_shard, self.rank, self.world_size),
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })
    }

    /// Like `read_proj`, but slicing an explicit offset range of the sharded dim.
    #[allow(clippy::too_many_arguments)]
    fn read_proj_offset(
        &self,
        canonical: (usize, usize, usize),
        transposed: (usize, usize, usize),
        name: &str,
        canonical_dim: usize,
        transposed_dim: usize,
        offset: usize,
        len: usize,
    ) -> Result<Tensor> {
        self.vb
            .get_with_hints(
                canonical,
                name,
                Shard::Offset {
                    dim: canonical_dim,
                    offset,
                    len,
                },
            )
            .or_else(|_| {
                self.vb
                    .get_with_hints(
                        transposed,
                        name,
                        Shard::Offset {
                            dim: transposed_dim,
                            offset,
                            len,
                        },
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })
    }
}
