mod experts;

use mistralrs_quant::Shard;

#[cfg(feature = "cuda")]
pub(crate) use experts::GROUPED_PREFILL_MIN_TOKENS;
pub(crate) use experts::{expert_stack_available, rebuild_expert_projection};
pub use experts::{prelog_moe_backend, ExpertProj, ExpertProjNames, MoEExperts, MoEExpertsConfig};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
