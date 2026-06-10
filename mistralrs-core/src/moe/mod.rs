mod experts;

use mistralrs_quant::Shard;

pub use experts::{ExpertProjNames, MoEExperts, MoEExpertsConfig};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
