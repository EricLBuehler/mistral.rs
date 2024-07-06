/// The higher-level manager of the blocks allocated. Operations performed by the block engine do
/// not directly change memory.
mod block_engine;
mod block_engine_sequence;
/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
mod cache_engine;
mod config;
mod layers;
mod scheduler;

pub use block_engine::{BlockEngine, LogicalTokenBlock};
pub use block_engine_sequence::BlockEngineSequence;
pub use cache_engine::{CacheConfig, CacheEngine};
pub use config::ModelConfigLike;
pub use layers::{InputMetadata, PagedAttention};
pub use scheduler::{
    PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
};
