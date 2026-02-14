mod default_scheduler;

use std::sync::Arc;

pub use default_scheduler::{DefaultScheduler, DefaultSchedulerMethod, DefaultSchedulerOutput};
use tokio::sync::Mutex;

use crate::{
    engine::IntervalLogger,
    paged_attention::{
        CacheConfig, KVCacheManager, PagedAttentionScheduler, PagedAttentionSchedulerConfig,
        PagedAttentionSchedulerOutput,
    },
    sequence::Sequence,
};

#[derive(Clone)]
pub enum SchedulerConfig {
    DefaultScheduler {
        method: DefaultSchedulerMethod,
    },
    PagedAttentionMeta {
        max_num_seqs: usize,
        config: CacheConfig,
    },
}

impl SchedulerConfig {
    pub fn into_scheduler(self) -> Arc<Mutex<dyn Scheduler>> {
        match self {
            Self::DefaultScheduler { method } => {
                Arc::new(Mutex::new(DefaultScheduler::new(method)))
            }
            Self::PagedAttentionMeta {
                max_num_seqs,
                config,
            } => Arc::new(Mutex::new(PagedAttentionScheduler::new(
                PagedAttentionSchedulerConfig { max_num_seqs },
                config,
            ))),
        }
    }
}

pub enum SchedulerOutput<'a> {
    DefaultScheduler {
        output: DefaultSchedulerOutput<'a>,
    },
    PagedAttention {
        output: PagedAttentionSchedulerOutput,
    },
}

pub trait Scheduler: Send + Sync {
    fn schedule(&mut self, logger: &IntervalLogger) -> SchedulerOutput<'_>;
    fn waiting_len(&self) -> usize;
    fn running_len(&self) -> usize;
    fn add_seq(&mut self, seq: Sequence);
    /// This may do nothing. It depends on the implementation
    fn free_finished_sequence_groups(&mut self);
    /// Get Mamba state pool indices of finished sequences for freeing.
    /// Called before free_finished_sequence_groups to allow cleanup of hybrid cache slots.
    fn get_finished_mamba_indices(&self) -> Vec<usize>;

    // PagedAttention metadata
    fn block_size(&self) -> Option<usize>;
    fn kv_cache_manager(&self) -> Option<Arc<Mutex<KVCacheManager>>>;

    /// Set whether prefix caching is enabled. Called by Engine after creation
    /// to synchronize with the global no_prefix_cache setting.
    fn set_prefix_caching_enabled(&mut self, enabled: bool);
}
