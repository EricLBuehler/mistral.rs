mod default_scheduler;

use std::sync::Arc;

pub use default_scheduler::{DefaultScheduler, DefaultSchedulerMethod, DefaultSchedulerOutput};
use tokio::sync::Mutex;

use crate::{
    engine::IntervalLogger,
    paged_attention::{
        BlockEngine, BlockTables, CacheConfig, PagedAttentionScheduler,
        PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
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

    // PagedAttention metadata
    fn block_tables(&self) -> Option<BlockTables>;
    fn block_size(&self) -> Option<usize>;
    fn block_engine(&self) -> Option<Arc<Mutex<BlockEngine>>>;
}
