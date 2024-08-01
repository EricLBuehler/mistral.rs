mod default_scheduler;

pub use default_scheduler::{DefaultScheduler, DefaultSchedulerMethod, DefaultSchedulerOutput};

use crate::{
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
    pub fn into_scheduler(self) -> Box<dyn Scheduler> {
        match self {
            Self::DefaultScheduler { method } => Box::new(DefaultScheduler::new(method)),
            Self::PagedAttentionMeta {
                max_num_seqs,
                config,
            } => Box::new(PagedAttentionScheduler::new(
                PagedAttentionSchedulerConfig { max_num_seqs },
                config,
            )),
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

pub trait Scheduler {
    fn schedule(&mut self) -> SchedulerOutput<'_>;
    fn waiting_len(&self) -> usize;
    fn add_seq(&mut self, seq: Sequence);
    /// This may do nothing. It depends on the implementation
    fn free_finished_sequence_groups(&mut self);

    // PagedAttention metadata
    fn block_tables(&self) -> Option<&BlockTables>;
    fn block_size(&self) -> Option<usize>;
    fn block_engine(&mut self) -> Option<&mut BlockEngine>;
}
