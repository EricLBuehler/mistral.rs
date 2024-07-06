mod default_scheduler;

pub use default_scheduler::{DefaultScheduler, DefaultSchedulerMethod, DefaultSchedulerOutput};

use crate::{paged_attention::PagedAttentionSchedulerOutput, sequence::Sequence};

#[derive(Clone)]
pub enum SchedulerConfig {
    DefaultScheduler { method: DefaultSchedulerMethod },
}

impl SchedulerConfig {
    pub fn into_scheduler(self) -> Box<dyn Scheduler> {
        match self {
            Self::DefaultScheduler { method } => Box::new(DefaultScheduler::new(method)),
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
}
