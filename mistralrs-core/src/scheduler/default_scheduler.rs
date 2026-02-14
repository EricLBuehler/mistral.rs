use std::{
    collections::{HashMap, VecDeque},
    num::NonZeroUsize,
    sync::{atomic::Ordering, Arc},
};

use crate::{
    engine::{IntervalLogger, TERMINATE_ALL_NEXT_STEP},
    paged_attention::KVCacheManager,
    sequence::{Sequence, SequenceState, StopReason},
};

use super::{Scheduler, SchedulerOutput};

pub trait FcfsBacker: Default {
    fn new() -> Self;
    fn add(&mut self, item: Sequence);
    fn into_iter(self) -> impl Iterator<Item = Sequence>;
    fn len(&self) -> usize;
    fn sort_ascending_ids(&mut self);
}

impl FcfsBacker for VecDeque<Sequence> {
    fn new() -> Self {
        Self::new()
    }
    fn add(&mut self, item: Sequence) {
        self.push_back(item)
    }
    fn into_iter(self) -> impl Iterator<Item = Sequence> {
        <Self as IntoIterator>::into_iter(self)
    }
    fn sort_ascending_ids(&mut self) {
        let slice = self.make_contiguous();
        slice.sort_by_key(|seq| *seq.id());
    }
    fn len(&self) -> usize {
        VecDeque::len(self)
    }
}

pub struct DefaultSchedulerOutput<'a> {
    pub completion: Box<[&'a mut Sequence]>,
    pub prompt: Box<[&'a mut Sequence]>,
}

/// The scheduler method controld how sequences are scheduled during each
/// step of the engine. For each scheduling step, the scheduler method is used if there
/// are not only running, only waiting sequences, or none. If is it used, then it
/// is used to allow waiting sequences to run.
#[derive(Clone)]
pub enum DefaultSchedulerMethod {
    Fixed(NonZeroUsize),
}

pub struct BucketedSeqs<Backer: FcfsBacker> {
    running: Vec<Sequence>,
    waiting: Backer,
}

pub trait BucketingManager<Backer: FcfsBacker>: Send + Sync {
    /// Bucket and waitlist running input sequences, returning the newly running sequences.
    fn bucket_and_waitlist_seqs_waiting(
        &mut self,
        running: Vec<Sequence>,
        waiting: Backer,
        discrete: bool,
    ) -> BucketedSeqs<Backer>;
}

// (cache length, (has_imgs && is_prompt), sequence offset)
// Bucket by that metric for images because if we are not a prompt, then this doesn't apply
type BucketKey = (usize, bool, usize);

struct FixedBucketingManager;

impl<Backer: FcfsBacker> BucketingManager<Backer> for FixedBucketingManager {
    /// Move the sequences into buckets, and run the ones with the shortest lengths.
    /// The others are moved to the waiting list (retaining high priority due to start time),
    /// without a state modification.
    fn bucket_and_waitlist_seqs_waiting(
        &mut self,
        running: Vec<Sequence>,
        mut waiting: Backer,
        discrete: bool,
    ) -> BucketedSeqs<Backer> {
        // Now, get the sequences with the smallest sequence lengths, and allow them to catch up.
        let mut seq_buckets: HashMap<BucketKey, Vec<Sequence>> = HashMap::new();
        let mut seq_priorities: HashMap<BucketKey, f64> = HashMap::new();
        for seq in running {
            let len = seq.len();
            match seq_buckets.get_mut(&(
                len,
                seq.images().is_some() && seq.is_prompt(),
                seq.token_offset(),
            )) {
                Some(bucket) => {
                    if !discrete {
                        *seq_priorities
                            .get_mut(&(
                                len,
                                seq.images().is_some() && seq.is_prompt(),
                                seq.token_offset(),
                            ))
                            .unwrap() += seq.compute_priority();
                    }
                    bucket.push(seq);
                }
                None => {
                    if !discrete {
                        seq_priorities.insert(
                            (
                                len,
                                seq.images().is_some() && seq.is_prompt(),
                                seq.token_offset(),
                            ),
                            seq.compute_priority(),
                        );
                    }
                    seq_buckets.insert(
                        (
                            len,
                            seq.images().is_some() && seq.is_prompt(),
                            seq.token_offset(),
                        ),
                        vec![seq],
                    );
                }
            }
        }
        let running = if seq_buckets.len() <= 1 {
            // Full steam ahead or have everything
            seq_buckets
                .into_iter()
                .flat_map(|(_, x)| x)
                .map(|s| s.reset_urgency())
                .collect::<Vec<_>>()
        } else {
            // Set the min seqs to be the running ones, and the rest to be waiting (but their states are not changed!)
            // Allow the min seqs to catch up.
            let min = *seq_buckets
                .keys()
                .min_by_key(|(x, _, _)| *x)
                .expect("No sequence buckets.");
            let len = if !discrete {
                seq_priorities
                    .iter()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(a, b)| (a, *b))
                    .unwrap_or_else(|| (&min, seq_priorities[&min]))
                    .0
            } else {
                &min
            };
            let highest_priority_seqs = seq_buckets
                .remove(len)
                .unwrap()
                .into_iter()
                .map(|s| s.reset_urgency())
                .collect();
            for (_, seqs) in seq_buckets {
                for seq in seqs {
                    waiting.add(seq.add_urgency());
                }
            }
            // Know min_seqs.len < running.len() <= max
            highest_priority_seqs
        };
        BucketedSeqs { running, waiting }
    }
}

pub struct DefaultScheduler<Backer: FcfsBacker> {
    waiting: Backer,
    running: Vec<Sequence>,
    method: DefaultSchedulerMethod,
    bucketing_manager: Box<dyn BucketingManager<Backer>>,
}

impl<Backer: FcfsBacker> DefaultScheduler<Backer> {
    pub fn new(method: DefaultSchedulerMethod) -> Self {
        let bucketing_manager: Box<dyn BucketingManager<_>> = match method {
            DefaultSchedulerMethod::Fixed(_) => Box::new(FixedBucketingManager),
        };
        Self {
            running: Vec::new(),
            waiting: Backer::new(),
            method,
            bucketing_manager,
        }
    }

    /// Move the sequences into buckets, and run the ones with the shortest lengths.
    /// The others are moved to the waiting list (retaining high priority due to start time),
    /// without a state modification.
    fn bucket_and_waitlist_seqs(&mut self, running: Vec<Sequence>) -> Vec<Sequence> {
        let waiting = std::mem::take(&mut self.waiting);
        let BucketedSeqs { running, waiting } = self
            .bucketing_manager
            .bucket_and_waitlist_seqs_waiting(running, waiting, true);
        self.waiting = waiting;
        running
    }

    /// Schedule all sequences based on their state and the available space.
    pub fn schedule(&mut self, logger: &IntervalLogger) -> DefaultSchedulerOutput<'_> {
        // Filter out all done sequences
        let running = std::mem::take(&mut self.running);
        let mut waiting = std::mem::take(&mut self.waiting);
        let mut running = running
            .into_iter()
            .filter(|seq| seq.is_running())
            .collect::<Vec<_>>();

        match (waiting.len(), running.len()) {
            (0, 0) => {
                self.running = running;
                logger.set_num_running(self.running.len());
                logger.set_num_waiting(self.waiting.len());
                return DefaultSchedulerOutput {
                    prompt: vec![].into(),
                    completion: vec![].into(),
                };
            }
            (_, 0) => {
                for seq in waiting.into_iter() {
                    seq.set_state(SequenceState::RunningPrompt);
                    self.running.push(seq);
                }
                self.waiting = Backer::new();
                let running = std::mem::take(&mut self.running);
                self.running = self.bucket_and_waitlist_seqs(running);
                logger.set_num_running(self.running.len());
                logger.set_num_waiting(self.waiting.len());
                return DefaultSchedulerOutput {
                    prompt: self.running.iter_mut().collect::<Vec<_>>().into(),
                    completion: vec![].into(),
                };
            }
            (0, _) => {
                self.running = self.bucket_and_waitlist_seqs(running);
                if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
                    self.running
                        .iter_mut()
                        .for_each(|seq| seq.set_state(SequenceState::Done(StopReason::Canceled)));
                    TERMINATE_ALL_NEXT_STEP.store(false, Ordering::SeqCst);
                }
                logger.set_num_running(self.running.len());
                logger.set_num_waiting(self.waiting.len());
                return DefaultSchedulerOutput {
                    prompt: vec![].into(),
                    completion: self.running.iter_mut().collect::<Vec<_>>().into(),
                };
            }
            _ => {}
        }

        // Sort the waiting seqs
        waiting.sort_ascending_ids();

        // If the waiting sequence will fit, add it. Otherwise remove it
        let mut new_waiting = Backer::new();
        for seq in waiting.into_iter() {
            if self.sequence_fits(&running, &seq) {
                if seq.is_waiting() {
                    seq.set_state(SequenceState::RunningPrompt);
                }
                running.push(seq);
            } else {
                new_waiting.add(seq);
            }
        }

        let BucketedSeqs {
            running,
            waiting: new_waiting,
        } = self
            .bucketing_manager
            .bucket_and_waitlist_seqs_waiting(running, new_waiting, false);

        self.running = running;
        self.waiting = new_waiting;

        logger.set_num_running(self.running.len());
        logger.set_num_waiting(self.waiting.len());

        let mut completion = Vec::new();
        let mut prompt = Vec::new();
        for seq in &mut self.running {
            if seq.is_completion() {
                completion.push(seq);
            } else {
                prompt.push(seq);
            }
        }

        DefaultSchedulerOutput {
            completion: completion.into(),
            prompt: prompt.into(),
        }
    }

    fn sequence_fits(&self, running: &[Sequence], _seq: &Sequence) -> bool {
        match &self.method {
            DefaultSchedulerMethod::Fixed(n) => (running.len() + 1) <= (*n).into(),
        }
    }
}

impl Scheduler for DefaultScheduler<VecDeque<Sequence>> {
    fn schedule(&mut self, logger: &IntervalLogger) -> SchedulerOutput<'_> {
        SchedulerOutput::DefaultScheduler {
            output: self.schedule(logger),
        }
    }
    fn waiting_len(&self) -> usize {
        self.waiting.len()
    }
    fn running_len(&self) -> usize {
        self.running.len()
    }
    fn add_seq(&mut self, seq: Sequence) {
        if seq.is_running() {
            // prefill case
            self.running.push(seq);
        } else {
            self.waiting.add(seq);
        }
    }
    fn block_size(&self) -> Option<usize> {
        None
    }
    fn free_finished_sequence_groups(&mut self) {
        // Remove finished sequences
        self.running.retain(|seq| !seq.is_finished_paged_attn());
    }
    fn get_finished_mamba_indices(&self) -> Vec<usize> {
        self.running
            .iter()
            .filter(|seq| seq.is_finished_paged_attn())
            .filter_map(|seq| seq.mamba_state_idx())
            .collect()
    }
    fn kv_cache_manager(&self) -> Option<Arc<tokio::sync::Mutex<KVCacheManager>>> {
        None
    }
    fn set_prefix_caching_enabled(&mut self, _enabled: bool) {
        // DefaultScheduler doesn't use PagedAttention prefix caching
    }
}
