use std::{
    collections::{HashMap, VecDeque},
    num::NonZeroUsize,
    sync::{atomic::Ordering, Arc},
};

use crate::{
    engine::{IntervalLogger, TERMINATE_ALL_NEXT_STEP},
    paged_attention::KVCacheManager,
    scheduler::modality_signature,
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
        requires_uniform_media_batch: bool,
    ) -> BucketedSeqs<Backer>;
}

// (cache length, media bucket, sequence offset, raw request)
type BucketKey = (usize, u8, usize, Option<usize>);

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
        requires_uniform_media_batch: bool,
    ) -> BucketedSeqs<Backer> {
        // Now, get the sequences with the smallest sequence lengths, and allow them to catch up.
        let mut seq_buckets: HashMap<BucketKey, Vec<Sequence>> = HashMap::new();
        let mut seq_priorities: HashMap<BucketKey, f64> = HashMap::new();
        for seq in running {
            let len = seq.len();
            let media = if requires_uniform_media_batch {
                modality_signature(&seq)
            } else {
                u8::from(seq.images().is_some() && seq.is_prompt())
            };
            let key = (
                len,
                media,
                seq.token_offset(),
                seq.return_raw_logits.then_some(*seq.id()),
            );
            match seq_buckets.get_mut(&key) {
                Some(bucket) => {
                    if !discrete {
                        *seq_priorities.get_mut(&key).unwrap() += seq.compute_priority();
                    }
                    bucket.push(seq);
                }
                None => {
                    if !discrete {
                        seq_priorities.insert(key, seq.compute_priority());
                    }
                    seq_buckets.insert(key, vec![seq]);
                }
            }
        }
        let running = if seq_buckets.len() <= 1 {
            // Full steam ahead or have everything
            seq_buckets
                .into_values()
                .flatten()
                .map(|s| s.reset_urgency())
                .collect::<Vec<_>>()
        } else {
            // Set the min seqs to be the running ones, and the rest to be waiting (but their states are not changed!)
            // Allow the min seqs to catch up.
            let min = *seq_buckets
                .keys()
                .min_by_key(|(len, _, _, _)| *len)
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
    requires_uniform_media_batch: bool,
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
            requires_uniform_media_batch: false,
        }
    }

    /// Move the sequences into buckets, and run the ones with the shortest lengths.
    /// The others are moved to the waiting list (retaining high priority due to start time),
    /// without a state modification.
    fn bucket_and_waitlist_seqs(&mut self, running: Vec<Sequence>) -> Vec<Sequence> {
        let waiting = std::mem::take(&mut self.waiting);
        let BucketedSeqs { running, waiting } =
            self.bucketing_manager.bucket_and_waitlist_seqs_waiting(
                running,
                waiting,
                true,
                self.requires_uniform_media_batch,
            );
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
        } = self.bucketing_manager.bucket_and_waitlist_seqs_waiting(
            running,
            new_waiting,
            false,
            self.requires_uniform_media_batch,
        );

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
    fn schedule(
        &mut self,
        logger: &IntervalLogger,
        _prefix_validator: Option<&mut dyn crate::scheduler::PagedPrefixCacheValidator>,
    ) -> SchedulerOutput<'_> {
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
    fn get_finished_recurrent_indices(&self) -> Vec<usize> {
        self.running
            .iter()
            .filter(|seq| seq.is_finished_paged_attn())
            .filter_map(|seq| seq.recurrent_state_idx())
            .collect()
    }
    fn get_finished_sequence_ids(&self) -> Vec<usize> {
        self.running
            .iter()
            .filter(|seq| seq.is_finished_paged_attn())
            .map(|seq| *seq.id())
            .collect()
    }
    fn kv_cache_manager(&self) -> Option<Arc<tokio::sync::Mutex<KVCacheManager>>> {
        None
    }
    fn set_prefix_caching_enabled(&mut self, _enabled: bool) {
        // DefaultScheduler doesn't use PagedAttention prefix caching
    }
    fn set_requires_uniform_media_batch(&mut self, required: bool) {
        self.requires_uniform_media_batch = required;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        paged_attention::block_hash::{
            MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind,
        },
        sampler::Sampler,
        sequence::{SeqStepType, SequenceGroup, SequenceRecognizer},
    };
    use tokio::sync::{mpsc::channel, Mutex as TokioMutex};

    fn test_sequence(id: usize, input_images: Option<Vec<image::DynamicImage>>) -> Sequence {
        let (tx, _rx) = channel(1);
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            32,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let group = Arc::new(TokioMutex::new(SequenceGroup::new(1, false, true, None)));
        let seq = Sequence::new_waiting(
            vec![1; 4],
            "prompt".to_string(),
            id,
            id as u128,
            1,
            tx,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            input_images,
            None,
            None,
            Some(8),
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            false,
            vec![],
        );
        seq.set_state(SequenceState::RunningCompletion);
        seq
    }

    fn persistent_image_sequence(id: usize) -> Sequence {
        let mut seq = test_sequence(id, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
        seq.set_mm_features(vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 0,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }]);
        seq.multimodal.has_changed_prompt = true;
        assert_eq!(seq.take_images().unwrap().len(), 1);
        assert!(!seq.has_images());
        seq
    }

    #[test]
    fn uniform_media_gate_uses_persistent_features() {
        let mut manager = FixedBucketingManager;
        let bucketed = manager.bucket_and_waitlist_seqs_waiting(
            vec![test_sequence(0, None), persistent_image_sequence(1)],
            VecDeque::new(),
            true,
            true,
        );

        assert_eq!(bucketed.running.len(), 1);
        assert_eq!(bucketed.waiting.len(), 1);
    }

    #[test]
    fn mixed_media_batches_can_share_a_completion_bucket() {
        let mut manager = FixedBucketingManager;
        let bucketed = manager.bucket_and_waitlist_seqs_waiting(
            vec![test_sequence(0, None), persistent_image_sequence(1)],
            VecDeque::new(),
            true,
            false,
        );

        assert_eq!(bucketed.running.len(), 2);
        assert_eq!(bucketed.waiting.len(), 0);
    }

    #[test]
    fn raw_logits_requests_do_not_share_default_buckets() {
        let mut manager = FixedBucketingManager;
        let normal = test_sequence(0, None);
        let mut raw = test_sequence(1, None);
        raw.return_raw_logits = true;
        let bucketed = manager.bucket_and_waitlist_seqs_waiting(
            vec![normal, raw],
            VecDeque::new(),
            true,
            false,
        );

        assert_eq!(bucketed.running.len(), 1);
        assert_eq!(bucketed.waiting.len(), 1);
    }

    #[test]
    fn raw_logits_requests_are_singleton_default_buckets() {
        let mut manager = FixedBucketingManager;
        let mut first = test_sequence(0, None);
        first.return_raw_logits = true;
        let mut second = test_sequence(1, None);
        second.return_raw_logits = true;
        let bucketed = manager.bucket_and_waitlist_seqs_waiting(
            vec![first, second],
            VecDeque::new(),
            true,
            false,
        );

        assert_eq!(bucketed.running.len(), 1);
        assert_eq!(bucketed.waiting.len(), 1);
    }
}
