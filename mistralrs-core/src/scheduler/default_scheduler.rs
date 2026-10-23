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

const MAX_COHORT_CATCH_UP_STEPS: usize = 8;

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

#[derive(Default)]
struct FixedBucketingManager {
    catch_up_steps: usize,
}

impl FixedBucketingManager {
    fn select_bucket(
        &mut self,
        buckets: &HashMap<BucketKey, Vec<Sequence>>,
        priorities: &HashMap<BucketKey, f64>,
        discrete: bool,
    ) -> BucketKey {
        if discrete {
            self.catch_up_steps = 0;
            return *buckets.keys().min().expect("No sequence buckets.");
        }
        let priority_key = *priorities
            .iter()
            .max_by(|(key_a, a), (key_b, b)| a.total_cmp(b).then_with(|| key_b.cmp(key_a)))
            .expect("No sequence priorities.")
            .0;
        // Let nearby decode cohorts converge without indefinitely delaying the priority winner.
        if self.catch_up_steps < MAX_COHORT_CATCH_UP_STEPS
            && buckets[&priority_key].iter().all(Sequence::is_completion)
        {
            let catch_up = buckets
                .iter()
                .filter(|(key, seqs)| {
                    key.0 < priority_key.0
                        && priority_key.0 - key.0 <= MAX_COHORT_CATCH_UP_STEPS
                        && (key.1, key.2, key.3) == (priority_key.1, priority_key.2, priority_key.3)
                        && seqs.iter().all(Sequence::is_completion)
                })
                .map(|(key, _)| *key)
                .min();
            if let Some(key) = catch_up {
                self.catch_up_steps += 1;
                return key;
            }
        }
        self.catch_up_steps = 0;
        priority_key
    }
}

impl<Backer: FcfsBacker> BucketingManager<Backer> for FixedBucketingManager {
    fn bucket_and_waitlist_seqs_waiting(
        &mut self,
        running: Vec<Sequence>,
        mut waiting: Backer,
        discrete: bool,
        requires_uniform_media_batch: bool,
    ) -> BucketedSeqs<Backer> {
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
            self.catch_up_steps = 0;
            // Full steam ahead or have everything
            seq_buckets
                .into_values()
                .flatten()
                .map(|s| s.reset_urgency())
                .collect::<Vec<_>>()
        } else {
            let key = self.select_bucket(&seq_buckets, &seq_priorities, discrete);
            let highest_priority_seqs = seq_buckets
                .remove(&key)
                .unwrap()
                .into_iter()
                .map(|s| s.reset_urgency())
                .collect();
            for (_, seqs) in seq_buckets {
                for seq in seqs {
                    waiting.add(seq.add_urgency());
                }
            }
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
            DefaultSchedulerMethod::Fixed(_) => Box::<FixedBucketingManager>::default(),
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
                    if seq.is_waiting() {
                        seq.set_state(SequenceState::RunningPrompt);
                    }
                    self.running.push(seq);
                }
                self.waiting = Backer::new();
                let running = std::mem::take(&mut self.running);
                self.running = self.bucket_and_waitlist_seqs(running);
                logger.set_num_running(self.running.len());
                logger.set_num_waiting(self.waiting.len());
                let (completion, prompt): (Vec<_>, Vec<_>) =
                    self.running.iter_mut().partition(|seq| seq.is_completion());
                return DefaultSchedulerOutput {
                    prompt: prompt.into(),
                    completion: completion.into(),
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
    fn cancel_closed_response_groups(&mut self) {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .filter(|seq| seq.response_is_closed() && !seq.is_finished_paged_attn())
            .for_each(|seq| seq.set_state(SequenceState::Done(StopReason::Canceled)));
    }
    fn block_size(&self) -> Option<usize> {
        None
    }
    fn free_finished_sequence_groups(&mut self) {
        self.running.retain(|seq| !seq.is_finished_paged_attn());
        self.waiting.retain(|seq| !seq.is_finished_paged_attn());
    }
    fn get_finished_recurrent_slots(&self) -> Vec<(usize, usize)> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .filter(|seq| seq.is_finished_paged_attn())
            .filter_map(|seq| seq.recurrent_state_idx().map(|slot| (*seq.id(), slot)))
            .collect()
    }
    fn get_finished_sequence_ids(&self) -> Vec<usize> {
        self.running
            .iter()
            .chain(self.waiting.iter())
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
    use std::time::Duration;

    use crate::{
        paged_attention::block_hash::{
            MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind,
        },
        sampler::{Logprobs, Sampler},
        sequence::{SeqStepType, SequenceGroup, SequenceRecognizer},
    };
    use tokio::sync::{mpsc::channel, Mutex as TokioMutex};

    const TEST_PROMPT_LEN: usize = 4;
    const TEST_COHORT_GAP: usize = 3;
    const TEST_COHORT_SIZES: [usize; 2] = [2, 4];
    const TEST_LOG_INTERVAL: Duration = Duration::from_secs(60);
    const TEST_TOKEN: u32 = 1;

    fn test_sequence(id: usize, input_images: Option<Vec<image::DynamicImage>>) -> Sequence {
        test_sequence_with_len(id, TEST_PROMPT_LEN, input_images)
    }

    fn test_sequence_with_len(
        id: usize,
        len: usize,
        input_images: Option<Vec<image::DynamicImage>>,
    ) -> Sequence {
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
            vec![TEST_TOKEN; len],
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
            None,
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
        let mut manager = FixedBucketingManager::default();
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
        let mut manager = FixedBucketingManager::default();
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
        let mut manager = FixedBucketingManager::default();
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
        let mut manager = FixedBucketingManager::default();
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

    fn advance_decode(seq: &mut Sequence) {
        seq.add_token(
            Logprobs {
                token: TEST_TOKEN,
                logprob: 0.0,
                bytes: None,
                top_logprobs: None,
            },
            Vec::new(),
            None,
        );
    }

    #[test]
    fn nearby_equal_sized_decode_cohorts_converge() {
        let logger = IntervalLogger::new(TEST_LOG_INTERVAL, None);
        for cohort_size in TEST_COHORT_SIZES {
            for gap in [1, TEST_COHORT_GAP, MAX_COHORT_CATCH_UP_STEPS] {
                let mut scheduler = DefaultScheduler::new(DefaultSchedulerMethod::Fixed(
                    NonZeroUsize::new(cohort_size * 2).unwrap(),
                ));
                for id in 0..cohort_size * 2 {
                    let len = TEST_PROMPT_LEN + usize::from(id >= cohort_size) * gap;
                    scheduler.add_seq(test_sequence_with_len(id, len, None));
                }
                for step in 0..gap {
                    let mut output = scheduler.schedule(&logger);
                    assert!(output.prompt.is_empty());
                    assert_eq!(output.completion.len(), cohort_size);
                    for seq in output.completion.iter_mut() {
                        assert_eq!(seq.len(), TEST_PROMPT_LEN + step);
                        advance_decode(seq);
                    }
                }
                let output = scheduler.schedule(&logger);
                assert!(output.prompt.is_empty());
                assert_eq!(output.completion.len(), cohort_size * 2);
                assert!(output
                    .completion
                    .iter()
                    .all(|seq| seq.len() == TEST_PROMPT_LEN + gap));
            }
        }
    }

    #[test]
    fn repeated_shorter_arrivals_cannot_starve_the_priority_winner() {
        let mut manager = FixedBucketingManager::default();
        let mut older = test_sequence_with_len(0, TEST_PROMPT_LEN + 1, None);
        for _ in 0..MAX_COHORT_CATCH_UP_STEPS * 2 {
            older = older.add_urgency();
        }
        for step in 0..=MAX_COHORT_CATCH_UP_STEPS {
            let bucketed = manager.bucket_and_waitlist_seqs_waiting(
                vec![older, test_sequence(step + 1, None)],
                VecDeque::new(),
                false,
                false,
            );
            assert_eq!(bucketed.running.len(), 1);
            assert_eq!(bucketed.waiting.len(), 1);
            if step == MAX_COHORT_CATCH_UP_STEPS {
                assert_eq!(*bucketed.running[0].id(), 0);
                return;
            }
            assert_eq!(*bucketed.running[0].id(), step + 1);
            let mut waiting = bucketed.waiting;
            older = waiting.pop_front().unwrap();
            assert_eq!(*older.id(), 0);
        }
        unreachable!();
    }

    #[test]
    fn catch_up_preserves_phase_media_offset_and_raw_boundaries() {
        let longer = || test_sequence_with_len(0, TEST_PROMPT_LEN + TEST_COHORT_GAP, None);
        let short_prompt = test_sequence(1, None);
        short_prompt.set_state(SequenceState::RunningPrompt);
        let long_prompt = longer();
        long_prompt.set_state(SequenceState::RunningPrompt);
        let mut offset = test_sequence(1, None).prefill_v2_normal(
            vec![None],
            vec![TEST_TOKEN; TEST_PROMPT_LEN],
            1,
        );
        offset.reset_prefill_toks();
        offset.set_state(SequenceState::RunningCompletion);
        let mut raw = test_sequence(1, None);
        raw.return_raw_logits = true;
        let too_far =
            test_sequence_with_len(0, TEST_PROMPT_LEN + MAX_COHORT_CATCH_UP_STEPS + 1, None);
        for (short, long, uniform_media) in [
            (short_prompt, longer(), false),
            (test_sequence(1, None), long_prompt, false),
            (persistent_image_sequence(1), longer(), true),
            (offset, longer(), false),
            (raw, longer(), false),
            (test_sequence(1, None), too_far, false),
        ] {
            let mut manager = FixedBucketingManager::default();
            let bucketed = manager.bucket_and_waitlist_seqs_waiting(
                vec![short, long],
                VecDeque::new(),
                false,
                uniform_media,
            );
            assert_eq!(bucketed.running.len(), 1);
            assert_eq!(*bucketed.running[0].id(), 0);
            assert_eq!(bucketed.waiting.len(), 1);
        }
    }

    #[test]
    fn deferred_decode_resumes_without_reprefilling_its_cache() -> candle_core::Result<()> {
        let logger = IntervalLogger::new(TEST_LOG_INTERVAL, None);
        let mut scheduler: DefaultScheduler<VecDeque<Sequence>> =
            DefaultScheduler::new(DefaultSchedulerMethod::Fixed(NonZeroUsize::new(1).unwrap()));
        let finished = test_sequence(0, None);
        finished.set_state(SequenceState::Done(StopReason::Canceled));
        scheduler.running.push(finished);
        let mut deferred = test_sequence(1, None);
        let cache = candle_core::Tensor::zeros(
            (1, 1, TEST_PROMPT_LEN - 1, 1),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )?;
        deferred.cache()[0] = Some((cache.clone(), cache));
        scheduler.waiting.add(deferred);

        let mut output = scheduler.schedule(&logger);
        assert!(output.prompt.is_empty());
        assert_eq!(output.completion.len(), 1);
        let seq = &mut output.completion[0];
        assert_eq!(*seq.id(), 1);
        assert!(seq.is_completion());
        assert_eq!(seq.len(), TEST_PROMPT_LEN);
        assert_eq!(
            seq.cache()[0].as_ref().unwrap().0.dim(2)?,
            TEST_PROMPT_LEN - 1
        );
        Ok(())
    }

    #[test]
    fn fresh_and_prefix_cached_requests_still_prefill() {
        let logger = IntervalLogger::new(TEST_LOG_INTERVAL, None);
        let fresh = test_sequence(0, None);
        fresh.set_state(SequenceState::Waiting);
        let prefix = test_sequence(1, None).prefill_v2_normal(
            vec![None],
            vec![TEST_TOKEN; TEST_PROMPT_LEN],
            1,
        );
        for request in [fresh, prefix] {
            let expected_offset = request.token_offset();
            let mut scheduler =
                DefaultScheduler::new(DefaultSchedulerMethod::Fixed(NonZeroUsize::new(1).unwrap()));
            scheduler.add_seq(request);
            let output = scheduler.schedule(&logger);
            assert!(output.completion.is_empty());
            assert_eq!(output.prompt.len(), 1);
            assert!(output.prompt[0].is_prompt());
            assert_eq!(output.prompt[0].token_offset(), expected_offset);
        }
    }

    #[test]
    fn closed_responses_are_removed_from_both_default_queues() {
        let mut scheduler =
            DefaultScheduler::new(DefaultSchedulerMethod::Fixed(NonZeroUsize::new(2).unwrap()));
        let running = test_sequence(10, None);
        let waiting = test_sequence(20, None);
        waiting.set_state(SequenceState::Waiting);
        scheduler.add_seq(running);
        scheduler.add_seq(waiting);

        Scheduler::cancel_closed_response_groups(&mut scheduler);
        assert_eq!(
            Scheduler::get_finished_sequence_ids(&scheduler),
            vec![10, 20]
        );

        scheduler.free_finished_sequence_groups();
        assert!(scheduler.running.is_empty());
        assert!(scheduler.waiting.is_empty());
    }
}
