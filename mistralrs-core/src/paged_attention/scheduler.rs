//! The Scheduler uses a BlockEngine to schedule and automatically batch sequences. The
//! primary method `schedule` returns the batched sequences as inputs, as well as the
//! operations to be executed on the cache by the CacheEngine.

type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

use std::{
    collections::{HashMap, VecDeque},
    sync::{atomic::Ordering, Arc, Mutex},
};

use tracing::{info, warn};

use crate::{
    engine::IntervalLogger,
    get_mut_arcmutex,
    paged_attention::BlockEngine,
    scheduler::{Scheduler, SchedulerOutput},
    sequence::{Sequence, SequenceState, StopReason},
    TERMINATE_ALL_NEXT_STEP,
};

use super::{block_engine::AllocStatus, BlockEngineSequence, BlockTables, CacheConfig};

/// Bucket key: (sequence length, has_images && is_prompt, token_offset)
/// We bucket sequences by these criteria to ensure all sequences in a batch have the same
/// length, avoiding padding issues with flash attention varlen.
type BucketKey = (usize, bool, usize);

/// Allow sequences to wait for 64 scheduling passes before warning of deprivation.
const WAITING_TIMEOUT: usize = 64;

pub struct PagedAttentionSchedulerOutput {
    /// Either ALL prompt or ALL completion.
    pub scheduled: Vec<Arc<Mutex<Sequence>>>,
    pub blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
}

pub struct PagedAttentionSchedulerConfig {
    pub max_num_seqs: usize,
}

pub struct PagedAttentionScheduler {
    waiting: VecDeque<Arc<Mutex<Sequence>>>,
    running: VecDeque<Arc<Mutex<Sequence>>>,
    config: PagedAttentionSchedulerConfig,
    pub block_engine: Arc<tokio::sync::Mutex<BlockEngine>>,
    block_size: usize,
    prefix_caching_enabled: bool,
}

impl PagedAttentionScheduler {
    pub fn new(config: PagedAttentionSchedulerConfig, cache_config: CacheConfig) -> Self {
        // Default to enabled, Engine::new will call set_prefix_caching_enabled
        // based on the global no_prefix_cache flag
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            block_engine: Arc::new(tokio::sync::Mutex::new(BlockEngine::new(
                cache_config.block_size,
                cache_config.num_gpu_blocks,
                true, // Default enabled, will be configured by Engine
            ))),
            block_size: cache_config.block_size,
            config,
            prefix_caching_enabled: true,
        }
    }

    /// Set whether prefix caching is enabled. This also updates the block engine.
    pub fn set_prefix_caching_enabled_sync(&mut self, enabled: bool) {
        self.prefix_caching_enabled = enabled;
        if enabled {
            info!("Prefix caching enabled (block-level, PagedAttention). Expect higher multi-turn throughput for both text and multimodal.");
        }
        // Update the block engine - we need to block on the async mutex
        // This is called once at startup, so blocking is acceptable
        let block_engine = self.block_engine.clone();
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                block_engine
                    .lock()
                    .await
                    .set_prefix_caching_enabled(enabled);
            });
        });
    }

    /// Bucket sequences by (length, has_images && is_prompt, token_offset).
    /// Returns the bucket with the shortest sequence length; sequences from other buckets
    /// are preempted (blocks freed, state set to Waiting, added to waiting queue).
    ///
    /// This ensures all sequences in a batch have the same length, which is required for
    /// correct flash attention varlen operation (avoiding soundness issues with padding).
    ///
    /// Also removes preempted sequences from self.running.
    fn bucket_and_preempt_sequences(
        &mut self,
        sequences: VecDeque<Arc<Mutex<Sequence>>>,
    ) -> VecDeque<Arc<Mutex<Sequence>>> {
        if sequences.len() <= 1 {
            return sequences;
        }

        let mut buckets: HashMap<BucketKey, VecDeque<Arc<Mutex<Sequence>>>> = HashMap::new();

        for seq in sequences {
            let seq_guard = get_mut_arcmutex!(seq);
            let key: BucketKey = (
                seq_guard.len(),
                seq_guard.images().is_some() && seq_guard.is_prompt(),
                seq_guard.token_offset(),
            );
            drop(seq_guard);

            buckets.entry(key).or_default().push_back(seq);
        }

        if buckets.len() == 1 {
            // All sequences are in the same bucket, return them all
            return buckets.into_values().next().unwrap();
        }

        // Find the bucket with the shortest sequence length
        let min_key = *buckets
            .keys()
            .min_by_key(|(len, _, _)| *len)
            .expect("No sequence buckets");

        let selected = buckets.remove(&min_key).unwrap();

        // Collect IDs of sequences to preempt
        let mut ids_to_preempt = Vec::new();

        // Preempt sequences from other buckets (free blocks, set state to Waiting, add to waiting)
        for (_, seqs) in buckets {
            for seq in seqs.into_iter().rev() {
                ids_to_preempt.push(get_mut_arcmutex!(seq).get_id());
                self._preempt_by_recompute(seq);
            }
        }

        // Remove preempted sequences from self.running
        self.running
            .retain(|seq| !ids_to_preempt.contains(&get_mut_arcmutex!(seq).get_id()));

        selected
    }

    pub fn schedule(&mut self, logger: &IntervalLogger) -> PagedAttentionSchedulerOutput {
        let mut scheduled: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        let mut for_waiting_again: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        let mut did_ignore = false;
        while !self.waiting.is_empty() {
            let seq = self.waiting.front().unwrap().clone();

            if self.running.len() >= self.config.max_num_seqs {
                break;
            }

            let can_allocate =
                get_mut_arcmutex!(self.block_engine).can_allocate(&mut *get_mut_arcmutex!(seq));
            match can_allocate {
                AllocStatus::Later { waitlisted_count } => {
                    if waitlisted_count > WAITING_TIMEOUT {
                        if let Some(seq_to_preempt) = self.running.pop_back() {
                            self._preempt_by_recompute(seq_to_preempt);
                            if !matches!(
                                get_mut_arcmutex!(self.block_engine)
                                    .can_allocate(&mut *get_mut_arcmutex!(seq)),
                                AllocStatus::Ok
                            ) {
                                let id = *get_mut_arcmutex!(seq).id();
                                let len = get_mut_arcmutex!(seq).get_toks().len();
                                warn!(
                                    "Sequence {id} with length of {len} tokens still exceeds KV cache size \
                                     even after evicting another sequence.",
                                );
                                get_mut_arcmutex!(seq).set_state(SequenceState::FinishedIgnored);
                                did_ignore = true;
                            }
                        } else {
                            let id = *get_mut_arcmutex!(seq).id();
                            let len = get_mut_arcmutex!(seq).get_toks().len();
                            warn!(
                                "Sequence {id} with length of {len} tokens is too long and exceeds KV cache size. \
                                 To fix, increase the maximum sequence length for the KV cache, for example with \
                                 `--max-seq-len`/ `max_seq_len` in automatic device mapping parameters.",
                            );
                            get_mut_arcmutex!(seq).set_state(SequenceState::FinishedIgnored);
                            did_ignore = true;
                        }
                    } else {
                        break;
                    }
                }
                AllocStatus::Impossible => {
                    let id = *get_mut_arcmutex!(seq).id();
                    let len = get_mut_arcmutex!(seq).get_toks().len();
                    warn!(
                        "Sequence {id} with length of {len} tokens is too long and exceeds KV cache size. To fix, increase the maximum sequence length for the KV cache, for example with `--max-seq-len`/ `max_seq_len` in automatic device mapping parameters.",
                    );
                    get_mut_arcmutex!(seq).set_state(SequenceState::FinishedIgnored);
                    did_ignore = true;
                }
                _ => {}
            }

            let new_seq_has_images = get_mut_arcmutex!(seq).has_images();
            if !scheduled.is_empty()
                && get_mut_arcmutex!(scheduled[0]).has_images() != new_seq_has_images
            {
                let seq = self.waiting.pop_front().unwrap();
                for_waiting_again.push_back(seq.clone());
                continue;
            }
            if !did_ignore {
                get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
                let mut seq_handle = get_mut_arcmutex!(seq);
                self._allocate(&mut seq_handle);
                // Check for prefix cache hit and report to logger
                let seq_id = seq_handle.get_id();
                if get_mut_arcmutex!(self.block_engine).last_allocate_had_cache_hit(seq_id) > 0 {
                    logger.add_prefix_cache_hit();
                }
            }

            let seq = self.waiting.pop_front().unwrap();
            self.running.push_back(seq.clone());
            if !did_ignore {
                scheduled.push_back(seq);
            }
        }
        self.waiting.extend(for_waiting_again);

        if !scheduled.is_empty() || did_ignore {
            // Bucket scheduled prompts by sequence length to ensure all sequences in a batch
            // have the same length (required for correct flash attention varlen operation).
            let scheduled = self.bucket_and_preempt_sequences(scheduled);

            logger.set_num_running(self.running.len());
            logger.set_num_waiting(self.waiting.len());

            return PagedAttentionSchedulerOutput {
                scheduled: scheduled.into_iter().collect(),
                blocks_to_copy: HashMap::new(),
            };
        }

        let mut blocks_to_copy = HashMap::new();

        // Reserve token slots for the running sequence groups, preempting the lowest (earliest) first.
        // Preempt lowest priority sequences that are in the running queue, forming a
        // new running queue that has the actually running sequences. Remember the preempted
        // sequences, which will be put into the waiting or swapped out state depending on
        // the preemption method (recompute or swap, respectively).

        // Sorts by creation time, in descending order so that earliest are latest (first come first serve).
        self.sort_running_by_priority_fcfs();

        let mut running: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        while !self.running.is_empty() {
            let seq = self.running.pop_front().unwrap();
            let mut finished_with_break = false;
            while !get_mut_arcmutex!(self.block_engine)
                .can_append_token_to_seq(&*get_mut_arcmutex!(seq))
            {
                // If we cannot, now we need to preempt some seqs
                if !self.running.is_empty() {
                    // There is something to preempt.
                    let seq_to_preempt = self.running.pop_back().unwrap();
                    self._preempt(seq_to_preempt);
                } else {
                    // Nothing to preempt, preempt ourselves. Also, do not bother looking at anything else.
                    self._preempt(seq.clone());
                    finished_with_break = true;
                    break;
                }
            }
            if !finished_with_break {
                {
                    // If we need to, append physical blocks for a new token. We do not need to if there is enough space.
                    // If we just got preempted, there is no reason to allocate
                    let seq_handle = get_mut_arcmutex!(seq);
                    self._append_token_slot_to_seq(&seq_handle, &mut blocks_to_copy);
                }
                let new_seq_has_images = get_mut_arcmutex!(seq).has_images();
                // Only add it if has_images matches either current or there are none.
                if running.is_empty()
                    || get_mut_arcmutex!(running[0]).has_images() == new_seq_has_images
                {
                    running.push_back(seq);
                } else {
                    self.running.push_back(seq);
                }
            }
        }
        self.running = running;

        // Bucket running completions by sequence length to ensure all sequences in a batch
        // have the same length (required for correct flash attention varlen operation).
        let running_for_bucket = std::mem::take(&mut self.running);
        let bucketed = self.bucket_and_preempt_sequences(running_for_bucket);
        self.running = bucketed;

        self.running
            .iter()
            .for_each(|seq| get_mut_arcmutex!(seq).set_state(SequenceState::RunningCompletion));

        if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
            self.running.iter().for_each(|seq| {
                get_mut_arcmutex!(seq).set_state(SequenceState::Done(StopReason::Canceled))
            });
            TERMINATE_ALL_NEXT_STEP.store(false, Ordering::SeqCst);
        }

        logger.set_num_running(self.running.len());
        logger.set_num_waiting(self.waiting.len());

        PagedAttentionSchedulerOutput {
            scheduled: self.running.clone().into_iter().collect(),
            blocks_to_copy,
        }
    }

    pub fn free_finished_sequence_groups(&mut self) {
        let mut to_free: Vec<(usize, Vec<super::LogicalTokenBlock>)> = Vec::new();
        self.running.retain(|seq| {
            if get_mut_arcmutex!(seq).is_finished_paged_attn() {
                let seq_guard = get_mut_arcmutex!(seq);
                let id = seq_guard.get_id();
                // Get logical blocks for caching (clone them since we're dropping the lock)
                let logical_blocks = seq_guard.logical_token_blocks().to_vec();
                drop(seq_guard);
                to_free.push((id, logical_blocks));
                false
            } else {
                true
            }
        });

        for (id, logical_blocks) in to_free {
            self._free_with_caching(id, Some(&logical_blocks));
        }
    }
}

impl PagedAttentionScheduler {
    #[allow(dead_code)]
    fn remove_seq(&mut self, seq_id: usize) -> Arc<Mutex<Sequence>> {
        // Remove it if it is in waiting
        if let Some(idx) = self
            .waiting
            .iter()
            .position(|other| get_mut_arcmutex!(other).get_id() == seq_id)
        {
            return self.waiting.remove(idx).unwrap();
        };
        // Remove it if it is in running
        if let Some(idx) = self
            .running
            .iter()
            .position(|other| get_mut_arcmutex!(other).get_id() == seq_id)
        {
            return self.running.remove(idx).unwrap();
        };
        panic!("Attempted to remove sequence id {seq_id} but it is not running or waiting.");
    }

    fn _append_token_slot_to_seq(
        &mut self,
        seq: &Sequence,
        blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) {
        let op = get_mut_arcmutex!(self.block_engine).append_token_slot_to_seq(seq);
        if let Some((src_block, dst_block)) = op {
            if let std::collections::hash_map::Entry::Vacant(e) = blocks_to_copy.entry(src_block) {
                e.insert(vec![dst_block]);
            } else {
                blocks_to_copy.get_mut(&src_block).unwrap().push(dst_block);
            }
        }
    }

    fn _abort_seq(&mut self, seq_id: usize) {
        let removed = self.remove_seq(seq_id);
        get_mut_arcmutex!(removed).set_state(SequenceState::FinishedAborted);
        self._free(seq_id);
    }

    /// Preempt sequences by dropping their cache and recomputing later.
    fn _preempt(&mut self, seq: Arc<Mutex<Sequence>>) {
        self._preempt_by_recompute(seq)
    }

    fn _preempt_by_recompute(&mut self, seq: Arc<Mutex<Sequence>>) {
        let seq_guard = get_mut_arcmutex!(seq);
        seq_guard.set_state(SequenceState::Waiting);
        let seq_id = seq_guard.get_id();
        // Get logical blocks for proper cache ref release
        let logical_blocks = seq_guard.logical_token_blocks().to_vec();
        drop(seq_guard);
        // Use free_with_caching to properly release prefix cache refs
        // (but don't add blocks to cache since we're preempting)
        self._free_for_preemption(seq_id, &logical_blocks);
        self.waiting.push_front(seq);
    }

    fn _allocate(&mut self, seq: &mut Sequence) {
        get_mut_arcmutex!(self.block_engine).allocate(seq)
    }

    fn _free(&mut self, seq_id: usize) {
        get_mut_arcmutex!(self.block_engine).free_sequence(seq_id);
    }

    fn _free_for_preemption(&mut self, seq_id: usize, logical_blocks: &[super::LogicalTokenBlock]) {
        get_mut_arcmutex!(self.block_engine).free_sequence_for_preemption(seq_id, logical_blocks);
    }

    fn _free_with_caching(
        &mut self,
        seq_id: usize,
        logical_blocks: Option<&[super::LogicalTokenBlock]>,
    ) {
        get_mut_arcmutex!(self.block_engine).free_sequence_with_caching(seq_id, logical_blocks);
    }

    fn sort_running_by_priority_fcfs(&mut self) {
        self.running
            .make_contiguous()
            .sort_by_key(|seq| get_mut_arcmutex!(seq).timestamp());
        self.running.make_contiguous().reverse();
    }
}

impl Scheduler for PagedAttentionScheduler {
    fn add_seq(&mut self, seq: Sequence) {
        self.waiting.push_back(Arc::new(Mutex::new(seq)));
    }
    fn schedule(&mut self, logger: &IntervalLogger) -> SchedulerOutput<'_> {
        SchedulerOutput::PagedAttention {
            output: self.schedule(logger),
        }
    }
    fn waiting_len(&self) -> usize {
        self.waiting.len()
    }
    fn running_len(&self) -> usize {
        self.running.len()
    }
    fn block_tables(&self) -> Option<BlockTables> {
        Some(get_mut_arcmutex!(self.block_engine).block_tables.clone())
    }
    fn block_size(&self) -> Option<usize> {
        Some(self.block_size)
    }
    fn free_finished_sequence_groups(&mut self) {
        self.free_finished_sequence_groups()
    }
    fn get_finished_mamba_indices(&self) -> Vec<usize> {
        self.running
            .iter()
            .filter(|seq| get_mut_arcmutex!(seq).is_finished_paged_attn())
            .filter_map(|seq| get_mut_arcmutex!(seq).mamba_state_idx())
            .collect()
    }
    fn block_engine(&self) -> Option<Arc<tokio::sync::Mutex<BlockEngine>>> {
        Some(self.block_engine.clone())
    }
    fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.set_prefix_caching_enabled_sync(enabled);
    }
}
