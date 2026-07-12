//! The Scheduler uses a KVCacheManager to schedule and automatically batch sequences.
//! The primary method `schedule` returns the batched sequences as inputs.

use std::{
    collections::{HashMap, VecDeque},
    sync::{atomic::Ordering, Arc, Mutex},
};

use tracing::{info, warn};

use crate::{
    engine::IntervalLogger,
    get_mut_arcmutex,
    paged_attention::{
        block_hash::{
            clamp_prefix_cache_hit_len, compute_block_hashes, compute_new_block_hashes, BlockHash,
            MultiModalFeature,
        },
        kv_cache_manager::KVCacheManager,
    },
    scheduler::{PagedPrefixCacheValidator, Scheduler, SchedulerOutput},
    sequence::{clamp_prefix_cache_len_for_mm_features, Sequence, SequenceState, StopReason},
    TERMINATE_ALL_NEXT_STEP,
};

use super::CacheConfig;

/// Bucket key: (sequence length, has_images, token_offset)
/// We bucket sequences by these criteria to ensure all sequences in a batch have the same
/// length, avoiding padding issues with flash attention varlen.
type BucketKey = (usize, bool, usize);
type SequenceQueue = VecDeque<Arc<Mutex<Sequence>>>;

/// Allow sequences to wait for 64 scheduling passes before warning of deprivation.
const WAITING_TIMEOUT: usize = 64;

/// (seq_id, tokens, mm_features, block_hash_revision, num_computed_tokens)
type SeqCacheInfo = (usize, Vec<u32>, Vec<MultiModalFeature>, u64, usize);

pub struct PagedAttentionSchedulerOutput {
    /// Either ALL prompt or ALL completion.
    pub scheduled: Vec<Arc<Mutex<Sequence>>>,
    /// Number of cached tokens per sequence (from prefix cache hits).
    /// Only populated for prompt scheduling when prefix caching is enabled.
    pub num_cached_tokens: Vec<usize>,
}

pub struct PagedAttentionSchedulerConfig {
    pub max_num_seqs: usize,
}

pub struct PagedAttentionScheduler {
    waiting: VecDeque<Arc<Mutex<Sequence>>>,
    running: VecDeque<Arc<Mutex<Sequence>>>,
    config: PagedAttentionSchedulerConfig,
    pub kv_cache_manager: Arc<tokio::sync::Mutex<KVCacheManager>>,
    block_size: usize,
    prefix_caching_enabled: bool,
    /// Block hashes per sequence for prefix caching.
    /// Computed incrementally as sequences grow.
    seq_block_hashes: HashMap<usize, Vec<BlockHash>>,
    seq_block_hash_revisions: HashMap<usize, u64>,
    /// Per-sequence waitlist counter for starvation detection.
    waiting_counts: HashMap<usize, usize>,
}

impl PagedAttentionScheduler {
    pub fn new(config: PagedAttentionSchedulerConfig, cache_config: CacheConfig) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            kv_cache_manager: Arc::new(tokio::sync::Mutex::new(KVCacheManager::new(
                cache_config.num_gpu_blocks,
                cache_config.block_size,
                true,
                cache_config.kv_cache_group_ids.clone(),
            ))),
            block_size: cache_config.block_size,
            config,
            prefix_caching_enabled: true,
            seq_block_hashes: HashMap::new(),
            seq_block_hash_revisions: HashMap::new(),
            waiting_counts: HashMap::new(),
        }
    }

    /// Set whether prefix caching is enabled. This also updates the KV cache manager.
    pub fn set_prefix_caching_enabled_sync(&mut self, enabled: bool) {
        self.prefix_caching_enabled = enabled;
        if enabled {
            info!("Prefix caching enabled (block-level, PagedAttention). Expect higher multi-turn throughput for both text and multimodal.");
        }
    }

    /// Compute or update block hashes for a sequence.
    ///
    /// `mm_features`: per-item multimodal feature positions. Each feature's content hash
    /// is included only in blocks whose token range overlaps with that feature's placeholder
    /// tokens, ensuring that adding a new image at the end of a conversation doesn't
    /// invalidate hashes for earlier (unchanged) blocks.
    fn ensure_block_hashes(
        &mut self,
        seq_id: usize,
        tokens: &[u32],
        mm_features: &[MultiModalFeature],
        revision: u64,
    ) {
        let known_revision = self.seq_block_hash_revisions.get(&seq_id).copied();
        let hashes = self.seq_block_hashes.entry(seq_id).or_default();
        if hashes.is_empty() || known_revision != Some(revision) {
            *hashes = compute_block_hashes(tokens, self.block_size, mm_features, &[]);
            self.seq_block_hash_revisions.insert(seq_id, revision);
        } else {
            let new = compute_new_block_hashes(tokens, self.block_size, hashes, mm_features, &[]);
            hashes.extend(new);
        }
    }

    fn bucket_key(seq: &Arc<Mutex<Sequence>>) -> BucketKey {
        let seq = get_mut_arcmutex!(seq);
        // Use effective length for prompts so sequences with different
        // prefix cache hits land in separate buckets. The causal offset
        // (seqlen_k - seqlen_q) is only correct when all Qs are the same.
        let effective_len = if seq.is_prompt() {
            seq.len().saturating_sub(seq.prefix_cache_len())
        } else {
            seq.len()
        };
        (effective_len, seq.has_images(), seq.token_offset())
    }

    fn selected_bucket_key(sequences: &VecDeque<Arc<Mutex<Sequence>>>) -> Option<BucketKey> {
        sequences
            .iter()
            .min_by_key(|seq| {
                let seq = get_mut_arcmutex!(seq);
                (seq.timestamp(), *seq.id())
            })
            .map(Self::bucket_key)
    }

    /// Partition sequences by (length, has_images, token_offset), selecting the bucket that
    /// contains the oldest sequence and deferring incompatible buckets without preemption.
    ///
    /// This ensures all sequences in a batch have the same length, which is required for
    /// correct flash attention varlen operation (avoiding soundness issues with padding).
    fn partition_compatible_sequences(sequences: SequenceQueue) -> (SequenceQueue, SequenceQueue) {
        let Some(selected_key) = Self::selected_bucket_key(&sequences) else {
            return (VecDeque::new(), VecDeque::new());
        };

        sequences
            .into_iter()
            .partition(|seq| Self::bucket_key(seq) == selected_key)
    }

    pub fn schedule(
        &mut self,
        logger: &IntervalLogger,
        mut prefix_validator: Option<&mut dyn PagedPrefixCacheValidator>,
    ) -> PagedAttentionSchedulerOutput {
        // Prompts admitted in an earlier step retain both their KV allocation and prompt
        // state while an incompatible prompt bucket runs. Schedule those before admitting
        // more work or entering completion scheduling.
        let pending_prompts: VecDeque<_> = self
            .running
            .iter()
            .filter(|seq| get_mut_arcmutex!(seq).is_prompt())
            .cloned()
            .collect();
        if !pending_prompts.is_empty() {
            let (scheduled, _) = Self::partition_compatible_sequences(pending_prompts);
            let num_cached_tokens = scheduled
                .iter()
                .map(|seq| get_mut_arcmutex!(seq).prefix_cache_len())
                .collect();

            logger.set_num_running(self.running.len());
            logger.set_num_waiting(self.waiting.len());

            return PagedAttentionSchedulerOutput {
                scheduled: scheduled.into_iter().collect(),
                num_cached_tokens,
            };
        }

        let mut scheduled: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        while !self.waiting.is_empty() {
            let mut did_ignore = false;
            let seq = self.waiting.front().unwrap().clone();

            if self.running.len() >= self.config.max_num_seqs {
                break;
            }

            let seq_guard = get_mut_arcmutex!(seq);
            let seq_id = *seq_guard.id();
            let tokens = seq_guard.get_toks().to_vec();
            let num_tokens = tokens.len();
            let mm_features = seq_guard.mm_features().to_vec();
            let block_hash_revision = seq_guard.block_hash_revision();
            drop(seq_guard);

            // Compute block hashes for prefix cache lookup
            self.ensure_block_hashes(seq_id, &tokens, &mm_features, block_hash_revision);
            let block_hashes = self
                .seq_block_hashes
                .get(&seq_id)
                .cloned()
                .unwrap_or_default();

            // Look up prefix cache hits
            let kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
            let mut computed = if self.prefix_caching_enabled {
                kv_mgr.get_computed_blocks(&block_hashes, num_tokens)
            } else {
                super::kv_cache_manager::ComputedBlocks {
                    block_ids: Vec::new(),
                    num_computed_tokens: 0,
                }
            };
            drop(kv_mgr);

            if computed.num_computed_tokens > 0 {
                if let Some(validator) = prefix_validator.as_deref_mut() {
                    let mut seq_guard = get_mut_arcmutex!(seq);
                    let valid_tokens = validator.validate_prefix_cache_hit(
                        &mut seq_guard,
                        &block_hashes,
                        computed.num_computed_tokens,
                        self.block_size,
                    );
                    drop(seq_guard);
                    if valid_tokens < computed.num_computed_tokens {
                        let valid_blocks = valid_tokens / self.block_size;
                        computed.block_ids.truncate(valid_blocks);
                        computed.num_computed_tokens = valid_blocks * self.block_size;
                    }
                }
            }

            let clamped = clamp_prefix_cache_len_for_mm_features(
                computed.num_computed_tokens,
                self.block_size,
                &mm_features,
            );
            let clamped = clamp_prefix_cache_hit_len(clamped, self.block_size, &mm_features);
            if clamped < computed.num_computed_tokens {
                computed.block_ids.truncate(clamped / self.block_size);
                computed.num_computed_tokens = clamped;
            }

            let num_computed = computed.num_computed_tokens;
            let computed_block_count = num_computed / self.block_size;
            let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
            let alloc_result = kv_mgr.allocate_slots(
                seq_id,
                num_tokens,
                &computed.block_ids[..computed_block_count],
            );
            drop(kv_mgr);

            match alloc_result {
                Some(_) => {
                    // Allocation succeeded
                    if num_computed > 0 {
                        logger.add_prefix_cache_hit();
                    }
                    // Reset waiting count on successful allocation
                    self.waiting_counts.remove(&seq_id);
                }
                None => {
                    // Not enough blocks, check starvation
                    let count = self.waiting_counts.entry(seq_id).or_insert(0);
                    *count += 1;

                    if *count > WAITING_TIMEOUT {
                        // Try to preempt a running sequence
                        if let Some(seq_to_preempt) = self.running.pop_back() {
                            self._preempt(seq_to_preempt);

                            // Retry allocation
                            let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                            let retry = kv_mgr.allocate_slots(
                                seq_id,
                                num_tokens,
                                &computed.block_ids[..computed_block_count],
                            );
                            drop(kv_mgr);

                            if retry.is_none() {
                                let id = seq_id;
                                warn!(
                                    "Sequence {id} with length of {num_tokens} tokens still exceeds KV cache size \
                                     even after evicting another sequence.",
                                );
                                get_mut_arcmutex!(seq).set_state(SequenceState::FinishedIgnored);
                                did_ignore = true;
                            } else {
                                self.waiting_counts.remove(&seq_id);
                            }
                        } else {
                            warn!(
                                "Sequence {seq_id} with length of {num_tokens} tokens is too long and exceeds KV cache size. \
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
            }

            if !did_ignore {
                let mut seq_guard = get_mut_arcmutex!(seq);
                seq_guard.set_state(SequenceState::RunningPrompt);
                seq_guard.set_prefix_cache_len(num_computed);
                seq_guard.set_num_computed_tokens(num_computed);
            }

            let seq = self.waiting.pop_front().unwrap();
            if did_ignore {
                // Sequence is terminal (FinishedIgnored), do NOT add to running queue.
                // Clean up associated state and free any allocated blocks.
                let seq_id = *get_mut_arcmutex!(seq).id();
                self.waiting_counts.remove(&seq_id);
                self.seq_block_hashes.remove(&seq_id);
                self.seq_block_hash_revisions.remove(&seq_id);
                let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                kv_mgr.free(seq_id);
                drop(kv_mgr);
                continue;
            }
            self.running.push_back(seq.clone());
            scheduled.push_back(seq);
        }

        if !scheduled.is_empty() {
            // Emit one compatible prompt bucket. Other admitted prompts remain in `running`
            // with their allocations and RunningPrompt state for a later scheduling step.
            let (scheduled, _) = Self::partition_compatible_sequences(scheduled);

            // Rebuild num_cached_tokens from the bucketed sequences.
            // prefix_cache_len was set per-sequence above, so this stays aligned
            // even after bucketing removes sequences from non-contiguous positions.
            let num_cached_tokens: Vec<usize> = scheduled
                .iter()
                .map(|seq| get_mut_arcmutex!(seq).prefix_cache_len())
                .collect();

            logger.set_num_running(self.running.len());
            logger.set_num_waiting(self.waiting.len());

            return PagedAttentionSchedulerOutput {
                scheduled: scheduled.into_iter().collect(),
                num_cached_tokens,
            };
        }

        // === Completion scheduling (decode) ===
        // Reserve token slots for running sequences, preempting lowest priority first.

        self.sort_running_by_priority_fcfs();

        let selected_bucket_key = Self::selected_bucket_key(&self.running);
        let mut running: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        let mut deferred_running: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        while !self.running.is_empty() {
            let seq = self.running.pop_front().unwrap();
            if Some(Self::bucket_key(&seq)) != selected_bucket_key {
                deferred_running.push_back(seq);
                continue;
            }
            let mut finished_with_break = false;

            let seq_guard = get_mut_arcmutex!(seq);
            let seq_id = *seq_guard.id();
            let staged_speculative = seq_guard.active_staged_speculative_len();
            let num_tokens = if staged_speculative > 0 {
                seq_guard.len() + staged_speculative
            } else if seq_guard.num_uncomputed_tokens() > 0 {
                seq_guard.len()
            } else {
                seq_guard.len() + 1 // +1 for the new token to be generated
            };
            drop(seq_guard);

            // Try to allocate for the new token
            loop {
                let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                if kv_mgr.allocate_slots(seq_id, num_tokens, &[]).is_some() {
                    break;
                }
                drop(kv_mgr);
                if !self.running.is_empty() {
                    let seq_to_preempt = self.running.pop_back().unwrap();
                    self._preempt(seq_to_preempt);
                } else {
                    self._preempt(seq.clone());
                    finished_with_break = true;
                    break;
                }
            }

            if !finished_with_break {
                running.push_back(seq);
            }
        }
        // Length, modality, and offset incompatibility are scheduling constraints, not memory
        // pressure: keep deferred sequences allocated and runnable instead of freeing and
        // re-prefilling them. They also do not reserve an extra decode slot until selected.
        let scheduled_running = running;
        self.running = scheduled_running.clone();
        self.running.extend(deferred_running);

        if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
            self.running.iter().for_each(|seq| {
                get_mut_arcmutex!(seq).set_state(SequenceState::Done(StopReason::Canceled))
            });
            TERMINATE_ALL_NEXT_STEP.store(false, Ordering::SeqCst);
        }

        // Eagerly cache any newly-full blocks so other requests can hit the prefix cache
        // sooner, rather than waiting until finish/preempt. cache_blocks is idempotent.
        if self.prefix_caching_enabled {
            // Collect sequence info first to avoid borrow conflict with self.ensure_block_hashes
            let seq_infos: Vec<SeqCacheInfo> = self
                .running
                .iter()
                .map(|seq| {
                    let seq_guard = get_mut_arcmutex!(seq);
                    let seq_id = *seq_guard.id();
                    let tokens = seq_guard.get_toks().to_vec();
                    let mm_features = seq_guard.mm_features().to_vec();
                    let block_hash_revision = seq_guard.block_hash_revision();
                    let num_computed_tokens = seq_guard.num_computed_tokens();
                    (
                        seq_id,
                        tokens,
                        mm_features,
                        block_hash_revision,
                        num_computed_tokens,
                    )
                })
                .collect();

            for (seq_id, tokens, mm_features, block_hash_revision, num_computed_tokens) in
                &seq_infos
            {
                self.ensure_block_hashes(*seq_id, tokens, mm_features, *block_hash_revision);
                if let Some(block_hashes) = self.seq_block_hashes.get(seq_id).cloned() {
                    let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                    kv_mgr.cache_blocks(*seq_id, &block_hashes, *num_computed_tokens);
                }
            }
        }

        logger.set_num_running(self.running.len());
        logger.set_num_waiting(self.waiting.len());

        PagedAttentionSchedulerOutput {
            scheduled: scheduled_running.into_iter().collect(),
            num_cached_tokens: Vec::new(), // No prefix cache for completion
        }
    }

    pub fn free_finished_sequence_groups(&mut self) {
        // Collect finished sequence info before modifying self.running
        let mut finished: Vec<SeqCacheInfo> = Vec::new();
        let mut cacheable_finished: Vec<SeqCacheInfo> = Vec::new();
        for seq in self.running.iter() {
            let seq_guard = get_mut_arcmutex!(seq);
            if seq_guard.is_finished_paged_attn() {
                let id = *seq_guard.id();
                let tokens = seq_guard.get_toks().to_vec();
                let mm_features = seq_guard.mm_features().to_vec();
                let block_hash_revision = seq_guard.block_hash_revision();
                let num_computed_tokens = seq_guard.num_computed_tokens();
                let info = (
                    id,
                    tokens,
                    mm_features,
                    block_hash_revision,
                    num_computed_tokens,
                );
                if !matches!(seq_guard.getstate(), SequenceState::Error) {
                    cacheable_finished.push(info.clone());
                }
                finished.push(info);
            }
        }

        // Remove finished sequences from running
        self.running
            .retain(|seq| !get_mut_arcmutex!(seq).is_finished_paged_attn());

        // Cache and free blocks for finished sequences
        if self.prefix_caching_enabled {
            for (id, tokens, mm_features, block_hash_revision, num_computed_tokens) in
                &cacheable_finished
            {
                self.ensure_block_hashes(*id, tokens, mm_features, *block_hash_revision);
                let block_hashes = self.seq_block_hashes.get(id).cloned().unwrap_or_default();
                let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                kv_mgr.cache_blocks(*id, &block_hashes, *num_computed_tokens);
                drop(kv_mgr);
            }
        }

        let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
        for (id, _, _, _, _) in finished {
            kv_mgr.free(id);
            self.seq_block_hashes.remove(&id);
            self.seq_block_hash_revisions.remove(&id);
            self.waiting_counts.remove(&id);
        }
    }
}

impl PagedAttentionScheduler {
    fn _preempt(&mut self, seq: Arc<Mutex<Sequence>>) {
        let mut seq_guard = get_mut_arcmutex!(seq);
        // Don't resurrect sequences that are already in a terminal state
        if seq_guard.is_finished_paged_attn() {
            return;
        }
        seq_guard.set_state(SequenceState::Waiting);
        seq_guard.set_prefix_cache_len(0);
        seq_guard.clear_staged_speculative_tokens();
        let seq_id = *seq_guard.id();
        let tokens = seq_guard.get_toks().to_vec();
        let mm_features = seq_guard.mm_features().to_vec();
        let block_hash_revision = seq_guard.block_hash_revision();
        let num_computed_tokens = seq_guard.num_computed_tokens();
        drop(seq_guard);

        // Ensure block hashes are up-to-date before freeing
        self.ensure_block_hashes(seq_id, &tokens, &mm_features, block_hash_revision);
        let block_hashes = self
            .seq_block_hashes
            .get(&seq_id)
            .cloned()
            .unwrap_or_default();

        // Cache all full blocks and free, blocks stay in cache for LRU reuse
        let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
        if self.prefix_caching_enabled {
            kv_mgr.cache_blocks(seq_id, &block_hashes, num_computed_tokens);
        }
        kv_mgr.free(seq_id);
        drop(kv_mgr);

        self.waiting.push_front(seq);
    }

    fn sort_running_by_priority_fcfs(&mut self) {
        self.running.make_contiguous().sort_by_key(|seq| {
            let seq = get_mut_arcmutex!(seq);
            (seq.timestamp(), *seq.id())
        });
    }
}

impl Scheduler for PagedAttentionScheduler {
    fn add_seq(&mut self, seq: Sequence) {
        self.waiting.push_back(Arc::new(Mutex::new(seq)));
    }
    fn schedule(
        &mut self,
        logger: &IntervalLogger,
        prefix_validator: Option<&mut dyn PagedPrefixCacheValidator>,
    ) -> SchedulerOutput<'_> {
        SchedulerOutput::PagedAttention {
            output: self.schedule(logger, prefix_validator),
        }
    }
    fn waiting_len(&self) -> usize {
        self.waiting.len()
    }
    fn running_len(&self) -> usize {
        self.running.len()
    }
    fn block_size(&self) -> Option<usize> {
        Some(self.block_size)
    }
    fn free_finished_sequence_groups(&mut self) {
        self.free_finished_sequence_groups()
    }
    fn get_finished_recurrent_indices(&self) -> Vec<usize> {
        self.running
            .iter()
            .filter(|seq| get_mut_arcmutex!(seq).is_finished_paged_attn())
            .filter_map(|seq| get_mut_arcmutex!(seq).recurrent_state_idx())
            .collect()
    }
    fn kv_cache_manager(&self) -> Option<Arc<tokio::sync::Mutex<KVCacheManager>>> {
        Some(self.kv_cache_manager.clone())
    }
    fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.set_prefix_caching_enabled_sync(enabled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sampler::Sampler,
        sequence::{SeqStepType, SequenceGroup, SequenceRecognizer},
    };
    use std::collections::HashMap;
    use tokio::sync::{mpsc::channel, Mutex as TokioMutex};

    fn test_sequence_with_images(
        id: usize,
        timestamp: u128,
        len: usize,
        has_images: bool,
        state: SequenceState,
    ) -> Arc<Mutex<Sequence>> {
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
        .expect("test sampler must build");
        let group = Arc::new(TokioMutex::new(SequenceGroup::new(1, false, true, None)));
        let input_images = has_images.then(|| vec![image::DynamicImage::new_rgb8(1, 1)]);
        let mut sequence = Sequence::new_waiting(
            vec![u32::try_from(id).expect("test id must fit in u32"); len],
            "prompt".to_string(),
            id,
            timestamp,
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
            Some(4),
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        );
        if state == SequenceState::RunningCompletion {
            sequence.set_num_computed_tokens(len);
        }
        sequence.set_state(state);
        Arc::new(Mutex::new(sequence))
    }

    fn test_sequence(
        id: usize,
        timestamp: u128,
        len: usize,
        state: SequenceState,
    ) -> Arc<Mutex<Sequence>> {
        test_sequence_with_images(id, timestamp, len, false, state)
    }

    fn test_scheduler_with_blocks(num_gpu_blocks: usize) -> PagedAttentionScheduler {
        PagedAttentionScheduler::new(
            PagedAttentionSchedulerConfig { max_num_seqs: 4 },
            CacheConfig {
                block_size: 4,
                num_gpu_blocks,
                cache_type: Default::default(),
                kv_cache_group_ids: vec![0],
            },
        )
    }

    fn test_scheduler() -> PagedAttentionScheduler {
        test_scheduler_with_blocks(16)
    }

    fn allocate_sequence(scheduler: &mut PagedAttentionScheduler, sequence: &Arc<Mutex<Sequence>>) {
        let sequence = get_mut_arcmutex!(sequence);
        let id = *sequence.id();
        let len = sequence.len();
        drop(sequence);

        get_mut_arcmutex!(scheduler.kv_cache_manager)
            .allocate_slots(id, len, &[])
            .expect("test sequence allocation must succeed");
    }

    fn sequence_id(sequence: &Arc<Mutex<Sequence>>) -> usize {
        *get_mut_arcmutex!(sequence).id()
    }

    #[test]
    fn bucket_selection_uses_oldest_sequence_without_preemption() {
        let older_longer = test_sequence(1, 1, 8, SequenceState::RunningCompletion);
        let newer_shorter = test_sequence(2, 2, 4, SequenceState::RunningCompletion);

        let (selected, deferred) =
            PagedAttentionScheduler::partition_compatible_sequences(VecDeque::from([
                newer_shorter.clone(),
                older_longer.clone(),
            ]));

        assert_eq!(selected.len(), 1);
        assert_eq!(sequence_id(&selected[0]), 1);
        assert_eq!(deferred.len(), 1);
        assert_eq!(sequence_id(&deferred[0]), 2);
        assert_eq!(
            get_mut_arcmutex!(older_longer).getstate(),
            SequenceState::RunningCompletion
        );
        assert_eq!(
            get_mut_arcmutex!(newer_shorter).getstate(),
            SequenceState::RunningCompletion
        );
    }

    #[test]
    fn bucket_selection_breaks_equal_timestamps_by_sequence_id() {
        let lower_id_longer = test_sequence(1, 1, 8, SequenceState::RunningCompletion);
        let higher_id_shorter = test_sequence(2, 1, 4, SequenceState::RunningCompletion);

        let (selected, deferred) =
            PagedAttentionScheduler::partition_compatible_sequences(VecDeque::from([
                higher_id_shorter,
                lower_id_longer,
            ]));

        assert_eq!(selected.len(), 1);
        assert_eq!(sequence_id(&selected[0]), 1);
        assert_eq!(deferred.len(), 1);
        assert_eq!(sequence_id(&deferred[0]), 2);
    }

    #[test]
    fn prompt_buckets_keep_allocations_and_prompt_state() {
        let mut scheduler = test_scheduler();
        let older_longer = test_sequence(1, 1, 8, SequenceState::Waiting);
        let newer_shorter = test_sequence(2, 2, 4, SequenceState::Waiting);
        scheduler.waiting = VecDeque::from([older_longer.clone(), newer_shorter.clone()]);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let output = scheduler.schedule(&logger, None);

        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(sequence_id(&output.scheduled[0]), 1);
        assert_eq!(output.num_cached_tokens, vec![0]);
        assert!(scheduler.waiting.is_empty());
        assert_eq!(scheduler.running.len(), 2);
        assert_eq!(
            get_mut_arcmutex!(older_longer).getstate(),
            SequenceState::RunningPrompt
        );
        assert_eq!(
            get_mut_arcmutex!(newer_shorter).getstate(),
            SequenceState::RunningPrompt
        );

        {
            let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
            assert!(kv_mgr.has_request(1));
            assert!(kv_mgr.has_request(2));
            assert_eq!(kv_mgr.num_blocks_for_request(1), 2);
            assert_eq!(kv_mgr.num_blocks_for_request(2), 1);
            assert_eq!(kv_mgr.num_free_blocks(), 12);
        }

        // Complete only the prompt that the engine actually executed. The deferred prompt
        // must be emitted next without converting the completed prompt back to prefill.
        {
            let mut older_longer = get_mut_arcmutex!(older_longer);
            crate::engine::transition_executed_paged_prompt(&mut older_longer);
        }
        assert_eq!(
            get_mut_arcmutex!(older_longer).getstate(),
            SequenceState::RunningCompletion
        );
        let next_output = scheduler.schedule(&logger, None);

        assert_eq!(next_output.scheduled.len(), 1);
        assert_eq!(sequence_id(&next_output.scheduled[0]), 2);
        assert_eq!(
            get_mut_arcmutex!(older_longer).getstate(),
            SequenceState::RunningCompletion
        );
        assert_eq!(
            get_mut_arcmutex!(newer_shorter).getstate(),
            SequenceState::RunningPrompt
        );
        let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert_eq!(kv_mgr.num_blocks_for_request(1), 2);
        assert_eq!(kv_mgr.num_blocks_for_request(2), 1);
        assert_eq!(kv_mgr.num_free_blocks(), 12);
        drop(kv_mgr);

        // After the deferred prompt really runs, it transitions independently and the next
        // scheduler pass is decode, rather than another prompt replay.
        {
            let mut newer_shorter = get_mut_arcmutex!(newer_shorter);
            crate::engine::transition_executed_paged_prompt(&mut newer_shorter);
        }
        let completion_output = scheduler.schedule(&logger, None);
        assert_eq!(completion_output.scheduled.len(), 1);
        assert_eq!(sequence_id(&completion_output.scheduled[0]), 1);
        assert_eq!(
            get_mut_arcmutex!(newer_shorter).getstate(),
            SequenceState::RunningCompletion
        );
    }

    #[test]
    fn completion_length_mismatch_keeps_allocations_and_completion_state() {
        let mut scheduler = test_scheduler();
        let older_longer = test_sequence(1, 1, 8, SequenceState::RunningCompletion);
        let newer_shorter = test_sequence(2, 2, 4, SequenceState::RunningCompletion);
        allocate_sequence(&mut scheduler, &older_longer);
        allocate_sequence(&mut scheduler, &newer_shorter);
        scheduler.running = VecDeque::from([older_longer.clone(), newer_shorter.clone()]);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let output = scheduler.schedule(&logger, None);

        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(sequence_id(&output.scheduled[0]), 1);
        assert!(scheduler.waiting.is_empty());
        assert_eq!(scheduler.running.len(), 2);
        assert_eq!(
            get_mut_arcmutex!(older_longer).getstate(),
            SequenceState::RunningCompletion
        );
        assert_eq!(
            get_mut_arcmutex!(newer_shorter).getstate(),
            SequenceState::RunningCompletion
        );

        {
            let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
            assert_eq!(kv_mgr.num_blocks_for_request(1), 3);
            assert_eq!(kv_mgr.num_blocks_for_request(2), 1);
            assert_eq!(kv_mgr.num_free_blocks(), 11);
        }

        // Re-scheduling without growth must neither free nor allocate either request.
        let next_output = scheduler.schedule(&logger, None);
        assert_eq!(sequence_id(&next_output.scheduled[0]), 1);
        assert!(scheduler.waiting.is_empty());
        {
            let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
            assert_eq!(kv_mgr.num_blocks_for_request(1), 3);
            assert_eq!(kv_mgr.num_blocks_for_request(2), 1);
            assert_eq!(kv_mgr.num_free_blocks(), 11);
        }

        // Once the oldest bucket finishes, the deferred completion progresses in place.
        get_mut_arcmutex!(older_longer).set_state(SequenceState::Done(StopReason::Canceled));
        scheduler.free_finished_sequence_groups();
        let deferred_output = scheduler.schedule(&logger, None);
        assert_eq!(deferred_output.scheduled.len(), 1);
        assert_eq!(sequence_id(&deferred_output.scheduled[0]), 2);
        assert_eq!(
            get_mut_arcmutex!(newer_shorter).getstate(),
            SequenceState::RunningCompletion
        );
        let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(!kv_mgr.has_request(1));
        assert_eq!(kv_mgr.num_blocks_for_request(2), 2);
    }

    #[test]
    fn completion_modality_mismatch_does_not_reinsert_or_free() {
        let mut scheduler = test_scheduler();
        let text_sequence = test_sequence(1, 1, 4, SequenceState::RunningCompletion);
        let image_sequence =
            test_sequence_with_images(2, 2, 4, true, SequenceState::RunningCompletion);
        allocate_sequence(&mut scheduler, &text_sequence);
        allocate_sequence(&mut scheduler, &image_sequence);
        scheduler.running = VecDeque::from([text_sequence.clone(), image_sequence.clone()]);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let output = scheduler.schedule(&logger, None);

        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(sequence_id(&output.scheduled[0]), 1);
        assert!(scheduler.waiting.is_empty());
        assert_eq!(scheduler.running.len(), 2);
        assert_eq!(
            get_mut_arcmutex!(image_sequence).getstate(),
            SequenceState::RunningCompletion
        );
        let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr.has_request(1));
        assert!(kv_mgr.has_request(2));
        assert_eq!(kv_mgr.num_blocks_for_request(1), 2);
        assert_eq!(kv_mgr.num_blocks_for_request(2), 1);
    }

    #[test]
    fn true_memory_pressure_still_preempts() {
        // Four total blocks means three usable blocks after the null block.
        let mut scheduler = test_scheduler_with_blocks(4);
        let older = test_sequence(1, 1, 4, SequenceState::RunningCompletion);
        let newer = test_sequence(2, 2, 4, SequenceState::RunningCompletion);
        allocate_sequence(&mut scheduler, &older);
        allocate_sequence(&mut scheduler, &newer);
        scheduler.running = VecDeque::from([older.clone(), newer.clone()]);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let output = scheduler.schedule(&logger, None);

        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(sequence_id(&output.scheduled[0]), 1);
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(sequence_id(&scheduler.waiting[0]), 2);
        assert_eq!(
            get_mut_arcmutex!(older).getstate(),
            SequenceState::RunningCompletion
        );
        assert_eq!(get_mut_arcmutex!(newer).getstate(), SequenceState::Waiting);
        let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr.has_request(1));
        assert!(!kv_mgr.has_request(2));
        assert_eq!(kv_mgr.num_blocks_for_request(1), 2);
        assert_eq!(kv_mgr.num_free_blocks(), 1);
    }

    #[test]
    fn equal_timestamp_memory_pressure_preempts_higher_id() {
        // Three usable blocks are initially consumed. Extending the selected lower-ID
        // bucket must evict the higher-ID request even when the input queue is reversed.
        let mut scheduler = test_scheduler_with_blocks(4);
        let lower_id_longer = test_sequence(1, 1, 8, SequenceState::RunningCompletion);
        let higher_id_shorter = test_sequence(2, 1, 4, SequenceState::RunningCompletion);
        allocate_sequence(&mut scheduler, &lower_id_longer);
        allocate_sequence(&mut scheduler, &higher_id_shorter);
        scheduler.running = VecDeque::from([higher_id_shorter.clone(), lower_id_longer.clone()]);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let output = scheduler.schedule(&logger, None);

        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(sequence_id(&output.scheduled[0]), 1);
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(sequence_id(&scheduler.waiting[0]), 2);
        assert_eq!(
            get_mut_arcmutex!(lower_id_longer).getstate(),
            SequenceState::RunningCompletion
        );
        assert_eq!(
            get_mut_arcmutex!(higher_id_shorter).getstate(),
            SequenceState::Waiting
        );
        let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr.has_request(1));
        assert!(!kv_mgr.has_request(2));
        assert_eq!(kv_mgr.num_blocks_for_request(1), 3);
        assert_eq!(kv_mgr.num_free_blocks(), 0);
    }

    #[test]
    fn idle_scheduler_returns_an_empty_batch() {
        let mut scheduler = test_scheduler();
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

        let output = scheduler.schedule(&logger, None);

        assert!(output.scheduled.is_empty());
        assert!(output.num_cached_tokens.is_empty());
        assert!(scheduler.running.is_empty());
        assert!(scheduler.waiting.is_empty());
    }
}
