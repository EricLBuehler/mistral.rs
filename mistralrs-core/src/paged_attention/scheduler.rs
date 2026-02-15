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
            compute_block_hashes, compute_new_block_hashes, BlockHash, MultiModalFeature,
        },
        kv_cache_manager::KVCacheManager,
    },
    scheduler::{Scheduler, SchedulerOutput},
    sequence::{Sequence, SequenceState, StopReason},
    TERMINATE_ALL_NEXT_STEP,
};

use super::CacheConfig;

/// Bucket key: (sequence length, has_images && is_prompt, token_offset)
/// We bucket sequences by these criteria to ensure all sequences in a batch have the same
/// length, avoiding padding issues with flash attention varlen.
type BucketKey = (usize, bool, usize);

/// Allow sequences to wait for 64 scheduling passes before warning of deprivation.
const WAITING_TIMEOUT: usize = 64;

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
                true, // Default enabled, will be configured by Engine
                vec![0],
            ))),
            block_size: cache_config.block_size,
            config,
            prefix_caching_enabled: true,
            seq_block_hashes: HashMap::new(),
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
    ) {
        let hashes = self.seq_block_hashes.entry(seq_id).or_default();
        if hashes.is_empty() {
            // Compute all hashes from scratch
            *hashes = compute_block_hashes(tokens, self.block_size, mm_features, &[]);
        } else {
            // Incrementally compute new block hashes
            let new = compute_new_block_hashes(tokens, self.block_size, hashes, mm_features, &[]);
            hashes.extend(new);
        }
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
            // Use effective length for prompts so sequences with different
            // prefix cache hits land in separate buckets. The causal offset
            // (seqlen_k - seqlen_q) is only correct when all Qs are the same.
            let effective_len = if seq_guard.is_prompt() {
                seq_guard.len().saturating_sub(seq_guard.prefix_cache_len())
            } else {
                seq_guard.len()
            };
            let key: BucketKey = (
                effective_len,
                seq_guard.images().is_some() && seq_guard.is_prompt(),
                seq_guard.token_offset(),
            );
            drop(seq_guard);

            buckets.entry(key).or_default().push_back(seq);
        }

        if buckets.len() == 1 {
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

        // Preempt sequences from other buckets
        for (_, seqs) in buckets {
            for seq in seqs.into_iter().rev() {
                ids_to_preempt.push(*get_mut_arcmutex!(seq).id());
                self._preempt(seq);
            }
        }

        // Remove preempted sequences from self.running
        self.running
            .retain(|seq| !ids_to_preempt.contains(get_mut_arcmutex!(seq).id()));

        selected
    }

    pub fn schedule(&mut self, logger: &IntervalLogger) -> PagedAttentionSchedulerOutput {
        let mut scheduled: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        let mut for_waiting_again: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
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
            drop(seq_guard);

            // Compute block hashes for prefix cache lookup
            self.ensure_block_hashes(seq_id, &tokens, &mm_features);
            let block_hashes = self
                .seq_block_hashes
                .get(&seq_id)
                .cloned()
                .unwrap_or_default();

            // Look up prefix cache hits
            let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
            let computed = if self.prefix_caching_enabled {
                kv_mgr.get_computed_blocks(&block_hashes, num_tokens)
            } else {
                super::kv_cache_manager::ComputedBlocks {
                    block_ids: Vec::new(),
                    num_computed_tokens: 0,
                }
            };
            let num_computed = computed.num_computed_tokens;
            // Try to allocate blocks
            let alloc_result = kv_mgr.allocate_slots(seq_id, num_tokens, &computed.block_ids);
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
                    // Not enough blocks — check starvation
                    let count = self.waiting_counts.entry(seq_id).or_insert(0);
                    *count += 1;

                    if *count > WAITING_TIMEOUT {
                        // Try to preempt a running sequence
                        if let Some(seq_to_preempt) = self.running.pop_back() {
                            self._preempt(seq_to_preempt);

                            // Retry allocation
                            let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                            let retry =
                                kv_mgr.allocate_slots(seq_id, num_tokens, &computed.block_ids);
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

            let new_seq_has_images = get_mut_arcmutex!(seq).has_images();
            if !scheduled.is_empty()
                && get_mut_arcmutex!(scheduled[0]).has_images() != new_seq_has_images
            {
                // Free allocated blocks before deferring this image-incompatible sequence
                if !did_ignore {
                    let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                    kv_mgr.free(seq_id);
                    drop(kv_mgr);
                }
                let seq = self.waiting.pop_front().unwrap();
                for_waiting_again.push_back(seq);
                continue;
            }

            if !did_ignore {
                get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
                // Set prefix cache len so the pipeline knows to skip cached tokens
                get_mut_arcmutex!(seq).set_prefix_cache_len(num_computed);
            }

            let seq = self.waiting.pop_front().unwrap();
            if did_ignore {
                // Sequence is terminal (FinishedIgnored) — do NOT add to running queue.
                // Clean up associated state and free any allocated blocks.
                let seq_id = *get_mut_arcmutex!(seq).id();
                self.waiting_counts.remove(&seq_id);
                self.seq_block_hashes.remove(&seq_id);
                let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                kv_mgr.free(seq_id);
                drop(kv_mgr);
                continue;
            }
            self.running.push_back(seq.clone());
            scheduled.push_back(seq);
        }
        self.waiting.extend(for_waiting_again);

        if !scheduled.is_empty() {
            // Bucket scheduled prompts by sequence length to ensure all sequences in a batch
            // have the same length (required for correct flash attention varlen operation).
            let scheduled = self.bucket_and_preempt_sequences(scheduled);

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

        let mut running: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        while !self.running.is_empty() {
            let seq = self.running.pop_front().unwrap();
            let mut finished_with_break = false;

            let seq_guard = get_mut_arcmutex!(seq);
            let seq_id = *seq_guard.id();
            let num_tokens = seq_guard.len() + 1; // +1 for the new token to be generated
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
                let new_seq_has_images = get_mut_arcmutex!(seq).has_images();
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

        // Bucket running completions by sequence length
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

        // Eagerly cache any newly-full blocks so other requests can hit the prefix cache
        // sooner, rather than waiting until finish/preempt. cache_blocks is idempotent.
        if self.prefix_caching_enabled {
            // Collect sequence info first to avoid borrow conflict with self.ensure_block_hashes
            let seq_infos: Vec<(usize, Vec<u32>, Vec<MultiModalFeature>)> = self
                .running
                .iter()
                .map(|seq| {
                    let seq_guard = get_mut_arcmutex!(seq);
                    let seq_id = *seq_guard.id();
                    let tokens = seq_guard.get_toks().to_vec();
                    let mm_features = seq_guard.mm_features().to_vec();
                    (seq_id, tokens, mm_features)
                })
                .collect();

            for (seq_id, tokens, mm_features) in &seq_infos {
                self.ensure_block_hashes(*seq_id, tokens, mm_features);
                if let Some(block_hashes) = self.seq_block_hashes.get(seq_id).cloned() {
                    let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                    kv_mgr.cache_blocks(*seq_id, &block_hashes, tokens.len());
                }
            }
        }

        logger.set_num_running(self.running.len());
        logger.set_num_waiting(self.waiting.len());

        PagedAttentionSchedulerOutput {
            scheduled: self.running.clone().into_iter().collect(),
            num_cached_tokens: Vec::new(), // No prefix cache for completion
        }
    }

    pub fn free_finished_sequence_groups(&mut self) {
        // Collect finished sequence info before modifying self.running
        let mut finished: Vec<(usize, Vec<u32>, Vec<MultiModalFeature>)> = Vec::new();
        for seq in self.running.iter() {
            let seq_guard = get_mut_arcmutex!(seq);
            if seq_guard.is_finished_paged_attn() {
                let id = *seq_guard.id();
                let tokens = seq_guard.get_toks().to_vec();
                let mm_features = seq_guard.mm_features().to_vec();
                finished.push((id, tokens, mm_features));
            }
        }

        // Remove finished sequences from running
        self.running
            .retain(|seq| !get_mut_arcmutex!(seq).is_finished_paged_attn());

        // Cache and free blocks for finished sequences
        if self.prefix_caching_enabled {
            for (id, tokens, mm_features) in &finished {
                self.ensure_block_hashes(*id, tokens, mm_features);
                let block_hashes = self.seq_block_hashes.get(id).cloned().unwrap_or_default();
                let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                kv_mgr.cache_blocks(*id, &block_hashes, tokens.len());
                drop(kv_mgr);
            }
        }

        let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
        for (id, _, _) in finished {
            kv_mgr.free(id);
            self.seq_block_hashes.remove(&id);
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
        let seq_id = *seq_guard.id();
        let tokens = seq_guard.get_toks().to_vec();
        let mm_features = seq_guard.mm_features().to_vec();
        drop(seq_guard);

        // Ensure block hashes are up-to-date before freeing
        self.ensure_block_hashes(seq_id, &tokens, &mm_features);
        let block_hashes = self
            .seq_block_hashes
            .get(&seq_id)
            .cloned()
            .unwrap_or_default();

        // Cache all full blocks and free — blocks stay in cache for LRU reuse
        let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
        if self.prefix_caching_enabled {
            kv_mgr.cache_blocks(seq_id, &block_hashes, tokens.len());
        }
        kv_mgr.free(seq_id);
        drop(kv_mgr);

        self.waiting.push_front(seq);
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
    fn kv_cache_manager(&self) -> Option<Arc<tokio::sync::Mutex<KVCacheManager>>> {
        Some(self.kv_cache_manager.clone())
    }
    fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.set_prefix_caching_enabled_sync(enabled);
    }
}
