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
            adapter_generation_key, clamp_prefix_cache_hit_len, compute_block_hashes,
            compute_new_block_hashes, BlockHash, MultiModalFeature,
        },
        kv_cache_manager::KVCacheManager,
    },
    scheduler::{PagedPrefixCacheValidator, Scheduler, SchedulerOutput},
    sequence::{clamp_prefix_cache_len_for_mm_features, Sequence, SequenceState, StopReason},
    AdapterGenerationId, TERMINATE_ALL_NEXT_STEP,
};

use super::CacheConfig;

/// Bucket key: (sequence length bucket, cached prefix, raw request, prompt images, token offset)
type BucketKey = (usize, usize, Option<usize>, bool, usize);

const RAGGED_PROMPT_BUCKET_TOKENS: usize = 256;

#[derive(Clone, Copy)]
enum BatchKind {
    Prompt,
    Completion,
}

/// Allow sequences to wait for 64 scheduling passes before warning of deprivation.
const WAITING_TIMEOUT: usize = 64;

/// (seq_id, tokens, mm_features, adapter_generation, block_hash_revision, num_computed_tokens)
type SeqCacheInfo = (
    usize,
    Vec<u32>,
    Vec<MultiModalFeature>,
    Option<AdapterGenerationId>,
    u64,
    usize,
);

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
    requires_uniform_prompt_batch: bool,
    requires_uniform_completion_batch: bool,
    completion_turn_due: bool,
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
            requires_uniform_prompt_batch: true,
            requires_uniform_completion_batch: true,
            completion_turn_due: false,
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
        adapter_generation: Option<AdapterGenerationId>,
        revision: u64,
    ) {
        let adapter_key = adapter_generation_key(adapter_generation);
        let known_revision = self.seq_block_hash_revisions.get(&seq_id).copied();
        let hashes = self.seq_block_hashes.entry(seq_id).or_default();
        if hashes.is_empty() || known_revision != Some(revision) {
            *hashes =
                compute_block_hashes(tokens, self.block_size, mm_features, adapter_key.as_slice());
            self.seq_block_hash_revisions.insert(seq_id, revision);
        } else {
            let new = compute_new_block_hashes(
                tokens,
                self.block_size,
                hashes,
                mm_features,
                adapter_key.as_slice(),
            );
            hashes.extend(new);
        }
    }

    /// Select the oldest compatible bucket and preempt the other sequences.
    fn bucket_and_preempt_sequences(
        &mut self,
        sequences: VecDeque<Arc<Mutex<Sequence>>>,
        batch_kind: BatchKind,
        require_uniform_length: bool,
    ) -> VecDeque<Arc<Mutex<Sequence>>> {
        if sequences.len() <= 1 {
            return sequences;
        }

        let mut keyed_sequences = Vec::with_capacity(sequences.len());
        let mut first_key = None;

        for seq in sequences {
            let seq_guard = get_mut_arcmutex!(seq);
            let effective_len = match (batch_kind, require_uniform_length) {
                (BatchKind::Prompt, false) => seq_guard.len().div_ceil(RAGGED_PROMPT_BUCKET_TOKENS),
                (BatchKind::Prompt, true) => {
                    seq_guard.len().saturating_sub(seq_guard.prefix_cache_len())
                }
                (BatchKind::Completion, false) => 0,
                (BatchKind::Completion, true) => seq_guard.len(),
            };
            let key: BucketKey = (
                effective_len,
                if matches!(batch_kind, BatchKind::Prompt) {
                    seq_guard.prefix_cache_len()
                } else {
                    0
                },
                seq_guard.return_raw_logits.then_some(*seq_guard.id()),
                seq_guard.has_images() && matches!(batch_kind, BatchKind::Prompt),
                seq_guard.token_offset(),
            );
            drop(seq_guard);

            first_key.get_or_insert(key);
            keyed_sequences.push((key, seq));
        }

        let first_key = first_key.unwrap();
        let mut selected = VecDeque::new();
        let mut rejected = Vec::new();
        for (key, seq) in keyed_sequences {
            if key == first_key {
                selected.push_back(seq);
            } else {
                rejected.push(seq);
            }
        }

        if rejected.is_empty() {
            return selected;
        }

        let ids_to_preempt: Vec<_> = rejected
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect();
        for seq in rejected.into_iter().rev() {
            self._preempt(seq);
        }

        self.running
            .retain(|seq| !ids_to_preempt.contains(get_mut_arcmutex!(seq).id()));

        selected
    }

    fn enforce_completion_compatibility(&mut self) {
        let running = std::mem::take(&mut self.running);
        self.running = self.bucket_and_preempt_sequences(
            running,
            BatchKind::Completion,
            self.requires_uniform_completion_batch,
        );
    }

    pub fn schedule(
        &mut self,
        logger: &IntervalLogger,
        mut prefix_validator: Option<&mut dyn PagedPrefixCacheValidator>,
    ) -> PagedAttentionSchedulerOutput {
        let mut scheduled: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        let mut for_waiting_again: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        let completion_turn_due = std::mem::take(&mut self.completion_turn_due);
        let completion_due = self
            .running
            .iter()
            .any(|seq| get_mut_arcmutex!(seq).is_prompt())
            || completion_turn_due && !self.running.is_empty();
        while !completion_due && !self.waiting.is_empty() {
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
            let adapter_generation = seq_guard.adapter_generation();
            let block_hash_revision = seq_guard.block_hash_revision();
            drop(seq_guard);

            // Compute block hashes for prefix cache lookup
            self.ensure_block_hashes(
                seq_id,
                &tokens,
                &mm_features,
                adapter_generation,
                block_hash_revision,
            );
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
        self.waiting.extend(for_waiting_again);

        if !scheduled.is_empty() {
            // Prefix-cache offsets and prompt tensor shapes must be uniform within a batch.
            let require_uniform_length = self.requires_uniform_prompt_batch
                || scheduled.iter().any(|seq| {
                    let seq = get_mut_arcmutex!(seq);
                    seq.return_raw_logits || seq.prefix_cache_len() > 0
                });
            let scheduled = self.bucket_and_preempt_sequences(
                scheduled,
                BatchKind::Prompt,
                require_uniform_length,
            );

            // Rebuild num_cached_tokens from the bucketed sequences.
            // prefix_cache_len was set per-sequence above, so this stays aligned
            // even after bucketing removes sequences from non-contiguous positions.
            let num_cached_tokens: Vec<usize> = scheduled
                .iter()
                .map(|seq| get_mut_arcmutex!(seq).prefix_cache_len())
                .collect();

            logger.set_num_running(self.running.len());
            logger.set_num_waiting(self.waiting.len());
            self.completion_turn_due = true;

            return PagedAttentionSchedulerOutput {
                scheduled: scheduled.into_iter().collect(),
                num_cached_tokens,
            };
        }

        // Reserve completion token slots, preempting lowest priority first.

        self.sort_running_by_priority_fcfs();

        let mut running: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        while !self.running.is_empty() {
            let seq = self.running.pop_front().unwrap();
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
                let new_seq_has_images = get_mut_arcmutex!(seq).has_images();
                if running.is_empty()
                    || get_mut_arcmutex!(running[0]).has_images() == new_seq_has_images
                {
                    running.push_back(seq);
                } else {
                    self._preempt(seq);
                }
            }
        }
        self.running = running;

        self.enforce_completion_compatibility();

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
            let seq_infos: Vec<SeqCacheInfo> = self
                .running
                .iter()
                .map(|seq| {
                    let seq_guard = get_mut_arcmutex!(seq);
                    let seq_id = *seq_guard.id();
                    let tokens = seq_guard.get_toks().to_vec();
                    let mm_features = seq_guard.mm_features().to_vec();
                    let adapter_generation = seq_guard.adapter_generation();
                    let block_hash_revision = seq_guard.block_hash_revision();
                    let num_computed_tokens = seq_guard.num_computed_tokens();
                    (
                        seq_id,
                        tokens,
                        mm_features,
                        adapter_generation,
                        block_hash_revision,
                        num_computed_tokens,
                    )
                })
                .collect();

            for (
                seq_id,
                tokens,
                mm_features,
                adapter_generation,
                block_hash_revision,
                num_computed_tokens,
            ) in &seq_infos
            {
                self.ensure_block_hashes(
                    *seq_id,
                    tokens,
                    mm_features,
                    *adapter_generation,
                    *block_hash_revision,
                );
                if let Some(block_hashes) = self.seq_block_hashes.get(seq_id).cloned() {
                    let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                    kv_mgr.cache_blocks(*seq_id, &block_hashes, *num_computed_tokens);
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
        let mut finished: Vec<SeqCacheInfo> = Vec::new();
        let mut cacheable_finished: Vec<SeqCacheInfo> = Vec::new();
        for seq in self.running.iter() {
            let seq_guard = get_mut_arcmutex!(seq);
            if seq_guard.is_finished_paged_attn() {
                let id = *seq_guard.id();
                let tokens = seq_guard.get_toks().to_vec();
                let mm_features = seq_guard.mm_features().to_vec();
                let adapter_generation = seq_guard.adapter_generation();
                let block_hash_revision = seq_guard.block_hash_revision();
                let num_computed_tokens = seq_guard.num_computed_tokens();
                let info = (
                    id,
                    tokens,
                    mm_features,
                    adapter_generation,
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
            for (
                id,
                tokens,
                mm_features,
                adapter_generation,
                block_hash_revision,
                num_computed_tokens,
            ) in &cacheable_finished
            {
                self.ensure_block_hashes(
                    *id,
                    tokens,
                    mm_features,
                    *adapter_generation,
                    *block_hash_revision,
                );
                let block_hashes = self.seq_block_hashes.get(id).cloned().unwrap_or_default();
                let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                kv_mgr.cache_blocks(*id, &block_hashes, *num_computed_tokens);
                drop(kv_mgr);
            }
        }

        let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
        for (id, _, _, _, _, _) in finished {
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
        let adapter_generation = seq_guard.adapter_generation();
        let block_hash_revision = seq_guard.block_hash_revision();
        let num_computed_tokens = seq_guard.num_computed_tokens();
        drop(seq_guard);

        // Ensure block hashes are up-to-date before freeing
        self.ensure_block_hashes(
            seq_id,
            &tokens,
            &mm_features,
            adapter_generation,
            block_hash_revision,
        );
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
        self.running
            .make_contiguous()
            .sort_by_key(|seq| get_mut_arcmutex!(seq).timestamp());
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
    fn set_requires_uniform_prompt_batch(&mut self, required: bool) {
        self.requires_uniform_prompt_batch = required;
    }
    fn set_requires_uniform_completion_batch(&mut self, required: bool) {
        self.requires_uniform_completion_batch = required;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        paged_attention::PagedCacheType,
        sampler::Sampler,
        sequence::{SeqStepType, SequenceGroup, SequenceRecognizer},
    };
    use tokio::sync::{mpsc::channel, Mutex as TokioMutex};

    fn test_scheduler() -> PagedAttentionScheduler {
        PagedAttentionScheduler::new(
            PagedAttentionSchedulerConfig { max_num_seqs: 8 },
            CacheConfig {
                block_size: 8,
                num_gpu_blocks: 128,
                cache_type: PagedCacheType::Auto,
                kv_cache_group_ids: vec![0],
            },
        )
    }

    fn test_sequence_with_images(
        id: usize,
        len: usize,
        input_images: Option<Vec<image::DynamicImage>>,
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
        .unwrap();
        let group = Arc::new(TokioMutex::new(SequenceGroup::new(1, false, true, None)));
        let seq = Sequence::new_waiting(
            vec![1; len],
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
            vec![],
        );
        seq.set_state(SequenceState::RunningCompletion);
        Arc::new(Mutex::new(seq))
    }

    fn test_sequence(id: usize, len: usize) -> Arc<Mutex<Sequence>> {
        test_sequence_with_images(id, len, None)
    }

    #[test]
    fn ragged_completion_batch_keeps_all_sequences_running() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_completion_batch = false;
        scheduler.running.push_back(test_sequence(0, 4));
        scheduler.running.push_back(test_sequence(1, 7));

        scheduler.enforce_completion_compatibility();

        assert_eq!(scheduler.running.len(), 2);
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn empty_image_list_does_not_split_text_completion_batch() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_completion_batch = false;
        scheduler.running.push_back(test_sequence(0, 4));
        let prompt = test_sequence_with_images(1, 7, Some(vec![]));
        get_mut_arcmutex!(prompt).set_state(SequenceState::RunningPrompt);
        scheduler.running.push_back(prompt);

        scheduler.enforce_completion_compatibility();

        assert_eq!(scheduler.running.len(), 2);
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn uniform_completion_batch_preempts_other_lengths() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_completion_batch = true;
        scheduler.running.push_back(test_sequence(0, 4));
        scheduler.running.push_back(test_sequence(1, 7));

        scheduler.enforce_completion_compatibility();

        assert_eq!(scheduler.running.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(get_mut_arcmutex!(scheduler.running[0]).len(), 4);
    }

    #[test]
    fn ragged_prompt_batch_keeps_compatible_sequences() {
        let mut scheduler = test_scheduler();
        let prompts = VecDeque::from([test_sequence(0, 4), test_sequence(1, 7)]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

        assert_eq!(scheduled.len(), 2);
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn ragged_prompt_batch_bounds_padding() {
        let mut scheduler = test_scheduler();
        let prompts = VecDeque::from([test_sequence(0, 4), test_sequence(1, 300)]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(get_mut_arcmutex!(scheduled[0]).len(), 4);
    }

    #[test]
    fn ragged_prompt_batch_preserves_fcfs_across_buckets() {
        let mut scheduler = test_scheduler();
        scheduler.waiting.push_back(test_sequence(4, 8));
        let prompts = VecDeque::from([
            test_sequence(0, 300),
            test_sequence(1, 4),
            test_sequence(2, 600),
            test_sequence(3, 7),
        ]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(scheduled[0]).id(), 0);
        let waiting_ids: Vec<_> = scheduler
            .waiting
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect();
        assert_eq!(waiting_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn cached_prompt_batch_requires_matching_prefix_offsets() {
        let mut scheduler = test_scheduler();
        let first = test_sequence(0, 100);
        let second = test_sequence(1, 132);
        get_mut_arcmutex!(first).set_prefix_cache_len(32);
        get_mut_arcmutex!(second).set_prefix_cache_len(64);
        let prompts = VecDeque::from([first, second]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn prompt_batch_separates_raw_logits_requests() {
        let mut scheduler = test_scheduler();
        let first = test_sequence(0, 4);
        let second = test_sequence(1, 4);
        get_mut_arcmutex!(second).return_raw_logits = true;
        let prompts = VecDeque::from([first, second]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn prompt_batch_singletonizes_raw_logits_requests() {
        let mut scheduler = test_scheduler();
        let first = test_sequence(0, 4);
        let second = test_sequence(1, 4);
        get_mut_arcmutex!(first).return_raw_logits = true;
        get_mut_arcmutex!(second).return_raw_logits = true;
        let prompts = VecDeque::from([first, second]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn prompt_batch_gets_a_completion_turn_before_refill() {
        let mut scheduler = test_scheduler();
        let running = test_sequence(0, 4);
        get_mut_arcmutex!(running).set_state(SequenceState::RunningPrompt);
        get_mut_arcmutex!(running).set_num_computed_tokens(4);
        scheduler.running.push_back(running);

        let waiting = test_sequence(1, 7);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let completion = scheduler.schedule(&logger, None);

        assert_eq!(completion.scheduled.len(), 1);
        assert!(!get_mut_arcmutex!(completion.scheduled[0]).is_prompt());
        assert_eq!(scheduler.waiting.len(), 1);

        let prompt = scheduler.schedule(&logger, None);

        assert_eq!(prompt.scheduled.len(), 1);
        assert!(get_mut_arcmutex!(prompt.scheduled[0]).is_prompt());
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn completions_do_not_delay_prompt_admission() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(0, 4));

        let waiting = test_sequence(1, 7);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let prompt = scheduler.schedule(&logger, None);

        assert_eq!(prompt.scheduled.len(), 1);
        assert!(get_mut_arcmutex!(prompt.scheduled[0]).is_prompt());
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn completion_turn_survives_finished_prompt_cleanup() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(0, 4));
        scheduler.completion_turn_due = true;

        let waiting = test_sequence(1, 7);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let completion = scheduler.schedule(&logger, None);

        assert_eq!(completion.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(completion.scheduled[0]).id(), 0);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn completion_turn_without_running_sequences_does_not_delay_prompts() {
        let mut scheduler = test_scheduler();
        scheduler.completion_turn_due = true;

        let waiting = test_sequence(1, 7);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let prompt = scheduler.schedule(&logger, None);

        assert_eq!(prompt.scheduled.len(), 1);
        assert!(get_mut_arcmutex!(prompt.scheduled[0]).is_prompt());
        assert!(scheduler.waiting.is_empty());
    }
}
