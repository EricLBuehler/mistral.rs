//! The Scheduler uses a KVCacheManager to schedule and automatically batch sequences.
//! The primary method `schedule` returns the batched sequences as inputs.

use std::{
    collections::{HashMap, HashSet, VecDeque},
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
    pipeline::prompt_chunks::{build_prompt_chunk_plan, next_prompt_chunk_group, PromptChunkPlan},
    scheduler::{
        modality_signature, PagedPrefixCacheValidation, PagedPrefixCacheValidator, Scheduler,
        SchedulerOutput,
    },
    sequence::{
        clamp_prefix_cache_len_for_mm_features, SeqStepType, Sequence, SequenceState, StopReason,
    },
    speculative::SpeculativePrefixCheckpointPolicy,
    AdapterGenerationId, Response, TERMINATE_ALL_NEXT_STEP,
};

use super::CacheConfig;

/// Bucket key: (sequence length bucket, cached prefix, raw request, media, token offset)
type BucketKey = (usize, usize, Option<usize>, u8, usize);

const RAGGED_PROMPT_BUCKET_TOKENS: usize = 256;
#[derive(Clone, Copy)]
enum BatchKind {
    Prompt,
    Completion,
}

struct PromptBatch {
    scheduled: VecDeque<Arc<Mutex<Sequence>>>,
    chunk_size: Option<usize>,
    chunks: Option<Vec<PromptChunkPlan>>,
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
    /// Per-sequence prompt chunk size selected from the batch-wide token budget.
    pub prompt_chunk_size: Option<usize>,
    /// Exact prompt ranges selected for this scheduler turn, aligned with `scheduled`.
    pub(crate) scheduled_prompt_chunks: Option<Vec<PromptChunkPlan>>,
}

impl PagedAttentionSchedulerOutput {
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn retain_prompt_prefix(&mut self, retained: usize) -> Option<usize> {
        assert!(retained > 0);
        assert_eq!(self.num_cached_tokens.len(), self.scheduled.len());
        if let Some(chunks) = self.scheduled_prompt_chunks.as_ref() {
            assert_eq!(chunks.len(), self.scheduled.len());
        }
        let first_omitted = self
            .scheduled
            .get(retained)
            .map(|seq| *get_mut_arcmutex!(seq).id())?;
        self.scheduled.truncate(retained);
        self.num_cached_tokens.truncate(retained);
        if let Some(chunks) = self.scheduled_prompt_chunks.as_mut() {
            chunks.truncate(retained);
        }
        Some(first_omitted)
    }
}

pub struct PagedAttentionSchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_prefill_chunk_tokens: usize,
    pub max_decode_steps_before_prefill: usize,
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
    requires_uniform_media_batch: bool,
    supports_packed_prefill: bool,
    scheduler_visible_prompt_chunks: bool,
    prompt_chunks_require_block_alignment: bool,
    prefix_policy: SpeculativePrefixCheckpointPolicy,
    decode_steps_since_prefill: usize,
    waiting_prompt_preemption_enabled: bool,
    next_prompt_sequence_id: Option<usize>,
    completion_cursor: usize,
    /// Block hashes per sequence for prefix caching.
    /// Computed incrementally as sequences grow.
    seq_block_hashes: HashMap<usize, Vec<BlockHash>>,
    seq_block_hash_revisions: HashMap<usize, u64>,
    /// Per-sequence waitlist counter for starvation detection.
    waiting_counts: HashMap<usize, usize>,
    finished_recurrent_slots: Vec<(usize, usize)>,
    preempted_sequence_ids: Vec<usize>,
}

impl PagedAttentionScheduler {
    pub fn new(config: PagedAttentionSchedulerConfig, cache_config: CacheConfig) -> Self {
        assert!(config.max_num_seqs > 0);
        assert!(config.max_num_batched_tokens > 0);
        assert!(config.max_prefill_chunk_tokens > 0);
        assert!(config.max_decode_steps_before_prefill > 0);
        info!(
            max_num_seqs = config.max_num_seqs,
            max_num_batched_tokens = config.max_num_batched_tokens,
            max_prefill_chunk_tokens = config.max_prefill_chunk_tokens,
            max_decode_steps_before_prefill = config.max_decode_steps_before_prefill,
            "Configured PagedAttention scheduler"
        );
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
            requires_uniform_media_batch: false,
            supports_packed_prefill: false,
            scheduler_visible_prompt_chunks: false,
            prompt_chunks_require_block_alignment: false,
            prefix_policy: SpeculativePrefixCheckpointPolicy::default(),
            decode_steps_since_prefill: 0,
            waiting_prompt_preemption_enabled: true,
            next_prompt_sequence_id: None,
            completion_cursor: 0,
            seq_block_hashes: HashMap::new(),
            seq_block_hash_revisions: HashMap::new(),
            waiting_counts: HashMap::new(),
            finished_recurrent_slots: Vec::new(),
            preempted_sequence_ids: Vec::new(),
        }
    }

    fn prompt_chunk_size(&self, batch_size: usize) -> Option<usize> {
        (batch_size > 0).then(|| (self.prefill_token_budget() / batch_size).max(1))
    }

    fn prefill_token_budget(&self) -> usize {
        if self.scheduler_visible_prompt_chunks {
            self.config
                .max_num_batched_tokens
                .min(self.config.max_prefill_chunk_tokens)
        } else {
            self.config.max_num_batched_tokens
        }
    }

    fn supports_scheduler_visible_prompt_chunks(&self, seq: &Sequence) -> bool {
        self.scheduler_visible_prompt_chunks
            && !seq.return_raw_logits
            && !seq.is_xlora()
            && matches!(seq.sequence_stepping_type(), SeqStepType::PromptAndDecode)
            && !seq.has_suffix_only_prefill_toks()
            && !((seq.has_images() || seq.has_audios() || seq.has_videos())
                && seq.mm_features().is_empty())
    }

    fn select_prompt_batch(
        &mut self,
        mut candidates: VecDeque<Arc<Mutex<Sequence>>>,
    ) -> PromptBatch {
        if candidates.is_empty() {
            self.next_prompt_sequence_id = None;
            return PromptBatch {
                scheduled: VecDeque::new(),
                chunk_size: None,
                chunks: None,
            };
        }
        if let Some(position) = self.next_prompt_sequence_id.and_then(|sequence_id| {
            candidates
                .iter()
                .position(|seq| *get_mut_arcmutex!(seq).id() == sequence_id)
        }) {
            candidates.rotate_left(position);
        }
        let rotation_candidates = candidates.clone();
        let first = candidates
            .front()
            .expect("non-empty prompt candidates disappeared");
        let first = get_mut_arcmutex!(first);
        let scheduler_visible = self.supports_scheduler_visible_prompt_chunks(&first);
        let first_modality = modality_signature(&first);
        let first_offset = first.token_offset();
        let first_cursor = first.num_computed_tokens();
        drop(first);
        let max_candidates = if scheduler_visible {
            self.prefill_token_budget()
        } else {
            usize::MAX
        };

        let candidates = candidates
            .into_iter()
            .filter(|seq| {
                let seq = get_mut_arcmutex!(seq);
                self.supports_scheduler_visible_prompt_chunks(&seq) == scheduler_visible
                    && (!(self.requires_uniform_prompt_batch || self.requires_uniform_media_batch)
                        || modality_signature(&seq) == first_modality)
                    && (!self.requires_uniform_prompt_batch
                        || (seq.token_offset() == first_offset
                            && (!scheduler_visible || seq.num_computed_tokens() == first_cursor)))
            })
            .take(max_candidates)
            .collect::<VecDeque<_>>();

        if !scheduler_visible {
            let require_uniform_length = self.requires_uniform_prompt_batch
                || candidates.iter().any(|seq| {
                    let seq = get_mut_arcmutex!(seq);
                    seq.return_raw_logits || seq.prefix_cache_len() > 0
                });
            let scheduled = self.bucket_and_preempt_sequences(
                candidates,
                BatchKind::Prompt,
                require_uniform_length,
            );
            let chunk_size = if self.scheduler_visible_prompt_chunks {
                None
            } else {
                self.prompt_chunk_size(scheduled.len())
            };
            self.advance_prompt_cursor(&rotation_candidates, &scheduled);
            return PromptBatch {
                scheduled,
                chunk_size,
                chunks: None,
            };
        }

        let chunk_size = self.prompt_chunk_size(candidates.len()).unwrap();
        let block_align = self
            .prompt_chunks_require_block_alignment
            .then_some(self.block_size);
        let chunk_plans = candidates
            .iter()
            .map(|seq| {
                let seq = get_mut_arcmutex!(seq);
                build_prompt_chunk_plan(
                    seq.get_toks().len(),
                    seq.num_computed_tokens(),
                    chunk_size,
                    block_align,
                    self.prefix_policy.replay_for(modality_signature(&seq)),
                    seq.mm_features(),
                )
            })
            .collect::<Vec<_>>();
        let plan_indices = vec![0; chunk_plans.len()];
        let require_uniform_query_len = self.requires_uniform_prompt_batch
            || !self.supports_packed_prefill
            || self.prompt_chunks_require_block_alignment;
        let (active_indices, _, _) =
            next_prompt_chunk_group(&plan_indices, &chunk_plans, require_uniform_query_len)
                .expect("running prompt has uncomputed tokens");

        let mut scheduled = VecDeque::with_capacity(active_indices.len());
        let mut scheduled_chunks = Vec::with_capacity(active_indices.len());
        for index in active_indices {
            scheduled.push_back(candidates[index].clone());
            scheduled_chunks.push(chunk_plans[index][0]);
        }
        self.advance_prompt_cursor(&rotation_candidates, &scheduled);
        PromptBatch {
            scheduled,
            chunk_size: Some(chunk_size),
            chunks: Some(scheduled_chunks),
        }
    }

    fn advance_prompt_cursor(
        &mut self,
        candidates: &VecDeque<Arc<Mutex<Sequence>>>,
        scheduled: &VecDeque<Arc<Mutex<Sequence>>>,
    ) {
        self.next_prompt_sequence_id = candidates
            .iter()
            .find(|candidate| {
                let candidate_id = *get_mut_arcmutex!(candidate).id();
                scheduled
                    .iter()
                    .all(|seq| *get_mut_arcmutex!(seq).id() != candidate_id)
            })
            .map(|seq| *get_mut_arcmutex!(seq).id());
    }

    fn completion_is_due(&self) -> bool {
        let has_completion = self
            .running
            .iter()
            .any(|seq| get_mut_arcmutex!(seq).is_completion());
        if !has_completion {
            return false;
        }
        let running_prompt_tokens = self
            .running
            .iter()
            .filter_map(|seq| {
                let seq = get_mut_arcmutex!(seq);
                seq.is_prompt().then(|| seq.num_uncomputed_tokens())
            })
            .sum::<usize>();
        if running_prompt_tokens > 0
            && running_prompt_tokens <= self.prefill_token_budget()
            && self.decode_steps_since_prefill > 0
        {
            return false;
        }

        let has_running_prompt = running_prompt_tokens > 0;
        let has_waiting_prompt = !self.waiting.is_empty();
        if !has_running_prompt
            && has_waiting_prompt
            && self.running.len() < self.config.max_num_seqs
            && self.waiting_prompt_fits_free_blocks()
            && self.decode_steps_since_prefill > 0
        {
            return false;
        }

        let has_prompt = has_running_prompt || has_waiting_prompt;
        !has_prompt || self.decode_steps_since_prefill < self.config.max_decode_steps_before_prefill
    }

    fn waiting_prompt_fits_free_blocks(&self) -> bool {
        let Some(prompt) = self.waiting.front() else {
            return false;
        };
        let prompt_blocks = get_mut_arcmutex!(prompt)
            .get_toks()
            .len()
            .div_ceil(self.block_size);
        let decode_reserve = self
            .running
            .iter()
            .filter(|seq| get_mut_arcmutex!(seq).is_completion())
            .count();
        prompt_blocks.saturating_add(decode_reserve)
            <= get_mut_arcmutex!(self.kv_cache_manager).num_free_blocks()
    }

    fn completion_token_cost(seq: &Sequence) -> usize {
        seq.num_uncomputed_tokens()
            .saturating_add(seq.active_staged_speculative_len())
            .max(1)
    }

    fn completion_batch_indices(
        rows: &[Arc<Mutex<Sequence>>],
        cursor: usize,
        token_budget: usize,
    ) -> (Vec<usize>, usize) {
        if rows.is_empty() {
            return (Vec::new(), cursor);
        }

        let len = rows.len();
        let start = cursor % len;
        let staged_width = get_mut_arcmutex!(rows[start]).active_staged_speculative_len();
        let mut remaining_tokens = token_budget;
        let mut selected = Vec::with_capacity(len);
        let mut last_selected = start;

        for offset in 0..len {
            let index = (start + offset) % len;
            let seq = get_mut_arcmutex!(rows[index]);
            if seq.active_staged_speculative_len() != staged_width {
                continue;
            }
            let token_cost = Self::completion_token_cost(&seq);
            drop(seq);
            if token_cost <= remaining_tokens || selected.is_empty() {
                remaining_tokens = remaining_tokens.saturating_sub(token_cost);
                selected.push(index);
                last_selected = index;
            }
        }

        (selected, (last_selected + 1) % len)
    }

    fn live_completion_rows(&self) -> Vec<Arc<Mutex<Sequence>>> {
        let mut rows = self
            .running
            .iter()
            .filter(|seq| get_mut_arcmutex!(seq).is_completion())
            .cloned()
            .collect::<Vec<_>>();
        rows.sort_by_key(|seq| get_mut_arcmutex!(seq).timestamp());
        rows
    }

    fn can_continue_decode_batch_inner(
        &self,
        sequence_ids: &[usize],
        terminate_all_pending: bool,
    ) -> bool {
        if sequence_ids.is_empty() || terminate_all_pending || !self.completion_is_due() {
            return false;
        }

        let rows = self.live_completion_rows();
        let (selected, _) = Self::completion_batch_indices(
            &rows,
            self.completion_cursor,
            self.config.max_num_batched_tokens,
        );
        selected.len() == sequence_ids.len()
            && selected
                .iter()
                .zip(sequence_ids)
                .all(|(&index, id)| *get_mut_arcmutex!(rows[index]).id() == *id)
    }

    fn record_decode_step(&mut self) {
        self.decode_steps_since_prefill = self
            .decode_steps_since_prefill
            .saturating_add(1)
            .min(self.config.max_decode_steps_before_prefill);
    }

    fn select_completion_batch(&mut self) -> Vec<Arc<Mutex<Sequence>>> {
        let rows = self.running.iter().cloned().collect::<Vec<_>>();
        let (selected, next_cursor) = Self::completion_batch_indices(
            &rows,
            self.completion_cursor,
            self.config.max_num_batched_tokens,
        );
        self.completion_cursor = next_cursor;
        selected
            .into_iter()
            .map(|index| rows[index].clone())
            .collect()
    }

    fn finish_ignored_sequence(
        &mut self,
        seq: Arc<Mutex<Sequence>>,
        response: Response,
        recurrent_state_released: bool,
    ) {
        let (seq_id, responder, recurrent_state_idx) = {
            let mut seq_guard = get_mut_arcmutex!(seq);
            seq_guard.set_state(SequenceState::FinishedIgnored);
            let recurrent_state_idx = seq_guard.recurrent_state_idx();
            if recurrent_state_released {
                seq_guard.set_recurrent_state_idx(None);
            }
            (*seq_guard.id(), seq_guard.responder(), recurrent_state_idx)
        };

        if let Some(slot_idx) = recurrent_state_idx {
            if !recurrent_state_released {
                self.finished_recurrent_slots.push((seq_id, slot_idx));
            }
        }

        self.waiting_counts.remove(&seq_id);
        self.seq_block_hashes.remove(&seq_id);
        self.seq_block_hash_revisions.remove(&seq_id);
        get_mut_arcmutex!(self.kv_cache_manager).free(seq_id);

        if responder.try_send(response).is_err() {
            warn!("Failed to deliver scheduling error for sequence {seq_id}");
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
                (BatchKind::Prompt, false) if self.supports_packed_prefill => 0,
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
                if self.requires_uniform_media_batch
                    || require_uniform_length && matches!(batch_kind, BatchKind::Prompt)
                {
                    modality_signature(&seq_guard)
                } else {
                    0
                },
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

    fn reject_front_of_waiting(
        &mut self,
        reason: String,
        metric_reason: &'static str,
        prefix_validator: &mut Option<&mut dyn PagedPrefixCacheValidator>,
    ) {
        warn!("{reason}");
        self.finish_front_of_waiting(
            Response::ValidationError(reason.into()),
            metric_reason,
            prefix_validator,
        );
    }

    fn fail_front_of_waiting(
        &mut self,
        reason: String,
        metric_reason: &'static str,
        prefix_validator: &mut Option<&mut dyn PagedPrefixCacheValidator>,
    ) {
        warn!("{reason}");
        self.finish_front_of_waiting(
            Response::InternalError(reason.into()),
            metric_reason,
            prefix_validator,
        );
    }

    fn finish_front_of_waiting(
        &mut self,
        response: Response,
        metric_reason: &'static str,
        prefix_validator: &mut Option<&mut dyn PagedPrefixCacheValidator>,
    ) {
        metrics::counter!("mistralrs_sequences_rejected_total", "reason" => metric_reason)
            .increment(1);
        let seq = self.waiting.pop_front().unwrap();
        let (sequence_id, recurrent_state_idx) = {
            let seq = get_mut_arcmutex!(seq);
            (*seq.id(), seq.recurrent_state_idx())
        };
        let recurrent_state_released = if let (Some(slot_idx), Some(validator)) =
            (recurrent_state_idx, prefix_validator.as_mut())
        {
            match validator.release_recurrent_state(sequence_id, slot_idx) {
                Ok(_) => true,
                Err(err) => {
                    tracing::error!(
                        "Failed to release recurrent state for sequence {sequence_id}: {err}"
                    );
                    false
                }
            }
        } else {
            false
        };
        self.finish_ignored_sequence(seq, response, recurrent_state_released);
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
        for seq in &self.running {
            let seq = get_mut_arcmutex!(seq);
            if seq.is_prompt() && seq.num_computed_tokens() == seq.len() {
                seq.set_state(SequenceState::RunningCompletion);
            }
        }
        let mut scheduled: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        let mut for_waiting_again: VecDeque<Arc<Mutex<Sequence>>> = VecDeque::new();
        let completion_due = self.completion_is_due();
        if !completion_due {
            scheduled.extend(
                self.running
                    .iter()
                    .filter(|seq| get_mut_arcmutex!(seq).is_prompt())
                    .cloned(),
            );
        }
        while !completion_due && !self.waiting.is_empty() {
            let mut ignore_reason = None;
            let seq = self.waiting.front().unwrap().clone();

            if self.running.len() >= self.config.max_num_seqs
                || scheduled.len() >= self.config.max_num_batched_tokens
            {
                break;
            }

            let seq_guard = get_mut_arcmutex!(seq);
            let seq_id = *seq_guard.id();
            let tokens = seq_guard.get_toks().to_vec();
            let num_tokens = tokens.len();
            let mm_features = seq_guard.mm_features().to_vec();
            let adapter_generation = seq_guard.adapter_generation();
            let block_hash_revision = seq_guard.block_hash_revision();
            let return_raw_logits = seq_guard.return_raw_logits;
            let new_seq_modality = modality_signature(&seq_guard);
            drop(seq_guard);

            // Reject prompts that can never fit instead of waiting out the starvation timeout.
            let total_token_capacity =
                get_mut_arcmutex!(self.kv_cache_manager).num_usable_blocks() * self.block_size;
            if num_tokens > total_token_capacity {
                self.reject_front_of_waiting(
                    format!(
                        "Sequence {seq_id} with {num_tokens} tokens exceeds the total KV cache \
                         capacity of {total_token_capacity} tokens."
                    ),
                    "over_total_capacity",
                    &mut prefix_validator,
                );
                continue;
            }

            if (self.requires_uniform_prompt_batch || self.requires_uniform_media_batch)
                && !scheduled.is_empty()
                && modality_signature(&*get_mut_arcmutex!(scheduled[0])) != new_seq_modality
            {
                let seq = self.waiting.pop_front().unwrap();
                for_waiting_again.push_back(seq);
                continue;
            }

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
            let mut computed = if self.prefix_caching_enabled && !return_raw_logits {
                kv_mgr.get_computed_blocks(&block_hashes, num_tokens)
            } else {
                super::kv_cache_manager::ComputedBlocks {
                    block_ids: Vec::new(),
                    num_computed_tokens: 0,
                }
            };
            drop(kv_mgr);

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
            let matched_prefix_tokens = computed.num_computed_tokens;
            let mut prefix_validation: Option<PagedPrefixCacheValidation> = None;
            if let Some(validator) = prefix_validator.as_mut() {
                let seq_guard = get_mut_arcmutex!(seq);
                let validation = validator.validate_prefix_cache_hit(
                    &seq_guard,
                    &block_hashes,
                    computed.num_computed_tokens,
                    self.block_size,
                );
                drop(seq_guard);
                let validation = match validation {
                    Ok(validation) => validation,
                    Err(err) => {
                        self.fail_front_of_waiting(
                            format!(
                                "Failed to prepare recurrent state for sequence {seq_id}: {err}"
                            ),
                            "recurrent_state",
                            &mut prefix_validator,
                        );
                        continue;
                    }
                };
                let valid_tokens = validation.valid_tokens();
                if valid_tokens < computed.num_computed_tokens {
                    let valid_blocks = valid_tokens / self.block_size;
                    computed.block_ids.truncate(valid_blocks);
                    computed.num_computed_tokens = valid_blocks * self.block_size;
                }
                prefix_validation = Some(validation);
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
                    // Reset waiting count on successful allocation
                    self.waiting_counts.remove(&seq_id);
                }
                None => {
                    // Not enough blocks, check starvation
                    let count = self.waiting_counts.entry(seq_id).or_insert(0);
                    *count += 1;

                    if *count > WAITING_TIMEOUT {
                        if !self.waiting_prompt_preemption_enabled {
                            break;
                        }
                        // A feasible sequence waits or evicts instead of failing on ordinary pressure.
                        let mut allocated = false;
                        loop {
                            let Some(seq_to_preempt) = self.running.pop_back() else {
                                break;
                            };
                            let preempted_id = *get_mut_arcmutex!(seq_to_preempt).id();
                            let waiting_seq = self.waiting.pop_front().unwrap();
                            self._preempt(seq_to_preempt);
                            self.waiting.push_front(waiting_seq);
                            scheduled.retain(|seq| *get_mut_arcmutex!(seq).id() != preempted_id);

                            let mut kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
                            let retry = kv_mgr.allocate_slots(
                                seq_id,
                                num_tokens,
                                &computed.block_ids[..computed_block_count],
                            );
                            drop(kv_mgr);
                            if retry.is_some() {
                                allocated = true;
                                break;
                            }
                        }
                        if allocated {
                            self.waiting_counts.remove(&seq_id);
                        } else {
                            // This cannot fail after the capacity check and full preemption.
                            ignore_reason = Some(format!(
                                "Sequence {seq_id} with {num_tokens} tokens cannot be scheduled: \
                                 KV cache exhausted even after preempting all running sequences."
                            ));
                        }
                    } else {
                        break;
                    }
                }
            }

            if let Some(reason) = ignore_reason {
                self.fail_front_of_waiting(reason, "cache_exhausted", &mut prefix_validator);
                continue;
            }

            if let Some(validation) = prefix_validation {
                let mut seq_guard = get_mut_arcmutex!(seq);
                let commit_result = validation.commit(&mut seq_guard);
                drop(seq_guard);
                if let Err(err) = commit_result {
                    get_mut_arcmutex!(self.kv_cache_manager).free(seq_id);
                    self.fail_front_of_waiting(
                        format!("Failed to commit recurrent state for sequence {seq_id}: {err}"),
                        "recurrent_state",
                        &mut prefix_validator,
                    );
                    continue;
                }
            }

            metrics::counter!("mistralrs_prefix_cache_tokens_matched_total").increment(
                u64::try_from(matched_prefix_tokens).expect("prefix length exceeds u64"),
            );
            metrics::counter!("mistralrs_prefix_cache_tokens_reused_total")
                .increment(u64::try_from(num_computed).expect("prefix length exceeds u64"));
            if num_computed > 0 && get_mut_arcmutex!(seq).record_prefix_cache_hit() {
                logger.add_prefix_cache_hit();
            }
            let mut seq_guard = get_mut_arcmutex!(seq);
            seq_guard.set_state(SequenceState::RunningPrompt);
            seq_guard.set_prefix_cache_len(num_computed);
            seq_guard.set_num_computed_tokens(num_computed);
            drop(seq_guard);

            let seq = self.waiting.pop_front().unwrap();
            self.running.push_back(seq.clone());
            scheduled.push_back(seq);
        }
        self.waiting.extend(for_waiting_again);

        if !scheduled.is_empty() {
            let PromptBatch {
                scheduled,
                chunk_size: prompt_chunk_size,
                chunks: scheduled_prompt_chunks,
            } = self.select_prompt_batch(scheduled);

            // Rebuild num_cached_tokens from the bucketed sequences.
            // prefix_cache_len was set per-sequence above, so this stays aligned
            // even after bucketing removes sequences from non-contiguous positions.
            let num_cached_tokens: Vec<usize> = scheduled
                .iter()
                .map(|seq| get_mut_arcmutex!(seq).prefix_cache_len())
                .collect();

            logger.set_num_running(self.running.len());
            logger.set_num_waiting(self.waiting.len());
            self.publish_kv_block_metrics();
            self.decode_steps_since_prefill = 0;

            return PagedAttentionSchedulerOutput {
                scheduled: scheduled.into_iter().collect(),
                num_cached_tokens,
                prompt_chunk_size,
                scheduled_prompt_chunks,
            };
        }

        // Reserve completion token slots, preempting lowest priority first.

        let all_running = std::mem::take(&mut self.running);
        let mut prompt_running = VecDeque::new();
        for seq in all_running {
            if get_mut_arcmutex!(seq).is_prompt() {
                prompt_running.push_back(seq);
            } else {
                self.running.push_back(seq);
            }
        }

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
                let new_seq_modality = modality_signature(&*get_mut_arcmutex!(seq));
                if !self.requires_uniform_media_batch
                    || running.is_empty()
                    || modality_signature(&*get_mut_arcmutex!(running[0])) == new_seq_modality
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
            self.running
                .iter()
                .chain(prompt_running.iter())
                .for_each(|seq| {
                    get_mut_arcmutex!(seq).set_state(SequenceState::Done(StopReason::Canceled))
                });
            TERMINATE_ALL_NEXT_STEP.store(false, Ordering::SeqCst);
            self.running.extend(prompt_running);
            logger.set_num_running(self.running.len());
            logger.set_num_waiting(self.waiting.len());
            self.publish_kv_block_metrics();
            return PagedAttentionSchedulerOutput {
                scheduled: Vec::new(),
                num_cached_tokens: Vec::new(),
                prompt_chunk_size: None,
                scheduled_prompt_chunks: None,
            };
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

        self.record_decode_step();
        let scheduled = self.select_completion_batch();
        self.running.extend(prompt_running);
        logger.set_num_running(self.running.len());
        logger.set_num_waiting(self.waiting.len());
        self.publish_kv_block_metrics();

        PagedAttentionSchedulerOutput {
            scheduled,
            num_cached_tokens: Vec::new(), // No prefix cache for completion
            prompt_chunk_size: None,
            scheduled_prompt_chunks: None,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn publish_kv_block_metrics(&self) {
        let kv_mgr = get_mut_arcmutex!(self.kv_cache_manager);
        let total = kv_mgr.num_usable_blocks();
        let active = kv_mgr.num_active_blocks();
        metrics::gauge!("mistralrs_kv_cache_blocks_used").set(active as f64);
        metrics::gauge!("mistralrs_kv_cache_blocks_total").set(total as f64);
        metrics::gauge!("mistralrs_kv_cache_blocks_active").set(active as f64);
        metrics::gauge!("mistralrs_kv_cache_blocks_prefix_cached")
            .set(kv_mgr.num_prefix_cached_blocks() as f64);
        metrics::gauge!("mistralrs_kv_cache_blocks_prefix_retained")
            .set(kv_mgr.num_retained_prefix_blocks() as f64);
    }

    pub fn free_finished_sequence_groups(&mut self) {
        let mut finished: Vec<SeqCacheInfo> = Vec::new();
        let mut cacheable_finished: Vec<SeqCacheInfo> = Vec::new();
        for seq in self.running.iter().chain(self.waiting.iter()) {
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
                if !matches!(seq_guard.getstate(), SequenceState::Error)
                    && num_computed_tokens >= self.block_size
                {
                    cacheable_finished.push(info.clone());
                }
                finished.push(info);
            }
        }

        self.running
            .retain(|seq| !get_mut_arcmutex!(seq).is_finished_paged_attn());
        self.waiting
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
        drop(kv_mgr);
        self.finished_recurrent_slots.clear();
        self.publish_kv_block_metrics();
    }
}

impl PagedAttentionScheduler {
    fn _preempt(&mut self, seq: Arc<Mutex<Sequence>>) {
        let mut seq_guard = get_mut_arcmutex!(seq);
        // Don't resurrect sequences that are already in a terminal state
        if seq_guard.is_finished_paged_attn() {
            return;
        }
        metrics::counter!("mistralrs_paged_preemptions_total").increment(1);
        if seq_guard.active_staged_speculative_len() > 0 {
            metrics::counter!("mistralrs_speculative_staged_drops_total").increment(1);
        }
        seq_guard.set_state(SequenceState::Waiting);
        seq_guard.set_prefix_cache_len(0);
        seq_guard.clear_staged_speculative_tokens();
        let seq_id = *seq_guard.id();
        self.preempted_sequence_ids.push(seq_id);
        let tokens = seq_guard.get_toks().to_vec();
        let mm_features = seq_guard.mm_features().to_vec();
        let adapter_generation = seq_guard.adapter_generation();
        let block_hash_revision = seq_guard.block_hash_revision();
        let num_computed_tokens = seq_guard.num_computed_tokens();
        seq_guard.set_num_computed_tokens(0);
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

    fn take_deferred_speculative_releases(&mut self) -> Vec<usize> {
        let running_ids = self
            .running
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<HashSet<_>>();
        let mut released_ids = HashSet::new();
        std::mem::take(&mut self.preempted_sequence_ids)
            .into_iter()
            .filter(|seq_id| !running_ids.contains(seq_id) && released_ids.insert(*seq_id))
            .collect()
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
    fn cancel_closed_response_groups(&mut self) {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .for_each(|seq| {
                let seq = get_mut_arcmutex!(seq);
                if seq.response_is_closed() && !seq.is_finished_paged_attn() {
                    seq.set_state(SequenceState::Done(StopReason::Canceled));
                }
            });
    }
    fn schedule(
        &mut self,
        logger: &IntervalLogger,
        prefix_validator: Option<&mut dyn PagedPrefixCacheValidator>,
    ) -> SchedulerOutput<'_> {
        let output = self.schedule(logger, prefix_validator);
        let preempted_sequence_ids = self.take_deferred_speculative_releases();
        SchedulerOutput::PagedAttention {
            output,
            preempted_sequence_ids,
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
    fn get_finished_recurrent_slots(&self) -> Vec<(usize, usize)> {
        let mut slots = self.finished_recurrent_slots.clone();
        slots.extend(
            self.running
                .iter()
                .chain(self.waiting.iter())
                .filter_map(|seq| {
                    let seq = get_mut_arcmutex!(seq);
                    seq.is_finished_paged_attn()
                        .then(|| seq.recurrent_state_idx().map(|slot| (*seq.id(), slot)))
                        .flatten()
                }),
        );
        slots
    }
    fn get_finished_sequence_ids(&self) -> Vec<usize> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .filter(|seq| get_mut_arcmutex!(seq).is_finished_paged_attn())
            .map(|seq| *get_mut_arcmutex!(seq).id())
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
    fn set_requires_uniform_media_batch(&mut self, required: bool) {
        self.requires_uniform_media_batch = required;
    }
    fn set_supports_packed_prefill(&mut self, supported: bool) {
        self.supports_packed_prefill = supported;
    }
    fn set_scheduler_visible_prompt_chunks(
        &mut self,
        enabled: bool,
        require_block_alignment: bool,
        prefix_policy: SpeculativePrefixCheckpointPolicy,
    ) {
        self.scheduler_visible_prompt_chunks = enabled;
        self.prompt_chunks_require_block_alignment = require_block_alignment;
        self.prefix_policy = prefix_policy;
    }

    fn defer_prompt_tail(&mut self, first_omitted_sequence_id: usize) {
        self.next_prompt_sequence_id = Some(first_omitted_sequence_id);
    }

    fn set_waiting_prompt_preemption_enabled(&mut self, enabled: bool) {
        self.waiting_prompt_preemption_enabled = enabled;
    }

    fn can_continue_decode_batch(&self, sequence_ids: &[usize]) -> bool {
        self.can_continue_decode_batch_inner(
            sequence_ids,
            TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst),
        )
    }

    fn record_decode_continuation(&mut self) {
        let rows = self.live_completion_rows();
        let (_, next_cursor) = Self::completion_batch_indices(
            &rows,
            self.completion_cursor,
            self.config.max_num_batched_tokens,
        );
        self.completion_cursor = next_cursor;
        self.record_decode_step();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        paged_attention::{block_hash::MultimodalKind, PagedCacheType},
        sampler::Sampler,
        scheduler::IMAGE_MODALITY,
        sequence::{SeqStepType, SequenceGroup, SequenceRecognizer},
        speculative::SpeculativePrefixReplay,
        AudioInput, VideoInput,
    };
    use tokio::sync::{mpsc::channel, Mutex as TokioMutex};

    fn test_scheduler() -> PagedAttentionScheduler {
        PagedAttentionScheduler::new(
            PagedAttentionSchedulerConfig {
                max_num_seqs: 8,
                max_num_batched_tokens: 4096,
                max_prefill_chunk_tokens: 4096,
                max_decode_steps_before_prefill: 8,
            },
            CacheConfig {
                block_size: 8,
                num_gpu_blocks: 128,
                cache_type: PagedCacheType::Auto,
                kv_cache_group_ids: vec![0],
            },
        )
    }

    #[derive(Default)]
    struct RecordingPrefixValidator {
        cached_tokens: Vec<usize>,
        validated_ids: Vec<usize>,
        committed_ids: Arc<Mutex<Vec<usize>>>,
        released_slots: Vec<(usize, usize)>,
    }

    impl PagedPrefixCacheValidator for RecordingPrefixValidator {
        fn validate_prefix_cache_hit(
            &mut self,
            seq: &Sequence,
            _block_hashes: &[BlockHash],
            cached_tokens: usize,
            _block_size: usize,
        ) -> candle_core::Result<PagedPrefixCacheValidation> {
            self.cached_tokens.push(cached_tokens);
            let sequence_id = *seq.id();
            self.validated_ids.push(sequence_id);
            let committed_ids = Arc::clone(&self.committed_ids);
            Ok(PagedPrefixCacheValidation::staged(
                cached_tokens,
                move |_| {
                    get_mut_arcmutex!(committed_ids).push(sequence_id);
                    Ok(())
                },
            ))
        }

        fn release_recurrent_state(
            &mut self,
            sequence_id: usize,
            slot_idx: usize,
        ) -> candle_core::Result<bool> {
            self.released_slots.push((sequence_id, slot_idx));
            Ok(true)
        }
    }

    #[derive(Default)]
    struct FailingPrefixValidator {
        released_slots: Vec<(usize, usize)>,
    }

    impl PagedPrefixCacheValidator for FailingPrefixValidator {
        fn validate_prefix_cache_hit(
            &mut self,
            _seq: &Sequence,
            _block_hashes: &[BlockHash],
            _cached_tokens: usize,
            _block_size: usize,
        ) -> candle_core::Result<PagedPrefixCacheValidation> {
            candle_core::bail!("injected recurrent state reset failure")
        }

        fn release_recurrent_state(
            &mut self,
            sequence_id: usize,
            slot_idx: usize,
        ) -> candle_core::Result<bool> {
            self.released_slots.push((sequence_id, slot_idx));
            Ok(true)
        }
    }

    #[derive(Default)]
    struct FailingCommitPrefixValidator {
        released_slots: Vec<(usize, usize)>,
    }

    impl PagedPrefixCacheValidator for FailingCommitPrefixValidator {
        fn validate_prefix_cache_hit(
            &mut self,
            _seq: &Sequence,
            _block_hashes: &[BlockHash],
            cached_tokens: usize,
            _block_size: usize,
        ) -> candle_core::Result<PagedPrefixCacheValidation> {
            Ok(PagedPrefixCacheValidation::staged(cached_tokens, |_| {
                candle_core::bail!("injected recurrent state commit failure")
            }))
        }

        fn release_recurrent_state(
            &mut self,
            sequence_id: usize,
            slot_idx: usize,
        ) -> candle_core::Result<bool> {
            self.released_slots.push((sequence_id, slot_idx));
            Ok(true)
        }
    }

    type TestSequenceMedia = (
        Option<Vec<image::DynamicImage>>,
        Option<Vec<AudioInput>>,
        Option<Vec<VideoInput>>,
    );

    fn test_sequence_with_media_sender_and_group(
        id: usize,
        len: usize,
        input_media: TestSequenceMedia,
        tx: tokio::sync::mpsc::Sender<Response>,
        group: Arc<TokioMutex<SequenceGroup>>,
    ) -> Arc<Mutex<Sequence>> {
        let (input_images, input_audios, input_videos) = input_media;
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
            input_audios,
            input_videos,
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
        Arc::new(Mutex::new(seq))
    }

    fn test_sequence_with_media_and_sender(
        id: usize,
        len: usize,
        input_images: Option<Vec<image::DynamicImage>>,
        input_audios: Option<Vec<AudioInput>>,
        input_videos: Option<Vec<VideoInput>>,
        tx: tokio::sync::mpsc::Sender<Response>,
    ) -> Arc<Mutex<Sequence>> {
        test_sequence_with_media_sender_and_group(
            id,
            len,
            (input_images, input_audios, input_videos),
            tx,
            Arc::new(TokioMutex::new(SequenceGroup::new(1, false, true, None))),
        )
    }

    fn test_sequence_with_media_and_receiver(
        id: usize,
        len: usize,
        input_images: Option<Vec<image::DynamicImage>>,
        input_audios: Option<Vec<AudioInput>>,
        input_videos: Option<Vec<VideoInput>>,
    ) -> (Arc<Mutex<Sequence>>, tokio::sync::mpsc::Receiver<Response>) {
        let (tx, rx) = channel(1);
        (
            test_sequence_with_media_and_sender(
                id,
                len,
                input_images,
                input_audios,
                input_videos,
                tx,
            ),
            rx,
        )
    }

    fn test_sequence_with_media(
        id: usize,
        len: usize,
        input_images: Option<Vec<image::DynamicImage>>,
        input_audios: Option<Vec<AudioInput>>,
        input_videos: Option<Vec<VideoInput>>,
    ) -> Arc<Mutex<Sequence>> {
        test_sequence_with_media_and_receiver(id, len, input_images, input_audios, input_videos).0
    }

    fn test_sequence_with_images(
        id: usize,
        len: usize,
        input_images: Option<Vec<image::DynamicImage>>,
    ) -> Arc<Mutex<Sequence>> {
        test_sequence_with_media(id, len, input_images, None, None)
    }

    fn test_sequence(id: usize, len: usize) -> Arc<Mutex<Sequence>> {
        test_sequence_with_images(id, len, None)
    }

    fn test_audio_sequence(id: usize, len: usize) -> Arc<Mutex<Sequence>> {
        test_sequence_with_media(
            id,
            len,
            None,
            Some(vec![AudioInput {
                samples: vec![0.0],
                sample_rate: 16_000,
                channels: 1,
            }]),
            None,
        )
    }

    fn test_video_sequence(id: usize, len: usize) -> Arc<Mutex<Sequence>> {
        test_sequence_with_media(
            id,
            len,
            None,
            None,
            Some(vec![VideoInput::from_frames(
                vec![image::DynamicImage::new_rgb8(1, 1)],
                24.0,
                None,
            )]),
        )
    }

    #[test]
    fn preempted_prompt_validates_zero_token_recurrent_state() {
        let mut scheduler = test_scheduler();
        let seq = test_sequence(0, 4);
        {
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_prefix_cache_len(4);
            seq.set_num_computed_tokens(4);
            seq.set_recurrent_state_idx(Some(7));
        }
        scheduler._preempt(seq.clone());

        {
            let seq = get_mut_arcmutex!(seq);
            assert_eq!(seq.getstate(), SequenceState::Waiting);
            assert_eq!(seq.prefix_cache_len(), 0);
            assert_eq!(seq.num_computed_tokens(), 0);
        }

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut validator = RecordingPrefixValidator::default();
        let output = scheduler.schedule(&logger, Some(&mut validator));

        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(validator.cached_tokens, vec![0]);
    }

    #[test]
    fn scheduler_output_reports_preempted_sequence_ids() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_completion_batch = true;
        scheduler.running.push_back(test_sequence(10, 4));
        scheduler.running.push_back(test_sequence(20, 7));

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let output = Scheduler::schedule(&mut scheduler, &logger, None);
        let SchedulerOutput::PagedAttention {
            output,
            preempted_sequence_ids,
        } = output
        else {
            panic!("paged scheduler returned a default scheduler output");
        };

        assert_eq!(preempted_sequence_ids, vec![20]);
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 10);
    }

    #[test]
    fn readmitted_prefix_restore_supersedes_deferred_speculative_release() {
        let mut scheduler = test_scheduler();
        let seq = test_sequence(20, 16);
        scheduler._preempt(seq);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut validator = RecordingPrefixValidator::default();
        let committed_ids = Arc::clone(&validator.committed_ids);
        let SchedulerOutput::PagedAttention {
            output,
            preempted_sequence_ids,
        } = Scheduler::schedule(&mut scheduler, &logger, Some(&mut validator))
        else {
            panic!("paged scheduler returned a default scheduler output");
        };

        assert_eq!(&*get_mut_arcmutex!(committed_ids), &[20]);
        assert!(preempted_sequence_ids.is_empty());
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 20);
    }

    #[test]
    fn deferred_speculative_releases_follow_final_state_and_are_unique() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(20, 4));
        scheduler.preempted_sequence_ids = vec![20, 30, 20, 30];

        assert_eq!(scheduler.take_deferred_speculative_releases(), vec![30]);
        assert!(scheduler.preempted_sequence_ids.is_empty());
    }

    #[test]
    fn finished_sequence_ids_are_visible_before_cleanup() {
        let mut scheduler = test_scheduler();
        let live = test_sequence(10, 4);
        let finished = test_sequence(20, 4);
        get_mut_arcmutex!(finished).set_state(SequenceState::Done(StopReason::Eos));
        scheduler.running.push_back(live);
        scheduler.running.push_back(finished);

        assert_eq!(Scheduler::get_finished_sequence_ids(&scheduler), vec![20]);
        scheduler.free_finished_sequence_groups();
        assert!(Scheduler::get_finished_sequence_ids(&scheduler).is_empty());
    }

    #[test]
    fn closed_response_group_cancels_waiting_prefill_and_decode() {
        let mut scheduler = test_scheduler();
        let initial_free_blocks = get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks();
        let (tx, rx) = channel(1);
        let group = Arc::new(TokioMutex::new(SequenceGroup::new(3, false, true, None)));

        let waiting = test_sequence_with_media_sender_and_group(
            10,
            4,
            (None, None, None),
            tx.clone(),
            group.clone(),
        );
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        get_mut_arcmutex!(waiting).set_recurrent_state_idx(Some(10));
        scheduler.waiting.push_back(waiting.clone());

        let prefill = test_sequence_with_media_sender_and_group(
            20,
            4,
            (None, None, None),
            tx.clone(),
            group.clone(),
        );
        get_mut_arcmutex!(prefill).set_state(SequenceState::RunningPrompt);
        get_mut_arcmutex!(prefill).set_recurrent_state_idx(Some(20));
        assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
            .allocate_slots(20, 4, &[])
            .is_some());
        scheduler.running.push_back(prefill.clone());

        let decode =
            test_sequence_with_media_sender_and_group(30, 4, (None, None, None), tx, group);
        get_mut_arcmutex!(decode).set_recurrent_state_idx(Some(30));
        assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
            .allocate_slots(30, 4, &[])
            .is_some());
        scheduler.running.push_back(decode.clone());
        assert_eq!(
            get_mut_arcmutex!(scheduler.kv_cache_manager).num_active_blocks(),
            2
        );

        drop(rx);
        Scheduler::cancel_closed_response_groups(&mut scheduler);

        for seq in [&waiting, &prefill, &decode] {
            assert_eq!(
                get_mut_arcmutex!(seq).getstate(),
                SequenceState::Done(StopReason::Canceled)
            );
        }
        assert_eq!(
            Scheduler::get_finished_sequence_ids(&scheduler),
            vec![20, 30, 10]
        );
        assert_eq!(
            Scheduler::get_finished_recurrent_slots(&scheduler),
            vec![(20, 20), (30, 30), (10, 10)]
        );
        assert!(!scheduler.can_continue_decode_batch(&[30]));

        scheduler.free_finished_sequence_groups();
        assert!(scheduler.waiting.is_empty());
        assert!(scheduler.running.is_empty());
        assert_eq!(
            get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks(),
            initial_free_blocks
        );
        assert_eq!(
            get_mut_arcmutex!(scheduler.kv_cache_manager).num_active_blocks(),
            0
        );
    }

    #[test]
    fn closed_response_stops_resident_decode_continuation() {
        let mut scheduler = test_scheduler();
        let (seq, rx) = test_sequence_with_media_and_receiver(10, 4, None, None, None);
        scheduler.running.push_back(seq);
        assert!(scheduler.can_continue_decode_batch(&[10]));

        drop(rx);
        Scheduler::cancel_closed_response_groups(&mut scheduler);

        assert!(!scheduler.can_continue_decode_batch(&[10]));
    }

    #[test]
    fn prefix_validator_observes_the_clamped_cache_boundary() {
        let mut scheduler = test_scheduler();
        let tokens = vec![1; 24];
        let features = vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 12,
            length: 8,
            attention_policy: crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
            splittable: false,
        }];
        let hashes = compute_block_hashes(&tokens, scheduler.block_size, &features, &[]);
        {
            let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
            assert!(kv_mgr.allocate_slots(99, tokens.len(), &[]).is_some());
            kv_mgr.cache_blocks(99, &hashes, 16);
            kv_mgr.free(99);
        }

        let seq = test_sequence(0, tokens.len());
        {
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_state(SequenceState::Waiting);
            seq.set_mm_features(features);
        }
        scheduler.waiting.push_back(seq);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut validator = RecordingPrefixValidator::default();
        let output = scheduler.schedule(&logger, Some(&mut validator));

        assert_eq!(validator.cached_tokens, vec![8]);
        assert_eq!(output.num_cached_tokens, vec![8]);
    }

    #[test]
    fn cache_pressure_discards_staged_prefix_admission_until_retry_succeeds() {
        let mut scheduler = PagedAttentionScheduler::new(
            PagedAttentionSchedulerConfig {
                max_num_seqs: 8,
                max_num_batched_tokens: 4096,
                max_prefill_chunk_tokens: 4096,
                max_decode_steps_before_prefill: 8,
            },
            CacheConfig {
                block_size: 8,
                num_gpu_blocks: 2,
                cache_type: PagedCacheType::Auto,
                kv_cache_group_ids: vec![0],
            },
        );
        assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
            .allocate_slots(99, 8, &[])
            .is_some());
        let seq = test_sequence(0, 8);
        {
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_state(SequenceState::Waiting);
            seq.set_recurrent_state_idx(Some(7));
        }
        scheduler.waiting.push_back(seq);
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut validator = RecordingPrefixValidator::default();

        let blocked = scheduler.schedule(&logger, Some(&mut validator));

        assert!(blocked.scheduled.is_empty());
        assert_eq!(validator.validated_ids, vec![0]);
        assert!(get_mut_arcmutex!(validator.committed_ids).is_empty());
        assert_eq!(scheduler.waiting.len(), 1);

        get_mut_arcmutex!(scheduler.kv_cache_manager).free(99);
        let admitted = scheduler.schedule(&logger, Some(&mut validator));

        assert_eq!(admitted.scheduled.len(), 1);
        assert_eq!(validator.validated_ids, vec![0, 0]);
        assert_eq!(*get_mut_arcmutex!(validator.committed_ids), vec![0]);
    }

    #[test]
    fn disabled_waiting_prompt_preemption_preserves_decode_state() {
        let mut scheduler = PagedAttentionScheduler::new(
            PagedAttentionSchedulerConfig {
                max_num_seqs: 8,
                max_num_batched_tokens: 4096,
                max_prefill_chunk_tokens: 4096,
                max_decode_steps_before_prefill: 8,
            },
            CacheConfig {
                block_size: 8,
                num_gpu_blocks: 5,
                cache_type: PagedCacheType::Auto,
                kv_cache_group_ids: vec![0],
            },
        );
        scheduler.decode_steps_since_prefill = scheduler.config.max_decode_steps_before_prefill;

        let completion = test_sequence(10, 7);
        get_mut_arcmutex!(completion).set_num_computed_tokens(7);
        assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
            .allocate_slots(10, 7, &[])
            .is_some());
        scheduler.running.push_back(completion.clone());

        let waiting = test_sequence(20, 32);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting.clone());
        scheduler.waiting_counts.insert(20, WAITING_TIMEOUT);
        Scheduler::set_waiting_prompt_preemption_enabled(&mut scheduler, false);

        let blocks_before = get_mut_arcmutex!(scheduler.kv_cache_manager)
            .get_block_ids(10)
            .unwrap()
            .to_vec();
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let SchedulerOutput::PagedAttention {
            output,
            preempted_sequence_ids,
        } = Scheduler::schedule(&mut scheduler, &logger, None)
        else {
            panic!("paged scheduler returned a default scheduler output");
        };

        assert!(preempted_sequence_ids.is_empty());
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 10);
        assert_eq!(get_mut_arcmutex!(completion).num_computed_tokens(), 7);
        assert_eq!(
            get_mut_arcmutex!(completion).getstate(),
            SequenceState::RunningCompletion
        );
        assert_eq!(
            get_mut_arcmutex!(waiting).getstate(),
            SequenceState::Waiting
        );
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(
            get_mut_arcmutex!(scheduler.kv_cache_manager)
                .get_block_ids(10)
                .unwrap(),
            blocks_before
        );
        assert_eq!(
            scheduler.waiting_counts.get(&20),
            Some(&(WAITING_TIMEOUT + 1))
        );

        Scheduler::set_waiting_prompt_preemption_enabled(&mut scheduler, true);
        let SchedulerOutput::PagedAttention {
            output,
            preempted_sequence_ids,
        } = Scheduler::schedule(&mut scheduler, &logger, None)
        else {
            panic!("paged scheduler returned a default scheduler output");
        };

        assert_eq!(preempted_sequence_ids, vec![10]);
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 20);
        assert!(!scheduler.waiting_counts.contains_key(&20));
    }

    #[test]
    fn modality_requeue_does_not_stage_or_commit_prefix_admission() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_media_batch = true;
        let text = test_sequence(0, 8);
        let image =
            test_sequence_with_images(1, 8, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
        for seq in [&text, &image] {
            get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
        }
        scheduler.waiting.push_back(text);
        scheduler.waiting.push_back(image);
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut validator = RecordingPrefixValidator::default();

        let text_batch = scheduler.schedule(&logger, Some(&mut validator));

        assert_eq!(text_batch.scheduled.len(), 1);
        assert_eq!(validator.validated_ids, vec![0]);
        assert_eq!(*get_mut_arcmutex!(validator.committed_ids), vec![0]);
        assert_eq!(scheduler.waiting.len(), 1);

        get_mut_arcmutex!(scheduler.kv_cache_manager).free(0);
        scheduler.running.clear();
        let image_batch = scheduler.schedule(&logger, Some(&mut validator));

        assert_eq!(image_batch.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(image_batch.scheduled[0]).id(), 1);
        assert_eq!(validator.validated_ids, vec![0, 1]);
        assert_eq!(*get_mut_arcmutex!(validator.committed_ids), vec![0, 1]);
    }

    #[test]
    fn ignored_prompt_finishes_before_modality_requeue() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_media_batch = true;

        let first = test_sequence(0, 4);
        get_mut_arcmutex!(first).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(first);

        let (oversized, mut receiver) = test_sequence_with_media_and_receiver(
            1,
            1_024,
            Some(vec![image::DynamicImage::new_rgb8(1, 1)]),
            None,
            None,
        );
        {
            let mut oversized = get_mut_arcmutex!(oversized);
            oversized.set_state(SequenceState::Waiting);
            oversized.set_recurrent_state_idx(Some(7));
        }
        scheduler.waiting.push_back(oversized.clone());
        scheduler.waiting_counts.insert(1, WAITING_TIMEOUT);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut validator = RecordingPrefixValidator::default();
        let output = scheduler.schedule(&logger, Some(&mut validator));

        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 0);
        assert!(scheduler.waiting.is_empty());
        assert_eq!(
            get_mut_arcmutex!(oversized).getstate(),
            SequenceState::FinishedIgnored
        );
        assert_eq!(get_mut_arcmutex!(oversized).recurrent_state_idx(), None);
        assert_eq!(validator.released_slots, vec![(1, 7)]);
        assert!(matches!(
            receiver.try_recv(),
            Ok(Response::ValidationError(_))
        ));
    }

    #[test]
    fn recurrent_prefix_failure_rejects_request_and_releases_slot() {
        let mut scheduler = test_scheduler();
        let (seq, mut receiver) = test_sequence_with_media_and_receiver(1, 16, None, None, None);
        {
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_state(SequenceState::Waiting);
            seq.set_recurrent_state_idx(Some(7));
        }
        scheduler.waiting.push_back(seq.clone());

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut validator = FailingPrefixValidator::default();
        let output = scheduler.schedule(&logger, Some(&mut validator));

        assert!(output.scheduled.is_empty());
        assert!(scheduler.waiting.is_empty());
        assert!(scheduler.running.is_empty());
        assert_eq!(validator.released_slots, vec![(1, 7)]);
        assert_eq!(get_mut_arcmutex!(seq).recurrent_state_idx(), None);
        assert_eq!(
            get_mut_arcmutex!(seq).getstate(),
            SequenceState::FinishedIgnored
        );
        let response = receiver.try_recv().unwrap();
        assert!(matches!(response, Response::InternalError(_)));
    }

    #[test]
    fn recurrent_prefix_commit_failure_frees_kv_and_releases_slot() {
        let mut scheduler = test_scheduler();
        let initial_free_blocks = get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks();
        let (seq, mut receiver) = test_sequence_with_media_and_receiver(1, 16, None, None, None);
        {
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_state(SequenceState::Waiting);
            seq.set_recurrent_state_idx(Some(7));
        }
        scheduler.waiting.push_back(seq.clone());

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut validator = FailingCommitPrefixValidator::default();
        let output = scheduler.schedule(&logger, Some(&mut validator));

        assert!(output.scheduled.is_empty());
        assert!(scheduler.waiting.is_empty());
        assert!(scheduler.running.is_empty());
        assert_eq!(validator.released_slots, vec![(1, 7)]);
        assert_eq!(get_mut_arcmutex!(seq).recurrent_state_idx(), None);
        assert_eq!(
            get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks(),
            initial_free_blocks
        );
        assert!(matches!(
            receiver.try_recv(),
            Ok(Response::InternalError(_))
        ));
    }

    #[test]
    fn ragged_completion_batch_keeps_all_sequences_running() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_completion_batch = false;
        scheduler.requires_uniform_media_batch = false;
        scheduler.running.push_back(test_sequence(0, 4));
        scheduler.running.push_back(test_sequence(1, 7));

        scheduler.enforce_completion_compatibility();

        assert_eq!(scheduler.running.len(), 2);
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn ragged_completion_batch_separates_incompatible_media() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_completion_batch = false;
        scheduler.requires_uniform_media_batch = true;
        scheduler.running.push_back(test_sequence(0, 4));
        scheduler.running.push_back(test_sequence_with_images(
            1,
            7,
            Some(vec![image::DynamicImage::new_rgb8(1, 1)]),
        ));

        scheduler.enforce_completion_compatibility();

        assert_eq!(scheduler.running.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn completion_media_signature_survives_consumed_prompt_inputs() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_completion_batch = false;
        scheduler.requires_uniform_media_batch = true;
        scheduler.running.push_back(test_sequence(0, 4));
        let image =
            test_sequence_with_images(1, 7, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
        {
            let mut image = get_mut_arcmutex!(image);
            image.set_mm_features(vec![MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![1],
                offset: 0,
                length: 1,
                attention_policy:
                    crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
                splittable: false,
            }]);
            image.multimodal.has_changed_prompt = true;
            assert_eq!(image.take_images().unwrap().len(), 1);
            assert!(!image.has_images());
            assert_eq!(modality_signature(&image), IMAGE_MODALITY);
        }
        scheduler.running.push_back(image);

        scheduler.enforce_completion_compatibility();

        assert_eq!(scheduler.running.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
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
    fn ragged_prompt_batch_mixes_media_and_text_sequences() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_media_batch = false;
        let prompts = VecDeque::from([
            test_sequence(0, 4),
            test_sequence_with_images(1, 7, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
        ]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

        assert_eq!(scheduled.len(), 2);
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn ragged_prompt_batch_mixes_distinct_media_modalities() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_media_batch = false;
        let prompts = VecDeque::from([
            test_sequence_with_images(0, 4, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
            test_audio_sequence(1, 7),
            test_video_sequence(2, 5),
        ]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

        assert_eq!(scheduled.len(), 3);
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn uniform_prompt_batch_separates_media_and_text_sequences() {
        let mut scheduler = test_scheduler();
        let prompts = VecDeque::from([
            test_sequence(0, 4),
            test_sequence_with_images(1, 4, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
        ]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn ragged_prompt_batch_separates_incompatible_media() {
        let mut scheduler = test_scheduler();
        scheduler.requires_uniform_prompt_batch = false;
        scheduler.requires_uniform_media_batch = true;
        let prompts = VecDeque::from([
            test_sequence(0, 4),
            test_sequence_with_images(1, 7, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
        ]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn uniform_prompt_batch_separates_image_and_audio_sequences() {
        let mut scheduler = test_scheduler();
        let prompts = VecDeque::from([
            test_sequence_with_images(0, 4, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
            test_audio_sequence(1, 4),
        ]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn uniform_prompt_batch_separates_image_and_video_sequences() {
        let mut scheduler = test_scheduler();
        let prompts = VecDeque::from([
            test_sequence_with_images(0, 4, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
            test_video_sequence(1, 4),
        ]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn ragged_prompt_batch_keeps_unequal_lengths() {
        let mut scheduler = test_scheduler();
        scheduler.supports_packed_prefill = true;
        let prompts = VecDeque::from([
            test_sequence(0, 4),
            test_sequence(1, 300),
            test_sequence(2, 600),
        ]);

        for seq in &prompts {
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
        }
        let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

        let scheduled_ids: Vec<_> = scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect();
        assert_eq!(scheduled_ids, vec![0, 1, 2]);
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn ragged_prompt_batch_preserves_fcfs_order() {
        let mut scheduler = test_scheduler();
        scheduler.supports_packed_prefill = true;
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

        let scheduled_ids: Vec<_> = scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect();
        assert_eq!(scheduled_ids, vec![0, 1, 2, 3]);
        let waiting_ids: Vec<_> = scheduler
            .waiting
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect();
        assert_eq!(waiting_ids, vec![4]);
    }

    #[test]
    fn padded_ragged_prompt_batch_bounds_padding() {
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
    fn raw_logits_prompt_bypasses_reusable_prefix_blocks() {
        let mut scheduler = test_scheduler();
        let tokens = vec![1; 16];
        let hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
        {
            let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
            assert!(kv_mgr.allocate_slots(99, tokens.len(), &[]).is_some());
            kv_mgr.cache_blocks(99, &hashes, scheduler.block_size);
            kv_mgr.free(99);
            assert_eq!(
                kv_mgr
                    .get_computed_blocks(&hashes, tokens.len())
                    .num_computed_tokens,
                scheduler.block_size
            );
        }

        let raw = test_sequence(0, tokens.len());
        get_mut_arcmutex!(raw).return_raw_logits = true;
        get_mut_arcmutex!(raw).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(raw);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let output = scheduler.schedule(&logger, None);

        assert_eq!(output.num_cached_tokens, vec![0]);
        assert_eq!(get_mut_arcmutex!(output.scheduled[0]).prefix_cache_len(), 0);
        let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert_eq!(
            kv_mgr
                .get_computed_blocks(&hashes, tokens.len())
                .num_computed_tokens,
            scheduler.block_size
        );
    }

    #[test]
    fn preempted_prefix_cache_hit_is_counted_once() {
        let mut scheduler = test_scheduler();
        let tokens = vec![1; 16];
        let hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
        {
            let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
            assert!(kv_mgr.allocate_slots(99, tokens.len(), &[]).is_some());
            kv_mgr.cache_blocks(99, &hashes, scheduler.block_size);
            kv_mgr.free(99);
        }

        let seq = test_sequence(0, tokens.len());
        get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(seq);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        logger.add_new_sequence();
        let first = scheduler.schedule(&logger, None);

        assert_eq!(first.num_cached_tokens, vec![scheduler.block_size]);
        assert_eq!(logger.prefix_cache_stats(), (1, 1));

        let seq = first.scheduled[0].clone();
        scheduler.running.clear();
        scheduler._preempt(seq);
        let second = scheduler.schedule(&logger, None);

        assert_eq!(second.num_cached_tokens, vec![scheduler.block_size]);
        assert_eq!(logger.prefix_cache_stats(), (1, 1));
    }

    #[test]
    fn underfilled_decode_gets_one_completion_turn_before_refill() {
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
    fn running_prompt_tail_within_budget_yields_after_prefill() {
        let mut scheduler = test_scheduler();
        let completion = test_sequence(0, 8);
        get_mut_arcmutex!(completion).set_num_computed_tokens(8);
        scheduler.running.push_back(completion);

        let prompt = test_sequence(1, 12);
        {
            let mut prompt = get_mut_arcmutex!(prompt);
            prompt.set_state(SequenceState::RunningPrompt);
            prompt.set_prefix_cache_len(8);
            prompt.set_num_computed_tokens(8);
        }
        scheduler.running.push_back(prompt);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let decode = scheduler.schedule(&logger, None);
        assert_eq!(decode.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(decode.scheduled[0]).id(), 0);

        let prompt = scheduler.schedule(&logger, None);
        assert_eq!(prompt.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(prompt.scheduled[0]).id(), 1);
        assert!(get_mut_arcmutex!(prompt.scheduled[0]).is_prompt());
    }

    #[test]
    fn incompatible_prompt_tails_yield_between_batches() {
        let mut scheduler = test_scheduler();
        scheduler.config.max_num_batched_tokens = 8;
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.running.push_back(test_sequence(0, 8));

        for (id, len) in [(1, 10), (2, 11)] {
            let prompt = test_sequence(id, len);
            {
                let mut prompt = get_mut_arcmutex!(prompt);
                prompt.set_state(SequenceState::RunningPrompt);
                prompt.set_prefix_cache_len(8);
                prompt.set_num_computed_tokens(8);
            }
            scheduler.running.push_back(prompt);
        }

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let decode = scheduler.schedule(&logger, None);
        assert_eq!(*get_mut_arcmutex!(decode.scheduled[0]).id(), 0);

        let first = scheduler.schedule(&logger, None);
        assert_eq!(first.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(first.scheduled[0]).id(), 1);
        get_mut_arcmutex!(first.scheduled[0]).set_num_computed_tokens(10);

        let decode = scheduler.schedule(&logger, None);
        assert_eq!(*get_mut_arcmutex!(decode.scheduled[0]).id(), 0);

        let second = scheduler.schedule(&logger, None);
        assert_eq!(second.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(second.scheduled[0]).id(), 2);
        assert!(get_mut_arcmutex!(second.scheduled[0]).is_prompt());
    }

    #[test]
    fn chunked_prompt_buckets_rotate_without_discarding_partial_state() {
        let mut scheduler = test_scheduler();
        scheduler.config.max_num_batched_tokens = 4;
        scheduler.scheduler_visible_prompt_chunks = true;

        for (id, computed) in [(0, 4), (1, 0), (2, 4)] {
            let prompt = test_sequence(id, 12);
            {
                let mut prompt = get_mut_arcmutex!(prompt);
                prompt.set_state(SequenceState::RunningPrompt);
                prompt.set_prefix_cache_len(computed);
                prompt.set_num_computed_tokens(computed);
            }
            scheduler.running.push_back(prompt);
        }

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let first = scheduler.schedule(&logger, None);
        assert_eq!(
            first
                .scheduled
                .iter()
                .map(|seq| *get_mut_arcmutex!(seq).id())
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        for seq in &first.scheduled {
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_prefix_cache_len(6);
            seq.set_num_computed_tokens(6);
        }

        let second = scheduler.schedule(&logger, None);
        assert_eq!(second.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(second.scheduled[0]).id(), 1);
        assert_eq!(scheduler.running.len(), 3);
        assert!(scheduler.waiting.is_empty());
        assert_eq!(
            scheduler
                .running
                .iter()
                .map(|seq| get_mut_arcmutex!(seq).num_computed_tokens())
                .collect::<Vec<_>>(),
            vec![6, 0, 6]
        );

        {
            let mut seq = get_mut_arcmutex!(second.scheduled[0]);
            seq.set_prefix_cache_len(4);
            seq.set_num_computed_tokens(4);
        }
        let third = scheduler.schedule(&logger, None);
        assert_eq!(
            third
                .scheduled
                .iter()
                .map(|seq| *get_mut_arcmutex!(seq).id())
                .collect::<Vec<_>>(),
            vec![2, 0]
        );
    }

    #[test]
    fn active_decodes_are_prioritized_over_new_prompts() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(0, 4));

        let waiting = test_sequence(1, 7);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting);

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let completion = scheduler.schedule(&logger, None);

        assert_eq!(completion.scheduled.len(), 1);
        assert!(!get_mut_arcmutex!(completion.scheduled[0]).is_prompt());
        assert_eq!(scheduler.waiting.len(), 1);
    }

    #[test]
    fn completion_turn_survives_finished_prompt_cleanup() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(0, 4));
        scheduler.decode_steps_since_prefill = 0;

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
        scheduler.decode_steps_since_prefill = 0;

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
    fn prompt_chunk_size_stays_within_the_batch_token_budget() {
        let mut scheduler = test_scheduler();
        scheduler.config.max_num_batched_tokens = 4096;

        assert_eq!(scheduler.prompt_chunk_size(0), None);
        assert_eq!(scheduler.prompt_chunk_size(1), Some(4096));
        assert_eq!(scheduler.prompt_chunk_size(8), Some(512));
        assert_eq!(scheduler.prompt_chunk_size(16), Some(256));
        assert_eq!(scheduler.prompt_chunk_size(7), Some(585));
        assert!(scheduler.prompt_chunk_size(7).unwrap() * 7 <= 4096);
    }

    #[test]
    fn prompt_chunk_size_uses_the_prefill_latency_budget() {
        let mut scheduler = test_scheduler();
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.config.max_num_batched_tokens = 4096;
        scheduler.config.max_prefill_chunk_tokens = 512;

        assert_eq!(scheduler.prompt_chunk_size(1), Some(512));
        assert_eq!(scheduler.prompt_chunk_size(8), Some(64));
        assert_eq!(scheduler.prompt_chunk_size(16), Some(32));
    }

    #[test]
    fn prefill_latency_budget_does_not_change_atomic_prompt_paths() {
        let mut scheduler = test_scheduler();
        scheduler.config.max_num_batched_tokens = 4096;
        scheduler.config.max_prefill_chunk_tokens = 512;

        assert_eq!(scheduler.prompt_chunk_size(1), Some(4096));
    }

    #[test]
    fn completion_priority_uses_the_prefill_latency_budget() {
        let mut scheduler = test_scheduler();
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.config.max_num_batched_tokens = 4096;
        scheduler.config.max_prefill_chunk_tokens = 512;
        scheduler.running.push_back(test_sequence(0, 4));
        let prompt = test_sequence(1, 1024);
        get_mut_arcmutex!(prompt).set_state(SequenceState::RunningPrompt);
        scheduler.running.push_back(prompt.clone());

        assert!(scheduler.completion_is_due());

        get_mut_arcmutex!(prompt).set_num_computed_tokens(512);
        assert!(scheduler.completion_is_due());

        scheduler.decode_steps_since_prefill = 1;
        assert!(!scheduler.completion_is_due());
    }

    #[test]
    fn prompt_batch_does_not_exceed_the_prefill_latency_budget() {
        let mut scheduler = test_scheduler();
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.config.max_num_batched_tokens = 16;
        scheduler.config.max_prefill_chunk_tokens = 2;
        let candidates = (0..3)
            .map(|id| {
                let seq = test_sequence(id, 8);
                get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
                seq
            })
            .collect::<VecDeque<_>>();

        let batch = scheduler.select_prompt_batch(candidates);

        assert_eq!(batch.scheduled.len(), 2);
        assert_eq!(batch.chunk_size, Some(1));
        assert_eq!(
            batch
                .chunks
                .unwrap()
                .into_iter()
                .map(|chunk| chunk.end - chunk.start)
                .sum::<usize>(),
            2
        );
    }

    #[test]
    fn token_budget_caps_prompt_admission() {
        let mut scheduler = test_scheduler();
        scheduler.config.max_num_batched_tokens = 3;
        for id in 0..5 {
            let seq = test_sequence(id, 4);
            get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
            scheduler.waiting.push_back(seq);
        }

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let prompt = scheduler.schedule(&logger, None);

        assert_eq!(prompt.scheduled.len(), 3);
        assert_eq!(prompt.prompt_chunk_size, Some(1));
        assert_eq!(scheduler.waiting.len(), 2);
    }

    #[test]
    fn deferred_prompt_tail_runs_first_without_reallocating_kv() {
        let mut scheduler = test_scheduler();
        Scheduler::set_scheduler_visible_prompt_chunks(
            &mut scheduler,
            true,
            false,
            SpeculativePrefixCheckpointPolicy::default(),
        );
        for id in 0..3 {
            let seq = test_sequence(id, 16);
            get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
            scheduler.waiting.push_back(seq);
        }

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let mut first = scheduler.schedule(&logger, None);
        assert_eq!(
            first
                .scheduled
                .iter()
                .map(|seq| *get_mut_arcmutex!(seq).id())
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let first_omitted = first.retain_prompt_prefix(1).unwrap();
        assert_eq!(first_omitted, 1);
        assert_eq!(first.scheduled.len(), 1);
        assert_eq!(first.num_cached_tokens.len(), 1);
        assert_eq!(first.scheduled_prompt_chunks.as_ref().unwrap().len(), 1);

        let kv_before = {
            let kv_manager = get_mut_arcmutex!(scheduler.kv_cache_manager);
            (0..3)
                .map(|id| kv_manager.get_block_ids(id).unwrap().to_vec())
                .collect::<Vec<_>>()
        };
        let states_before = scheduler
            .running
            .iter()
            .map(|seq| get_mut_arcmutex!(seq).getstate())
            .collect::<Vec<_>>();

        Scheduler::defer_prompt_tail(&mut scheduler, first_omitted);

        let kv_after_deferral = {
            let kv_manager = get_mut_arcmutex!(scheduler.kv_cache_manager);
            (0..3)
                .map(|id| kv_manager.get_block_ids(id).unwrap().to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(kv_after_deferral, kv_before);
        assert_eq!(
            scheduler
                .running
                .iter()
                .map(|seq| get_mut_arcmutex!(seq).getstate())
                .collect::<Vec<_>>(),
            states_before
        );

        let next = scheduler.schedule(&logger, None);
        assert_eq!(
            next.scheduled
                .iter()
                .map(|seq| *get_mut_arcmutex!(seq).id())
                .collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        let kv_after_schedule = {
            let kv_manager = get_mut_arcmutex!(scheduler.kv_cache_manager);
            (0..3)
                .map(|id| kv_manager.get_block_ids(id).unwrap().to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(kv_after_schedule, kv_before);
        assert_eq!(
            scheduler
                .running
                .iter()
                .map(|seq| get_mut_arcmutex!(seq).getstate())
                .collect::<Vec<_>>(),
            states_before
        );
    }

    #[test]
    fn token_budget_fairly_rotates_completion_batches() {
        let mut scheduler = test_scheduler();
        scheduler.config.max_num_batched_tokens = 2;
        for id in 0..3 {
            let seq = test_sequence(id, 4);
            get_mut_arcmutex!(seq).set_num_computed_tokens(4);
            scheduler.running.push_back(seq);
        }

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let first = scheduler.schedule(&logger, None);
        let first_ids = first
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>();
        assert_eq!(first_ids, vec![0, 1]);

        let second = scheduler.schedule(&logger, None);
        let second_ids = second
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>();
        assert_eq!(second_ids, vec![2, 0]);
    }

    #[test]
    fn completion_batches_bootstrap_new_rows_without_dropping_staged_rows() {
        let mut scheduler = test_scheduler();
        for id in 0..2 {
            let seq = test_sequence(id, 4);
            let mut seq_guard = get_mut_arcmutex!(seq);
            seq_guard.set_num_computed_tokens(4);
            seq_guard.set_staged_speculative(vec![10, 11], None);
            drop(seq_guard);
            scheduler.running.push_back(seq);
        }
        let newcomer = test_sequence(2, 4);
        get_mut_arcmutex!(newcomer).set_num_computed_tokens(4);
        scheduler.running.push_back(newcomer.clone());

        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
        let staged = scheduler.schedule(&logger, None);
        let staged_ids = staged
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>();
        assert_eq!(staged_ids, vec![0, 1]);
        assert_eq!(
            scheduler
                .running
                .iter()
                .map(|seq| get_mut_arcmutex!(seq).active_staged_speculative_len())
                .collect::<Vec<_>>(),
            vec![2, 2, 0]
        );

        let bootstrap = scheduler.schedule(&logger, None);
        assert_eq!(bootstrap.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(bootstrap.scheduled[0]).id(), 2);
        get_mut_arcmutex!(newcomer).set_staged_speculative(vec![12, 13], None);

        let joined = scheduler.schedule(&logger, None);
        let joined_ids = joined
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>();
        assert_eq!(joined_ids, vec![0, 1, 2]);
    }

    #[test]
    fn resident_decode_continuation_requires_exact_live_batch() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(0, 4));
        scheduler.running.push_back(test_sequence(1, 4));

        let cursor = scheduler.completion_cursor;
        let decode_steps = scheduler.decode_steps_since_prefill;
        assert!(scheduler.can_continue_decode_batch(&[0, 1]));
        assert!(!scheduler.can_continue_decode_batch(&[1, 0]));
        assert!(!scheduler.can_continue_decode_batch(&[0]));
        assert!(!scheduler.can_continue_decode_batch(&[0, 1, 2]));
        assert_eq!(scheduler.completion_cursor, cursor);
        assert_eq!(scheduler.decode_steps_since_prefill, decode_steps);

        get_mut_arcmutex!(scheduler.running[1])
            .set_state(SequenceState::Done(StopReason::Canceled));
        assert!(!scheduler.can_continue_decode_batch(&[0, 1]));
    }

    #[test]
    fn resident_decode_continuation_respects_prompt_fairness() {
        let mut scheduler = test_scheduler();
        for id in 0..scheduler.config.max_num_seqs {
            scheduler.running.push_back(test_sequence(id, 4));
        }
        let waiting = test_sequence(scheduler.config.max_num_seqs, 4);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting);
        scheduler.decode_steps_since_prefill = scheduler.config.max_decode_steps_before_prefill - 1;
        let sequence_ids = (0..scheduler.config.max_num_seqs).collect::<Vec<_>>();

        assert!(scheduler.can_continue_decode_batch(&sequence_ids));
        let before = scheduler.decode_steps_since_prefill;
        scheduler.record_decode_continuation();
        assert_eq!(scheduler.decode_steps_since_prefill, before + 1);
        assert!(!scheduler.can_continue_decode_batch(&sequence_ids));
    }

    #[test]
    fn underfilled_resident_decode_yields_after_one_turn() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(0, 4));
        let waiting = test_sequence(1, 4);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting);
        scheduler.decode_steps_since_prefill = 1;

        let cursor = scheduler.completion_cursor;
        let steps = scheduler.decode_steps_since_prefill;
        assert!(!scheduler.can_continue_decode_batch(&[0]));
        assert_eq!(scheduler.completion_cursor, cursor);
        assert_eq!(scheduler.decode_steps_since_prefill, steps);
    }

    #[test]
    fn kv_blocked_prompt_does_not_interrupt_resident_decode() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(0, 4));
        let waiting = test_sequence(1, 4);
        get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(waiting);
        scheduler.decode_steps_since_prefill = 1;
        {
            let mut kv_manager = get_mut_arcmutex!(scheduler.kv_cache_manager);
            let free_tokens = kv_manager.num_free_blocks() * scheduler.block_size;
            assert!(kv_manager.allocate_slots(99, free_tokens, &[]).is_some());
            assert_eq!(kv_manager.num_free_blocks(), 0);
        }

        assert!(scheduler.can_continue_decode_batch(&[0]));
    }

    #[test]
    fn resident_decode_continuation_stops_for_pending_termination() {
        let mut scheduler = test_scheduler();
        scheduler.running.push_back(test_sequence(0, 4));

        assert!(scheduler.can_continue_decode_batch_inner(&[0], false));
        assert!(!scheduler.can_continue_decode_batch_inner(&[0], true));
    }

    #[test]
    fn resident_decode_continuation_preserves_subset_rotation() {
        let mut scheduler = test_scheduler();
        scheduler.config.max_num_batched_tokens = 2;
        for id in 0..3 {
            let seq = test_sequence(id, 4);
            get_mut_arcmutex!(seq).set_num_computed_tokens(4);
            scheduler.running.push_back(seq);
        }

        assert!(scheduler.can_continue_decode_batch(&[0, 1]));
        assert_eq!(scheduler.completion_cursor, 0);
        scheduler.record_decode_continuation();
        assert_eq!(scheduler.completion_cursor, 2);
        assert!(!scheduler.can_continue_decode_batch(&[0, 1]));
        assert!(scheduler.can_continue_decode_batch(&[2, 0]));

        scheduler.record_decode_continuation();
        assert_eq!(scheduler.completion_cursor, 1);
        assert!(scheduler.can_continue_decode_batch(&[1, 2]));
    }

    #[test]
    fn scheduler_visible_prefill_uses_exact_long_prompt_ranges() {
        let mut scheduler = test_scheduler();
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.config.max_num_batched_tokens = 4;
        let seq = test_sequence(0, 10);
        get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(seq.clone());
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

        for expected in [(0, 4), (4, 8), (8, 10)] {
            let output = scheduler.schedule(&logger, None);
            let chunks = output.scheduled_prompt_chunks.unwrap();
            assert_eq!(output.scheduled.len(), 1);
            assert_eq!((chunks[0].start, chunks[0].end), expected);
            assert_eq!(chunks[0].end - chunks[0].start, expected.1 - expected.0);
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_prefix_cache_len(chunks[0].end);
            seq.set_num_computed_tokens(chunks[0].end);
        }
    }

    #[test]
    fn hybrid_prefill_schedules_the_maximum_reusable_prefix_boundary() {
        let mut scheduler = test_scheduler();
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.prompt_chunks_require_block_alignment = true;
        scheduler.config.max_num_batched_tokens = 32;
        let seq = test_sequence(0, 64);
        get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(seq.clone());
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

        for expected in [(0, 32), (32, 56), (56, 64)] {
            let output = scheduler.schedule(&logger, None);
            let chunks = output.scheduled_prompt_chunks.unwrap();
            assert_eq!(output.scheduled.len(), 1);
            assert_eq!((chunks[0].start, chunks[0].end), expected);
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_prefix_cache_len(chunks[0].end);
            seq.set_num_computed_tokens(chunks[0].end);
        }
    }

    #[test]
    fn hybrid_prefill_schedules_the_suffix_replay_boundary() {
        let mut scheduler = test_scheduler();
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.prompt_chunks_require_block_alignment = true;
        scheduler.prefix_policy =
            SpeculativePrefixCheckpointPolicy::new(SpeculativePrefixReplay::Suffix(16), false);
        scheduler.config.max_num_batched_tokens = 32;
        let seq = test_sequence(0, 64);
        get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(seq.clone());
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

        for expected in [(0, 32), (32, 40), (40, 64)] {
            let output = scheduler.schedule(&logger, None);
            let chunks = output.scheduled_prompt_chunks.unwrap();
            assert_eq!(output.scheduled.len(), 1);
            assert_eq!((chunks[0].start, chunks[0].end), expected);
            let mut seq = get_mut_arcmutex!(seq);
            seq.set_prefix_cache_len(chunks[0].end);
            seq.set_num_computed_tokens(chunks[0].end);
        }
    }

    #[test]
    fn text_auxiliary_policy_keeps_multimodal_suffix_replay() {
        let policy =
            SpeculativePrefixCheckpointPolicy::new(SpeculativePrefixReplay::Suffix(16), true);
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

        let mut text_scheduler = test_scheduler();
        text_scheduler.scheduler_visible_prompt_chunks = true;
        text_scheduler.prompt_chunks_require_block_alignment = true;
        text_scheduler.prefix_policy = policy;
        text_scheduler.config.max_num_batched_tokens = 32;
        let text = test_sequence(0, 64);
        get_mut_arcmutex!(text).set_state(SequenceState::Waiting);
        text_scheduler.waiting.push_back(text.clone());
        let first = text_scheduler.schedule(&logger, None);
        let first_chunk = first.scheduled_prompt_chunks.unwrap()[0];
        get_mut_arcmutex!(text).set_num_computed_tokens(first_chunk.end);
        let second = text_scheduler.schedule(&logger, None);
        assert_eq!(second.scheduled_prompt_chunks.unwrap()[0].end, 56);

        let mut media_scheduler = test_scheduler();
        media_scheduler.scheduler_visible_prompt_chunks = true;
        media_scheduler.prompt_chunks_require_block_alignment = true;
        media_scheduler.prefix_policy = policy;
        media_scheduler.config.max_num_batched_tokens = 32;
        let media =
            test_sequence_with_images(1, 64, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
        assert_eq!(
            policy.replay_for(modality_signature(&*get_mut_arcmutex!(media))),
            SpeculativePrefixReplay::Suffix(16)
        );
        {
            let mut media = get_mut_arcmutex!(media);
            media.set_mm_features(vec![MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![1],
                offset: 0,
                length: 8,
                attention_policy:
                    crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
                splittable: false,
            }]);
            media.set_state(SequenceState::Waiting);
        }
        media_scheduler.waiting.push_back(media.clone());
        let first = media_scheduler.schedule(&logger, None);
        let first_chunk = first.scheduled_prompt_chunks.unwrap()[0];
        get_mut_arcmutex!(media).set_num_computed_tokens(first_chunk.end);
        let second = media_scheduler.schedule(&logger, None);
        assert_eq!(second.scheduled_prompt_chunks.unwrap()[0].end, 40);
    }

    #[test]
    fn partial_prefill_runs_after_the_decode_turn_budget() {
        let mut scheduler = test_scheduler();
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.config.max_num_batched_tokens = 4;
        scheduler.config.max_decode_steps_before_prefill = 3;

        let completion = test_sequence(0, 8);
        get_mut_arcmutex!(completion).set_num_computed_tokens(8);
        scheduler.running.push_back(completion);
        let prompt = test_sequence(1, 12);
        {
            let mut prompt = get_mut_arcmutex!(prompt);
            prompt.set_state(SequenceState::RunningPrompt);
            prompt.set_prefix_cache_len(4);
            prompt.set_num_computed_tokens(4);
        }
        scheduler.running.push_back(prompt);
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

        for _ in 0..scheduler.config.max_decode_steps_before_prefill {
            let output = scheduler.schedule(&logger, None);
            assert_eq!(output.scheduled.len(), 1);
            assert!(get_mut_arcmutex!(output.scheduled[0]).is_completion());
            assert!(output.scheduled_prompt_chunks.is_none());
        }

        let output = scheduler.schedule(&logger, None);
        assert_eq!(output.scheduled.len(), 1);
        assert!(get_mut_arcmutex!(output.scheduled[0]).is_prompt());
        let chunks = output.scheduled_prompt_chunks.unwrap();
        assert_eq!((chunks[0].start, chunks[0].end), (4, 8));
    }

    #[test]
    fn partial_prefill_yields_to_decode_between_quanta() {
        let mut scheduler = test_scheduler();
        scheduler.scheduler_visible_prompt_chunks = true;
        scheduler.config.max_num_batched_tokens = 16;
        scheduler.config.max_prefill_chunk_tokens = 4;
        scheduler.config.max_decode_steps_before_prefill = 3;

        let completion = test_sequence(0, 8);
        get_mut_arcmutex!(completion).set_num_computed_tokens(8);
        scheduler.running.push_back(completion);
        let prompt = test_sequence(1, 12);
        {
            let mut prompt = get_mut_arcmutex!(prompt);
            prompt.set_state(SequenceState::RunningPrompt);
            prompt.set_prefix_cache_len(4);
            prompt.set_num_computed_tokens(4);
        }
        scheduler.running.push_back(prompt.clone());
        scheduler.decode_steps_since_prefill = scheduler.config.max_decode_steps_before_prefill;
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

        let prefill = scheduler.schedule(&logger, None);
        let chunks = prefill.scheduled_prompt_chunks.unwrap();
        assert_eq!((chunks[0].start, chunks[0].end), (4, 8));
        get_mut_arcmutex!(prompt).set_num_computed_tokens(8);

        let decode = scheduler.schedule(&logger, None);
        assert_eq!(decode.scheduled.len(), 1);
        assert_eq!(*get_mut_arcmutex!(decode.scheduled[0]).id(), 0);
        assert!(decode.scheduled_prompt_chunks.is_none());
    }

    #[test]
    fn unsupported_prompt_paths_remain_atomic() {
        let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

        let mut raw_scheduler = test_scheduler();
        raw_scheduler.scheduler_visible_prompt_chunks = true;
        raw_scheduler.config.max_num_batched_tokens = 4;
        let raw = test_sequence(0, 10);
        {
            let mut raw = get_mut_arcmutex!(raw);
            raw.return_raw_logits = true;
            raw.set_state(SequenceState::Waiting);
        }
        raw_scheduler.waiting.push_back(raw);
        let raw_output = raw_scheduler.schedule(&logger, None);
        assert!(raw_output.scheduled_prompt_chunks.is_none());
        assert!(raw_output.prompt_chunk_size.is_none());

        let mut media_scheduler = test_scheduler();
        media_scheduler.scheduler_visible_prompt_chunks = true;
        media_scheduler.config.max_num_batched_tokens = 4;
        let media =
            test_sequence_with_images(1, 10, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
        get_mut_arcmutex!(media).set_state(SequenceState::Waiting);
        media_scheduler.waiting.push_back(media);
        let media_output = media_scheduler.schedule(&logger, None);
        assert!(media_output.scheduled_prompt_chunks.is_none());
        assert!(media_output.prompt_chunk_size.is_none());
    }
}
