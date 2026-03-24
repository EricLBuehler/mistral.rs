use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::Result as anyhowResult;
use candle_core::{Device, IndexOp, Result, Tensor};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    get_mut_arcmutex,
    kv_cache::{CacheManager, HybridCacheManager, NormalCacheManager, RecurrentStateSnapshot},
    layers_masker::PastKvLenCache,
    pipeline::sampling::{
        finish_or_add_toks_to_seq, sample_sequence, sample_target_sequence_speculative,
    },
    prefix_cacher::PrefixCacheManagerV2,
    sequence::Sequence,
    DeviceMapSetting, Loader, ModelKind, PagedAttentionConfig, Pipeline, TokenSource, TryIntoDType,
};

use crate::utils::progress::ProgressScopeGuard;

use super::{
    chat_template::ChatTemplate, sampling::SpeculativeSample, AnyMoePipelineMixin,
    CacheBackendMetadata, CacheInstruction, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, IsqPipelineMixin, MetadataMixin, ModelCategory, ModelPaths,
    PreProcessingMixin,
};

/// A loader for a speculative pipeline using 2 [`Loader`]s.
pub struct SpeculativeLoader {
    pub target: Box<dyn Loader>,
    pub draft: Box<dyn Loader>,
    pub config: SpeculativeConfig,
}

impl Loader for SpeculativeLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> anyhowResult<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);

        let target = self.target.load_model_from_hf(
            revision.clone(),
            token_source.clone(),
            dtype,
            device,
            silent,
            mapper.clone(),
            in_situ_quant,
            paged_attn_config,
        )?;
        let draft = self.draft.load_model_from_hf(
            revision,
            token_source,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )?;
        Ok(Arc::new(tokio::sync::Mutex::new(SpeculativePipeline::new(
            target,
            draft,
            self.config,
        )?)))
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> anyhowResult<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);

        let target = self.target.load_model_from_path(
            paths,
            dtype,
            device,
            silent,
            mapper.clone(),
            in_situ_quant,
            paged_attn_config,
        )?;
        let draft = self.draft.load_model_from_path(
            paths,
            dtype,
            device,
            silent,
            mapper.clone(),
            in_situ_quant,
            paged_attn_config,
        )?;
        Ok(Arc::new(tokio::sync::Mutex::new(SpeculativePipeline::new(
            target,
            draft,
            self.config,
        )?)))
    }
    fn get_id(&self) -> String {
        format!(
            "Speculative: tgt = `{}`, draft = `{}`, gamma = `{}`",
            self.target.get_id(),
            self.draft.get_id(),
            self.config.gamma,
        )
    }
    fn get_kind(&self) -> ModelKind {
        ModelKind::Speculative {
            target: Box::new(self.target.get_kind()),
            draft: Box::new(self.draft.get_kind()),
        }
    }
}

/// Speculative decoding pipeline: <https://arxiv.org/pdf/2211.17192>
///
/// # Algorithm
/// Given draft model q and target model p with probability distributions \
/// q_i(x) and p_i(x) for each token
///
/// - Keep the sample for token i if q_i(x) <= p_i(x)
///     - This means the target model agrees
/// - Else (q_i(x) > p_i(x)) accept that token with prob p_i(x)/q_i(x)
///     - If rejected, sample token from from p'_i(x) = norm(max(0, p(x) − q(x))) and do not take any more'
///
pub struct SpeculativePipeline {
    target: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    draft: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    // Exposes the target model cache to engine-level cache checks/management.
    target_cache: EitherCache,
    // Draft hybrid slot index per sequence id.
    draft_recurrent_slots: Mutex<HashMap<usize, usize>>,
    gamma: usize,
    metadata: Arc<GeneralMetadata>,
    category: ModelCategory,
}

#[derive(Copy, Clone)]
/// Metadata for a speculative pipeline
pub struct SpeculativeConfig {
    /// γ completions to run of the draft model
    pub gamma: usize,
}

impl SpeculativePipeline {
    fn swap_in_draft_recurrent_indices(&self, seqs: &mut [&mut Sequence]) -> Vec<Option<usize>> {
        let slots = self.draft_recurrent_slots.lock().unwrap();
        let mut originals = Vec::with_capacity(seqs.len());
        for seq in seqs.iter_mut() {
            originals.push(seq.recurrent_state_idx());
            seq.set_recurrent_state_idx(slots.get(seq.id()).copied());
        }
        originals
    }

    fn capture_draft_recurrent_indices(&self, seqs: &mut [&mut Sequence]) {
        let mut slots = self.draft_recurrent_slots.lock().unwrap();
        for seq in seqs.iter_mut() {
            if let Some(idx) = seq.recurrent_state_idx() {
                slots.insert(*seq.id(), idx);
            } else {
                slots.remove(seq.id());
            }
        }
    }

    fn restore_recurrent_indices(seqs: &mut [&mut Sequence], originals: Vec<Option<usize>>) {
        for (seq, original) in seqs.iter_mut().zip(originals) {
            seq.set_recurrent_state_idx(original);
        }
    }

    fn cleanup_finished_draft_slots(&self, seqs: &mut [&mut Sequence]) {
        let mut to_free = Vec::new();
        {
            let mut slots = self.draft_recurrent_slots.lock().unwrap();
            for seq in seqs.iter_mut() {
                if seq.is_finished_paged_attn() {
                    if let Some(idx) = slots.remove(seq.id()) {
                        to_free.push(idx);
                    }
                }
            }
        }

        if to_free.is_empty() {
            return;
        }

        let draft = get_mut_arcmutex!(self.draft);
        if matches!(draft.cache(), EitherCache::Hybrid(_)) {
            let mut hybrid = draft.cache().hybrid();
            for idx in to_free {
                hybrid.free_seq(idx);
            }
        }
    }

    pub fn new(
        target: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        draft: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        config: SpeculativeConfig,
    ) -> Result<Self> {
        if get_mut_arcmutex!(target)
            .tokenizer()
            .as_ref()
            .ok_or(candle_core::Error::Msg(
                "`SpeculativePipeline::new` requires the target pipeline to have a token trie"
                    .to_string(),
            ))?
            .get_vocab(true)
            != get_mut_arcmutex!(draft)
                .tokenizer()
                .as_ref()
                .ok_or(candle_core::Error::Msg(
                    "`SpeculativePipeline::new` requires the draft pipeline to have a token trie"
                        .to_string(),
                ))?
                .get_vocab(true)
        {
            candle_core::bail!("Target and draft models' tokenizer vocab do not match. This is required for speculative decoding.");
        }
        if get_mut_arcmutex!(target).category() != get_mut_arcmutex!(draft).category() {
            candle_core::bail!("Target and draft models' category do not match. This is required for speculative decoding.");
        }
        if get_mut_arcmutex!(target)
            .get_processor()
            .inputs_processor()
            .get_type()
            != get_mut_arcmutex!(draft)
                .get_processor()
                .inputs_processor()
                .get_type()
        {
            candle_core::bail!("Target and draft models' input processors do not match. This is required for speculative decoding.");
        }
        let metadata = get_mut_arcmutex!(target).get_metadata().clone();
        let category = get_mut_arcmutex!(target).category();
        let target_cache = get_mut_arcmutex!(target).cache().clone();
        // TODO: some checks or relaxation here?
        Ok(Self {
            target,
            draft,
            target_cache,
            draft_recurrent_slots: Mutex::new(HashMap::new()),
            gamma: config.gamma,
            metadata,
            category,
        })
    }
}

impl PreProcessingMixin for SpeculativePipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        get_mut_arcmutex!(self.target).get_chat_template()
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        get_mut_arcmutex!(self.target).get_input_processor_config()
    }
}

impl IsqPipelineMixin for SpeculativePipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> anyhow::Result<()> {
        get_mut_arcmutex!(self.target).re_isq_model(dtype)?;
        get_mut_arcmutex!(self.draft).re_isq_model(dtype)
    }
}

impl CacheManagerMixin for SpeculativePipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        {
            let draft = get_mut_arcmutex!(self.draft);
            if matches!(draft.cache(), EitherCache::Hybrid(_)) {
                // Use draft-specific recurrent slots and keep sequence-owned
                // recurrent_state_idx reserved for the target pipeline.
                let originals = self.swap_in_draft_recurrent_indices(seqs);
                HybridCacheManager.clone_in_cache(&*draft, seqs, true);
                self.capture_draft_recurrent_indices(seqs);
                Self::restore_recurrent_indices(seqs, originals);
            } else {
                NormalCacheManager.clone_in_cache(&*draft, seqs, true);
            }
        }
        {
            let target = get_mut_arcmutex!(self.target);
            if matches!(target.cache(), EitherCache::Hybrid(_)) {
                HybridCacheManager.clone_in_cache(&*target, seqs, false);
            } else {
                NormalCacheManager.clone_in_cache(&*target, seqs, false);
            }
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        {
            let draft = get_mut_arcmutex!(self.draft);
            if matches!(draft.cache(), EitherCache::Hybrid(_)) {
                let originals = self.swap_in_draft_recurrent_indices(seqs);
                HybridCacheManager.clone_out_cache(&*draft, seqs, true);
                self.capture_draft_recurrent_indices(seqs);
                Self::restore_recurrent_indices(seqs, originals);
            } else {
                NormalCacheManager.clone_out_cache(&*draft, seqs, true);
            }
        }
        {
            let target = get_mut_arcmutex!(self.target);
            if matches!(target.cache(), EitherCache::Hybrid(_)) {
                HybridCacheManager.clone_out_cache(&*target, seqs, false);
            } else {
                NormalCacheManager.clone_out_cache(&*target, seqs, false);
            }
        }
        self.cleanup_finished_draft_slots(seqs);
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        {
            let draft = get_mut_arcmutex!(self.draft);
            if matches!(draft.cache(), EitherCache::Hybrid(_)) {
                HybridCacheManager.set_none_cache(
                    &*draft,
                    seqs,
                    modify_draft_cache,
                    load_preallocated_cache,
                );
                self.draft_recurrent_slots.lock().unwrap().clear();
            } else {
                NormalCacheManager.set_none_cache(
                    &*draft,
                    seqs,
                    modify_draft_cache,
                    load_preallocated_cache,
                );
            }
        }
        {
            let target = get_mut_arcmutex!(self.target);
            if matches!(target.cache(), EitherCache::Hybrid(_)) {
                HybridCacheManager.set_none_cache(&*target, seqs, false, load_preallocated_cache);
            } else {
                NormalCacheManager.set_none_cache(&*target, seqs, false, load_preallocated_cache);
            }
        }
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        &self.target_cache
    }
    fn do_preallocated_cache(&self) -> bool {
        // KV cache size is not the same (necessarily)
        false
    }
}

impl MetadataMixin for SpeculativePipeline {
    fn device(&self) -> Device {
        get_mut_arcmutex!(self.target).device()
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        get_mut_arcmutex!(self.target).tokenizer()
    }
    fn name(&self) -> String {
        format!(
            "Speculative: tgt = `{}`, draft = `{}`, gamma = `{}`",
            get_mut_arcmutex!(self.target).name(),
            get_mut_arcmutex!(self.draft).name(),
            self.gamma,
        )
    }
    fn reset_non_granular_state(&self) {
        get_mut_arcmutex!(self.target).reset_non_granular_state();
        get_mut_arcmutex!(self.draft).reset_non_granular_state();
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for SpeculativePipeline {
    fn forward_inputs(
        &mut self,
        _inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> Result<ForwardInputsResult> {
        unreachable!()
    }
    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<()> {
        unreachable!()
    }
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        _return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
    ) -> Result<Duration> {
        if input_seqs.len() > 1 {
            // Fallback batching path: process each sequence independently.
            // This enables speculative decoding parity with scheduler batching
            // semantics while preserving existing single-sequence logic.
            let mut total_duration = Duration::ZERO;
            for i in 0..input_seqs.len() {
                let (_, tail) = input_seqs.split_at_mut(i);
                let seq = &mut tail[0];
                total_duration += self
                    .step(
                        std::slice::from_mut(seq),
                        is_prompt,
                        _return_raw_logits,
                        prefix_cacher,
                        disable_eos_stop,
                        rng.clone(),
                        backend_metadata.clone(),
                    )
                    .await?;
            }
            return Ok(total_duration);
        }

        let (paged_attn_metadata, post_op) = match backend_metadata {
            CacheBackendMetadata::DefaultInstructions { pre_op, post_op } => {
                match pre_op {
                    CacheInstruction::In => self.clone_in_cache(input_seqs),
                    CacheInstruction::Nothing => (),
                    CacheInstruction::Reset {
                        reset_non_granular,
                        load_preallocated_cache,
                    } => self.set_none_cache(
                        input_seqs,
                        reset_non_granular,
                        true,
                        load_preallocated_cache,
                    ),
                    _ => unreachable!("Unreachable PRE cache op."),
                }
                (None, Some(post_op))
            }
            CacheBackendMetadata::PagedAttention { metadata } => {
                // Speculative decoding relies on clone_in/out for hybrid recurrent slots.
                self.clone_in_cache(input_seqs);
                (Some(metadata), None)
            }
        };
        let using_paged_attn = paged_attn_metadata.is_some();

        let start = Instant::now();
        assert_eq!(input_seqs.len(), 1);

        let seq = &mut input_seqs[0];
        let seq_id = *seq.id();
        let base_seq_len = seq.get_toks().len();
        let mut gamma = self.gamma;

        if let Some(metadata) = paged_attn_metadata.as_ref() {
            let mut kv_mgr = get_mut_arcmutex!(metadata.kv_cache_manager);

            // Scheduler reserves decode slots for +1 token. Speculative decoding may
            // need up to `gamma` lookahead tokens, so reserve additional capacity here.
            while gamma > 0
                && kv_mgr
                    .allocate_slots(seq_id, base_seq_len + gamma, &[])
                    .is_none()
            {
                gamma -= 1;
            }
        }

        if gamma == 0 {
            candle_core::bail!(
                "Speculative decoding could not reserve paged-attention slots for sequence {seq_id}."
            );
        }

        // Detect hybrid caches and capture initial state for snapshot/restore
        let draft_is_hybrid = matches!(
            get_mut_arcmutex!(self.draft).cache(),
            EitherCache::Hybrid(_)
        );
        let target_is_hybrid = matches!(
            get_mut_arcmutex!(self.target).cache(),
            EitherCache::Hybrid(_)
        );

        let initial_draft_cache_len = if draft_is_hybrid {
            get_mut_arcmutex!(self.draft)
                .cache()
                .hybrid()
                .get_past_kv_len()
                .unwrap_or(0)
        } else {
            0
        };

        // Snapshot draft recurrent state before draft token generation
        let draft_recurrent_snapshot: Option<Vec<RecurrentStateSnapshot>> = if draft_is_hybrid {
            let slot_idx = self
                .draft_recurrent_slots
                .lock()
                .unwrap()
                .get(seq.id())
                .copied()
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "Hybrid draft is missing recurrent slot for sequence {}",
                        seq.id()
                    ))
                })?;
            Some(
                get_mut_arcmutex!(self.draft)
                    .cache()
                    .hybrid()
                    .snapshot_recurrent_state(slot_idx)?,
            )
        } else {
            None
        };

        // ======================= Run draft model gamma times producing tokens ============================
        // ======================= Sample the `gamma` logits. ============================
        let mut draft_samples = Vec::new();
        for i in 0..gamma {
            let is_xlora = get_mut_arcmutex!(self.draft).get_metadata().is_xlora;
            let device = get_mut_arcmutex!(self.draft).device();
            let no_kv_cache = get_mut_arcmutex!(self.draft).get_metadata().no_kv_cache;
            let inputs = self
                .get_processor()
                .inputs_processor()
                .process_inputs(
                    self.tokenizer(),
                    &mut [seq],
                    is_prompt && i == 0, // Only prompt (no kv cache) if first
                    is_xlora,
                    &device,
                    no_kv_cache,
                    None,
                    false,
                    None,
                    paged_attn_metadata.clone(),
                    get_mut_arcmutex!(self.draft).device_mapper(),
                )
                .unwrap()
                .inputs;
            let logits = get_mut_arcmutex!(self.draft).forward_inputs(inputs, false)?;
            #[allow(irrefutable_let_patterns)]
            let ForwardInputsResult::CausalGeneration { logits } = logits
            else {
                candle_core::bail!(
                    "Speculative decoding requires `CausalGeneration` forward results"
                );
            };

            let sample = sample_sequence(
                logits.clone(),
                seq,
                seq.return_logprobs(),
                rng.clone(),
                false, // todo tune
                true,
                false,
            )
            .await?;
            seq.add_tmp_tok(sample.token);
            draft_samples.push(SpeculativeSample { sample });
        }
        seq.remove_tmp_tok(gamma);

        // ======================= Add all draft tokens but the last one. Add the last from the seq. ============================
        let mut draft_prefill_tokens = if is_prompt {
            seq.get_toks().to_vec()
        } else {
            vec![*seq.get_toks().last().unwrap()]
        };
        for (i, sample) in draft_samples.iter().enumerate() {
            if i == draft_samples.len() - 1 {
                continue;
            }
            draft_prefill_tokens.push(sample.sample.token);
        }
        // Clone before move — needed for hybrid cache replay after rejection
        let draft_prefill_tokens_clone = if draft_is_hybrid || target_is_hybrid {
            Some(draft_prefill_tokens.clone())
        } else {
            None
        };
        seq.set_prefill_toks(draft_prefill_tokens);

        // ======================= Run the model with all draft tokens. ============================

        let initial_cache_len = match get_mut_arcmutex!(self.target).cache() {
            EitherCache::Full(full) => full.lock()[0]
                .as_ref()
                .map(|(k, _)| k.dims()[2])
                .unwrap_or(0),
            EitherCache::Normal(normal) => normal.lock().unwrap().0[0].current_seq_len(),
            EitherCache::Hybrid(hybrid) => hybrid.lock().unwrap().get_past_kv_len().unwrap_or(0),
        };

        // Snapshot target recurrent state before target forward
        let target_recurrent_snapshot: Option<Vec<RecurrentStateSnapshot>> = if target_is_hybrid {
            let slot_idx = seq.recurrent_state_idx().ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "Hybrid target is missing recurrent slot for sequence {}",
                    seq.id()
                ))
            })?;
            Some(
                get_mut_arcmutex!(self.target)
                    .cache()
                    .hybrid()
                    .snapshot_recurrent_state(slot_idx)?,
            )
        } else {
            None
        };

        // ========= Run the model ============
        let is_xlora = get_mut_arcmutex!(self.target).get_metadata().is_xlora;
        let device = get_mut_arcmutex!(self.target).device();
        let no_kv_cache = get_mut_arcmutex!(self.target).get_metadata().no_kv_cache;
        let inputs = self
            .get_processor()
            .inputs_processor()
            .process_inputs(
                self.tokenizer(),
                &mut [seq],
                true, // use the "prefill" tokens
                is_xlora,
                &device,
                no_kv_cache,
                Some((gamma, initial_cache_len)), // Get the last gamma, see above
                false,
                None,
                paged_attn_metadata.clone(),
                get_mut_arcmutex!(self.target).device_mapper(),
            )
            .unwrap()
            .inputs;

        let logits = get_mut_arcmutex!(self.target).forward_inputs(inputs, false)?;
        #[allow(irrefutable_let_patterns)]
        let ForwardInputsResult::CausalGeneration { logits } = logits
        else {
            candle_core::bail!("Speculative decoding requires `CausalGeneration` forward results");
        };

        // Reset the prefill tokens
        seq.reset_prefill_toks();

        // ======================= Rejection sampling. ============================
        // Map from each target sample to corresponding in draft sample
        // this will first rollback LLG state if any, and then advance for the accepted tokens only
        let samples = sample_target_sequence_speculative(
            logits.clone(),
            seq,
            seq.return_logprobs(),
            rng.clone(),
            &draft_samples,
        )
        .await?;

        let accepted_tokens = samples.into_iter().map(|s| s.sample).collect::<Vec<_>>();

        // ======================= Narrow caches to account for rejections ============================
        let n_not_accepted = gamma - accepted_tokens.len();

        if draft_is_hybrid {
            if n_not_accepted > 0 {
                let slot_idx = self
                    .draft_recurrent_slots
                    .lock()
                    .unwrap()
                    .get(seq.id())
                    .copied()
                    .ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "Hybrid draft is missing recurrent slot for sequence {}",
                            seq.id()
                        ))
                    })?;
                {
                    let draft_guard = get_mut_arcmutex!(self.draft);
                    let mut hybrid = draft_guard.cache().hybrid();
                    hybrid.restore_recurrent_state(
                        slot_idx,
                        draft_recurrent_snapshot.as_ref().unwrap(),
                    )?;
                    hybrid.truncate_attention_to(initial_draft_cache_len)?;
                }

                // Replay accepted tokens through draft to advance cache
                let draft_prefill_toks = draft_prefill_tokens_clone.as_ref().unwrap();
                let n_replay = draft_prefill_toks.len().saturating_sub(n_not_accepted);
                if n_replay > 0 {
                    let replay_toks = draft_prefill_toks[..n_replay].to_vec();
                    seq.set_prefill_toks(replay_toks);
                    let is_xlora = get_mut_arcmutex!(self.draft).get_metadata().is_xlora;
                    let device = get_mut_arcmutex!(self.draft).device();
                    let no_kv_cache = get_mut_arcmutex!(self.draft).get_metadata().no_kv_cache;
                    let inputs = self
                        .get_processor()
                        .inputs_processor()
                        .process_inputs(
                            self.tokenizer(),
                            &mut [seq],
                            true,
                            is_xlora,
                            &device,
                            no_kv_cache,
                            Some((n_replay, initial_draft_cache_len)),
                            false,
                            None,
                            paged_attn_metadata.clone(),
                            get_mut_arcmutex!(self.draft).device_mapper(),
                        )
                        .unwrap()
                        .inputs;
                    let _ = get_mut_arcmutex!(self.draft).forward_inputs(inputs, false)?;
                    seq.reset_prefill_toks();
                }
            }
        } else if !using_paged_attn {
            match get_mut_arcmutex!(self.draft).cache() {
                EitherCache::Full(full) => {
                    for (k, v) in full.lock().iter_mut().flatten() {
                        *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                        *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
                    }
                }
                EitherCache::Normal(normal) => {
                    for cache in &mut *normal.lock().unwrap().0 {
                        cache
                            .set_len(cache.current_seq_len() - n_not_accepted)
                            .map_err(|_| candle_core::Error::msg("KV cache set_len failed."))?;
                    }
                }
                EitherCache::Hybrid(_) => unreachable!(),
            }
        }
        if get_mut_arcmutex!(self.draft).get_metadata().is_xlora {
            match get_mut_arcmutex!(self.draft).cache() {
                EitherCache::Full(full) => {
                    for (k, v) in full.xlora_lock().iter_mut().flatten() {
                        *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                        *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
                    }
                }
                EitherCache::Normal(_) | EitherCache::Hybrid(_) => {
                    candle_core::bail!(
                        "Speculative decoding X-LoRA path requires full cache backend."
                    )
                }
            }
        }
        if target_is_hybrid {
            if n_not_accepted > 0 {
                let slot_idx = seq.recurrent_state_idx().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "Hybrid target is missing recurrent slot for sequence {}",
                        seq.id()
                    ))
                })?;
                {
                    let target_guard = get_mut_arcmutex!(self.target);
                    let mut hybrid = target_guard.cache().hybrid();
                    hybrid.restore_recurrent_state(
                        slot_idx,
                        target_recurrent_snapshot.as_ref().unwrap(),
                    )?;
                    hybrid.truncate_attention_to(initial_cache_len)?;
                }

                // Replay accepted tokens through target to advance cache
                let draft_prefill_toks = draft_prefill_tokens_clone.as_ref().unwrap();
                let n_replay = draft_prefill_toks.len().saturating_sub(n_not_accepted);
                if n_replay > 0 {
                    let replay_toks = draft_prefill_toks[..n_replay].to_vec();
                    seq.set_prefill_toks(replay_toks);
                    let is_xlora = get_mut_arcmutex!(self.target).get_metadata().is_xlora;
                    let device = get_mut_arcmutex!(self.target).device();
                    let no_kv_cache = get_mut_arcmutex!(self.target).get_metadata().no_kv_cache;
                    let inputs = self
                        .get_processor()
                        .inputs_processor()
                        .process_inputs(
                            self.tokenizer(),
                            &mut [seq],
                            true,
                            is_xlora,
                            &device,
                            no_kv_cache,
                            Some((n_replay, initial_cache_len)),
                            false,
                            None,
                            paged_attn_metadata.clone(),
                            get_mut_arcmutex!(self.target).device_mapper(),
                        )
                        .unwrap()
                        .inputs;
                    let _ = get_mut_arcmutex!(self.target).forward_inputs(inputs, false)?;
                    seq.reset_prefill_toks();
                }
            }
        } else if !using_paged_attn {
            match get_mut_arcmutex!(self.target).cache() {
                EitherCache::Full(full) => {
                    for (k, v) in full.lock().iter_mut().flatten() {
                        *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                        *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
                    }
                }
                EitherCache::Normal(normal) => {
                    for cache in &mut *normal.lock().unwrap().0 {
                        cache
                            .set_len(cache.current_seq_len() - n_not_accepted)
                            .map_err(|_| candle_core::Error::msg("KV cache set_len failed."))?;
                    }
                }
                EitherCache::Hybrid(_) => unreachable!(),
            }
        }
        if get_mut_arcmutex!(self.target).get_metadata().is_xlora {
            match get_mut_arcmutex!(self.target).cache() {
                EitherCache::Full(full) => {
                    for (k, v) in full.xlora_lock().iter_mut().flatten() {
                        *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                        *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
                    }
                }
                EitherCache::Normal(_) | EitherCache::Hybrid(_) => {
                    candle_core::bail!(
                        "Speculative decoding X-LoRA path requires full cache backend."
                    )
                }
            }
        }

        let eos_owned = get_mut_arcmutex!(self.target)
            .get_metadata()
            .eos_tok
            .clone();
        let eos_tok = if disable_eos_stop {
            None
        } else {
            Some(&eos_owned[..])
        };
        // Add the tokens to the seq and the trie
        for accepted in accepted_tokens {
            finish_or_add_toks_to_seq(self, prefix_cacher, seq, accepted.clone(), eos_tok, true)
                .await?;
        }

        if let Some(metadata) = paged_attn_metadata.as_ref() {
            let mut kv_mgr = get_mut_arcmutex!(metadata.kv_cache_manager);
            kv_mgr.trim_request_to_num_tokens(seq_id, seq.get_toks().len());
        }

        // Trick to improve lower bounds. Sample last token in multinomial
        /*
        let sample = sample_sequence(
            logits.clone(),
            seq,
            seq.return_logprobs(),
            rng.clone(),
            false, // todo tune
            true, // do not add to tok trie yet
            true,
        )
        .await?;
        finish_or_add_toks_to_seq(self, prefix_cacher, seq, sample, eos_tok, false);
        */
        let end = Instant::now();
        let exec_duration = end.duration_since(start);

        if using_paged_attn {
            self.clone_out_cache(input_seqs);
        } else if let Some(post_op) = post_op {
            match post_op {
                CacheInstruction::Out => {
                    self.clone_out_cache(input_seqs);
                }
                CacheInstruction::Nothing => (),
                CacheInstruction::Reset {
                    reset_non_granular,
                    load_preallocated_cache,
                } => self.set_none_cache(
                    input_seqs,
                    reset_non_granular,
                    true,
                    load_preallocated_cache,
                ),
                _ => unreachable!("Unreachable pre cache op."),
            }
        }

        // Done! We have:
        // - Run the draft model gamma times
        // - Reset draft model cache fully
        // - Sampled draft model's distributions
        // - Run target model
        // - Execute speculative decoding algorithm on the resulting distributions
        // - Added the accepted tokens to buffer and trie
        // - Maybe fixed up cache of base model based on accepted tokens.

        Ok(exec_duration)
    }
    fn category(&self) -> ModelCategory {
        self.category.clone()
    }
}

impl AnyMoePipelineMixin for SpeculativePipeline {}
