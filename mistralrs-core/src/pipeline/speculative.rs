use std::{
    any::Any,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::Result as anyhowResult;
use candle_core::{Device, IndexOp, Result, Tensor};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;
use tracing::warn;

use crate::{
    device_map::DeviceMapper,
    get_mut_arcmutex,
    kv_cache::NormalCacheManager,
    pipeline::sampling::{
        finish_or_add_toks_to_seq, sample_sequence, sample_target_sequence_speculative,
    },
    prefix_cacher::PrefixCacheManagerV2,
    sequence::Sequence,
    DeviceMapSetting, Loader, ModelKind, PagedAttentionConfig, Pipeline, TokenSource, TryIntoDType,
};

use crate::kv_cache::CacheManager;
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
        let paged_attn_config = if paged_attn_config.is_none() {
            warn!(
                "Speculative decoding does not currently support PagedAttention, running without"
            );
            None
        } else {
            paged_attn_config
        };

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
        let paged_attn_config = if paged_attn_config.is_none() {
            warn!(
                "Speculative decoding does not currently support PagedAttention, running without"
            );
            None
        } else {
            paged_attn_config
        };

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
        // TODO: some checks or relaxation here?
        Ok(Self {
            target,
            draft,
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
        NormalCacheManager.clone_in_cache(&*get_mut_arcmutex!(self.draft), seqs, true);
        NormalCacheManager.clone_in_cache(&*get_mut_arcmutex!(self.target), seqs, false);
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        NormalCacheManager.clone_out_cache(&*get_mut_arcmutex!(self.draft), seqs, true);
        NormalCacheManager.clone_out_cache(&*get_mut_arcmutex!(self.target), seqs, false);
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        NormalCacheManager.set_none_cache(
            &*get_mut_arcmutex!(self.draft),
            seqs,
            modify_draft_cache,
            load_preallocated_cache,
        );
        NormalCacheManager.set_none_cache(
            &*get_mut_arcmutex!(self.target),
            seqs,
            false,
            load_preallocated_cache,
        );
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        unreachable!()
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
        match backend_metadata {
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

                let start = Instant::now();
                assert_eq!(input_seqs.len(), 1);

                let seq = &mut input_seqs[0];

                // ======================= Run draft model gamma times producing tokens ============================
                // ======================= Sample the `gamma` logits. ============================
                let mut draft_samples = Vec::new();
                for i in 0..self.gamma {
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
                            None, // TODO: get block tables/handle it
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
                seq.remove_tmp_tok(self.gamma);

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
                seq.set_prefill_toks(draft_prefill_tokens);

                // ======================= Run the model with all draft tokens. ============================

                let initial_cache_len = match get_mut_arcmutex!(self.target).cache() {
                    EitherCache::Full(full) => full.lock()[0]
                        .as_ref()
                        .map(|(k, _)| k.dims()[2])
                        .unwrap_or(0),
                    EitherCache::Normal(normal) => normal.lock().unwrap().0[0].current_seq_len(),
                    EitherCache::Hybrid(_) => {
                        unreachable!("Speculative decoding is not supported with hybrid caches")
                    }
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
                        Some((self.gamma, initial_cache_len)), // Get the last gamma, see above
                        false,
                        None,
                        None, // TODO: get block tables/handle it
                        get_mut_arcmutex!(self.target).device_mapper(),
                    )
                    .unwrap()
                    .inputs;

                let logits = get_mut_arcmutex!(self.target).forward_inputs(inputs, false)?;
                #[allow(irrefutable_let_patterns)]
                let ForwardInputsResult::CausalGeneration { logits } = logits
                else {
                    candle_core::bail!(
                        "Speculative decoding requires `CausalGeneration` forward results"
                    );
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
                let n_not_accepted = self.gamma - accepted_tokens.len();

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
                    EitherCache::Hybrid(_) => {
                        unreachable!("Speculative decoding is not supported with hybrid caches")
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
                            unreachable!()
                        }
                    }
                }
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
                    EitherCache::Hybrid(_) => {
                        unreachable!("Speculative decoding is not supported with hybrid caches")
                    }
                }
                if get_mut_arcmutex!(self.draft).get_metadata().is_xlora {
                    match get_mut_arcmutex!(self.target).cache() {
                        EitherCache::Full(full) => {
                            for (k, v) in full.xlora_lock().iter_mut().flatten() {
                                *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                                *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
                            }
                        }
                        EitherCache::Normal(_) | EitherCache::Hybrid(_) => {
                            unreachable!()
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
                    // Do not use the prefix cacher
                    finish_or_add_toks_to_seq(
                        self,
                        prefix_cacher,
                        seq,
                        accepted.clone(),
                        eos_tok,
                        false,
                    )
                    .await?;
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
            CacheBackendMetadata::PagedAttention { .. } => unreachable!(),
        }
    }
    fn category(&self) -> ModelCategory {
        self.category.clone()
    }
}

impl AnyMoePipelineMixin for SpeculativePipeline {}
