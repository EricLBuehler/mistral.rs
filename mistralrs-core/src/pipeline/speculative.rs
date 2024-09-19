use std::{
    any::Any,
    iter::zip,
    sync::{Arc, Mutex},
};

use anyhow::Result as anyhowResult;
use candle_core::{Device, IndexOp, Result, Tensor};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;
use tracing::warn;

use crate::{
    get_mut_arcmutex,
    pipeline::{
        sampling::{
            finish_or_add_toks_to_seq, sample_sequence, sample_target_sequence_speculative,
        },
        AdapterInstruction, Cache,
    },
    prefix_cacher::PrefixCacheManager,
    sequence::{Sequence, SequenceRecognizer},
    DeviceMapMetadata, Loader, ModelKind, PagedAttentionConfig, Pipeline, TokenSource,
    TryIntoDType,
};

use super::{
    cache_manager::DefaultCacheManager, chat_template::ChatTemplate, sampling::SpeculativeSample,
    AdapterActivationMixin, AnyMoePipelineMixin, CacheBackendMetadata, CacheInstruction,
    CacheManager, CacheManagerMixin, ForwardInputsResult, GeneralMetadata, IsqPipelineMixin,
    MetadataMixin, ModelCategory, ModelPaths, PreProcessingMixin,
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
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> anyhowResult<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
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
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> anyhowResult<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
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
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_in_cache(
            &*get_mut_arcmutex!(self.draft),
            seqs,
            modify_draft_cache,
        );
        DefaultCacheManager.clone_in_cache(&*get_mut_arcmutex!(self.target), seqs, false);
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_out_cache(
            &*get_mut_arcmutex!(self.draft),
            seqs,
            modify_draft_cache,
        );
        DefaultCacheManager.clone_out_cache(&*get_mut_arcmutex!(self.target), seqs, false);
    }
    fn set_none_cache(&self, reset_non_granular: bool, modify_draft_cache: bool) {
        DefaultCacheManager.set_none_cache(&*get_mut_arcmutex!(self.draft), modify_draft_cache);
        DefaultCacheManager.set_none_cache(&*get_mut_arcmutex!(self.target), false);
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &Cache {
        unreachable!()
    }
}

impl AdapterActivationMixin for SpeculativePipeline {
    /// Returns the number of activated adapters.
    fn activate_adapters(&mut self, adapters: Vec<String>) -> anyhow::Result<usize> {
        let mut res = 0;
        res += get_mut_arcmutex!(self.draft).activate_adapters(adapters.clone())?;
        res += get_mut_arcmutex!(self.target).activate_adapters(adapters)?;
        Ok(res)
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
}

#[async_trait::async_trait]
impl Pipeline for SpeculativePipeline {
    fn forward_inputs(&mut self, _inputs: Box<dyn Any>) -> Result<ForwardInputsResult> {
        unreachable!()
    }
    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManager,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<()> {
        unreachable!()
    }
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata<'_>,
    ) -> Result<()> {
        match backend_metadata {
            CacheBackendMetadata::DefaultInstructions { pre_op, post_op } => {
                match pre_op {
                    CacheInstruction::In(adapter_inst) => {
                        match adapter_inst {
                            AdapterInstruction::Activate(adapters) => {
                                self.activate_adapters(adapters).map_err(|e| {
                                    candle_core::Error::msg(<anyhow::Error as AsRef<
                                        dyn std::error::Error,
                                    >>::as_ref(
                                        &e
                                    ))
                                })?
                            }
                            AdapterInstruction::None => 0,
                        };
                        self.clone_in_cache(input_seqs, false)
                    }
                    CacheInstruction::Nothing(adapter_inst) => {
                        match adapter_inst {
                            AdapterInstruction::Activate(adapters) => {
                                self.activate_adapters(adapters).map_err(|e| {
                                    candle_core::Error::msg(<anyhow::Error as AsRef<
                                        dyn std::error::Error,
                                    >>::as_ref(
                                        &e
                                    ))
                                })?
                            }
                            AdapterInstruction::None => 0,
                        };
                    }
                    CacheInstruction::Reset {
                        reset_non_granular,
                        adapter_inst,
                    } => {
                        match adapter_inst {
                            AdapterInstruction::Activate(adapters) => {
                                self.activate_adapters(adapters).map_err(|e| {
                                    candle_core::Error::msg(<anyhow::Error as AsRef<
                                        dyn std::error::Error,
                                    >>::as_ref(
                                        &e
                                    ))
                                })?
                            }
                            AdapterInstruction::None => 0,
                        };
                        self.set_none_cache(reset_non_granular, false)
                    }
                    _ => unreachable!("Unreachable PRE cache op."),
                }

                assert_eq!(input_seqs.len(), 1);

                let seq = &mut input_seqs[0];

                // ======================= Run draft model gamma times producing tokens ============================
                // ======================= Sample the `gamma` logits. ============================
                let mut draft_samples = Vec::new();
                for i in 0..self.gamma {
                    let is_xlora = get_mut_arcmutex!(self.draft).get_metadata().is_xlora;
                    let device = get_mut_arcmutex!(self.draft).device();
                    let has_no_kv_cache =
                        get_mut_arcmutex!(self.draft).get_metadata().has_no_kv_cache;
                    let inputs = self
                        .get_processor()
                        .inputs_processor()
                        .process_inputs(
                            self.tokenizer(),
                            &mut [seq],
                            is_prompt && i == 0, // Only prompt (no kv cache) if first
                            is_xlora,
                            &device,
                            has_no_kv_cache,
                            None,
                            None,
                            None, // TODO: get block tables/handle it
                            None, // TODO: do we support???
                        )
                        .nth(0)
                        .unwrap()
                        .unwrap();
                    let logits = get_mut_arcmutex!(self.draft).forward_inputs(Box::new(inputs))?;
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
                        false, // do not add to tok trie yet
                        true,
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

                let initial_cache_len = get_mut_arcmutex!(self.target).cache().lock()[0]
                    .as_ref()
                    .map(|(k, _)| k.dims()[2])
                    .unwrap_or(0);

                // ========= Run the model ============
                let is_xlora = get_mut_arcmutex!(self.target).get_metadata().is_xlora;
                let device = get_mut_arcmutex!(self.target).device();
                let has_no_kv_cache = get_mut_arcmutex!(self.target)
                    .get_metadata()
                    .has_no_kv_cache;
                let inputs = self
                    .get_processor()
                    .inputs_processor()
                    .process_inputs(
                        self.tokenizer(),
                        &mut [seq],
                        true, // use the "prefill" tokens
                        is_xlora,
                        &device,
                        has_no_kv_cache,
                        Some((self.gamma, initial_cache_len)), // Get the last gamma, see above
                        None,
                        None, // TODO: get block tables/handle it
                        None, // TODO: do we support???
                    )
                    .nth(0)
                    .unwrap()
                    .unwrap();

                let logits = get_mut_arcmutex!(self.target).forward_inputs(Box::new(inputs))?;
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
                let samples = sample_target_sequence_speculative(
                    logits.clone(),
                    seq,
                    seq.return_logprobs(),
                    rng.clone(),
                    self.gamma,
                )
                .await?;

                let mut accepted_tokens = Vec::new();
                for (target_sample, draft_sample) in zip(samples, draft_samples) {
                    let tok = target_sample.sample.token;
                    accepted_tokens.push(target_sample.sample);
                    if draft_sample.sample.token != tok {
                        break;
                    }
                }

                // ======================= Narrow caches to account for rejections ============================
                let n_not_accepted = self.gamma - accepted_tokens.len();
                for (k, v) in get_mut_arcmutex!(self.draft)
                    .cache()
                    .lock()
                    .iter_mut()
                    .flatten()
                {
                    *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                    *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
                }
                if get_mut_arcmutex!(self.draft).get_metadata().is_xlora {
                    for (k, v) in get_mut_arcmutex!(self.draft)
                        .cache()
                        .xlora_lock()
                        .iter_mut()
                        .flatten()
                    {
                        *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                        *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
                    }
                }
                for (k, v) in get_mut_arcmutex!(self.target)
                    .cache()
                    .lock()
                    .iter_mut()
                    .flatten()
                {
                    *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                    *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
                }
                if get_mut_arcmutex!(self.draft).get_metadata().is_xlora {
                    for (k, v) in get_mut_arcmutex!(self.target)
                        .cache()
                        .xlora_lock()
                        .iter_mut()
                        .flatten()
                    {
                        *k = k.i((.., .., ..k.dims()[2] - n_not_accepted, ..))?;
                        *v = v.i((.., .., ..v.dims()[2] - n_not_accepted, ..))?;
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
                    match seq.recognizer {
                        SequenceRecognizer::Regex(ref mut rx) => {
                            get_mut_arcmutex!(self.target)
                                .get_metadata()
                                .tok_trie
                                .as_ref()
                                .ok_or(candle_core::Error::Msg(
                                    "`SpeculativePipeline::step` requires a token trie".to_string(),
                                ))?
                                .append_token(rx.as_mut(), accepted.token)
                                .map_err(candle_core::Error::msg)?;
                        }
                        SequenceRecognizer::Cfg(ref mut cfg) => {
                            get_mut_arcmutex!(self.target)
                                .get_metadata()
                                .tok_trie
                                .as_ref()
                                .ok_or(candle_core::Error::Msg(
                                    "`SpeculativePipeline::step` requires a token trie".to_string(),
                                ))?
                                .append_token(cfg.as_mut(), accepted.token)
                                .map_err(candle_core::Error::msg)?;
                        }
                        SequenceRecognizer::None => {}
                    }
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

                match post_op {
                    CacheInstruction::Out => {
                        self.clone_out_cache(input_seqs, true);
                    }
                    CacheInstruction::Nothing(_) => (),
                    CacheInstruction::Reset {
                        reset_non_granular,
                        adapter_inst: _,
                    } => self.set_none_cache(reset_non_granular, true),
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

                Ok(())
            }
            CacheBackendMetadata::PagedAttention {
                metadata: _,
                blocks_to_copy: _,
                blocks_to_swap_in: _,
                blocks_to_swap_out: _,
            } => unreachable!(),
        }
    }
    fn category(&self) -> ModelCategory {
        self.category
    }
}

// TODO
impl AnyMoePipelineMixin for SpeculativePipeline {}
