use std::sync::{Arc, Mutex};

use candle_core::{quantized::GgmlDType, Device, Result, Tensor};
use rand::Rng;
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;

use crate::{
    finish_and_add_tokens_to_seq, get_mut_arcmutex,
    models::Cache,
    pipeline::{sample_sequence, sampling::sample_target_sequence_speculative},
    prefix_cacher::PrefixCacheManager,
    sequence::{Sequence, SequenceRecognizer, SequenceState},
    DeviceMapMetadata, Loader, ModelKind, Pipeline, TokenSource,
};

use super::{
    cache_manager::DefaultCacheManager, calculate_inputs, chat_template::ChatTemplate,
    sampling::SpeculativeSample, CacheInstruction, CacheManager, GeneralMetadata, ModelInputs,
};

pub struct SpeculativeLoader {
    pub target: Box<dyn Loader>,
    pub draft: Box<dyn Loader>,
    pub config: SpeculativeConfig,
}

impl Loader for SpeculativeLoader {
    fn load_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: Option<candle_core::DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> anyhow::Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let target = self.target.load_model(
            revision.clone(),
            token_source.clone(),
            dtype,
            device,
            silent,
            mapper.clone(),
            in_situ_quant,
        )?;
        let draft = self.draft.load_model(
            revision,
            token_source,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
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

/// Speculative decoding pipeline: https://arxiv.org/pdf/2211.17192
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
    rt: Runtime,
    metadata: GeneralMetadata,
}

#[derive(Copy, Clone)]
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
        if get_mut_arcmutex!(target).tokenizer().get_vocab(true)
            != get_mut_arcmutex!(draft).tokenizer().get_vocab(true)
        {
            candle_core::bail!("Target and draft models' tokenzier vocab do not match. This is required for speculative decoding.");
        }
        let metadata = get_mut_arcmutex!(target).get_metadata().clone();
        // todo: some checks here?
        Ok(Self {
            target,
            draft,
            gamma: config.gamma,
            rt: Runtime::new().expect("Failed to create runtime"),
            metadata,
        })
    }
}

#[async_trait::async_trait]
impl Pipeline for SpeculativePipeline {
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
        pre_op: CacheInstruction,
        post_op: CacheInstruction,
    ) -> Result<()> {
        let n_seqs = input_seqs.len();

        match pre_op {
            CacheInstruction::In => self.clone_in_cache(input_seqs),
            CacheInstruction::Nonthing => (),
            CacheInstruction::Reset { reset_non_granular } => {
                self.set_none_cache(reset_non_granular)
            }
            _ => unreachable!("Unreachable pre cache op."),
        }

        let mut draft_samples = vec![Vec::new(); n_seqs];

        let repeat_last_n = get_mut_arcmutex!(self.draft).get_metadata().repeat_last_n;
        // Get the draft tokens
        for i in 0..self.gamma {
            let inputs = calculate_inputs(
                input_seqs,
                i == 0,
                get_mut_arcmutex!(self.draft).get_metadata().is_xlora,
                &get_mut_arcmutex!(self.draft).device(),
                get_mut_arcmutex!(self.draft).get_metadata().has_no_kv_cache,
                None,
            )
            .unwrap();
            let logits = get_mut_arcmutex!(self.draft).forward_inputs(inputs.clone())?;
            for ((i, seq), logits) in input_seqs
                .iter_mut()
                .enumerate()
                .zip(logits.to_device(&Device::Cpu)?.chunk(n_seqs, 0)?)
            {
                let t = get_mut_arcmutex!(self.draft)
                    .get_metadata()
                    .tok_trie
                    .clone();
                let r = rng.clone();
                let sampled = self.rt.block_on(async {
                    sample_sequence(
                        logits.clone(),
                        seq,
                        seq.return_logprobs(),
                        repeat_last_n,
                        t,
                        r,
                        n_seqs > 1,
                        false, // Do not add to trie
                    )
                    .await
                })?;
                seq.add_tmp_token(sampled.token);
                draft_samples[i].push(SpeculativeSample {
                    sample: sampled,
                    distribution: logits,
                });
            }
        }

        // Reset the cache (xlora and normal), and reset non granular state
        // So no more cache at all
        get_mut_arcmutex!(self.draft).set_none_cache(true);

        // =========== Now run base model with draft tokens ============

        // Add the tokens
        for seq in input_seqs.iter_mut() {
            seq.set_state(SequenceState::RunningPrefillPrompt); // This is really what it is
        }

        // Make inputs for target model
        let inputs_target = calculate_inputs(
            input_seqs,
            is_prompt,
            get_mut_arcmutex!(self.target).get_metadata().is_xlora,
            &get_mut_arcmutex!(self.target).device(),
            get_mut_arcmutex!(self.target)
                .get_metadata()
                .has_no_kv_cache,
            Some(self.gamma),
        )
        .unwrap();
        let target_logits = get_mut_arcmutex!(self.target).forward_inputs(inputs_target)?;

        // Sample the tokens for each one we're testing and apply the algorithm
        // Remove γ raw tokens
        let repeat_last_n = get_mut_arcmutex!(self.target).get_metadata().repeat_last_n;
        let mut target_samples = Vec::new();
        for (seq, target_logits) in input_seqs.iter_mut().zip(target_logits.chunk(n_seqs, 0)?) {
            seq.set_state(SequenceState::RunningCompletion); // Back to normal
            seq.remove_tmp_tokens(self.gamma);

            let t = get_mut_arcmutex!(self.target)
                .get_metadata()
                .tok_trie
                .clone();
            let r = rng.clone();
            // Sample. Do not add results to trie yet, and do not add tokens to the seq
            let samples = self.rt.block_on(async {
                sample_target_sequence_speculative(
                    target_logits,
                    seq,
                    seq.return_logprobs(),
                    repeat_last_n,
                    t,
                    r,
                    self.gamma,
                )
                .await
            })?;
            target_samples.push(samples);
        }

        // Note: know that all seqs here have same cache length
        let initial_cache_len = input_seqs[0].cache()[0]
            .as_ref()
            .map(|(k, _)| k.dims()[2])
            .unwrap_or(0);

        let mut n_accepted_toks = Vec::new();
        for (seq, (draft_samples, tgt_samples)) in input_seqs
            .iter_mut()
            .zip(draft_samples.into_iter().zip(target_samples))
        {
            let mut accepted_tokens = Vec::new();
            for (draft_sample, tgt_sample) in draft_samples.into_iter().zip(tgt_samples) {
                if draft_sample.sample.token == tgt_sample.sample.token {
                    if draft_sample.sample.logprob <= tgt_sample.sample.logprob {
                        // Target model agrees.
                        accepted_tokens.push(tgt_sample.sample);
                    } else {
                        // Target model disagrees.
                        let acceptance_prob =
                            tgt_sample.sample.logprob / draft_sample.sample.logprob;
                        let is_accepted = get_mut_arcmutex!(rng).gen_bool(acceptance_prob as f64);
                        if is_accepted {
                            accepted_tokens.push(tgt_sample.sample);
                        } else {
                            // Do not accept. Resample with updated prob dist relu(p(x) − q(x))
                            let corrected_distribution =
                                (tgt_sample.distribution - draft_sample.distribution)?.relu()?;
                            let t = get_mut_arcmutex!(self.target)
                                .get_metadata()
                                .tok_trie
                                .clone();
                            let r = rng.clone();
                            let sampled = self.rt.block_on(async {
                                sample_sequence(
                                    corrected_distribution,
                                    seq,
                                    seq.return_logprobs(),
                                    repeat_last_n,
                                    t,
                                    r,
                                    n_seqs > 1,
                                    false, // Do not add to trie
                                )
                                .await
                            })?;
                            accepted_tokens.push(sampled);
                            break;
                        }
                    }
                } else {
                    // Did not agree. Use the target model's choice. Return it.
                    accepted_tokens.push(tgt_sample.sample);
                    break;
                }
            }

            n_accepted_toks.push(accepted_tokens.len());
            // Add the tokens to the seq and the trie
            for accepted in accepted_tokens {
                let eos_owned = get_mut_arcmutex!(self.target)
                    .get_metadata()
                    .eos_tok
                    .clone();
                let eos_tok = if disable_eos_stop {
                    None
                } else {
                    Some(&eos_owned[..])
                };
                finish_and_add_tokens_to_seq!(self, prefix_cacher, seq, accepted, eos_tok);
                match seq.recognizer {
                    SequenceRecognizer::Regex(ref mut rx) => {
                        get_mut_arcmutex!(self.target)
                            .get_metadata()
                            .tok_trie
                            .append_token(rx.as_mut(), accepted.token);
                    }
                    SequenceRecognizer::Cfg(ref mut cfg) => {
                        get_mut_arcmutex!(self.target)
                            .get_metadata()
                            .tok_trie
                            .append_token(cfg.as_mut(), accepted.token);
                    }
                    SequenceRecognizer::None => {}
                }
            }
        }

        match post_op {
            CacheInstruction::Out => {
                // Fix up the kv cache of the base model based on accepted tokens
                get_mut_arcmutex!(self.target).clone_out_cache(input_seqs);
                for (seq, n_accepted) in input_seqs.iter_mut().zip(n_accepted_toks) {
                    let cache = seq.cache();
                    for (k, v) in cache.iter_mut().flatten() {
                        let computed_len = initial_cache_len + n_accepted;
                        *k = k.narrow(2, 0, computed_len)?;
                        *v = v.narrow(2, 0, computed_len)?;
                    }
                    if seq.is_xlora() {
                        let cache = seq.xlora_cache();
                        for (k, v) in cache.iter_mut().flatten() {
                            let computed_len = initial_cache_len + n_accepted;
                            *k = k.narrow(2, 0, computed_len)?;
                            *v = v.narrow(2, 0, computed_len)?;
                        }
                    }
                }
            }
            CacheInstruction::Nonthing => (),
            CacheInstruction::Reset { reset_non_granular } => {
                self.set_none_cache(reset_non_granular)
            }
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
    fn forward_inputs(&mut self, _: ModelInputs) -> anyhow::Result<Tensor, candle_core::Error> {
        unreachable!("Speculative pipeline handles the forward pass in `.step`")
    }
    async fn sample(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Tensor,
        _prefix_cacher: &mut PrefixCacheManager,
        _disable_eos_stop: bool,
        __rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<()> {
        unreachable!("Speculative pipeline handles sampling in `.step`")
    }
    fn device(&self) -> Device {
        get_mut_arcmutex!(self.target).device()
    }
    fn tokenizer(&self) -> Arc<Tokenizer> {
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
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        get_mut_arcmutex!(self.target).get_chat_template()
    }
    fn reset_non_granular_state(&self) {
        get_mut_arcmutex!(self.target).reset_non_granular_state();
        get_mut_arcmutex!(self.draft).reset_non_granular_state();
    }
    fn re_isq_model(&mut self, dtype: GgmlDType) -> anyhow::Result<()> {
        get_mut_arcmutex!(self.target).re_isq_model(dtype)?;
        get_mut_arcmutex!(self.draft).re_isq_model(dtype)
    }
    fn get_metadata(&self) -> &GeneralMetadata {
        &self.metadata
    }
    fn clone_in_cache(&mut self, seqs: &mut [&mut Sequence]) {
        DefaultCacheManager.clone_in_cache(&mut *get_mut_arcmutex!(self.target), seqs)
    }
    fn clone_out_cache(&mut self, seqs: &mut [&mut Sequence]) {
        DefaultCacheManager.clone_out_cache(&mut *get_mut_arcmutex!(self.target), seqs)
    }
    fn set_none_cache(&mut self, reset_non_granular: bool) {
        DefaultCacheManager.set_none_cache(&mut *get_mut_arcmutex!(self.target));
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &Cache {
        unreachable!()
    }
}
