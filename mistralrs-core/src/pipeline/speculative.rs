use std::sync::{Arc, Mutex};

use candle_core::{quantized::GgmlDType, Device, Result, Tensor};
use rand::Rng;
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;

use crate::{
    get_mut_arcmutex,
    pipeline::{sample_sequence, sampling::sample_target_sequence_speculative},
    prefix_cacher::PrefixCacheManager,
    sequence::{Sequence, SequenceState},
    Pipeline,
};

use super::{calculate_inputs, chat_template::ChatTemplate, GeneralMetadata, ModelInputs};

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
    target: Arc<Mutex<dyn Pipeline>>,
    draft: Arc<Mutex<dyn Pipeline>>,
    gamma: usize,
    rt: Runtime,
    metadata: GeneralMetadata,
}

pub struct SpeculativeConfig {
    /// γ completions to run of the draft model
    pub gamma: usize,
}

impl SpeculativePipeline {
    pub fn new(
        target: Arc<Mutex<dyn Pipeline>>,
        draft: Arc<Mutex<dyn Pipeline>>,
        config: SpeculativeConfig,
    ) -> Result<Self> {
        if get_mut_arcmutex!(target).tokenizer().get_vocab(true)
            != get_mut_arcmutex!(draft).tokenizer().get_vocab(true)
        {
            candle_core::bail!("Target and draft models' tokenzier vocab do not match. This is required for speculative decoding.");
        }
        let metadata = get_mut_arcmutex!(target).get_metadata().clone();
        // todo: some checks here
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
    ) -> Result<()> {
        let n_seqs = input_seqs.len();
        let mut draft_model = get_mut_arcmutex!(self.draft);
        let mut target_model = get_mut_arcmutex!(self.target);
        let draft_cache_len = draft_model.get_metadata().num_hidden_layers;

        let mut draft_samples = vec![Vec::new(); n_seqs];

        let repeat_last_n = draft_model.get_metadata().repeat_last_n;
        // Get the draft tokens
        for i in 0..self.gamma {
            let inputs = calculate_inputs(
                input_seqs,
                i == 0,
                draft_model.get_metadata().is_xlora,
                &draft_model.device(),
                draft_model.get_metadata().has_no_kv_cache,
                None,
            )
            .unwrap();
            let logits = draft_model.forward_inputs(inputs.clone())?;
            for ((i, seq), logits) in input_seqs
                .iter_mut()
                .enumerate()
                .zip(logits.to_device(&Device::Cpu)?.chunk(n_seqs, 0)?)
            {
                let t = draft_model.get_metadata().tok_trie.clone(); // TODO: do we need to reset it?
                let r = rng.clone();
                let sampled = self.rt.block_on(async {
                    sample_sequence(
                        logits,
                        *seq,
                        seq.return_logprobs(),
                        repeat_last_n,
                        t,
                        r,
                        n_seqs > 1,
                    )
                    .await
                })?;
                seq.add_tmp_token(sampled.token);
                draft_samples[i].push(sampled);
            }
        }
        // Reset the cache
        draft_model.set_none_cache();
        // TODO: xlora cache reset too

        // =========== Now run base model with draft tokens ============

        // Add the tokens
        for seq in input_seqs.iter_mut() {
            seq.set_state(SequenceState::RunningPrefillPrompt); // This is really what it is
        }

        // Make inputs for target model
        let inputs_target = calculate_inputs(
            input_seqs,
            is_prompt,
            target_model.get_metadata().is_xlora,
            &target_model.device(),
            target_model.get_metadata().has_no_kv_cache,
            Some(self.gamma),
        )
        .unwrap();
        let target_logits = target_model.forward_inputs(inputs_target)?;

        // Sample the tokens for each one we're testing and apply the algorithm
        // Remove γ raw tokens
        let repeat_last_n = target_model.get_metadata().repeat_last_n;
        let mut target_samples = Vec::new();
        for (seq, target_logits) in input_seqs.iter_mut().zip(target_logits.chunk(n_seqs, 0)?) {
            seq.set_state(SequenceState::RunningCompletion); // Back to normal
            seq.remove_tmp_tokens(self.gamma);

            let t = target_model.get_metadata().tok_trie.clone();
            let r = rng.clone();
            let samples = self.rt.block_on(async {
                sample_target_sequence_speculative(
                    target_logits,
                    *seq,
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

        for (draft_samples, tgt_samples) in draft_samples.into_iter().zip(target_samples) {
            let mut accepted_tokens = Vec::new();
            for (draft_sample, tgt_sample) in draft_samples.into_iter().zip(tgt_samples) {
                if draft_sample.token == tgt_sample.token {
                    if draft_sample.logprob <= tgt_sample.logprob {
                        // Target model agrees.
                        accepted_tokens.push(tgt_sample);
                    } else {
                        // Target model disagrees.
                        let acceptance_prob = tgt_sample.logprob / draft_sample.logprob;
                        let is_accepted = get_mut_arcmutex!(rng).gen_bool(acceptance_prob as f64);
                        if is_accepted {
                            accepted_tokens.push(tgt_sample);
                        } else {
                            // Do not accept. Resample with updated prob dist norm(max(0, p(x) − q(x)))
                            // todo: resample and done with seq
                        }
                    }
                } else {
                    // Did not agree. Use the target model's choice. Return it.
                    accepted_tokens.push(tgt_sample);
                }
            }
        }

        // todo: fix up the kv cache of the base model based on accepted tokens

        todo!()
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
            "Speculative: tgt = `{}`, draft = `{}`",
            get_mut_arcmutex!(self.target).name(),
            get_mut_arcmutex!(self.draft).name()
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
        get_mut_arcmutex!(self.draft).clone_in_cache(seqs);
        get_mut_arcmutex!(self.target).clone_in_cache(seqs);
    }
    fn clone_out_cache(&mut self, seqs: &mut [&mut Sequence]) {
        get_mut_arcmutex!(self.draft).clone_out_cache(seqs);
        get_mut_arcmutex!(self.target).clone_out_cache(seqs);
    }
    fn set_none_cache(&mut self) {
        get_mut_arcmutex!(self.draft).set_none_cache();
        get_mut_arcmutex!(self.target).set_none_cache();
    }
}
