use std::sync::{Arc, Mutex};

use candle_core::{quantized::GgmlDType, Device, Result, Tensor};
use rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;

use crate::{
    aici::toktree::TokTrie,
    get_mut_arcmutex,
    models::Cache,
    pipeline::{sample_sequence, sampling::sample_target_sequence_speculative},
    sequence::{Sequence, SequenceState},
    xlora_models::NonGranularState,
    Pipeline,
};

use super::{calculate_inputs, chat_template::ChatTemplate, ModelInputs};

const SEED: u64 = 0;

/// Speculative decoding pipeline: https://arxiv.org/pdf/2211.17192
///
/// # Algorithm
/// Given draft model q and target model p with probability distributions \
/// q_i(x) and p_i(x) for each token
///
/// - Keep the sample for token i if q_i(x) <= p_i(x)
///     - This means the target model agrees
/// - Else (q_i(x) > p_i(x)) accept that token with prob p_i(x)/q_i(x)
///     - If rejected, sample token from from p'_i(x) = norm(max(0, p(x) − q(x))) and do not take any more
pub struct SpeculativePipeline {
    target: Arc<Mutex<dyn Pipeline>>,
    draft: Arc<Mutex<dyn Pipeline>>,
    gamma: usize,
    rng: Arc<Mutex<Isaac64Rng>>,
    rt: Runtime,
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
        // todo: some checks here
        Ok(Self {
            target,
            draft,
            gamma: config.gamma,
            rng: Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(SEED))),
            rt: Runtime::new().expect("Failed to create runtime"),
        })
    }
}

impl Pipeline for SpeculativePipeline {
    fn forward(&mut self, input_seqs: &mut [&mut Sequence], is_prompt: bool) -> Result<Tensor> {
        let n_seqs = input_seqs.len();
        let mut draft_model = get_mut_arcmutex!(self.draft);
        let mut target_model = get_mut_arcmutex!(self.target);
        let draft_cache_len = draft_model.cache().lock().len();

        let mut draft_samples = vec![Vec::new(); n_seqs];

        let repeat_last_n = draft_model.get_repeat_last_n();
        // Get the draft tokens
        for i in 0..self.gamma {
            let inputs = calculate_inputs(
                input_seqs,
                i == 0,
                self.is_xlora(),
                self.device(),
                self.has_no_kv_cache(),
                None,
            )
            .unwrap();
            let logits = draft_model.forward_inputs(inputs.clone())?;
            for ((i, seq), logits) in input_seqs
                .iter_mut()
                .enumerate()
                .zip(logits.to_device(&Device::Cpu)?.chunk(n_seqs, 0)?)
            {
                let t = draft_model.tok_trie().clone(); // TODO: do we need to reset it?
                let r = self.rng.clone();
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
        *draft_model.cache().lock() = vec![None; draft_cache_len];
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
            self.is_xlora(),
            self.device(),
            self.has_no_kv_cache(),
            Some(self.gamma),
        )
        .unwrap();
        let target_logits = target_model.forward_inputs(inputs_target)?;

        // Sample the tokens for each one we're testing and apply the algorithm
        // Remove γ raw tokens
        let repeat_last_n = target_model.get_repeat_last_n();
        let mut target_samples = Vec::new();
        for (seq, target_logits) in input_seqs.iter_mut().zip(target_logits.chunk(n_seqs, 0)?) {
            seq.set_state(SequenceState::RunningCompletion); // Back to normal
            seq.remove_tmp_tokens(self.gamma);

            let t = target_model.tok_trie().clone();
            let r = self.rng.clone();
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
            for (draft_sample, tgt_sample) in draft_samples.into_iter().zip(tgt_samples) {
                if draft_sample.token == tgt_sample.token {
                    if draft_sample.logprob <= tgt_sample.logprob {
                        // Target model agrees.
                        // todo: accepted
                    } else {
                        // Target model disagrees.
                        let acceptance_prob = tgt_sample.logprob / draft_sample.logprob;
                        let is_accepted =
                            get_mut_arcmutex!(self.rng).gen_bool(acceptance_prob as f64);
                        if is_accepted {
                            // todo: accepted
                        } else {
                            // Do not accept. Resample with updated prob dist norm(max(0, p(x) − q(x)))
                            // todo: resample and done with seq
                        }
                    }
                } else {
                    // Did not agree. Use the target model's choice. Return it.
                    // todo: done with seq
                }
            }
        }

        // todo: fix up the kv cache of the base model based on accepted tokens

        todo!()
    }
    fn forward_inputs(&mut self, _: ModelInputs) -> anyhow::Result<Tensor, candle_core::Error> {
        unreachable!()
    }
    fn device(&self) -> &Device {
        get_mut_arcmutex!(self.target).device()
    }
    fn num_hidden_layers(&self) -> usize {
        self.cache().lock().len()
    }
    fn cache(&self) -> &Cache {
        get_mut_arcmutex!(self.target).cache()
    }
    fn get_repeat_last_n(&self) -> usize {
        get_mut_arcmutex!(self.target).get_repeat_last_n()
    }
    fn tokenizer(&self) -> Arc<Tokenizer> {
        get_mut_arcmutex!(self.target).tokenizer()
    }
    fn eos_tok(&self) -> &[u32] {
        get_mut_arcmutex!(self.target).eos_tok()
    }
    fn name(&self) -> String {
        format!(
            "Speculative: p = {}, q = {}",
            get_mut_arcmutex!(self.target).name(),
            get_mut_arcmutex!(self.draft).name()
        )
    }
    fn get_max_seq_len(&self) -> usize {
        get_mut_arcmutex!(self.target).get_max_seq_len()
    }
    fn is_xlora(&self) -> bool {
        get_mut_arcmutex!(self.target).is_xlora()
    }
    fn has_no_kv_cache(&self) -> bool {
        get_mut_arcmutex!(self.target).has_no_kv_cache()
    }
    fn get_chat_template(&self) -> &ChatTemplate {
        get_mut_arcmutex!(self.target).get_chat_template()
    }
    fn get_non_granular_state(&self) -> &Option<NonGranularState> {
        get_mut_arcmutex!(self.target).get_non_granular_state()
    }
    fn tok_trie(&self) -> Arc<TokTrie> {
        get_mut_arcmutex!(self.target).tok_trie()
    }
    fn re_isq_model(&mut self, dtype: GgmlDType) -> anyhow::Result<()> {
        get_mut_arcmutex!(self.target).re_isq_model(dtype)?;
        get_mut_arcmutex!(self.draft).re_isq_model(dtype)
    }
}
