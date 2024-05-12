#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    ops::Add,
    sync::{Arc, Mutex},
};

use candle_core::{quantized::GgmlDType, DType, Device, IndexOp, Result, Tensor, D};
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;

use crate::{
    finish_and_add_tokens_to_seq, get_mut_arcmutex,
    models::Cache,
    pipeline::sampling::{sample_sequence, sample_target_sequence_speculative},
    prefix_cacher::PrefixCacheManager,
    sequence::{Sequence, SequenceRecognizer},
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
    metadata: GeneralMetadata,
    latest_logit_cache: Option<Tensor>,
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
        // todo: some checks or relaxation here?
        Ok(Self {
            target,
            draft,
            gamma: config.gamma,
            metadata,
            latest_logit_cache: None,
        })
    }
}

fn find_first_true(x: &Tensor) -> Result<Tensor> {
    (x.to_dtype(DType::F32)?.cumsum(D::Minus1)?.eq(0.0)?).sum(D::Minus1)
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
        match pre_op {
            CacheInstruction::In => self.clone_in_cache(input_seqs, false),
            CacheInstruction::Nonthing => (),
            CacheInstruction::Reset { reset_non_granular } => {
                self.set_none_cache(reset_non_granular, false)
            }
            _ => unreachable!("Unreachable PRE cache op."),
        }

        assert_eq!(input_seqs.len(), 1);

        let seq = &mut input_seqs[0];

        // ======================= Run draft model gamma times producing tokens ============================
        // ======================= Sample the `gamma` logits. ============================
        let mut draft_samples = Vec::new();
        let mut q_sampled_out = Vec::new();
        let mut small_logits = Vec::new();
        let repeat_last_n = get_mut_arcmutex!(self.draft).get_metadata().repeat_last_n;
        for i in 0..self.gamma {
            let is_xlora = get_mut_arcmutex!(self.draft).get_metadata().is_xlora;
            let device = get_mut_arcmutex!(self.draft).device();
            let has_no_kv_cache = get_mut_arcmutex!(self.draft).get_metadata().has_no_kv_cache;
            let inputs = calculate_inputs(
                &[seq],
                is_prompt && i == 0, // Only prompt (no kv cache) if first
                is_xlora,
                &device,
                has_no_kv_cache,
                None,
            )
            .unwrap();
            let logits = get_mut_arcmutex!(self.draft)
                .forward_inputs(inputs)?
                .to_dtype(candle_core::DType::F32)?;

            let sample = sample_sequence(
                logits.clone(),
                seq,
                seq.return_logprobs(),
                repeat_last_n,
                get_mut_arcmutex!(self.draft)
                    .get_metadata()
                    .tok_trie
                    .clone(),
                rng.clone(),
                false, // todo tune
                false, // do not add to tok trie yet
                true,
            )
            .await?;
            seq.add_tmp_tok(sample.token);
            q_sampled_out.push(Tensor::new(vec![sample.token], logits.device())?);
            small_logits.push(logits.narrow(1, logits.dim(1)? - 1, 1)?);
            draft_samples.push(SpeculativeSample {
                sample,
                distribution: logits.clone(),
            });
        }
        let q_sampled_out = Tensor::cat(&q_sampled_out, 0)?;
        let small_logits = Tensor::cat(&small_logits, D::Minus2)?;
        seq.remove_tmp_tok(self.gamma);

        // ======================= Add all draft tokens but the last one. Add the last from the seq. ============================
        let mut draft_prefill_tokens = if is_prompt {
            seq.get_toks().to_vec()
        } else {
            vec![*seq.get_toks().last().unwrap()]
        };
        for sample in draft_samples {
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
        let inputs = calculate_inputs(
            &[seq],
            true, // use the "prefill" tokens
            is_xlora,
            &device,
            has_no_kv_cache,
            Some((self.gamma + 1, initial_cache_len)), // Get the last gamma, see above
        )
        .unwrap();

        let logits = get_mut_arcmutex!(self.target)
            .forward_inputs(inputs)?
            .to_dtype(candle_core::DType::F32)?;

        let logits = logits.i((.., ..logits.dim(1)? - 1, ..))?;
        let prob_next = logits.i((.., logits.dim(1)? - 1, ..))?.unsqueeze(1)?;

        // Reset the prefill tokens
        seq.reset_prefill_toks();

        let p = logits
            .gather(&q_sampled_out.unsqueeze(1)?.unsqueeze(0)?, D::Minus1)?
            .to_dtype(candle_core::DType::F32)?;
        let q = small_logits.gather(&q_sampled_out.unsqueeze(1)?.unsqueeze(0)?, D::Minus1)?;

        let rand_uniform = q.rand_like(0.0, 1.0)?;
        let accepted = find_first_true(&rand_uniform.gt(&p.div(&q)?)?.squeeze(2)?)?;

        let num_rejected = accepted
            .to_dtype(DType::F32)?
            .neg()?
            .add(self.gamma as f64)?;
        let has_rejected = num_rejected.gt(0.0)?;
        let num_rejected = num_rejected.to_dtype(DType::U8)?.to_vec1::<u8>()?;

        let accepted = accepted.clamp(0u32, (self.gamma - 1) as u32)?;
        let adjusted_prob = logits
            .index_select(&accepted.to_dtype(DType::U8)?, 1)?
            .sub(&small_logits.index_select(&accepted.to_dtype(DType::U8)?, 1)?)?
            .relu()?;
        let adjusted_prob = adjusted_prob.broadcast_div(&adjusted_prob.sum_keepdim(D::Minus1)?)?;

        let prob_next = Tensor::where_cond(
            &has_rejected.unsqueeze(1)?.broadcast_as(prob_next.shape())?,
            &adjusted_prob,
            &prob_next,
        )?;
        // ======================= Rejection sampling. ============================
        // Map from each target sample to corresponding in draft sample
        let samples = sample_target_sequence_speculative(
            Tensor::cat(
                &[
                    &logits
                        .i((
                            ..,
                            ..(logits.dim(1)? - num_rejected[0] as usize).saturating_sub(1),
                            ..,
                        ))
                        .unwrap(),
                    &prob_next,
                ],
                1,
            )?,
            seq,
            seq.return_logprobs(),
            repeat_last_n,
            get_mut_arcmutex!(self.draft)
                .get_metadata()
                .tok_trie
                .clone(),
            rng.clone(),
            self.gamma + 1 - num_rejected[0] as usize,
        )
        .await?;

        let mut accepted_tokens = Vec::new();
        for target_sample in samples {
            accepted_tokens.push(target_sample.sample);
        }

        // ======================= Narrow caches to account for rejections ============================
        let n_not_accepted = num_rejected[0] as usize + 1;
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
            finish_and_add_tokens_to_seq!(self, prefix_cacher, seq, accepted, eos_tok, false);
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

        // Trick to improve lower bounds. Sample last token in multinomial
        /*
        let sample = sample_sequence(
            logits.clone(),
            seq,
            seq.return_logprobs(),
            repeat_last_n,
            get_mut_arcmutex!(self.draft)
                .get_metadata()
                .tok_trie
                .clone(),
            rng.clone(),
            false, // todo tune
            true, // do not add to tok trie yet
            true,
        )
        .await?;
        finish_and_add_tokens_to_seq!(self, prefix_cacher, seq, sample, eos_tok, false);
        */

        match post_op {
            CacheInstruction::Out => {
                self.clone_out_cache(input_seqs, true);
            }
            CacheInstruction::Nonthing => (),
            CacheInstruction::Reset { reset_non_granular } => {
                self.set_none_cache(reset_non_granular, true)
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
    fn clone_in_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_in_cache(
            &mut *get_mut_arcmutex!(self.draft),
            seqs,
            modify_draft_cache,
        );
        DefaultCacheManager.clone_in_cache(&mut *get_mut_arcmutex!(self.target), seqs, false);
    }
    fn clone_out_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_out_cache(
            &mut *get_mut_arcmutex!(self.draft),
            seqs,
            modify_draft_cache,
        );
        DefaultCacheManager.clone_out_cache(&mut *get_mut_arcmutex!(self.target), seqs, false);
    }
    fn set_none_cache(&mut self, reset_non_granular: bool, modify_draft_cache: bool) {
        DefaultCacheManager.set_none_cache(&mut *get_mut_arcmutex!(self.draft), modify_draft_cache);
        DefaultCacheManager.set_none_cache(&mut *get_mut_arcmutex!(self.target), false);
        if reset_non_granular {
            self.reset_non_granular_state()
        }
        self.latest_logit_cache = None;
    }
    fn cache(&self) -> &Cache {
        unreachable!()
    }
}
