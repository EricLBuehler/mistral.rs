#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use core::f64;
use std::{
    collections::HashMap,
    iter::zip,
    sync::{Arc, Mutex},
};

use candle_core::{bail, DType, Device, Error, IndexOp, Result, Tensor, D};
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use rand::distributions::{Distribution, WeightedIndex};
use rand_isaac::Isaac64Rng;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::layers_masker::masked_fill;

#[derive(Clone, Debug)]
/// Stop sequences or ids.
pub enum StopTokens {
    Seqs(Vec<String>),
    Ids(Vec<u32>),
}

#[derive(Clone, Debug)]
/// Sampling params are used to control sampling.
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub top_n_logprobs: usize,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_toks: Option<StopTokens>,
    pub max_len: Option<usize>,
    pub logits_bias: Option<HashMap<u32, f32>>,
    pub n_choices: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            top_n_logprobs: 0,
            frequency_penalty: None,
            presence_penalty: None,
            stop_toks: None,
            max_len: None,
            logits_bias: None,
            n_choices: 1,
        }
    }
}

/// Sampler for sampling.
#[derive(Clone)]
pub struct Sampler {
    temperature: Option<f64>,
    top_n_logprobs: usize,
    tokenizer: Arc<Tokenizer>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    logits_bias: Option<Tensor>,
    top_k: i64,
    top_p: f64,
    min_p: f64,
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Top-n logprobs element
pub struct TopLogprob {
    pub token: u32,
    pub logprob: f32,
    pub bytes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprobs {
    pub token: u32,
    pub logprob: f32,
    pub bytes: String,
    pub top_logprobs: Option<Vec<TopLogprob>>,
}

#[derive(Clone)]
pub struct SamplingMetadata<'a> {
    pub output_tokens_tensor: Tensor,
    pub presence_penalties: Tensor,
    pub freq_penalties: Tensor,
    pub topk: Tensor,
    pub topp: Tensor,
    pub minp: Tensor,
    pub temperature: Tensor,
    pub tokenizers: Vec<&'a Tokenizer>,
}

fn argmax_sample_last_dim(logits: &Tensor) -> Result<Tensor> {
    logits.argmax(D::Minus1)
}

impl Sampler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temperature: Option<f64>,
        top_n_logprobs: usize,
        tokenizer: Arc<Tokenizer>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        logits_bias: Option<Tensor>,
        top_k: i64,
        top_p: f64,
        min_p: f64,
    ) -> Self {
        let temperature = if temperature.map_or(true, |v| v < 1e-7) {
            None
        } else {
            temperature
        };
        Self {
            temperature,
            top_n_logprobs,
            tokenizer,
            frequency_penalty,
            presence_penalty,
            logits_bias,
            top_k,
            top_p,
            min_p,
        }
    }

    fn get_top_logprobs(
        &self,
        probs: &[f32],
        argsort_indices: &[usize],
    ) -> Result<Vec<TopLogprob>> {
        let mut argsort_indices_sorted = argsort_indices.to_vec();
        // Sort by descending prob
        argsort_indices_sorted
            .sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).expect("No ordering."));
        // These are where the top n are
        let top_n_toks_range = 0..self.top_n_logprobs;
        // The top n's values
        let top_n_logprobs = argsort_indices_sorted[top_n_toks_range.clone()]
            .iter()
            .map(|x| probs[*x].log(10.0))
            .collect::<Vec<_>>();
        // Find where they actually are in the logits
        let mut top_n_toks = Vec::new();
        for val in top_n_toks_range {
            top_n_toks.push(argsort_indices[val]);
        }

        let mut bytes = Vec::new();
        for tok in &top_n_toks {
            bytes.push(
                self.tokenizer
                    .decode(&[*tok as u32], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            );
        }
        Ok(zip(bytes, zip(top_n_toks, top_n_logprobs))
            .map(|(bytes, (token, logprob))| TopLogprob {
                token: token as u32,
                logprob,
                bytes,
            })
            .collect::<Vec<_>>())
    }

    fn sample_argmax(&self, logits: Tensor, return_logprobs: bool) -> Result<Logprobs> {
        let next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

        let probs: Vec<f32> = logits.to_vec1()?;

        let argsort_indices = (0..probs.len()).collect::<Vec<_>>();
        let logprob = probs[next_token as usize].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(&probs, &argsort_indices)?)
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs,
            bytes: self
                .tokenizer
                .decode(&[next_token], false)
                .map_err(|x| Error::Msg(x.to_string()))?,
        })
    }

    fn sample_speculative_top_kp_min_p(
        &self,
        logits: Tensor,
        return_logprobs: bool,
        top_k: i64,
        top_p: f32,
        min_p: f32,
    ) -> Result<Logprobs> {
        let mut probs: Vec<f32> = logits.to_vec1()?;
        let mut argsort_indices = (0..probs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices
            .sort_unstable_by(|&i, &j| probs[j].partial_cmp(&probs[i]).expect("No ordering."));

        if top_k > 0 {
            // Clamp smaller probabilities to zero.
            for (index, val) in argsort_indices.iter().enumerate() {
                if index >= top_k as usize {
                    probs[*val] = 0.0;
                }
            }
        }

        // TOP P

        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                probs[*index] = 0.0;
            } else {
                cumsum += probs[*index];
            }
        }

        let max_p = probs[argsort_indices[0]];

        // MIN P

        // min-p sampling samples from the tokens whose prob are greater than
        // (max prob of token in dist) * min_p

        // Clamp smaller probabilities to zero.
        for index in &argsort_indices {
            if max_p * min_p >= probs[*index] {
                probs[*index] = 0.0;
            }
        }

        let logits = Tensor::from_slice(&probs, logits.shape(), &Device::Cpu)?;

        let next_token = argmax_sample_last_dim(&logits)?.to_scalar::<u32>()?;

        let logprob = probs[next_token as usize].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(&probs, &argsort_indices)?)
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs,
            bytes: self
                .tokenizer
                .decode(&[next_token], false)
                .map_err(|x| Error::Msg(x.to_string()))?,
        })
    }

    fn sample_multinomial(
        &self,
        probs: &mut Vec<f32>,
        argsort_indices: Vec<usize>,
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        let distr = WeightedIndex::new(&*probs).map_err(Error::wrap)?;

        let mut mut_ref_rng = &mut *rng.lock().expect("could not lock rng mutex");
        let next_token = distr.sample(&mut mut_ref_rng); // "Find the first item which has a weight *higher* than the chosen weight."
        let logprob = probs[next_token].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(probs, &argsort_indices)?)
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token as u32,
            logprob,
            top_logprobs,
            bytes: self
                .tokenizer
                .decode(&[next_token.try_into().unwrap()], false)
                .map_err(|x| Error::Msg(x.to_string()))?,
        })
    }

    fn sample_top_kp_min_p(
        &self,
        probs: &mut Vec<f32>,
        top_k: i64,
        top_p: f32,
        min_p: f32,
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        let mut argsort_indices = (0..probs.len()).collect::<Vec<_>>();
        // Sort by descending probability.
        argsort_indices
            .sort_unstable_by(|&i, &j| probs[j].partial_cmp(&probs[i]).expect("No ordering."));

        if top_k > 0 {
            // Clamp smaller probabilities to zero.
            for (index, val) in argsort_indices.iter().enumerate() {
                if index >= top_k as usize {
                    probs[*val] = 0.0;
                }
            }
        }

        if top_p <= 0.0 || top_p >= 1.0 {
            return self.sample_multinomial(probs, argsort_indices, return_logprobs, rng);
        }

        // TOP P

        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                probs[*index] = 0.0;
            } else {
                cumsum += probs[*index];
            }
        }

        if min_p <= 0.0 || min_p >= 1.0 {
            return self.sample_multinomial(probs, argsort_indices, return_logprobs, rng);
        }

        let max_p = probs[argsort_indices[0]];

        // MIN P

        // min-p sampling samples from the tokens whose prob are greater than
        // (max prob of token in dist) * min_p

        // Clamp smaller probabilities to zero.
        for index in &argsort_indices {
            if max_p * min_p >= probs[*index] {
                probs[*index] = 0.0;
            }
        }

        // Sample with clamped probabilities.
        self.sample_multinomial(probs, argsort_indices, return_logprobs, rng)
    }

    fn apply_penalties(&self, mut logits: Vec<f32>, context: Option<&[u32]>) -> Result<Tensor> {
        if self.frequency_penalty.is_some() || self.presence_penalty.is_some() {
            if context.is_none() {
                bail!("Must specify penalty context.");
            }
            let context = context.as_ref().unwrap();
            let frequency_penalty = self.frequency_penalty.unwrap_or(0.);
            let presence_penalty = self.presence_penalty.unwrap_or(0.);

            //mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence

            let mut counts = vec![0.0f32; logits.len()];
            for ctx in context.iter() {
                counts[*ctx as usize] += 1.0;
            }

            for (token_id, logit) in logits.iter_mut().enumerate() {
                let count = counts[token_id];
                *logit = *logit
                    - count * frequency_penalty
                    - if count > 0.0 { 1. } else { 0. } * presence_penalty;
            }
        }
        let vocab_size = logits.len();
        Tensor::from_vec(logits, vocab_size, &Device::Cpu)
    }

    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    /// If `frequency_penalty.is_some()` or `presence_penalty.is_some()`, then `penalty_ctxt` must be provided.
    pub fn sample(
        &self,
        logits: Tensor,
        penalty_ctxt: Option<&[u32]>,
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
        sample_speculative: bool,
    ) -> Result<Logprobs> {
        let logits = self.apply_penalties(logits.to_vec1()?, penalty_ctxt)?;
        let logits = match self.logits_bias {
            Some(ref bias) => (logits + bias)?,
            None => logits,
        };
        let next_token = if sample_speculative {
            match self.temperature {
                None => self.sample_speculative_top_kp_min_p(
                    logits,
                    return_logprobs,
                    self.top_k,
                    self.top_p as f32,
                    self.min_p as f32,
                )?,
                Some(temperature) => {
                    let logits = (&logits / temperature)?;
                    let probs = candle_nn::ops::softmax_last_dim(&logits)?;

                    self.sample_speculative_top_kp_min_p(
                        probs,
                        return_logprobs,
                        self.top_k,
                        self.top_p as f32,
                        self.min_p as f32,
                    )?
                }
            }
        } else {
            match self.temperature {
                None => self.sample_argmax(logits, return_logprobs)?,
                Some(temperature) => {
                    let logits = (&logits / temperature)?;
                    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
                    let mut probs: Vec<f32> = probs.to_vec1()?;

                    self.sample_top_kp_min_p(
                        &mut probs,
                        self.top_k,
                        self.top_p as f32,
                        self.min_p as f32,
                        return_logprobs,
                        rng,
                    )?
                }
            }
        };
        Ok(next_token)
    }
}

pub struct NewSampler;

impl NewSampler {
    pub fn sample_on_gpu(
        &self,
        logits: Tensor,
        metadata: SamplingMetadata,
    ) -> Result<Vec<Logprobs>> {
        let logits = logits.to_dtype(DType::F32)?;

        let logits = apply_penalties(
            logits,
            &metadata.output_tokens_tensor,
            &metadata.presence_penalties,
            &metadata.freq_penalties,
        )?;

        let logits = (logits / metadata.temperature)?;

        let logits = apply_topk_topp(&logits, &metadata.topp, &metadata.topk)?;
        let logits = apply_minp(&logits, &metadata.minp)?;

        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let log_probs = probs.log()?;

        let sampled = sample_gumbel(&logits)?;

        let sampled_toks = sampled.to_vec1::<u32>()?;
        let logprobs = log_probs.to_vec1::<f32>()?;

        let mut logprobs_collector = Vec::new();
        for (tok, (logprob, tokenizer)) in sampled_toks
            .into_iter()
            .zip(logprobs.into_iter().zip(metadata.tokenizers))
        {
            logprobs_collector.push(Logprobs {
                token: tok,
                logprob,
                top_logprobs: None, // TODO
                bytes: tokenizer
                    .decode(&[tok.try_into().unwrap()], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            });
        }
        Ok(logprobs_collector)
    }
}

fn get_bin_counts_and_mask(
    tokens: &Tensor,
    vocab_size: usize,
    num_seqs: usize,
) -> Result<(Tensor, Tensor)> {
    // https://github.com/vllm-project/vllm/blob/740374d456a638df98ffbc7d9dab328752330e62/vllm/model_executor/layers/sampler.py#L179
    let bin_counts = Tensor::zeros((num_seqs, vocab_size + 1), DType::I64, tokens.device())?;
    let bin_counts = bin_counts.scatter_add(tokens, &tokens.ones_like()?, 1)?;
    let bin_counts = bin_counts.i((.., ..vocab_size))?;
    let mask = bin_counts.gt(0f64)?;
    Ok((bin_counts, mask))
}

fn apply_penalties(
    mut logits: Tensor,
    output_tokens_tensor: &Tensor,
    presence_penalties: &Tensor,
    freq_penalties: &Tensor,
) -> Result<Tensor> {
    // https://github.com/vllm-project/vllm/blob/740374d456a638df98ffbc7d9dab328752330e62/vllm/model_executor/layers/sampler.py#L243
    let (num_seqs, vocab_size) = logits.dims2()?;
    let (output_bin_counts, output_mask) =
        get_bin_counts_and_mask(output_tokens_tensor, vocab_size, num_seqs)?;

    logits = (logits - (freq_penalties.unsqueeze(D::Minus1)? * output_bin_counts)?)?;
    logits = (logits - (presence_penalties.unsqueeze(D::Minus1)? * output_mask)?)?;
    Ok(logits)
}

fn apply_topk_topp(logits: &Tensor, p: &Tensor, k: &Tensor) -> Result<Tensor> {
    // https://github.com/vllm-project/vllm/blob/740374d456a638df98ffbc7d9dab328752330e62/vllm/model_executor/layers/sampler.py#L266
    let (logits_sort, logits_idx) = logits.sort_last_dim(true)?;

    // Apply topk
    let topk_mask = (k.to_dtype(DType::I64)?.neg()? + logits_sort.dim(1)? as f64)?;
    // Get all topk values
    let topk_mask = logits_sort.gather(&topk_mask.unsqueeze(1)?, 1)?;
    let topk_mask = logits_sort.lt(&topk_mask)?;
    let logits_sort = masked_fill(&logits_sort, &topk_mask, f64::NEG_INFINITY)?;

    // Apply topp
    let probs_sort = candle_nn::ops::softmax_last_dim(&logits_sort)?;
    let probs_sum = probs_sort.cumsum(D::Minus1)?;
    let topp_mask = probs_sum.le(&(p.unsqueeze(1)?.neg()? + 1f64)?)?;
    // At least one
    // Equivalent of Pytorch `topp_mask[:, -1] = False`
    let topp_mask = topp_mask.slice_assign(
        &[&.., &(topk_mask.dim(1)? - 1..)],
        &Tensor::zeros(
            (topk_mask.dim(0)?, 1),
            topp_mask.dtype(),
            topp_mask.device(),
        )?,
    )?;
    let logits_sort = masked_fill(&logits_sort, &topp_mask, f64::NEG_INFINITY)?;

    // Resort the probs
    let src = Tensor::arange(0, logits_idx.dim(D::Minus1)? as u32, logits_idx.device())?;
    let logits_idx_inv = logits_idx
        .zeros_like()?
        .scatter_add(&logits_idx, &src, D::Minus1)?;

    logits_sort.gather(&logits_idx_inv, D::Minus1)
}

fn apply_minp(logits: &Tensor, minp: &Tensor) -> Result<Tensor> {
    // https://github.com/vllm-project/vllm/blob/740374d456a638df98ffbc7d9dab328752330e62/vllm/model_executor/layers/sampler.py#L298
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let top_probs = probs.max_keepdim(D::Minus1)?;
    let scaled_minp = (minp.unsqueeze(1)? * top_probs)?;
    let toks_to_remove = probs.lt(&scaled_minp)?;
    masked_fill(logits, &toks_to_remove, f64::NEG_INFINITY)
}

// (bs, vocab) -> (bs)
/// Approximation of multinomial
fn sample_gumbel(probs: &Tensor) -> Result<Tensor> {
    // https://github.com/lucidrains/speculative-decoding/blob/main/speculative_decoding/speculative_decoding.py#L36
    let uniform_dist = Tensor::rand(0f32, 1f32, probs.shape(), probs.device())?;
    let gumbel_noise = uniform_dist.log()?.neg()?.log()?.neg()?;
    // Already applied temperature, no need to again
    candle_nn::ops::softmax_last_dim(&(probs + gumbel_noise)?)?.argmax(D::Minus1)
}

mod tests {
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    use tokenizers::Tokenizer;

    #[allow(dead_code)]
    fn get_tokenizer() -> Tokenizer {
        let api = ApiBuilder::new().with_progress(true).build().unwrap();
        let api = api.repo(Repo::with_revision(
            "EricB/mistralrs_tests".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = api.get("tokenizer.json").unwrap();
        Tokenizer::from_file(tokenizer_filename).unwrap()
    }

    #[test]
    fn test_argmax() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::Arc;
        use std::sync::Mutex;

        let sampler = Sampler::new(
            None,
            10,
            get_tokenizer().into(),
            None,
            None,
            None,
            32,
            0.1,
            0.05,
        );
        let logits = Tensor::arange(0f32, 1024f32, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler.sample(logits, None, false, rng, false).unwrap();
        assert_eq!(res.token, 1023);
        assert_eq!(res.top_logprobs, None);
        assert_eq!(res.logprob, 1023f64.log(10.) as f32)
    }

    #[test]
    fn test_gumbel_speculative() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::Arc;
        use std::sync::Mutex;

        let sampler = Sampler::new(
            None,
            10,
            get_tokenizer().into(),
            None,
            None,
            None,
            32,
            0.1,
            0.05,
        );
        let logits = Tensor::arange(0f32, 1024f32, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler.sample(logits, None, false, rng, true).unwrap();
        assert_eq!(res.token, 1023);
        assert_eq!(res.top_logprobs, None);
        assert_eq!(res.logprob, 1023f64.log(10.) as f32)
    }
}
