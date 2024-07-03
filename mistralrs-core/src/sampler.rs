#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    iter::zip,
    sync::{Arc, Mutex},
};

use candle_core::{bail, Device, Error, Result, Tensor, D};
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use rand::distributions::{Distribution, WeightedIndex};
use rand_isaac::Isaac64Rng;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

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
    topk: i64,
    topp: f64,
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
        topk: i64,
        topp: f64,
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
            topk,
            topp,
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

    fn sample_speculative_topkp(
        &self,
        logits: Tensor,
        return_logprobs: bool,
        top_k: i64,
        top_p: f32,
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

    fn sample_topkp(
        &self,
        probs: &mut Vec<f32>,
        top_k: i64,
        top_p: f32,
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
                None => self.sample_speculative_topkp(
                    logits,
                    return_logprobs,
                    self.topk,
                    self.topp as f32,
                )?,
                Some(temperature) => {
                    let logits = (&logits / temperature)?;
                    let probs = candle_nn::ops::softmax_last_dim(&logits)?;

                    self.sample_speculative_topkp(
                        probs,
                        return_logprobs,
                        self.topk,
                        self.topp as f32,
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

                    self.sample_topkp(
                        &mut probs,
                        self.topk,
                        self.topp as f32,
                        return_logprobs,
                        rng,
                    )?
                }
            }
        };
        Ok(next_token)
    }
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

        let sampler = Sampler::new(None, 10, get_tokenizer().into(), None, None, None, 32, 0.1);
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

        let sampler = Sampler::new(None, 10, get_tokenizer().into(), None, None, None, 32, 0.1);
        let logits = Tensor::arange(0f32, 1024f32, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler.sample(logits, None, false, rng, true).unwrap();
        assert_eq!(res.token, 1023);
        assert_eq!(res.top_logprobs, None);
        assert_eq!(res.logprob, 1023f64.log(10.) as f32)
    }
}
