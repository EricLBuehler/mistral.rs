#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, iter::zip};

use candle_core::{bail, DType, Error, Result, Tensor, D};
use rand::{
    distributions::{Distribution, WeightedIndex},
    SeedableRng,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub enum StopTokens {
    Seqs(Vec<String>),
    Ids(Vec<u32>),
}

#[derive(Clone, Debug)]
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

/// Sampler for sampling.
#[derive(Clone)]
pub struct Sampler {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    top_n_logprobs: usize,
    tokenizer: Tokenizer,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    logits_bias: Option<HashMap<u32, f32>>,
    topk: i64,
    topp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
// Top-n logprobs element
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

impl Sampler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        top_n_logprobs: usize,
        tokenizer: Tokenizer,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        logits_bias: Option<HashMap<u32, f32>>,
        topk: i64,
        topp: f64,
    ) -> Self {
        let temperature = if temperature.map_or(true, |v| v < 1e-7) {
            None
        } else {
            temperature
        };
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
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

    fn apply_logit_bias(&self, probs: &mut [f32]) -> Result<()> {
        if let Some(ref bias) = self.logits_bias {
            for (id, bias_v) in bias {
                let idx = probs.get_mut(*id as usize);
                if let Some(idx) = idx {
                    *idx += bias_v;
                } else {
                    candle_core::bail!(
                        "Token ID `{id}` out of range for probs of length `{}`.",
                        probs.len()
                    );
                }
            }
        }
        Ok(())
    }

    fn get_top_logprobs(
        &self,
        probs: &[f32],
        argsort_indices: &[usize],
    ) -> Result<Vec<TopLogprob>> {
        let mut argsort_indices_sorted = argsort_indices.to_vec();
        // Sort by descending prob
        argsort_indices_sorted.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());
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
                    .decode(&[*tok as u32], true)
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

    fn sample_argmax(&mut self, logits: Tensor, return_logprobs: bool) -> Result<Logprobs> {
        let mut probs: Vec<f32> = logits.to_vec1()?;
        let argsort_indices = (0..probs.len()).collect::<Vec<_>>();

        self.apply_logit_bias(&mut probs)?;

        let next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

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
                .decode(&[next_token], true)
                .map_err(|x| Error::Msg(x.to_string()))?,
        })
    }

    fn sample_multinomial(
        &mut self,
        probs: &mut Vec<f32>,
        argsort_indices: Vec<usize>,
        return_logprobs: bool,
    ) -> Result<Logprobs> {
        self.apply_logit_bias(probs)?;

        let distr = WeightedIndex::new(&*probs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng); // "Find the first item which has a weight *higher* than the chosen weight."
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
                .decode(&[next_token.try_into().unwrap()], true)
                .map_err(|x| Error::Msg(x.to_string()))?,
        })
    }

    fn sample_topkp(
        &mut self,
        probs: &mut Vec<f32>,
        top_k: i64,
        top_p: f32,
        return_logprobs: bool,
    ) -> Result<Logprobs> {
        let mut argsort_indices = (0..probs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        if top_k > 0 {
            // Clamp smaller probabilities to zero.
            for (index, val) in argsort_indices.iter().enumerate() {
                if index >= top_k as usize {
                    probs[*val] = 0.0;
                }
            }
        }

        if top_p <= 0.0 || top_p >= 1.0 {
            return self.sample_multinomial(probs, argsort_indices, return_logprobs);
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
        self.sample_multinomial(probs, argsort_indices, return_logprobs)
    }

    fn apply_repeat_presence_penalty(
        logits: &Tensor,
        presence_penalty: f32,
        frequency_penalty: f32,
        context: &[u32],
    ) -> Result<Tensor> {
        //mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
        let device = logits.device();
        let mut logits = logits.to_vec1::<f32>()?;
        for (token_id, logit) in logits.iter_mut().enumerate() {
            let count = context.iter().filter(|x| **x as usize == token_id).count();
            *logit = *logit
                - count as f32 * frequency_penalty
                - if count > 0 { 1. } else { 0. } * presence_penalty;
        }
        let logits_len = logits.len();
        Tensor::from_vec(logits, logits_len, device)
    }

    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    /// If `frequency_penalty.is_some()` or `presence_penalty.is_some()`, then `penalty_ctxt` must be provided.
    pub fn sample(
        &mut self,
        logits: &Tensor,
        penalty_ctxt: Option<&[u32]>,
        return_logprobs: bool,
    ) -> Result<Logprobs> {
        let logits = logits.to_dtype(DType::F32)?;

        let logits = if self.frequency_penalty.is_none() && self.presence_penalty.is_none() {
            logits
        } else {
            if penalty_ctxt.is_none() {
                bail!("Must specify penalty context.");
            }
            Self::apply_repeat_presence_penalty(
                &logits,
                self.presence_penalty.unwrap_or(0.),
                self.frequency_penalty.unwrap_or(0.),
                penalty_ctxt.unwrap(),
            )?
        };

        let next_token = match self.temperature {
            None => self.sample_argmax(logits, return_logprobs)?,
            Some(temperature) => {
                let logits = (&logits / temperature)?;
                let probs = candle_nn::ops::softmax_last_dim(&logits)?;
                let mut probs: Vec<f32> = probs.to_vec1()?;
                self.sample_topkp(&mut probs, self.topk, self.topp as f32, return_logprobs)?
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
            "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
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

        let mut sampler = Sampler::new(0, None, 10, get_tokenizer(), None, None, None, 32, 0.1);

        let logits = Tensor::arange(0f64, 1024f64, &Device::Cpu).unwrap();
        let res = sampler.sample(&logits, None, false).unwrap();
        assert_eq!(res.token, 1023);
        assert_eq!(res.top_logprobs, None);
        assert_eq!(res.logprob, 1023f64.log(10.) as f32)
    }
}
