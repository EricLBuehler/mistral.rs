#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::{HashMap, HashSet},
    iter::zip,
    sync::{Arc, Mutex},
};

use candle_core::{Device, Error, Result, Tensor, D};
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use once_cell::sync::Lazy;
use rand::distributions::{Distribution, WeightedIndex};
use rand_isaac::Isaac64Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

static DRY_SEQUENCE_BREAKERS: Lazy<Vec<String>> =
    Lazy::new(|| ["\n", ":", "\"", "*"].map(String::from).to_vec());

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
    pub dry_params: Option<DrySamplingParams>,
}

impl SamplingParams {
    /// This sets up the parameters so that there is:
    /// - No temperature, topk, topp, minp
    /// - No penalties, stop tokens, or logit bias
    /// - No maximum length
    pub fn deterministic() -> Self {
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
            dry_params: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DrySamplingParams {
    pub sequence_breakers: Vec<String>,
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
}

impl DrySamplingParams {
    pub fn new_with_defaults(
        multiplier: f32,
        sequence_breakers: Option<Vec<String>>,
        base: Option<f32>,
        allowed_length: Option<usize>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            base: base.unwrap_or(1.75),
            allowed_length: allowed_length.unwrap_or(2),
            sequence_breakers: sequence_breakers.unwrap_or(DRY_SEQUENCE_BREAKERS.clone()),
            multiplier,
        })
    }
}

impl Default for DrySamplingParams {
    fn default() -> Self {
        Self {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            sequence_breakers: DRY_SEQUENCE_BREAKERS.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct DrySamplingParamsInner {
    pub sequence_breakers: HashSet<u32>,
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
}

impl DrySamplingParamsInner {
    pub fn from(other: DrySamplingParams, tokenizer: &Tokenizer) -> anyhow::Result<Self> {
        Ok(Self {
            base: other.base,
            allowed_length: other.allowed_length,
            sequence_breakers: HashSet::from_iter(
                other
                    .sequence_breakers
                    .into_iter()
                    .map(|breaker| {
                        tokenizer
                            // Prefix with 'a' to get the correct encoding of the token at the end of a text.
                            //
                            // FIXME: This is a hack. See https://github.com/LostRuins/koboldcpp/pull/982
                            //        for the correct solution which covers multi-token sequence breakers
                            //        and ambiguous encodings.
                            .encode(["a", &breaker].concat(), true)
                            .map_err(anyhow::Error::msg)
                            .map(|enc| {
                                let ids = enc.get_ids();
                                if !ids.is_empty() {
                                    None
                                } else {
                                    Some(ids[ids.len() - 1])
                                }
                            })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            ),
            multiplier: other.multiplier,
        })
    }
}

/// Customizable logits processor.
///
/// # Example
/// ```rust
/// use std::{sync::Arc, ops::Mul};
/// use mistralrs_core::CustomLogitsProcessor;
/// use candle_core::{Result, Tensor};
///
/// struct ThresholdLogitsProcessor;
/// impl CustomLogitsProcessor for ThresholdLogitsProcessor {
///     fn apply(&self, logits: &Tensor, _context: &[u32]) -> Result<Tensor> {
///         // Mask is 1 for true, 0 for false.
///         let mask = logits.ge(0.5)?;
///         logits.broadcast_mul(&mask.to_dtype(logits.dtype())?)
///     }
/// }
/// let processor1: Arc<dyn CustomLogitsProcessor> = Arc::new(|logits: &Tensor, _context: &[u32]| logits * 1.23);
/// let processor2: Arc<dyn CustomLogitsProcessor> = Arc::new(ThresholdLogitsProcessor);
/// ```
pub trait CustomLogitsProcessor: Send + Sync {
    /// Logits and sequence context (prompt and generated tokens), returning modified tokens.
    fn apply(&self, logits: &Tensor, context: &[u32]) -> Result<Tensor>;
}

impl<T: Fn(&Tensor, &[u32]) -> Result<Tensor> + Send + Sync> CustomLogitsProcessor for T {
    fn apply(&self, logits: &Tensor, context: &[u32]) -> Result<Tensor> {
        self(logits, context)
    }
}

/// Sampler for sampling.
#[derive(Clone)]
pub struct Sampler {
    temperature: Option<f64>,
    top_n_logprobs: usize,
    tokenizer: Option<Arc<Tokenizer>>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    dry_params: Option<DrySamplingParamsInner>,
    top_k: i64,
    top_p: f64,
    min_p: f64,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Top-n logprobs element
pub struct TopLogprob {
    pub token: u32,
    pub logprob: f32,
    pub bytes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprobs {
    pub token: u32,
    pub logprob: f32,
    pub bytes: Option<String>,
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
        tokenizer: Option<Arc<Tokenizer>>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        dry_params: Option<DrySamplingParams>,
        top_k: i64,
        top_p: f64,
        min_p: f64,
        logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    ) -> anyhow::Result<Self> {
        let temperature = if temperature.map_or(true, |v| v < 1e-7) {
            None
        } else {
            temperature
        };
        let dry_params = if let Some(ref tokenizer) = tokenizer {
            dry_params.map(|params| DrySamplingParamsInner::from(params, tokenizer))
        } else {
            None
        };
        let dry_params = match dry_params {
            Some(fallible) => Some(fallible?),
            None => None,
        };
        Ok(Self {
            temperature,
            top_n_logprobs,
            tokenizer,
            frequency_penalty,
            presence_penalty,
            dry_params,
            top_k,
            top_p,
            min_p,
            logits_processors,
        })
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

        if let Some(tokenizer) = &self.tokenizer {
            let mut bytes = Vec::new();
            for tok in &top_n_toks {
                bytes.push(
                    tokenizer
                        .decode(&[*tok as u32], false)
                        .map_err(|x| Error::Msg(x.to_string()))?,
                );
            }

            Ok(zip(bytes, zip(top_n_toks, top_n_logprobs))
                .map(|(bytes, (token, logprob))| TopLogprob {
                    token: token as u32,
                    logprob,
                    bytes: Some(bytes),
                })
                .collect::<Vec<_>>())
        } else {
            Ok(zip(top_n_toks, top_n_logprobs)
                .map(|(token, logprob)| TopLogprob {
                    token: token as u32,
                    logprob,
                    bytes: None,
                })
                .collect::<Vec<_>>())
        }
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

        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[next_token], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs,
            bytes,
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

        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[next_token], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs,
            bytes,
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

        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[next_token.try_into().unwrap()], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token as u32,
            logprob,
            top_logprobs,
            bytes,
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

    fn apply_penalties(&self, mut logits: Vec<f32>, context: &[u32]) -> Result<Tensor> {
        if context.is_empty() {
            candle_core::bail!("Penalty context is empty, this should not happen.");
        }

        // Dry penalty
        self.apply_dry_penalty(&mut logits, context)?;

        // Frequency and Presence penalty
        self.apply_freq_presc_penalty(&mut logits, context)?;

        let vocab_size = logits.len();
        Tensor::from_vec(logits, vocab_size, &Device::Cpu)
    }

    fn apply_freq_presc_penalty(&self, logits: &mut [f32], context: &[u32]) -> Result<()> {
        if self.frequency_penalty.is_some() || self.presence_penalty.is_some() {
            let frequency_penalty = self.frequency_penalty.unwrap_or(0.);
            let presence_penalty = self.presence_penalty.unwrap_or(0.);

            //mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence

            let mut counts = vec![0.0f32; logits.len()];
            for ctx in context.iter() {
                // Llama 3.2 uses a hack triggering this error... we wouldn't want a weight on it anyway
                if *ctx as usize >= logits.len() {
                    continue;
                }
                counts[*ctx as usize] += 1.0;
            }

            for (token_id, logit) in logits.iter_mut().enumerate() {
                let count = counts[token_id];
                *logit = *logit
                    - count * frequency_penalty
                    - if count > 0.0 { 1. } else { 0. } * presence_penalty;
            }
        }
        Ok(())
    }

    fn apply_dry_penalty(&self, logits: &mut [f32], context: &[u32]) -> Result<()> {
        if let Some(ref params) = self.dry_params {
            let match_indices = context
                .par_iter()
                .enumerate()
                .take(context.len() - 1)
                .filter(|(_i, x)| *context.last().unwrap() == **x)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            let mut match_lengths = HashMap::new();

            for i in match_indices {
                let next_token = context[i + 1];

                if params.sequence_breakers.contains(&next_token) {
                    continue;
                }

                let mut match_length = 1;

                // Limit match length to avoid quadratic runtime and potential DoS with adversarial inputs.
                while match_length < 50 {
                    if match_length > i {
                        // Start of input
                        break;
                    }

                    let j = i - match_length;

                    let prev_tok = context[context.len() - (match_length + 1)];
                    if context[j] != prev_tok {
                        // Start of match reached
                        break;
                    }

                    if params.sequence_breakers.contains(&prev_tok) {
                        // Seq breaking tok reached
                        break;
                    }

                    match_length += 1;
                }

                #[allow(clippy::map_entry)]
                if match_lengths.contains_key(&next_token) {
                    match_lengths.insert(next_token, match_length.max(match_lengths[&next_token]));
                } else {
                    match_lengths.insert(next_token, match_length);
                }
            }

            // Actually apply penalties
            for (tok, match_len) in match_lengths {
                if match_len >= params.allowed_length {
                    // Llama 3.2 uses a hack triggering this error... we wouldn't want a weight on it anyway
                    if tok as usize >= logits.len() {
                        continue;
                    }
                    let penalty = params.multiplier
                        * params.base.powf((match_len - params.allowed_length) as f32);
                    logits[tok as usize] -= penalty;
                }
            }
        }
        Ok(())
    }

    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    pub fn sample(
        &self,
        logits: Tensor,
        context: &[u32],
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
        sample_speculative: bool,
    ) -> Result<Logprobs> {
        let logits = logits.to_vec1()?;
        let mut logits = self.apply_penalties(logits, context)?;
        for processor in &self.logits_processors {
            logits = processor.apply(&logits, context)?;
        }
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
            Some(get_tokenizer().into()),
            None,
            None,
            None,
            32,
            0.1,
            0.05,
            vec![],
        )
        .unwrap();
        let logits = Tensor::arange(0f32, 1024f32, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler
            .sample(logits, &(0..1024).collect::<Vec<_>>(), false, rng, false)
            .unwrap();
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
            Some(get_tokenizer().into()),
            None,
            None,
            None,
            32,
            0.1,
            0.05,
            vec![],
        )
        .unwrap();
        let logits = Tensor::arange(0f32, 1024f32, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler
            .sample(logits, &(0..1024).collect::<Vec<_>>(), false, rng, true)
            .unwrap();
        assert_eq!(res.token, 1023);
        assert_eq!(res.top_logprobs, None);
        assert_eq!(res.logprob, 1023f64.log(10.) as f32)
    }
}
