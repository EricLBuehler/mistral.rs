#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock, Mutex},
};

use candle_core::{DType, Device, Error, Result, Tensor, D};
use mistralrs_quant::{CumSumOp, SortOp};
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use rand::distr::{weighted::WeightedIndex, Distribution};
use rand_isaac::Isaac64Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

static DRY_SEQUENCE_BREAKERS: LazyLock<Vec<String>> =
    LazyLock::new(|| ["\n", ":", "\"", "*"].map(String::from).to_vec());

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Stop sequences or ids.
pub enum StopTokens {
    Seqs(Vec<String>),
    Ids(Vec<u32>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Sampling params are used to control sampling.
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub top_n_logprobs: usize,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
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
            top_k: Some(1),
            top_p: None,
            min_p: None,
            top_n_logprobs: 0,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_toks: None,
            max_len: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        }
    }
}

/// Parameters for DRY (Don't Repeat Yourself) sampling to reduce repetition.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
                            .encode_fast(["a", &breaker].concat(), true)
                            .map_err(anyhow::Error::msg)
                            .map(|enc| {
                                let ids = enc.get_ids();
                                if !ids.is_empty() {
                                    Some(ids[ids.len() - 1])
                                } else {
                                    None
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
    repetition_penalty: Option<f32>,
    dry_params: Option<DrySamplingParamsInner>,
    top_k: i64,
    top_p: f64,
    min_p: f64,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    /// Cached Gumbel noise tensor to avoid reallocating it.
    gumbel_cache: Arc<Mutex<Option<Tensor>>>,
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

/// Comparator for descending order by probability (second element of tuple).
#[inline]
fn cmp_desc_by_prob(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
}

/// Returns the top-k (index, probability) pairs from `probs`, sorted in descending order.
/// Uses partial sort (O(n) + O(k log k)) instead of full sort (O(n log n)).
///
/// If `k >= probs.len()`, returns all elements sorted.
/// Also zeros out elements in `probs` beyond top-k if `zero_rest` is true.
fn partial_sort_top_k(probs: &mut [f32], k: usize, zero_rest: bool) -> Vec<(u32, f32)> {
    let n = probs.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    // Build (index, probability) pairs
    let mut idx_probs: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, probs[i as usize])).collect();

    let k = k.min(n);

    if k < n {
        // Partial sort: partition so top k elements are in first k positions
        // select_nth_unstable_by places the k-1th largest at position k-1,
        // with all larger elements before it (unsorted) and smaller after
        idx_probs.select_nth_unstable_by(k - 1, cmp_desc_by_prob);

        if zero_rest {
            // Zero out elements beyond top-k
            for (idx, _) in idx_probs[k..].iter() {
                probs[*idx as usize] = 0.0;
            }
        }

        // Truncate to top k
        idx_probs.truncate(k);
    }

    // Sort just the top k elements (descending by probability)
    idx_probs.sort_unstable_by(cmp_desc_by_prob);

    idx_probs
}

/// Find the index of the maximum element in a slice. O(n) scan.
#[inline]
fn argmax_f32(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

impl Sampler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temperature: Option<f64>,
        top_n_logprobs: usize,
        tokenizer: Option<Arc<Tokenizer>>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        repetition_penalty: Option<f32>,
        dry_params: Option<DrySamplingParams>,
        top_k: i64,
        top_p: f64,
        min_p: f64,
        logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    ) -> anyhow::Result<Self> {
        let temperature = if temperature.is_none_or(|v| v < 1e-7) {
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
            repetition_penalty,
            dry_params,
            top_k,
            top_p,
            min_p,
            logits_processors,
            gumbel_cache: Arc::new(Mutex::new(None)),
        })
    }

    fn get_top_logprobs(&self, probs: &[f32]) -> Result<Vec<TopLogprob>> {
        let k = self.top_n_logprobs.min(probs.len());
        if k == 0 {
            return Ok(Vec::new());
        }

        // Use partial sort helper (doesn't modify probs since we pass a copy)
        let mut probs_copy = probs.to_vec();
        let top_k = partial_sort_top_k(&mut probs_copy, k, false);

        // Build the result vector with log10 of probabilities and optional decoding
        let mut result = Vec::with_capacity(k);
        if let Some(tokenizer) = &self.tokenizer {
            for (token, prob) in top_k {
                let decoded = tokenizer
                    .decode(&[token], false)
                    .map_err(|e| Error::Msg(e.to_string()))?;
                result.push(TopLogprob {
                    token,
                    logprob: prob.log(10.0),
                    bytes: Some(decoded),
                });
            }
        } else {
            for (token, prob) in top_k {
                result.push(TopLogprob {
                    token,
                    logprob: prob.log(10.0),
                    bytes: None,
                });
            }
        }
        Ok(result)
    }

    fn sample_argmax(&self, logits: Tensor, return_logprobs: bool) -> Result<Logprobs> {
        let probs: Vec<f32> = logits.to_vec1()?;
        let next_token = argmax_f32(&probs);
        let logprob = probs[next_token as usize].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(&probs)?)
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

    #[allow(unused)]
    fn sample_fast(
        &self,
        logits: Tensor,
        context: &[u32],
        return_logprobs: bool,
        top_k: i64,
        top_p: f64,
        min_p: f64,
    ) -> Result<Logprobs> {
        let mut probs = logits.to_dtype(DType::F32)?;

        for processor in &self.logits_processors {
            probs = processor.apply(&probs, context)?;
        }

        let context = Tensor::new(context, logits.device())?;
        let mut counts = logits.zeros_like()?;
        counts = counts.scatter_add(
            &context,
            &context.ones_like()?.to_dtype(counts.dtype())?,
            D::Minus1,
        )?;

        let presence = counts
            .gt(0.)?
            .where_cond(&counts.ones_like()?, &counts.zeros_like()?)?;

        match self.frequency_penalty {
            Some(freq_penalty) if freq_penalty != 0. => {
                probs = (probs - (freq_penalty as f64 * counts)?)?;
            }
            _ => (),
        }

        match self.presence_penalty {
            Some(pres_penalty) if pres_penalty != 0. => {
                probs = (probs - (pres_penalty as f64 * &presence)?)?;
            }
            _ => (),
        }

        match self.repetition_penalty {
            Some(rep_penalty) if rep_penalty != 1. => {
                let pos_mask = probs.gt(0.)?;
                let scaled_pos = (&probs / (rep_penalty as f64))?;
                let scaled_neg = (&probs * (rep_penalty as f64))?;
                let modified = pos_mask.where_cond(&scaled_pos, &scaled_neg)?;

                let pres_mask = presence.gt(0.)?;
                probs = pres_mask.where_cond(&modified, &probs)?;
            }
            _ => (),
        }

        probs = candle_nn::ops::softmax_last_dim(&(probs / self.temperature.unwrap_or(1.))?)?;

        // Top-K
        if top_k > 0 {
            let sorted_values = probs.fast_sort_asc(D::Minus1)?;
            let topk_values = sorted_values.narrow(
                D::Minus1,
                sorted_values.dim(D::Minus1)? - top_k as usize,
                top_k as usize,
            )?;

            // select the kth largest value as threshold
            let threshold = topk_values.get_on_dim(D::Minus1, 0)?.unsqueeze(0)?;
            let mask_topk = probs.broadcast_ge(&threshold)?;
            probs = mask_topk.where_cond(&probs, &Tensor::zeros_like(&probs)?)?;
        }

        // Top-P (nucleus)
        if top_p > 0.0 && top_p < 1.0 {
            let sorted_probs = probs.fast_sort_asc(D::Minus1)?;

            let cumsum = sorted_probs.fast_cumsum(D::Minus1)?;

            let mask_topp = cumsum.le(top_p)?;

            let masked_sorted =
                mask_topp.where_cond(&sorted_probs, &Tensor::zeros_like(&sorted_probs)?)?;

            let threshold = masked_sorted.max(D::Minus1)?;
            let threshold = threshold.unsqueeze(D::Minus1)?;
            let mask_full = probs.broadcast_ge(&threshold)?;
            probs = mask_full.where_cond(&probs, &Tensor::zeros_like(&probs)?)?;
        }

        // Min-P
        if min_p > 0.0 && min_p < 1.0 {
            let max_vals = probs.max(D::Minus1)?;
            let threshold_min = (max_vals.unsqueeze(D::Minus1)? * min_p)?;
            let mask_minp = probs.broadcast_gt(&threshold_min)?;
            probs = mask_minp.where_cond(&probs, &Tensor::zeros_like(&probs)?)?;
        }

        // Sample using the Gumbel-max trick fully on-device.
        let log_probs = probs.log()?;
        // Generate cached Gumbel noise (-log(-log(u))) once.
        let gumbel = {
            let mut guard = self.gumbel_cache.lock().unwrap();
            if guard.is_none() {
                let uniform = Tensor::rand(0f32, 1f32, log_probs.shape(), log_probs.device())?;
                let noise = uniform
                    .clamp(1e-20, 1.0)?
                    .log()? // ln(u)
                    .neg()? // -ln(u)
                    .log()? // ln(-ln(u))
                    .neg()?; // -ln(-ln(u))
                *guard = Some(noise);
            }
            guard.as_ref().unwrap().clone()
        };

        let gumbel_logits = (&log_probs + &gumbel)?;
        let next_token = gumbel_logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

        // Extract the top‑n log‑probs if the caller asked for them.
        let (top_logprobs, logprob) = if return_logprobs {
            let k = self.top_n_logprobs;

            let sorted_values = probs.fast_sort_asc(D::Minus1)?;
            let topk_values = sorted_values
                .narrow(
                    D::Minus1,
                    sorted_values.dim(D::Minus1)? - top_k as usize,
                    top_k as usize,
                )?
                .to_vec1::<f32>()?;

            let sorted_idxs = probs.fast_argsort_asc(D::Minus1)?;
            let topk_idxs = sorted_idxs
                .narrow(
                    D::Minus1,
                    sorted_values.dim(D::Minus1)? - top_k as usize,
                    top_k as usize,
                )?
                .to_vec1::<u32>()?;

            let mut result = Vec::with_capacity(k);
            if let Some(tokenizer) = &self.tokenizer {
                for (prob, token) in topk_values.iter().zip(topk_idxs) {
                    let decoded = tokenizer
                        .decode(&[token], false)
                        .map_err(|e| Error::Msg(e.to_string()))?;
                    result.push(TopLogprob {
                        token,
                        logprob: prob.log(10.0),
                        bytes: Some(decoded),
                    });
                }
            } else {
                for (prob, token) in topk_values.iter().zip(topk_idxs) {
                    result.push(TopLogprob {
                        token,
                        logprob: prob.log(10.0),
                        bytes: None,
                    });
                }
            }

            let logprob = result.last().map(|res| res.logprob).unwrap_or(1.);

            (Some(result), logprob)
        } else {
            (None, 1.)
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

        // Determine how many elements we need for partial sort
        let k = if top_k > 0 {
            top_k as usize
        } else {
            probs.len()
        };

        // Get sorted top-k indices with partial sort, zeroing out rest
        let idx_probs = partial_sort_top_k(&mut probs, k, true);

        // TOP P
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for (index, prob) in &idx_probs {
            if cumsum >= top_p {
                probs[*index as usize] = 0.0;
            } else {
                cumsum += prob;
            }
        }

        // Get max_p from first sorted element
        let max_p = idx_probs.first().map(|(_, p)| *p).unwrap_or(0.0);

        // MIN P
        // min-p sampling samples from the tokens whose prob are greater than
        // (max prob of token in dist) * min_p

        // Clamp smaller probabilities to zero.
        let min_p_threshold = max_p * min_p;
        for (index, prob) in &idx_probs {
            if min_p_threshold >= *prob {
                probs[*index as usize] = 0.0;
            }
        }

        // Find argmax directly on the Vec (O(n) scan, no Tensor creation)
        let next_token = argmax_f32(&probs);
        let logprob = probs[next_token as usize].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(&probs)?)
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
        probs: &[f32],
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        let distr = WeightedIndex::new(probs).map_err(Error::wrap)?;

        let mut mut_ref_rng = &mut *rng.lock().expect("could not lock rng mutex");
        let next_token = distr.sample(&mut mut_ref_rng); // "Find the first item which has a weight *higher* than the chosen weight."
        let logprob = probs[next_token].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(probs)?)
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

    #[allow(clippy::too_many_arguments)]
    fn sample_top_kp_min_p(
        &self,
        probs: &mut [f32],
        top_k: i64,
        top_p: f32,
        min_p: f32,
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        // Determine how many elements we need for partial sort
        let k = if top_k > 0 {
            top_k as usize
        } else {
            probs.len()
        };

        // Get sorted top-k indices with partial sort, zeroing out rest
        let idx_probs = partial_sort_top_k(probs, k, true);

        if top_p <= 0.0 || top_p >= 1.0 {
            return self.sample_multinomial(probs, return_logprobs, rng);
        }

        // TOP P

        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for (index, prob) in &idx_probs {
            if cumsum >= top_p {
                probs[*index as usize] = 0.0;
            } else {
                cumsum += prob;
            }
        }

        if min_p <= 0.0 || min_p >= 1.0 {
            return self.sample_multinomial(probs, return_logprobs, rng);
        }

        // Get max_p from first sorted element
        let max_p = idx_probs.first().map(|(_, p)| *p).unwrap_or(0.0);

        // MIN P

        // min-p sampling samples from the tokens whose prob are greater than
        // (max prob of token in dist) * min_p

        // Clamp smaller probabilities to zero.
        let min_p_threshold = max_p * min_p;
        for (index, prob) in &idx_probs {
            if min_p_threshold >= *prob {
                probs[*index as usize] = 0.0;
            }
        }

        // Sample with clamped probabilities.
        self.sample_multinomial(probs, return_logprobs, rng)
    }

    fn apply_penalties(&self, mut logits: Vec<f32>, context: &[u32]) -> Result<Tensor> {
        if context.is_empty() {
            candle_core::bail!("Penalty context is empty, this should not happen.");
        }

        // Dry penalty
        self.apply_dry_penalty(&mut logits, context)?;

        // Frequency, presence, repetition penalty
        self.apply_freq_pres_rep_penalty(&mut logits, context)?;

        let vocab_size = logits.len();
        Tensor::from_vec(logits, vocab_size, &Device::Cpu)
    }

    fn apply_freq_pres_rep_penalty(&self, logits: &mut [f32], context: &[u32]) -> Result<()> {
        if self.frequency_penalty.is_some()
            || self.presence_penalty.is_some()
            || self.repetition_penalty.is_some()
        {
            let frequency_penalty = self.frequency_penalty.unwrap_or(0.);
            let presence_penalty = self.presence_penalty.unwrap_or(0.);
            let repetition_penalty = self.repetition_penalty.unwrap_or(1.);

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

                if repetition_penalty != 1.0 && count > 0.0 {
                    if *logit > 0.0 {
                        *logit /= repetition_penalty;
                    } else {
                        *logit *= repetition_penalty;
                    }
                }
            }
        }
        Ok(())
    }

    /// Threshold for using parallel iteration in dry penalty.
    /// Below this, sequential is faster due to parallel overhead.
    const DRY_PENALTY_PAR_THRESHOLD: usize = 1024;

    fn apply_dry_penalty(&self, logits: &mut [f32], context: &[u32]) -> Result<()> {
        if let Some(ref params) = self.dry_params {
            if params.multiplier == 0. {
                return Ok(());
            }

            let last_token = *context.last().unwrap();

            // Use parallel iteration only for large contexts
            let match_indices: Vec<usize> = if context.len() > Self::DRY_PENALTY_PAR_THRESHOLD {
                context
                    .par_iter()
                    .enumerate()
                    .take(context.len() - 1)
                    .filter(|(_i, x)| last_token == **x)
                    .map(|(i, _)| i)
                    .collect()
            } else {
                context
                    .iter()
                    .enumerate()
                    .take(context.len() - 1)
                    .filter(|(_i, x)| last_token == **x)
                    .map(|(i, _)| i)
                    .collect()
            };

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

    #[allow(unused)]
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
        multiple_sequences: bool,
    ) -> Result<Logprobs> {
        // if cfg!(feature = "metal") && !multiple_sequences {
        //     return self.sample_fast(
        //         logits,
        //         context,
        //         return_logprobs,
        //         self.top_k,
        //         self.top_p,
        //         self.min_p,
        //     );
        // }

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
            None,
            None,
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
            .sample(
                logits,
                &(0..1024).collect::<Vec<_>>(),
                false,
                rng,
                false,
                false,
            )
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
            None,
            None,
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
            .sample(
                logits,
                &(0..1024).collect::<Vec<_>>(),
                false,
                rng,
                true,
                false,
            )
            .unwrap();
        assert_eq!(res.token, 1023);
        assert_eq!(res.top_logprobs, None);
        assert_eq!(res.logprob, 1023f64.log(10.) as f32)
    }
}
