#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock, Mutex},
};

use candle_core::{Device, Error, Result, Tensor};
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use rand::distr::{weighted::WeightedIndex, Distribution};
use rand_isaac::Isaac64Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

static DRY_SEQUENCE_BREAKERS: LazyLock<Vec<String>> =
    LazyLock::new(|| ["\n", ":", "\"", "*"].map(String::from).to_vec());
const SUPPRESS_TOKEN_LOGIT_BIAS: f32 = -1.0e9;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
/// Optional generation defaults parsed from a model's `generation_config.json`.
///
/// These defaults are descriptive and opt-in: consumers may choose to apply them,
/// partially apply them, or ignore them entirely.
pub struct ModelGenerationDefaults {
    pub do_sample: Option<bool>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub max_new_tokens: Option<usize>,
    pub max_length: Option<usize>,
    pub suppress_tokens: Option<Vec<u32>>,
}

impl ModelGenerationDefaults {
    pub fn is_empty(&self) -> bool {
        self.do_sample.is_none()
            && self.temperature.is_none()
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.repetition_penalty.is_none()
            && self.max_new_tokens.is_none()
            && self.max_length.is_none()
            && self.suppress_tokens.is_none()
    }
}

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
    #[serde(default)]
    pub ignore_eos: bool,
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
    ///
    /// Unlike [`Self::deterministic`], this does not force `top_k = 1`.
    pub fn neutral() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            top_n_logprobs: 0,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_toks: None,
            ignore_eos: false,
            max_len: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        }
    }

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
            ignore_eos: false,
            max_len: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        }
    }

    /// Fills the sampling fields a request left unset from the model's generation defaults.
    pub fn fill_model_defaults(&mut self, defaults: &ModelGenerationDefaults) {
        let mut filled = SamplingParams::neutral();
        filled.apply_model_defaults(defaults);
        self.temperature = self.temperature.or(filled.temperature);
        self.top_k = self.top_k.or(filled.top_k);
        self.top_p = self.top_p.or(filled.top_p);
        self.min_p = self.min_p.or(filled.min_p);
        self.repetition_penalty = self.repetition_penalty.or(filled.repetition_penalty);
        if let Some(defaults_bias) = filled.logits_bias {
            let logits_bias = self.logits_bias.get_or_insert_with(HashMap::new);
            for (token, bias) in defaults_bias {
                logits_bias.entry(token).or_insert(bias);
            }
        }
    }

    /// Applies model-level generation defaults onto this request-local sampler config.
    ///
    /// This is opt-in and only updates fields that the model default explicitly provides.
    pub fn apply_model_defaults(&mut self, defaults: &ModelGenerationDefaults) {
        if defaults.do_sample == Some(false) {
            self.temperature = None;
            self.top_k = Some(1);
            self.top_p = None;
            self.min_p = None;
        } else {
            if let Some(temperature) = defaults.temperature {
                self.temperature = Some(temperature);
            }
            if let Some(top_k) = defaults.top_k {
                self.top_k = if top_k == 0 { None } else { Some(top_k) };
            }
            if let Some(top_p) = defaults.top_p {
                self.top_p = Some(top_p);
            }
            if let Some(min_p) = defaults.min_p {
                self.min_p = Some(min_p);
            }
        }
        if let Some(repetition_penalty) = defaults.repetition_penalty {
            self.repetition_penalty = Some(repetition_penalty);
        }
        if let Some(max_new_tokens) = defaults.max_new_tokens {
            self.max_len = Some(max_new_tokens);
        }
        if let Some(suppress_tokens) = &defaults.suppress_tokens {
            let logits_bias = self.logits_bias.get_or_insert_with(HashMap::new);
            for token in suppress_tokens {
                logits_bias
                    .entry(*token)
                    .or_insert(SUPPRESS_TOKEN_LOGIT_BIAS);
            }
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
    logits_bias: HashMap<u32, f32>,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    #[cfg(feature = "cuda")]
    top1_cache: Arc<Mutex<Option<crate::ops::CudaTop1LogitsWorkspace>>>,
    #[cfg(feature = "cuda")]
    topk_sampling_cache: Arc<Mutex<Option<crate::ops::CudaTopKSamplingWorkspace>>>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub(crate) enum CudaBatchSamplingKind {
    Greedy,
    TopK { k: usize },
    Categorical,
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTop1BatchCompletion {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) packed: Option<Vec<[f32; crate::ops::CUDA_TOP1_PACKED_WIDTH]>>,
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTop1BatchSubmission {
    cache: Arc<Mutex<Option<crate::ops::CudaTop1LogitsWorkspace>>>,
    submission: Option<crate::ops::CudaTop1Submission>,
}

#[cfg(feature = "cuda")]
impl CudaTop1BatchSubmission {
    pub(crate) fn batch_size(&self) -> usize {
        self.submission
            .as_ref()
            .expect("CUDA top-1 submission was already completed")
            .batch_size()
    }

    pub(crate) fn wait_on(
        &self,
        stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let mut cache = self.cache.lock().unwrap();
        crate::ops::cuda_top1_device_tokens_wait_on(
            cache
                .as_mut()
                .expect("CUDA top-1 workspace exists while its submission is active"),
            self.submission
                .as_ref()
                .expect("CUDA top-1 submission was already completed"),
            stream,
        )
    }

    pub(crate) fn release_after(
        &self,
        stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let mut cache = self.cache.lock().unwrap();
        crate::ops::cuda_top1_device_tokens_release_after(
            cache
                .as_mut()
                .expect("CUDA top-1 workspace exists while its submission is active"),
            self.submission
                .as_ref()
                .expect("CUDA top-1 submission was already completed"),
            stream,
        )
    }

    pub(crate) fn complete(mut self) -> Result<CudaTop1BatchCompletion> {
        let submission = self
            .submission
            .as_ref()
            .expect("CUDA top-1 submission was already completed");
        submission.wait()?;
        let (token_ids, packed) = {
            let mut cache = self.cache.lock().unwrap();
            let completion = crate::ops::cuda_top1_submission_complete(
                cache
                    .as_mut()
                    .expect("CUDA top-1 workspace exists while its submission is active"),
                submission,
            )?;
            let token_ids = completion.token_ids().to_vec();
            let packed = completion.packed().map(|values| {
                values
                    .as_chunks::<{ crate::ops::CUDA_TOP1_PACKED_WIDTH }>()
                    .0
                    .iter()
                    .map(|row| [row[0], row[1]])
                    .collect()
            });
            (token_ids, packed)
        };
        self.submission = None;
        if packed.is_none() && token_ids.contains(&crate::ops::CUDA_TOP1_INVALID_TOKEN) {
            candle_core::bail!("invalid CUDA top-1 output");
        }
        Ok(CudaTop1BatchCompletion { token_ids, packed })
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaTop1BatchSubmission {
    fn drop(&mut self) {
        let Some(submission) = self.submission.take() else {
            return;
        };
        let _ = submission.wait();
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(workspace) = cache.as_mut() {
            let _ = crate::ops::cuda_top1_submission_cancel(workspace, &submission);
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTopKBatchSubmission {
    cache: Arc<Mutex<Option<crate::ops::CudaTopKSamplingWorkspace>>>,
    submission: Option<crate::ops::CudaTopKSamplingSubmission>,
}

#[cfg(feature = "cuda")]
impl CudaTopKBatchSubmission {
    pub(crate) fn batch_size(&self) -> usize {
        self.submission
            .as_ref()
            .expect("CUDA top-k submission was already completed")
            .batch_size()
    }

    pub(crate) fn wait_on(
        &self,
        stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let mut cache = self.cache.lock().unwrap();
        crate::ops::cuda_topk_sampling_device_tokens_wait_on(
            cache
                .as_mut()
                .expect("CUDA top-k workspace exists while its submission is active"),
            self.submission
                .as_ref()
                .expect("CUDA top-k submission was already completed"),
            stream,
        )
    }

    pub(crate) fn release_after(
        &self,
        stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let mut cache = self.cache.lock().unwrap();
        crate::ops::cuda_topk_sampling_device_tokens_release_after(
            cache
                .as_mut()
                .expect("CUDA top-k workspace exists while its submission is active"),
            self.submission
                .as_ref()
                .expect("CUDA top-k submission was already completed"),
            stream,
        )
    }

    pub(crate) fn complete(mut self) -> Result<Vec<u32>> {
        let submission = self
            .submission
            .as_ref()
            .expect("CUDA top-k submission was already completed");
        submission.wait()?;
        let token_ids = {
            let mut cache = self.cache.lock().unwrap();
            crate::ops::cuda_topk_sampling_submission_complete(
                cache
                    .as_mut()
                    .expect("CUDA top-k workspace exists while its submission is active"),
                submission,
            )?
            .token_ids()
            .to_vec()
        };
        self.submission = None;
        Ok(token_ids)
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaTopKBatchSubmission {
    fn drop(&mut self) {
        let Some(submission) = self.submission.take() else {
            return;
        };
        let _ = submission.wait();
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(workspace) = cache.as_mut() {
            let _ = crate::ops::cuda_topk_sampling_submission_cancel(workspace, &submission);
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaBatchSamplingKind {
    pub(crate) fn is_argmax(self) -> bool {
        matches!(self, Self::Greedy | Self::TopK { k: 1 })
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaBatchSamplingPlan {
    pub(crate) kind: CudaBatchSamplingKind,
    pub(crate) inverse_temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) min_p: f32,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaSpeculativeSamplingPlan {
    pub(crate) inverse_temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) min_p: f32,
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

pub(crate) struct SpeculativeProbs {
    pub sampling: Vec<f32>,
    pub reporting: Vec<f32>,
}

/// Comparator for descending order by probability (second element of tuple).
#[inline]
fn cmp_desc_by_prob(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
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
fn argmax_f32(values: &[f32]) -> Result<u32> {
    let mut best_index = None;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value.is_nan() || value == f32::INFINITY {
            candle_core::bail!("argmax received invalid logits");
        }
        if value > best_value {
            best_index = Some(index);
            best_value = value;
        }
    }
    if best_value == f32::NEG_INFINITY {
        candle_core::bail!("argmax received no finite logits");
    }
    Ok(best_index.expect("finite argmax value exists") as u32)
}

/// Nucleus mass measured on the distribution renormalized over the surviving top-k set (HF, vLLM, llama.cpp).
fn top_p_cutoff(top_p: f32, kept_probs: impl Iterator<Item = f32>) -> f32 {
    top_p * kept_probs.sum::<f32>()
}

#[cfg(all(feature = "cuda", test))]
fn weighted_index_from_unit_f32(weights: &[f32], unit: f32) -> Result<usize> {
    if weights.is_empty() || !unit.is_finite() || !(0.0..1.0).contains(&unit) {
        candle_core::bail!("invalid resident sampling weights or uniform");
    }
    let mut total = 0.0f32;
    for &weight in weights {
        if !weight.is_finite() || weight < 0.0 {
            candle_core::bail!("invalid resident sampling weight");
        }
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        candle_core::bail!("resident sampling weights have no positive mass");
    }

    let chosen = unit * total;
    let mut cumulative = 0.0f32;
    let mut last_positive = None;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if weight > 0.0 {
            last_positive = Some(index);
        }
        if cumulative > chosen {
            return Ok(index);
        }
    }
    Ok(last_positive.expect("positive sampling mass was checked above"))
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn sparse_token_counts(
    context: &[u32],
    vocab_size: usize,
    device: &Device,
) -> Result<Option<(Tensor, Tensor)>> {
    let mut counts = HashMap::<u32, f32>::with_capacity(context.len().min(vocab_size));
    for &token_id in context {
        if token_id as usize >= vocab_size {
            continue;
        }
        *counts.entry(token_id).or_insert(0.0) += 1.0;
    }
    if counts.is_empty() {
        return Ok(None);
    }
    let n_tokens = counts.len();
    let (token_ids, token_counts): (Vec<u32>, Vec<f32>) = counts.into_iter().unzip();
    Ok(Some((
        Tensor::from_vec(token_ids, n_tokens, device)?,
        Tensor::from_vec(token_counts, n_tokens, device)?,
    )))
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
        logits_bias: HashMap<u32, f32>,
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
            logits_bias,
            logits_processors,
            #[cfg(feature = "cuda")]
            top1_cache: Arc::new(Mutex::new(None)),
            #[cfg(feature = "cuda")]
            topk_sampling_cache: Arc::new(Mutex::new(None)),
        })
    }

    pub fn is_argmax(&self) -> bool {
        self.temperature.is_none()
    }

    pub(crate) fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_batch_sampling_plan(
        &self,
        return_logprobs: bool,
    ) -> Option<CudaBatchSamplingPlan> {
        let has_penalties = self.frequency_penalty.unwrap_or(0.0) != 0.0
            || self.presence_penalty.unwrap_or(0.0) != 0.0
            || self.repetition_penalty.unwrap_or(1.0) != 1.0;
        let has_dry_penalty = self
            .dry_params
            .as_ref()
            .is_some_and(|params| params.multiplier != 0.0);
        if return_logprobs
            || has_penalties
            || has_dry_penalty
            || !self.logits_bias.is_empty()
            || !self.logits_processors.is_empty()
        {
            return None;
        }

        match self.temperature {
            None => Some(CudaBatchSamplingPlan {
                kind: CudaBatchSamplingKind::Greedy,
                inverse_temperature: 1.0,
                top_p: 1.0,
                min_p: 0.0,
            }),
            Some(temperature) if temperature.is_finite() && temperature > 0.0 => {
                let inverse_temperature = (1.0 / temperature) as f32;
                if !inverse_temperature.is_finite() || inverse_temperature <= 0.0 {
                    return None;
                }
                let has_top_p_filter = self.top_p > 0.0 && self.top_p < 1.0;
                let has_min_p_filter = self.min_p > 0.0 && self.min_p < 1.0;
                let kind = if self.top_k > 0 {
                    let k = usize::try_from(self.top_k).ok()?;
                    if k > crate::ops::CUDA_TOPK_MAX_K {
                        return None;
                    }
                    CudaBatchSamplingKind::TopK { k }
                } else if !(has_top_p_filter || has_min_p_filter) {
                    CudaBatchSamplingKind::Categorical
                } else {
                    return None;
                };
                Some(CudaBatchSamplingPlan {
                    kind,
                    inverse_temperature,
                    top_p: self.top_p as f32,
                    min_p: self.min_p as f32,
                })
            }
            Some(_) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_speculative_sampling_plan(
        &self,
        return_logprobs: bool,
    ) -> Option<CudaSpeculativeSamplingPlan> {
        let plan = self.cuda_batch_sampling_plan(return_logprobs)?;
        let top_k = match plan.kind {
            CudaBatchSamplingKind::Greedy => return None,
            CudaBatchSamplingKind::Categorical => 0,
            CudaBatchSamplingKind::TopK { k } => k,
        };
        let top_p = plan.top_p;
        let min_p = plan.min_p;
        if !top_p.is_finite() || !min_p.is_finite() {
            return None;
        }
        Some(CudaSpeculativeSamplingPlan {
            inverse_temperature: plan.inverse_temperature,
            top_k,
            top_p,
            min_p,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_resident_sampling_plan(
        &self,
        return_logprobs: bool,
    ) -> Option<CudaBatchSamplingPlan> {
        let plan = self.cuda_batch_sampling_plan(return_logprobs)?;
        if matches!(plan.kind, CudaBatchSamplingKind::Categorical)
            || !plan.top_p.is_finite()
            || !plan.min_p.is_finite()
        {
            return None;
        }
        Some(plan)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn draw_cuda_resident_uniform(rng: &mut Isaac64Rng) -> f32 {
        use rand::distr::Uniform;

        Uniform::new(0.0f32, 1.0f32)
            .expect("valid unit uniform distribution")
            .sample(rng)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn sample_cuda_topk_packed_row(
        &self,
        packed: &[f32],
        packed_k: usize,
        plan: CudaBatchSamplingPlan,
        rng: &mut Isaac64Rng,
    ) -> Result<Logprobs> {
        let expected = 2 * packed_k + 2;
        if packed.len() != expected {
            candle_core::bail!(
                "invalid batched CUDA top-k row length {}, expected {expected}",
                packed.len()
            );
        }
        let row_k = match plan.kind {
            CudaBatchSamplingKind::Greedy => 1,
            CudaBatchSamplingKind::TopK { k } => k.min(packed_k),
            CudaBatchSamplingKind::Categorical => {
                candle_core::bail!("categorical plan cannot parse CUDA top-k output")
            }
        };
        let top_values = &packed[..row_k];
        let top_indices = packed[packed_k..packed_k + row_k]
            .iter()
            .map(|idx| *idx as u32)
            .collect::<Vec<_>>();
        let denom = packed[2 * packed_k];
        let global_max = packed[2 * packed_k + 1];
        if denom <= 0.0 || !denom.is_finite() || !global_max.is_finite() {
            candle_core::bail!("invalid batched CUDA top-k softmax normalizer");
        }

        let reporting_probs = top_values
            .iter()
            .map(|value| ((*value * plan.inverse_temperature - global_max).exp()) / denom)
            .collect::<Vec<_>>();
        self.sample_cuda_topk_probabilities(
            &top_indices,
            reporting_probs,
            matches!(plan.kind, CudaBatchSamplingKind::Greedy) || row_k == 1,
            rng,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn sample_cuda_ranked_topk_packed_row(
        &self,
        packed: &[f32],
        packed_k: usize,
        plan: CudaBatchSamplingPlan,
        rng: &mut Isaac64Rng,
    ) -> Result<Logprobs> {
        let expected = 2 * packed_k;
        if packed.len() != expected {
            candle_core::bail!(
                "invalid batched CUDA ranked top-k row length {}, expected {expected}",
                packed.len()
            );
        }
        let row_k = match plan.kind {
            CudaBatchSamplingKind::Greedy => 1,
            CudaBatchSamplingKind::TopK { k } => k.min(packed_k),
            CudaBatchSamplingKind::Categorical => {
                candle_core::bail!("categorical plan cannot parse CUDA ranked top-k output")
            }
        };
        let top_values = &packed[..row_k];
        let top_indices = packed[packed_k..packed_k + row_k]
            .iter()
            .map(|idx| *idx as u32)
            .collect::<Vec<_>>();
        let scaled_max =
            top_values.first().copied().unwrap_or(f32::NEG_INFINITY) * plan.inverse_temperature;
        if !scaled_max.is_finite() {
            candle_core::bail!("invalid batched CUDA ranked top-k maximum");
        }
        let mut reporting_probs = top_values
            .iter()
            .map(|value| (*value * plan.inverse_temperature - scaled_max).exp())
            .collect::<Vec<_>>();
        let denominator = reporting_probs.iter().sum::<f32>();
        if denominator <= 0.0 || !denominator.is_finite() {
            candle_core::bail!("invalid batched CUDA ranked top-k normalizer");
        }
        for probability in &mut reporting_probs {
            *probability /= denominator;
        }
        self.sample_cuda_topk_probabilities(
            &top_indices,
            reporting_probs,
            matches!(plan.kind, CudaBatchSamplingKind::Greedy) || row_k == 1,
            rng,
        )
    }

    #[cfg(feature = "cuda")]
    fn sample_cuda_topk_probabilities(
        &self,
        top_indices: &[u32],
        reporting_probs: Vec<f32>,
        greedy: bool,
        rng: &mut Isaac64Rng,
    ) -> Result<Logprobs> {
        let selected = if greedy {
            0
        } else {
            let mut sampling_probs = reporting_probs.clone();
            if self.top_p > 0.0 && self.top_p < 1.0 {
                let cutoff = top_p_cutoff(self.top_p as f32, sampling_probs.iter().copied());
                let mut cumsum = 0.0f32;
                for prob in &mut sampling_probs {
                    if cumsum >= cutoff {
                        *prob = 0.0;
                    } else {
                        cumsum += *prob;
                    }
                }
            }
            if self.min_p > 0.0 && self.min_p < 1.0 {
                let threshold = sampling_probs.first().copied().unwrap_or(0.0) * self.min_p as f32;
                for prob in &mut sampling_probs {
                    if threshold >= *prob {
                        *prob = 0.0;
                    }
                }
            }
            WeightedIndex::new(&sampling_probs)
                .map_err(|err| {
                    Error::Msg(format!(
                        "Failed to construct CUDA top-k multinomial sampler: {err}"
                    ))
                })?
                .sample(rng)
        };
        let next_token = top_indices[selected];

        Ok(Logprobs {
            token: next_token,
            logprob: reporting_probs[selected].ln(),
            top_logprobs: None,
            bytes: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn sample_cuda_categorical_row(&self, packed: &[f32]) -> Result<Logprobs> {
        if packed.len() != crate::ops::CUDA_CATEGORICAL_PACKED_WIDTH {
            candle_core::bail!(
                "invalid batched CUDA categorical row length {}, expected {}",
                packed.len(),
                crate::ops::CUDA_CATEGORICAL_PACKED_WIDTH
            );
        }
        let token = packed[0];
        let logprob = packed[1];
        if !token.is_finite() || token < 0.0 || token.fract() != 0.0 || !logprob.is_finite() {
            candle_core::bail!("invalid batched CUDA categorical output");
        }
        let next_token = token as u32;
        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs: None,
            bytes: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn sample_cuda_top1_row(&self, packed: &[f32]) -> Result<Logprobs> {
        if packed.len() != crate::ops::CUDA_TOP1_PACKED_WIDTH {
            candle_core::bail!(
                "invalid batched CUDA top-1 row length {}, expected {}",
                packed.len(),
                crate::ops::CUDA_TOP1_PACKED_WIDTH
            );
        }
        Ok(Logprobs {
            token: Self::cuda_top1_token([packed[0], packed[1]])?,
            logprob: 0.0,
            top_logprobs: None,
            bytes: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn sample_cuda_top1_batch(&self, logits: &Tensor) -> Result<Vec<[f32; 2]>> {
        self.submit_cuda_top1_batch(logits, true)?
            .complete()?
            .packed
            .ok_or_else(|| candle_core::Error::Msg("missing CUDA top-1 packed output".to_string()))
    }

    #[cfg(feature = "cuda")]
    fn submit_cuda_top1_batch(
        &self,
        logits: &Tensor,
        packed: bool,
    ) -> Result<CudaTop1BatchSubmission> {
        let submission = {
            let mut cache = self.top1_cache.lock().unwrap();
            if packed {
                crate::ops::cuda_top1_logits_submit_batched_packed(logits, &mut cache)?
            } else {
                crate::ops::cuda_top1_logits_submit_batched(logits, &mut cache)?
            }
        };
        Ok(CudaTop1BatchSubmission {
            cache: self.top1_cache.clone(),
            submission: Some(submission),
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn submit_cuda_top1_batch_owned(
        &self,
        logits: &Tensor,
    ) -> Result<CudaTop1BatchSubmission> {
        self.submit_cuda_top1_batch(logits, false)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn submit_cuda_top1_batch_into(
        &self,
        logits: &Tensor,
        token_ids_dst: &Tensor,
    ) -> Result<CudaTop1BatchSubmission> {
        let submission = {
            let mut cache = self.top1_cache.lock().unwrap();
            crate::ops::cuda_top1_logits_submit_batched_into(logits, token_ids_dst, &mut cache)?
        };
        Ok(CudaTop1BatchSubmission {
            cache: self.top1_cache.clone(),
            submission: Some(submission),
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn submit_cuda_topk_batch_owned(
        &self,
        logits: &Tensor,
        params: &[crate::ops::CudaTopKSamplingParams],
    ) -> Result<CudaTopKBatchSubmission> {
        let submission = {
            let mut cache = self.topk_sampling_cache.lock().unwrap();
            crate::ops::cuda_topk_sampling_submit_batched(logits, params, &mut cache)?
        };
        Ok(CudaTopKBatchSubmission {
            cache: self.topk_sampling_cache.clone(),
            submission: Some(submission),
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn submit_cuda_topk_batch_into(
        &self,
        logits: &Tensor,
        token_ids_dst: &Tensor,
        params: &[crate::ops::CudaTopKSamplingParams],
    ) -> Result<CudaTopKBatchSubmission> {
        let submission = {
            let mut cache = self.topk_sampling_cache.lock().unwrap();
            crate::ops::cuda_topk_sampling_submit_batched_into(
                logits,
                token_ids_dst,
                params,
                &mut cache,
            )?
        };
        Ok(CudaTopKBatchSubmission {
            cache: self.topk_sampling_cache.clone(),
            submission: Some(submission),
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

        // Build the result vector with natural log of probabilities and optional decoding
        let mut result = Vec::with_capacity(k);
        if let Some(tokenizer) = &self.tokenizer {
            for (token, prob) in top_k {
                let decoded = tokenizer
                    .decode(&[token], false)
                    .map_err(|e| Error::Msg(e.to_string()))?;
                result.push(TopLogprob {
                    token,
                    logprob: prob.ln(),
                    bytes: Some(decoded),
                });
            }
        } else {
            for (token, prob) in top_k {
                result.push(TopLogprob {
                    token,
                    logprob: prob.ln(),
                    bytes: None,
                });
            }
        }
        Ok(result)
    }

    fn sample_argmax(&self, logits: Tensor, return_logprobs: bool) -> Result<Logprobs> {
        let next_token = argmax_f32(&logits.to_vec1::<f32>()?)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?.to_vec1::<f32>()?;
        let logprob = probs[next_token as usize].ln();

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

    fn sample_speculative_top_kp_min_p(
        &self,
        logits: Tensor,
        return_logprobs: bool,
        top_k: i64,
        top_p: f32,
        min_p: f32,
    ) -> Result<Logprobs> {
        let mut probs: Vec<f32> = logits.to_vec1()?;
        let reporting_probs = probs.clone();

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
        let cutoff = top_p_cutoff(top_p, idx_probs.iter().map(|(_, p)| *p));
        let mut cumsum = 0.;
        for (index, prob) in &idx_probs {
            if cumsum >= cutoff {
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
        let next_token = argmax_f32(&probs)?;
        let logprob = reporting_probs[next_token as usize].ln();

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(&reporting_probs)?)
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
        sampling_probs: &[f32],
        reporting_probs: &[f32],
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        let distr = match WeightedIndex::new(sampling_probs) {
            Ok(distr) => distr,
            Err(e) => {
                if let Some((idx, prob)) = sampling_probs
                    .iter()
                    .enumerate()
                    .find(|(_, prob)| !prob.is_finite() || **prob < 0.0)
                {
                    return Err(Error::Msg(format!(
                        "Invalid sampling probability at index {idx}: {prob}. The model likely produced NaN/Inf logits."
                    )));
                }

                let positive_weight_sum: f64 = sampling_probs
                    .iter()
                    .copied()
                    .filter(|prob| prob.is_finite() && *prob > 0.0)
                    .map(f64::from)
                    .sum();

                if positive_weight_sum == 0.0 {
                    return Err(Error::Msg(
                        "All sampling probabilities are zero after filtering (top-k/top-p/min-p)."
                            .to_string(),
                    ));
                }

                return Err(Error::Msg(format!(
                    "Failed to construct multinomial sampler: {e}"
                )));
            }
        };

        let mut mut_ref_rng = &mut *rng.lock().expect("could not lock rng mutex");
        let next_token = distr.sample(&mut mut_ref_rng); // "Find the first item which has a weight *higher* than the chosen weight."
        let logprob = reporting_probs[next_token].ln();

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(reporting_probs)?)
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

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn can_sample_topk_on_device(
        &self,
        return_logprobs: bool,
        sample_speculative: bool,
        multiple_sequences: bool,
        supports_logits_bias: bool,
    ) -> bool {
        const MAX_DEVICE_TOP_K: i64 = 128;

        !return_logprobs
            && !sample_speculative
            && !multiple_sequences
            && self.temperature.is_some()
            && self.top_k > 0
            && self.top_k <= MAX_DEVICE_TOP_K
            && (supports_logits_bias || self.logits_bias.is_empty())
            && self.logits_processors.is_empty()
            && self
                .dry_params
                .as_ref()
                .is_none_or(|params| params.multiplier == 0.0)
    }

    #[cfg(feature = "cuda")]
    fn can_sample_greedy_on_device(
        &self,
        return_logprobs: bool,
        sample_speculative: bool,
        multiple_sequences: bool,
    ) -> bool {
        !return_logprobs
            && !sample_speculative
            && !multiple_sequences
            && self.temperature.is_none()
            && self.logits_processors.is_empty()
            && self
                .dry_params
                .as_ref()
                .is_none_or(|params| params.multiplier == 0.0)
    }

    #[cfg(feature = "cuda")]
    fn apply_device_sparse_penalties_if_needed(
        &self,
        logits: Tensor,
        context: &[u32],
        prompt_len: usize,
    ) -> Result<Tensor> {
        let frequency_penalty = self.frequency_penalty.unwrap_or(0.0);
        let presence_penalty = self.presence_penalty.unwrap_or(0.0);
        let repetition_penalty = self.repetition_penalty.unwrap_or(1.0);
        let needs_penalty =
            frequency_penalty != 0.0 || presence_penalty != 0.0 || repetition_penalty != 1.0;

        if !needs_penalty {
            return Ok(logits);
        }
        if context.is_empty() {
            candle_core::bail!("Penalty context is empty, this should not happen.");
        }

        let vocab_size = logits.elem_count();
        let mut logits = logits;
        if frequency_penalty != 0.0 || presence_penalty != 0.0 {
            if let Some((token_ids, token_counts)) = sparse_token_counts(
                &context[prompt_len.min(context.len())..],
                vocab_size,
                logits.device(),
            )? {
                logits = crate::ops::cuda_apply_sparse_penalties_f32(
                    &logits,
                    &token_ids,
                    &token_counts,
                    frequency_penalty,
                    presence_penalty,
                    1.0,
                )?;
            }
        }
        if repetition_penalty != 1.0 {
            if let Some((token_ids, token_counts)) =
                sparse_token_counts(context, vocab_size, logits.device())?
            {
                logits = crate::ops::cuda_apply_sparse_penalties_f32(
                    &logits,
                    &token_ids,
                    &token_counts,
                    0.0,
                    0.0,
                    repetition_penalty,
                )?;
            }
        }
        Ok(logits)
    }

    #[cfg(feature = "cuda")]
    fn apply_device_logits_bias_if_needed(&self, logits: Tensor) -> Result<Tensor> {
        if self.logits_bias.is_empty() {
            return Ok(logits);
        }

        let vocab_size = logits.elem_count();
        let mut token_ids = Vec::with_capacity(self.logits_bias.len());
        let mut biases = Vec::with_capacity(self.logits_bias.len());
        for (&token_id, &bias) in &self.logits_bias {
            if token_id as usize >= vocab_size || bias == 0.0 {
                continue;
            }
            token_ids.push(token_id);
            biases.push(bias);
        }
        if token_ids.is_empty() {
            return Ok(logits);
        }

        let n_tokens = token_ids.len();
        let device = logits.device();
        let token_ids = Tensor::from_vec(token_ids, n_tokens, device)?;
        let biases = Tensor::from_vec(biases, n_tokens, device)?;
        crate::ops::cuda_apply_sparse_logits_bias_f32(&logits, &token_ids, &biases)
    }

    #[cfg(feature = "cuda")]
    fn sample_topk_on_device(
        &self,
        logits: Tensor,
        temperature: f64,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        if self.top_k == 1 {
            let packed = {
                let mut cache = self.top1_cache.lock().unwrap();
                crate::ops::cuda_top1_logits_f32_cached(&logits, &mut cache)?
            };
            return self.sample_cuda_top1_row(&packed);
        }

        let topk =
            crate::ops::cuda_topk_logits_f32_packed(&logits, self.top_k as usize, temperature)?;
        let packed = topk.packed.to_vec1::<f32>()?;
        let k = topk.k;
        if packed.len() != 2 * k + 2 {
            candle_core::bail!(
                "invalid CUDA top-k packed output length {}, expected {}",
                packed.len(),
                2 * k + 2
            );
        }
        let top_values = &packed[..k];
        let top_indices = packed[k..2 * k]
            .iter()
            .map(|idx| *idx as u32)
            .collect::<Vec<_>>();
        let softmax_info = &packed[2 * k..2 * k + 2];

        let denom = softmax_info[0];
        let global_max = softmax_info[1];
        if denom <= 0.0 || !denom.is_finite() || !global_max.is_finite() {
            candle_core::bail!("invalid CUDA top-k softmax normalizer");
        }

        let inv_temperature = (1.0 / temperature) as f32;
        let reporting_probs = top_values
            .iter()
            .map(|value| ((*value * inv_temperature - global_max).exp()) / denom)
            .collect::<Vec<_>>();
        let mut probs = reporting_probs.clone();

        if self.top_p > 0.0 && self.top_p < 1.0 {
            let cutoff = top_p_cutoff(self.top_p as f32, probs.iter().copied());
            let mut cumsum = 0.0f32;
            for prob in &mut probs {
                if cumsum >= cutoff {
                    *prob = 0.0;
                } else {
                    cumsum += *prob;
                }
            }
        }

        if self.min_p > 0.0 && self.min_p < 1.0 {
            let max_p = probs.first().copied().unwrap_or(0.0);
            let min_p_threshold = max_p * self.min_p as f32;
            for prob in &mut probs {
                if min_p_threshold >= *prob {
                    *prob = 0.0;
                }
            }
        }

        let distr = match WeightedIndex::new(&probs) {
            Ok(distr) => distr,
            Err(e) => {
                let positive_weight_sum: f64 = probs
                    .iter()
                    .copied()
                    .filter(|prob| prob.is_finite() && *prob > 0.0)
                    .map(f64::from)
                    .sum();
                if positive_weight_sum == 0.0 {
                    return Err(Error::Msg(
                        "All sampling probabilities are zero after CUDA top-k filtering."
                            .to_string(),
                    ));
                }

                return Err(Error::Msg(format!(
                    "Failed to construct CUDA top-k multinomial sampler: {e}"
                )));
            }
        };

        let mut mut_ref_rng = &mut *rng.lock().expect("could not lock rng mutex");
        let selected = distr.sample(&mut mut_ref_rng);
        let next_token = top_indices[selected];
        let logprob = reporting_probs[selected].ln();

        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs: None,
            bytes: None,
        })
    }

    #[cfg(feature = "cuda")]
    fn sample_greedy_on_device(&self, logits: Tensor) -> Result<Logprobs> {
        let packed = {
            let mut cache = self.top1_cache.lock().unwrap();
            crate::ops::cuda_top1_logits_f32_cached(&logits, &mut cache)?
        };
        self.sample_cuda_top1_row(&packed)
    }

    #[cfg(feature = "cuda")]
    fn cuda_top1_token(packed: [f32; 2]) -> Result<u32> {
        if !packed[0].is_finite()
            || !packed[1].is_finite()
            || packed[1] < 0.0
            || packed[1].fract() != 0.0
        {
            candle_core::bail!(
                "invalid CUDA top-1 output: max_logit={} argmax={}",
                packed[0],
                packed[1]
            );
        }
        Ok(packed[1] as u32)
    }

    #[cfg(feature = "metal")]
    fn apply_device_sparse_penalties_if_needed_metal(
        &self,
        logits: Tensor,
        context: &[u32],
        prompt_len: usize,
    ) -> Result<Tensor> {
        let frequency_penalty = self.frequency_penalty.unwrap_or(0.0);
        let presence_penalty = self.presence_penalty.unwrap_or(0.0);
        let repetition_penalty = self.repetition_penalty.unwrap_or(1.0);
        let needs_penalty = frequency_penalty.abs() > f32::EPSILON
            || presence_penalty.abs() > f32::EPSILON
            || (repetition_penalty - 1.0).abs() > f32::EPSILON;
        if !needs_penalty || context.is_empty() {
            return Ok(logits);
        }
        let vocab_size = logits.elem_count();
        let mut logits = logits;
        if frequency_penalty.abs() > f32::EPSILON || presence_penalty.abs() > f32::EPSILON {
            if let Some((token_ids, token_counts)) = sparse_token_counts(
                &context[prompt_len.min(context.len())..],
                vocab_size,
                logits.device(),
            )? {
                logits = crate::ops::metal_apply_sparse_penalties(
                    &logits,
                    &token_ids,
                    &token_counts,
                    frequency_penalty,
                    presence_penalty,
                    1.0,
                )?;
            }
        }
        if (repetition_penalty - 1.0).abs() > f32::EPSILON {
            if let Some((token_ids, token_counts)) =
                sparse_token_counts(context, vocab_size, logits.device())?
            {
                logits = crate::ops::metal_apply_sparse_penalties(
                    &logits,
                    &token_ids,
                    &token_counts,
                    0.0,
                    0.0,
                    repetition_penalty,
                )?;
            }
        }
        Ok(logits)
    }

    #[cfg(feature = "metal")]
    fn sample_topk_on_device_metal(
        &self,
        logits: Tensor,
        temperature: f64,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        let topk = crate::ops::metal_topk_logits_packed(&logits, self.top_k as usize, temperature)?;
        let packed = topk.packed.to_vec1::<f32>()?;
        let k = topk.k;
        if packed.len() != 2 * k + 2 {
            candle_core::bail!(
                "invalid Metal top-k packed output length {}, expected {}",
                packed.len(),
                2 * k + 2
            );
        }
        let top_values = &packed[..k];
        let top_indices = packed[k..2 * k]
            .iter()
            .map(|idx| *idx as u32)
            .collect::<Vec<_>>();
        let softmax_info = &packed[2 * k..2 * k + 2];
        let denom = softmax_info[0];
        let global_max = softmax_info[1];
        if denom <= 0.0 || !denom.is_finite() || !global_max.is_finite() {
            candle_core::bail!("invalid Metal top-k softmax normalizer");
        }

        let inv_temperature = (1.0 / temperature) as f32;
        let mut probs = top_values
            .iter()
            .map(|value| ((*value * inv_temperature - global_max).exp()) / denom)
            .collect::<Vec<_>>();

        if self.top_p > 0.0 && self.top_p < 1.0 {
            let cutoff = top_p_cutoff(self.top_p as f32, probs.iter().copied());
            let mut cumsum = 0.0f32;
            for prob in &mut probs {
                if cumsum >= cutoff {
                    *prob = 0.0;
                } else {
                    cumsum += *prob;
                }
            }
        }
        if self.min_p > 0.0 && self.min_p < 1.0 {
            let max_p = probs.first().copied().unwrap_or(0.0);
            let min_p_threshold = max_p * self.min_p as f32;
            for prob in &mut probs {
                if min_p_threshold >= *prob {
                    *prob = 0.0;
                }
            }
        }

        let distr = match WeightedIndex::new(&probs) {
            Ok(distr) => distr,
            Err(e) => {
                let positive_weight_sum: f64 = probs
                    .iter()
                    .copied()
                    .filter(|prob| prob.is_finite() && *prob > 0.0)
                    .map(f64::from)
                    .sum();
                if positive_weight_sum == 0.0 {
                    return Err(Error::Msg(
                        "All sampling probabilities are zero after Metal top-k filtering."
                            .to_string(),
                    ));
                }
                return Err(Error::Msg(format!(
                    "Failed to construct Metal top-k multinomial sampler: {e}"
                )));
            }
        };

        let mut mut_ref_rng = &mut *rng.lock().expect("could not lock rng mutex");
        let selected = distr.sample(&mut mut_ref_rng);
        let next_token = top_indices[selected];
        let logprob = probs[selected].ln();
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
            top_logprobs: None,
            bytes,
        })
    }

    fn filter_top_kp_min_p(&self, probs: &mut [f32]) {
        let k = if self.top_k > 0 {
            self.top_k as usize
        } else {
            probs.len()
        };

        let idx_probs = partial_sort_top_k(probs, k, true);

        if self.top_p > 0.0 && self.top_p < 1.0 {
            let cutoff = top_p_cutoff(self.top_p as f32, idx_probs.iter().map(|(_, p)| *p));
            let mut cumsum = 0.0f32;
            for (index, prob) in &idx_probs {
                if cumsum >= cutoff {
                    probs[*index as usize] = 0.0;
                } else {
                    cumsum += prob;
                }
            }
        }

        if self.min_p <= 0.0 || self.min_p >= 1.0 {
            return;
        }

        let max_p = idx_probs.first().map(|(_, p)| *p).unwrap_or(0.0);
        let min_p_threshold = max_p * self.min_p as f32;
        for (index, prob) in &idx_probs {
            if min_p_threshold >= *prob {
                probs[*index as usize] = 0.0;
            }
        }
    }

    fn normalize_probs(probs: &mut [f32]) -> Result<()> {
        let sum: f32 = probs
            .iter()
            .copied()
            .filter(|prob| prob.is_finite() && *prob > 0.0)
            .sum();
        if sum <= 0.0 {
            candle_core::bail!("all probabilities are zero in speculative sampling");
        }
        for prob in probs.iter_mut() {
            if prob.is_finite() && *prob > 0.0 {
                *prob /= sum;
            } else {
                *prob = 0.0;
            }
        }
        Ok(())
    }

    pub(crate) fn speculative_target_probs(
        &self,
        logits: Tensor,
        context: &[u32],
        prompt_len: usize,
    ) -> Result<SpeculativeProbs> {
        self.speculative_probs(logits, context, prompt_len)
    }

    pub(crate) fn speculative_candidate_probs(
        &self,
        logits: Tensor,
        context: &[u32],
        prompt_len: usize,
    ) -> Result<Vec<f32>> {
        Ok(self
            .speculative_probs(logits, context, prompt_len)?
            .sampling)
    }

    fn speculative_probs(
        &self,
        logits: Tensor,
        context: &[u32],
        prompt_len: usize,
    ) -> Result<SpeculativeProbs> {
        let logits = logits.to_vec1()?;
        let mut logits = self.apply_penalties(logits, context, prompt_len)?;
        for processor in &self.logits_processors {
            logits = processor.apply(&logits, context)?;
        }

        let greedy_token = match self.temperature {
            None => Some(argmax_f32(&logits.to_vec1::<f32>()?)?),
            Some(_) => None,
        };
        let reporting = match self.temperature {
            None => candle_nn::ops::softmax_last_dim(&logits)?.to_vec1::<f32>()?,
            Some(temperature) => {
                let logits = (&logits / temperature)?;
                candle_nn::ops::softmax_last_dim(&logits)?.to_vec1::<f32>()?
            }
        };
        let mut sampling = match self.temperature {
            None => {
                let mut sampling = vec![0.0; reporting.len()];
                sampling[greedy_token.expect("greedy token exists") as usize] = 1.0;
                sampling
            }
            Some(_) => reporting.clone(),
        };
        self.filter_top_kp_min_p(&mut sampling);
        Self::normalize_probs(&mut sampling)?;
        Ok(SpeculativeProbs {
            sampling,
            reporting,
        })
    }

    pub(crate) fn logprobs_from_probs(
        &self,
        token: u32,
        probs: &[f32],
        return_logprobs: bool,
    ) -> Result<Logprobs> {
        let prob = probs.get(token as usize).copied().unwrap_or(0.0);
        let logprob = if prob > 0.0 {
            prob.ln()
        } else {
            f32::NEG_INFINITY
        };
        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(probs)?)
        } else {
            None
        };
        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[token], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };
        Ok(Logprobs {
            token,
            logprob,
            top_logprobs,
            bytes,
        })
    }

    pub(crate) fn sample_from_probs(
        &self,
        sampling_probs: &[f32],
        reporting_probs: &[f32],
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        self.sample_multinomial(sampling_probs, reporting_probs, return_logprobs, rng)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_top_kp_min_p(
        &self,
        reporting_probs: &[f32],
        top_k: i64,
        top_p: f32,
        min_p: f32,
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        if top_k <= 0 && !(top_p > 0.0 && top_p < 1.0) && !(min_p > 0.0 && min_p < 1.0) {
            return self.sample_multinomial(reporting_probs, reporting_probs, return_logprobs, rng);
        }
        let mut sampling_probs = reporting_probs.to_vec();
        // Determine how many elements we need for partial sort
        let k = if top_k > 0 {
            top_k as usize
        } else {
            sampling_probs.len()
        };

        // Get sorted top-k indices with partial sort, zeroing out rest
        let idx_probs = partial_sort_top_k(&mut sampling_probs, k, true);

        if top_p > 0.0 && top_p < 1.0 {
            let cutoff = top_p_cutoff(top_p, idx_probs.iter().map(|(_, p)| *p));
            let mut cumsum = 0.;
            for (index, prob) in &idx_probs {
                if cumsum >= cutoff {
                    sampling_probs[*index as usize] = 0.0;
                } else {
                    cumsum += prob;
                }
            }
        }

        if min_p <= 0.0 || min_p >= 1.0 {
            return self.sample_multinomial(&sampling_probs, reporting_probs, return_logprobs, rng);
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
                sampling_probs[*index as usize] = 0.0;
            }
        }

        // Sample with clamped probabilities.
        self.sample_multinomial(&sampling_probs, reporting_probs, return_logprobs, rng)
    }

    fn apply_penalties(
        &self,
        mut logits: Vec<f32>,
        context: &[u32],
        prompt_len: usize,
    ) -> Result<Tensor> {
        if context.is_empty() {
            candle_core::bail!("Penalty context is empty, this should not happen.");
        }

        self.apply_dry_penalty(&mut logits, context)?;
        self.apply_freq_pres_rep_penalty(&mut logits, context, prompt_len)?;
        self.apply_logits_bias(&mut logits);

        let vocab_size = logits.len();
        Tensor::from_vec(logits, vocab_size, &Device::Cpu)
    }

    fn apply_logits_bias(&self, logits: &mut [f32]) {
        for (&token_id, &bias) in &self.logits_bias {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                *logit += bias;
            }
        }
    }

    // Frequency/presence penalties count sampled tokens only (OpenAI, vLLM); repetition penalty spans the prompt too (HF)
    fn apply_freq_pres_rep_penalty(
        &self,
        logits: &mut [f32],
        context: &[u32],
        prompt_len: usize,
    ) -> Result<()> {
        if self.frequency_penalty.is_some()
            || self.presence_penalty.is_some()
            || self.repetition_penalty.is_some()
        {
            let frequency_penalty = self.frequency_penalty.unwrap_or(0.);
            let presence_penalty = self.presence_penalty.unwrap_or(0.);
            let repetition_penalty = self.repetition_penalty.unwrap_or(1.);

            //mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence

            let mut generated_counts = vec![0.0f32; logits.len()];
            let mut seen = vec![false; logits.len()];
            for (idx, ctx) in context.iter().enumerate() {
                // Llama 3.2 uses a hack triggering this error... we wouldn't want a weight on it anyway
                if *ctx as usize >= logits.len() {
                    continue;
                }
                seen[*ctx as usize] = true;
                if idx >= prompt_len {
                    generated_counts[*ctx as usize] += 1.0;
                }
            }

            for (token_id, logit) in logits.iter_mut().enumerate() {
                let count = generated_counts[token_id];
                *logit = *logit
                    - count * frequency_penalty
                    - if count > 0.0 { 1. } else { 0. } * presence_penalty;

                if repetition_penalty != 1.0 && seen[token_id] {
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
    /// `context` is the full token history (prompt + generated); `prompt_len` marks where sampling started.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        logits: Tensor,
        context: &[u32],
        prompt_len: usize,
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
        sample_speculative: bool,
        multiple_sequences: bool,
    ) -> Result<Logprobs> {
        #[cfg(feature = "cuda")]
        if logits.device().is_cuda()
            && self.can_sample_greedy_on_device(
                return_logprobs,
                sample_speculative,
                multiple_sequences,
            )
        {
            let logits =
                self.apply_device_sparse_penalties_if_needed(logits, context, prompt_len)?;
            let logits = self.apply_device_logits_bias_if_needed(logits)?;
            return self.sample_greedy_on_device(logits);
        }

        #[cfg(feature = "cuda")]
        if logits.device().is_cuda()
            && self.can_sample_topk_on_device(
                return_logprobs,
                sample_speculative,
                multiple_sequences,
                true,
            )
        {
            if let Some(temperature) = self.temperature {
                let logits =
                    self.apply_device_sparse_penalties_if_needed(logits, context, prompt_len)?;
                let logits = self.apply_device_logits_bias_if_needed(logits)?;
                return self.sample_topk_on_device(logits, temperature, rng);
            }
        }

        #[cfg(feature = "metal")]
        if logits.device().is_metal()
            && self.can_sample_topk_on_device(
                return_logprobs,
                sample_speculative,
                multiple_sequences,
                false,
            )
        {
            if let Some(temperature) = self.temperature {
                let logits = self
                    .apply_device_sparse_penalties_if_needed_metal(logits, context, prompt_len)?;
                return self.sample_topk_on_device_metal(logits, temperature, rng);
            }
        }

        let logits = logits.to_vec1()?;
        let mut logits = self.apply_penalties(logits, context, prompt_len)?;
        for processor in &self.logits_processors {
            logits = processor.apply(&logits, context)?;
        }
        let next_token = if sample_speculative {
            match self.temperature {
                None => self.sample_speculative_top_kp_min_p(
                    candle_nn::ops::softmax_last_dim(&logits)?,
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
                    let probs: Vec<f32> = probs.to_vec1()?;

                    self.sample_top_kp_min_p(
                        &probs,
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

#[cfg(test)]
mod tests {
    use super::{argmax_f32, partial_sort_top_k, ModelGenerationDefaults, SamplingParams};
    use std::collections::HashMap;

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
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let logits = Tensor::from_vec(vec![-3.0f32, -1.0, -2.0], 3, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler
            .sample(logits, &[0, 1, 2], 0, true, rng, false, false)
            .unwrap();
        assert_eq!(res.token, 1);
        let expected = -1.0f32 - ((-3.0f32).exp() + (-1.0f32).exp() + (-2.0f32).exp()).ln();
        assert!((res.logprob - expected).abs() < 1e-6);
        let selected = res
            .top_logprobs
            .unwrap()
            .into_iter()
            .find(|top| top.token == 1)
            .unwrap();
        assert!((selected.logprob - expected).abs() < 1e-6);
    }

    #[test]
    fn argmax_uses_first_maximum_and_rejects_invalid_rows() {
        assert_eq!(argmax_f32(&[-2.0, 4.0, 4.0, 1.0]).unwrap(), 1);
        assert_eq!(argmax_f32(&[f32::NEG_INFINITY, -3.0, 2.0]).unwrap(), 2);
        assert!(argmax_f32(&[0.0, f32::NAN]).is_err());
        assert!(argmax_f32(&[0.0, f32::INFINITY]).is_err());
        assert!(argmax_f32(&[f32::NEG_INFINITY, f32::NEG_INFINITY]).is_err());
        assert!(argmax_f32(&[]).is_err());
    }

    #[test]
    fn top_k_ties_prefer_lower_token_ids_at_the_boundary() {
        let mut probs = vec![0.1, 0.5, 0.5, 0.5, 0.2];
        let selected = partial_sort_top_k(&mut probs, 2, true);

        assert_eq!(selected, vec![(1, 0.5), (2, 0.5)]);
        assert_eq!(probs, vec![0.0, 0.5, 0.5, 0.0, 0.0]);
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
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let logits = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], 3, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler
            .sample(logits, &[0, 1, 2], 0, false, rng, true, false)
            .unwrap();
        assert_eq!(res.token, 2);
        assert_eq!(res.top_logprobs, None);
        let expected = 2.0f32 - (0.0f32.exp() + 1.0f32.exp() + 2.0f32.exp()).ln();
        assert!((res.logprob - expected).abs() < 1e-6);
    }

    #[test]
    fn test_speculative_candidate_probs_use_sampling_filters() {
        use super::Sampler;
        use candle_core::{Device, Tensor};

        let sampler = Sampler::new(
            Some(1.0),
            10,
            None,
            None,
            None,
            None,
            None,
            1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let logits = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], 3, &Device::Cpu).unwrap();
        let context = [0u32];
        let target_probs = sampler
            .speculative_target_probs(logits.clone(), &context, 0)
            .unwrap();
        let candidate_probs = sampler
            .speculative_candidate_probs(logits, &context, 0)
            .unwrap();

        assert_eq!(candidate_probs, target_probs.sampling);
        assert_eq!(candidate_probs, vec![0.0, 0.0, 1.0]);
        let expected = [
            1.0f32 / (1.0 + 1.0f32.exp() + 2.0f32.exp()),
            1.0f32.exp() / (1.0 + 1.0f32.exp() + 2.0f32.exp()),
            2.0f32.exp() / (1.0 + 1.0f32.exp() + 2.0f32.exp()),
        ];
        for (actual, expected) in target_probs.reporting.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_min_p_applies_without_top_p() {
        use super::Sampler;
        use candle_core::{Device, Tensor};

        let sampler = Sampler::new(
            Some(1.0),
            0,
            None,
            None,
            None,
            None,
            None,
            3,
            1.0,
            0.4,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let logits =
            Tensor::from_vec(vec![0.0f32, 2.0f32.ln(), 4.0f32.ln()], 3, &Device::Cpu).unwrap();
        let probs = sampler
            .speculative_candidate_probs(logits, &[0], 0)
            .unwrap();

        assert!((probs[0] - 0.0).abs() < 1e-6);
        assert!((probs[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((probs[2] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_logprobs_use_unfiltered_distribution() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::{Arc, Mutex};

        let sampler = Sampler::new(
            Some(1.0),
            3,
            None,
            None,
            None,
            None,
            None,
            1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let logits =
            Tensor::from_vec(vec![0.0f32, 2.0f32.ln(), 3.0f32.ln()], 3, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));

        let result = sampler
            .sample(logits, &[0], 0, true, rng, false, false)
            .unwrap();

        assert_eq!(result.token, 2);
        let top = result.top_logprobs.unwrap();
        assert_eq!(top.len(), 3);
        let expected = [1.0f32 / 6.0, 2.0 / 6.0, 3.0 / 6.0];
        for item in top {
            assert!((item.logprob - expected[item.token as usize].ln()).abs() < 1e-6);
        }
    }

    #[test]
    fn test_logits_bias_suppresses_argmax_token() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::Arc;
        use std::sync::Mutex;

        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            HashMap::from([(2, -1.0e9)]),
            vec![],
        )
        .unwrap();
        let logits = Tensor::from_vec(vec![0.0f32, 1.0, 10.0], 3, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler
            .sample(logits, &[0], 0, false, rng, false, false)
            .unwrap();

        assert_eq!(res.token, 1);
    }

    #[test]
    fn test_apply_model_defaults() {
        let mut params = SamplingParams::neutral();
        params.apply_model_defaults(&ModelGenerationDefaults {
            do_sample: Some(true),
            temperature: Some(1.0),
            top_k: Some(32),
            top_p: Some(0.9),
            min_p: Some(0.05),
            repetition_penalty: Some(1.1),
            max_new_tokens: Some(256),
            max_length: None,
            suppress_tokens: Some(vec![258882, 258883]),
        });

        assert_eq!(params.temperature, Some(1.0));
        assert_eq!(params.top_k, Some(32));
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.min_p, Some(0.05));
        assert_eq!(params.repetition_penalty, Some(1.1));
        assert_eq!(params.max_len, Some(256));
        assert_eq!(
            params.logits_bias.as_ref().unwrap().get(&258882),
            Some(&-1.0e9)
        );
        assert_eq!(
            params.logits_bias.as_ref().unwrap().get(&258883),
            Some(&-1.0e9)
        );
    }

    #[test]
    fn test_apply_model_defaults_disables_sampling_when_requested() {
        let mut params = SamplingParams {
            temperature: Some(0.7),
            top_k: Some(40),
            top_p: Some(0.9),
            min_p: Some(0.1),
            ..SamplingParams::neutral()
        };
        params.apply_model_defaults(&ModelGenerationDefaults {
            do_sample: Some(false),
            ..Default::default()
        });

        assert_eq!(params.temperature, None);
        assert_eq!(params.top_k, Some(1));
        assert_eq!(params.top_p, None);
        assert_eq!(params.min_p, None);
    }

    #[test]
    fn sampling_params_ignore_eos_is_backward_compatible() {
        let mut serialized = serde_json::to_value(SamplingParams::neutral()).unwrap();
        serialized.as_object_mut().unwrap().remove("ignore_eos");

        let params: SamplingParams = serde_json::from_value(serialized).unwrap();
        assert!(!params.ignore_eos);

        let mut params = SamplingParams::neutral();
        params.ignore_eos = true;
        let params: SamplingParams =
            serde_json::from_value(serde_json::to_value(params).unwrap()).unwrap();
        assert!(params.ignore_eos);
    }

    #[test]
    fn test_do_sample_false_overrides_sampling_defaults() {
        let mut params = SamplingParams::neutral();
        params.apply_model_defaults(&ModelGenerationDefaults {
            do_sample: Some(false),
            temperature: Some(0.6),
            top_k: Some(20),
            top_p: Some(0.9),
            min_p: Some(0.05),
            ..Default::default()
        });

        assert_eq!(params.temperature, None);
        assert_eq!(params.top_k, Some(1));
        assert_eq!(params.top_p, None);
        assert_eq!(params.min_p, None);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_plan_covers_greedy_top_k_and_categorical() {
        use super::{CudaBatchSamplingKind, Sampler};

        let greedy = Sampler::new(
            None,
            0,
            None,
            Some(0.0),
            Some(0.0),
            Some(1.0),
            None,
            -1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let greedy_plan = greedy.cuda_batch_sampling_plan(false).unwrap();
        assert!(matches!(greedy_plan.kind, CudaBatchSamplingKind::Greedy));
        assert_eq!(greedy_plan.inverse_temperature, 1.0);
        assert!(greedy.cuda_resident_sampling_plan(false).is_some());
        assert!(greedy.cuda_speculative_sampling_plan(false).is_none());

        let mut penalized = greedy.clone();
        penalized.repetition_penalty = Some(1.1);
        assert!(penalized.cuda_batch_sampling_plan(false).is_none());
        let mut biased = greedy.clone();
        biased.logits_bias.insert(1, 1.0);
        assert!(biased.cuda_batch_sampling_plan(false).is_none());
        let mut processed = greedy.clone();
        processed.logits_processors.push(std::sync::Arc::new(
            |logits: &candle_core::Tensor, _context: &[u32]| Ok(logits.clone()),
        ));
        assert!(processed.cuda_batch_sampling_plan(false).is_none());

        let top_k = Sampler::new(
            Some(0.5),
            0,
            None,
            None,
            None,
            None,
            None,
            64,
            0.9,
            0.05,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let top_k_plan = top_k.cuda_batch_sampling_plan(false).unwrap();
        assert!(matches!(
            top_k_plan.kind,
            CudaBatchSamplingKind::TopK { k: 64 }
        ));
        assert_eq!(top_k_plan.inverse_temperature, 2.0);
        assert!(top_k.cuda_resident_sampling_plan(false).is_some());
        assert!(top_k.cuda_batch_sampling_plan(true).is_none());
        let speculative_top_k = top_k.cuda_speculative_sampling_plan(false).unwrap();
        assert_eq!(speculative_top_k.inverse_temperature, 2.0);
        assert_eq!(speculative_top_k.top_k, 64);
        assert_eq!(speculative_top_k.top_p, 0.9);
        assert_eq!(speculative_top_k.min_p, 0.05);

        let top_one = Sampler::new(
            Some(1.0),
            0,
            None,
            None,
            None,
            None,
            None,
            1,
            0.9,
            0.05,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let top_one_plan = top_one.cuda_batch_sampling_plan(false).unwrap();
        assert!(top_one_plan.kind.is_argmax());

        let unbounded = Sampler::new(
            Some(1.0),
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let unbounded_plan = unbounded.cuda_batch_sampling_plan(false).unwrap();
        assert!(matches!(
            unbounded_plan.kind,
            CudaBatchSamplingKind::Categorical
        ));
        assert!(unbounded.cuda_resident_sampling_plan(false).is_none());
        let speculative_unbounded = unbounded.cuda_speculative_sampling_plan(false).unwrap();
        assert_eq!(speculative_unbounded.top_k, 0);
        assert_eq!(speculative_unbounded.top_p, 1.0);
        assert_eq!(speculative_unbounded.min_p, 0.0);

        let mut filtered = unbounded;
        filtered.top_p = 0.9;
        assert!(filtered.cuda_batch_sampling_plan(false).is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_packed_row_uses_full_distribution_logprob() {
        use super::Sampler;
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;

        let sampler = Sampler::new(
            Some(2.0),
            0,
            None,
            None,
            None,
            None,
            None,
            1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let plan = sampler.cuda_batch_sampling_plan(false).unwrap();
        let packed = [4.0, 2.0, 1.0, 7.0, 8.0, 9.0, 2.0, 2.0];
        let mut rng = Isaac64Rng::seed_from_u64(42);

        let sampled = sampler
            .sample_cuda_topk_packed_row(&packed, 3, plan, &mut rng)
            .unwrap();

        assert_eq!(sampled.token, 7);
        assert!((sampled.logprob - 0.5f32.ln()).abs() < 1e-6);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_ranked_topk_matches_packed_reference_for_fixed_seeds() {
        use super::Sampler;
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;

        for k in [20, 32] {
            let sampler = Sampler::new(
                Some(0.8),
                0,
                None,
                None,
                None,
                None,
                None,
                k as i64,
                0.37,
                0.05,
                HashMap::new(),
                vec![],
            )
            .unwrap();
            let plan = sampler.cuda_batch_sampling_plan(false).unwrap();
            let values = (0..k)
                .map(|slot| 5.0 - (slot / 4) as f32 * 0.25)
                .collect::<Vec<_>>();
            let indices = (0..k).map(|slot| (1000 + slot) as f32).collect::<Vec<_>>();
            let scaled_max = values[0] * plan.inverse_temperature;
            let probabilities = values
                .iter()
                .map(|value| (*value * plan.inverse_temperature - scaled_max).exp())
                .collect::<Vec<_>>();
            let denominator = probabilities.iter().sum::<f32>();
            let cutoff = 0.37 * denominator;
            let mut cumulative = 0.0f32;
            let mut support = 0;
            for probability in &probabilities {
                if cumulative >= cutoff {
                    break;
                }
                cumulative += probability;
                support += 1;
            }

            let mut ranked = Vec::with_capacity(2 * k);
            ranked.extend_from_slice(&values);
            ranked.extend_from_slice(&indices);
            let mut full = ranked.clone();
            full.extend_from_slice(&[denominator, scaled_max]);

            for seed in 0..256 {
                let mut reference_rng = Isaac64Rng::seed_from_u64(seed);
                let reference = sampler
                    .sample_cuda_topk_packed_row(&full, k, plan, &mut reference_rng)
                    .unwrap();
                let mut ranked_rng = Isaac64Rng::seed_from_u64(seed);
                let actual = sampler
                    .sample_cuda_ranked_topk_packed_row(&ranked, k, plan, &mut ranked_rng)
                    .unwrap();

                assert_eq!(actual.token, reference.token);
                assert_eq!(actual.logprob, reference.logprob);
                assert!((actual.token as usize) < 1000 + support);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resident_unit_uniform_matches_weighted_index_for_fixed_seeds() {
        use super::weighted_index_from_unit_f32;
        use rand::{
            distr::{weighted::WeightedIndex, Distribution, Uniform},
            RngCore, SeedableRng,
        };
        use rand_isaac::Isaac64Rng;

        let top_p_truncated = |mut weights: Vec<f32>, top_p: f32| {
            let cutoff = top_p * weights.iter().sum::<f32>();
            let mut cumulative = 0.0f32;
            for weight in &mut weights {
                if cumulative >= cutoff {
                    *weight = 0.0;
                } else {
                    cumulative += *weight;
                }
            }
            weights
        };
        let distributions = [
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.0, 0.125, 0.0, 8.0, 1.0],
            vec![1.0e-7, 3.0e-5, 0.25, 9.0],
            top_p_truncated(vec![0.42, 0.31, 0.17, 0.07, 0.03], 0.72),
            top_p_truncated(vec![1.0; 20], 0.95),
        ];
        let unit = Uniform::new(0.0f32, 1.0f32).unwrap();

        for weights in distributions {
            let reference = WeightedIndex::new(&weights).unwrap();
            for seed in 0..4096 {
                let mut reference_rng = Isaac64Rng::seed_from_u64(seed);
                let expected = reference.sample(&mut reference_rng);
                let expected_next = reference_rng.next_u64();

                let mut resident_rng = Isaac64Rng::seed_from_u64(seed);
                let uniform = unit.sample(&mut resident_rng);
                let actual = weighted_index_from_unit_f32(&weights, uniform).unwrap();
                let actual_next = resident_rng.next_u64();

                assert_eq!(actual, expected, "seed={seed}, weights={weights:?}");
                assert_eq!(actual_next, expected_next, "seed={seed}");
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_categorical_row_parses_token_and_logprob() {
        use super::Sampler;

        let sampler = Sampler::new(
            Some(1.0),
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let sampled = sampler.sample_cuda_categorical_row(&[17.0, -2.5]).unwrap();

        assert_eq!(sampled.token, 17);
        assert_eq!(sampled.logprob, -2.5);
        assert!(sampler
            .sample_cuda_categorical_row(&[f32::NAN, f32::NAN])
            .is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_top1_parser_rejects_invalid_output() {
        use super::Sampler;

        assert_eq!(Sampler::cuda_top1_token([3.0, 9.0]).unwrap(), 9);
        assert!(Sampler::cuda_top1_token([f32::NAN, f32::NAN]).is_err());
        assert!(Sampler::cuda_top1_token([3.0, -1.0]).is_err());
        assert!(Sampler::cuda_top1_token([3.0, 1.5]).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_top1_row_uses_zero_logprob_and_validates_shape() {
        use super::Sampler;

        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let sampled = sampler.sample_cuda_top1_row(&[4.5, 17.0]).unwrap();

        assert_eq!(sampled.token, 17);
        assert_eq!(sampled.logprob, 0.0);
        assert!(sampler.sample_cuda_top1_row(&[4.5]).is_err());
        assert!(sampler.sample_cuda_top1_row(&[f32::NAN, f32::NAN]).is_err());
    }
}
