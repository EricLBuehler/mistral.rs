use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use candle_core::{Result, Tensor};
#[cfg(any(feature = "cuda", feature = "metal", test))]
use llm_watermarking::tensor::IndexedCandidates;
use llm_watermarking::{
    exponential, inverse_transform, kgw, mpac, sampling, semstamp, synthid, unigram,
};
use serde::{Deserialize, Serialize};

use super::{SynthIdTextWatermarkConfig, HASH_DOMAIN};

const DEFAULT_CONTEXT_WIDTH: usize = 1;
const DEFAULT_GREEN_FRACTION: f64 = 0.5;
const DEFAULT_DELTA: f64 = 2.0;
const DEFAULT_SEQUENCE_LEN: usize = 1024;
const DEFAULT_RADIX: usize = 2;
const DEFAULT_HYPERPLANES: usize = 8;
const DEFAULT_SEMANTIC_FRACTION: f64 = 0.25;
const DEFAULT_MARGIN: f64 = 0.02;
const DEFAULT_ATTEMPTS: usize = 100;
const SEMSTAMP_GENERATION_ERROR: &str = "SemStamp requires sentence embeddings and sentence rejection sampling; it cannot be attached to token generation. Use the embedding watermark API.";

fn context_width() -> usize {
    DEFAULT_CONTEXT_WIDTH
}
fn green_fraction() -> f64 {
    DEFAULT_GREEN_FRACTION
}
fn delta() -> f64 {
    DEFAULT_DELTA
}
fn sequence_len() -> usize {
    DEFAULT_SEQUENCE_LEN
}
fn radix() -> usize {
    DEFAULT_RADIX
}
fn hyperplanes() -> usize {
    DEFAULT_HYPERPLANES
}
fn semantic_fraction() -> f64 {
    DEFAULT_SEMANTIC_FRACTION
}
fn margin() -> f64 {
    DEFAULT_MARGIN
}
fn attempts() -> usize {
    DEFAULT_ATTEMPTS
}
fn enabled() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum SynthIdGenerationPolicy {
    #[default]
    ProbabilityUpdates,
    Tournament,
}

/// Selects a library watermark; vocabulary sizes refer to model output rows, including padding.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "scheme", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum WatermarkConfig {
    Synthid {
        key: String,
        #[serde(default = "super::default_ngram_len")]
        ngram_len: usize,
        #[serde(default = "super::default_depth")]
        depth: usize,
        #[serde(default)]
        generation_policy: SynthIdGenerationPolicy,
    },
    Kgw {
        key: String,
        vocab_size: usize,
        #[serde(default = "context_width")]
        context_width: usize,
        #[serde(default = "green_fraction")]
        green_fraction: f64,
        #[serde(default = "delta")]
        delta: f64,
        #[serde(default = "enabled")]
        ignore_repeated_ngrams: bool,
    },
    Unigram {
        key: String,
        vocab_size: usize,
        #[serde(default = "green_fraction")]
        green_fraction: f64,
        #[serde(default = "delta")]
        delta: f64,
        #[serde(default = "enabled")]
        ignore_repeated_tokens: bool,
    },
    Exponential {
        key: String,
        vocab_size: usize,
        #[serde(default = "sequence_len")]
        sequence_len: usize,
        #[serde(default)]
        start_position: usize,
    },
    InverseTransform {
        key: String,
        vocab_size: usize,
        #[serde(default = "sequence_len")]
        sequence_len: usize,
        #[serde(default)]
        start_position: usize,
    },
    Mpac {
        key: String,
        vocab_size: usize,
        payload: Vec<u8>,
        #[serde(default = "radix")]
        radix: usize,
        #[serde(default = "context_width")]
        context_width: usize,
        #[serde(default = "delta")]
        delta: f64,
        #[serde(default = "enabled")]
        ignore_repeated_ngrams: bool,
    },
    Semstamp {
        key: String,
        embedding_dim: usize,
        #[serde(default = "hyperplanes")]
        num_hyperplanes: usize,
        #[serde(default = "semantic_fraction")]
        green_fraction: f64,
        #[serde(default = "margin")]
        margin: f64,
        #[serde(default = "attempts")]
        max_attempts: usize,
        #[serde(default = "enabled")]
        ignore_repeated_transitions: bool,
    },
}

impl fmt::Debug for WatermarkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WatermarkConfig")
            .field("scheme", &self.scheme())
            .field("key", &"[redacted]")
            .finish_non_exhaustive()
    }
}

impl From<SynthIdTextWatermarkConfig> for WatermarkConfig {
    fn from(config: SynthIdTextWatermarkConfig) -> Self {
        Self::Synthid {
            key: config.key,
            ngram_len: config.ngram_len,
            depth: config.depth,
            generation_policy: SynthIdGenerationPolicy::default(),
        }
    }
}

impl WatermarkConfig {
    pub fn uses_tournament(&self) -> bool {
        matches!(
            self,
            Self::Synthid {
                generation_policy: SynthIdGenerationPolicy::Tournament,
                ..
            }
        )
    }

    pub fn scheme(&self) -> &'static str {
        match self {
            Self::Synthid { .. } => "synthid",
            Self::Kgw { .. } => "kgw",
            Self::Unigram { .. } => "unigram",
            Self::Exponential { .. } => "exponential",
            Self::InverseTransform { .. } => "inverse_transform",
            Self::Mpac { .. } => "mpac",
            Self::Semstamp { .. } => "semstamp",
        }
    }

    pub fn vocab_size(&self) -> Option<usize> {
        match self {
            Self::Kgw { vocab_size, .. }
            | Self::Unigram { vocab_size, .. }
            | Self::Exponential { vocab_size, .. }
            | Self::InverseTransform { vocab_size, .. }
            | Self::Mpac { vocab_size, .. } => Some(*vocab_size),
            _ => None,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        self.algorithm_config().map(|_| ())
    }

    pub fn validate_generation(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !matches!(self, Self::Semstamp { .. }),
            SEMSTAMP_GENERATION_ERROR
        );
        self.validate()?;
        if self.uses_tournament() {
            Watermark::new(self)?.production_sampler()?;
        }
        Ok(())
    }

    pub fn deserialize_option<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Option<Self>, D::Error> {
        let value = Option::<serde_json::Value>::deserialize(deserializer)?;
        value
            .map(|mut value| {
                if let Some(object) = value.as_object_mut() {
                    object.entry("scheme").or_insert_with(|| "synthid".into());
                }
                serde_json::from_value(value).map_err(serde::de::Error::custom)
            })
            .transpose()
    }

    fn algorithm_config(&self) -> anyhow::Result<AlgorithmConfig> {
        let result = match self {
            Self::Synthid {
                key,
                ngram_len,
                depth,
                ..
            } => {
                let config = synthid::SynthIdConfig {
                    key: decode_key(key)?,
                    ngram_len: *ngram_len,
                    depth: *depth,
                };
                config.validate()?;
                AlgorithmConfig::Synthid(config)
            }
            Self::Kgw {
                key,
                vocab_size,
                context_width,
                green_fraction,
                delta,
                ignore_repeated_ngrams,
            } => {
                let config = kgw::KgwConfig {
                    key: decode_key(key)?,
                    vocab_size: *vocab_size,
                    context_width: *context_width,
                    green_fraction: *green_fraction,
                    delta: *delta,
                    ignore_repeated_ngrams: *ignore_repeated_ngrams,
                };
                config.validate()?;
                AlgorithmConfig::Kgw(config)
            }
            Self::Unigram {
                key,
                vocab_size,
                green_fraction,
                delta,
                ignore_repeated_tokens,
            } => {
                let config = unigram::UnigramConfig {
                    key: decode_key(key)?,
                    vocab_size: *vocab_size,
                    green_fraction: *green_fraction,
                    delta: *delta,
                    ignore_repeated_tokens: *ignore_repeated_tokens,
                };
                config.validate()?;
                AlgorithmConfig::Unigram(config)
            }
            Self::Exponential {
                key,
                vocab_size,
                sequence_len,
                start_position,
            }
            | Self::InverseTransform {
                key,
                vocab_size,
                sequence_len,
                start_position,
            } => {
                let config = sampling::SamplingConfig {
                    key: decode_key(key)?,
                    vocab_size: *vocab_size,
                    sequence_len: *sequence_len,
                };
                config.validate()?;
                if matches!(self, Self::Exponential { .. }) {
                    AlgorithmConfig::Exponential(config, *start_position)
                } else {
                    AlgorithmConfig::InverseTransform(config, *start_position)
                }
            }
            Self::Mpac {
                key,
                vocab_size,
                payload,
                radix,
                context_width,
                delta,
                ignore_repeated_ngrams,
            } => {
                let config = mpac::MpacConfig {
                    key: decode_key(key)?,
                    vocab_size: *vocab_size,
                    payload_len: payload.len(),
                    radix: *radix,
                    context_width: *context_width,
                    delta: *delta,
                    ignore_repeated_ngrams: *ignore_repeated_ngrams,
                };
                config.validate()?;
                anyhow::ensure!(
                    payload.iter().all(|symbol| usize::from(*symbol) < *radix),
                    "MPAC payload symbols must be smaller than radix"
                );
                AlgorithmConfig::Mpac(config, payload.clone())
            }
            Self::Semstamp {
                key,
                embedding_dim,
                num_hyperplanes,
                green_fraction,
                margin,
                max_attempts,
                ignore_repeated_transitions,
            } => {
                let config = semstamp::SemStampConfig {
                    key: decode_key(key)?,
                    embedding_dim: *embedding_dim,
                    num_hyperplanes: *num_hyperplanes,
                    green_fraction: *green_fraction,
                    margin: *margin,
                    max_attempts: *max_attempts,
                    ignore_repeated_transitions: *ignore_repeated_transitions,
                };
                config.validate()?;
                AlgorithmConfig::Semstamp(config)
            }
        };
        Ok(result)
    }
}

fn decode_key(key: &str) -> anyhow::Result<[u8; synthid::KEY_BYTES]> {
    anyhow::ensure!(
        key.len() == synthid::KEY_BYTES * 2 && key.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "watermark key must contain exactly 64 hexadecimal characters"
    );
    let mut bytes = [0; synthid::KEY_BYTES];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&key[index * 2..index * 2 + 2], 16)?;
    }
    Ok(bytes)
}

enum AlgorithmConfig {
    Synthid(synthid::SynthIdConfig),
    Kgw(kgw::KgwConfig),
    Unigram(unigram::UnigramConfig),
    Exponential(sampling::SamplingConfig, usize),
    InverseTransform(sampling::SamplingConfig, usize),
    Mpac(mpac::MpacConfig, Vec<u8>),
    Semstamp(semstamp::SemStampConfig),
}

#[derive(Clone)]
enum Algorithm {
    Synthid(synthid::SynthIdText),
    Kgw(kgw::Kgw),
    Unigram(unigram::Unigram),
    Exponential(exponential::ExponentialRace, usize, usize),
    InverseTransform(inverse_transform::InverseTransform, usize, usize),
    Mpac(mpac::Mpac, Vec<u8>),
    Semstamp(semstamp::SemStamp),
}

/// Library adapter with scheme-specific detection and device-preserving tensor transforms.
#[derive(Clone)]
pub struct Watermark {
    algorithm: Algorithm,
}

/// Probability transforms retain categorical selection; keyed samplers require argmax of their scores.
pub enum WatermarkTensor {
    Probabilities(Tensor),
    SelectionScores(Tensor),
}

impl Watermark {
    pub(crate) fn production_sampler(
        &self,
    ) -> Result<synthid::generation_tournament::ProductionTournamentSampler<'_>> {
        match &self.algorithm {
            Algorithm::Synthid(w) => w.tournament_sampler().map_err(candle_core::Error::wrap),
            _ => candle_core::bail!("generation tournament sampling requires SynthID"),
        }
    }

    pub(crate) fn tournament_demo(
        &self,
        weights: &[f32],
        context: &[u32],
        prompt_len: usize,
        options: &synthid::tournament::TournamentOptions,
    ) -> Result<synthid::tournament::TournamentDemo> {
        match &self.algorithm {
            Algorithm::Synthid(w) => w
                .tournament_demo(weights, context, prompt_len, options)
                .map_err(candle_core::Error::wrap),
            _ => candle_core::bail!("teaching_tournament requires a SynthID watermark"),
        }
    }

    pub(crate) fn apply_traced(
        &self,
        probs: &mut [f32],
        context: &[u32],
        prompt_len: usize,
        options: &llm_watermarking::trace::TraceOptions,
    ) -> Result<llm_watermarking::trace::ScalarSamplingTrace> {
        let generated = context
            .len()
            .checked_sub(prompt_len)
            .ok_or_else(|| candle_core::Error::Msg("prompt_len exceeds token count".into()))?;
        let result = match &self.algorithm {
            Algorithm::Synthid(w) => w.apply_traced(probs, context, prompt_len, options),
            Algorithm::Kgw(w) => w.apply_traced(probs, context, prompt_len, options),
            Algorithm::Unigram(w) => w.apply_traced(probs, options),
            Algorithm::Mpac(w, payload) => {
                w.apply_traced(probs, context, prompt_len, payload, options)
            }
            Algorithm::Exponential(w, start, period) => w
                .sample_traced(probs, position(*start, generated, *period), options)
                .map(|(token, trace)| {
                    probs.fill(0.0);
                    probs[token as usize] = 1.0;
                    trace
                }),
            Algorithm::InverseTransform(w, start, period) => w
                .sample_traced(probs, position(*start, generated, *period), options)
                .map(|(token, trace)| {
                    probs.fill(0.0);
                    probs[token as usize] = 1.0;
                    trace
                }),
            Algorithm::Semstamp(_) => candle_core::bail!("{SEMSTAMP_GENERATION_ERROR}"),
        };
        result.map_err(candle_core::Error::wrap)
    }

    pub fn new(config: &WatermarkConfig) -> anyhow::Result<Self> {
        let algorithm = match config.algorithm_config()? {
            AlgorithmConfig::Synthid(c) => {
                Algorithm::Synthid(synthid::SynthIdText::with_domain(&c, HASH_DOMAIN)?)
            }
            AlgorithmConfig::Kgw(c) => Algorithm::Kgw(kgw::Kgw::new(&c)?),
            AlgorithmConfig::Unigram(c) => Algorithm::Unigram(unigram::Unigram::new(&c)?),
            AlgorithmConfig::Exponential(c, start) => Algorithm::Exponential(
                exponential::ExponentialRace::new(&c)?,
                start,
                c.sequence_len,
            ),
            AlgorithmConfig::InverseTransform(c, start) => Algorithm::InverseTransform(
                inverse_transform::InverseTransform::new(&c)?,
                start,
                c.sequence_len,
            ),
            AlgorithmConfig::Mpac(c, payload) => Algorithm::Mpac(mpac::Mpac::new(&c)?, payload),
            AlgorithmConfig::Semstamp(c) => Algorithm::Semstamp(semstamp::SemStamp::new(&c)?),
        };
        Ok(Self { algorithm })
    }

    /// Mutates host probabilities; keyed selection becomes a point mass for speculative acceptance.
    pub(crate) fn apply(
        &self,
        probs: &mut [f32],
        context: &[u32],
        prompt_len: usize,
    ) -> Result<()> {
        let generated = context
            .len()
            .checked_sub(prompt_len)
            .ok_or_else(|| candle_core::Error::Msg("prompt_len exceeds token count".into()))?;
        let token = match &self.algorithm {
            Algorithm::Synthid(w) => {
                w.apply(probs, context, prompt_len)
                    .map_err(candle_core::Error::wrap)?;
                None
            }
            Algorithm::Kgw(w) => {
                w.apply(probs, context, prompt_len)
                    .map_err(candle_core::Error::wrap)?;
                None
            }
            Algorithm::Unigram(w) => {
                w.apply(probs).map_err(candle_core::Error::wrap)?;
                None
            }
            Algorithm::Mpac(w, payload) => {
                w.apply(probs, context, prompt_len, payload)
                    .map_err(candle_core::Error::wrap)?;
                None
            }
            Algorithm::Exponential(w, start, period) => Some(
                w.sample(probs, position(*start, generated, *period))
                    .map_err(candle_core::Error::wrap)?,
            ),
            Algorithm::InverseTransform(w, start, period) => Some(
                w.sample(probs, position(*start, generated, *period))
                    .map_err(candle_core::Error::wrap)?,
            ),
            Algorithm::Semstamp(_) => candle_core::bail!("{SEMSTAMP_GENERATION_ERROR}"),
        };
        if let Some(token) = token {
            probs.fill(0.0);
            probs[token as usize] = 1.0;
        }
        Ok(())
    }

    /// The caller guarantees finite nonnegative weights with positive mass; no probability readback occurs.
    pub fn apply_tensor(
        &self,
        probs: &Tensor,
        context: &[u32],
        prompt_len: usize,
    ) -> Result<WatermarkTensor> {
        let generated = context
            .len()
            .checked_sub(prompt_len)
            .ok_or_else(|| candle_core::Error::Msg("prompt_len exceeds token count".into()))?;
        let device = probs.device();
        let prepared = match &self.algorithm {
            Algorithm::Synthid(w) => {
                w.prepare_tensor(probs.dims1()?, context, prompt_len, device)?
            }
            Algorithm::Kgw(w) => w.prepare_tensor(context, prompt_len, device)?,
            Algorithm::Unigram(w) => w.prepare_tensor(device)?,
            Algorithm::Mpac(w, payload) => {
                w.prepare_tensor(context, prompt_len, payload, device)?
            }
            Algorithm::Exponential(w, start, period) => {
                return Ok(WatermarkTensor::SelectionScores(
                    w.prepare_tensor(position(*start, generated, *period), device)?
                        .apply_trusted(probs)?,
                ))
            }
            Algorithm::InverseTransform(w, start, period) => {
                return Ok(WatermarkTensor::SelectionScores(
                    w.prepare_tensor(position(*start, generated, *period), device)?
                        .apply_trusted(probs)?,
                ))
            }
            Algorithm::Semstamp(_) => candle_core::bail!("{SEMSTAMP_GENERATION_ERROR}"),
        };
        Ok(WatermarkTensor::Probabilities(
            prepared.apply_trusted(probs)?,
        ))
    }

    #[cfg(any(feature = "cuda", feature = "metal", test))]
    pub(crate) fn apply_indexed(
        &self,
        probs: &Tensor,
        candidates: &IndexedCandidates,
        context: &[u32],
        prompt_len: usize,
    ) -> Result<WatermarkTensor> {
        let generated = context
            .len()
            .checked_sub(prompt_len)
            .ok_or_else(|| candle_core::Error::Msg("prompt_len exceeds token count".into()))?;
        let prepared = match &self.algorithm {
            Algorithm::Synthid(w) => w.prepare_indexed(candidates, context, prompt_len)?,
            Algorithm::Kgw(w) => w.prepare_indexed(candidates, context, prompt_len)?,
            Algorithm::Unigram(w) => w.prepare_indexed(candidates)?,
            Algorithm::Mpac(w, payload) => {
                w.prepare_indexed(candidates, context, prompt_len, payload)?
            }
            Algorithm::Exponential(w, start, period) => {
                return Ok(WatermarkTensor::SelectionScores(
                    w.prepare_indexed(candidates, position(*start, generated, *period))?
                        .apply_trusted(probs)?,
                ))
            }
            Algorithm::InverseTransform(w, start, period) => {
                return Ok(WatermarkTensor::SelectionScores(
                    w.prepare_indexed(candidates, position(*start, generated, *period))?
                        .apply_trusted(probs)?,
                ))
            }
            Algorithm::Semstamp(_) => candle_core::bail!("{SEMSTAMP_GENERATION_ERROR}"),
        };
        Ok(WatermarkTensor::Probabilities(
            prepared.apply_trusted(probs)?,
        ))
    }

    pub fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos: &[u32],
    ) -> anyhow::Result<WatermarkEvidence> {
        Ok(match &self.algorithm {
            Algorithm::Synthid(w) => {
                let e = w.detect(tokens, prompt_len, eos)?;
                WatermarkEvidence::Synthid {
                    tokens_scored: e.tokens_scored,
                    mean_g_value: e.mean_g_value,
                }
            }
            Algorithm::Kgw(w) => {
                WatermarkEvidence::count("kgw", w.detect(tokens, prompt_len, eos)?)
            }
            Algorithm::Unigram(w) => {
                WatermarkEvidence::count("unigram", w.detect(tokens, prompt_len, eos)?)
            }
            Algorithm::Exponential(w, start, _) => WatermarkEvidence::sampling(
                "exponential",
                w.detect(tokens, prompt_len, eos, *start)?,
            ),
            Algorithm::InverseTransform(w, start, _) => WatermarkEvidence::sampling(
                "inverse_transform",
                w.detect(tokens, prompt_len, eos, *start)?,
            ),
            Algorithm::Mpac(w, _) => {
                let e = w.detect(tokens, prompt_len, eos)?;
                WatermarkEvidence::Mpac {
                    tokens_scored: e.tokens_scored,
                    payload: e.payload,
                    votes: e.votes,
                    winning_fraction: e.winning_fraction,
                }
            }
            Algorithm::Semstamp(_) => {
                anyhow::bail!("SemStamp detection requires sentence embeddings, not token IDs")
            }
        })
    }

    pub fn semstamp(&self) -> anyhow::Result<&semstamp::SemStamp> {
        match &self.algorithm {
            Algorithm::Semstamp(w) => Ok(w),
            _ => anyhow::bail!("embedding watermark operations require SemStamp"),
        }
    }

    pub fn detect_embeddings(
        &self,
        embeddings: &[Vec<f32>],
        prompt_len: usize,
    ) -> anyhow::Result<WatermarkEvidence> {
        Ok(WatermarkEvidence::semantic(
            self.semstamp()?.detect(embeddings, prompt_len)?,
        ))
    }

    pub fn detect_embeddings_tensor(
        &self,
        embeddings: &Tensor,
        prompt_len: usize,
    ) -> anyhow::Result<WatermarkEvidence> {
        Ok(WatermarkEvidence::semantic(
            self.semstamp()?.detect_tensor(embeddings, prompt_len)?,
        ))
    }
}

fn position(start: usize, generated: usize, period: usize) -> usize {
    (start % period + generated % period) % period
}

/// Uncalibrated, scheme-specific evidence; scores are not probabilities of authorship.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum WatermarkEvidence {
    Synthid {
        tokens_scored: usize,
        mean_g_value: Option<f64>,
    },
    Count {
        scheme: String,
        trials: usize,
        successes: usize,
        expected_rate: f64,
        observed_rate: Option<f64>,
        z_score: Option<f64>,
    },
    Sampling {
        scheme: String,
        tokens_scored: usize,
        mean_cost: Option<f64>,
    },
    Mpac {
        tokens_scored: usize,
        payload: Vec<Option<u8>>,
        votes: Vec<Vec<usize>>,
        winning_fraction: Option<f64>,
    },
    Semstamp {
        sentences_scored: usize,
        valid_sentences: usize,
        expected_rate: f64,
        valid_fraction: Option<f64>,
        z_score: Option<f64>,
    },
}

impl WatermarkEvidence {
    fn count(scheme: &str, e: llm_watermarking::CountDetection) -> Self {
        Self::Count {
            scheme: scheme.into(),
            trials: e.trials,
            successes: e.successes,
            expected_rate: e.expected_rate,
            observed_rate: e.observed_rate,
            z_score: e.z_score,
        }
    }
    fn sampling(scheme: &str, e: sampling::SamplingDetection) -> Self {
        Self::Sampling {
            scheme: scheme.into(),
            tokens_scored: e.tokens_scored,
            mean_cost: e.mean_cost,
        }
    }
    fn semantic(e: semstamp::SemStampDetection) -> Self {
        Self::Semstamp {
            sentences_scored: e.sentences_scored,
            valid_sentences: e.valid_sentences,
            expected_rate: e.expected_rate,
            valid_fraction: e.valid_fraction,
            z_score: e.z_score,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RequestWatermark {
    config: WatermarkConfig,
    inner: Arc<OnceLock<Watermark>>,
}

impl RequestWatermark {
    pub(crate) fn uses_tournament(&self) -> bool {
        self.config.uses_tournament()
    }

    pub(crate) fn synthid_depth(&self) -> Option<usize> {
        match &self.config {
            WatermarkConfig::Synthid { depth, .. } => Some(*depth),
            _ => None,
        }
    }

    pub(crate) fn teaching_depth(&self) -> Result<usize> {
        match &self.config {
            WatermarkConfig::Synthid { depth, .. } => Ok(*depth),
            _ => candle_core::bail!("teaching_tournament requires a SynthID watermark"),
        }
    }

    pub(crate) fn scheme(&self) -> &'static str {
        self.config.scheme()
    }

    pub(crate) fn new(config: &WatermarkConfig) -> anyhow::Result<Self> {
        config.validate_generation()?;
        let inner = OnceLock::new();
        if config.uses_tournament() {
            let _ = inner.set(Watermark::new(config)?);
        }
        Ok(Self {
            config: config.clone(),
            inner: Arc::new(inner),
        })
    }

    pub(crate) fn resolve(&self, vocab_size: usize) -> Result<&Watermark> {
        if let Some(expected) = self.config.vocab_size() {
            if expected != vocab_size {
                candle_core::bail!("watermark vocab_size={expected} does not match model logits vocabulary {vocab_size}");
            }
        }
        if self.inner.get().is_none() {
            let watermark = Watermark::new(&self.config).map_err(candle_core::Error::wrap)?;
            let _ = self.inner.set(watermark);
        }
        Ok(self.inner.get().expect("watermark initialized"))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_core::Device;

    pub(crate) fn token_configs(vocab_size: usize) -> Vec<WatermarkConfig> {
        [
            "synthid",
            "kgw",
            "unigram",
            "exponential",
            "inverse_transform",
            "mpac",
        ]
        .into_iter()
        .map(|scheme| {
            let mut value = serde_json::json!({"scheme": scheme, "key": "42".repeat(32)});
            if scheme != "synthid" {
                value["vocab_size"] = vocab_size.into();
            }
            if scheme == "mpac" {
                value["payload"] = serde_json::json!([1, 0, 1]);
            }
            serde_json::from_value(value).unwrap()
        })
        .collect()
    }

    fn tensor_parity(device: &Device) -> anyhow::Result<()> {
        let initial = [0.0f32, 0.1, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0];
        let context = [1, 2, 3, 4, 5];
        for config in token_configs(initial.len()) {
            let watermark = Watermark::new(&config)?;
            let mut expected = initial;
            watermark.apply(&mut expected, &context, 4)?;
            let input = Tensor::new(&initial, device)?;
            let actual = match watermark.apply_tensor(&input, &context, 4)? {
                WatermarkTensor::Probabilities(output) => {
                    assert!(output.device().same_device(device));
                    output.to_vec1::<f32>()?
                }
                WatermarkTensor::SelectionScores(scores) => {
                    assert!(scores.device().same_device(device));
                    let token = scores.argmax(0)?.to_scalar::<u32>()?;
                    let mut probs = vec![0.0; initial.len()];
                    probs[token as usize] = 1.0;
                    probs
                }
            };
            assert_eq!(input.to_vec1::<f32>()?, initial);
            for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 2e-4,
                    "{} token {i}: {actual} != {expected}",
                    config.scheme()
                );
                if initial[i] == 0.0 {
                    assert_eq!(*actual, 0.0);
                }
            }
            let mut replay = initial;
            watermark.clone().apply(&mut replay, &context, 4)?;
            assert_eq!(replay, expected);
            let evidence = watermark.detect(&[1, 2, 3, 4, 5, 6], 4, &[6])?;
            let value = serde_json::to_value(evidence)?;
            let count = value
                .get("tokens_scored")
                .or_else(|| value.get("trials"))
                .unwrap();
            assert_eq!(count, 1);
        }
        Ok(())
    }

    #[test]
    fn watermark_schemes_cpu_tensor_matches_scalar_and_preserves_support() -> anyhow::Result<()> {
        tensor_parity(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn watermark_schemes_metal_tensor_matches_scalar() -> anyhow::Result<()> {
        tensor_parity(&Device::new_metal(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn watermark_schemes_cuda_tensor_matches_scalar() -> anyhow::Result<()> {
        tensor_parity(&Device::new_cuda(0)?)
    }

    #[test]
    fn watermark_configs_validate_without_allocating_vocabulary_tables() {
        for config in token_configs(8) {
            config.validate_generation().unwrap();
            assert!(!format!("{config:?}").contains(&"42".repeat(32)));
            let mut value = serde_json::to_value(&config).unwrap();
            value["key"] = "invalid".into();
            assert!(serde_json::from_value::<WatermarkConfig>(value)
                .unwrap()
                .validate()
                .is_err());
            let mut value = serde_json::to_value(&config).unwrap();
            value["typo"] = true.into();
            assert!(serde_json::from_value::<WatermarkConfig>(value).is_err());
            let pending = RequestWatermark::new(&config).unwrap();
            if config.vocab_size().is_some() {
                assert!(pending.resolve(9).is_err());
                assert!(pending.inner.get().is_none());
            }
        }
        let invalid: WatermarkConfig = serde_json::from_value(serde_json::json!({"scheme":"mpac", "key":"42".repeat(32), "vocab_size":8, "payload":[2], "radix":2})).unwrap();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn watermark_sampling_offsets_wrap_and_exclude_prompt() -> anyhow::Result<()> {
        for mut config in token_configs(8) {
            match &mut config {
                WatermarkConfig::Exponential {
                    start_position,
                    sequence_len,
                    ..
                }
                | WatermarkConfig::InverseTransform {
                    start_position,
                    sequence_len,
                    ..
                } => {
                    *start_position = usize::MAX;
                    *sequence_len = 7;
                }
                _ => continue,
            }
            let watermark = Watermark::new(&config)?;
            let mut a = [0.125; 8];
            let mut b = a;
            watermark.apply(&mut a, &[1, 2, 3, 4, 5], 4)?;
            watermark.apply(&mut b, &[6], 0)?;
            assert_eq!(a, b);
            assert!(watermark.apply(&mut b, &[], 1).is_err());
        }
        Ok(())
    }

    fn semstamp_parity(device: &Device) -> anyhow::Result<()> {
        let config: WatermarkConfig = serde_json::from_value(
            serde_json::json!({"scheme":"semstamp", "key":"42".repeat(32), "embedding_dim":3, "num_hyperplanes":2, "margin":0.0}),
        )?;
        config.validate()?;
        assert!(config
            .validate_generation()
            .unwrap_err()
            .to_string()
            .contains("sentence embeddings"));
        let watermark = Watermark::new(&config)?;
        let embeddings = vec![
            vec![1.0f32, 0.3, 0.5],
            vec![-0.2, 0.9, 0.1],
            vec![0.4, -0.1, 1.0],
        ];
        let input = Tensor::new(embeddings.clone(), device)?;
        let prepared = watermark.semstamp()?.prepare_tensor(device)?;
        let signatures = prepared.signatures_trusted(&input)?;
        assert!(signatures.device().same_device(device));
        let expected = embeddings
            .iter()
            .map(|e| watermark.semstamp().unwrap().signature(e).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(signatures.to_vec1::<u32>()?, expected);
        assert_eq!(
            serde_json::to_value(watermark.detect_embeddings(&embeddings, 1)?)?,
            serde_json::to_value(watermark.detect_embeddings_tensor(&input, 1)?)?
        );
        Ok(())
    }

    #[test]
    fn watermark_semstamp_cpu_embeddings_and_generation_rejection() -> anyhow::Result<()> {
        semstamp_parity(&Device::Cpu)
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn watermark_semstamp_metal_embeddings() -> anyhow::Result<()> {
        semstamp_parity(&Device::new_metal(0)?)
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn watermark_semstamp_cuda_embeddings() -> anyhow::Result<()> {
        semstamp_parity(&Device::new_cuda(0)?)
    }
}
