use std::any::Any;
use std::sync::{Arc, Mutex};

use candle_core::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::text_models_inputs_processor::{
    FlashParams, PagedAttentionInputMetadata, PagedAttentionMeta,
};
use crate::sequence::Sequence;

pub type TargetTokenEmbedder<'a> = dyn Fn(&Tensor) -> Result<Tensor> + 'a;

pub trait SpeculativeProposePreparation: Any + Send {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any + Send> SpeculativeProposePreparation for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct SpeculativeProposePrepareCtx<'a> {
    pub seq_ids: &'a [usize],
    pub base_lens: &'a [usize],
    pub target_rows: &'a [(usize, usize)],
}

pub enum SpeculativeKvCache<'a> {
    Paged {
        metadata: &'a PagedAttentionMeta,
        kv_cache: &'a [(Tensor, Tensor)],
    },
}

pub struct SpeculativeProposeBatchCtx<'a> {
    pub proposal_len: usize,
    pub sampled_tokens: &'a [u32],
    pub sampled_tokens_emitted: bool,
    pub seq_ids: &'a [usize],
    pub base_lens: &'a [usize],
    pub sequences: &'a [&'a Sequence],
    pub cache: SpeculativeKvCache<'a>,
    pub target_hiddens: Option<Tensor>,
    /// Per active sequence: its row in the last target forward's batch and how many leading rows of
    /// that forward (the sampled anchor plus accepted drafts) the proposer must consume.
    pub target_rows: &'a [(usize, usize)],
    pub preparation: Option<&'a dyn SpeculativeProposePreparation>,
    pub rng: Arc<Mutex<Isaac64Rng>>,
}

/// Outcome of verifying one sequence's staged drafts, in target-batch order.
#[derive(Clone, Copy, Debug)]
pub struct SpeculativeCommitRow {
    pub batch_idx: usize,
    /// Rows of the verify forward that stay committed: the anchor plus accepted drafts.
    pub keep_rows: usize,
    pub accepted_all: bool,
    pub terminal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpeculativePrefillCaptureLayout {
    Dense,
    Packed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpeculativeTapSpan {
    capture_batch_idx: usize,
    capture_row_start: usize,
    rows: usize,
}

impl SpeculativeTapSpan {
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SpeculativeTapRouting {
    spans: Vec<SpeculativeTapSpan>,
    capture_rows: usize,
}

impl SpeculativeTapRouting {
    pub(crate) fn new(
        layout: SpeculativePrefillCaptureLayout,
        capture_batch: usize,
        capture_rows: usize,
        target_batch_indices: &[usize],
        chunk_ranges: &[(usize, usize)],
    ) -> Result<Self> {
        if target_batch_indices.len() != chunk_ranges.len() {
            candle_core::bail!(
                "speculative tap routing has {} batch indices for {} prompt rows",
                target_batch_indices.len(),
                chunk_ranges.len()
            );
        }
        let row_counts = chunk_ranges
            .iter()
            .map(|&(start, end)| {
                let rows = end.checked_sub(start).ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "speculative prompt range ({start}, {end}) is reversed"
                    ))
                })?;
                if rows == 0 {
                    candle_core::bail!("speculative prompt range ({start}, {end}) is empty");
                }
                Ok(rows)
            })
            .collect::<Result<Vec<_>>>()?;

        let spans = match layout {
            SpeculativePrefillCaptureLayout::Dense => target_batch_indices
                .iter()
                .zip(row_counts)
                .map(|(&batch_idx, rows)| {
                    if batch_idx >= capture_batch {
                        candle_core::bail!(
                            "speculative tap batch row {batch_idx} exceeds capture batch {capture_batch}"
                        );
                    }
                    if rows > capture_rows {
                        candle_core::bail!(
                            "speculative prompt has {rows} rows but dense tap capture has {capture_rows}"
                        );
                    }
                    Ok(SpeculativeTapSpan {
                        capture_batch_idx: batch_idx,
                        capture_row_start: 0,
                        rows,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            SpeculativePrefillCaptureLayout::Packed => {
                if capture_batch != 1 {
                    candle_core::bail!(
                        "packed speculative tap capture has batch {capture_batch}, expected 1"
                    );
                }
                for (logical_idx, &batch_idx) in target_batch_indices.iter().enumerate() {
                    if batch_idx != logical_idx {
                        candle_core::bail!(
                            "packed speculative tap row {logical_idx} maps to target batch {batch_idx}"
                        );
                    }
                }
                let mut row_start = 0usize;
                let mut spans = Vec::with_capacity(row_counts.len());
                for rows in row_counts {
                    spans.push(SpeculativeTapSpan {
                        capture_batch_idx: 0,
                        capture_row_start: row_start,
                        rows,
                    });
                    row_start = row_start.checked_add(rows).ok_or_else(|| {
                        candle_core::Error::msg("packed speculative tap row count overflow")
                    })?;
                }
                if row_start != capture_rows {
                    candle_core::bail!(
                        "packed speculative prompts have {row_start} rows but tap capture has {capture_rows}"
                    );
                }
                spans
            }
        };
        Ok(Self {
            spans,
            capture_rows,
        })
    }

    pub(crate) fn spans(&self) -> &[SpeculativeTapSpan] {
        &self.spans
    }

    pub(crate) fn flat_row_indices(&self) -> Result<Vec<u32>> {
        let total_rows = self.spans.iter().try_fold(0usize, |total, span| {
            total.checked_add(span.rows).ok_or_else(|| {
                candle_core::Error::msg("speculative tap routing row count overflow")
            })
        })?;
        let mut indices = Vec::with_capacity(total_rows);
        for span in &self.spans {
            let start = span
                .capture_batch_idx
                .checked_mul(self.capture_rows)
                .and_then(|row| row.checked_add(span.capture_row_start))
                .ok_or_else(|| candle_core::Error::msg("speculative tap row index overflow"))?;
            let end = start
                .checked_add(span.rows)
                .ok_or_else(|| candle_core::Error::msg("speculative tap row index overflow"))?;
            for row in start..end {
                indices.push(u32::try_from(row).map_err(candle_core::Error::wrap)?);
            }
        }
        Ok(indices)
    }
}

/// One prompt chunk the target just processed; proposers that keep their own KV cache use it to
/// catch up over the prompt (rows are `[start, end)` per sequence, in target-batch order).
pub struct SpeculativePrefillCtx<'a> {
    pub seq_ids: &'a [usize],
    pub batch_indices: &'a [usize],
    pub tokens: &'a [&'a [u32]],
    pub chunk_ranges: &'a [(usize, usize)],
    pub is_final_prompt_chunk: bool,
    pub cache: SpeculativeKvCache<'a>,
    /// The target's own attention inputs for this chunk; a drafter sharing the block table can run one prefill with them.
    pub target_attention: Option<TargetAttentionInputs<'a>>,
}

#[derive(Clone, Copy)]
pub struct TargetAttentionInputs<'a> {
    pub metadata: &'a PagedAttentionInputMetadata,
    pub flash_params: &'a FlashParams,
}

impl SpeculativePrefillCtx<'_> {
    pub(crate) fn capture_layout(&self) -> SpeculativePrefillCaptureLayout {
        self.target_attention
            .map_or(SpeculativePrefillCaptureLayout::Dense, |target| {
                if target.flash_params.packed {
                    SpeculativePrefillCaptureLayout::Packed
                } else {
                    SpeculativePrefillCaptureLayout::Dense
                }
            })
    }
}

#[derive(Clone, Debug)]
pub enum SpeculativeTokens {
    Host(Vec<u32>),
    Device(Tensor),
}

impl Default for SpeculativeTokens {
    fn default() -> Self {
        Self::Host(Vec::new())
    }
}

impl From<Vec<u32>> for SpeculativeTokens {
    fn from(tokens: Vec<u32>) -> Self {
        Self::Host(tokens)
    }
}

impl SpeculativeTokens {
    pub fn from_device(tokens: Tensor) -> Result<Self> {
        if tokens.rank() != 1 {
            candle_core::bail!(
                "device speculative tokens must have rank 1, got {:?}",
                tokens.dims()
            );
        }
        Ok(Self::Device(
            tokens.to_dtype(candle_core::DType::U32)?.contiguous()?,
        ))
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Host(tokens) => tokens.len(),
            Self::Device(tokens) => tokens
                .dim(0)
                .expect("device speculative token shape was validated"),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_host(&self) -> Option<&[u32]> {
        match self {
            Self::Host(tokens) => Some(tokens),
            Self::Device(_) => None,
        }
    }

    pub fn as_device(&self) -> Option<&Tensor> {
        match self {
            Self::Host(_) => None,
            Self::Device(tokens) => Some(tokens),
        }
    }

    pub fn materialize(&mut self) -> Result<&[u32]> {
        if let Self::Device(tokens) = self {
            let tokens = tokens.to_vec1::<u32>()?;
            *self = Self::Host(tokens);
        }
        Ok(self
            .as_host()
            .expect("materialized speculative tokens are host-backed"))
    }

    pub fn into_vec(self) -> Result<Vec<u32>> {
        match self {
            Self::Host(tokens) => Ok(tokens),
            Self::Device(tokens) => tokens.to_vec1::<u32>(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeculativeProposal {
    pub tokens: SpeculativeTokens,
    pub distribution: Option<SpeculativeProposalDistribution>,
}

#[derive(Clone, Debug)]
pub enum SpeculativeProposalDistribution {
    Logits(Tensor),
    SparseProbs(SparseSpeculativeProbs),
}

#[derive(Clone, Debug)]
pub struct SparseSpeculativeProbs {
    token_ids: Tensor,
    probs: Tensor,
}

impl SparseSpeculativeProbs {
    pub fn new(token_ids: Tensor, probs: Tensor) -> Result<Self> {
        let [positions, candidates] = token_ids.dims() else {
            candle_core::bail!(
                "sparse speculative token ids must have shape [positions, candidates], got {:?}",
                token_ids.dims()
            );
        };
        if probs.dims() != [*positions, *candidates] {
            candle_core::bail!(
                "sparse speculative probability shape {:?} does not match token ids {:?}",
                probs.dims(),
                token_ids.dims()
            );
        }
        if *candidates == 0 {
            candle_core::bail!("sparse speculative probabilities must contain candidates");
        }
        Ok(Self {
            token_ids: token_ids.to_dtype(candle_core::DType::U32)?,
            probs: probs.to_dtype(candle_core::DType::F32)?,
        })
    }

    pub fn positions(&self) -> usize {
        self.token_ids
            .dim(0)
            .expect("validated sparse probability rank")
    }

    pub fn token_ids(&self) -> &Tensor {
        &self.token_ids
    }

    pub fn probs(&self) -> &Tensor {
        &self.probs
    }
}

impl SpeculativeProposal {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self {
            tokens: tokens.into(),
            distribution: None,
        }
    }

    pub fn from_device(tokens: Tensor) -> Result<Self> {
        Ok(Self {
            tokens: SpeculativeTokens::from_device(tokens)?,
            distribution: None,
        })
    }

    pub fn with_logits(tokens: Vec<u32>, logits: Tensor) -> Self {
        Self {
            tokens: tokens.into(),
            distribution: Some(SpeculativeProposalDistribution::Logits(logits)),
        }
    }

    pub fn with_sparse_probs(tokens: Vec<u32>, token_ids: Tensor, probs: Tensor) -> Result<Self> {
        let sparse = SparseSpeculativeProbs::new(token_ids, probs)?;
        if sparse.positions() != tokens.len() {
            candle_core::bail!(
                "sparse speculative probabilities have {} positions for {} tokens",
                sparse.positions(),
                tokens.len()
            );
        }
        Ok(Self {
            tokens: tokens.into(),
            distribution: Some(SpeculativeProposalDistribution::SparseProbs(sparse)),
        })
    }

    pub fn with_device_sparse_probs(
        tokens: Tensor,
        token_ids: Tensor,
        probs: Tensor,
    ) -> Result<Self> {
        let tokens = SpeculativeTokens::from_device(tokens)?;
        let sparse = SparseSpeculativeProbs::new(token_ids, probs)?;
        if sparse.positions() != tokens.len() {
            candle_core::bail!(
                "sparse speculative probabilities have {} positions for {} tokens",
                sparse.positions(),
                tokens.len()
            );
        }
        Ok(Self {
            tokens,
            distribution: Some(SpeculativeProposalDistribution::SparseProbs(sparse)),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub struct SpeculativeProposalBatch {
    pub proposals: Vec<SpeculativeProposal>,
}

impl SpeculativeProposalBatch {
    pub fn new(proposals: Vec<SpeculativeProposal>) -> Self {
        Self { proposals }
    }
}

pub trait SpeculativeProposer {
    fn proposal_len(&self) -> usize;

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch>;
}

/// Sample one draft token per row of `logits` (`[rows, vocab]`) with each row's own sampler.
pub fn sample_draft_rows(
    logits: &Tensor,
    sequences: &[&Sequence],
    contexts: &mut [Vec<u32>],
    rng: &Arc<Mutex<Isaac64Rng>>,
) -> Result<Vec<u32>> {
    let batch = sequences.len();
    if contexts.len() != batch || logits.dim(0)? != batch {
        candle_core::bail!(
            "draft sampling batch mismatch: logits={}, contexts={}, sequences={batch}",
            logits.dim(0)?,
            contexts.len()
        );
    }
    #[cfg(feature = "cuda")]
    if batch > 1 && logits.device().is_cuda() {
        let samplers = sequences
            .iter()
            .map(|seq| seq.sampler())
            .collect::<Vec<_>>();
        if samplers.iter().all(|sampler| {
            sampler.cuda_batch_sampling_plan(false).is_some_and(|plan| {
                matches!(plan.kind, crate::sampler::CudaBatchSamplingKind::Greedy)
            })
        }) {
            let packed = samplers[0].sample_cuda_top1_batch(&logits.contiguous()?)?;
            let tokens = packed
                .iter()
                .zip(&samplers)
                .map(|(row, sampler)| Ok(sampler.sample_cuda_top1_row(row)?.token))
                .collect::<Result<Vec<_>>>()?;
            for (context, token) in contexts.iter_mut().zip(&tokens) {
                context.push(*token);
            }
            return Ok(tokens);
        }
    }
    let mut tokens = Vec::with_capacity(batch);
    for (row, seq) in sequences.iter().enumerate() {
        let row_logits = logits.get(row)?.to_dtype(candle_core::DType::F32)?;
        let sequence_rng = seq.sampling_rng(rng);
        let sampled = seq.sampler().sample(
            row_logits,
            &contexts[row],
            seq.prompt_tokens(),
            false,
            sequence_rng,
            false,
            batch > 1,
        )?;
        contexts[row].push(sampled.token);
        tokens.push(sampled.token);
    }
    Ok(tokens)
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_draft_sampling_tests {
    use std::{collections::HashMap, sync::Arc};

    use candle_core::{DType, Device, Result, Tensor};
    use rand::{RngCore, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use crate::{
        sampler::{CustomLogitsProcessor, Sampler, SamplingParams},
        sequence::{SeqStepType, Sequence, SequenceGroup, SequenceRecognizer},
    };

    use super::sample_draft_rows;

    const VOCAB: usize = 4097;
    const RNG_SEED: u64 = 42;

    fn sequence(
        params: SamplingParams,
        processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    ) -> Sequence {
        let sampler = Sampler::new(
            params.temperature,
            params.top_n_logprobs,
            None,
            params.frequency_penalty,
            params.presence_penalty,
            params.repetition_penalty,
            params.dry_params,
            params.top_k.map_or(-1, |k| i64::try_from(k).unwrap()),
            params.top_p.unwrap_or(1.0),
            params.min_p.unwrap_or(0.0),
            params.logits_bias.unwrap_or_default(),
            processors,
        )
        .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        Sequence::new_waiting(
            vec![0, 0],
            "prompt".to_string(),
            0,
            0,
            0,
            tx,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
                1, false, true, None,
            ))),
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            false,
            vec![],
            None,
        )
    }

    #[test]
    fn draft_rows_cuda_greedy_matches_cpu_with_masks_padding_and_context_updates() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(RNG_SEED)));
        for batch in [1, 6, 8] {
            let sequences = (0..batch)
                .map(|_| sequence(SamplingParams::deterministic(), vec![]))
                .collect::<Vec<_>>();
            let sequences = sequences.iter().collect::<Vec<_>>();
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                let mut cpu_contexts = vec![vec![0, 0]; batch];
                let mut cuda_contexts = cpu_contexts.clone();
                for phase in [0, 1] {
                    let stride = VOCAB + 2;
                    let mut values = vec![1000.0f32; (batch + 2) * stride];
                    let mut expected = Vec::with_capacity(batch);
                    for row in 0..batch {
                        let begin = (row + 1) * stride + 1;
                        values[begin..begin + VOCAB].fill(f32::NEG_INFINITY);
                        let winner = 17 + row * 37 + phase;
                        values[begin + winner] = 8.0;
                        values[begin + VOCAB - 1] = 8.0;
                        expected.push(u32::try_from(winner).unwrap());
                    }
                    let backing = Tensor::from_vec(values, (batch + 2, stride), &Device::Cpu)?;
                    let cpu = backing.narrow(0, 1, batch)?.narrow(1, 1, VOCAB)?;
                    let cuda = backing
                        .to_device(&device)?
                        .to_dtype(dtype)?
                        .narrow(0, 1, batch)?
                        .narrow(1, 1, VOCAB)?;
                    if batch > 1 {
                        assert!(!cuda.is_contiguous());
                    }
                    let reference = sample_draft_rows(&cpu, &sequences, &mut cpu_contexts, &rng)?;
                    let actual = sample_draft_rows(&cuda, &sequences, &mut cuda_contexts, &rng)?;
                    assert_eq!(reference, expected);
                    assert_eq!(actual, expected);
                    assert_eq!(cuda_contexts, cpu_contexts);
                    assert!(cuda_contexts
                        .iter()
                        .all(|context| context.len() == phase + 3));
                }
            }
        }
        assert_eq!(
            rng.lock().unwrap().next_u64(),
            Isaac64Rng::seed_from_u64(RNG_SEED).next_u64()
        );
        Ok(())
    }

    #[test]
    fn draft_rows_cuda_preserves_mixed_sampler_penalties_bias_and_processors() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(RNG_SEED)));
        let logits = Tensor::new(&[[5.0f32, 4.0], [5.0, 4.0]], &Device::Cpu)?;
        let cuda = logits.to_device(&device)?;
        let mut variants = Vec::new();
        for field in ["frequency", "presence", "repetition", "bias", "processor"] {
            let mut params = SamplingParams::deterministic();
            let mut processors: Vec<Arc<dyn CustomLogitsProcessor>> = Vec::new();
            match field {
                "frequency" => params.frequency_penalty = Some(2.0),
                "presence" => params.presence_penalty = Some(2.0),
                "repetition" => params.repetition_penalty = Some(2.0),
                "bias" => params.logits_bias = Some(HashMap::from([(1, 2.0)])),
                "processor" => processors.push(Arc::new(|logits: &Tensor, _context: &[u32]| {
                    Tensor::new(&[0.0f32, 9.0], logits.device())
                })),
                _ => unreachable!(),
            }
            variants.push(sequence(params, processors));
        }
        let greedy = sequence(SamplingParams::deterministic(), vec![]);
        for modified in &variants {
            let sequences = [&greedy, modified];
            let mut cpu_contexts = vec![vec![0, 0, 0]; 2];
            let mut cuda_contexts = cpu_contexts.clone();
            let reference = sample_draft_rows(&logits, &sequences, &mut cpu_contexts, &rng)?;
            let actual = sample_draft_rows(&cuda, &sequences, &mut cuda_contexts, &rng)?;
            assert_eq!(reference, vec![0, 1]);
            assert_eq!(actual, reference);
            assert_eq!(cuda_contexts, cpu_contexts);
        }
        Ok(())
    }

    #[test]
    fn draft_rows_cuda_preserves_stochastic_rng_consumption() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let cpu_rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(RNG_SEED)));
        let cuda_rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(RNG_SEED)));
        let mut stochastic = SamplingParams::deterministic();
        stochastic.temperature = Some(1.0);
        let sequences = [
            sequence(SamplingParams::deterministic(), vec![]),
            sequence(stochastic, vec![]),
        ];
        let sequences = sequences.iter().collect::<Vec<_>>();
        let logits = Tensor::new(&[[5.0f32, 4.0], [5.0, 4.0]], &Device::Cpu)?;
        let cuda = logits.to_device(&device)?;
        let mut cpu_contexts = vec![vec![0, 0]; 2];
        let mut cuda_contexts = cpu_contexts.clone();
        let reference = sample_draft_rows(&logits, &sequences, &mut cpu_contexts, &cpu_rng)?;
        let actual = sample_draft_rows(&cuda, &sequences, &mut cuda_contexts, &cuda_rng)?;
        assert_eq!(actual, reference);
        assert_eq!(cuda_contexts, cpu_contexts);
        let next = cpu_rng.lock().unwrap().next_u64();
        assert_eq!(cuda_rng.lock().unwrap().next_u64(), next);
        assert_ne!(Isaac64Rng::seed_from_u64(RNG_SEED).next_u64(), next);
        Ok(())
    }

    #[test]
    fn draft_rows_cuda_rejects_invalid_logits_before_updating_contexts() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(RNG_SEED)));
        let sequences = [
            sequence(SamplingParams::deterministic(), vec![]),
            sequence(SamplingParams::deterministic(), vec![]),
        ];
        let sequences = sequences.iter().collect::<Vec<_>>();
        for invalid in [
            [0.0f32, f32::NAN],
            [0.0, f32::INFINITY],
            [f32::NEG_INFINITY, f32::NEG_INFINITY],
        ] {
            let logits = Tensor::new(&[[0.0, 1.0], invalid], &device)?;
            let mut contexts = vec![vec![0, 0]; 2];
            assert!(sample_draft_rows(&logits, &sequences, &mut contexts, &rng).is_err());
            assert_eq!(contexts, vec![vec![0, 0]; 2]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{
        SparseSpeculativeProbs, SpeculativePrefillCaptureLayout, SpeculativeProposal,
        SpeculativeTapRouting, SpeculativeTapSpan, SpeculativeTokens,
    };

    #[test]
    fn dense_speculative_tap_routing_preserves_target_batch_rows() {
        let routing = SpeculativeTapRouting::new(
            SpeculativePrefillCaptureLayout::Dense,
            3,
            5,
            &[2, 0],
            &[(7, 10), (4, 9)],
        )
        .unwrap();

        assert_eq!(
            routing.spans(),
            &[
                SpeculativeTapSpan {
                    capture_batch_idx: 2,
                    capture_row_start: 0,
                    rows: 3,
                },
                SpeculativeTapSpan {
                    capture_batch_idx: 0,
                    capture_row_start: 0,
                    rows: 5,
                },
            ]
        );
        assert_eq!(
            routing.flat_row_indices().unwrap(),
            vec![10, 11, 12, 0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn packed_speculative_tap_routing_uses_cumulative_rows() {
        let routing = SpeculativeTapRouting::new(
            SpeculativePrefillCaptureLayout::Packed,
            1,
            8,
            &[0, 1],
            &[(7, 10), (4, 9)],
        )
        .unwrap();

        assert_eq!(
            routing.spans(),
            &[
                SpeculativeTapSpan {
                    capture_batch_idx: 0,
                    capture_row_start: 0,
                    rows: 3,
                },
                SpeculativeTapSpan {
                    capture_batch_idx: 0,
                    capture_row_start: 3,
                    rows: 5,
                },
            ]
        );
        assert_eq!(
            routing.flat_row_indices().unwrap(),
            (0..8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn speculative_tap_routing_rejects_inconsistent_packed_metadata() {
        assert!(SpeculativeTapRouting::new(
            SpeculativePrefillCaptureLayout::Packed,
            2,
            8,
            &[0, 1],
            &[(0, 3), (0, 5)],
        )
        .is_err());
        assert!(SpeculativeTapRouting::new(
            SpeculativePrefillCaptureLayout::Packed,
            1,
            9,
            &[0, 1],
            &[(0, 3), (0, 5)],
        )
        .is_err());
        assert!(SpeculativeTapRouting::new(
            SpeculativePrefillCaptureLayout::Packed,
            1,
            8,
            &[1, 0],
            &[(0, 3), (0, 5)],
        )
        .is_err());
    }

    #[test]
    fn sparse_probabilities_validate_shape_and_proposal_length() {
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (2, 2), &Device::Cpu).unwrap();
        let probs = Tensor::from_vec(vec![0.25f32, 0.75, 0.6, 0.4], (2, 2), &Device::Cpu).unwrap();
        let proposal =
            SpeculativeProposal::with_sparse_probs(vec![2, 3], ids.clone(), probs.clone()).unwrap();
        assert!(proposal.distribution.is_some());

        assert!(
            SpeculativeProposal::with_sparse_probs(vec![2], ids.clone(), probs.clone()).is_err()
        );
        assert!(SparseSpeculativeProbs::new(
            ids,
            Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu).unwrap(),
        )
        .is_err());
        assert!(SparseSpeculativeProbs::new(
            Tensor::zeros((2, 0), candle_core::DType::U32, &Device::Cpu).unwrap(),
            Tensor::zeros((2, 0), candle_core::DType::F32, &Device::Cpu).unwrap(),
        )
        .is_err());
    }

    #[test]
    fn device_tokens_materialize_without_changing_length() {
        let tensor = Tensor::from_vec(vec![7u32, 8, 9], 3, &Device::Cpu).unwrap();
        let mut tokens = SpeculativeTokens::from_device(tensor).unwrap();
        assert_eq!(tokens.len(), 3);
        assert!(tokens.as_host().is_none());
        assert_eq!(tokens.materialize().unwrap(), &[7, 8, 9]);
        assert!(tokens.as_device().is_none());
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn device_tokens_require_one_row() {
        let tensor = Tensor::zeros((1, 3), candle_core::DType::U32, &Device::Cpu).unwrap();
        assert!(SpeculativeTokens::from_device(tensor).is_err());
    }
}
