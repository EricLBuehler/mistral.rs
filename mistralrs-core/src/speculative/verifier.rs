use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use rand::Rng;
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::Pipeline;
use crate::prefix_cacher::PrefixCacheManagerV2;
#[cfg(feature = "cuda")]
use crate::sampler::CudaSpeculativeSamplingPlan;
use crate::sampler::{Logprobs, Sampler};
use crate::sequence::{Sequence, SequenceRecognizer, SequenceState};

#[cfg(feature = "cuda")]
use super::proposer::SpeculativeTokens;
use super::proposer::{SparseSpeculativeProbs, SpeculativeProposalDistribution};

pub(crate) fn can_greedy_device_verify(seq: &Sequence) -> bool {
    #[cfg(feature = "cuda")]
    {
        !seq.return_logprobs()
            && !seq.sampling_logprob_required()
            && matches!(seq.recognizer, SequenceRecognizer::None)
            && seq.tool_call_state.is_none()
            && seq
                .sampler()
                .cuda_batch_sampling_plan(false)
                .is_some_and(|plan| plan.kind.is_argmax())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = seq;
        false
    }
}

pub(crate) fn can_batch_greedy_device_verify(seqs: &[&mut Sequence]) -> bool {
    crate::speculative::staging::staged_batch_width(seqs).is_some()
        && seqs.iter().all(|seq| can_greedy_device_verify(seq))
}

#[cfg(feature = "cuda")]
fn sparse_rejection_plan(seq: &Sequence) -> Option<CudaSpeculativeSamplingPlan> {
    if seq.return_logprobs()
        || seq.sampling_logprob_required()
        || !stochastic_verification_allowed_for_sequence(seq)
    {
        return None;
    }
    seq.sampler().cuda_speculative_sampling_plan(false)
}

#[cfg(feature = "cuda")]
fn sparse_distribution_is_cuda_eligible(sparse: &SparseSpeculativeProbs, drafts: usize) -> bool {
    let [positions, q_width] = *sparse.token_ids().dims() else {
        return false;
    };
    drafts > 0
        && positions == drafts
        && q_width > 0
        && q_width <= crate::cuda::speculative_rejection::SPARSE_REJECTION_MAX_Q_WIDTH
        && sparse.probs().dims() == [drafts, q_width]
        && sparse.token_ids().device().is_cuda()
        && sparse
            .token_ids()
            .device()
            .same_device(sparse.probs().device())
}

pub(crate) fn can_batch_device_verify(seqs: &[&mut Sequence]) -> bool {
    if can_batch_greedy_device_verify(seqs) {
        return true;
    }
    #[cfg(feature = "cuda")]
    {
        crate::speculative::staging::staged_batch_width(seqs).is_some()
            && seqs.iter().all(|seq| {
                if can_greedy_device_verify(seq) {
                    return true;
                }
                let Some(_) = sparse_rejection_plan(seq) else {
                    return false;
                };
                match seq.staged_speculative_distribution() {
                    None => true,
                    Some(SpeculativeProposalDistribution::SparseProbs(sparse)) => {
                        sparse_distribution_is_cuda_eligible(
                            sparse,
                            seq.active_staged_speculative_len(),
                        )
                    }
                    Some(SpeculativeProposalDistribution::Logits(_)) => false,
                }
            })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = seqs;
        false
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct GreedyDeviceVerifyInput<'a> {
    pub(crate) seq: &'a Sequence,
    pub(crate) logits: &'a Tensor,
}

#[cfg(feature = "cuda")]
pub(crate) fn greedy_device_verify_batch(
    inputs: &[GreedyDeviceVerifyInput<'_>],
) -> Result<Vec<Option<Vec<u32>>>> {
    let mut outputs = std::iter::repeat_with(|| None)
        .take(inputs.len())
        .collect::<Vec<_>>();
    let mut selected_indices = Vec::new();
    let mut selected_logits = Vec::new();
    let mut row_counts = Vec::new();
    let mut common_device: Option<candle_core::Device> = None;
    let mut common_dtype = None;
    let mut common_vocab = None;

    for (idx, input) in inputs.iter().enumerate() {
        if !input.logits.device().is_cuda() || !can_greedy_device_verify(input.seq) {
            continue;
        }
        let logits = match *input.logits.dims() {
            [1, rows, vocab] => input.logits.reshape((rows, vocab))?,
            [_, _] => input.logits.clone(),
            _ => continue,
        };
        let [rows, vocab] = *logits.dims() else {
            unreachable!("verify logits were normalized to rank two")
        };
        if rows == 0 || vocab == 0 {
            continue;
        }

        match (&common_device, common_dtype, common_vocab) {
            (Some(device), Some(dtype), Some(expected_vocab))
                if !device.same_device(logits.device())
                    || dtype != logits.dtype()
                    || expected_vocab != vocab =>
            {
                continue;
            }
            (Some(_), Some(_), Some(_)) => {}
            (None, None, None) => {
                common_device = Some(logits.device().clone());
                common_dtype = Some(logits.dtype());
                common_vocab = Some(vocab);
            }
            _ => unreachable!("CUDA verify batch properties are initialized together"),
        }
        selected_indices.push(idx);
        selected_logits.push(logits);
        row_counts.push(rows);
    }

    let Some(first_idx) = selected_indices.first().copied() else {
        return Ok(outputs);
    };
    let logits = if let [logits] = selected_logits.as_slice() {
        logits.clone()
    } else {
        Tensor::cat(&selected_logits.iter().collect::<Vec<_>>(), 0)?
    }
    .contiguous()?;
    let token_ids = inputs[first_idx]
        .seq
        .sampler()
        .submit_cuda_top1_batch_owned(&logits)?
        .complete()?
        .token_ids;
    let tokens = partition_device_tokens(token_ids, &row_counts)?;
    for (idx, tokens) in selected_indices.into_iter().zip(tokens) {
        outputs[idx] = Some(tokens);
    }
    Ok(outputs)
}

#[cfg(feature = "cuda")]
pub(crate) struct SparseRejectionVerifyInput<'a> {
    pub(crate) seq: &'a Sequence,
    pub(crate) logits: &'a Tensor,
    pub(crate) proposal: &'a SpeculativeTokens,
    pub(crate) distribution: Option<&'a SpeculativeProposalDistribution>,
}

#[cfg(feature = "cuda")]
enum SparseRejectionCandidateDistribution {
    Deterministic,
    Sparse {
        token_ids: Tensor,
        probs: Tensor,
        width: usize,
    },
}

#[cfg(feature = "cuda")]
impl SparseRejectionCandidateDistribution {
    fn width(&self) -> usize {
        match self {
            Self::Deterministic => 1,
            Self::Sparse { width, .. } => *width,
        }
    }

    fn is_deterministic(&self) -> bool {
        matches!(self, Self::Deterministic)
    }
}

#[cfg(feature = "cuda")]
struct SparseRejectionCandidate<'a> {
    input_idx: usize,
    logits: Tensor,
    proposal: &'a SpeculativeTokens,
    distribution: SparseRejectionCandidateDistribution,
    plan: CudaSpeculativeSamplingPlan,
    drafts: usize,
    vocab: usize,
}

#[cfg(feature = "cuda")]
enum SparseRejectionDraftTokens {
    Host(Vec<u32>),
    Device(Tensor),
}

#[cfg(feature = "cuda")]
impl SparseRejectionDraftTokens {
    fn as_input(&self) -> crate::cuda::speculative_rejection::SparseRejectionDraftInput<'_> {
        match self {
            Self::Host(tokens) => {
                crate::cuda::speculative_rejection::SparseRejectionDraftInput::Host(tokens)
            }
            Self::Device(tokens) => {
                crate::cuda::speculative_rejection::SparseRejectionDraftInput::Device(tokens)
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct SparseRejectionDeviceVerifyOutput {
    pub(crate) verifications: Vec<Option<DeviceVerification>>,
    pub(crate) materialized_proposals: Vec<Option<Vec<u32>>>,
}

#[cfg(any(feature = "cuda", test))]
fn packed_target_shape_matches(dims: &[usize], batch: usize, drafts: usize, vocab: usize) -> bool {
    drafts
        .checked_add(1)
        .is_some_and(|rows| dims == [batch, rows, vocab])
}

#[cfg(any(feature = "cuda", test))]
fn is_complete_ordered_sparse_group(
    input_count: usize,
    candidate_input_indices: impl Iterator<Item = usize>,
    group_indices: impl Iterator<Item = usize>,
) -> bool {
    candidate_input_indices.eq(0..input_count) && group_indices.eq(0..input_count)
}

#[cfg(feature = "cuda")]
fn batch_sparse_rejection_logits(logits: Vec<Tensor>) -> Result<Tensor> {
    match logits.as_slice() {
        [logits] => Ok(logits.clone()),
        _ => Tensor::cat(&logits.iter().collect::<Vec<_>>(), 0),
    }
}

#[cfg(feature = "cuda")]
fn sparse_rejection_candidate<'a>(
    input_idx: usize,
    input: &'a SparseRejectionVerifyInput<'a>,
) -> Result<Option<SparseRejectionCandidate<'a>>> {
    let Some(plan) = sparse_rejection_plan(input.seq) else {
        return Ok(None);
    };
    let drafts = input.proposal.len();
    let distribution = match input.distribution {
        None => SparseRejectionCandidateDistribution::Deterministic,
        Some(SpeculativeProposalDistribution::SparseProbs(sparse)) => {
            if !sparse_distribution_is_cuda_eligible(sparse, drafts) {
                return Ok(None);
            }
            SparseRejectionCandidateDistribution::Sparse {
                token_ids: sparse.token_ids().contiguous()?,
                probs: sparse.probs().contiguous()?,
                width: sparse.token_ids().dim(1)?,
            }
        }
        Some(SpeculativeProposalDistribution::Logits(_)) => return Ok(None),
    };
    let logits = match input.logits.dims() {
        [1, rows, _] if *rows == drafts + 1 => input.logits.clone(),
        [rows, _] if *rows == drafts + 1 => input.logits.unsqueeze(0)?,
        _ => return Ok(None),
    };
    let [_, _, vocab] = *logits.dims() else {
        unreachable!("sparse rejection logits were normalized to rank three")
    };
    if plan.top_k > 0 && !matches!(logits.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }
    if !logits.device().is_cuda()
        || matches!(
            input.proposal.as_device(),
            Some(tokens)
                if tokens.dtype() != DType::U32
                    || tokens.dims() != [drafts]
                    || !tokens.is_contiguous()
                    || !logits.device().same_device(tokens.device())
        )
        || matches!(
            &distribution,
            SparseRejectionCandidateDistribution::Sparse { token_ids, .. }
                if !logits.device().same_device(token_ids.device())
        )
    {
        return Ok(None);
    }
    Ok(Some(SparseRejectionCandidate {
        input_idx,
        logits,
        proposal: input.proposal,
        distribution,
        plan,
        drafts,
        vocab,
    }))
}

#[cfg(feature = "cuda")]
struct SparseRejectionPendingCandidate {
    input_idx: usize,
    drafts: usize,
    vocab: usize,
    accept_uniforms: Vec<f32>,
    sample_uniform: f32,
}

#[cfg(feature = "cuda")]
pub(crate) struct SparseRejectionDeviceBatchSubmission {
    input_count: usize,
    candidates: Vec<SparseRejectionPendingCandidate>,
    submission: crate::cuda::speculative_rejection::CudaSparseRejectionSubmission,
}

#[cfg(feature = "cuda")]
fn decode_sparse_rejection_row(
    candidate: SparseRejectionPendingCandidate,
    row: crate::cuda::speculative_rejection::SparseRejectionRow,
) -> Result<DeviceVerification> {
    use crate::cuda::speculative_rejection::{
        SPARSE_REJECTION_INVALID_VALUE, SPARSE_REJECTION_STATUS_INVALID_Q,
        SPARSE_REJECTION_STATUS_INVALID_RNG, SPARSE_REJECTION_STATUS_INVALID_TARGET,
        SPARSE_REJECTION_STATUS_NEEDS_CPU, SPARSE_REJECTION_STATUS_OK,
    };

    match row.status {
        SPARSE_REJECTION_STATUS_OK => {
            let accepted_drafts =
                usize::try_from(row.accepted_count).map_err(candle_core::Error::wrap)?;
            if row.accepted_count == SPARSE_REJECTION_INVALID_VALUE
                || row.continuation == SPARSE_REJECTION_INVALID_VALUE
                || accepted_drafts > candidate.drafts
                || usize::try_from(row.continuation).map_err(candle_core::Error::wrap)?
                    >= candidate.vocab
            {
                candle_core::bail!(
                    "sparse CUDA verification returned an invalid outcome: accepted={}, continuation={}, drafts={}, vocab={}",
                    row.accepted_count,
                    row.continuation,
                    candidate.drafts,
                    candidate.vocab
                );
            }
            metrics::counter!("mistralrs_speculative_sparse_gpu_verify_total").increment(1);
            Ok(DeviceVerification::SparseRejection {
                accepted_drafts,
                continuation_token: row.continuation,
            })
        }
        SPARSE_REJECTION_STATUS_NEEDS_CPU => {
            metrics::counter!("mistralrs_speculative_sparse_gpu_fallback_total").increment(1);
            Ok(DeviceVerification::SparseRejectionCpuFallback(
                SparseRejectionFallbackUniforms {
                    accept: candidate.accept_uniforms,
                    sample: candidate.sample_uniform,
                },
            ))
        }
        SPARSE_REJECTION_STATUS_INVALID_Q => {
            candle_core::bail!("sparse CUDA verification rejected invalid proposal probabilities");
        }
        SPARSE_REJECTION_STATUS_INVALID_TARGET => {
            candle_core::bail!(
                "sparse CUDA verification rejected invalid target logits or sampling parameters"
            );
        }
        SPARSE_REJECTION_STATUS_INVALID_RNG => {
            candle_core::bail!("sparse CUDA verification received an invalid random draw");
        }
        status => candle_core::bail!("sparse CUDA verification returned unknown status {status}"),
    }
}

#[cfg(feature = "cuda")]
fn decode_sparse_rejection_completion(
    input_count: usize,
    candidates: Vec<SparseRejectionPendingCandidate>,
    completion: crate::cuda::speculative_rejection::SparseRejectionCompletion,
) -> Result<SparseRejectionDeviceVerifyOutput> {
    if completion.rows.len() != candidates.len()
        || completion.draft_tokens.len() != candidates.len()
    {
        candle_core::bail!("sparse CUDA verification returned the wrong batch size");
    }
    let mut outputs = std::iter::repeat_with(|| None)
        .take(input_count)
        .collect::<Vec<_>>();
    let mut materialized_proposals = std::iter::repeat_with(|| None)
        .take(input_count)
        .collect::<Vec<_>>();
    for ((candidate, row), proposal) in candidates
        .into_iter()
        .zip(completion.rows)
        .zip(completion.draft_tokens)
    {
        let input_idx = candidate.input_idx;
        materialized_proposals[input_idx] = Some(proposal);
        outputs[input_idx] = Some(decode_sparse_rejection_row(candidate, row)?);
    }
    Ok(SparseRejectionDeviceVerifyOutput {
        verifications: outputs,
        materialized_proposals,
    })
}

#[cfg(feature = "cuda")]
fn sparse_rejection_candidates<'a>(
    inputs: &'a [SparseRejectionVerifyInput<'a>],
) -> Result<Vec<SparseRejectionCandidate<'a>>> {
    inputs
        .iter()
        .enumerate()
        .filter_map(|(input_idx, input)| sparse_rejection_candidate(input_idx, input).transpose())
        .collect()
}

#[cfg(feature = "cuda")]
fn sparse_rejection_same_group(
    candidate: &SparseRejectionCandidate<'_>,
    seed: &SparseRejectionCandidate<'_>,
) -> bool {
    let top_k_mode = seed.plan.top_k > 0;
    candidate.drafts == seed.drafts
        && candidate.vocab == seed.vocab
        && candidate.distribution.width() == seed.distribution.width()
        && candidate.distribution.is_deterministic() == seed.distribution.is_deterministic()
        && (candidate.plan.top_k > 0) == top_k_mode
        && (!top_k_mode || candidate.logits.dtype() == seed.logits.dtype())
        && candidate.logits.device().same_device(seed.logits.device())
        && match (candidate.proposal.as_device(), seed.proposal.as_device()) {
            (Some(candidate), Some(seed)) => candidate.device().same_device(seed.device()),
            (None, None) => true,
            _ => false,
        }
}

#[cfg(feature = "cuda")]
fn sparse_rejection_uniforms(
    candidates: &[SparseRejectionCandidate<'_>],
    inputs: &[SparseRejectionVerifyInput<'_>],
    rng: &Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Vec<(Vec<f32>, f32)> {
    candidates
        .iter()
        .map(|candidate| {
            let rng = inputs[candidate.input_idx].seq.sampling_rng(rng);
            let mut rng = rng.lock().expect("could not lock rng mutex");
            (
                (0..candidate.drafts).map(|_| rng.random::<f32>()).collect(),
                rng.random::<f32>(),
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn submit_sparse_rejection_group(
    input_count: usize,
    candidates: &[SparseRejectionCandidate<'_>],
    group_indices: &[usize],
    candidate_uniforms: &[(Vec<f32>, f32)],
    batched_target_logits: Option<&Tensor>,
    workspace: &mut Option<crate::cuda::speculative_rejection::CudaSparseRejectionWorkspace>,
) -> Result<SparseRejectionDeviceBatchSubmission> {
    use crate::cuda::speculative_rejection::{
        sparse_rejection_cuda_submit, SparseRejectionMode, SparseRejectionProposalInput,
        SparseRejectionWorkspaceInput,
    };

    let seed = &candidates[group_indices[0]];
    let group = group_indices
        .iter()
        .map(|&idx| &candidates[idx])
        .collect::<Vec<_>>();
    let batch = group.len();
    let drafts = seed.drafts;
    let device = seed.logits.device();
    let mode = if seed.plan.top_k > 0 {
        SparseRejectionMode::BoundedTopK {
            max_top_k: group
                .iter()
                .map(|row| row.plan.top_k)
                .max()
                .expect("sparse rejection group is non-empty"),
        }
    } else {
        SparseRejectionMode::Categorical
    };
    let batched_target_logits = batched_target_logits.filter(|batched| {
        is_complete_ordered_sparse_group(
            input_count,
            candidates.iter().map(|candidate| candidate.input_idx),
            group_indices.iter().copied(),
        ) && packed_target_shape_matches(batched.dims(), batch, drafts, seed.vocab)
            && batched.device().is_cuda()
            && batched.device().same_device(device)
            && group.iter().all(|candidate| {
                candidate.logits.dtype() == batched.dtype()
                    && candidate.logits.device().same_device(batched.device())
            })
    });
    let target_logits = match (mode, batched_target_logits) {
        (SparseRejectionMode::BoundedTopK { .. }, Some(batched)) => batched.contiguous()?,
        (SparseRejectionMode::BoundedTopK { .. }, None) => {
            let logits = group
                .iter()
                .map(|row| row.logits.contiguous())
                .collect::<Result<Vec<_>>>()?;
            batch_sparse_rejection_logits(logits)?
        }
        (SparseRejectionMode::Categorical, Some(batched)) => {
            batched.to_dtype(DType::F32)?.contiguous()?
        }
        (SparseRejectionMode::Categorical, None) => {
            let logits = group
                .iter()
                .map(|row| row.logits.to_dtype(DType::F32)?.contiguous())
                .collect::<Result<Vec<_>>>()?;
            batch_sparse_rejection_logits(logits)?
        }
    };
    let sparse_q = if seed.distribution.is_deterministic() {
        None
    } else {
        let q_token_rows = group
            .iter()
            .map(|row| match &row.distribution {
                SparseRejectionCandidateDistribution::Sparse { token_ids, .. } => {
                    token_ids.unsqueeze(0)
                }
                SparseRejectionCandidateDistribution::Deterministic => {
                    unreachable!("sparse rejection groups have one proposal distribution kind")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let q_prob_rows = group
            .iter()
            .map(|row| match &row.distribution {
                SparseRejectionCandidateDistribution::Sparse { probs, .. } => probs.unsqueeze(0),
                SparseRejectionCandidateDistribution::Deterministic => {
                    unreachable!("sparse rejection groups have one proposal distribution kind")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Some((
            Tensor::cat(&q_token_rows.iter().collect::<Vec<_>>(), 0)?,
            Tensor::cat(&q_prob_rows.iter().collect::<Vec<_>>(), 0)?,
        ))
    };
    let proposal = sparse_q.as_ref().map_or(
        SparseRejectionProposalInput::Deterministic,
        |(token_ids, probs)| SparseRejectionProposalInput::Sparse { token_ids, probs },
    );
    let draft_tokens = if seed.proposal.as_device().is_some() {
        let rows = group
            .iter()
            .map(|row| {
                row.proposal
                    .as_device()
                    .expect("sparse rejection group has one proposal storage kind")
            })
            .collect::<Vec<_>>();
        let tokens = match rows.as_slice() {
            [row] => row.unsqueeze(0)?.contiguous()?,
            _ => Tensor::stack(&rows, 0)?.contiguous()?,
        };
        SparseRejectionDraftTokens::Device(tokens)
    } else {
        SparseRejectionDraftTokens::Host(
            group
                .iter()
                .flat_map(|row| {
                    row.proposal
                        .as_host()
                        .expect("sparse rejection group has one proposal storage kind")
                        .iter()
                        .copied()
                })
                .collect(),
        )
    };
    let inverse_temperatures = group
        .iter()
        .map(|row| row.plan.inverse_temperature)
        .collect::<Vec<_>>();
    let target_top_k = group
        .iter()
        .map(|row| u32::try_from(row.plan.top_k).map_err(candle_core::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    let top_p = group.iter().map(|row| row.plan.top_p).collect::<Vec<_>>();
    let min_p = group.iter().map(|row| row.plan.min_p).collect::<Vec<_>>();
    let accept_uniforms = group_indices
        .iter()
        .flat_map(|&idx| candidate_uniforms[idx].0.iter().copied())
        .collect::<Vec<_>>();
    let sample_uniforms = group_indices
        .iter()
        .map(|&idx| candidate_uniforms[idx].1)
        .collect::<Vec<_>>();
    let submission = sparse_rejection_cuda_submit(
        SparseRejectionWorkspaceInput {
            target_logits: &target_logits,
            proposal,
            draft_tokens: draft_tokens.as_input(),
            inverse_temperatures: &inverse_temperatures,
            target_top_k: &target_top_k,
            top_p: &top_p,
            min_p: &min_p,
            accept_uniforms: &accept_uniforms,
            sample_uniforms: &sample_uniforms,
            mode,
        },
        workspace,
    )?;
    Ok(SparseRejectionDeviceBatchSubmission {
        input_count,
        candidates: group_indices
            .iter()
            .map(|&idx| SparseRejectionPendingCandidate {
                input_idx: candidates[idx].input_idx,
                drafts: candidates[idx].drafts,
                vocab: candidates[idx].vocab,
                accept_uniforms: candidate_uniforms[idx].0.clone(),
                sample_uniform: candidate_uniforms[idx].1,
            })
            .collect(),
        submission,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn complete_sparse_rejection_device_verify_batch(
    pending: SparseRejectionDeviceBatchSubmission,
    workspace: &mut Option<crate::cuda::speculative_rejection::CudaSparseRejectionWorkspace>,
) -> Result<SparseRejectionDeviceVerifyOutput> {
    use crate::cuda::speculative_rejection::sparse_rejection_cuda_complete;

    let completion = sparse_rejection_cuda_complete(workspace, &pending.submission)?;
    decode_sparse_rejection_completion(pending.input_count, pending.candidates, completion)
}

#[cfg(feature = "cuda")]
pub(crate) fn try_submit_sparse_rejection_device_verify_batch(
    inputs: &[SparseRejectionVerifyInput<'_>],
    batched_target_logits: Option<&Tensor>,
    rng: &Arc<std::sync::Mutex<Isaac64Rng>>,
    workspace: &mut Option<crate::cuda::speculative_rejection::CudaSparseRejectionWorkspace>,
) -> Result<Option<SparseRejectionDeviceBatchSubmission>> {
    let candidates = sparse_rejection_candidates(inputs)?;
    let Some(seed) = candidates.first() else {
        return Ok(None);
    };
    if candidates.len() != inputs.len()
        || !candidates
            .iter()
            .all(|candidate| sparse_rejection_same_group(candidate, seed))
    {
        return Ok(None);
    }
    let candidate_uniforms = sparse_rejection_uniforms(&candidates, inputs, rng);
    let group_indices = (0..candidates.len()).collect::<Vec<_>>();
    submit_sparse_rejection_group(
        inputs.len(),
        &candidates,
        &group_indices,
        &candidate_uniforms,
        batched_target_logits,
        workspace,
    )
    .map(Some)
}

#[cfg(feature = "cuda")]
pub(crate) fn sparse_rejection_device_verify_batch(
    inputs: &[SparseRejectionVerifyInput<'_>],
    batched_target_logits: Option<&Tensor>,
    rng: &Arc<std::sync::Mutex<Isaac64Rng>>,
    workspace: &mut Option<crate::cuda::speculative_rejection::CudaSparseRejectionWorkspace>,
) -> Result<SparseRejectionDeviceVerifyOutput> {
    let candidates = sparse_rejection_candidates(inputs)?;
    let candidate_uniforms = sparse_rejection_uniforms(&candidates, inputs, rng);
    let mut outputs = std::iter::repeat_with(|| None)
        .take(inputs.len())
        .collect::<Vec<_>>();
    let mut materialized_proposals = std::iter::repeat_with(|| None)
        .take(inputs.len())
        .collect::<Vec<_>>();
    let mut remaining = vec![true; candidates.len()];
    while let Some(seed_idx) = remaining.iter().position(|pending| *pending) {
        let seed = &candidates[seed_idx];
        let group_indices = candidates
            .iter()
            .enumerate()
            .filter_map(|(idx, candidate)| {
                (remaining[idx] && sparse_rejection_same_group(candidate, seed)).then_some(idx)
            })
            .collect::<Vec<_>>();
        for &idx in &group_indices {
            remaining[idx] = false;
        }
        let pending = submit_sparse_rejection_group(
            inputs.len(),
            &candidates,
            &group_indices,
            &candidate_uniforms,
            batched_target_logits,
            workspace,
        )?;
        let completed = complete_sparse_rejection_device_verify_batch(pending, workspace)?;
        for (idx, verification) in completed.verifications.into_iter().enumerate() {
            if verification.is_some() {
                outputs[idx] = verification;
            }
        }
        for (idx, proposal) in completed.materialized_proposals.into_iter().enumerate() {
            if proposal.is_some() {
                materialized_proposals[idx] = proposal;
            }
        }
    }
    Ok(SparseRejectionDeviceVerifyOutput {
        verifications: outputs,
        materialized_proposals,
    })
}

fn device_token_logprobs(token: u32) -> Logprobs {
    Logprobs {
        token,
        logprob: 0.0,
        top_logprobs: None,
        bytes: None,
    }
}

pub struct VerificationOutcome {
    pub accepted_drafts: usize,
    pub proposed_drafts: usize,
    pub keep_len: usize,
    pub continuation_token: Option<u32>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) enum DeviceVerification {
    TargetTokens(Vec<u32>),
    SparseRejection {
        accepted_drafts: usize,
        continuation_token: u32,
    },
    SparseRejectionCpuFallback(SparseRejectionFallbackUniforms),
}

pub(crate) struct SparseRejectionFallbackUniforms {
    accept: Vec<f32>,
    sample: f32,
}

pub(crate) struct VerificationInput {
    pub(crate) verify_logits: Tensor,
    pub(crate) proposal: Vec<u32>,
    pub(crate) proposal_distribution: Option<SpeculativeProposalDistribution>,
    pub(crate) base_len: usize,
    pub(crate) anchor_to_emit: Option<Logprobs>,
    pub(crate) device_verification: Option<DeviceVerification>,
}

pub(crate) async fn finish_verified_step<P: Pipeline>(
    pipeline: &P,
    seq: &mut Sequence,
    input: VerificationInput,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<VerificationOutcome> {
    let rng = seq.sampling_rng(&rng);
    let VerificationInput {
        verify_logits,
        proposal,
        proposal_distribution,
        base_len,
        anchor_to_emit,
        device_verification,
    } = input;
    let general_metadata = pipeline.get_metadata();
    let eos_tok = seq.effective_eos_tokens(&general_metadata.eos_tok, disable_eos_stop);
    let return_logprobs = seq.return_logprobs();

    if let Some(anchor) = anchor_to_emit {
        finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, anchor, eos_tok, true).await?;
        if matches!(seq.getstate(), SequenceState::Done(_)) {
            let keep_len = base_len + 1;
            seq.clear_staged_speculative_tokens();
            return Ok(VerificationOutcome {
                accepted_drafts: 0,
                proposed_drafts: proposal.len(),
                keep_len,
                continuation_token: None,
            });
        }
    }

    validate_device_verification(device_verification.as_ref(), proposal.len())?;

    let fallback_uniforms = match &device_verification {
        Some(DeviceVerification::SparseRejectionCpuFallback(uniforms)) => Some(uniforms),
        _ => None,
    };
    let stochastic = fallback_uniforms.is_some()
        || (device_verification.is_none() && proposal_distribution.is_some());
    if stochastic && stochastic_verification_allowed_for_sequence(seq) {
        return finish_verified_step_stochastic(
            pipeline,
            seq,
            verify_logits,
            proposal,
            proposal_distribution,
            base_len,
            prefix_cacher,
            eos_tok,
            return_logprobs,
            rng,
            fallback_uniforms,
        )
        .await;
    }
    if fallback_uniforms.is_some() {
        candle_core::bail!("sparse CUDA verification cannot fall back under constrained sampling");
    }

    let mut accepted = 0usize;
    for (idx, draft) in proposal.iter().copied().enumerate() {
        let (sampled, accepted_by_device) = match &device_verification {
            Some(DeviceVerification::TargetTokens(tokens)) => {
                (device_token_logprobs(tokens[idx]), None)
            }
            Some(DeviceVerification::SparseRejection {
                accepted_drafts,
                continuation_token,
            }) => {
                if idx < *accepted_drafts {
                    (device_token_logprobs(draft), Some(true))
                } else {
                    (device_token_logprobs(*continuation_token), Some(false))
                }
            }
            Some(DeviceVerification::SparseRejectionCpuFallback(_)) => {
                unreachable!("sparse CPU fallback returned before token verification")
            }
            None => {
                let row = logit_row(&verify_logits, idx)?;
                (
                    sample_sequence(
                        row.clone(),
                        seq,
                        return_logprobs,
                        eos_tok,
                        general_metadata.llg_factory.clone(),
                        general_metadata.max_seq_len,
                        rng.clone(),
                        false,
                        false,
                        false,
                    )
                    .await?,
                    None,
                )
            }
        };
        let sampled_token = sampled.token;
        if accepted_by_device.unwrap_or(sampled_token == draft) {
            accepted += 1;
            finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
            if matches!(seq.getstate(), SequenceState::Done(_)) {
                let keep_len = base_len + 1 + accepted;
                seq.clear_staged_speculative_tokens();
                return Ok(VerificationOutcome {
                    accepted_drafts: accepted,
                    proposed_drafts: proposal.len(),
                    keep_len,
                    continuation_token: None,
                });
            }
        } else {
            let keep_len = base_len + 1 + accepted;
            finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
            if matches!(seq.getstate(), SequenceState::Done(_)) {
                seq.clear_staged_speculative_tokens();
                return Ok(VerificationOutcome {
                    accepted_drafts: accepted,
                    proposed_drafts: proposal.len(),
                    keep_len,
                    continuation_token: None,
                });
            }
            return Ok(VerificationOutcome {
                accepted_drafts: accepted,
                proposed_drafts: proposal.len(),
                keep_len,
                continuation_token: Some(sampled_token),
            });
        }
    }

    let continuation = match &device_verification {
        Some(DeviceVerification::TargetTokens(tokens)) => device_token_logprobs(tokens[accepted]),
        Some(DeviceVerification::SparseRejection {
            continuation_token, ..
        }) => device_token_logprobs(*continuation_token),
        Some(DeviceVerification::SparseRejectionCpuFallback(_)) => {
            unreachable!("sparse CPU fallback returned before continuation sampling")
        }
        None => {
            let row = logit_row(&verify_logits, accepted)?;
            sample_sequence(
                row.clone(),
                seq,
                return_logprobs,
                eos_tok,
                general_metadata.llg_factory.clone(),
                general_metadata.max_seq_len,
                rng.clone(),
                false,
                false,
                false,
            )
            .await?
        }
    };
    let continuation_token = continuation.token;
    finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, continuation, eos_tok, true).await?;

    let keep_len = base_len + 1 + accepted;
    let continuation_token = if matches!(seq.getstate(), SequenceState::Done(_)) {
        seq.clear_staged_speculative_tokens();
        None
    } else {
        Some(continuation_token)
    };

    Ok(VerificationOutcome {
        accepted_drafts: accepted,
        proposed_drafts: proposal.len(),
        keep_len,
        continuation_token,
    })
}

fn validate_device_verification(
    verification: Option<&DeviceVerification>,
    proposal_len: usize,
) -> Result<()> {
    match verification {
        Some(DeviceVerification::TargetTokens(tokens)) if tokens.len() < proposal_len + 1 => {
            candle_core::bail!(
                "speculative CUDA verification returned fewer tokens than required: got {}, need {}",
                tokens.len(),
                proposal_len + 1
            );
        }
        Some(DeviceVerification::SparseRejection {
            accepted_drafts, ..
        }) if *accepted_drafts > proposal_len => {
            candle_core::bail!(
                "speculative CUDA verification accepted {accepted_drafts} of {proposal_len} drafts"
            );
        }
        Some(DeviceVerification::SparseRejectionCpuFallback(uniforms))
            if uniforms.accept.len() != proposal_len
                || uniforms.accept.iter().any(|draw| !valid_uniform(*draw))
                || !valid_uniform(uniforms.sample) =>
        {
            candle_core::bail!("sparse CUDA verification returned invalid fallback uniforms");
        }
        _ => {}
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn partition_device_tokens(tokens: Vec<u32>, row_counts: &[usize]) -> Result<Vec<Vec<u32>>> {
    let expected = row_counts.iter().try_fold(0usize, |total, &rows| {
        total.checked_add(rows).ok_or_else(|| {
            candle_core::Error::Msg("speculative CUDA verify row count overflow".to_string())
        })
    })?;
    if tokens.len() != expected {
        candle_core::bail!(
            "speculative CUDA verification returned {} tokens for {expected} rows",
            tokens.len()
        );
    }
    let mut offset = 0;
    Ok(row_counts
        .iter()
        .map(|&rows| {
            let end = offset + rows;
            let row = tokens[offset..end].to_vec();
            offset = end;
            row
        })
        .collect())
}

fn stochastic_verification_allowed(
    is_argmax: bool,
    has_constraint: bool,
    has_tool_call_state: bool,
) -> bool {
    !is_argmax && !has_constraint && !has_tool_call_state
}

pub(crate) fn stochastic_verification_allowed_for_sequence(seq: &Sequence) -> bool {
    stochastic_verification_allowed(
        seq.sampler().is_argmax(),
        !matches!(seq.recognizer, SequenceRecognizer::None),
        seq.tool_call_state.is_some(),
    )
}

enum PreparedProposalDistribution {
    Deterministic,
    Logits(Tensor),
    Sparse {
        token_ids: Vec<Vec<u32>>,
        probs: Vec<Vec<f32>>,
    },
}

impl PreparedProposalDistribution {
    fn new(distribution: SpeculativeProposalDistribution, positions: usize) -> Result<Self> {
        match distribution {
            SpeculativeProposalDistribution::Logits(logits) => Ok(Self::Logits(logits)),
            SpeculativeProposalDistribution::SparseProbs(sparse) => {
                validate_sparse_positions(&sparse, positions)?;
                Ok(Self::Sparse {
                    token_ids: sparse.token_ids().to_vec2::<u32>()?,
                    probs: sparse.probs().to_vec2::<f32>()?,
                })
            }
        }
    }

    fn probability_row(
        &self,
        row: usize,
        draft: u32,
        sampler: &Sampler,
        context: &[u32],
        prompt_len: usize,
        vocab: usize,
    ) -> Result<ProposalProbabilityRow> {
        match self {
            Self::Deterministic => Ok(ProposalProbabilityRow::Sparse(vec![(draft as usize, 1.0)])),
            Self::Logits(logits) => {
                let candidate_row = logit_row(logits, row)?;
                let probs = sampler.speculative_candidate_probs(
                    flat_logits(candidate_row)?,
                    context,
                    prompt_len,
                )?;
                if probs.len() != vocab {
                    candle_core::bail!(
                        "speculative target/candidate vocab mismatch: target={vocab}, candidate={}",
                        probs.len()
                    );
                }
                Ok(ProposalProbabilityRow::Dense(probs))
            }
            Self::Sparse { token_ids, probs } => {
                let token_ids = token_ids.get(row).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "sparse speculative probability row {row} is out of range"
                    ))
                })?;
                let probs = probs.get(row).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "sparse speculative probability row {row} is out of range"
                    ))
                })?;
                Ok(ProposalProbabilityRow::Sparse(normalize_sparse_row(
                    row, token_ids, probs, vocab,
                )?))
            }
        }
    }
}

fn validate_sparse_positions(sparse: &SparseSpeculativeProbs, positions: usize) -> Result<()> {
    if sparse.positions() != positions {
        candle_core::bail!(
            "sparse speculative probabilities have {} positions for {positions} tokens",
            sparse.positions()
        );
    }
    Ok(())
}

enum ProposalProbabilityRow {
    Dense(Vec<f32>),
    Sparse(Vec<(usize, f32)>),
}

impl ProposalProbabilityRow {
    fn probability(&self, token: usize) -> f32 {
        match self {
            Self::Dense(probs) => probs.get(token).copied().unwrap_or(0.0),
            Self::Sparse(entries) => entries
                .iter()
                .find_map(|(candidate, prob)| (*candidate == token).then_some(*prob))
                .unwrap_or(0.0),
        }
    }

    fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    fn subtract_from(&self, target: &mut [f32]) -> Result<()> {
        match self {
            Self::Dense(probs) => {
                if probs.len() != target.len() {
                    candle_core::bail!(
                        "speculative target/candidate vocab mismatch: target={}, candidate={}",
                        target.len(),
                        probs.len()
                    );
                }
                for (target, candidate) in target.iter_mut().zip(probs) {
                    *target -= candidate;
                }
            }
            Self::Sparse(entries) => {
                for &(token, prob) in entries {
                    target[token] -= prob;
                }
            }
        }
        Ok(())
    }
}

fn normalize_sparse_row(
    row: usize,
    token_ids: &[u32],
    probs: &[f32],
    vocab: usize,
) -> Result<Vec<(usize, f32)>> {
    if token_ids.len() != probs.len() || token_ids.is_empty() {
        candle_core::bail!(
            "invalid sparse speculative probability row {row}: ids={}, probabilities={}",
            token_ids.len(),
            probs.len()
        );
    }

    let mut sum = 0.0f64;
    for (&token, &prob) in token_ids.iter().zip(probs) {
        if token as usize >= vocab {
            candle_core::bail!(
                "sparse speculative token id {token} at position {row} exceeds vocab {vocab}"
            );
        }
        if !prob.is_finite() || prob < 0.0 {
            candle_core::bail!("invalid sparse speculative probability {prob} at position {row}");
        }
        sum += prob as f64;
    }
    if !sum.is_finite() || sum <= 0.0 {
        candle_core::bail!(
            "sparse speculative probabilities sum to an invalid value at position {row}"
        );
    }

    let mut entries = Vec::<(usize, f32)>::with_capacity(token_ids.len());
    for (&token, &prob) in token_ids.iter().zip(probs) {
        if prob == 0.0 {
            continue;
        }
        let normalized = (prob as f64 / sum) as f32;
        let token = token as usize;
        if let Some((_, existing)) = entries
            .iter_mut()
            .find(|(candidate, _)| *candidate == token)
        {
            *existing += normalized;
        } else {
            entries.push((token, normalized));
        }
    }
    Ok(entries)
}

#[allow(clippy::too_many_arguments)]
async fn finish_verified_step_stochastic<P: Pipeline>(
    pipeline: &P,
    seq: &mut Sequence,
    verify_logits: Tensor,
    proposal: Vec<u32>,
    proposal_distribution: Option<SpeculativeProposalDistribution>,
    base_len: usize,
    prefix_cacher: &mut PrefixCacheManagerV2,
    eos_tok: Option<&[u32]>,
    return_logprobs: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    fallback_uniforms: Option<&SparseRejectionFallbackUniforms>,
) -> Result<VerificationOutcome> {
    let proposal_distribution = match proposal_distribution {
        Some(distribution) => PreparedProposalDistribution::new(distribution, proposal.len())?,
        None => PreparedProposalDistribution::Deterministic,
    };
    let mut accepted = 0usize;
    for (idx, draft) in proposal.iter().copied().enumerate() {
        let target_row = logit_row(&verify_logits, idx)?;
        let sampler = seq.sampler();
        let target_probs = sampler.speculative_target_probs(
            flat_logits(target_row.clone())?,
            seq.get_toks(),
            seq.prompt_tokens(),
        )?;
        let candidate_probs = proposal_distribution.probability_row(
            idx,
            draft,
            &sampler,
            seq.get_toks(),
            seq.prompt_tokens(),
            target_probs.sampling.len(),
        )?;
        let draft_idx = draft as usize;
        let p_i = target_probs.sampling.get(draft_idx).copied().unwrap_or(0.0);
        let q_i = candidate_probs.probability(draft_idx);
        if candidate_probs.is_sparse() && q_i <= 0.0 {
            candle_core::bail!(
                "sparse speculative proposal token {draft} has zero probability at position {idx}"
            );
        }
        let accept_prob = if q_i <= 0.0 {
            if p_i > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            (p_i / q_i).min(1.0)
        };
        let draw = match fallback_uniforms {
            Some(uniforms) => uniforms.accept[idx],
            None => {
                let mut rng = rng.lock().expect("could not lock rng mutex");
                rng.random::<f32>()
            }
        };

        if accepts_draft(draw, accept_prob) {
            accepted += 1;
            let sampled =
                sampler.logprobs_from_probs(draft, &target_probs.reporting, return_logprobs)?;
            finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
            if matches!(seq.getstate(), SequenceState::Done(_)) {
                let keep_len = base_len + 1 + accepted;
                seq.clear_staged_speculative_tokens();
                return Ok(VerificationOutcome {
                    accepted_drafts: accepted,
                    proposed_drafts: proposal.len(),
                    keep_len,
                    continuation_token: None,
                });
            }
            continue;
        }

        let mut adjusted_probs = target_probs.sampling.clone();
        candidate_probs.subtract_from(&mut adjusted_probs)?;
        for prob in &mut adjusted_probs {
            *prob = prob.max(0.0);
        }
        if normalize_probs(&mut adjusted_probs).is_err() {
            adjusted_probs = target_probs.sampling;
        }
        let sampled = match fallback_uniforms {
            Some(uniforms) => sample_from_probs_with_uniform(
                &sampler,
                &adjusted_probs,
                &target_probs.reporting,
                return_logprobs,
                uniforms.sample,
            )?,
            None => sampler.sample_from_probs(
                &adjusted_probs,
                &target_probs.reporting,
                return_logprobs,
                rng.clone(),
            )?,
        };
        let sampled_token = sampled.token;
        let keep_len = base_len + 1 + accepted;
        finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
        if matches!(seq.getstate(), SequenceState::Done(_)) {
            seq.clear_staged_speculative_tokens();
            return Ok(VerificationOutcome {
                accepted_drafts: accepted,
                proposed_drafts: proposal.len(),
                keep_len,
                continuation_token: None,
            });
        }
        return Ok(VerificationOutcome {
            accepted_drafts: accepted,
            proposed_drafts: proposal.len(),
            keep_len,
            continuation_token: Some(sampled_token),
        });
    }

    let row = logit_row(&verify_logits, accepted)?;
    let sampler = seq.sampler();
    let target_probs = sampler.speculative_target_probs(
        flat_logits(row.clone())?,
        seq.get_toks(),
        seq.prompt_tokens(),
    )?;
    let continuation = match fallback_uniforms {
        Some(uniforms) => sample_from_probs_with_uniform(
            &sampler,
            &target_probs.sampling,
            &target_probs.reporting,
            return_logprobs,
            uniforms.sample,
        )?,
        None => sampler.sample_from_probs(
            &target_probs.sampling,
            &target_probs.reporting,
            return_logprobs,
            rng,
        )?,
    };
    let continuation_token = continuation.token;
    finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, continuation, eos_tok, true).await?;

    let keep_len = base_len + 1 + accepted;
    let continuation_token = if matches!(seq.getstate(), SequenceState::Done(_)) {
        seq.clear_staged_speculative_tokens();
        None
    } else {
        Some(continuation_token)
    };

    Ok(VerificationOutcome {
        accepted_drafts: accepted,
        proposed_drafts: proposal.len(),
        keep_len,
        continuation_token,
    })
}

fn logit_row(logits: &Tensor, row: usize) -> Result<Tensor> {
    match logits.dims() {
        [_, rows, _] => {
            if row >= *rows {
                candle_core::bail!("speculative logit row {row} is out of range for {rows} rows");
            }
            logits.narrow(1, row, 1)
        }
        [rows, _] => {
            if row >= *rows {
                candle_core::bail!("speculative logit row {row} is out of range for {rows} rows");
            }
            logits.narrow(0, row, 1)
        }
        shape => candle_core::bail!("speculative logits have unsupported shape {shape:?}"),
    }
}

fn flat_logits(logits: Tensor) -> Result<Tensor> {
    match logits.dims() {
        [1, 1, _] => logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32),
        [1, _] => logits.squeeze(0)?.to_dtype(DType::F32),
        [_] => logits.to_dtype(DType::F32),
        dims => candle_core::bail!("speculative logit row must flatten to rank 1, got {dims:?}"),
    }
}

fn normalize_probs(probs: &mut [f32]) -> Result<()> {
    let sum: f32 = probs
        .iter()
        .copied()
        .filter(|prob| prob.is_finite() && *prob > 0.0)
        .sum();
    if sum <= 0.0 {
        candle_core::bail!("all probabilities are zero in speculative adjusted distribution");
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

fn sample_from_probs_with_uniform(
    sampler: &Sampler,
    sampling_probs: &[f32],
    reporting_probs: &[f32],
    return_logprobs: bool,
    uniform: f32,
) -> Result<Logprobs> {
    if !valid_uniform(uniform) {
        candle_core::bail!("invalid speculative fallback sampling uniform {uniform}");
    }
    let mut total = 0.0f32;
    let mut last_positive = None;
    for (token, probability) in sampling_probs.iter().copied().enumerate() {
        if !probability.is_finite() || probability < 0.0 {
            candle_core::bail!(
                "invalid speculative fallback sampling probability {probability} at token {token}"
            );
        }
        if probability > 0.0 {
            last_positive = Some(token);
        }
        total += probability;
    }
    if !total.is_finite() || total <= 0.0 {
        candle_core::bail!("all speculative fallback sampling probabilities are zero");
    }
    let target = uniform * total;
    let mut cumulative = 0.0f32;
    let token = sampling_probs
        .iter()
        .copied()
        .enumerate()
        .find_map(|(token, probability)| {
            cumulative += probability;
            (target < cumulative).then_some(token)
        })
        .or(last_positive)
        .expect("positive fallback probability was validated");
    sampler.logprobs_from_probs(token as u32, reporting_probs, return_logprobs)
}

fn valid_uniform(uniform: f32) -> bool {
    uniform.is_finite() && (0.0..1.0).contains(&uniform)
}

fn accepts_draft(draw: f32, accept_prob: f32) -> bool {
    draw < accept_prob
}

#[cfg(test)]
mod tests {
    use super::{
        accepts_draft, normalize_probs, normalize_sparse_row, packed_target_shape_matches,
        partition_device_tokens, stochastic_verification_allowed, validate_device_verification,
        DeviceVerification, ProposalProbabilityRow, SparseRejectionFallbackUniforms,
    };

    #[test]
    fn packed_sparse_target_requires_the_complete_homogeneous_shape() {
        assert!(packed_target_shape_matches(
            &[16, 8, 248_320],
            16,
            7,
            248_320
        ));
        assert!(!packed_target_shape_matches(
            &[8, 8, 248_320],
            16,
            7,
            248_320
        ));
        assert!(!packed_target_shape_matches(
            &[16, 7, 248_320],
            16,
            7,
            248_320
        ));
        assert!(!packed_target_shape_matches(
            &[16, 8, 32_000],
            16,
            7,
            248_320
        ));
        assert!(!packed_target_shape_matches(
            &[1, 16, 8, 248_320],
            16,
            7,
            248_320
        ));
    }

    #[test]
    fn packed_sparse_target_rejects_partial_or_reordered_groups() {
        use super::is_complete_ordered_sparse_group;

        assert!(is_complete_ordered_sparse_group(
            4,
            [0, 1, 2, 3].into_iter(),
            [0, 1, 2, 3].into_iter(),
        ));
        assert!(!is_complete_ordered_sparse_group(
            4,
            [0, 1, 3].into_iter(),
            [0, 1, 2].into_iter(),
        ));
        assert!(!is_complete_ordered_sparse_group(
            4,
            [0, 1, 2, 3].into_iter(),
            [0, 2, 1, 3].into_iter(),
        ));
    }

    #[test]
    fn stochastic_verification_excludes_argmax_constraints_and_tools() {
        assert!(stochastic_verification_allowed(false, false, false));
        assert!(!stochastic_verification_allowed(false, false, true));
        assert!(!stochastic_verification_allowed(false, true, false));
        assert!(!stochastic_verification_allowed(true, false, false));
    }

    #[test]
    fn zero_probability_drafts_are_never_accepted() {
        assert!(!accepts_draft(0.0, 0.0));
        assert!(accepts_draft(0.0, 1.0));
        assert!(accepts_draft(0.999, 1.0));
    }

    #[test]
    fn partitions_batched_device_tokens_in_sequence_order() {
        let tokens = partition_device_tokens(vec![1, 2, 3, 4, 5, 6], &[2, 1, 3]).unwrap();
        assert_eq!(tokens, vec![vec![1, 2], vec![3], vec![4, 5, 6]]);
        assert!(partition_device_tokens(vec![1, 2], &[1, 2]).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sparse_cuda_fallback_reuses_uniforms_in_input_order() -> candle_core::Result<()> {
        use super::{decode_sparse_rejection_completion, SparseRejectionPendingCandidate};
        use crate::cuda::speculative_rejection::{
            SparseRejectionCompletion, SparseRejectionRow, SPARSE_REJECTION_INVALID_VALUE,
            SPARSE_REJECTION_STATUS_NEEDS_CPU, SPARSE_REJECTION_STATUS_OK,
        };

        let completion = SparseRejectionCompletion {
            rows: vec![
                SparseRejectionRow {
                    accepted_count: 1,
                    continuation: 8,
                    status: SPARSE_REJECTION_STATUS_OK,
                },
                SparseRejectionRow {
                    accepted_count: SPARSE_REJECTION_INVALID_VALUE,
                    continuation: SPARSE_REJECTION_INVALID_VALUE,
                    status: SPARSE_REJECTION_STATUS_NEEDS_CPU,
                },
                SparseRejectionRow {
                    accepted_count: 2,
                    continuation: 9,
                    status: SPARSE_REJECTION_STATUS_OK,
                },
            ],
            draft_tokens: vec![vec![21, 22], vec![1, 2], vec![11, 12]],
        };
        let output = decode_sparse_rejection_completion(
            3,
            vec![
                SparseRejectionPendingCandidate {
                    input_idx: 2,
                    drafts: 2,
                    vocab: 10,
                    accept_uniforms: vec![0.21, 0.22],
                    sample_uniform: 0.23,
                },
                SparseRejectionPendingCandidate {
                    input_idx: 0,
                    drafts: 2,
                    vocab: 10,
                    accept_uniforms: vec![0.01, 0.02],
                    sample_uniform: 0.03,
                },
                SparseRejectionPendingCandidate {
                    input_idx: 1,
                    drafts: 2,
                    vocab: 10,
                    accept_uniforms: vec![0.11, 0.12],
                    sample_uniform: 0.13,
                },
            ],
            completion,
        )?;

        let Some(DeviceVerification::SparseRejectionCpuFallback(uniforms)) =
            output.verifications[0].as_ref()
        else {
            panic!("input zero must retain its CPU fallback draws")
        };
        assert_eq!(uniforms.accept, [0.01, 0.02]);
        assert_eq!(uniforms.sample, 0.03);
        assert!(matches!(
            output.verifications[1],
            Some(DeviceVerification::SparseRejection {
                accepted_drafts: 2,
                continuation_token: 9,
            })
        ));
        assert!(matches!(
            output.verifications[2],
            Some(DeviceVerification::SparseRejection {
                accepted_drafts: 1,
                continuation_token: 8,
            })
        ));
        assert_eq!(
            output.materialized_proposals,
            vec![Some(vec![1, 2]), Some(vec![11, 12]), Some(vec![21, 22])]
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sparse_cuda_eligibility_rejects_invalid_inputs() -> candle_core::Result<()> {
        use super::sparse_distribution_is_cuda_eligible;
        use crate::speculative::SparseSpeculativeProbs;
        use candle_core::{Device, Tensor};

        let cuda = Device::new_cuda(0)?;
        let token_ids = Tensor::from_vec(vec![0u32; 8], (2, 4), &cuda)?;
        let probs = Tensor::from_vec(vec![0.125f32; 8], (2, 4), &cuda)?;
        let valid = SparseSpeculativeProbs::new(token_ids.clone(), probs.clone())?;
        assert!(sparse_distribution_is_cuda_eligible(&valid, 2));
        assert!(!sparse_distribution_is_cuda_eligible(&valid, 1));

        let cpu = Device::Cpu;
        let cpu_sparse = SparseSpeculativeProbs::new(
            Tensor::from_vec(vec![0u32; 8], (2, 4), &cpu)?,
            Tensor::from_vec(vec![0.125f32; 8], (2, 4), &cpu)?,
        )?;
        assert!(!sparse_distribution_is_cuda_eligible(&cpu_sparse, 2));

        let split_device = SparseSpeculativeProbs::new(
            token_ids,
            Tensor::from_vec(vec![0.125f32; 8], (2, 4), &cpu)?,
        )?;
        assert!(!sparse_distribution_is_cuda_eligible(&split_device, 2));

        let too_wide = SparseSpeculativeProbs::new(
            Tensor::from_vec(vec![0u32; 2 * 129], (2, 129), &cuda)?,
            Tensor::from_vec(vec![1.0f32; 2 * 129], (2, 129), &cuda)?,
        )?;
        assert!(!sparse_distribution_is_cuda_eligible(&too_wide, 2));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sparse_cuda_eligibility_allows_noncontiguous_inputs() -> candle_core::Result<()> {
        use super::sparse_distribution_is_cuda_eligible;
        use crate::speculative::SparseSpeculativeProbs;
        use candle_core::{Device, Tensor};

        let cuda = Device::new_cuda(0)?;
        let token_ids = Tensor::from_vec(vec![0u32; 8], (4, 2), &cuda)?.transpose(0, 1)?;
        let probs = Tensor::from_vec(vec![0.125f32; 8], (4, 2), &cuda)?.transpose(0, 1)?;
        let sparse = SparseSpeculativeProbs::new(token_ids, probs)?;
        assert!(!sparse.token_ids().is_contiguous());
        assert!(!sparse.probs().is_contiguous());
        assert!(sparse_distribution_is_cuda_eligible(&sparse, 2));
        Ok(())
    }

    #[test]
    fn device_verification_requires_a_continuation_token() {
        assert!(validate_device_verification(None, 7).is_ok());
        assert!(validate_device_verification(
            Some(&DeviceVerification::TargetTokens(vec![0; 8])),
            7
        )
        .is_ok());
        assert!(validate_device_verification(
            Some(&DeviceVerification::TargetTokens(vec![0; 7])),
            7
        )
        .is_err());
        assert!(validate_device_verification(
            Some(&DeviceVerification::SparseRejection {
                accepted_drafts: 7,
                continuation_token: 0,
            }),
            7
        )
        .is_ok());
        assert!(validate_device_verification(
            Some(&DeviceVerification::SparseRejection {
                accepted_drafts: 8,
                continuation_token: 0,
            }),
            7
        )
        .is_err());
        assert!(validate_device_verification(
            Some(&DeviceVerification::SparseRejectionCpuFallback(
                SparseRejectionFallbackUniforms {
                    accept: vec![0.25; 7],
                    sample: 0.5,
                },
            )),
            7,
        )
        .is_ok());
        assert!(validate_device_verification(
            Some(&DeviceVerification::SparseRejectionCpuFallback(
                SparseRejectionFallbackUniforms {
                    accept: vec![0.25; 6],
                    sample: 0.5,
                },
            )),
            7,
        )
        .is_err());
        assert!(validate_device_verification(
            Some(&DeviceVerification::SparseRejectionCpuFallback(
                SparseRejectionFallbackUniforms {
                    accept: vec![0.25; 7],
                    sample: 1.0,
                },
            )),
            7,
        )
        .is_err());
    }

    #[test]
    fn sparse_probability_rows_normalize_and_merge_duplicate_tokens() {
        let row = normalize_sparse_row(0, &[1, 3, 3], &[1.0, 1.0, 2.0], 4).unwrap();
        let row = ProposalProbabilityRow::Sparse(row);
        assert!((row.probability(1) - 0.25).abs() < f32::EPSILON);
        assert!((row.probability(3) - 0.75).abs() < f32::EPSILON);
        assert_eq!(row.probability(2), 0.0);

        let mut target = vec![0.1, 0.4, 0.2, 0.3];
        row.subtract_from(&mut target).unwrap();
        assert!((target[0] - 0.1).abs() < f32::EPSILON);
        assert!((target[1] - 0.15).abs() < f32::EPSILON);
        assert!((target[2] - 0.2).abs() < f32::EPSILON);
        assert!((target[3] + 0.45).abs() < f32::EPSILON);

        for prob in &mut target {
            *prob = prob.max(0.0);
        }
        normalize_probs(&mut target).unwrap();
        assert!((target[0] - 2.0 / 9.0).abs() < 1e-6);
        assert!((target[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((target[2] - 4.0 / 9.0).abs() < 1e-6);
        assert_eq!(target[3], 0.0);
    }

    #[test]
    fn sparse_probability_rows_reject_invalid_distributions() {
        assert!(normalize_sparse_row(0, &[1], &[], 2).is_err());
        assert!(normalize_sparse_row(0, &[1], &[0.0], 2).is_err());
        assert!(normalize_sparse_row(0, &[1], &[-0.1], 2).is_err());
        assert!(normalize_sparse_row(0, &[1], &[f32::NAN], 2).is_err());
        assert!(normalize_sparse_row(0, &[2], &[1.0], 2).is_err());
    }
}
