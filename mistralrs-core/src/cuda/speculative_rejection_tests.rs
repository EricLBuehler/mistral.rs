use super::*;
use candle_core::Device;

struct HostCase {
    batch: usize,
    drafts: usize,
    vocab: usize,
    q_width: usize,
    target_logits: Vec<f32>,
    draft_tokens: Vec<u32>,
    q_token_ids: Vec<u32>,
    q_probs: Vec<f32>,
    inverse_temperatures: Vec<f32>,
    target_top_k: Vec<u32>,
    top_p: Vec<f32>,
    min_p: Vec<f32>,
    accept_uniforms: Vec<f32>,
    sample_uniforms: Vec<f32>,
    mode: SparseRejectionMode,
}

struct DeviceCase {
    target_logits: Tensor,
    draft_tokens: Tensor,
    q_token_ids: Tensor,
    q_probs: Tensor,
    inverse_temperatures: Tensor,
    target_top_k: Tensor,
    top_p: Tensor,
    min_p: Tensor,
    accept_uniforms: Tensor,
    sample_uniforms: Tensor,
}

impl DeviceCase {
    fn run_with_target(
        &self,
        target_logits: &Tensor,
        mode: SparseRejectionMode,
    ) -> Result<Vec<SparseRejectionRow>> {
        sparse_rejection_cuda(SparseRejectionInput {
            target_logits,
            draft_tokens: &self.draft_tokens,
            q_token_ids: &self.q_token_ids,
            q_probs: &self.q_probs,
            inverse_temperatures: &self.inverse_temperatures,
            target_top_k: &self.target_top_k,
            top_p: &self.top_p,
            min_p: &self.min_p,
            accept_uniforms: &self.accept_uniforms,
            sample_uniforms: &self.sample_uniforms,
            mode,
        })?
        .to_rows()
    }
}

impl HostCase {
    fn to_device(&self, device: &Device) -> Result<DeviceCase> {
        let rows = self.drafts + 1;
        Ok(DeviceCase {
            target_logits: Tensor::from_vec(
                self.target_logits.clone(),
                (self.batch, rows, self.vocab),
                device,
            )?,
            draft_tokens: Tensor::from_vec(
                self.draft_tokens.clone(),
                (self.batch, self.drafts),
                device,
            )?,
            q_token_ids: Tensor::from_vec(
                self.q_token_ids.clone(),
                (self.batch, self.drafts, self.q_width),
                device,
            )?,
            q_probs: Tensor::from_vec(
                self.q_probs.clone(),
                (self.batch, self.drafts, self.q_width),
                device,
            )?,
            inverse_temperatures: Tensor::from_vec(
                self.inverse_temperatures.clone(),
                self.batch,
                device,
            )?,
            target_top_k: Tensor::from_vec(self.target_top_k.clone(), self.batch, device)?,
            top_p: Tensor::from_vec(self.top_p.clone(), self.batch, device)?,
            min_p: Tensor::from_vec(self.min_p.clone(), self.batch, device)?,
            accept_uniforms: Tensor::from_vec(
                self.accept_uniforms.clone(),
                (self.batch, self.drafts),
                device,
            )?,
            sample_uniforms: Tensor::from_vec(self.sample_uniforms.clone(), self.batch, device)?,
        })
    }

    fn run(&self, device: &Device) -> Result<Vec<SparseRejectionRow>> {
        let tensors = self.to_device(device)?;
        tensors.run_with_target(&tensors.target_logits, self.mode)
    }

    fn run_workspace(
        &self,
        device: &Device,
        workspace: &mut Option<CudaSparseRejectionWorkspace>,
        deterministic: bool,
    ) -> Result<Vec<SparseRejectionRow>> {
        let tensors = self.to_device(device)?;
        let proposal = if deterministic {
            SparseRejectionProposalInput::Deterministic
        } else {
            SparseRejectionProposalInput::Sparse {
                token_ids: &tensors.q_token_ids,
                probs: &tensors.q_probs,
            }
        };
        let submission = sparse_rejection_cuda_submit(
            SparseRejectionWorkspaceInput {
                target_logits: &tensors.target_logits,
                proposal,
                draft_tokens: SparseRejectionDraftInput::Host(&self.draft_tokens),
                inverse_temperatures: &self.inverse_temperatures,
                target_top_k: &self.target_top_k,
                top_p: &self.top_p,
                min_p: &self.min_p,
                accept_uniforms: &self.accept_uniforms,
                sample_uniforms: &self.sample_uniforms,
                mode: self.mode,
            },
            workspace,
        )?;
        Ok(sparse_rejection_cuda_complete(workspace, &submission)?.rows)
    }
}

fn active_filter(value: f32) -> bool {
    value > 0.0 && value < 1.0
}

fn target_distribution(case: &HostCase, sequence: usize, row: usize) -> Vec<f32> {
    let offset = (sequence * (case.drafts + 1) + row) * case.vocab;
    let logits = &case.target_logits[offset..offset + case.vocab];
    let inverse_temperature = case.inverse_temperatures[sequence];
    let scaled_max = logits
        .iter()
        .map(|value| value * inverse_temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = logits
        .iter()
        .map(|value| (value * inverse_temperature - scaled_max).exp())
        .collect::<Vec<_>>();

    if matches!(case.mode, SparseRejectionMode::BoundedTopK { .. }) {
        let mut sorted = (0..case.vocab).collect::<Vec<_>>();
        sorted.sort_by(|left, right| {
            logits[*right]
                .total_cmp(&logits[*left])
                .then_with(|| left.cmp(right))
        });
        let count = (case.target_top_k[sequence] as usize).min(case.vocab);
        for token in sorted.iter().skip(count) {
            weights[*token] = 0.0;
        }
        let kept = &sorted[..count];
        if active_filter(case.top_p[sequence]) {
            let cutoff =
                case.top_p[sequence] * kept.iter().map(|token| weights[*token]).sum::<f32>();
            let mut cumulative = 0.0f32;
            for token in kept {
                if cumulative >= cutoff {
                    weights[*token] = 0.0;
                } else {
                    cumulative += weights[*token];
                }
            }
        }
        if active_filter(case.min_p[sequence]) {
            let threshold = case.min_p[sequence];
            for token in kept {
                if threshold >= weights[*token] {
                    weights[*token] = 0.0;
                }
            }
        }
    }

    let denominator = weights.iter().sum::<f32>();
    for weight in &mut weights {
        *weight /= denominator;
    }
    weights
}

fn q_distribution(case: &HostCase, sequence: usize, row: usize) -> Vec<f32> {
    let offset = (sequence * case.drafts + row) * case.q_width;
    let ids = &case.q_token_ids[offset..offset + case.q_width];
    let probabilities = &case.q_probs[offset..offset + case.q_width];
    let denominator = probabilities
        .iter()
        .map(|probability| *probability as f64)
        .sum::<f64>();
    let mut normalized = vec![0.0f32; case.vocab];
    for (&id, &probability) in ids.iter().zip(probabilities) {
        normalized[id as usize] += (probability as f64 / denominator) as f32;
    }
    normalized
}

fn sample_distribution(probabilities: &[f32], uniform: f32) -> u32 {
    let total = probabilities.iter().sum::<f32>();
    let target = uniform * total;
    let mut cumulative = 0.0f32;
    for (token, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if target < cumulative {
            return token as u32;
        }
    }
    probabilities
        .iter()
        .rposition(|probability| *probability > 0.0)
        .expect("reference distribution has positive mass") as u32
}

fn reference(case: &HostCase) -> Vec<SparseRejectionRow> {
    (0..case.batch)
        .map(|sequence| {
            let mut accepted_count = 0u32;
            for row in 0..case.drafts {
                let target = target_distribution(case, sequence, row);
                let q = q_distribution(case, sequence, row);
                let draft = case.draft_tokens[sequence * case.drafts + row] as usize;
                let accept_probability = (target[draft] / q[draft]).min(1.0);
                let uniform = case.accept_uniforms[sequence * case.drafts + row];
                if uniform < accept_probability {
                    accepted_count += 1;
                    continue;
                }
                let residual = target
                    .iter()
                    .zip(q)
                    .map(|(target, q)| (target - q).max(0.0))
                    .collect::<Vec<_>>();
                let continuation = if residual.iter().sum::<f32>() > 0.0 {
                    sample_distribution(&residual, case.sample_uniforms[sequence])
                } else {
                    sample_distribution(&target, case.sample_uniforms[sequence])
                };
                return SparseRejectionRow {
                    accepted_count,
                    continuation,
                    status: SPARSE_REJECTION_STATUS_OK,
                };
            }
            let target = target_distribution(case, sequence, case.drafts);
            SparseRejectionRow {
                accepted_count,
                continuation: sample_distribution(&target, case.sample_uniforms[sequence]),
                status: SPARSE_REJECTION_STATUS_OK,
            }
        })
        .collect()
}

fn logits(probability_rows: &[[f32; 5]]) -> Vec<f32> {
    probability_rows
        .iter()
        .flat_map(|row| row.iter().map(|probability| probability.ln()))
        .collect()
}

fn deterministic_workspace_case(batch: usize, drafts: usize) -> HostCase {
    let vocab = 5;
    let draft_tokens = (0..batch * drafts)
        .map(|idx| (idx % vocab) as u32)
        .collect::<Vec<_>>();
    let target_logits = (0..batch * (drafts + 1))
        .flat_map(|row| {
            (0..vocab).map(move |token| ((row * 3 + token * 2) % 11) as f32 * 0.25 - 1.0)
        })
        .collect::<Vec<_>>();
    HostCase {
        batch,
        drafts,
        vocab,
        q_width: 1,
        target_logits,
        q_token_ids: draft_tokens.clone(),
        q_probs: vec![1.0; batch * drafts],
        draft_tokens,
        inverse_temperatures: vec![1.0; batch],
        target_top_k: vec![0; batch],
        top_p: vec![0.0; batch],
        min_p: vec![0.0; batch],
        accept_uniforms: (0..batch * drafts)
            .map(|idx| 0.1 + 0.8 * (idx % 7) as f32 / 7.0)
            .collect(),
        sample_uniforms: (0..batch)
            .map(|idx| 0.2 + 0.6 * (idx % 5) as f32 / 5.0)
            .collect(),
        mode: SparseRejectionMode::Categorical,
    }
}

#[test]
fn dense_rows_require_shared_ordered_tight_backing() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let packed = Tensor::from_vec((0..24u32).collect(), (4, 3, 2), &device)?;
    let rows = (0..4)
        .map(|row| packed.get(row))
        .collect::<Result<Vec<_>>>()?;
    let dense = dense_cuda_rows(&rows, DType::U32, &[3, 2], &device)?
        .expect("packed row views must remain dense");
    assert_eq!(dense.shape.dims(), [4, 3, 2]);

    let reordered = vec![
        rows[1].clone(),
        rows[0].clone(),
        rows[2].clone(),
        rows[3].clone(),
    ];
    assert!(dense_cuda_rows(&reordered, DType::U32, &[3, 2], &device)?.is_none());
    assert!(matches!(
        SparseRejectionDeviceTensor::from_rows(&reordered, DType::U32, &[3, 2], &device)?,
        SparseRejectionDeviceTensor::Tensor(_)
    ));

    let gapped = vec![rows[0].clone(), rows[2].clone()];
    assert!(dense_cuda_rows(&gapped, DType::U32, &[3, 2], &device)?.is_none());

    let separate = (0..4)
        .map(|row| Tensor::from_vec(vec![row as u32; 6], (3, 2), &device))
        .collect::<Result<Vec<_>>>()?;
    assert!(dense_cuda_rows(&separate, DType::U32, &[3, 2], &device)?.is_none());

    let transposed = rows
        .iter()
        .map(|row| row.transpose(0, 1))
        .collect::<Result<Vec<_>>>()?;
    assert!(dense_cuda_rows(&transposed, DType::U32, &[2, 3], &device)?.is_none());
    Ok(())
}

#[test]
fn dense_rows_match_packed_inputs_and_survive_owner_drop() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for bounded_topk in [false, true] {
        let mut case = deterministic_workspace_case(4, 3);
        if bounded_topk {
            case.target_top_k.fill(case.vocab as u32);
            case.mode = SparseRejectionMode::BoundedTopK {
                max_top_k: case.vocab,
            };
        }
        let tensors = case.to_device(&device)?;
        let mut packed_workspace = None;
        let packed_submission = sparse_rejection_cuda_submit(
            SparseRejectionWorkspaceInput {
                target_logits: &tensors.target_logits,
                proposal: SparseRejectionProposalInput::Sparse {
                    token_ids: &tensors.q_token_ids,
                    probs: &tensors.q_probs,
                },
                draft_tokens: SparseRejectionDraftInput::Device(&tensors.draft_tokens),
                inverse_temperatures: &case.inverse_temperatures,
                target_top_k: &case.target_top_k,
                top_p: &case.top_p,
                min_p: &case.min_p,
                accept_uniforms: &case.accept_uniforms,
                sample_uniforms: &case.sample_uniforms,
                mode: case.mode,
            },
            &mut packed_workspace,
        )?;
        let packed = sparse_rejection_cuda_complete(&mut packed_workspace, &packed_submission)?;

        let draft_rows = (0..case.batch)
            .map(|row| tensors.draft_tokens.get(row))
            .collect::<Result<Vec<_>>>()?;
        let q_token_rows = (0..case.batch)
            .map(|row| tensors.q_token_ids.get(row))
            .collect::<Result<Vec<_>>>()?;
        let q_prob_rows = (0..case.batch)
            .map(|row| tensors.q_probs.get(row))
            .collect::<Result<Vec<_>>>()?;
        assert!(dense_cuda_rows(&draft_rows, DType::U32, &[case.drafts], &device)?.is_some());
        assert!(dense_cuda_rows(
            &q_token_rows,
            DType::U32,
            &[case.drafts, case.q_width],
            &device
        )?
        .is_some());
        assert!(dense_cuda_rows(
            &q_prob_rows,
            DType::F32,
            &[case.drafts, case.q_width],
            &device
        )?
        .is_some());

        let mut row_workspace = None;
        let row_submission = sparse_rejection_cuda_submit(
            SparseRejectionWorkspaceInput {
                target_logits: &tensors.target_logits,
                proposal: SparseRejectionProposalInput::SparseRows {
                    token_ids: &q_token_rows,
                    probs: &q_prob_rows,
                },
                draft_tokens: SparseRejectionDraftInput::DeviceRows(&draft_rows),
                inverse_temperatures: &case.inverse_temperatures,
                target_top_k: &case.target_top_k,
                top_p: &case.top_p,
                min_p: &case.min_p,
                accept_uniforms: &case.accept_uniforms,
                sample_uniforms: &case.sample_uniforms,
                mode: case.mode,
            },
            &mut row_workspace,
        )?;
        drop(draft_rows);
        drop(q_token_rows);
        drop(q_prob_rows);
        drop(tensors);
        let rows = sparse_rejection_cuda_complete(&mut row_workspace, &row_submission)?;
        assert_eq!(rows.rows, packed.rows);
        assert_eq!(rows.draft_tokens, packed.draft_tokens);
        assert_eq!(rows.rows, reference(&case));
    }
    Ok(())
}

#[test]
fn workspace_reuses_grows_and_supports_deterministic_proposals() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut workspace = None;
    let first = deterministic_workspace_case(2, 2);
    let expected = reference(&first);
    assert_eq!(
        first.run_workspace(&device, &mut workspace, true)?,
        expected
    );
    let first_workspace = workspace.as_ref().expect("workspace was allocated");
    let first_id = first_workspace.id;
    let first_batch_capacity = first_workspace.capacity_batch;
    let first_draft_capacity = first_workspace.capacity_drafts;

    assert_eq!(
        first.run_workspace(&device, &mut workspace, false)?,
        expected
    );
    let reused = workspace.as_ref().expect("workspace was retained");
    assert_eq!(reused.id, first_id);
    assert_eq!(reused.capacity_batch, first_batch_capacity);
    assert_eq!(reused.capacity_drafts, first_draft_capacity);

    let larger = deterministic_workspace_case(3, 3);
    assert_eq!(
        larger.run_workspace(&device, &mut workspace, true)?,
        reference(&larger)
    );
    let grown = workspace.as_ref().expect("workspace was grown");
    assert_ne!(grown.id, first_id);
    assert!(grown.capacity_batch >= larger.batch);
    assert!(grown.capacity_drafts >= larger.drafts);
    Ok(())
}

#[test]
fn workspace_accepts_device_drafts_and_materializes_with_outcomes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let case = deterministic_workspace_case(3, 4);
    let tensors = case.to_device(&device)?;
    let mut workspace = None;
    let submission = sparse_rejection_cuda_submit(
        SparseRejectionWorkspaceInput {
            target_logits: &tensors.target_logits,
            proposal: SparseRejectionProposalInput::Deterministic,
            draft_tokens: SparseRejectionDraftInput::Device(&tensors.draft_tokens),
            inverse_temperatures: &case.inverse_temperatures,
            target_top_k: &case.target_top_k,
            top_p: &case.top_p,
            min_p: &case.min_p,
            accept_uniforms: &case.accept_uniforms,
            sample_uniforms: &case.sample_uniforms,
            mode: case.mode,
        },
        &mut workspace,
    )?;
    let completion = sparse_rejection_cuda_complete(&mut workspace, &submission)?;
    assert_eq!(completion.rows, reference(&case));
    assert_eq!(
        completion.draft_tokens,
        case.draft_tokens
            .chunks_exact(case.drafts)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn workspace_reuses_bounded_topk_scratch_and_row_temperatures() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut workspace = None;
    let mut case = deterministic_workspace_case(2, 2);
    case.inverse_temperatures = vec![0.5, 1.25];
    case.target_top_k = vec![5; case.batch];
    case.mode = SparseRejectionMode::BoundedTopK { max_top_k: 5 };
    let expected = reference(&case);

    assert_eq!(case.run_workspace(&device, &mut workspace, true)?, expected);
    let topk = workspace
        .as_ref()
        .and_then(|workspace| workspace.topk.as_ref())
        .expect("bounded top-k workspace was allocated") as *const _;

    assert_eq!(
        case.run_workspace(&device, &mut workspace, false)?,
        expected
    );
    let reused = workspace
        .as_ref()
        .and_then(|workspace| workspace.topk.as_ref())
        .expect("bounded top-k workspace was retained") as *const _;
    assert_eq!(topk, reused);
    Ok(())
}

#[test]
fn workspace_rejects_overlap_and_stale_completion() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let case = deterministic_workspace_case(2, 2);
    let tensors = case.to_device(&device)?;
    let input = || SparseRejectionWorkspaceInput {
        target_logits: &tensors.target_logits,
        proposal: SparseRejectionProposalInput::Deterministic,
        draft_tokens: SparseRejectionDraftInput::Host(&case.draft_tokens),
        inverse_temperatures: &case.inverse_temperatures,
        target_top_k: &case.target_top_k,
        top_p: &case.top_p,
        min_p: &case.min_p,
        accept_uniforms: &case.accept_uniforms,
        sample_uniforms: &case.sample_uniforms,
        mode: case.mode,
    };
    let mut workspace = None;
    let submission = sparse_rejection_cuda_submit(input(), &mut workspace)?;
    let overlap = match sparse_rejection_cuda_submit(input(), &mut workspace) {
        Ok(_) => candle_core::bail!("overlapping submissions must fail"),
        Err(error) => error,
    };
    assert!(overlap.to_string().contains("pending submission"));
    sparse_rejection_cuda_complete(&mut workspace, &submission)?;
    let next = sparse_rejection_cuda_submit(input(), &mut workspace)?;
    let stale = sparse_rejection_cuda_complete(&mut workspace, &submission)
        .expect_err("an old generation must not complete a new submission");
    assert!(stale.to_string().contains("stale submission"));
    sparse_rejection_cuda_complete(&mut workspace, &next)?;
    let inactive = sparse_rejection_cuda_complete(&mut workspace, &next)
        .expect_err("a submission must complete once");
    assert!(inactive.to_string().contains("inactive submission"));
    Ok(())
}

#[test]
fn categorical_matches_cpu_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let target_logits = logits(&[
        [0.05, 0.20, 0.20, 0.30, 0.25],
        [0.10, 0.20, 0.30, 0.20, 0.20],
        [0.20, 0.20, 0.20, 0.20, 0.20],
        [0.55, 0.10, 0.10, 0.15, 0.10],
        [0.10, 0.03, 0.40, 0.25, 0.22],
        [0.10, 0.20, 0.30, 0.25, 0.15],
        [0.10, 0.10, 0.60, 0.10, 0.10],
        [0.10, 0.10, 0.10, 0.60, 0.10],
        [0.05, 0.10, 0.15, 0.20, 0.50],
    ]);
    let case = HostCase {
        batch: 3,
        drafts: 2,
        vocab: 5,
        q_width: 3,
        target_logits,
        draft_tokens: vec![0, 4, 0, 1, 2, 3],
        q_token_ids: vec![0, 1, 2, 4, 1, 2, 0, 1, 2, 1, 2, 3, 2, 0, 4, 3, 1, 4],
        q_probs: vec![
            0.90, 0.05, 0.05, 0.50, 0.25, 0.25, 0.20, 0.40, 0.40, 0.80, 0.10, 0.10, 0.40, 0.30,
            0.30, 0.50, 0.25, 0.25,
        ],
        inverse_temperatures: vec![1.0, 1.0, 1.0],
        target_top_k: vec![0, 0, 0],
        top_p: vec![0.0, 0.0, 0.0],
        min_p: vec![0.0, 0.0, 0.0],
        accept_uniforms: vec![0.80, 0.20, 0.90, 0.70, 0.99, 0.99],
        sample_uniforms: vec![0.67, 0.45, 0.76],
        mode: SparseRejectionMode::Categorical,
    };
    let expected = reference(&case);
    assert_eq!(
        expected
            .iter()
            .map(|row| row.accepted_count)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(case.run(&device)?, expected);
    Ok(())
}

#[test]
fn bounded_topk_matches_cpu_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let case = HostCase {
        batch: 2,
        drafts: 2,
        vocab: 7,
        q_width: 4,
        target_logits: [
            [0.0, 1.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 0.0, 7.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 5.0],
            [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]
        .concat(),
        draft_tokens: vec![2, 2, 0, 1],
        q_token_ids: vec![2, 0, 1, 3, 2, 4, 5, 6, 0, 1, 2, 3, 1, 6, 0, 2],
        q_probs: vec![
            0.10, 0.30, 0.30, 0.30, 0.10, 0.30, 0.30, 0.30, 0.20, 0.30, 0.30, 0.20, 0.70, 0.10,
            0.10, 0.10,
        ],
        inverse_temperatures: vec![1.0, 1.0],
        target_top_k: vec![1, 4],
        top_p: vec![0.0, 0.70],
        min_p: vec![0.0, 0.15],
        accept_uniforms: vec![0.99, 0.99, 0.90, 0.50],
        sample_uniforms: vec![0.30, 0.80],
        mode: SparseRejectionMode::BoundedTopK { max_top_k: 4 },
    };
    let expected = reference(&case);
    assert_eq!(
        expected
            .iter()
            .map(|row| row.accepted_count)
            .collect::<Vec<_>>(),
        vec![2, 1]
    );
    assert_eq!(case.run(&device)?, expected);

    let tensors = case.to_device(&device)?;
    for dtype in [DType::BF16, DType::F16] {
        let low_precision = tensors.target_logits.to_dtype(dtype)?.contiguous()?;
        let reference = low_precision.to_dtype(DType::F32)?.contiguous()?;
        let expected = tensors.run_with_target(&reference, case.mode)?;
        let actual = tensors.run_with_target(&low_precision, case.mode)?;
        assert_eq!(actual, expected);
    }

    let bf16 = tensors.target_logits.to_dtype(DType::BF16)?.contiguous()?;
    let error = match tensors.run_with_target(&bf16, SparseRejectionMode::Categorical) {
        Ok(_) => candle_core::bail!("categorical sparse rejection accepted BF16 logits"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("categorical mode requires F32 target logits"));
    Ok(())
}

#[test]
fn ranked_topk_20_32_matches_cpu_reference_with_ties_and_top_p() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for max_top_k in [20, 32] {
        let batch = 2;
        let drafts = 2;
        let vocab = 64;
        let draft_tokens = vec![0, 7, 13, 31];
        let case = HostCase {
            batch,
            drafts,
            vocab,
            q_width: 1,
            target_logits: (0..batch * (drafts + 1))
                .flat_map(|row| {
                    (0..vocab).map(move |token| ((token * 7 + row * 3) % 13) as f32 * 0.2 - 1.0)
                })
                .collect(),
            draft_tokens: draft_tokens.clone(),
            q_token_ids: draft_tokens,
            q_probs: vec![1.0; batch * drafts],
            inverse_temperatures: vec![0.8, 1.2],
            target_top_k: vec![max_top_k as u32; batch],
            top_p: vec![0.55, 0.72],
            min_p: vec![0.05, 0.10],
            accept_uniforms: vec![0.99; batch * drafts],
            sample_uniforms: vec![0.27, 0.83],
            mode: SparseRejectionMode::BoundedTopK { max_top_k },
        };
        let expected = reference(&case);
        assert_eq!(case.run(&device)?, expected);

        let mut workspace = None;
        assert_eq!(case.run_workspace(&device, &mut workspace, true)?, expected);
    }
    Ok(())
}

#[test]
fn reports_per_row_fallback_and_validation_status() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut q_probs = vec![0.5f32; 8];
    q_probs[2] = -1.0;
    q_probs[3] = 2.0;
    let case = HostCase {
        batch: 4,
        drafts: 1,
        vocab: 3,
        q_width: 2,
        target_logits: vec![0.0; 24],
        draft_tokens: vec![0; 4],
        q_token_ids: vec![0, 1, 0, 1, 0, 1, 0, 1],
        q_probs,
        inverse_temperatures: vec![1.0, 1.0, 0.0, 1.0],
        target_top_k: vec![1, 0, 0, 0],
        top_p: vec![0.0; 4],
        min_p: vec![0.0; 4],
        accept_uniforms: vec![0.0; 4],
        sample_uniforms: vec![0.5, 0.5, 0.5, 1.0],
        mode: SparseRejectionMode::Categorical,
    };
    let rows = case.run(&device)?;
    assert_eq!(
        rows.iter().map(|row| row.status).collect::<Vec<_>>(),
        vec![
            SPARSE_REJECTION_STATUS_NEEDS_CPU,
            SPARSE_REJECTION_STATUS_INVALID_Q,
            SPARSE_REJECTION_STATUS_INVALID_TARGET,
            SPARSE_REJECTION_STATUS_INVALID_RNG,
        ]
    );
    assert!(rows.iter().all(|row| {
        row.accepted_count == SPARSE_REJECTION_INVALID_VALUE
            && row.continuation == SPARSE_REJECTION_INVALID_VALUE
    }));
    Ok(())
}
