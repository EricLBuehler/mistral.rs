#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{
    maybe_wrap_dynamic_lora, with_lora_execution, LoraExecution, LoraLayerRegistry, LoraLinearSpec,
    LoraWeights, QuantMethod, QuantMethodConfig, ShardedSafeTensors, UnquantLinear,
};

use super::MergedDenseProjection;

const MERGED_PADDED_OUTPUT_ROWS: usize = 16;
#[cfg(feature = "cuda")]
const CUDA_F32_REL_TOLERANCE: f32 = 1e-5;
#[cfg(feature = "cuda")]
const CUDA_BF16_ABS_TOLERANCE: f32 = 2e-2;
#[cfg(feature = "cuda")]
const CUDA_LOGPROB_REL_TOLERANCE: f32 = 1e-4;

#[cfg(feature = "cuda")]
fn assert_close(actual: f32, expected: f32, relative_tolerance: f32) {
    let tolerance = relative_tolerance * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[cfg(feature = "cuda")]
fn topk_sampling_reference(
    logits: &[f32],
    inverse_temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    uniform: f32,
) -> u32 {
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        logits[right]
            .partial_cmp(&logits[left])
            .unwrap()
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(top_k.min(logits.len()));

    let scaled_max = logits[indices[0]] * inverse_temperature;
    let mut probabilities = indices
        .iter()
        .map(|&index| (logits[index] * inverse_temperature - scaled_max).exp())
        .collect::<Vec<_>>();
    let denominator = probabilities.iter().sum::<f32>();
    for probability in &mut probabilities {
        *probability /= denominator;
    }
    if top_p > 0.0 && top_p < 1.0 {
        let cutoff = top_p * probabilities.iter().sum::<f32>();
        let mut cumulative = 0.0f32;
        for probability in &mut probabilities {
            if cumulative >= cutoff {
                *probability = 0.0;
            } else {
                cumulative += *probability;
            }
        }
    }
    if min_p > 0.0 && min_p < 1.0 {
        let threshold = probabilities[0] * min_p;
        for probability in &mut probabilities {
            if threshold >= *probability {
                *probability = 0.0;
            }
        }
    }

    let chosen = uniform * probabilities.iter().sum::<f32>();
    let mut cumulative = 0.0f32;
    let mut selected = None;
    for (rank, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if probability > 0.0 {
            selected = Some(rank);
        }
        if cumulative > chosen {
            selected = Some(rank);
            break;
        }
    }
    indices[selected.unwrap()] as u32
}

#[cfg(feature = "cuda")]
fn ranked_topk(packed: Tensor, k: usize) -> super::RankedTopKPackedOutput {
    super::RankedTopKPackedOutput {
        packed,
        k,
        _workspace: Vec::new(),
    }
}

#[cfg(feature = "cuda")]
struct DFlashSelectorReference<'a> {
    packed_topk: &'a [f32],
    hidden: &'a [f32],
    predecessor_codebook: &'a [f32],
    successor_codebook: &'a [f32],
    anchors: &'a [u32],
    positions: usize,
    rank: usize,
    vocab: usize,
    k: usize,
}

#[cfg(feature = "cuda")]
fn dflash_selector_reference(input: DFlashSelectorReference<'_>) -> Vec<u32> {
    let packed_width = 2 * input.k;
    let mut selected = Vec::with_capacity(input.anchors.len() * input.positions);
    for (batch, anchor) in input.anchors.iter().enumerate() {
        let mut predecessor = *anchor as usize;
        for position in 0..input.positions {
            let row = batch * input.positions + position;
            let packed = &input.packed_topk[row * packed_width..(row + 1) * packed_width];
            let hidden = &input.hidden[row * input.rank..(row + 1) * input.rank];
            let pred = &input.predecessor_codebook
                [predecessor * input.rank..(predecessor + 1) * input.rank];
            let mut best_score = f32::NEG_INFINITY;
            let mut best_token = packed[input.k] as u32;
            for candidate_slot in 0..input.k {
                let candidate = packed[input.k + candidate_slot] as usize;
                assert!(candidate < input.vocab);
                let succ =
                    &input.successor_codebook[candidate * input.rank..(candidate + 1) * input.rank];
                let dot = pred
                    .iter()
                    .zip(hidden)
                    .zip(succ)
                    .map(|((pred, hidden), succ)| pred * hidden * succ)
                    .sum::<f32>();
                let score = packed[candidate_slot] + dot;
                if score > best_score {
                    best_score = score;
                    best_token = candidate as u32;
                }
            }
            selected.push(best_token);
            predecessor = best_token as usize;
        }
    }
    selected
}

#[cfg(feature = "cuda")]
fn dflash_sample_selector_reference(
    input: DFlashSelectorReference<'_>,
    inverse_temperatures: &[f32],
    uniforms: &[f32],
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let packed_width = 2 * input.k;
    let mut selected = Vec::with_capacity(input.anchors.len() * input.positions);
    let mut candidate_ids = Vec::with_capacity(selected.capacity() * input.k);
    let mut candidate_probs = Vec::with_capacity(candidate_ids.capacity());
    for (batch, anchor) in input.anchors.iter().enumerate() {
        let mut predecessor = *anchor as usize;
        for position in 0..input.positions {
            let row = batch * input.positions + position;
            let packed = &input.packed_topk[row * packed_width..(row + 1) * packed_width];
            let hidden = &input.hidden[row * input.rank..(row + 1) * input.rank];
            let pred = &input.predecessor_codebook
                [predecessor * input.rank..(predecessor + 1) * input.rank];
            let mut scores = Vec::with_capacity(input.k);
            for candidate_slot in 0..input.k {
                let candidate = packed[input.k + candidate_slot] as usize;
                let succ =
                    &input.successor_codebook[candidate * input.rank..(candidate + 1) * input.rank];
                let dot = pred
                    .iter()
                    .zip(hidden)
                    .zip(succ)
                    .map(|((pred, hidden), succ)| pred * hidden * succ)
                    .sum::<f32>();
                candidate_ids.push(candidate as u32);
                scores.push(packed[candidate_slot] + dot);
            }

            let inverse_temperature = inverse_temperatures[batch];
            let selected_slot = if inverse_temperature <= 0.0 {
                let mut selected_slot = 0;
                let mut best_score = f32::NEG_INFINITY;
                for (candidate_slot, score) in scores.iter().enumerate() {
                    if *score > best_score {
                        best_score = *score;
                        selected_slot = candidate_slot;
                    }
                }
                candidate_probs.extend((0..input.k).map(|candidate_slot| {
                    if candidate_slot == selected_slot {
                        1.0
                    } else {
                        0.0
                    }
                }));
                selected_slot
            } else {
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| ((score - max_score) * inverse_temperature).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                candidate_probs.extend(weights.iter().map(|weight| weight / denominator));
                let target = uniforms[row] * denominator;
                let mut cumulative = 0.0f32;
                weights
                    .iter()
                    .position(|weight| {
                        cumulative += weight;
                        target < cumulative
                    })
                    .unwrap()
            };
            predecessor = packed[input.k + selected_slot] as usize;
            selected.push(predecessor as u32);
        }
    }
    (selected, candidate_ids, candidate_probs)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_add_rms_norm_matches_separate_ops() -> candle_core::Result<()> {
    const ROWS: usize = 2;
    const COLS: usize = 16;
    const EPS: f32 = 1e-6;

    let device = Device::new_cuda(0)?;
    let input = Tensor::from_vec(
        (0..ROWS * COLS)
            .map(|index| index as f32 * 0.03125 - 0.4)
            .collect::<Vec<_>>(),
        (ROWS, COLS),
        &device,
    )?
    .to_dtype(DType::BF16)?;
    let residual = Tensor::from_vec(
        (0..ROWS * COLS)
            .map(|index| 0.25 - index as f32 * 0.015625)
            .collect::<Vec<_>>(),
        (ROWS, COLS),
        &device,
    )?
    .to_dtype(DType::BF16)?;
    let weight = Tensor::from_vec(
        (0..COLS)
            .map(|index| 0.75 + index as f32 * 0.01)
            .collect::<Vec<_>>(),
        COLS,
        &device,
    )?
    .to_dtype(DType::BF16)?;

    let expected_sum = (&input + &residual)?;
    let expected_norm = candle_nn::ops::rms_norm(&expected_sum.contiguous()?, &weight, EPS)?;
    let (actual_sum, actual_norm) = super::cuda_add_rms_norm(&input, &residual, &weight, EPS)?;

    let expected_sum = expected_sum
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual_sum = actual_sum
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual_sum, expected_sum);

    let expected_norm = expected_norm
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual_norm = actual_norm
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (actual, expected) in actual_norm.into_iter().zip(expected_norm) {
        assert!((actual - expected).abs() <= CUDA_BF16_ABS_TOLERANCE);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[allow(clippy::cast_precision_loss)]
fn cuda_qk_norm_rope_writes_token_major_from_packed_projection() -> candle_core::Result<()> {
    const BATCH: usize = 2;
    const SEQ_LEN: usize = 3;
    const Q_HEADS: usize = 2;
    const K_HEADS: usize = 1;
    const HEAD_DIM: usize = 8;
    const EPS: f32 = 1e-6;

    let device = Device::new_cuda(0)?;
    let packed_width = (Q_HEADS + K_HEADS + K_HEADS) * HEAD_DIM;
    let packed = Tensor::arange(0f32, (BATCH * SEQ_LEN * packed_width) as f32, &device)?
        .affine(0.003, -0.4)?
        .to_dtype(DType::BF16)?
        .reshape((BATCH, SEQ_LEN, packed_width))?;
    let q = packed
        .narrow(2, 0, Q_HEADS * HEAD_DIM)?
        .reshape((BATCH, SEQ_LEN, Q_HEADS, HEAD_DIM))?
        .transpose(1, 2)?;
    let k = packed
        .narrow(2, Q_HEADS * HEAD_DIM, K_HEADS * HEAD_DIM)?
        .reshape((BATCH, SEQ_LEN, K_HEADS, HEAD_DIM))?
        .transpose(1, 2)?;
    let q_weight = Tensor::arange(0f32, HEAD_DIM as f32, &device)?
        .affine(0.02, 0.8)?
        .to_dtype(DType::BF16)?;
    let k_weight = Tensor::arange(0f32, HEAD_DIM as f32, &device)?
        .affine(-0.015, 1.1)?
        .to_dtype(DType::BF16)?;
    let angles = Tensor::arange(0f32, (BATCH * SEQ_LEN * HEAD_DIM / 2) as f32, &device)?
        .affine(0.01, 0.0)?
        .reshape((BATCH, SEQ_LEN, HEAD_DIM / 2))?;
    let cos = angles.cos()?.to_dtype(DType::BF16)?;
    let sin = angles.sin()?.to_dtype(DType::BF16)?;

    let expected_q = candle_nn::rotary_emb::rope(
        &candle_nn::ops::rms_norm(&q.contiguous()?, &q_weight, EPS)?,
        &cos,
        &sin,
    )?
    .transpose(1, 2)?
    .contiguous()?;
    let expected_k = candle_nn::rotary_emb::rope(
        &candle_nn::ops::rms_norm(&k.contiguous()?, &k_weight, EPS)?,
        &cos,
        &sin,
    )?
    .transpose(1, 2)?
    .contiguous()?;
    let (actual_q, actual_k) = super::try_cuda_qk_rms_norm_rope(
        &q,
        Some(&k),
        &q_weight,
        Some(&k_weight),
        EPS,
        EPS,
        &cos.reshape((BATCH * SEQ_LEN, HEAD_DIM / 2))?,
        &sin.reshape((BATCH * SEQ_LEN, HEAD_DIM / 2))?,
        true,
        super::QkRopeOutputLayout::TokensFirst,
    )?
    .expect("supported CUDA Q/K fusion");
    let actual_k = actual_k.expect("K output");
    assert_eq!(actual_q.dims4()?, (BATCH, SEQ_LEN, Q_HEADS, HEAD_DIM));
    assert_eq!(actual_k.dims4()?, (BATCH, SEQ_LEN, K_HEADS, HEAD_DIM));

    for (actual, expected) in actual_q
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .zip(
            expected_q
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        )
        .chain(
            actual_k
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .zip(
                    expected_k
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                ),
        )
    {
        assert!((actual - expected).abs() <= CUDA_BF16_ABS_TOLERANCE);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_qk_norm_rope_positions_preserves_projection_layout() -> candle_core::Result<()> {
    const BATCH: usize = 2;
    const SEQ_LEN: usize = 3;
    const Q_HEADS: usize = 3;
    const K_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const PADDING_HEADS: usize = 1;
    const CACHE_ROWS: usize = 11;
    const EPS: f32 = 1e-6;
    const BF16_TOLERANCE: f32 = 0.008;
    const F16_TOLERANCE: f32 = 0.001;
    const F32_TOLERANCE: f32 = 1e-5;
    const POSITIONS: [u32; BATCH * SEQ_LEN] = [9, 2, 7, 1, 10, 4];

    let device = Device::new_cuda(0)?;
    let packed_heads = PADDING_HEADS + Q_HEADS + K_HEADS;
    let width = packed_heads * HEAD_DIM;
    let raw = (0..BATCH * SEQ_LEN * width)
        .map(|index| ((index * 29 + index / 7) % 277) as f32 / 71.0 - 1.7)
        .collect::<Vec<_>>();
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let packed = Tensor::from_vec(
            raw.clone(),
            (BATCH, SEQ_LEN, packed_heads, HEAD_DIM),
            &device,
        )?
        .to_dtype(dtype)?;
        let q = packed.narrow(2, PADDING_HEADS, Q_HEADS)?.transpose(1, 2)?;
        let k = packed
            .narrow(2, PADDING_HEADS + Q_HEADS, K_HEADS)?
            .transpose(1, 2)?;
        for projection in [&q, &k] {
            assert!(projection.storage_and_layout().1.start_offset() > 0);
            assert_eq!(projection.stride()[2], width);
        }
        let weight = Tensor::from_vec(
            (0..HEAD_DIM)
                .map(|index| 0.7 + index as f32 / 211.0)
                .collect::<Vec<_>>(),
            HEAD_DIM,
            &device,
        )?
        .to_dtype(dtype)?;
        let positions = Tensor::new(POSITIONS.as_slice(), &device)?;
        for rot_dim in [HEAD_DIM / 4, HEAD_DIM / 2] {
            let angles = Tensor::from_vec(
                (0..CACHE_ROWS * rot_dim)
                    .map(|index| index as f32 * 0.023)
                    .collect::<Vec<_>>(),
                (CACHE_ROWS, rot_dim),
                &device,
            )?;
            let cos = angles.cos()?.to_dtype(dtype)?;
            let sin = angles.sin()?.to_dtype(dtype)?;
            let values = |tensor: &Tensor| -> candle_core::Result<Vec<f32>> {
                tensor
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1()
            };
            let weights = values(&weight)?;
            let cos_values = values(&cos)?;
            let sin_values = values(&sin)?;
            for is_neox in [false, true] {
                let reference = |input: &Tensor| -> candle_core::Result<Vec<f32>> {
                    let input_values = values(input)?;
                    let heads = input.dim(1)?;
                    let mut expected = vec![0f32; input_values.len()];
                    for batch in 0..BATCH {
                        for head in 0..heads {
                            for seq in 0..SEQ_LEN {
                                let row = ((batch * heads + head) * SEQ_LEN + seq) * HEAD_DIM;
                                let squares = input_values[row..row + HEAD_DIM]
                                    .iter()
                                    .map(|x| f64::from(*x).powi(2))
                                    .sum::<f64>();
                                let inv_rms =
                                    (squares / HEAD_DIM as f64 + f64::from(EPS)).sqrt().recip();
                                let normalized = (0..HEAD_DIM)
                                    .map(|col| {
                                        f64::from(input_values[row + col])
                                            * inv_rms
                                            * f64::from(weights[col])
                                    })
                                    .collect::<Vec<_>>();
                                for col in 0..HEAD_DIM {
                                    expected[row + col] = normalized[col] as f32;
                                }
                                let cache_row = POSITIONS[batch * SEQ_LEN + seq] as usize * rot_dim;
                                for col in 0..rot_dim {
                                    let (left, right) = if is_neox {
                                        (col, col + rot_dim)
                                    } else {
                                        (2 * col, 2 * col + 1)
                                    };
                                    let c = f64::from(cos_values[cache_row + col]);
                                    let s = f64::from(sin_values[cache_row + col]);
                                    expected[row + left] =
                                        (normalized[left] * c - normalized[right] * s) as f32;
                                    expected[row + right] =
                                        (normalized[right] * c + normalized[left] * s) as f32;
                                }
                            }
                        }
                    }
                    values(
                        &Tensor::from_vec(expected, input.shape(), &Device::Cpu)?
                            .to_dtype(dtype)?,
                    )
                };
                let expected_q = reference(&q)?;
                let expected_k = reference(&k)?;
                for token_major in [false, true] {
                    let q = if token_major {
                        q.clone()
                    } else {
                        q.contiguous()?
                    };
                    let k = if token_major {
                        k.clone()
                    } else {
                        k.contiguous()?
                    };
                    for with_k in [false, true] {
                        let (actual_q, actual_k) = super::try_cuda_qk_rms_norm_rope_positions(
                            &q,
                            with_k.then_some(&k),
                            &weight,
                            with_k.then_some(&weight),
                            EPS,
                            EPS,
                            &cos,
                            &sin,
                            &positions,
                            is_neox,
                        )?
                        .expect("supported CUDA Q/K normalization and RoPE");
                        assert_eq!(actual_q.shape(), q.shape());
                        assert_eq!(actual_k.is_some(), with_k);
                        let outputs = std::iter::once((&actual_q, &expected_q))
                            .chain(actual_k.as_ref().map(|tensor| (tensor, &expected_k)));
                        for (actual, expected) in outputs {
                            if token_major {
                                assert!(actual.transpose(1, 2)?.is_contiguous());
                            } else {
                                assert!(actual.is_contiguous());
                            }
                            let tolerance = match dtype {
                                DType::BF16 => BF16_TOLERANCE,
                                DType::F16 => F16_TOLERANCE,
                                _ => F32_TOLERANCE,
                            };
                            for (index, (actual, expected)) in
                                values(actual)?.into_iter().zip(expected).enumerate()
                            {
                                assert!((actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                                        "{dtype:?} token_major={token_major} neox={is_neox} with_k={with_k} output[{index}]={actual}, expected={expected}");
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn packed_reference(logits: &[f32], k: usize, inverse_temperature: f32) -> Vec<f32> {
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&lhs, &rhs| logits[rhs].total_cmp(&logits[lhs]));
    indices.truncate(k.min(logits.len()));

    let global_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inverse_temperature;
    let denominator = logits
        .iter()
        .map(|value| (value * inverse_temperature - global_max).exp())
        .sum::<f32>();
    let mut packed = indices
        .iter()
        .map(|&index| logits[index])
        .collect::<Vec<_>>();
    packed.extend(
        indices
            .into_iter()
            .map(|index| f32::from(u16::try_from(index).expect("test vocabulary fits u16"))),
    );
    packed.extend([denominator, global_max]);
    packed
}

#[cfg(feature = "cuda")]
fn categorical_reference(logits: &[f32], inverse_temperature: f32, uniform: f32) -> [f32; 2] {
    let global_max = logits
        .iter()
        .map(|value| value * inverse_temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let weights = logits
        .iter()
        .map(|value| (value * inverse_temperature - global_max).exp())
        .collect::<Vec<_>>();
    let denominator = weights.iter().sum::<f32>();
    let target = uniform * denominator;
    let mut cumulative = 0.0f32;
    let token = weights
        .iter()
        .position(|weight| {
            cumulative += weight;
            target < cumulative
        })
        .expect("valid categorical distribution");
    [
        f32::from(u16::try_from(token).expect("test vocabulary fits u16")),
        logits[token] * inverse_temperature - global_max - denominator.ln(),
    ]
}

#[test]
fn merged_projection_uses_dynamic_lora_constituents_when_active() -> candle_core::Result<()> {
    let registry = Arc::new(LoraLayerRegistry::new());
    let vb = ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
        .with_lora_registry(registry.clone());
    let packed_weight = Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.], [1., -1.]], &Device::Cpu)?;
    let packed_weight = Tensor::cat(
        &[
            &packed_weight,
            &Tensor::zeros(
                (MERGED_PADDED_OUTPUT_ROWS - packed_weight.dim(0)?, 2),
                DType::F32,
                &Device::Cpu,
            )?,
        ],
        0,
    )?;
    let packed = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
        Linear::new(packed_weight.clone(), None),
    ))?) as Arc<dyn QuantMethod>;
    let gate_view = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
        Linear::new(packed_weight.narrow(0, 0, 2)?, None),
    ))?) as Arc<dyn QuantMethod>;
    let up_view = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
        Linear::new(packed_weight.narrow(0, 2, 2)?, None),
    ))?) as Arc<dyn QuantMethod>;
    let gate =
        maybe_wrap_dynamic_lora(&vb.pp("gate"), gate_view, LoraLinearSpec::replicated(2, 2))?;
    let up = maybe_wrap_dynamic_lora(&vb.pp("up"), up_view, LoraLinearSpec::replicated(2, 2))?;
    registry.finalize()?;
    let merged = MergedDenseProjection::from_packed(&mistralrs_quant::PackedColumnParallel {
        packed,
        constituents: vec![gate, up],
        rows_per_rank: vec![2, 2],
    });
    let input = Tensor::new(&[[2f32, 3.]], &Device::Cpu)?;
    let base = merged.forward(&input)?;
    assert_eq!(base[0].stride(), &[MERGED_PADDED_OUTPUT_ROWS, 1]);
    assert_eq!(base[0].to_vec2::<f32>()?, vec![vec![2., 3.]]);
    assert_eq!(base[1].to_vec2::<f32>()?, vec![vec![5., -1.]]);

    let gate_site = registry
        .sites()
        .into_iter()
        .find(|site| site.key().path() == "gate")
        .expect("gate site");
    let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
    execution.insert(
        &gate_site,
        0,
        LoraWeights::new(
            Tensor::new(&[[1f32, 0.]], &Device::Cpu)?,
            Tensor::new(&[[1f32], [0.]], &Device::Cpu)?,
            2.0,
        )?,
    )?;
    let active = with_lora_execution(Some(Arc::new(execution)), || merged.forward(&input))?;
    assert_eq!(active[0].to_vec2::<f32>()?, vec![vec![6., 3.]]);
    assert_eq!(active[1].to_vec2::<f32>()?, vec![vec![5., -1.]]);
    Ok(())
}

#[test]
fn merged_projection_keeps_multirow_gate_up_packed() -> candle_core::Result<()> {
    let packed_weight = Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.], [1., -1.]], &Device::Cpu)?;
    let packed = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
        Linear::new(packed_weight, None),
    ))?) as Arc<dyn QuantMethod>;
    let dummy = || {
        Arc::new(mistralrs_quant::DummyLayer::placeholder(
            mistralrs_quant::DummyLayerInfo::unknown(),
        )) as Arc<dyn QuantMethod>
    };
    let merged = MergedDenseProjection::from_packed(&mistralrs_quant::PackedLinear {
        packed,
        constituents: vec![dummy(), dummy()],
        rows_per_rank: vec![2, 2],
    });
    let input = Tensor::new(&[[2f32, 3.], [4., 5.]], &Device::Cpu)?;
    let packed_output = merged
        .forward_packed(&input)?
        .expect("inactive constituents keep the packed path");
    assert_eq!(packed_output.dims(), &[2, 4]);
    let actual = super::split_mul_and_act(&packed_output, 2, crate::layers::Activation::Silu)?;
    let gate = packed_output
        .narrow(candle_core::D::Minus1, 0, 2)?
        .contiguous()?;
    let up = packed_output
        .narrow(candle_core::D::Minus1, 2, 2)?
        .contiguous()?;
    let expected = super::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
    assert_eq!(actual.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    Ok(())
}

#[test]
fn test_topk() {
    use crate::ops::{TopKLastDimOp, TopKOutput};
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    //  [[1, 3, 5],
    //   [2, 4, 6]]
    let x = Tensor::arange(1f32, 7f32, &device)
        .unwrap()
        .reshape((3, 2))
        .unwrap()
        .t()
        .unwrap()
        .contiguous()
        .unwrap();
    let TopKOutput { values, indices } = x.topk(2).unwrap();
    assert_eq!(
        x.to_vec2::<f32>().unwrap(),
        vec![vec![1f32, 3f32, 5f32], vec![2f32, 4f32, 6f32]]
    );
    assert_eq!(
        values.to_vec2::<f32>().unwrap(),
        vec![vec![5f32, 3f32], vec![6f32, 4f32]]
    );
    assert_eq!(
        indices.to_vec2::<u32>().unwrap(),
        vec![vec![2u32, 1u32], vec![2u32, 1u32]]
    );
}

#[test]
fn test_topk_unsorted_cpu() -> candle_core::Result<()> {
    use crate::ops::{TopKLastDimOp, TopKOutput};
    use candle_core::Tensor;
    let x = Tensor::new(&[[1f32, 3., 5.], [6., 2., 4.]], &candle_core::Device::Cpu)?;
    let TopKOutput { values, indices } = x.topk_unsorted(2)?;
    assert_eq!(values.to_vec2::<f32>()?, vec![vec![3f32, 5.], vec![6., 4.]]);
    assert_eq!(indices.to_vec2::<u32>()?, vec![vec![1u32, 2], vec![0, 2]]);
    Ok(())
}

#[test]
fn test_repeat_interleave() -> candle_core::Result<()> {
    use crate::ops::RepeatInterleaveOp;
    use candle_core::{Device, Tensor};

    let input = Tensor::new(
        vec![vec![vec![1f32, 2., 3.], vec![4f32, 5., 6.]]],
        &Device::Cpu,
    )?;

    let repeat_interleaved = input.repeat_interleave(2, 2)?;
    assert_eq!(
        repeat_interleaved.to_vec3::<f32>()?,
        vec![vec![
            vec![1., 1., 2., 2., 3., 3.],
            vec![4., 4., 5., 5., 6., 6.]
        ]]
    );

    Ok(())
}

#[test]
fn test_repeat_interleave_flat() -> candle_core::Result<()> {
    use crate::ops::RepeatInterleaveOp;
    use candle_core::{Device, Tensor};

    let input = Tensor::new(vec![1., 2., 3., 4.], &Device::Cpu)?;

    let repeat_interleaved = input.repeat_interleave_flat(vec![1u32, 2u32, 3u32, 4u32])?;
    assert_eq!(
        repeat_interleaved.to_vec1::<f64>()?,
        vec![1., 2., 2., 3., 3., 3., 4., 4., 4., 4.]
    );

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_topk_matches_cpu_with_offsets_and_mixed_temperatures() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let logits = Tensor::new(
        &[
            [90.0f32, 91.0, 92.0, 93.0],
            [-1.0, 4.0, 0.0, 2.0],
            [3.0, 1.0, 5.0, -2.0],
        ],
        &device,
    )?
    .narrow(0, 1, 2)?;
    let inverse_temperatures = Tensor::new(&[99.0f32, 1.0, 0.5], &device)?.narrow(0, 1, 2)?;

    let output = super::cuda_topk_logits_f32_packed_batched(&logits, 8, &inverse_temperatures)?;
    assert_eq!(output.k, 4);
    let actual = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let expected = [
        packed_reference(&[-1.0, 4.0, 0.0, 2.0], 4, 1.0),
        packed_reference(&[3.0, 1.0, 5.0, -2.0], 4, 0.5),
    ];

    for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
        for (&actual, &expected) in actual_row.iter().zip(expected_row.iter()) {
            assert_close(actual, expected, CUDA_F32_REL_TOLERANCE);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_topk_rejects_nan_distribution() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let logits = Tensor::new(&[[1.0f32, f32::NAN, 3.0, 2.0]], &device)?;
    let inverse_temperatures = Tensor::new(&[1.0f32], &device)?;
    let output = super::cuda_topk_logits_f32_packed_batched(&logits, 2, &inverse_temperatures)?;
    let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    assert!(packed[0][2 * output.k].is_nan());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_topk_sampling_matches_filtered_reference() -> candle_core::Result<()> {
    const BATCH: usize = 4;
    const VOCAB: usize = 2051;

    let device = Device::new_cuda(0)?;
    let mut rows = Vec::with_capacity(BATCH * VOCAB);
    for row in 0..BATCH {
        rows.extend((0..VOCAB).map(|column| {
            let shuffled = (column * 977 + row * 131) % VOCAB;
            shuffled as f32 * 0.0031 + column as f32 * 1e-7 - row as f32 * 0.17
        }));
    }
    let inverse_temperatures = [1.0f32, 0.5, 1.7, 0.9];
    let top_ks = [20u32, 7, 13, 3];
    let top_ps = [0.95f32, 0.55, 1.0, 0.2];
    let min_ps = [0.0f32, 0.1, 0.05, 0.0];
    let uniforms = [0.0f32, 0.347, 0.891, 0.777];
    let params = (0..BATCH)
        .map(|row| super::CudaTopKSamplingParams {
            inverse_temperature: inverse_temperatures[row],
            top_k: top_ks[row] as usize,
            top_p: top_ps[row],
            min_p: min_ps[row],
            uniform: uniforms[row],
        })
        .collect::<Vec<_>>();
    let expected = (0..BATCH)
        .map(|row| {
            topk_sampling_reference(
                &rows[row * VOCAB..(row + 1) * VOCAB],
                inverse_temperatures[row],
                top_ks[row] as usize,
                top_ps[row],
                min_ps[row],
                uniforms[row],
            )
        })
        .collect::<Vec<_>>();
    let logits = Tensor::from_vec(rows, (BATCH, VOCAB), &device)?;
    let mut workspace = None;
    let submission = super::cuda_topk_sampling_submit_batched(&logits, &params, &mut workspace)?;
    let actual =
        super::cuda_topk_sampling_submission_complete(workspace.as_mut().unwrap(), &submission)?
            .token_ids()
            .to_vec();

    assert_eq!(actual, expected);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_topk_sampling_reuses_ring_and_destination() -> candle_core::Result<()> {
    const BATCH: usize = 2;

    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let resident_input = Tensor::zeros((4, 1), DType::U32, &device)?;
    let first_logits = Tensor::new(&[[9.0f32, 8.0, 1.0, 0.0], [0.0, 1.0, 9.0, 8.0]], &device)?;
    let second_logits = Tensor::new(&[[8.0f32, 9.0, 0.0, 1.0], [1.0, 0.0, 8.0, 9.0]], &device)?;
    let inverse_temperatures = [1.0f32; BATCH];
    let top_ks = [2u32; BATCH];
    let top_ps = [1.0f32; BATCH];
    let min_ps = [0.0f32; BATCH];
    let first_uniforms = [0.01f32; BATCH];
    let second_uniforms = [0.99f32; BATCH];
    let first_params = (0..BATCH)
        .map(|row| super::CudaTopKSamplingParams {
            inverse_temperature: inverse_temperatures[row],
            top_k: top_ks[row] as usize,
            top_p: top_ps[row],
            min_p: min_ps[row],
            uniform: first_uniforms[row],
        })
        .collect::<Vec<_>>();
    let second_params = (0..BATCH)
        .map(|row| super::CudaTopKSamplingParams {
            inverse_temperature: inverse_temperatures[row],
            top_k: top_ks[row] as usize,
            top_p: top_ps[row],
            min_p: min_ps[row],
            uniform: second_uniforms[row],
        })
        .collect::<Vec<_>>();
    let mut workspace = None;
    let first = super::cuda_topk_sampling_submit_batched_into(
        &first_logits,
        &resident_input,
        &first_params,
        &mut workspace,
    )?;
    let first_slot = first.token.reservation.slot;
    super::cuda_topk_sampling_device_tokens_wait_on(workspace.as_mut().unwrap(), &first, &stream)?;
    super::cuda_topk_sampling_device_tokens_release_after(
        workspace.as_mut().unwrap(),
        &first,
        &stream,
    )?;
    let second = super::cuda_topk_sampling_submit_batched_into(
        &second_logits,
        &resident_input,
        &second_params,
        &mut workspace,
    )?;
    assert_ne!(first_slot, second.token.reservation.slot);
    assert!(super::cuda_topk_sampling_submit_batched(
        &Tensor::zeros((BATCH, 4), DType::F32, &device)?,
        &first_params,
        &mut workspace,
    )
    .is_err());
    super::cuda_topk_sampling_device_tokens_release_after(
        workspace.as_mut().unwrap(),
        &second,
        &stream,
    )?;

    let first_tokens =
        super::cuda_topk_sampling_submission_complete(workspace.as_mut().unwrap(), &first)?
            .token_ids()
            .to_vec();
    let second_tokens =
        super::cuda_topk_sampling_submission_complete(workspace.as_mut().unwrap(), &second)?
            .token_ids()
            .to_vec();
    assert_eq!(first_tokens, [0, 2]);
    assert_eq!(second_tokens, [0, 2]);

    let third =
        super::cuda_topk_sampling_submit_batched(&first_logits, &first_params, &mut workspace)?;
    assert_eq!(third.token.reservation.slot, first_slot);
    super::cuda_topk_sampling_submission_cancel(workspace.as_mut().unwrap(), &third)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cached_batched_top1_tracks_batch_shape() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut workspace = None;
    let first = Tensor::new(&[[1.0f32, 4.0, 3.0], [8.0, 2.0, 5.0]], &device)?;
    let actual = super::cuda_top1_logits_f32_packed_batched_cached(&first, &mut workspace)?;
    assert_eq!(actual, vec![[4.0, 1.0], [8.0, 0.0]]);
    assert_eq!(workspace.as_ref().unwrap().capacity_rows, 2);

    let second = Tensor::new(&[[0.0f32, -2.0, 7.0]], &device)?;
    let actual = super::cuda_top1_logits_f32_packed_batched_cached(&second, &mut workspace)?;
    assert_eq!(actual, vec![[7.0, 2.0]]);
    assert_eq!(workspace.as_ref().unwrap().capacity_rows, 2);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_async_top1_device_and_host_tokens_match() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let logits = Tensor::new(&[[1.0f32, 7.0, 3.0], [9.0, 2.0, 5.0]], &device)?;
    let resident_input = Tensor::zeros((4, 1), DType::U32, &device)?;
    let mut workspace = None;
    let submission =
        super::cuda_top1_logits_submit_batched_into(&logits, &resident_input, &mut workspace)?;

    assert_eq!(submission.batch_size(), 2);
    assert_eq!(submission.device_tokens().dims(), &[4, 1]);
    let device_tokens = submission
        .device_tokens()
        .narrow(0, 0, 2)?
        .to_vec2::<u32>()?;
    let completion =
        super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &submission)?;

    assert_eq!(device_tokens, [[1], [0]]);
    assert_eq!(completion.token_ids(), &[1, 0]);
    assert!(completion.packed().is_none());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_async_top1_queues_two_submissions_and_reuses_slots() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let first = Tensor::new(&[[1.0f32, 4.0, 3.0], [8.0, 2.0, 5.0]], &device)?;
    let second = Tensor::new(&[[6.0f32, 4.0, 3.0], [1.0, 2.0, 9.0]], &device)?;
    let mut workspace = None;
    let first = super::cuda_top1_logits_submit_batched(&first, &mut workspace)?;
    let first_slot = first.token.reservation.slot;
    let second = super::cuda_top1_logits_submit_batched(&second, &mut workspace)?;

    assert_ne!(first_slot, second.token.reservation.slot);
    assert!(super::cuda_top1_logits_submit_batched(
        &Tensor::zeros((2, 3), DType::F32, &device)?,
        &mut workspace,
    )
    .is_err());
    let first_tokens = super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &first)?
        .token_ids()
        .to_vec();
    let second_tokens = super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &second)?
        .token_ids()
        .to_vec();
    assert_eq!(first_tokens, [1, 0]);
    assert_eq!(second_tokens, [0, 2]);

    let third = Tensor::new(&[[1.0f32, 2.0, 8.0], [3.0, 7.0, 4.0]], &device)?;
    let third = super::cuda_top1_logits_submit_batched(&third, &mut workspace)?;
    assert_eq!(third.token.reservation.slot, first_slot);
    let third_tokens = super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &third)?
        .token_ids()
        .to_vec();
    assert_eq!(third_tokens, [2, 1]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_async_top1_releases_resident_target_before_host_completion() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let resident_input = Tensor::zeros((2, 1), DType::U32, &device)?;
    let first_logits = Tensor::new(&[[1.0f32, 7.0], [9.0, 2.0]], &device)?;
    let second_logits = Tensor::new(&[[8.0f32, 1.0], [3.0, 6.0]], &device)?;
    let mut workspace = None;
    let first = super::cuda_top1_logits_submit_batched_into(
        &first_logits,
        &resident_input,
        &mut workspace,
    )?;
    super::cuda_top1_device_tokens_wait_on(workspace.as_mut().unwrap(), &first, &stream)?;
    super::cuda_top1_device_tokens_release_after(workspace.as_mut().unwrap(), &first, &stream)?;
    let second = super::cuda_top1_logits_submit_batched_into(
        &second_logits,
        &resident_input,
        &mut workspace,
    )?;
    super::cuda_top1_device_tokens_release_after(workspace.as_mut().unwrap(), &second, &stream)?;

    let first_tokens = super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &first)?
        .token_ids()
        .to_vec();
    let second_tokens = super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &second)?
        .token_ids()
        .to_vec();
    assert_eq!(first_tokens, [1, 0]);
    assert_eq!(second_tokens, [0, 1]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_async_top1_resizes_after_completion() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut workspace = None;
    let first = Tensor::new(&[[1.0f32, 4.0], [8.0, 2.0]], &device)?;
    let submission = super::cuda_top1_logits_submit_batched(&first, &mut workspace)?;
    super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &submission)?;
    let first_workspace_id = workspace.as_ref().unwrap().token_ring.id;

    let second = Tensor::new(&[[1.0f32, 9.0]], &device)?;
    let submission = super::cuda_top1_logits_submit_batched(&second, &mut workspace)?;
    assert_eq!(
        workspace.as_ref().unwrap().token_ring.id,
        first_workspace_id
    );
    assert_eq!(submission.batch_size(), 1);
    let completion =
        super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &submission)?;
    assert_eq!(completion.token_ids(), &[1]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_async_top1_marks_nan_token_invalid() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let logits = Tensor::new(&[[1.0f32, f32::NAN, 3.0]], &device)?;
    let mut workspace = None;
    let submission = super::cuda_top1_logits_submit_batched(&logits, &mut workspace)?;
    let completion =
        super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &submission)?;

    assert_eq!(completion.token_ids(), &[super::CUDA_TOP1_INVALID_TOKEN]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_low_precision_top1_matches_f32_across_ties_and_nonfinite_values() -> candle_core::Result<()>
{
    const BACKING_ROWS: usize = 6;
    const ROWS: usize = 5;
    const VOCAB: usize = 4097;

    let device = Device::new_cuda(0)?;
    let mut values = vec![-32.0f32; BACKING_ROWS * VOCAB];
    let row = |row: usize, column: usize| row * VOCAB + column;
    for column in [1, 256, 2048, 3000] {
        values[row(1, column)] = 7.0;
    }
    values[row(2, 2047)] = -2.0;
    values[row(2, 2048)] = 3.5;
    values[row(3, 17)] = f32::NAN;
    values[row(4, 9)] = f32::INFINITY;
    for value in &mut values[row(5, 0)..row(5, VOCAB)] {
        *value = f32::NEG_INFINITY;
    }
    let logits = Tensor::from_vec(values, (BACKING_ROWS, VOCAB), &device)?;

    for dtype in [DType::BF16, DType::F16] {
        let native = logits.to_dtype(dtype)?.narrow(0, 1, ROWS)?;
        let reference = native.to_dtype(DType::F32)?.contiguous()?;
        let mut workspace = None;

        let native_submission =
            super::cuda_top1_logits_submit_batched_packed(&native, &mut workspace)?;
        let (native_tokens, native_packed) = {
            let completion = super::cuda_top1_submission_complete(
                workspace.as_mut().unwrap(),
                &native_submission,
            )?;
            (
                completion.token_ids().to_vec(),
                completion.packed().unwrap().to_vec(),
            )
        };
        let reference_submission =
            super::cuda_top1_logits_submit_batched_packed(&reference, &mut workspace)?;
        let (reference_tokens, reference_packed) = {
            let completion = super::cuda_top1_submission_complete(
                workspace.as_mut().unwrap(),
                &reference_submission,
            )?;
            (
                completion.token_ids().to_vec(),
                completion.packed().unwrap().to_vec(),
            )
        };

        assert_eq!(native_tokens, reference_tokens);
        assert_eq!(native_tokens[0], 1);
        assert_eq!(native_tokens[1], 2048);
        assert_eq!(native_tokens[2], super::CUDA_TOP1_INVALID_TOKEN);
        assert_eq!(native_tokens[3], 9);
        assert_eq!(native_tokens[4], 0);
        for (native, reference) in native_packed.iter().zip(reference_packed) {
            assert!(native == &reference || (native.is_nan() && reference.is_nan()));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_topk_orders_ties_by_lowest_index() -> candle_core::Result<()> {
    const VOCAB: usize = 4097;

    let device = Device::new_cuda(0)?;
    let mut row = vec![-10.0f32; VOCAB];
    for index in [1, 256, 300, 2048] {
        row[index] = 5.0;
    }
    let logits = Tensor::from_vec(row, (1, VOCAB), &device)?;
    let inverse_temperatures = Tensor::new(&[1.0f32], &device)?;
    let output = super::cuda_topk_logits_f32_packed_batched(&logits, 4, &inverse_temperatures)?;
    let packed = output.packed.to_vec2::<f32>()?;

    assert_eq!(
        &packed[0][output.k..2 * output.k],
        &[1.0, 256.0, 300.0, 2048.0]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_topk_low_precision_inputs_match_f32() -> candle_core::Result<()> {
    const ROWS: usize = 3;
    const VOCAB: usize = 4097;
    const K: usize = 17;

    let device = Device::new_cuda(0)?;
    let values = (0..ROWS * VOCAB)
        .map(|index| (((index * 37) % 257) as f32 - 128.0) / 8.0)
        .collect::<Vec<_>>();
    let logits = Tensor::from_vec(values, (ROWS, VOCAB), &device)?;
    let inverse_temperatures = Tensor::new(&[2.0f32, 0.75, 0.125], &device)?.narrow(0, 1, 2)?;

    for dtype in [DType::BF16, DType::F16] {
        let low_precision = logits.to_dtype(dtype)?.narrow(0, 1, 2)?;
        let reference = low_precision.to_dtype(DType::F32)?.contiguous()?;
        let actual =
            super::cuda_topk_logits_packed_batched(&low_precision, K, &inverse_temperatures)?;
        let expected =
            super::cuda_topk_logits_f32_packed_batched(&reference, K, &inverse_temperatures)?;

        assert_eq!(actual.k, expected.k);
        assert_eq!(
            actual.packed.to_vec2::<f32>()?,
            expected.packed.to_vec2::<f32>()?
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_topk_workspace_reuses_and_grows() -> candle_core::Result<()> {
    const ROWS: usize = 4;
    const VOCAB: usize = 4097;

    let device = Device::new_cuda(0)?;
    let values = (0..ROWS * VOCAB)
        .map(|index| (((index * 37) % 257) as f32 - 128.0) / 8.0)
        .collect::<Vec<_>>();
    let logits = Tensor::from_vec(values, (ROWS, VOCAB), &device)?;
    let inverse_temperatures = Tensor::new(&[1.0f32, 0.75, 0.5, 0.25], &device)?;
    let mut workspace = None;

    let first_logits = logits.narrow(0, 0, 2)?;
    let first_temperatures = inverse_temperatures.narrow(0, 0, 2)?;
    let first = super::cuda_topk_logits_packed_batched_with_workspace(
        &first_logits,
        17,
        &first_temperatures,
        &mut workspace,
    )?;
    let expected = super::cuda_topk_logits_packed_batched(&first_logits, 17, &first_temperatures)?;
    assert_eq!(
        first.packed.to_vec2::<f32>()?,
        expected.packed.to_vec2::<f32>()?
    );
    drop(first);
    drop(expected);
    let first_workspace = workspace.as_ref().expect("workspace was allocated");
    let first_id = first_workspace.id;
    assert_eq!(first_workspace.capacity_rows, 2);
    assert_eq!(first_workspace.capacity_k, 32);

    let smaller_logits = logits.narrow(0, 1, 1)?;
    let smaller_temperatures = inverse_temperatures.narrow(0, 1, 1)?;
    super::cuda_topk_logits_packed_batched_with_workspace(
        &smaller_logits,
        8,
        &smaller_temperatures,
        &mut workspace,
    )?;
    assert_eq!(
        workspace.as_ref().expect("workspace was reused").id,
        first_id
    );

    let grown = super::cuda_topk_logits_packed_batched_with_workspace(
        &logits,
        33,
        &inverse_temperatures,
        &mut workspace,
    )?;
    let expected = super::cuda_topk_logits_packed_batched(&logits, 33, &inverse_temperatures)?;
    assert_eq!(
        grown.packed.to_vec2::<f32>()?,
        expected.packed.to_vec2::<f32>()?
    );
    let grown_workspace = workspace.as_ref().expect("workspace was grown");
    assert_ne!(grown_workspace.id, first_id);
    assert_eq!(grown_workspace.capacity_rows, 4);
    assert_eq!(grown_workspace.capacity_k, 64);

    let changed_vocab = Tensor::zeros((1, 2049), DType::F32, &device)?;
    let one_temperature = inverse_temperatures.narrow(0, 0, 1)?;
    let grown_id = grown_workspace.id;
    super::cuda_topk_logits_packed_batched_with_workspace(
        &changed_vocab,
        20,
        &one_temperature,
        &mut workspace,
    )?;
    assert_ne!(
        workspace.as_ref().expect("shape change was applied").id,
        grown_id
    );

    let wrong_temperatures = inverse_temperatures.narrow(0, 0, 2)?;
    let error = match super::cuda_topk_logits_packed_batched_with_workspace(
        &changed_vocab,
        20,
        &wrong_temperatures,
        &mut workspace,
    ) {
        Ok(_) => candle_core::bail!("row temperature shape mismatch must fail"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("inverse temperatures with shape"));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ranked_topk_matches_cpu_across_dtypes_ties_and_offsets() -> candle_core::Result<()> {
    const BACKING_ROWS: usize = 4;
    const ROWS: usize = 2;
    const VOCAB: usize = 4097;
    const K: usize = 8;

    let device = Device::new_cuda(0)?;
    let mut values = (0..BACKING_ROWS * VOCAB)
        .map(|index| ((index % VOCAB) % 127) as f32 / 8.0)
        .collect::<Vec<_>>();
    for (row, peaks) in [
        &[
            (1, 50.0),
            (256, 50.0),
            (300, 50.0),
            (2048, 50.0),
            (4096, 50.0),
        ][..],
        &[(0, 60.0), (255, 60.0), (1023, 60.0), (3000, 60.0)][..],
    ]
    .into_iter()
    .enumerate()
    {
        let row = row + 1;
        for &(index, value) in peaks {
            values[row * VOCAB + index] = value;
        }
    }
    let logits = Tensor::from_vec(values, (BACKING_ROWS, VOCAB), &device)?;

    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let input = logits.to_dtype(dtype)?.narrow(0, 1, ROWS)?;
        let (_storage, layout) = input.storage_and_layout();
        assert!(layout.start_offset() > 0);
        let reference = input
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let output = super::cuda_topk_ranked_packed_batched(&input, K)?;
        assert_eq!(output.k, K);
        assert_eq!(output.packed.dims(), &[ROWS, 2 * K]);
        let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        for (packed_row, reference_row) in packed.iter().zip(&reference) {
            let mut indices = (0..VOCAB).collect::<Vec<_>>();
            indices.sort_unstable_by(|&left, &right| {
                reference_row[right]
                    .total_cmp(&reference_row[left])
                    .then_with(|| left.cmp(&right))
            });
            for (slot, &index) in indices.iter().take(K).enumerate() {
                assert_eq!(packed_row[slot], reference_row[index]);
                assert_eq!(packed_row[K + slot], index as f32);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ranked_topk_radix_matches_realistic_vocab_and_cross_chunk_ties() -> candle_core::Result<()>
{
    const BACKING_ROWS: usize = 3;
    const ROWS: usize = 2;
    const VOCAB: usize = 248_320;

    let device = Device::new_cuda(0)?;
    let mut values = vec![-32.0f32; BACKING_ROWS * VOCAB];
    for index in 0..VOCAB {
        values[VOCAB + index] = ((index * 37) % 4096) as f32 / 32.0 - 64.0;
        values[2 * VOCAB + index] = 3.0;
    }
    for index in [
        1, 82_775, 82_776, 120_001, 165_551, 165_552, 220_003, 248_319,
    ] {
        values[VOCAB + index] = 256.0;
    }
    let logits = Tensor::from_vec(values, (BACKING_ROWS, VOCAB), &device)?;

    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let input = logits.to_dtype(dtype)?.narrow(0, 1, ROWS)?;
        let reference = input
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        for k in [20, 32] {
            let output = super::cuda_topk_ranked_packed_batched(&input, k)?;
            let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

            for (packed_row, reference_row) in packed.iter().zip(&reference) {
                let mut indices = (0..VOCAB).collect::<Vec<_>>();
                indices.sort_unstable_by(|&left, &right| {
                    reference_row[right]
                        .total_cmp(&reference_row[left])
                        .then_with(|| left.cmp(&right))
                });
                for (slot, &index) in indices.iter().take(k).enumerate() {
                    assert_eq!(packed_row[slot], reference_row[index]);
                    assert_eq!(packed_row[k + slot], index as f32);
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ranked_topk_radix_preserves_special_value_contract() -> candle_core::Result<()> {
    const VOCAB: usize = 4097;
    const K: usize = 16;

    let device = Device::new_cuda(0)?;
    let mut row = vec![f32::NEG_INFINITY; VOCAB];
    row[1] = f32::NAN;
    row[2] = f32::INFINITY;
    row[3] = -0.0;
    row[4] = 0.0;
    row[5] = -1.0;
    row[6] = 1.0;
    row[1024] = 7.0;
    row[2048] = 7.0;
    row[4096] = 7.0;
    let logits = Tensor::from_vec(row, (1, VOCAB), &device)?;

    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let input = logits.to_dtype(dtype)?;
        let output = super::cuda_topk_ranked_packed_batched(&input, K)?;
        let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let values = &packed[0][..K];
        let indices = &packed[0][K..];

        assert_eq!(
            &indices[..8],
            &[2.0, 1024.0, 2048.0, 4096.0, 6.0, 4.0, 3.0, 5.0]
        );
        assert!(values[0].is_infinite() && values[0].is_sign_positive());
        assert_eq!(&indices[8..], &[0.0; K - 8]);
        assert!(values[8..]
            .iter()
            .all(|value| value.is_infinite() && value.is_sign_negative()));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ranked_topk_cooperative_boundaries_and_fallback_match_cpu() -> candle_core::Result<()> {
    const VOCAB: usize = 248_320;

    let device = Device::new_cuda(0)?;
    let values = (0..VOCAB)
        .map(|index| ((index * 104_729) % VOCAB) as f32 / 64.0)
        .collect::<Vec<_>>();
    let logits = Tensor::from_vec(values.clone(), (1, VOCAB), &device)?;
    let mut expected_indices = (0..VOCAB).collect::<Vec<_>>();
    expected_indices.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });

    for k in [7, 8, 16, 17, 20, 32, 33, 128] {
        let output = super::cuda_topk_ranked_packed_batched(&logits, k)?;
        let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        for (slot, &index) in expected_indices.iter().take(k).enumerate() {
            assert_eq!(packed[0][slot], values[index]);
            assert_eq!(packed[0][k + slot], index as f32);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dflash_selector_matches_reference_with_bf16_codebooks() -> candle_core::Result<()> {
    const BATCH: usize = 2;
    const POSITIONS: usize = 3;
    const K: usize = 4;
    const RANK: usize = 7;
    const VOCAB: usize = 13;

    let rows = BATCH * POSITIONS;
    let packed_width = 2 * K;
    let mut packed = vec![0.0f32; rows * packed_width];
    for row in 0..rows {
        for candidate_slot in 0..K {
            packed[row * packed_width + candidate_slot] =
                ((row * 3 + candidate_slot * 5) % 7) as f32 * 0.25 - 0.75;
            packed[row * packed_width + K + candidate_slot] =
                ((row * 3 + candidate_slot * 2 + 1) % VOCAB) as f32;
        }
    }
    let hidden = (0..rows * RANK)
        .map(|index| ((index * 7) % 9) as f32 * 0.25 - 1.0)
        .collect::<Vec<_>>();
    let predecessor = (0..VOCAB * RANK)
        .map(|index| ((index * 5) % 11) as f32 * 0.125 - 0.625)
        .collect::<Vec<_>>();
    let successor = (0..VOCAB * RANK)
        .map(|index| ((index * 3) % 13) as f32 * 0.125 - 0.75)
        .collect::<Vec<_>>();
    let anchors = [2u32, 7];
    let expected = dflash_selector_reference(DFlashSelectorReference {
        packed_topk: &packed,
        hidden: &hidden,
        predecessor_codebook: &predecessor,
        successor_codebook: &successor,
        anchors: &anchors,
        positions: POSITIONS,
        rank: RANK,
        vocab: VOCAB,
        k: K,
    });

    let device = Device::new_cuda(0)?;
    let topk = ranked_topk(Tensor::from_vec(packed, (rows, packed_width), &device)?, K);
    let actual = super::cuda_dflash_greedy_select(
        &topk,
        &Tensor::from_vec(hidden, (rows, RANK), &device)?,
        &Tensor::from_vec(predecessor, (VOCAB, RANK), &device)?.to_dtype(DType::BF16)?,
        &Tensor::from_vec(successor, (VOCAB, RANK), &device)?.to_dtype(DType::BF16)?,
        &Tensor::new(&anchors, &device)?,
    )?
    .to_vec2::<u32>()?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    assert_eq!(actual, expected);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dflash_selector_supports_max_k_and_stable_ties() -> candle_core::Result<()> {
    const POSITIONS: usize = 2;
    const K: usize = super::CUDA_DFLASH_SELECTOR_MAX_K;
    const RANK: usize = 3;
    const VOCAB: usize = K;

    let packed_width = 2 * K;
    let mut packed = vec![0.0f32; POSITIONS * packed_width];
    for position in 0..POSITIONS {
        for candidate_slot in 0..K {
            packed[position * packed_width + candidate_slot] = 1.0;
            packed[position * packed_width + K + candidate_slot] = (K - candidate_slot - 1) as f32;
        }
    }

    let device = Device::new_cuda(0)?;
    let topk = ranked_topk(
        Tensor::from_vec(packed, (POSITIONS, packed_width), &device)?,
        K,
    );
    let actual = super::cuda_dflash_greedy_select(
        &topk,
        &Tensor::zeros((POSITIONS, RANK), DType::BF16, &device)?,
        &Tensor::zeros((VOCAB, RANK), DType::F32, &device)?,
        &Tensor::zeros((VOCAB, RANK), DType::F32, &device)?,
        &Tensor::new(&[0u32], &device)?,
    )?
    .to_vec2::<u32>()?;

    assert_eq!(actual, [vec![(K - 1) as u32; POSITIONS]]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dflash_sample_selector_matches_sequential_reference() -> candle_core::Result<()> {
    const BATCH: usize = 2;
    const POSITIONS: usize = 3;
    const K: usize = 3;
    const RANK: usize = 2;
    const VOCAB: usize = 7;

    let rows = BATCH * POSITIONS;
    let packed_width = 2 * K;
    let mut packed = vec![0.0f32; rows * packed_width];
    for row in 0..rows {
        for candidate_slot in 0..K {
            packed[row * packed_width + candidate_slot] =
                ((row * 5 + candidate_slot * 3) % 11) as f32 * 0.2 - 0.8;
            packed[row * packed_width + K + candidate_slot] =
                ((row + candidate_slot * 2 + 1) % VOCAB) as f32;
        }
    }
    let hidden = (0..rows * RANK)
        .map(|index| ((index * 3) % 7) as f32 * 0.25 - 0.5)
        .collect::<Vec<_>>();
    let predecessor = (0..VOCAB * RANK)
        .map(|index| ((index * 5) % 9) as f32 * 0.125 - 0.375)
        .collect::<Vec<_>>();
    let successor = (0..VOCAB * RANK)
        .map(|index| ((index * 7) % 11) as f32 * 0.1 - 0.4)
        .collect::<Vec<_>>();
    let anchors = [2u32, 5];
    let inverse_temperatures = [0.0f32, 0.75];
    let uniforms = [f32::NAN, f32::NAN, f32::NAN, 0.1, 0.7, 0.4];
    let (expected_tokens, expected_ids, expected_probs) = dflash_sample_selector_reference(
        DFlashSelectorReference {
            packed_topk: &packed,
            hidden: &hidden,
            predecessor_codebook: &predecessor,
            successor_codebook: &successor,
            anchors: &anchors,
            positions: POSITIONS,
            rank: RANK,
            vocab: VOCAB,
            k: K,
        },
        &inverse_temperatures,
        &uniforms,
    );

    let device = Device::new_cuda(0)?;
    let topk = ranked_topk(Tensor::from_vec(packed, (rows, packed_width), &device)?, K);
    let output = super::cuda_dflash_sample_select(super::DFlashSelectorSampleInput {
        topk: &topk,
        projected_hidden: &Tensor::from_vec(hidden, (rows, RANK), &device)?,
        predecessor_codebook: &Tensor::from_vec(predecessor, (VOCAB, RANK), &device)?,
        successor_codebook: &Tensor::from_vec(successor, (VOCAB, RANK), &device)?,
        anchors: &Tensor::new(&anchors, &device)?,
        inverse_temperatures: &Tensor::new(&inverse_temperatures, &device)?,
        uniforms: &Tensor::from_vec(uniforms.to_vec(), (BATCH, POSITIONS), &device)?,
    })?;
    let actual_tokens = output
        .tokens
        .to_vec2::<u32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let actual_ids = output
        .candidate_ids
        .to_vec3::<u32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();
    let actual_probs = output
        .candidate_probs
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();

    assert_eq!(actual_tokens, expected_tokens);
    assert_eq!(actual_ids, expected_ids);
    for (actual, expected) in actual_probs.into_iter().zip(expected_probs) {
        assert_close(actual, expected, CUDA_LOGPROB_REL_TOLERANCE);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dflash_sample_selector_marks_invalid_sampling_params() -> candle_core::Result<()> {
    const K: usize = 2;
    const VOCAB: usize = 2;
    const PACKED_WIDTH: usize = 2 * K;

    let device = Device::new_cuda(0)?;
    let topk = ranked_topk(
        Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, PACKED_WIDTH), &device)?,
        K,
    );
    let output = super::cuda_dflash_sample_select(super::DFlashSelectorSampleInput {
        topk: &topk,
        projected_hidden: &Tensor::zeros((1, 1), DType::F32, &device)?,
        predecessor_codebook: &Tensor::zeros((VOCAB, 1), DType::F32, &device)?,
        successor_codebook: &Tensor::zeros((VOCAB, 1), DType::F32, &device)?,
        anchors: &Tensor::new(&[0u32], &device)?,
        inverse_temperatures: &Tensor::new(&[f32::INFINITY], &device)?,
        uniforms: &Tensor::new(&[[0.5f32]], &device)?,
    })?;

    assert_eq!(
        output.tokens.to_vec2::<u32>()?,
        [vec![super::CUDA_DFLASH_SELECTOR_INVALID_TOKEN]]
    );
    assert!(output.candidate_probs.to_vec3::<f32>()?[0][0]
        .iter()
        .all(|probability| probability.is_nan()));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_categorical_matches_reference_across_chunks() -> candle_core::Result<()> {
    const VOCAB: usize = 2051;

    let device = Device::new_cuda(0)?;
    let mut backing = vec![90.0f32; 3 * VOCAB];
    let mut first = vec![-10.0f32; VOCAB];
    first[..3].copy_from_slice(&[0.0, 1.0, 2.0]);
    let mut second = vec![-20.0f32; VOCAB];
    second[2049] = 4.0;
    backing[VOCAB..2 * VOCAB].copy_from_slice(&first);
    backing[2 * VOCAB..].copy_from_slice(&second);
    let logits = Tensor::from_vec(backing, (3, VOCAB), &device)?.narrow(0, 1, 2)?;
    let inverse_temperatures = Tensor::new(&[99.0f32, 1.0, 0.5], &device)?.narrow(0, 1, 2)?;
    let uniforms = Tensor::new(&[0.99f32, 0.2, 0.5], &device)?.narrow(0, 1, 2)?;

    let output = super::cuda_categorical_logits_f32_packed_batched(
        &logits,
        &inverse_temperatures,
        &uniforms,
    )?;
    let actual = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let expected = [
        categorical_reference(&first, 1.0, 0.2),
        categorical_reference(&second, 0.5, 0.5),
    ];

    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual[0], expected[0]);
        assert_close(actual[1], expected[1], CUDA_LOGPROB_REL_TOLERANCE);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_categorical_marks_invalid_distribution() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let logits = Tensor::new(&[[1.0f32, f32::NAN, 3.0]], &device)?;
    let inverse_temperatures = Tensor::new(&[1.0f32], &device)?;
    let uniforms = Tensor::new(&[0.5f32], &device)?;
    let output = super::cuda_categorical_logits_f32_packed_batched(
        &logits,
        &inverse_temperatures,
        &uniforms,
    )?;
    let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

    assert!(packed[0].iter().all(|value| value.is_nan()));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_categorical_selects_at_upper_boundary() -> candle_core::Result<()> {
    const VOCAB: usize = 2048;

    let device = Device::new_cuda(0)?;
    let logits = Tensor::zeros((1, VOCAB), DType::F32, &device)?;
    let inverse_temperatures = Tensor::new(&[1.0f32], &device)?;
    let upper = f32::from_bits(1.0f32.to_bits() - 1);
    let uniforms = Tensor::new(&[upper], &device)?;
    let output = super::cuda_categorical_logits_f32_packed_batched(
        &logits,
        &inverse_temperatures,
        &uniforms,
    )?;
    let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

    assert_eq!(packed[0][0], 2047.0);
    assert_close(packed[0][1], -(2048.0f32).ln(), CUDA_LOGPROB_REL_TOLERANCE);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cached_top1_honors_view_offset() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let logits = Tensor::new(
        &[[90.0f32, 91.0, 92.0, 93.0], [-1.0, 4.0, 0.0, 2.0]],
        &device,
    )?
    .narrow(0, 1, 1)?;
    let mut workspace = None;
    let actual = super::cuda_top1_logits_f32_cached(&logits, &mut workspace)?;

    assert_eq!(actual, [4.0, 1.0]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_top1_uses_first_maximum_across_lanes_and_chunks() -> candle_core::Result<()> {
    const VOCAB: usize = 4097;

    let device = Device::new_cuda(0)?;
    let mut first = vec![-10.0f32; VOCAB];
    first[1] = 5.0;
    first[256] = 5.0;
    first[300] = 5.0;
    let mut second = vec![-10.0f32; VOCAB];
    second[2047] = 7.0;
    second[2048] = 7.0;
    second[3000] = 7.0;

    let mut workspace = None;
    let single = Tensor::from_vec(first.clone(), VOCAB, &device)?;
    let single = super::cuda_top1_logits_f32_cached(&single, &mut workspace)?;
    assert_eq!(single, [5.0, 1.0]);

    first.extend(second);
    let batched = Tensor::from_vec(first, (2, VOCAB), &device)?;
    let packed = super::cuda_top1_logits_f32_packed_batched(&batched)?
        .packed
        .to_vec2::<f32>()?;
    assert_eq!(packed, [[5.0, 1.0], [7.0, 2047.0]]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cached_top1_marks_nan_distribution() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let logits = Tensor::new(&[1.0f32, f32::NAN, 3.0], &device)?;
    let mut workspace = None;
    let actual = super::cuda_top1_logits_f32_cached(&logits, &mut workspace)?;

    assert!(actual.iter().all(|value| value.is_nan()));
    Ok(())
}
