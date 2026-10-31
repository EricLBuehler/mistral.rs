use super::*;
use candle_core::{Device, IndexOp};

#[derive(Clone, Copy)]
struct RecurrenceCase {
    bh: usize,
    seq_len: usize,
    k_dim: usize,
    v_dim: usize,
}

fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as u16 as f32;
            ((x / 128.0) - 1.0) * scale + offset
        })
        .collect()
}

fn tensor2(data: Vec<f32>, shape: (usize, usize), dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(data, shape, dev)
}

fn tensor3(data: Vec<f32>, shape: (usize, usize, usize), dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(data, shape, dev)
}

fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> (f32, usize, f32, f32) {
    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    let mut lhs_at_max = 0.0f32;
    let mut rhs_at_max = 0.0f32;
    for (idx, (&left, &right)) in lhs.iter().zip(rhs).enumerate() {
        let diff = (left - right).abs();
        if diff > max_diff || diff.is_nan() {
            max_diff = diff;
            max_idx = idx;
            lhs_at_max = left;
            rhs_at_max = right;
        }
    }
    (max_diff, max_idx, lhs_at_max, rhs_at_max)
}

fn assert_close(label: &str, lhs: &[f32], rhs: &[f32], tol: f32) {
    let lhs_nan = lhs.iter().filter(|x| x.is_nan()).count();
    let rhs_nan = rhs.iter().filter(|x| x.is_nan()).count();
    let (max_diff, max_idx, lhs_at_max, rhs_at_max) = max_abs_diff(lhs, rhs);
    assert!(
            lhs_nan == 0 && rhs_nan == 0 && max_diff <= tol,
            "{label}: max_diff={max_diff} at {max_idx}, lhs={lhs_at_max}, rhs={rhs_at_max}, lhs_nan={lhs_nan}, rhs_nan={rhs_nan}"
        );
}

fn run_case(case: RecurrenceCase, dev: &Device) -> Result<()> {
    let q = tensor3(
        patterned(case.bh * case.seq_len * case.k_dim, 1, 0.02, 0.0),
        (case.bh, case.seq_len, case.k_dim),
        dev,
    )?;
    let k = tensor3(
        patterned(case.bh * case.seq_len * case.k_dim, 2, 0.02, 0.0),
        (case.bh, case.seq_len, case.k_dim),
        dev,
    )?;
    let v = tensor3(
        patterned(case.bh * case.seq_len * case.v_dim, 3, 0.05, 0.0),
        (case.bh, case.seq_len, case.v_dim),
        dev,
    )?;
    let g = tensor2(
        patterned(case.bh * case.seq_len, 4, 0.03, -0.08),
        (case.bh, case.seq_len),
        dev,
    )?;
    let beta = tensor2(
        patterned(case.bh * case.seq_len, 5, 0.15, 0.5),
        (case.bh, case.seq_len),
        dev,
    )?;
    let state = patterned(case.bh * case.k_dim * case.v_dim, 6, 0.01, 0.0);

    let mut state_scalar = tensor3(state.clone(), (case.bh, case.k_dim, case.v_dim), dev)?;
    let scalar = gated_delta_rule_recurrence_cuda(
        RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        },
        &mut state_scalar,
        GdnStateSlots::Gathered,
    )?;
    let mut state_chunked = tensor3(state.clone(), (case.bh, case.k_dim, case.v_dim), dev)?;
    let chunked = chunked_gated_delta_rule_recurrence_cuda(
        RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        },
        &mut state_chunked,
        GdnStateSlots::Gathered,
    )?;
    let mut state_warp = tensor3(state, (case.bh, case.k_dim, case.v_dim), dev)?;
    let warp = warp_gated_delta_rule_recurrence_cuda(
        RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        },
        &mut state_warp,
        GdnStateSlots::Gathered,
    )?;

    let scalar_flat = flat(&scalar)?;
    let scalar_state_flat = flat(&state_scalar)?;
    let chunked_flat = flat(&chunked)?;
    let chunked_state_flat = flat(&state_chunked)?;
    let warp_flat = flat(&warp)?;
    let warp_state_flat = flat(&state_warp)?;

    let name = format!(
        "bh={},seq={},k={},v={}",
        case.bh, case.seq_len, case.k_dim, case.v_dim
    );
    assert_close(
        &format!("{name} chunked output"),
        &scalar_flat,
        &chunked_flat,
        3.0e-4,
    );
    assert_close(
        &format!("{name} chunked state"),
        &scalar_state_flat,
        &chunked_state_flat,
        3.0e-4,
    );
    assert_close(
        &format!("{name} warp output"),
        &scalar_flat,
        &warp_flat,
        3.0e-4,
    );
    assert_close(
        &format!("{name} warp state"),
        &scalar_state_flat,
        &warp_state_flat,
        3.0e-4,
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn warp_recurrence_matches_scalar_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for case in [
        RecurrenceCase {
            bh: 1,
            seq_len: 1,
            k_dim: 64,
            v_dim: 64,
        },
        RecurrenceCase {
            bh: 1,
            seq_len: 65,
            k_dim: 64,
            v_dim: 64,
        },
        RecurrenceCase {
            bh: 2,
            seq_len: 128,
            k_dim: 128,
            v_dim: 64,
        },
        RecurrenceCase {
            bh: 8,
            seq_len: 256,
            k_dim: 128,
            v_dim: 64,
        },
        RecurrenceCase {
            bh: 32,
            seq_len: 512,
            k_dim: 128,
            v_dim: 128,
        },
    ] {
        run_case(case, &dev)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum FusedPrefillStateSource {
    Gathered,
    Pooled,
}

#[derive(Clone, Copy, Debug)]
enum FusedPrefillProvider {
    Mma,
    MmaFused,
    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    FlashInferSm90,
    #[cfg(feature = "cutile")]
    Cutile,
}

#[derive(Clone, Copy)]
struct FusedPrefillCase {
    provider: FusedPrefillProvider,
    batch_size: usize,
    seq_len: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    tiled_v_heads: bool,
    state_source: FusedPrefillStateSource,
}

fn run_fused_prefill_case(dev: &Device, case: FusedPrefillCase) -> Result<()> {
    const HEAD_DIM: usize = 128;
    const POOLED_PADDING_BATCH: usize = 1;

    let key_dim = case.num_k_heads * HEAD_DIM;
    let value_dim = case.num_v_heads * HEAD_DIM;
    let conv_dim = 2 * key_dim + value_dim;
    let mixed_qkv = tensor3(
        patterned(case.batch_size * case.seq_len * conv_dim, 140, 0.08, 0.01),
        (case.batch_size, case.seq_len, conv_dim),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let b = tensor3(
        patterned(
            case.batch_size * case.seq_len * case.num_v_heads,
            141,
            0.2,
            0.1,
        ),
        (case.batch_size, case.seq_len, case.num_v_heads),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let a = tensor3(
        patterned(
            case.batch_size * case.seq_len * case.num_v_heads,
            142,
            0.18,
            -0.04,
        ),
        (case.batch_size, case.seq_len, case.num_v_heads),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let a_log = Tensor::from_vec(
        patterned(case.num_v_heads, 143, 0.05, -0.2),
        (case.num_v_heads,),
        dev,
    )?;
    let dt_bias = Tensor::from_vec(
        patterned(case.num_v_heads, 144, 0.1, 0.3),
        (case.num_v_heads,),
        dev,
    )?;
    let (state_rows, slots_tensor, active_rows) = match case.state_source {
        FusedPrefillStateSource::Gathered => {
            (case.batch_size, None, (0..case.batch_size).collect())
        }
        FusedPrefillStateSource::Pooled => {
            assert!(case.batch_size >= 3);
            let capacity = case.batch_size + 2;
            let mut slots = (0..case.batch_size)
                .map(|batch| (capacity - 1 - batch) as u32)
                .collect::<Vec<_>>();
            slots[POOLED_PADDING_BATCH] = GDN_PAD_SLOT;
            let active_rows = slots
                .iter()
                .filter(|&&slot| slot != GDN_PAD_SLOT)
                .map(|&slot| slot as usize)
                .collect::<Vec<_>>();
            (
                capacity,
                Some(Tensor::from_vec(slots, (case.batch_size,), dev)?),
                active_rows,
            )
        }
    };
    let initial = Tensor::from_vec(
        patterned(
            state_rows * case.num_v_heads * HEAD_DIM * HEAD_DIM,
            145,
            0.01,
            0.0,
        ),
        (state_rows, case.num_v_heads, HEAD_DIM, HEAD_DIM),
        dev,
    )?
    .to_dtype(DType::F32)?;
    let initial = match case.state_source {
        FusedPrefillStateSource::Gathered => {
            initial.reshape((case.batch_size * case.num_v_heads, HEAD_DIM, HEAD_DIM))?
        }
        FusedPrefillStateSource::Pooled => initial,
    };
    let initial_host = flat(&initial.to_dtype(DType::F32)?)?;
    let slots = GdnStateSlots::from_option(slots_tensor.as_ref());
    let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
        &mixed_qkv,
        &b,
        &a,
        &a_log,
        &dt_bias,
        case.batch_size,
        case.seq_len,
        case.num_k_heads,
        case.num_v_heads,
        HEAD_DIM,
        HEAD_DIM,
        case.tiled_v_heads,
    )?;
    let mut reference_state = initial.copy()?;
    let reference = launch_recurrence(
        RecurrenceKernel::ValueMajorWarp,
        RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        },
        &mut reference_state,
        slots,
    )?;
    let mut fused_state = initial.copy()?;
    let launch = FusedPrefillRecurrence {
        mixed_qkv: &mixed_qkv,
        b: &b,
        a: &a,
        a_log: &a_log,
        dt_bias: &dt_bias,
        state: &mut fused_state,
        batch_size: case.batch_size,
        num_k_heads: case.num_k_heads,
        num_v_heads: case.num_v_heads,
        head_k_dim: HEAD_DIM,
        head_v_dim: HEAD_DIM,
        tiled_v_heads: case.tiled_v_heads,
        state_layout: RecurrentStateLayout::GdnValueMajor,
        slots,
    };
    let fused_output = match case.provider {
        FusedPrefillProvider::Mma => mma_prefill(launch)?,
        FusedPrefillProvider::MmaFused => mma_fused_prefill(launch)?,
        #[cfg(has_flashinfer_gdn_sm90_kernel)]
        FusedPrefillProvider::FlashInferSm90 => flashinfer_sm90_prefill_dispatch(launch)?,
        #[cfg(feature = "cutile")]
        FusedPrefillProvider::Cutile => cutile_prefill(launch)?,
    };
    let fused_output = match fused_output {
        FusedPrefillOutput::TokenMajor(output) => output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((case.batch_size * case.num_v_heads, case.seq_len, HEAD_DIM))?,
    };

    let source = match case.state_source {
        FusedPrefillStateSource::Gathered => "gathered",
        FusedPrefillStateSource::Pooled => "pooled",
    };
    let label = format!(
        "{:?} B={} S={} HK={} HV={} tiled={} {:?} {source}",
        case.provider,
        case.batch_size,
        case.seq_len,
        case.num_k_heads,
        case.num_v_heads,
        case.tiled_v_heads,
        DType::F32,
    );
    let (output_tolerance, state_tolerance) = match case.provider {
        FusedPrefillProvider::Mma | FusedPrefillProvider::MmaFused => (5.0e-4, 5.0e-5),
        #[cfg(has_flashinfer_gdn_sm90_kernel)]
        FusedPrefillProvider::FlashInferSm90 => (2.0e-2, 1.0e-2),
        #[cfg(feature = "cutile")]
        FusedPrefillProvider::Cutile => (2.0e-2, 1.0e-2),
    };
    assert_close(
        &format!("{label} output"),
        &flat(&fused_output.to_dtype(DType::F32)?)?,
        &flat(&reference)?,
        output_tolerance,
    );
    let fused_state_host = flat(&fused_state.to_dtype(DType::F32)?)?;
    let reference_state_host = flat(&reference_state.to_dtype(DType::F32)?)?;
    assert_close(
        &format!("{label} state"),
        &fused_state_host,
        &reference_state_host,
        state_tolerance,
    );

    let row_span = case.num_v_heads * HEAD_DIM * HEAD_DIM;
    for &row in &active_rows {
        for key_tile in 0..HEAD_DIM / 16 {
            let mut tile_changed = false;
            for head in 0..case.num_v_heads {
                for value in 0..HEAD_DIM {
                    let start = row * row_span
                        + head * HEAD_DIM * HEAD_DIM
                        + value * HEAD_DIM
                        + key_tile * 16;
                    if reference_state_host[start..start + 16]
                        .iter()
                        .zip(&initial_host[start..start + 16])
                        .any(|(&updated, &initial)| (updated - initial).abs() > 1.0e-6)
                    {
                        tile_changed = true;
                        break;
                    }
                }
                if tile_changed {
                    break;
                }
            }
            assert!(
                tile_changed,
                "{label} state K tile {key_tile} was not exercised"
            );
        }
    }

    if matches!(case.state_source, FusedPrefillStateSource::Pooled) {
        assert_zero(
            &format!("{label} padding output"),
            &fused_output.narrow(0, POOLED_PADDING_BATCH * case.num_v_heads, case.num_v_heads)?,
        )?;
        for row in 0..state_rows {
            if active_rows.contains(&row) {
                continue;
            }
            assert_close(
                &format!("{label} untouched row {row}"),
                &fused_state_host[row * row_span..(row + 1) * row_span],
                &initial_host[row * row_span..(row + 1) * row_span],
                0.0,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM80 or newer CUDA device"]
fn mma_prefill_matches_sequential_recurrence() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    assert!(mma_prefill_device_supported(dev.as_cuda_device()?)?);
    for provider in [FusedPrefillProvider::Mma, FusedPrefillProvider::MmaFused] {
        for seq_len in [1, 15, 16, 17, 63, 64, 65, 129, 512] {
            for tiled_v_heads in [false, true] {
                run_fused_prefill_case(
                    &dev,
                    FusedPrefillCase {
                        provider,
                        batch_size: 1,
                        seq_len,
                        num_k_heads: 16,
                        num_v_heads: 48,
                        tiled_v_heads,
                        state_source: FusedPrefillStateSource::Gathered,
                    },
                )?;
            }
        }
        run_fused_prefill_case(
            &dev,
            FusedPrefillCase {
                provider,
                batch_size: 3,
                seq_len: 65,
                num_k_heads: 16,
                num_v_heads: 48,
                tiled_v_heads: false,
                state_source: FusedPrefillStateSource::Pooled,
            },
        )?;
    }
    Ok(())
}

const MMA_STRESS_KEY_HEADS: usize = 2;
const MMA_STRESS_VALUE_HEADS: usize = 6;
const MMA_STRESS_HEAD_DIM: usize = 128;
const MMA_STRESS_OUTPUT_TOLERANCE: f32 = 5.0e-4;
const MMA_STRESS_STATE_TOLERANCE: f32 = 5.0e-5;

struct MmaStressInputs {
    mixed: Tensor,
    b: Tensor,
    a: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
}

impl MmaStressInputs {
    fn new(batch: usize, sequence: usize, device: &Device) -> Result<Self> {
        let width = (2 * MMA_STRESS_KEY_HEADS + MMA_STRESS_VALUE_HEADS) * MMA_STRESS_HEAD_DIM;
        let values = (0..batch * sequence * width)
            .map(|index| {
                let token = index / width;
                let channel = index % width;
                let phase = (channel % MMA_STRESS_HEAD_DIM) as f32 * 0.11;
                if channel < 2 * MMA_STRESS_KEY_HEADS * MMA_STRESS_HEAD_DIM {
                    phase.cos() * 0.2 + (token as f32 * 0.07).sin() * 0.01
                } else {
                    (phase + token as f32 * 0.13).sin() * 0.5
                }
            })
            .collect();
        let gates = (0..batch * sequence * MMA_STRESS_VALUE_HEADS)
            .map(|index| [-8.0f32, 0.0, 8.0][index % 3])
            .collect();
        Ok(Self {
            mixed: Tensor::from_vec(values, (batch, sequence, width), device)?
                .to_dtype(DType::BF16)?,
            b: Tensor::from_vec(gates, (batch, sequence, MMA_STRESS_VALUE_HEADS), device)?
                .to_dtype(DType::BF16)?,
            a: Tensor::full(-12.0f32, (batch, sequence, MMA_STRESS_VALUE_HEADS), device)?
                .to_dtype(DType::BF16)?,
            a_log: Tensor::zeros(MMA_STRESS_VALUE_HEADS, DType::F32, device)?,
            dt_bias: Tensor::zeros(MMA_STRESS_VALUE_HEADS, DType::F32, device)?,
        })
    }

    fn chunk(&self, start: usize, length: usize) -> Result<Self> {
        Ok(Self {
            mixed: self.mixed.narrow(1, start, length)?.contiguous()?,
            b: self.b.narrow(1, start, length)?.contiguous()?,
            a: self.a.narrow(1, start, length)?.contiguous()?,
            a_log: self.a_log.clone(),
            dt_bias: self.dt_bias.clone(),
        })
    }

    fn forward(
        &self,
        state: &mut Tensor,
        slots: GdnStateSlots<'_>,
        tiled: bool,
        provider: Option<FusedPrefillProvider>,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = self.mixed.dims3()?;
        if let Some(provider) = provider {
            let launch = FusedPrefillRecurrence {
                mixed_qkv: &self.mixed,
                b: &self.b,
                a: &self.a,
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
                state,
                batch_size: batch,
                num_k_heads: MMA_STRESS_KEY_HEADS,
                num_v_heads: MMA_STRESS_VALUE_HEADS,
                head_k_dim: MMA_STRESS_HEAD_DIM,
                head_v_dim: MMA_STRESS_HEAD_DIM,
                tiled_v_heads: tiled,
                state_layout: RecurrentStateLayout::GdnValueMajor,
                slots,
            };
            let FusedPrefillOutput::TokenMajor(output) = match provider {
                FusedPrefillProvider::Mma => mma_prefill(launch)?,
                FusedPrefillProvider::MmaFused => mma_fused_prefill(launch)?,
                #[cfg(has_flashinfer_gdn_sm90_kernel)]
                FusedPrefillProvider::FlashInferSm90 => unreachable!(),
                #[cfg(feature = "cutile")]
                FusedPrefillProvider::Cutile => unreachable!(),
            };
            return Ok(output);
        }
        let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
            &self.mixed,
            &self.b,
            &self.a,
            &self.a_log,
            &self.dt_bias,
            batch,
            sequence,
            MMA_STRESS_KEY_HEADS,
            MMA_STRESS_VALUE_HEADS,
            MMA_STRESS_HEAD_DIM,
            MMA_STRESS_HEAD_DIM,
            tiled,
        )?;
        launch_recurrence(
            RecurrenceKernel::ValueMajorWarp,
            RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            },
            state,
            slots,
        )?
        .reshape((batch, MMA_STRESS_VALUE_HEADS, sequence, MMA_STRESS_HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()
    }
}

#[test]
#[ignore = "requires an SM80 or newer CUDA device"]
fn mma_prefill_preserves_slow_decay_state_across_prompt_chunks() -> Result<()> {
    const CHUNKS: [usize; 4] = [63, 64, 257, 641];
    let device = Device::new_cuda_with_stream(0)?;
    assert!(mma_prefill_device_supported(device.as_cuda_device()?)?);
    let inputs = MmaStressInputs::new(1, CHUNKS.iter().sum(), &device)?;
    let initial = Tensor::from_vec(
        patterned(
            MMA_STRESS_VALUE_HEADS * MMA_STRESS_HEAD_DIM * MMA_STRESS_HEAD_DIM,
            156,
            0.03,
            0.01,
        ),
        (
            MMA_STRESS_VALUE_HEADS,
            MMA_STRESS_HEAD_DIM,
            MMA_STRESS_HEAD_DIM,
        ),
        &device,
    )?;
    for tiled in [false, true] {
        let mut expected_state = initial.copy()?;
        let expected = inputs.forward(&mut expected_state, GdnStateSlots::Gathered, tiled, None)?;
        for provider in [FusedPrefillProvider::Mma, FusedPrefillProvider::MmaFused] {
            let mut whole_state = initial.copy()?;
            let whole = inputs.forward(
                &mut whole_state,
                GdnStateSlots::Gathered,
                tiled,
                Some(provider),
            )?;
            assert_close(
                "MMA slow-decay full output",
                &flat(&whole.to_dtype(DType::F32)?)?,
                &flat(&expected)?,
                MMA_STRESS_OUTPUT_TOLERANCE,
            );
            assert_close(
                "MMA slow-decay full state",
                &flat(&whole_state)?,
                &flat(&expected_state)?,
                MMA_STRESS_STATE_TOLERANCE,
            );
            let mut chunked_state = initial.copy()?;
            let mut outputs = Vec::new();
            let mut start = 0;
            for length in CHUNKS {
                outputs.push(inputs.chunk(start, length)?.forward(
                    &mut chunked_state,
                    GdnStateSlots::Gathered,
                    tiled,
                    Some(provider),
                )?);
                start += length;
            }
            let output = Tensor::cat(&outputs, 1)?;
            assert_close(
                "MMA slow-decay chunk output",
                &flat(&output.to_dtype(DType::F32)?)?,
                &flat(&expected)?,
                MMA_STRESS_OUTPUT_TOLERANCE,
            );
            assert_close(
                "MMA slow-decay chunk state",
                &flat(&chunked_state)?,
                &flat(&expected_state)?,
                MMA_STRESS_STATE_TOLERANCE,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM80 or newer CUDA device"]
fn mma_prefill_orders_cross_stream_state_updates() -> Result<()> {
    const SEQUENCE: usize = 1_025;
    let input_device = Device::new_cuda_with_stream(0)?;
    let state_device = Device::new_cuda_with_stream(0)?;
    assert!(mma_prefill_device_supported(
        input_device.as_cuda_device()?
    )?);
    let inputs = MmaStressInputs::new(1, SEQUENCE, &input_device)?;
    let shape = (
        MMA_STRESS_VALUE_HEADS,
        MMA_STRESS_HEAD_DIM,
        MMA_STRESS_HEAD_DIM,
    );
    let values = patterned(
        MMA_STRESS_VALUE_HEADS * MMA_STRESS_HEAD_DIM * MMA_STRESS_HEAD_DIM,
        158,
        0.03,
        0.01,
    );
    let mut reference_state = Tensor::from_vec(values.clone(), shape, &input_device)?;
    let _ = inputs.forward(&mut reference_state, GdnStateSlots::Gathered, false, None)?;
    let expected = flat(&reference_state)?;
    for provider in [FusedPrefillProvider::Mma, FusedPrefillProvider::MmaFused] {
        let mut state = Tensor::from_vec(values.clone(), shape, &state_device)?;
        let _output = inputs.forward(&mut state, GdnStateSlots::Gathered, false, Some(provider))?;
        assert_close(
            "MMA cross-stream state",
            &flat(&state)?,
            &expected,
            MMA_STRESS_STATE_TOLERANCE,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM80 or newer CUDA device"]
fn mma_prefill_graph_replay_tracks_changed_pooled_slots() -> Result<()> {
    use candle_core::{cuda::cudarc::driver::sys, Var};
    const BATCH: usize = 3;
    const SEQUENCE: usize = 65;
    const CAPACITY: usize = 5;
    const SLOT_CASES: [[u32; BATCH]; 4] = [
        [4, GDN_PAD_SLOT, 1],
        [0, 3, GDN_PAD_SLOT],
        [GDN_PAD_SLOT, 2, 4],
        [4, GDN_PAD_SLOT, 1],
    ];
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    assert!(mma_prefill_device_supported(cuda)?);
    let _htod_cache = cuda.enable_cuda_graph_htod_cache();
    let inputs = MmaStressInputs::new(BATCH, SEQUENCE, &device)?;
    let initial = Tensor::from_vec(
        patterned(
            CAPACITY * MMA_STRESS_VALUE_HEADS * MMA_STRESS_HEAD_DIM * MMA_STRESS_HEAD_DIM,
            157,
            0.03,
            0.01,
        ),
        (
            CAPACITY,
            MMA_STRESS_VALUE_HEADS,
            MMA_STRESS_HEAD_DIM,
            MMA_STRESS_HEAD_DIM,
        ),
        &device,
    )?;
    for provider in [FusedPrefillProvider::Mma, FusedPrefillProvider::MmaFused] {
        let slots = Var::from_vec(SLOT_CASES[0].to_vec(), BATCH, &device)?;
        let state_variable = Var::from_tensor(&initial)?;
        let mut state = state_variable.as_tensor().clone();
        let _ = inputs.forward(
            &mut state,
            GdnStateSlots::Pooled(&slots),
            true,
            Some(provider),
        )?;
        state_variable.set(&initial)?;
        device.synchronize()?;
        let stream = cuda.cuda_stream();
        let tracking = stream.context().is_event_tracking();
        if tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        if let Err(error) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::wrap(error));
        }
        let output = inputs.forward(
            &mut state,
            GdnStateSlots::Pooled(&slots),
            true,
            Some(provider),
        );
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?;
        let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
        let mut expected_state = initial.copy()?;
        for slot_values in SLOT_CASES {
            slots.set(&Tensor::from_vec(slot_values.to_vec(), BATCH, &device)?)?;
            graph.launch().map_err(candle_core::Error::wrap)?;
            device.synchronize()?;
            let expected = inputs.forward(
                &mut expected_state,
                GdnStateSlots::Pooled(&slots),
                true,
                None,
            )?;
            assert_close(
                "MMA replay output",
                &flat(&output.to_dtype(DType::F32)?)?,
                &flat(&expected)?,
                MMA_STRESS_OUTPUT_TOLERANCE,
            );
            assert_close(
                "MMA replay state pool",
                &flat(&state)?,
                &flat(&expected_state)?,
                MMA_STRESS_STATE_TOLERANCE,
            );
        }
    }
    Ok(())
}

#[cfg(has_flashinfer_gdn_sm90_kernel)]
#[test]
fn flashinfer_sm90_gathered_workspace_excludes_packed_state() {
    const BATCH_SIZE: i32 = 3;
    const SEQ_LEN: i32 = 65;
    const NUM_K_HEADS: i32 = 16;
    const NUM_V_HEADS: i32 = 48;
    const SM_COUNT: i32 = 132;

    let gathered = unsafe {
        crate::cuda::ffi::mistralrs_flashinfer_gdn_sm90_workspace_size(
            BATCH_SIZE,
            SEQ_LEN,
            NUM_K_HEADS,
            NUM_V_HEADS,
            SM_COUNT,
            0,
        )
    };
    let pooled = unsafe {
        crate::cuda::ffi::mistralrs_flashinfer_gdn_sm90_workspace_size(
            BATCH_SIZE,
            SEQ_LEN,
            NUM_K_HEADS,
            NUM_V_HEADS,
            SM_COUNT,
            1,
        )
    };
    let packed_state_bytes = BATCH_SIZE as u64
        * NUM_V_HEADS as u64
        * GDN_DECODE_K_DIM as u64
        * GDN_DECODE_V_DIM as u64
        * std::mem::size_of::<f32>() as u64;

    assert_ne!(gathered, 0);
    assert_eq!(pooled - gathered, packed_state_bytes);
}

#[cfg(has_flashinfer_gdn_sm90_kernel)]
#[test]
#[ignore = "requires an SM90 CUDA device"]
fn flashinfer_sm90_prefill_matches_sequential_recurrence() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for case in [
        FusedPrefillCase {
            provider: FusedPrefillProvider::FlashInferSm90,
            batch_size: 3,
            seq_len: 65,
            num_k_heads: 16,
            num_v_heads: 48,
            tiled_v_heads: false,
            state_source: FusedPrefillStateSource::Pooled,
        },
        FusedPrefillCase {
            provider: FusedPrefillProvider::FlashInferSm90,
            batch_size: 1,
            seq_len: 129,
            num_k_heads: 16,
            num_v_heads: 48,
            tiled_v_heads: false,
            state_source: FusedPrefillStateSource::Gathered,
        },
    ] {
        run_fused_prefill_case(&dev, case)?;
    }
    Ok(())
}

#[cfg(feature = "cutile")]
#[test]
#[ignore = "requires a CUDA device"]
fn cutile_prefill_matches_sequential_recurrence() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for case in [
        FusedPrefillCase {
            provider: FusedPrefillProvider::Cutile,
            batch_size: 3,
            seq_len: 65,
            num_k_heads: 16,
            num_v_heads: 48,
            tiled_v_heads: false,
            state_source: FusedPrefillStateSource::Pooled,
        },
        FusedPrefillCase {
            provider: FusedPrefillProvider::Cutile,
            batch_size: 1,
            seq_len: 129,
            num_k_heads: 16,
            num_v_heads: 48,
            tiled_v_heads: false,
            state_source: FusedPrefillStateSource::Gathered,
        },
        FusedPrefillCase {
            provider: FusedPrefillProvider::Cutile,
            batch_size: 2,
            seq_len: 512,
            num_k_heads: 16,
            num_v_heads: 48,
            tiled_v_heads: true,
            state_source: FusedPrefillStateSource::Gathered,
        },
    ] {
        run_fused_prefill_case(&dev, case)?;
    }
    Ok(())
}

fn run_speculative_state_commit_case(
    dev: &Device,
    state_layout: RecurrentStateLayout,
    state_dtype: DType,
) -> Result<()> {
    let batch_size = 3;
    let seq_len = 4;
    let num_k_heads = 1;
    let num_v_heads = 2;
    let head_k_dim = 128;
    let head_v_dim = 128;
    let kernel_size = 4;
    let capacity = 5;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let row_state_elements = num_v_heads * head_k_dim * head_v_dim;
    let row_conv_elements = conv_dim * kernel_size;
    let keep_rows_host = vec![1u32, 3, 0];
    let slots_host = vec![4u32, 1, 3];

    let mixed_qkv = tensor3(
        patterned(batch_size * seq_len * conv_dim, 110, 0.08, 0.01),
        (batch_size, seq_len, conv_dim),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let conv_weight = tensor2(
        patterned(conv_dim * kernel_size, 111, 0.05, -0.01),
        (conv_dim, kernel_size),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let initial_conv_state = tensor3(
        patterned(batch_size * row_conv_elements, 112, 0.03, 0.0),
        (batch_size, conv_dim, kernel_size),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let (convolved_qkv, _) = causal_conv1d_cuda(
        &mixed_qkv,
        &conv_weight,
        &initial_conv_state,
        kernel_size,
        false,
        GdnStateSlots::Gathered,
    )?;
    let b = tensor3(
        patterned(batch_size * seq_len * num_v_heads, 113, 0.2, 0.1),
        (batch_size, seq_len, num_v_heads),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let a = tensor3(
        patterned(batch_size * seq_len * num_v_heads, 114, 0.18, -0.04),
        (batch_size, seq_len, num_v_heads),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let a_log = Tensor::from_vec(patterned(num_v_heads, 115, 0.05, -0.2), (num_v_heads,), dev)?;
    let dt_bias = Tensor::from_vec(patterned(num_v_heads, 116, 0.1, 0.3), (num_v_heads,), dev)?;
    let state_shape = match state_layout {
        RecurrentStateLayout::GdnKeyMajor => {
            vec![batch_size, num_v_heads, head_k_dim, head_v_dim]
        }
        RecurrentStateLayout::GdnValueMajor => {
            vec![batch_size, num_v_heads, head_v_dim, head_k_dim]
        }
        RecurrentStateLayout::Opaque => unreachable!(),
    };
    let pool_shape = match state_layout {
        RecurrentStateLayout::GdnKeyMajor => {
            vec![capacity, num_v_heads, head_k_dim, head_v_dim]
        }
        RecurrentStateLayout::GdnValueMajor => {
            vec![capacity, num_v_heads, head_v_dim, head_k_dim]
        }
        RecurrentStateLayout::Opaque => unreachable!(),
    };
    let initial_recurrent_state = Tensor::from_vec(
        patterned(batch_size * row_state_elements, 117, 0.02, 0.0),
        state_shape,
        dev,
    )?
    .to_dtype(state_dtype)?;
    let conv_state_pool = Tensor::from_vec(
        patterned(capacity * row_conv_elements, 118, 0.04, 0.0),
        (capacity, conv_dim, kernel_size),
        dev,
    )?
    .to_dtype(DType::BF16)?;
    let recurrent_state_pool = Tensor::from_vec(
        patterned(capacity * row_state_elements, 119, 0.02, 0.0),
        pool_shape,
        dev,
    )?
    .to_dtype(state_dtype)?;
    let mut expected_conv = flat(&conv_state_pool.to_dtype(DType::F32)?)?;
    let mut expected_recurrent = flat(&recurrent_state_pool.to_dtype(DType::F32)?)?;

    for batch_idx in 0..batch_size {
        let rows = keep_rows_host[batch_idx] as usize;
        if rows == 0 {
            continue;
        }
        let mixed_row = mixed_qkv.narrow(0, batch_idx, 1)?.narrow(1, 0, rows)?;
        let initial_conv_row = initial_conv_state.narrow(0, batch_idx, 1)?;
        let (_, conv_state) = causal_conv1d_cuda(
            &mixed_row,
            &conv_weight,
            &initial_conv_row,
            kernel_size,
            false,
            GdnStateSlots::Gathered,
        )?;
        let conv_state = flat(&conv_state.to_dtype(DType::F32)?)?;
        let conv_destination = slots_host[batch_idx] as usize * row_conv_elements;
        expected_conv[conv_destination..conv_destination + row_conv_elements]
            .copy_from_slice(&conv_state);

        let convolved_row = convolved_qkv.narrow(0, batch_idx, 1)?.narrow(1, 0, rows)?;
        let b_row = b.narrow(0, batch_idx, 1)?.narrow(1, 0, rows)?;
        let a_row = a.narrow(0, batch_idx, 1)?.narrow(1, 0, rows)?;
        let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
            &convolved_row,
            &b_row,
            &a_row,
            &a_log,
            &dt_bias,
            1,
            rows,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            false,
        )?;
        let mut state = initial_recurrent_state.narrow(0, batch_idx, 1)?.copy()?;
        let inputs = RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        };
        if state_layout == RecurrentStateLayout::GdnValueMajor {
            vmajor_warp_gated_delta_rule_recurrence_cuda(
                inputs,
                &mut state,
                GdnStateSlots::Gathered,
            )?;
        } else {
            gated_delta_rule_recurrence_cuda(inputs, &mut state, GdnStateSlots::Gathered)?;
        }
        let state = flat(&state.to_dtype(DType::F32)?)?;
        let state_destination = slots_host[batch_idx] as usize * row_state_elements;
        expected_recurrent[state_destination..state_destination + row_state_elements]
            .copy_from_slice(&state);
    }

    speculative_state_commit_cuda(GdnSpeculativeStateCommit {
        mixed_qkv: &mixed_qkv,
        convolved_qkv: &convolved_qkv,
        b: &b,
        a: &a,
        initial_conv_state: &initial_conv_state,
        initial_recurrent_state: &initial_recurrent_state,
        a_log: &a_log,
        dt_bias: &dt_bias,
        conv_state_pool: &conv_state_pool,
        recurrent_state_pool: &recurrent_state_pool,
        keep_rows: &Tensor::from_vec(keep_rows_host, (batch_size,), dev)?,
        slot_indices: &Tensor::from_vec(slots_host.clone(), (batch_size,), dev)?,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tiled_v_heads: false,
        state_layout,
    })?;
    assert_close(
        "speculative conv state",
        &flat(&conv_state_pool.to_dtype(DType::F32)?)?,
        &expected_conv,
        0.0,
    );
    assert_close(
        "speculative recurrent state",
        &flat(&recurrent_state_pool.to_dtype(DType::F32)?)?,
        &expected_recurrent,
        2.0e-4,
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn speculative_state_commit_matches_prefix_replay_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for state_dtype in [DType::F32, DType::BF16, DType::F16] {
        run_speculative_state_commit_case(&dev, RecurrentStateLayout::GdnKeyMajor, state_dtype)?;
        run_speculative_state_commit_case(&dev, RecurrentStateLayout::GdnValueMajor, state_dtype)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn speculative_checkpoint_kernels_match_serial_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let batch_size = 3;
    let seq_len = 8;
    let checkpoint_lanes = 8;
    let capacity = batch_size * checkpoint_lanes;
    let active_slots_host = vec![7u32, 14, GDN_PAD_SLOT];
    let active_slots = Tensor::from_vec(active_slots_host.clone(), (batch_size,), &dev)?;

    let conv_dim = 37;
    let conv_storage_dim = conv_dim + 5;
    let kernel_size = 4;
    let x = tensor3(
        patterned(batch_size * seq_len * conv_storage_dim, 130, 0.08, 0.01),
        (batch_size, seq_len, conv_storage_dim),
        &dev,
    )?
    .to_dtype(DType::BF16)?
    .narrow(2, 2, conv_dim)?;
    assert!(!x.is_contiguous());
    let weight = tensor2(
        patterned(conv_dim * kernel_size, 131, 0.05, -0.01),
        (conv_dim, kernel_size),
        &dev,
    )?
    .to_dtype(DType::BF16)?;
    let conv_pool = tensor3(
        patterned(capacity * conv_dim * kernel_size, 132, 0.03, 0.0),
        (capacity, conv_dim, kernel_size),
        &dev,
    )?
    .to_dtype(DType::BF16)?;
    let mut expected_conv_pool = flat(&conv_pool.to_dtype(DType::F32)?)?;
    let mut expected_conv_outputs = Vec::with_capacity(batch_size);
    for (batch_idx, &active_slot) in active_slots_host.iter().enumerate() {
        if active_slot == GDN_PAD_SLOT {
            expected_conv_outputs.push(Tensor::zeros((1, seq_len, conv_dim), DType::BF16, &dev)?);
            continue;
        }
        let input = x.narrow(0, batch_idx, 1)?;
        let initial = conv_pool.narrow(0, active_slot as usize, 1)?.copy()?;
        let (output, _) = causal_conv1d_cuda(
            &input,
            &weight,
            &initial,
            kernel_size,
            false,
            GdnStateSlots::Gathered,
        )?;
        expected_conv_outputs.push(output);
        let base_slot = active_slot as usize / checkpoint_lanes * checkpoint_lanes;
        for position in 0..seq_len {
            let (_, state) = causal_conv1d_cuda(
                &input.narrow(1, 0, position + 1)?,
                &weight,
                &initial,
                kernel_size,
                false,
                GdnStateSlots::Gathered,
            )?;
            let state = flat(&state.to_dtype(DType::F32)?)?;
            let destination = (base_slot + position) * conv_dim * kernel_size;
            expected_conv_pool[destination..destination + state.len()].copy_from_slice(&state);
        }
    }
    let expected_conv_output = Tensor::cat(&expected_conv_outputs, 0)?;
    let actual_conv_output = speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
        x: &x,
        weight: &weight,
        state_pool: &conv_pool,
        active_slots: &active_slots,
        checkpoint_lanes,
        write_checkpoints: true,
        pending: None,
    })?;
    assert_close(
        "speculative checkpoint convolution output",
        &flat(&actual_conv_output.to_dtype(DType::F32)?)?,
        &flat(&expected_conv_output.to_dtype(DType::F32)?)?,
        2.0e-3,
    );
    assert_close(
        "speculative checkpoint convolution state",
        &flat(&conv_pool.to_dtype(DType::F32)?)?,
        &expected_conv_pool,
        0.0,
    );

    let num_k_heads = 2;
    let num_v_heads = 4;
    let head_k_dim = 128;
    let head_v_dim = 128;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let recurrent_conv_dim = 2 * key_dim + value_dim;
    let mixed_qkv = tensor3(
        patterned(batch_size * seq_len * recurrent_conv_dim, 133, 0.08, 0.01),
        (batch_size, seq_len, recurrent_conv_dim),
        &dev,
    )?
    .to_dtype(DType::BF16)?;
    let b_storage_width = num_v_heads + 3;
    let b = tensor3(
        patterned(batch_size * seq_len * b_storage_width, 134, 0.2, 0.1),
        (batch_size, seq_len, b_storage_width),
        &dev,
    )?
    .to_dtype(DType::BF16)?
    .narrow(2, 2, num_v_heads)?;
    let a_storage_width = num_v_heads + 5;
    let a = tensor3(
        patterned(batch_size * seq_len * a_storage_width, 135, 0.18, -0.04),
        (batch_size, seq_len, a_storage_width),
        &dev,
    )?
    .to_dtype(DType::BF16)?
    .narrow(2, 3, num_v_heads)?;
    let a_log = Tensor::from_vec(
        patterned(num_v_heads, 136, 0.05, -0.2),
        (num_v_heads,),
        &dev,
    )?;
    let dt_bias = Tensor::from_vec(patterned(num_v_heads, 137, 0.1, 0.3), (num_v_heads,), &dev)?;
    let norm_eps = 1.0e-6;
    let norm_weight = Tensor::from_vec(patterned(head_v_dim, 139, 0.1, 1.0), (head_v_dim,), &dev)?
        .to_dtype(DType::BF16)?;
    let gate = Tensor::from_vec(
        patterned(
            batch_size * seq_len * num_v_heads * head_v_dim,
            140,
            0.2,
            0.0,
        ),
        (batch_size, seq_len, num_v_heads, head_v_dim),
        &dev,
    )?
    .to_dtype(DType::BF16)?;
    let state_elements = num_v_heads * head_v_dim * head_k_dim;
    for state_layout in [
        RecurrentStateLayout::GdnKeyMajor,
        RecurrentStateLayout::GdnValueMajor,
    ] {
        for state_dtype in [DType::F32, DType::BF16, DType::F16] {
            let recurrent_pool = Tensor::from_vec(
                patterned(capacity * state_elements, 138, 0.02, 0.0),
                (capacity, num_v_heads, head_v_dim, head_k_dim),
                &dev,
            )?
            .to_dtype(state_dtype)?;
            let mut expected_recurrent_pool = flat(&recurrent_pool.to_dtype(DType::F32)?)?;
            let mut expected_recurrent_outputs = Vec::with_capacity(batch_size);
            for (batch_idx, &active_slot) in active_slots_host.iter().enumerate() {
                if active_slot == GDN_PAD_SLOT {
                    expected_recurrent_outputs.push(Tensor::zeros(
                        (num_v_heads, seq_len, head_v_dim),
                        DType::F32,
                        &dev,
                    )?);
                    continue;
                }
                let mixed_row = mixed_qkv.narrow(0, batch_idx, 1)?;
                let b_row = b.narrow(0, batch_idx, 1)?;
                let a_row = a.narrow(0, batch_idx, 1)?;
                let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
                    &mixed_row,
                    &b_row,
                    &a_row,
                    &a_log,
                    &dt_bias,
                    1,
                    seq_len,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    true,
                )?;
                let initial = recurrent_pool.narrow(0, active_slot as usize, 1)?.copy()?;
                let mut final_state = initial.copy()?;
                let inputs = RecurrenceInputs {
                    q: &q,
                    k: &k,
                    v: &v,
                    g: &g,
                    beta: &beta,
                };
                let output = if state_layout == RecurrentStateLayout::GdnValueMajor {
                    vmajor_warp_gated_delta_rule_recurrence_cuda(
                        inputs,
                        &mut final_state,
                        GdnStateSlots::Gathered,
                    )?
                } else {
                    gated_delta_rule_recurrence_cuda(
                        inputs,
                        &mut final_state,
                        GdnStateSlots::Gathered,
                    )?
                };
                expected_recurrent_outputs.push(output);

                let base_slot = active_slot as usize / checkpoint_lanes * checkpoint_lanes;
                for position in 0..seq_len {
                    let prefix_len = position + 1;
                    let q_prefix = q.narrow(1, 0, prefix_len)?.contiguous()?;
                    let k_prefix = k.narrow(1, 0, prefix_len)?.contiguous()?;
                    let v_prefix = v.narrow(1, 0, prefix_len)?.contiguous()?;
                    let g_prefix = g.narrow(1, 0, prefix_len)?.contiguous()?;
                    let beta_prefix = beta.narrow(1, 0, prefix_len)?.contiguous()?;
                    let mut checkpoint_state = initial.copy()?;
                    let prefix_inputs = RecurrenceInputs {
                        q: &q_prefix,
                        k: &k_prefix,
                        v: &v_prefix,
                        g: &g_prefix,
                        beta: &beta_prefix,
                    };
                    if state_layout == RecurrentStateLayout::GdnValueMajor {
                        vmajor_warp_gated_delta_rule_recurrence_cuda(
                            prefix_inputs,
                            &mut checkpoint_state,
                            GdnStateSlots::Gathered,
                        )?;
                    } else {
                        gated_delta_rule_recurrence_cuda(
                            prefix_inputs,
                            &mut checkpoint_state,
                            GdnStateSlots::Gathered,
                        )?;
                    }
                    let checkpoint_state = flat(&checkpoint_state.to_dtype(DType::F32)?)?;
                    let destination = (base_slot + position) * state_elements;
                    expected_recurrent_pool[destination..destination + checkpoint_state.len()]
                        .copy_from_slice(&checkpoint_state);
                }
            }
            let expected_recurrent_output = Tensor::cat(&expected_recurrent_outputs, 0)?;
            let fused_pool = recurrent_pool.copy()?;
            let quantized_pool = (state_layout == RecurrentStateLayout::GdnValueMajor
                && state_dtype == DType::F32)
                .then(|| fused_pool.copy())
                .transpose()?;
            let actual_recurrent_output =
                speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                    mixed_qkv: &mixed_qkv,
                    b: &b,
                    a: &a,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    state_pool: &recurrent_pool,
                    active_slots: &active_slots,
                    checkpoint_lanes,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    tiled_v_heads: true,
                    state_layout,
                    post_op: None,
                    record_transitions: false,
                    pending: None,
                })?;
            let actual_recurrent_output = actual_recurrent_output.output.into_tensor()?;
            let label = format!("{state_layout:?} {state_dtype:?}");
            assert_close(
                &format!("{label} speculative checkpoint recurrence output"),
                &flat(&actual_recurrent_output.to_dtype(DType::F32)?)?,
                &flat(&expected_recurrent_output)?,
                3.0e-3,
            );
            assert_close(
                &format!("{label} speculative checkpoint recurrence state"),
                &flat(&recurrent_pool.to_dtype(DType::F32)?)?,
                &expected_recurrent_pool,
                2.0e-3,
            );
            if state_layout == RecurrentStateLayout::GdnValueMajor {
                let expected_normalized = expected_recurrent_output
                    .reshape((batch_size, num_v_heads, seq_len, head_v_dim))?
                    .transpose(1, 2)?
                    .to_dtype(DType::BF16)?;
                let expected_normalized = rmsnorm_gated_cuda(
                    &expected_normalized,
                    &gate,
                    &norm_weight,
                    norm_eps,
                    GdnGateActivation::Silu,
                )?;
                let actual_normalized =
                    speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                        mixed_qkv: &mixed_qkv,
                        b: &b,
                        a: &a,
                        a_log: &a_log,
                        dt_bias: &dt_bias,
                        state_pool: &fused_pool,
                        active_slots: &active_slots,
                        checkpoint_lanes,
                        num_k_heads,
                        num_v_heads,
                        head_k_dim,
                        head_v_dim,
                        tiled_v_heads: true,
                        state_layout,
                        post_op: Some(GdnSpeculativeRmsNormGate {
                            sigmoid_gate: false,
                            gate: &gate,
                            weight: &norm_weight,
                            eps: norm_eps,
                            quantization: None,
                        }),
                        record_transitions: false,
                        pending: None,
                    })?;
                let actual_normalized = actual_normalized.output.into_tensor()?;
                assert_close(
                    &format!("{label} fused speculative normalization state"),
                    &flat(&fused_pool.to_dtype(DType::F32)?)?,
                    &expected_recurrent_pool,
                    2.0e-3,
                );
                assert_close(
                    &format!("{label} fused speculative normalization output"),
                    &flat(&actual_normalized.to_dtype(DType::F32)?)?,
                    &flat(&expected_normalized.to_dtype(DType::F32)?)?,
                    3.0e-2,
                );
                if let Some(quantized_pool) = quantized_pool.as_ref() {
                    let scale_layout = ActivationScaleLayout::GroupMajor {
                        row_alignment: std::num::NonZeroUsize::new(4).unwrap(),
                    };
                    let quantization = GdnFp8OutputSpec::new(
                        [batch_size, seq_len, value_dim],
                        ActivationQuantizationScheme {
                            dtype: DType::F8E4M3,
                            scale_encoding: mistralrs_quant::ActivationScaleEncoding::F32,
                            block_shape: [1, GDN_FP8_GROUP_SIZE],
                        },
                        scale_layout,
                        num_v_heads,
                        head_v_dim,
                    )
                    .unwrap();
                    let actual_quantized = speculative_recurrence_checkpoints_cuda(
                        GdnSpeculativeRecurrenceCheckpoints {
                            mixed_qkv: &mixed_qkv,
                            b: &b,
                            a: &a,
                            a_log: &a_log,
                            dt_bias: &dt_bias,
                            state_pool: quantized_pool,
                            active_slots: &active_slots,
                            checkpoint_lanes,
                            num_k_heads,
                            num_v_heads,
                            head_k_dim,
                            head_v_dim,
                            tiled_v_heads: true,
                            state_layout,
                            post_op: Some(GdnSpeculativeRmsNormGate {
                                sigmoid_gate: false,
                                gate: &gate,
                                weight: &norm_weight,
                                eps: norm_eps,
                                quantization: Some(quantization),
                            }),
                            record_transitions: false,
                            pending: None,
                        },
                    )?;
                    let GdnPostOpOutput::Quantized(actual_quantized) = actual_quantized.output
                    else {
                        panic!("speculative FP8 post-op returned BF16 output")
                    };
                    assert_gdn_quantized_matches_bf16(
                        &format!("{label} quantized speculative normalization"),
                        &actual_quantized,
                        &actual_normalized,
                        batch_size * seq_len,
                        num_v_heads,
                        scale_layout,
                    )?;
                    assert_close(
                        &format!("{label} quantized speculative normalization state"),
                        &flat(&quantized_pool.to_dtype(DType::F32)?)?,
                        &expected_recurrent_pool,
                        2.0e-3,
                    );
                }
            }
        }
    }

    Ok(())
}

fn run_speculative_transition_commit_case(
    dev: &Device,
    seq_len: usize,
    activation_dtype: DType,
    state_dtype: DType,
    tiled_v_heads: bool,
) -> Result<()> {
    struct LayerCase {
        conv_input: Tensor,
        weight: Tensor,
        b: Tensor,
        a: Tensor,
        a_log: Tensor,
        dt_bias: Tensor,
        gate: Tensor,
        norm_weight: Tensor,
        transitions: GdnSpeculativeTransitions,
        conv_state: Tensor,
        recurrent_state: Tensor,
        pending_conv_input: Tensor,
        pending_key: Tensor,
        pending_key_banks: Tensor,
        pending_key_bank: Tensor,
        pending_delta: Tensor,
        pending_decay: Tensor,
        pending_keep_rows: Tensor,
        pending_epochs: Tensor,
        conv_applied_epochs: Tensor,
        recurrent_applied_epochs: Tensor,
        staged_conv_state: Tensor,
        staged_recurrent_state: Tensor,
        lazy_conv_applied_epochs: Tensor,
        lazy_recurrent_applied_epochs: Tensor,
        lazy_conv_state: Tensor,
        lazy_recurrent_state: Tensor,
        expected_conv_state: Tensor,
        expected_recurrent_state: Tensor,
    }

    let batch_size = seq_len + 2;
    let capacity = 2 * batch_size + 3;
    let num_k_heads = 2;
    let num_v_heads = 4;
    let head_k_dim = 128;
    let head_v_dim = 128;
    let conv_width = 4;
    let conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    let mut active_slots_host = (0..=seq_len)
        .map(|row| ((row * 2 + 1) % capacity) as u32)
        .collect::<Vec<_>>();
    active_slots_host.push(GDN_PAD_SLOT);
    let mut keep_rows_host = (1..=seq_len).map(|rows| rows as u32).collect::<Vec<_>>();
    keep_rows_host.push(0);
    keep_rows_host.push(0);
    let active_slots = Tensor::from_vec(active_slots_host.clone(), (batch_size,), dev)?;
    let keep_rows = Tensor::from_vec(keep_rows_host.clone(), (batch_size,), dev)?;
    let mut cases = Vec::new();

    for layer_idx in 0..2 {
        let seed = 200 + layer_idx * 20 + seq_len;
        let conv_input = tensor3(
            patterned(batch_size * seq_len * conv_dim, seed, 0.08, 0.01),
            (batch_size, seq_len, conv_dim),
            dev,
        )?
        .to_dtype(activation_dtype)?;
        let weight = tensor2(
            patterned(conv_dim * conv_width, seed + 1, 0.05, -0.01),
            (conv_dim, conv_width),
            dev,
        )?
        .to_dtype(activation_dtype)?;
        let conv_state = tensor3(
            patterned(capacity * conv_dim * conv_width, seed + 2, 0.03, 0.0),
            (capacity, conv_dim, conv_width),
            dev,
        )?
        .to_dtype(activation_dtype)?;
        let recurrent_state = Tensor::from_vec(
            patterned(
                capacity * num_v_heads * head_v_dim * head_k_dim,
                seed + 3,
                0.02,
                0.0,
            ),
            (capacity, num_v_heads, head_v_dim, head_k_dim),
            dev,
        )?
        .to_dtype(state_dtype)?;
        let initial_conv = flat(&conv_state.to_dtype(DType::F32)?)?;
        let initial_recurrent = flat(&recurrent_state.to_dtype(DType::F32)?)?;
        let convolved = speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
            x: &conv_input,
            weight: &weight,
            state_pool: &conv_state,
            active_slots: &active_slots,
            checkpoint_lanes: 1,
            write_checkpoints: false,
            pending: None,
        })?;
        assert_close(
            "transition convolution leaves state untouched",
            &flat(&conv_state.to_dtype(DType::F32)?)?,
            &initial_conv,
            0.0,
        );

        let b = tensor3(
            patterned(batch_size * seq_len * num_v_heads, seed + 4, 0.2, 0.1),
            (batch_size, seq_len, num_v_heads),
            dev,
        )?
        .to_dtype(activation_dtype)?;
        let a = tensor3(
            patterned(batch_size * seq_len * num_v_heads, seed + 5, 0.18, -0.04),
            (batch_size, seq_len, num_v_heads),
            dev,
        )?
        .to_dtype(activation_dtype)?;
        let a_log = Tensor::from_vec(
            patterned(num_v_heads, seed + 6, 0.05, -0.2),
            (num_v_heads,),
            dev,
        )?;
        let dt_bias = Tensor::from_vec(
            patterned(num_v_heads, seed + 7, 0.1, 0.3),
            (num_v_heads,),
            dev,
        )?;
        let gate = Tensor::from_vec(
            patterned(
                batch_size * seq_len * num_v_heads * head_v_dim,
                seed + 8,
                0.2,
                0.0,
            ),
            (batch_size, seq_len, num_v_heads, head_v_dim),
            dev,
        )?
        .to_dtype(activation_dtype)?;
        let norm_weight = Tensor::from_vec(
            patterned(head_v_dim, seed + 9, 0.1, 1.0),
            (head_v_dim,),
            dev,
        )?
        .to_dtype(activation_dtype)?;
        let recurrence =
            speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                mixed_qkv: &convolved,
                b: &b,
                a: &a,
                a_log: &a_log,
                dt_bias: &dt_bias,
                state_pool: &recurrent_state,
                active_slots: &active_slots,
                checkpoint_lanes: 1,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                tiled_v_heads,
                state_layout: RecurrentStateLayout::GdnValueMajor,
                post_op: Some(GdnSpeculativeRmsNormGate {
                    sigmoid_gate: false,
                    gate: &gate,
                    weight: &norm_weight,
                    eps: 1.0e-6,
                    quantization: None,
                }),
                record_transitions: true,
                pending: None,
            })?;
        assert_close(
            "transition recurrence leaves state untouched",
            &flat(&recurrent_state.to_dtype(DType::F32)?)?,
            &initial_recurrent,
            0.0,
        );

        let expected_conv_state = conv_state.copy()?;
        let expected_recurrent_state = recurrent_state.copy()?;
        let initial_conv_rows = active_slots_host
            .iter()
            .map(|&slot| {
                if slot == GDN_PAD_SLOT {
                    Tensor::zeros((1, conv_dim, conv_width), activation_dtype, dev)
                } else {
                    conv_state.narrow(0, slot as usize, 1)?.copy()
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let initial_recurrent_rows = active_slots_host
            .iter()
            .map(|&slot| {
                if slot == GDN_PAD_SLOT {
                    Tensor::zeros((1, num_v_heads, head_v_dim, head_k_dim), state_dtype, dev)
                } else {
                    recurrent_state.narrow(0, slot as usize, 1)?.copy()
                }
            })
            .collect::<Result<Vec<_>>>()?;
        speculative_state_commit_cuda(GdnSpeculativeStateCommit {
            mixed_qkv: &conv_input,
            convolved_qkv: &convolved,
            b: &b,
            a: &a,
            initial_conv_state: &Tensor::cat(&initial_conv_rows, 0)?,
            initial_recurrent_state: &Tensor::cat(&initial_recurrent_rows, 0)?,
            a_log: &a_log,
            dt_bias: &dt_bias,
            conv_state_pool: &expected_conv_state,
            recurrent_state_pool: &expected_recurrent_state,
            keep_rows: &keep_rows,
            slot_indices: &active_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            state_layout: RecurrentStateLayout::GdnValueMajor,
        })?;
        let pending_conv_input =
            Tensor::zeros((capacity, seq_len, conv_dim), activation_dtype, dev)?;
        let pending_key_banks = Tensor::zeros(
            (2, capacity, seq_len, num_k_heads, head_k_dim),
            DType::F32,
            dev,
        )?;
        let pending_key = pending_key_banks.i(0)?;
        let pending_key_bank = Tensor::zeros(capacity, DType::U32, dev)?;
        let pending_delta = Tensor::zeros(
            (capacity, seq_len, num_v_heads, head_v_dim),
            DType::F32,
            dev,
        )?;
        let pending_decay = Tensor::zeros((capacity, seq_len, num_v_heads), DType::F32, dev)?;
        let pending_keep_rows = Tensor::zeros(capacity, DType::U32, dev)?;
        let pending_epochs = Tensor::zeros(capacity, DType::U32, dev)?;
        let conv_applied_epochs = Tensor::zeros(
            (capacity, conv_dim.div_ceil(GDN_CHANNEL_BLOCK_SIZE)),
            DType::U32,
            dev,
        )?;
        let recurrent_applied_epochs = Tensor::zeros((capacity, num_v_heads), DType::U32, dev)?;
        cases.push(LayerCase {
            conv_input,
            weight,
            b,
            a,
            a_log,
            dt_bias,
            gate,
            norm_weight,
            transitions: recurrence
                .transitions
                .expect("record_transitions produced no transition tensors"),
            staged_conv_state: conv_state.copy()?,
            staged_recurrent_state: recurrent_state.copy()?,
            lazy_conv_applied_epochs: conv_applied_epochs.copy()?,
            lazy_recurrent_applied_epochs: recurrent_applied_epochs.copy()?,
            lazy_conv_state: conv_state.copy()?,
            lazy_recurrent_state: recurrent_state.copy()?,
            conv_state,
            recurrent_state,
            pending_conv_input,
            pending_key,
            pending_key_banks,
            pending_key_bank,
            pending_delta,
            pending_decay,
            pending_keep_rows,
            pending_epochs,
            conv_applied_epochs,
            recurrent_applied_epochs,
            expected_conv_state,
            expected_recurrent_state,
        });
    }

    let layers = cases
        .iter()
        .map(|case| GdnSpeculativeTransitionLayer {
            conv_input: &case.conv_input,
            key: &case.transitions.key,
            delta: &case.transitions.delta,
            decay: &case.transitions.decay,
            conv_state: &case.conv_state,
            recurrent_state: &case.recurrent_state,
        })
        .collect::<Vec<_>>();
    speculative_transition_commit_batched_cuda(GdnSpeculativeTransitionCommit {
        layers: &layers,
        keep_rows: &keep_rows,
        active_slots: &active_slots,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_dim,
        conv_width,
        tiled_v_heads,
        state_layout: RecurrentStateLayout::GdnValueMajor,
    })?;
    let stage_layers = cases
        .iter()
        .map(|case| GdnSpeculativeTransitionStageLayer {
            conv_input: &case.conv_input,
            key: &case.transitions.key,
            delta: &case.transitions.delta,
            decay: &case.transitions.decay,
            pending_conv_input: &case.pending_conv_input,
            pending_key: &case.pending_key,
            pending_delta: &case.pending_delta,
            pending_decay: &case.pending_decay,
            pending_keep_rows: &case.pending_keep_rows,
            pending_epochs: &case.pending_epochs,
        })
        .collect::<Vec<_>>();
    speculative_transition_stage_batched_cuda(GdnSpeculativeTransitionStage {
        layers: &stage_layers,
        keep_rows: &keep_rows,
        destination_slots: &active_slots,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_dim,
    })?;
    let apply_layers = cases
        .iter()
        .map(|case| GdnPendingTransitionApplyLayer {
            pending_conv_input: &case.pending_conv_input,
            pending_key_banks: &case.pending_key_banks,
            pending_key_bank: &case.pending_key_bank,
            pending_delta: &case.pending_delta,
            pending_decay: &case.pending_decay,
            pending_keep_rows: &case.pending_keep_rows,
            pending_epochs: &case.pending_epochs,
            conv_applied_epochs: &case.conv_applied_epochs,
            recurrent_applied_epochs: &case.recurrent_applied_epochs,
            conv_state: &case.staged_conv_state,
            recurrent_state: &case.staged_recurrent_state,
        })
        .collect::<Vec<_>>();
    let apply = || {
        pending_transition_apply_batched_cuda(GdnPendingTransitionApply {
            layers: &apply_layers,
            active_slots: &active_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            conv_width,
            tiled_v_heads,
            state_layout: RecurrentStateLayout::GdnValueMajor,
        })
    };
    apply()?;
    apply()?;
    for case in &cases {
        assert_close(
            "batched transition convolution state",
            &flat(&case.conv_state.to_dtype(DType::F32)?)?,
            &flat(&case.expected_conv_state.to_dtype(DType::F32)?)?,
            0.0,
        );
        assert_close(
            "batched transition recurrent state",
            &flat(&case.recurrent_state.to_dtype(DType::F32)?)?,
            &flat(&case.expected_recurrent_state.to_dtype(DType::F32)?)?,
            3.0e-3,
        );
        assert_close(
            "staged transition convolution state",
            &flat(&case.staged_conv_state.to_dtype(DType::F32)?)?,
            &flat(&case.expected_conv_state.to_dtype(DType::F32)?)?,
            0.0,
        );
        assert_close(
            "staged transition recurrent state",
            &flat(&case.staged_recurrent_state.to_dtype(DType::F32)?)?,
            &flat(&case.expected_recurrent_state.to_dtype(DType::F32)?)?,
            3.0e-3,
        );

        let expected_convolved =
            speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
                x: &case.conv_input,
                weight: &case.weight,
                state_pool: &case.expected_conv_state,
                active_slots: &active_slots,
                checkpoint_lanes: 1,
                write_checkpoints: false,
                pending: None,
            })?;
        let expected_recurrence =
            speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                mixed_qkv: &expected_convolved,
                b: &case.b,
                a: &case.a,
                a_log: &case.a_log,
                dt_bias: &case.dt_bias,
                state_pool: &case.expected_recurrent_state,
                active_slots: &active_slots,
                checkpoint_lanes: 1,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                tiled_v_heads,
                state_layout: RecurrentStateLayout::GdnValueMajor,
                post_op: Some(GdnSpeculativeRmsNormGate {
                    sigmoid_gate: false,
                    gate: &case.gate,
                    weight: &case.norm_weight,
                    eps: 1.0e-6,
                    quantization: None,
                }),
                record_transitions: true,
                pending: None,
            })?;
        let expected_output = expected_recurrence.output.as_tensor()?;
        let run_lazy = || {
            let convolved = speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
                x: &case.conv_input,
                weight: &case.weight,
                state_pool: &case.lazy_conv_state,
                active_slots: &active_slots,
                checkpoint_lanes: 1,
                write_checkpoints: false,
                pending: Some(GdnPendingSpeculativeConv {
                    conv_input: &case.pending_conv_input,
                    keep_rows: &case.pending_keep_rows,
                    pending_epochs: &case.pending_epochs,
                    applied_epochs: &case.lazy_conv_applied_epochs,
                }),
            })?;
            let output =
                speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                    mixed_qkv: &convolved,
                    b: &case.b,
                    a: &case.a,
                    a_log: &case.a_log,
                    dt_bias: &case.dt_bias,
                    state_pool: &case.lazy_recurrent_state,
                    active_slots: &active_slots,
                    checkpoint_lanes: 1,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    tiled_v_heads,
                    state_layout: RecurrentStateLayout::GdnValueMajor,
                    post_op: Some(GdnSpeculativeRmsNormGate {
                        sigmoid_gate: false,
                        gate: &case.gate,
                        weight: &case.norm_weight,
                        eps: 1.0e-6,
                        quantization: None,
                    }),
                    record_transitions: true,
                    pending: Some(GdnPendingSpeculativeRecurrence {
                        key_banks: &case.pending_key_banks,
                        key_bank: &case.pending_key_bank,
                        delta: &case.pending_delta,
                        decay: &case.pending_decay,
                        keep_rows: &case.pending_keep_rows,
                        pending_epochs: &case.pending_epochs,
                        applied_epochs: &case.lazy_recurrent_applied_epochs,
                    }),
                })?
                .output
                .into_tensor()?;
            Ok::<_, candle_core::Error>((convolved, output))
        };
        let (lazy_convolved, lazy_output) = run_lazy()?;
        let (retried_convolved, retried_output) = run_lazy()?;
        assert_close(
            "lazy transition convolution state",
            &flat(&case.lazy_conv_state.to_dtype(DType::F32)?)?,
            &flat(&case.expected_conv_state.to_dtype(DType::F32)?)?,
            0.0,
        );
        assert_close(
            "lazy transition recurrent state",
            &flat(&case.lazy_recurrent_state.to_dtype(DType::F32)?)?,
            &flat(&case.expected_recurrent_state.to_dtype(DType::F32)?)?,
            3.0e-3,
        );
        for (label, actual, expected, tolerance) in [
            (
                "lazy transition convolution output",
                &lazy_convolved,
                &expected_convolved,
                0.0,
            ),
            (
                "retried transition convolution output",
                &retried_convolved,
                &expected_convolved,
                0.0,
            ),
            (
                "lazy transition recurrence output",
                &lazy_output,
                expected_output,
                3.0e-2,
            ),
            (
                "retried transition recurrence output",
                &retried_output,
                expected_output,
                3.0e-2,
            ),
        ] {
            assert_close(
                label,
                &flat(&actual.to_dtype(DType::F32)?)?,
                &flat(&expected.to_dtype(DType::F32)?)?,
                tolerance,
            );
        }

        let expected_second_conv_state = case.expected_conv_state.copy()?;
        let expected_second_recurrent_state = case.expected_recurrent_state.copy()?;
        let expected_transitions = expected_recurrence
            .transitions
            .as_ref()
            .expect("record_transitions produced no transition tensors");
        let expected_transition_layers = [GdnSpeculativeTransitionLayer {
            conv_input: &case.conv_input,
            key: &expected_transitions.key,
            delta: &expected_transitions.delta,
            decay: &expected_transitions.decay,
            conv_state: &expected_second_conv_state,
            recurrent_state: &expected_second_recurrent_state,
        }];
        speculative_transition_commit_batched_cuda(GdnSpeculativeTransitionCommit {
            layers: &expected_transition_layers,
            keep_rows: &keep_rows,
            active_slots: &active_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            conv_width,
            tiled_v_heads,
            state_layout: RecurrentStateLayout::GdnValueMajor,
        })?;

        let publish_layers = [GdnPendingTransitionPublishLayer {
            pending_keep_rows: &case.pending_keep_rows,
            pending_epochs: &case.pending_epochs,
            pending_key_bank: &case.pending_key_bank,
        }];
        pending_transition_publish_batched_cuda(GdnPendingTransitionPublish {
            layers: &publish_layers,
            keep_rows: &keep_rows,
            destination_slots: &active_slots,
            max_rows: seq_len,
            destination_capacity: capacity,
        })?;

        let expected_second_convolved =
            speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
                x: &case.conv_input,
                weight: &case.weight,
                state_pool: &expected_second_conv_state,
                active_slots: &active_slots,
                checkpoint_lanes: 1,
                write_checkpoints: false,
                pending: None,
            })?;
        let expected_second_output =
            speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                mixed_qkv: &expected_second_convolved,
                b: &case.b,
                a: &case.a,
                a_log: &case.a_log,
                dt_bias: &case.dt_bias,
                state_pool: &expected_second_recurrent_state,
                active_slots: &active_slots,
                checkpoint_lanes: 1,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                tiled_v_heads,
                state_layout: RecurrentStateLayout::GdnValueMajor,
                post_op: Some(GdnSpeculativeRmsNormGate {
                    sigmoid_gate: false,
                    gate: &case.gate,
                    weight: &case.norm_weight,
                    eps: 1.0e-6,
                    quantization: None,
                }),
                record_transitions: true,
                pending: None,
            })?
            .output
            .into_tensor()?;
        let (second_convolved, second_output) = run_lazy()?;
        assert_close(
            "second-epoch transition convolution state",
            &flat(&case.lazy_conv_state.to_dtype(DType::F32)?)?,
            &flat(&expected_second_conv_state.to_dtype(DType::F32)?)?,
            0.0,
        );
        assert_close(
            "second-epoch transition recurrent state",
            &flat(&case.lazy_recurrent_state.to_dtype(DType::F32)?)?,
            &flat(&expected_second_recurrent_state.to_dtype(DType::F32)?)?,
            3.0e-3,
        );
        assert_close(
            "second-epoch transition convolution output",
            &flat(&second_convolved.to_dtype(DType::F32)?)?,
            &flat(&expected_second_convolved.to_dtype(DType::F32)?)?,
            0.0,
        );
        assert_close(
            "second-epoch transition recurrence output",
            &flat(&second_output.to_dtype(DType::F32)?)?,
            &flat(&expected_second_output.to_dtype(DType::F32)?)?,
            3.0e-2,
        );
        let published_banks = case
            .pending_key_bank
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        let published_epochs = case
            .pending_epochs
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        for &slot in active_slots_host
            .iter()
            .filter(|&&slot| slot != GDN_PAD_SLOT)
        {
            assert_eq!(published_banks[slot as usize], 1);
            assert_eq!(published_epochs[slot as usize], 2);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn speculative_transition_commit_matches_prefix_replay_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for seq_len in [4, 5, 8] {
        for activation_dtype in [DType::F16, DType::BF16] {
            for state_dtype in [DType::F32, DType::BF16, DType::F16] {
                for tiled_v_heads in [false, true] {
                    run_speculative_transition_commit_case(
                        &dev,
                        seq_len,
                        activation_dtype,
                        state_dtype,
                        tiled_v_heads,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn pending_transition_publish_is_slot_indexed_cuda() -> Result<()> {
    const CAPACITY: usize = 6;
    const MAX_ROWS: usize = 8;

    let dev = Device::new_cuda(0)?;
    let keep_rows = Tensor::from_vec(vec![3u32, 8, 5], (3,), &dev)?;
    let destination_slots = Tensor::from_vec(vec![4u32, 1, GDN_PAD_SLOT], (3,), &dev)?;
    let layer_keep = [
        Tensor::from_vec(vec![10u32, 11, 12, 13, 14, 15], (CAPACITY,), &dev)?,
        Tensor::from_vec(vec![20u32, 21, 22, 23, 24, 25], (CAPACITY,), &dev)?,
    ];
    let layer_epochs = [
        Tensor::from_vec(vec![30u32, 31, 32, 33, u32::MAX, 35], (CAPACITY,), &dev)?,
        Tensor::from_vec(vec![40u32, 41, 42, 43, u32::MAX, 45], (CAPACITY,), &dev)?,
    ];
    let layer_key_bank = [
        Tensor::from_vec(vec![0u32, 1, 0, 1, 0, 1], (CAPACITY,), &dev)?,
        Tensor::from_vec(vec![1u32, 0, 1, 0, 1, 0], (CAPACITY,), &dev)?,
    ];
    let layers = (0..layer_keep.len())
        .map(|idx| GdnPendingTransitionPublishLayer {
            pending_keep_rows: &layer_keep[idx],
            pending_epochs: &layer_epochs[idx],
            pending_key_bank: &layer_key_bank[idx],
        })
        .collect::<Vec<_>>();

    pending_transition_publish_batched_cuda(GdnPendingTransitionPublish {
        layers: &layers,
        keep_rows: &keep_rows,
        destination_slots: &destination_slots,
        max_rows: MAX_ROWS,
        destination_capacity: CAPACITY,
    })?;

    for (idx, (keep, epochs)) in layer_keep.iter().zip(&layer_epochs).enumerate() {
        let keep = keep.to_device(&Device::Cpu)?.to_vec1::<u32>()?;
        let epochs = epochs.to_device(&Device::Cpu)?.to_vec1::<u32>()?;
        let key_bank = layer_key_bank[idx]
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        let keep_base = if idx == 0 { 10 } else { 20 };
        let epoch_base = if idx == 0 { 30 } else { 40 };
        assert_eq!(
            keep,
            vec![keep_base, 8, keep_base + 2, keep_base + 3, 3, keep_base + 5]
        );
        assert_eq!(
            epochs,
            vec![
                epoch_base,
                epoch_base + 2,
                epoch_base + 2,
                epoch_base + 3,
                1,
                epoch_base + 5,
            ]
        );
        assert_eq!(
            key_bank,
            if idx == 0 {
                vec![0, 0, 0, 1, 1, 1]
            } else {
                vec![1, 1, 1, 0, 0, 0]
            }
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ConvShape {
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    kernel_size: usize,
}

fn causal_conv_reference(
    x: &[f32],
    weight: &[f32],
    initial_state: &[f32],
    shape: ConvShape,
) -> (Vec<f32>, Vec<f32>) {
    let ConvShape {
        batch_size,
        seq_len,
        conv_dim,
        kernel_size,
    } = shape;
    let mut state = initial_state.to_vec();
    let mut output = vec![0.0f32; batch_size * seq_len * conv_dim];
    for b in 0..batch_size {
        for pos in 0..seq_len {
            for ch in 0..conv_dim {
                let state_base = (b * conv_dim + ch) * kernel_size;
                state.copy_within(state_base + 1..state_base + kernel_size, state_base);
                state[state_base + kernel_size - 1] = x[(b * seq_len + pos) * conv_dim + ch];
                let weight_base = ch * kernel_size;
                let mut sum = 0.0f32;
                for k in 0..kernel_size {
                    sum += state[state_base + k] * weight[weight_base + k];
                }
                output[(b * seq_len + pos) * conv_dim + ch] = sum / (1.0 + (-sum).exp());
            }
        }
    }
    (output, state)
}

#[test]
#[ignore = "requires a CUDA device"]
fn causal_conv1d_strided_nonzero_offset_matches_reference_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let batch_size = 3;
    let conv_dim = 19;
    let kernel_size = 4;
    let prefix = 3;
    let physical_dim = conv_dim + 7;

    for seq_len in [1usize, 5] {
        let logical = patterned(batch_size * seq_len * conv_dim, 60 + seq_len, 0.08, 0.01);
        let mut packed = vec![-7.0f32; batch_size * seq_len * physical_dim];
        for b in 0..batch_size {
            for pos in 0..seq_len {
                let logical_base = (b * seq_len + pos) * conv_dim;
                let packed_base = (b * seq_len + pos) * physical_dim + prefix;
                packed[packed_base..packed_base + conv_dim]
                    .copy_from_slice(&logical[logical_base..logical_base + conv_dim]);
            }
        }
        let x = Tensor::from_vec(packed, (batch_size, seq_len, physical_dim), &dev)?
            .to_dtype(DType::F16)?
            .narrow(2, prefix, conv_dim)?;
        assert!(!x.is_contiguous());
        assert!(x.layout().start_offset() > 0);

        let weight = tensor2(
            patterned(conv_dim * kernel_size, 70 + seq_len, 0.05, -0.01),
            (conv_dim, kernel_size),
            &dev,
        )?
        .to_dtype(DType::F16)?;
        let state = tensor3(
            patterned(batch_size * conv_dim * kernel_size, 80 + seq_len, 0.03, 0.0),
            (batch_size, conv_dim, kernel_size),
            &dev,
        )?
        .to_dtype(DType::F16)?;
        let x_host = flat(&x.to_dtype(DType::F32)?.contiguous()?)?;
        let weight_host = flat(&weight.to_dtype(DType::F32)?)?;
        let state_host = flat(&state.to_dtype(DType::F32)?)?;
        let (expected, expected_state) = causal_conv_reference(
            &x_host,
            &weight_host,
            &state_host,
            ConvShape {
                batch_size,
                seq_len,
                conv_dim,
                kernel_size,
            },
        );

        let (actual, actual_state) = causal_conv1d_cuda(
            &x,
            &weight,
            &state,
            kernel_size,
            seq_len == 1,
            GdnStateSlots::Gathered,
        )?;
        assert_eq!(actual.dims3()?, (batch_size, seq_len, conv_dim));
        assert!(actual.is_contiguous());
        assert_close(
            "strided causal conv output",
            &flat(&actual.to_dtype(DType::F32)?)?,
            &expected,
            2.0e-3,
        );
        assert_close(
            "strided causal conv state",
            &flat(&actual_state.to_dtype(DType::F32)?)?,
            &expected_state,
            5.0e-4,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn rmsnorm_gated_strided_nonzero_offset_matches_reference_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (batch_size, seq_len, heads, hidden_dim) = (3, 2, 5, 17);
    let x_physical_dim = hidden_dim + 7;
    let value_dim = heads * hidden_dim;
    let gate_physical_dim = value_dim + 7;
    let x = Tensor::from_vec(
        patterned(batch_size * seq_len * heads * x_physical_dim, 91, 0.2, 0.01),
        (batch_size, heads, seq_len, x_physical_dim),
        &dev,
    )?
    .to_dtype(DType::BF16)?
    .narrow(3, 2, hidden_dim)?
    .transpose(1, 2)?;
    let gate = Tensor::from_vec(
        patterned(batch_size * seq_len * gate_physical_dim, 92, 0.3, -0.02),
        (batch_size, seq_len, gate_physical_dim),
        &dev,
    )?
    .to_dtype(DType::BF16)?
    .narrow(2, 3, value_dim)?;
    let weight = Tensor::from_vec(patterned(hidden_dim, 93, 0.1, 1.0), (hidden_dim,), &dev)?
        .to_dtype(DType::BF16)?;
    assert!(!x.is_contiguous());
    assert!(!gate.is_contiguous());
    assert!(x.layout().start_offset() > 0 && gate.layout().start_offset() > 0);

    let rows = batch_size * seq_len * heads;
    let x_host = flat(&x.to_dtype(DType::F32)?.contiguous()?)?;
    let gate_host = flat(&gate.to_dtype(DType::F32)?.contiguous()?)?;
    let weight_host = flat(&weight.to_dtype(DType::F32)?)?;
    let eps = 1.0e-6;
    let mut expected = vec![0.0f32; rows * hidden_dim];
    for row in 0..rows {
        let base = row * hidden_dim;
        let mean_square = x_host[base..base + hidden_dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / hidden_dim as f32;
        let inv_rms = (mean_square + eps as f32).sqrt().recip();
        for col in 0..hidden_dim {
            let gate_value = gate_host[base + col];
            let silu_gate = gate_value / (1.0 + (-gate_value).exp());
            expected[base + col] = x_host[base + col] * inv_rms * weight_host[col] * silu_gate;
        }
    }

    let actual = rmsnorm_gated_cuda(&x, &gate, &weight, eps, GdnGateActivation::Silu)?;
    assert_eq!(actual.shape(), x.shape());
    assert!(actual.is_contiguous());
    assert_close(
        "strided gated RMSNorm",
        &flat(&actual.to_dtype(DType::F32)?)?,
        &expected,
        2.0e-3,
    );
    Ok(())
}

fn assert_gdn_quantized_matches_bf16(
    label: &str,
    activation: &QuantizedActivation,
    expected: &Tensor,
    projection_rows: usize,
    groups: usize,
    scale_layout: ActivationScaleLayout,
) -> Result<()> {
    use float8::F8E4M3;
    use half::bf16;

    let expected = expected
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<bf16>()?;
    let quantized = activation
        .quantized()
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<F8E4M3>()?;
    let scales = activation
        .scales()
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let scale_stride_m = match scale_layout {
        ActivationScaleLayout::RowMajor => projection_rows,
        ActivationScaleLayout::GroupMajor { row_alignment } => {
            projection_rows.div_ceil(row_alignment.get()) * row_alignment.get()
        }
    };
    for projection_row in 0..projection_rows {
        for group in 0..groups {
            let start = (projection_row * groups + group) * GDN_FP8_GROUP_SIZE;
            let maximum = expected[start..start + GDN_FP8_GROUP_SIZE]
                .iter()
                .map(|value| value.to_f32().abs())
                .fold(0.0f32, f32::max);
            let (quant_scale, expected_scale, scale_offset) = match scale_layout {
                ActivationScaleLayout::RowMajor => {
                    let scale = maximum.max(1.0e-10) / 448.0;
                    (1.0 / scale, scale, group * scale_stride_m + projection_row)
                }
                ActivationScaleLayout::GroupMajor { .. } => {
                    let quant_scale = if maximum == 0.0 { 1.0 } else { 448.0 / maximum };
                    (
                        quant_scale,
                        1.0 / quant_scale,
                        group * scale_stride_m + projection_row,
                    )
                }
            };
            let actual_scale = scales[scale_offset];
            let scale_tolerance = 1.0e-12 + expected_scale.abs() * 2.0e-6;
            assert!(
                    (actual_scale - expected_scale).abs() <= scale_tolerance,
                    "{label}: row={projection_row}, group={group}, expected scale {expected_scale}, got {actual_scale}"
                );
            for column in 0..GDN_FP8_GROUP_SIZE {
                let index = start + column;
                let scaled = if scale_layout == ActivationScaleLayout::RowMajor {
                    expected[index].to_f32() / expected_scale
                } else {
                    expected[index].to_f32() * quant_scale
                };
                let expected_quantized = F8E4M3::from_f32(scaled.clamp(-448.0, 448.0)).to_f32();
                let actual_quantized = quantized[index].to_f32();
                let magnitude = expected_quantized.abs();
                let quantization_step = if magnitude < 2.0f32.powi(-6) {
                    2.0f32.powi(-9)
                } else {
                    2.0f32.powi(magnitude.log2().floor() as i32 - 3)
                };
                assert!(
                        (actual_quantized - expected_quantized).abs()
                            <= quantization_step * (1.0 + 2.0e-6),
                        "{label}: row={projection_row}, group={group}, column={column}, expected {expected_quantized}, got {actual_quantized}"
                    );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn rmsnorm_gated_quantized_matches_bf16_rounding_cuda() -> Result<()> {
    use std::num::NonZeroUsize;

    const BATCH_SIZE: usize = 1;
    const NUM_V_HEADS: usize = 4;
    const HEAD_DIM: usize = GDN_FP8_GROUP_SIZE;
    const EPS: f64 = 1.0e-6;

    let dev = Device::new_cuda(0)?;
    for seq_len in [3usize, 257] {
        let mut x_values = patterned(
            BATCH_SIZE * seq_len * NUM_V_HEADS * HEAD_DIM,
            seq_len + 401,
            0.2,
            0.01,
        );
        x_values[..HEAD_DIM].fill(0.0);
        let x = Tensor::from_vec(x_values, (BATCH_SIZE, seq_len, NUM_V_HEADS, HEAD_DIM), &dev)?
            .to_dtype(DType::BF16)?;
        let gate = Tensor::from_vec(
            patterned(
                BATCH_SIZE * seq_len * NUM_V_HEADS * HEAD_DIM,
                seq_len + 402,
                0.3,
                -0.02,
            ),
            (BATCH_SIZE, seq_len, NUM_V_HEADS, HEAD_DIM),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let weight =
            Tensor::from_vec(patterned(HEAD_DIM, seq_len + 403, 0.1, 1.0), HEAD_DIM, &dev)?
                .to_dtype(DType::BF16)?;
        let expected = rmsnorm_gated_cuda(&x, &gate, &weight, EPS, GdnGateActivation::Silu)?;
        for scale_layout in [
            ActivationScaleLayout::RowMajor,
            ActivationScaleLayout::GroupMajor {
                row_alignment: NonZeroUsize::new(4).unwrap(),
            },
        ] {
            let source_shape = [BATCH_SIZE, seq_len, NUM_V_HEADS * HEAD_DIM];
            let spec = GdnFp8OutputSpec::new(
                source_shape,
                ActivationQuantizationScheme {
                    dtype: DType::F8E4M3,
                    scale_encoding: mistralrs_quant::ActivationScaleEncoding::F32,
                    block_shape: [1, GDN_FP8_GROUP_SIZE],
                },
                scale_layout,
                NUM_V_HEADS,
                HEAD_DIM,
            )
            .unwrap();
            let actual = rmsnorm_gated_quantized_cuda(
                &x,
                &gate,
                &weight,
                EPS,
                GdnGateActivation::Silu,
                &spec,
                NUM_V_HEADS,
                HEAD_DIM,
            )?;
            dev.synchronize()?;
            assert_eq!(actual.source_shape(), source_shape);
            assert_eq!(actual.source_dtype(), DType::BF16);
            assert_eq!(actual.scale_layout(), scale_layout);
            assert_gdn_quantized_matches_bf16(
                &format!("seq_len={seq_len} {scale_layout:?}"),
                &actual,
                &expected,
                BATCH_SIZE * seq_len,
                NUM_V_HEADS,
                scale_layout,
            )?;
        }
    }
    Ok(())
}

fn assert_zero(label: &str, tensor: &Tensor) -> Result<()> {
    let values = flat(&tensor.to_dtype(DType::F32)?)?;
    let nonzero = values.iter().position(|value| *value != 0.0);
    assert!(nonzero.is_none(), "{label}: nonzero value at {nonzero:?}");
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn pooled_causal_conv_padding_rows_are_zero_and_stateless_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let capacity = 5usize;
    let batch = 3usize;
    let conv_dim = 7usize;
    let slots_host = vec![3u32, GDN_PAD_SLOT, 1];
    let slots = Tensor::from_vec(slots_host, (batch,), &dev)?;
    let real_batch = Tensor::from_vec(vec![0u32, 2], (2,), &dev)?;
    let real_slots = Tensor::from_vec(vec![3u32, 1], (2,), &dev)?;

    for dtype in [DType::F16, DType::BF16] {
        for kernel_size in [3usize, 4] {
            for (seq_len, is_update) in [(1usize, true), (3, false)] {
                let x = tensor3(
                    patterned(batch * seq_len * conv_dim, 80, 0.08, 0.01),
                    (batch, seq_len, conv_dim),
                    &dev,
                )?
                .to_dtype(dtype)?;
                let weight = tensor2(
                    patterned(conv_dim * kernel_size, 81, 0.05, -0.01),
                    (conv_dim, kernel_size),
                    &dev,
                )?
                .to_dtype(dtype)?;
                let initial = tensor3(
                    patterned(capacity * conv_dim * kernel_size, 82, 0.03, 0.0),
                    (capacity, conv_dim, kernel_size),
                    &dev,
                )?
                .to_dtype(dtype)?;
                let x_real = x.index_select(&real_batch, 0)?.contiguous()?;
                let gathered_initial = initial.index_select(&real_slots, 0)?.contiguous()?;
                let (expected_output, expected_state) = causal_conv1d_cuda(
                    &x_real,
                    &weight,
                    &gathered_initial,
                    kernel_size,
                    is_update,
                    GdnStateSlots::Gathered,
                )?;
                let pool = initial.copy()?;
                let (actual_output, actual_state) = causal_conv1d_cuda(
                    &x,
                    &weight,
                    &pool,
                    kernel_size,
                    is_update,
                    GdnStateSlots::Pooled(&slots),
                )?;

                assert_zero(
                    "causal convolution padded output",
                    &actual_output.narrow(0, 1, 1)?,
                )?;
                assert_close(
                    "causal convolution real output",
                    &flat(
                        &actual_output
                            .index_select(&real_batch, 0)?
                            .to_dtype(DType::F32)?,
                    )?,
                    &flat(&expected_output.to_dtype(DType::F32)?)?,
                    0.0,
                );
                assert_close(
                    "causal convolution real state",
                    &flat(
                        &actual_state
                            .index_select(&real_slots, 0)?
                            .to_dtype(DType::F32)?,
                    )?,
                    &flat(&expected_state.to_dtype(DType::F32)?)?,
                    0.0,
                );
                for row in [0usize, 2, 4] {
                    assert_close(
                        "causal convolution untouched state",
                        &flat(&actual_state.narrow(0, row, 1)?.to_dtype(DType::F32)?)?,
                        &flat(&initial.narrow(0, row, 1)?.to_dtype(DType::F32)?)?,
                        0.0,
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn pooled_decomposed_recurrence_padding_rows_are_zero_and_stateless_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let capacity = 5usize;
    let batch = 3usize;
    let num_heads = 2usize;
    let seq_len = 3usize;
    let k_dim = 128usize;
    let v_dim = 128usize;
    let bh = batch * num_heads;
    let real_bh = 2 * num_heads;
    let slots = Tensor::from_vec(vec![4u32, GDN_PAD_SLOT, 1], (batch,), &dev)?;
    let real_slots = Tensor::from_vec(vec![4u32, 1], (2,), &dev)?;
    let real_heads = Tensor::from_vec(vec![0u32, 1, 4, 5], (real_bh,), &dev)?;
    let q = tensor3(
        patterned(bh * seq_len * k_dim, 90, 0.02, 0.0),
        (bh, seq_len, k_dim),
        &dev,
    )?;
    let k = tensor3(
        patterned(bh * seq_len * k_dim, 91, 0.02, 0.0),
        (bh, seq_len, k_dim),
        &dev,
    )?;
    let v = tensor3(
        patterned(bh * seq_len * v_dim, 92, 0.05, 0.0),
        (bh, seq_len, v_dim),
        &dev,
    )?;
    let g = tensor2(
        patterned(bh * seq_len, 93, 0.03, -0.08),
        (bh, seq_len),
        &dev,
    )?;
    let beta = tensor2(patterned(bh * seq_len, 94, 0.15, 0.5), (bh, seq_len), &dev)?;
    let real_q = q.index_select(&real_heads, 0)?.contiguous()?;
    let real_k = k.index_select(&real_heads, 0)?.contiguous()?;
    let real_v = v.index_select(&real_heads, 0)?.contiguous()?;
    let real_g = g.index_select(&real_heads, 0)?.contiguous()?;
    let real_beta = beta.index_select(&real_heads, 0)?.contiguous()?;
    let pooled_inputs = RecurrenceInputs {
        q: &q,
        k: &k,
        v: &v,
        g: &g,
        beta: &beta,
    };
    let gathered_inputs = RecurrenceInputs {
        q: &real_q,
        k: &real_k,
        v: &real_v,
        g: &real_g,
        beta: &real_beta,
    };
    let initial = Tensor::from_vec(
        patterned(capacity * num_heads * k_dim * v_dim, 95, 0.01, 0.0),
        (capacity, num_heads, k_dim, v_dim),
        &dev,
    )?;

    for (kernel, label) in [
        (RecurrenceKernel::Scalar, "scalar"),
        (RecurrenceKernel::Warp, "warp"),
        (RecurrenceKernel::Chunked, "chunked"),
    ] {
        let mut pooled_state = initial.copy()?;
        let mut gathered_state = initial
            .index_select(&real_slots, 0)?
            .reshape((real_bh, k_dim, v_dim))?
            .contiguous()?;
        let actual = launch_recurrence(
            kernel,
            pooled_inputs,
            &mut pooled_state,
            GdnStateSlots::Pooled(&slots),
        )?;
        let expected = launch_recurrence(
            kernel,
            gathered_inputs,
            &mut gathered_state,
            GdnStateSlots::Gathered,
        )?;
        assert_zero(
            &format!("{label} padded output"),
            &actual.narrow(0, num_heads, num_heads)?,
        )?;
        assert_close(
            &format!("{label} real output"),
            &flat(&actual.index_select(&real_heads, 0)?)?,
            &flat(&expected)?,
            1.0e-6,
        );
        assert_close(
            &format!("{label} real state"),
            &flat(
                &pooled_state
                    .index_select(&real_slots, 0)?
                    .reshape((real_bh, k_dim, v_dim))?,
            )?,
            &flat(&gathered_state)?,
            1.0e-6,
        );
        for row in [0usize, 2, 3] {
            assert_close(
                &format!("{label} untouched state"),
                &flat(&pooled_state.narrow(0, row, 1)?)?,
                &flat(&initial.narrow(0, row, 1)?)?,
                0.0,
            );
        }
    }

    let initial = initial.transpose(2, 3)?.contiguous()?;
    let mut pooled_state = initial.copy()?;
    let mut gathered_state = initial
        .index_select(&real_slots, 0)?
        .reshape((real_bh, v_dim, k_dim))?
        .contiguous()?;
    let actual = launch_recurrence(
        RecurrenceKernel::ValueMajorWarp,
        pooled_inputs,
        &mut pooled_state,
        GdnStateSlots::Pooled(&slots),
    )?;
    let expected = launch_recurrence(
        RecurrenceKernel::ValueMajorWarp,
        gathered_inputs,
        &mut gathered_state,
        GdnStateSlots::Gathered,
    )?;
    assert_zero(
        "value-major warp padded output",
        &actual.narrow(0, num_heads, num_heads)?,
    )?;
    assert_close(
        "value-major warp real output",
        &flat(&actual.index_select(&real_heads, 0)?)?,
        &flat(&expected)?,
        1.0e-6,
    );
    assert_close(
        "value-major warp real state",
        &flat(
            &pooled_state
                .index_select(&real_slots, 0)?
                .reshape((real_bh, v_dim, k_dim))?,
        )?,
        &flat(&gathered_state)?,
        1.0e-6,
    );
    for row in [0usize, 2, 3] {
        assert_close(
            "value-major warp untouched state",
            &flat(&pooled_state.narrow(0, row, 1)?)?,
            &flat(&initial.narrow(0, row, 1)?)?,
            0.0,
        );
    }
    Ok(())
}

fn run_fused_decode_padding_case(
    dev: &Device,
    kernel: GdnDecodeKernel,
    head_k_dim: usize,
    head_v_dim: usize,
    dtype: DType,
    state_layout: RecurrentStateLayout,
) -> Result<()> {
    let capacity = 4usize;
    let batch = 3usize;
    let num_k_heads = 1usize;
    let num_v_heads = 1usize;
    let conv_dim = 2 * head_k_dim + head_v_dim;
    let slots = Tensor::from_vec(vec![2u32, GDN_PAD_SLOT, 0], (batch,), dev)?;
    let real_batch = Tensor::from_vec(vec![0u32, 2], (2,), dev)?;
    let real_slots = Tensor::from_vec(vec![2u32, 0], (2,), dev)?;
    let mixed_qkv = tensor3(
        patterned(batch * conv_dim, 100, 0.08, 0.01),
        (batch, 1, conv_dim),
        dev,
    )?
    .to_dtype(dtype)?;
    let b = tensor3(
        patterned(batch * num_v_heads, 101, 0.2, 0.1),
        (batch, 1, num_v_heads),
        dev,
    )?
    .to_dtype(dtype)?;
    let a = tensor3(
        patterned(batch * num_v_heads, 102, 0.18, -0.04),
        (batch, 1, num_v_heads),
        dev,
    )?
    .to_dtype(dtype)?;
    let a_log = Tensor::from_vec(patterned(num_v_heads, 103, 0.05, -0.2), (num_v_heads,), dev)?;
    let dt_bias = Tensor::from_vec(patterned(num_v_heads, 104, 0.1, 0.3), (num_v_heads,), dev)?;
    let initial = Tensor::from_vec(
        patterned(
            capacity * num_v_heads * head_k_dim * head_v_dim,
            105,
            0.02,
            0.0,
        ),
        (capacity, num_v_heads, head_k_dim, head_v_dim),
        dev,
    )?;
    let initial = if state_layout == RecurrentStateLayout::GdnValueMajor {
        initial.transpose(2, 3)?.contiguous()?
    } else {
        initial
    };
    let mut pooled_state = initial.copy()?;
    let mut gathered_state = initial.index_select(&real_slots, 0)?.contiguous()?;
    let actual = fused_decode_recurrence_cuda_impl(GdnDecodeLaunch {
        mixed_qkv: &mixed_qkv,
        b: &b,
        a: &a,
        a_log: &a_log,
        dt_bias: &dt_bias,
        state: &mut pooled_state,
        batch_size: batch,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tiled_v_heads: false,
        state_layout,
        slots: GdnStateSlots::Pooled(&slots),
        requested_kernel: Some(kernel),
    })?;
    let mixed_qkv_real = mixed_qkv.index_select(&real_batch, 0)?.contiguous()?;
    let b_real = b.index_select(&real_batch, 0)?.contiguous()?;
    let a_real = a.index_select(&real_batch, 0)?.contiguous()?;
    let expected = fused_decode_recurrence_cuda_impl(GdnDecodeLaunch {
        mixed_qkv: &mixed_qkv_real,
        b: &b_real,
        a: &a_real,
        a_log: &a_log,
        dt_bias: &dt_bias,
        state: &mut gathered_state,
        batch_size: 2,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tiled_v_heads: false,
        state_layout,
        slots: GdnStateSlots::Gathered,
        requested_kernel: Some(kernel),
    })?;

    assert_zero("fused decode padded output", &actual.narrow(0, 1, 1)?)?;
    assert_close(
        "fused decode real output",
        &flat(&actual.index_select(&real_batch, 0)?.to_dtype(DType::F32)?)?,
        &flat(&expected.to_dtype(DType::F32)?)?,
        0.0,
    );
    assert_close(
        "fused decode real state",
        &flat(&pooled_state.index_select(&real_slots, 0)?)?,
        &flat(&gathered_state)?,
        0.0,
    );
    for row in [1usize, 3] {
        assert_close(
            "fused decode untouched state",
            &flat(&pooled_state.narrow(0, row, 1)?)?,
            &flat(&initial.narrow(0, row, 1)?)?,
            0.0,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn fused_decode_dispatches_zero_padding_without_touching_state_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    run_fused_decode_padding_case(
        &dev,
        GdnDecodeKernel::Baseline,
        64,
        64,
        DType::F16,
        RecurrentStateLayout::GdnKeyMajor,
    )?;
    run_fused_decode_padding_case(
        &dev,
        GdnDecodeKernel::Baseline,
        96,
        64,
        DType::BF16,
        RecurrentStateLayout::GdnKeyMajor,
    )?;
    run_fused_decode_padding_case(
        &dev,
        GdnDecodeKernel::Baseline,
        128,
        128,
        DType::BF16,
        RecurrentStateLayout::GdnKeyMajor,
    )?;

    let properties = gdn_cuda_device_properties(dev.as_cuda_device()?)?;
    if properties.compute_major >= GDN_DECODE_MIN_COMPUTE_MAJOR {
        for kernel in [GdnDecodeKernel::Cooperative, GdnDecodeKernel::Pipelined] {
            run_fused_decode_padding_case(
                &dev,
                kernel,
                128,
                128,
                DType::BF16,
                RecurrentStateLayout::GdnKeyMajor,
            )?;
        }
    }
    if properties.compute_major >= GDN_DECODE_MIN_COMPUTE_MAJOR {
        for kernel in [GdnDecodeKernel::ValueMajor4, GdnDecodeKernel::ValueMajor32] {
            run_fused_decode_padding_case(
                &dev,
                kernel,
                128,
                128,
                DType::BF16,
                RecurrentStateLayout::GdnValueMajor,
            )?;
        }
    }
    Ok(())
}
