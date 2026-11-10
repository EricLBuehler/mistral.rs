use super::*;
use candle_core::{Device, Result as CandleResult};

const ASSERT_EPS: f32 = 5e-5;
const TEST_RMS_NORM_EPS: f64 = 1e-6;

fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as f32;
            ((x / 128.0) - 1.0) * scale + offset
        })
        .collect()
}

fn dims_with_layout(
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    v_head_layout: GdnVHeadLayout,
) -> GdnDims {
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    GdnDims {
        hidden_size: value_dim,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size: 4,
        key_dim,
        value_dim,
        conv_dim: key_dim * 2 + value_dim,
        v_per_group: num_v_heads / num_k_heads,
        v_head_layout,
    }
}

fn dims(num_k_heads: usize, num_v_heads: usize, head_k_dim: usize, head_v_dim: usize) -> GdnDims {
    dims_with_layout(
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        GdnVHeadLayout::Grouped,
    )
}

fn assert_close(lhs: &Tensor, rhs: &Tensor) -> CandleResult<()> {
    assert_eq!(lhs.shape(), rhs.shape());
    let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (&lhs, &rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= ASSERT_EPS,
            "idx={idx} lhs={lhs} rhs={rhs} diff={diff}"
        );
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_token_major_prefill_conv_matches_legacy_and_chunked_cpu() -> CandleResult<()> {
    const BATCH: usize = 2;
    const LENGTHS: [usize; 6] = [1, 3, 4, 7, 53, 512];
    let metal = Device::new_metal(0)?;
    for (dtype, tolerance) in [(DType::F16, 1e-3f32), (DType::BF16, 5e-3)] {
        for kernel_size in [1, 4] {
            let mut dims = dims(2, 4, 64, 65);
            dims.conv_kernel_size = kernel_size;
            for seq_len in LENGTHS {
                let input = Tensor::from_vec(
                    patterned(BATCH * (seq_len + 2) * (dims.conv_dim + 2), 1, 0.5, 0.1),
                    (BATCH, seq_len + 2, dims.conv_dim + 2),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?;
                let slice_input =
                    |input: &Tensor| input.narrow(1, 1, seq_len)?.narrow(2, 1, dims.conv_dim);
                let x_cpu = slice_input(&input)?.to_dtype(DType::F32)?;
                let x = slice_input(&input.to_device(&metal)?)?;
                let weight_cpu = Tensor::from_vec(
                    patterned(dims.conv_dim * kernel_size, 2, 0.4, -0.1),
                    (dims.conv_dim, 1, kernel_size),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?;
                let initial_state = Tensor::from_vec(
                    patterned(BATCH * dims.conv_dim * kernel_size, 3, 0.5, 0.2),
                    (BATCH, dims.conv_dim, kernel_size),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?;
                let cache = |device: &Device, dtype| -> CandleResult<GdnLayerCache> {
                    Ok(GdnLayerCache::gathered(
                        initial_state.to_dtype(dtype)?.to_device(device)?,
                        Tensor::zeros(
                            (BATCH, dims.num_v_heads, dims.head_k_dim, dims.head_v_dim),
                            DType::F32,
                            device,
                        )?,
                        RecurrentStateLayout::GdnKeyMajor,
                    ))
                };
                let mut cpu_cache = cache(&Device::Cpu, DType::F32)?;
                let expected = causal_conv1d_full(&x_cpu, &weight_cpu, &dims, &mut cpu_cache)?;
                let weight = weight_cpu.to_dtype(dtype)?.to_device(&metal)?;
                let mut metal_cache = cache(&metal, dtype)?;
                let actual = causal_conv1d_full(&x, &weight, &dims, &mut metal_cache)?;
                assert!(actual.is_contiguous());

                let (legacy, legacy_state) = crate::metal::gdn::causal_conv1d_metal(
                    &x.transpose(1, 2)?.contiguous()?,
                    &weight.squeeze(1)?,
                    &initial_state.to_dtype(dtype)?.to_device(&metal)?,
                    false,
                    kernel_size,
                )?;
                let values = |tensor: &Tensor| -> CandleResult<Vec<f32>> {
                    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
                };
                let actual_values = values(&actual)?;
                assert_eq!(actual_values, values(&legacy.transpose(1, 2)?)?);
                assert_eq!(values(&metal_cache.conv_state)?, values(&legacy_state)?);
                assert_eq!(
                    values(&metal_cache.conv_state)?,
                    values(&cpu_cache.conv_state)?
                );
                for (actual, expected) in actual_values.iter().zip(values(&expected)?) {
                    assert!(
                            (actual - expected).abs() <= tolerance,
                            "{dtype:?} T={seq_len} W={kernel_size}: actual={actual}, expected={expected}"
                        );
                }

                let mut chunked_cache = cache(&metal, dtype)?;
                let mut outputs = Vec::new();
                let mut offset = 0;
                for chunk_len in [1, 2, seq_len] {
                    let chunk_len = chunk_len.min(seq_len - offset);
                    if chunk_len == 0 {
                        break;
                    }
                    outputs.push(causal_conv1d_full(
                        &x.narrow(1, offset, chunk_len)?,
                        &weight,
                        &dims,
                        &mut chunked_cache,
                    )?);
                    offset += chunk_len;
                }
                assert_eq!(actual_values, values(&Tensor::cat(&outputs, 1)?)?);
                assert_eq!(
                    values(&metal_cache.conv_state)?,
                    values(&chunked_cache.conv_state)?
                );
            }
        }
    }
    Ok(())
}

#[cfg(any(feature = "metal", feature = "cuda"))]
struct RecurrenceParityCase {
    layout: GdnVHeadLayout,
    dtype: DType,
    pooled: bool,
    tolerance: f32,
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn pooled_decode_from_convolved_matches_cpu(
    device: &Device,
    cases: &[RecurrenceParityCase],
) -> CandleResult<()> {
    const POOL_ROWS: usize = 6;
    let cpu = Device::Cpu;
    let slots = [2u32, 0, 5];
    let batch_size = slots.len();
    for &RecurrenceParityCase {
        layout,
        dtype,
        tolerance: tol,
        ..
    } in cases
    {
        let dims = dims_with_layout(2, 4, 128, 128, layout);
        let mixed = Tensor::from_vec(
            patterned(batch_size * dims.conv_dim, 1, 0.4, 0.05),
            (batch_size, 1, dims.conv_dim),
            &cpu,
        )?;
        let b = Tensor::from_vec(
            patterned(batch_size * dims.num_v_heads, 2, 0.8, 0.1),
            (batch_size, 1, dims.num_v_heads),
            &cpu,
        )?;
        let a = Tensor::from_vec(
            patterned(batch_size * dims.num_v_heads, 3, 0.8, -0.05),
            (batch_size, 1, dims.num_v_heads),
            &cpu,
        )?;
        let a_log = Tensor::from_vec(
            patterned(dims.num_v_heads, 4, 0.2, -0.2),
            (dims.num_v_heads,),
            &cpu,
        )?;
        let dt_bias = Tensor::from_vec(
            patterned(dims.num_v_heads, 5, 0.3, 0.3),
            (dims.num_v_heads,),
            &cpu,
        )?;
        let state_len = dims.num_v_heads * dims.head_k_dim * dims.head_v_dim;
        let pool = Tensor::from_vec(
            patterned(POOL_ROWS * state_len, 6, 0.05, 0.0),
            (
                POOL_ROWS,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            &cpu,
        )?;
        let slot_rows = slots.iter().map(|&s| s as usize).collect::<Vec<_>>();
        let gathered = Tensor::cat(
            &slot_rows
                .iter()
                .map(|&s| pool.narrow(0, s, 1))
                .collect::<CandleResult<Vec<_>>>()?,
            0,
        )?;
        // Round inputs through the tested dtype so both sides see the same values.
        let (mixed, b, a) = (
            mixed.to_dtype(dtype)?.to_dtype(DType::F32)?,
            b.to_dtype(dtype)?.to_dtype(DType::F32)?,
            a.to_dtype(dtype)?.to_dtype(DType::F32)?,
        );
        let mut cpu_cache = GdnLayerCache {
            conv_state: Tensor::zeros(
                (batch_size, dims.conv_dim, dims.conv_kernel_size),
                DType::F32,
                &cpu,
            )?,
            recurrent_state: gathered,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
        };
        let expected = apply_recurrence_from_convolved(
            &mixed,
            &b,
            &a,
            &a_log,
            &dt_bias,
            &dims,
            batch_size,
            1,
            &mut cpu_cache,
            DType::F32,
        )?;

        let mut device_cache = GdnLayerCache {
            conv_state: Tensor::zeros(
                (POOL_ROWS, dims.conv_dim, dims.conv_kernel_size),
                DType::F32,
                device,
            )?,
            recurrent_state: pool.to_device(device)?,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: Some(Tensor::new(&slots, device)?),
            pending_transitions: None,
            deferred_state: None,
        };
        let actual = apply_recurrence_from_convolved(
            &mixed.to_dtype(dtype)?.to_device(device)?,
            &b.to_dtype(dtype)?.to_device(device)?,
            &a.to_dtype(dtype)?.to_device(device)?,
            &a_log.to_device(device)?,
            &dt_bias.to_device(device)?,
            &dims,
            batch_size,
            1,
            &mut device_cache,
            dtype,
        )?;
        let max_diff = |x: &Tensor, y: &Tensor| -> CandleResult<f32> {
            (x.to_dtype(DType::F32)?.to_device(&cpu)? - y.to_dtype(DType::F32)?.to_device(&cpu)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()
        };
        let out_diff = max_diff(&actual, &expected)?;
        let pool_after = device_cache.recurrent_state.to_device(&cpu)?;
        let mut state_diff = 0f32;
        for (row, &slot) in slot_rows.iter().enumerate() {
            state_diff = state_diff.max(max_diff(
                &pool_after.narrow(0, slot, 1)?,
                &cpu_cache.recurrent_state.narrow(0, row, 1)?,
            )?);
        }
        let untouched = max_diff(&pool_after.narrow(0, 1, 1)?, &pool.narrow(0, 1, 1)?)?;
        assert!(
            out_diff < tol,
            "{layout:?} {dtype:?}: output diff {out_diff}"
        );
        assert!(
            state_diff < 1e-3,
            "{layout:?} {dtype:?}: state diff {state_diff}"
        );
        assert_eq!(untouched, 0.0, "unselected pool rows must not change");
    }
    Ok(())
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn multi_token_from_convolved_matches_cpu(
    device: &Device,
    cases: &[RecurrenceParityCase],
) -> CandleResult<()> {
    const POOL_ROWS: usize = 5;
    const SEQ: usize = 7;
    let cpu = Device::Cpu;
    let slots = [3u32, 0];
    let batch_size = slots.len();
    for &RecurrenceParityCase {
        layout,
        dtype,
        pooled,
        tolerance: tol,
    } in cases
    {
        let dims = dims_with_layout(2, 4, 128, 128, layout);
        let round = |t: Tensor| -> CandleResult<Tensor> { t.to_dtype(dtype)?.to_dtype(DType::F32) };
        let mixed = round(Tensor::from_vec(
            patterned(batch_size * SEQ * dims.conv_dim, 1, 0.4, 0.05),
            (batch_size, SEQ, dims.conv_dim),
            &cpu,
        )?)?;
        let b = round(Tensor::from_vec(
            patterned(batch_size * SEQ * dims.num_v_heads, 2, 0.8, 0.1),
            (batch_size, SEQ, dims.num_v_heads),
            &cpu,
        )?)?;
        let a = round(Tensor::from_vec(
            patterned(batch_size * SEQ * dims.num_v_heads, 3, 0.8, -0.05),
            (batch_size, SEQ, dims.num_v_heads),
            &cpu,
        )?)?;
        let a_log = Tensor::from_vec(
            patterned(dims.num_v_heads, 4, 0.2, -0.2),
            (dims.num_v_heads,),
            &cpu,
        )?;
        let dt_bias = Tensor::from_vec(
            patterned(dims.num_v_heads, 5, 0.3, 0.3),
            (dims.num_v_heads,),
            &cpu,
        )?;
        let state_len = dims.num_v_heads * dims.head_k_dim * dims.head_v_dim;
        let pool = Tensor::from_vec(
            patterned(POOL_ROWS * state_len, 6, 0.05, 0.0),
            (
                POOL_ROWS,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            &cpu,
        )?;
        let rows = slots.iter().map(|&s| s as usize).collect::<Vec<_>>();
        let gathered = Tensor::cat(
            &rows
                .iter()
                .map(|&r| pool.narrow(0, r, 1))
                .collect::<CandleResult<Vec<_>>>()?,
            0,
        )?;
        let mut cpu_cache = GdnLayerCache {
            conv_state: Tensor::zeros(
                (batch_size, dims.conv_dim, dims.conv_kernel_size),
                DType::F32,
                &cpu,
            )?,
            recurrent_state: gathered.clone(),
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
        };
        let expected = apply_recurrence_from_convolved(
            &mixed,
            &b,
            &a,
            &a_log,
            &dt_bias,
            &dims,
            batch_size,
            SEQ,
            &mut cpu_cache,
            DType::F32,
        )?;

        let (state, slot_tensor) = if pooled {
            (pool.to_device(device)?, Some(Tensor::new(&slots, device)?))
        } else {
            (gathered.to_device(device)?, None)
        };
        let mut device_cache = GdnLayerCache {
            conv_state: Tensor::zeros(
                (1, dims.conv_dim, dims.conv_kernel_size),
                DType::F32,
                device,
            )?,
            recurrent_state: state,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: slot_tensor,
            pending_transitions: None,
            deferred_state: None,
        };
        let actual = apply_recurrence_from_convolved(
            &mixed.to_dtype(dtype)?.to_device(device)?,
            &b.to_dtype(dtype)?.to_device(device)?,
            &a.to_dtype(dtype)?.to_device(device)?,
            &a_log.to_device(device)?,
            &dt_bias.to_device(device)?,
            &dims,
            batch_size,
            SEQ,
            &mut device_cache,
            dtype,
        )?;
        let max_diff = |x: &Tensor, y: &Tensor| -> CandleResult<f32> {
            (x.to_dtype(DType::F32)?.to_device(&cpu)? - y.to_dtype(DType::F32)?.to_device(&cpu)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()
        };
        let tag = format!("{layout:?} {dtype:?} pooled={pooled}");
        let out_diff = max_diff(&actual, &expected)?;
        assert!(out_diff < tol, "{tag}: output diff {out_diff}");
        let state_after = device_cache.recurrent_state.to_device(&cpu)?;
        for (i, &r) in rows.iter().enumerate() {
            let got = if pooled {
                state_after.narrow(0, r, 1)?
            } else {
                state_after.narrow(0, i, 1)?
            };
            let d = max_diff(&got, &cpu_cache.recurrent_state.narrow(0, i, 1)?)?;
            assert!(d < 1e-3, "{tag}: state diff {d}");
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_pooled_decode_from_convolved_matches_cpu() -> CandleResult<()> {
    let case = |layout, dtype, tolerance| RecurrenceParityCase {
        layout,
        dtype,
        pooled: true,
        tolerance,
    };
    pooled_decode_from_convolved_matches_cpu(
        &Device::new_metal(0)?,
        &[
            case(GdnVHeadLayout::Grouped, DType::F32, 1e-4),
            case(GdnVHeadLayout::Tiled, DType::F32, 1e-4),
            case(GdnVHeadLayout::Grouped, DType::BF16, 2e-2),
        ],
    )
}

#[cfg(feature = "metal")]
#[test]
fn metal_multi_token_from_convolved_matches_cpu() -> CandleResult<()> {
    let case = |layout, dtype, pooled, tolerance| RecurrenceParityCase {
        layout,
        dtype,
        pooled,
        tolerance,
    };
    multi_token_from_convolved_matches_cpu(
        &Device::new_metal(0)?,
        &[
            case(GdnVHeadLayout::Grouped, DType::F32, false, 1e-4),
            case(GdnVHeadLayout::Tiled, DType::F32, false, 1e-4),
            case(GdnVHeadLayout::Grouped, DType::BF16, true, 3e-2),
        ],
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_pooled_decode_from_convolved_matches_cpu() -> CandleResult<()> {
    let case = |layout, dtype, tolerance| RecurrenceParityCase {
        layout,
        dtype,
        pooled: true,
        tolerance,
    };
    pooled_decode_from_convolved_matches_cpu(
        &Device::new_cuda(0)?,
        &[
            case(GdnVHeadLayout::Grouped, DType::BF16, 2e-2),
            case(GdnVHeadLayout::Tiled, DType::BF16, 2e-2),
            case(GdnVHeadLayout::Grouped, DType::F16, 2e-3),
        ],
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_multi_token_from_convolved_matches_cpu() -> CandleResult<()> {
    let case = |layout, dtype, tolerance| RecurrenceParityCase {
        layout,
        dtype,
        pooled: true,
        tolerance,
    };
    multi_token_from_convolved_matches_cpu(
        &Device::new_cuda(0)?,
        &[
            case(GdnVHeadLayout::Grouped, DType::BF16, 3e-2),
            case(GdnVHeadLayout::Tiled, DType::BF16, 3e-2),
            case(GdnVHeadLayout::Grouped, DType::F16, 3e-3),
        ],
    )
}

fn converter_tiled_heads(dims: &GdnDims) -> Vec<usize> {
    (0..dims.v_per_group)
        .flat_map(|within_group| {
            (0..dims.num_k_heads).map(move |key_head| key_head * dims.v_per_group + within_group)
        })
        .collect()
}

fn expand_head_map(heads: &[usize], head_dim: usize) -> Vec<usize> {
    heads
        .iter()
        .flat_map(|head| (0..head_dim).map(move |feature| head * head_dim + feature))
        .collect()
}

fn runtime_from_grouped(
    tensor: &Tensor,
    dim: usize,
    runtime_to_grouped: &[usize],
) -> CandleResult<Tensor> {
    let indices = runtime_to_grouped
        .iter()
        .map(|&index| index as u32)
        .collect::<Vec<_>>();
    tensor
        .contiguous()?
        .index_select(
            &Tensor::from_vec(indices, (runtime_to_grouped.len(),), tensor.device())?,
            dim,
        )?
        .contiguous()
}

fn grouped_from_runtime(
    tensor: &Tensor,
    dim: usize,
    runtime_to_grouped: &[usize],
) -> CandleResult<Tensor> {
    let mut grouped_to_runtime = vec![0; runtime_to_grouped.len()];
    for (runtime, &grouped) in runtime_to_grouped.iter().enumerate() {
        grouped_to_runtime[grouped] = runtime;
    }
    runtime_from_grouped(tensor, dim, &grouped_to_runtime)
}

struct BackendStep<'a> {
    mixed_qkv: &'a Tensor,
    b: &'a Tensor,
    a: &'a Tensor,
    a_log: &'a Tensor,
    dt_bias: &'a Tensor,
    conv_weight: &'a Tensor,
    dims: &'a GdnDims,
}

fn run_backend_step(
    step: BackendStep<'_>,
    cache: &mut GdnLayerCache,
    batch_kind: RecurrentBatchKind,
) -> CandleResult<(Tensor, Tensor)> {
    let (batch_size, seq_len, _) = step.mixed_qkv.dims3()?;
    let convolved = causal_conv1d(
        step.mixed_qkv,
        step.conv_weight,
        step.dims,
        cache,
        batch_kind,
    )?;
    let recurrent = apply_recurrence_from_convolved(
        &convolved,
        step.b,
        step.a,
        step.a_log,
        step.dt_bias,
        step.dims,
        batch_size,
        seq_len,
        cache,
        DType::F32,
    )?;
    Ok((convolved, recurrent))
}

fn finish_backend_step(
    recurrent: &Tensor,
    z: &Tensor,
    norm_weight: &Tensor,
    out_weight: &Tensor,
    dims: &GdnDims,
) -> CandleResult<Tensor> {
    let (batch_size, seq_len, num_v_heads, head_v_dim) = z.dims4()?;
    assert_eq!(num_v_heads, dims.num_v_heads);
    assert_eq!(head_v_dim, dims.head_v_dim);
    let recurrent = recurrent.reshape(((), dims.head_v_dim))?;
    let gate = candle_nn::ops::silu(&z.reshape(((), dims.head_v_dim))?)?;
    let variance = recurrent.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = recurrent.broadcast_div(&(variance + TEST_RMS_NORM_EPS)?.sqrt()?)?;
    let normalized = normalized
        .broadcast_mul(norm_weight)?
        .broadcast_mul(&gate)?
        .reshape((batch_size * seq_len, dims.value_dim))?;
    normalized
        .matmul(&out_weight.t()?)?
        .reshape((batch_size, seq_len, dims.hidden_size))
}

fn assert_layout_equivalent(
    grouped: &Tensor,
    tiled: &Tensor,
    dim: usize,
    runtime_to_grouped: &[usize],
) -> CandleResult<()> {
    assert_close(
        grouped,
        &grouped_from_runtime(tiled, dim, runtime_to_grouped)?,
    )
}

fn assert_cache_layout_equivalent(
    grouped: &GdnLayerCache,
    tiled: &GdnLayerCache,
    conv_runtime_to_grouped: &[usize],
    head_runtime_to_grouped: &[usize],
) -> CandleResult<()> {
    assert_layout_equivalent(
        &grouped.conv_state,
        &tiled.conv_state,
        1,
        conv_runtime_to_grouped,
    )?;
    assert_layout_equivalent(
        &grouped.recurrent_state,
        &tiled.recurrent_state,
        1,
        head_runtime_to_grouped,
    )
}

fn run_decode_case(dims: GdnDims, batch_size: usize) -> CandleResult<()> {
    let dev = Device::Cpu;
    let seq_len = 1;
    let mixed = Tensor::from_vec(
        patterned(batch_size * dims.conv_dim, 1, 0.08, 0.01),
        (batch_size, seq_len, dims.conv_dim),
        &dev,
    )?;
    let b = Tensor::from_vec(
        patterned(batch_size * dims.num_v_heads, 2, 0.2, 0.1),
        (batch_size, seq_len, dims.num_v_heads),
        &dev,
    )?;
    let a = Tensor::from_vec(
        patterned(batch_size * dims.num_v_heads, 3, 0.2, -0.05),
        (batch_size, seq_len, dims.num_v_heads),
        &dev,
    )?;
    let a_log = Tensor::from_vec(
        patterned(dims.num_v_heads, 4, 0.05, -0.2),
        (dims.num_v_heads,),
        &dev,
    )?;
    let dt_bias = Tensor::from_vec(
        patterned(dims.num_v_heads, 5, 0.1, 0.3),
        (dims.num_v_heads,),
        &dev,
    )?;
    let initial_state = Tensor::from_vec(
        patterned(
            batch_size * dims.num_v_heads * dims.head_k_dim * dims.head_v_dim,
            6,
            0.02,
            0.0,
        ),
        (
            batch_size,
            dims.num_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        ),
        &dev,
    )?;
    let conv_state = Tensor::zeros(
        (batch_size, dims.conv_dim, dims.conv_kernel_size),
        DType::F32,
        &dev,
    )?;
    let mut fast_cache = GdnLayerCache {
        conv_state: conv_state.clone(),
        recurrent_state: initial_state.clone(),
        state_layout: RecurrentStateLayout::GdnKeyMajor,
        slots: None,
        pending_transitions: None,
        deferred_state: None,
    };
    let fast = decode_recurrence_cpu_from_convolved(
        &mixed,
        &b,
        &a,
        &a_log,
        &dt_bias,
        &dims,
        batch_size,
        &mut fast_cache,
        DType::F32,
    )?;

    let q = mixed.narrow(D::Minus1, 0, dims.key_dim)?;
    let k = mixed.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
    let v = mixed.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;
    let q = q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let k = k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let v = v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?;
    let q = expand_k_heads(&q, &dims, batch_size, seq_len)?;
    let k = expand_k_heads(&k, &dims, batch_size, seq_len)?;
    let (beta, g) = compute_beta_g(&b, &a, &a_log, &dt_bias, DType::F32)?;
    let q = l2_norm(&q, QK_NORM_EPS)?;
    let k = l2_norm(&k, QK_NORM_EPS)?;
    let mut reference_cache = GdnLayerCache {
        conv_state,
        recurrent_state: initial_state,
        state_layout: RecurrentStateLayout::GdnKeyMajor,
        slots: None,
        pending_transitions: None,
        deferred_state: None,
    };
    let reference =
        gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut reference_cache.recurrent_state)?;

    assert_close(&fast, &reference)?;
    assert_close(
        &fast_cache.recurrent_state,
        &reference_cache.recurrent_state,
    )
}

fn causal_conv1d_update_reference(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> CandleResult<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    let x_t = x.transpose(1, 2)?.contiguous()?;
    let state_len = cache.conv_state.dim(2)?;
    let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
    let new_len = hidden_new.dim(2)?;
    cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(hidden_new.dtype())?;
    let mut conv_outputs = Vec::with_capacity(seq_len);
    let total_len = hidden_new.dim(2)?;
    for i in (total_len - seq_len)..total_len {
        let window = hidden_new.narrow(2, i + 1 - dims.conv_kernel_size, dims.conv_kernel_size)?;
        let out = window
            .broadcast_mul(&weight.unsqueeze(0)?)?
            .sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    candle_nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}

#[test]
fn key_heads_expand_in_grouped_and_tiled_order() -> CandleResult<()> {
    let dev = Device::Cpu;
    let keys = Tensor::from_vec(vec![10.0f32, 20.0], (1, 1, 2, 1), &dev)?;
    let grouped = dims_with_layout(2, 4, 1, 1, GdnVHeadLayout::Grouped);
    let tiled = dims_with_layout(2, 4, 1, 1, GdnVHeadLayout::Tiled);

    assert_eq!(
        expand_k_heads(&keys, &grouped, 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [10.0, 10.0, 20.0, 20.0]
    );
    assert_eq!(
        expand_k_heads(&keys, &tiled, 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        [10.0, 20.0, 10.0, 20.0]
    );
    Ok(())
}

#[test]
fn converter_tiled_backend_matches_grouped_prefill_and_decode() -> CandleResult<()> {
    let dev = Device::Cpu;
    let batch_size = 2;
    let prefill_len = 5;
    let mut grouped_dims = dims_with_layout(2, 6, 3, 2, GdnVHeadLayout::Grouped);
    grouped_dims.hidden_size = 7;
    let mut tiled_dims = grouped_dims;
    tiled_dims.v_head_layout = GdnVHeadLayout::Tiled;

    let head_runtime_to_grouped = converter_tiled_heads(&grouped_dims);
    let value_runtime_to_grouped =
        expand_head_map(&head_runtime_to_grouped, grouped_dims.head_v_dim);
    let mut conv_runtime_to_grouped = (0..2 * grouped_dims.key_dim).collect::<Vec<_>>();
    conv_runtime_to_grouped.extend(
        value_runtime_to_grouped
            .iter()
            .map(|&index| 2 * grouped_dims.key_dim + index),
    );
    assert_eq!(head_runtime_to_grouped, [0, 3, 1, 4, 2, 5]);
    assert_eq!(
        value_runtime_to_grouped,
        [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]
    );
    assert_eq!(conv_runtime_to_grouped.len(), grouped_dims.conv_dim);

    let conv_weight_grouped = Tensor::from_vec(
        patterned(
            grouped_dims.conv_dim * grouped_dims.conv_kernel_size,
            31,
            0.09,
            -0.01,
        ),
        (grouped_dims.conv_dim, 1, grouped_dims.conv_kernel_size),
        &dev,
    )?;
    let conv_weight_tiled =
        runtime_from_grouped(&conv_weight_grouped, 0, &conv_runtime_to_grouped)?;
    let a_log_grouped = Tensor::from_vec(
        patterned(grouped_dims.num_v_heads, 32, 0.08, -0.2),
        (grouped_dims.num_v_heads,),
        &dev,
    )?;
    let a_log_tiled = runtime_from_grouped(&a_log_grouped, 0, &head_runtime_to_grouped)?;
    let dt_bias_grouped = Tensor::from_vec(
        patterned(grouped_dims.num_v_heads, 33, 0.12, 0.3),
        (grouped_dims.num_v_heads,),
        &dev,
    )?;
    let dt_bias_tiled = runtime_from_grouped(&dt_bias_grouped, 0, &head_runtime_to_grouped)?;
    let norm_weight = Tensor::from_vec(
        patterned(grouped_dims.head_v_dim, 34, 0.15, 1.0),
        (grouped_dims.head_v_dim,),
        &dev,
    )?;
    let out_weight_grouped = Tensor::from_vec(
        patterned(
            grouped_dims.hidden_size * grouped_dims.value_dim,
            35,
            0.1,
            0.01,
        ),
        (grouped_dims.hidden_size, grouped_dims.value_dim),
        &dev,
    )?;
    let out_weight_tiled = runtime_from_grouped(&out_weight_grouped, 1, &value_runtime_to_grouped)?;

    let conv_state_grouped = Tensor::from_vec(
        patterned(
            batch_size * grouped_dims.conv_dim * grouped_dims.conv_kernel_size,
            36,
            0.03,
            0.0,
        ),
        (
            batch_size,
            grouped_dims.conv_dim,
            grouped_dims.conv_kernel_size,
        ),
        &dev,
    )?;
    let recurrent_state_grouped = Tensor::from_vec(
        patterned(
            batch_size
                * grouped_dims.num_v_heads
                * grouped_dims.head_k_dim
                * grouped_dims.head_v_dim,
            37,
            0.025,
            0.0,
        ),
        (
            batch_size,
            grouped_dims.num_v_heads,
            grouped_dims.head_k_dim,
            grouped_dims.head_v_dim,
        ),
        &dev,
    )?;
    let mut grouped_cache = GdnLayerCache {
        conv_state: conv_state_grouped.clone(),
        recurrent_state: recurrent_state_grouped.clone(),
        state_layout: RecurrentStateLayout::GdnKeyMajor,
        slots: None,
        pending_transitions: None,
        deferred_state: None,
    };
    let mut tiled_cache = GdnLayerCache {
        conv_state: runtime_from_grouped(&conv_state_grouped, 1, &conv_runtime_to_grouped)?,
        recurrent_state: runtime_from_grouped(
            &recurrent_state_grouped,
            1,
            &head_runtime_to_grouped,
        )?,
        state_layout: RecurrentStateLayout::GdnKeyMajor,
        slots: None,
        pending_transitions: None,
        deferred_state: None,
    };

    let prefill_mixed_grouped = Tensor::from_vec(
        patterned(
            batch_size * prefill_len * grouped_dims.conv_dim,
            38,
            0.08,
            0.01,
        ),
        (batch_size, prefill_len, grouped_dims.conv_dim),
        &dev,
    )?;
    let prefill_mixed_tiled =
        runtime_from_grouped(&prefill_mixed_grouped, 2, &conv_runtime_to_grouped)?;
    let prefill_b_grouped = Tensor::from_vec(
        patterned(
            batch_size * prefill_len * grouped_dims.num_v_heads,
            39,
            0.2,
            0.1,
        ),
        (batch_size, prefill_len, grouped_dims.num_v_heads),
        &dev,
    )?;
    let prefill_b_tiled = runtime_from_grouped(&prefill_b_grouped, 2, &head_runtime_to_grouped)?;
    let prefill_a_grouped = Tensor::from_vec(
        patterned(
            batch_size * prefill_len * grouped_dims.num_v_heads,
            40,
            0.18,
            -0.04,
        ),
        (batch_size, prefill_len, grouped_dims.num_v_heads),
        &dev,
    )?;
    let prefill_a_tiled = runtime_from_grouped(&prefill_a_grouped, 2, &head_runtime_to_grouped)?;
    let prefill_z_grouped = Tensor::from_vec(
        patterned(
            batch_size * prefill_len * grouped_dims.value_dim,
            41,
            0.14,
            0.02,
        ),
        (
            batch_size,
            prefill_len,
            grouped_dims.num_v_heads,
            grouped_dims.head_v_dim,
        ),
        &dev,
    )?;
    let prefill_z_tiled = runtime_from_grouped(&prefill_z_grouped, 2, &head_runtime_to_grouped)?;

    let (grouped_beta, grouped_g) = compute_beta_g(
        &prefill_b_grouped,
        &prefill_a_grouped,
        &a_log_grouped,
        &dt_bias_grouped,
        DType::F32,
    )?;
    let (tiled_beta, tiled_g) = compute_beta_g(
        &prefill_b_tiled,
        &prefill_a_tiled,
        &a_log_tiled,
        &dt_bias_tiled,
        DType::F32,
    )?;
    assert_layout_equivalent(&grouped_beta, &tiled_beta, 2, &head_runtime_to_grouped)?;
    assert_layout_equivalent(&grouped_g, &tiled_g, 2, &head_runtime_to_grouped)?;

    let (grouped_conv, grouped_recurrent) = run_backend_step(
        BackendStep {
            mixed_qkv: &prefill_mixed_grouped,
            b: &prefill_b_grouped,
            a: &prefill_a_grouped,
            a_log: &a_log_grouped,
            dt_bias: &dt_bias_grouped,
            conv_weight: &conv_weight_grouped,
            dims: &grouped_dims,
        },
        &mut grouped_cache,
        RecurrentBatchKind::Prefill,
    )?;
    let (tiled_conv, tiled_recurrent) = run_backend_step(
        BackendStep {
            mixed_qkv: &prefill_mixed_tiled,
            b: &prefill_b_tiled,
            a: &prefill_a_tiled,
            a_log: &a_log_tiled,
            dt_bias: &dt_bias_tiled,
            conv_weight: &conv_weight_tiled,
            dims: &tiled_dims,
        },
        &mut tiled_cache,
        RecurrentBatchKind::Prefill,
    )?;
    assert_layout_equivalent(&grouped_conv, &tiled_conv, 2, &conv_runtime_to_grouped)?;
    for offset in [0, grouped_dims.key_dim] {
        let grouped_keys = grouped_conv
            .narrow(2, offset, grouped_dims.key_dim)?
            .reshape((
                batch_size,
                prefill_len,
                grouped_dims.num_k_heads,
                grouped_dims.head_k_dim,
            ))?;
        let tiled_keys = tiled_conv.narrow(2, offset, tiled_dims.key_dim)?.reshape((
            batch_size,
            prefill_len,
            tiled_dims.num_k_heads,
            tiled_dims.head_k_dim,
        ))?;
        let grouped_expanded =
            expand_k_heads(&grouped_keys, &grouped_dims, batch_size, prefill_len)?;
        let tiled_expanded = expand_k_heads(&tiled_keys, &tiled_dims, batch_size, prefill_len)?;
        assert_layout_equivalent(
            &grouped_expanded,
            &tiled_expanded,
            2,
            &head_runtime_to_grouped,
        )?;
    }
    assert_layout_equivalent(
        &grouped_recurrent,
        &tiled_recurrent,
        2,
        &head_runtime_to_grouped,
    )?;
    let grouped_prefill_output = finish_backend_step(
        &grouped_recurrent,
        &prefill_z_grouped,
        &norm_weight,
        &out_weight_grouped,
        &grouped_dims,
    )?;
    let tiled_prefill_output = finish_backend_step(
        &tiled_recurrent,
        &prefill_z_tiled,
        &norm_weight,
        &out_weight_tiled,
        &tiled_dims,
    )?;
    assert_close(&grouped_prefill_output, &tiled_prefill_output)?;
    assert_cache_layout_equivalent(
        &grouped_cache,
        &tiled_cache,
        &conv_runtime_to_grouped,
        &head_runtime_to_grouped,
    )?;

    let decode_mixed_grouped = Tensor::from_vec(
        patterned(batch_size * grouped_dims.conv_dim, 42, 0.08, 0.01),
        (batch_size, 1, grouped_dims.conv_dim),
        &dev,
    )?;
    let decode_mixed_tiled =
        runtime_from_grouped(&decode_mixed_grouped, 2, &conv_runtime_to_grouped)?;
    let decode_b_grouped = Tensor::from_vec(
        patterned(batch_size * grouped_dims.num_v_heads, 43, 0.2, 0.1),
        (batch_size, 1, grouped_dims.num_v_heads),
        &dev,
    )?;
    let decode_b_tiled = runtime_from_grouped(&decode_b_grouped, 2, &head_runtime_to_grouped)?;
    let decode_a_grouped = Tensor::from_vec(
        patterned(batch_size * grouped_dims.num_v_heads, 44, 0.18, -0.04),
        (batch_size, 1, grouped_dims.num_v_heads),
        &dev,
    )?;
    let decode_a_tiled = runtime_from_grouped(&decode_a_grouped, 2, &head_runtime_to_grouped)?;
    let decode_z_grouped = Tensor::from_vec(
        patterned(batch_size * grouped_dims.value_dim, 45, 0.14, 0.02),
        (
            batch_size,
            1,
            grouped_dims.num_v_heads,
            grouped_dims.head_v_dim,
        ),
        &dev,
    )?;
    let decode_z_tiled = runtime_from_grouped(&decode_z_grouped, 2, &head_runtime_to_grouped)?;

    let (grouped_conv, grouped_recurrent) = run_backend_step(
        BackendStep {
            mixed_qkv: &decode_mixed_grouped,
            b: &decode_b_grouped,
            a: &decode_a_grouped,
            a_log: &a_log_grouped,
            dt_bias: &dt_bias_grouped,
            conv_weight: &conv_weight_grouped,
            dims: &grouped_dims,
        },
        &mut grouped_cache,
        RecurrentBatchKind::Decode,
    )?;
    let (tiled_conv, tiled_recurrent) = run_backend_step(
        BackendStep {
            mixed_qkv: &decode_mixed_tiled,
            b: &decode_b_tiled,
            a: &decode_a_tiled,
            a_log: &a_log_tiled,
            dt_bias: &dt_bias_tiled,
            conv_weight: &conv_weight_tiled,
            dims: &tiled_dims,
        },
        &mut tiled_cache,
        RecurrentBatchKind::Decode,
    )?;
    assert_layout_equivalent(&grouped_conv, &tiled_conv, 2, &conv_runtime_to_grouped)?;
    assert_layout_equivalent(
        &grouped_recurrent,
        &tiled_recurrent,
        2,
        &head_runtime_to_grouped,
    )?;
    let grouped_decode_output = finish_backend_step(
        &grouped_recurrent,
        &decode_z_grouped,
        &norm_weight,
        &out_weight_grouped,
        &grouped_dims,
    )?;
    let tiled_decode_output = finish_backend_step(
        &tiled_recurrent,
        &decode_z_tiled,
        &norm_weight,
        &out_weight_tiled,
        &tiled_dims,
    )?;
    assert_close(&grouped_decode_output, &tiled_decode_output)?;
    assert_cache_layout_equivalent(
        &grouped_cache,
        &tiled_cache,
        &conv_runtime_to_grouped,
        &head_runtime_to_grouped,
    )
}

#[test]
fn causal_conv1d_update_cpu_matches_tensor_path() -> CandleResult<()> {
    let dev = Device::Cpu;
    let dims = dims(2, 4, 5, 3);
    let batch_size = 2;
    let x = Tensor::from_vec(
        patterned(batch_size * dims.conv_dim, 7, 0.08, 0.01),
        (batch_size, 1, dims.conv_dim),
        &dev,
    )?;
    let weight = Tensor::from_vec(
        patterned(dims.conv_dim * dims.conv_kernel_size, 8, 0.05, -0.01),
        (dims.conv_dim, 1, dims.conv_kernel_size),
        &dev,
    )?;
    let initial_state = Tensor::from_vec(
        patterned(
            batch_size * dims.conv_dim * dims.conv_kernel_size,
            9,
            0.03,
            0.0,
        ),
        (batch_size, dims.conv_dim, dims.conv_kernel_size),
        &dev,
    )?;
    let recurrent_state = Tensor::zeros(
        (
            batch_size,
            dims.num_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        ),
        DType::F32,
        &dev,
    )?;
    let mut fast_cache = GdnLayerCache {
        conv_state: initial_state.clone(),
        recurrent_state: recurrent_state.clone(),
        state_layout: RecurrentStateLayout::GdnKeyMajor,
        slots: None,
        pending_transitions: None,
        deferred_state: None,
    };
    let mut reference_cache = GdnLayerCache {
        conv_state: initial_state,
        recurrent_state,
        state_layout: RecurrentStateLayout::GdnKeyMajor,
        slots: None,
        pending_transitions: None,
        deferred_state: None,
    };

    let fast = causal_conv1d_update_cpu(&x, &weight, &dims, &mut fast_cache)?;
    let reference = causal_conv1d_update_reference(&x, &weight, &dims, &mut reference_cache)?;

    assert_close(&fast, &reference)?;
    assert_close(&fast_cache.conv_state, &reference_cache.conv_state)
}

#[test]
fn causal_conv1d_prefill_continues_from_existing_state() -> CandleResult<()> {
    let dev = Device::Cpu;
    let dims = dims(2, 4, 5, 3);
    let batch_size = 2;
    let seq_len = 7;
    let split = 3;
    let x = Tensor::from_vec(
        patterned(batch_size * seq_len * dims.conv_dim, 10, 0.08, 0.01),
        (batch_size, seq_len, dims.conv_dim),
        &dev,
    )?;
    let weight = Tensor::from_vec(
        patterned(dims.conv_dim * dims.conv_kernel_size, 11, 0.05, -0.01),
        (dims.conv_dim, 1, dims.conv_kernel_size),
        &dev,
    )?;
    let conv_state = Tensor::from_vec(
        patterned(
            batch_size * dims.conv_dim * dims.conv_kernel_size,
            12,
            0.03,
            0.0,
        ),
        (batch_size, dims.conv_dim, dims.conv_kernel_size),
        &dev,
    )?;
    let recurrent_state = Tensor::zeros(
        (
            batch_size,
            dims.num_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        ),
        DType::F32,
        &dev,
    )?;
    let mut one_shot_cache = GdnLayerCache {
        conv_state: conv_state.clone(),
        recurrent_state: recurrent_state.clone(),
        state_layout: RecurrentStateLayout::GdnKeyMajor,
        slots: None,
        pending_transitions: None,
        deferred_state: None,
    };
    let mut chunked_cache = GdnLayerCache {
        conv_state,
        recurrent_state,
        state_layout: RecurrentStateLayout::GdnKeyMajor,
        slots: None,
        pending_transitions: None,
        deferred_state: None,
    };

    let one_shot = causal_conv1d_full(&x, &weight, &dims, &mut one_shot_cache)?;
    let first = causal_conv1d_full(&x.narrow(1, 0, split)?, &weight, &dims, &mut chunked_cache)?;
    let second = causal_conv1d_full(
        &x.narrow(1, split, seq_len - split)?,
        &weight,
        &dims,
        &mut chunked_cache,
    )?;

    assert_close(&one_shot, &Tensor::cat(&[first, second], 1)?)?;
    assert_close(&one_shot_cache.conv_state, &chunked_cache.conv_state)
}

#[test]
fn decode_recurrence_cpu_matches_tensor_path() -> CandleResult<()> {
    run_decode_case(dims(2, 4, 5, 3), 2)?;
    run_decode_case(dims(3, 3, 4, 2), 1)?;
    run_decode_case(dims_with_layout(2, 4, 5, 3, GdnVHeadLayout::Tiled), 2)
}
