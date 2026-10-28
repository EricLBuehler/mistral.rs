use super::*;

#[derive(Clone, Copy, Debug)]
struct Case {
    rows: usize,
    n: usize,
    k: usize,
    routed: bool,
    per_route: bool,
    weight_offset: usize,
}

fn output(case: Case, dtype: DType, device: &Device) -> Result<Tensor> {
    let experts = if case.routed { 3 } else { 1 };
    let topk = if case.routed { 3 } else { 1 };
    let input_rows = case.rows * if case.per_route { topk } else { 1 };
    let elements = experts * case.n * case.k;
    let weight = Tensor::from_vec(
        (0..elements + case.weight_offset)
            .map(|i| F8E4M3::from_f32(((i * 13 % 47) as f32 - 23.0) * 0.125))
            .collect::<Vec<_>>(),
        elements + case.weight_offset,
        device,
    )?
    .narrow(0, case.weight_offset, elements)?;
    let scale_count = elements / MXFP8_BLOCK_SIZE;
    let scales = Tensor::from_vec(
        (0..scale_count + 1)
            .map(|i| 123 + (i % 7) as u8)
            .collect::<Vec<_>>(),
        scale_count + 1,
        device,
    )?
    .narrow(0, 1, scale_count)?;
    let (weight, scales) = if case.routed {
        (
            weight.reshape((experts, case.n, case.k))?,
            scales.reshape((experts, case.n, case.k / MXFP8_BLOCK_SIZE))?,
        )
    } else {
        (
            weight.reshape((case.n, case.k))?,
            scales.reshape((case.n, case.k / MXFP8_BLOCK_SIZE))?,
        )
    };
    let input_elements = input_rows * case.k;
    let input = Tensor::from_vec(
        (0..input_elements + 1)
            .map(|i| ((i * 19 % 97) as f32 - 48.0) * 0.03125)
            .collect::<Vec<_>>(),
        input_elements + 1,
        device,
    )?
    .to_dtype(dtype)?
    .narrow(0, 1, input_elements)?
    .reshape((input_rows, case.k))?;
    let indices = case
        .routed
        .then(|| {
            Tensor::from_vec(
                (0..case.rows * topk)
                    .map(|i| [2u32, 0, 2, 1, 2, 0][i % 6])
                    .collect::<Vec<_>>(),
                (case.rows, topk),
                device,
            )
        })
        .transpose()?;
    input.apply_op3_no_bwd(&weight, &scales, &Matmul { indices })
}

#[test]
fn cuda_vector_gemv_matches_cpu_for_views_routes_and_reduction_tails() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (rows, n, k) in [(1, 1, 512), (1, 3, 640), (1, 17, 1056), (2, 3, 160)] {
            for (routed, per_route) in [(false, false), (true, false), (true, true)] {
                for weight_offset in [0, 1] {
                    let case = Case {
                        rows,
                        n,
                        k,
                        routed,
                        per_route,
                        weight_offset,
                    };
                    let reference = output(case, dtype, &Device::Cpu)?;
                    let actual = output(case, dtype, &device)?;
                    assert_eq!(actual.dims(), reference.dims());
                    let reference = reference
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let actual = actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let tolerance = match dtype {
                        DType::F32 => 1e-4,
                        DType::F16 => 0.002,
                        _ => 0.015,
                    };
                    for (index, (actual, reference)) in actual.iter().zip(reference).enumerate() {
                        assert!(
                            (actual - reference).abs() <= tolerance * reference.abs().max(1.0),
                            "{dtype:?} {case:?} element {index}: {actual} vs {reference}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
