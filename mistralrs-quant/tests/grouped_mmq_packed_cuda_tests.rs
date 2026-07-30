#![cfg(feature = "cuda")]

use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, Result, Storage, Tensor,
};
use mistralrs_quant::{
    grouped_moe_mmq, grouped_moe_mmq_from_glu_packed, grouped_moe_mmq_from_glu_sorted_pair,
    grouped_moe_mmq_pair_packed, moe_dispatch_build, GluActivationType,
};

const NUM_EXPERTS: usize = 3;
const NUM_TOKENS: usize = 4;
const TOPK: usize = 2;
const TOTAL_ASSIGNMENTS: usize = NUM_TOKENS * TOPK;
const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 96;
const TOLERANCE: f32 = 5e-4;

fn patterned(shape: impl Into<candle_core::Shape>, salt: usize, scale: f32) -> Result<Tensor> {
    let shape = shape.into();
    let values = (0..shape.elem_count())
        .map(|index| {
            let value = (index.wrapping_mul(37) + salt.wrapping_mul(19)) % 211;
            (value as f32 / 105.0 - 1.0) * scale
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape, &Device::Cpu)
}

fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual
        .contiguous()?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .contiguous()?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max_error = 0.0f32;
    let mut max_index = 0usize;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(actual.is_finite(), "non-finite actual value at {index}");
        assert!(expected.is_finite(), "non-finite expected value at {index}");
        let error = (actual - expected).abs() / (1.0 + expected.abs());
        if error > max_error {
            max_error = error;
            max_index = index;
        }
    }
    assert!(
        max_error <= TOLERANCE,
        "relative error {max_error} at {max_index} exceeds {TOLERANCE}"
    );
    Ok(())
}

#[test]
fn packed_gate_up_and_sorted_pair_glu_preserve_route_order() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    let dev = cuda.as_cuda_device()?;
    let xs = patterned((NUM_TOKENS, HIDDEN), 3, 0.7)?.to_device(&cuda)?;
    let gate = QTensor::quantize_onto(
        &patterned((NUM_EXPERTS, INTERMEDIATE, HIDDEN), 11, 0.11)?,
        GgmlDType::Q4_0,
        &cuda,
    )?;
    let up = QTensor::quantize_onto(
        &patterned((NUM_EXPERTS, INTERMEDIATE, HIDDEN), 29, 0.09)?,
        GgmlDType::Q4_0,
        &cuda,
    )?;
    let down = QTensor::quantize_onto(
        &patterned((NUM_EXPERTS, HIDDEN, INTERMEDIATE), 47, 0.1)?,
        GgmlDType::Q4_0,
        &cuda,
    )?;

    let route_experts = [2u32, 0, 1, 2, 0, 1, 2, 1];
    let topk_ids = Tensor::from_slice(&route_experts, (NUM_TOKENS, TOPK), &cuda)?;
    let topk_ids = topk_ids.flatten_all()?.contiguous()?;
    let (topk_storage, topk_layout) = topk_ids.storage_and_layout();
    assert_eq!(topk_layout.start_offset(), 0);
    let Storage::Cuda(topk_cuda) = &*topk_storage else {
        unreachable!()
    };
    let topk_slice = topk_cuda.as_cuda_slice::<u32>()?;
    let (expert_bounds, sorted_token_ids, sorted_source_ids) =
        moe_dispatch_build(topk_slice, TOTAL_ASSIGNMENTS, NUM_EXPERTS, TOPK, dev)?;

    let sorted_routes = dev.clone_dtoh(&sorted_token_ids)?;
    assert_ne!(
        sorted_routes,
        (0..TOTAL_ASSIGNMENTS as u32).collect::<Vec<_>>()
    );
    let sorted_experts = sorted_routes
        .iter()
        .map(|&route| route_experts[route as usize])
        .collect::<Vec<_>>();
    assert!(sorted_experts.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(sorted_experts.first(), Some(&0));
    assert_eq!(sorted_experts.last(), Some(&2));

    let packed = grouped_moe_mmq_pair_packed(
        &gate,
        &up,
        &xs,
        &sorted_source_ids,
        &sorted_token_ids,
        &expert_bounds,
        TOTAL_ASSIGNMENTS,
        TOPK,
        NUM_EXPERTS,
        dev,
    )?;
    assert_eq!(packed.dims2()?, (TOTAL_ASSIGNMENTS, 2 * INTERMEDIATE));
    assert!(packed.is_contiguous());

    let gate_flat = grouped_moe_mmq(
        &gate,
        &xs,
        &sorted_source_ids,
        &sorted_token_ids,
        &expert_bounds,
        TOTAL_ASSIGNMENTS,
        NUM_TOKENS,
        NUM_EXPERTS,
        dev,
    )?;
    let up_flat = grouped_moe_mmq(
        &up,
        &xs,
        &sorted_source_ids,
        &sorted_token_ids,
        &expert_bounds,
        TOTAL_ASSIGNMENTS,
        NUM_TOKENS,
        NUM_EXPERTS,
        dev,
    )?;
    let packed_gate = packed.narrow(1, 0, INTERMEDIATE)?;
    let packed_up = packed.narrow(1, INTERMEDIATE, INTERMEDIATE)?;
    assert_close(&packed_gate, &gate_flat)?;
    assert_close(&packed_up, &up_flat)?;
    let gate_up_difference = (&gate_flat - &up_flat)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(gate_up_difference > 1e-3);

    let identity = Tensor::from_vec(
        (0..TOTAL_ASSIGNMENTS as u32).collect::<Vec<_>>(),
        (TOTAL_ASSIGNMENTS,),
        &cuda,
    )?;
    let (identity_storage, identity_layout) = identity.storage_and_layout();
    assert_eq!(identity_layout.start_offset(), 0);
    let Storage::Cuda(identity_cuda) = &*identity_storage else {
        unreachable!()
    };
    let identity_slice = identity_cuda.as_cuda_slice::<u32>()?;
    let gate_sorted = grouped_moe_mmq(
        &gate,
        &xs,
        &sorted_source_ids,
        identity_slice,
        &expert_bounds,
        TOTAL_ASSIGNMENTS,
        NUM_TOKENS,
        NUM_EXPERTS,
        dev,
    )?;
    let up_sorted = grouped_moe_mmq(
        &up,
        &xs,
        &sorted_source_ids,
        identity_slice,
        &expert_bounds,
        TOTAL_ASSIGNMENTS,
        NUM_TOKENS,
        NUM_EXPERTS,
        dev,
    )?;

    let down_from_packed = grouped_moe_mmq_from_glu_packed(
        &down,
        &packed,
        &sorted_token_ids,
        &sorted_token_ids,
        &expert_bounds,
        TOTAL_ASSIGNMENTS,
        NUM_TOKENS,
        NUM_EXPERTS,
        GluActivationType::Silu as i32,
        dev,
    )?;
    let down_from_sorted_pair = grouped_moe_mmq_from_glu_sorted_pair(
        &down,
        &gate_sorted,
        &up_sorted,
        &sorted_token_ids,
        &expert_bounds,
        TOTAL_ASSIGNMENTS,
        NUM_TOKENS,
        NUM_EXPERTS,
        GluActivationType::Silu as i32,
        dev,
    )?;
    assert_close(&down_from_sorted_pair, &down_from_packed)
}
