#![cfg(feature = "cuda")]

use candle_core::{
    quantized::GgmlDType, quantized::QTensor, DType, Device, Result, Storage, Tensor,
};
use half::f16;
use mistralrs_quant::{
    moe_weighted_reduce_flat_same_dtype, IndexedMoeLoraDecode, IndexedMoeLoraWeights,
    IndexedMoeRouting,
};

fn values(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| ((index as f32 * 0.013 + phase).sin() * 0.2).clamp(-0.2, 0.2))
        .collect()
}

fn tensor(shape: impl Into<candle_core::Shape>, phase: f32) -> Result<Tensor> {
    let shape = shape.into();
    Tensor::from_vec(values(shape.elem_count(), phase), shape, &Device::Cpu)
}

fn routed_reference(weight: &QTensor, input: &Tensor, ids: &[u32]) -> Result<Tensor> {
    let weight = weight.dequantize(&Device::Cpu)?;
    let input = input.to_dtype(DType::F32)?;
    let rows = ids
        .iter()
        .enumerate()
        .map(|(row, expert)| {
            input
                .get(row)?
                .unsqueeze(0)?
                .matmul(&weight.get(*expert as usize)?.t()?)?
                .squeeze(0)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::stack(&rows, 0)
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    let actual = actual.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let expected = expected.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let error = (&actual - &expected)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let scale = expected.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        error <= tolerance * (1.0 + scale),
        "max error {error} exceeds tolerance at reference scale {scale}"
    );
    Ok(())
}

fn check_route_outputs(
    activation_dtype: DType,
    quant_dtype: GgmlDType,
    hidden: usize,
    intermediate: usize,
    tolerance: f32,
) -> Result<()> {
    const NUM_EXPERTS: usize = 3;
    const BATCH: usize = 2;
    const TOPK: usize = 2;

    let cuda = Device::new_cuda(0)?;
    let dev = cuda.as_cuda_device()?;
    let gate_cpu = tensor((NUM_EXPERTS, intermediate, hidden), 0.1)?;
    let up_cpu = tensor((NUM_EXPERTS, intermediate, hidden), 0.7)?;
    let down_cpu = tensor((NUM_EXPERTS, hidden, intermediate), 1.3)?;
    let gate = QTensor::quantize_onto(&gate_cpu, quant_dtype, &cuda)?;
    let up = QTensor::quantize_onto(&up_cpu, quant_dtype, &cuda)?;
    let down = QTensor::quantize_onto(&down_cpu, quant_dtype, &cuda)?;

    let ids = [0u32, 2, 1, 0];
    let ids_tensor = Tensor::from_slice(&ids, (BATCH, TOPK), &cuda)?;
    let (ids_storage, ids_layout) = ids_tensor.storage_and_layout();
    assert_eq!(ids_layout.start_offset(), 0);
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        unreachable!()
    };
    let ids_slice = ids_cuda.as_cuda_slice::<u32>()?;
    let weights = IndexedMoeLoraWeights::new(&gate, &up, &down);
    let routing = IndexedMoeRouting::new(ids_slice, BATCH, TOPK, NUM_EXPERTS, dev);
    let decode = IndexedMoeLoraDecode::new(weights, routing)?.expect("supported quant dtype");

    let input_cpu = tensor((BATCH, hidden), 1.9)?.to_dtype(activation_dtype)?;
    let input = input_cpu.to_device(&cuda)?;
    let gate_up = decode.gate_up(&input)?.expect("supported activation dtype");
    assert_eq!(gate_up.dtype(), activation_dtype);
    let routed_input = input_cpu
        .unsqueeze(1)?
        .broadcast_as((BATCH, TOPK, hidden))?
        .reshape((BATCH * TOPK, hidden))?;
    let gate_ref = routed_reference(&gate, &routed_input, &ids)?;
    let up_ref = routed_reference(&up, &routed_input, &ids)?;
    let gate_up_ref =
        Tensor::cat(&[&gate_ref, &up_ref], 1)?.reshape((BATCH, TOPK, 2 * intermediate))?;
    assert_close(&gate_up, &gate_up_ref, tolerance)?;

    let down_input_cpu = tensor((BATCH * TOPK, intermediate), 2.5)?.to_dtype(activation_dtype)?;
    let down_input = down_input_cpu.to_device(&cuda)?;
    let down_output = decode
        .down(&down_input)?
        .expect("supported activation dtype");
    assert_eq!(down_output.dtype(), activation_dtype);
    let down_ref =
        routed_reference(&down, &down_input_cpu, &ids)?.reshape((BATCH, TOPK, hidden))?;
    assert_close(&down_output, &down_ref, tolerance)?;

    let route_weights_cpu =
        Tensor::from_slice(&[0.7f32, 0.3, 0.4, 0.6], (BATCH, TOPK), &Device::Cpu)?;
    let route_weights = route_weights_cpu.to_device(&cuda)?;
    let reduced = moe_weighted_reduce_flat_same_dtype(
        &down_output.reshape((BATCH * TOPK, hidden))?,
        &route_weights,
        BATCH,
        TOPK,
        dev,
    )?;
    assert_eq!(reduced.dtype(), activation_dtype);
    let reduced_ref = down_output
        .to_dtype(DType::F32)?
        .broadcast_mul(&route_weights.unsqueeze(2)?)?
        .sum(1)?;
    assert_close(&reduced, &reduced_ref, 0.01)
}

#[test]
fn f16_q4_0_route_outputs_with_odd_block_counts() -> Result<()> {
    check_route_outputs(DType::F16, GgmlDType::Q4_0, 288, 544, 0.08)
}

#[test]
fn bf16_q4k_route_outputs() -> Result<()> {
    check_route_outputs(DType::BF16, GgmlDType::Q4K, 256, 512, 0.12)
}

#[test]
fn invalid_expert_routes_are_zeroed() -> Result<()> {
    const NUM_EXPERTS: usize = 2;
    const HIDDEN: usize = 64;
    const INTERMEDIATE: usize = 96;
    const TOPK: usize = 2;

    let cuda = Device::new_cuda(0)?;
    let dev = cuda.as_cuda_device()?;
    let gate = QTensor::quantize_onto(
        &tensor((NUM_EXPERTS, INTERMEDIATE, HIDDEN), 0.1)?,
        GgmlDType::Q4_0,
        &cuda,
    )?;
    let up = QTensor::quantize_onto(
        &tensor((NUM_EXPERTS, INTERMEDIATE, HIDDEN), 0.7)?,
        GgmlDType::Q4_0,
        &cuda,
    )?;
    let down = QTensor::quantize_onto(
        &tensor((NUM_EXPERTS, HIDDEN, INTERMEDIATE), 1.3)?,
        GgmlDType::Q4_0,
        &cuda,
    )?;
    let ids = Tensor::from_slice(&[0u32, u32::MAX], (1, TOPK), &cuda)?;
    let (ids_storage, _) = ids.storage_and_layout();
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        unreachable!()
    };
    let routing =
        IndexedMoeRouting::new(ids_cuda.as_cuda_slice::<u32>()?, 1, TOPK, NUM_EXPERTS, dev);
    let decode = IndexedMoeLoraDecode::new(IndexedMoeLoraWeights::new(&gate, &up, &down), routing)?
        .expect("Q4_0 uses the indexed decode path");

    let gate_up = decode.gate_up(&Tensor::ones((1, HIDDEN), DType::F16, &cuda)?)?;
    let gate_up = gate_up.expect("F16 output is supported");
    let invalid_gate_up = gate_up.get(0)?.get(1)?.abs()?.max_all()?;
    assert_eq!(invalid_gate_up.to_scalar::<f16>()?.to_bits(), 0);

    let down = decode.down(&Tensor::ones((TOPK, INTERMEDIATE), DType::F16, &cuda)?)?;
    let down = down.expect("F16 output is supported");
    let invalid_down = down.get(0)?.get(1)?.abs()?.max_all()?;
    assert_eq!(invalid_down.to_scalar::<f16>()?.to_bits(), 0);
    Ok(())
}
