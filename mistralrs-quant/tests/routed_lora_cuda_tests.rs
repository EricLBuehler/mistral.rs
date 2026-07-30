#![cfg(feature = "cuda")]

use std::sync::Arc;

use candle_core::{
    cuda::cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr},
    CudaDevice, DType, Device, Result, Tensor, D,
};
use half::{bf16, f16};
use mistralrs_quant::{
    add_expert_delta_reference, launch_routed_lora_grouped, with_lora_execution, LoraExecution,
    LoraExecutionArena, LoraExpertDelta, LoraExpertExecution, LoraExpertInputMode,
    LoraExpertProjection, LoraExpertProjectionNames, LoraExpertProjectionWeights,
    LoraExpertSiteHandle, LoraExpertSiteSpec, LoraExpertWeights, LoraGateUpOrder,
    LoraLayerRegistry, LoraSiteKey, RoutedLoraAdapterWeight, RoutedLoraCudaMetadata,
    RoutedLoraCudaWeightTable, RoutedLoraGroupedLaunch, RoutedLoraInputMode,
    RoutedLoraMetadataLayout, RoutedLoraProjectionLayout, Shard, ROUTED_LORA_WMMA_RANK_CAP,
};

const NUM_EXPERTS: usize = 4;
const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 12;
const RANK: usize = 4;
const ARENA_GROUPED_RANK: usize = ROUTED_LORA_WMMA_RANK_CAP + 1;
const TOP_K: usize = 2;

fn values(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = (index.wrapping_mul(37) + salt.wrapping_mul(17)) % 101;
            (value as f32 / 50.0 - 1.0) * scale
        })
        .collect()
}

trait RawElement: DeviceRepr + Copy + Clone + 'static {
    const DTYPE: DType;

    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl RawElement for f16 {
    const DTYPE: DType = DType::F16;

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl RawElement for bf16 {
    const DTYPE: DType = DType::BF16;

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

struct RawProjection<T> {
    a: Vec<T>,
    b: Vec<T>,
    input_features: usize,
    output_features: usize,
    rank: usize,
    rank_stride: usize,
    scale: f32,
}

fn raw_projection<T: RawElement>(
    num_experts: usize,
    input_features: usize,
    output_features: usize,
    rank: usize,
    rank_stride: usize,
    salt: usize,
) -> RawProjection<T> {
    RawProjection {
        a: values(num_experts * rank_stride * input_features, salt, 0.035)
            .into_iter()
            .map(T::from_f32)
            .collect(),
        b: values(num_experts * output_features * rank_stride, salt + 1, 0.03)
            .into_iter()
            .map(T::from_f32)
            .collect(),
        input_features,
        output_features,
        rank,
        rank_stride,
        scale: 0.65,
    }
}

fn upload<T: DeviceRepr + Clone + 'static>(
    device: &CudaDevice,
    values: &[T],
) -> Result<CudaSlice<T>> {
    let mut slice = unsafe { device.alloc::<T>(values.len())? };
    device.memcpy_htod(values, &mut slice)?;
    Ok(slice)
}

fn device_address<T: DeviceRepr>(device: &CudaDevice, slice: &CudaSlice<T>) -> u64 {
    let stream = device.cuda_stream();
    let (pointer, guard) = slice.device_ptr(&stream);
    drop(guard);
    pointer
}

struct RawReferenceRoute<'a, T> {
    output: &'a mut [T],
    output_row: usize,
    output_row_stride: usize,
    output_offset: usize,
    input: &'a [T],
    input_row: usize,
    expert: usize,
    projection: &'a RawProjection<T>,
    route_scale: f32,
}

fn add_raw_reference<T: RawElement>(route: RawReferenceRoute<'_, T>) {
    let RawReferenceRoute {
        output,
        output_row,
        output_row_stride,
        output_offset,
        input,
        input_row,
        expert,
        projection,
        route_scale,
    } = route;
    let mut hidden = vec![T::from_f32(0.0); projection.rank];
    for (rank, hidden_value) in hidden.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for column in 0..projection.input_features {
            let input = input[input_row * projection.input_features + column].to_f32();
            let weight = projection.a
                [(expert * projection.rank_stride + rank) * projection.input_features + column]
                .to_f32();
            sum += input * weight;
        }
        *hidden_value = T::from_f32(sum);
    }
    for column in 0..projection.output_features {
        let mut sum = 0.0f32;
        for (rank, hidden) in hidden.iter().enumerate() {
            let weight = projection.b
                [(expert * projection.output_features + column) * projection.rank_stride + rank]
                .to_f32();
            sum += hidden.to_f32() * weight;
        }
        let index = output_row * output_row_stride + output_offset + column;
        output[index] = T::from_f32(output[index].to_f32() + projection.scale * route_scale * sum);
    }
}

fn assert_raw_close<T: RawElement>(actual: &[T], expected: &[T], tolerance: f32) {
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.to_f32() - expected.to_f32()).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= tolerance,
        "grouped WMMA max error {max_error} exceeds {tolerance}"
    );
}

fn check_grouped_wmma_padded<T: RawElement>(rank: usize, rank_stride: usize) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?;
    let num_experts = 128;
    let num_tokens = 1;
    let top_k = 8;
    let num_routes = num_tokens * top_k;
    let hidden = 65;
    let intermediate = 97;
    let expert_ids = (0..num_routes)
        .map(|route| ((route * 17 + 3) % num_experts) as u32)
        .collect::<Vec<_>>();
    let expert_ids_device = upload(cuda, &expert_ids)?;
    let token_slots_device = upload(cuda, &vec![0u32; num_tokens])?;

    let gate = raw_projection::<T>(num_experts, hidden, intermediate, rank, rank_stride, 301);
    let up = raw_projection::<T>(num_experts, hidden, intermediate, rank, rank_stride, 307);
    let down = raw_projection::<T>(num_experts, intermediate, hidden, rank, rank_stride, 313);
    let gate_a = upload(cuda, &gate.a)?;
    let gate_b = upload(cuda, &gate.b)?;
    let up_a = upload(cuda, &up.a)?;
    let up_b = upload(cuda, &up.b)?;
    let down_a = upload(cuda, &down.a)?;
    let down_b = upload(cuda, &down.b)?;
    let descriptors = [
        RoutedLoraAdapterWeight {
            a: device_address(cuda, &gate_a),
            b: device_address(cuda, &gate_b),
            scales: 0,
            rank: rank as u32,
            rank_stride: rank_stride as u32,
            scale: gate.scale,
            flags: 0,
        },
        RoutedLoraAdapterWeight {
            a: device_address(cuda, &up_a),
            b: device_address(cuda, &up_b),
            scales: 0,
            rank: rank as u32,
            rank_stride: rank_stride as u32,
            scale: up.scale,
            flags: 0,
        },
        RoutedLoraAdapterWeight {
            a: device_address(cuda, &down_a),
            b: device_address(cuda, &down_b),
            scales: 0,
            rank: rank as u32,
            rank_stride: rank_stride as u32,
            scale: down.scale,
            flags: 0,
        },
    ];
    let table = RoutedLoraCudaWeightTable::new(cuda, &descriptors, 3, 1)?;
    let metadata = RoutedLoraMetadataLayout::new(num_tokens, top_k, num_experts, 1)?;
    let mut grouped_metadata = RoutedLoraCudaMetadata::new(cuda, metadata)?;
    unsafe {
        grouped_metadata.build(
            device_address(cuda, &token_slots_device),
            device_address(cuda, &expert_ids_device),
        )?;
    }
    let scratch = unsafe { cuda.alloc::<f32>(metadata.hidden_elements(2, rank)?)? };

    let gate_input = values(num_tokens * hidden, 317, 0.2)
        .into_iter()
        .map(T::from_f32)
        .collect::<Vec<_>>();
    let gate_input_device = upload(cuda, &gate_input)?;
    let gate_row_stride = intermediate * 2;
    let gate_initial = values(num_routes * gate_row_stride, 331, 0.02)
        .into_iter()
        .map(T::from_f32)
        .collect::<Vec<_>>();
    let gate_output_device = upload(cuda, &gate_initial)?;
    let gate_layout = RoutedLoraProjectionLayout::new(
        hidden,
        intermediate,
        gate_row_stride,
        intermediate,
        2,
        rank,
        RoutedLoraInputMode::TokenRows,
    )?;
    unsafe {
        launch_routed_lora_grouped(
            &grouped_metadata,
            &table,
            RoutedLoraGroupedLaunch {
                input: device_address(cuda, &gate_input_device),
                hidden: device_address(cuda, &scratch),
                output: device_address(cuda, &gate_output_device),
                route_input_rows: 0,
                route_output_rows: 0,
                route_output_scales: 0,
                dtype: T::DTYPE,
                projection: gate_layout,
                weight_slice_offset: 0,
            },
        )?;
    }
    device.synchronize()?;
    let gate_actual = cuda.clone_dtoh(&gate_output_device)?;
    let mut gate_expected = gate_initial;
    for (route, expert) in expert_ids.iter().copied().enumerate() {
        add_raw_reference(RawReferenceRoute {
            output: &mut gate_expected,
            output_row: route,
            output_row_stride: gate_row_stride,
            output_offset: 0,
            input: &gate_input,
            input_row: 0,
            expert: expert as usize,
            projection: &gate,
            route_scale: 1.0,
        });
        add_raw_reference(RawReferenceRoute {
            output: &mut gate_expected,
            output_row: route,
            output_row_stride: gate_row_stride,
            output_offset: intermediate,
            input: &gate_input,
            input_row: 0,
            expert: expert as usize,
            projection: &up,
            route_scale: 1.0,
        });
    }
    let tolerance = if T::DTYPE == DType::BF16 { 0.5 } else { 0.08 };
    assert_raw_close(&gate_actual, &gate_expected, tolerance);

    let down_input = values(num_routes * intermediate, 337, 0.18)
        .into_iter()
        .map(T::from_f32)
        .collect::<Vec<_>>();
    let down_input_device = upload(cuda, &down_input)?;
    let route_scales = (0..num_routes)
        .map(|route| 0.2 + route as f32 * 0.07)
        .collect::<Vec<_>>();
    let route_scales_device = upload(cuda, &route_scales)?;
    let down_initial = values(num_routes * hidden, 347, 0.02)
        .into_iter()
        .map(T::from_f32)
        .collect::<Vec<_>>();
    let down_output_device = upload(cuda, &down_initial)?;
    let down_layout = RoutedLoraProjectionLayout::new(
        intermediate,
        hidden,
        hidden,
        0,
        1,
        rank,
        RoutedLoraInputMode::RoutedRows,
    )?;
    unsafe {
        launch_routed_lora_grouped(
            &grouped_metadata,
            &table,
            RoutedLoraGroupedLaunch {
                input: device_address(cuda, &down_input_device),
                hidden: device_address(cuda, &scratch),
                output: device_address(cuda, &down_output_device),
                route_input_rows: 0,
                route_output_rows: 0,
                route_output_scales: device_address(cuda, &route_scales_device),
                dtype: T::DTYPE,
                projection: down_layout,
                weight_slice_offset: 2,
            },
        )?;
    }
    device.synchronize()?;
    let down_actual = cuda.clone_dtoh(&down_output_device)?;
    let mut down_expected = down_initial;
    for (route, expert) in expert_ids.iter().copied().enumerate() {
        add_raw_reference(RawReferenceRoute {
            output: &mut down_expected,
            output_row: route,
            output_row_stride: hidden,
            output_offset: 0,
            input: &down_input,
            input_row: route,
            expert: expert as usize,
            projection: &down,
            route_scale: route_scales[route],
        });
    }
    assert_raw_close(&down_actual, &down_expected, tolerance);
    Ok(())
}

fn projection(
    input: usize,
    output: usize,
    salt: usize,
    scales: &[f32],
    dtype: DType,
    device: &Device,
) -> Result<LoraExpertProjectionWeights> {
    let a = Tensor::from_vec(
        values(NUM_EXPERTS * RANK * input, salt, 0.2),
        (NUM_EXPERTS, RANK, input),
        device,
    )?
    .to_dtype(dtype)?;
    let b = Tensor::from_vec(
        values(NUM_EXPERTS * output * RANK, salt + 1, 0.15),
        (NUM_EXPERTS, output, RANK),
        device,
    )?
    .to_dtype(dtype)?;
    let scales = Tensor::from_vec(scales.to_vec(), NUM_EXPERTS, device)?;
    LoraExpertProjectionWeights::new(a, b, scales)
}

fn adapter(
    site: &LoraExpertSiteHandle,
    salt: usize,
    missing_gate: bool,
    dtype: DType,
    device: &Device,
) -> Result<LoraExpertWeights> {
    let sparse_scales = [0.7, 0.0, -0.3, 0.5];
    let gate = if missing_gate {
        None
    } else {
        Some(projection(
            HIDDEN,
            INTERMEDIATE,
            salt,
            &sparse_scales,
            dtype,
            device,
        )?)
    };
    LoraExpertWeights::new(
        site,
        gate,
        Some(projection(
            HIDDEN,
            INTERMEDIATE,
            salt + 2,
            &sparse_scales,
            dtype,
            device,
        )?),
        Some(projection(
            INTERMEDIATE,
            HIDDEN,
            salt + 4,
            &sparse_scales,
            dtype,
            device,
        )?),
    )
}

fn setup_with_order(
    num_tokens: usize,
    dtype: DType,
    device: &Device,
    gate_up_order: LoraGateUpOrder,
) -> Result<(Arc<LoraExecution>, Arc<LoraExpertSiteHandle>, Tensor)> {
    let registry = LoraLayerRegistry::new();
    let site = registry.register_expert(
        LoraSiteKey::new("model.layers.0.mlp.experts"),
        LoraExpertSiteSpec::new(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
            Shard::default(),
            Shard::default(),
        )?
        .with_gate_up_order(gate_up_order),
        dtype,
        device.clone(),
    )?;
    registry.finalize()?;
    let row_slots = (0..num_tokens)
        .map(|token| match token % 3 {
            0 => Some(3),
            1 => None,
            _ => Some(7),
        })
        .collect();
    let mut execution = LoraExecution::new(registry.runtime_id(), row_slots);
    execution.insert_expert(&site, 3, adapter(&site, 11, false, dtype, device)?)?;
    execution.insert_expert(&site, 7, adapter(&site, 29, true, dtype, device)?)?;
    let ids = (0..num_tokens * TOP_K)
        .map(|route| ((route * 3 + route / TOP_K) % NUM_EXPERTS) as u32)
        .collect::<Vec<_>>();
    let topk_ids = Tensor::from_vec(ids, (num_tokens, TOP_K), device)?;
    Ok((Arc::new(execution), site, topk_ids))
}

fn assert_close(lhs: &Tensor, rhs: &Tensor, tolerance: f32) -> Result<()> {
    let lhs = lhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let max_error = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= tolerance,
        "routed LoRA max error {max_error} exceeds {tolerance}"
    );
    Ok(())
}

fn check_gate_up_execution(
    execution: Arc<LoraExecution>,
    site: &Arc<LoraExpertSiteHandle>,
    input: &Tensor,
    base_gate_up: &Tensor,
    topk_ids: &Tensor,
    tolerance: f32,
) -> Result<()> {
    let (num_tokens, top_k) = topk_ids.dims2()?;
    let shaped = base_gate_up.reshape((num_tokens, top_k, INTERMEDIATE * 2))?;
    let expected_gate = add_expert_delta_reference(
        &execution,
        site,
        LoraExpertDelta::new(
            LoraExpertProjection::Gate,
            input,
            shaped.narrow(D::Minus1, 0, INTERMEDIATE)?.contiguous()?,
            topk_ids,
            LoraExpertInputMode::TokenRows,
        ),
    )?;
    let expected_up = add_expert_delta_reference(
        &execution,
        site,
        LoraExpertDelta::new(
            LoraExpertProjection::Up,
            input,
            shaped
                .narrow(D::Minus1, INTERMEDIATE, INTERMEDIATE)?
                .contiguous()?,
            topk_ids,
            LoraExpertInputMode::TokenRows,
        ),
    )?;
    let (gate, up) = with_lora_execution(Some(execution), || -> Result<_> {
        LoraExpertExecution::current(site)?
            .expect("expert LoRA site is active")
            .add_gate_up_delta(input, base_gate_up.clone(), topk_ids)
    })?;
    assert_close(&gate, &expected_gate, tolerance)?;
    assert_close(&up, &expected_up, tolerance)
}

fn check_case(num_tokens: usize, dtype: DType, tolerance: f32) -> Result<()> {
    check_case_with_order(num_tokens, dtype, tolerance, LoraGateUpOrder::Concatenated)
}

fn check_case_with_order(
    num_tokens: usize,
    dtype: DType,
    tolerance: f32,
    gate_up_order: LoraGateUpOrder,
) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (execution, site, topk_ids) = setup_with_order(num_tokens, dtype, &device, gate_up_order)?;
    let input = Tensor::from_vec(
        values(num_tokens * HIDDEN, 41, 0.8),
        (num_tokens, HIDDEN),
        &device,
    )?
    .to_dtype(dtype)?;
    let base_gate_up = Tensor::from_vec(
        values(num_tokens * TOP_K * INTERMEDIATE * 2, 43, 0.3),
        (num_tokens * TOP_K, INTERMEDIATE * 2),
        &device,
    )?
    .to_dtype(dtype)?;
    let shaped_gate_up = base_gate_up.reshape((num_tokens, TOP_K, INTERMEDIATE * 2))?;
    let (base_gate, base_up) = match gate_up_order {
        LoraGateUpOrder::Concatenated => (
            shaped_gate_up
                .narrow(D::Minus1, 0, INTERMEDIATE)?
                .contiguous()?,
            shaped_gate_up
                .narrow(D::Minus1, INTERMEDIATE, INTERMEDIATE)?
                .contiguous()?,
        ),
        LoraGateUpOrder::Interleaved => {
            let gate_up = shaped_gate_up.reshape((num_tokens, TOP_K, INTERMEDIATE, 2))?;
            (
                gate_up
                    .narrow(D::Minus1, 0, 1)?
                    .squeeze(D::Minus1)?
                    .contiguous()?,
                gate_up
                    .narrow(D::Minus1, 1, 1)?
                    .squeeze(D::Minus1)?
                    .contiguous()?,
            )
        }
    };
    let expected_gate = add_expert_delta_reference(
        &execution,
        &site,
        LoraExpertDelta::new(
            LoraExpertProjection::Gate,
            &input,
            base_gate,
            &topk_ids,
            LoraExpertInputMode::TokenRows,
        ),
    )?;
    let expected_up = add_expert_delta_reference(
        &execution,
        &site,
        LoraExpertDelta::new(
            LoraExpertProjection::Up,
            &input,
            base_up,
            &topk_ids,
            LoraExpertInputMode::TokenRows,
        ),
    )?;
    let (gate, up) = with_lora_execution(Some(execution.clone()), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_gate_up_delta(&input, base_gate_up, &topk_ids)
    })?;
    assert_close(&gate, &expected_gate, tolerance)?;
    assert_close(&up, &expected_up, tolerance)?;

    let down_input = Tensor::from_vec(
        values(num_tokens * TOP_K * INTERMEDIATE, 47, 0.5),
        (num_tokens, TOP_K, INTERMEDIATE),
        &device,
    )?
    .to_dtype(dtype)?;
    let base_down = Tensor::from_vec(
        values(num_tokens * TOP_K * HIDDEN, 53, 0.25),
        (num_tokens, TOP_K, HIDDEN),
        &device,
    )?
    .to_dtype(dtype)?;
    let routed_weights = Tensor::from_vec(
        values(num_tokens * TOP_K, 59, 0.4),
        (num_tokens, TOP_K),
        &device,
    )?;
    let expected_down = add_expert_delta_reference(
        &execution,
        &site,
        LoraExpertDelta::new(
            LoraExpertProjection::Down,
            &down_input,
            base_down.clone(),
            &topk_ids,
            LoraExpertInputMode::RoutedRows,
        )
        .with_routed_weights(&routed_weights),
    )?;
    let down = with_lora_execution(Some(execution.clone()), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_delta(
                LoraExpertProjection::Down,
                &down_input,
                base_down,
                &topk_ids,
                Some(&routed_weights),
                LoraExpertInputMode::RoutedRows,
            )
    })?;
    assert_close(&down, &expected_down, tolerance)?;

    let token_down_input = Tensor::from_vec(
        values(num_tokens * INTERMEDIATE, 61, 0.5),
        (num_tokens, INTERMEDIATE),
        &device,
    )?
    .to_dtype(dtype)?;
    let token_base_down = Tensor::from_vec(
        values(num_tokens * TOP_K * HIDDEN, 67, 0.25),
        (num_tokens, TOP_K, HIDDEN),
        &device,
    )?
    .to_dtype(dtype)?;
    let expected_token_down = add_expert_delta_reference(
        &execution,
        &site,
        LoraExpertDelta::new(
            LoraExpertProjection::Down,
            &token_down_input,
            token_base_down.clone(),
            &topk_ids,
            LoraExpertInputMode::TokenRows,
        )
        .with_routed_weights(&routed_weights),
    )?;
    let token_down = with_lora_execution(Some(execution), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_delta(
                LoraExpertProjection::Down,
                &token_down_input,
                token_base_down,
                &topk_ids,
                Some(&routed_weights),
                LoraExpertInputMode::TokenRows,
            )
    })?;
    assert_close(&token_down, &expected_token_down, tolerance)
}

fn check_rank_boundary_case(
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    rank: usize,
    dtype: DType,
) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let hidden = 65;
    let intermediate = 97;
    let registry = LoraLayerRegistry::new();
    let site = registry.register_expert(
        LoraSiteKey::new("model.layers.0.mlp.boundary_experts"),
        LoraExpertSiteSpec::new(
            num_experts,
            hidden,
            intermediate,
            LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
            Shard::default(),
            Shard::default(),
        )?,
        dtype,
        device.clone(),
    )?;
    registry.finalize()?;
    let make_projection = |input: usize, output: usize, salt: usize| -> Result<_> {
        LoraExpertProjectionWeights::new(
            Tensor::from_vec(
                values(num_experts * rank * input, salt, 0.04),
                (num_experts, rank, input),
                &device,
            )?
            .to_dtype(dtype)?,
            Tensor::from_vec(
                values(num_experts * output * rank, salt + 1, 0.04),
                (num_experts, output, rank),
                &device,
            )?
            .to_dtype(dtype)?,
            Tensor::from_vec(
                (0..num_experts)
                    .map(|expert| [0.7, -0.25, 0.0][expert % 3])
                    .collect::<Vec<f32>>(),
                num_experts,
                &device,
            )?,
        )
    };
    let weights = LoraExpertWeights::new(
        &site,
        Some(make_projection(hidden, intermediate, 151)?),
        Some(make_projection(hidden, intermediate, 157)?),
        Some(make_projection(intermediate, hidden, 163)?),
    )?;
    let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3); num_tokens]);
    execution.insert_expert(&site, 3, weights)?;
    let execution = Arc::new(execution);
    let topk_ids = Tensor::from_vec(
        (0..num_tokens * top_k)
            .map(|route| ((route * 2 + route / top_k) % num_experts) as u32)
            .collect::<Vec<_>>(),
        (num_tokens, top_k),
        &device,
    )?;
    let input = Tensor::from_vec(
        values(num_tokens * hidden, 167, 0.2),
        (num_tokens, hidden),
        &device,
    )?
    .to_dtype(dtype)?;
    let base_gate_up = Tensor::from_vec(
        values(num_tokens * top_k * intermediate * 2, 173, 0.1),
        (num_tokens, top_k, intermediate * 2),
        &device,
    )?
    .to_dtype(dtype)?;
    let expected_gate = add_expert_delta_reference(
        &execution,
        &site,
        LoraExpertDelta::new(
            LoraExpertProjection::Gate,
            &input,
            base_gate_up
                .narrow(D::Minus1, 0, intermediate)?
                .contiguous()?,
            &topk_ids,
            LoraExpertInputMode::TokenRows,
        ),
    )?;
    let expected_up = add_expert_delta_reference(
        &execution,
        &site,
        LoraExpertDelta::new(
            LoraExpertProjection::Up,
            &input,
            base_gate_up
                .narrow(D::Minus1, intermediate, intermediate)?
                .contiguous()?,
            &topk_ids,
            LoraExpertInputMode::TokenRows,
        ),
    )?;
    let (gate, up) = with_lora_execution(Some(execution.clone()), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_gate_up_delta(&input, base_gate_up, &topk_ids)
    })?;
    assert_close(&gate, &expected_gate, 0.3)?;
    assert_close(&up, &expected_up, 0.3)?;

    let down_input = Tensor::from_vec(
        values(num_tokens * top_k * intermediate, 179, 0.15),
        (num_tokens, top_k, intermediate),
        &device,
    )?
    .to_dtype(dtype)?;
    let base_down = Tensor::from_vec(
        values(num_tokens * top_k * hidden, 181, 0.1),
        (num_tokens, top_k, hidden),
        &device,
    )?
    .to_dtype(dtype)?;
    let route_weights = Tensor::from_vec(
        values(num_tokens * top_k, 191, 0.5),
        (num_tokens, top_k),
        &device,
    )?;
    let expected_down = add_expert_delta_reference(
        &execution,
        &site,
        LoraExpertDelta::new(
            LoraExpertProjection::Down,
            &down_input,
            base_down.clone(),
            &topk_ids,
            LoraExpertInputMode::RoutedRows,
        )
        .with_routed_weights(&route_weights),
    )?;
    let down = with_lora_execution(Some(execution), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_delta(
                LoraExpertProjection::Down,
                &down_input,
                base_down,
                &topk_ids,
                Some(&route_weights),
                LoraExpertInputMode::RoutedRows,
            )
    })?;
    assert_close(&down, &expected_down, 0.3)
}

#[test]
fn direct_f16_matches_reference_with_mixed_adapters() -> Result<()> {
    check_case(4, DType::F16, 0.04)
}

#[test]
fn direct_bf16_matches_reference_with_mixed_adapters() -> Result<()> {
    check_case(4, DType::BF16, 0.12)
}

#[test]
fn grouped_f16_matches_reference_with_mixed_adapters() -> Result<()> {
    check_case(24, DType::F16, 0.08)
}

#[test]
fn grouped_bf16_matches_reference_with_mixed_adapters() -> Result<()> {
    check_case(24, DType::BF16, 0.3)
}

#[test]
fn grouped_f32_matches_reference_with_mixed_adapters() -> Result<()> {
    check_case(24, DType::F32, 2e-4)
}

#[test]
fn direct_bf16_interleaved_gate_up_matches_reference() -> Result<()> {
    check_case_with_order(4, DType::BF16, 0.12, LoraGateUpOrder::Interleaved)
}

#[test]
fn grouped_bf16_interleaved_gate_up_matches_reference() -> Result<()> {
    check_case_with_order(24, DType::BF16, 0.3, LoraGateUpOrder::Interleaved)
}

#[test]
fn persistent_arena_reuses_allocations_and_reconfigures_mapping() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let registry = LoraLayerRegistry::new();
    let site = registry.register_expert(
        LoraSiteKey::new("model.layers.0.mlp.experts"),
        LoraExpertSiteSpec::new(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
            Shard::default(),
            Shard::default(),
        )?,
        DType::F16,
        device.clone(),
    )?;
    registry.finalize()?;
    let arena = Arc::new(LoraExecutionArena::new());
    let make_projection = |input: usize, output: usize, salt: usize| -> Result<_> {
        LoraExpertProjectionWeights::new(
            Tensor::from_vec(
                values(NUM_EXPERTS * ARENA_GROUPED_RANK * input, salt, 0.2),
                (NUM_EXPERTS, ARENA_GROUPED_RANK, input),
                &device,
            )?
            .to_dtype(DType::F16)?,
            Tensor::from_vec(
                values(NUM_EXPERTS * output * ARENA_GROUPED_RANK, salt + 1, 0.15),
                (NUM_EXPERTS, output, ARENA_GROUPED_RANK),
                &device,
            )?
            .to_dtype(DType::F16)?,
            Tensor::from_vec(vec![0.7f32, 0.0, -0.3, 0.5], NUM_EXPERTS, &device)?,
        )
    };
    let make_adapter = |salt: usize, missing_gate: bool| -> Result<_> {
        LoraExpertWeights::new(
            &site,
            if missing_gate {
                None
            } else {
                Some(make_projection(HIDDEN, INTERMEDIATE, salt)?)
            },
            Some(make_projection(HIDDEN, INTERMEDIATE, salt + 2)?),
            Some(make_projection(INTERMEDIATE, HIDDEN, salt + 4)?),
        )
    };
    let first_three = Arc::new(make_adapter(11, false)?);
    let first_seven = Arc::new(make_adapter(29, true)?);
    let num_tokens = 24;
    let topk_ids = Tensor::from_vec(
        (0..num_tokens * TOP_K)
            .map(|route| ((route * 3 + route / TOP_K) % NUM_EXPERTS) as u32)
            .collect::<Vec<_>>(),
        (num_tokens, TOP_K),
        &device,
    )?;
    let input = Tensor::from_vec(
        values(num_tokens * HIDDEN, 73, 0.8),
        (num_tokens, HIDDEN),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let base = Tensor::from_vec(
        values(num_tokens * TOP_K * INTERMEDIATE * 2, 79, 0.3),
        (num_tokens * TOP_K, INTERMEDIATE * 2),
        &device,
    )?
    .to_dtype(DType::F16)?;

    let make_execution = |rows: Vec<Option<u32>>,
                          three: Arc<LoraExpertWeights>,
                          seven: Arc<LoraExpertWeights>|
     -> Result<Arc<LoraExecution>> {
        let mut execution =
            LoraExecution::new_with_arena(registry.runtime_id(), rows, arena.clone());
        execution.insert_expert_shared(&site, 3, three)?;
        execution.insert_expert_shared(&site, 7, seven)?;
        Ok(Arc::new(execution))
    };
    let first = make_execution(
        (0..num_tokens)
            .map(|token| [Some(3), None, Some(7)][token % 3])
            .collect(),
        first_three.clone(),
        first_seven.clone(),
    )?;
    check_gate_up_execution(first, &site, &input, &base, &topk_ids, 0.08)?;
    let first_stats = arena.cuda_stats();
    assert_eq!(first_stats.cached_routing_resources, 1);
    assert_eq!(first_stats.weight_table_uploads, 1);
    assert_eq!(first_stats.token_slot_uploads, 1);
    assert_eq!(first_stats.metadata_builds, 1);

    let second_rows = (0..num_tokens)
        .map(|token| [Some(7), Some(3), None][token % 3])
        .collect::<Vec<_>>();
    let second = make_execution(
        second_rows.clone(),
        first_three.clone(),
        first_seven.clone(),
    )?;
    check_gate_up_execution(second, &site, &input, &base, &topk_ids, 0.08)?;
    let second_stats = arena.cuda_stats();
    assert_eq!(second_stats.weight_table_uploads, 1);
    assert_eq!(second_stats.token_slot_uploads, 2);
    assert_eq!(second_stats.metadata_builds, 2);

    let replacement_three = Arc::new(make_adapter(101, false)?);
    let replacement_seven = Arc::new(make_adapter(131, false)?);
    let third = make_execution(second_rows, replacement_three, replacement_seven)?;
    check_gate_up_execution(third, &site, &input, &base, &topk_ids, 0.08)?;
    let third_stats = arena.cuda_stats();
    assert_eq!(third_stats.cached_routing_resources, 1);
    assert_eq!(third_stats.cached_weight_tables, 2);
    assert_eq!(third_stats.weight_table_uploads, 2);
    assert_eq!(third_stats.token_slot_uploads, 2);
    assert_eq!(third_stats.metadata_builds, 3);
    Ok(())
}

#[test]
fn rank_padding_tails_and_dispatch_boundaries_match_reference() -> Result<()> {
    for (num_tokens, rank) in [(8, 64), (9, 64), (24, 17), (24, 128), (24, 129)] {
        check_rank_boundary_case(num_tokens, TOP_K, 3, rank, DType::F16)?;
    }
    for rank in [96, 128] {
        for dtype in [DType::F16, DType::BF16] {
            check_rank_boundary_case(1, 8, 128, rank, dtype)?;
        }
    }
    Ok(())
}

#[test]
fn grouped_wmma_padded_rank_stride_matches_reference() -> Result<()> {
    check_grouped_wmma_padded::<f16>(96, 112)?;
    check_grouped_wmma_padded::<bf16>(96, 112)?;
    check_grouped_wmma_padded::<f16>(128, 144)?;
    check_grouped_wmma_padded::<bf16>(128, 144)
}

#[test]
fn grouped_owned_api_accumulates_in_place_while_safe_api_preserves_aliases() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let num_tokens = 24;
    let (execution, site, topk_ids) = setup_with_order(
        num_tokens,
        DType::F16,
        &device,
        LoraGateUpOrder::Concatenated,
    )?;
    let input = Tensor::from_vec(
        values(num_tokens * HIDDEN, 211, 0.4),
        (num_tokens, HIDDEN),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let base = Tensor::from_vec(
        values(num_tokens * TOP_K * INTERMEDIATE * 2, 223, 0.2),
        (num_tokens * TOP_K, INTERMEDIATE * 2),
        &device,
    )?
    .to_dtype(DType::F16)?;

    let safe_base = base.copy()?;
    let safe_alias = safe_base.clone();
    let _ = with_lora_execution(Some(execution.clone()), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_gate_up_delta(&input, safe_base, &topk_ids)
    })?;
    assert_close(&safe_alias, &base, 0.0)?;

    let owned_base = base.copy()?;
    let owned_alias = owned_base.clone();
    let (gate, up) = with_lora_execution(Some(execution.clone()), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_gate_up_delta_owned(&input, owned_base, &topk_ids)
    })?;
    let packed =
        Tensor::cat(&[&gate, &up], D::Minus1)?.reshape((num_tokens * TOP_K, INTERMEDIATE * 2))?;
    assert_close(&owned_alias, &packed, 0.0)?;

    let combined_base = base.copy()?;
    let combined_alias = combined_base.clone();
    let combined = with_lora_execution(Some(execution.clone()), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_gate_up_delta_combined_owned(&input, combined_base, &topk_ids)
    })?;
    assert_close(
        &combined_alias,
        &combined.reshape((num_tokens * TOP_K, INTERMEDIATE * 2))?,
        0.0,
    )?;
    assert_close(&combined_alias, &packed, 0.0)?;

    let down_input = Tensor::from_vec(
        values(num_tokens * TOP_K * INTERMEDIATE, 227, 0.3),
        (num_tokens, TOP_K, INTERMEDIATE),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let down_base = Tensor::from_vec(
        values(num_tokens * TOP_K * HIDDEN, 229, 0.2),
        (num_tokens, TOP_K, HIDDEN),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let route_weights = Tensor::from_vec(
        values(num_tokens * TOP_K, 233, 0.5),
        (num_tokens, TOP_K),
        &device,
    )?;
    let safe_down = down_base.copy()?;
    let safe_down_alias = safe_down.clone();
    let _ = with_lora_execution(Some(execution.clone()), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_delta(
                LoraExpertProjection::Down,
                &down_input,
                safe_down,
                &topk_ids,
                Some(&route_weights),
                LoraExpertInputMode::RoutedRows,
            )
    })?;
    assert_close(&safe_down_alias, &down_base, 0.0)?;

    let owned_down = down_base.copy()?;
    let owned_down_alias = owned_down.clone();
    let down = with_lora_execution(Some(execution), || -> Result<_> {
        LoraExpertExecution::current(&site)?
            .expect("expert LoRA site is active")
            .add_delta_owned(
                LoraExpertProjection::Down,
                &down_input,
                owned_down,
                &topk_ids,
                Some(&route_weights),
                LoraExpertInputMode::RoutedRows,
            )
    })?;
    assert_close(&owned_down_alias, &down, 0.0)
}
