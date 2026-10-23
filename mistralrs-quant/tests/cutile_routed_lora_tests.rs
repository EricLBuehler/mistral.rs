#![cfg(all(feature = "cuda", feature = "cutile"))]

use candle_core::cuda::cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr};
use candle_core::{CudaDevice, DType, Device, Result};
use half::bf16;
use mistralrs_quant::cutile::{
    try_cutile_routed_lora, try_cutile_routed_lora_no_sort, CutileRoutedLoraLaunch,
    CutileRoutedLoraStatus, CutileRoutedLoraUnsupported,
};
use mistralrs_quant::{
    RoutedLoraAdapterWeight, RoutedLoraCudaMetadata, RoutedLoraCudaWeightTable,
    RoutedLoraInputMode, RoutedLoraMetadataLayout, RoutedLoraProjectionLayout,
};

struct HostProjection {
    a: Vec<bf16>,
    b: Vec<bf16>,
    expert_scales: Option<Vec<f32>>,
    rank: usize,
    rank_stride: usize,
    scale: f32,
}

struct DeviceProjection {
    _a: CudaSlice<bf16>,
    _b: CudaSlice<bf16>,
    _expert_scales: Option<CudaSlice<f32>>,
}

struct ProjectionSpec {
    num_experts: usize,
    input_features: usize,
    output_features: usize,
    rank: usize,
    rank_stride: usize,
    salt: usize,
    scale: f32,
    expert_scales: Option<Vec<f32>>,
}

struct ReferenceRoute<'a> {
    output_row: usize,
    output_row_stride: usize,
    output_offset: usize,
    input: &'a [bf16],
    input_row: usize,
    input_features: usize,
    output_features: usize,
    expert: usize,
    projection: &'a HostProjection,
    route_scale: f32,
}

fn patterned_bf16(len: usize, salt: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|index| {
            let value = ((index.wrapping_mul(37) + salt.wrapping_mul(19)) % 251) as f32;
            bf16::from_f32((value / 125.0 - 1.0) * scale)
        })
        .collect()
}

fn upload<T: DeviceRepr + Clone + 'static>(dev: &CudaDevice, values: &[T]) -> Result<CudaSlice<T>> {
    let mut slice = unsafe { dev.alloc::<T>(values.len())? };
    dev.memcpy_htod(values, &mut slice)?;
    Ok(slice)
}

fn synchronize(dev: &CudaDevice) -> Result<()> {
    dev.cuda_stream()
        .synchronize()
        .map_err(|error| candle_core::Error::Msg(format!("CUDA synchronize: {error:?}")))
}

fn address<T: DeviceRepr>(dev: &CudaDevice, slice: &CudaSlice<T>) -> u64 {
    let stream = dev.cuda_stream();
    let (address, guard) = slice.device_ptr(&stream);
    drop(guard);
    address
}

fn projection(spec: ProjectionSpec) -> HostProjection {
    HostProjection {
        a: patterned_bf16(
            spec.num_experts * spec.rank_stride * spec.input_features,
            spec.salt,
            0.055,
        ),
        b: patterned_bf16(
            spec.num_experts * spec.output_features * spec.rank_stride,
            spec.salt + 1,
            0.045,
        ),
        expert_scales: spec.expert_scales,
        rank: spec.rank,
        rank_stride: spec.rank_stride,
        scale: spec.scale,
    }
}

fn weight_table(
    dev: &CudaDevice,
    projections: &[Option<HostProjection>],
    num_slices: usize,
    num_adapter_slots: usize,
) -> Result<(RoutedLoraCudaWeightTable, Vec<Option<DeviceProjection>>)> {
    let mut descriptors = Vec::with_capacity(projections.len());
    let mut allocations = Vec::with_capacity(projections.len());
    for projection in projections {
        let Some(projection) = projection else {
            descriptors.push(RoutedLoraAdapterWeight::empty());
            allocations.push(None);
            continue;
        };
        let a = upload(dev, &projection.a)?;
        let b = upload(dev, &projection.b)?;
        let expert_scales = projection
            .expert_scales
            .as_ref()
            .map(|scales| upload(dev, scales))
            .transpose()?;
        descriptors.push(RoutedLoraAdapterWeight {
            a: address(dev, &a),
            b: address(dev, &b),
            scales: expert_scales
                .as_ref()
                .map(|scales| address(dev, scales))
                .unwrap_or(0),
            rank: projection.rank as u32,
            rank_stride: projection.rank_stride as u32,
            scale: projection.scale,
            flags: 0,
        });
        allocations.push(Some(DeviceProjection {
            _a: a,
            _b: b,
            _expert_scales: expert_scales,
        }));
    }
    let table = RoutedLoraCudaWeightTable::new(dev, &descriptors, num_slices, num_adapter_slots)?;
    Ok((table, allocations))
}

fn projection_scale(projection: &HostProjection, expert: usize) -> f32 {
    projection
        .expert_scales
        .as_ref()
        .map(|scales| scales[expert])
        .unwrap_or(projection.scale)
}

fn add_reference_route(output: &mut [bf16], route: ReferenceRoute<'_>) {
    let mut hidden = vec![bf16::ZERO; route.projection.rank];
    for (rank, hidden_value) in hidden.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for column in 0..route.input_features {
            let x = route.input[route.input_row * route.input_features + column].to_f32();
            let a = route.projection.a[(route.expert * route.projection.rank_stride + rank)
                * route.input_features
                + column]
                .to_f32();
            sum += x * a;
        }
        *hidden_value = bf16::from_f32(sum);
    }
    let scale = projection_scale(route.projection, route.expert) * route.route_scale;
    for output_column in 0..route.output_features {
        let mut sum = 0.0f32;
        for (rank, hidden_value) in hidden.iter().enumerate() {
            let b = route.projection.b[(route.expert * route.output_features + output_column)
                * route.projection.rank_stride
                + rank]
                .to_f32();
            sum += hidden_value.to_f32() * b;
        }
        let index =
            route.output_row * route.output_row_stride + route.output_offset + output_column;
        output[index] = bf16::from_f32(output[index].to_f32() + scale * sum);
    }
}

fn assert_close(actual: &[bf16], expected: &[bf16], tolerance: f32) {
    let mut max_difference = 0.0f32;
    let mut max_index = 0;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual.to_f32() - expected.to_f32()).abs();
        if difference > max_difference {
            max_difference = difference;
            max_index = index;
        }
    }
    assert!(
        max_difference <= tolerance,
        "max difference {max_difference} at {max_index}"
    );
}

fn launched_or_platform_skip(status: CutileRoutedLoraStatus, context: &str) -> Result<bool> {
    match status {
        CutileRoutedLoraStatus::Launched => Ok(true),
        CutileRoutedLoraStatus::Unsupported(
            reason @ (CutileRoutedLoraUnsupported::Device
            | CutileRoutedLoraUnsupported::JitUnavailable),
        ) => {
            eprintln!("skipping {context}: {reason:?}");
            Ok(false)
        }
        CutileRoutedLoraStatus::Unsupported(reason) => {
            candle_core::bail!("{context} unsupported: {reason:?}")
        }
    }
}

fn require_launched(status: CutileRoutedLoraStatus, context: &str) -> Result<()> {
    match status {
        CutileRoutedLoraStatus::Launched => Ok(()),
        CutileRoutedLoraStatus::Unsupported(reason) => {
            candle_core::bail!(
                "{context} unsupported after cuTile support was established: {reason:?}"
            )
        }
    }
}

#[test]
fn cutile_routed_lora_matches_reference_for_w13_and_weighted_w2() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;
    let num_tokens = 37usize;
    let top_k = 2usize;
    let num_routes = num_tokens * top_k;
    let num_experts = 4usize;
    let num_adapter_slots = 2usize;
    let token_adapter_slots: Vec<u32> = (0..num_tokens)
        .map(|token| if token % 3 == 0 { 0 } else { 1 })
        .collect();
    let mut expert_ids = Vec::with_capacity(num_routes);
    for token in 0..num_tokens {
        let first = if token % 5 == 0 {
            ((token / 5) % num_experts) as u32
        } else {
            0
        };
        let second = (1 + token % (num_experts - 1)) as u32;
        expert_ids.extend([first, second]);
    }
    let token_slots_device = upload(dev, &token_adapter_slots)?;
    let expert_ids_device = upload(dev, &expert_ids)?;
    let metadata_layout =
        RoutedLoraMetadataLayout::new(num_tokens, top_k, num_experts, num_adapter_slots)?;
    let mut metadata = RoutedLoraCudaMetadata::new(dev, metadata_layout)?;
    unsafe {
        metadata.build(
            address(dev, &token_slots_device),
            address(dev, &expert_ids_device),
        )?;
    }

    let w13_input_features = 48usize;
    let w13_output_features = 40usize;
    let w13_input = patterned_bf16(num_tokens * w13_input_features, 11, 0.55);
    let w13_input_device = upload(dev, &w13_input)?;
    let w13_projections = vec![
        Some(projection(ProjectionSpec {
            num_experts,
            input_features: w13_input_features,
            output_features: w13_output_features,
            rank: 4,
            rank_stride: 8,
            salt: 20,
            scale: 0.5,
            expert_scales: Some(vec![0.65, 0.8, 0.0, 0.7]),
        })),
        Some(projection(ProjectionSpec {
            num_experts,
            input_features: w13_input_features,
            output_features: w13_output_features,
            rank: 12,
            rank_stride: 16,
            salt: 30,
            scale: 0.4,
            expert_scales: None,
        })),
        Some(projection(ProjectionSpec {
            num_experts,
            input_features: w13_input_features,
            output_features: w13_output_features,
            rank: 8,
            rank_stride: 8,
            salt: 40,
            scale: -0.25,
            expert_scales: None,
        })),
        None,
    ];
    let (w13_table, _w13_allocations) = weight_table(dev, &w13_projections, 2, num_adapter_slots)?;
    let w13_row_stride = 2 * w13_output_features;
    let w13_initial = patterned_bf16(num_routes * w13_row_stride, 50, 0.03);
    let mut w13_expected = w13_initial.clone();
    for (route, &expert_id) in expert_ids.iter().enumerate() {
        let token = route / top_k;
        let slot = token_adapter_slots[token] as usize;
        let expert = expert_id as usize;
        for slice in 0..2 {
            if let Some(projection) = &w13_projections[slice * num_adapter_slots + slot] {
                add_reference_route(
                    &mut w13_expected,
                    ReferenceRoute {
                        output_row: route,
                        output_row_stride: w13_row_stride,
                        output_offset: slice * w13_output_features,
                        input: &w13_input,
                        input_row: token,
                        input_features: w13_input_features,
                        output_features: w13_output_features,
                        expert,
                        projection,
                        route_scale: 1.0,
                    },
                );
            }
        }
    }
    let w13_output_device = upload(dev, &w13_initial)?;
    let w13_layout = RoutedLoraProjectionLayout::new(
        w13_input_features,
        w13_output_features,
        w13_row_stride,
        w13_output_features,
        2,
        12,
        RoutedLoraInputMode::TokenRows,
    )?;
    let w13_status = unsafe {
        try_cutile_routed_lora(
            dev,
            &metadata,
            &w13_table,
            CutileRoutedLoraLaunch {
                input: address(dev, &w13_input_device),
                output: address(dev, &w13_output_device),
                route_input_rows: 0,
                route_output_rows: 0,
                route_output_scales: 0,
                dtype: DType::BF16,
                projection: w13_layout,
                weight_slice_offset: 0,
            },
        )?
    };
    if !launched_or_platform_skip(w13_status, "cuTile W13 test path")? {
        return Ok(());
    }
    synchronize(dev)?;
    let w13_actual = dev.clone_dtoh(&w13_output_device)?;
    assert_close(&w13_actual, &w13_expected, 0.035);

    let w2_input_features = 32usize;
    let w2_output_features = 24usize;
    let w2_input = patterned_bf16(num_routes * w2_input_features, 60, 0.45);
    let w2_input_device = upload(dev, &w2_input)?;
    let w2_projections = vec![
        None,
        None,
        Some(projection(ProjectionSpec {
            num_experts,
            input_features: w2_input_features,
            output_features: w2_output_features,
            rank: 6,
            rank_stride: 8,
            salt: 70,
            scale: 0.35,
            expert_scales: Some(vec![0.45, 0.5, 0.4, 0.55]),
        })),
        Some(projection(ProjectionSpec {
            num_experts,
            input_features: w2_input_features,
            output_features: w2_output_features,
            rank: 16,
            rank_stride: 16,
            salt: 80,
            scale: -0.3,
            expert_scales: None,
        })),
    ];
    let (w2_table, _w2_allocations) = weight_table(dev, &w2_projections, 2, num_adapter_slots)?;
    let route_input_rows: Vec<u32> = (0..num_routes)
        .map(|route| ((route * 17) % num_routes) as u32)
        .collect();
    let route_output_rows: Vec<u32> = (0..num_routes)
        .map(|route| ((route * 19) % num_routes) as u32)
        .collect();
    let route_scales: Vec<f32> = (0..num_routes)
        .map(|route| if route % top_k == 0 { 0.72 } else { 0.28 })
        .collect();
    let route_input_rows_device = upload(dev, &route_input_rows)?;
    let route_output_rows_device = upload(dev, &route_output_rows)?;
    let route_scales_device = upload(dev, &route_scales)?;
    let w2_initial = patterned_bf16(num_routes * w2_output_features, 90, 0.025);
    let mut w2_expected = w2_initial.clone();
    for (route, &expert_id) in expert_ids.iter().enumerate() {
        let token = route / top_k;
        let slot = token_adapter_slots[token] as usize;
        let expert = expert_id as usize;
        add_reference_route(
            &mut w2_expected,
            ReferenceRoute {
                output_row: route_output_rows[route] as usize,
                output_row_stride: w2_output_features,
                output_offset: 0,
                input: &w2_input,
                input_row: route_input_rows[route] as usize,
                input_features: w2_input_features,
                output_features: w2_output_features,
                expert,
                projection: w2_projections[num_adapter_slots + slot].as_ref().unwrap(),
                route_scale: route_scales[route],
            },
        );
    }
    let w2_output_device = upload(dev, &w2_initial)?;
    let w2_layout = RoutedLoraProjectionLayout::new(
        w2_input_features,
        w2_output_features,
        w2_output_features,
        0,
        1,
        16,
        RoutedLoraInputMode::RoutedRows,
    )?;
    let w2_status = unsafe {
        try_cutile_routed_lora(
            dev,
            &metadata,
            &w2_table,
            CutileRoutedLoraLaunch {
                input: address(dev, &w2_input_device),
                output: address(dev, &w2_output_device),
                route_input_rows: address(dev, &route_input_rows_device),
                route_output_rows: address(dev, &route_output_rows_device),
                route_output_scales: address(dev, &route_scales_device),
                dtype: DType::BF16,
                projection: w2_layout,
                weight_slice_offset: 1,
            },
        )?
    };
    require_launched(w2_status, "cuTile W2 test path")?;
    synchronize(dev)?;
    let w2_actual = dev.clone_dtoh(&w2_output_device)?;
    assert_close(&w2_actual, &w2_expected, 0.035);
    Ok(())
}

fn check_no_sort_rank(dev: &CudaDevice, rank: usize, allow_platform_skip: bool) -> Result<bool> {
    let num_tokens = 1usize;
    let top_k = 8usize;
    let num_routes = num_tokens * top_k;
    let num_experts = 128usize;
    let num_adapter_slots = 2usize;
    let token_adapter_slots = vec![1u32];
    let expert_ids = vec![0u32, 1, 7, 31, 63, 95, 126, 127];
    let token_slots_device = upload(dev, &token_adapter_slots)?;
    let expert_ids_device = upload(dev, &expert_ids)?;
    let metadata_layout =
        RoutedLoraMetadataLayout::new(num_tokens, top_k, num_experts, num_adapter_slots)?;
    let rank_stride = rank + 16;

    let gate_input_features = 67usize;
    let gate_output_features = 73usize;
    let gate_slice_stride = gate_output_features + 3;
    let gate_row_stride = gate_slice_stride + gate_output_features + 5;
    let gate_scales = (0..num_experts)
        .map(|expert| {
            if expert == 31 {
                0.0
            } else {
                (expert % 9) as f32 * 0.05 - 0.2
            }
        })
        .collect();
    let gate_projections = vec![
        None,
        Some(projection(ProjectionSpec {
            num_experts,
            input_features: gate_input_features,
            output_features: gate_output_features,
            rank,
            rank_stride,
            salt: 200 + rank,
            scale: 0.35,
            expert_scales: Some(gate_scales),
        })),
        None,
        Some(projection(ProjectionSpec {
            num_experts,
            input_features: gate_input_features,
            output_features: gate_output_features,
            rank,
            rank_stride,
            salt: 300 + rank,
            scale: -0.25,
            expert_scales: None,
        })),
    ];
    let (gate_table, _gate_allocations) =
        weight_table(dev, &gate_projections, 2, num_adapter_slots)?;
    let gate_input = patterned_bf16(num_tokens * gate_input_features, 401 + rank, 0.35);
    let gate_input_device = upload(dev, &gate_input)?;
    let gate_initial = patterned_bf16(num_routes * gate_row_stride, 501 + rank, 0.02);
    let mut gate_expected = gate_initial.clone();
    for (route, &expert) in expert_ids.iter().enumerate() {
        for slice in 0..2 {
            add_reference_route(
                &mut gate_expected,
                ReferenceRoute {
                    output_row: route,
                    output_row_stride: gate_row_stride,
                    output_offset: slice * gate_slice_stride,
                    input: &gate_input,
                    input_row: route / top_k,
                    input_features: gate_input_features,
                    output_features: gate_output_features,
                    expert: expert as usize,
                    projection: gate_projections[slice * num_adapter_slots + 1]
                        .as_ref()
                        .unwrap(),
                    route_scale: 1.0,
                },
            );
        }
    }
    let gate_output_device = upload(dev, &gate_initial)?;
    let gate_layout = RoutedLoraProjectionLayout::new(
        gate_input_features,
        gate_output_features,
        gate_row_stride,
        gate_slice_stride,
        2,
        rank,
        RoutedLoraInputMode::TokenRows,
    )?;
    let gate_status = unsafe {
        try_cutile_routed_lora_no_sort(
            dev,
            metadata_layout,
            Some(&token_slots_device),
            &expert_ids_device,
            0,
            &gate_table,
            CutileRoutedLoraLaunch {
                input: address(dev, &gate_input_device),
                output: address(dev, &gate_output_device),
                route_input_rows: 0,
                route_output_rows: 0,
                route_output_scales: 0,
                dtype: DType::BF16,
                projection: gate_layout,
                weight_slice_offset: 0,
            },
        )?
    };
    let gate_context = format!("cuTile no-sort gate/up rank {rank}");
    if allow_platform_skip {
        if !launched_or_platform_skip(gate_status, &gate_context)? {
            return Ok(false);
        }
    } else {
        require_launched(gate_status, &gate_context)?;
    }
    synchronize(dev)?;
    let gate_actual = dev.clone_dtoh(&gate_output_device)?;
    assert_close(&gate_actual, &gate_expected, 0.08);

    let down_input_features = 75usize;
    let down_output_features = 69usize;
    let down_row_stride = down_output_features + 7;
    let down_scales = (0..num_experts)
        .map(|expert| {
            if expert == 95 {
                0.0
            } else {
                0.15 + (expert % 7) as f32 * 0.04
            }
        })
        .collect();
    let down_projections = vec![
        None,
        Some(projection(ProjectionSpec {
            num_experts,
            input_features: down_input_features,
            output_features: down_output_features,
            rank,
            rank_stride,
            salt: 600 + rank,
            scale: 0.3,
            expert_scales: Some(down_scales),
        })),
    ];
    let (down_table, _down_allocations) =
        weight_table(dev, &down_projections, 1, num_adapter_slots)?;
    let down_input = patterned_bf16(num_routes * down_input_features, 701 + rank, 0.3);
    let down_input_device = upload(dev, &down_input)?;
    let route_scales = vec![0.7f32, -0.4, 1.1, 0.25, -0.8, 0.6, 0.9, -0.3];
    let route_scales_device = upload(dev, &route_scales)?;
    let down_initial = patterned_bf16(num_routes * down_row_stride, 801 + rank, 0.02);
    let mut down_expected = down_initial.clone();
    for (route, &expert) in expert_ids.iter().enumerate() {
        add_reference_route(
            &mut down_expected,
            ReferenceRoute {
                output_row: route,
                output_row_stride: down_row_stride,
                output_offset: 0,
                input: &down_input,
                input_row: route,
                input_features: down_input_features,
                output_features: down_output_features,
                expert: expert as usize,
                projection: down_projections[1].as_ref().unwrap(),
                route_scale: route_scales[route],
            },
        );
    }
    let down_output_device = upload(dev, &down_initial)?;
    let down_layout = RoutedLoraProjectionLayout::new(
        down_input_features,
        down_output_features,
        down_row_stride,
        0,
        1,
        rank,
        RoutedLoraInputMode::RoutedRows,
    )?;
    let down_status = unsafe {
        try_cutile_routed_lora_no_sort(
            dev,
            metadata_layout,
            Some(&token_slots_device),
            &expert_ids_device,
            0,
            &down_table,
            CutileRoutedLoraLaunch {
                input: address(dev, &down_input_device),
                output: address(dev, &down_output_device),
                route_input_rows: 0,
                route_output_rows: 0,
                route_output_scales: address(dev, &route_scales_device),
                dtype: DType::BF16,
                projection: down_layout,
                weight_slice_offset: 0,
            },
        )?
    };
    require_launched(
        down_status,
        &format!("cuTile no-sort weighted down rank {rank}"),
    )?;
    synchronize(dev)?;
    let down_actual = dev.clone_dtoh(&down_output_device)?;
    assert_close(&down_actual, &down_expected, 0.08);
    Ok(true)
}

#[test]
fn cutile_no_sort_rank96_and_rank128_match_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;
    if !check_no_sort_rank(dev, 96, true)? {
        return Ok(());
    }
    assert!(check_no_sort_rank(dev, 128, false)?);
    Ok(())
}
