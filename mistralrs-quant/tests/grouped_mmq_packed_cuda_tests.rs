#![cfg(feature = "cuda")]

use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Result, Storage, Tensor,
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
const GRAPH_NUM_EXPERTS: usize = 512;
const GRAPH_TOKEN_COUNTS: [usize; 5] = [24, 28, 32, 35, 40];
const GRAPH_GROW_TOKENS: usize = 128;
const GRAPH_TOPK: usize = 10;
const GRAPH_HIDDEN: usize = 256;
const GRAPH_INTERMEDIATE: usize = 128;
const GRAPH_ROUTE_TOKEN_STRIDE: usize = 17;
const GRAPH_ROUTE_EXPERT_STRIDE: usize = 53;
const GRAPH_ROUTE_VARIANT_OFFSET: usize = 137;
const GRAPH_REPLAY_CASES: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
const GRAPH_MIN_VARIATION: f32 = 1e-4;

// Capture changes event tracking on the shared primary CUDA context.
static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn patterned(shape: impl Into<candle_core::Shape>, salt: usize, scale: f32) -> Result<Tensor> {
    let shape = shape.into();
    let values = (0..shape.elem_count())
        .map(|index| {
            let value = (index.wrapping_mul(37) + salt.wrapping_mul(19)) % 211;
            (f32::from(u16::try_from(value).unwrap()) / 105.0 - 1.0) * scale
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
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
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

struct GraphMoeWeights {
    gate: QTensor,
    up: QTensor,
    down: QTensor,
}

impl GraphMoeWeights {
    fn new(device: &Device) -> Result<Self> {
        let gate_shape = (GRAPH_NUM_EXPERTS, GRAPH_INTERMEDIATE, GRAPH_HIDDEN);
        Ok(Self {
            gate: QTensor::quantize_onto(
                &patterned(gate_shape, 11, 0.11)?,
                GgmlDType::Q4K,
                device,
            )?,
            up: QTensor::quantize_onto(&patterned(gate_shape, 29, 0.09)?, GgmlDType::Q4K, device)?,
            down: QTensor::quantize_onto(
                &patterned(
                    (GRAPH_NUM_EXPERTS, GRAPH_HIDDEN, GRAPH_INTERMEDIATE),
                    47,
                    0.1,
                )?,
                GgmlDType::Q4_1,
                device,
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, routes: &Tensor) -> Result<Tensor> {
        let num_tokens = xs.dim(0)?;
        let assignments = num_tokens * GRAPH_TOPK;
        let dev = xs.device().as_cuda_device()?;
        let ids = routes.flatten_all()?.contiguous()?;
        let (storage, layout) = ids.storage_and_layout();
        assert_eq!(layout.start_offset(), 0);
        let Storage::Cuda(storage) = &*storage else {
            unreachable!()
        };
        let (bounds, sorted_tokens, sorted_sources) = moe_dispatch_build(
            storage.as_cuda_slice::<u32>()?,
            assignments,
            GRAPH_NUM_EXPERTS,
            GRAPH_TOPK,
            dev,
        )?;
        let gate_up = grouped_moe_mmq_pair_packed(
            &self.gate,
            &self.up,
            xs,
            &sorted_sources,
            &sorted_tokens,
            &bounds,
            assignments,
            GRAPH_TOPK,
            GRAPH_NUM_EXPERTS,
            dev,
        )?;
        grouped_moe_mmq_from_glu_packed(
            &self.down,
            &gate_up,
            &sorted_tokens,
            &sorted_tokens,
            &bounds,
            assignments,
            num_tokens,
            GRAPH_NUM_EXPERTS,
            GluActivationType::Silu as i32,
            dev,
        )
    }
}

fn graph_routes(num_tokens: usize, variant: usize, device: &Device) -> Result<Tensor> {
    let routes = (0..num_tokens)
        .flat_map(|token| {
            (0..GRAPH_TOPK).map(move |rank| {
                u32::try_from(
                    (token * GRAPH_ROUTE_TOKEN_STRIDE
                        + rank * GRAPH_ROUTE_EXPERT_STRIDE
                        + variant * GRAPH_ROUTE_VARIANT_OFFSET)
                        % GRAPH_NUM_EXPERTS,
                )
                .unwrap()
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(routes, (num_tokens, GRAPH_TOPK), device)
}

#[test]
fn grouped_mmq_graph_replay_updates_inputs_and_routes_after_workspace_growth() -> Result<()> {
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    for num_tokens in GRAPH_TOKEN_COUNTS {
        check_graph_replay(num_tokens)?;
    }
    Ok(())
}

fn check_graph_replay(num_tokens: usize) -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    let cuda = Device::new_cuda(0)?;
    let dev = cuda.as_cuda_device()?;
    let stream = dev.cuda_stream();
    let weights = GraphMoeWeights::new(&cuda)?;
    let inputs = [
        patterned((num_tokens, GRAPH_HIDDEN), 3, 0.7)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?,
        patterned((num_tokens, GRAPH_HIDDEN), 97, 0.4)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?,
    ];
    let routes = [
        graph_routes(num_tokens, 0, &cuda)?,
        graph_routes(num_tokens, 1, &cuda)?,
    ];
    let expected = GRAPH_REPLAY_CASES
        .iter()
        .map(|&(input, route)| {
            weights
                .forward(&inputs[input], &routes[route])?
                .to_device(&Device::Cpu)
        })
        .collect::<Result<Vec<_>>>()?;
    for changed in &expected[1..] {
        let difference = (&expected[0] - changed)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(difference > GRAPH_MIN_VARIATION);
    }

    let xs = Tensor::zeros((num_tokens, GRAPH_HIDDEN), DType::BF16, &cuda)?;
    let ids = Tensor::zeros((num_tokens, GRAPH_TOPK), DType::U32, &cuda)?;
    xs.slice_set(&inputs[0], 0, 0)?;
    ids.slice_set(&routes[0], 0, 0)?;
    let _htod_cache_guard = dev.enable_cuda_graph_htod_cache();
    drop(weights.forward(&xs, &ids)?);
    cuda.synchronize()?;

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
        return Err(candle_core::Error::msg(error.to_string()));
    }
    let captured = weights.forward(&xs, &ids);
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = captured?;
    let graph = graph
        .map_err(|error| candle_core::Error::msg(error.to_string()))?
        .ok_or_else(|| candle_core::Error::msg("grouped MMQ capture produced no graph"))?;

    for grow_workspace in [false, true] {
        if grow_workspace {
            let larger_input = patterned((GRAPH_GROW_TOKENS, GRAPH_HIDDEN), 61, 0.6)?
                .to_dtype(DType::BF16)?
                .to_device(&cuda)?;
            let larger_routes = graph_routes(GRAPH_GROW_TOKENS, 1, &cuda)?;
            drop(weights.forward(&larger_input, &larger_routes)?);
            cuda.synchronize()?;
        }
        for (&(input, route), expected) in GRAPH_REPLAY_CASES.iter().zip(&expected) {
            xs.slice_set(&inputs[input], 0, 0)?;
            ids.slice_set(&routes[route], 0, 0)?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            cuda.synchronize()?;
            assert_close(&output, expected)?;
        }
    }
    drop(output);
    cuda.synchronize()?;
    drop(graph);
    cuda.synchronize()
}
