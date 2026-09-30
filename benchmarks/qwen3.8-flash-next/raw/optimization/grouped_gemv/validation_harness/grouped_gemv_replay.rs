use super::*;
use candle_core::cuda::cudarc::driver::{sys, CudaGraph, CudaSlice, DevicePtrMut};
use std::ffi::c_void;

const ROW_PADDING: usize = 512;
const QUANTIZE_THREADS: usize = 256;
const BACKENDS: [&str; 4] = ["default", "indexed_gemv", "grouped_2", "grouped_4"];
const SAMPLE_ENV: &str = "MISTRALRS_MOE_REPLAY_SAMPLE";

struct F32PrecisionGuard(bool);

impl F32PrecisionGuard {
    fn new() -> Self {
        let previous = candle_core::cuda::gemm_reduced_precision_f32();
        candle_core::cuda::set_gemm_reduced_precision_f32(false);
        Self(previous)
    }
}

impl Drop for F32PrecisionGuard {
    fn drop(&mut self) {
        candle_core::cuda::set_gemm_reduced_precision_f32(self.0);
    }
}

fn validate_weights(weights: &Weights) {
    assert_eq!(weights.gate.dtype(), GgmlDType::Q4K);
    assert_eq!(weights.up.dtype(), GgmlDType::Q4K);
    assert_eq!(weights.down.dtype(), GgmlDType::Q4_1);
    assert_eq!(
        weights.gate.shape().dims(),
        &[EXPERTS, INTERMEDIATE, HIDDEN]
    );
    assert_eq!(weights.up.shape().dims(), &[EXPERTS, INTERMEDIATE, HIDDEN]);
    assert_eq!(
        weights.down.shape().dims(),
        &[EXPERTS, HIDDEN, INTERMEDIATE]
    );
}

#[repr(C)]
struct GroupedGemvArgs {
    gate: *const c_void,
    up: *const c_void,
    x: *const c_void,
    bounds: *const u32,
    sorted_ids: *const u32,
    route_weights: *const f32,
    output: *mut f32,
    n: i32,
    k: i32,
    k_padded: i32,
    topk: i32,
    experts: i32,
}

extern "C" {
    fn launch_grouped_gemv_gate_up(
        args: *const GroupedGemvArgs,
        width: i32,
        stream: *mut c_void,
    ) -> i32;
    fn launch_grouped_gemv_down(
        args: *const GroupedGemvArgs,
        width: i32,
        batch: i32,
        stream: *mut c_void,
    ) -> i32;
    fn launch_quantize_q8_1_bf16(
        x: *const c_void,
        out: *mut c_void,
        k: i32,
        padded: i32,
        rows: i32,
        stream: *mut c_void,
    );
    fn launch_quantize_q8_1(
        x: *const f32,
        out: *mut c_void,
        k: i32,
        padded: i32,
        blocks: i32,
        rows: i32,
        stream: *mut c_void,
    );
}

struct Candidate {
    quant: CudaSlice<u8>,
    activated: Tensor,
    output: Tensor,
    width: i32,
}

impl Candidate {
    fn new(inputs: &Inputs, width: i32) -> Result<Self> {
        let rows = inputs.xs.dim(0)?;
        assert_eq!(inputs.xs.dtype(), DType::BF16);
        assert_eq!(inputs.xs.dims(), &[rows, HIDDEN]);
        assert!(inputs.xs.is_contiguous());
        assert_eq!(inputs.ids.dtype(), DType::U32);
        assert_eq!(inputs.ids.dims(), &[rows, TOPK]);
        assert!(inputs.ids.is_contiguous());
        assert_eq!(inputs.weights.dtype(), DType::F32);
        assert_eq!(inputs.weights.dims(), &[rows, TOPK]);
        assert!(inputs.weights.is_contiguous());
        assert!([2, 4].contains(&width));
        let q8_bytes = |rows: usize, k: usize| {
            rows * k.div_ceil(ROW_PADDING) * ROW_PADDING / GgmlDType::Q8_1.block_size()
                * GgmlDType::Q8_1.type_size()
        };
        let device = inputs.xs.device();
        Ok(Self {
            quant: unsafe {
                device
                    .as_cuda_device()?
                    .alloc::<u8>(q8_bytes(rows, HIDDEN).max(q8_bytes(rows * TOPK, INTERMEDIATE)))?
            },
            activated: Tensor::zeros((rows * TOPK, INTERMEDIATE), DType::F32, device)?,
            output: Tensor::zeros((rows, HIDDEN), DType::F32, device)?,
            width,
        })
    }

    fn forward(&mut self, inputs: &Inputs, weights: &Weights) -> Result<Tensor> {
        let rows = inputs.xs.dim(0)?;
        let dev = inputs.xs.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let stream_ptr = stream.cu_stream() as *mut c_void;
        let (ids_storage, _) = inputs.ids.storage_and_layout();
        let Storage::Cuda(ids) = &*ids_storage else {
            unreachable!()
        };
        let (bounds, sorted, _sources) =
            moe_dispatch_build(ids.as_cuda_slice::<u32>()?, rows * TOPK, EXPERTS, TOPK, dev)?;
        let (bounds_ptr, _bg) = bounds.device_ptr(&stream);
        let (sorted_ptr, _sg) = sorted.device_ptr(&stream);
        let (xs_storage, layout) = inputs.xs.storage_and_layout();
        assert_eq!(layout.start_offset(), 0);
        let Storage::Cuda(xs) = &*xs_storage else {
            unreachable!()
        };
        let (input_ptr, _ig) = xs.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
        let (weight_storage, _) = inputs.weights.storage_and_layout();
        let Storage::Cuda(routes) = &*weight_storage else {
            unreachable!()
        };
        let (routes_ptr, _rg) = routes.as_cuda_slice::<f32>()?.device_ptr(&stream);
        let (gate_ptr, _gg) = weights.gate.device_ptr_with_guard(&stream)?;
        let (up_ptr, _ug) = weights.up.device_ptr_with_guard(&stream)?;
        let (down_ptr, _dg) = weights.down.device_ptr_with_guard(&stream)?;
        let (activated_storage, _) = self.activated.storage_and_layout();
        let Storage::Cuda(activated) = &*activated_storage else {
            unreachable!()
        };
        let (activated_ptr, _ag) = activated.as_cuda_slice::<f32>()?.device_ptr(&stream);
        let (output_storage, _) = self.output.storage_and_layout();
        let Storage::Cuda(output) = &*output_storage else {
            unreachable!()
        };
        let (output_ptr, _og) = output.as_cuda_slice::<f32>()?.device_ptr(&stream);
        let (quant_ptr, _qg) = self.quant.device_ptr_mut(&stream);
        let gate_pad = HIDDEN.div_ceil(ROW_PADDING) * ROW_PADDING;
        let down_pad = INTERMEDIATE.div_ceil(ROW_PADDING) * ROW_PADDING;
        let mut args = GroupedGemvArgs {
            gate: gate_ptr as *const c_void,
            up: up_ptr as *const c_void,
            x: quant_ptr as *const c_void,
            bounds: bounds_ptr as *const u32,
            sorted_ids: sorted_ptr as *const u32,
            route_weights: routes_ptr as *const f32,
            output: activated_ptr as *mut f32,
            n: INTERMEDIATE as i32,
            k: HIDDEN as i32,
            k_padded: gate_pad as i32,
            topk: TOPK as i32,
            experts: EXPERTS as i32,
        };
        unsafe {
            launch_quantize_q8_1_bf16(
                input_ptr as *const c_void,
                quant_ptr as *mut c_void,
                HIDDEN as i32,
                gate_pad as i32,
                rows as i32,
                stream_ptr,
            );
            assert_eq!(
                launch_grouped_gemv_gate_up(&args, self.width, stream_ptr),
                0
            );
            launch_quantize_q8_1(
                activated_ptr as *const f32,
                quant_ptr as *mut c_void,
                INTERMEDIATE as i32,
                down_pad as i32,
                down_pad.div_ceil(QUANTIZE_THREADS) as i32,
                (rows * TOPK) as i32,
                stream_ptr,
            );
            args.gate = down_ptr as *const c_void;
            args.up = std::ptr::null();
            args.output = output_ptr as *mut f32;
            args.n = HIDDEN as i32;
            args.k = INTERMEDIATE as i32;
            args.k_padded = down_pad as i32;
            assert_eq!(
                launch_grouped_gemv_down(&args, self.width, rows as i32, stream_ptr),
                0
            );
        }
        self.output.to_dtype(DType::BF16)
    }
}

fn forward(
    index: usize,
    inputs: &Inputs,
    weights: &Weights,
    candidates: &mut [Candidate; 2],
) -> Result<Tensor> {
    match index {
        0 => inputs.forward(weights, inputs.xs.dim(0)? >= 32),
        1 => inputs.forward(weights, false),
        2 | 3 => candidates[index - 2].forward(inputs, weights),
        _ => unreachable!(),
    }
}

fn capture(
    inputs: &Inputs,
    mut forward: impl FnMut() -> Result<Tensor>,
) -> Result<(CudaGraph, Tensor)> {
    let dev = inputs.xs.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let _cache = dev.enable_cuda_graph_htod_cache();
    drop(forward()?);
    inputs.xs.device().synchronize()?;
    let tracking = stream.context().is_event_tracking();
    if tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    let started = stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED);
    if let Err(error) = started {
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        return Err(candle_core::Error::wrap(error));
    }
    let output = forward();
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = output?;
    let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
    graph.launch().map_err(candle_core::Error::wrap)?;
    inputs.xs.device().synchronize()?;
    Ok((graph, output))
}

fn strict(reference: &Tensor, actual: &Tensor) -> Result<Value> {
    let m = comparison_metrics(reference, actual)?;
    assert!(
        m["relative_rms"].as_f64().unwrap() < MAX_BOUND_RELATIVE_RMS
            && m["cosine"].as_f64().unwrap() > MIN_BOUND_COSINE,
        "strict GEMV/graph comparison failed: {m}"
    );
    Ok(m)
}

#[test]
#[ignore = "requires exact captured operands and exclusive GPU access"]
fn flash_next_grouped_gemv_real_routing() -> Result<()> {
    let directory = PathBuf::from(std::env::var(REPLAY_DIRECTORY_ENV).unwrap());
    let output = PathBuf::from(std::env::var("MISTRALRS_MOE_BENCH_OUTPUT").unwrap());
    let archive = PathBuf::from(std::env::var(REPLAY_OUTPUT_DIRECTORY_ENV).unwrap());
    assert!(!output.exists());
    std::fs::create_dir(&archive)?;
    let selected = std::env::var(SAMPLE_ENV).ok();
    let device = Device::new_cuda(0)?;
    let weights = Weights::gguf(&directory, &device)?;
    validate_weights(&weights);
    let mut paths = std::fs::read_dir(&directory)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    let mut results = Vec::new();
    let mut changed_route_checked = false;
    for path in paths {
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap();
        if selected.as_deref().is_some_and(|s| s != name) {
            continue;
        }
        let metadata: Value =
            serde_json::from_reader(File::open(&path)?).map_err(candle_core::Error::wrap)?;
        let Some(tensor_file) = metadata["tensor_file"].as_str() else {
            continue;
        };
        if metadata["query_len"] != REPLAY_QUERY_LEN
            || !REPLAY_TILE_BATCHES.contains(&metadata["batch"].as_u64().unwrap())
        {
            continue;
        }
        assert_eq!(metadata["derived"], false);
        assert_eq!(metadata["layer"], LAYER);
        let tensors = candle_core::safetensors::load(directory.join(tensor_file), &device)?;
        let full_rows = tensors["xs"].dim(0)?;
        let mut selections = vec![("native", (0..full_rows as u32).collect::<Vec<_>>())];
        if full_rows == REPLAY_DERIVE_FROM_ROWS && selected.is_none() {
            selections.extend(
                REPLAY_SEQUENCE_QUERY_COUNTS
                    .into_iter()
                    .map(|q| ("sequence_query_prefix", sequence_query_prefix_rows(q))),
            );
        }
        for (selection, source_rows) in selections {
            let row_indices = Tensor::new(source_rows.as_slice(), &device)?;
            let inputs = replay_inputs(&tensors, &row_indices)?;
            let rows = source_rows.len();
            let mut candidates = [Candidate::new(&inputs, 2)?, Candidate::new(&inputs, 4)?];
            let eager = (0..BACKENDS.len())
                .map(|i| forward(i, &inputs, &weights, &mut candidates))
                .collect::<Result<Vec<_>>>()?;
            let candidate_vs_gemv = [strict(&eager[1], &eager[2])?, strict(&eager[1], &eager[3])?];
            let candidate_vs_default = [
                comparison_metrics(&eager[0], &eager[2])?,
                comparison_metrics(&eager[0], &eager[3])?,
            ];
            let native_guard = if selection == "native" {
                let captured = tensors["output"]
                    .index_select(&row_indices, 0)?
                    .contiguous()?;
                Some(strict(&captured, &eager[0])?)
            } else {
                None
            };
            let graphs = (0..BACKENDS.len())
                .map(|i| capture(&inputs, || forward(i, &inputs, &weights, &mut candidates)))
                .collect::<Result<Vec<_>>>()?;
            let graph_checks = graphs
                .iter()
                .zip(&eager)
                .map(|((_, actual), expected)| strict(expected, actual))
                .collect::<Result<Vec<_>>>()?;
            let mut changed_route_checks = Vec::new();
            if !changed_route_checked {
                let original = inputs.ids.copy()?;
                let alternate = inputs
                    .ids
                    .flatten_all()?
                    .to_vec1::<u32>()?
                    .into_iter()
                    .map(|id| (id + 1) % EXPERTS as u32)
                    .collect::<Vec<_>>();
                let alternate = Tensor::from_vec(alternate, (rows, TOPK), &device)?;
                inputs.ids.slice_set(&alternate, 0, 0)?;
                let reference = inputs.forward(&weights, false)?;
                for i in [2usize, 3] {
                    graphs[i].0.launch().map_err(candle_core::Error::wrap)?;
                    device.synchronize()?;
                    let eager_alternate = forward(i, &inputs, &weights, &mut candidates)?;
                    changed_route_checks.push(json!({"backend":BACKENDS[i], "vs_gemv":strict(&reference, &graphs[i].1)?, "vs_eager":strict(&eager_alternate, &graphs[i].1)?}));
                }
                inputs.ids.slice_set(&original, 0, 0)?;
                for (i, (graph, actual)) in graphs.iter().enumerate() {
                    graph.launch().map_err(candle_core::Error::wrap)?;
                    device.synchronize()?;
                    strict(&eager[i], actual)?;
                }
                changed_route_checked = true;
            }
            for _ in 0..0 {
                for (i, (graph, _)) in graphs.iter().enumerate() {
                    forward(i, &inputs, &weights, &mut candidates)?;
                    graph.launch().map_err(candle_core::Error::wrap)?;
                }
            }
            let mut eager_timings: [Vec<Value>; 4] = std::array::from_fn(|_| Vec::new());
            let mut graph_timings: [Vec<Value>; 4] = std::array::from_fn(|_| Vec::new());
            for round in 0..0 {
                for offset in 0..BACKENDS.len() {
                    let index = (round + offset) % BACKENDS.len();
                    eager_timings[index].push(measure(&inputs, || {
                        forward(index, &inputs, &weights, &mut candidates)
                    })?);
                    graph_timings[index].push(measure(&inputs, || {
                        graphs[index].0.launch().map_err(candle_core::Error::wrap)?;
                        Ok(graphs[index].1.clone())
                    })?);
                }
            }
            let tensor_name = format!("{name}.{selection}.{rows}.safetensors");
            let outputs = BACKENDS
                .iter()
                .zip(&eager)
                .map(|(&name, tensor)| Ok((name.to_string(), tensor.to_device(&Device::Cpu)?)))
                .collect::<Result<std::collections::HashMap<_, _>>>()?;
            candle_core::safetensors::save(&outputs, archive.join(&tensor_name))?;
            let timings = BACKENDS
                .iter()
                .enumerate()
                .map(|(i, &backend)| {
                    (
                        backend.to_string(),
                        json!({"eager":eager_timings[i], "graph":graph_timings[i]}),
                    )
                })
                .collect::<serde_json::Map<_, _>>();
            let result = json!({"source_metadata":name,"rows":rows,"selection":selection,"derived":selection!="native","source_row_indices":source_rows,"capture":metadata,"active_experts":inputs.active_experts,"max_expert_rows":inputs.max_expert_rows,"candidate_vs_gemv":candidate_vs_gemv,"candidate_vs_default":candidate_vs_default,"native_guard":native_guard,"graph_checks":graph_checks,"changed_route_checks":changed_route_checks,"timings":timings,"output_file":tensor_name});
            eprintln!("GROUPED_GEMV_RESULT {result}");
            results.push(result);
        }
    }
    assert_eq!(results.len(), if selected.is_some() { 1 } else { 25 });
    let report = json!({"backends":BACKENDS,"validation_only":true,"warmups":0,"rounds":0,"repetitions":0,"candidate_contract":"same Q8_1 activations and F32 intermediates as indexed GEMV; down atomic reduction may reorder","results":results});
    serde_json::to_writer_pretty(File::create(output)?, &report)
        .map_err(candle_core::Error::wrap)?;
    Ok(())
}

#[test]
#[ignore = "requires completed grouped GEMV replay outputs and exclusive GPU access"]
fn flash_next_grouped_gemv_f32_oracle() -> Result<()> {
    let precision = F32PrecisionGuard::new();
    let directory = PathBuf::from(std::env::var(REPLAY_DIRECTORY_ENV).unwrap());
    let archive = PathBuf::from(std::env::var(REPLAY_OUTPUT_DIRECTORY_ENV).unwrap());
    let report: Value =
        serde_json::from_reader(File::open(std::env::var(ORACLE_RESULTS_ENV).unwrap())?)
            .map_err(candle_core::Error::wrap)?;
    let output = PathBuf::from(std::env::var(ORACLE_OUTPUT_ENV).unwrap());
    assert!(!output.exists());
    let device = Device::new_cuda(0)?;
    let weights = Weights::gguf(&directory, &device)?;
    validate_weights(&weights);
    let oracle = F32ExpertOracle::new(&weights, &device)?;
    let mut results = Vec::new();
    for sample in report["results"].as_array().unwrap() {
        let tensors = candle_core::safetensors::load(
            directory.join(sample["capture"]["tensor_file"].as_str().unwrap()),
            &device,
        )?;
        let rows = sample["source_row_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let inputs = replay_inputs(&tensors, &Tensor::new(rows.as_slice(), &device)?)?;
        let reference = oracle.forward(&inputs)?;
        let outputs = candle_core::safetensors::load(
            archive.join(sample["output_file"].as_str().unwrap()),
            &Device::Cpu,
        )?;
        let metrics = BACKENDS
            .iter()
            .map(|&name| {
                Ok((
                    name.to_string(),
                    comparison_metrics(&reference, &outputs[name])?,
                ))
            })
            .collect::<Result<serde_json::Map<_, _>>>()?;
        results.push(json!({"source_metadata":sample["source_metadata"],"selection":sample["selection"],"source_row_indices":rows,"metrics":metrics}));
    }
    serde_json::to_writer_pretty(File::create(output)?, &json!({"reference":"original dequantized FP32 weights, FP32 GEMMs with TF32 disabled, FP64 CPU weighted reduction; no new tolerance claim","reduced_f32_before":precision.0,"reduced_f32_during":false,"results":results})).map_err(candle_core::Error::wrap)?;
    Ok(())
}
