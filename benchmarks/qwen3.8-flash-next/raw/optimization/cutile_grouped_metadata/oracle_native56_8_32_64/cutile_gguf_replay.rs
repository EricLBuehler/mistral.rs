use super::*;
use candle_core::cuda::cudarc::driver::{sys, CudaGraph};
use mistralrs_quant::{
    cutile::gguf_moe::{gguf_moe_projection, GgufMoeConfig, GgufMoeProjection},
    fused_glu,
    moe::cuda::moe_align,
};

const TILE_ENV: &str = "MISTRALRS_GGUF_MOE_TILE";
const SAMPLE_ENV: &str = "MISTRALRS_MOE_REPLAY_SAMPLE";
const GROUPED_MIN_ROWS: usize = 32;

fn config() -> Result<GgufMoeConfig> {
    let Some(value) = std::env::var_os(TILE_ENV) else {
        return Ok(GgufMoeConfig::default());
    };
    let parts = value
        .to_str()
        .unwrap()
        .split(',')
        .map(str::parse::<i32>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(candle_core::Error::wrap)?;
    assert_eq!(parts.len(), 3, "tile must be BM,BN,BK");
    Ok(GgufMoeConfig {
        bm: parts[0],
        bn: parts[1],
        bk: parts[2],
    })
}

fn forward(inputs: &Inputs, weights: &Weights, config: GgufMoeConfig) -> Result<Tensor> {
    let rows = inputs.xs.dim(0)?;
    let dev = inputs.xs.device().as_cuda_device()?;
    let (ids_storage, _) = inputs.ids.storage_and_layout();
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        unreachable!()
    };
    let (sorted, experts, padded, capacity) = moe_align(
        ids_cuda.as_cuda_slice::<u32>()?,
        rows,
        EXPERTS,
        TOPK,
        config.bm,
        dev,
    )?;
    let project = |input: &Tensor, weight: &QTensor, top_k| {
        gguf_moe_projection(GgufMoeProjection {
            input,
            weight,
            sorted_token_ids: &sorted,
            expert_ids: &experts,
            num_tokens_post_pad: &padded,
            padded_capacity: capacity,
            assignments: rows * TOPK,
            top_k,
            config,
        })
    };
    let gate = project(&inputs.xs, &weights.gate, TOPK)?;
    let up = project(&inputs.xs, &weights.up, TOPK)?;
    let activated = fused_glu(&gate, &up, GluActivationType::Silu)?.to_dtype(DType::BF16)?;
    let down = project(&activated, &weights.down, 1)?;
    let (weight_storage, _) = inputs.weights.storage_and_layout();
    let Storage::Cuda(weight_cuda) = &*weight_storage else {
        unreachable!()
    };
    let weight_slice = weight_cuda.as_cuda_slice::<f32>()?;
    let (weight_ptr, _guard) = weight_slice.device_ptr(weight_slice.stream());
    unsafe { moe_weighted_reduce_flat_bf16(&down, weight_ptr as *const f32, rows, TOPK, dev) }
}

fn capture(
    inputs: &Inputs,
    mut forward: impl FnMut() -> Result<Tensor>,
) -> Result<(CudaGraph, Tensor)> {
    let dev = inputs.xs.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let _htod_cache_guard = dev.enable_cuda_graph_htod_cache();
    drop(forward()?);
    inputs.xs.device().synchronize()?;
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

#[test]
#[ignore = "requires exact captured Flash-Next operands and exclusive GPU access"]
fn flash_next_cutile_gguf_real_routing() -> Result<()> {
    let directory = PathBuf::from(std::env::var(REPLAY_DIRECTORY_ENV).unwrap());
    let output = PathBuf::from(std::env::var("MISTRALRS_MOE_BENCH_OUTPUT").unwrap());
    let output_directory = PathBuf::from(std::env::var(REPLAY_OUTPUT_DIRECTORY_ENV).unwrap());
    assert!(!output.exists(), "preserve prior benchmark output");
    std::fs::create_dir(&output_directory)?;
    let selected = std::env::var(SAMPLE_ENV).ok();
    let config = config()?;
    let device = Device::new_cuda(0)?;
    let weights = Weights::gguf(&directory, &device)?;
    let mut paths = std::fs::read_dir(&directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    let mut results = Vec::new();
    for path in paths {
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap();
        if selected.as_deref().is_some_and(|selected| name != selected) {
            continue;
        }
        let metadata: Value =
            serde_json::from_reader(File::open(&path)?).map_err(candle_core::Error::wrap)?;
        let Some(tensor_file) = metadata["tensor_file"].as_str() else {
            continue;
        };
        assert_eq!(metadata["derived"], false);
        assert_eq!(metadata["layer"], LAYER);
        let tensors = candle_core::safetensors::load(directory.join(tensor_file), &device)?;
        let full_rows = tensors["xs"].dim(0)?;
        let mut selections = vec![("native", (0..full_rows as u32).collect::<Vec<_>>())];
        if full_rows == REPLAY_DERIVE_FROM_ROWS && selected.is_none() {
            selections.extend(
                REPLAY_SEQUENCE_QUERY_COUNTS
                    .into_iter()
                    .map(|queries| ("sequence_query_prefix", sequence_query_prefix_rows(queries))),
            );
        }
        for (selection, source_rows) in selections {
            let inputs = replay_inputs(&tensors, &Tensor::new(source_rows.as_slice(), &device)?)?;
            let rows = inputs.xs.dim(0)?;
            let grouped = rows >= GROUPED_MIN_ROWS;
            let baseline = inputs.forward(&weights, grouped)?;
            let captured_vs_baseline = if selection == "native" {
                assert_eq!(
                    metadata["source_dispatch"],
                    if grouped {
                        "grouped_mmq"
                    } else {
                        "indexed_gemv"
                    }
                );
                let comparison = comparison_metrics(&tensors["output"], &baseline)?;
                assert!(comparison["relative_rms"].as_f64().unwrap() < MAX_BOUND_RELATIVE_RMS);
                assert!(comparison["cosine"].as_f64().unwrap() > MIN_BOUND_COSINE);
                Some(comparison)
            } else {
                None
            };
            let candidate = forward(&inputs, &weights, config)?;
            let numerical = comparison_metrics(&baseline, &candidate)?;
            let (baseline_graph, baseline_output) =
                capture(&inputs, || inputs.forward(&weights, grouped))?;
            let (candidate_graph, candidate_output) =
                capture(&inputs, || forward(&inputs, &weights, config))?;
            let baseline_replay = comparison_metrics(&baseline, &baseline_output)?;
            let candidate_replay = comparison_metrics(&candidate, &candidate_output)?;
            assert!(baseline_replay["relative_rms"].as_f64().unwrap() < MAX_BOUND_RELATIVE_RMS);
            assert!(candidate_replay["relative_rms"].as_f64().unwrap() < MAX_BOUND_RELATIVE_RMS);
            for _ in 0..WARMUPS {
                drop(inputs.forward(&weights, grouped)?);
                drop(forward(&inputs, &weights, config)?);
                baseline_graph.launch().map_err(candle_core::Error::wrap)?;
                candidate_graph.launch().map_err(candle_core::Error::wrap)?;
            }
            let mut baseline_eager = Vec::new();
            let mut candidate_eager = Vec::new();
            let mut baseline_replay_times = Vec::new();
            let mut candidate_replay_times = Vec::new();
            for round in 0..ROUNDS {
                for candidate_first in [round % 2 == 0, round % 2 != 0] {
                    if candidate_first {
                        candidate_eager
                            .push(measure(&inputs, || forward(&inputs, &weights, config))?);
                        candidate_replay_times.push(measure(&inputs, || {
                            candidate_graph.launch().map_err(candle_core::Error::wrap)?;
                            Ok(candidate_output.clone())
                        })?);
                    } else {
                        baseline_eager
                            .push(measure(&inputs, || inputs.forward(&weights, grouped))?);
                        baseline_replay_times.push(measure(&inputs, || {
                            baseline_graph.launch().map_err(candle_core::Error::wrap)?;
                            Ok(baseline_output.clone())
                        })?);
                    }
                }
            }
            let saved_name = format!("{name}.{selection}.{rows}.safetensors");
            candle_core::safetensors::save(
                &std::collections::HashMap::from([
                    ("baseline".to_owned(), baseline),
                    ("cutile".to_owned(), candidate),
                ]),
                output_directory.join(&saved_name),
            )?;
            let result = json!({
                "source_metadata": name, "capture": metadata, "selection": selection,
                "source_row_indices": source_rows, "rows": rows, "derived": selection != "native",
                "active_experts": inputs.active_experts, "max_expert_rows": inputs.max_expert_rows,
                "baseline_dispatch": if grouped {"grouped_mmq"} else {"indexed_gemv"},
                "captured_vs_baseline": captured_vs_baseline,
                "numerical": numerical, "baseline_graph_vs_eager": baseline_replay,
                "candidate_graph_vs_eager": candidate_replay,
                "baseline_eager": baseline_eager, "candidate_eager": candidate_eager,
                "baseline_graph": baseline_replay_times, "candidate_graph": candidate_replay_times,
                "saved_outputs": saved_name,
            });
            eprintln!("CUTILE_GGUF_RESULT {result}");
            results.push(result);
            drop((baseline_output, candidate_output));
            device.synchronize()?;
            drop((baseline_graph, candidate_graph));
            device.synchronize()?;
        }
    }
    assert!(!results.is_empty());
    serde_json::to_writer_pretty(File::create(output)?, &json!({
        "config": {"bm":config.bm,"bn":config.bn,"bk":config.bk},
        "layer": LAYER, "experts": EXPERTS, "topk": TOPK, "hidden": HIDDEN,
        "intermediate": INTERMEDIATE, "warmups": WARMUPS, "rounds": ROUNDS,
        "repetitions": REPETITIONS, "results": results,
        "output_directory": output_directory,
        "scope": "Whole routed expert forward including GPU routing, gate/up, FP32 SiLU product, BF16 intermediate, down and weighted reduction; original compressed weights; eager and captured graph timings; excludes JIT and capture; warmed repeated layer, not serving throughput.",
        "numerics": "cuTile rounds decoded weights to BF16, uses original BF16 activations and FP32 accumulation. Existing MMQ/GEMV retain their own activation quantization and scale rounding; byte equality is not expected.",
    })).map_err(candle_core::Error::wrap)?;
    Ok(())
}

struct RoundedOracle(F32ExpertOracle);

impl RoundedOracle {
    fn new(weights: &Weights, device: &Device) -> Result<Self> {
        let rounded = |weight: &QTensor| {
            weight
                .dequantize(device)?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)
        };
        Ok(Self(F32ExpertOracle {
            gate: rounded(&weights.gate)?,
            up: rounded(&weights.up)?,
            down: rounded(&weights.down)?,
        }))
    }

    fn forward(&self, inputs: &Inputs) -> Result<Tensor> {
        let rows = inputs.xs.dim(0)?;
        let xs = inputs.xs.to_dtype(DType::F32)?;
        let ids = inputs.ids.to_vec2::<u32>()?;
        let weights = inputs.weights.to_vec2::<f32>()?;
        let mut assignments = vec![Vec::<(usize, usize)>::new(); EXPERTS];
        for (row, experts) in ids.iter().enumerate() {
            for (slot, &expert) in experts.iter().enumerate() {
                assignments[usize::try_from(expert).unwrap()].push((row, slot));
            }
        }
        let mut output = vec![0.0_f64; rows * HIDDEN];
        for (expert, assignments) in assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let indices = assignments
                .iter()
                .map(|&(row, _)| u32::try_from(row).unwrap())
                .collect::<Vec<_>>();
            let selected = xs.index_select(&Tensor::new(indices.as_slice(), xs.device())?, 0)?;
            let gate = selected.matmul(&self.0.gate.get(expert)?.t()?.contiguous()?)?;
            let up = selected.matmul(&self.0.up.get(expert)?.t()?.contiguous()?)?;
            let activated = (candle_nn::ops::silu(&gate)? * up)?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?;
            let values = activated
                .matmul(&self.0.down.get(expert)?.t()?.contiguous()?)?
                .to_vec2::<f32>()?;
            for (&(row, slot), values) in assignments.iter().zip(&values) {
                for (column, &value) in values.iter().enumerate() {
                    output[row * HIDDEN + column] +=
                        f64::from(weights[row][slot]) * f64::from(value);
                }
            }
        }
        Tensor::from_vec(output, (rows, HIDDEN), &Device::Cpu)
    }
}

#[test]
#[ignore = "requires completed cuTile real-route replay and exclusive GPU access"]
fn flash_next_cutile_gguf_oracle() -> Result<()> {
    let directory = PathBuf::from(std::env::var(REPLAY_DIRECTORY_ENV).unwrap());
    let report_path = PathBuf::from(std::env::var(ORACLE_RESULTS_ENV).unwrap());
    let output = PathBuf::from(std::env::var(ORACLE_OUTPUT_ENV).unwrap());
    assert!(!output.exists(), "preserve prior oracle output");
    let report: Value =
        serde_json::from_reader(File::open(&report_path)?).map_err(candle_core::Error::wrap)?;
    let saved_outputs = PathBuf::from(report["output_directory"].as_str().unwrap());
    let device = Device::new_cuda(0)?;
    candle_core::cuda::set_gemm_reduced_precision_f32(false);
    let weights = Weights::gguf(&directory, &device)?;
    let fp32 = F32ExpertOracle::new(&weights, &device)?;
    let rounded = RoundedOracle::new(&weights, &device)?;
    let mut results = Vec::new();
    for sample in report["results"].as_array().unwrap() {
        let tensors = candle_core::safetensors::load(
            directory.join(sample["capture"]["tensor_file"].as_str().unwrap()),
            &device,
        )?;
        let source_rows = sample["source_row_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| u32::try_from(r.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>();
        let inputs = replay_inputs(&tensors, &Tensor::new(source_rows.as_slice(), &device)?)?;
        let outputs = candle_core::safetensors::load(
            saved_outputs.join(sample["saved_outputs"].as_str().unwrap()),
            &Device::Cpu,
        )?;
        let original_reference = fp32.forward(&inputs)?;
        let rounded_reference = rounded.forward(&inputs)?;
        let rounded_output = rounded_reference
            .to_dtype(DType::F32)?
            .to_dtype(DType::BF16)?;
        let record = json!({
            "source_metadata":sample["source_metadata"], "selection":sample["selection"],
            "source_row_indices":source_rows, "rows":sample["rows"], "derived":sample["derived"],
            "original_reference_vs_baseline":comparison_metrics(&original_reference, &outputs["baseline"])?,
            "original_reference_vs_cutile":comparison_metrics(&original_reference, &outputs["cutile"])?,
            "rounded_reference_vs_cutile":comparison_metrics(&rounded_reference, &outputs["cutile"])?,
            "rounded_reference_bf16_output_vs_cutile":comparison_metrics(&rounded_output, &outputs["cutile"])?,
        });
        eprintln!("CUTILE_GGUF_ORACLE {record}");
        results.push(record);
    }
    serde_json::to_writer_pretty(File::create(output)?, &json!({
        "replay_results":report_path,"results":results,"reduced_f32":false,
        "original_reference":"Original quantized weights dequantized to FP32; unquantized FP32 FFN with CPU F64 routing-weight reduction.",
        "rounded_reference":"Decoded weights rounded to BF16 then represented as FP32; BF16-rounded intermediate activation; FP32 matmuls with TF32 disabled and CPU F64 weighted reduction. Reports both unrounded F64 output and output cast through FP32 to BF16; differences still include accumulation and reduction ordering. Distinguishes the cuTile numerical contract from original-weight rounding.",
        "scope":"Separate reference pass after timing. Expanded weights exist only in the reference implementation, not in the cuTile kernel or timed forward. Projection tests independently validate decoding against CPU dequantization. These metrics are not a language-model quality evaluation.",
    })).map_err(candle_core::Error::wrap)?;
    Ok(())
}
