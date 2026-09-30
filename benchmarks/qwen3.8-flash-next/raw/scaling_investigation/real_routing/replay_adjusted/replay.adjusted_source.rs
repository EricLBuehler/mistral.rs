#![cfg(feature = "cuda")]

use std::{fs::File, path::Path, time::Instant};

use candle_core::{
    cuda::cudarc::{driver::sys::CUevent_flags, driver::DevicePtr},
    quantized::{GgmlDType, QTensor},
    safetensors::MmapedSafetensors,
    DType, Device, Result, Storage, Tensor,
};
use mistralrs_quant::{
    grouped_moe_mmq, grouped_moe_mmq_from_glu_packed, grouped_moe_mmq_from_glu_pair,
    grouped_moe_mmq_pair_packed, indexed_moe_fused_decode, moe_dispatch_build,
    moe_weighted_reduce_flat_bf16, GluActivationType, ACT_SILU,
};
use serde_json::{json, Value};

const EXPERTS: usize = 512;
const TOPK: usize = 10;
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 640;
const LAYER: usize = 8;
const ROW_COUNTS: [usize; 8] = [1, 8, 24, 32, 40, 48, 56, 64];
const WARMUPS: usize = 3;
const ROUNDS: usize = 7;
const REPETITIONS: usize = 10;
const CORRELATED_ROWS: usize = 7;
const MAX_RELATIVE_RMS: f64 = 0.03;
const MIN_COSINE: f64 = 0.999;
const TIGHT_COLUMN_BOUND: usize = 8;
const MAX_BOUND_RELATIVE_RMS: f64 = 1e-3;
const MIN_BOUND_COSINE: f64 = 0.999999;
const REPLAY_DIRECTORY_ENV: &str = "MISTRALRS_MOE_REPLAY_DIR";
const REPLAY_DERIVED_ROWS: [usize; 6] = [1, 6, 8, 24, 32, 40];
const REPLAY_DERIVE_FROM_ROWS: usize = 56;
const REPLAY_SEQUENCE_ROWS: [usize; 3] = [1, 6, 8];
const REPLAY_QUERY_LEN: usize = 7;
const REPLAY_BATCH: usize = 8;
const ORACLE_RESULTS_ENV: &str = "MISTRALRS_MOE_REPLAY_RESULTS";
const ORACLE_OUTPUT_ENV: &str = "MISTRALRS_MOE_ORACLE_OUTPUT";

struct Weights {
    gate: QTensor,
    up: QTensor,
    down: QTensor,
}

impl Weights {
    fn gguf(directory: &Path, device: &Device) -> Result<Self> {
        let paths = std::fs::read_dir(directory)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|path| {
                path.extension().and_then(|extension| extension.to_str()) == Some("gguf")
            })
            .collect::<Vec<_>>();
        let archive = mistralrs_quant::GgufArchive::open(paths)?;
        let mut weights = Vec::new();
        for projection in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let name = format!("blk.{LAYER}.{projection}.weight");
            eprintln!("loading {name}");
            weights.push(archive.load_qtensor(&name, device)?);
        }
        let down = weights.pop().unwrap();
        let up = weights.pop().unwrap();
        let gate = weights.pop().unwrap();
        Ok(Self { gate, up, down })
    }

    fn isq(directory: &Path, device: &Device) -> Result<Self> {
        let index: Value =
            serde_json::from_reader(File::open(directory.join("model.safetensors.index.json"))?)
                .map_err(candle_core::Error::wrap)?;
        let prefix = format!("model.language_model.layers.{LAYER}.mlp.experts");
        let load = |suffix: &str| -> Result<Tensor> {
            let name = format!("{prefix}.{suffix}");
            let shard = index["weight_map"][&name].as_str().unwrap();
            eprintln!("loading {name} from {shard}");
            let mapped = unsafe { MmapedSafetensors::new(directory.join(shard))? };
            mapped.load(&name, &Device::Cpu)
        };
        let gate_up = load("gate_up_proj")?;
        assert_eq!(gate_up.dims(), &[EXPERTS, INTERMEDIATE * 2, HIDDEN]);
        let gate = QTensor::quantize_onto(
            &gate_up.narrow(1, 0, INTERMEDIATE)?.contiguous()?,
            GgmlDType::Q4K,
            device,
        )?;
        let up = QTensor::quantize_onto(
            &gate_up
                .narrow(1, INTERMEDIATE, INTERMEDIATE)?
                .contiguous()?,
            GgmlDType::Q4K,
            device,
        )?;
        drop(gate_up);
        let down = QTensor::quantize_onto(&load("down_proj")?, GgmlDType::Q4_1, device)?;
        Ok(Self { gate, up, down })
    }
}

struct Inputs {
    xs: Tensor,
    ids: Tensor,
    weights: Tensor,
    active_experts: usize,
    max_expert_rows: usize,
}

impl Inputs {
    fn new(rows: usize, correlated: bool, device: &Device) -> Result<Self> {
        let xs = (0..rows * HIDDEN)
            .map(|i| {
                let value = u16::try_from((i * 37 + i / HIDDEN * 73) % 211).unwrap();
                f32::from(value) / 60.0 - 1.75
            })
            .collect::<Vec<_>>();
        let mut counts = [0usize; EXPERTS];
        let mut ids = Vec::new();
        for row in 0..rows {
            let group = if correlated {
                row / CORRELATED_ROWS
            } else {
                row
            };
            for rank in 0..TOPK {
                let expert = (group * 131 + rank * 53 + 17) % EXPERTS;
                counts[expert] += 1;
                ids.push(u32::try_from(expert).unwrap());
            }
        }
        let weights = (0..rows * TOPK)
            .map(|i| f32::from(u16::try_from(TOPK - i % TOPK).unwrap()) / 55.0)
            .collect::<Vec<_>>();
        Ok(Self {
            xs: Tensor::from_vec(xs, (rows, HIDDEN), device)?.to_dtype(DType::BF16)?,
            ids: Tensor::from_vec(ids, (rows, TOPK), device)?,
            weights: Tensor::from_vec(weights, (rows, TOPK), device)?,
            active_experts: counts.iter().filter(|&&n| n > 0).count(),
            max_expert_rows: *counts.iter().max().unwrap(),
        })
    }

    fn forward(&self, weights: &Weights, grouped: bool) -> Result<Tensor> {
        let rows = self.xs.dim(0)?;
        let dev = self.xs.device().as_cuda_device()?;
        let (ids_storage, _) = self.ids.storage_and_layout();
        let Storage::Cuda(ids_cuda) = &*ids_storage else {
            unreachable!()
        };
        let ids = ids_cuda.as_cuda_slice::<u32>()?;
        let (weights_storage, _) = self.weights.storage_and_layout();
        let Storage::Cuda(weights_cuda) = &*weights_storage else {
            unreachable!()
        };
        let weights_slice = weights_cuda.as_cuda_slice::<f32>()?;
        let (weights_pointer, _guard) = weights_slice.device_ptr(weights_slice.stream());
        let weights_pointer = weights_pointer as *const f32;
        if !grouped {
            return unsafe {
                indexed_moe_fused_decode(
                    &weights.gate,
                    &weights.up,
                    &weights.down,
                    &self.xs,
                    ids,
                    weights_pointer,
                    rows,
                    TOPK,
                    ACT_SILU,
                    dev,
                )
            }?
            .to_dtype(DType::BF16);
        }
        let assignments = rows * TOPK;
        let (bounds, sorted_ids, sorted_sources) =
            moe_dispatch_build(ids, assignments, EXPERTS, TOPK, dev)?;
        let gate_up = grouped_moe_mmq_pair_packed(
            &weights.gate,
            &weights.up,
            &self.xs,
            &sorted_sources,
            &sorted_ids,
            &bounds,
            assignments,
            TOPK,
            EXPERTS,
            dev,
        )?;
        let down = grouped_moe_mmq_from_glu_packed(
            &weights.down,
            &gate_up,
            &sorted_ids,
            &sorted_ids,
            &bounds,
            assignments,
            rows,
            EXPERTS,
            GluActivationType::Silu as i32,
            dev,
        )?;
        unsafe { moe_weighted_reduce_flat_bf16(&down, weights_pointer, rows, TOPK, dev) }
    }

    fn validate_tight_bound(&self) -> Result<()> {
        let dev = self.xs.device().as_cuda_device()?;
        let (storage, _) = self.ids.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            unreachable!()
        };
        let ids = cuda.as_cuda_slice::<u32>()?;
        let assignments = self.xs.dim(0)? * TOPK;
        let (bounds, _, _) = moe_dispatch_build(ids, assignments, EXPERTS, TOPK, dev)?;
        let bounds = dev.clone_dtoh(&bounds)?;
        assert_eq!(bounds[0], 0);
        assert_eq!(bounds[EXPERTS], u32::try_from(assignments)?);
        let counts = bounds[..=EXPERTS]
            .windows(2)
            .map(|pair| {
                assert!(pair[1] >= pair[0]);
                usize::try_from(pair[1] - pair[0]).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(*counts.iter().max().unwrap(), self.max_expert_rows);
        assert_eq!(
            counts.iter().filter(|&&n| n > 0).count(),
            self.active_experts
        );
        assert!(self.max_expert_rows <= TIGHT_COLUMN_BOUND);
        Ok(())
    }

    fn projection_forward(&self, weights: &Weights, column_bound: usize) -> Result<Tensor> {
        assert!(self.max_expert_rows <= column_bound);
        let rows = self.xs.dim(0)?;
        let assignments = rows * TOPK;
        let dev = self.xs.device().as_cuda_device()?;
        let (ids_storage, _) = self.ids.storage_and_layout();
        let Storage::Cuda(ids_cuda) = &*ids_storage else {
            unreachable!()
        };
        let ids = ids_cuda.as_cuda_slice::<u32>()?;
        let (weights_storage, _) = self.weights.storage_and_layout();
        let Storage::Cuda(weights_cuda) = &*weights_storage else {
            unreachable!()
        };
        let weights_slice = weights_cuda.as_cuda_slice::<f32>()?;
        let (weights_pointer, _guard) = weights_slice.device_ptr(weights_slice.stream());
        let (bounds, sorted_ids, sorted_sources) =
            moe_dispatch_build(ids, assignments, EXPERTS, TOPK, dev)?;
        let gate = grouped_moe_mmq(
            &weights.gate,
            &self.xs,
            &sorted_sources,
            &sorted_ids,
            &bounds,
            assignments,
            column_bound,
            EXPERTS,
            dev,
        )?;
        let up = grouped_moe_mmq(
            &weights.up,
            &self.xs,
            &sorted_sources,
            &sorted_ids,
            &bounds,
            assignments,
            column_bound,
            EXPERTS,
            dev,
        )?;
        let down = grouped_moe_mmq_from_glu_pair(
            &weights.down,
            &gate,
            &up,
            &sorted_ids,
            &sorted_ids,
            &bounds,
            assignments,
            column_bound,
            EXPERTS,
            GluActivationType::Silu as i32,
            dev,
        )?;
        unsafe {
            moe_weighted_reduce_flat_bf16(&down, weights_pointer as *const f32, rows, TOPK, dev)
        }
    }
}

fn comparison_metrics(a: &Tensor, b: &Tensor) -> Result<Value> {
    assert_eq!(a.dims(), b.dims());
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let (mut aa, mut bb, mut ab, mut diff, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in a.iter().zip(&b) {
        assert!(a.is_finite() && b.is_finite());
        let (a, b) = (f64::from(a), f64::from(b));
        aa += a * a;
        bb += b * b;
        ab += a * b;
        diff += (a - b).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let relative_rms = (diff / aa).sqrt();
    let cosine = ab / (aa * bb).sqrt();
    assert!(relative_rms.is_finite() && cosine.is_finite());
    Ok(json!({"relative_rms": relative_rms, "cosine": cosine, "max_abs": max_abs}))
}

fn cross_kernel_guard_passed(metrics: &Value) -> bool {
    metrics["relative_rms"].as_f64().unwrap() < MAX_RELATIVE_RMS
        && metrics["cosine"].as_f64().unwrap() > MIN_COSINE
}

fn compare(a: &Tensor, b: &Tensor) -> Result<Value> {
    let metrics = comparison_metrics(a, b)?;
    assert!(
        cross_kernel_guard_passed(&metrics),
        "cross-kernel difference: {metrics}"
    );
    Ok(metrics)
}

fn measure(inputs: &Inputs, mut forward: impl FnMut() -> Result<Tensor>) -> Result<Value> {
    let dev = inputs.xs.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    stream.synchronize().map_err(candle_core::Error::wrap)?;
    let start = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    let host_start = Instant::now();
    for _ in 0..REPETITIONS {
        std::hint::black_box(forward()?);
    }
    let end = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    end.synchronize().map_err(candle_core::Error::wrap)?;
    let denominator = f64::from(u32::try_from(REPETITIONS).unwrap());
    let host_ms = host_start.elapsed().as_secs_f64() * 1000.0 / denominator;
    let gpu_ms = f64::from(start.elapsed_ms(&end).map_err(candle_core::Error::wrap)?) / denominator;
    Ok(json!({"stream_elapsed_ms_per_layer": gpu_ms, "host_elapsed_ms_per_layer": host_ms}))
}

#[test]
#[ignore = "requires local Flash-Next weights and exclusive GPU access"]
fn flash_next_expert_dispatch_microbenchmark() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut results = Vec::new();
    for (scheme, variable) in [
        ("gguf", "MISTRALRS_MOE_BENCH_GGUF"),
        ("isq", "MISTRALRS_MOE_BENCH_SAFETENSORS"),
    ] {
        let Ok(directory) = std::env::var(variable) else {
            continue;
        };
        let weights = if scheme == "gguf" {
            Weights::gguf(Path::new(&directory), &device)?
        } else {
            Weights::isq(Path::new(&directory), &device)?
        };
        assert_eq!(
            weights.gate.shape().dims(),
            &[EXPERTS, INTERMEDIATE, HIDDEN]
        );
        assert_eq!(weights.up.shape().dims(), &[EXPERTS, INTERMEDIATE, HIDDEN]);
        assert_eq!(
            weights.down.shape().dims(),
            &[EXPERTS, HIDDEN, INTERMEDIATE]
        );
        eprintln!(
            "weights ready {scheme}: {:?}/{:?}/{:?}",
            weights.gate.dtype(),
            weights.up.dtype(),
            weights.down.dtype()
        );
        for correlated in [false, true] {
            for rows in ROW_COUNTS {
                let inputs = Inputs::new(rows, correlated, &device)?;
                let numerical = compare(
                    &inputs.forward(&weights, false)?,
                    &inputs.forward(&weights, true)?,
                )?;
                for _ in 0..WARMUPS {
                    inputs.forward(&weights, false)?;
                    inputs.forward(&weights, true)?;
                }
                let mut gemv = Vec::new();
                let mut grouped = Vec::new();
                for round in 0..ROUNDS {
                    for selected in [round % 2 == 0, round % 2 != 0] {
                        let timing = measure(&inputs, || inputs.forward(&weights, selected))?;
                        if selected {
                            grouped.push(timing)
                        } else {
                            gemv.push(timing)
                        }
                    }
                }
                let result = json!({"scheme": scheme, "source_directory": directory,
                    "gate_dtype": format!("{:?}", weights.gate.dtype()), "down_dtype": format!("{:?}", weights.down.dtype()),
                    "rows": rows, "routing": if correlated { "groups_of_7" } else { "independent" },
                    "active_experts": inputs.active_experts, "max_expert_rows": inputs.max_expert_rows,
                    "numerical": numerical, "gemv": gemv, "grouped": grouped});
                eprintln!("RESULT {result}");
                results.push(result);
            }
        }
    }
    assert!(
        !results.is_empty(),
        "set a MISTRALRS_MOE_BENCH_* weight directory"
    );
    let output =
        std::env::var("MISTRALRS_MOE_BENCH_OUTPUT").expect("set MISTRALRS_MOE_BENCH_OUTPUT");
    let report = json!({"layer": LAYER, "experts": EXPERTS, "topk": TOPK, "hidden": HIDDEN,
        "intermediate": INTERMEDIATE, "warmups": WARMUPS, "rounds": ROUNDS, "repetitions": REPETITIONS,
        "timing_scope": "eager full expert pipeline, warmed repeated layer; stream elapsed includes launch gaps; synthetic activations and routes, real weights",
        "results": results});
    serde_json::to_writer_pretty(File::create(output)?, &report)
        .map_err(candle_core::Error::wrap)?;
    Ok(())
}

#[test]
#[ignore = "requires local Flash-Next weights and exclusive GPU access"]
fn flash_next_known_occupancy_microbenchmark() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut results = Vec::new();
    for (scheme, variable) in [
        ("gguf", "MISTRALRS_MOE_BENCH_GGUF"),
        ("isq", "MISTRALRS_MOE_BENCH_SAFETENSORS"),
    ] {
        let Ok(directory) = std::env::var(variable) else {
            continue;
        };
        let weights = if scheme == "gguf" {
            Weights::gguf(Path::new(&directory), &device)?
        } else {
            Weights::isq(Path::new(&directory), &device)?
        };
        assert_eq!(
            weights.gate.shape().dims(),
            &[EXPERTS, INTERMEDIATE, HIDDEN]
        );
        assert_eq!(weights.up.shape().dims(), &[EXPERTS, INTERMEDIATE, HIDDEN]);
        assert_eq!(
            weights.down.shape().dims(),
            &[EXPERTS, HIDDEN, INTERMEDIATE]
        );
        eprintln!(
            "weights ready {scheme}: {:?}/{:?}/{:?}",
            weights.gate.dtype(),
            weights.up.dtype(),
            weights.down.dtype()
        );
        for correlated in [false, true] {
            for rows in ROW_COUNTS {
                let inputs = Inputs::new(rows, correlated, &device)?;
                inputs.validate_tight_bound()?;
                let normal = inputs.projection_forward(&weights, rows)?;
                let tight = inputs.projection_forward(&weights, TIGHT_COLUMN_BOUND)?;
                let numerical = compare(&normal, &tight)?;
                assert!(
                    numerical["relative_rms"].as_f64().unwrap() < MAX_BOUND_RELATIVE_RMS,
                    "bound changes output: {numerical}"
                );
                assert!(
                    numerical["cosine"].as_f64().unwrap() > MIN_BOUND_COSINE,
                    "bound changes output: {numerical}"
                );
                let packed_reference = compare(&normal, &inputs.forward(&weights, true)?)?;
                for _ in 0..WARMUPS {
                    inputs.projection_forward(&weights, rows)?;
                    inputs.projection_forward(&weights, TIGHT_COLUMN_BOUND)?;
                }
                let mut normal_times = Vec::new();
                let mut tight_times = Vec::new();
                for round in 0..ROUNDS {
                    for selected in [round % 2 == 0, round % 2 != 0] {
                        let bound = if selected { TIGHT_COLUMN_BOUND } else { rows };
                        let timing =
                            measure(&inputs, || inputs.projection_forward(&weights, bound))?;
                        if selected {
                            tight_times.push(timing)
                        } else {
                            normal_times.push(timing)
                        }
                    }
                }
                let result = json!({"scheme": scheme, "source_directory": directory,
                    "gate_dtype": format!("{:?}", weights.gate.dtype()), "down_dtype": format!("{:?}", weights.down.dtype()),
                    "rows": rows, "routing": if correlated { "groups_of_7" } else { "independent" },
                    "active_experts": inputs.active_experts, "max_expert_rows": inputs.max_expert_rows,
                    "normal_column_bound": rows, "tight_column_bound": TIGHT_COLUMN_BOUND,
                    "numerical": numerical, "packed_reference": packed_reference,
                    "normal_bound": normal_times, "tight_bound": tight_times});
                eprintln!("RESULT {result}");
                results.push(result);
            }
        }
    }
    assert!(
        !results.is_empty(),
        "set a MISTRALRS_MOE_BENCH_* weight directory"
    );
    let output =
        std::env::var("MISTRALRS_MOE_BENCH_OUTPUT").expect("set MISTRALRS_MOE_BENCH_OUTPUT");
    let report = json!({"layer": LAYER, "experts": EXPERTS, "topk": TOPK, "hidden": HIDDEN,
        "intermediate": INTERMEDIATE, "warmups": WARMUPS, "rounds": ROUNDS, "repetitions": REPETITIONS,
        "timing_scope": "same eager unfused expert projection pipeline on both sides; synthetic routing GPU bounds verified <=8 before timing; normal ncols_max=token rows versus ideal known ncols_max=8; not a generally safe bound or packed production pipeline",
        "results": results});
    serde_json::to_writer_pretty(File::create(output)?, &report)
        .map_err(candle_core::Error::wrap)?;
    Ok(())
}

fn replay_inputs(
    tensors: &std::collections::HashMap<String, Tensor>,
    row_indices: &Tensor,
) -> Result<Inputs> {
    let rows = row_indices.elem_count();
    let xs = tensors["xs"].index_select(row_indices, 0)?.contiguous()?;
    let ids = tensors["ids"].index_select(row_indices, 0)?.contiguous()?;
    let weights = tensors["weights"]
        .index_select(row_indices, 0)?
        .contiguous()?;
    assert_eq!(xs.dims(), &[rows, HIDDEN]);
    assert_eq!(xs.dtype(), DType::BF16);
    assert_eq!(ids.dims(), &[rows, TOPK]);
    assert_eq!(ids.dtype(), DType::U32);
    assert_eq!(weights.dims(), &[rows, TOPK]);
    assert_eq!(weights.dtype(), DType::F32);
    let mut occupancy = [0usize; EXPERTS];
    for row in ids.to_vec2::<u32>()? {
        let mut distinct = std::collections::HashSet::new();
        for expert in row {
            let expert = usize::try_from(expert).unwrap();
            assert!(expert < EXPERTS && distinct.insert(expert));
            occupancy[expert] += 1;
        }
    }
    assert!(weights
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0));
    Ok(Inputs {
        xs,
        ids,
        weights,
        active_experts: occupancy.iter().filter(|&&count| count > 0).count(),
        max_expert_rows: *occupancy.iter().max().unwrap(),
    })
}

#[test]
#[ignore = "requires captured Flash-Next MoE operands and exclusive GPU access"]
fn flash_next_real_routing_replay() -> Result<()> {
    let output =
        std::env::var("MISTRALRS_MOE_BENCH_OUTPUT").expect("set MISTRALRS_MOE_BENCH_OUTPUT");
    assert!(
        !Path::new(&output).exists(),
        "preserve prior benchmark output"
    );
    let directory = std::env::var(REPLAY_DIRECTORY_ENV).expect("set MISTRALRS_MOE_REPLAY_DIR");
    let directory = Path::new(&directory);
    let device = Device::new_cuda(0)?;
    let weights = Weights::gguf(directory, &device)?;
    assert_eq!(
        weights.gate.shape().dims(),
        &[EXPERTS, INTERMEDIATE, HIDDEN]
    );
    assert_eq!(weights.up.shape().dims(), &[EXPERTS, INTERMEDIATE, HIDDEN]);
    assert_eq!(
        weights.down.shape().dims(),
        &[EXPERTS, HIDDEN, INTERMEDIATE]
    );
    let mut paths = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    let mut results = Vec::new();
    for path in paths {
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let metadata: Value =
            serde_json::from_reader(File::open(&path)?).map_err(candle_core::Error::wrap)?;
        let Some(tensor_file) = metadata["tensor_file"].as_str() else {
            continue;
        };
        assert_eq!(metadata["derived"], false);
        assert_eq!(metadata["layer"], LAYER);
        assert_eq!(
            metadata["gate_dtype"],
            format!("{:?}", weights.gate.dtype())
        );
        assert_eq!(metadata["up_dtype"], format!("{:?}", weights.up.dtype()));
        assert_eq!(
            metadata["down_dtype"],
            format!("{:?}", weights.down.dtype())
        );
        let tensors = candle_core::safetensors::load(directory.join(tensor_file), &device)?;
        let full_rows = tensors["xs"].dim(0)?;
        assert_eq!(
            metadata["rows"].as_u64().unwrap(),
            u64::try_from(full_rows).unwrap()
        );
        assert_eq!(tensors["output"].dtype(), DType::BF16);
        assert_eq!(tensors["output"].dims(), &[full_rows, HIDDEN]);
        assert_eq!(metadata["output_dtype"], "BF16");
        assert_eq!(metadata["output_shape"], json!([full_rows, HIDDEN]));
        let indices = |rows: usize| {
            (0..rows)
                .map(|row| u32::try_from(row).unwrap())
                .collect::<Vec<_>>()
        };
        let mut selections = vec![("native", indices(full_rows))];
        if full_rows == REPLAY_DERIVE_FROM_ROWS {
            for rows in REPLAY_DERIVED_ROWS {
                selections.push(("prefix", indices(rows)));
            }
            if metadata["batch"] == REPLAY_BATCH && metadata["query_len"] == REPLAY_QUERY_LEN {
                for rows in REPLAY_SEQUENCE_ROWS {
                    selections.push((
                        "sequence_query_zero",
                        (0..rows)
                            .map(|row| u32::try_from(row * REPLAY_QUERY_LEN).unwrap())
                            .collect(),
                    ));
                }
            }
        }
        for (selection, source_rows) in selections {
            let rows = source_rows.len();
            let row_indices = Tensor::new(source_rows.as_slice(), &device)?;
            let inputs = replay_inputs(&tensors, &row_indices)?;
            let captured = tensors["output"]
                .index_select(&row_indices, 0)?
                .contiguous()?;
            let gemv_output = inputs.forward(&weights, false)?;
            let grouped_output = inputs.forward(&weights, true)?;
            let numerical = comparison_metrics(&gemv_output, &grouped_output)?;
            let cross_guard_passed = cross_kernel_guard_passed(&numerical);
            let captured_vs_gemv = comparison_metrics(&captured, &gemv_output)?;
            let captured_vs_grouped = comparison_metrics(&captured, &grouped_output)?;
            let derived = selection != "native";
            if !derived {
                let native_comparison = match metadata["source_dispatch"].as_str().unwrap() {
                    "indexed_gemv" => &captured_vs_gemv,
                    "grouped_mmq" => &captured_vs_grouped,
                    other => panic!("unexpected source dispatch {other}"),
                };
                assert!(
                    native_comparison["relative_rms"].as_f64().unwrap() < MAX_BOUND_RELATIVE_RMS,
                    "source replay differs: {native_comparison}"
                );
                assert!(
                    native_comparison["cosine"].as_f64().unwrap() > MIN_BOUND_COSINE,
                    "source replay differs: {native_comparison}"
                );
            }
            for _ in 0..WARMUPS {
                inputs.forward(&weights, false)?;
                inputs.forward(&weights, true)?;
            }
            let mut gemv = Vec::new();
            let mut grouped = Vec::new();
            for round in 0..ROUNDS {
                for selected in [round % 2 == 0, round % 2 != 0] {
                    let timing = measure(&inputs, || inputs.forward(&weights, selected))?;
                    if selected {
                        grouped.push(timing);
                    } else {
                        gemv.push(timing);
                    }
                }
            }
            let mut occupancy = [0usize; EXPERTS];
            for expert in inputs.ids.flatten_all()?.to_vec1::<u32>()? {
                occupancy[usize::try_from(expert).unwrap()] += 1;
            }
            if !derived {
                assert_eq!(metadata["expert_occupancy"], json!(occupancy.to_vec()));
            }
            let result = json!({
                "source_metadata": path.file_name().unwrap().to_str().unwrap(),
                "capture": metadata, "rows": rows, "derived": derived,
                "selection": selection, "source_row_indices": source_rows,
                "derivation": match selection {
                    "prefix" => "first N flattened rows of the captured 56-row batch; not an independently observed batch",
                    "sequence_query_zero" => "one row at query position zero per sequence in native B8xQ7; not an observed decode batch",
                    _ => "none",
                },
                "active_experts": inputs.active_experts, "max_expert_rows": inputs.max_expert_rows,
                "expert_occupancy": occupancy.to_vec(),
                "numerical": numerical, "cross_kernel_guard_passed": cross_guard_passed,
                "native_replay_guard_passed": (!derived).then_some(true),
                "captured_vs_gemv": captured_vs_gemv,
                "captured_vs_grouped": captured_vs_grouped,
                "gemv": gemv, "grouped": grouped,
            });
            eprintln!("REAL_ROUTE_RESULT {result}");
            results.push(result);
        }
    }
    assert!(!results.is_empty(), "no captured samples found");
    let report = json!({
        "layer": LAYER, "experts": EXPERTS, "topk": TOPK, "hidden": HIDDEN,
        "intermediate": INTERMEDIATE, "warmups": WARMUPS, "rounds": ROUNDS,
        "repetitions": REPETITIONS,
        "cross_kernel_guard": {"relative_rms_max": MAX_RELATIVE_RMS, "cosine_min": MIN_COSINE},
        "cross_kernel_equivalence_asserted": false,
        "guard_failure_meaning": "Alternate kernel does not satisfy the existing numerical guard; timing alone does not establish it as a safe replacement.",
        "timing_scope": "eager full expert pipeline, warmed repeated layer; stream elapsed includes launch gaps; exact captured model inputs, routes and post-ISQ weights; derived prefix and per-sequence query-zero selections explicitly marked; not end-to-end throughput or natural autotuner occupancy",
        "results": results,
    });
    serde_json::to_writer_pretty(File::create(output)?, &report)
        .map_err(candle_core::Error::wrap)?;
    Ok(())
}

struct F32ExpertOracle {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

impl F32ExpertOracle {
    fn new(weights: &Weights, device: &Device) -> Result<Self> {
        assert!(!candle_core::cuda::gemm_reduced_precision_f32());
        Ok(Self {
            gate: weights.gate.dequantize(device)?.to_dtype(DType::F32)?,
            up: weights.up.dequantize(device)?.to_dtype(DType::F32)?,
            down: weights.down.dequantize(device)?.to_dtype(DType::F32)?,
        })
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
            let row_indices = assignments
                .iter()
                .map(|&(row, _)| u32::try_from(row).unwrap())
                .collect::<Vec<_>>();
            let row_indices = Tensor::from_vec(row_indices, assignments.len(), xs.device())?;
            let selected = xs.index_select(&row_indices, 0)?;
            let gate = selected.matmul(&self.gate.get(expert)?.t()?.contiguous()?)?;
            let up = selected.matmul(&self.up.get(expert)?.t()?.contiguous()?)?;
            let activated = (candle_nn::ops::silu(&gate)? * up)?;
            let expert_output = activated
                .matmul(&self.down.get(expert)?.t()?.contiguous()?)?
                .to_vec2::<f32>()?;
            for (&(row, slot), values) in assignments.iter().zip(&expert_output) {
                let weight = f64::from(weights[row][slot]);
                for (column, &value) in values.iter().enumerate() {
                    assert!(value.is_finite());
                    output[row * HIDDEN + column] += weight * f64::from(value);
                }
            }
        }
        assert!(output.iter().all(|value| value.is_finite()));
        Tensor::from_vec(output, (rows, HIDDEN), &Device::Cpu)
    }
}

#[test]
#[ignore = "requires captured operands, completed replay timings and exclusive GPU access"]
fn flash_next_real_routing_oracle() -> Result<()> {
    let directory = std::env::var(REPLAY_DIRECTORY_ENV).expect("set MISTRALRS_MOE_REPLAY_DIR");
    let replay_results =
        std::env::var(ORACLE_RESULTS_ENV).expect("set MISTRALRS_MOE_REPLAY_RESULTS");
    let output = std::env::var(ORACLE_OUTPUT_ENV).expect("set MISTRALRS_MOE_ORACLE_OUTPUT");
    assert!(!Path::new(&output).exists(), "preserve prior oracle output");
    let replay: Value =
        serde_json::from_reader(File::open(&replay_results)?).map_err(candle_core::Error::wrap)?;
    let directory = Path::new(&directory);
    let device = Device::new_cuda(0)?;
    let weights = Weights::gguf(directory, &device)?;
    let reduced_f32_before = candle_core::cuda::gemm_reduced_precision_f32();
    candle_core::cuda::set_gemm_reduced_precision_f32(false);
    let oracle = F32ExpertOracle::new(&weights, &device)?;
    let mut results = Vec::new();
    for sample in replay["results"].as_array().unwrap() {
        let flagged = !sample["cross_kernel_guard_passed"].as_bool().unwrap();
        let native = !sample["derived"].as_bool().unwrap();
        if !flagged && !native {
            continue;
        }
        let tensors = candle_core::safetensors::load(
            directory.join(sample["capture"]["tensor_file"].as_str().unwrap()),
            &device,
        )?;
        let source_rows = sample["source_row_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>();
        let row_indices = Tensor::new(source_rows.as_slice(), &device)?;
        let inputs = replay_inputs(&tensors, &row_indices)?;
        let reference = oracle.forward(&inputs)?;
        let gemv = inputs.forward(&weights, false)?;
        let grouped = inputs.forward(&weights, true)?;
        let captured = tensors["output"]
            .index_select(&row_indices, 0)?
            .contiguous()?;
        let record = json!({
            "source_metadata": sample["source_metadata"], "selection": sample["selection"],
            "rows": sample["rows"], "source_row_indices": source_rows,
            "derived": sample["derived"], "original_cross_kernel_guard_passed": !flagged,
            "reference_vs_gemv": comparison_metrics(&reference, &gemv)?,
            "reference_vs_grouped": comparison_metrics(&reference, &grouped)?,
            "reference_vs_captured": comparison_metrics(&reference, &captured)?,
        });
        eprintln!("FP32_ORACLE_RESULT {record}");
        results.push(record);
    }
    assert!(!results.is_empty(), "no oracle samples selected");
    let report = json!({
        "replay_results": replay_results,
        "scope": "All native captures and every cross-kernel-guard failure, evaluated separately after timings.",
        "reference": "Exact dumped QTensors dequantized to F32; F32 cuBLAS gate/up, SiLU product and down projection with reduced-F32/TF32 disabled; routing-weighted accumulation on CPU in F64, then F32 comparison.",
        "activation_quantization": "none in the oracle; both production paths retain their own activation quantization",
        "no_new_accuracy_tolerance_asserted": true,
        "reduced_f32_before": reduced_f32_before, "reduced_f32_for_oracle": false,
        "results": results,
    });
    serde_json::to_writer_pretty(File::create(output)?, &report)
        .map_err(candle_core::Error::wrap)?;
    Ok(())
}
