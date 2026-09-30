#![cfg(feature = "cuda")]

use std::{fs::File, path::Path, time::Instant};

use candle_core::{
    cuda::cudarc::{driver::sys::CUevent_flags, driver::DevicePtr},
    quantized::{GgmlDType, QTensor},
    safetensors::MmapedSafetensors,
    DType, Device, Result, Storage, Tensor,
};
use mistralrs_quant::{
    grouped_moe_mmq_from_glu_packed, grouped_moe_mmq_pair_packed, indexed_moe_fused_decode,
    moe_dispatch_build, moe_weighted_reduce_flat_bf16, GluActivationType,
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
                    1,
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
}

fn compare(a: &Tensor, b: &Tensor) -> Result<Value> {
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
    assert!(
        relative_rms < MAX_RELATIVE_RMS,
        "relative RMS {relative_rms}"
    );
    assert!(cosine > MIN_COSINE, "cosine {cosine}");
    Ok(json!({"relative_rms": relative_rms, "cosine": cosine, "max_abs": max_abs}))
}

fn measure(inputs: &Inputs, weights: &Weights, grouped: bool) -> Result<Value> {
    let dev = inputs.xs.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    stream.synchronize().map_err(candle_core::Error::wrap)?;
    let start = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    let host_start = Instant::now();
    for _ in 0..REPETITIONS {
        std::hint::black_box(inputs.forward(weights, grouped)?);
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
                        let timing = measure(&inputs, &weights, selected)?;
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
