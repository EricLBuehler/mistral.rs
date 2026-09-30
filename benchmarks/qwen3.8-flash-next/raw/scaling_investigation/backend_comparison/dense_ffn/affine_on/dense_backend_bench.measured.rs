#![cfg(all(feature = "cuda", feature = "cutile"))]

use std::{fs::File, path::Path, sync::Arc, time::Instant};

use candle_core::{
    cuda::cudarc::driver::sys::{CUdevice_attribute, CUevent_flags},
    quantized::{GgmlDType, QTensor},
    safetensors::MmapedSafetensors,
    DType, Device, Result, Tensor,
};
use float8::F8E4M3;
use mistralrs_quant::{
    fused_split_glu, try_fused_quantized_ffn, try_fused_quantized_gate_up, BlockwiseFP8Linear,
    Fp8ActivationScheme, GgufMatMul, GluActivationType, QuantMethod, QuantMethodConfig,
};
use serde_json::{json, Value};

const LAYER: usize = 0;
const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 17408;
const BLOCK_SIZE: usize = 128;
const ROW_COUNTS: [usize; 7] = [1, 6, 8, 24, 32, 40, 56];
const WARMUPS: usize = 3;
const ROUNDS: usize = 7;
const REPETITIONS: usize = 10;
const MAX_REFERENCE_RELATIVE_RMS: f64 = 0.06;
const MIN_REFERENCE_COSINE: f64 = 0.995;
const FP8_MMA_MAX_ROWS: usize = 32;
const Q4K_MMVQ_MAX_ROWS: usize = 8;
const TARGET_COMPUTE_CAPABILITY: (i32, i32) = (12, 1);
const GGUF_BACKEND_ENV: &str = "MISTRALRS_GGUF_AFFINE_BACKEND";

struct CheckpointProjection {
    weight: Tensor,
    scales: Tensor,
}

impl CheckpointProjection {
    fn load(directory: &Path, index: &Value, projection: &str) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{LAYER}.mlp.{projection}");
        let load = |suffix: &str| -> Result<Tensor> {
            let name = format!("{prefix}.{suffix}");
            let shard = index["weight_map"][&name]
                .as_str()
                .expect("checkpoint tensor is indexed");
            eprintln!("loading {name} from {shard}");
            let mapped = unsafe { MmapedSafetensors::new(directory.join(shard))? };
            mapped.load(&name, &Device::Cpu)
        };
        let weight = load("weight")?;
        assert_eq!(weight.dtype(), DType::F8E4M3);
        let scales = load("weight_scale_inv")?.to_dtype(DType::F32)?;
        Ok(Self { weight, scales })
    }

    fn pack(gate: Self, up: Self) -> Result<Self> {
        assert_eq!(gate.weight.dims(), &[INTERMEDIATE, HIDDEN]);
        assert_eq!(gate.weight.dims(), up.weight.dims());
        let mut values = gate.weight.flatten_all()?.to_vec1::<F8E4M3>()?;
        values.extend(up.weight.flatten_all()?.to_vec1::<F8E4M3>()?);
        Ok(Self {
            weight: Tensor::from_vec(values, (INTERMEDIATE * 2, HIDDEN), &Device::Cpu)?,
            scales: Tensor::cat(&[&gate.scales, &up.scales], 0)?,
        })
    }

    fn into_layer(self, device: &Device) -> Result<BlockwiseFP8Linear> {
        BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
            weight: self.weight.to_device(device)?,
            weight_scale_inv: self.scales.to_device(device)?,
            bias: None,
            dequant_dtype: DType::BF16,
            weight_block_size: vec![BLOCK_SIZE, BLOCK_SIZE],
            activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        })
    }
}

enum GateUp {
    Packed(Box<dyn QuantMethod>),
    Split {
        gate: Box<dyn QuantMethod>,
        up: Box<dyn QuantMethod>,
    },
}

struct Pipeline {
    gate_up: GateUp,
    down: Box<dyn QuantMethod>,
}

impl Pipeline {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let packed = match &self.gate_up {
            GateUp::Packed(projection) => projection,
            GateUp::Split { gate, up } => {
                if let Some(output) = try_fused_quantized_ffn(
                    input,
                    &**gate,
                    &**up,
                    &*self.down,
                    GluActivationType::Silu,
                )? {
                    return Ok(output);
                }
                let intermediate =
                    try_fused_quantized_gate_up(input, &**gate, &**up, GluActivationType::Silu)?
                        .expect("Q4K gate/up has a CUDA fused path");
                return self.down.forward(&intermediate);
            }
        };
        let gate_up = packed.forward(input)?;
        if let Some(output) = self.down.try_forward_fused_split_glu(
            &gate_up,
            INTERMEDIATE,
            GluActivationType::Silu,
        )? {
            return Ok(output);
        }
        self.down.forward(&fused_split_glu(
            &gate_up,
            INTERMEDIATE,
            GluActivationType::Silu,
        )?)
    }

    fn q4k_from_common_weights(&self, device: &Device) -> Result<Self> {
        let quantize = |common: &Tensor| -> Result<Box<dyn QuantMethod>> {
            let weight = QTensor::quantize_onto(&common.contiguous()?, GgmlDType::Q4K, device)?;
            assert_eq!(weight.dtype(), GgmlDType::Q4K);
            Ok(Box::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(weight),
                b: None,
            })?))
        };
        let GateUp::Packed(projection) = &self.gate_up else {
            unreachable!("common source is the packed FP8 projection")
        };
        let common = projection.dequantize_w()?.to_device(&Device::Cpu)?;
        let gate = quantize(&common.narrow(0, 0, INTERMEDIATE)?)?;
        let up = quantize(&common.narrow(0, INTERMEDIATE, INTERMEDIATE)?)?;
        drop(common);
        let down = quantize(&self.down.dequantize_w()?.to_device(&Device::Cpu)?)?;
        Ok(Self {
            gate_up: GateUp::Split { gate, up },
            down,
        })
    }

    fn reference(&self) -> Result<Reference> {
        let gate_up = match &self.gate_up {
            GateUp::Packed(projection) => projection.dequantize_w()?.to_dtype(DType::F32)?,
            GateUp::Split { gate, up } => Tensor::cat(
                &[
                    &gate.dequantize_w()?.to_dtype(DType::F32)?,
                    &up.dequantize_w()?.to_dtype(DType::F32)?,
                ],
                0,
            )?,
        };
        Ok(Reference {
            gate_up: gate_up.t()?.contiguous()?,
            down: self
                .down
                .dequantize_w()?
                .to_dtype(DType::F32)?
                .t()?
                .contiguous()?,
        })
    }

    fn dispatch(&self, input: &Tensor, scheme: &str) -> Result<Value> {
        let describe = |method: &dyn QuantMethod, features: usize| -> Result<Value> {
            let rows = input.dim(0)?;
            let activation = Tensor::zeros((rows, features), DType::BF16, input.device())?;
            let affine = {
                #[cfg(has_marlin_kernels)]
                {
                    method.prepare_gguf_affine_raw(rows, DType::BF16, input.device())?
                }
                #[cfg(not(has_marlin_kernels))]
                {
                    false
                }
            };
            let expected_kernel_family = if scheme == "fp8" {
                assert!(method
                    .activation_quantization_scheme_for(&activation)
                    .is_some());
                if rows <= FP8_MMA_MAX_ROWS {
                    "fp8_mma_gemv_kernel"
                } else {
                    "fp8_blockwise_gemm (cuTile)"
                }
            } else if affine {
                "packed GGUF affine (Marlin)"
            } else if rows <= Q4K_MMVQ_MAX_ROWS {
                "mmvq_gguf Q4K"
            } else {
                "mul_mat_q Q4K (MMQ)"
            };
            Ok(json!({
                "method": method.name(),
                "activation_scheme": format!("{:?}", method.activation_quantization_scheme_for(&activation)),
                "activation_scale_layout": format!("{:?}", method.preferred_activation_scale_layout_for(&activation)),
                "packed_affine_prepared": affine,
                "expected_kernel_family_from_source_dispatch": expected_kernel_family,
            }))
        };
        let gate_up = match &self.gate_up {
            GateUp::Packed(projection) => json!({"packed": describe(&**projection, HIDDEN)?}),
            GateUp::Split { gate, up } => json!({
                "gate": describe(&**gate, HIDDEN)?, "up": describe(&**up, HIDDEN)?,
                "fusion": "try_fused_quantized_ffn, then fused gate/up fallback, as in the production unpacked MLP",
            }),
        };
        Ok(json!({
            "gate_up": gate_up,
            "down": describe(&*self.down, INTERMEDIATE)?,
            "source_dispatch_is_not_a_measured_kernel_trace": true,
        }))
    }
}

struct Reference {
    gate_up: Tensor,
    down: Tensor,
}

impl Reference {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate_up = input.to_dtype(DType::F32)?.matmul(&self.gate_up)?;
        let gate = gate_up.narrow(1, 0, INTERMEDIATE)?.contiguous()?;
        let up = gate_up
            .narrow(1, INTERMEDIATE, INTERMEDIATE)?
            .contiguous()?;
        (candle_nn::ops::silu(&gate)? * up)?.matmul(&self.down)
    }
}

fn input(rows: usize, device: &Device) -> Result<Tensor> {
    let values = (0..rows * HIDDEN)
        .map(|index| {
            let value = u16::try_from((index * 37 + index / HIDDEN * 73) % 211).unwrap();
            f32::from(value) / 60.0 - 1.75
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (rows, HIDDEN), device)?.to_dtype(DType::BF16)
}

fn compare(reference: &Tensor, output: &Tensor, check_reference: bool) -> Result<Value> {
    assert_eq!(reference.dims(), output.dims());
    let reference = reference
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let output = output
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (mut rr, mut yy, mut ry, mut difference, mut max_abs) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (&reference, &output) in reference.iter().zip(&output) {
        assert!(reference.is_finite() && output.is_finite());
        let (reference, output) = (f64::from(reference), f64::from(output));
        rr += reference * reference;
        yy += output * output;
        ry += reference * output;
        difference += (reference - output).powi(2);
        max_abs = max_abs.max((reference - output).abs());
    }
    assert!(rr > 0.0 && yy > 0.0);
    let relative_rms = (difference / rr).sqrt();
    let cosine = ry / (rr * yy).sqrt();
    if check_reference {
        assert!(
            relative_rms < MAX_REFERENCE_RELATIVE_RMS,
            "relative RMS {relative_rms}"
        );
        assert!(cosine > MIN_REFERENCE_COSINE, "cosine {cosine}");
    }
    Ok(json!({"relative_rms": relative_rms, "cosine": cosine, "max_abs": max_abs}))
}

fn measure(pipeline: &Pipeline, input: &Tensor) -> Result<Value> {
    let stream = input.device().as_cuda_device()?.cuda_stream();
    stream.synchronize().map_err(candle_core::Error::wrap)?;
    let start = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    let host_start = Instant::now();
    for _ in 0..REPETITIONS {
        std::hint::black_box(pipeline.forward(input)?);
    }
    let end = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    end.synchronize().map_err(candle_core::Error::wrap)?;
    let repetitions = f64::from(u32::try_from(REPETITIONS).unwrap());
    Ok(json!({
        "stream_elapsed_ms_per_layer": f64::from(start.elapsed_ms(&end).map_err(candle_core::Error::wrap)?) / repetitions,
        "host_elapsed_ms_per_layer": host_start.elapsed().as_secs_f64() * 1000.0 / repetitions,
    }))
}

#[test]
#[ignore = "requires local dense Qwen3.8-27B FP8 weights and exclusive GPU access"]
fn dense_qwen38_fp8_vs_q4k_microbenchmark() -> Result<()> {
    let directory = std::env::var("MISTRALRS_DENSE_BENCH_SAFETENSORS")
        .expect("set MISTRALRS_DENSE_BENCH_SAFETENSORS to the pinned FP8 snapshot");
    let output = std::env::var("MISTRALRS_DENSE_BENCH_OUTPUT")
        .expect("set MISTRALRS_DENSE_BENCH_OUTPUT to a new JSON result path");
    assert!(
        !Path::new(&output).exists(),
        "preserve prior benchmark output"
    );
    let directory = Path::new(&directory);
    let index: Value =
        serde_json::from_reader(File::open(directory.join("model.safetensors.index.json"))?)
            .map_err(candle_core::Error::wrap)?;
    let gate = CheckpointProjection::load(directory, &index, "gate_proj")?;
    let up = CheckpointProjection::load(directory, &index, "up_proj")?;
    let down = CheckpointProjection::load(directory, &index, "down_proj")?;
    assert_eq!(down.weight.dims(), &[HIDDEN, INTERMEDIATE]);
    let device = Device::new_cuda(0)?;
    let context = device.as_cuda_device()?.cuda_stream().context().clone();
    let major = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(candle_core::Error::wrap)?;
    let minor = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(candle_core::Error::wrap)?;
    assert_eq!(
        (major, minor),
        TARGET_COMPUTE_CAPABILITY,
        "this dispatch comparison targets GB10"
    );
    let fp8 = Pipeline {
        gate_up: GateUp::Packed(Box::new(
            CheckpointProjection::pack(gate, up)?.into_layer(&device)?,
        )),
        down: Box::new(down.into_layer(&device)?),
    };
    let q4k = fp8.q4k_from_common_weights(&device)?;
    let references = [fp8.reference()?, q4k.reference()?];
    let pipelines = [(&fp8, "fp8"), (&q4k, "q4k")];
    let mut results = Vec::new();
    for rows in ROW_COUNTS {
        let input = input(rows, &device)?;
        let mut numerical = Vec::new();
        let mut dispatch = Vec::new();
        for ((pipeline, scheme), reference) in pipelines.iter().zip(&references) {
            dispatch.push(pipeline.dispatch(&input, scheme)?);
            numerical.push(compare(
                &reference.forward(&input)?,
                &pipeline.forward(&input)?,
                true,
            )?);
        }
        let cross_backend = compare(&fp8.forward(&input)?, &q4k.forward(&input)?, false)?;
        for _ in 0..WARMUPS {
            fp8.forward(&input)?;
            q4k.forward(&input)?;
        }
        let mut timings = [Vec::new(), Vec::new()];
        for round in 0..ROUNDS {
            for selected in [round % 2, 1 - round % 2] {
                timings[selected].push(measure(pipelines[selected].0, &input)?);
            }
        }
        let result = json!({
            "rows": rows,
            "fp8": {"dispatch": dispatch[0], "own_dequantized_weight_reference": numerical[0], "timings": timings[0]},
            "q4k": {"dispatch": dispatch[1], "own_dequantized_weight_reference": numerical[1], "timings": timings[1]},
            "cross_backend_difference_not_an_equivalence_assertion": cross_backend,
        });
        eprintln!("{result}");
        results.push(result);
    }
    let artifact = json!({
        "source_directory": directory,
        "layer": LAYER,
        "dimensions": {"hidden": HIDDEN, "intermediate": INTERMEDIATE},
        "compute_capability": [major, minor],
        "warmups": WARMUPS, "rounds": ROUNDS, "repetitions": REPETITIONS,
        "reference_tolerance": {"relative_rms_max": MAX_REFERENCE_RELATIVE_RMS, "cosine_min": MIN_REFERENCE_COSINE},
        "gguf_affine_env": std::env::var(GGUF_BACKEND_ENV).ok(),
        "pipeline": "gate/up projections, SiLU product, down projection through production QuantMethod dispatch/fusion hooks; FP8 packs gate/up, Q4K uses separate weights with fused FFN/gate-up kernels as production ISQ does",
        "limitations": [
            "Q4K weights are quantized from dequantized FP8 checkpoint weights, not the original BF16 checkpoint. This double-quantized control is for backend timing, not model quality.",
            "Each backend is checked against FP32 matmul with its own dequantized weights; cross-backend equality is not expected.",
            "Activations are deterministic synthetic BF16 rows, not captured model activations.",
            "Timing starts from BF16 inputs and includes activation quantization; a full model may fuse that quantization with upstream normalization.",
            "Repeated isolated FFN weights have different cache behavior from a full model; timings do not predict end-to-end speed.",
            "CUDA-event intervals include host launch gaps. This benchmark does not use CUDA graphs.",
            "Recorded kernel families are inferred from checked dispatch branches, not an instruction-level profile.",
        ],
        "results": results,
    });
    serde_json::to_writer_pretty(File::create(output)?, &artifact)
        .map_err(candle_core::Error::wrap)?;
    Ok(())
}
