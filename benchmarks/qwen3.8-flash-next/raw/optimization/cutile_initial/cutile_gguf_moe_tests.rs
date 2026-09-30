#![cfg(all(feature = "cuda", feature = "cutile"))]

use candle_core::{
    cuda::cudarc::driver::{sys, CudaSlice},
    quantized::{GgmlDType, QTensor},
    DType, Device, Result, Storage, Tensor,
};
use mistralrs_quant::cutile::gguf_moe::{gguf_moe_projection, GgufMoeConfig, GgufMoeProjection};

const EXPERTS: usize = 8;
const OUTPUT_COLUMNS: usize = 67;
const ROUTED_ROWS: usize = 19;
const ROUTED_TOPK: usize = 2;
const Q4K_COLUMNS: usize = 512;
const Q4_1_COLUMNS: usize = 96;
const PROJECTION_RELATIVE_RMS: f64 = 2e-5;
const PROJECTION_MAX_ERROR: f64 = 2e-3;
const IDENTITY_MAX_ERROR: f64 = 2e-6;
const GRAPH_INPUTS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

struct Projection {
    gpu: QTensor,
    original: Vec<f32>,
    rounded: Vec<f32>,
    columns: usize,
}

impl Projection {
    fn new(dtype: GgmlDType, columns: usize, device: &Device) -> Result<Self> {
        let values = (0..EXPERTS * OUTPUT_COLUMNS * columns)
            .map(|i| {
                let group = (i / 32) % 8;
                let scale = (group + 1) as f32 * 0.027;
                (((i * 71 + i / columns * 13) % 257) as f32 - 130.0) * scale
                    + (group as f32 - 3.0) * 0.17
            })
            .collect::<Vec<_>>();
        let source = Tensor::from_vec(values, (EXPERTS, OUTPUT_COLUMNS, columns), &Device::Cpu)?;
        let cpu = QTensor::quantize(&source, dtype)?;
        let gpu = QTensor::quantize_onto(&source, dtype, device)?;
        assert_eq!(cpu.data()?.as_ref(), gpu.data()?.as_ref());
        let original = cpu
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rounded = original
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        Ok(Self {
            gpu,
            original,
            rounded,
            columns,
        })
    }

    fn forward(&self, input: &Tensor, ids: &Tensor, cfg: GgufMoeConfig) -> Result<Tensor> {
        let (rows, topk) = ids.dims2()?;
        let dev = input.device().as_cuda_device()?;
        let (storage, _) = ids.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            unreachable!()
        };
        let (sorted, experts, padded, capacity) = mistralrs_quant::moe::cuda::moe_align(
            storage.as_cuda_slice::<u32>()?,
            rows,
            EXPERTS,
            topk,
            cfg.bm,
            dev,
        )?;
        gguf_moe_projection(GgufMoeProjection {
            input,
            weight: &self.gpu,
            sorted_token_ids: &sorted,
            expert_ids: &experts,
            num_tokens_post_pad: &padded,
            padded_capacity: capacity,
            assignments: rows * topk,
            top_k: topk,
            config: cfg,
        })
    }

    fn reference(&self, input: &Tensor, ids: &Tensor, rounded: bool) -> Result<Vec<f32>> {
        let (rows, topk) = ids.dims2()?;
        let ids = ids
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let input = input
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let weights = if rounded {
            &self.rounded
        } else {
            &self.original
        };
        let mut output = vec![0.0; rows * topk * OUTPUT_COLUMNS];
        for assignment in 0..rows * topk {
            let expert = ids[assignment] as usize;
            for out in 0..OUTPUT_COLUMNS {
                let base = (expert * OUTPUT_COLUMNS + out) * self.columns;
                let mut sum = 0.0f64;
                for k in 0..self.columns {
                    sum += f64::from(input[assignment / topk * self.columns + k])
                        * f64::from(weights[base + k]);
                }
                output[assignment * OUTPUT_COLUMNS + out] = sum as f32;
            }
        }
        Ok(output)
    }
}

fn compare(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut error = 0.0f64;
    let mut norm = 0.0f64;
    let mut max_error = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        let delta = f64::from(actual) - f64::from(expected);
        error += delta * delta;
        norm += f64::from(expected).powi(2);
        max_error = max_error.max(delta.abs());
    }
    ((error / norm.max(f64::MIN_POSITIVE)).sqrt(), max_error)
}

fn assert_projection(actual: &Tensor, expected: &[f32]) -> Result<()> {
    let actual = actual
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (rms, maximum) = compare(&actual, expected);
    assert!(
        rms < PROJECTION_RELATIVE_RMS && maximum < PROJECTION_MAX_ERROR,
        "relative RMS {rms}, max error {maximum}"
    );
    Ok(())
}

fn routed_input(columns: usize, variant: usize, device: &Device) -> Result<Tensor> {
    let input = (0..ROUTED_ROWS * columns)
        .map(|i| (((i * 17 + variant * 31) % 101) as f32 - 50.0) / 81.0)
        .collect::<Vec<_>>();
    Tensor::from_vec(input, (ROUTED_ROWS, columns), device)?.to_dtype(DType::BF16)
}

fn routed_ids(variant: usize, device: &Device) -> Result<Tensor> {
    let ids = (0..ROUTED_ROWS)
        .flat_map(|row| {
            if variant == 0 {
                [0u32, 7]
            } else {
                [1, 2 + (row % 5) as u32]
            }
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(ids, (ROUTED_ROWS, ROUTED_TOPK), device)
}

#[test]
fn cutile_gguf_decodes_every_nibble_and_scale_group() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (dtype, columns) in [
        (GgmlDType::Q4K, Q4K_COLUMNS),
        (GgmlDType::Q4_1, Q4_1_COLUMNS),
    ] {
        let projection = Projection::new(dtype, columns, &device)?;
        let mut eye = vec![0.0f32; columns * columns];
        for k in 0..columns {
            eye[k * columns + k] = 1.0;
        }
        let input = Tensor::from_vec(eye, (columns, columns), &device)?.to_dtype(DType::BF16)?;
        let ids = Tensor::from_vec(vec![(EXPERTS - 1) as u32; columns], (columns, 1), &device)?;
        let expected = projection.reference(&input, &ids, true)?;
        let actual = projection
            .forward(&input, &ids, GgufMoeConfig::default())?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let (_, maximum) = compare(&actual, &expected);
        assert!(
            maximum < IDENTITY_MAX_ERROR,
            "{dtype:?} decoding max error {maximum}"
        );
    }
    Ok(())
}

#[test]
fn cutile_gguf_projection_matches_rounded_weight_oracle() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (dtype, columns) in [
        (GgmlDType::Q4K, Q4K_COLUMNS),
        (GgmlDType::Q4_1, Q4_1_COLUMNS),
    ] {
        let projection = Projection::new(dtype, columns, &device)?;
        let input = routed_input(columns, 0, &device)?;
        let ids = routed_ids(0, &device)?;
        let rounded = projection.reference(&input, &ids, true)?;
        let original = projection.reference(&input, &ids, false)?;
        let (rounding_rms, rounding_maximum) = compare(&rounded, &original);
        eprintln!("{dtype:?} BF16 weight rounding alone: relative RMS {rounding_rms}, max {rounding_maximum}");
        for bm in [8, 16, 32] {
            let cfg = GgufMoeConfig {
                bm,
                ..GgufMoeConfig::default()
            };
            assert_projection(&projection.forward(&input, &ids, cfg)?, &rounded)?;
        }
    }
    Ok(())
}

#[test]
fn cutile_gguf_projection_graph_replays_changed_routes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;
    let stream = dev.cuda_stream();
    let _htod_cache = dev.enable_cuda_graph_htod_cache();
    for (dtype, columns) in [
        (GgmlDType::Q4K, Q4K_COLUMNS),
        (GgmlDType::Q4_1, Q4_1_COLUMNS),
    ] {
        let projection = Projection::new(dtype, columns, &device)?;
        let inputs = [
            routed_input(columns, 0, &device)?,
            routed_input(columns, 1, &device)?,
        ];
        let routes = [routed_ids(0, &device)?, routed_ids(1, &device)?];
        let expected = GRAPH_INPUTS
            .iter()
            .map(|&(x, r)| projection.reference(&inputs[x], &routes[r], true))
            .collect::<Result<Vec<_>>>()?;
        let input = Tensor::zeros((ROUTED_ROWS, columns), DType::BF16, &device)?;
        let ids = Tensor::zeros((ROUTED_ROWS, ROUTED_TOPK), DType::U32, &device)?;
        input.slice_set(&inputs[0], 0, 0)?;
        ids.slice_set(&routes[0], 0, 0)?;
        let cfg = GgufMoeConfig::default();
        drop(projection.forward(&input, &ids, cfg)?);
        device.synchronize()?;
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
        let output = projection.forward(&input, &ids, cfg);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("cuTile GGUF capture returned no graph"))?;
        for (&(x, r), expected) in GRAPH_INPUTS
            .iter()
            .zip(expected.iter())
            .cycle()
            .take(GRAPH_INPUTS.len() * 2)
        {
            input.slice_set(&inputs[x], 0, 0)?;
            ids.slice_set(&routes[r], 0, 0)?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            device.synchronize()?;
            assert_projection(&output, expected)?;
        }
        drop(output);
        device.synchronize()?;
        drop(graph);
        device.synchronize()?;
    }
    Ok(())
}

#[test]
fn cutile_gguf_nonlocal_expert_writes_zero() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;
    let cfg = GgufMoeConfig::default();
    let projection = Projection::new(GgmlDType::Q4K, Q4K_COLUMNS, &device)?;
    let input = Tensor::ones((1, Q4K_COLUMNS), DType::BF16, &device)?;
    let upload = |values: &[i32]| -> Result<CudaSlice<i32>> {
        let mut slice = unsafe { dev.alloc::<i32>(values.len())? };
        dev.memcpy_htod(values, &mut slice)?;
        Ok(slice)
    };
    let mut sorted = vec![1i32; cfg.bm as usize];
    sorted[0] = 0;
    let sorted = upload(&sorted)?;
    let expert = upload(&[-1])?;
    let padded = upload(&[cfg.bm])?;
    let output = gguf_moe_projection(GgufMoeProjection {
        input: &input,
        weight: &projection.gpu,
        sorted_token_ids: &sorted,
        expert_ids: &expert,
        num_tokens_post_pad: &padded,
        padded_capacity: cfg.bm as usize,
        assignments: 1,
        top_k: 1,
        config: cfg,
    })?;
    assert_eq!(output.abs()?.max_all()?.to_scalar::<f32>()?, 0.0);
    Ok(())
}
