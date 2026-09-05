use std::{env, hint::black_box, time::Instant};

use candle_core::{cuda::cudarc::driver::sys, DType, Device, Result, Tensor};
use float8::F8E4M3;
use mistralrs_quant::cutile::{cutile_nvfp4, cutile_nvfp4_gather, Nvfp4GemmArgs};

const DEFAULT_ITERATIONS: usize = 20;
const WARMUP_ITERATIONS: usize = 4;
const SAMPLES: usize = 5;
const BLOCK_SIZE: usize = 16;
const EXPERTS: usize = 8;
const TOPK: usize = 4;
const ACTIVATIONS: [f32; 4] = [6.0, 3.0, 1.5, 0.0];
const DENSE_SHAPES: [(usize, usize, usize); 9] = [
    (1, 5120, 5120),
    (1, 1024, 5120),
    (1, 17408, 5120),
    (1, 5120, 17408),
    (4, 5120, 5120),
    (16, 5120, 5120),
    (32, 5120, 5120),
    (128, 5120, 5120),
    (512, 5120, 5120),
];
const ROUTED_SHAPES: [(usize, usize, usize); 4] = [
    (1, 2048, 512),
    (4, 2048, 512),
    (32, 2048, 512),
    (128, 2048, 512),
];

struct Options {
    iterations: usize,
    suite: String,
    dtype: DType,
    graph: bool,
    a4: bool,
}

impl Options {
    fn parse() -> Result<Self> {
        let mut options = Self {
            iterations: DEFAULT_ITERATIONS,
            suite: "all".into(),
            dtype: DType::BF16,
            graph: false,
            a4: true,
        };
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--iterations" => {
                    options.iterations = args
                        .next()
                        .ok_or_else(|| candle_core::Error::msg("missing iteration count"))?
                        .parse()
                        .map_err(candle_core::Error::msg)?;
                }
                "--suite" => {
                    options.suite = args
                        .next()
                        .ok_or_else(|| candle_core::Error::msg("missing suite"))?;
                    if !matches!(options.suite.as_str(), "all" | "dense" | "moe" | "decode") {
                        candle_core::bail!("suite must be all, dense, moe, or decode");
                    }
                }
                "--f16" => options.dtype = DType::F16,
                "--w4a16" => options.a4 = false,
                "--graph" => options.graph = true,
                _ => candle_core::bail!("unknown option {arg}"),
            }
        }
        if options.iterations == 0 {
            candle_core::bail!("iterations must be positive");
        }
        Ok(options)
    }
}

fn measure(
    device: &Device,
    options: &Options,
    mut launch: impl FnMut() -> Result<Tensor>,
) -> Result<(f64, f64, Tensor)> {
    let cuda = device.as_cuda_device()?;
    let _htod_cache = options.graph.then(|| cuda.enable_cuda_graph_htod_cache());
    for _ in 0..WARMUP_ITERATIONS {
        black_box(launch()?);
    }
    device.synchronize()?;
    let stream = cuda.cuda_stream();
    let mut output = None;
    let graph = if options.graph {
        let tracking = stream.context().is_event_tracking();
        if tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        let begin = stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED);
        if let Err(error) = begin {
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(error.to_string()));
        }
        let captured = launch();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        output = Some(captured?);
        Some(
            graph
                .map_err(|e| candle_core::Error::msg(e.to_string()))?
                .ok_or_else(|| candle_core::Error::msg("capture produced no graph"))?,
        )
    } else {
        None
    };
    let mut gpu_samples = Vec::with_capacity(SAMPLES);
    let mut host_samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let begin = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        let start = Instant::now();
        for _ in 0..options.iterations {
            if let Some(graph) = &graph {
                graph
                    .launch()
                    .map_err(|e| candle_core::Error::msg(e.to_string()))?;
            } else {
                output = Some(launch()?);
            }
        }
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        end.synchronize()
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        gpu_samples.push(
            begin
                .elapsed_ms(&end)
                .map_err(|e| candle_core::Error::msg(e.to_string()))? as f64
                / options.iterations as f64,
        );
        host_samples.push(start.elapsed().as_secs_f64() * 1000.0 / options.iterations as f64);
    }
    gpu_samples.sort_by(f64::total_cmp);
    host_samples.sort_by(f64::total_cmp);
    let result = output
        .as_ref()
        .unwrap()
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    drop(output);
    device.synchronize()?;
    drop(graph);
    Ok((gpu_samples[SAMPLES / 2], host_samples[SAMPLES / 2], result))
}

fn run_case(
    device: &Device,
    options: &Options,
    shape: (usize, usize, usize),
    routed: bool,
    per_route: bool,
) -> Result<()> {
    let (tokens, n, k) = shape;
    let experts = if routed { EXPERTS } else { 1 };
    let topk = if routed { TOPK } else { 1 };
    let input_rows = if per_route { tokens * topk } else { tokens };
    let input = (0..input_rows)
        .flat_map(|row| std::iter::repeat_n(ACTIVATIONS[row % ACTIVATIONS.len()], k))
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(input, (input_rows, k), device)?.to_dtype(options.dtype)?;
    let x = if routed {
        x.reshape((tokens, if per_route { topk } else { 1 }, k))?
    } else {
        x
    };
    let weights = Tensor::full(0x22u8, (experts, n, k / 2), device)?;
    let scales = Tensor::full(F8E4M3::from_f32(1.0), (experts, n, k / BLOCK_SIZE), device)?;
    let globals = Tensor::from_vec(
        (0..experts)
            .flat_map(|expert| std::iter::repeat_n((expert + 1) as f32 / k as f32, n))
            .collect::<Vec<_>>(),
        (experts, n),
        device,
    )?;
    let activation_global = Tensor::ones(experts, DType::F32, device)?;
    let (weights, scales, globals, activation_global) = if routed {
        (weights, scales, globals, activation_global)
    } else {
        (
            weights.squeeze(0)?,
            scales.squeeze(0)?,
            globals.squeeze(0)?,
            activation_global.reshape(())?,
        )
    };
    let ids = (0..tokens * topk)
        .map(|route| (route % experts) as u32)
        .collect::<Vec<_>>();
    let indices = Tensor::from_vec(ids.clone(), (tokens, topk), device)?;
    let args = Nvfp4GemmArgs {
        weights: &weights,
        weight_scales: &scales,
        weight_global_scale: &globals,
        activation_global_scale: options.a4.then_some(&activation_global),
    };
    let (gpu_ms, host_ms, output) = measure(device, options, || {
        if routed {
            cutile_nvfp4_gather(&x, &indices, args)
        } else {
            cutile_nvfp4(&x, args)
        }
    })?;
    let values = output.flatten_all()?.to_vec1::<f32>()?;
    for (route, row) in values.chunks_exact(n).enumerate() {
        let input_row = if per_route { route } else { route / topk };
        let expected = ACTIVATIONS[input_row % ACTIVATIONS.len()] * (ids[route] + 1) as f32;
        if row.iter().any(|&value| value != expected) {
            candle_core::bail!("benchmark output mismatch at route {route}, expected {expected}");
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "tokens": tokens, "n": n, "k": k, "experts": experts, "topk": topk,
            "per_route": per_route, "a4": options.a4, "dtype": format!("{:?}", options.dtype),
            "graph": options.graph, "iterations": options.iterations, "samples": SAMPLES,
            "gpu_ms_median": gpu_ms, "host_ms_median": host_ms,
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    let options = Options::parse()?;
    let device = Device::new_cuda(0)?;
    if options.suite != "moe" {
        for shape in DENSE_SHAPES {
            if options.suite != "decode" || shape.0 <= 4 {
                run_case(&device, &options, shape, false, false)?;
            }
        }
    }
    if matches!(options.suite.as_str(), "all" | "moe") {
        for shape in ROUTED_SHAPES {
            for per_route in [false, true] {
                run_case(&device, &options, shape, true, per_route)?;
            }
        }
    }
    Ok(())
}
