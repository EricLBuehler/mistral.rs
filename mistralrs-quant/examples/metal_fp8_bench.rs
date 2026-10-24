use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use mistralrs_quant::{
    fp8_blockwise_dequantize, BlockwiseFP8Linear, QuantMethod, QuantMethodConfig,
};

const BLOCK_SIZE: usize = 128;
const WARMUP_TIME: Duration = Duration::from_millis(100);
const SAMPLES: usize = 5;
const ITERATIONS: usize = 20;
const DECODE_ROWS: usize = 4;
const PROJECTIONS: [(usize, usize); 3] = [(4096, 4096), (14336, 4096), (4096, 14336)];
const ROWS: [usize; 13] = [1, 2, 4, 8, 16, 32, 33, 128, 256, 257, 512, 1024, 4096];

fn measure(
    device: &Device,
    serialized: bool,
    mut forward: impl FnMut() -> Result<Tensor>,
) -> Result<f64> {
    let warmup = Instant::now();
    loop {
        for _ in 0..ITERATIONS {
            black_box(forward()?);
        }
        device.synchronize()?;
        if warmup.elapsed() >= WARMUP_TIME {
            break;
        }
    }
    let mut timings = [0.0; SAMPLES];
    for timing in &mut timings {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(forward()?);
            if serialized {
                device.synchronize()?;
            }
        }
        device.synchronize()?;
        *timing = start.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64;
    }
    timings.sort_by(f64::total_cmp);
    Ok(timings[SAMPLES / 2])
}

fn main() -> Result<()> {
    let mut dtype = DType::BF16;
    let mut serialized = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "bf16" => dtype = DType::BF16,
            "f16" => dtype = DType::F16,
            "f32" => dtype = DType::F32,
            "serial" => serialized = true,
            _ => candle_core::bail!("usage: metal_fp8_bench [bf16|f16|f32] [serial]"),
        }
    }
    let device = Device::new_metal(0)?;
    println!(
        "{dtype:?}, block={BLOCK_SIZE}; median of {SAMPLES} batches of {ITERATIONS}; serialized={serialized}; host dispatch + GPU completion"
    );
    println!("M,N,K,fp8_us,dequant_matmul_us,cached_matmul_us,max_abs_error");
    for (n, k) in PROJECTIONS {
        let weights: Vec<_> = (0..n * k)
            .map(|i| F8E4M3::from_f32(((i * 17 % 251) as f32 - 125.0) / 32.0))
            .collect();
        let weight = Tensor::from_vec(weights, (n, k), &device)?;
        let block = [BLOCK_SIZE, BLOCK_SIZE];
        let scale_shape = (n.div_ceil(block[0]), k.div_ceil(block[1]));
        let scales = Tensor::from_vec(
            (0..scale_shape.0 * scale_shape.1)
                .map(|i| 0.003 + (i % 31) as f32 * 0.0001)
                .collect::<Vec<_>>(),
            scale_shape,
            &device,
        )?;
        let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
            weight: weight.clone(),
            weight_scale_inv: scales.clone(),
            bias: None,
            dequant_dtype: dtype,
            weight_block_size: block.to_vec(),
            activation_scheme: None,
        })?;
        layer.prepare_metal_kernels(dtype)?;
        let dense = fp8_blockwise_dequantize(&weight, &scales, block.to_vec(), dtype)?;
        for m in ROWS {
            if serialized && m > DECODE_ROWS {
                continue;
            }
            let input = Tensor::from_vec(
                (0..m * k)
                    .map(|i| ((i * 13 % 127) as f32 - 63.0) / 64.0)
                    .collect::<Vec<_>>(),
                (m, k),
                &device,
            )?
            .to_dtype(dtype)?;
            let expected = input.matmul(&dense.t()?)?.to_dtype(DType::F32)?;
            let actual = layer.forward(&input)?.to_dtype(DType::F32)?;
            let error = (&actual - &expected)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let direct = measure(&device, serialized, || layer.forward(&input))?;
            let dequant = measure(&device, serialized, || {
                let dense = fp8_blockwise_dequantize(&weight, &scales, block.to_vec(), dtype)?;
                input.matmul(&dense.t()?)
            })?;
            let cached = measure(&device, serialized, || input.matmul(&dense.t()?))?;
            println!("{m},{n},{k},{direct:.2},{dequant:.2},{cached:.2},{error:.6}");
        }
    }
    Ok(())
}
