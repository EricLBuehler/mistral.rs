use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;

use super::{dense, gather_grouped, Fp8ScaleEncoding, Fp8WeightView};

const TOPK: usize = 8;
const HOT_EXPERT: u32 = 5;
const FP8_CODE_STD: f32 = 96.0;
const FP8_MAX: f32 = 448.0;
const E8M0_MIN_CODE: u32 = 117;
const E8M0_CODE_SPAN: u32 = 6;
const MX_GROUP: usize = 32;
const FP8_BLOCK: usize = 128;
const TOLERANCE: f32 = 1.5e-2;

#[derive(Clone, Copy, Debug)]
enum Scales {
    Tensor,
    Channel,
    Block,
    Mx,
}

impl Scales {
    fn block(self, n: usize, k: usize) -> [usize; 2] {
        match self {
            Self::Tensor => [n, k],
            Self::Channel => [1, k],
            Self::Block => [FP8_BLOCK, FP8_BLOCK],
            Self::Mx => [1, MX_GROUP],
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn uniform(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32
    }

    fn normal(&mut self) -> f32 {
        let u = self.uniform().max(1e-7);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

struct Weights {
    codes: Tensor,
    scales: Tensor,
    encoding: Fp8ScaleEncoding,
    block: [usize; 2],
    deq: Tensor,
}

impl Weights {
    fn view(&self) -> Fp8WeightView<'_> {
        Fp8WeightView {
            weight: &self.codes,
            scales: &self.scales,
            encoding: self.encoding,
            block: self.block,
        }
    }
}

fn weights(experts: usize, n: usize, k: usize, kind: Scales, dev: &Device) -> Result<Weights> {
    let mut rng = Lcg(experts as u64 * 7919 + n as u64 * 31 + k as u64);
    let codes = (0..experts * n * k)
        .map(|_| F8E4M3::from_f32((rng.normal() * FP8_CODE_STD).clamp(-FP8_MAX, FP8_MAX)))
        .collect::<Vec<_>>();
    let [bn, bk] = kind.block(n, k);
    let (rows, cols) = (n.div_ceil(bn), k.div_ceil(bk));
    let count = experts * rows * cols;
    let scale_values = (0..count)
        .map(|_| match kind {
            Scales::Mx => f32::from_bits((E8M0_MIN_CODE + rng.next() % E8M0_CODE_SPAN) << 23),
            _ => (0.5 + rng.uniform()) / FP8_MAX,
        })
        .collect::<Vec<_>>();
    let mut deq = vec![0f32; experts * n * k];
    for e in 0..experts {
        for row in 0..n {
            for col in 0..k {
                let index = (e * n + row) * k + col;
                let scale = scale_values[(e * rows + row / bn) * cols + col / bk];
                deq[index] = codes[index].to_f32() * scale;
            }
        }
    }
    let (scales, encoding) = match kind {
        Scales::Mx => (
            Tensor::from_vec(
                scale_values
                    .iter()
                    .map(|s| (s.to_bits() >> 23) as u8)
                    .collect::<Vec<_>>(),
                (experts, rows, cols),
                dev,
            )?,
            Fp8ScaleEncoding::E8m0,
        ),
        _ => (
            Tensor::from_vec(scale_values, (experts, rows, cols), dev)?,
            Fp8ScaleEncoding::F32,
        ),
    };
    let codes = Tensor::from_vec(codes, (experts, n, k), &Device::Cpu)?.to_device(dev)?;
    let deq = Tensor::from_vec(deq, (experts, n, k), dev)?;
    Ok(Weights {
        codes,
        scales,
        encoding,
        block: [bn, bk],
        deq,
    })
}

fn routing(tokens: usize, experts: usize, unbalanced: bool) -> Vec<u32> {
    let mut rng = Lcg(tokens as u64 * 131 + experts as u64);
    let pool = if unbalanced { experts / 2 } else { experts };
    let mut ids = Vec::with_capacity(tokens * TOPK);
    for _ in 0..tokens {
        let mut picked = Vec::with_capacity(TOPK);
        if unbalanced {
            picked.push(HOT_EXPERT);
        }
        while picked.len() < TOPK {
            let e = rng.next() % pool as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    ids
}

// per-expert f32 GEMMs on the device; route r reads input row r / divisor
fn reference(x: &Tensor, divisor: usize, deq: &Tensor, ids: &[u32], n: usize) -> Result<Vec<f32>> {
    let experts = deq.dim(0)?;
    let mut out = vec![0f32; ids.len() * n];
    for e in 0..experts {
        let routes = (0..ids.len())
            .filter(|&r| ids[r] as usize == e)
            .collect::<Vec<_>>();
        if routes.is_empty() {
            continue;
        }
        let rows = routes
            .iter()
            .map(|&r| (r / divisor) as u32)
            .collect::<Vec<_>>();
        let rows = Tensor::from_vec(rows, routes.len(), x.device())?;
        let y = x
            .index_select(&rows, 0)?
            .matmul(&deq.get(e)?.t()?)?
            .to_vec2::<f32>()?;
        for (route, values) in routes.into_iter().zip(y) {
            out[route * n..(route + 1) * n].copy_from_slice(&values);
        }
    }
    Ok(out)
}

fn assert_close(label: &str, y: &Tensor, reference: &[f32]) -> Result<()> {
    let y = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(y.len(), reference.len(), "{label}");
    let scale = reference.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = y
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst / scale <= TOLERANCE,
        "{label}: max error {worst} at reference scale {scale}"
    );
    Ok(())
}

fn input(rows: usize, k: usize, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut rng = Lcg(rows as u64 * 17 + k as u64);
    let values = (0..rows * k).map(|_| rng.normal()).collect::<Vec<_>>();
    let x = Tensor::from_vec(values, (rows, k), dev)?.to_dtype(dtype)?;
    let rounded = x.to_dtype(DType::F32)?;
    Ok((x, rounded))
}

struct Case {
    tokens: usize,
    experts: usize,
    n: usize,
    k: usize,
    kind: Scales,
    dtype: DType,
    per_route_input: bool,
    unbalanced: bool,
}

fn run_gather(case: Case, dev: &Device) -> Result<()> {
    let label = format!(
        "T={} E={} N={} K={} {:?} {:?} per_route={} unbalanced={}",
        case.tokens,
        case.experts,
        case.n,
        case.k,
        case.kind,
        case.dtype,
        case.per_route_input,
        case.unbalanced
    );
    let w = weights(case.experts, case.n, case.k, case.kind, dev)?;
    let ids = routing(case.tokens, case.experts, case.unbalanced);
    let x_routes = if case.per_route_input { TOPK } else { 1 };
    let (x, xf) = input(case.tokens * x_routes, case.k, case.dtype, dev)?;
    let x = x.reshape((case.tokens, x_routes, case.k))?;
    let indices = Tensor::from_vec(ids.clone(), (case.tokens, TOPK), dev)?;
    let y = gather_grouped(&x, &indices, w.view())?.expect(&label);
    assert_eq!(y.dims(), [case.tokens, TOPK, case.n], "{label}");
    assert_eq!(y.dtype(), case.dtype, "{label}");
    let divisor = if case.per_route_input { 1 } else { TOPK };
    assert_close(&label, &y, &reference(&xf, divisor, &w.deq, &ids, case.n)?)
}

#[test]
fn grouped_w8a16_matches_dequantized_reference() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for kind in [Scales::Tensor, Scales::Channel, Scales::Block, Scales::Mx] {
        let k = if matches!(kind, Scales::Mx) {
            1056
        } else {
            1040
        };
        for tokens in [1, 31, 512, 2048] {
            for experts in [8, 128] {
                for n in [200, 203] {
                    run_gather(
                        Case {
                            tokens,
                            experts,
                            n,
                            k,
                            kind,
                            dtype: DType::BF16,
                            per_route_input: false,
                            unbalanced: experts > TOPK,
                        },
                        &dev,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn grouped_w8a16_covers_f16_and_per_route_inputs() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for kind in [Scales::Channel, Scales::Block, Scales::Mx] {
        let k = if matches!(kind, Scales::Mx) {
            1024
        } else {
            1040
        };
        for tokens in [31, 512] {
            for (dtype, per_route_input) in
                [(DType::F16, false), (DType::BF16, true), (DType::F16, true)]
            {
                run_gather(
                    Case {
                        tokens,
                        experts: 128,
                        n: 256,
                        k,
                        kind,
                        dtype,
                        per_route_input,
                        unbalanced: false,
                    },
                    &dev,
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn dense_w8a16_matches_dequantized_reference() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for kind in [Scales::Tensor, Scales::Channel, Scales::Block, Scales::Mx] {
        let k = if matches!(kind, Scales::Mx) {
            1056
        } else {
            1040
        };
        let n = 203;
        let w = weights(1, n, k, kind, &dev)?;
        let codes = w.codes.squeeze(0)?;
        let view = Fp8WeightView {
            weight: &codes,
            ..w.view()
        };
        for rows in [1, 31, 512] {
            for dtype in [DType::BF16, DType::F16] {
                let label = format!("dense M={rows} {kind:?} {dtype:?}");
                let (x, xf) = input(rows, k, dtype, &dev)?;
                let x = x.reshape((1, rows, k))?;
                let y = dense(&x, view)?.expect(&label);
                assert_eq!(y.dims(), [1, rows, n], "{label}");
                let expected = reference(&xf, 1, &w.deq, &vec![0u32; rows], n)?;
                assert_close(&label, &y, &expected)?;
            }
        }
    }
    Ok(())
}

#[test]
fn sparse_routing_defers_to_per_route_kernels() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let w = weights(128, 256, 1024, Scales::Channel, &dev)?;
    let (x, _) = input(4, 1024, DType::BF16, &dev)?;
    let indices = Tensor::from_vec(routing(4, 128, false), (4, TOPK), &dev)?;
    assert!(super::gather(&x.reshape((4, 1, 1024))?, &indices, w.view())?.is_none());
    Ok(())
}

const BENCH_EXPERTS: usize = 128;
const BENCH_HIDDEN: usize = 2048;
const BENCH_INTER: usize = 768;
const BENCH_ITERS: usize = 20;
// tileiras aborts when spawned under ncu's injection, so profiling runs drop the cuTile baseline
const BENCH_SKIP_CUTILE_ENV: &str = "GROUPED_FP8_BENCH_SKIP_CUTILE";

fn time_ms(dev: &Device, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    f()?;
    dev.synchronize()?;
    let start = std::time::Instant::now();
    for _ in 0..BENCH_ITERS {
        f()?;
    }
    dev.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / BENCH_ITERS as f64)
}

// Qwen3-30B-A3B gate projection vs the bf16 grouped MoE GEMM and the per-route FP8 kernel
#[test]
#[ignore]
fn bench_grouped_w8a16_qwen3_30b() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (e, n, k) = (BENCH_EXPERTS, BENCH_INTER, BENCH_HIDDEN);
    let kinds = [Scales::Channel, Scales::Block, Scales::Mx];
    let built = kinds
        .iter()
        .map(|&kind| weights(e, n, k, kind, &dev))
        .collect::<Result<Vec<_>>>()?;
    #[cfg(feature = "cutile")]
    let bf16 = built[0].deq.to_dtype(DType::BF16)?;
    let bytes = (e * n * k) as f64;
    for tokens in [8, 16, 32, 64, 128, 512, 2048] {
        let (x, _) = input(tokens, k, DType::BF16, &dev)?;
        let x = x.reshape((tokens, 1, k))?;
        let ids = routing(tokens, e, false);
        let touched = ids.iter().collect::<std::collections::HashSet<_>>().len();
        let indices = Tensor::from_vec(ids, (tokens, TOPK), &dev)?;
        let mut line = format!("T={tokens:5}");
        #[cfg(feature = "cutile")]
        if std::env::var_os(BENCH_SKIP_CUTILE_ENV).is_none() {
            let cuda = dev.as_cuda_device()?.clone();
            let cfg = crate::cutile::get_default_config(tokens, e);
            let ms = time_ms(&dev, || {
                let flat = indices.flatten_all()?;
                let (storage, layout) = flat.storage_and_layout();
                let candle_core::Storage::Cuda(storage) = &*storage else {
                    unreachable!()
                };
                let view = storage
                    .as_cuda_slice::<u32>()?
                    .slice(layout.start_offset()..layout.start_offset() + tokens * TOPK);
                let (sids, eids, ntpp, em) =
                    super::super::cuda::moe_align(&view, tokens, e, TOPK, cfg.bm, &cuda)?;
                crate::cutile::cutile_grouped_gemm(
                    &x.reshape((tokens, k))?,
                    &bf16,
                    &sids,
                    &eids,
                    &ntpp,
                    None,
                    em,
                    tokens * TOPK,
                    TOPK,
                    false,
                    cfg,
                    &cuda,
                )
            })?;
            line += &format!("  bf16-cutile {ms:7.3}ms");
        }
        for (kind, w) in kinds.iter().zip(&built) {
            let grouped = time_ms(
                &dev,
                || Ok(gather_grouped(&x, &indices, w.view())?.unwrap()),
            )?;
            let per_route = if matches!(kind, Scales::Mx) {
                f64::NAN
            } else {
                time_ms(&dev, || {
                    crate::blockwise_fp8::fp8_indexed_moe_gemm(
                        &x, &w.codes, &w.scales, &indices, &w.block,
                    )
                })?
            };
            let gbs = bytes * touched as f64 / e as f64 / (grouped * 1e-3) / 1e9;
            line +=
                &format!("  {kind:?} {grouped:6.3}ms ({gbs:4.0} GB/s) per-route {per_route:6.3}ms");
        }
        println!("{line}");
    }
    Ok(())
}
