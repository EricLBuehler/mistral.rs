use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use half::bf16;

use super::{cuda, RoutedCodec, RoutedWeights};
use crate::GluActivationType;

const FP4_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const FP4_SIGN: u8 = 8;
const E8M0_BIAS: i32 = 127;
const E8M0_SPAN: u64 = 7;
const E8M0_LOW: i32 = 122;
const NVFP4_GROUP: usize = 16;
const MX_GROUP: usize = 32;
const BF16_TOL: f32 = 1e-2;
const F32_TOL: f32 = 1e-4;
const ACTIVATIONS: [GluActivationType; 5] = [
    GluActivationType::Silu,
    GluActivationType::Gelu,
    GluActivationType::Relu,
    GluActivationType::GeluErf,
    GluActivationType::Sigmoid,
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn uniform(&mut self) -> f32 {
        (self.next() % 2_000_001) as f32 / 1_000_000.0 - 1.0
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }

    fn e4m3(&mut self, magnitude: f32) -> u8 {
        F8E4M3::from_f32(self.uniform() * magnitude).to_bits()
    }

    fn e8m0(&mut self) -> u8 {
        (E8M0_LOW + (self.next() % E8M0_SPAN) as i32) as u8
    }
}

fn fp4(nibble: u8) -> f32 {
    let value = FP4_VALUES[(nibble & 7) as usize];
    if nibble & FP4_SIGN != 0 {
        -value
    } else {
        value
    }
}

fn e8m0(value: u8) -> f32 {
    2f32.powi(value as i32 - E8M0_BIAS)
}

/// Device-side routed weights plus the host dequantization `[E, N, K]` they encode.
struct Case {
    weights: RoutedWeights,
    dense: Vec<f32>,
}

fn u8_tensor(data: Vec<u8>, shape: &[usize], dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(data, shape, dev)
}

fn nvfp4(rng: &mut Rng, [e, n, k]: [usize; 3], dev: &Device) -> Result<Case> {
    let packed = rng.bytes(e * n * k / 2);
    let scales = (0..e * n * k / NVFP4_GROUP)
        .map(|_| F8E4M3::from_f32(0.25 + (rng.next() % 16) as f32 / 8.0).to_bits())
        .collect::<Vec<_>>();
    let global = (0..e * n)
        .map(|_| 0.01 + rng.uniform().abs() * 0.02)
        .collect::<Vec<_>>();
    let mut dense = Vec::with_capacity(e * n * k);
    for (i, byte) in packed.iter().enumerate() {
        for (j, nibble) in [byte & 15, byte >> 4].into_iter().enumerate() {
            let index = 2 * i + j;
            let scale = F8E4M3::from_bits(scales[index / NVFP4_GROUP]).to_f32();
            dense.push(fp4(nibble) * scale * global[index / k]);
        }
    }
    // the codec reads raw bytes, so U8 storage of the E4M3 bit patterns stands in for F8E4M3
    let scales = u8_tensor(scales, &[e, n, k / NVFP4_GROUP], dev)?;
    Ok(Case {
        weights: RoutedWeights::new(
            RoutedCodec::Nvfp4,
            u8_tensor(packed, &[e, n, k / 2], dev)?,
            scales,
            Some(Tensor::from_vec(global, (e, n), dev)?),
            None,
            [e, n, k],
        ),
        dense,
    })
}

fn mx(rng: &mut Rng, [e, n, k]: [usize; 3], fp8: bool, dev: &Device) -> Result<Case> {
    let scales = (0..e * n * k / MX_GROUP)
        .map(|_| rng.e8m0())
        .collect::<Vec<_>>();
    let (data, dense, data_shape) = if fp8 {
        let data = (0..e * n * k).map(|_| rng.e4m3(4.0)).collect::<Vec<_>>();
        let dense = data
            .iter()
            .enumerate()
            .map(|(i, &b)| F8E4M3::from_bits(b).to_f32() * e8m0(scales[i / MX_GROUP]))
            .collect();
        (data, dense, vec![e, n, k])
    } else {
        let data = rng.bytes(e * n * k / 2);
        let mut dense = Vec::with_capacity(e * n * k);
        for (i, byte) in data.iter().enumerate() {
            for (j, nibble) in [byte & 15, byte >> 4].into_iter().enumerate() {
                dense.push(fp4(nibble) * e8m0(scales[(2 * i + j) / MX_GROUP]));
            }
        }
        (data, dense, vec![e, n, k / 2])
    };
    Ok(Case {
        weights: RoutedWeights::new(
            if fp8 {
                RoutedCodec::Mxfp8
            } else {
                RoutedCodec::Mxfp4
            },
            u8_tensor(data, &data_shape, dev)?,
            u8_tensor(scales, &[e, n, k / MX_GROUP], dev)?,
            None,
            None,
            [e, n, k],
        ),
        dense,
    })
}

fn fp8_block(
    rng: &mut Rng,
    [e, n, k]: [usize; 3],
    [block_rows, block_cols]: [usize; 2],
    dev: &Device,
) -> Result<Case> {
    let data = (0..e * n * k).map(|_| rng.e4m3(8.0)).collect::<Vec<_>>();
    let (rows, cols) = (n.div_ceil(block_rows), k.div_ceil(block_cols));
    let scales = (0..e * rows * cols)
        .map(|_| 0.001 + rng.uniform().abs() * 0.004)
        .collect::<Vec<_>>();
    let dense = data
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let (expert, row, col) = (i / (n * k), (i / k) % n, i % k);
            F8E4M3::from_bits(b).to_f32()
                * scales[(expert * rows + row / block_rows) * cols + col / block_cols]
        })
        .collect();
    Ok(Case {
        weights: RoutedWeights::new(
            RoutedCodec::Fp8 {
                block_rows,
                block_cols,
            },
            u8_tensor(data, &[e, n, k], dev)?,
            Tensor::from_vec(scales, (e, rows, cols), dev)?,
            None,
            None,
            [e, n, k],
        ),
        dense,
    })
}

fn afq(
    rng: &mut Rng,
    [e, n, k]: [usize; 3],
    bits: usize,
    group: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Case> {
    let codes = (0..e * n * k)
        .map(|_| (rng.next() % (1 << bits)) as u32)
        .collect::<Vec<_>>();
    let mut packed = vec![0u8; e * n * k * bits / 8];
    for (i, &code) in codes.iter().enumerate() {
        for bit in 0..bits {
            if code >> bit & 1 == 1 {
                let at = i * bits + bit;
                packed[at / 8] |= 1 << (at % 8);
            }
        }
    }
    let groups = e * n * k / group;
    let round = |v: f32| bf16::from_f32(v).to_f32();
    let scales = (0..groups)
        .map(|_| round(0.002 + rng.uniform().abs() * 0.01))
        .collect::<Vec<_>>();
    // centered like real affine quantization so GLU gates are not all saturated
    let mid = (1u32 << (bits - 1)) as f32;
    let biases = scales
        .iter()
        .map(|s| round(-mid * s + rng.uniform() * 0.002))
        .collect::<Vec<_>>();
    let dense = codes
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * scales[i / group] + biases[i / group])
        .collect();
    let affine = |v: Vec<f32>| Tensor::from_vec(v, (e, n, k / group), dev)?.to_dtype(dtype);
    Ok(Case {
        weights: RoutedWeights::new(
            RoutedCodec::Afq { bits, group },
            u8_tensor(packed, &[e, n, k * bits / 8], dev)?,
            affine(scales)?,
            Some(affine(biases)?),
            None,
            [e, n, k],
        ),
        dense,
    })
}

fn routing(rng: &mut Rng, tokens: usize, topk: usize, used: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(tokens * topk);
    for _ in 0..tokens {
        let mut picked = Vec::with_capacity(topk);
        while picked.len() < topk {
            let e = (rng.next() as usize % used) as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    ids
}

fn random(rng: &mut Rng, shape: &[usize], dtype: DType, dev: &Device) -> Result<Tensor> {
    let len = shape.iter().product::<usize>();
    let values = (0..len).map(|_| rng.uniform()).collect::<Vec<_>>();
    Tensor::from_vec(values, shape, dev)?.to_dtype(dtype)
}

// rows [routes, K] f32 against each route's dense expert -> [routes, N]
fn reference(case: &Case, rows: &[Vec<f32>], ids: &[u32], bias: Option<&[f32]>) -> Vec<Vec<f32>> {
    let (n, k) = (case.weights.n, case.weights.k);
    ids.iter()
        .zip(rows)
        .map(|(&e, x)| {
            (0..n)
                .map(|row| {
                    let w = &case.dense[(e as usize * n + row) * k..][..k];
                    let dot = w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
                    dot + bias.map_or(0.0, |b| b[e as usize * n + row])
                })
                .collect()
        })
        .collect()
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn rows_of(x: &Tensor) -> Result<Vec<Vec<f32>>> {
    let k = x.dim(candle_core::D::Minus1)?;
    Ok(host(x)?.chunks(k).map(<[f32]>::to_vec).collect())
}

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let err = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let norm = want.iter().map(|b| b * b).sum::<f32>().sqrt().max(1e-12);
    assert!(
        err / norm < tol,
        "{name}: rel err {} >= {tol}; got {:?} want {:?}",
        err / norm,
        &got[..got.len().min(6)],
        &want[..want.len().min(6)]
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: non-finite output"
    );
}

fn activate(g: f32, act: GluActivationType) -> f32 {
    match act {
        GluActivationType::Silu => g / (1.0 + (-g).exp()),
        GluActivationType::Gelu => {
            0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh())
        }
        GluActivationType::Relu => g.max(0.0),
        GluActivationType::GeluErf => 0.5 * g * (1.0 + erf(g / std::f32::consts::SQRT_2)),
        GluActivationType::Sigmoid => 1.0 / (1.0 + (-g).exp()),
    }
}

// Abramowitz-Stegun 7.1.26, |err| < 1.5e-7
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152_1 + t * 1.061_405_4))));
    let y = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

struct Shape {
    tokens: usize,
    topk: usize,
    used: usize,
    per_route: bool,
}

const SHAPES: [Shape; 5] = [
    Shape {
        tokens: 1,
        topk: 1,
        used: 4,
        per_route: false,
    },
    Shape {
        tokens: 1,
        topk: 8,
        used: 12,
        per_route: false,
    },
    Shape {
        tokens: 3,
        topk: 2,
        used: 5,
        per_route: true,
    },
    Shape {
        tokens: 5,
        topk: 10,
        used: 12,
        per_route: false,
    },
    Shape {
        tokens: 8,
        topk: 8,
        used: 9,
        per_route: true,
    },
];

fn exercise(
    name: &str,
    make: &dyn Fn(&mut Rng, &Device) -> Result<Case>,
    dtype: DType,
) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let tol = if dtype == DType::F32 {
        F32_TOL
    } else {
        BF16_TOL
    };
    let mut rng = Rng(0x5eed);
    let gate = make(&mut rng, &dev)?;
    let up = make(&mut rng, &dev)?;
    let (n, k) = (gate.weights.n, gate.weights.k);
    for prefetch in [false, true] {
        for lanes in [8, 16, 32] {
            cuda::FORCE_PREFETCH.with(|f| f.set(Some(prefetch)));
            cuda::FORCE_LANES.with(|f| f.set(Some(lanes)));
            for shape in &SHAPES {
                let tag = format!(
                    "{name} {dtype:?} prefetch={prefetch} lanes={lanes} t={} topk={}",
                    shape.tokens, shape.topk
                );
                let ids = routing(&mut rng, shape.tokens, shape.topk, shape.used);
                let idx = Tensor::from_vec(ids.clone(), (shape.tokens, shape.topk), &dev)?;
                let routes = if shape.per_route { shape.topk } else { 1 };
                let x = random(&mut rng, &[shape.tokens, routes, k], dtype, &dev)?;
                let x_rows = rows_of(&x)?;
                let route_rows = (0..ids.len())
                    .map(|r| x_rows[if shape.per_route { r } else { r / shape.topk }].clone())
                    .collect::<Vec<_>>();
                let plain = super::try_routed_gather(&x, &idx, &gate.weights)?
                    .expect("routed gather declined a supported case");
                assert_eq!(plain.dims(), &[shape.tokens, shape.topk, n]);
                let want = reference(&gate, &route_rows, &ids, None);
                check(&format!("{tag} plain"), &host(&plain)?, &want.concat(), tol);

                let routing_weights =
                    random(&mut rng, &[shape.tokens, shape.topk], DType::F32, &dev)?;
                let rw = host(&routing_weights)?;
                let reduced = cuda::reduce(&x, &idx, &routing_weights, &gate.weights)?
                    .expect("routed reduce declined a supported case");
                let want_reduce = (0..shape.tokens)
                    .flat_map(|t| {
                        let want = &want;
                        let rw = &rw;
                        (0..n).map(move |row| {
                            (0..shape.topk)
                                .map(|s| rw[t * shape.topk + s] * want[t * shape.topk + s][row])
                                .sum::<f32>()
                        })
                    })
                    .collect::<Vec<_>>();
                check(
                    &format!("{tag} reduce"),
                    &host(&reduced)?,
                    &want_reduce,
                    tol,
                );

                if shape.per_route {
                    continue;
                }
                let want_up = reference(&up, &route_rows, &ids, None);
                for act in ACTIVATIONS {
                    let glu = cuda::glu(&x, &idx, &gate.weights, &up.weights, act)?
                        .expect("routed GLU declined a supported case");
                    let want_glu = want
                        .concat()
                        .iter()
                        .zip(want_up.concat())
                        .map(|(g, u)| activate(*g, act) * u)
                        .collect::<Vec<_>>();
                    check(&format!("{tag} glu {act:?}"), &host(&glu)?, &want_glu, tol);
                }
            }
        }
    }
    cuda::FORCE_PREFETCH.with(|f| f.set(None));
    cuda::FORCE_LANES.with(|f| f.set(None));
    Ok(())
}

#[test]
fn nvfp4_matches_dequantized_reference() -> Result<()> {
    for shape in [[12, 37, 128], [12, 70, 80]] {
        exercise(
            &format!("nvfp4 {shape:?}"),
            &|r, d| nvfp4(r, shape, d),
            DType::BF16,
        )?;
    }
    exercise("nvfp4 f32", &|r, d| nvfp4(r, [12, 33, 64], d), DType::F32)
}

#[test]
fn mxfp4_matches_dequantized_reference() -> Result<()> {
    exercise("mxfp4", &|r, d| mx(r, [12, 45, 160], false, d), DType::BF16)?;
    exercise(
        "mxfp4 f16",
        &|r, d| mx(r, [12, 19, 96], false, d),
        DType::F16,
    )
}

#[test]
fn mxfp8_matches_dequantized_reference() -> Result<()> {
    exercise("mxfp8", &|r, d| mx(r, [12, 41, 192], true, d), DType::BF16)?;
    exercise(
        "mxfp8 f32",
        &|r, d| mx(r, [12, 17, 64], true, d),
        DType::F32,
    )
}

#[test]
fn fp8_matches_dequantized_reference() -> Result<()> {
    exercise(
        "fp8 block",
        &|r, d| fp8_block(r, [12, 75, 208], [32, 64], d),
        DType::BF16,
    )?;
    exercise(
        "fp8 channel",
        &|r, d| fp8_block(r, [12, 29, 144], [1, 144], d),
        DType::BF16,
    )?;
    exercise(
        "fp8 tensor",
        &|r, d| fp8_block(r, [12, 29, 112], [29, 112], d),
        DType::F16,
    )
}

#[test]
fn afq_matches_dequantized_reference() -> Result<()> {
    for bits in [2, 3, 4, 6, 8] {
        for group in [32, 64, 128] {
            exercise(
                &format!("afq{bits} g{group}"),
                &|r, d| afq(r, [12, 23, 256], bits, group, DType::BF16, d),
                DType::BF16,
            )?;
        }
    }
    exercise(
        "afq4 f32",
        &|r, d| afq(r, [12, 9, 64], 4, 32, DType::F32, d),
        DType::F32,
    )
}

#[test]
fn rejects_tokens_past_decode_limit() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let mut rng = Rng(7);
    let case = mx(&mut rng, [4, 8, 64], true, &dev)?;
    let tokens = cuda::MAX_TOKENS + 1;
    let idx = Tensor::zeros((tokens, 2), DType::U32, &dev)?;
    let x = random(&mut rng, &[tokens, 1, 64], DType::BF16, &dev)?;
    assert!(super::try_routed_gather(&x, &idx, &case.weights)?.is_none());
    Ok(())
}
