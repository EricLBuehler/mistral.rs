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
const OFFSET_WEIGHT_SHAPE: [usize; 3] = [4, 17, 64];
const OFFSET_TOKENS: usize = 2;
const OFFSET_TOPK: usize = 2;
const INPUT_OFFSETS: [usize; 4] = [1, 3, 8, 17];
const OFFSET_IDS: [u32; 4] = [3, 1, 2, 0];
const STREAM_ROWS: usize = 65_536;
const STREAM_COLUMNS: usize = 128;

struct Rng(u64);

#[test]
fn routed_gather_keeps_cross_stream_input_alive() -> Result<()> {
    use candle_core::{CudaStorage, Storage};

    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let consumer = cuda.cuda_stream();
    let producer = consumer
        .context()
        .new_stream_with_priority(-1)
        .map_err(candle_core::Error::wrap)?;
    let values = vec![F8E4M3::from_f32(1.).to_bits(); STREAM_ROWS * STREAM_COLUMNS];
    let data = producer
        .clone_htod(&values)
        .map_err(candle_core::Error::wrap)?;
    let data = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(data, cuda.clone())),
        (1, STREAM_ROWS, STREAM_COLUMNS),
    ));
    let weights = RoutedWeights::new(
        RoutedCodec::Fp8 {
            block_rows: 1,
            block_cols: STREAM_COLUMNS,
        },
        data,
        Tensor::ones((1, STREAM_ROWS, 1), DType::F32, &device)?,
        None,
        None,
        [1, STREAM_ROWS, STREAM_COLUMNS],
    );
    let input = Tensor::ones((cuda::MAX_TOKENS, 1, STREAM_COLUMNS), DType::F32, &device)?;
    let ids = Tensor::zeros((cuda::MAX_TOKENS, 1), DType::U32, &device)?;
    let _delay = crate::utils::cuda_test::delay_stream(&consumer)?;
    let output = cuda::gather(&input, &ids, &weights)?.expect("routed FP8 gather");
    drop(weights);
    let _replacement = producer
        .alloc_zeros::<u8>(values.len())
        .map_err(candle_core::Error::wrap)?;
    producer.synchronize().map_err(candle_core::Error::wrap)?;
    assert!(output
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|&v| v == STREAM_COLUMNS as f32));
    Ok(())
}

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

#[test]
fn offset_inputs_match_dequantized_reference() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let mut rng = Rng(0x5eed);
    let k = OFFSET_WEIGHT_SHAPE[2];
    let indices = Tensor::from_vec(OFFSET_IDS.to_vec(), (OFFSET_TOKENS, OFFSET_TOPK), &dev)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let mut cases = vec![
            nvfp4(&mut rng, OFFSET_WEIGHT_SHAPE, &dev)?,
            mx(&mut rng, OFFSET_WEIGHT_SHAPE, false, &dev)?,
            mx(&mut rng, OFFSET_WEIGHT_SHAPE, true, &dev)?,
            fp8_block(&mut rng, OFFSET_WEIGHT_SHAPE, [1, MX_GROUP], &dev)?,
        ];
        for bits in [2, 3, 4, 6, 8] {
            cases.push(afq(
                &mut rng,
                OFFSET_WEIGHT_SHAPE,
                bits,
                MX_GROUP,
                dtype,
                &dev,
            )?);
        }
        for offset in INPUT_OFFSETS {
            let elements = OFFSET_TOKENS * k;
            let x = random(&mut rng, &[elements + offset], dtype, &dev)?
                .narrow(0, offset, elements)?
                .reshape((OFFSET_TOKENS, 1, k))?;
            assert!(x.is_contiguous());
            assert_eq!(x.layout().start_offset(), offset);
            let rows = rows_of(&x)?;
            let route_rows = (0..OFFSET_IDS.len())
                .map(|route| rows[route / OFFSET_TOPK].clone())
                .collect::<Vec<_>>();
            for case in &cases {
                let actual = super::try_routed_gather(&x, &indices, &case.weights)?
                    .expect("routed gather declined an offset input");
                let expected = reference(case, &route_rows, &OFFSET_IDS, None);
                check(
                    &format!("{:?} {dtype:?} offset={offset}", case.weights.codec),
                    &host(&actual)?,
                    &expected.concat(),
                    if dtype == DType::F32 {
                        F32_TOL
                    } else {
                        BF16_TOL
                    },
                );
            }
        }
    }
    Ok(())
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

#[test]
fn fused_fp8_glu_and_reduce_match_reference() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let mut rng = Rng(7);
    let gate = fp8_block(&mut rng, OFFSET_WEIGHT_SHAPE, [1, MX_GROUP], &device)?;
    let up = fp8_block(&mut rng, OFFSET_WEIGHT_SHAPE, [1, MX_GROUP], &device)?;
    let input = random(
        &mut rng,
        &[OFFSET_TOKENS, 1, OFFSET_WEIGHT_SHAPE[2]],
        DType::BF16,
        &device,
    )?;
    let indices = Tensor::from_vec(OFFSET_IDS.to_vec(), (OFFSET_TOKENS, OFFSET_TOPK), &device)?;
    let rows = rows_of(&input)?;
    let rows = (0..OFFSET_IDS.len())
        .map(|i| rows[i / OFFSET_TOPK].clone())
        .collect::<Vec<_>>();
    let expected = reference(&gate, &rows, &OFFSET_IDS, None);
    let expected_up = reference(&up, &rows, &OFFSET_IDS, None);
    let glu = cuda::glu(
        &input,
        &indices,
        &gate.weights,
        &up.weights,
        GluActivationType::Relu,
    )?
    .unwrap();
    let expected_glu = expected
        .iter()
        .flatten()
        .zip(expected_up.iter().flatten())
        .map(|(gate, up)| gate.max(0.0) * up)
        .collect::<Vec<_>>();
    check("fused GLU", &host(&glu)?, &expected_glu, BF16_TOL);
    let routing = Tensor::new(&[[0.25f32, 0.75], [0.5, 0.5]], &device)?;
    let weights = routing.to_vec2::<f32>()?;
    let expected_reduce = (0..OFFSET_TOKENS)
        .flat_map(|token| {
            let expected = &expected;
            let weights = &weights;
            (0..OFFSET_WEIGHT_SHAPE[1]).map(move |row| {
                (0..OFFSET_TOPK)
                    .map(|route| weights[token][route] * expected[token * OFFSET_TOPK + route][row])
                    .sum::<f32>()
            })
        })
        .collect::<Vec<_>>();
    let reduced = cuda::reduce(&input, &indices, &routing, &gate.weights)?.unwrap();
    check("fused reduce", &host(&reduced)?, &expected_reduce, BF16_TOL);
    Ok(())
}
