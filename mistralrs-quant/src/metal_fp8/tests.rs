use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use half::{bf16, f16};

use super::linear;
use crate::{BlockwiseFP8Linear, Fp8ActivationScheme, QuantMethod, QuantMethodConfig};

fn rounded(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => f16::from_f32(value).to_f32(),
        DType::BF16 => bf16::from_f32(value).to_f32(),
        DType::F32 => value,
        _ => unreachable!(),
    }
}

fn offset_tensor(values: &[f32], shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
    // Candle's strided copy requires four-byte buffer offsets.
    const BUFFER_ALIGNMENT: usize = 4;
    let offset = BUFFER_ALIGNMENT / dtype.size_in_bytes();
    let mut padded = vec![-123.0; offset];
    padded.extend_from_slice(values);
    Tensor::from_vec(padded, values.len() + offset, device)?
        .to_dtype(dtype)?
        .narrow(0, offset, values.len())?
        .reshape(shape)
}

fn assert_close(actual: &Tensor, expected: &[f32], context: &str) -> Result<()> {
    let (absolute, relative) = match actual.dtype() {
        DType::F32 => (2e-5, 2e-5),
        DType::F16 => (2e-5, 1e-3),
        DType::BF16 => (2e-5, 8e-3),
        _ => unreachable!(),
    };
    let values = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(values.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in values.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{context}, element {index}: got {actual}, expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn metal_e4m3_decode_matches_all_encodings() -> Result<()> {
    let device = Device::new_metal(0)?;
    let encodings: Vec<_> = (u8::MIN..=u8::MAX).map(F8E4M3::from_bits).collect();
    let input = Tensor::from_vec(encodings.clone(), encodings.len(), &device)?;
    let decoded = crate::scalar_fp8::ops::fp8_to_dtype(&input, DType::F32)?.to_vec1::<f32>()?;
    for (bits, (&actual, encoded)) in decoded.iter().zip(&encodings).enumerate() {
        let expected = encoded.to_f32();
        if expected.is_nan() {
            assert!(actual.is_nan(), "encoding {bits:#04x}: got {actual}");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "encoding {bits:#04x}");
        }
    }
    Ok(())
}

struct LinearCase {
    name: &'static str,
    shape: &'static [usize],
    n: usize,
    block: [usize; 2],
    strided: bool,
}

fn reference(
    case: &LinearCase,
    input: &[f32],
    weight: &[F8E4M3],
    scales: &[f32],
    dtype: DType,
) -> Vec<f32> {
    let k = *case.shape.last().unwrap();
    input
        .chunks_exact(k)
        .flat_map(|row| {
            (0..case.n).map(move |n| {
                let mut sum = 0.0_f32;
                for (column, &x) in row.iter().enumerate() {
                    let index = n * k + column;
                    let [by, bx] = case.block;
                    let scale = (n / by) * k.div_ceil(bx) + column / bx;
                    let w = rounded(weight[index].to_f32() * scales[scale], dtype);
                    sum = rounded(x, dtype).mul_add(w, sum);
                }
                rounded(sum, dtype)
            })
        })
        .collect()
}

#[test]
fn metal_fp8_linear_matches_cpu_reference() -> Result<()> {
    let device = Device::new_metal(0)?;
    let cases = [
        LinearCase {
            name: "scalar decode",
            shape: &[1, 128],
            n: 32,
            block: [32, 128],
            strided: false,
        },
        LinearCase {
            name: "channel batched decode",
            shape: &[4, 257],
            n: 35,
            block: [1, 257],
            strided: false,
        },
        LinearCase {
            name: "block partial batched decode",
            shape: &[3, 257],
            n: 129,
            block: [128, 128],
            strided: false,
        },
        LinearCase {
            name: "block strided prefill",
            shape: &[5, 37],
            n: 35,
            block: [16, 16],
            strided: true,
        },
        LinearCase {
            name: "row-block tiled prefill",
            shape: &[65, 257],
            n: 35,
            block: [1, 128],
            strided: false,
        },
        LinearCase {
            name: "block aligned short prefill",
            shape: &[8, 128],
            n: 64,
            block: [128, 128],
            strided: false,
        },
        LinearCase {
            name: "block aligned medium prefill",
            shape: &[16, 128],
            n: 64,
            block: [128, 128],
            strided: false,
        },
        LinearCase {
            name: "block aligned regular prefill",
            shape: &[32, 128],
            n: 64,
            block: [128, 128],
            strided: false,
        },
        LinearCase {
            name: "block aligned long prefill",
            shape: &[256, 384],
            n: 64,
            block: [128, 128],
            strided: false,
        },
        LinearCase {
            name: "channel aligned prefill",
            shape: &[65, 256],
            n: 64,
            block: [1, 256],
            strided: false,
        },
        LinearCase {
            name: "row-block aligned prefill",
            shape: &[129, 256],
            n: 64,
            block: [1, 128],
            strided: false,
        },
        LinearCase {
            name: "block aligned partial rows",
            shape: &[33, 128],
            n: 64,
            block: [128, 128],
            strided: false,
        },
        LinearCase {
            name: "block rank three",
            shape: &[2, 3, 129],
            n: 129,
            block: [128, 128],
            strided: false,
        },
        LinearCase {
            name: "row-block decode",
            shape: &[1, 257],
            n: 35,
            block: [1, 128],
            strided: false,
        },
    ];
    for case in &cases {
        let k = *case.shape.last().unwrap();
        let input_count = case.shape.iter().product::<usize>();
        let input: Vec<_> = (0..input_count)
            .map(|i| ((i * 17 + 3) % 43) as f32 / 19.0 - 1.0)
            .collect();
        let weight: Vec<_> = (0..case.n * k)
            .map(|i| {
                let magnitude = ((i * 29 + 3) % 97) as u8;
                F8E4M3::from_bits(magnitude | if i % 3 == 0 { 0x80 } else { 0 })
            })
            .collect();
        let [by, bx] = case.block;
        let scale_shape = vec![case.n.div_ceil(by), k.div_ceil(bx)];
        let scales: Vec<_> = (0..scale_shape.iter().product::<usize>())
            .map(|i| 0.03125 + (i % 7) as f32 * 0.0043)
            .collect();
        let mut padded_weight = vec![F8E4M3::from_f32(-448.0)];
        padded_weight.extend_from_slice(&weight);
        let weight_tensor = Tensor::from_vec(padded_weight, weight.len() + 1, &device)?
            .narrow(0, 1, weight.len())?
            .reshape((case.n, k))?;
        let scale_tensor = offset_tensor(&scales, &scale_shape, DType::F32, &device)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let input_tensor = if case.strided {
                let m = input_count / k;
                let transposed: Vec<_> = (0..k)
                    .flat_map(|column| {
                        let input = &input;
                        (0..m).map(move |row| input[row * k + column])
                    })
                    .collect();
                offset_tensor(&transposed, &[k, m], dtype, &device)?.t()?
            } else {
                offset_tensor(&input, case.shape, dtype, &device)?
            };
            let output = linear(
                &input_tensor,
                &weight_tensor,
                &scale_tensor,
                case.block,
                dtype,
            )?
            .expect("supported Metal FP8 linear");
            let mut shape = case.shape.to_vec();
            *shape.last_mut().unwrap() = case.n;
            assert_eq!(output.dims(), shape);
            assert_eq!(output.dtype(), dtype);
            let expected = reference(case, &input, &weight, &scales, dtype);
            assert_close(&output, &expected, &format!("{}, {dtype:?}", case.name))?;
        }
    }

    for (n, shape) in [(35, &[65, 37][..]), (64, &[65, 256][..])] {
        let k = shape[1];
        let case = LinearCase {
            name: "BF16 range",
            shape,
            n,
            block: [n, k],
            strided: false,
        };
        let input: Vec<_> = (0..65 * k)
            .map(|i| ((i * 13 % 19) as f32 - 9.0) * 1e20)
            .collect();
        let weight: Vec<_> = (0..case.n * k)
            .map(|i| F8E4M3::from_f32((i % 7) as f32 / 4.0 - 0.75))
            .collect();
        let scales = [1e-20];
        let output = linear(
            &offset_tensor(&input, case.shape, DType::BF16, &device)?,
            &Tensor::from_vec(weight.clone(), (case.n, k), &device)?,
            &Tensor::new(scales[0], &device)?,
            case.block,
            DType::BF16,
        )?
        .expect("supported Metal FP8 linear");
        let expected = reference(&case, &input, &weight, &scales, DType::BF16);
        assert_close(&output, &expected, case.name)?;
    }
    Ok(())
}

#[test]
fn metal_fp8_layer_preserves_bias_and_activation_metadata_semantics() -> Result<()> {
    let device = Device::new_metal(0)?;
    let weight = [1.0, -2.0, 0.5, -0.5, 1.5, 2.0].map(F8E4M3::from_f32);
    let scales = [0.37, 0.19];
    let bias = [0.11, -0.23];
    for shape in [&[2, 3][..], &[256, 3], &[257, 3], &[2, 129, 3]] {
        let case = LinearCase {
            name: "layer dispatch with bias",
            shape,
            n: 2,
            block: [1, 3],
            strided: false,
        };
        let input: Vec<_> = [0.31, -0.73, 1.13, -1.17, 0.43, 0.89]
            .into_iter()
            .cycle()
            .take(shape.iter().product())
            .collect();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let input_tensor = offset_tensor(&input, shape, dtype, &device)?;
            let weight_tensor = Tensor::from_vec(weight.to_vec(), (case.n, 3), &device)?;
            let scales_tensor = Tensor::from_vec(scales.to_vec(), (case.n, 1), &device)?;
            let bias_tensor = Tensor::from_vec(bias.to_vec(), case.n, &device)?.to_dtype(dtype)?;
            let expected: Vec<_> = reference(&case, &input, &weight, &scales, dtype)
                .into_iter()
                .enumerate()
                .map(|(i, value)| rounded(value + rounded(bias[i % case.n], dtype), dtype))
                .collect();
            for activation_scheme in [None, Some(Fp8ActivationScheme::Dynamic)] {
                let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
                    weight: weight_tensor.clone(),
                    weight_scale_inv: scales_tensor.clone(),
                    weight_block_size: vec![1, 3],
                    activation_scheme,
                    bias: Some(bias_tensor.clone()),
                    dequant_dtype: dtype,
                })?;
                assert_close(&layer.forward(&input_tensor)?, &expected, case.name)?;
            }
        }
    }
    Ok(())
}
