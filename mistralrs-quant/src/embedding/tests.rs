use std::sync::Arc;

use candle_core::{quantized::QTensor, DType, Device, Result, Tensor};
use candle_nn::Linear;
use float8::F8E4M3;

use super::*;
use crate::{
    AfqBits, AfqGroupSize, AfqLayer, BlockwiseFP8Linear, FP8Linear, GgufMatMul, MXFP4Layer,
    Nvfp4ActivationMode, Nvfp4Layer, Nvfp4LayerParts, QuantMethod, QuantMethodConfig,
    UnquantLinear,
};

const ROWS: usize = 37;
const ODD_COLUMNS: usize = 161;
const BLOCK_COLUMNS: usize = 160;
const K_COLUMNS: usize = 768;
const FP8_BLOCK: usize = 128;
const FP8_ROWS: usize = 300;
const FP8_COLUMNS: usize = 384;
const SEED: u64 = 0x9e37_79b9;
const TOLERANCE: f32 = 1e-5;
const SCALE: f64 = 45.25;
const E4M3_NAN_MASK: u8 = 0x7f;
const E8M0_BASE: u8 = 118;
const E8M0_SPAN: u8 = 16;
const NVFP4_SCALE_BASE: u8 = 0x28;
const NVFP4_SCALE_SPAN: u8 = 0x20;
const FP8_SCALE_GROUPS: [usize; 9] = [1, 2, 3, 4, 7, 8, 12, 16, 128];
const FP8_SCALE_TEST_COLUMNS: [usize; 4] = [16, 17, 24, 129];

fn ids(rows: usize) -> Vec<u32> {
    let last = rows as u32 - 1;
    vec![last, 0, 3, 3, 1, last, 2, 0]
}

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    }
}

fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut next = lcg(seed);
    (0..n).map(|_| next() as u8).collect()
}

fn values(rows: usize, columns: usize) -> Result<Tensor> {
    let mut next = lcg(SEED);
    let v = (0..rows * columns)
        .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
        .collect::<Vec<_>>();
    Tensor::from_vec(v, (rows, columns), &Device::Cpu)
}

fn e4m3(n: usize, seed: u64) -> Result<Tensor> {
    let codes = bytes(n, seed)
        .into_iter()
        .map(|b| {
            let b = if b & E4M3_NAN_MASK == E4M3_NAN_MASK {
                b - 1
            } else {
                b
            };
            F8E4M3::from_bits(b)
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(codes, n, &Device::Cpu)
}

fn ranged(n: usize, base: u8, span: u8, seed: u64) -> Vec<u8> {
    bytes(n, seed)
        .into_iter()
        .map(|b| base + b % span)
        .collect()
}

fn unquant(w: Tensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(Linear::new(w, None)),
    )?))
}

struct Case {
    name: String,
    layer: Arc<dyn QuantMethod>,
}

fn cases() -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    let mut push = |name: String, layer: Arc<dyn QuantMethod>| cases.push(Case { name, layer });
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for columns in [ODD_COLUMNS, BLOCK_COLUMNS] {
            push(
                format!("dense {dtype:?} x{columns}"),
                unquant(values(ROWS, columns)?.to_dtype(dtype)?)?,
            );
        }
    }
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ] {
        let columns = if dtype.block_size() == 256 {
            K_COLUMNS
        } else {
            BLOCK_COLUMNS
        };
        let q = QTensor::quantize(&values(ROWS, columns)?, dtype)?;
        push(
            format!("gguf {dtype:?}"),
            Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(q),
                b: None,
            })?),
        );
    }
    for bits in [
        AfqBits::Two,
        AfqBits::Three,
        AfqBits::Four,
        AfqBits::Six,
        AfqBits::Eight,
    ] {
        for group_size in [AfqGroupSize::Low, AfqGroupSize::High] {
            push(
                format!("afq {bits:?} {group_size:?}"),
                Arc::new(AfqLayer::new(QuantMethodConfig::Afq {
                    weight: values(ROWS, BLOCK_COLUMNS * 4)?,
                    bias: None,
                    bits,
                    group_size,
                })?),
            );
        }
    }
    push(
        "mxfp4".into(),
        Arc::new(MXFP4Layer::new(QuantMethodConfig::MXFP4 {
            blocks: Tensor::from_vec(
                bytes(ROWS * BLOCK_COLUMNS / 2, 1),
                (ROWS, BLOCK_COLUMNS / 2),
                &Device::Cpu,
            )?,
            scales: Tensor::from_vec(
                ranged(ROWS * BLOCK_COLUMNS / 32, E8M0_BASE, E8M0_SPAN, 2),
                (ROWS, BLOCK_COLUMNS / 32),
                &Device::Cpu,
            )?,
            bias: None,
        })?),
    );
    push(
        "mxfp8".into(),
        Arc::new(crate::mxfp8::Mxfp8Linear::from_parts(
            e4m3(ROWS * BLOCK_COLUMNS, 3)?.reshape((ROWS, BLOCK_COLUMNS))?,
            Tensor::from_vec(
                ranged(ROWS * BLOCK_COLUMNS / 32, E8M0_BASE, E8M0_SPAN, 4),
                (ROWS, BLOCK_COLUMNS / 32),
                &Device::Cpu,
            )?,
            None,
            DType::F32,
        )?),
    );
    push(
        "nvfp4".into(),
        Arc::new(Nvfp4Layer::from_parts(Nvfp4LayerParts {
            weight: Tensor::from_vec(
                bytes(ROWS * BLOCK_COLUMNS / 2, 5),
                (ROWS, BLOCK_COLUMNS / 2),
                &Device::Cpu,
            )?,
            scales: Tensor::from_vec(
                ranged(
                    ROWS * BLOCK_COLUMNS / 16,
                    NVFP4_SCALE_BASE,
                    NVFP4_SCALE_SPAN,
                    6,
                )
                .into_iter()
                .map(F8E4M3::from_bits)
                .collect(),
                (ROWS, BLOCK_COLUMNS / 16),
                &Device::Cpu,
            )?,
            global_scales: Tensor::from_vec(
                (0..ROWS).map(|r| 0.001 * (r + 1) as f32).collect(),
                ROWS,
                &Device::Cpu,
            )?,
            input_scale: None,
            activation: Nvfp4ActivationMode::None,
            bias: None,
            dtype: DType::F32,
        })?),
    );
    for columns in [ODD_COLUMNS, BLOCK_COLUMNS] {
        push(
            format!("fp8 tensor x{columns}"),
            Arc::new(FP8Linear::new(QuantMethodConfig::FP8 {
                lin: Linear::new(values(ROWS, columns)?, None),
                dtype: DType::F8E4M3,
            })?),
        );
    }
    let scale_rows = FP8_ROWS.div_ceil(FP8_BLOCK);
    let scale_columns = FP8_COLUMNS.div_ceil(FP8_BLOCK);
    push(
        "fp8 block128".into(),
        Arc::new(BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
            weight: e4m3(FP8_ROWS * FP8_COLUMNS, 7)?.reshape((FP8_ROWS, FP8_COLUMNS))?,
            weight_scale_inv: (values(scale_rows, scale_columns)?.abs()? + 0.01)?,
            bias: None,
            dequant_dtype: DType::F32,
            weight_block_size: vec![FP8_BLOCK, FP8_BLOCK],
            activation_scheme: None,
        })?),
    );
    Ok(cases)
}

fn reference(case: &Case, ids: &Tensor) -> Result<Tensor> {
    // GgufMatMul::dequantize_w round-trips through f16, so GGUF rows come from the stored tensor.
    let dense = case
        .layer
        .embedding_table()
        .and_then(|t| match (t.codec, t.storage) {
            (
                RowCodec::Dense(_),
                TableStorage::Device {
                    data: TableBytes::Tensor(w),
                    ..
                },
            ) => Some(w.clone()),
            _ => None,
        });
    let table = match dense {
        Some(w) => w.to_dtype(DType::F32)?,
        None => match case.layer.get_qtensor() {
            Some(q) => q.dequantize(&Device::Cpu)?,
            None => case.layer.dequantize_w()?.to_dtype(DType::F32)?,
        },
    };
    let columns = table.dim(1)?;
    table
        .index_select(&ids.flatten_all()?, 0)?
        .reshape((ids.dim(0)?, ids.dim(1)?, columns))
}

fn assert_close(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{name}");
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            a == e || (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "{name} element {i}: {a} != {e}"
        );
    }
    Ok(())
}

#[test]
fn host_gather_matches_dequantized_index_select_for_every_codec() -> Result<()> {
    for case in cases()? {
        let table = case
            .layer
            .embedding_table()
            .unwrap_or_else(|| panic!("{} has no row codec", case.name));
        let rows = table.rows;
        let ids = Tensor::from_vec(ids(rows), (2, 4), &Device::Cpu)?;
        let expected = reference(&case, &ids)?;
        let actual = case.layer.embedding_forward(&ids, DType::F32.into())?;
        assert_close(&case.name, &actual, &expected, TOLERANCE)?;
        let scaled = case
            .layer
            .embedding_forward(&ids, EmbeddingOutput::scaled(DType::BF16, SCALE))?;
        assert_eq!(scaled.dtype(), DType::BF16);
        assert_close(
            &format!("{} scaled", case.name),
            &scaled,
            &(expected * SCALE)?,
            1e-2,
        )?;
        let empty = case.layer.embedding_forward(
            &Tensor::zeros((0, 3), DType::U32, &Device::Cpu)?,
            DType::F32.into(),
        )?;
        assert_eq!(empty.dims(), &[0, 3, table.columns], "{}", case.name);
        assert!(
            table
                .gather_host(&[rows as u64], DType::F32.into())
                .is_err(),
            "{}",
            case.name
        );
    }
    Ok(())
}

fn sym_int_table(bits: usize) -> Result<(Tensor, Tensor, Vec<f32>)> {
    let group = 32;
    let row_bytes = BLOCK_COLUMNS * bits / 8;
    let data = bytes(ROWS * row_bytes, 11);
    let scales = (0..ROWS * BLOCK_COLUMNS / group)
        .map(|i| half::f16::from_f32(0.01 * (i % 13 + 1) as f32))
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(ROWS * BLOCK_COLUMNS);
    for r in 0..ROWS {
        for c in 0..BLOCK_COLUMNS {
            let q = if bits == 8 {
                f32::from(data[r * row_bytes + c] as i8)
            } else {
                f32::from((data[r * row_bytes + c / 2] >> (4 * (c % 2))) & 15) - 8.0
            };
            expected.push(q * scales[(r * BLOCK_COLUMNS + c) / group].to_f32());
        }
    }
    Ok((
        Tensor::from_vec(data, (ROWS, row_bytes), &Device::Cpu)?,
        Tensor::from_vec(scales, (ROWS, BLOCK_COLUMNS / group), &Device::Cpu)?,
        expected,
    ))
}

fn per_channel_fp8() -> Result<(Tensor, Tensor, Vec<f32>)> {
    let codes = e4m3(ROWS * ODD_COLUMNS, 13)?.reshape((ROWS, ODD_COLUMNS))?;
    let scales = Tensor::from_vec(
        (0..ROWS).map(|r| 0.5 + r as f32 * 0.25).collect(),
        ROWS,
        &Device::Cpu,
    )?;
    let expected = codes
        .flatten_all()?
        .to_vec1::<F8E4M3>()?
        .into_iter()
        .enumerate()
        .map(|(i, q)| q.to_f32() * (0.5 + (i / ODD_COLUMNS) as f32 * 0.25))
        .collect();
    Ok((codes, scales, expected))
}

fn check_table(
    name: &str,
    table: &EmbeddingTable<'_>,
    expected: &[f32],
    device: &Device,
) -> Result<()> {
    let ids_host = ids(table.rows);
    let gathered = ids_host
        .iter()
        .flat_map(|&r| {
            expected[r as usize * table.columns..(r as usize + 1) * table.columns].to_vec()
        })
        .collect::<Vec<_>>();
    let expected = Tensor::from_vec(gathered, (2, 4, table.columns), &Device::Cpu)?;
    let ids = Tensor::from_vec(ids_host, (2, 4), device)?;
    assert_close(
        name,
        &table
            .gather(&ids, DType::F32.into())?
            .to_device(&Device::Cpu)?,
        &expected,
        TOLERANCE,
    )
}

fn raw_u8(bytes: TableBytes<'_>, device: &Device) -> Result<Tensor> {
    let host = HostBytes::new(bytes)?;
    let bytes = host.bytes()?;
    Tensor::from_slice(bytes, bytes.len(), device)
}

fn check_codecs_without_a_layer(device: &Device) -> Result<()> {
    for bits in [4, 8] {
        let (data, scales, expected) = sym_int_table(bits)?;
        let (data, scales) = (data.to_device(device)?, scales.to_device(device)?);
        let codec = RowCodec::SymInt {
            bits,
            grid: ScaleGrid::per_row(ScaleType::F16, 32),
        };
        let table = EmbeddingTable::new(codec, ROWS, BLOCK_COLUMNS, &data).with_scales(&scales);
        check_table(&format!("sym int{bits}"), &table, &expected, device)?;
    }
    let (codes, scales, expected) = per_channel_fp8()?;
    let codes = raw_u8(TableBytes::Tensor(&codes), device)?;
    let scales = scales.to_device(device)?;
    let table = EmbeddingTable::new(RowCodec::Fp8(None), ROWS, ODD_COLUMNS, &codes)
        .with_row_scale(RowScale::PerRow(&scales));
    check_table("fp8 channel", &table, &expected, device)
}

#[test]
fn host_gather_handles_codecs_without_a_layer() -> Result<()> {
    check_codecs_without_a_layer(&Device::Cpu)
}

fn check_fp8_scale_boundaries(device: &Device) -> Result<()> {
    for group in FP8_SCALE_GROUPS {
        for columns in FP8_SCALE_TEST_COLUMNS {
            for rows_per_scale in [1, 4] {
                let scale_columns = columns.div_ceil(group);
                let scales: Vec<f32> = (0..ROWS.div_ceil(rows_per_scale) * scale_columns)
                    .map(|i| (i + 1) as f32 * 0.25)
                    .collect();
                let expected: Vec<f32> = (0..ROWS * columns)
                    .map(|i| {
                        scales[(i / columns / rows_per_scale) * scale_columns + i % columns / group]
                    })
                    .collect();
                let data = Tensor::from_vec(
                    vec![F8E4M3::from_f32(1.0).to_bits(); ROWS * columns],
                    (ROWS, columns),
                    device,
                )?;
                let scales = Tensor::from_vec(scales.clone(), scales.len(), device)?;
                let codec = RowCodec::Fp8(Some(ScaleGrid {
                    ty: ScaleType::F32,
                    group,
                    rows_per_scale,
                }));
                let table = EmbeddingTable::new(codec, ROWS, columns, &data).with_scales(&scales);
                check_table(
                    &format!("fp8 group={group} columns={columns} scale_rows={rows_per_scale}"),
                    &table,
                    &expected,
                    device,
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn host_gather_handles_fp8_scale_boundaries() -> Result<()> {
    check_fp8_scale_boundaries(&Device::Cpu)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
mod gpu {
    use super::*;

    struct DeviceCopy {
        data: Tensor,
        scales: Option<Tensor>,
        aux: Option<Tensor>,
        row_scale: Option<Tensor>,
    }

    fn opt_u8(t: Option<&Tensor>, device: &Device) -> Result<Option<Tensor>> {
        t.map(|t| raw_u8(TableBytes::Tensor(t), device)).transpose()
    }

    fn upload(table: &EmbeddingTable<'_>, device: &Device) -> Result<DeviceCopy> {
        let TableStorage::Device { data, scales, aux } = table.storage else {
            unreachable!()
        };
        let row_scale = match table.row_scale {
            RowScale::Scalar(t) | RowScale::PerRow(t) => Some(t.to_device(device)?),
            _ => None,
        };
        Ok(DeviceCopy {
            data: raw_u8(data, device)?,
            scales: opt_u8(scales, device)?,
            aux: opt_u8(aux, device)?,
            row_scale,
        })
    }

    fn on_device<'a>(table: &EmbeddingTable<'_>, copy: &'a DeviceCopy) -> EmbeddingTable<'a> {
        let mut out = EmbeddingTable::new(table.codec, table.rows, table.columns, &copy.data);
        if let Some(s) = &copy.scales {
            out = out.with_scales(s);
        }
        if let Some(a) = &copy.aux {
            out = out.with_aux(a);
        }
        out.row_scale = match (table.row_scale, &copy.row_scale) {
            (RowScale::Scalar(_), Some(t)) => RowScale::Scalar(t),
            (RowScale::PerRow(_), Some(t)) => RowScale::PerRow(t),
            (RowScale::Value(v), _) => RowScale::Value(v),
            _ => RowScale::None,
        };
        out
    }

    fn check_every_codec(device: Device) -> Result<()> {
        for case in cases()? {
            let table = case.layer.embedding_table().unwrap();
            let copy = upload(&table, &device)?;
            let cuda = on_device(&table, &copy);
            let rows = table.rows;
            let host_ids = Tensor::from_vec(ids(rows), (2, 4), &Device::Cpu)?;
            let expected = reference(&case, &host_ids)?;
            for ids in [
                host_ids.to_device(&device)?,
                host_ids.to_dtype(DType::I64)?.to_device(&device)?,
            ] {
                let actual = cuda.gather(&ids, DType::F32.into())?;
                assert_close(
                    &case.name,
                    &actual.to_device(&Device::Cpu)?,
                    &expected,
                    TOLERANCE,
                )?;
            }
            let ids = host_ids.to_device(&device)?;
            for dtype in [DType::F16, DType::BF16] {
                let actual = cuda.gather(&ids, EmbeddingOutput::scaled(dtype, SCALE))?;
                assert_eq!(actual.dtype(), dtype);
                assert_close(
                    &format!("{} {dtype:?}", case.name),
                    &actual.to_device(&Device::Cpu)?,
                    &(&expected * SCALE)?.to_dtype(dtype)?.to_dtype(DType::F32)?,
                    1e-2,
                )?;
            }
            let past = Tensor::from_vec(vec![rows as u32], (1, 1), &device)?;
            let zeros = cuda.gather(&past, DType::F32.into())?;
            assert_eq!(
                zeros.abs()?.sum_all()?.to_scalar::<f32>()?,
                0.0,
                "{}",
                case.name
            );
            let empty = cuda.gather(
                &Tensor::zeros((0, 3), DType::U32, &device)?,
                DType::F32.into(),
            )?;
            assert_eq!(empty.dims(), &[0, 3, table.columns]);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gather_matches_dequantized_index_select_for_every_codec() -> Result<()> {
        check_every_codec(Device::new_cuda(0)?)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_gather_matches_dequantized_index_select_for_every_codec() -> Result<()> {
        check_every_codec(Device::new_metal(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gather_handles_codecs_without_a_layer() -> Result<()> {
        check_codecs_without_a_layer(&Device::new_cuda(0)?)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_gather_handles_codecs_without_a_layer() -> Result<()> {
        check_codecs_without_a_layer(&Device::new_metal(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gather_handles_fp8_scale_boundaries() -> Result<()> {
        check_fp8_scale_boundaries(&Device::new_cuda(0)?)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_gather_handles_fp8_scale_boundaries() -> Result<()> {
        check_fp8_scale_boundaries(&Device::new_metal(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_layers_gather_through_the_hook() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let w = values(ROWS, BLOCK_COLUMNS)?;
        let host_ids = Tensor::from_vec(ids(ROWS), (2, 4), &Device::Cpu)?;
        let layers: Vec<(&str, Arc<dyn QuantMethod>)> = vec![
            (
                "bf16",
                unquant(w.to_dtype(DType::BF16)?.to_device(&device)?)?,
            ),
            (
                "q4k",
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(QTensor::quantize_onto(
                        &values(ROWS, K_COLUMNS)?,
                        GgmlDType::Q4K,
                        &device,
                    )?),
                    b: None,
                })?),
            ),
        ];
        for (name, layer) in layers {
            let expected = layer
                .dequantize_w()?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .index_select(&host_ids.flatten_all()?, 0)?;
            let actual =
                layer.embedding_forward(&host_ids.to_device(&device)?, DType::F32.into())?;
            assert!(actual.device().is_cuda());
            assert_close(
                name,
                &actual.to_device(&Device::Cpu)?.flatten_to(1)?,
                &expected,
                1e-3,
            )?;
        }
        Ok(())
    }
}
