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
const STAGED_SHARD_ROWS: [usize; 2] = [3, 4];
const STAGED_COLUMNS: usize = 16;
const STAGED_SCALE_GROUP: usize = 8;
const STAGED_ROWS_PER_SCALE: usize = 2;
const STAGED_IDS: [u64; 8] = [6, 0, 2, 3, 3, 4, 1, 5];
#[cfg(feature = "cuda")]
const STREAM_ROWS: usize = 262_144;
#[cfg(feature = "cuda")]
const STREAM_COLUMNS: usize = 64;

#[cfg(feature = "cuda")]
#[test]
fn cuda_gather_keeps_cross_stream_input_alive() -> Result<()> {
    use candle_core::{CudaStorage, Storage};

    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let consumer = cuda.cuda_stream();
    let producer = consumer
        .context()
        .new_stream_with_priority(-1)
        .map_err(candle_core::Error::wrap)?;
    let ids = Tensor::from_vec(
        (0..STREAM_ROWS as u32).collect::<Vec<_>>(),
        STREAM_ROWS,
        &device,
    )?;
    let values = vec![1f32; STREAM_ROWS * STREAM_COLUMNS];
    let data = producer
        .clone_htod(&values)
        .map_err(candle_core::Error::wrap)?;
    let weight = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(data, cuda.clone())),
        (STREAM_ROWS, STREAM_COLUMNS),
    ));
    let _delay = crate::utils::cuda_test::delay_stream(&consumer)?;
    let output = EmbeddingTable::new(
        RowCodec::Dense(DType::F32),
        STREAM_ROWS,
        STREAM_COLUMNS,
        &weight,
    )
    .gather(&ids, DType::F32.into())?;
    drop(weight);
    let _replacement = producer
        .alloc_zeros::<f32>(values.len())
        .map_err(candle_core::Error::wrap)?;
    producer.synchronize().map_err(candle_core::Error::wrap)?;
    assert!(output
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|&v| v == 1.));
    Ok(())
}

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

fn check_staged_codecs(device: &Device) -> Result<()> {
    for case in cases()? {
        let table = case.layer.embedding_table().unwrap();
        let host_ids = Tensor::from_vec(ids(table.rows), (2, 4), &Device::Cpu)?;
        let expected = reference(&case, &host_ids)?.reshape((8, table.columns))?;
        let ids = ids(table.rows)
            .into_iter()
            .map(u64::from)
            .collect::<Vec<_>>();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let actual =
                table.gather_staged(&ids, EmbeddingOutput::scaled(dtype, SCALE), device)?;
            assert_eq!(actual.dtype(), dtype, "{}", case.name);
            assert!(actual.device().same_device(device));
            assert_close(
                &format!("staged {} {dtype:?}", case.name),
                &actual.to_device(&Device::Cpu)?,
                &(&expected * SCALE)?.to_dtype(dtype)?.to_dtype(DType::F32)?,
                if dtype == DType::F32 { TOLERANCE } else { 1e-2 },
            )?;
        }
        assert!(table
            .gather_staged(&[table.rows as u64], DType::F32.into(), device)
            .is_err());
        assert_eq!(
            table.gather_staged(&[], DType::F32.into(), device)?.dims(),
            &[0, table.columns]
        );
    }
    for bits in [4, 8] {
        let (data, scales, expected) = sym_int_table(bits)?;
        let table = EmbeddingTable::new(
            RowCodec::SymInt {
                bits,
                grid: ScaleGrid::per_row(ScaleType::F16, 32),
            },
            ROWS,
            BLOCK_COLUMNS,
            &data,
        )
        .with_scales(&scales);
        let ids = ids(ROWS).into_iter().map(u64::from).collect::<Vec<_>>();
        let expected = ids
            .iter()
            .flat_map(|&row| {
                expected[row as usize * BLOCK_COLUMNS..(row as usize + 1) * BLOCK_COLUMNS].to_vec()
            })
            .collect::<Vec<_>>();
        assert_close(
            "staged symmetric",
            &table
                .gather_staged(&ids, DType::F32.into(), device)?
                .to_device(&Device::Cpu)?,
            &Tensor::from_vec(expected, (ids.len(), BLOCK_COLUMNS), &Device::Cpu)?,
            TOLERANCE,
        )?;
    }
    Ok(())
}

fn check_staged_shared_rows(device: &Device) -> Result<()> {
    let grid = ScaleGrid {
        ty: ScaleType::F32,
        group: STAGED_SCALE_GROUP,
        rows_per_scale: STAGED_ROWS_PER_SCALE,
    };
    let columns = STAGED_COLUMNS;
    let scale_columns = columns / grid.group;
    let rows = STAGED_SHARD_ROWS.iter().sum::<usize>();
    let scales = (0..rows.div_ceil(grid.rows_per_scale) * scale_columns)
        .map(|i| (i + 1) as f32 * 0.5)
        .collect::<Vec<_>>();
    let scales = Tensor::from_vec(
        scales,
        (rows.div_ceil(grid.rows_per_scale), scale_columns),
        &Device::Cpu,
    )?;
    let aux = (&scales + 0.25)?;
    let multipliers = Tensor::from_vec(
        (0..rows).map(|r| (r + 1) as f32).collect::<Vec<_>>(),
        rows,
        &Device::Cpu,
    )?;
    let scalar = Tensor::new(0.5f32, &Device::Cpu)?;
    for (codec, code, value) in [
        (RowCodec::Fp4(grid), 0x22u8, 1.0),
        (RowCodec::Fp8(Some(grid)), 0x38, 1.0),
        (RowCodec::Afq { bits: 3, grid }, 0xff, 7.0),
        (RowCodec::SymInt { bits: 4, grid }, 0x99, 1.0),
        (RowCodec::SymInt { bits: 8, grid }, 1, 1.0),
    ] {
        let data = Tensor::from_vec(
            vec![code; rows * codec.row_bytes(columns)],
            (rows, codec.row_bytes(columns)),
            &Device::Cpu,
        )?;
        let table = EmbeddingTable::new(codec, rows, columns, &data)
            .with_scales(&scales)
            .with_aux(&aux);
        for row_scale in [
            RowScale::None,
            RowScale::Value(0.5),
            RowScale::Scalar(&scalar),
            RowScale::PerRow(&multipliers),
        ] {
            let table = table.with_row_scale(row_scale);
            let expected = STAGED_IDS
                .iter()
                .flat_map(|&row| {
                    (0..columns).map(move |column| {
                        let scale = ((row as usize / grid.rows_per_scale * scale_columns
                            + column / grid.group)
                            + 1) as f32
                            * 0.5;
                        let decoded = value * scale
                            + if matches!(codec, RowCodec::Afq { .. }) {
                                scale + 0.25
                            } else {
                                0.0
                            };
                        let multiplier = match row_scale {
                            RowScale::None => 1.0,
                            RowScale::PerRow(_) => (row + 1) as f32,
                            _ => 0.5,
                        };
                        decoded * multiplier * 2.0
                    })
                })
                .collect::<Vec<_>>();
            assert_close(
                &format!("staged shared rows {codec:?} {row_scale:?}"),
                &table
                    .gather_staged(
                        &STAGED_IDS,
                        EmbeddingOutput::scaled(DType::F32, 2.0),
                        device,
                    )?
                    .to_device(&Device::Cpu)?,
                &Tensor::from_vec(expected, (STAGED_IDS.len(), columns), &Device::Cpu)?,
                TOLERANCE,
            )?;
        }
    }
    let owner = Arc::new(STAGED_SHARD_ROWS.map(|rows| {
        (
            vec![0x22u8; rows * columns / 2],
            (0..rows.div_ceil(grid.rows_per_scale) * scale_columns)
                .map(|i| (rows * 10 + i) as f32)
                .collect::<Vec<_>>(),
        )
    }));
    let mut first_row = 0;
    let shards = owner
        .iter()
        .zip(STAGED_SHARD_ROWS)
        .map(|((data, scales), rows)| {
            let shard = MappedShard {
                data: data.as_ptr(),
                scales: scales.as_ptr().cast(),
                first_row,
                rows,
            };
            first_row += rows;
            shard
        })
        .collect();
    let mapped = unsafe { MappedRows::new(shards, owner) };
    let table = EmbeddingTable::mapped(RowCodec::Fp4(grid), columns, &mapped)
        .with_row_scale(RowScale::Value(0.5));
    let expected = STAGED_IDS
        .iter()
        .flat_map(|&row| {
            (0..columns).map(move |column| {
                let (local, shard_rows) = if row < STAGED_SHARD_ROWS[0] as u64 {
                    (row as usize, STAGED_SHARD_ROWS[0])
                } else {
                    (row as usize - STAGED_SHARD_ROWS[0], STAGED_SHARD_ROWS[1])
                };
                (shard_rows * 10
                    + local / grid.rows_per_scale * scale_columns
                    + column / grid.group) as f32
            })
        })
        .collect::<Vec<_>>();
    let expected = Tensor::from_vec(expected, (STAGED_IDS.len(), columns), &Device::Cpu)?;
    for dtype in [DType::F32, DType::BF16] {
        assert_close(
            "staged mapped shared scales",
            &table
                .gather_staged(&STAGED_IDS, EmbeddingOutput::scaled(dtype, 2.0), device)?
                .to_device(&Device::Cpu)?,
            &expected.to_dtype(dtype)?.to_dtype(DType::F32)?,
            0.0,
        )?;
    }
    Ok(())
}

#[test]
fn host_staged_gather_preserves_codecs_and_shared_rows() -> Result<()> {
    check_staged_codecs(&Device::Cpu)?;
    check_staged_shared_rows(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_staged_gather_preserves_codecs_and_shared_rows() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    check_staged_codecs(&device)?;
    check_staged_shared_rows(&device)
}

#[cfg(feature = "metal")]
#[test]
fn metal_staged_gather_preserves_codecs_and_shared_rows() -> Result<()> {
    let device = Device::new_metal(0)?;
    check_staged_codecs(&device)?;
    check_staged_shared_rows(&device)
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
