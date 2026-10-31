use super::*;
#[cfg(feature = "cuda")]
use crate::GgufArchive;
use crate::ShardedVarBuilder;
use candle_core::IndexOp;

const TABLE_PACKING: EmbeddingPacking = EmbeddingPacking::Symmetric {
    preferred_bits: 8,
    fallback_bits: 4,
};
const PLANNED_RESIDENT_BITS: usize = 4;
const DEVICE_TEST_ROWS: usize = 2;
const DEVICE_TEST_COLUMNS: usize = 2;
#[cfg(feature = "cuda")]
const RESIDENT_BITS: [usize; 2] = [8, 4];
#[cfg(feature = "cuda")]
const GGUF_TABLE_NAME: &str = "embedding.weight";
#[cfg(feature = "cuda")]
const MAPPED_RELEASE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
#[cfg(feature = "cuda")]
const MAPPED_RELEASE_POLL: std::time::Duration = std::time::Duration::from_millis(1);
#[cfg(feature = "cuda")]
const LIFETIME_ROWS: usize = 1024;
#[cfg(feature = "cuda")]
const LIFETIME_COLUMNS: usize = 64;

fn device_test_rows() -> MappedRows {
    let owner = Arc::new([1f32, 2., 3., 4.]);
    let shard = MappedShard {
        data: owner.as_ptr().cast(),
        scales: std::ptr::null(),
        first_row: 0,
        rows: DEVICE_TEST_ROWS,
    };
    unsafe { MappedRows::new(vec![shard], owner) }
}

#[test]
fn mapped_device_index_rejects_cpu() {
    let rows = device_test_rows();
    assert!(
        EmbeddingTable::mapped(RowCodec::Dense(DType::F32), DEVICE_TEST_COLUMNS, &rows,)
            .gathers_on(&Device::Cpu)
    );
    assert!(rows.with_device_index(&Device::Cpu).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn mapped_device_index_dispatch_checks_device() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let other_device = Device::new_cuda_with_stream(0)?;
    let rows = device_test_rows();
    let ids = Tensor::new(&[1u32, 0], &other_device)?;
    let gather = |rows: &MappedRows| -> Result<()> {
        let table = EmbeddingTable::mapped(RowCodec::Dense(DType::F32), DEVICE_TEST_COLUMNS, rows);
        assert!(!table.gathers_on(&other_device));
        let output = table.gather(&ids, DType::F32.into())?;
        assert!(output.device().same_device(&other_device));
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![3., 4.], vec![1., 2.]]);
        Ok(())
    };
    gather(&rows)?;
    if !device_reads_pageable_memory(&device)? {
        assert!(rows.with_device_index(&device).is_err());
        return Ok(());
    }
    let rows = rows.with_device_index(&device)?;
    assert!(
        EmbeddingTable::mapped(RowCodec::Dense(DType::F32), DEVICE_TEST_COLUMNS, &rows,)
            .gathers_on(&device)
    );
    gather(&rows)
}

#[cfg(feature = "cuda")]
fn mapped_lifetime_table(
    device: &Device,
) -> Result<(ManagedEmbedding, std::sync::Weak<MmapedSafetensors>)> {
    let file = tempfile::NamedTempFile::new().map_err(candle_core::Error::wrap)?;
    candle_core::safetensors::save(
        &std::collections::HashMap::from([(
            "weight".to_owned(),
            Tensor::ones((LIFETIME_ROWS, LIFETIME_COLUMNS), DType::F32, &Device::Cpu)?,
        )]),
        file.path(),
    )?;
    let raw = Arc::new(unsafe { MmapedSafetensors::new(file.path())? });
    let owner = Arc::downgrade(&raw);
    let source = EmbeddingSource::dense_shards(raw, vec!["weight".to_owned()], LIFETIME_COLUMNS)?;
    Ok((ManagedEmbedding::new(source, device)?, owner))
}

#[cfg(feature = "cuda")]
#[test]
fn mapped_table_drop_waits_for_eager_cuda_reads() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !device_reads_pageable_memory(&device)? {
        return Ok(());
    }
    let (table, owner) = mapped_lifetime_table(&device)?;
    let ids = Tensor::new(&[0u32, (LIFETIME_ROWS - 1) as u32], &device)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let _delay = crate::utils::cuda_test::delay_stream(&stream)?;
    let output = table.gather(&ids, DType::F32.into(), EmbeddingGatherMode::Decode)?;
    drop(table);
    assert!(owner.upgrade().is_none());
    assert_eq!(
        output.to_vec2::<f32>()?,
        vec![vec![1.; LIFETIME_COLUMNS]; 2]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn captured_mapped_table_outlives_its_owner_until_graph_release() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    let device = Device::new_cuda_with_stream(0)?;
    if !device_reads_pageable_memory(&device)? {
        return Ok(());
    }
    let (table, owner) = mapped_lifetime_table(&device)?;
    let ids = Tensor::new(&[0u32, (LIFETIME_ROWS - 1) as u32], &device)?;
    let gather = || table.gather(&ids, DType::F32.into(), EmbeddingGatherMode::Decode);
    drop(gather()?);
    device.synchronize()?;
    let stream = device.as_cuda_device()?.cuda_stream();
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
        return Err(candle_core::Error::wrap(error));
    }
    let output = gather().and_then(|first| first + gather()?);
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = output?;
    let graph = graph
        .map_err(candle_core::Error::wrap)?
        .expect("captured mapped gathers");
    drop(table);
    assert!(owner.upgrade().is_some());
    for _ in 0..2 {
        graph.launch().map_err(candle_core::Error::wrap)?;
        assert_eq!(
            output.to_vec2::<f32>()?,
            vec![vec![2.; LIFETIME_COLUMNS]; 2]
        );
    }
    drop(output);
    drop(graph);
    let started = std::time::Instant::now();
    while owner.upgrade().is_some() && started.elapsed() < MAPPED_RELEASE_TIMEOUT {
        std::thread::sleep(MAPPED_RELEASE_POLL);
    }
    assert!(
        owner.upgrade().is_none(),
        "graph release kept the mapped table alive"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(feature = "cuda")]
fn managed_table(
    storage: TableSource,
    columns: usize,
    device: &Device,
) -> Result<ManagedEmbedding> {
    ManagedEmbedding::new(
        EmbeddingSource {
            storage,
            columns,
            packing: TABLE_PACKING,
        },
        device,
    )
}

fn nvfp4_test_table(
    vb: &ShardedVarBuilder,
    columns: usize,
    device: &Device,
) -> Result<ManagedEmbedding> {
    let raw = vb.raw_safetensors().unwrap();
    let prefix = vb.prefix();
    let shards = (0..2)
        .map(|shard| {
            (
                format!("{prefix}.shard_{shard}.weight"),
                format!("{prefix}.shard_{shard}.weight_scale"),
            )
        })
        .collect();
    ManagedEmbedding::new(
        EmbeddingSource::nvfp4(Nvfp4Embedding::new(
            raw,
            shards,
            columns,
            &format!("{prefix}.weight_scale_2"),
        )?),
        device,
    )
}

#[test]
fn source_metadata_tracks_packing_and_real_shard_sizes() -> Result<()> {
    const COLUMNS: usize = 64;
    const ROWS: [usize; 2] = [3, 5];
    let file = tempfile::NamedTempFile::new().map_err(candle_core::Error::wrap)?;
    let names = vec!["part_a".to_owned(), "part_b".to_owned()];
    candle_core::safetensors::save(
        &std::collections::HashMap::from([
            (
                names[0].clone(),
                Tensor::ones((ROWS[0], COLUMNS), DType::F16, &Device::Cpu)?,
            ),
            (
                names[1].clone(),
                Tensor::ones((ROWS[1], COLUMNS), DType::F16, &Device::Cpu)?.affine(2.0, 0.0)?,
            ),
        ]),
        file.path(),
    )?;
    let raw = Arc::new(unsafe { MmapedSafetensors::new(file.path())? });
    let source = EmbeddingSource::dense_shards(raw.clone(), names.clone(), COLUMNS)?;
    let metadata = source.metadata();
    assert_eq!(metadata.rows, ROWS.iter().sum::<usize>());
    assert_eq!(metadata.columns, COLUMNS);
    assert_eq!(
        metadata.device_bytes,
        ROWS.iter().sum::<usize>() * COLUMNS * 2
    );
    assert_eq!(metadata.host_backing, EmbeddingBacking::FileMapped);
    let table = ManagedEmbedding::new(source, &Device::Cpu)?;
    let ids = Tensor::from_slice(&[7u32, 0, 3, 2], (2, 2), &Device::Cpu)?;
    let expected = table.gather_host(&[7, 0, 3, 2], &Device::Cpu)?;
    for mode in [EmbeddingGatherMode::Prefill, EmbeddingGatherMode::Decode] {
        let actual = table.gather(&ids, DType::F16.into(), mode)?;
        assert_eq!(actual.dims(), &[2, 2, COLUMNS]);
        assert_eq!(
            actual.flatten_all()?.to_vec1::<half::f16>()?,
            expected.flatten_all()?.to_vec1::<half::f16>()?
        );
    }
    let packed = EmbeddingSource::dense_shards(raw, names, COLUMNS)?.with_packing(TABLE_PACKING)?;
    let metadata = packed.metadata();
    assert_eq!(metadata.host_backing, EmbeddingBacking::Owned);
    assert_eq!(
        metadata.device_bytes,
        resident_bytes(ROWS.iter().sum(), COLUMNS, PLANNED_RESIDENT_BITS)
    );
    Ok(())
}

#[test]
fn nvfp4_shards_load_without_requantizing() -> Result<()> {
    const HEAD_DIM: usize = 160;
    const SHARD_ROWS: usize = 3;
    const SHARDS: usize = 2;
    const PREFIX: &str = "encoder.embedding";
    const GLOBAL_SCALE: f32 = 0.251;
    let mut tensors = std::collections::HashMap::new();
    for shard in 0..SHARDS {
        tensors.insert(
            format!("{PREFIX}.shard_{shard}.weight"),
            Tensor::from_vec(
                (0..SHARD_ROWS * HEAD_DIM / 2)
                    .map(|i| {
                        let value = ((i / (HEAD_DIM / 2) + shard * SHARD_ROWS) % 4 + 2) as u8;
                        value | ((value + 1) << 4)
                    })
                    .collect::<Vec<_>>(),
                (SHARD_ROWS, HEAD_DIM / 2),
                &Device::Cpu,
            )?,
        );
        tensors.insert(
            format!("{PREFIX}.shard_{shard}.weight_scale"),
            Tensor::ones((SHARD_ROWS, HEAD_DIM / 16), DType::F32, &Device::Cpu)?
                .to_dtype(DType::F8E4M3)?,
        );
    }
    tensors.insert(
        format!("{PREFIX}.weight_scale_2"),
        Tensor::from_slice(&[GLOBAL_SCALE], 1, &Device::Cpu)?,
    );
    let file = tempfile::NamedTempFile::new().map_err(candle_core::Error::wrap)?;
    candle_core::safetensors::save(&tensors, file.path())?;
    let vb = unsafe {
        crate::ShardedSafeTensors::sharded(
            &[file.path()],
            DType::BF16,
            &Device::Cpu,
            None,
            Arc::new(|_| true),
        )?
    }
    .pp("encoder")
    .pp("embedding");
    let table = nvfp4_test_table(&vb, HEAD_DIM, &Device::Cpu)?;
    assert!(matches!(table.source, TableSource::Nvfp4(_)));
    let ids = [5u64, 0, 3, 2, 3, 1];
    let expected = table.gather_host(&ids, &Device::Cpu)?;
    let first = expected.i(0)?.to_vec1::<f32>()?;
    assert_eq!(first[0], 1.5 * GLOBAL_SCALE);
    assert_eq!(first[1], 2.0 * GLOBAL_SCALE);
    assert_ne!(
        expected
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        expected
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
    );
    #[cfg(feature = "cuda")]
    {
        let device = Device::new_cuda_with_stream(0)?;
        let mut table = nvfp4_test_table(&vb, HEAD_DIM, &device)?;
        let ids = Tensor::from_slice(&ids.map(|i| i as i64), (2, 3), &device)?;
        if table.mapped.has_device_index() {
            for prepared in [false, true] {
                if prepared {
                    table.prepare_mapped(&device, SHARD_ROWS, 0)?;
                }
                for dtype in [DType::BF16, DType::F16, DType::F32] {
                    let expected = expected
                        .to_dtype(dtype)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for mode in [EmbeddingGatherMode::Decode, EmbeddingGatherMode::Prefill] {
                        let actual = table.gather(&ids, dtype.into(), mode)?;
                        assert_eq!(actual.dtype(), dtype);
                        assert_eq!(actual.dims(), &[2, 3, HEAD_DIM]);
                        assert_eq!(
                            actual
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?,
                            expected
                        );
                    }
                }
            }
        }
        let bytes = SHARD_ROWS * SHARDS * (HEAD_DIM / 2 + HEAD_DIM / 16);
        table.make_resident(
            &device,
            WeightAllocationBudget {
                persistent_bytes: bytes,
                execution_bytes: bytes + RESIDENT_RESERVE_BYTES,
            },
        )?;
        let Some(resident) = &table.resident else {
            panic!("packed table was not made resident");
        };
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let actual = resident
                .table(HEAD_DIM)?
                .gather(&ids, dtype.into())?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = expected
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
const GGML_TYPE_IQ4_NL: u32 = 20;
#[cfg(feature = "cuda")]
const IQ4_NL_BLOCK: usize = 32;
#[cfg(feature = "cuda")]
const IQ4_NL_BLOCK_BYTES: usize = 18;
#[cfg(feature = "cuda")]
const KVALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[cfg(feature = "cuda")]
fn iq4_nl_row_bytes(head_dim: usize) -> usize {
    head_dim / IQ4_NL_BLOCK * IQ4_NL_BLOCK_BYTES
}

/// ggml `dequantize_row_iq4_nl` for one row.
#[cfg(feature = "cuda")]
fn dequantize_iq4_nl_row(src: &[u8], out: &mut [f32]) {
    let (blocks, _) = src.as_chunks::<IQ4_NL_BLOCK_BYTES>();
    let (dsts, _) = out.as_chunks_mut::<IQ4_NL_BLOCK>();
    for (block, dst) in blocks.iter().zip(dsts) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let (lo, hi) = dst.split_at_mut(IQ4_NL_BLOCK / 2);
        for (j, &byte) in block[2..].iter().enumerate() {
            lo[j] = d * f32::from(KVALUES_IQ4_NL[usize::from(byte & 0xf)]);
            hi[j] = d * f32::from(KVALUES_IQ4_NL[usize::from(byte >> 4)]);
        }
    }
}

#[cfg(feature = "cuda")]
const GGUF_TEST_ALIGNMENT: usize = 32;

#[cfg(feature = "cuda")]
fn mapped_bytes(bytes: &[u8]) -> Result<memmap2::Mmap> {
    use std::io::Write;
    let mut file = tempfile::tempfile().map_err(candle_core::Error::wrap)?;
    file.write_all(bytes).map_err(candle_core::Error::wrap)?;
    unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(candle_core::Error::wrap)
}

#[cfg(feature = "cuda")]
fn mapped_iq4_table(
    data: &[u8],
    rows: usize,
    head_dim: usize,
    device: &Device,
) -> Result<ManagedEmbedding> {
    use std::io::Write;
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(GGUF_TABLE_NAME.len() as u64).to_le_bytes());
    bytes.extend_from_slice(GGUF_TABLE_NAME.as_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&(head_dim as u64).to_le_bytes());
    bytes.extend_from_slice(&(rows as u64).to_le_bytes());
    bytes.extend_from_slice(&GGML_TYPE_IQ4_NL.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.resize(
        bytes.len().div_ceil(GGUF_TEST_ALIGNMENT) * GGUF_TEST_ALIGNMENT,
        0,
    );
    bytes.extend_from_slice(data);
    bytes.resize(
        bytes.len().div_ceil(GGUF_TEST_ALIGNMENT) * GGUF_TEST_ALIGNMENT,
        0,
    );
    let mut file = tempfile::NamedTempFile::new().map_err(candle_core::Error::wrap)?;
    file.write_all(&bytes).map_err(candle_core::Error::wrap)?;
    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    ManagedEmbedding::new(
        EmbeddingSource::gguf(GgufEmbedding::new(archive, GGUF_TABLE_NAME)?),
        device,
    )
}

#[cfg(feature = "cuda")]
#[test]
fn resident_quantization_matches_host() -> Result<()> {
    const HEAD_DIM: usize = 160;
    const ROWS: usize = 37;
    const CHUNK_ROWS: usize = 11;
    const GUARD_BYTE: u8 = 0xa5;
    let device = Device::new_cuda(0)?;
    for dtype in [DType::F16, DType::BF16] {
        for bits in RESIDENT_BITS {
            let qmax = if bits == 8 { 127.0 } else { 7.0 };
            let row_bytes = HEAD_DIM * bits / 8;
            let groups = HEAD_DIM / RESIDENT_GROUP;
            let values = (0..ROWS * HEAD_DIM)
                .map(|i| match i / RESIDENT_GROUP {
                    0 => 0.0,
                    1 => match i % RESIDENT_GROUP {
                        0 => qmax,
                        1 => -qmax,
                        j => (j % 8) as f32 / 2.0 - 1.5,
                    },
                    2 => (i as f32 - 80.0) * 1e-9,
                    _ => {
                        let scale = [1e-5, 1e-2, 1.0, 128.0][i / RESIDENT_GROUP % 4];
                        ((splitmix64(i as u64) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
                    }
                })
                .collect::<Vec<_>>();
            let source =
                Tensor::from_vec(values, (ROWS, HEAD_DIM), &Device::Cpu)?.to_dtype(dtype)?;
            let values = source
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let source = source.to_device(&device)?;
            let data = Tensor::from_vec(
                vec![GUARD_BYTE; (ROWS + 2) * row_bytes],
                (ROWS + 2, row_bytes),
                &device,
            )?;
            let scales = Tensor::zeros((ROWS + 2, groups), DType::F16, &device)?;
            for first in (0..ROWS).step_by(CHUNK_ROWS) {
                let count = CHUNK_ROWS.min(ROWS - first);
                super::super::packing::quantize(
                    &source.narrow(0, first, count)?,
                    &data.narrow(0, first + 1, count)?,
                    &scales.narrow(0, first + 1, count)?,
                    bits,
                )?;
            }
            let got = data.flatten_all()?.to_vec1::<u8>()?;
            let got_scales = scales.flatten_all()?.to_vec1::<half::f16>()?;
            assert!(got[..row_bytes].iter().all(|v| *v == GUARD_BYTE));
            assert!(got[(ROWS + 1) * row_bytes..]
                .iter()
                .all(|v| *v == GUARD_BYTE));
            assert!(got_scales[..groups].iter().all(|v| *v == half::f16::ZERO));
            assert!(got_scales[(ROWS + 1) * groups..]
                .iter()
                .all(|v| *v == half::f16::ZERO));
            let mut expected = vec![0u8; ROWS * row_bytes];
            for (g, values) in values.as_chunks::<RESIDENT_GROUP>().0.iter().enumerate() {
                let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = half::f16::from_f32(amax / qmax);
                assert_eq!(got_scales[groups + g], scale, "{dtype:?} Q{bits} group {g}");
                let inv = if scale.to_f32() > 0.0 {
                    1.0 / scale.to_f32()
                } else {
                    0.0
                };
                for (j, value) in values.iter().enumerate() {
                    let q = (value * inv).round().clamp(-qmax, qmax) as i32;
                    let i = g * RESIDENT_GROUP + j;
                    if bits == 8 {
                        expected[i] = q as i8 as u8;
                    } else {
                        expected[i / 2] |= ((q + 8) as u8) << (4 * (i % 2));
                    }
                }
            }
            assert_eq!(
                &got[row_bytes..(ROWS + 1) * row_bytes],
                &expected,
                "{dtype:?} Q{bits}"
            );
            if device_reads_pageable_memory(&device)? {
                let mapped_data = mapped_bytes(&got)?;
                let mapped_scales = mapped_bytes(bytemuck::cast_slice(&got_scales))?;
                const LOOKUPS: [i64; 6] = [0, 36, 5, 5, 17, 1];
                let rows = Tensor::from_slice(&LOOKUPS, (2, 3), &device)?;
                let shard = MappedShard {
                    data: mapped_data[row_bytes..].as_ptr(),
                    scales: mapped_scales[groups * 2..].as_ptr(),
                    first_row: 0,
                    rows: ROWS,
                };
                let mapped =
                    unsafe { MappedRows::new(vec![shard], Arc::new((mapped_data, mapped_scales))) }
                        .with_device_index(&device)?;
                let data = data.narrow(0, 1, ROWS)?;
                let scales = scales.narrow(0, 1, ROWS)?;
                let resident = EmbeddingTable::new(packed_codec(bits), ROWS, HEAD_DIM, &data)
                    .with_scales(&scales);
                let expected = resident
                    .gather(&rows, DType::BF16.into())?
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?;
                let (host_data, host_scales) = (
                    data.to_device(&Device::Cpu)?,
                    scales.to_device(&Device::Cpu)?,
                );
                let host = EmbeddingTable::new(packed_codec(bits), ROWS, HEAD_DIM, &host_data)
                    .with_scales(&host_scales)
                    .gather_host(&LOOKUPS.map(|row| row as u64), DType::BF16.into())?
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?;
                assert_eq!(host, expected, "host {dtype:?} Q{bits}");
                let actual = EmbeddingTable::mapped(packed_codec(bits), HEAD_DIM, &mapped)
                    .gather(&rows, DType::BF16.into())?
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?;
                assert_eq!(actual, expected, "mapped {dtype:?} Q{bits}");
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mapped_quantization_preserves_shard_and_chunk_offsets() -> Result<()> {
    const HEAD_DIM: usize = 160;
    const ROWS: usize = 37;
    const FIRST_SHARD_ROWS: usize = 19;
    const CHUNK_ROWS: usize = 11;
    let device = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::F16, DType::BF16] {
        let source = Tensor::from_vec(
            (0..ROWS * HEAD_DIM)
                .map(|i| (i as f32 * 0.3).sin())
                .collect::<Vec<_>>(),
            (ROWS, HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?;
        let names = vec!["shard_0.weight".to_string(), "shard_1.weight".to_string()];
        let starts = vec![0, FIRST_SHARD_ROWS];
        let file = tempfile::NamedTempFile::new().map_err(candle_core::Error::wrap)?;
        candle_core::safetensors::save(
            &std::collections::HashMap::from([
                (
                    names[0].clone(),
                    source.narrow(0, 0, FIRST_SHARD_ROWS)?.force_contiguous()?,
                ),
                (
                    names[1].clone(),
                    source
                        .narrow(0, FIRST_SHARD_ROWS, ROWS - FIRST_SHARD_ROWS)?
                        .force_contiguous()?,
                ),
            ]),
            file.path(),
        )?;
        let raw = Arc::new(unsafe { MmapedSafetensors::new(file.path())? });
        let mut resident_table = managed_table(
            TableSource::Shards {
                raw: raw.clone(),
                names: names.clone(),
                starts: starts.clone(),
                dtype,
            },
            HEAD_DIM,
            &Device::Cpu,
        )?;
        let mut table = managed_table(
            TableSource::Shards {
                raw,
                names,
                starts,
                dtype,
            },
            HEAD_DIM,
            &Device::Cpu,
        )?;
        table.make_resident(
            &device,
            WeightAllocationBudget {
                persistent_bytes: 0,
                execution_bytes: 0,
            },
        )?;
        assert!(table.resident.is_none());
        assert!(table.prepare_mapped(&device, CHUNK_ROWS, 0).is_err());
        assert!(matches!(table.source, TableSource::Shards { .. }));
        assert!(table.resident.is_none());
        let host_budget =
            resident_bytes(ROWS, HEAD_DIM, PLANNED_RESIDENT_BITS) + RESIDENT_RESERVE_BYTES;
        let resident_budget = WeightAllocationBudget {
            persistent_bytes: host_budget - RESIDENT_RESERVE_BYTES,
            execution_bytes: host_budget,
        };
        resident_table.make_resident(
            &device,
            WeightAllocationBudget {
                persistent_bytes: resident_budget.persistent_bytes - 1,
                ..resident_budget
            },
        )?;
        assert!(resident_table.resident.is_none());
        resident_table.make_resident(
            &device,
            WeightAllocationBudget {
                execution_bytes: resident_budget.execution_bytes - 1,
                ..resident_budget
            },
        )?;
        assert!(resident_table.resident.is_none());
        resident_table.make_resident(&device, resident_budget)?;
        assert!(resident_table
            .resident
            .as_ref()
            .is_some_and(|r| r.codec == packed_codec(PLANNED_RESIDENT_BITS)));
        assert!(table
            .prepare_mapped(&device, CHUNK_ROWS, host_budget - 1)
            .is_err());
        assert!(matches!(table.source, TableSource::Shards { .. }));
        assert!(!table.mapped.has_device_index());
        table.prepare_mapped(&device, CHUNK_ROWS, host_budget)?;
        file.close().map_err(candle_core::Error::wrap)?;
        let TableSource::Packed {
            map, data_bytes, ..
        } = &table.source
        else {
            unreachable!()
        };
        let row_bytes = HEAD_DIM / 2;
        let groups = HEAD_DIM / RESIDENT_GROUP;
        let data = Tensor::zeros((ROWS, row_bytes), DType::U8, &device)?;
        let scales = Tensor::zeros((ROWS, groups), DType::F16, &device)?;
        super::super::packing::quantize(
            &source.to_device(&device)?,
            &data,
            &scales,
            PLANNED_RESIDENT_BITS,
        )?;
        assert_eq!(&map[..*data_bytes], data.flatten_all()?.to_vec1::<u8>()?);
        assert_eq!(
            &map[*data_bytes..],
            bytemuck::cast_slice::<_, u8>(&scales.flatten_all()?.to_vec1::<half::f16>()?)
        );
        let Some(ResidentTable {
            data: resident_data,
            scales: Some(resident_scales),
            ..
        }) = &resident_table.resident
        else {
            unreachable!()
        };
        assert_eq!(
            resident_data.flatten_all()?.to_vec1::<u8>()?,
            data.flatten_all()?.to_vec1::<u8>()?,
        );
        assert_eq!(
            resident_scales.flatten_all()?.to_vec1::<half::f16>()?,
            scales.flatten_all()?.to_vec1::<half::f16>()?,
        );
        let lookups = [0u64, 36, 18, 19, 10, 11];
        let host = table
            .gather_host(&lookups, &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        if device_reads_pageable_memory(&device)? {
            let rows = Tensor::from_slice(&lookups.map(|row| row as i64), (2, 3), &device)?;
            for mode in [EmbeddingGatherMode::Decode, EmbeddingGatherMode::Prefill] {
                let actual = table.gather(&rows, DType::BF16.into(), mode)?;
                assert!(actual.device().same_device(&device));
                assert_eq!(actual.dims(), &[2, 3, HEAD_DIM]);
                assert_eq!(actual.flatten_all()?.to_vec1::<half::bf16>()?, host);
            }
            assert_mapped_graph_replay(&table, &data, &scales, &device)?;
        }
        assert_eq!(table.reclaimable_weight_cache_bytes(), None);
        table.resident = Some(ResidentTable {
            codec: packed_codec(PLANNED_RESIDENT_BITS),
            data,
            scales: Some(scales),
            row_scale: None,
        });
        assert_eq!(table.reclaimable_weight_cache_bytes(), None);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn preparation_chunks_respect_host_and_device_workspace() -> Result<()> {
    const WIDTHS: [usize; 2] = [160, 576];
    const LIMITED_ROWS: usize = 7;
    for head_dim in WIDTHS {
        let source_row_bytes = head_dim * DType::BF16.size_in_bytes();
        let packed_row_bytes = resident_bytes(1, head_dim, PLANNED_RESIDENT_BITS);
        let host_row_bytes = 2 * source_row_bytes + 3 * packed_row_bytes;
        let device_row_bytes = source_row_bytes + packed_row_bytes;
        let rows = preparation_chunk_rows(
            RESIDENT_COPY_ROWS,
            source_row_bytes,
            packed_row_bytes,
            RESIDENT_RESERVE_BYTES,
        )?;
        assert!(rows > 0 && rows <= RESIDENT_COPY_ROWS);
        assert!(rows * host_row_bytes <= RESIDENT_RESERVE_BYTES);
        assert!(rows * device_row_bytes <= RESIDENT_RESERVE_BYTES);
        if head_dim == WIDTHS[0] {
            assert_eq!(rows, RESIDENT_COPY_ROWS);
        } else {
            assert!(rows < RESIDENT_COPY_ROWS);
        }
        assert_eq!(
            preparation_chunk_rows(
                RESIDENT_COPY_ROWS,
                source_row_bytes,
                packed_row_bytes,
                LIMITED_ROWS * device_row_bytes,
            )?,
            LIMITED_ROWS
        );
        assert!(preparation_chunk_rows(
            RESIDENT_COPY_ROWS,
            source_row_bytes,
            packed_row_bytes,
            device_row_bytes - 1,
        )
        .is_err());
        assert!(
            preparation_chunk_rows(RESIDENT_COPY_ROWS, source_row_bytes, packed_row_bytes, 0,)
                .is_err()
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn assert_mapped_graph_replay(
    table: &ManagedEmbedding,
    data: &Tensor,
    scales: &Tensor,
    device: &Device,
) -> Result<()> {
    use candle_core::{cuda::cudarc::driver::sys, Var};

    const IDS: [[i64; 6]; 4] = [
        [0, 36, 18, 19, 10, 11],
        [36, 0, 19, 18, 11, 10],
        [5, 5, 0, 0, 36, 36],
        [0, 36, 18, 19, 10, 11],
    ];
    let cuda = device.as_cuda_device()?;
    let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
    let rows = Var::zeros((2, 3), DType::I64, device)?;
    let gather = || table.gather(&rows, DType::BF16.into(), EmbeddingGatherMode::Decode);
    let _ = gather()?;
    device.synchronize()?;
    let stream = cuda.cuda_stream();
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
        return Err(candle_core::Error::wrap(error));
    }
    let output = gather();
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = output?;
    let graph = graph
        .map_err(candle_core::Error::wrap)?
        .ok_or_else(|| candle_core::Error::msg("Embedding capture returned no graph"))?;
    for ids in IDS {
        rows.set(&Tensor::from_slice(&ids, (2, 3), device)?)?;
        graph.launch().map_err(candle_core::Error::wrap)?;
        let expected = EmbeddingTable::new(
            packed_codec(PLANNED_RESIDENT_BITS),
            data.dim(0)?,
            table.columns,
            data,
        )
        .with_scales(scales)
        .gather(&rows, DType::BF16.into())?;
        for dtype in [DType::BF16, DType::F16] {
            let values = |tensor: &Tensor| {
                tensor
                    .to_dtype(dtype)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()
            };
            assert_eq!(values(&output)?, values(&expected)?, "{dtype:?}: {ids:?}");
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn iq4_nl_gather_matches_host_dequant() -> Result<()> {
    const HEAD_DIM: usize = 160;
    const ROWS: usize = 37;
    let device = Device::new_cuda(0)?;
    let row_bytes = iq4_nl_row_bytes(HEAD_DIM);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut table = vec![0u8; ROWS * row_bytes];
    for block in table.as_chunks_mut::<IQ4_NL_BLOCK_BYTES>().0 {
        for byte in block.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        let scale = half::f16::from_f32(((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5);
        block[..2].copy_from_slice(&scale.to_le_bytes());
    }
    let lookups: Vec<i64> = vec![0, 36, 5, 5, 17, 1, 30, 22];
    let data = Tensor::from_slice(&table, (ROWS, row_bytes), &device)?;
    let rows = Tensor::from_slice(&lookups, (2, 4), &device)?;
    let mapped = mapped_iq4_table(&table, ROWS, HEAD_DIM, &device)?;
    let TableSource::Gguf(embedding) = &mapped.source else {
        unreachable!()
    };
    let got = embedding
        .resident(&data)
        .gather(&rows, DType::BF16.into())?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut expected = vec![0f32; lookups.len() * HEAD_DIM];
    for (dst, &row) in expected
        .as_chunks_mut::<HEAD_DIM>()
        .0
        .iter_mut()
        .zip(&lookups)
    {
        let row = row as usize;
        dequantize_iq4_nl_row(&table[row * row_bytes..(row + 1) * row_bytes], dst);
    }
    for (g, e) in got.iter().zip(&expected) {
        assert_eq!(*g, half::bf16::from_f32(*e).to_f32());
    }
    if device_reads_pageable_memory(&device)? {
        let actual = mapped
            .table()
            .gather(&rows, DType::BF16.into())?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, got);
        let table = mapped_iq4_table(&table, ROWS, HEAD_DIM, &device)?;
        for mode in [EmbeddingGatherMode::Decode, EmbeddingGatherMode::Prefill] {
            let actual = table.gather(&rows, DType::BF16.into(), mode)?;
            assert!(actual.device().same_device(&device));
            assert_eq!(actual.dims(), &[2, 4, HEAD_DIM]);
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                got
            );
        }
    }
    Ok(())
}
