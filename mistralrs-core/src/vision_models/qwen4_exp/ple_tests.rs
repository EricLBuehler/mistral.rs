use super::*;
use crate::vision_models::qwen4_exp::config::tests::flash_next_text_config;

const GGML_TYPE_IQ4_NL: u32 = 20;
const IQ4_NL_BLOCK: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const KVALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn iq4_nl_row_bytes(head_dim: usize) -> usize {
    head_dim / IQ4_NL_BLOCK * IQ4_NL_BLOCK_BYTES
}

/// ggml `dequantize_row_iq4_nl` for one row.
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
) -> Result<PleTable> {
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
    PleTable::load_gguf(archive, head_dim, device)
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
                crate::cuda::qwen4_exp::ple_quantize_table(
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
            for (g, values) in values.chunks_exact(RESIDENT_GROUP).enumerate() {
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
                let rows = Tensor::from_slice(&[0i64, 36, 5, 5, 17, 1], (2, 3), &device)?;
                let format = if bits == 8 {
                    crate::cuda::qwen4_exp::PleTableFormat::Q8
                } else {
                    crate::cuda::qwen4_exp::PleTableFormat::Q4
                };
                let expected = crate::cuda::qwen4_exp::ple_gather_quant(
                    &rows,
                    &data.narrow(0, 1, ROWS)?,
                    Some(&scales.narrow(0, 1, ROWS)?),
                    HEAD_DIM,
                    format,
                )?
                .flatten_all()?
                .to_vec1::<half::bf16>()?;
                let actual = crate::cuda::qwen4_exp::ple_gather_quant_mapped(
                    &rows,
                    &mapped_data[row_bytes..(ROWS + 1) * row_bytes],
                    Some(&mapped_scales[groups * 2..(ROWS + 1) * groups * 2]),
                    HEAD_DIM,
                    format,
                )?
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
    let device = Device::new_cuda(0)?;
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
        let mut table = PleTable {
            source: TableSource::Shards {
                raw,
                names,
                starts,
                dtype,
            },
            head_dim: HEAD_DIM,
            device_index: None,
            resident: None,
        };
        table.make_resident(&device, 0)?;
        assert!(table.resident.is_none());
        table.prepare_mapped(&device, CHUNK_ROWS)?;
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
        crate::cuda::qwen4_exp::ple_quantize_table(
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
        let lookups = [0u64, 36, 18, 19, 10, 11];
        let host = table
            .gather_host(&lookups, &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        if device_reads_pageable_memory(&device)? {
            let rows = Tensor::from_slice(&lookups.map(|row| row as i64), (2, 3), &device)?;
            for layout in [
                crate::cuda::qwen4_exp::TokenLayout::rectangular(2, 1),
                crate::cuda::qwen4_exp::TokenLayout::from_host(&[(0, 2)], &[2], 2, &device)?,
            ] {
                let actual = table.gather_mapped(&rows, &layout)?;
                assert!(actual.device().same_device(&device));
                assert_eq!(actual.dims(), &[2, 3 * HEAD_DIM]);
                assert_eq!(actual.flatten_all()?.to_vec1::<half::bf16>()?, host);
            }
            assert_eq!(
                table.reclaimable_weight_cache_bytes(),
                Some(
                    resident_bytes(ROWS, HEAD_DIM, PLANNED_RESIDENT_BITS) + RESIDENT_RESERVE_BYTES
                )
            );
        }
        table.resident = Some(ResidentTable::Native {
            data,
            scales: Some(scales),
            format: crate::cuda::qwen4_exp::PleTableFormat::Q4,
        });
        assert_eq!(table.reclaimable_weight_cache_bytes(), None);
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
    let got = unsafe { embedding.gather_resident(&rows, &data) }?
        .to_dtype(DType::BF16)?
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
        let actual = unsafe { embedding.gather_mapped(&rows) }?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, got);
        let table = mapped_iq4_table(&table, ROWS, HEAD_DIM, &device)?;
        for layout in [
            crate::cuda::qwen4_exp::TokenLayout::rectangular(2, 1),
            crate::cuda::qwen4_exp::TokenLayout::from_host(&[(0, 2)], &[2], 2, &device)?,
        ] {
            let actual = table.gather_mapped(&rows, &layout)?;
            assert!(actual.device().same_device(&device));
            assert_eq!(actual.dims(), &[2, 4 * HEAD_DIM]);
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

#[cfg(feature = "cuda")]
#[test]
fn planned_table_matches_checkpoint_rows() {
    // The checkpoint pads the hashed rows up to a multiple of its shard count
    const CHECKPOINT_ROWS: usize = 320_001_536;
    const CHECKPOINT_SHARDS: usize = 128;
    let cfg = flash_next_text_config(4);
    let ple = cfg.ple().unwrap().unwrap();
    let row_bytes = ple.head_dim() / 2 + ple.head_dim() / RESIDENT_GROUP * 2;
    let rows = planned_resident_table_bytes(&cfg, &ple) / row_bytes;
    assert_eq!(
        rows.div_ceil(CHECKPOINT_SHARDS) * CHECKPOINT_SHARDS,
        CHECKPOINT_ROWS
    );
}

#[test]
fn hashed_rows_match_reference() {
    let cfg = flash_next_text_config(4);
    let ple = cfg.ple().unwrap().unwrap();
    let hash = PleHash::new(&cfg, &ple, 0);
    assert_eq!(
        hash.multipliers,
        [23703573157769, 20109073645365, 8052911324071]
    );
    assert_eq!(&hash.vocab_sizes[..3], &[20000003, 20000023, 20000033]);
    assert_eq!(&hash.offsets[..3], &[0, 20000003, 40000026]);

    // Rows 0, 1, 7, 8 and 15 per token from `Qwen4ExpTextNGramEmbedding` with no history
    let tokens = [9707, 11, 1879, 248044, 553, 1024, 7, 248044, 248044, 42];
    let expected = [
        [16410909, 39682429, 152932897, 169641436, 300984276],
        [18158303, 36390029, 159566246, 175132467, 307699687],
        [6380558, 26411572, 148942556, 164226950, 312121804],
        [18827606, 27700912, 142052741, 163793416, 318943450],
        [12851972, 31952205, 159255184, 173644045, 312013235],
        [12082042, 20450345, 149364883, 165982112, 316678718],
        [17604418, 20085892, 154993589, 175674761, 313648512],
        [10204458, 27984170, 158727320, 162586317, 310194357],
        [9663979, 26558231, 151022725, 170054832, 300986548],
        [4064167, 21428206, 156147803, 175176707, 314347972],
    ];
    let mut rows = vec![0u64; hash.num_heads()];
    for (local, expected) in expected.iter().enumerate() {
        hash.rows(&tokens, local, &[None, None], &mut rows);
        let picked = [rows[0], rows[1], rows[7], rows[8], rows[15]];
        assert_eq!(&picked, expected, "token {local}");
    }

    // A chunk boundary: the history carries the context the reference sees in-sequence
    let mut split = vec![0u64; hash.num_heads()];
    hash.rows(&tokens[5..], 0, &[Some(248044), Some(553)], &mut split);
    hash.rows(&tokens, 5, &[None, None], &mut rows);
    assert_eq!(split, rows);
}
