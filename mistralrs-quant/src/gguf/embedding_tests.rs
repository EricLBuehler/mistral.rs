use std::{io::Write, sync::Arc};

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use candle_core::{quantized::QTensor, DType, Device, Result, Tensor};
use tempfile::NamedTempFile;

use super::{
    archive::{GgufArchive, GgufDType},
    embedding::GgufEmbedding,
};

const FORMATS: [u32; 16] = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30];
const ROWS: usize = 7;
const ALIGNMENT: usize = 32;
const TABLE_NAME: &str = "embeddings";
const IQ4_NL: u32 = 20;
const IQ4_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
const SELECTION: [u64; 5] = [6, 0, 3, 3, 1];
const TOLERANCE: f32 = 1e-5;

struct Fixture {
    _file: NamedTempFile,
    embedding: GgufEmbedding,
    values: Vec<f32>,
    columns: usize,
}

fn widths(format: u32) -> [usize; 2] {
    match GgufDType::new(format).block_size().unwrap() {
        1 => [161, 545],
        32 => [160, 544],
        _ => [256, 768],
    }
}

fn encoded_table(format: u32, columns: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    if format == IQ4_NL {
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for block in 0..ROWS * columns / ALIGNMENT {
            let scale = half::f16::from_f32((block % 11 + 1) as f32 * 0.03125);
            bytes.extend_from_slice(&scale.to_le_bytes());
            let indices: Vec<_> = (0..ALIGNMENT)
                .map(|index| (index * 3 + block * 7) % IQ4_VALUES.len())
                .collect();
            for index in 0..ALIGNMENT / 2 {
                bytes.push(indices[index] as u8 | ((indices[index + ALIGNMENT / 2] as u8) << 4));
            }
            values.extend(
                indices
                    .into_iter()
                    .map(|index| scale.to_f32() * f32::from(IQ4_VALUES[index])),
            );
        }
        return Ok((bytes, values));
    }
    let values: Vec<_> = (0..ROWS * columns)
        .map(|index| ((index * 37 % 101) as f32 - 50.0) * 0.09375)
        .collect();
    let tensor = Tensor::from_vec(values, (ROWS, columns), &Device::Cpu)?;
    let quantized = QTensor::quantize(&tensor, GgufDType::new(format).candle_dtype()?)?;
    Ok((
        quantized.data()?.to_vec(),
        quantized
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?,
    ))
}

fn write_string<O: ByteOrder>(bytes: &mut Vec<u8>, value: &str) -> std::io::Result<()> {
    bytes.write_u64::<O>(value.len() as u64)?;
    bytes.write_all(value.as_bytes())
}

fn write_fixture<O: ByteOrder>(
    data: &[u8],
    columns: usize,
    format: u32,
    unaligned: bool,
) -> Result<NamedTempFile> {
    let alignment = if unaligned { 1 } else { ALIGNMENT };
    let mut bytes = Vec::new();
    let magic = if O::read_u32(&1u32.to_le_bytes()) == 1 {
        b"GGUF"
    } else {
        b"FUGG"
    };
    bytes.write_all(magic)?;
    bytes.write_u32::<O>(3)?;
    bytes.write_u64::<O>(1)?;
    bytes.write_u64::<O>(1)?;
    write_string::<O>(&mut bytes, "general.alignment")?;
    bytes.write_u32::<O>(4)?;
    bytes.write_u32::<O>(alignment as u32)?;
    write_string::<O>(&mut bytes, TABLE_NAME)?;
    bytes.write_u32::<O>(2)?;
    bytes.write_u64::<O>(columns as u64)?;
    bytes.write_u64::<O>(ROWS as u64)?;
    bytes.write_u32::<O>(format)?;
    bytes.write_u64::<O>(0)?;
    bytes.resize(bytes.len().div_ceil(alignment) * alignment, 0);
    bytes.extend_from_slice(data);
    bytes.resize(bytes.len().div_ceil(alignment) * alignment, 0);
    let mut file = NamedTempFile::new()?;
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(file)
}

fn fixture(format: u32, columns: usize, unaligned: bool) -> Result<Fixture> {
    let (data, values) = encoded_table(format, columns)?;
    let file = write_fixture::<LittleEndian>(&data, columns, format, unaligned)?;
    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    let embedding = GgufEmbedding::new(archive, TABLE_NAME)?;
    if unaligned {
        assert_ne!(
            embedding.bytes()?.as_ptr() as usize % std::mem::align_of::<f32>(),
            0
        );
    }
    Ok(Fixture {
        _file: file,
        embedding,
        values,
        columns,
    })
}

fn assert_rows(actual: &Tensor, fixture: &Fixture, rows: &[u64], format: u32) -> Result<()> {
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), rows.len() * fixture.columns);
    for (index, actual) in actual.into_iter().enumerate() {
        let row = rows[index / fixture.columns] as usize;
        let expected = fixture.values[row * fixture.columns + index % fixture.columns];
        assert!(
            (actual - expected).abs() <= TOLERANCE * (1.0 + expected.abs()),
            "format={format} columns={} element={index}: {actual} != {expected}",
            fixture.columns
        );
    }
    Ok(())
}

#[test]
fn embedding_cpu_gathers_all_formats_from_unaligned_storage() -> Result<()> {
    for format in FORMATS {
        for columns in widths(format) {
            let fixture = fixture(format, columns, true)?;
            assert_eq!(fixture.embedding.rows(), ROWS);
            assert_eq!(fixture.embedding.columns(), columns);
            let rows = fixture.embedding.mapped_rows()?;
            let table = fixture.embedding.mapped(&rows);
            let out = DType::F32.into();
            let empty = table.gather_host(&[], out)?;
            assert_eq!(empty.dims(), &[0, columns]);
            assert_rows(
                &table.gather_host(&SELECTION, out)?,
                &fixture,
                &SELECTION,
                format,
            )?;
            assert!(table.gather_host(&[ROWS as u64], out).is_err());
        }
    }
    Ok(())
}

#[test]
fn embedding_rejects_big_endian_tensor_storage() -> Result<()> {
    let (data, _) = encoded_table(0, widths(0)[0])?;
    let file = write_fixture::<byteorder::BigEndian>(&data, widths(0)[0], 0, false)?;
    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    assert!(GgufEmbedding::new(archive, TABLE_NAME).is_err());
    Ok(())
}

#[test]
fn embedding_rejects_zero_width_rows() -> Result<()> {
    for format in FORMATS {
        let file = write_fixture::<LittleEndian>(&[], 0, format, false)?;
        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        assert!(GgufEmbedding::new(archive, TABLE_NAME).is_err());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn can_read_mapped(device: &Device) -> Result<bool> {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let ordinal = device.as_cuda_device()?.cuda_stream().context().ordinal();
    let dev = result::device::get(ordinal as i32).map_err(candle_core::Error::wrap)?;
    Ok(unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        )
    }
    .map_err(candle_core::Error::wrap)?
        == 1)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn embedding_cuda_gathers_all_formats_resident_and_mapped() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let mapped = can_read_mapped(&device)?;
    let ids: Vec<_> = SELECTION.into_iter().map(|id| id as u32).collect();
    let ids = Tensor::from_vec(ids, SELECTION.len(), &device)?;
    let empty_ids = Tensor::from_vec(Vec::<u32>::new(), 0, &device)?;
    for format in FORMATS {
        for columns in widths(format) {
            let fixture = fixture(format, columns, false)?;
            let mut bytes = vec![0u8; ALIGNMENT];
            bytes.extend_from_slice(fixture.embedding.bytes()?);
            let len = bytes.len();
            let data =
                Tensor::from_vec(bytes, len, &device)?.narrow(0, ALIGNMENT, len - ALIGNMENT)?;
            let out = DType::F32.into();
            let resident = fixture.embedding.resident(&data);
            let empty = resident.gather(&empty_ids, out)?;
            assert_eq!(empty.dims(), &[0, columns]);
            assert_rows(&resident.gather(&ids, out)?, &fixture, &SELECTION, format)?;
            if mapped && fixture.embedding.mapped_gather_aligned()? {
                let rows = fixture
                    .embedding
                    .mapped_rows()?
                    .with_device_index(&device)?;
                let table = fixture.embedding.mapped(&rows);
                let empty = table.gather(&empty_ids, out)?;
                assert_eq!(empty.dims(), &[0, columns]);
                let original_bytes = fixture.embedding.bytes()?.to_vec();
                crate::cuda_memory::prefault_mapped_memory(fixture.embedding.bytes()?, &device)?;
                assert_eq!(fixture.embedding.bytes()?, original_bytes);
                assert_rows(&table.gather(&ids, out)?, &fixture, &SELECTION, format)?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn embedding_cuda_graph_replay_tracks_changed_strided_ids() -> Result<()> {
    use candle_core::{cuda::cudarc::driver::sys, Var};
    const ID_ROWS: usize = 2;
    const ID_COLUMNS: usize = SELECTION.len() + 1;
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let _htod_cache = cuda.enable_cuda_graph_htod_cache();
    let mapped = can_read_mapped(&device)?;
    for format in FORMATS {
        let fixture = fixture(format, widths(format)[1], false)?;
        let bytes = fixture.embedding.bytes()?.to_vec();
        let len = bytes.len();
        let data = Tensor::from_vec(bytes, len, &device)?;
        let resident = fixture.embedding.resident(&data);
        let rows = if mapped {
            Some(
                fixture
                    .embedding
                    .mapped_rows()?
                    .with_device_index(&device)?,
            )
        } else {
            None
        };
        let mapped_table = rows.as_ref().map(|rows| fixture.embedding.mapped(rows));
        for from_mapping in [false, true] {
            if from_mapping && mapped_table.is_none() {
                continue;
            }
            let backing = Var::zeros((ID_ROWS, ID_COLUMNS), DType::U32, &device)?;
            let ids = backing.narrow(1, 1, SELECTION.len())?;
            assert!(!ids.is_contiguous());
            assert!(ids.layout().start_offset() > 0);
            let gather = || {
                let table = if from_mapping {
                    mapped_table.as_ref().unwrap()
                } else {
                    &resident
                };
                table.gather(&ids, DType::F32.into())
            };
            let _ = gather()?;
            device.synchronize()?;
            let stream = cuda.cuda_stream();
            let tracking = stream.context().is_event_tracking();
            if tracking {
                unsafe { stream.context().disable_event_tracking() };
            }
            let captured =
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED);
            if let Err(error) = captured {
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
            let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
            for shift in [0, 3, 5, 1, 0] {
                let values: Vec<_> = (0..ID_ROWS * ID_COLUMNS)
                    .map(|index| ((index * 3 + shift) % ROWS) as u32)
                    .collect();
                let selected: Vec<_> = values
                    .as_chunks::<ID_COLUMNS>()
                    .0
                    .iter()
                    .flat_map(|row| row[1..].iter().map(|&id| u64::from(id)))
                    .collect();
                backing.set(&Tensor::from_vec(values, (ID_ROWS, ID_COLUMNS), &device)?)?;
                graph.launch().map_err(candle_core::Error::wrap)?;
                device.synchronize()?;
                assert_rows(&output, &fixture, &selected, format)?;
            }
        }
    }
    Ok(())
}
