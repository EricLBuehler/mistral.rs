use std::{fs, io::Write, sync::Arc};

use byteorder::{LittleEndian, WriteBytesExt};
use candle_core::quantized::GgmlDType;
use tempfile::NamedTempFile;

use super::*;

struct TestTensor {
    name: &'static str,
    shape: Vec<u64>,
    dtype: u32,
    data: Vec<u8>,
    offset: Option<u64>,
}

fn write_string(writer: &mut impl Write, value: &str) {
    writer
        .write_u64::<LittleEndian>(value.len() as u64)
        .unwrap();
    writer.write_all(value.as_bytes()).unwrap();
}

fn value_type(value: &Value) -> u32 {
    match value {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn write_value(writer: &mut impl Write, value: &Value) {
    match value {
        Value::U8(value) => writer.write_u8(*value).unwrap(),
        Value::I8(value) => writer.write_i8(*value).unwrap(),
        Value::U16(value) => writer.write_u16::<LittleEndian>(*value).unwrap(),
        Value::I16(value) => writer.write_i16::<LittleEndian>(*value).unwrap(),
        Value::U32(value) => writer.write_u32::<LittleEndian>(*value).unwrap(),
        Value::I32(value) => writer.write_i32::<LittleEndian>(*value).unwrap(),
        Value::U64(value) => writer.write_u64::<LittleEndian>(*value).unwrap(),
        Value::I64(value) => writer.write_i64::<LittleEndian>(*value).unwrap(),
        Value::F32(value) => writer.write_f32::<LittleEndian>(*value).unwrap(),
        Value::F64(value) => writer.write_f64::<LittleEndian>(*value).unwrap(),
        Value::Bool(value) => writer.write_u8(u8::from(*value)).unwrap(),
        Value::String(value) => write_string(writer, value),
        Value::Array(values) => {
            let element_type = values.first().map(value_type).unwrap_or(4);
            writer.write_u32::<LittleEndian>(element_type).unwrap();
            writer
                .write_u64::<LittleEndian>(values.len() as u64)
                .unwrap();
            for value in values {
                assert_eq!(value_type(value), element_type);
                write_value(writer, value);
            }
        }
    }
}

fn write_test_gguf(
    metadata: &[(&str, Value)],
    tensors: &[TestTensor],
    alignment: usize,
) -> NamedTempFile {
    let mut offsets = Vec::new();
    let mut next_offset = 0usize;
    for tensor in tensors {
        let offset = tensor.offset.unwrap_or(next_offset as u64);
        offsets.push(offset);
        next_offset = usize::try_from(offset).unwrap() + tensor.data.len();
        next_offset = next_offset.div_ceil(alignment) * alignment;
    }

    let mut bytes = Vec::new();
    bytes.write_all(b"GGUF").unwrap();
    bytes.write_u32::<LittleEndian>(3).unwrap();
    bytes
        .write_u64::<LittleEndian>(tensors.len() as u64)
        .unwrap();
    bytes
        .write_u64::<LittleEndian>(metadata.len() as u64)
        .unwrap();
    for (key, value) in metadata {
        write_string(&mut bytes, key);
        bytes.write_u32::<LittleEndian>(value_type(value)).unwrap();
        write_value(&mut bytes, value);
    }
    for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
        write_string(&mut bytes, tensor.name);
        bytes
            .write_u32::<LittleEndian>(tensor.shape.len() as u32)
            .unwrap();
        for dimension in tensor.shape.iter().rev() {
            bytes.write_u64::<LittleEndian>(*dimension).unwrap();
        }
        bytes.write_u32::<LittleEndian>(tensor.dtype).unwrap();
        bytes.write_u64::<LittleEndian>(*offset).unwrap();
    }
    let data_offset = bytes.len().div_ceil(alignment) * alignment;
    bytes.resize(data_offset, 0);
    for (tensor, offset) in tensors.iter().zip(offsets) {
        let start = data_offset + usize::try_from(offset).unwrap();
        bytes.resize(start, 0);
        bytes.extend_from_slice(&tensor.data);
        let padded = bytes.len().div_ceil(alignment) * alignment;
        bytes.resize(padded, 0);
    }

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();
    file
}

fn base_metadata() -> Vec<(&'static str, Value)> {
    vec![
        ("general.architecture", Value::String("qwen3".to_string())),
        (GENERAL_ALIGNMENT, Value::U32(DEFAULT_ALIGNMENT as u32)),
        (
            "tokenizer.ggml.tokens",
            Value::Array(vec![
                Value::String("hello".to_string()),
                Value::String("world".to_string()),
            ]),
        ),
    ]
}

#[test]
fn supported_dtype_catalog_matches_candle() {
    for (raw, dtype) in [
        (0, GgmlDType::F32),
        (1, GgmlDType::F16),
        (2, GgmlDType::Q4_0),
        (3, GgmlDType::Q4_1),
        (6, GgmlDType::Q5_0),
        (7, GgmlDType::Q5_1),
        (8, GgmlDType::Q8_0),
        (9, GgmlDType::Q8_1),
        (10, GgmlDType::Q2K),
        (11, GgmlDType::Q3K),
        (12, GgmlDType::Q4K),
        (13, GgmlDType::Q5K),
        (14, GgmlDType::Q6K),
        (15, GgmlDType::Q8K),
        (30, GgmlDType::BF16),
    ] {
        let catalog = GgufDType::new(raw);
        assert_eq!(catalog.block_size(), Some(dtype.block_size()), "{dtype:?}");
        assert_eq!(catalog.type_size(), Some(dtype.type_size()), "{dtype:?}");
    }
}

#[test]
fn stages_unaligned_quantized_data() -> Result<()> {
    let mut bytes = vec![0u8; GgmlDType::Q4_0.type_size() + 1];
    let aligned = ggml_dtype_alignment(GgmlDType::Q4_0);
    let start = usize::from(bytes.as_ptr().align_offset(aligned) == 0);
    let data = &mut bytes[start..start + GgmlDType::Q4_0.type_size()];
    assert_ne!(data.as_ptr().align_offset(aligned), 0);

    let tensor = qtensor_from_gguf_data(GgmlDType::Q4_0, data, vec![32], &Device::Cpu)?;
    assert_eq!(tensor.dtype(), GgmlDType::Q4_0);
    assert_eq!(tensor.dequantize(&Device::Cpu)?.dims(), [32]);
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn mapped_cuda_load_preserves_bytes_after_source_reclaim_and_unlink() -> Result<()> {
    const EXPERTS: usize = 48;
    const ROWS: usize = 640;
    const COLS: usize = 2560;
    let device = Device::new_cuda(0)?;
    for (raw, dtype, shape) in [
        (12, GgmlDType::Q4K, vec![EXPERTS, ROWS, COLS]),
        (7, GgmlDType::Q5_1, vec![EXPERTS, COLS, ROWS]),
    ] {
        let bytes = EXPERTS * ROWS * COLS / dtype.block_size() * dtype.type_size();
        assert!(bytes >= MAPPED_UPLOAD_ADVICE_THRESHOLD_BYTES);
        let tensors = [
            TestTensor {
                name: "before",
                shape: vec![8],
                dtype: 0,
                data: vec![23; 32],
                offset: None,
            },
            TestTensor {
                name: "experts",
                shape: shape.iter().map(|&dim| dim as u64).collect(),
                dtype: raw,
                data: (0..bytes)
                    .map(|i| ((i * 131 + i / 97) % 251) as u8)
                    .collect(),
                offset: None,
            },
            TestTensor {
                name: "after",
                shape: vec![16],
                dtype: 30,
                data: vec![29; 32],
                offset: None,
            },
        ];
        let file = write_test_gguf(&base_metadata(), &tensors, DEFAULT_ALIGNMENT);
        file.as_file().sync_all().map_err(Error::wrap)?;
        let archive = GgufArchive::open_file(file.path())?;
        file.close().map_err(Error::wrap)?;
        for _ in 0..2 {
            let tensor = archive.load_qtensor("experts", &device)?;
            device.synchronize()?;
            archive.release_source_cache()?;
            assert_eq!(tensor.shape().dims(), shape);
            assert_eq!(tensor.dtype(), dtype);
            assert_eq!(&*tensor.data()?, tensors[1].data.as_slice());
        }
        for tensor in &tensors {
            assert_eq!(archive.tensor_data(tensor.name)?.bytes(), tensor.data);
        }
    }
    Ok(())
}

#[test]
fn catalogs_tensors_and_loads_supported_qtensor() -> Result<()> {
    let q4_data = vec![7; 36];
    let q8_1_data = vec![3; 36];
    let q8_k_data = vec![5; 292];
    let f32_data = (0..8)
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let file = write_test_gguf(
        &base_metadata(),
        &[
            TestTensor {
                name: "blk.0.attn_q.weight",
                shape: vec![2, 32],
                dtype: 2,
                data: q4_data.clone(),
                offset: None,
            },
            TestTensor {
                name: "blk.0.attn_k.weight",
                shape: vec![1, 32],
                dtype: 9,
                data: q8_1_data.clone(),
                offset: None,
            },
            TestTensor {
                name: "blk.0.attn_v.weight",
                shape: vec![1, 256],
                dtype: 15,
                data: q8_k_data.clone(),
                offset: None,
            },
            TestTensor {
                name: "output_norm.weight",
                shape: vec![8],
                dtype: 0,
                data: f32_data.clone(),
                offset: None,
            },
        ],
        DEFAULT_ALIGNMENT,
    );

    let archive = GgufArchive::open_file(file.path())?;
    assert_eq!(archive.shards().len(), 1);
    assert_eq!(
        archive
            .metadata_value("general.architecture")
            .unwrap()
            .to_string()?,
        "qwen3"
    );
    let q4 = archive.tensor_info("blk.0.attn_q.weight")?;
    assert_eq!(q4.shape(), [2, 32]);
    assert_eq!(q4.dtype().raw(), 2);
    assert_eq!(q4.byte_len(), Some(36));
    assert_eq!(archive.tensor_data(q4.name())?.bytes(), q4_data);
    assert_eq!(
        q4.data_range().unwrap().start,
        archive.shards()[0].tensor_data_offset()
    );
    let q8_1 = archive.tensor_info("blk.0.attn_k.weight")?;
    assert_eq!(q8_1.byte_len(), Some(36));
    assert_eq!(
        archive.load_qtensor(q8_1.name(), &Device::Cpu)?.dtype(),
        GgmlDType::Q8_1
    );
    assert_eq!(archive.tensor_data(q8_1.name())?.bytes(), q8_1_data);
    let q8_k = archive.tensor_info("blk.0.attn_v.weight")?;
    assert_eq!(q8_k.byte_len(), Some(292));
    assert_eq!(
        archive.load_qtensor(q8_k.name(), &Device::Cpu)?.dtype(),
        GgmlDType::Q8K
    );
    assert_eq!(archive.tensor_data(q8_k.name())?.bytes(), q8_k_data);

    let dense = archive.load_qtensor("output_norm.weight", &Device::Cpu)?;
    assert_eq!(dense.dtype(), GgmlDType::F32);
    assert_eq!(dense.shape().dims(), [8]);
    assert_eq!(archive.tensor_data("output_norm.weight")?.bytes(), f32_data);
    Ok(())
}

#[test]
fn retains_iq_and_future_dtype_codes() -> Result<()> {
    let file = write_test_gguf(
        &base_metadata(),
        &[
            TestTensor {
                name: "iq.weight",
                shape: vec![256],
                dtype: 16,
                data: vec![1; 66],
                offset: None,
            },
            TestTensor {
                name: "future.weight",
                shape: vec![32],
                dtype: 999,
                data: vec![2; 17],
                offset: None,
            },
        ],
        DEFAULT_ALIGNMENT,
    );
    let archive = GgufArchive::open_file(file.path())?;

    let iq = archive.tensor_info("iq.weight")?;
    assert_eq!(iq.dtype().raw(), 16);
    assert_eq!(iq.byte_len(), Some(66));
    assert_eq!(archive.tensor_data(iq.name())?.bytes().len(), 66);
    assert!(archive.load_qtensor(iq.name(), &Device::Cpu).is_err());

    let future = archive.tensor_info("future.weight")?;
    assert_eq!(future.dtype().raw(), 999);
    assert_eq!(future.byte_len(), None);
    assert!(archive.tensor_data(future.name()).is_err());
    assert_eq!(
        archive.tensor_storage_data(future.name())?.bytes().len(),
        DEFAULT_ALIGNMENT
    );
    Ok(())
}

#[test]
fn orders_and_validates_split_shards() -> Result<()> {
    let first = write_test_gguf(
        &[
            ("general.architecture", Value::String("qwen3".to_string())),
            (GENERAL_ALIGNMENT, Value::U32(DEFAULT_ALIGNMENT as u32)),
            (SPLIT_NO, Value::U16(0)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::I32(2)),
        ],
        &[TestTensor {
            name: "first.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![0; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let second = write_test_gguf(
        &[
            (SPLIT_NO, Value::U16(1)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::I32(2)),
        ],
        &[TestTensor {
            name: "second.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![1; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );

    let archive = GgufArchive::open([second.path(), first.path()])?;
    assert_eq!(archive.shards()[0].path(), first.path());
    assert_eq!(archive.shards()[1].path(), second.path());
    assert!(archive.contains_tensor("first.weight"));
    assert!(archive.contains_tensor("second.weight"));
    assert_eq!(
        archive
            .metadata_value("general.architecture")
            .unwrap()
            .to_string()?,
        "qwen3"
    );
    Ok(())
}

#[test]
fn merges_split_mmproj_component() -> Result<()> {
    let main_file = write_test_gguf(
        &[
            ("general.architecture", Value::String("qwen3".to_string())),
            (GENERAL_TYPE, Value::String("model".to_string())),
            ("general.name", Value::String("text".to_string())),
            ("shared.value", Value::U32(7)),
        ],
        &[TestTensor {
            name: "token_embd.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![1; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let component_first = write_test_gguf(
        &[
            ("general.architecture", Value::String("clip".to_string())),
            (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
            ("general.name", Value::String("vision".to_string())),
            (SPLIT_NO, Value::U16(0)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::U32(2)),
            (
                "clip.projector_type",
                Value::String("qwen3vl_merger".to_string()),
            ),
            ("shared.value", Value::U32(7)),
        ],
        &[TestTensor {
            name: "v.patch_embd.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![2; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let component_second = write_test_gguf(
        &[
            (SPLIT_NO, Value::U16(1)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::U32(2)),
        ],
        &[TestTensor {
            name: "mm.0.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![3; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );

    let mut archive = GgufArchive::open_file(main_file.path())?;
    let component = GgufArchive::open([component_second.path(), component_first.path()])?;
    archive.merge_component(component)?;

    assert_eq!(archive.shards().len(), 3);
    assert_eq!(archive.shards()[0].path(), main_file.path());
    assert_eq!(archive.shards()[1].path(), component_first.path());
    assert_eq!(archive.shards()[2].path(), component_second.path());
    assert_eq!(archive.tensor_info("token_embd.weight")?.shard_index(), 0);
    assert_eq!(archive.tensor_info("v.patch_embd.weight")?.shard_index(), 1);
    assert_eq!(archive.tensor_info("mm.0.weight")?.shard_index(), 2);
    assert_eq!(
        archive.tensor_data("v.patch_embd.weight")?.bytes(),
        vec![2; 32]
    );
    assert_eq!(archive.tensor_data("mm.0.weight")?.bytes(), vec![3; 32]);
    assert_eq!(
        archive
            .metadata_value("general.architecture")
            .unwrap()
            .to_string()?,
        "qwen3"
    );
    assert_eq!(
        archive.metadata_value(GENERAL_TYPE).unwrap().to_string()?,
        "model"
    );
    assert_eq!(
        archive
            .metadata_value("general.name")
            .unwrap()
            .to_string()?,
        "text"
    );
    assert_eq!(
        archive
            .metadata_value("clip.projector_type")
            .unwrap()
            .to_string()?,
        "qwen3vl_merger"
    );
    assert!(archive.metadata_value(SPLIT_COUNT).is_none());
    Ok(())
}

#[test]
fn merges_multiple_mmproj_components() -> Result<()> {
    let main_file = write_test_gguf(&base_metadata(), &[], DEFAULT_ALIGNMENT);
    let vision_file = write_test_gguf(
        &[
            ("general.architecture", Value::String("clip".to_string())),
            (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
            ("clip.has_vision_encoder", Value::Bool(true)),
        ],
        &[TestTensor {
            name: "v.patch_embd.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![1; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let audio_file = write_test_gguf(
        &[
            ("general.architecture", Value::String("clip".to_string())),
            (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
            ("clip.has_audio_encoder", Value::Bool(true)),
        ],
        &[TestTensor {
            name: "a.position_embd.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![2; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );

    let mut archive = GgufArchive::open_file(main_file.path())?;
    archive.merge_components([
        GgufArchive::open_file(vision_file.path())?,
        GgufArchive::open_file(audio_file.path())?,
    ])?;

    assert_eq!(archive.tensor_info("v.patch_embd.weight")?.shard_index(), 1);
    assert_eq!(
        archive.tensor_info("a.position_embd.weight")?.shard_index(),
        2
    );
    assert!(matches!(
        archive.metadata_value("clip.has_vision_encoder"),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        archive.metadata_value("clip.has_audio_encoder"),
        Some(Value::Bool(true))
    ));
    Ok(())
}

#[test]
fn opens_split_and_independent_mmproj_components() -> Result<()> {
    let main_file = write_test_gguf(&base_metadata(), &[], DEFAULT_ALIGNMENT);
    let vision_first = write_test_gguf(
        &[
            ("general.architecture", Value::String("clip".to_string())),
            (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
            (SPLIT_NO, Value::U16(0)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::U32(2)),
        ],
        &[TestTensor {
            name: "v.patch_embd.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![1; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let vision_second = write_test_gguf(
        &[
            (SPLIT_NO, Value::U16(1)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::U32(2)),
        ],
        &[TestTensor {
            name: "mm.0.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![2; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let audio = write_test_gguf(
        &[
            ("general.architecture", Value::String("clip".to_string())),
            (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
        ],
        &[TestTensor {
            name: "a.position_embd.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![3; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );

    let components =
        GgufArchive::open_components([vision_second.path(), vision_first.path(), audio.path()])?;

    assert_eq!(components.len(), 2);
    assert_eq!(components[0].shards().len(), 2);
    assert!(components[0].contains_tensor("v.patch_embd.weight"));
    assert!(components[0].contains_tensor("mm.0.weight"));
    assert_eq!(components[1].shards().len(), 1);
    assert!(components[1].contains_tensor("a.position_embd.weight"));

    let mut archive = GgufArchive::open_file(main_file.path())?;
    archive.merge_components(components)?;
    assert_eq!(archive.shards().len(), 4);
    assert_eq!(archive.tensor_info("v.patch_embd.weight")?.shard_index(), 1);
    assert_eq!(archive.tensor_info("mm.0.weight")?.shard_index(), 2);
    assert_eq!(
        archive.tensor_info("a.position_embd.weight")?.shard_index(),
        3
    );
    Ok(())
}

#[test]
fn rejects_invalid_or_duplicate_mmproj_components() {
    let main_file = write_test_gguf(
        &base_metadata(),
        &[TestTensor {
            name: "duplicate.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![0; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let wrong_type_file = write_test_gguf(
        &[(GENERAL_TYPE, Value::String("model".to_string()))],
        &[],
        DEFAULT_ALIGNMENT,
    );
    let duplicate_file = write_test_gguf(
        &[(GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string()))],
        &[TestTensor {
            name: "duplicate.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![1; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );

    let mut archive = GgufArchive::open_file(main_file.path()).unwrap();
    let error = archive
        .merge_component(GgufArchive::open_file(wrong_type_file.path()).unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("must be `mmproj`"));
    let error = archive
        .merge_component(GgufArchive::open_file(duplicate_file.path()).unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("duplicated across components"));
    assert_eq!(archive.shards().len(), 1);
}

#[test]
fn rejects_conflicting_component_metadata_atomically() {
    let main_file = write_test_gguf(&base_metadata(), &[], DEFAULT_ALIGNMENT);
    let first_file = write_test_gguf(
        &[
            (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
            (
                "clip.projector_type",
                Value::String("qwen3vl_merger".to_string()),
            ),
        ],
        &[TestTensor {
            name: "first.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![1; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let second_file = write_test_gguf(
        &[
            (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
            ("clip.projector_type", Value::String("gemma3".to_string())),
        ],
        &[TestTensor {
            name: "second.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![2; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );

    let mut archive = GgufArchive::open_file(main_file.path()).unwrap();
    let error = archive
        .merge_components([
            GgufArchive::open_file(first_file.path()).unwrap(),
            GgufArchive::open_file(second_file.path()).unwrap(),
        ])
        .unwrap_err();
    assert!(error.to_string().contains("component metadata key"));
    assert_eq!(archive.shards().len(), 1);
    assert!(!archive.contains_tensor("first.weight"));
    assert!(!archive.contains_tensor("second.weight"));
    assert!(archive.metadata_value("clip.projector_type").is_none());
}

#[test]
fn rejects_duplicate_split_tensor_names() {
    let first = write_test_gguf(
        &[
            (SPLIT_NO, Value::U16(0)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::I32(2)),
        ],
        &[TestTensor {
            name: "duplicate.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![0; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let second = write_test_gguf(
        &[
            (SPLIT_NO, Value::U16(1)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::I32(2)),
        ],
        &[TestTensor {
            name: "duplicate.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![1; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let error = GgufArchive::open([first.path(), second.path()]).unwrap_err();
    assert!(error.to_string().contains("duplicated across shards"));
}

#[test]
fn rejects_incomplete_splits_and_wrong_total_count() {
    let first = write_test_gguf(
        &[
            (SPLIT_NO, Value::U16(0)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::I32(1)),
        ],
        &[TestTensor {
            name: "first.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![0; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let error = GgufArchive::open_file(first.path()).unwrap_err();
    assert!(error
        .to_string()
        .contains("were supplied, but split metadata declares 2"));

    let first = write_test_gguf(
        &[
            (SPLIT_NO, Value::U16(0)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::I32(3)),
        ],
        &[TestTensor {
            name: "first.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![0; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let second = write_test_gguf(
        &[
            (SPLIT_NO, Value::U16(1)),
            (SPLIT_COUNT, Value::U16(2)),
            (SPLIT_TENSORS_COUNT, Value::I32(3)),
        ],
        &[TestTensor {
            name: "second.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![0; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let error = GgufArchive::open([first.path(), second.path()]).unwrap_err();
    assert!(error.to_string().contains("declares 3 tensors"));
}

#[test]
fn rejects_truncated_storage_and_unaligned_offsets() {
    let file = write_test_gguf(
        &base_metadata(),
        &[TestTensor {
            name: "truncated.weight",
            shape: vec![8],
            dtype: 0,
            data: vec![0; 32],
            offset: None,
        }],
        DEFAULT_ALIGNMENT,
    );
    let shortened_len = fs::metadata(file.path()).unwrap().len() - 1;
    file.as_file().set_len(shortened_len).unwrap();
    let error = GgufArchive::open_file(file.path()).unwrap_err();
    assert!(error.to_string().contains("storage ends"));

    let file = write_test_gguf(
        &base_metadata(),
        &[
            TestTensor {
                name: "aligned.weight",
                shape: vec![32],
                dtype: 999,
                data: vec![0; 17],
                offset: Some(0),
            },
            TestTensor {
                name: "unaligned.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![0; 32],
                offset: Some(33),
            },
        ],
        DEFAULT_ALIGNMENT,
    );
    let error = GgufArchive::open_file(file.path()).unwrap_err();
    assert!(error.to_string().contains("not aligned"));
}

#[test]
fn archive_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GgufArchive>();

    let file = write_test_gguf(&base_metadata(), &[], DEFAULT_ALIGNMENT);
    let archive = Arc::new(GgufArchive::open_file(file.path()).unwrap());
    let copy = archive.clone();
    std::thread::spawn(move || {
        assert_eq!(copy.shards().len(), 1);
    })
    .join()
    .unwrap();
}
