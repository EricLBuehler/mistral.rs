use std::collections::HashMap;

use super::*;
use crate::{CheckpointLinearSpec, Nvfp4Layer, QuantMethod, QuantizedConfig, ShardedSafeTensors};

const ASCENDING_CODES: [u8; 8] = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
const DESCENDING_CODES: [u8; 8] = [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01];
const TEST_ROWS: usize = 32;
const TEST_SCALE_COLS: usize = 16;
const TEST_INPUT_DIM: usize = TEST_SCALE_COLS * crate::NVFP4_BLOCK_SIZE;

#[test]
fn slab_layout_preserves_all_fp8_bits_and_exception_bytes() -> Result<()> {
    let mut fixed = (0..SLAB_ROWS)
        .map(|row| (row * 16) as u8)
        .collect::<Vec<_>>();
    fixed.extend(ASCENDING_CODES.repeat(SLAB_ROWS));
    fixed.extend((0..SLAB_ROWS).rev().map(|row| (row * 16) as u8));
    fixed.extend(DESCENDING_CODES.repeat(SLAB_ROWS));
    let mut expected = (0..=u8::MAX).chain((0..=u8::MAX).rev()).collect::<Vec<_>>();
    let exceptions = [0x8000_0000, 0x0000_00ff, 0xff00_0100, 0x7f00_01ff];
    for (index, byte) in [(0, 0x80), (255, 0x00), (256, 0xff), (511, 0x7f)] {
        expected[index] = byte;
    }
    let decoded = decode(&fixed, &exceptions, TEST_SCALE_COLS)?;
    assert_eq!(
        decoded
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    Ok(())
}

#[test]
fn malformed_bases_and_exception_positions_are_rejected() {
    let mut fixed = vec![0; SLAB_ROWS * (1 + TEST_SCALE_COLS / CODES_PER_BYTE)];
    fixed[0] = MAX_BASE + 1;
    assert!(decode(&fixed, &[], TEST_SCALE_COLS)
        .unwrap_err()
        .to_string()
        .contains("row base"));
    fixed[0] = 0;
    assert!(decode(&fixed, &[256], TEST_SCALE_COLS)
        .unwrap_err()
        .to_string()
        .contains("position 256"));
    for exceptions in [[1, 1], [2, 1]] {
        assert!(decode(&fixed, &exceptions, TEST_SCALE_COLS)
            .unwrap_err()
            .to_string()
            .contains("strictly increasing"));
    }
}

fn fixture(shape: &[usize], compressed: bool) -> Result<HashMap<String, Tensor>> {
    let rows = shape[..shape.len() - 1].iter().product::<usize>();
    let mut packed_shape = shape.to_vec();
    *packed_shape.last_mut().unwrap() /= 2;
    let mut scale_shape = shape.to_vec();
    *scale_shape.last_mut().unwrap() = TEST_SCALE_COLS;
    let mut tensors = HashMap::from([(
        "layer.weight".to_string(),
        Tensor::full(0x22u8, packed_shape, &Device::Cpu)?,
    )]);
    let experts = if shape.len() == 3 { shape[0] } else { 1 };
    tensors.insert(
        "layer.weight_scale_2".to_string(),
        Tensor::from_vec(
            (0..experts)
                .map(|expert| (expert + 1) as f32 * 0.25)
                .collect(),
            experts,
            &Device::Cpu,
        )?,
    );
    tensors.insert(
        "layer.input_scale".to_string(),
        Tensor::full(0.25f32, experts, &Device::Cpu)?,
    );
    if compressed {
        let mut fixed = Vec::new();
        for _ in 0..rows / SLAB_ROWS {
            fixed.extend((0..SLAB_ROWS).map(|row| 0x30 + row as u8));
            fixed.extend(ASCENDING_CODES.repeat(SLAB_ROWS));
        }
        tensors.insert(
            format!("layer.weight_scale.{FIXED_SUFFIX}"),
            Tensor::from_vec(
                fixed,
                (
                    rows / SLAB_ROWS,
                    SLAB_ROWS * (1 + TEST_SCALE_COLS / CODES_PER_BYTE),
                ),
                &Device::Cpu,
            )?,
        );
        tensors.insert(
            format!("layer.weight_scale.{EXCEPTIONS_SUFFIX}"),
            Tensor::new(
                &[
                    0x7000_0000u32,
                    0x1000_0000 | (rows * TEST_SCALE_COLS - 1) as u32,
                ],
                &Device::Cpu,
            )?,
        );
    } else {
        let mut bytes = (0..rows)
            .flat_map(|row| {
                (0..TEST_SCALE_COLS).map(move |col| 0x30 + (row % SLAB_ROWS + col) as u8)
            })
            .collect::<Vec<_>>();
        bytes[0] = 0x70;
        *bytes.last_mut().unwrap() = 0x10;
        tensors.insert(
            "layer.weight_scale".to_string(),
            Tensor::from_vec(
                bytes.into_iter().map(F8E4M3::from_bits).collect(),
                scale_shape,
                &Device::Cpu,
            )?,
        );
    }
    Ok(tensors)
}

fn checkpoint(shape: &[usize], compressed: bool, shard: Shard) -> Result<Nvfp4Layer> {
    let config: QuantizedConfig = serde_json::from_value(serde_json::json!({
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": crate::NVFP4_BLOCK_SIZE,
        "weight_scale_encoding": if compressed { Some("csf") } else { None }
    }))
    .map_err(candle_core::Error::msg)?;
    let Some(CheckpointLinearSpec::Nvfp4(spec)) = config.resolve_checkpoint("layer")? else {
        panic!("expected NVFP4 fixture");
    };
    let vb =
        ShardedSafeTensors::wrap(fixture(shape, compressed)?, DType::F32, Device::Cpu).pp("layer");
    if shape.len() == 2 {
        Nvfp4Layer::load(shape[1], shape[0], spec, false, shard, vb)
    } else {
        Nvfp4Layer::load_stacked(shape[0], shape[2], shape[1], spec, false, shard, vb)
    }
}

#[test]
fn compressed_checkpoint_matches_plain_scales_before_and_after_sharding() -> Result<()> {
    for shape in [
        vec![TEST_ROWS, TEST_INPUT_DIM],
        vec![2, TEST_ROWS, TEST_INPUT_DIM],
    ] {
        let plain = checkpoint(&shape, false, Shard::default())?.dequantize_w()?;
        let mut shards = vec![Shard::default()];
        for (dim, &size) in shape.iter().enumerate() {
            let (offset, len) = if dim == shape.len() - 1 {
                (crate::NVFP4_BLOCK_SIZE, crate::NVFP4_BLOCK_SIZE * 3)
            } else if size == 2 {
                (1, 1)
            } else {
                (7, 19)
            };
            shards.push(Shard::Offset { dim, offset, len });
        }
        for shard in shards {
            let expected = match crate::shard_range(shard, &shape)? {
                Some((axis, offset, len)) => plain.narrow(axis, offset, len)?,
                None => plain.clone(),
            };
            let actual = checkpoint(&shape, true, shard)?.dequantize_w()?;
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
    }
    Ok(())
}

#[test]
fn compressed_scales_preserve_dense_and_routed_forward_results() -> Result<()> {
    let input = Tensor::from_vec(
        (0..3 * TEST_INPUT_DIM)
            .map(|i| (i % 17) as f32 * 0.125 - 1.0)
            .collect(),
        (3, TEST_INPUT_DIM),
        &Device::Cpu,
    )?;
    let shape = [TEST_ROWS, TEST_INPUT_DIM];
    let plain = checkpoint(&shape, false, Shard::default())?;
    let compressed = checkpoint(&shape, true, Shard::default())?;
    assert_eq!(
        plain.forward(&input)?.to_vec2::<f32>()?,
        compressed.forward(&input)?.to_vec2::<f32>()?
    );
    let shape = [2, TEST_ROWS, TEST_INPUT_DIM];
    let plain = checkpoint(&shape, false, Shard::default())?;
    let compressed = checkpoint(&shape, true, Shard::default())?;
    let indices = Tensor::new(&[[0u32, 1], [1, 0], [1, 1]], &Device::Cpu)?;
    assert_eq!(
        plain.gather_forward(&input, &indices)?.to_vec3::<f32>()?,
        compressed
            .gather_forward(&input, &indices)?
            .to_vec3::<f32>()?
    );
    Ok(())
}
