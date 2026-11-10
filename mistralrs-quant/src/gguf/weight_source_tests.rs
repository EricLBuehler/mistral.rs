use std::io::Write;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use tempfile::NamedTempFile;

use super::*;
use crate::MXFP4Layer;

const OUT_DIM: usize = 4;
const IN_DIM: usize = 64;
const EXPERTS: usize = 3;
const EXPERT_ROWS: usize = 2;
type Mxfp4Fixture = (NamedTempFile, Arc<GgufWeightSource>, Vec<u8>, Vec<u8>);

fn patterned(rows: usize, cols: usize, offset: usize) -> Result<Tensor> {
    let values = (0..rows * cols)
        .map(|index| ((index + offset) % 29) as f32 / 7.0 - 2.0)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (rows, cols), &Device::Cpu)
}

fn patterned_experts(offset: usize) -> Result<Tensor> {
    let values = (0..EXPERTS * EXPERT_ROWS * IN_DIM)
        .map(|index| ((index + offset) % 31) as f32 / 9.0 - 1.5)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (EXPERTS, EXPERT_ROWS, IN_DIM), &Device::Cpu)
}

fn align(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn push_tensor_header(bytes: &mut Vec<u8>, name: &str, offset: u64) {
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    for dim in [64u64, 2, 2] {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    bytes.extend_from_slice(&39u32.to_le_bytes());
    bytes.extend_from_slice(&offset.to_le_bytes());
}

fn mxfp4_data(seed: u8) -> Vec<u8> {
    let mut data = Vec::new();
    for block in 0..8 {
        data.push(124 + ((block as u8 + seed) % 4));
        for index in 0..16 {
            let low = seed.wrapping_add((block * 3 + index) as u8) & 0x0f;
            let high = seed.wrapping_add((block * 3 + index + 7) as u8) & 0x0f;
            data.push(low | (high << 4));
        }
    }
    data
}

fn unsupported_dtype_archive() -> Result<(NamedTempFile, Arc<GgufArchive>)> {
    let name = "unsupported.weight";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&256u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.resize(align(bytes.len(), 32), 0);
    bytes.extend_from_slice(&[0; 66]);
    bytes.resize(align(bytes.len(), 32), 0);

    let mut file = NamedTempFile::new().map_err(Error::wrap)?;
    file.as_file_mut().write_all(&bytes).map_err(Error::wrap)?;
    file.as_file_mut().flush().map_err(Error::wrap)?;
    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    Ok((file, archive))
}

fn big_endian_archive() -> Result<(NamedTempFile, Arc<GgufArchive>)> {
    let name = "big.weight";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"FUGG");
    bytes.extend_from_slice(&3u32.to_be_bytes());
    bytes.extend_from_slice(&1u64.to_be_bytes());
    bytes.extend_from_slice(&0u64.to_be_bytes());
    bytes.extend_from_slice(&(name.len() as u64).to_be_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&1u32.to_be_bytes());
    bytes.extend_from_slice(&1u64.to_be_bytes());
    bytes.extend_from_slice(&0u32.to_be_bytes());
    bytes.extend_from_slice(&0u64.to_be_bytes());
    bytes.resize(align(bytes.len(), 32), 0);
    bytes.extend_from_slice(&0f32.to_be_bytes());
    bytes.resize(align(bytes.len(), 32), 0);

    let mut file = NamedTempFile::new().map_err(Error::wrap)?;
    file.as_file_mut().write_all(&bytes).map_err(Error::wrap)?;
    file.as_file_mut().flush().map_err(Error::wrap)?;
    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    Ok((file, archive))
}

fn native_mxfp4_blocks(data: &[u8]) -> Vec<u8> {
    let mut blocks = Vec::new();
    for block in data.as_chunks::<17>().0 {
        for index in (0..32).step_by(2) {
            let nibble = |index: usize| {
                if index < 16 {
                    block[index + 1] & 0x0f
                } else {
                    block[index - 15] >> 4
                }
            };
            blocks.push(nibble(index) | (nibble(index + 1) << 4));
        }
    }
    blocks
}

fn interleave_rows(lhs: &[u8], rhs: &[u8], outer: usize, rows: usize, width: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(lhs.len() + rhs.len());
    for outer_index in 0..outer {
        for row in 0..rows {
            let start = (outer_index * rows + row) * width;
            values.extend_from_slice(&lhs[start..start + width]);
            values.extend_from_slice(&rhs[start..start + width]);
        }
    }
    values
}

fn mxfp4_source() -> Result<Mxfp4Fixture> {
    let gate = mxfp4_data(0);
    let up = mxfp4_data(5);
    let second_offset = align(gate.len(), 32);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    push_tensor_header(&mut bytes, "blk.0.ffn_gate_exps.weight", 0);
    push_tensor_header(&mut bytes, "blk.0.ffn_up_exps.weight", second_offset as u64);
    bytes.resize(align(bytes.len(), 32), 0);
    bytes.extend_from_slice(&gate);
    bytes.resize(align(bytes.len(), 32), 0);
    bytes.extend_from_slice(&up);
    bytes.resize(align(bytes.len(), 32), 0);

    let mut file = NamedTempFile::new().map_err(Error::wrap)?;
    file.as_file_mut().write_all(&bytes).map_err(Error::wrap)?;
    file.as_file_mut().flush().map_err(Error::wrap)?;

    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    let gate_blocks = GgufTensorBinding::mxfp4_blocks("blk.0.ffn_gate_exps.weight");
    let up_blocks = GgufTensorBinding::mxfp4_blocks("blk.0.ffn_up_exps.weight");
    let gate_scales = GgufTensorBinding::mxfp4_scales("blk.0.ffn_gate_exps.weight");
    let up_scales = GgufTensorBinding::mxfp4_scales("blk.0.ffn_up_exps.weight");
    let bindings = GgufBindingMap::new()
        .with_binding("gate.blocks", gate_blocks.clone())
        .with_binding("gate.scales", gate_scales.clone())
        .with_binding(
            "native.gate_up_proj_blocks",
            GgufTensorBinding::interleave(vec![gate_blocks, up_blocks], 1)
                .reshape(vec![2, 4, 2, 16]),
        )
        .with_binding(
            "native.gate_up_proj_scales",
            GgufTensorBinding::interleave(vec![gate_scales, up_scales], 1),
        );
    let source = Arc::new(GgufWeightSource::new(archive, &bindings, DType::BF16)?);
    Ok((file, source, gate, up))
}

fn test_source() -> Result<(NamedTempFile, Arc<GgufWeightSource>, Tensor, Tensor)> {
    let weight = patterned(OUT_DIM, IN_DIM, 0)?;
    let other = patterned(OUT_DIM, IN_DIM, 11)?;
    let single_block = patterned(1, GgmlDType::Q4_0.block_size(), 7)?;
    let bias = Tensor::from_vec(vec![0.25f32, 0.5, 0.75, 1.0], OUT_DIM, &Device::Cpu)?;
    let norm = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], OUT_DIM, &Device::Cpu)?;

    let q_weight = QTensor::quantize(&weight, GgmlDType::Q4_0)?;
    let q_other = QTensor::quantize(&other, GgmlDType::Q4_0)?;
    let q_single_block = QTensor::quantize(&single_block, GgmlDType::Q4_0)?;
    let q_bias = QTensor::quantize(&bias, GgmlDType::F32)?;
    let q_norm = QTensor::quantize(&norm, GgmlDType::F32)?;
    let expected_weight = q_weight.dequantize(&Device::Cpu)?;
    let expected_other = q_other.dequantize(&Device::Cpu)?;
    let float_weight = QTensor::quantize(&expected_weight, GgmlDType::F32)?;

    let mut file = NamedTempFile::new().map_err(Error::wrap)?;
    gguf_file::write(
        file.as_file_mut(),
        &[],
        &[
            ("blk.0.weight", &q_weight),
            ("blk.1.weight", &q_other),
            ("blk.0.single.weight", &q_single_block),
            ("blk.float.weight", &float_weight),
            ("blk.0.bias", &q_bias),
            ("norm.weight", &q_norm),
        ],
    )?;
    file.as_file_mut().flush().map_err(Error::wrap)?;

    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    let bindings = GgufBindingMap::new()
        .with_binding(
            "model.linear.weight",
            GgufTensorBinding::tensor("blk.0.weight"),
        )
        .with_binding("model.linear.bias", GgufTensorBinding::tensor("blk.0.bias"))
        .with_binding(
            "model.float.weight",
            GgufTensorBinding::tensor("blk.float.weight"),
        )
        .with_binding(
            "model.single.weight",
            GgufTensorBinding::tensor("blk.0.single.weight"),
        )
        .with_binding(
            "model.slice.weight",
            GgufTensorBinding::tensor("blk.0.weight").slice(0, 1, 2),
        )
        .with_binding(
            "model.concat.weight",
            GgufTensorBinding::concat(
                vec![
                    GgufTensorBinding::tensor("blk.0.weight").slice(0, 0, 2),
                    GgufTensorBinding::tensor("blk.1.weight").slice(0, 2, 2),
                ],
                0,
            ),
        )
        .with_binding(
            "model.stack.weight",
            GgufTensorBinding::stack(
                vec![
                    GgufTensorBinding::tensor("blk.0.weight"),
                    GgufTensorBinding::tensor("blk.1.weight"),
                ],
                0,
            ),
        )
        .with_binding(
            "model.norm.weight",
            GgufTensorBinding::tensor("norm.weight").affine(1.0, 1.0),
        )
        .with_binding(
            "model.log.weight",
            GgufTensorBinding::tensor("norm.weight").log(),
        )
        .with_binding(
            "model.inverse_softplus.weight",
            GgufTensorBinding::tensor("norm.weight").inverse_softplus(),
        )
        .with_binding(
            "model.precise.weight",
            GgufTensorBinding::tensor("norm.weight").affine(1.0, 0.001),
        )
        .with_binding(
            "model.cast.weight",
            GgufTensorBinding::tensor("blk.float.weight").cast(DType::F32),
        )
        .with_binding(
            "model.layers.0.linear.weight",
            GgufTensorBinding::tensor("blk.0.weight"),
        )
        .with_binding(
            "model.layers.0.single.weight",
            GgufTensorBinding::tensor("blk.0.single.weight"),
        )
        .with_binding(
            "model.layers.1.linear.weight",
            GgufTensorBinding::tensor("blk.0.weight").slice(0, 0, OUT_DIM),
        );
    let source = Arc::new(GgufWeightSource::new(archive, &bindings, DType::F32)?);
    Ok((file, source, expected_weight, expected_other))
}

fn expert_source() -> Result<(NamedTempFile, Arc<GgufWeightSource>, Tensor, Tensor)> {
    let gate = QTensor::quantize(&patterned_experts(0)?, GgmlDType::Q4_0)?;
    let up = QTensor::quantize(&patterned_experts(17)?, GgmlDType::Q4_0)?;
    let gate_bias = Tensor::from_vec(
        (0..EXPERTS * EXPERT_ROWS)
            .map(|index| index as f32 / 10.0)
            .collect::<Vec<_>>(),
        (EXPERTS, EXPERT_ROWS),
        &Device::Cpu,
    )?;
    let up_bias = Tensor::from_vec(
        (0..EXPERTS * EXPERT_ROWS)
            .map(|index| 1.0 + index as f32 / 10.0)
            .collect::<Vec<_>>(),
        (EXPERTS, EXPERT_ROWS),
        &Device::Cpu,
    )?;
    let q_gate_bias = QTensor::quantize(&gate_bias, GgmlDType::F32)?;
    let q_up_bias = QTensor::quantize(&up_bias, GgmlDType::F32)?;
    let expected = Tensor::cat(
        &[gate.dequantize(&Device::Cpu)?, up.dequantize(&Device::Cpu)?],
        1,
    )?;
    let expected_bias = Tensor::cat(&[gate_bias, up_bias], 1)?;

    let mut file = NamedTempFile::new().map_err(Error::wrap)?;
    gguf_file::write(
        file.as_file_mut(),
        &[],
        &[
            ("experts.gate.weight", &gate),
            ("experts.up.weight", &up),
            ("experts.gate.bias", &q_gate_bias),
            ("experts.up.bias", &q_up_bias),
        ],
    )?;
    file.as_file_mut().flush().map_err(Error::wrap)?;
    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    let bindings = GgufBindingMap::new()
        .with_binding(
            "model.experts.gate_up.weight",
            GgufTensorBinding::concat(
                vec![
                    GgufTensorBinding::tensor("experts.gate.weight"),
                    GgufTensorBinding::tensor("experts.up.weight"),
                ],
                1,
            ),
        )
        .with_binding(
            "model.experts.gate_up.bias",
            GgufTensorBinding::concat(
                vec![
                    GgufTensorBinding::tensor("experts.gate.bias"),
                    GgufTensorBinding::tensor("experts.up.bias"),
                ],
                1,
            ),
        )
        .with_binding(
            "model.experts.gate.weight",
            GgufTensorBinding::tensor("experts.gate.weight").slice(1, 0, EXPERT_ROWS),
        );
    let source = Arc::new(GgufWeightSource::new(archive, &bindings, DType::F32)?);
    Ok((file, source, expected, expected_bias))
}

fn assert_close(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    assert_eq!(lhs.dims(), rhs.dims());
    let max_diff = (lhs - rhs)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_diff <= 0.003, "max diff {max_diff}");
    Ok(())
}

#[test]
fn direct_bindings_stay_packed_and_shard_before_loading() -> Result<()> {
    let (_file, source, expected, _) = test_source()?;

    let full = source
        .load_linear("model.linear", &Device::Cpu, Shard::default())?
        .unwrap();
    assert_eq!(
        full.get_qtensor().unwrap().shape().dims(),
        [OUT_DIM, IN_DIM]
    );
    assert!(full.has_bias());
    assert_close(&full.dequantize_w()?, &expected)?;

    let output_shard = source
        .load_linear(
            "model.linear",
            &Device::Cpu,
            Shard::Simple {
                dim: 0,
                rank: 1,
                world_size: 2,
            },
        )?
        .unwrap();
    assert_eq!(
        output_shard.get_qtensor().unwrap().shape().dims(),
        [OUT_DIM / 2, IN_DIM]
    );
    assert!(output_shard.has_bias());
    assert_close(
        &output_shard.dequantize_w()?,
        &expected.narrow(0, OUT_DIM / 2, OUT_DIM / 2)?,
    )?;
    let zeros = Tensor::zeros((1, IN_DIM), DType::F32, &Device::Cpu)?;
    let output = output_shard.forward(&zeros)?;
    let expected_bias = Tensor::from_vec(vec![0.75f32, 1.0], (1, 2), &Device::Cpu)?;
    assert_close(&output, &expected_bias)?;

    let input_shard = source
        .load_linear(
            "model.linear",
            &Device::Cpu,
            Shard::Simple {
                dim: 1,
                rank: 0,
                world_size: 2,
            },
        )?
        .unwrap();
    assert_eq!(
        input_shard.get_qtensor().unwrap().shape().dims(),
        [OUT_DIM, IN_DIM / 2]
    );
    assert!(!input_shard.has_bias());
    assert_eq!(source.shard_alignment("model.linear")?, 32);
    assert_eq!(
        source.pack_factor_for("model.linear.weight", DType::F32)?,
        Some(7)
    );
    assert_eq!(
        source.pack_factor_for("model.single.weight", DType::F32)?,
        Some(6)
    );

    let with_weight_suffix = source
        .load_linear("model.linear.weight", &Device::Cpu, Shard::default())?
        .unwrap();
    assert!(with_weight_suffix.has_bias());

    let float = source
        .load_linear("model.float", &Device::Cpu, Shard::default())?
        .unwrap();
    assert!(float.get_qtensor().is_none());
    assert_close(&float.dequantize_w()?, &expected)?;
    assert_eq!(source.pack_factor_for("model.float", DType::F32)?, Some(1));
    assert_eq!(source.pack_factor(DType::F32)?, 6);

    let uniform_bindings = GgufBindingMap::new().with_binding(
        "model.layers.0.linear.weight",
        GgufTensorBinding::tensor("blk.0.weight"),
    );
    let uniform = GgufWeightSource::new(source.archive().clone(), &uniform_bindings, DType::F32)?;
    assert_eq!(uniform.pack_factor(DType::F32)?, 7);

    let multimodal_bindings = GgufBindingMap::new()
        .with_binding(
            "model.language_model.layers.0.linear.weight",
            GgufTensorBinding::tensor("blk.0.weight"),
        )
        .with_binding(
            "model.vision_tower.encoder.layers.0.linear.weight",
            GgufTensorBinding::tensor("blk.float.weight"),
        );
    let multimodal =
        GgufWeightSource::new(source.archive().clone(), &multimodal_bindings, DType::F32)?;
    assert_eq!(multimodal.pack_factor(DType::F32)?, 7);
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn shared_binding_cache_release_preserves_uploaded_aliases_and_raw_neighbors() -> Result<()> {
    const ROWS: usize = 128;
    const COLUMNS: usize = 256;
    const ALIAS_ROWS: usize = ROWS / 4;
    const LAST_ALIAS: usize = ROWS - ALIAS_ROWS;
    const NEIGHBOR_ROWS: usize = 64;
    let device = Device::new_cuda_with_stream(0)?;
    let weight = QTensor::quantize(&patterned(ROWS, COLUMNS, 5)?, GgmlDType::Q4_0)?;
    let neighbor = QTensor::quantize(&patterned(NEIGHBOR_ROWS, COLUMNS, 17)?, GgmlDType::F32)?;
    let expected = weight.dequantize(&Device::Cpu)?;
    let expected_dense = expected
        .narrow(0, ALIAS_ROWS, ALIAS_ROWS)?
        .affine(0.5, 0.25)?;
    let expected_packed = expected.narrow(0, LAST_ALIAS, ALIAS_ROWS)?;
    let row_bytes = COLUMNS / weight.dtype().block_size() * weight.dtype().type_size();
    let weight_bytes = weight.data()?.into_owned();
    let neighbor_bytes = neighbor.data()?.into_owned();
    let mut file = NamedTempFile::new().map_err(Error::wrap)?;
    gguf_file::write(
        file.as_file_mut(),
        &[],
        &[("shared.weight", &weight), ("raw.neighbor", &neighbor)],
    )?;
    file.as_file_mut().flush().map_err(Error::wrap)?;
    file.as_file().sync_all().map_err(Error::wrap)?;
    let archive = Arc::new(GgufArchive::open_file(file.path())?);
    let shared = GgufTensorBinding::tensor("shared.weight");
    let bindings = GgufBindingMap::new()
        .with_binding(
            "dense.weight",
            shared
                .clone()
                .slice(0, ALIAS_ROWS, 2 * ALIAS_ROWS)
                .affine(0.5, 0.25),
        )
        .with_binding(
            "packed.weight",
            GgufTensorBinding::concat(
                vec![
                    shared.clone().slice(0, 0, ALIAS_ROWS),
                    shared.slice(0, LAST_ALIAS, ALIAS_ROWS),
                ],
                0,
            ),
        );
    let source = GgufWeightSource::new(archive.clone(), &bindings, DType::F32)?;
    let raw = super::super::embedding::GgufEmbedding::new(archive.clone(), "raw.neighbor")?;
    let mut file = Some(file);
    for _ in 0..2 {
        let dense = source
            .load_linear(
                "dense",
                &device,
                Shard::Simple {
                    dim: 0,
                    rank: 0,
                    world_size: 2,
                },
            )?
            .unwrap();
        let packed = source
            .load_linear(
                "packed",
                &device,
                Shard::Simple {
                    dim: 0,
                    rank: 1,
                    world_size: 2,
                },
            )?
            .unwrap();
        device
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .synchronize()
            .map_err(Error::wrap)?;
        source.release_source_cache()?;
        source.release_source_cache()?;
        assert!(dense.get_qtensor().is_none());
        assert_close(
            &dense.dequantize_w()?.to_device(&Device::Cpu)?,
            &expected_dense,
        )?;
        assert_close(
            &packed.dequantize_w()?.to_device(&Device::Cpu)?,
            &expected_packed,
        )?;
        assert_eq!(
            &*packed.get_qtensor().unwrap().data()?,
            &weight_bytes[LAST_ALIAS * row_bytes..ROWS * row_bytes],
        );
        assert_eq!(raw.bytes()?, neighbor_bytes);
        assert_eq!(archive.tensor_data("shared.weight")?.bytes(), weight_bytes);
        if let Some(file) = file.take() {
            file.close().map_err(Error::wrap)?;
        }
    }
    Ok(())
}

#[test]
fn structural_bindings_stay_packed() -> Result<()> {
    let (_file, source, expected, other) = test_source()?;

    let sliced = source
        .load_linear("model.slice", &Device::Cpu, Shard::default())?
        .unwrap();
    assert!(sliced.get_qtensor().is_some());
    assert_close(&sliced.dequantize_w()?, &expected.narrow(0, 1, 2)?)?;

    let concatenated = source
        .load_linear("model.concat", &Device::Cpu, Shard::default())?
        .unwrap();
    let expected_concat = Tensor::cat(&[expected.narrow(0, 0, 2)?, other.narrow(0, 2, 2)?], 0)?;
    assert!(concatenated.get_qtensor().is_some());
    assert_close(&concatenated.dequantize_w()?, &expected_concat)?;
    assert_eq!(source.pack_factor_for("model.concat", DType::F32)?, Some(7));

    let stacked = source
        .load_linear("model.stack", &Device::Cpu, Shard::default())?
        .unwrap();
    assert_eq!(stacked.dequantize_w()?.dims(), [2, OUT_DIM, IN_DIM]);
    assert!(stacked.get_qtensor().is_none());
    Ok(())
}

#[test]
fn rank_three_concat_preserves_layout_and_selects_expert_bias() -> Result<()> {
    let (_file, source, expected, expected_bias) = expert_source()?;
    let layer = source
        .load_linear("model.experts.gate_up", &Device::Cpu, Shard::default())?
        .unwrap();
    assert_eq!(
        layer.get_qtensor().unwrap().shape().dims(),
        [EXPERTS, EXPERT_ROWS * 2, IN_DIM]
    );
    assert_close(&layer.dequantize_w()?, &expected)?;

    let indices = Tensor::from_vec(vec![0u32, 2, 1, 0], (2, 2), &Device::Cpu)?;
    let zeros = Tensor::zeros((2, 1, IN_DIM), DType::F32, &Device::Cpu)?;
    let actual = layer.gather_forward(&zeros, &indices)?;
    let selected_bias = expected_bias
        .index_select(&indices.flatten_all()?, 0)?
        .reshape((2, 2, EXPERT_ROWS * 2))?;
    assert_close(&actual, &selected_bias)?;

    let gate = source
        .load_linear("model.experts.gate", &Device::Cpu, Shard::default())?
        .unwrap();
    assert!(gate.get_qtensor().is_some());
    assert_close(&gate.dequantize_w()?, &expected.narrow(1, 0, EXPERT_ROWS)?)?;

    let output_shard = source
        .load_linear(
            "model.experts.gate_up",
            &Device::Cpu,
            Shard::Simple {
                dim: 1,
                rank: 1,
                world_size: 2,
            },
        )?
        .unwrap();
    let actual = output_shard.gather_forward(&zeros, &indices)?;
    let expected_bias = expected_bias
        .narrow(1, EXPERT_ROWS, EXPERT_ROWS)?
        .contiguous()?
        .index_select(&indices.flatten_all()?, 0)?
        .reshape((2, 2, EXPERT_ROWS))?;
    assert_close(&actual, &expected_bias)
}

#[test]
fn lazy_backend_materializes_bound_residuals() -> Result<()> {
    let (_file, source, _, _) = test_source()?;
    let expected = Tensor::from_vec(vec![2f32, 3.0, 4.0, 5.0], OUT_DIM, &Device::Cpu)?;

    assert_close(
        &source.materialize_tensor("model.norm.weight", &Device::Cpu)?,
        &expected,
    )?;
    let expected_log = Tensor::from_vec(
        vec![0f32, 2f32.ln(), 3f32.ln(), 4f32.ln()],
        OUT_DIM,
        &Device::Cpu,
    )?;
    assert_close(
        &source.materialize_tensor("model.log.weight", &Device::Cpu)?,
        &expected_log,
    )?;
    let inverse_softplus =
        source.materialize_tensor("model.inverse_softplus.weight", &Device::Cpu)?;
    let reconstructed = inverse_softplus.exp()?.affine(1.0, 1.0)?.log()?;
    let original = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], OUT_DIM, &Device::Cpu)?;
    assert_close(&reconstructed, &original)?;
    let vb = source.sharded_var_builder(Device::Cpu);
    assert_eq!(
        vb.tensor_shape("model.norm.weight"),
        Some([OUT_DIM].as_slice())
    );
    let actual = vb.pp("model").pp("norm").get(OUT_DIM, "weight")?;
    assert_close(&actual, &expected)?;
    assert!(vb.contains_tensor("model.norm.weight"));

    let bf16_source = Arc::new(GgufWeightSource::new(
        source.archive().clone(),
        source.bindings(),
        DType::BF16,
    )?);
    let precise = bf16_source
        .sharded_var_builder(Device::Cpu)
        .pp("model")
        .pp("precise")
        .get_with_hints_dtype(OUT_DIM, "weight", Shard::default(), DType::F32)?;
    let expected_precise =
        Tensor::from_vec(vec![1.001f32, 2.001, 3.001, 4.001], OUT_DIM, &Device::Cpu)?;
    let max_diff = (&precise - expected_precise)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(max_diff < 0.0001, "max diff {max_diff}");

    let cast = bf16_source
        .load_linear("model.cast", &Device::Cpu, Shard::default())?
        .unwrap();
    assert_eq!(cast.dequantize_w()?.dtype(), DType::F32);
    Ok(())
}

#[test]
fn mxfp4_bindings_extract_repack_and_interleave_raw_bytes() -> Result<()> {
    let (_file, source, gate_data, up_data) = mxfp4_source()?;
    let expected_gate_blocks = native_mxfp4_blocks(&gate_data);
    let expected_up_blocks = native_mxfp4_blocks(&up_data);
    let expected_gate_scales = gate_data
        .as_chunks::<17>()
        .0
        .iter()
        .map(|block| block[0])
        .collect::<Vec<_>>();
    let expected_up_scales = up_data
        .as_chunks::<17>()
        .0
        .iter()
        .map(|block| block[0])
        .collect::<Vec<_>>();

    let gate_blocks = source.materialize_tensor("gate.blocks", &Device::Cpu)?;
    let gate_scales = source.materialize_tensor("gate.scales", &Device::Cpu)?;
    assert_eq!(gate_blocks.dtype(), DType::U8);
    assert_eq!(gate_blocks.dims(), [2, 2, 32]);
    assert_eq!(
        gate_blocks.flatten_all()?.to_vec1::<u8>()?,
        expected_gate_blocks
    );
    assert_eq!(gate_scales.dtype(), DType::U8);
    assert_eq!(gate_scales.dims(), [2, 2, 2]);
    assert_eq!(
        gate_scales.flatten_all()?.to_vec1::<u8>()?,
        expected_gate_scales
    );

    let dequantized = MXFP4Layer::from_parts(gate_blocks, gate_scales, None).dequantize_w()?;
    let fp4 = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let mut expected_dequantized = Vec::new();
    for block in gate_data.as_chunks::<17>().0 {
        let scale = f32::from_bits(u32::from(block[0]) << 23);
        for packed in &block[1..] {
            expected_dequantized.push(fp4[(packed & 0x0f) as usize] * scale);
        }
        for packed in &block[1..] {
            expected_dequantized.push(fp4[(packed >> 4) as usize] * scale);
        }
    }
    assert_close(
        &dequantized.to_dtype(DType::F32)?,
        &Tensor::from_vec(expected_dequantized, (2, 2, 64), &Device::Cpu)?,
    )?;

    let expected_interleaved_blocks =
        interleave_rows(&expected_gate_blocks, &expected_up_blocks, 2, 2, 32);
    let expected_interleaved_scales =
        interleave_rows(&expected_gate_scales, &expected_up_scales, 2, 2, 2);
    let vb = source.sharded_var_builder(Device::Cpu).pp("native");
    let blocks = vb.get_with_hints_dtype(
        (2, 4, 2, 16),
        "gate_up_proj_blocks",
        Shard::default(),
        DType::U8,
    )?;
    let scales = vb.get_with_hints_dtype(
        (2, 4, 2),
        "gate_up_proj_scales",
        Shard::default(),
        DType::U8,
    )?;
    assert_eq!(
        blocks.flatten_all()?.to_vec1::<u8>()?,
        expected_interleaved_blocks
    );
    assert_eq!(
        scales.flatten_all()?.to_vec1::<u8>()?,
        expected_interleaved_scales
    );
    Ok(())
}

#[test]
fn invalid_binding_shapes_are_rejected_at_construction() -> Result<()> {
    let (_file, source, _, _) = test_source()?;
    let bindings = GgufBindingMap::new().with_binding(
        "bad.weight",
        GgufTensorBinding::tensor("blk.0.weight").slice(0, OUT_DIM, 1),
    );
    assert!(GgufWeightSource::new(source.archive().clone(), &bindings, DType::F32).is_err());
    Ok(())
}

#[test]
fn unsupported_dtypes_are_rejected_at_construction() -> Result<()> {
    let (_file, archive) = unsupported_dtype_archive()?;
    let bindings = GgufBindingMap::new().with_binding(
        "model.layers.0.linear.weight",
        GgufTensorBinding::tensor("unsupported.weight"),
    );
    let error = GgufWeightSource::new(archive, &bindings, DType::F32)
        .expect_err("IQ2_XXS must be rejected before model construction")
        .to_string();
    assert!(error.contains("unsupported.weight"));
    assert!(error.contains("IQ2_XXS (16)"));
    assert!(error.contains("Q2_K through Q8_K"));
    Ok(())
}

#[test]
fn big_endian_archives_are_rejected_at_construction() -> Result<()> {
    let (file, archive) = big_endian_archive()?;
    let bindings = GgufBindingMap::new().with_binding(
        "model.layers.0.linear.weight",
        GgufTensorBinding::tensor("big.weight"),
    );
    let error = GgufWeightSource::new(archive, &bindings, DType::F32)
        .expect_err("big-endian GGUF must be rejected before model construction")
        .to_string();
    assert!(error.contains("big-endian"));
    assert!(error.contains(&file.path().display().to_string()));
    Ok(())
}
