use std::collections::HashMap;

use candle_core::Device;
use float8::F8E4M3;

use super::*;

const ROWS: usize = 11;
const COLUMNS: usize = 160;
const FIRST_SHARD_ROWS: usize = 4;
const GLOBAL_SCALE: f32 = 3.3242362e-5;
const FP4_REFERENCE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
const IDS: [u64; 8] = [10, 0, 4, 3, 4, 8, 2, 7];

struct Fixture {
    _file: tempfile::NamedTempFile,
    table: Nvfp4Embedding,
    expected: Vec<f32>,
    #[cfg(feature = "cuda")]
    data: Vec<u8>,
    #[cfg(feature = "cuda")]
    scales: Vec<u8>,
}

fn fixture() -> Result<Fixture> {
    let groups = COLUMNS / NVFP4_BLOCK_SIZE;
    let data = (0..ROWS * COLUMNS / 2)
        .map(|i| {
            let first = (i * 2 + i / (COLUMNS / 2)) % FP4_REFERENCE.len();
            (first | ((first + 1) % FP4_REFERENCE.len()) << 4) as u8
        })
        .collect::<Vec<_>>();
    let scales = (0..ROWS * groups)
        .map(|i| ((i * 7) % 127) as u8)
        .collect::<Vec<_>>();
    let expected = (0..ROWS * COLUMNS)
        .map(|i| {
            let code = (data[i / 2] >> (4 * (i % 2))) & 15;
            let bits = scales[i / NVFP4_BLOCK_SIZE];
            let exponent = (bits >> 3) & 15;
            let fraction = f32::from(bits & 7) / 8.0;
            let scale = if exponent == 0 {
                fraction * 2f32.powi(-6)
            } else {
                (1.0 + fraction) * 2f32.powi(i32::from(exponent) - 7)
            };
            (FP4_REFERENCE[usize::from(code)] * scale) * GLOBAL_SCALE
        })
        .collect::<Vec<_>>();
    let mut tensors = HashMap::new();
    let mut shards = Vec::new();
    for (shard, (first, count)) in [
        (0, FIRST_SHARD_ROWS),
        (FIRST_SHARD_ROWS, ROWS - FIRST_SHARD_ROWS),
    ]
    .into_iter()
    .enumerate()
    {
        let weight = format!("embedding.shard_{shard}.weight");
        let scale = format!("embedding.shard_{shard}.weight_scale");
        tensors.insert(
            weight.clone(),
            Tensor::from_slice(
                &data[first * COLUMNS / 2..(first + count) * COLUMNS / 2],
                (count, COLUMNS / 2),
                &Device::Cpu,
            )?,
        );
        let fp8 = scales[first * groups..(first + count) * groups]
            .iter()
            .copied()
            .map(F8E4M3::from_bits)
            .collect::<Vec<_>>();
        tensors.insert(
            scale.clone(),
            Tensor::from_vec(fp8, (count, groups), &Device::Cpu)?,
        );
        shards.push((weight, scale));
    }
    tensors.insert(
        "embedding.weight_scale_2".to_string(),
        Tensor::from_slice(&[GLOBAL_SCALE], 1, &Device::Cpu)?,
    );
    let file = tempfile::NamedTempFile::new().map_err(candle_core::Error::wrap)?;
    candle_core::safetensors::save(&tensors, file.path())?;
    let raw = Arc::new(unsafe { MmapedSafetensors::new(file.path())? });
    let table = Nvfp4Embedding::new(raw, shards, COLUMNS, "embedding.weight_scale_2")?;
    Ok(Fixture {
        _file: file,
        table,
        expected,
        #[cfg(feature = "cuda")]
        data,
        #[cfg(feature = "cuda")]
        scales,
    })
}

fn expected_rows(fixture: &Fixture) -> Vec<f32> {
    IDS.iter()
        .flat_map(|&row| {
            fixture.expected[row as usize * COLUMNS..(row as usize + 1) * COLUMNS]
                .iter()
                .copied()
        })
        .collect()
}

fn assert_exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "element {i}: {actual} != {expected}"
        );
    }
}

#[test]
fn nvfp4_embedding_sharded_host_gather_matches_reference() -> Result<()> {
    let fixture = fixture()?;
    assert_eq!(fixture.table.rows(), ROWS);
    assert_eq!(
        fixture.table.row_bytes(),
        COLUMNS / 2 + COLUMNS / NVFP4_BLOCK_SIZE
    );
    let rows = fixture.table.mapped_rows()?;
    let table = fixture.table.mapped(&rows);
    let out = DType::F32.into();
    let actual = table
        .gather_host(&IDS, out)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_exact(&actual, &expected_rows(&fixture));
    assert!(table.gather_host(&[ROWS as u64], out).is_err());
    assert!(table.gather_host(&[u64::MAX], out).is_err());
    assert_eq!(table.gather_host(&[], out)?.dims(), &[0, COLUMNS]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn nvfp4_embedding_cuda_resident_and_mapped_match_reference() -> Result<()> {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let fixture = fixture()?;
    let device = Device::new_cuda_with_stream(0)?;
    let groups = COLUMNS / NVFP4_BLOCK_SIZE;
    let mut data = vec![0u8; COLUMNS / 2];
    data.extend(&fixture.data);
    let mut scales = vec![0u8; groups];
    scales.extend(&fixture.scales);
    let data = Tensor::from_vec(data, (ROWS + 1, COLUMNS / 2), &device)?.narrow(0, 1, ROWS)?;
    let scales = Tensor::from_vec(scales, (ROWS + 1, groups), &device)?.narrow(0, 1, ROWS)?;
    let resident = fixture.table.resident(&data, &scales);
    let ids = IDS.map(|i| i as i64);
    let mut padded_ids = vec![0i64];
    padded_ids.extend(ids);
    let ids = Tensor::from_vec(padded_ids, IDS.len() + 1, &device)?
        .narrow(0, 1, IDS.len())?
        .reshape((2, 4))?;
    let expected = Tensor::from_vec(expected_rows(&fixture), (IDS.len(), COLUMNS), &Device::Cpu)?;
    let gpu = result::device::get(0).map_err(candle_core::Error::wrap)?;
    let pageable = unsafe {
        result::device::get_attribute(
            gpu,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        )
    }
    .map_err(candle_core::Error::wrap)?
        == 1;
    let rows = if pageable {
        Some(fixture.table.mapped_rows()?.with_device_index(&device)?)
    } else {
        None
    };
    let mapped = rows.as_ref().map(|rows| fixture.table.mapped(rows));
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let expected = expected
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for table in std::iter::once(&resident).chain(mapped.as_ref()) {
            let actual = table.gather(&ids, dtype.into())?;
            assert_eq!(actual.dims(), &[2, 4, COLUMNS]);
            let actual = actual
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_exact(&actual, &expected);
        }
    }
    Ok(())
}
