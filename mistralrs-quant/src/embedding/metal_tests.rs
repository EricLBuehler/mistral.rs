use candle_core::{
    quantized::{GgmlDType, QStorage, QTensor},
    DType, Device, Result, Tensor,
};

use super::{EmbeddingOutput, EmbeddingTable, RowCodec, RowScale};
use crate::{GgufMatMul, QuantMethod};

const ROWS: usize = 5;
const COLUMNS: usize = 256;
const OUTPUT_SCALE: f64 = 1.5;
const ABS_TOLERANCE: f32 = 2e-5;

fn weights() -> Result<Tensor> {
    Tensor::from_vec(
        (0..ROWS * COLUMNS)
            .map(|i| ((i * 13 % 127) as f32 - 63.0) * 0.03125)
            .collect(),
        (ROWS, COLUMNS),
        &Device::Cpu,
    )
}

fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
    let actual = actual.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let expected = expected.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    assert_eq!(actual.dims(), expected.dims());
    let max_error = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_error <= ABS_TOLERANCE, "maximum error {max_error}");
    Ok(())
}

#[test]
fn gguf_layer_gathers_quantized_metal_rows() -> Result<()> {
    let device = Device::new_metal(0)?;
    let ids = Tensor::from_vec(vec![4u32, 1, 1, 0], (2, 2), &device)?;
    let host_ids = ids.flatten_all()?.to_device(&Device::Cpu)?;
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let host = QTensor::quantize(&weights()?, dtype)?;
        let expected = host
            .dequantize(&Device::Cpu)?
            .index_select(&host_ids, 0)?
            .reshape((2, 2, COLUMNS))?;
        let quantized = QTensor::new(
            QStorage::from_data(host.data()?, &device, dtype)?,
            host.shape().clone(),
        )?;
        let layer = GgufMatMul::from_qtensor(quantized, None);
        assert!(layer.embedding_table().unwrap().gathers_on(&device));
        for output_dtype in [DType::F32, DType::F16, DType::BF16] {
            let actual = layer
                .embedding_forward(&ids, EmbeddingOutput::scaled(output_dtype, OUTPUT_SCALE))?;
            let expected = (&expected * OUTPUT_SCALE)?.to_dtype(output_dtype)?;
            assert_close(&actual, &expected)?;
        }
        let empty_ids = ids.narrow(0, 0, 0)?;
        let empty = layer.embedding_forward(&empty_ids, DType::F32.into())?;
        assert_eq!(empty.dims(), &[0, 2, COLUMNS]);
    }
    Ok(())
}

#[test]
fn metal_dense_gather_handles_storage_offsets_and_strided_ids() -> Result<()> {
    let device = Device::new_metal(0)?;
    let ids = Tensor::from_vec(vec![0i64, 4, 1, 3, 2, 1], (2, 3), &device)?.t()?;
    let selected = ids
        .flatten_all()?
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let data = weights()?.to_dtype(dtype)?;
        let prefix = Tensor::zeros(1, dtype, &Device::Cpu)?;
        let packed = Tensor::cat(&[&prefix, &data.flatten_all()?], 0)?.to_device(&device)?;
        let offset_data = packed
            .narrow(0, 1, ROWS * COLUMNS)?
            .reshape((ROWS, COLUMNS))?;
        let scales = Tensor::from_vec(vec![99f32, 0.5, 1.0, 1.5, 2.0, 2.5], ROWS + 1, &device)?
            .narrow(0, 1, ROWS)?;
        let table = EmbeddingTable::new(RowCodec::Dense(dtype), ROWS, COLUMNS, &offset_data)
            .with_row_scale(RowScale::PerRow(&scales));
        let expected = data
            .to_dtype(DType::F32)?
            .broadcast_mul(&scales.to_device(&Device::Cpu)?.unsqueeze(1)?)?
            .index_select(&selected, 0)?
            .reshape((3, 2, COLUMNS))?;
        let actual = table.gather(&ids, DType::F32.into())?;
        assert_close(&actual, &expected)?;
    }
    Ok(())
}
