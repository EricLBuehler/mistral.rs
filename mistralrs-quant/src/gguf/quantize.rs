use candle_core::{
    quantized::{k_quants::*, GgmlDType, GgmlType, QStorage, QTensor},
    CpuStorage, DType, Device, Result, Storage, Tensor, D,
};
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;

const QUANTIZE_CHUNK_ELEMENTS: usize = 64 * 1024;

#[derive(Clone, Copy)]
struct QuantizeChunks {
    elements: usize,
    parallel: bool,
}

pub(crate) fn quantize_cpu(
    source: &Tensor,
    dtype: GgmlDType,
    imatrix: Option<&[f32]>,
) -> Result<QTensor> {
    let source = source.to_device(&Device::Cpu)?.contiguous()?;
    let row_size = source.dim(D::Minus1)?;
    if let Some(weights) = imatrix {
        if weights.len() != row_size {
            candle_core::bail!(
                "imatrix length {} does not match row width {row_size}",
                weights.len()
            );
        }
    }
    if !row_size.is_multiple_of(dtype.block_size()) {
        candle_core::bail!(
            "row width {row_size} is not divisible by {:?} block size",
            dtype
        );
    }
    let (storage, layout) = source.storage_and_layout();
    let Storage::Cpu(storage) = &*storage else {
        unreachable!()
    };
    let (first, last) = layout
        .contiguous_offsets()
        .expect("contiguous quantization source");
    let chunk_elements = if imatrix.is_some() {
        (QUANTIZE_CHUNK_ELEMENTS / row_size).max(1) * row_size
    } else {
        QUANTIZE_CHUNK_ELEMENTS
    }
    .min(source.elem_count().max(dtype.block_size()));
    let chunks = QuantizeChunks {
        elements: chunk_elements,
        parallel: source.elem_count() > chunk_elements
            && std::env::var_os("MISTRALRS_ISQ_SINGLETHREAD").is_none(),
    };
    macro_rules! quantize {
        ($block:ty) => {{
            let mut blocks = vec![<$block>::zeros(); source.elem_count() / <$block>::BLCK_SIZE];
            let quantize = |values: &[f32], output: &mut [$block]| match imatrix {
                Some(weights) => <$block>::from_float_imatrix(values, output, weights, row_size),
                None => <$block>::from_float(values, output),
            };
            let block_chunk = chunk_elements / <$block>::BLCK_SIZE;
            match storage {
                CpuStorage::F32(values) => {
                    let values = &values[first..last];
                    if chunks.parallel {
                        blocks
                            .par_chunks_mut(block_chunk)
                            .zip(values.par_chunks(chunk_elements))
                            .for_each(|(output, values)| quantize(values, output));
                    } else {
                        for (output, values) in blocks
                            .chunks_mut(block_chunk)
                            .zip(values.chunks(chunk_elements))
                        {
                            quantize(values, output);
                        }
                    }
                }
                CpuStorage::F16(values) => quantize_half(
                    &values[first..last],
                    &mut blocks,
                    chunks,
                    |values, floats| values.convert_to_f32_slice(floats),
                    quantize,
                ),
                CpuStorage::BF16(values) => quantize_half(
                    &values[first..last],
                    &mut blocks,
                    chunks,
                    |values, floats| values.convert_to_f32_slice(floats),
                    quantize,
                ),
                _ => return quantize_cpu(&source.to_dtype(DType::F32)?, dtype, imatrix),
            }
            QTensor::new(QStorage::Cpu(Box::new(blocks)), source.shape())
        }};
    }
    match dtype {
        GgmlDType::Q4_0 => quantize!(BlockQ4_0),
        GgmlDType::Q4_1 => quantize!(BlockQ4_1),
        GgmlDType::Q5_0 => quantize!(BlockQ5_0),
        GgmlDType::Q5_1 => quantize!(BlockQ5_1),
        GgmlDType::Q8_0 => quantize!(BlockQ8_0),
        GgmlDType::Q8_1 => quantize!(BlockQ8_1),
        GgmlDType::Q2K => quantize!(BlockQ2K),
        GgmlDType::Q3K => quantize!(BlockQ3K),
        GgmlDType::Q4K => quantize!(BlockQ4K),
        GgmlDType::Q5K => quantize!(BlockQ5K),
        GgmlDType::Q6K => quantize!(BlockQ6K),
        GgmlDType::Q8K => quantize!(BlockQ8K),
        _ => QTensor::quantize(&source.force_contiguous()?, dtype),
    }
}

fn quantize_half<T: Sync, Q: GgmlType>(
    source: &[T],
    output: &mut [Q],
    chunks: QuantizeChunks,
    convert: impl Fn(&[T], &mut [f32]) + Sync,
    quantize: impl Fn(&[f32], &mut [Q]) + Sync,
) {
    let chunk_elements = chunks.elements;
    let block_chunk = chunk_elements / Q::BLCK_SIZE;
    let run = |scratch: &mut Vec<f32>, (output, values): (&mut [Q], &[T])| {
        let floats = &mut scratch[..values.len()];
        convert(values, floats);
        quantize(floats, output);
    };
    if chunks.parallel {
        output
            .par_chunks_mut(block_chunk)
            .zip(source.par_chunks(chunk_elements))
            .for_each_init(|| vec![0f32; chunk_elements], run);
    } else {
        let mut scratch = vec![0f32; chunk_elements];
        for chunk in output
            .chunks_mut(block_chunk)
            .zip(source.chunks(chunk_elements))
        {
            run(&mut scratch, chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IsqType;

    const ROWS: usize = 97;
    const COLS: usize = 768;
    const FORMATS: [(GgmlDType, IsqType); 12] = [
        (GgmlDType::Q4_0, IsqType::Q4_0),
        (GgmlDType::Q4_1, IsqType::Q4_1),
        (GgmlDType::Q5_0, IsqType::Q5_0),
        (GgmlDType::Q5_1, IsqType::Q5_1),
        (GgmlDType::Q8_0, IsqType::Q8_0),
        (GgmlDType::Q8_1, IsqType::Q8_1),
        (GgmlDType::Q2K, IsqType::Q2K),
        (GgmlDType::Q3K, IsqType::Q3K),
        (GgmlDType::Q4K, IsqType::Q4K),
        (GgmlDType::Q5K, IsqType::Q5K),
        (GgmlDType::Q6K, IsqType::Q6K),
        (GgmlDType::Q8K, IsqType::Q8K),
    ];

    fn values(elements: usize) -> Vec<f32> {
        (0..elements)
            .map(|i| match i / COLS % 7 {
                0 => 0.0,
                1 => -0.125,
                _ => ((i * 41 % 1009) as f32 - 504.0) * 0.0031,
            })
            .collect()
    }

    #[test]
    fn parallel_quantization_preserves_candle_bytes() -> Result<()> {
        for source_dtype in [DType::F32, DType::F16, DType::BF16] {
            let source =
                Tensor::from_vec(values((ROWS + 2) * COLS), (ROWS + 2, COLS), &Device::Cpu)?
                    .to_dtype(source_dtype)?
                    .narrow(0, 1, ROWS)?;
            let reference = source.force_contiguous()?;
            let weights = (0..COLS).map(|i| (i % 11) as f32 / 5.0).collect::<Vec<_>>();
            for (dtype, ty) in FORMATS {
                let expected = QTensor::quantize(&reference, dtype)?;
                let actual = quantize_cpu(&source, dtype, None)?;
                assert_eq!(
                    actual.data()?,
                    expected.data()?,
                    "{source_dtype:?} {dtype:?}"
                );
                if ty.supports_imatrix() {
                    let expected = QTensor::quantize_imatrix(&reference, &weights, dtype)?;
                    let actual = quantize_cpu(&source, dtype, Some(&weights))?;
                    assert_eq!(
                        actual.data()?,
                        expected.data()?,
                        "imatrix {source_dtype:?} {dtype:?}"
                    );
                }
            }
        }
        Ok(())
    }
}
