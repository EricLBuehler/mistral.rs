use candle_core::{
    quantized::{
        k_quants::{
            BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1,
            BlockQ6K, BlockQ8K, BlockQ8_0, GgmlType,
        },
        GgmlDType, QMatMul, QTensor,
    },
    DType, Device, Result, Tensor, D,
};
use half::{bf16, f16};

pub(crate) fn embedding_forward(w: &QMatMul, ids: &Tensor) -> Result<Tensor> {
    match w {
        QMatMul::Tensor(w) | QMatMul::TensorF16(w) => dense_embedding_forward(w, ids),
        QMatMul::QTensor(w) => qtensor_embedding_forward(w, ids),
    }
}

fn dense_embedding_forward(w: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let mut final_dims = ids.dims().to_vec();
    final_dims.push(w.dim(D::Minus1)?);
    let ids = ids.to_device(w.device())?.flatten_all()?;
    w.index_select(&ids, 0)?.reshape(final_dims)
}

fn qtensor_embedding_forward(w: &QTensor, ids: &Tensor) -> Result<Tensor> {
    if !w.device().is_cpu() {
        candle_core::bail!("GGUF embedding_forward for QTensor is only supported on CPU");
    }
    let (n_rows, hidden) = w.shape().dims2()?;
    let ids_vec = ids
        .to_device(&Device::Cpu)?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;

    let dtype = w.dtype();
    if !hidden.is_multiple_of(dtype.block_size()) {
        candle_core::bail!(
            "GGUF embedding hidden size {hidden} is not divisible by block size {}",
            dtype.block_size()
        );
    }

    let data = w.data()?;
    let data = data.as_ref();
    let mut out = vec![0f32; ids_vec.len() * hidden];
    match dtype {
        GgmlDType::F32 => dequant_rows::<f32>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::F16 => dequant_rows::<f16>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::BF16 => dequant_rows::<bf16>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q4_0 => dequant_rows::<BlockQ4_0>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q4_1 => dequant_rows::<BlockQ4_1>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q5_0 => dequant_rows::<BlockQ5_0>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q5_1 => dequant_rows::<BlockQ5_1>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q8_0 => dequant_rows::<BlockQ8_0>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q8_1 => dequant_q8_1_rows(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q2K => dequant_rows::<BlockQ2K>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q3K => dequant_rows::<BlockQ3K>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q4K => dequant_rows::<BlockQ4K>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q5K => dequant_rows::<BlockQ5K>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q6K => dequant_rows::<BlockQ6K>(data, &ids_vec, n_rows, hidden, &mut out)?,
        GgmlDType::Q8K => dequant_rows::<BlockQ8K>(data, &ids_vec, n_rows, hidden, &mut out)?,
    }

    let mut out_shape = ids.dims().to_vec();
    out_shape.push(hidden);
    Tensor::from_vec(out, out_shape, &Device::Cpu)
}

fn dequant_rows<T: GgmlType>(
    data: &[u8],
    ids: &[u32],
    n_rows: usize,
    hidden: usize,
    out: &mut [f32],
) -> Result<()> {
    let row_blocks = hidden / T::BLCK_SIZE;
    let blocks = cast_slice::<T>(data)?;
    if blocks.len() != n_rows * row_blocks {
        candle_core::bail!(
            "GGUF tensor data has {} blocks, expected {}",
            blocks.len(),
            n_rows * row_blocks
        );
    }

    for (out_row, &row_id) in ids.iter().enumerate() {
        let row = row_id as usize;
        if row >= n_rows {
            candle_core::bail!("Embedding id {row} is out of range for {n_rows} rows.");
        }
        let src = &blocks[row * row_blocks..(row + 1) * row_blocks];
        let dst = &mut out[out_row * hidden..(out_row + 1) * hidden];
        T::to_float(src, dst);
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ8_1Row {
    d: f16,
    _s: f16,
    qs: [i8; 32],
}

fn dequant_q8_1_rows(
    data: &[u8],
    ids: &[u32],
    n_rows: usize,
    hidden: usize,
    out: &mut [f32],
) -> Result<()> {
    const BLOCK: usize = 32;
    let row_blocks = hidden / BLOCK;
    let blocks = cast_slice::<BlockQ8_1Row>(data)?;
    if blocks.len() != n_rows * row_blocks {
        candle_core::bail!(
            "GGUF Q8_1 tensor data has {} blocks, expected {}",
            blocks.len(),
            n_rows * row_blocks
        );
    }

    for (out_row, &row_id) in ids.iter().enumerate() {
        let row = row_id as usize;
        if row >= n_rows {
            candle_core::bail!("Embedding id {row} is out of range for {n_rows} rows.");
        }
        let src = &blocks[row * row_blocks..(row + 1) * row_blocks];
        let dst = &mut out[out_row * hidden..(out_row + 1) * hidden];
        for (block_idx, block) in src.iter().enumerate() {
            let d = block.d.to_f32();
            let start = block_idx * BLOCK;
            for i in 0..BLOCK {
                dst[start + i] = block.qs[i] as f32 * d;
            }
        }
    }
    Ok(())
}

fn cast_slice<T>(data: &[u8]) -> Result<&[T]> {
    if !data.len().is_multiple_of(std::mem::size_of::<T>()) {
        candle_core::bail!(
            "GGUF tensor data length {} is not divisible by block size {}",
            data.len(),
            std::mem::size_of::<T>()
        );
    }
    let (prefix, body, suffix) = unsafe { data.align_to::<T>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        candle_core::bail!("GGUF tensor data is not aligned for selected dtype");
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::embedding_forward;
    use candle_core::{
        quantized::{GgmlDType, QMatMul, QTensor},
        Device, Result, Tensor,
    };

    fn weight(device: &Device) -> Result<Tensor> {
        let values = (0..(8 * 256))
            .map(|i| {
                let x = i as f32;
                (x * 0.003).sin() * 0.5 + (x * 0.007).cos() * 0.25
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (8, 256), device)
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32) -> Result<()> {
        let a = a.flatten_all()?.to_vec1::<f32>()?;
        let b = b.flatten_all()?.to_vec1::<f32>()?;
        for (a, b) in a.iter().zip(b.iter()) {
            assert!((a - b).abs() <= tol, "{a} != {b}");
        }
        Ok(())
    }

    fn run_quantized_embedding(dtype: GgmlDType) -> Result<()> {
        let device = Device::Cpu;
        let w = weight(&device)?;
        let ids = Tensor::from_vec(vec![3u32, 1, 3, 7], (2, 2), &device)?;
        let q = QTensor::quantize(&w, dtype)?;
        let got = embedding_forward(&QMatMul::QTensor(std::sync::Arc::new(q)), &ids)?;

        if dtype == GgmlDType::Q8_1 {
            let expected = w
                .index_select(&ids.flatten_all()?, 0)?
                .reshape((2, 2, 256))?;
            assert_close(&got, &expected, 0.01)
        } else {
            let q = QTensor::quantize(&w, dtype)?;
            let expected = q
                .dequantize(&device)?
                .index_select(&ids.flatten_all()?, 0)?
                .reshape((2, 2, 256))?;
            assert_close(&got, &expected, 1e-6)
        }
    }

    #[test]
    fn test_cpu_ggml_embedding_supported_dtypes() -> Result<()> {
        for dtype in [
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
            run_quantized_embedding(dtype)?;
        }
        Ok(())
    }
}
