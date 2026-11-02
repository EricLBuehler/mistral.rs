use candle_core::{DType, Device, MetalStorage, Result, Shape, Storage, Tensor, D};

use crate::metal_kernels::{
    fp8::{self, Fp8LinearDispatch, Fp8LinearParams},
    utils::BufferOffset,
    Kernels,
};

const MAX_DIRECT_ROWS: usize = 256;

#[cfg(test)]
mod tests;

pub(crate) fn supported(weight: &Tensor, scales: &Tensor, dtype: DType) -> bool {
    weight.device().is_metal()
        && weight.device().same_device(scales.device())
        && weight.rank() == 2
        && weight.dtype() == DType::F8E4M3
        && scales.dtype() == DType::F32
        && weight.is_contiguous()
        && scales.is_contiguous()
        && matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
}

fn params(m: usize, n: usize, k: usize, block: [usize; 2]) -> Fp8LinearParams {
    let [block_y, block_x] = block;
    Fp8LinearParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        block_y: block_y as u32,
        block_x: block_x as u32,
        scale_stride: k.div_ceil(block_x) as u32,
    }
}

pub(crate) fn prepare(weight: &Tensor, block: [usize; 2], dtype: DType) -> Result<()> {
    let Device::Metal(device) = weight.device() else {
        return Ok(());
    };
    let (n, k) = weight.dims2()?;
    if n == 0 || k == 0 {
        return Ok(());
    }
    fp8::prepare_fp8_linear(device, dtype, params(1, n, k, block)).map_err(candle_core::Error::wrap)
}

pub(crate) fn linear(
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    block: [usize; 2],
    dequant_dtype: DType,
) -> Result<Option<Tensor>> {
    if !supported(weight, scales, dequant_dtype)
        || input.dtype() != dequant_dtype
        || !input.device().same_device(weight.device())
    {
        return Ok(None);
    }
    let (n, k) = weight.dims2()?;
    if input.dim(D::Minus1)? != k {
        candle_core::bail!("FP8 linear input width does not match weight width {k}");
    }
    let mut shape = input.dims().to_vec();
    let m = shape[..shape.len() - 1].iter().product::<usize>();
    *shape.last_mut().unwrap() = n;
    if m == 0 || n == 0 || k == 0 {
        return Tensor::zeros(shape, dequant_dtype, input.device()).map(Some);
    }
    // Beyond this crossover, repeated tile conversion costs more than one dense dequantization.
    if m > MAX_DIRECT_ROWS {
        return Ok(None);
    }
    let input = input.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let (scale_storage, scale_layout) = scales.storage_and_layout();
    let (
        Storage::Metal(input_storage),
        Storage::Metal(weight_storage),
        Storage::Metal(scale_storage),
    ) = (&*input_storage, &*weight_storage, &*scale_storage)
    else {
        unreachable!()
    };
    let Device::Metal(device) = input.device() else {
        unreachable!()
    };
    let output = device.new_buffer(m * n, dequant_dtype, "fp8-linear")?;
    let encoder = device.command_encoder()?;
    fp8::call_fp8_linear(
        device.device(),
        &encoder,
        Kernels::global(),
        Fp8LinearDispatch {
            dtype: dequant_dtype,
            params: params(m, n, k, block),
            input: BufferOffset {
                buffer: input_storage.buffer(),
                offset_in_bytes: input_layout.start_offset() * dequant_dtype.size_in_bytes(),
            },
            weight: BufferOffset {
                buffer: weight_storage.buffer(),
                offset_in_bytes: weight_layout.start_offset(),
            },
            scales: BufferOffset {
                buffer: scale_storage.buffer(),
                offset_in_bytes: scale_layout.start_offset() * DType::F32.size_in_bytes(),
            },
            output: &output,
        },
    )
    .map_err(candle_core::Error::wrap)?;
    let storage = MetalStorage::new(output, device.clone(), m * n, dequant_dtype);
    Ok(Some(Tensor::from((
        Storage::Metal(storage),
        Shape::from(shape),
    ))))
}
