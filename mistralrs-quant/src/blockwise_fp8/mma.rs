use std::{
    collections::HashMap,
    ffi::CStr,
    sync::{LazyLock, Mutex},
};

use candle_core::{
    cuda::cudarc::driver::sys::CUdevice_attribute, CudaDevice, CudaStorage, DType, Device, Result,
    Shape, Storage, Tensor,
};
use float8::F8E4M3;
use half::bf16;

use super::ffi;
use crate::{utils::slice_ptr, ActivationScaleLayout};

pub(super) const MMA_GEMV_MAX_ROWS: usize = 32;
pub(super) const GROUP_SIZE: usize = 128;
const TILE_ROWS: usize = 16;
const MIN_COMPUTE_CAPABILITY: i32 = 89;

static DEVICE_SUPPORT: LazyLock<Mutex<HashMap<candle_core::cuda::DeviceId, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(super) fn device_supported(dev: &CudaDevice) -> bool {
    if let Some(supported) = DEVICE_SUPPORT.lock().unwrap().get(&dev.id()) {
        return *supported;
    }
    let stream = dev.cuda_stream();
    let context = stream.context();
    let major = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    let supported = major * 10 + minor >= MIN_COMPUTE_CAPABILITY;
    DEVICE_SUPPORT.lock().unwrap().insert(dev.id(), supported);
    supported
}

pub(super) fn weight_supported(weight: &Tensor, scales: &Tensor, block_size: &[usize]) -> bool {
    let Device::Cuda(dev) = weight.device() else {
        return false;
    };
    let (Ok((n, k)), Ok((scale_n, scale_k))) = (weight.dims2(), scales.dims2()) else {
        return false;
    };
    weight.dtype() == DType::F8E4M3
        && scales.dtype() == DType::F32
        && block_size == [GROUP_SIZE, GROUP_SIZE]
        && n.is_multiple_of(TILE_ROWS)
        && k.is_multiple_of(GROUP_SIZE)
        && scale_n == n.div_ceil(GROUP_SIZE)
        && scale_k == k / GROUP_SIZE
        && weight.is_contiguous()
        && scales.is_contiguous()
        && device_supported(dev)
}

fn check_status(operation: &str, status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(ffi::mistralrs_fp8_mma_error_string(status)) }
        .to_string_lossy()
        .into_owned();
    candle_core::bail!("{operation} failed: {message}")
}

struct ScaleStrides {
    row: usize,
    group: usize,
    shape: (usize, usize),
}

// RowMajor scales carry a [rows, groups] shape but are stored group-major like the CUTLASS and
// GDN producers write them; GroupMajor pads the row count to the requested alignment.
fn scale_strides(layout: ActivationScaleLayout, rows: usize, groups: usize) -> ScaleStrides {
    match layout {
        ActivationScaleLayout::RowMajor => ScaleStrides {
            row: 1,
            group: rows,
            shape: (rows, groups),
        },
        ActivationScaleLayout::GroupMajor { row_alignment } => {
            let padded_rows = rows.div_ceil(row_alignment.get()) * row_alignment.get();
            ScaleStrides {
                row: 1,
                group: padded_rows,
                shape: (groups, padded_rows),
            }
        }
    }
}

fn cuda_device(tensor: &Tensor) -> Result<&CudaDevice> {
    match tensor.device() {
        Device::Cuda(dev) => Ok(dev),
        _ => candle_core::bail!("FP8 tensor-core kernels require CUDA tensors"),
    }
}

pub(super) fn quantize_activation(
    x: &Tensor,
    layout: ActivationScaleLayout,
) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = x.dims2()?;
    let strides = scale_strides(layout, rows, cols / GROUP_SIZE);
    // group-major consumers read whole aligned row blocks, so the FP8 storage is padded like the scales
    let (quantized, scales) = quantize_rows(x, strides.group.max(rows), strides)?;
    Ok((quantized.narrow(0, 0, rows)?, scales))
}

/// Quantizes into `padded_rows` zero-filled rows with group-major scales `[K/128, padded_rows]`,
/// the operand shape the cuTile FP8 GEMM consumes.
pub(crate) fn quantize_activation_padded(
    x: &Tensor,
    padded_rows: usize,
) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = x.dims2()?;
    if padded_rows < rows {
        candle_core::bail!("padded row count {padded_rows} is below the {rows} activation rows")
    }
    let strides = ScaleStrides {
        row: 1,
        group: padded_rows,
        shape: (cols / GROUP_SIZE, padded_rows),
    };
    quantize_rows(x, padded_rows, strides)
}

fn quantize_rows(x: &Tensor, alloc_rows: usize, strides: ScaleStrides) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = x.dims2()?;
    if x.dtype() != DType::BF16 || !cols.is_multiple_of(GROUP_SIZE) {
        candle_core::bail!(
            "FP8 activation quantization needs BF16 rows with a multiple of {GROUP_SIZE} columns"
        )
    }
    let dev = cuda_device(x)?;
    let x = x.contiguous()?;
    let quantized = if alloc_rows == rows {
        unsafe { dev.alloc::<F8E4M3>(rows * cols)? }
    } else {
        dev.alloc_zeros::<F8E4M3>(alloc_rows * cols)?
    };
    let scales = dev.alloc_zeros::<f32>(strides.shape.0 * strides.shape.1)?;
    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let x_slice = match &*x_storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<bf16>()?,
            _ => unreachable!(),
        };
        let (x_ptr, _x_guard) = slice_ptr(x_slice, x_layout.start_offset());
        let (quantized_ptr, _quantized_guard) = slice_ptr(&quantized, 0);
        let (scales_ptr, _scales_guard) = slice_ptr(&scales, 0);
        let status = unsafe {
            ffi::mistralrs_fp8_mma_quantize_bf16(
                x_ptr as *const _,
                quantized_ptr as *mut _,
                scales_ptr as *mut _,
                rows as i32,
                cols as i32,
                strides.row as i32,
                strides.group as i32,
                dev.cuda_stream().cu_stream(),
            )
        };
        check_status("FP8 activation quantization", status)?;
    }
    let quantized = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(quantized, dev.clone())),
        Shape::from_dims(&[alloc_rows, cols]),
    ));
    let scales = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(scales, dev.clone())),
        Shape::from_dims(&[strides.shape.0, strides.shape.1]),
    ));
    Ok((quantized, scales))
}

pub(super) fn gemv(
    activation: &Tensor,
    activation_scales: &Tensor,
    layout: ActivationScaleLayout,
    weight: &Tensor,
    weight_scales: &Tensor,
) -> Result<Tensor> {
    let (rows, k) = activation.dims2()?;
    let (n, weight_k) = weight.dims2()?;
    if weight_k != k || rows == 0 || rows > MMA_GEMV_MAX_ROWS {
        candle_core::bail!(
            "FP8 tensor-core GEMV got {rows} rows of K={k} against a {n}x{weight_k} weight"
        )
    }
    if activation.dtype() != DType::F8E4M3 || activation_scales.dtype() != DType::F32 {
        candle_core::bail!("FP8 tensor-core GEMV needs E4M3 activations with F32 scales")
    }
    let strides = scale_strides(layout, rows, k / GROUP_SIZE);
    if activation_scales.dims2()? != strides.shape {
        candle_core::bail!(
            "FP8 activation scale shape {:?} does not match {:?} for {layout:?}",
            activation_scales.dims(),
            strides.shape
        )
    }
    let dev = cuda_device(weight)?;
    let activation = activation.contiguous()?;
    let activation_scales = activation_scales.contiguous()?;
    let weight_scale_stride = weight_scales.dim(1)?;
    let output = unsafe { dev.alloc::<bf16>(rows * n)? };
    {
        let (activation_storage, activation_layout) = activation.storage_and_layout();
        let (activation_scales_storage, activation_scales_layout) =
            activation_scales.storage_and_layout();
        let (weight_storage, weight_layout) = weight.storage_and_layout();
        let (weight_scales_storage, weight_scales_layout) = weight_scales.storage_and_layout();
        let activation_slice = match &*activation_storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<F8E4M3>()?,
            _ => unreachable!(),
        };
        let activation_scales_slice = match &*activation_scales_storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
            _ => unreachable!(),
        };
        let weight_slice = match &*weight_storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<F8E4M3>()?,
            _ => unreachable!(),
        };
        let weight_scales_slice = match &*weight_scales_storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
            _ => unreachable!(),
        };
        let (activation_ptr, _activation_guard) =
            slice_ptr(activation_slice, activation_layout.start_offset());
        let (activation_scales_ptr, _activation_scales_guard) = slice_ptr(
            activation_scales_slice,
            activation_scales_layout.start_offset(),
        );
        let (weight_ptr, _weight_guard) = slice_ptr(weight_slice, weight_layout.start_offset());
        let (weight_scales_ptr, _weight_scales_guard) =
            slice_ptr(weight_scales_slice, weight_scales_layout.start_offset());
        let (output_ptr, _output_guard) = slice_ptr(&output, 0);
        let status = unsafe {
            ffi::mistralrs_fp8_mma_gemv(
                weight_ptr as *const _,
                activation_ptr as *const _,
                activation_scales_ptr as *const _,
                weight_scales_ptr as *const _,
                output_ptr as *mut _,
                rows as i32,
                n as i32,
                k as i32,
                weight_scale_stride as i32,
                strides.row as i32,
                strides.group as i32,
                dev.cuda_stream().cu_stream(),
            )
        };
        check_status("FP8 tensor-core GEMV", status)?;
    }
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
        Shape::from_dims(&[rows, n]),
    )))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use candle_core::{DType, Device, Result, Tensor};

    use super::{gemv, quantize_activation, GROUP_SIZE};
    use crate::{blockwise_fp8::ops, ActivationScaleLayout};

    fn patterned(len: usize, seed: usize, amplitude: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 7919 + seed * 104729) % 2001) as f32 / 1000.0 - 1.0)
            .map(|value| value * amplitude + offset)
            .collect()
    }

    fn dequantize_rows(
        quantized: &Tensor,
        scales: &Tensor,
        layout: ActivationScaleLayout,
    ) -> Result<Tensor> {
        let (rows, cols) = quantized.dims2()?;
        let groups = cols / GROUP_SIZE;
        let cpu = Device::Cpu;
        let values = quantized.to_device(&cpu)?.to_dtype(DType::F32)?;
        let scales = scales.to_device(&cpu)?;
        let row_scales = match layout {
            ActivationScaleLayout::RowMajor => scales.reshape((groups, rows))?.t()?.contiguous()?,
            ActivationScaleLayout::GroupMajor { .. } => {
                scales.t()?.narrow(0, 0, rows)?.contiguous()?
            }
        };
        values
            .reshape((rows, groups, GROUP_SIZE))?
            .broadcast_mul(&row_scales.reshape((rows, groups, 1))?)?
            .reshape((rows, cols))
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn tensor_core_gemv_matches_dequantized_reference() -> Result<()> {
        const N: usize = 272;
        const K: usize = 1024;
        let dev = Device::new_cuda(0)?;
        let weight =
            Tensor::from_vec(patterned(N * K, 3, 2.0, 0.1), (N, K), &dev)?.to_dtype(DType::BF16)?;
        let (weight, weight_scales) =
            ops::fp8_blockwise_quantize(&weight, vec![GROUP_SIZE, GROUP_SIZE])?;
        let weight_ref = ops::fp8_blockwise_dequantize(
            &weight,
            &weight_scales,
            vec![GROUP_SIZE, GROUP_SIZE],
            DType::F32,
        )?
        .to_device(&Device::Cpu)?;
        let layouts = [
            ActivationScaleLayout::RowMajor,
            ActivationScaleLayout::GroupMajor {
                row_alignment: NonZeroUsize::new(4).unwrap(),
            },
        ];
        for rows in [1, 3, 8, 9, 16, 24, 32] {
            let x = Tensor::from_vec(patterned(rows * K, rows, 3.0, -0.2), (rows, K), &dev)?
                .to_dtype(DType::BF16)?;
            for layout in layouts {
                let (quantized, scales) = quantize_activation(&x, layout)?;
                let output = gemv(&quantized, &scales, layout, &weight, &weight_scales)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?;
                let reference =
                    dequantize_rows(&quantized, &scales, layout)?.matmul(&weight_ref.t()?)?;
                let max_error = (output.sub(&reference)?.abs()?.max_all()?).to_scalar::<f32>()?;
                let max_reference = reference.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(
                    max_error <= 1.0e-2 * max_reference,
                    "rows={rows} {layout:?}: max error {max_error} vs reference {max_reference}"
                );
                let requant = dequantize_rows(&quantized, &scales, layout)?;
                let source = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                let quant_error = (requant.sub(&source)?.abs()?.max_all()?).to_scalar::<f32>()?;
                let source_max = source.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(
                    quant_error <= source_max / 16.0,
                    "rows={rows} {layout:?}: activation quantization error {quant_error}"
                );
            }
        }
        Ok(())
    }
}
