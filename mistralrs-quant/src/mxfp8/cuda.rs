use candle_core::{
    backend::BackendStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor,
};
use float8::F8E4M3;
use half::{bf16, f16};

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

use super::MXFP8_BLOCK_SIZE;

fn wmma_supported(dev: &candle_core::CudaDevice) -> bool {
    cfg!(has_mxfp4_wmma_kernels) && crate::cuda_device::cuda_device_info(dev).has_bf16_mma()
}

macro_rules! ffi {
    ($ty:ty, $quantize:ident, $quantize_packed:ident, $dequantize:ident, $matmul:ident) => {
        extern "C" {
            fn $quantize(
                input: *const $ty,
                output: *mut $ty,
                elements: usize,
                stream: candle_core::cuda::cudarc::driver::sys::CUstream,
            ) -> i32;
            fn $quantize_packed(
                input: *const $ty,
                output: *mut u8,
                scales: *mut u8,
                elements: usize,
                stream: candle_core::cuda::cudarc::driver::sys::CUstream,
            ) -> i32;
            fn $dequantize(
                weight: *const u8,
                scales: *const u8,
                output: *mut $ty,
                elements: usize,
                stream: candle_core::cuda::cudarc::driver::sys::CUstream,
            ) -> i32;
            fn $matmul(
                input: *const $ty,
                weight: *const u8,
                scales: *const u8,
                indices: *const u32,
                output: *mut $ty,
                rows: i32,
                n: i32,
                k: i32,
                topk: i32,
                input_per_expert: bool,
                tensor_cores: bool,
                stream: candle_core::cuda::cudarc::driver::sys::CUstream,
            ) -> i32;
        }
    };
}
ffi!(
    f32,
    mxfp8_quantize_f32,
    mxfp8_quantize_packed_f32,
    mxfp8_dequantize_f32,
    mxfp8_matmul_f32
);
ffi!(
    f16,
    mxfp8_quantize_f16,
    mxfp8_quantize_packed_f16,
    mxfp8_dequantize_f16,
    mxfp8_matmul_f16
);
ffi!(
    bf16,
    mxfp8_quantize_bf16,
    mxfp8_quantize_packed_bf16,
    mxfp8_dequantize_bf16,
    mxfp8_matmul_bf16
);

fn check_status(status: i32) -> Result<()> {
    if status != 0 {
        candle_core::bail!("MXFP8 CUDA kernel launch failed with error {status}");
    }
    Ok(())
}

pub(super) fn quantize(storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
    let dev = storage.device();
    let stream = dev.cuda_stream();
    let elements = layout.shape().elem_count();
    macro_rules! run {
        ($ty:ty, $kernel:ident) => {{
            let mut output = unsafe { dev.alloc::<$ty>(elements)? };
            let (input_ptr, input_guard) = slice_ptr_on_stream(
                storage.as_cuda_slice::<$ty>()?,
                layout.start_offset(),
                &stream,
            );
            let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let status = unsafe {
                $kernel(
                    input_ptr as *const $ty,
                    output_ptr as *mut $ty,
                    elements,
                    stream.cu_stream(),
                )
            };
            check_status(status)?;
            drop((input_guard, output_guard));
            CudaStorage::wrap_cuda_slice(output, dev.clone())
        }};
    }
    let output = match storage.dtype() {
        DType::F32 => run!(f32, mxfp8_quantize_f32),
        DType::F16 => run!(f16, mxfp8_quantize_f16),
        DType::BF16 => run!(bf16, mxfp8_quantize_bf16),
        _ => candle_core::bail!("MXFP8 activation quantization requires F32, F16 or BF16"),
    };
    Ok((output, layout.shape().clone()))
}

pub(super) fn quantize_packed(
    storage: &CudaStorage,
    layout: &Layout,
) -> Result<(CudaStorage, CudaStorage)> {
    let dev = storage.device();
    let stream = dev.cuda_stream();
    let elements = layout.shape().elem_count();
    let mut output = unsafe { dev.alloc::<F8E4M3>(elements)? };
    let mut scales = unsafe { dev.alloc::<u8>(elements / MXFP8_BLOCK_SIZE)? };
    let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
    let (scales_ptr, scales_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
    macro_rules! run {
        ($ty:ty, $kernel:ident) => {{
            let (input_ptr, input_guard) = slice_ptr_on_stream(
                storage.as_cuda_slice::<$ty>()?,
                layout.start_offset(),
                &stream,
            );
            let status = unsafe {
                $kernel(
                    input_ptr as *const $ty,
                    output_ptr as *mut u8,
                    scales_ptr as *mut u8,
                    elements,
                    stream.cu_stream(),
                )
            };
            check_status(status)?;
            drop(input_guard);
        }};
    }
    match storage.dtype() {
        DType::F32 => run!(f32, mxfp8_quantize_packed_f32),
        DType::F16 => run!(f16, mxfp8_quantize_packed_f16),
        DType::BF16 => run!(bf16, mxfp8_quantize_packed_bf16),
        _ => candle_core::bail!("MXFP8 activation quantization requires F32, F16 or BF16"),
    }
    drop((output_guard, scales_guard));
    Ok((
        CudaStorage::wrap_cuda_slice(output, dev.clone()),
        CudaStorage::wrap_cuda_slice(scales, dev.clone()),
    ))
}

pub(super) fn dequantize(
    weight: &CudaStorage,
    wl: &Layout,
    scales: &CudaStorage,
    sl: &Layout,
    dtype: DType,
) -> Result<(CudaStorage, Shape)> {
    let dev = weight.device();
    let stream = dev.cuda_stream();
    let elements = wl.shape().elem_count();
    let (weight_ptr, weight_guard) = slice_ptr_on_stream(
        weight.as_cuda_slice::<F8E4M3>()?,
        wl.start_offset(),
        &stream,
    );
    let (scale_ptr, scale_guard) =
        slice_ptr_on_stream(scales.as_cuda_slice::<u8>()?, sl.start_offset(), &stream);
    macro_rules! run {
        ($ty:ty, $kernel:ident) => {{
            let mut output = unsafe { dev.alloc::<$ty>(elements)? };
            let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let status = unsafe {
                $kernel(
                    weight_ptr as *const u8,
                    scale_ptr as *const u8,
                    output_ptr as *mut $ty,
                    elements,
                    stream.cu_stream(),
                )
            };
            check_status(status)?;
            drop(output_guard);
            CudaStorage::wrap_cuda_slice(output, dev.clone())
        }};
    }
    let output = match dtype {
        DType::F32 => run!(f32, mxfp8_dequantize_f32),
        DType::F16 => run!(f16, mxfp8_dequantize_f16),
        DType::BF16 => run!(bf16, mxfp8_dequantize_bf16),
        _ => candle_core::bail!("MXFP8 dequantization requires F32, F16 or BF16"),
    };
    drop((weight_guard, scale_guard));
    Ok((output, wl.shape().clone()))
}

pub(super) struct Operands<'a> {
    pub input: &'a CudaStorage,
    pub input_layout: &'a Layout,
    pub weight: &'a CudaStorage,
    pub weight_layout: &'a Layout,
    pub scales: &'a CudaStorage,
    pub scale_layout: &'a Layout,
    pub indices: Option<&'a Tensor>,
}

pub(super) fn matmul(args: Operands<'_>) -> Result<(CudaStorage, Shape)> {
    let dev = args.input.device();
    let stream = dev.cuda_stream();
    let (input_rows, k) = args.input_layout.shape().dims2()?;
    let n = args.weight_layout.dims()[args.weight_layout.dims().len() - 2];
    let (rows, topk) = args
        .indices
        .map(Tensor::dims2)
        .transpose()?
        .unwrap_or((input_rows, 1));
    let (weight_ptr, weight_guard) = slice_ptr_on_stream(
        args.weight.as_cuda_slice::<F8E4M3>()?,
        args.weight_layout.start_offset(),
        &stream,
    );
    let (scale_ptr, scale_guard) = slice_ptr_on_stream(
        args.scales.as_cuda_slice::<u8>()?,
        args.scale_layout.start_offset(),
        &stream,
    );
    let index_storage = args.indices.map(Tensor::storage_and_layout);
    let indices = match &index_storage {
        Some((storage, layout)) => {
            let Storage::Cuda(storage) = &**storage else {
                candle_core::bail!("MXFP8 CUDA routing indices must be on CUDA")
            };
            Some(slice_ptr_on_stream(
                storage.as_cuda_slice::<u32>()?,
                layout.start_offset(),
                &stream,
            ))
        }
        None => None,
    };
    let index_ptr = indices
        .as_ref()
        .map_or(std::ptr::null(), |(pointer, _)| *pointer as *const u32);
    let elements = rows * topk * n;
    macro_rules! run {
        ($ty:ty, $kernel:ident) => {{
            let mut output = unsafe { dev.alloc::<$ty>(elements)? };
            let (input_ptr, input_guard) = slice_ptr_on_stream(
                args.input.as_cuda_slice::<$ty>()?,
                args.input_layout.start_offset(),
                &stream,
            );
            let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let status = unsafe {
                $kernel(
                    input_ptr as *const $ty,
                    weight_ptr as *const u8,
                    scale_ptr as *const u8,
                    index_ptr,
                    output_ptr as *mut $ty,
                    rows as i32,
                    n as i32,
                    k as i32,
                    topk as i32,
                    input_rows != rows,
                    wmma_supported(dev),
                    stream.cu_stream(),
                )
            };
            check_status(status)?;
            drop((input_guard, output_guard));
            CudaStorage::wrap_cuda_slice(output, dev.clone())
        }};
    }
    let output = match args.input.dtype() {
        DType::F32 => run!(f32, mxfp8_matmul_f32),
        DType::F16 => run!(f16, mxfp8_matmul_f16),
        DType::BF16 => run!(bf16, mxfp8_matmul_bf16),
        _ => candle_core::bail!("MXFP8 matmul requires F32, F16 or BF16"),
    };
    drop((weight_guard, scale_guard, indices));
    let shape = if args.indices.is_some() {
        Shape::from((rows, topk, n))
    } else {
        Shape::from((rows, n))
    };
    Ok((output, shape))
}
