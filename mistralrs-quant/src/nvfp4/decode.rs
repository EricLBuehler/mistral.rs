use std::ffi::c_void;

use candle_core::{CudaStorage, DType, Result, Storage, Tensor};
use float8::F8E4M3;
use half::{bf16, f16};

use crate::{
    cutile::{
        device_multiprocessor_count, nvfp4_gemv_geometry, nvfp4_grouped_min_tokens, Nvfp4GemmArgs,
    },
    utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream},
    NVFP4_BLOCK_SIZE,
};

const MAX_TOKENS: usize = 32;
const MAX_ROUTES: usize = u16::MAX as usize;

#[repr(C)]
struct Launch {
    input: *const c_void,
    weight: *const u8,
    scales: *const u8,
    weight_global: *const f32,
    input_global: *const f32,
    indices: *const u32,
    quantized: *mut u8,
    input_scales: *mut u8,
    output: *mut c_void,
    routes: i32,
    n: i32,
    k: i32,
    input_stride: i32,
}

unsafe extern "C" {
    fn mistralrs_nvfp4_decode_f16(
        args: *const Launch,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    ) -> i32;
    fn mistralrs_nvfp4_decode_bf16(
        args: *const Launch,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    ) -> i32;
}

pub(super) fn supports(
    tokens: usize,
    routes: usize,
    dtype: DType,
    args: Nvfp4GemmArgs<'_>,
) -> bool {
    let multiprocessors = if tokens > 1 {
        args.weights
            .device()
            .as_cuda_device()
            .map(device_multiprocessor_count)
            .unwrap_or(1)
    } else {
        1
    };
    supports_on_device(tokens, routes, dtype, args, multiprocessors)
}

fn supports_on_device(
    tokens: usize,
    routes: usize,
    dtype: DType,
    args: Nvfp4GemmArgs<'_>,
    multiprocessors: usize,
) -> bool {
    let dims = args.weights.dims();
    let n = dims[dims.len() - 2];
    let k = dims[dims.len() - 1] * 2;
    (1..=MAX_TOKENS).contains(&tokens)
        && routes
            .checked_mul(tokens)
            .is_some_and(|routes| (1..=MAX_ROUTES).contains(&routes))
        && (tokens == 1
            || (dims.len() == 3
                && tokens < nvfp4_grouped_min_tokens(dims[0], routes)
                && tokens * routes * n.div_ceil(nvfp4_gemv_geometry(n, k).0) > multiprocessors))
        && n >= k
        && k > 0
        && matches!(dtype, DType::BF16 | DType::F16)
        && args.activation_global_scale.is_some()
}

pub(super) fn forward(
    x: &Tensor,
    indices: Option<&Tensor>,
    args: Nvfp4GemmArgs<'_>,
) -> Result<Tensor> {
    let dims = args.weights.dims();
    let n = dims[dims.len() - 2];
    let k = dims[dims.len() - 1] * 2;
    let routes = indices.map_or(1, Tensor::elem_count);
    let input_rows = x.elem_count() / k;
    let x = x.contiguous()?;
    let indices = indices.map(Tensor::contiguous).transpose()?;
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let input_global = args.activation_global_scale.unwrap();
    let (x_storage, x_layout) = x.storage_and_layout();
    let (w_storage, w_layout) = args.weights.storage_and_layout();
    let (s_storage, s_layout) = args.weight_scales.storage_and_layout();
    let (wg_storage, wg_layout) = args.weight_global_scale.storage_and_layout();
    let (ag_storage, ag_layout) = input_global.storage_and_layout();
    let (
        Storage::Cuda(x_storage),
        Storage::Cuda(w_storage),
        Storage::Cuda(s_storage),
        Storage::Cuda(wg_storage),
        Storage::Cuda(ag_storage),
    ) = (
        &*x_storage,
        &*w_storage,
        &*s_storage,
        &*wg_storage,
        &*ag_storage,
    )
    else {
        unreachable!()
    };
    let (weight_ptr, _weight_guard) = slice_ptr_on_stream(
        w_storage.as_cuda_slice::<u8>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (scales_ptr, _scales_guard) = slice_ptr_on_stream(
        s_storage.as_cuda_slice::<F8E4M3>()?,
        s_layout.start_offset(),
        &stream,
    );
    let (weight_global_ptr, _weight_global_guard) = slice_ptr_on_stream(
        wg_storage.as_cuda_slice::<f32>()?,
        wg_layout.start_offset(),
        &stream,
    );
    let (input_global_ptr, _input_global_guard) = slice_ptr_on_stream(
        ag_storage.as_cuda_slice::<f32>()?,
        ag_layout.start_offset(),
        &stream,
    );
    let index_storage = indices.as_ref().map(Tensor::storage_and_layout);
    let index_pointer = index_storage
        .as_ref()
        .map(|(storage, layout)| {
            let Storage::Cuda(storage) = &**storage else {
                unreachable!()
            };
            Ok::<_, candle_core::Error>(slice_ptr_on_stream(
                storage.as_cuda_slice::<u32>()?,
                layout.start_offset(),
                &stream,
            ))
        })
        .transpose()?;
    let mut packed = unsafe { dev.alloc::<u8>(routes * k / 2)? };
    let mut input_scales = unsafe { dev.alloc::<F8E4M3>(routes * k / NVFP4_BLOCK_SIZE)? };
    let (packed_ptr, _packed_guard) = slice_ptr_mut_on_stream(&mut packed, 0, &stream);
    let (input_scales_ptr, _input_scales_guard) =
        slice_ptr_mut_on_stream(&mut input_scales, 0, &stream);
    macro_rules! launch {
        ($ty:ty, $kernel:ident) => {{
            let (input_ptr, _input_guard) = slice_ptr_on_stream(
                x_storage.as_cuda_slice::<$ty>()?,
                x_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<$ty>(routes * n)? };
            let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let status = unsafe {
                $kernel(
                    &Launch {
                        input: input_ptr as *const _,
                        weight: weight_ptr as *const _,
                        scales: scales_ptr as *const _,
                        weight_global: weight_global_ptr as *const _,
                        input_global: input_global_ptr as *const _,
                        indices: index_pointer
                            .as_ref()
                            .map_or(std::ptr::null(), |(ptr, _)| *ptr as *const _),
                        quantized: packed_ptr as *mut _,
                        input_scales: input_scales_ptr as *mut _,
                        output: output_ptr as *mut _,
                        routes: i32::try_from(routes)?,
                        n: i32::try_from(n)?,
                        k: i32::try_from(k)?,
                        input_stride: i32::try_from(routes / input_rows)?,
                    },
                    stream.cu_stream(),
                )
            };
            if status != 0 {
                candle_core::bail!("NVFP4 decode CUDA launch failed with error {status}");
            }
            drop(output_guard);
            CudaStorage::wrap_cuda_slice(output, dev.clone())
        }};
    }
    let output = match x.dtype() {
        DType::F16 => launch!(f16, mistralrs_nvfp4_decode_f16),
        DType::BF16 => launch!(bf16, mistralrs_nvfp4_decode_bf16),
        _ => unreachable!(),
    };
    let output = Tensor::from((Storage::Cuda(output), (routes, n)));
    if let Some(indices) = indices.as_ref() {
        let (tokens, topk) = indices.dims2()?;
        output.reshape((tokens, topk, n))
    } else {
        Ok(output)
    }
}
