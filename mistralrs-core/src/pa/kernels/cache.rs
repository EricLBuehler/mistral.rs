use std::collections::HashMap;

use candle_core::{
    cuda_backend::{cudarc::driver::sys::CUstream, WrapErr},
    Device, Tensor,
};
use core::ffi::c_void;

use super::{
    cuda_get_default_stream, get_scalar_type, get_tensor_device_ptr, DeviceDataPtr, ScalarType,
};

#[repr(C)]
pub struct CopyBlocksParams {
    pub num_layers: i32,
    pub num_pairs: i32,
    pub numel_per_block: i32,
    pub dtype: ScalarType,
}

extern "C" {
    fn copy_blocks(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        params: CopyBlocksParams,
    );
}

#[repr(C)]
struct RehshapeCacheKernelParams {
    value_stride: i32,
    key_stride: i32,
    num_tokens: i32,
    num_heads: i32,
    head_size: i32,
    block_size: i32,
    x: i32,
    dtype: ScalarType,
}

extern "C" {
    fn reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        slot_mapping: *const c_void,
        params: RehshapeCacheKernelParams,
    );
}

enum CopyAction<'a> {
    D2D(&'a Device),
    D2H(&'a Device),
    H2D(&'a Device),
}

impl<'a> CopyAction<'a> {
    fn copy(
        &self,
        dst: DeviceDataPtr,
        src: DeviceDataPtr,
        num_bytes: usize,
        _stream: CUstream,
    ) -> candle_core::Result<()> {
        match self {
            CopyAction::D2D(cuda) => unsafe {
                candle_core::cuda_backend::cudarc::driver::result::memcpy_dtod_async(
                    dst.as_ptr_int(),
                    src.as_ptr_int(),
                    num_bytes,
                    cuda_get_default_stream(cuda)?,
                )
                .w()
            },
            CopyAction::D2H(cuda) => unsafe {
                candle_core::cuda_backend::cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    dst.as_ffi_ptr(),
                    src.as_ptr_int(),
                    num_bytes,
                    cuda_get_default_stream(cuda)?,
                )
                .result()
                .w()
            },
            CopyAction::H2D(cuda) => unsafe {
                candle_core::cuda_backend::cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    dst.as_ptr_int(),
                    src.as_ffi_ptr(),
                    num_bytes,
                    cuda_get_default_stream(cuda)?,
                )
                .result()
                .w()
            },
        }
    }
}

pub fn swap_blocks(
    src: &Tensor,
    dst: &Tensor,
    block_mapping: &HashMap<usize, usize>,
    stream: CUstream,
) -> candle_core::Result<()> {
    let copy_action = if src.device().same_device(dst.device()) {
        if src.device().is_cuda() {
            CopyAction::D2D(src.device())
        } else {
            candle_core::bail!("unsupported device to d2d swap")
        }
    } else if src.device().is_cuda() && dst.device().is_cpu() {
        CopyAction::D2H(src.device())
    } else if src.device().is_cpu() && dst.device().is_cuda() {
        CopyAction::H2D(dst.device())
    } else {
        candle_core::bail!("Invalid device combination to d2h/h2d swap")
    };
    let mut src_data = get_tensor_device_ptr(src)?;
    let mut dst_data = get_tensor_device_ptr(dst)?;

    // const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
    let block_size_in_bytes = src.dtype().size_in_bytes() * (src.elem_count() / src.dims()[0]);
    for (src_block_number, dst_block_number) in block_mapping {
        let src_offset = *src_block_number as usize * block_size_in_bytes;
        let dst_offset = *dst_block_number as usize * block_size_in_bytes;
        copy_action.copy(dst_data, src_data, block_size_in_bytes, stream)?;
        dst_data.advance(dst_offset);
        src_data.advance(src_offset);
    }
    Ok(())
}

pub fn apply_copy_blocks(
    key_caches: Vec<&Tensor>,
    value_caches: Vec<&Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) -> candle_core::Result<()> {
    let num_layers = key_caches.len();
    let mut key_ptrs = Vec::with_capacity(num_layers);
    let mut value_ptrs = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        let key_ptr_int = get_tensor_device_ptr(key_caches[layer_idx])?.as_ptr_int();
        let value_ptr_int = get_tensor_device_ptr(value_caches[layer_idx])?.as_ptr_int();
        key_ptrs.push(key_ptr_int as i64);
        value_ptrs.push(value_ptr_int as i64);
    }

    // Create block mapping array.
    let mut block_mapping_vec = Vec::with_capacity(block_mapping.len());
    for (src_block_number, v) in block_mapping {
        for dst_block_number in v {
            block_mapping_vec.push(*src_block_number as i64);
            block_mapping_vec.push(*dst_block_number as i64);
        }
    }
    let num_pairs = block_mapping_vec.len() / 2;
    let device = key_caches[0].device();
    let key_cache_ptrs_tensor = Tensor::from_vec(key_ptrs, num_layers, device)?;
    let value_cache_ptrs_tensor = Tensor::from_vec(value_ptrs, num_layers, device)?;
    let block_mapping_tensor = Tensor::from_vec(block_mapping_vec, num_pairs * 2, device)?;

    //  const int numel_per_block = key_caches[0][0].numel();
    let numel_per_block = key_caches[0].elem_count() / key_caches[0].dims()[0];
    let key_cache_ptrs_tensor_data = get_tensor_device_ptr(&key_cache_ptrs_tensor)?;
    let value_cache_ptrs_tensor_data = get_tensor_device_ptr(&value_cache_ptrs_tensor)?;
    let block_mapping_tensor_data = get_tensor_device_ptr(&block_mapping_tensor)?;

    let params = CopyBlocksParams {
        num_layers: num_layers as i32,
        num_pairs: num_pairs as i32,
        numel_per_block: numel_per_block as i32,
        dtype: get_scalar_type(key_caches[0].dtype()),
    };
    unsafe {
        copy_blocks(
            key_cache_ptrs_tensor_data.as_ffi_ptr(),
            value_cache_ptrs_tensor_data.as_ffi_ptr(),
            block_mapping_tensor_data.as_ffi_ptr(),
            params,
        );
    }
    Ok(())
}

pub fn apply_reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> candle_core::Result<()> {
    let num_tokens = key.shape().dims()[0];
    let num_heads = key.shape().dims()[1];
    let head_size = key.shape().dims()[2];
    let block_size = key_cache.shape().dims()[3];
    let x = key_cache.shape().dims()[4];
    let key_stride = key.stride()[0];
    let value_stride = value.stride()[0];

    let params = RehshapeCacheKernelParams {
        value_stride: value_stride as i32,
        key_stride: key_stride as i32,
        num_tokens: num_tokens as i32,
        num_heads: num_heads as i32,
        head_size: head_size as i32,
        block_size: block_size as i32,
        x: x as i32,
        dtype: get_scalar_type(key.dtype()),
    };

    let key_data = get_tensor_device_ptr(key)?;
    let value_data = get_tensor_device_ptr(value)?;
    let key_cache_data = get_tensor_device_ptr(key_cache)?;
    let value_cache_data = get_tensor_device_ptr(value_cache)?;
    let slot_mapping_data = get_tensor_device_ptr(slot_mapping)?;
    unsafe {
        reshape_and_cache(
            key_data.as_ffi_ptr(),
            value_data.as_ffi_ptr(),
            key_cache_data.as_ffi_ptr(),
            value_cache_data.as_ffi_ptr(),
            slot_mapping_data.as_ffi_ptr(),
            params,
        );
    }
    Ok(())
}
