use candle_core::{DType, Device, Tensor};

use std::sync::Arc;

// use crate::model_executor::ops::PosEncoding;
use std::sync::OnceLock; //use candle-rotary instead

use candle_core::{cuda_backend::cudarc::driver::sys::CUstream, D};
use core::ffi::c_void;

use super::get_scalar_type;
use super::get_tensor_device_ptr;
use super::ScalarType;

fn compute_inv_freq(base: f32, rotary_dim: usize, device: &Device) -> candle_core::Result<Tensor> {
    let inv_freq: Vec<_> = (0..rotary_dim as u32)
        .step_by(2)
        .map(|i| 1f32 / base.powf(i as f32 / rotary_dim as f32))
        .collect();
    let inv_freq_len = inv_freq.len();
    Tensor::from_vec(inv_freq, (1, inv_freq_len), device)
}

pub fn compute_cos_sin_cache(
    base: f32,
    rotary_dim: usize,
    max_position_embeddings: usize,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let inv_freq = compute_inv_freq(base, rotary_dim, device)?;
    let t = Tensor::arange(0u32, max_position_embeddings as u32, inv_freq.device())?
        .to_dtype(DType::F32)?
        .reshape((max_position_embeddings, 1))?;
    let freqs = t.matmul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    let cache = Tensor::cat(&[cos, sin], D::Minus1)?
        .to_dtype(dtype)?
        .contiguous()?;
    Ok(cache)
}

#[test]
fn test_compute_inv_freq() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = compute_inv_freq(10000.0, 4096, &device)?;

    println!("out:{:?}/{:?}", a.shape(), a.dtype());
    println!("out:{:?}", a.to_string());
    Ok(())
}

#[test]
fn test_compute_cos_sin_cache() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = compute_cos_sin_cache(10000.0, 32, 4096, DType::F16, &device)?;

    println!("out:{:?}/{:?}/{}", a.shape(), a.dtype(), a.is_contiguous());
    println!("out:{:?}", a.to_string());
    Ok(())
}

#[derive(Debug)]
#[repr(C)]
pub struct RotaryEmbeddingKernelParams {
    pub stream: CUstream,
    pub head_size: i32,
    pub num_tokens: i32,
    pub rot_dim: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub query_stride: i32,
    pub key_stride: i32,
    pub is_neox: bool,
    pub scalar_type: ScalarType,
}

extern "C" {
    fn rotary_embedding(
        positions: *const c_void,
        query: *mut c_void,
        key: *mut c_void,
        cos_sin_cache: *const c_void,
        params: RotaryEmbeddingKernelParams,
    );
}

pub fn apply_rotary_embedding(
    position: &Tensor,
    query: &Tensor,
    key: &Tensor,
    cos_sin_cache: &Tensor,
    head_size: usize,
    is_neox: bool,
) -> candle_core::Result<()> {
    //   int64_t num_tokens = query.numel() / query.size(-1);
    //   int rot_dim = cos_sin_cache.size(1);
    //   int num_heads = query.size(-1) / head_size;
    //   int num_kv_heads = key.size(-1) / head_size;
    //   int64_t query_stride = query.stride(-2);
    //   int64_t key_stride = key.stride(-2);

    // let query_last_size = *query.dims().last().unwrap();
    // let num_tokens = query.elem_count() / query_last_size;
    // let num_heads = query_last_size / head_size;
    // let num_kv_heads = *key.dims().last().unwrap();
    let query_strides = query.stride();
    let query_stride = query_strides[query_strides.len() - 2];
    let key_strides = key.stride();
    let key_stride: usize = key_strides[key_strides.len() - 2];

    let rot_dim = cos_sin_cache.dims()[1];
    let (_, _, query_last_size) = query.shape().dims3()?;
    let num_tokens = query.elem_count() / query_last_size;
    let num_heads = query_last_size / head_size;
    let num_kv_heads = *key.dims().last().unwrap() / head_size;
    // let query_stride = query.stride()[0];
    // let key_stride = key.stride()[0];

    // let (num_tokens, num_heads, head_size) = query.shape().dims3()?;
    // let (num_tokens_kv, num_kv_heads, head_size_kv) = key.shape().dims3()?;
    // let query_stride = query.stride()[0];
    // let key_stride = key.stride()[0];

    let params = RotaryEmbeddingKernelParams {
        stream: std::ptr::null_mut(),
        head_size: head_size as i32,
        num_tokens: num_tokens as i32,
        rot_dim: rot_dim as i32,
        num_heads: num_heads as i32,
        num_kv_heads: num_kv_heads as i32,
        query_stride: query_stride as i32,
        key_stride: key_stride as i32,
        is_neox,
        scalar_type: get_scalar_type(query.dtype()),
    };
    // println!("params:{:?}, {:?}", params, key.shape(),);
    let position_data = get_tensor_device_ptr(position)?;
    let query_data = get_tensor_device_ptr(query)?;
    let key_data = get_tensor_device_ptr(key)?;
    let cos_sin_cache_data = get_tensor_device_ptr(cos_sin_cache)?;
    unsafe {
        rotary_embedding(
            position_data.as_ffi_ptr(),
            query_data.as_ffi_ptr(),
            key_data.as_ffi_ptr(),
            cos_sin_cache_data.as_ffi_ptr(),
            params,
        );
    }
    Ok(())
}

// pub fn rotary_embedding_tensor(
//     position: &Tensor,
//     query: &mut Tensor,
//     key: &mut Tensor,
//     cos_sin_cache: &Tensor,
//     mut params: RotaryEmbeddingKernelParams,
// ) -> candle_core::Result<()> {
//     let position_data = get_cuda_device_ptr!(position);
//     let query_data = get_cuda_device_ptr!(query);
//     let key_data = get_cuda_device_ptr!(key);
//     let cos_sin_cache_data = get_cuda_device_ptr!(cos_sin_cache);

//     if params.stream.is_null() {
//         if let Device::Cuda(cuda_dev) = position.device() {
//             params.stream = *(cuda_dev.cu_stream());
//         }
//     }

//     unsafe {
//         rotary_embedding(
//             position_data,
//             query_data,
//             key_data,
//             cos_sin_cache_data,
//             params,
//         );
//     }
//     Ok(())
// }
