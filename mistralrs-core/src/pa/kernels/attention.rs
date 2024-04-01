use candle_core::{DType, Tensor};
use core::ffi::c_void;

use super::{get_scalar_type, get_tensor_device_ptr, DeviceDataPtr, ScalarType};

const _PARTITION_SIZE: usize = 512;

#[repr(C)]
struct PagedAttentionV1Params {
    block_size: i32,
    num_seqs: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_size: i32,
    max_num_blocks_per_seq: i32,
    q_stride: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
    max_context_len: i32,
    scale: f32,
    scalar_type: ScalarType,
}
extern "C" {
    fn paged_attention_v1(
        out: *mut c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const c_void,
        context_lens: *const c_void,
        alibi_slopes: *const c_void,
        params: PagedAttentionV1Params,
    );
}
#[allow(clippy::too_many_arguments)]
pub fn apply_paged_attention_v1(
    scale: f32,
    max_context_len: usize,
    num_kv_heads: usize,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    apply_paged_attention_v1_(
        scale,
        max_context_len,
        num_kv_heads,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        alibi_slopes,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn apply_paged_attention_v1_(
    scale: f32,
    max_context_len: usize,
    num_kv_heads: usize,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    //let output = query.zeros_like()?;
    let output = unsafe { Tensor::empty(query.shape(), query.dtype(), query.device())? };
    let block_size = value_cache.shape().dims()[3] as i32;
    let num_seqs = query.shape().dims()[0] as i32;
    let num_heads = query.shape().dims()[1] as i32;
    let head_size = query.shape().dims()[2] as i32;

    let max_num_blocks_per_seq = block_tables.shape().dims()[1] as i32;
    let q_stride = query.stride()[0] as i32;
    let kv_block_stride = key_cache.stride()[0] as i32;
    let kv_head_stride = key_cache.stride()[1] as i32;

    let params = PagedAttentionV1Params {
        block_size,
        num_seqs,
        num_heads,
        num_kv_heads: num_kv_heads as i32,
        head_size,
        max_num_blocks_per_seq,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        max_context_len: max_context_len as i32,
        scale,
        scalar_type: get_scalar_type(query.dtype()),
    };

    let out_data = get_tensor_device_ptr(&output)?;
    let query_data = get_tensor_device_ptr(query)?;
    let key_cache_data = get_tensor_device_ptr(key_cache)?;
    let value_cache_data = get_tensor_device_ptr(value_cache)?;
    let block_tables_data = get_tensor_device_ptr(block_tables)?;
    let context_lens_data = get_tensor_device_ptr(context_lens)?;
    let alibi_slopes_data = if let Some(alibi_slopes) = alibi_slopes {
        get_tensor_device_ptr(alibi_slopes)?
    } else {
        DeviceDataPtr::null()
    };

    unsafe {
        paged_attention_v1(
            out_data.as_ffi_ptr(),
            query_data.as_ffi_ptr(),
            key_cache_data.as_ffi_ptr(),
            value_cache_data.as_ffi_ptr(),
            block_tables_data.as_ffi_ptr(),
            context_lens_data.as_ffi_ptr(),
            alibi_slopes_data.as_ffi_ptr(),
            params,
        );
    }
    Ok(output)
}

#[repr(C)]
struct PagedAttentionV2Params {
    block_size: i32,
    num_seqs: i32,
    num_heads: i32,
    head_size: i32,
    num_kv_heads: i32,
    max_num_blocks_per_seq: i32,
    q_stride: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
    max_context_len: i32,
    scale: f32,
    scalar_type: ScalarType,
}

extern "C" {
    fn paged_attention_v2(
        out: *mut c_void,
        exp_sums: *mut c_void,
        max_logits: *mut c_void,
        tmp_out: *mut c_void,
        query: *mut c_void,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        block_tables: *mut c_void,
        context_lens: *mut c_void,
        alibi_slopes: *mut c_void,
        params: PagedAttentionV2Params,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn apply_paged_attention_v2(
    scale: f32,
    max_context_len: usize,
    num_kv_heads: usize,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    apply_paged_attention_v2_(
        scale,
        max_context_len,
        num_kv_heads,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        alibi_slopes,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn apply_paged_attention_v2_(
    scale: f32,
    max_context_len: usize,
    num_kv_heads: usize,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    // let output = query.zeros_like()?;
    let output = unsafe { Tensor::empty(query.shape(), query.dtype(), query.device())? };
    let num_seqs = query.shape().dims()[0];
    let num_heads = query.shape().dims()[1];
    let head_size = query.shape().dims()[2];
    let max_num_blocks_per_seq = block_tables.shape().dims()[1];
    let q_stride = query.stride()[0] as i32;
    let kv_block_stride = key_cache.stride()[0] as i32;
    let kv_head_stride = key_cache.stride()[1] as i32;
    let block_size = value_cache.shape().dims()[3] as i32;

    let max_num_partitions = (max_context_len + _PARTITION_SIZE - 1) / _PARTITION_SIZE;
    // let tmp_out = Tensor::zeros(
    //     (num_seqs, num_heads, max_num_partitions, head_size),
    //     output.dtype(),
    //     output.device(),
    // )?;
    let tmp_out = unsafe {
        Tensor::empty(
            (num_seqs, num_heads, max_num_partitions, head_size),
            output.dtype(),
            output.device(),
        )?
    };
    // let exp_sums = Tensor::zeros(
    //     (num_seqs, num_heads, max_num_partitions),
    //     DType::F32,
    //     output.device(),
    // )?;
    let exp_sums = unsafe {
        Tensor::empty(
            (num_seqs, num_heads, max_num_partitions),
            DType::F32,
            output.device(),
        )?
    };
    // let max_logits = exp_sums.zeros_like()?;
    let max_logits =
        unsafe { Tensor::empty(exp_sums.shape(), exp_sums.dtype(), exp_sums.device())? };

    let params = PagedAttentionV2Params {
        block_size,
        max_context_len: max_context_len as i32,
        num_seqs: num_seqs as i32,
        num_heads: num_heads as i32,
        head_size: head_size as i32,
        max_num_blocks_per_seq: max_num_blocks_per_seq as i32,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        num_kv_heads: num_kv_heads as i32,
        scale,
        scalar_type: get_scalar_type(query.dtype()),
    };

    let alibi_slopes_data = if let Some(alibi_slopes) = alibi_slopes {
        get_tensor_device_ptr(alibi_slopes)?
    } else {
        DeviceDataPtr::null()
    };
    let out_data = get_tensor_device_ptr(&output)?;
    let query_data = get_tensor_device_ptr(query)?;
    let exp_sums_data = get_tensor_device_ptr(&exp_sums)?;
    let max_logits_data = get_tensor_device_ptr(&max_logits)?;
    let tmp_out_data = get_tensor_device_ptr(&tmp_out)?;
    let key_cache_data = get_tensor_device_ptr(key_cache)?;
    let value_cache_data = get_tensor_device_ptr(value_cache)?;
    let block_tables_data = get_tensor_device_ptr(block_tables)?;
    let context_lens_data = get_tensor_device_ptr(context_lens)?;
    unsafe {
        paged_attention_v2(
            out_data.as_ffi_ptr(),
            exp_sums_data.as_ffi_ptr(),
            max_logits_data.as_ffi_ptr(),
            tmp_out_data.as_ffi_ptr(),
            query_data.as_ffi_ptr(),
            key_cache_data.as_ffi_ptr(),
            value_cache_data.as_ffi_ptr(),
            block_tables_data.as_ffi_ptr(),
            context_lens_data.as_ffi_ptr(),
            alibi_slopes_data.as_ffi_ptr(),
            params,
        );
    }
    Ok(output)
}
