use super::{EncoderProvider, Kernels, MetalKernelError};
use candle_core::{DType, Layout, Shape};
use metal::{Buffer, ComputeCommandEncoderRef, FunctionConstantValues, MTLSize};
use std::os::raw::c_void;

#[derive(Debug)]
#[repr(C)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: i64,
    batch_stride_b: i64,
    batch_stride_d: i64,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

#[allow(clippy::too_many_arguments)]
pub fn call_gather_mm_rhs(
    device: &metal::Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    a_shape: &Shape,
    a_layout: &Layout,
    a_buffer: &Buffer,
    a_offset: usize,
    b_shape: &Shape,
    b_layout: &Layout,
    b_buffer: &Buffer,
    b_offset: usize,
    indices_buffer: &Buffer,
    indices_offset: usize,
    out_buffer: &Buffer,
    out_offset: usize,
) -> Result<(), MetalKernelError> {
    let m = a_shape.dims()[a_shape.dims().len() - 2];
    let k = a_shape.dims()[a_shape.dims().len() - 1];
    let n = b_shape.dims()[b_shape.dims().len() - 1];
    
    let a_strides = a_layout.stride();
    let b_strides = b_layout.stride();
    
    // Check transpose by examining strides
    let (transpose_a, lda) = if a_strides[a_strides.len() - 1] == 1 {
        (false, k as i32)
    } else if a_strides[a_strides.len() - 2] == 1 {
        (true, m as i32)
    } else {
        return Err(MetalKernelError::LoadLibraryError(
            "Matrix A must be contiguous in last dimension".to_string(),
        ));
    };
    
    let (transpose_b, ldb) = if b_strides[b_strides.len() - 1] == 1 {
        (false, n as i32)
    } else if b_strides[b_strides.len() - 2] == 1 {
        (true, k as i32)
    } else {
        return Err(MetalKernelError::LoadLibraryError(
            "Matrix B must be contiguous in last dimension".to_string(),
        ));
    };
    
    // Kernel parameters
    let (bm, bn, bk, wm, wn) = (16, 64, 16, 1, 2);
    
    let align_m = m % bm == 0;
    let align_n = n % bn == 0;
    let align_k = k % bk == 0;
    
    let dtype_str = match dtype {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        _ => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: dtype,
            })
        }
    };
    
    let trans_str = match (transpose_a, transpose_b) {
        (false, false) => "nn",
        (false, true) => "nt",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(
                "gather_mm_rhs only supports nn and nt transposes".to_string(),
            ))
        }
    };
    
    let kernel_name = format!(
        "gather_mm_rhs_{}_{}_{}_bm{}_bn{}_bk{}_wm{}_wn{}",
        trans_str, dtype_str, dtype_str, bm, bn, bk, wm, wn
    );
    
    let batch_stride_b = if b_shape.dims().len() > 2 {
        b_strides[b_strides.len() - 3] as i64
    } else {
        (n * k) as i64
    };
    
    let params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda,
        ldb,
        ldd: n as i32,
        tiles_n: ((n + bn - 1) / bn) as i32,
        tiles_m: ((m + bm - 1) / bm) as i32,
        batch_stride_a: 0,
        batch_stride_b,
        batch_stride_d: 0,
        swizzle_log: 0,
        gemm_k_iterations_aligned: (k / bk) as i32,
        batch_ndim: 0,
    };
    
    // Set up function constants
    let constants = FunctionConstantValues::new();
    let false_val = false;
    constants.set_constant_value_at_index(&false_val as *const bool as *const c_void, metal::MTLDataType::Bool, 10);
    constants.set_constant_value_at_index(&align_m as *const bool as *const c_void, metal::MTLDataType::Bool, 200);
    constants.set_constant_value_at_index(&align_n as *const bool as *const c_void, metal::MTLDataType::Bool, 201);
    constants.set_constant_value_at_index(&align_k as *const bool as *const c_void, metal::MTLDataType::Bool, 202);
    
    let func = kernels.load_function(device, &kernel_name, Some(constants))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
    
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a_buffer), a_offset as u64);
    encoder.set_buffer(1, Some(b_buffer), b_offset as u64);
    encoder.set_buffer(2, Some(indices_buffer), indices_offset as u64);
    encoder.set_buffer(3, Some(out_buffer), out_offset as u64);
    encoder.set_bytes(
        4,
        std::mem::size_of::<GemmParams>() as u64,
        &params as *const GemmParams as *const c_void,
    );
    
    let grid_size = MTLSize {
        width: params.tiles_n as u64,
        height: params.tiles_m as u64,
        depth: 1,
    };
    let group_size = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    
    encoder.use_resource(a_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(b_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(indices_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(out_buffer, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_gather_mm(
    device: &metal::Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    a_shape: &Shape,
    a_layout: &Layout,
    a_buffer: &Buffer,
    a_offset: usize,
    b_shape: &Shape,
    b_layout: &Layout,
    b_buffer: &Buffer,
    b_offset: usize,
    lhs_indices_shape: &Shape,
    lhs_indices_layout: &Layout,
    lhs_indices_buffer: &Buffer,
    lhs_indices_offset: usize,
    rhs_indices_shape: &Shape,
    rhs_indices_layout: &Layout,
    rhs_indices_buffer: &Buffer,
    rhs_indices_offset: usize,
    out_shape: &Shape,
    out_buffer: &Buffer,
    out_offset: usize,
) -> Result<(), MetalKernelError> {
    let m = a_shape.dims()[a_shape.dims().len() - 2];
    let k = a_shape.dims()[a_shape.dims().len() - 1];
    let n = b_shape.dims()[b_shape.dims().len() - 1];
    
    let a_strides = a_layout.stride();
    let b_strides = b_layout.stride();
    
    // Check transpose by examining strides
    let (transpose_a, lda) = if a_strides[a_strides.len() - 1] == 1 {
        (false, k as i32)
    } else if a_strides[a_strides.len() - 2] == 1 {
        (true, m as i32)
    } else {
        return Err(MetalKernelError::LoadLibraryError(
            "Matrix A must be contiguous in last dimension".to_string(),
        ));
    };
    
    let (transpose_b, ldb) = if b_strides[b_strides.len() - 1] == 1 {
        (false, n as i32)
    } else if b_strides[b_strides.len() - 2] == 1 {
        (true, k as i32)
    } else {
        return Err(MetalKernelError::LoadLibraryError(
            "Matrix B must be contiguous in last dimension".to_string(),
        ));
    };
    
    // Select kernel parameters based on architecture
    let (bm, bn, bk, wm, wn) = (64, 64, 16, 2, 2);
    
    let align_m = m % bm == 0;
    let align_n = n % bn == 0;
    let align_k = k % bk == 0;
    
    let dtype_str = match dtype {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        _ => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: dtype,
            })
        }
    };
    
    let trans_str = match (transpose_a, transpose_b) {
        (false, false) => "nn",
        (false, true) => "nt",
        (true, false) => "tn",
        (true, true) => "tt",
    };
    
    let kernel_name = format!(
        "gather_mm_{}_{}_{}_bm{}_bn{}_bk{}_wm{}_wn{}",
        trans_str, dtype_str, dtype_str, bm, bn, bk, wm, wn
    );
    
    let batch_size_out = out_shape.elem_count() / (m * n);
    let batch_ndim = out_shape.dims().len() - 2;
    let batch_ndim_a = a_shape.dims().len() - 2;
    let batch_ndim_b = b_shape.dims().len() - 2;
    
    let has_batch = batch_ndim > 1;
    
    let lhs_indices_strides = lhs_indices_layout.stride();
    let rhs_indices_strides = rhs_indices_layout.stride();
    
    let params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda,
        ldb,
        ldd: n as i32,
        tiles_n: ((n + bn - 1) / bn) as i32,
        tiles_m: ((m + bm - 1) / bm) as i32,
        batch_stride_a: if batch_ndim > 0 { lhs_indices_strides[0] as i64 } else { 0 },
        batch_stride_b: if batch_ndim > 0 { rhs_indices_strides[0] as i64 } else { 0 },
        batch_stride_d: (m * n) as i64,
        swizzle_log: 0,
        gemm_k_iterations_aligned: (k / bk) as i32,
        batch_ndim: batch_ndim as i32,
    };
    
    // Set up function constants
    let constants = FunctionConstantValues::new();
    constants.set_constant_value_at_index(&has_batch as *const bool as *const c_void, metal::MTLDataType::Bool, 10);
    constants.set_constant_value_at_index(&align_m as *const bool as *const c_void, metal::MTLDataType::Bool, 200);
    constants.set_constant_value_at_index(&align_n as *const bool as *const c_void, metal::MTLDataType::Bool, 201);
    constants.set_constant_value_at_index(&align_k as *const bool as *const c_void, metal::MTLDataType::Bool, 202);
    
    let func = kernels.load_function(device, &kernel_name, Some(constants))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
    
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a_buffer), a_offset as u64);
    encoder.set_buffer(1, Some(b_buffer), b_offset as u64);
    encoder.set_buffer(2, Some(lhs_indices_buffer), lhs_indices_offset as u64);
    encoder.set_buffer(3, Some(rhs_indices_buffer), rhs_indices_offset as u64);
    encoder.set_buffer(4, Some(out_buffer), out_offset as u64);
    encoder.set_bytes(
        5,
        std::mem::size_of::<GemmParams>() as u64,
        &params as *const GemmParams as *const c_void,
    );
    
    // Set indices shape and strides
    let indices_shape_vec: Vec<i32> = lhs_indices_shape.dims().iter().map(|&x| x as i32).collect();
    let lhs_strides_vec: Vec<i64> = lhs_indices_strides.iter().map(|&x| x as i64).collect();
    let rhs_strides_vec: Vec<i64> = rhs_indices_strides.iter().map(|&x| x as i64).collect();
    
    // Use set_bytes for array data
    encoder.set_bytes(
        6,
        (std::mem::size_of::<i32>() * indices_shape_vec.len()) as u64,
        indices_shape_vec.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        7,
        (std::mem::size_of::<i64>() * lhs_strides_vec.len()) as u64,
        lhs_strides_vec.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        8,
        (std::mem::size_of::<i64>() * rhs_strides_vec.len()) as u64,
        rhs_strides_vec.as_ptr() as *const c_void,
    );
    
    // Set batch info for A
    encoder.set_bytes(9, 4, &batch_ndim_a as *const usize as *const c_void);
    let a_shape_vec: Vec<i32> = a_shape.dims().iter().map(|&x| x as i32).collect();
    let a_strides_vec: Vec<i64> = a_strides.iter().map(|&x| x as i64).collect();
    encoder.set_bytes(
        10,
        (std::mem::size_of::<i32>() * a_shape_vec.len()) as u64,
        a_shape_vec.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        11,
        (std::mem::size_of::<i64>() * a_strides_vec.len()) as u64,
        a_strides_vec.as_ptr() as *const c_void,
    );
    
    // Set batch info for B
    encoder.set_bytes(12, 4, &batch_ndim_b as *const usize as *const c_void);
    let b_shape_vec: Vec<i32> = b_shape.dims().iter().map(|&x| x as i32).collect();
    let b_strides_vec: Vec<i64> = b_strides.iter().map(|&x| x as i64).collect();
    encoder.set_bytes(
        13,
        (std::mem::size_of::<i32>() * b_shape_vec.len()) as u64,
        b_shape_vec.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        14,
        (std::mem::size_of::<i64>() * b_strides_vec.len()) as u64,
        b_strides_vec.as_ptr() as *const c_void,
    );
    
    let grid_size = MTLSize {
        width: params.tiles_n as u64,
        height: params.tiles_m as u64,
        depth: batch_size_out as u64,
    };
    let group_size = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    
    encoder.use_resource(a_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(b_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(lhs_indices_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_indices_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(out_buffer, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    
    Ok(())
}