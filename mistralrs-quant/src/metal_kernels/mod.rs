use candle_core::{DType, MetalDevice};
use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLSize,
};
use std::os::raw::c_void;
use std::sync::RwLock;
use std::{collections::HashMap, sync::OnceLock};

pub mod utils;
use utils::{
    get_2d_grid_dims, get_2d_grid_dims_divisor, get_block_dims, linear_split, EncoderParam,
    EncoderProvider,
};

use crate::set_params;

#[cfg(target_os = "macos")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mistralrs_quant.metallib"));
#[cfg(target_os = "ios")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mistralrs_quant_ios.metallib"));

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0:?}")]
    LoadFunctionError(String),
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error("dtype mismatch, got {got:?}, expected {expected:?}")]
    DTypeMismatch { expected: Vec<DType>, got: DType },
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type Pipelines = HashMap<String, ComputePipelineState>;

static LIBRARY: OnceLock<Library> = OnceLock::new();

#[derive(Debug)]
pub struct Kernels {
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let pipelines = RwLock::new(Pipelines::new());
        Self { pipelines }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(&self, device: &Device) -> Result<Library, MetalKernelError> {
        if let Some(lib) = LIBRARY.get() {
            Ok(lib.clone())
        } else {
            let source_data = KERNELS;
            let lib = device.new_library_with_data(source_data).map_err(|e| {
                MetalKernelError::LoadLibraryError(format!(
                    "Metal requires macosx > 13.0 or higher, cannot load candle metal library: {e}"
                ))
            })?;
            Ok(LIBRARY.get_or_init(|| lib).clone())
        }
    }

    fn load_function(
        &self,
        device: &Device,
        name: impl ToString,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device)?
            .get_function(&name.to_string(), constants)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        name: impl ToString,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = name.to_string();
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let name = key;
            let func = self.load_function(device, &name, None)?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert(name, pipeline.clone());

            Ok(pipeline)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_8bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    weight: &Buffer,
    scale: &Buffer,
    zero: &Buffer,
    h: u32,
    w: u32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "dequantize_8bit_float",
        DType::BF16 => "dequantize_8bit_bfloat",
        DType::F16 => "dequantize_8bit_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let length = h * w;

    set_params!(encoder, (weight, scale, zero, output, h, w));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length as usize);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_4bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    weight: &Buffer,
    scale: &Buffer,
    zero: &Buffer,
    h: u32,
    w: u32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "dequantize_4bit_float",
        DType::BF16 => "dequantize_4bit_bfloat",
        DType::F16 => "dequantize_4bit_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let length = h * w;

    set_params!(encoder, (weight, scale, zero, output, h, w));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length as usize);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_2bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    weight: &Buffer,
    scale: &Buffer,
    zero: &Buffer,
    h: u32,
    w: u32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "dequantize_2bit_float",
        DType::BF16 => "dequantize_2bit_bfloat",
        DType::F16 => "dequantize_2bit_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let length = h * w;

    set_params!(encoder, (weight, scale, zero, output, h, w));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length as usize);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_1bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    weight: &Buffer,
    scale: &Buffer,
    zero: &Buffer,
    h: u32,
    w: u32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "dequantize_1bit_float",
        DType::BF16 => "dequantize_1bit_bfloat",
        DType::F16 => "dequantize_1bit_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let length = h * w;

    set_params!(encoder, (weight, scale, zero, output, h, w));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length as usize);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_3bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    weight: &Buffer,
    scale: &Buffer,
    zero: &Buffer,
    h: u32,
    w: u32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "dequantize_3bit_float",
        DType::BF16 => "dequantize_3bit_bfloat",
        DType::F16 => "dequantize_3bit_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let length = h * w;

    set_params!(encoder, (weight, scale, zero, output, h, w));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length as usize);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_bitwise_not(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    a: &Buffer,
    a_offset: usize,
    length: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::U8 => "bitwise_not_uint8_t",
        DType::U32 => "bitwise_not_uint32_t",
        DType::I64 => "bitwise_not_int64_t",
        DType::I32 => "bitwise_not_int",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::U8, DType::U32, DType::I64, DType::I32],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, ((a, a_offset), output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_bitwise_or(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    a: &Buffer,
    b: &Buffer,
    a_offset: usize,
    b_offset: usize,
    length: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::U8 => "bitwise_or_uint8_t",
        DType::U32 => "bitwise_or_uint32_t",
        DType::I64 => "bitwise_or_int64_t",
        DType::I32 => "bitwise_or_int",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::U8, DType::U32, DType::I64, DType::I32],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, ((a, a_offset), (b, b_offset), output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_bitwise_and(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    a: &Buffer,
    b: &Buffer,
    a_offset: usize,
    b_offset: usize,
    length: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::U8 => "bitwise_and_uint8_t",
        DType::U32 => "bitwise_and_uint32_t",
        DType::I64 => "bitwise_and_int64_t",
        DType::I32 => "bitwise_and_int",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::U8, DType::U32, DType::I64, DType::I32],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, ((a, a_offset), (b, b_offset), output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_bitwise_xor(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    a: &Buffer,
    b: &Buffer,
    a_offset: usize,
    b_offset: usize,
    length: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::U8 => "bitwise_xor_uint8_t",
        DType::U32 => "bitwise_xor_uint32_t",
        DType::I64 => "bitwise_xor_int64_t",
        DType::I32 => "bitwise_xor_int",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::U8, DType::U32, DType::I64, DType::I32],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, ((a, a_offset), (b, b_offset), output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_bitwise_leftshift(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    a: &Buffer,
    a_offset: usize,
    k: u32,
    length: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::U8 => "bitwise_leftshift_uint8_t",
        DType::U32 => "bitwise_leftshift_uint32_t",
        DType::I64 => "bitwise_leftshift_int64_t",
        DType::I32 => "bitwise_leftshift_int",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::U8, DType::U32, DType::I64, DType::I32],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, ((a, a_offset), output, k));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_bnb_nf4(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    input: &Buffer,
    absmax: &Buffer,
    code: &Buffer,
    output: &Buffer,
    blocksize: usize,
    n: usize,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "kernel_dequantize_nf4_float",
        DType::BF16 => "kernel_dequantize_nf4_bfloat16_t",
        DType::F16 => "kernel_dequantize_nf4_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (code, input, absmax, output, blocksize as i32, n as i32)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n.div_ceil(blocksize));
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_bnb_fp4(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    input: &Buffer,
    absmax: &Buffer,
    code: &Buffer,
    output: &Buffer,
    blocksize: usize,
    n: usize,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "kernel_dequantize_fp4_float",
        DType::BF16 => "kernel_dequantize_fp4_bfloat16_t",
        DType::F16 => "kernel_dequantize_fp4_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (code, input, absmax, output, blocksize as i32, n as i32)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n.div_ceil(blocksize));
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_bnb_int8(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    input: &Buffer,
    absmax: &Buffer,
    code: &Buffer,
    output: &Buffer,
    blocksize: usize,
    n: usize,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "kernel_dequantize_int8_float",
        DType::BF16 => "kernel_dequantize_int8_bfloat16_t",
        DType::F16 => "kernel_dequantize_int8_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (code, input, absmax, output, blocksize as i32, n as i32)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n.div_ceil(blocksize));
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_affine_quantize(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    full_ty: DType,
    input: &Buffer,
    input_offset: usize,
    input_dims: &[usize],
    input_strides: &[usize],
    output: &Buffer,
    output_dims: &[usize],
    scales: &Buffer,
    biases: &Buffer,
    dequantize: bool,
    group_size: usize,
    bits: usize,
) -> Result<(), MetalKernelError> {
    let type_string = match full_ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let kernel_func = if dequantize {
        "affine_dequantize"
    } else {
        "affine_quantize"
    };
    let name = format!("{kernel_func}_{type_string}_gs_{group_size}_b_{bits}");

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Treat uint32 as uint8 in kernel
    let uint8_per_uint32 = 4;
    let simd_size = 32;
    let packs_per_int = match bits {
        3 => 8,
        6 => 4,
        _ => 8 / bits,
    };
    let per_thread = if dequantize {
        packs_per_int
    } else {
        group_size / simd_size
    };
    let nthreads = if dequantize {
        output_dims.iter().product::<usize>() / packs_per_int
    } else {
        input_dims.iter().product::<usize>() / per_thread
    };

    let thread_group_size = (pipeline.max_total_threads_per_threadgroup() as usize).min(nthreads);
    let group_dims = MTLSize {
        width: thread_group_size as u64,
        height: 1,
        depth: 1,
    };
    let use_2d = nthreads > u32::MAX as usize;
    let mut grid_shape = input_dims.to_vec();
    if dequantize {
        *grid_shape.last_mut().unwrap() *= uint8_per_uint32;
    } else {
        *grid_shape.last_mut().unwrap() /= per_thread;
    }
    let grid_dims = if use_2d {
        get_2d_grid_dims(&grid_shape, input_strides)
    } else {
        MTLSize {
            width: nthreads as u64,
            height: 1,
            depth: 1,
        }
    };

    if dequantize {
        set_params!(encoder, ((input, input_offset), scales, biases, output));
    } else {
        set_params!(encoder, ((input, input_offset), output, scales, biases));
    }

    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_afq_qmm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: &Buffer,
    x_shape: &[usize],
    x_stride: &[usize],
    w: &Buffer,
    w_shape: &[usize],
    w_stride: &[usize],
    scales: &Buffer,
    s_stride: &[usize],
    biases: &Buffer,
    b_stride: &[usize],
    out: &Buffer,
    out_shape: &[usize],
    gather_lhs_rhs_indices: Option<(&Buffer, &Buffer)>,
    gather_lhs_shape: Option<&[usize]>,
    gather_lhs_rhs_strides: Option<(&[usize], &[usize])>,
    transpose: bool,
    bits: usize,
    group_size: usize,
) -> Result<(), MetalKernelError> {
    let gather = gather_lhs_rhs_indices.is_some();

    let batched = !gather && w_shape.len() > 2;

    let d = x_shape[x_shape.len() - 1];
    let o = out_shape[out_shape.len() - 1];
    // For the unbatched W case, avoid `adjust_matrix_offsets`
    // for a small performance gain.
    let b = if batched || gather {
        x_shape[x_shape.len() - 2]
    } else {
        x_shape.iter().product::<usize>() / d
    };
    let n = if batched || gather {
        out_shape.iter().product::<usize>() / b / o
    } else {
        1
    };

    let mut name = if gather {
        "bs_".to_string()
    } else {
        "".to_string()
    };
    let mut matrix = false;
    let mut aligned = false;
    let mut quad = false;

    let (group_dims, grid_dims) = if transpose {
        if b < 6 && (d == 128 || d == 64) && bits.is_power_of_two() {
            name.push_str("qmv_quad");
            let quads_per_simd = 8;
            let results_per_simdgroup = 8;
            let bo = quads_per_simd * results_per_simdgroup;
            let simdgroup_size = 32;
            quad = true;
            let group_dims = MTLSize {
                width: simdgroup_size as u64,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: o.div_ceil(bo) as u64,
                height: b as u64,
                depth: n as u64,
            };
            (group_dims, grid_dims)
        } else if b < 6 && o % 8 == 0 && d % 512 == 0 && d >= 512 {
            name.push_str("qmv_fast");
            let bo = 8;
            let bd = 32;
            let group_dims = MTLSize {
                width: bd,
                height: 2,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: (o / bo) as u64,
                height: b as u64,
                depth: n as u64,
            };
            (group_dims, grid_dims)
        } else if b < 6 {
            name.push_str("qmv");
            let bo = 8;
            let bd = 32;
            let group_dims = MTLSize {
                width: bd,
                height: 2,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: o.div_ceil(bo) as u64,
                height: b as u64,
                depth: n as u64,
            };
            (group_dims, grid_dims)
        } else {
            name.push_str("qmm_t");
            let wn = 2;
            let wm = 2;
            let bm = 32;
            let bn = 32;
            let group_dims = MTLSize {
                width: 32,
                height: wn as u64,
                depth: wm as u64,
            };
            let grid_dims = MTLSize {
                width: o.div_ceil(bn) as u64,
                height: b.div_ceil(bm) as u64,
                depth: n as u64,
            };
            matrix = true;
            aligned = true;
            (group_dims, grid_dims)
        }
    } else {
        /*if b < 4 && d >= 1024 {
            todo!("qvm_split_k");
        } else */
        if b < 4 {
            name.push_str("qvm");
            let bo = 64;
            let bd = 32;
            let group_dims = MTLSize {
                width: bd,
                height: 2,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: (o / bo) as u64,
                height: b as u64,
                depth: n as u64,
            };
            (group_dims, grid_dims)
        } else {
            name.push_str("qmm_n");
            let wn = 2;
            let wm = 2;
            let bm = 32;
            let bn = 32;
            let group_dims = MTLSize {
                width: 32,
                height: wn as u64,
                depth: wm as u64,
            };
            let grid_dims = MTLSize {
                width: (o / bn) as u64,
                height: b.div_ceil(bm) as u64,
                depth: n as u64,
            };
            matrix = true;
            if o % bn != 0 {
                panic!("output size should be divisible by {bn} but received {o}.");
            }
            (group_dims, grid_dims)
        }
    };

    let aligned_n = if o % 32 == 0 { "true" } else { "false" };

    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };

    name = format!("{name}_{type_string}_gs_{group_size}_b_{bits}");
    if quad {
        name.push_str(&format!("_d_{d}"));
    }
    if aligned {
        name.push_str(&format!("_alN_{aligned_n}"));
    }
    if !gather {
        name.push_str(&format!("_batch_{}", batched as usize));
    }

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (w, scales, biases, x, out, d as i32, o as i32));

    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(scales), 0);
    encoder.set_buffer(2, Some(biases), 0);
    encoder.set_buffer(3, Some(x), 0);
    encoder.set_buffer(4, Some(out), 0);
    <i32 as EncoderParam>::set_param(encoder, 5, d as i32);
    <i32 as EncoderParam>::set_param(encoder, 6, o as i32);

    let mut offset = 7;
    if matrix {
        <i32 as EncoderParam>::set_param(encoder, 7, b as i32);
        offset += 1;
    }

    let x_batch_ndims = x_shape.len() - 2;
    let w_batch_ndims = w_shape.len() - 2;

    if batched || gather {
        <i32 as EncoderParam>::set_param(encoder, offset, x_batch_ndims as i32);
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 1,
            &x_shape.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 2,
            &x_stride.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <i32 as EncoderParam>::set_param(encoder, offset + 3, w_batch_ndims as i32);
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 4,
            &w_shape.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 5,
            &w_stride.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 6,
            &s_stride.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 7,
            &b_stride.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
    }
    if gather {
        let (lhs_indices, rhs_indices) = gather_lhs_rhs_indices.unwrap();
        let batch_shape = gather_lhs_shape.unwrap();
        let batch_ndims = batch_shape.len();
        let (lhs_strides, rhs_strides) = gather_lhs_rhs_strides.unwrap();

        <i32 as EncoderParam>::set_param(encoder, offset + 8, batch_ndims as i32);
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 9,
            &batch_shape.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        encoder.set_buffer(offset + 10, Some(lhs_indices), 0);
        encoder.set_buffer(offset + 11, Some(rhs_indices), 0);
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 12,
            &lhs_strides.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 13,
            &rhs_strides.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
    }

    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_blockwise_fp8(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    weight: &Buffer,
    scales: &Buffer,
    output: &Buffer,
    weight_height: u32,
    weight_width: u32,
    weight_row_stride: u32,
    scale_stride: u32,
    block_size_y: u32,
    block_size_x: u32,
) -> Result<(), MetalKernelError> {
    #[repr(C)]
    struct DequantParams {
        weight_height: u32,
        weight_width: u32,
        weight_row_stride: u32,
        scale_stride: u32,
        block_size_y: u32,
        block_size_x: u32,
    }

    let name = match ty {
        DType::F32 => "dequant_fp8_blockwise_float",
        DType::BF16 => "dequant_fp8_blockwise_bfloat16_t",
        DType::F16 => "dequant_fp8_blockwise_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let dequant_params = DequantParams {
        weight_height,
        weight_width,
        weight_row_stride,
        scale_stride,
        block_size_y,
        block_size_x,
    };

    impl EncoderParam for &DequantParams {
        fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
            encoder.set_bytes(
                position,
                core::mem::size_of_val(data) as u64,
                &data as *const _ as *const c_void,
            );
        }
    }

    set_params!(encoder, (weight, scales, output, &dequant_params));

    let tg = MTLSize {
        width: 32,
        height: 32,
        depth: 1,
    };
    let blocks = MTLSize {
        width: weight_width.div_ceil(dequant_params.block_size_x) as u64,
        height: weight_height.div_ceil(dequant_params.block_size_y) as u64,
        depth: 1,
    };

    encoder.dispatch_thread_groups(blocks, tg);
    Ok(())
}

#[allow(dead_code)]
pub enum ScanType {
    Sum,
    Prod,
    Max,
    Min,
    LogAddExp,
}

#[allow(clippy::too_many_arguments)]
pub fn call_scan(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    op: ScanType,
    xs: &Buffer,
    xs_offset: usize,
    axis: usize,
    shape: &[usize],
    strides: &[usize],
    reverse: bool,
    inclusive: bool,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let contiguous = strides[axis] == 1;
    let mut name = if contiguous {
        "contig_".to_string()
    } else {
        "strided_".to_string()
    };
    name.push_str("scan_");
    if reverse {
        name.push_str("reverse_");
    }
    if inclusive {
        name.push_str("inclusive_");
    } else {
        name.push_str("exclusive_");
    }
    match op {
        ScanType::Sum => name.push_str("sum_"),
        ScanType::Prod => name.push_str("prod_"),
        ScanType::Max => name.push_str("max_"),
        ScanType::Min => name.push_str("min_"),
        ScanType::LogAddExp => name.push_str("logaddexp_"),
    }

    let type_name = match ty {
        DType::F32 => "float32",
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        DType::U8 => "uint8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![
                    DType::F32,
                    DType::F16,
                    DType::BF16,
                    DType::U8,
                    DType::I16,
                    DType::I32,
                    DType::I64,
                ],
                got: other,
            })
        }
    };
    name.push_str(&format!("{type_name}_{type_name}"));

    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    if contiguous {
        encoder.set_buffer(0, Some(xs), xs_offset as u64);
        encoder.set_buffer(1, Some(output), 0);

        let size = shape[axis];
        encoder.set_bytes(
            2,
            std::mem::size_of::<usize>() as u64,
            &size as *const usize as *const _,
        );

        // Compute the thread grid
        let n_reads = if ty.size_in_bytes() <= 4 { 4 } else { 2 };
        let simd_size = 32;
        let elements_per_simd = n_reads * simd_size;
        let mut thread_group_size = pipeline.max_total_threads_per_threadgroup() as usize;
        if size <= n_reads * 1024 {
            thread_group_size = size.div_ceil(elements_per_simd) * simd_size;
        } else if size <= n_reads * 2048 {
            thread_group_size = (size / 2).div_ceil(elements_per_simd) * simd_size;
        }
        thread_group_size =
            thread_group_size.min(pipeline.max_total_threads_per_threadgroup() as usize);
        let tmp_grid_dims = get_2d_grid_dims_divisor(shape, strides, size);

        let grid_dims = MTLSize {
            width: thread_group_size as u64,
            height: tmp_grid_dims.width,
            depth: tmp_grid_dims.height,
        };
        let group_dims = MTLSize {
            width: thread_group_size as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_dims, group_dims);
    } else {
        let size = shape[axis];
        let stride = strides[size];
        let _bm = 32;
        let bn = 32;
        let stride_blocks = stride.div_ceil(bn);

        encoder.set_buffer(0, Some(xs), xs_offset as u64);
        encoder.set_buffer(1, Some(output), 0);

        encoder.set_bytes(
            2,
            std::mem::size_of::<usize>() as u64,
            &size as *const usize as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<usize>() as u64,
            &stride as *const usize as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<usize>() as u64,
            &stride_blocks as *const usize as *const _,
        );

        // Compute the thread grid
        let n_reads = if ty.size_in_bytes() <= 4 { 4 } else { 2 };
        let n_simdgroups = bn / n_reads;
        let thread_group_size = n_simdgroups * 32;
        let mut tmp_grid_dims = get_2d_grid_dims_divisor(shape, strides, size * stride);
        if tmp_grid_dims.width.saturating_mul(stride_blocks as u64) <= u64::MAX {
            tmp_grid_dims.width *= stride_blocks as u64;
        } else {
            tmp_grid_dims.height *= stride_blocks as u64;
        }

        let grid_dims = MTLSize {
            width: thread_group_size as u64,
            height: tmp_grid_dims.width,
            depth: tmp_grid_dims.height,
        };
        let group_dims = MTLSize {
            width: thread_group_size as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_dims, group_dims);
    }

    Ok(())
}

/// Immutable parameters that fully describe a single `sort` / `argsort`
/// operation.  Bundling them keeps the public API readable as the kernel
/// signatures grow.
///
/// All slice references point into caller‑owned data – the struct itself
/// owns **no** memory.
#[derive(Debug)]
pub struct SortArgs<'a> {
    /// Axis along which the values are sorted.
    pub axis: usize,
    /// Shape of the input tensor.
    pub shape: &'a [usize],
    /// Strides (element‑wise) for the input tensor.
    pub strides: &'a [usize],
    /// Shape of the output tensor.
    pub out_shape: &'a [usize],
    /// Strides (element‑wise) for the output tensor.
    pub out_strides: &'a [usize],
    /// Whether the input tensor is already contiguous in memory.
    pub in_contiguous: bool,
    /// Element type of the input tensor.
    pub in_ty: DType,
    /// Element type of the output tensor (`u32` for `argsort`).
    pub out_ty: DType,
    /// GPU buffer holding the input tensor elements.
    pub src: &'a Buffer,
    /// Element offset into `src` where the region to be sorted begins.
    pub src_offset: usize,
    /// GPU buffer that will receive the sorted values / indices.
    pub dst: &'a Buffer,
}

/// Scratch buffers that can be reused between consecutive multi‑block sort
/// calls.  Providing them avoids the internal allocations performed each
/// time by `call_multi_block_sort`.
///
/// The enum is kept deliberately non‑exhaustive so that additional cached
/// resources can be introduced without breaking existing code.
#[derive(Debug)]
pub enum MultiBlockSortCache<'a> {
    /// Let the helper allocate fresh scratch buffers for this invocation.
    None,
    /// Re‑use caller‑provided temporary buffers.
    External {
        dev_vals_ping: &'a Buffer,
        dev_vals_pong: &'a Buffer,
        dev_idxs_ping: &'a Buffer,
        dev_idxs_pong: &'a Buffer,
        block_partitions: &'a Buffer,
    },
}

/// How the copy kernel should behave.
#[derive(Copy, Clone)]
enum CopyType {
    /// The last axis is contiguous – we can treat the tensor as a 1‑D slice.
    Vector,
    /// Arbitrary layout – we need to jump using the supplied strides.
    General,
}

fn type_to_name(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U32 => "uint32",
        DType::U8 => "uint8",
        _ => "unknown",
    }
}

#[allow(clippy::too_many_arguments)]
fn call_copy_gpu_inplace(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    shape: &[usize],
    src_strides: &[usize],
    dst_strides: &[usize],
    src: &Buffer,
    src_offset: usize,
    dst: &Buffer,
    dst_offset: usize,
    copy_type: CopyType,
) -> Result<(), MetalKernelError> {
    // === Constants & helpers =================================================
    const MAX_COPY_SPECIALIZED_DIMS: usize = 3;

    /// https://github.com/ml-explore/mlx/blob/eebe73001affcb424171e9d49657e508f70a9201/mlx/backend/metal/utils.h#L87
    #[inline]
    fn work_per_thread_for_dtype(dt: DType) -> usize {
        let wpt = 8 / dt.size_in_bytes().max(1);
        wpt.max(1)
    }

    #[inline]
    fn ceil_div(x: usize, y: usize) -> usize {
        (x + y - 1) / y
    }

    // === Sanity checks =======================================================
    if shape.is_empty() {
        // Nothing to do – mimic early‑return in the C++ version.
        return Ok(());
    }
    assert!(
        shape.len() == src_strides.len() && shape.len() == dst_strides.len(),
        "shape/stride rank mismatch in call_copy_gpu_inplace"
    );

    // === Derived sizes / flags ==============================================
    let elem_count: usize = shape.iter().product();
    let large = match copy_type {
        CopyType::General => {
            // Allow negative strides – original code switches to 32‑bit indexing once strides
            // are flattened, but here we only care about the element count threshold.
            elem_count > i32::MAX as usize
        }
        CopyType::Vector => elem_count > u32::MAX as usize,
    };
    let work_per_thread = match copy_type {
        CopyType::Vector => work_per_thread_for_dtype(ty),
        CopyType::General => {
            if shape.len() > MAX_COPY_SPECIALIZED_DIMS {
                if large {
                    4
                } else {
                    2
                }
            } else {
                1
            }
        }
    };

    // === Kernel name construction ===========================================
    let mut kernel_name = match copy_type {
        CopyType::Vector => {
            if large {
                "v2".to_owned()
            } else {
                "v".to_owned()
            }
        }
        CopyType::General => "g".to_owned(),
    };

    if matches!(copy_type, CopyType::General) {
        if shape.len() <= MAX_COPY_SPECIALIZED_DIMS {
            kernel_name.push_str(&shape.len().to_string());
        } else {
            kernel_name.push_str(&format!("n{}", work_per_thread));
        }
        if large {
            kernel_name.push_str("large");
        }
    }

    // The original MLX kernels encode source/destination dtypes in the mangled
    // name, even when they are the same.
    kernel_name.push_str(&format!("_copy{}{}", type_to_name(ty), type_to_name(ty)));

    // === Pipeline & encoder ==================================================
    let pipeline = kernels.load_pipeline(device, &kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // === Buffers (slots 0/1) =================================================
    let byte_offset_src = src_offset * ty.size_in_bytes();
    let byte_offset_dst = dst_offset * ty.size_in_bytes();
    encoder.set_buffer(0, Some(src), byte_offset_src as u64);
    encoder.set_buffer(1, Some(dst), byte_offset_dst as u64);

    // === Specialisation for each CopyType ===================================
    match copy_type {
        // ---------------------------------------------------------------------
        CopyType::Vector => {
            // Slot 2 – total elements (32‑ or 64‑bit depending on `large`)
            if large {
                <i64 as EncoderParam>::set_param(encoder, 2, elem_count as i64);
            } else {
                <i32 as EncoderParam>::set_param(encoder, 2, elem_count as i32);
            }

            // Grid
            let nthreads = ceil_div(elem_count, work_per_thread);
            let tg_size = pipeline.max_total_threads_per_threadgroup() as usize;
            let tg_size = tg_size.min(nthreads);

            let group_dims = MTLSize {
                width: tg_size as u64,
                height: 1,
                depth: 1,
            };
            let grid_dims = if large {
                // 64‑bit indexing path – fall back to 2‑D tiling helper.
                get_2d_grid_dims_divisor(shape, dst_strides, work_per_thread)
            } else {
                MTLSize {
                    width: nthreads as u64,
                    height: 1,
                    depth: 1,
                }
            };
            encoder.dispatch_threads(grid_dims, group_dims);
        }

        // ---------------------------------------------------------------------
        CopyType::General => {
            let ndim = shape.len() as i32;

            // ---- Shape / stride descriptors ---------------------------------
            if shape.len() > 3 {
                let shape_i32: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
                encoder.set_bytes(
                    2,
                    (std::mem::size_of::<i32>() * shape_i32.len()) as u64,
                    shape_i32.as_ptr() as *const _,
                );
            }
            // Strides – always required (slot 3)
            let strides_in_i64: Vec<i64> = src_strides.iter().map(|&x| x as i64).collect();
            encoder.set_bytes(
                3,
                (std::mem::size_of::<i64>() * strides_in_i64.len()) as u64,
                strides_in_i64.as_ptr() as *const _,
            );

            // If the kernel is the generic “gn*” variant we also pass `ndim`
            if shape.len() > MAX_COPY_SPECIALIZED_DIMS {
                <i32 as EncoderParam>::set_param(encoder, 5, ndim);
            }

            // ---- Thread/work‑grid ------------------------------------------
            let mut dim0 = *shape.last().unwrap_or(&1);
            let dim1 = if shape.len() > 1 {
                shape[shape.len() - 2]
            } else {
                1
            };
            let rest = elem_count / (dim0 * dim1);
            if shape.len() > MAX_COPY_SPECIALIZED_DIMS {
                dim0 = ceil_div(dim0, work_per_thread);
            }

            assert_eq!(
                pipeline.max_total_threads_per_threadgroup(),
                1024,
                "copy kernels expect a 1024-lane threadgroup"
            );
            let group_dims = get_block_dims(dim0, dim1, rest, 10);
            let grid_dims = MTLSize {
                width: dim0 as u64,
                height: dim1 as u64,
                depth: rest as u64,
            };
            encoder.dispatch_threads(grid_dims, group_dims);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_single_block_sort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    bn: usize,
    tn: usize,
    argsort: bool,
) -> Result<(), MetalKernelError> {
    // --- destructure helper -------------------------------------------------
    let axis = args.axis;
    let shape = args.shape;
    let strides = args.strides;
    let out_strides = args.out_strides;
    let in_contiguous = args.in_contiguous;
    let in_ty = args.in_ty;
    let out_ty = args.out_ty;
    let src = args.src;
    let src_offset = args.src_offset;
    let dst = args.dst;
    let n_rows = shape.iter().product::<usize>() / shape[axis];

    let mut in_nc_str = strides.iter().map(|x| *x as i64).collect::<Vec<_>>();
    in_nc_str.remove(axis);

    let mut out_nc_str = out_strides.iter().map(|x| *x as i64).collect::<Vec<_>>();
    out_nc_str.remove(axis);

    let mut nc_shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
    nc_shape.remove(axis);

    let nc_dim = nc_shape.len();

    let size_sorted_axis = shape[axis];
    let in_stride_sorted_axis = strides[axis];
    let out_stride_sorted_axis = out_strides[axis];

    macro_rules! check_strides {
        ($strides:expr, $sort_stride:expr) => {{
            let min_stride = $strides.iter().min().unwrap();
            let max_stride = $strides.iter().max().unwrap();
            $sort_stride == *min_stride || $sort_stride == *max_stride
        }};
    }
    let contiguous = in_contiguous
        && check_strides!(strides, in_stride_sorted_axis)
        && check_strides!(out_strides, out_stride_sorted_axis);

    use std::fmt::Write as _;
    let mut name = String::new();
    if contiguous {
        name.push_str("c");
    } else {
        name.push_str("nc");
    }
    if argsort {
        name.push_str("arg");
    }
    write!(
        &mut name,
        "_block_sort_{}_{}_bn{}_tn{}",
        type_to_name(in_ty),
        type_to_name(out_ty),
        bn,
        tn
    )
    .unwrap();

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Buffers
    encoder.set_buffer(0, Some(src), src_offset as u64);
    encoder.set_buffer(1, Some(dst), 0);

    // Scalar params
    <i32 as EncoderParam>::set_param(encoder, 2, size_sorted_axis as i32);
    <i32 as EncoderParam>::set_param(encoder, 3, in_stride_sorted_axis as i32);
    <i32 as EncoderParam>::set_param(encoder, 4, out_stride_sorted_axis as i32);

    if contiguous {
        let mut in_stride_segment_axis = i32::MAX;
        let mut out_stride_segment_axis = i32::MAX;
        for i in 0..in_nc_str.len() {
            if nc_shape[i] == 1 {
                continue;
            }
            if in_nc_str[i] > i32::MAX as i64 || out_nc_str[i] > i32::MAX as i64 {
                panic!(
                    "Stride too large in single_block_sort in {} out {}",
                    in_nc_str[i], out_nc_str[i]
                );
            }
            in_stride_segment_axis = in_stride_segment_axis.min(in_nc_str[i] as i32);
            out_stride_segment_axis = out_stride_segment_axis.min(out_nc_str[i] as i32);
        }

        <i32 as EncoderParam>::set_param(encoder, 5, in_stride_segment_axis);
        <i32 as EncoderParam>::set_param(encoder, 6, out_stride_segment_axis);
    } else {
        <i32 as EncoderParam>::set_param(encoder, 5, nc_dim as i32);
        if nc_shape.is_empty() {
            let shape = 0i32;
            let stride = 0i64;

            <i32 as EncoderParam>::set_param(encoder, 6, shape as i32);
            <i64 as EncoderParam>::set_param(encoder, 7, stride as i64);
            <i64 as EncoderParam>::set_param(encoder, 8, stride as i64);
        } else {
            encoder.set_bytes(
                6,
                (std::mem::size_of::<i32>() * nc_shape.len()) as u64,
                nc_shape.as_ptr() as *const _,
            );
            encoder.set_bytes(
                7,
                (std::mem::size_of::<i64>() * in_nc_str.len()) as u64,
                in_nc_str.as_ptr() as *const _,
            );
            encoder.set_bytes(
                8,
                (std::mem::size_of::<i64>() * out_nc_str.len()) as u64,
                out_nc_str.as_ptr() as *const _,
            );
        }
    }

    // Dispatch
    let group_dims = MTLSize {
        width: bn as u64,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: 1,
        height: n_rows as u64,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_multi_block_sort<'a>(
    metal_device: &MetalDevice,
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    bn: usize,
    tn: usize,
    n_blocks: usize,
    argsort: bool,
    _cache: &MultiBlockSortCache<'a>,
) -> Result<(), MetalKernelError> {
    // --- destructure helper -------------------------------------------------
    let axis = args.axis;
    let shape = args.shape;
    let strides = args.strides;
    let out_strides = args.out_strides;
    let in_ty = args.in_ty;
    let out_ty = args.out_ty;
    let src = args.src;
    let src_offset = args.src_offset;
    let dst = args.dst;
    let n_rows = shape.iter().product::<usize>() / shape[axis];

    let mut nc_str = strides.iter().map(|x| *x as i64).collect::<Vec<_>>();
    nc_str.remove(axis);

    let mut nc_shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
    nc_shape.remove(axis);

    let nc_dim = nc_shape.len();

    if nc_dim == 0 {
        nc_shape = vec![0];
        nc_str = vec![1];
    }

    let size_sorted_axis = shape[axis];
    let stride_sorted_axis = strides[axis];

    // Make temporary copies
    let dev_vals_0 = metal_device
        .new_buffer(n_rows * size_sorted_axis, in_ty, "dev_vals_0")
        .unwrap();
    let dev_vals_1 = metal_device
        .new_buffer(n_rows * size_sorted_axis, in_ty, "dev_vals_1")
        .unwrap();

    let dev_idxs_0 = metal_device
        .new_buffer(n_rows * size_sorted_axis, DType::U32, "dev_idxs_0")
        .unwrap();
    let dev_idxs_1 = metal_device
        .new_buffer(n_rows * size_sorted_axis, DType::U32, "dev_idxs_1")
        .unwrap();

    let block_partitions = metal_device
        .new_buffer(n_rows * (n_blocks + 1), DType::U32, "block_partitions")
        .unwrap();

    // Do blockwise sort
    {
        let name = format!(
            "sort_mbsort_{}_{}_bn{bn}_tn{tn}",
            type_to_name(in_ty),
            type_to_name(DType::U32)
        );

        let pipeline = kernels.load_pipeline(device, &name)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(0, Some(src), src_offset as u64);
        encoder.set_buffer(1, Some(&dev_vals_0), 0);
        encoder.set_buffer(2, Some(&dev_idxs_0), 0);
        <i32 as EncoderParam>::set_param(encoder, 3, size_sorted_axis as i32);
        <i32 as EncoderParam>::set_param(encoder, 4, stride_sorted_axis as i32);
        <i32 as EncoderParam>::set_param(encoder, 5, nc_dim as i32);
        encoder.set_bytes(
            6,
            (std::mem::size_of::<i32>() * nc_shape.len()) as u64,
            nc_shape.as_ptr() as *const _,
        );
        encoder.set_bytes(
            7,
            (std::mem::size_of::<i64>() * nc_str.len()) as u64,
            nc_str.as_ptr() as *const _,
        );

        // Dispatch
        let group_dims = MTLSize {
            width: bn as u64,
            height: 1,
            depth: 1,
        };
        let grid_dims = MTLSize {
            width: n_blocks as u64,
            height: n_rows as u64,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Do merges
    let mut ping = false;
    #[allow(unused_assignments)]
    let mut dev_vals_in = dev_vals_0.clone();
    #[allow(unused_assignments)]
    let mut dev_idxs_in = dev_idxs_0.clone();
    let mut dev_vals_out = dev_vals_1.clone();
    let mut dev_idxs_out = dev_idxs_1.clone();

    let n_thread_per_group = std::cmp::min(n_blocks + 1, 1024);

    let mut merge_tiles = 2;
    while (merge_tiles / 2) < n_blocks {
        dev_vals_in = if ping {
            dev_vals_1.clone()
        } else {
            dev_vals_0.clone()
        };
        dev_idxs_in = if ping {
            dev_idxs_1.clone()
        } else {
            dev_idxs_0.clone()
        };
        dev_vals_out = if ping {
            dev_vals_0.clone()
        } else {
            dev_vals_1.clone()
        };
        dev_idxs_out = if ping {
            dev_idxs_0.clone()
        } else {
            dev_idxs_1.clone()
        };
        ping = !ping;

        // Do partition
        {
            let name = format!(
                "partition_mbsort_{}_{}_bn{bn}_tn{tn}",
                type_to_name(in_ty),
                type_to_name(DType::U32)
            );

            let pipeline = kernels.load_pipeline(device, &name)?;

            let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);

            encoder.set_buffer(0, Some(&block_partitions), 0);
            encoder.set_buffer(1, Some(&dev_vals_in), 0);
            encoder.set_buffer(2, Some(&dev_idxs_in), 0);
            <i32 as EncoderParam>::set_param(encoder, 3, size_sorted_axis as i32);
            <i32 as EncoderParam>::set_param(encoder, 4, merge_tiles as i32);
            <i32 as EncoderParam>::set_param(encoder, 5, n_blocks as i32);

            // Dispatch
            let group_dims = MTLSize {
                width: n_thread_per_group as u64,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: 1,
                height: n_rows as u64,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid_dims, group_dims);
        }

        // Do merge
        {
            let name = format!(
                "merge_mbsort_{}_{}_bn{bn}_tn{tn}",
                type_to_name(in_ty),
                type_to_name(DType::U32)
            );

            let pipeline = kernels.load_pipeline(device, &name)?;

            let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);

            encoder.set_buffer(0, Some(&block_partitions), 0);
            encoder.set_buffer(1, Some(&dev_vals_in), 0);
            encoder.set_buffer(2, Some(&dev_idxs_in), 0);
            encoder.set_buffer(3, Some(&dev_vals_out), 0);
            encoder.set_buffer(4, Some(&dev_idxs_out), 0);
            <i32 as EncoderParam>::set_param(encoder, 5, size_sorted_axis as i32);
            <i32 as EncoderParam>::set_param(encoder, 6, merge_tiles as i32);
            <i32 as EncoderParam>::set_param(encoder, 7, n_blocks as i32);

            // Dispatch
            let group_dims = MTLSize {
                width: bn as u64,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: n_blocks as u64,
                height: n_rows as u64,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid_dims, group_dims);
        }

        merge_tiles *= 2;
    }

    // Copy outputs with appropriate strides
    let mut strides = out_strides.to_vec();
    for ax in axis + 1..strides.len() {
        strides[ax] *= args.out_shape[axis];
    }
    strides[axis] = 1;

    let copy_src = if argsort { dev_idxs_out } else { dev_vals_out };

    call_copy_gpu_inplace(
        device,
        ep,
        kernels,
        out_ty,
        args.out_shape,
        &strides,
        out_strides,
        &copy_src,
        0,
        dst,
        0,
        if axis == shape.len() - 1 {
            CopyType::Vector
        } else {
            CopyType::General
        },
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_block_sort<'a>(
    metal_device: &MetalDevice,
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    argsort: bool,
    cache: &MultiBlockSortCache<'a>,
) -> Result<(), MetalKernelError> {
    // --- destructure helper -------------------------------------------------
    let axis = args.axis;
    let shape = args.shape;
    let in_ty = args.in_ty;

    let size_sorted_axis = shape[axis];

    let tn = 4usize;
    let mut bn = match (size_sorted_axis + tn - 1) / tn {
        v if v > 256 => 512,
        v if v > 128 => 256,
        v if v > 64 => 128,
        v if v > 32 => 64,
        _ => 32,
    };
    if bn == 512 && in_ty.size_in_bytes() > 4 {
        bn = 256;
    }
    let n_per_block = bn * tn;
    let n_blocks = (size_sorted_axis + n_per_block - 1) / n_per_block;
    if n_blocks > 1 {
        // multi‑block path
        call_multi_block_sort(
            metal_device,
            device,
            ep,
            kernels,
            args,
            bn,
            tn,
            n_blocks,
            argsort,
            cache,
        )
    } else {
        // single‑block path
        call_single_block_sort(device, ep, kernels, args, bn, tn, argsort)
    }
}

/// Sorts values along the last axis of a contiguous tensor.
pub fn call_sort<'a>(
    metal_device: &MetalDevice,
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    cache: &MultiBlockSortCache<'a>,
) -> Result<(), MetalKernelError> {
    call_block_sort(
        metal_device,
        device,
        ep,
        kernels,
        args,
        /* argsort = */ false,
        cache,
    )
}

/// Returns the indices that would sort `src` along the last axis.
pub fn call_argsort<'a>(
    metal_device: &MetalDevice,
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    cache: &MultiBlockSortCache<'a>,
) -> Result<(), MetalKernelError> {
    call_block_sort(
        metal_device,
        device,
        ep,
        kernels,
        args,
        /* argsort = */ true,
        cache,
    )
}
