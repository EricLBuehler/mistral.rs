use candle_core::DType;
use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLSize,
};
use std::collections::HashMap;
use std::sync::RwLock;

pub mod utils;
use utils::{linear_split, EncoderProvider};

use crate::set_params;

const HQQ_DEQUANTIZE: &str = include_str!("hqq_dequantize.metal");
const BNB_DEQUANTIZE: &str = include_str!("bnb_dequantize.metal");
const BITWISE: &str = include_str!("bitwise.metal");
const RMS_NORM: &str = include_str!("rms_norm.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    HqqDequant,
    BnbDequant,
    Bitwise,
    RmsNorm,
}

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

type Libraries = HashMap<Source, Library>;
type Pipelines = HashMap<String, ComputePipelineState>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let pipelines = RwLock::new(Pipelines::new());
        Self {
            libraries,
            pipelines,
        }
    }

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::HqqDequant => HQQ_DEQUANTIZE,
            Source::BnbDequant => BNB_DEQUANTIZE,
            Source::Bitwise => BITWISE,
            Source::RmsNorm => RMS_NORM,
        }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let mut libraries = self.libraries.write()?;
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let lib = {
                let source_content = self.get_library_source(source);
                device
                    .new_library_with_source(source_content, &CompileOptions::new())
                    .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?
            };
            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &str,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(name, constants)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: &str,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = name.to_string();
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let name = key;
            let func = self.load_function(device, source, &name, None)?;
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
    let pipeline = kernels.load_pipeline(device, Source::HqqDequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::HqqDequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::HqqDequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::HqqDequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::HqqDequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::Bitwise, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::Bitwise, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::BnbDequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::BnbDequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::BnbDequant, name)?;

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
pub fn call_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: &Buffer,
    x_offset: usize,
    x_shape: &[usize],
    w: &Buffer,
    w_offset: usize,
    w_strides: &[usize],
    output: &Buffer,
    output_offset: usize,
    eps: f32,
) -> Result<(), MetalKernelError> {
    let axis_size = x_shape[x_shape.len() - 1];
    let n_rows = x_shape.iter().product::<usize>() / axis_size;

    let w_stride = w_strides[0];

    const SIMD_SIZE: usize = 32;
    const N_READS: usize = 4;
    const LOOPED_LIMIT: usize = 4096;

    let mut name = "rms".to_string();
    if axis_size > LOOPED_LIMIT {
        name.push_str("_looped");
    }
    match ty {
        DType::F32 => name.push_str("float"),
        DType::BF16 => name.push_str("bfloat16"),
        DType::F16 => name.push_str("half"),
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::RmsNorm, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let threadgroup_size = if axis_size <= LOOPED_LIMIT {
        let threadgroup_needed = (axis_size + N_READS - 1) / N_READS; //axis_size.div_ceil(N_READS);
        let simds_needed = (threadgroup_needed + SIMD_SIZE - 1) / SIMD_SIZE; //threadgroup_needed.div_ceil(SIMD_SIZE);
        let threadgroup_size = SIMD_SIZE * simds_needed;
        assert!(threadgroup_size <= pipeline.max_total_threads_per_threadgroup() as usize);
        threadgroup_size
    } else {
        pipeline.max_total_threads_per_threadgroup() as usize
    };

    let nthreads = n_rows * threadgroup_size;
    let grid_dims = MTLSize {
        width: nthreads as u64,
        height: 1,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: threadgroup_size as u64,
        height: 1,
        depth: 1,
    };

    set_params!(
        encoder,
        (
            (x, x_offset),
            (w, w_offset),
            (output, output_offset),
            eps,
            axis_size as u32,
            w_stride as u32
        )
    );

    encoder.dispatch_threads(grid_dims, group_dims);

    Ok(())
}
