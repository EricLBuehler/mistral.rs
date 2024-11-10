use candle_core::DType;
use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library,
};
use std::collections::HashMap;
use std::sync::RwLock;

pub mod utils;
use utils::{linear_split, EncoderProvider};

use crate::set_params;

const DEQUANTIZE: &str = include_str!("dequantize.metal");
const BITWISE: &str = include_str!("bitwise.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Dequant,
    Bitwise,
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
type Pipelines = HashMap<&'static str, ComputePipelineState>;

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
            Source::Dequant => DEQUANTIZE,
            Source::Bitwise => BITWISE,
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
        name: &'static str,
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
        name: &'static str,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = name;
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let name = key;
            let func = self.load_function(device, source, name, None)?;
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
    let pipeline = kernels.load_pipeline(device, Source::Dequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::Dequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::Dequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::Dequant, name)?;

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
    let pipeline = kernels.load_pipeline(device, Source::Dequant, name)?;

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

    set_params!(encoder, (a, b, output));

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

    set_params!(encoder, (a, output, k as u32));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
