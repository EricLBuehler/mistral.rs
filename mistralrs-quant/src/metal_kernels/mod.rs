use candle_core::DType;
use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLSize,
};
use std::ops::Deref;
use std::sync::{Arc, RwLock};
use std::{collections::HashMap, sync::OnceLock};

pub mod utils;
use utils::{get_2d_grid_dims, linear_split, EncoderParam, EncoderProvider};

use crate::set_params;

const HQQ_DEQUANTIZE: &str = include_str!("hqq_dequantize.metal");
const BNB_DEQUANTIZE: &str = include_str!("bnb_dequantize.metal");
const BITWISE: &str = include_str!("bitwise.metal");
const QUANTIZED: &str = include_str!("quantized.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    HqqDequant,
    BnbDequant,
    Bitwise,
    Quantized,
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

static KERNELS: OnceLock<Arc<Kernels_>> = OnceLock::new();

#[derive(Debug)]
pub struct Kernels_ {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels_ {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels_ {
    fn new() -> Self {
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
            Source::Quantized => QUANTIZED,
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
        name: impl ToString,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
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
        source: Source,
        name: impl ToString,
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

pub struct Kernels(Arc<Kernels_>);

impl Kernels {
    pub fn new() -> Self {
        let kernels = KERNELS.get_or_init(|| Arc::new(Kernels_::new()));
        Self(kernels.clone())
    }
}

impl Deref for Kernels {
    type Target = Kernels_;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
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
pub fn call_affine_quantize(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    full_ty: DType,
    input: &Buffer,
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

    let pipeline = kernels.load_pipeline(device, Source::Quantized, &name)?;

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
        set_params!(encoder, (input, scales, biases, output));
    } else {
        set_params!(encoder, (input, output, scales, biases));
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

    let pipeline = kernels.load_pipeline(device, Source::Quantized, &name)?;

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
