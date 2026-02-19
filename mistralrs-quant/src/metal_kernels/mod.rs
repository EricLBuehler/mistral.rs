// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use candle_core::{DType, MetalDevice};
use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, ConstantValues, Device, Function, Library,
};
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLMathMode, MTLSize};
use std::os::raw::c_void;
use std::sync::{Arc, RwLock};
use std::{collections::HashMap, sync::OnceLock};

pub mod utils;
use utils::{
    get_2d_grid_dims, get_2d_grid_dims_divisor, get_block_dims, linear_split, EncoderParam,
    EncoderProvider, RawBytesEncoder,
};

use crate::set_params;

// Backward-compatible aliases to ease the transition from the `metal` crate API.
type ComputeCommandEncoderRef = ComputeCommandEncoder;
type ComputePipelineState = ComputePipeline;

#[cfg(target_os = "macos")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mistralrs_quant.metallib"));
#[cfg(target_os = "ios")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mistralrs_quant_ios.metallib"));
#[cfg(target_os = "tvos")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mistralrs_quant_tvos.metallib"));

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading function: {0:?}")]
    LoadFunctionError(String),
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error("dtype mismatch, got {got:?}, expected {expected:?}")]
    DTypeMismatch { expected: Vec<DType>, got: DType },
    #[error("Failed to compile Metal shader: {0}")]
    CompilationError(String),
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type Pipelines = HashMap<String, ComputePipeline>;

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

    /// Load the library from precompiled metallib, falling back to runtime compilation if needed.
    /// If this has been previously loaded it will just fetch it from cache.
    #[allow(clippy::const_is_empty)] // KERNELS can be empty when MISTRALRS_METAL_PRECOMPILE=0
    pub fn load_library(&self, device: &Device) -> Result<Library, MetalKernelError> {
        if let Some(lib) = LIBRARY.get() {
            Ok(lib.clone())
        } else {
            // Try to load precompiled metallib first (faster startup)
            let lib = if !KERNELS.is_empty() {
                // Load precompiled metallib directly from embedded bytes via DispatchData.
                // This avoids writing to a temp file, which can fail in sandboxed
                // environments (e.g. macOS apps distributed via TestFlight).
                // https://github.com/EricLBuehler/mistral.rs/issues/1897
                let data = dispatch2::DispatchData::from_static_bytes(KERNELS);

                let raw_lib = device
                    .as_ref()
                    .newLibraryWithData_error(&data)
                    .map_err(|e| {
                        MetalKernelError::CompilationError(format!(
                            "Failed to load precompiled metallib: {e}"
                        ))
                    })?;
                Library::new(raw_lib)
            } else {
                // Fall back to runtime compilation if precompiled lib is not available
                // (e.g., when MISTRALRS_METAL_PRECOMPILE=0)
                self.compile_kernels_at_runtime(device)?
            };
            Ok(LIBRARY.get_or_init(|| lib).clone())
        }
    }

    fn compile_kernels_at_runtime(&self, device: &Device) -> Result<Library, MetalKernelError> {
        use std::collections::{HashMap, HashSet};

        // Create a virtual filesystem with all our Metal sources
        let mut file_system = HashMap::new();
        file_system.insert("bitwise.metal", include_str!("bitwise.metal"));
        file_system.insert("blockwise_fp8.metal", include_str!("blockwise_fp8.metal"));
        file_system.insert("bnb_dequantize.metal", include_str!("bnb_dequantize.metal"));
        file_system.insert("fused_glu.metal", include_str!("fused_glu.metal"));
        file_system.insert("hqq_dequantize.metal", include_str!("hqq_dequantize.metal"));
        file_system.insert("mxfp4.metal", include_str!("mxfp4.metal"));
        file_system.insert("quantized.metal", include_str!("quantized.metal"));
        file_system.insert("scan.metal", include_str!("scan.metal"));
        file_system.insert("sort.metal", include_str!("sort.metal"));
        file_system.insert("copy.metal", include_str!("copy.metal"));
        file_system.insert("utils.metal", include_str!("utils.metal"));
        file_system.insert("bf16.metal", include_str!("bf16.metal"));
        file_system.insert("scan_impl.metal", include_str!("scan_impl.metal"));
        file_system.insert("sort_impl.metal", include_str!("sort_impl.metal"));
        file_system.insert("copy_impl.metal", include_str!("copy_impl.metal"));
        file_system.insert("float8.metal", include_str!("float8.metal"));
        file_system.insert("float4.metal", include_str!("float4.metal"));
        file_system.insert("scalar_fp8.metal", include_str!("scalar_fp8.metal"));
        file_system.insert(
            "sdpa_with_sinks.metal",
            include_str!("sdpa_with_sinks.metal"),
        );
        file_system.insert(
            "softmax_with_sinks.metal",
            include_str!("softmax_with_sinks.metal"),
        );

        // Recursive include preprocessor
        fn preprocess_includes(
            content: &str,
            current_file: &str,
            file_system: &HashMap<&str, &str>,
            included_files: &mut HashSet<String>,
            include_stack: &mut Vec<String>,
        ) -> Result<String, String> {
            // Check for circular includes
            if include_stack.contains(&current_file.to_string()) {
                return Err(format!(
                    "Circular include detected: {} -> {}",
                    include_stack.join(" -> "),
                    current_file
                ));
            }

            include_stack.push(current_file.to_string());

            let mut result = String::new();
            let lines = content.lines();

            for line in lines {
                let trimmed = line.trim();

                // Check for #include directive
                if trimmed.starts_with("#include") {
                    // Extract the included filename
                    if let Some(start) = trimmed.find('"') {
                        if let Some(end) = trimmed[start + 1..].find('"') {
                            let include_file = &trimmed[start + 1..start + 1 + end];

                            // Check if this is one of our local files
                            if let Some(included_content) = file_system.get(include_file) {
                                // Only include each file once (like #pragma once)
                                if !included_files.contains(include_file) {
                                    included_files.insert(include_file.to_string());

                                    // Recursively process the included file
                                    let processed = preprocess_includes(
                                        included_content,
                                        include_file,
                                        file_system,
                                        included_files,
                                        include_stack,
                                    )?;

                                    result.push_str(&format!(
                                        "\n// ===== Start of {include_file} =====\n"
                                    ));
                                    result.push_str(&processed);
                                    result.push_str(&format!(
                                        "\n// ===== End of {include_file} =====\n"
                                    ));
                                }
                                // Skip the original #include line
                                continue;
                            } else if !trimmed.contains('<') {
                                // This is a quoted include but not one of our files
                                // Skip it to avoid "file not found" errors
                                continue;
                            }
                        }
                    }
                    // For system includes (with < >), keep them
                    if trimmed.contains('<') {
                        result.push_str(line);
                        result.push('\n');
                    }
                } else if trimmed == "#pragma once" {
                    // Skip #pragma once as we handle it differently
                    continue;
                } else {
                    // Fix backslash-newline warnings by removing trailing spaces
                    if line.ends_with("\\ ") || line.ends_with("\\\t") {
                        let cleaned = line.trim_end();
                        let without_backslash = cleaned.trim_end_matches('\\');
                        result.push_str(without_backslash);
                        result.push_str(" \\");
                    } else {
                        result.push_str(line);
                    }
                    result.push('\n');
                }
            }

            include_stack.pop();
            Ok(result)
        }

        // Start with a clean slate
        let mut included_files = HashSet::new();
        let mut include_stack = Vec::new();

        // Build the main source file
        let mut main_source = String::new();

        // Add standard Metal includes first
        main_source.push_str("#include <metal_stdlib>\n");
        main_source.push_str("#include <metal_common>\n");
        main_source.push_str("#include <metal_math>\n");
        main_source.push_str("#include <metal_integer>\n");
        main_source.push_str("#include <metal_simdgroup>\n");
        main_source.push_str("#include <metal_simdgroup_matrix>\n");
        main_source.push_str("\nusing namespace metal;\n\n");

        // Process only the top-level files that contain kernel definitions
        // The implementation files (_impl.metal) and utility files will be included
        // automatically through the preprocessor when processing these files
        // Note: bf16.metal is excluded as it only contains type definitions that
        // are already in utils.metal, which would cause duplicate definitions
        let main_files = vec![
            "bitwise.metal",        // Bitwise operations
            "blockwise_fp8.metal",  // FP8 blockwise operations (includes float8.metal, utils.metal)
            "bnb_dequantize.metal", // BitsAndBytes dequantization (includes utils.metal)
            "fused_glu.metal",      // Fused GLU operations (activation(a) * b)
            "hqq_dequantize.metal", // HQQ dequantization
            "mxfp4.metal",          // MXFP4 kernels
            "quantized.metal",      // Quantization operations (includes utils.metal)
            "copy.metal",           // Copy operations (includes utils.metal, copy_impl.metal)
            "scan.metal",           // Scan operations (includes utils.metal, scan_impl.metal)
            "sort.metal",           // Sort operations (includes utils.metal, sort_impl.metal)
        ];

        for file in main_files {
            if !included_files.contains(file) {
                if let Some(content) = file_system.get(file) {
                    match preprocess_includes(
                        content,
                        file,
                        &file_system,
                        &mut included_files,
                        &mut include_stack,
                    ) {
                        Ok(processed) => {
                            main_source.push_str(&format!("\n// ===== {file} =====\n"));
                            main_source.push_str(&processed);
                        }
                        Err(e) => {
                            return Err(MetalKernelError::CompilationError(format!(
                                "Failed to preprocess {file}: {e}"
                            )));
                        }
                    }
                }
            }
        }

        // Compile the preprocessed source
        let compile_options = {
            let opts = MTLCompileOptions::new();
            opts.setMathMode(MTLMathMode::Fast);
            opts
        };
        device
            .new_library_with_source(&main_source, Some(&compile_options))
            .map_err(|e| {
                MetalKernelError::CompilationError(format!(
                    "Failed to compile Metal kernels at runtime: {e}"
                ))
            })
    }

    fn load_function(
        &self,
        device: &Device,
        name: impl ToString,
        constants: Option<&ConstantValues>,
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
        40 => 2, // mxfp4: 2 FP4 values per byte
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
        width: thread_group_size as usize,
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
            width: nthreads,
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
    x_offset: usize,
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
                width: simdgroup_size as usize,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: o.div_ceil(bo),
                height: b,
                depth: n,
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
                width: (o / bo),
                height: b,
                depth: n,
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
                width: o.div_ceil(bo),
                height: b,
                depth: n,
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
                height: wn as usize,
                depth: wm as usize,
            };
            let grid_dims = MTLSize {
                width: o.div_ceil(bn),
                height: b.div_ceil(bm),
                depth: n,
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
                width: (o / bo),
                height: b,
                depth: n,
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
                height: wn as usize,
                depth: wm as usize,
            };
            let grid_dims = MTLSize {
                width: (o / bn),
                height: b.div_ceil(bm),
                depth: n,
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

    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(scales), 0);
    encoder.set_buffer(2, Some(biases), 0);
    encoder.set_buffer(3, Some(x), x_offset);
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
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 2,
            &x_stride.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        <i32 as EncoderParam>::set_param(encoder, offset + 3, w_batch_ndims as i32);
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 4,
            &w_shape.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 5,
            &w_stride.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 6,
            &s_stride.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 7,
            &b_stride.iter().map(|x| *x as i64).collect::<Vec<_>>(),
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
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 12,
            &lhs_strides.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 13,
            &rhs_strides.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
    }

    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mxfp4_matmul(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: (&Buffer, usize),
    w: (&Buffer, usize),
    scales: (&Buffer, usize),
    bias: (&Buffer, usize),
    out: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    has_bias: bool,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F16 => "mxfp4_matmul_f16",
        DType::BF16 => "mxfp4_matmul_bf16",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F16, DType::BF16],
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
        (
            x,
            w,
            scales,
            bias,
            out,
            m as i32,
            n as i32,
            k as i32,
            has_bias as i32
        )
    );

    // 8 simdgroups * 32 = 256 threads. Grid.x tiles N in blocks of 8.
    let group_dims = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(8),
        height: m,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mxfp4_moe_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: (&Buffer, usize),
    w: (&Buffer, usize),
    scales: (&Buffer, usize),
    biases: (&Buffer, usize),
    indices: (&Buffer, usize),
    out: &Buffer,
    num_tokens: usize,
    topk: usize,
    num_experts: usize,
    n: usize,
    k: usize,
    has_bias: bool,
    input_has_topk_dim: bool,
    reuse_topk: bool,
) -> Result<(), MetalKernelError> {
    let name = match (reuse_topk, ty) {
        (true, DType::F16) => "mxfp4_moe_gemm_reuse_f16",
        (true, DType::BF16) => "mxfp4_moe_gemm_reuse_bf16",
        (false, DType::F16) => "mxfp4_moe_gemm_split_f16",
        (false, DType::BF16) => "mxfp4_moe_gemm_split_bf16",
        (_, other) => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F16, DType::BF16],
                got: other,
            })
        }
    };

    let pipeline = kernels.load_pipeline(device, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    if reuse_topk {
        set_params!(
            encoder,
            (
                x,
                w,
                scales,
                biases,
                indices,
                out,
                num_tokens as i32,
                topk as i32,
                num_experts as i32,
                n as i32,
                k as i32,
                has_bias as i32
            )
        );
    } else {
        set_params!(
            encoder,
            (
                x,
                w,
                scales,
                biases,
                indices,
                out,
                num_tokens as i32,
                topk as i32,
                num_experts as i32,
                n as i32,
                k as i32,
                has_bias as i32,
                input_has_topk_dim as i32
            )
        );
    }

    let group_dims = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(8),
        height: num_tokens,
        depth: if reuse_topk { 1 } else { topk },
    };
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
        fn set_param(encoder: &ComputeCommandEncoderRef, position: usize, data: Self) {
            encoder.set_bytes_raw(
                position,
                core::mem::size_of_val(data),
                data as *const _ as *const c_void,
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
        width: weight_width.div_ceil(dequant_params.block_size_x) as usize,
        height: weight_height.div_ceil(dequant_params.block_size_y) as usize,
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
        encoder.set_buffer(0, Some(xs), xs_offset);
        encoder.set_buffer(1, Some(output), 0);

        let size = shape[axis];
        encoder.set_bytes_raw(
            2,
            std::mem::size_of::<usize>(),
            &size as *const usize as *const _,
        );

        // Compute the thread grid
        let n_reads = if ty.size_in_bytes() <= 4 { 4 } else { 2 };
        let simd_size = 32;
        let elements_per_simd = n_reads * simd_size;
        let mut thread_group_size = pipeline.max_total_threads_per_threadgroup();
        if size <= n_reads * 1024 {
            thread_group_size = size.div_ceil(elements_per_simd) * simd_size;
        } else if size <= n_reads * 2048 {
            thread_group_size = (size / 2).div_ceil(elements_per_simd) * simd_size;
        }
        thread_group_size = thread_group_size.min(pipeline.max_total_threads_per_threadgroup());
        let tmp_grid_dims = get_2d_grid_dims_divisor(shape, strides, size);

        let grid_dims = MTLSize {
            width: thread_group_size,
            height: tmp_grid_dims.width,
            depth: tmp_grid_dims.height,
        };
        let group_dims = MTLSize {
            width: thread_group_size,
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

        encoder.set_buffer(0, Some(xs), xs_offset);
        encoder.set_buffer(1, Some(output), 0);

        encoder.set_bytes_raw(
            2,
            std::mem::size_of::<usize>(),
            &size as *const usize as *const _,
        );
        encoder.set_bytes_raw(
            3,
            std::mem::size_of::<usize>(),
            &stride as *const usize as *const _,
        );
        encoder.set_bytes_raw(
            4,
            std::mem::size_of::<usize>(),
            &stride_blocks as *const usize as *const _,
        );

        // Compute the thread grid
        let n_reads = if ty.size_in_bytes() <= 4 { 4 } else { 2 };
        let n_simdgroups = bn / n_reads;
        let thread_group_size = n_simdgroups * 32;
        let mut tmp_grid_dims = get_2d_grid_dims_divisor(shape, strides, size * stride);
        if tmp_grid_dims.width <= (u32::MAX as usize) / stride_blocks {
            tmp_grid_dims.width *= stride_blocks;
        } else {
            tmp_grid_dims.height *= stride_blocks;
        }

        let grid_dims = MTLSize {
            width: thread_group_size,
            height: tmp_grid_dims.width,
            depth: tmp_grid_dims.height,
        };
        let group_dims = MTLSize {
            width: thread_group_size,
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
    pub bn: usize,
    pub tn: usize,
    pub n_blocks: usize,
}

/// Scratch buffers that can be reused between consecutive multi‑block sort
/// calls.  Providing them avoids the internal allocations performed each
/// time by `call_multi_block_sort`.
#[derive(Debug)]
pub struct MultiBlockSortCache {
    dev_vals_ping: Arc<Buffer>,
    dev_vals_pong: Arc<Buffer>,
    dev_idxs_ping: Arc<Buffer>,
    dev_idxs_pong: Arc<Buffer>,
    block_partitions: Arc<Buffer>,
}

// --------------------------------------------------------------------------
// Simple LRU cache for scratch buffers used by multi‑block sort / argsort
// --------------------------------------------------------------------------

use std::collections::VecDeque;

/// Uniquely identifies a scratch‑buffer layout.
///
/// We key only on the dimensions that impact required buffer sizes.  If two
/// calls have the same `(rows, cols, dtype_size, blocks)` tuple they can
/// safely reuse the same scratch buffers.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct CacheKey {
    n_rows: usize,
    size_sorted_axis: usize,
    dtype_size: usize,
    n_blocks: usize,
}

/// The buffers that can be reused between kernel launches.
#[derive(Debug)]
struct CachedBuffers {
    dev_vals_ping: Arc<Buffer>,
    dev_vals_pong: Arc<Buffer>,
    dev_idxs_ping: Arc<Buffer>,
    dev_idxs_pong: Arc<Buffer>,
    block_partitions: Arc<Buffer>,
}

/// Thread‑safe LRU cache with a fixed maximum number of entries.
///
/// *   The cache is **thread‑safe** (internally uses a `RwLock`).
/// *   Reuses scratch buffers across launches that share the same
///     (`rows`, `cols`, `dtype`, `blocks`) tuple.
pub struct SortScratchCache {
    cap: usize,
    map: RwLock<HashMap<CacheKey, CachedBuffers>>,
    order: RwLock<VecDeque<CacheKey>>, // most‑recent access at the back
}

impl SortScratchCache {
    /// Create a new cache that holds at most `cap` distinct buffer sets.
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            map: RwLock::new(HashMap::new()),
            order: RwLock::new(VecDeque::new()),
        }
    }

    /// Retrieve a set of scratch buffers (creating them if absent) and return
    /// a `MultiBlockSortCache::External` pointing to them.
    ///
    /// The borrow lasts for `'a` (caller’s scope) and is safe because the
    /// buffers live inside `self`.
    pub fn checkout(
        &self,
        metal_device: &MetalDevice,
        n_rows: usize,
        size_sorted_axis: usize,
        dtype: DType,
        n_blocks: usize,
    ) -> MultiBlockSortCache {
        let key = CacheKey {
            n_rows,
            size_sorted_axis,
            dtype_size: dtype.size_in_bytes(),
            n_blocks,
        };

        // Fast path – try read‑lock first
        if let Some(buffers) = self.map.read().unwrap().get(&key) {
            // Touch LRU order (needs write lock)
            self.touch_key(key);
            return MultiBlockSortCache {
                dev_vals_ping: buffers.dev_vals_ping.clone(),
                dev_vals_pong: buffers.dev_vals_pong.clone(),
                dev_idxs_ping: buffers.dev_idxs_ping.clone(),
                dev_idxs_pong: buffers.dev_idxs_pong.clone(),
                block_partitions: buffers.block_partitions.clone(),
            };
        }

        // Slow path – allocate new buffers
        let mut map_guard = self.map.write().unwrap();
        let mut order_guard = self.order.write().unwrap();

        // Evict least‑recently used if we’re at capacity
        if map_guard.len() == self.cap {
            if let Some(oldest) = order_guard.pop_front() {
                map_guard.remove(&oldest);
            }
        }

        // Allocate fresh buffers
        let elem_vals = n_rows * size_sorted_axis;
        let cached = CachedBuffers {
            dev_vals_ping: metal_device
                .new_buffer(elem_vals, dtype, "cache_vals_ping")
                .expect("alloc dev_vals_ping"),
            dev_vals_pong: metal_device
                .new_buffer(elem_vals, dtype, "cache_vals_pong")
                .expect("alloc dev_vals_pong"),
            dev_idxs_ping: metal_device
                .new_buffer(elem_vals, DType::U32, "cache_idxs_ping")
                .expect("alloc dev_idxs_ping"),
            dev_idxs_pong: metal_device
                .new_buffer(elem_vals, DType::U32, "cache_idxs_pong")
                .expect("alloc dev_idxs_pong"),
            block_partitions: metal_device
                .new_buffer(n_rows * (n_blocks + 1), DType::U32, "cache_partitions")
                .expect("alloc block_partitions"),
        };

        map_guard.insert(key, cached);
        order_guard.push_back(key);

        // SAFETY: we must re‑borrow from the fresh entry since `map_guard`
        // holds ownership of the buffers.
        let buffers = map_guard.get(&key).unwrap();
        MultiBlockSortCache {
            dev_vals_ping: buffers.dev_vals_ping.clone(),
            dev_vals_pong: buffers.dev_vals_pong.clone(),
            dev_idxs_ping: buffers.dev_idxs_ping.clone(),
            dev_idxs_pong: buffers.dev_idxs_pong.clone(),
            block_partitions: buffers.block_partitions.clone(),
        }
    }

    /// Move `key` to the back of the LRU order list.
    fn touch_key(&self, key: CacheKey) {
        let mut order = self.order.write().unwrap();
        if let Some(pos) = order.iter().position(|k| *k == key) {
            order.remove(pos);
        }
        order.push_back(key);
    }
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
        x.div_ceil(y)
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
            kernel_name.push_str(&format!("n{work_per_thread}"));
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
    encoder.set_buffer(0, Some(src), byte_offset_src);
    encoder.set_buffer(1, Some(dst), byte_offset_dst);

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
            let tg_size = pipeline.max_total_threads_per_threadgroup();
            let tg_size = tg_size.min(nthreads);

            let group_dims = MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            };
            let grid_dims = if large {
                // 64‑bit indexing path – fall back to 2‑D tiling helper.
                get_2d_grid_dims_divisor(shape, dst_strides, work_per_thread)
            } else {
                MTLSize {
                    width: nthreads,
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
                encoder.set_bytes_raw(
                    2,
                    std::mem::size_of::<i32>() * shape_i32.len(),
                    shape_i32.as_ptr() as *const _,
                );
            }
            // Strides – always required (slot 3)
            let strides_in_i64: Vec<i64> = src_strides.iter().map(|&x| x as i64).collect();
            encoder.set_bytes_raw(
                3,
                std::mem::size_of::<i64>() * strides_in_i64.len(),
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
                width: dim0,
                height: dim1,
                depth: rest,
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
        name.push('c');
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
    encoder.set_buffer(0, Some(src), src_offset);
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

            <i32 as EncoderParam>::set_param(encoder, 6, shape);
            <i64 as EncoderParam>::set_param(encoder, 7, stride);
            <i64 as EncoderParam>::set_param(encoder, 8, stride);
        } else {
            encoder.set_bytes_raw(
                6,
                std::mem::size_of::<i32>() * nc_shape.len(),
                nc_shape.as_ptr() as *const _,
            );
            encoder.set_bytes_raw(
                7,
                std::mem::size_of::<i64>() * in_nc_str.len(),
                in_nc_str.as_ptr() as *const _,
            );
            encoder.set_bytes_raw(
                8,
                std::mem::size_of::<i64>() * out_nc_str.len(),
                out_nc_str.as_ptr() as *const _,
            );
        }
    }

    // Dispatch
    let group_dims = MTLSize {
        width: bn,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: 1,
        height: n_rows,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_multi_block_sort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    bn: usize,
    tn: usize,
    n_blocks: usize,
    argsort: bool,
    cache: &MultiBlockSortCache,
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

    // ------------------------------------------------------------------
    // Acquire scratch buffers (either cached or freshly allocated)
    // ------------------------------------------------------------------

    // Scratch buffers supplied by the caller (from SortScratchCache)
    let dev_vals_0 = cache.dev_vals_ping.clone();
    let dev_vals_1 = cache.dev_vals_pong.clone();
    let dev_idxs_0 = cache.dev_idxs_ping.clone();
    let dev_idxs_1 = cache.dev_idxs_pong.clone();
    let block_partitions = cache.block_partitions.clone();

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

        encoder.set_buffer(0, Some(src), src_offset);
        encoder.set_buffer(1, Some(&*dev_vals_0), 0);
        encoder.set_buffer(2, Some(&*dev_idxs_0), 0);
        <i32 as EncoderParam>::set_param(encoder, 3, size_sorted_axis as i32);
        <i32 as EncoderParam>::set_param(encoder, 4, stride_sorted_axis as i32);
        <i32 as EncoderParam>::set_param(encoder, 5, nc_dim as i32);
        encoder.set_bytes_raw(
            6,
            std::mem::size_of::<i32>() * nc_shape.len(),
            nc_shape.as_ptr() as *const _,
        );
        encoder.set_bytes_raw(
            7,
            std::mem::size_of::<i64>() * nc_str.len(),
            nc_str.as_ptr() as *const _,
        );

        // Dispatch
        let group_dims = MTLSize {
            width: bn,
            height: 1,
            depth: 1,
        };
        let grid_dims = MTLSize {
            width: n_blocks,
            height: n_rows,
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

            encoder.set_buffer(0, Some(&*block_partitions), 0);
            encoder.set_buffer(1, Some(&*dev_vals_in), 0);
            encoder.set_buffer(2, Some(&*dev_idxs_in), 0);
            <i32 as EncoderParam>::set_param(encoder, 3, size_sorted_axis as i32);
            <i32 as EncoderParam>::set_param(encoder, 4, merge_tiles as i32);
            <i32 as EncoderParam>::set_param(encoder, 5, n_blocks as i32);

            // Dispatch
            let group_dims = MTLSize {
                width: n_thread_per_group as usize,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: 1,
                height: n_rows,
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

            encoder.set_buffer(0, Some(&*block_partitions), 0);
            encoder.set_buffer(1, Some(&*dev_vals_in), 0);
            encoder.set_buffer(2, Some(&*dev_idxs_in), 0);
            encoder.set_buffer(3, Some(&*dev_vals_out), 0);
            encoder.set_buffer(4, Some(&*dev_idxs_out), 0);
            <i32 as EncoderParam>::set_param(encoder, 5, size_sorted_axis as i32);
            <i32 as EncoderParam>::set_param(encoder, 6, merge_tiles as i32);
            <i32 as EncoderParam>::set_param(encoder, 7, n_blocks as i32);

            // Dispatch
            let group_dims = MTLSize {
                width: bn,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: n_blocks,
                height: n_rows,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid_dims, group_dims);
        }

        merge_tiles *= 2;
    }

    // Copy outputs with appropriate strides
    let mut strides = out_strides.to_vec();
    for strides_ax in strides.iter_mut().skip(axis + 1) {
        *strides_ax *= args.out_shape[axis];
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
        &copy_src, // deref Arc
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
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    argsort: bool,
    cache: &MultiBlockSortCache,
) -> Result<(), MetalKernelError> {
    // --- destructure helper -------------------------------------------------
    let bn = args.bn;
    let tn = args.tn;
    let n_blocks = args.n_blocks;

    if n_blocks > 1 {
        // multi‑block path
        call_multi_block_sort(device, ep, kernels, args, bn, tn, n_blocks, argsort, cache)
    } else {
        // single‑block path
        call_single_block_sort(device, ep, kernels, args, bn, tn, argsort)
    }
}

/// Sorts values along the last axis of a contiguous tensor.
pub fn call_sort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    cache: &MultiBlockSortCache,
) -> Result<(), MetalKernelError> {
    call_block_sort(device, ep, kernels, args, /* argsort = */ false, cache)
}

/// Returns the indices that would sort `src` along the last axis.
pub fn call_argsort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    cache: &MultiBlockSortCache,
) -> Result<(), MetalKernelError> {
    call_block_sort(device, ep, kernels, args, /* argsort = */ true, cache)
}

// HQQ Bitpacking functions

pub fn call_hqq_pack_8bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_8bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, output, num_elements));

    let grid_size = MTLSize {
        width: num_elements.div_ceil(256),
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

pub fn call_hqq_pack_4bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    height: usize,
    width: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_4bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, output, height, width));

    let step = height / 2;
    let grid_size = MTLSize {
        width: step.div_ceil(16),
        height: width.div_ceil(16),
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

#[allow(dead_code)]
pub fn call_hqq_pack_2bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    height: usize,
    width: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_2bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, output, height, width));

    let step = height / 4;
    let grid_size = MTLSize {
        width: step.div_ceil(16),
        height: width.div_ceil(16),
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

#[allow(dead_code)]
pub fn call_hqq_pack_3bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    height: usize,
    width: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_3bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, output, height, width));

    let step = height / 10;
    let grid_size = MTLSize {
        width: step.div_ceil(16),
        height: width.div_ceil(16),
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

#[allow(dead_code)]
pub fn call_hqq_pack_1bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    height: usize,
    width: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_1bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, output, height, width));

    let step = height / 8;
    let grid_size = MTLSize {
        width: step.div_ceil(16),
        height: width.div_ceil(16),
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_fused_glu(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    a: &Buffer,
    b: &Buffer,
    a_offset: usize,
    b_offset: usize,
    n_elements: usize,
    activation: i32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "fused_glu_float",
        DType::F16 => "fused_glu_half",
        DType::BF16 => "fused_glu_bfloat",
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
        (
            (a, a_offset),
            (b, b_offset),
            output,
            n_elements as u32,
            activation
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

// ============================================================================
// Softmax with sinks kernel
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub fn call_softmax_with_sinks(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    logits: &Buffer,
    logits_offset: usize,
    sinks: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    num_heads: u32,
    q_len: u32,
    k_len: u32,
    total_rows: usize,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "softmax_with_sinks_float",
        DType::F16 => "softmax_with_sinks_half",
        DType::BF16 => "softmax_with_sinks_bfloat",
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

    // Choose thread group size based on k_len
    let threads_per_group: usize = if k_len <= 64 {
        64
    } else if k_len <= 128 {
        128
    } else if k_len <= 256 {
        256
    } else {
        512
    };

    // Shared memory: s_max(1) + s_sum(1) + warp_scratch(threads_per_group / 32)
    let num_simdgroups = (threads_per_group + 31) / 32;
    let shared_mem_size = (2 + num_simdgroups) * std::mem::size_of::<f32>();
    encoder.set_threadgroup_memory_length(0, shared_mem_size);

    set_params!(
        encoder,
        (
            (logits, logits_offset),
            (sinks, sinks_offset),
            output,
            num_heads,
            q_len,
            k_len
        )
    );

    let thread_groups_count = MTLSize {
        width: total_rows,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

// ============================================================================
// SDPA with sinks (fused attention) kernels
// ============================================================================

fn sdpa_with_sinks_dtype_name(ty: DType) -> Result<&'static str, MetalKernelError> {
    match ty {
        DType::F32 => Ok("float"),
        DType::F16 => Ok("half"),
        DType::BF16 => Ok("bfloat16_t"),
        other => Err(MetalKernelError::DTypeMismatch {
            expected: vec![DType::F32, DType::F16, DType::BF16],
            got: other,
        }),
    }
}

/// Fused SDPA with sinks for decode (q_len == 1, single-pass).
/// Dispatches `sdpa_vector_with_sinks` Metal kernel.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_with_sinks(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    sinks_buffer: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    head_dim: usize,
    gqa_factor: i32,
    n: i32, // k_len
    k_stride: usize,
    v_stride: usize,
    scale: f32,
    b: usize, // batch * num_heads
) -> Result<(), MetalKernelError> {
    let type_name = sdpa_with_sinks_dtype_name(ty)?;
    let name = format!("sdpa_vector_with_sinks_{type_name}_{head_dim}");

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let k_stride = k_stride as u64;
    let v_stride = v_stride as u64;

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            (sinks_buffer, sinks_offset),
            output,
            gqa_factor,
            n,
            k_stride,
            v_stride,
            scale
        )
    );

    let grid_dims = MTLSize {
        width: 1,
        height: b,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: 1024, // 32 simdgroups * 32 threads
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Fused SDPA with sinks for decode, two-pass variant (k_len >= 1024).
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_with_sinks_2pass(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    sinks_buffer: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    intermediate: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    head_dim: usize,
    gqa_factor: i32,
    n: i32, // k_len
    k_stride: usize,
    v_stride: usize,
    scale: f32,
    b: usize, // batch * num_heads
) -> Result<(), MetalKernelError> {
    let type_name = sdpa_with_sinks_dtype_name(ty)?;
    let blocks: u64 = 32;

    // Pass 1: compute partial outputs per block
    {
        let name = format!("sdpa_vector_with_sinks_2pass_1_{type_name}_{head_dim}");
        let pipeline = kernels.load_pipeline(device, &name)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        let k_stride = k_stride as u64;
        let v_stride = v_stride as u64;

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                intermediate,
                sums,
                maxs,
                gqa_factor,
                n,
                k_stride,
                v_stride,
                scale
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: b,
            depth: blocks as usize,
        };
        let group_dims = MTLSize {
            width: 8 * 32, // BN=8 simdgroups * 32 threads
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Pass 2: reduce across blocks, integrate sinks
    {
        let name = format!("sdpa_vector_with_sinks_2pass_2_{type_name}_{head_dim}");
        let pipeline = kernels.load_pipeline(device, &name)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        set_params!(
            encoder,
            (
                intermediate,
                sums,
                maxs,
                (sinks_buffer, sinks_offset),
                output
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: b,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 1024, // 32 * 32
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(())
}

/// Fused flash attention with sinks for prefill (q_len > 1).
/// Dispatches `flash_attn_sinks_kernel` Metal kernel.
#[allow(clippy::too_many_arguments)]
pub fn call_flash_attn_sinks_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    sinks_buffer: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    scale: f32,
    batch_size: usize,
    q_len: usize,
    k_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
) -> Result<(), MetalKernelError> {
    let type_name = sdpa_with_sinks_dtype_name(ty)?;

    let br: usize = 8; // simdgroups per threadgroup
    let bc: usize = match head_dim {
        64 => 64,
        80 | 96 | 112 | 128 => 32,
        192 | 256 => 16,
        _ => {
            return Err(MetalKernelError::CompilationError(format!(
                "flash_attn_sinks: unsupported head_dim={head_dim}"
            )))
        }
    };

    let name = format!("flash_attn_sinks_{type_name}_hd{head_dim}_br{br}_bc{bc}");
    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Shared memory: k_smem[BC * D_PAD] + v_smem[BC * D_PAD] in float32
    let d_pad = ((head_dim + 31) / 32) * 32;
    let shared_mem_size = 2 * bc * d_pad * std::mem::size_of::<f32>();
    encoder.set_threadgroup_memory_length(0, shared_mem_size);

    let q_len_i32 = q_len as i32;
    let k_len_i32 = k_len as i32;
    let num_heads_i32 = num_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let window_size_i32 = window_size as i32;

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            (sinks_buffer, sinks_offset),
            output,
            scale,
            q_len_i32,
            k_len_i32,
            num_heads_i32,
            num_kv_heads_i32,
            window_size_i32
        )
    );

    let grid_dims = MTLSize {
        width: num_heads,
        height: batch_size,
        depth: (q_len + br - 1) / br,
    };
    let group_dims = MTLSize {
        width: br * 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Dispatches `flash_attn_sinks_varlen_kernel` Metal kernel.
#[allow(clippy::too_many_arguments)]
pub fn call_flash_attn_sinks_varlen_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    sinks_buffer: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    cu_seqlens_q_buffer: &Buffer,
    cu_seqlens_q_offset: usize,
    cu_seqlens_k_buffer: &Buffer,
    cu_seqlens_k_offset: usize,
    scale: f32,
    batch_size: usize,
    max_q_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
) -> Result<(), MetalKernelError> {
    let type_name = sdpa_with_sinks_dtype_name(ty)?;

    let br: usize = 8;
    let bc: usize = match head_dim {
        64 => 64,
        80 | 96 | 112 | 128 => 32,
        192 | 256 => 16,
        _ => {
            return Err(MetalKernelError::CompilationError(format!(
                "flash_attn_sinks_varlen: unsupported head_dim={head_dim}"
            )))
        }
    };

    let name = format!("flash_attn_sinks_varlen_{type_name}_hd{head_dim}_br{br}_bc{bc}");
    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let d_pad = ((head_dim + 31) / 32) * 32;
    let shared_mem_size = 2 * bc * d_pad * std::mem::size_of::<f32>();
    encoder.set_threadgroup_memory_length(0, shared_mem_size);

    let max_q_len_i32 = max_q_len as i32;
    let num_heads_i32 = num_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let window_size_i32 = window_size as i32;

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            (sinks_buffer, sinks_offset),
            output,
            (cu_seqlens_q_buffer, cu_seqlens_q_offset),
            (cu_seqlens_k_buffer, cu_seqlens_k_offset),
            scale,
            max_q_len_i32,
            num_heads_i32,
            num_kv_heads_i32,
            window_size_i32
        )
    );

    let grid_dims = MTLSize {
        width: num_heads,
        height: batch_size,
        depth: (max_q_len + br - 1) / br,
    };
    let group_dims = MTLSize {
        width: br * 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

// ============================================================================
// Scalar FP8 conversion kernels
// ============================================================================

/// Convert FP8 E4M3 tensor to another dtype (F32, F16, BF16)
#[allow(clippy::too_many_arguments)]
pub fn call_fp8_to_dtype(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    out_ty: DType,
    input: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let name = match out_ty {
        DType::F32 => "fp8_to_dtype_float",
        DType::F16 => "fp8_to_dtype_half",
        DType::BF16 => "fp8_to_dtype_bfloat16_t",
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

    set_params!(encoder, (input, output, num_elements as u32));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Convert tensor (F32, F16, BF16) to FP8 E4M3 with clamping
#[allow(clippy::too_many_arguments)]
pub fn call_dtype_to_fp8(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    in_ty: DType,
    input: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let name = match in_ty {
        DType::F32 => "dtype_to_fp8_float",
        DType::F16 => "dtype_to_fp8_half",
        DType::BF16 => "dtype_to_fp8_bfloat16_t",
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

    set_params!(encoder, (input, output, num_elements as u32));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Per-tensor FP8 dequantization: output = fp8_weight * scale_inv
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn call_fp8_pertensor_dequant(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    out_ty: DType,
    weight: &Buffer,
    scale_inv: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let name = match out_ty {
        DType::F32 => "fp8_pertensor_dequant_float",
        DType::F16 => "fp8_pertensor_dequant_half",
        DType::BF16 => "fp8_pertensor_dequant_bfloat16_t",
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

    set_params!(encoder, (weight, scale_inv, output, num_elements as u32));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Vector FP8 dequantization: output[i] = fp8_weight[i] * scale[i / 128]
/// Each group of 128 elements shares one scale
#[allow(clippy::too_many_arguments)]
pub fn call_fp8_vector_dequant(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    out_ty: DType,
    weight: &Buffer,
    scale: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let name = match out_ty {
        DType::F32 => "fp8_vector_dequant_float",
        DType::F16 => "fp8_vector_dequant_half",
        DType::BF16 => "fp8_vector_dequant_bfloat16_t",
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

    set_params!(encoder, (weight, scale, output, num_elements as u32));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
