use candle_core::{DType, MetalStorage};
use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLDataType, MTLSize, NSUInteger,
};
use std::sync::{OnceLock, RwLock};
use std::{collections::HashMap, ffi::c_void};

pub mod utils;
use utils::EncoderProvider;

use crate::set_params;

#[cfg(target_os = "macos")]
const KERNELS: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/mistralrs_paged_attention.metallib"
));
#[cfg(target_os = "ios")]
const KERNELS: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/mistralrs_paged_attention_ios.metallib"
));

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
    #[error("Failed to compile Metal shader: {0}")]
    CompilationError(String),
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type Pipelines = HashMap<(String, Option<ConstantValues>), ComputePipelineState>;

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
            // Check if the precompiled library is empty (which indicates runtime compilation is needed)
            let lib = if source_data.is_empty() {
                // Runtime compilation path
                self.compile_kernels_at_runtime(device)?
            } else {
                // Precompiled path
                device.new_library_with_data(source_data).map_err(|e| {
                    MetalKernelError::LoadLibraryError(format!(
                        "Metal requires macosx > 13.0 or higher, cannot load candle metal library: {e}"
                    ))
                })?
            };
            Ok(LIBRARY.get_or_init(|| lib).clone())
        }
    }

    fn compile_kernels_at_runtime(&self, device: &Device) -> Result<Library, MetalKernelError> {
        use std::collections::{HashMap, HashSet};

        // Create a virtual filesystem with all our Metal sources
        let mut file_system = HashMap::new();
        file_system.insert("copy_blocks.metal", include_str!("copy_blocks.metal"));
        file_system.insert("pagedattention.metal", include_str!("pagedattention.metal"));
        file_system.insert(
            "reshape_and_cache.metal",
            include_str!("reshape_and_cache.metal"),
        );
        file_system.insert("utils.metal", include_str!("utils.metal"));
        file_system.insert("float8.metal", include_str!("float8.metal"));

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
            let mut lines = content.lines();

            while let Some(line) = lines.next() {
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
                                        "\n// ===== Start of {} =====\n",
                                        include_file
                                    ));
                                    result.push_str(&processed);
                                    result.push_str(&format!(
                                        "\n// ===== End of {} =====\n",
                                        include_file
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
        main_source.push_str("#include <metal_simdgroup>\n");
        main_source.push_str("\nusing namespace metal;\n\n");

        // Process all the main implementation files
        // Order matters - we need to ensure dependencies are included first
        let main_files = vec![
            "float8.metal",      // Float8 types
            "utils.metal",       // Utility functions (depends on float8)
            "copy_blocks.metal", // Main implementations
            "pagedattention.metal",
            "reshape_and_cache.metal",
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
                            main_source.push_str(&format!("\n// ===== {} =====\n", file));
                            main_source.push_str(&processed);
                        }
                        Err(e) => {
                            return Err(MetalKernelError::CompilationError(format!(
                                "Failed to preprocess {}: {}",
                                file, e
                            )));
                        }
                    }
                }
            }
        }

        // Compile the preprocessed source
        let compile_options = metal::CompileOptions::new();
        device
            .new_library_with_source(&main_source, &compile_options)
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
    /// that source
    fn load_pipeline_with_constants(
        &self,
        device: &Device,
        name: String,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = (name, constants);
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(
                device,
                name.clone(),
                constants.as_ref().map(|c| c.function_constant_values()),
            )?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert((name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        name: String,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        self.load_pipeline_with_constants(device, name, None)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_copy_blocks(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    key_cache: &Buffer,
    key_cache_offset: usize,
    value_cache: &Buffer,
    value_cache_offset: usize,
    block_mapping: &Buffer,
    block_mapping_offset: usize,
    num_pairs: u64,
    numel_per_block: u64,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "copy_blocks_float",
        DType::BF16 => "copy_blocks_bfloat16_t",
        DType::F16 => "copy_blocks_half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (key_cache, key_cache_offset),
            (value_cache, value_cache_offset),
            (block_mapping, block_mapping_offset),
            numel_per_block
        )
    );

    let thread_groups_count = MTLSize {
        width: num_pairs,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: numel_per_block.min(1024),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

#[derive(Debug)]
pub enum PagedAttentionDType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F8E4M3 = 3,
}

impl PagedAttentionDType {
    fn to_repr(&self) -> &'static str {
        match self {
            PagedAttentionDType::F32 => "float",
            PagedAttentionDType::BF16 => "bfloat16_t",
            PagedAttentionDType::F16 => "half",
            PagedAttentionDType::F8E4M3 => "uchar",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_reshape_and_cache(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kv_ty: PagedAttentionDType,
    cache_ty: PagedAttentionDType,
    key: &Buffer,
    key_offset: usize,
    value: &Buffer,
    value_offset: usize,
    key_cache: &Buffer,
    key_cache_offset: usize,
    value_cache: &Buffer,
    value_cache_offset: usize,
    slot_mapping: &Buffer,
    slot_mapping_offset: usize,
    k_v_scale: Option<(Buffer, Buffer)>,
    num_tokens: i32,
    num_heads: i32,
    head_size: i32,
    block_size: i32,
    x: i32,
    key_stride: i32,
    value_stride: i32,
) -> Result<(), MetalKernelError> {
    let name = format!(
        "reshape_and_cache_kv_{}_cache_{}",
        kv_ty.to_repr(),
        cache_ty.to_repr()
    );

    let constants = Some(ConstantValues::new(vec![(
        10,
        Value::Bool(/* use_fp8_scales */ k_v_scale.is_some()),
    )]));

    let pipeline = kernels.load_pipeline_with_constants(device, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(key), key_offset as NSUInteger);
    encoder.set_buffer(1, Some(value), value_offset as NSUInteger);
    encoder.set_buffer(2, Some(key_cache), key_cache_offset as NSUInteger);
    encoder.set_buffer(3, Some(value_cache), value_cache_offset as NSUInteger);
    encoder.set_buffer(4, Some(slot_mapping), slot_mapping_offset as NSUInteger);
    if let Some((k_scale, v_scale)) = k_v_scale {
        encoder.set_buffer(5, Some(&k_scale), 0 as NSUInteger);
        encoder.set_buffer(6, Some(&v_scale), 0 as NSUInteger);
    }
    encoder.set_bytes(
        7,
        core::mem::size_of_val(&key_stride) as u64,
        &key_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        8,
        core::mem::size_of_val(&value_stride) as u64,
        &value_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        9,
        core::mem::size_of_val(&num_heads) as u64,
        &num_heads as *const _ as *const c_void,
    );
    encoder.set_bytes(
        10,
        core::mem::size_of_val(&head_size) as u64,
        &head_size as *const _ as *const c_void,
    );
    encoder.set_bytes(
        11,
        core::mem::size_of_val(&block_size) as u64,
        &block_size as *const _ as *const c_void,
    );
    encoder.set_bytes(
        12,
        core::mem::size_of_val(&x) as u64,
        &x as *const _ as *const c_void,
    );

    let thread_groups_count = MTLSize {
        width: num_tokens as u64,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: (num_heads * head_size).min(512) as u64,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[derive(Debug, PartialEq)]
pub enum Value {
    Bool(bool),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::Bool(v) => {
                    f.set_constant_value_at_index(
                        v as *const bool as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
            }
        }
        f
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_paged_attention_v1(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    cache_ty: PagedAttentionDType,
    q: &Buffer,
    q_offset: usize,
    k_cache: &Buffer,
    k_cache_offset: usize,
    v_cache: &Buffer,
    v_cache_offset: usize,
    block_tables: &Buffer,
    block_tables_offset: usize,
    context_lens: &Buffer,
    context_lens_offset: usize,
    k_v_scale: Option<(Buffer, Buffer)>,
    alibi_storage_and_offset: Option<(MetalStorage, usize)>,
    output: &Buffer,
    num_kv_heads: i32,
    scale: f32,
    softcapping: f32,
    block_size: i32,
    max_context_len: i32,
    num_seqs: i32,
    num_heads: i32,
    head_size: i32,
    max_num_blocks_per_seq: i32,
    q_stride: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: u64 = 256;
    const NUM_SIMD_LANES: u64 = 32;

    // v1 has no partition
    let name = format!(
        "paged_attention_{}_cache_{}_hs{head_size}_bs{block_size}_nt{NUM_THREADS}_nsl{NUM_SIMD_LANES}_ps0",
        ty.to_repr(), cache_ty.to_repr()
    );

    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* use_partitioning */ false)),
        (
            20,
            Value::Bool(/* use_alibi */ alibi_storage_and_offset.is_some()),
        ),
        (30, Value::Bool(/* use_fp8_scales */ k_v_scale.is_some())),
    ]));

    let pipeline = kernels.load_pipeline_with_constants(device, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    assert_eq!(pipeline.thread_execution_width(), NUM_SIMD_LANES);

    let num_simds = NUM_THREADS / NUM_SIMD_LANES;
    let padded_max_context_len = ((max_context_len + block_size - 1) / block_size) * block_size;
    let logits_size = padded_max_context_len * std::mem::size_of::<f32>() as i32;
    let outputs_size = (num_simds as i32 / 2) * head_size * std::mem::size_of::<f32>() as i32;
    let shared_mem_size = logits_size.max(outputs_size);
    encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

    encoder.set_buffer(2, Some(output), 0 as NSUInteger);
    encoder.set_buffer(3, Some(q), q_offset as NSUInteger);
    encoder.set_buffer(4, Some(k_cache), k_cache_offset as NSUInteger);
    encoder.set_buffer(5, Some(v_cache), v_cache_offset as NSUInteger);
    if let Some((k_scale, v_scale)) = &k_v_scale {
        encoder.set_buffer(6, Some(k_scale), 0 as NSUInteger);
        encoder.set_buffer(7, Some(v_scale), 0 as NSUInteger);
    }
    encoder.set_bytes(
        8,
        core::mem::size_of_val(&num_kv_heads) as u64,
        &num_kv_heads as *const _ as *const c_void,
    );
    encoder.set_bytes(
        9,
        core::mem::size_of_val(&scale) as u64,
        &scale as *const _ as *const c_void,
    );
    encoder.set_bytes(
        10,
        core::mem::size_of_val(&softcapping) as u64,
        &softcapping as *const _ as *const c_void,
    );
    encoder.set_buffer(11, Some(block_tables), block_tables_offset as NSUInteger);
    encoder.set_buffer(12, Some(context_lens), context_lens_offset as NSUInteger);
    encoder.set_bytes(
        13,
        core::mem::size_of_val(&max_num_blocks_per_seq) as u64,
        &max_num_blocks_per_seq as *const _ as *const c_void,
    );
    if let Some((alibi, alibi_offset)) = alibi_storage_and_offset {
        encoder.set_buffer(14, Some(alibi.buffer()), alibi_offset as NSUInteger);
    }
    encoder.set_bytes(
        15,
        core::mem::size_of_val(&q_stride) as u64,
        &q_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        16,
        core::mem::size_of_val(&kv_block_stride) as u64,
        &kv_block_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        17,
        core::mem::size_of_val(&kv_head_stride) as u64,
        &kv_head_stride as *const _ as *const c_void,
    );

    let thread_groups_count = MTLSize {
        width: num_heads as u64,
        height: num_seqs as u64,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: NUM_THREADS,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_paged_attention_v2(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    cache_ty: PagedAttentionDType,
    exp_sums: &Buffer,
    max_logits: &Buffer,
    q: &Buffer,
    q_offset: usize,
    k_cache: &Buffer,
    k_cache_offset: usize,
    v_cache: &Buffer,
    v_cache_offset: usize,
    block_tables: &Buffer,
    block_tables_offset: usize,
    context_lens: &Buffer,
    context_lens_offset: usize,
    k_v_scale: Option<(Buffer, Buffer)>,
    alibi_storage_and_offset: Option<(MetalStorage, usize)>,
    tmp_out: &Buffer,
    output: &Buffer,
    num_kv_heads: i32,
    scale: f32,
    softcapping: f32,
    block_size: i32,
    max_context_len: i32,
    num_seqs: i32,
    num_heads: i32,
    head_size: i32,
    max_num_blocks_per_seq: i32,
    q_stride: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: u64 = 256;
    const PARTITION_SIZE: u64 = 512;
    const NUM_SIMD_LANES: u64 = 32;

    // Initial paged attention kernel
    {
        // v2 has partition.
        let name = format!(
            "paged_attention_{}_cache_{}_hs{head_size}_bs{block_size}_nt{NUM_THREADS}_nsl{NUM_SIMD_LANES}_ps{PARTITION_SIZE}",
            ty.to_repr(), cache_ty.to_repr()
        );

        // v2 has partition.
        // Handle alibi
        let constants = Some(ConstantValues::new(vec![
            (10, Value::Bool(/* use_partitioning */ true)),
            (
                20,
                Value::Bool(/* use_alibi */ alibi_storage_and_offset.is_some()),
            ),
            (30, Value::Bool(/* use_fp8_scales */ k_v_scale.is_some())),
        ]));

        let pipeline = kernels.load_pipeline_with_constants(device, name, constants)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        assert_eq!(pipeline.thread_execution_width(), NUM_SIMD_LANES);

        let num_simds = NUM_THREADS / NUM_SIMD_LANES;
        let max_num_partitions =
            (max_context_len + PARTITION_SIZE as i32 - 1) / PARTITION_SIZE as i32;
        let logits_size = PARTITION_SIZE as i32 * std::mem::size_of::<f32>() as i32;
        let outputs_size = (num_simds as i32 / 2) * head_size * std::mem::size_of::<f32>() as i32;
        let shared_mem_size = logits_size.max(outputs_size);
        encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

        encoder.set_buffer(0, Some(exp_sums), 0 as NSUInteger);
        encoder.set_buffer(1, Some(max_logits), 0 as NSUInteger);
        encoder.set_buffer(2, Some(tmp_out), 0 as NSUInteger);
        encoder.set_buffer(3, Some(q), q_offset as NSUInteger);
        encoder.set_buffer(4, Some(k_cache), k_cache_offset as NSUInteger);
        encoder.set_buffer(5, Some(v_cache), v_cache_offset as NSUInteger);
        if let Some((k_scale, v_scale)) = &k_v_scale {
            encoder.set_buffer(6, Some(k_scale), 0 as NSUInteger);
            encoder.set_buffer(7, Some(v_scale), 0 as NSUInteger);
        }
        encoder.set_bytes(
            8,
            core::mem::size_of_val(&num_kv_heads) as u64,
            &num_kv_heads as *const _ as *const c_void,
        );
        encoder.set_bytes(
            9,
            core::mem::size_of_val(&scale) as u64,
            &scale as *const _ as *const c_void,
        );
        encoder.set_bytes(
            10,
            core::mem::size_of_val(&softcapping) as u64,
            &softcapping as *const _ as *const c_void,
        );
        encoder.set_buffer(11, Some(block_tables), block_tables_offset as NSUInteger);
        encoder.set_buffer(12, Some(context_lens), context_lens_offset as NSUInteger);
        encoder.set_bytes(
            13,
            core::mem::size_of_val(&max_num_blocks_per_seq) as u64,
            &max_num_blocks_per_seq as *const _ as *const c_void,
        );
        if let Some((alibi, alibi_offset)) = alibi_storage_and_offset {
            encoder.set_buffer(14, Some(alibi.buffer()), alibi_offset as NSUInteger);
        }
        encoder.set_bytes(
            15,
            core::mem::size_of_val(&q_stride) as u64,
            &q_stride as *const _ as *const c_void,
        );
        encoder.set_bytes(
            16,
            core::mem::size_of_val(&kv_block_stride) as u64,
            &kv_block_stride as *const _ as *const c_void,
        );
        encoder.set_bytes(
            17,
            core::mem::size_of_val(&kv_head_stride) as u64,
            &kv_head_stride as *const _ as *const c_void,
        );

        let thread_groups_count = MTLSize {
            width: num_heads as u64,
            height: num_seqs as u64,
            depth: max_num_partitions as u64,
        };
        let thread_group_size = MTLSize {
            width: NUM_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    }

    // Paged attention reduce kernel
    {
        let name = match ty {
            PagedAttentionDType::F32 => "paged_attention_v2_reduce_float",
            PagedAttentionDType::BF16 => "paged_attention_v2_reduce_bfloat16_t",
            PagedAttentionDType::F16 => "paged_attention_v2_reduce_half",
            PagedAttentionDType::F8E4M3 => {
                return Err(MetalKernelError::DTypeMismatch {
                    expected: vec![DType::F32, DType::F16, DType::BF16],
                    got: DType::F8E4M3,
                })
            }
        };
        let mut name = name.to_string();
        name.push_str(&format!("_hs{head_size}"));
        name.push_str(&format!("_nt{NUM_THREADS}"));
        name.push_str(&format!("_nsl{}", NUM_SIMD_LANES));
        name.push_str(&format!("_ps{}", PARTITION_SIZE));

        let pipeline = kernels.load_pipeline(device, name)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        assert_eq!(pipeline.thread_execution_width(), NUM_SIMD_LANES);

        let max_num_partitions =
            (max_context_len + PARTITION_SIZE as i32 - 1) / PARTITION_SIZE as i32;
        let reduce_shared_mem_size = 2 * max_num_partitions * std::mem::size_of::<f32>() as i32;
        encoder.set_threadgroup_memory_length(0, reduce_shared_mem_size as u64);

        encoder.set_buffer(0, Some(output), 0 as NSUInteger);
        encoder.set_buffer(1, Some(exp_sums), 0 as NSUInteger);
        encoder.set_buffer(2, Some(max_logits), 0 as NSUInteger);
        encoder.set_buffer(3, Some(tmp_out), 0 as NSUInteger);
        encoder.set_buffer(4, Some(context_lens), context_lens_offset as NSUInteger);
        encoder.set_bytes(
            5,
            core::mem::size_of_val(&max_num_partitions) as u64,
            &max_num_partitions as *const _ as *const c_void,
        );

        let thread_groups_count = MTLSize {
            width: num_heads as u64,
            height: num_seqs as u64,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: NUM_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    }
    Ok(())
}
