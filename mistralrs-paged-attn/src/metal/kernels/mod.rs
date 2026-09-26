use candle_core::{DType, MetalStorage};
use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, ConstantValues, Device, Function, Library,
    Value,
};
use objc2_metal::{MTLDevice, MTLSize};
use std::sync::{OnceLock, RwLock};
use std::{collections::HashMap, ffi::c_void};

pub mod utils;
use utils::{EncoderProvider, RawBytesEncoder};

// Backward-compatible aliases to ease migration from the `metal` crate API.
type ComputeCommandEncoderRef = ComputeCommandEncoder;
type ComputePipelineState = ComputePipeline;

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
#[cfg(target_os = "tvos")]
const KERNELS: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/mistralrs_paged_attention_tvos.metallib"
));

include!("source_set.rs");

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

type Pipelines = HashMap<(String, Option<ConstantValues>), ComputePipelineState>;

static LIBRARY: OnceLock<Library> = OnceLock::new();

// Process-wide: callers construct `Kernels` per dispatch, so a per-instance cache rebuilt pipelines every call.
static PIPELINES: OnceLock<RwLock<Pipelines>> = OnceLock::new();

fn pipelines() -> &'static RwLock<Pipelines> {
    PIPELINES.get_or_init(|| RwLock::new(Pipelines::new()))
}

#[derive(Debug)]
pub struct Kernels;

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        Self
    }

    /// Load the library from precompiled metallib, falling back to runtime compilation if needed.
    /// If this has been previously loaded it will just fetch it from cache.
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
        mistralrs_metal_compile::compile_runtime_library(device, &PAGED_ATTENTION_METAL_SOURCE_SET)
            .map_err(MetalKernelError::CompilationError)
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
    /// that source
    fn load_pipeline_with_constants(
        &self,
        device: &Device,
        name: String,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = pipelines().write()?;
        let key = (name, constants);
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(device, name.clone(), constants.as_ref())?;
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
    numel_per_block_key: u64,
    numel_per_block_value: u64,
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

    assert_eq!(
        numel_per_block_key, numel_per_block_value,
        "key and value blocks must be the same size"
    );
    // key_cache and value_cache are read+written by copy_blocks; mark as outputs.
    encoder.set_output_buffer(0, Some(key_cache), key_cache_offset);
    encoder.set_output_buffer(1, Some(value_cache), value_cache_offset);
    encoder.set_input_buffer(2, Some(block_mapping), block_mapping_offset);
    encoder.set_bytes(3, &numel_per_block_key);
    encoder.set_bytes(4, &numel_per_block_value);

    let thread_groups_count = MTLSize {
        width: num_pairs as usize,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: (numel_per_block_key.min(1024)) as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedAttentionDType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F8E4M3 = 3,
}

impl PagedAttentionDType {
    fn to_repr(self) -> &'static str {
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

    encoder.set_input_buffer(0, Some(key), key_offset);
    encoder.set_input_buffer(1, Some(value), value_offset);
    encoder.set_output_buffer(2, Some(key_cache), key_cache_offset);
    encoder.set_output_buffer(3, Some(value_cache), value_cache_offset);
    encoder.set_input_buffer(4, Some(slot_mapping), slot_mapping_offset);
    if let Some((k_scale, v_scale)) = k_v_scale {
        encoder.set_input_buffer(5, Some(&k_scale), 0_usize);
        encoder.set_input_buffer(6, Some(&v_scale), 0_usize);
    }
    encoder.set_bytes_raw(
        7,
        core::mem::size_of_val(&key_stride),
        &key_stride as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        8,
        core::mem::size_of_val(&value_stride),
        &value_stride as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        9,
        core::mem::size_of_val(&num_heads),
        &num_heads as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        10,
        core::mem::size_of_val(&head_size),
        &head_size as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        11,
        core::mem::size_of_val(&block_size),
        &block_size as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        12,
        core::mem::size_of_val(&x),
        &x as *const _ as *const c_void,
    );

    let thread_groups_count = MTLSize {
        width: num_tokens as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: (num_heads * head_size).min(512) as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
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
    sinks: Option<(&Buffer, usize)>,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: usize = 256;
    const NUM_SIMD_LANES: usize = 32;

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
        (40, Value::Bool(/* use_sinks */ sinks.is_some())),
    ]));

    let pipeline = kernels.load_pipeline_with_constants(device, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let num_simds = NUM_THREADS / NUM_SIMD_LANES;
    let padded_max_context_len = ((max_context_len + block_size - 1) / block_size) * block_size;
    let logits_size = padded_max_context_len * std::mem::size_of::<f32>() as i32;
    let outputs_size = (num_simds as i32 / 2) * head_size * std::mem::size_of::<f32>() as i32;
    let shared_mem_size = logits_size.max(outputs_size);
    encoder.set_threadgroup_memory_length(0, shared_mem_size as usize);

    encoder.set_output_buffer(2, Some(output), 0_usize);
    encoder.set_input_buffer(3, Some(q), q_offset);
    encoder.set_input_buffer(4, Some(k_cache), k_cache_offset);
    encoder.set_input_buffer(5, Some(v_cache), v_cache_offset);
    if let Some((k_scale, v_scale)) = &k_v_scale {
        encoder.set_input_buffer(6, Some(k_scale), 0_usize);
        encoder.set_input_buffer(7, Some(v_scale), 0_usize);
    }
    encoder.set_bytes_raw(
        8,
        core::mem::size_of_val(&num_kv_heads),
        &num_kv_heads as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        9,
        core::mem::size_of_val(&scale),
        &scale as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        10,
        core::mem::size_of_val(&softcapping),
        &softcapping as *const _ as *const c_void,
    );
    encoder.set_input_buffer(11, Some(block_tables), block_tables_offset);
    encoder.set_input_buffer(12, Some(context_lens), context_lens_offset);
    encoder.set_bytes_raw(
        13,
        core::mem::size_of_val(&max_num_blocks_per_seq),
        &max_num_blocks_per_seq as *const _ as *const c_void,
    );
    if let Some((alibi, alibi_offset)) = alibi_storage_and_offset {
        encoder.set_input_buffer(14, Some(alibi.buffer()), alibi_offset);
    }
    encoder.set_bytes_raw(
        15,
        core::mem::size_of_val(&q_stride),
        &q_stride as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        16,
        core::mem::size_of_val(&kv_block_stride),
        &kv_block_stride as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        17,
        core::mem::size_of_val(&kv_head_stride),
        &kv_head_stride as *const _ as *const c_void,
    );
    if let Some((sinks_buf, sinks_offset)) = sinks {
        encoder.set_input_buffer(18, Some(sinks_buf), sinks_offset);
    }

    let thread_groups_count = MTLSize {
        width: num_heads as usize,
        height: num_seqs as usize,
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
    sinks: Option<(&Buffer, usize)>,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: usize = 256;
    const PARTITION_SIZE: usize = 512;
    const NUM_SIMD_LANES: usize = 32;

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
            (40, Value::Bool(/* use_sinks */ sinks.is_some())),
        ]));

        let pipeline = kernels.load_pipeline_with_constants(device, name, constants)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        let num_simds = NUM_THREADS / NUM_SIMD_LANES;
        let max_num_partitions =
            (max_context_len + PARTITION_SIZE as i32 - 1) / PARTITION_SIZE as i32;
        let logits_size = PARTITION_SIZE as i32 * std::mem::size_of::<f32>() as i32;
        let outputs_size = (num_simds as i32 / 2) * head_size * std::mem::size_of::<f32>() as i32;
        let shared_mem_size = logits_size.max(outputs_size);
        encoder.set_threadgroup_memory_length(0, shared_mem_size as usize);

        encoder.set_output_buffer(0, Some(exp_sums), 0_usize);
        encoder.set_output_buffer(1, Some(max_logits), 0_usize);
        encoder.set_output_buffer(2, Some(tmp_out), 0_usize);
        encoder.set_input_buffer(3, Some(q), q_offset);
        encoder.set_input_buffer(4, Some(k_cache), k_cache_offset);
        encoder.set_input_buffer(5, Some(v_cache), v_cache_offset);
        if let Some((k_scale, v_scale)) = &k_v_scale {
            encoder.set_input_buffer(6, Some(k_scale), 0_usize);
            encoder.set_input_buffer(7, Some(v_scale), 0_usize);
        }
        encoder.set_bytes_raw(
            8,
            core::mem::size_of_val(&num_kv_heads),
            &num_kv_heads as *const _ as *const c_void,
        );
        encoder.set_bytes_raw(
            9,
            core::mem::size_of_val(&scale),
            &scale as *const _ as *const c_void,
        );
        encoder.set_bytes_raw(
            10,
            core::mem::size_of_val(&softcapping),
            &softcapping as *const _ as *const c_void,
        );
        encoder.set_input_buffer(11, Some(block_tables), block_tables_offset);
        encoder.set_input_buffer(12, Some(context_lens), context_lens_offset);
        encoder.set_bytes_raw(
            13,
            core::mem::size_of_val(&max_num_blocks_per_seq),
            &max_num_blocks_per_seq as *const _ as *const c_void,
        );
        if let Some((alibi, alibi_offset)) = alibi_storage_and_offset {
            encoder.set_input_buffer(14, Some(alibi.buffer()), alibi_offset);
        }
        encoder.set_bytes_raw(
            15,
            core::mem::size_of_val(&q_stride),
            &q_stride as *const _ as *const c_void,
        );
        encoder.set_bytes_raw(
            16,
            core::mem::size_of_val(&kv_block_stride),
            &kv_block_stride as *const _ as *const c_void,
        );
        encoder.set_bytes_raw(
            17,
            core::mem::size_of_val(&kv_head_stride),
            &kv_head_stride as *const _ as *const c_void,
        );
        if let Some((sinks_buf, sinks_offset)) = sinks {
            encoder.set_input_buffer(18, Some(sinks_buf), sinks_offset);
        }

        let thread_groups_count = MTLSize {
            width: num_heads as usize,
            height: num_seqs as usize,
            depth: max_num_partitions as usize,
        };
        let thread_group_size = MTLSize {
            width: NUM_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    }

    encode_v2_reduce(
        device,
        ep,
        kernels,
        ty,
        V2Reduce {
            exp_sums,
            max_logits,
            tmp_out,
            output,
            context_lens,
            context_lens_offset,
            sinks,
            partition_size: PARTITION_SIZE,
            max_context_len,
            num_seqs,
            num_heads,
            head_size,
        },
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_gather_kv_cache(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    cache_ty: PagedAttentionDType,
    out_ty: PagedAttentionDType,
    key_cache: &Buffer,
    key_cache_offset: usize,
    value_cache: &Buffer,
    value_cache_offset: usize,
    k_out: &Buffer,
    k_out_offset: usize,
    v_out: &Buffer,
    v_out_offset: usize,
    k_v_scale: Option<(&Buffer, &Buffer)>,
    block_table: &Buffer,
    block_table_offset: usize,
    cu_seq_lens: &Buffer,
    cu_seq_lens_offset: usize,
    num_tokens: i32,
    num_seqs: i32,
    block_size: i32,
    block_table_stride: i32,
    num_kv_heads: i32,
    head_size: i32,
    x: i32,
) -> Result<(), MetalKernelError> {
    let name = format!(
        "gather_kv_cache_cache_{}_out_{}",
        cache_ty.to_repr(),
        out_ty.to_repr()
    );

    let constants = Some(ConstantValues::new(vec![(
        10,
        Value::Bool(/* use_fp8_scales */ k_v_scale.is_some()),
    )]));

    let pipeline = kernels.load_pipeline_with_constants(device, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(key_cache), key_cache_offset);
    encoder.set_input_buffer(1, Some(value_cache), value_cache_offset);
    encoder.set_output_buffer(2, Some(k_out), k_out_offset);
    encoder.set_output_buffer(3, Some(v_out), v_out_offset);
    if let Some((k_scale, v_scale)) = k_v_scale {
        encoder.set_input_buffer(4, Some(k_scale), 0_usize);
        encoder.set_input_buffer(5, Some(v_scale), 0_usize);
    }
    encoder.set_input_buffer(6, Some(block_table), block_table_offset);
    encoder.set_input_buffer(7, Some(cu_seq_lens), cu_seq_lens_offset);
    encoder.set_bytes_raw(
        8,
        core::mem::size_of_val(&num_tokens),
        &num_tokens as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        9,
        core::mem::size_of_val(&num_seqs),
        &num_seqs as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        10,
        core::mem::size_of_val(&block_size),
        &block_size as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        11,
        core::mem::size_of_val(&block_table_stride),
        &block_table_stride as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        12,
        core::mem::size_of_val(&num_kv_heads),
        &num_kv_heads as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        13,
        core::mem::size_of_val(&head_size),
        &head_size as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        14,
        core::mem::size_of_val(&x),
        &x as *const _ as *const c_void,
    );

    let thread_groups_count = MTLSize {
        width: num_tokens as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: (num_kv_heads * head_size).min(512) as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_kv_scale_update(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    k: &Buffer,
    k_offset: usize,
    v: &Buffer,
    v_offset: usize,
    k_scale: &Buffer,
    v_scale: &Buffer,
    num_elements: i64,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        PagedAttentionDType::F32 => "kv_scale_update_float",
        PagedAttentionDType::BF16 => "kv_scale_update_bfloat16_t",
        PagedAttentionDType::F16 => "kv_scale_update_half",
        PagedAttentionDType::F8E4M3 => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: DType::F8E4M3,
            })
        }
    };

    let pipeline = kernels.load_pipeline(device, name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(k), k_offset);
    encoder.set_input_buffer(1, Some(v), v_offset);
    encoder.set_output_buffer(2, Some(k_scale), 0);
    encoder.set_output_buffer(3, Some(v_scale), 0);
    encoder.set_bytes_raw(
        4,
        core::mem::size_of_val(&num_elements),
        &num_elements as *const _ as *const c_void,
    );

    const THREADS_PER_GROUP: usize = 512;
    let num_groups = (num_elements as usize)
        .div_ceil(THREADS_PER_GROUP)
        .min(65535);

    // Shared memory for reduction: THREADS_PER_GROUP floats each for k and v maxima
    encoder.set_threadgroup_memory_length(0, THREADS_PER_GROUP * std::mem::size_of::<f32>());
    encoder.set_threadgroup_memory_length(1, THREADS_PER_GROUP * std::mem::size_of::<f32>());

    let thread_groups_count = MTLSize {
        width: num_groups,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: THREADS_PER_GROUP,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

/// Inputs for the v2 reduce pass.
pub struct V2Reduce<'a> {
    pub exp_sums: &'a Buffer,
    pub max_logits: &'a Buffer,
    pub tmp_out: &'a Buffer,
    pub output: &'a Buffer,
    pub context_lens: &'a Buffer,
    pub context_lens_offset: usize,
    pub sinks: Option<(&'a Buffer, usize)>,
    pub partition_size: usize,
    pub max_context_len: i32,
    pub num_seqs: i32,
    pub num_heads: i32,
    pub head_size: i32,
}

/// Merges 512-token partition partials into the final output; shared by v2 and the GQA decode kernel.
fn encode_v2_reduce(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
    reduce: V2Reduce<'_>,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: usize = 256;
    const NUM_SIMD_LANES: usize = 32;
    let V2Reduce {
        exp_sums,
        max_logits,
        tmp_out,
        output,
        context_lens,
        context_lens_offset,
        sinks,
        partition_size,
        max_context_len,
        num_seqs,
        num_heads,
        head_size,
    } = reduce;
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
    name.push_str(&format!("_nsl{NUM_SIMD_LANES}"));
    name.push_str(&format!("_ps{partition_size}"));

    let reduce_constants = Some(ConstantValues::new(vec![(
        40,
        Value::Bool(/* use_sinks */ sinks.is_some()),
    )]));
    let pipeline = kernels.load_pipeline_with_constants(device, name, reduce_constants)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let max_num_partitions = (max_context_len + partition_size as i32 - 1) / partition_size as i32;
    let reduce_shared_mem_size = 2 * max_num_partitions * std::mem::size_of::<f32>() as i32;
    encoder.set_threadgroup_memory_length(0, reduce_shared_mem_size as usize);

    encoder.set_output_buffer(0, Some(output), 0_usize);
    encoder.set_input_buffer(1, Some(exp_sums), 0_usize);
    encoder.set_input_buffer(2, Some(max_logits), 0_usize);
    encoder.set_input_buffer(3, Some(tmp_out), 0_usize);
    encoder.set_input_buffer(4, Some(context_lens), context_lens_offset);
    encoder.set_bytes_raw(
        5,
        core::mem::size_of_val(&max_num_partitions),
        &max_num_partitions as *const _ as *const c_void,
    );
    if let Some((sinks_buf, sinks_offset)) = sinks {
        encoder.set_input_buffer(6, Some(sinks_buf), sinks_offset);
    }

    let thread_groups_count = MTLSize {
        width: num_heads as usize,
        height: num_seqs as usize,
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

/// Block size, head sizes and query/KV head ratios the GQA decode kernel is instantiated for.
pub const PAGED_DECODE_BLOCK_SIZE: usize = 32;
pub const PAGED_DECODE_HEAD_SIZES: [usize; 5] = [64, 96, 128, 192, 256];
pub const PAGED_DECODE_MAX_GQA: usize = 8;
const PAGED_DECODE_THREADS: usize = 128;
// Reduce kernels are instantiated for exactly these partition sizes.
const PAGED_DECODE_LONG_PARTITION: usize = 512;
const PAGED_DECODE_SHORT_PARTITION: usize = 128;
// Below this many threadgroups a 40-core GPU idles, so shorter partitions trade reduce work for occupancy.
const PAGED_DECODE_MIN_THREADGROUPS: usize = 64;

/// Inputs for [`call_paged_decode_gqa`]; the cache must use the [blocks, kv_heads, D/8, 32, 8] key layout.
pub struct PagedDecodeArgs<'a> {
    pub ty: PagedAttentionDType,
    pub q: (&'a Buffer, usize),
    pub k_cache: (&'a Buffer, usize),
    pub v_cache: (&'a Buffer, usize),
    pub block_tables: (&'a Buffer, usize),
    pub context_lens: (&'a Buffer, usize),
    pub output: &'a Buffer,
    /// Required when the longest context spans more than one 512-token partition.
    pub partials: Option<PagedDecodePartials<'a>>,
    pub scale: f32,
    /// From [`paged_decode_partition_size`]; the partials must be sized for it.
    pub partition_size: usize,
    pub max_context_len: i32,
    pub num_seqs: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_size: i32,
    pub max_num_blocks_per_seq: i32,
    pub q_stride: i32,
    pub kv_block_stride: i32,
    pub kv_head_stride: i32,
}

/// Per-partition scratch in the v2 layout: [seqs, heads, parts] sums and maxima, [seqs, heads, parts, D] outputs.
pub struct PagedDecodePartials<'a> {
    pub exp_sums: &'a Buffer,
    pub max_logits: &'a Buffer,
    pub tmp_out: &'a Buffer,
}

pub fn paged_decode_partition_size(
    max_context_len: usize,
    num_seqs: usize,
    num_kv_heads: usize,
) -> usize {
    let long_threadgroups =
        max_context_len.div_ceil(PAGED_DECODE_LONG_PARTITION) * num_seqs * num_kv_heads;
    if long_threadgroups < PAGED_DECODE_MIN_THREADGROUPS {
        PAGED_DECODE_SHORT_PARTITION
    } else {
        PAGED_DECODE_LONG_PARTITION
    }
}

/// GQA-aware paged decode: one threadgroup per (partition, kv head, sequence) reads each KV block once.
pub fn call_paged_decode_gqa(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: PagedDecodeArgs<'_>,
) -> Result<(), MetalKernelError> {
    let max_num_partitions = (args.max_context_len as usize).div_ceil(args.partition_size);
    let gqa = args.num_heads / args.num_kv_heads;
    let name = format!(
        "paged_decode_gqa_{}_hs{}_g{gqa}",
        args.ty.to_repr(),
        args.head_size
    );
    let pipeline = kernels.load_pipeline_with_constants(device, name, None)?;

    let (exp_sums, max_logits, tmp_out) = match &args.partials {
        Some(p) => (p.exp_sums, p.max_logits, p.tmp_out),
        None => (args.output, args.output, args.output),
    };

    {
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_output_buffer(0, Some(args.output), 0);
        encoder.set_output_buffer(1, Some(exp_sums), 0);
        encoder.set_output_buffer(2, Some(max_logits), 0);
        encoder.set_output_buffer(3, Some(tmp_out), 0);
        encoder.set_input_buffer(4, Some(args.q.0), args.q.1);
        encoder.set_input_buffer(5, Some(args.k_cache.0), args.k_cache.1);
        encoder.set_input_buffer(6, Some(args.v_cache.0), args.v_cache.1);
        encoder.set_input_buffer(7, Some(args.block_tables.0), args.block_tables.1);
        encoder.set_input_buffer(8, Some(args.context_lens.0), args.context_lens.1);
        let max_parts = max_num_partitions as i32;
        let partition_size = args.partition_size as i32;
        for (index, value) in [
            (9, args.max_num_blocks_per_seq),
            (10, args.q_stride),
            (11, args.kv_block_stride),
            (12, args.kv_head_stride),
            (14, max_parts),
            (15, partition_size),
        ] {
            encoder.set_bytes_raw(
                index,
                core::mem::size_of_val(&value),
                &value as *const _ as *const c_void,
            );
        }
        encoder.set_bytes_raw(
            13,
            core::mem::size_of_val(&args.scale),
            &args.scale as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: max_num_partitions,
                height: args.num_kv_heads as usize,
                depth: args.num_seqs as usize,
            },
            MTLSize {
                width: PAGED_DECODE_THREADS,
                height: 1,
                depth: 1,
            },
        );
    }

    if let Some(partials) = &args.partials {
        encode_v2_reduce(
            device,
            ep,
            kernels,
            args.ty,
            V2Reduce {
                exp_sums: partials.exp_sums,
                max_logits: partials.max_logits,
                tmp_out: partials.tmp_out,
                output: args.output,
                context_lens: args.context_lens.0,
                context_lens_offset: args.context_lens.1,
                sinks: None,
                partition_size: args.partition_size,
                max_context_len: args.max_context_len,
                num_seqs: args.num_seqs,
                num_heads: args.num_heads,
                head_size: args.head_size,
            },
        )?;
    }
    Ok(())
}
