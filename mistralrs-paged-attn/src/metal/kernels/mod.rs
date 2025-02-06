use candle_core::{DType, MetalStorage};
use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLDataType, MTLSize, NSUInteger,
};
use once_cell::sync::OnceCell;
use std::sync::RwLock;
use std::{collections::HashMap, ffi::c_void};

pub mod utils;
use utils::EncoderProvider;

use crate::set_params;

const COPY_BLOCKS: &str = include_str!("copy_blocks.metal");
const RESHAPE_AND_CACHE: &str = include_str!("reshape_and_cache.metal");
const PAGEDATTENTION: &str = include_str!("pagedattention.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    CopyBlocks,
    ReshapeAndCache,
    PagedAttention,
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
type Pipelines = HashMap<(String, Option<ConstantValues>), ComputePipelineState>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

pub(crate) static G_KERNEL: OnceCell<Kernels> = OnceCell::new();

impl Kernels {
    pub fn default() -> &'static Kernels {
        G_KERNEL.get_or_init(Kernels::new)
    }

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
            Source::CopyBlocks => COPY_BLOCKS,
            Source::ReshapeAndCache => RESHAPE_AND_CACHE,
            Source::PagedAttention => PAGEDATTENTION,
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
        name: String,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(&name, constants)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    fn load_pipeline_with_constants(
        &self,
        device: &Device,
        source: Source,
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
                source,
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
        source: Source,
        name: String,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        self.load_pipeline_with_constants(device, source, name, None)
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
    let pipeline = kernels.load_pipeline(device, Source::CopyBlocks, name.to_string())?;
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
}

#[allow(clippy::too_many_arguments)]
pub fn call_reshape_and_cache(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: PagedAttentionDType,
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
    num_tokens: i32,
    num_heads: i32,
    head_size: i32,
    block_size: i32,
    x: i32,
    key_stride: i32,
    value_stride: i32,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        PagedAttentionDType::F32 => "reshape_and_cache_float",
        PagedAttentionDType::BF16 => "reshape_and_cache_bfloat16_t",
        PagedAttentionDType::F16 => "reshape_and_cache_half",
    };
    let pipeline = kernels.load_pipeline(device, Source::ReshapeAndCache, name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (key, key_offset),
            (value, value_offset),
            (key_cache, key_cache_offset),
            (value_cache, value_cache_offset),
            (slot_mapping, slot_mapping_offset),
            key_stride,
            value_stride,
            num_heads,
            head_size,
            block_size,
            x
        )
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

    let name = match ty {
        PagedAttentionDType::F32 => "paged_attention_float",
        PagedAttentionDType::BF16 => "paged_attention_bfloat16_t",
        PagedAttentionDType::F16 => "paged_attention_half",
    };
    let mut name = name.to_string();
    name.push_str(&format!("_hs{head_size}"));
    name.push_str(&format!("_bs{block_size}"));
    name.push_str(&format!("_nt{NUM_THREADS}"));
    name.push_str(&format!("_nsl{}", NUM_SIMD_LANES));
    // v1 has no partition
    name.push_str(&format!("_ps{}", 0));

    // v1 has no partition.
    // Handle alibi
    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* use_partitioning */ false)),
        (
            20,
            Value::Bool(/* use_alibi */ alibi_storage_and_offset.is_some()),
        ),
    ]));

    let pipeline =
        kernels.load_pipeline_with_constants(device, Source::PagedAttention, name, constants)?;
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
    encoder.set_bytes(
        6,
        core::mem::size_of_val(&num_kv_heads) as u64,
        &num_kv_heads as *const _ as *const c_void,
    );
    encoder.set_bytes(
        7,
        core::mem::size_of_val(&scale) as u64,
        &scale as *const _ as *const c_void,
    );
    encoder.set_bytes(
        8,
        core::mem::size_of_val(&softcapping) as u64,
        &softcapping as *const _ as *const c_void,
    );
    encoder.set_buffer(9, Some(block_tables), block_tables_offset as NSUInteger);
    encoder.set_buffer(10, Some(context_lens), context_lens_offset as NSUInteger);
    encoder.set_bytes(
        11,
        core::mem::size_of_val(&max_num_blocks_per_seq) as u64,
        &max_num_blocks_per_seq as *const _ as *const c_void,
    );
    if let Some((alibi, alibi_offset)) = alibi_storage_and_offset {
        encoder.set_buffer(12, Some(alibi.buffer()), alibi_offset as NSUInteger);
    }
    encoder.set_bytes(
        13,
        core::mem::size_of_val(&q_stride) as u64,
        &q_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        14,
        core::mem::size_of_val(&kv_block_stride) as u64,
        &kv_block_stride as *const _ as *const c_void,
    );
    encoder.set_bytes(
        15,
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
        let name = match ty {
            PagedAttentionDType::F32 => "paged_attention_float",
            PagedAttentionDType::BF16 => "paged_attention_bfloat16_t",
            PagedAttentionDType::F16 => "paged_attention_half",
        };
        let mut name = name.to_string();
        name.push_str(&format!("_hs{head_size}"));
        name.push_str(&format!("_bs{block_size}"));
        name.push_str(&format!("_nt{NUM_THREADS}"));
        name.push_str(&format!("_nsl{}", NUM_SIMD_LANES));
        // v2 has partition.
        name.push_str(&format!("_ps{}", PARTITION_SIZE));

        // v2 has partition.
        // Handle alibi
        let constants = Some(ConstantValues::new(vec![
            (10, Value::Bool(/* use_partitioning */ true)),
            (
                20,
                Value::Bool(/* use_alibi */ alibi_storage_and_offset.is_some()),
            ),
        ]));

        let pipeline = kernels.load_pipeline_with_constants(
            device,
            Source::PagedAttention,
            name,
            constants,
        )?;

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
        encoder.set_bytes(
            6,
            core::mem::size_of_val(&num_kv_heads) as u64,
            &num_kv_heads as *const _ as *const c_void,
        );
        encoder.set_bytes(
            7,
            core::mem::size_of_val(&scale) as u64,
            &scale as *const _ as *const c_void,
        );
        encoder.set_bytes(
            8,
            core::mem::size_of_val(&softcapping) as u64,
            &softcapping as *const _ as *const c_void,
        );
        encoder.set_buffer(9, Some(block_tables), block_tables_offset as NSUInteger);
        encoder.set_buffer(10, Some(context_lens), context_lens_offset as NSUInteger);
        encoder.set_bytes(
            11,
            core::mem::size_of_val(&max_num_blocks_per_seq) as u64,
            &max_num_blocks_per_seq as *const _ as *const c_void,
        );
        if let Some((alibi, alibi_offset)) = alibi_storage_and_offset {
            encoder.set_buffer(12, Some(alibi.buffer()), alibi_offset as NSUInteger);
        }
        encoder.set_bytes(
            13,
            core::mem::size_of_val(&q_stride) as u64,
            &q_stride as *const _ as *const c_void,
        );
        encoder.set_bytes(
            14,
            core::mem::size_of_val(&kv_block_stride) as u64,
            &kv_block_stride as *const _ as *const c_void,
        );
        encoder.set_bytes(
            15,
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
        };
        let mut name = name.to_string();
        name.push_str(&format!("_hs{head_size}"));
        name.push_str(&format!("_nt{NUM_THREADS}"));
        name.push_str(&format!("_nsl{}", NUM_SIMD_LANES));
        name.push_str(&format!("_ps{}", PARTITION_SIZE));

        let pipeline = kernels.load_pipeline(device, Source::PagedAttention, name)?;

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
