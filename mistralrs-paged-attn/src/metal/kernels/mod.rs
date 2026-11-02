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

const CACHE_LAYOUT_CONSTANT: usize = 50;
const PAGED_ATTENTION_THREADS: usize = 256;
const PAGED_ATTENTION_SIMD_LANES: usize = 32;

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

type Pipelines = HashMap<(u64, String, Option<ConstantValues>), ComputePipelineState>;

static LIBRARIES: OnceLock<RwLock<HashMap<u64, Library>>> = OnceLock::new();

// Process-wide: callers construct `Kernels` per dispatch, so a per-instance cache rebuilt pipelines every call.
static PIPELINES: OnceLock<RwLock<Pipelines>> = OnceLock::new();

fn pipelines() -> &'static RwLock<Pipelines> {
    PIPELINES.get_or_init(|| RwLock::new(Pipelines::new()))
}

#[cfg(test)]
pub(crate) fn cached_pipeline_count(device: &Device) -> usize {
    let device_id = device.registry_id();
    pipelines()
        .read()
        .unwrap()
        .keys()
        .filter(|(id, _, _)| *id == device_id)
        .count()
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
        let libraries = LIBRARIES.get_or_init(|| RwLock::new(HashMap::new()));
        let device_id = device.registry_id();
        if let Some(lib) = libraries.read()?.get(&device_id) {
            return Ok(lib.clone());
        }
        let mut libraries = libraries.write()?;
        if let Some(lib) = libraries.get(&device_id) {
            return Ok(lib.clone());
        }
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
        libraries.insert(device_id, lib.clone());
        Ok(lib)
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
    pub(crate) fn load_pipeline_with_constants(
        &self,
        device: &Device,
        name: String,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let key = (device.registry_id(), name, constants);
        if let Some(pipeline) = pipelines().read()?.get(&key) {
            return Ok(pipeline.clone());
        }
        let mut pipelines = pipelines().write()?;
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (device_id, name, constants) = key;
            let func = self.load_function(device, name.clone(), constants.as_ref())?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert((device_id, name, constants), pipeline.clone());

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
    let pipeline = copy_pipeline_key(ty.try_into()?).load(kernels, device)?;
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

impl TryFrom<DType> for PagedAttentionDType {
    type Error = MetalKernelError;

    fn try_from(dtype: DType) -> Result<Self, Self::Error> {
        match dtype {
            DType::F16 => Ok(Self::F16),
            DType::BF16 => Ok(Self::BF16),
            DType::F32 => Ok(Self::F32),
            DType::F8E4M3 => Ok(Self::F8E4M3),
            got => Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F16, DType::BF16, DType::F32, DType::F8E4M3],
                got,
            }),
        }
    }
}

pub(crate) struct PipelineKey {
    name: String,
    constants: Option<ConstantValues>,
}

impl PipelineKey {
    pub(crate) fn load(
        self,
        kernels: &Kernels,
        device: &Device,
    ) -> Result<ComputePipeline, MetalKernelError> {
        kernels.load_pipeline_with_constants(device, self.name, self.constants)
    }
}

fn cache_pipeline_key(name: String, scales: bool, hnd: bool) -> PipelineKey {
    PipelineKey {
        name,
        constants: Some(ConstantValues::new(vec![
            (10, Value::Bool(scales)),
            (CACHE_LAYOUT_CONSTANT, Value::Bool(hnd)),
        ])),
    }
}

pub(crate) fn reshape_pipeline_key(
    ty: PagedAttentionDType,
    cache_ty: PagedAttentionDType,
    scales: bool,
    hnd: bool,
) -> PipelineKey {
    cache_pipeline_key(
        format!(
            "reshape_and_cache_kv_{}_cache_{}",
            ty.to_repr(),
            cache_ty.to_repr()
        ),
        scales,
        hnd,
    )
}

pub(crate) fn gather_pipeline_key(
    ty: PagedAttentionDType,
    cache_ty: PagedAttentionDType,
    scales: bool,
    hnd: bool,
) -> PipelineKey {
    cache_pipeline_key(
        format!(
            "gather_kv_cache_cache_{}_out_{}",
            cache_ty.to_repr(),
            ty.to_repr()
        ),
        scales,
        hnd,
    )
}

pub(crate) fn copy_pipeline_key(ty: PagedAttentionDType) -> PipelineKey {
    PipelineKey {
        name: format!("copy_blocks_{}", ty.to_repr()),
        constants: None,
    }
}

pub(crate) fn scale_pipeline_key(ty: PagedAttentionDType) -> PipelineKey {
    PipelineKey {
        name: format!("kv_scale_update_{}", ty.to_repr()),
        constants: None,
    }
}

pub(crate) struct AttentionPipelineKey {
    pub ty: PagedAttentionDType,
    pub cache_ty: PagedAttentionDType,
    pub head_size: usize,
    pub block_size: usize,
    pub partitioned: bool,
    pub alibi: bool,
    pub scales: bool,
    pub sinks: bool,
    pub hnd: bool,
}

impl AttentionPipelineKey {
    pub(crate) fn key(self) -> PipelineKey {
        let partition = if self.partitioned {
            PAGED_DECODE_LONG_PARTITION
        } else {
            0
        };
        PipelineKey {
            name: format!(
                "paged_attention_{}_cache_{}_hs{}_bs{}_nt{PAGED_ATTENTION_THREADS}_nsl{PAGED_ATTENTION_SIMD_LANES}_ps{partition}",
                self.ty.to_repr(),
                self.cache_ty.to_repr(),
                self.head_size,
                self.block_size
            ),
            constants: Some(ConstantValues::new(vec![
                (10, Value::Bool(self.partitioned)),
                (20, Value::Bool(self.alibi)),
                (30, Value::Bool(self.scales)),
                (40, Value::Bool(self.sinks)),
                (CACHE_LAYOUT_CONSTANT, Value::Bool(self.hnd)),
            ])),
        }
    }
}

pub(crate) fn reduce_pipeline_key(
    ty: PagedAttentionDType,
    head_size: usize,
    partition_size: usize,
    sinks: bool,
) -> PipelineKey {
    PipelineKey {
        name: format!(
            "paged_attention_v2_reduce_{}_hs{head_size}_nt{PAGED_ATTENTION_THREADS}_nsl{PAGED_ATTENTION_SIMD_LANES}_ps{partition_size}",
            ty.to_repr()
        ),
        constants: Some(ConstantValues::new(vec![(40, Value::Bool(sinks))])),
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
    hnd: bool,
) -> Result<(), MetalKernelError> {
    let pipeline =
        reshape_pipeline_key(kv_ty, cache_ty, k_v_scale.is_some(), hnd).load(kernels, device)?;
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
    hnd: bool,
    sliding_window: i32,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: usize = PAGED_ATTENTION_THREADS;
    const NUM_SIMD_LANES: usize = PAGED_ATTENTION_SIMD_LANES;

    let pipeline = AttentionPipelineKey {
        ty,
        cache_ty,
        head_size: head_size as usize,
        block_size: block_size as usize,
        partitioned: false,
        alibi: alibi_storage_and_offset.is_some(),
        scales: k_v_scale.is_some(),
        sinks: sinks.is_some(),
        hnd,
    }
    .key()
    .load(kernels, device)?;
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

    encoder.set_bytes_raw(
        19,
        core::mem::size_of_val(&sliding_window),
        &sliding_window as *const _ as *const c_void,
    );
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
    hnd: bool,
    sliding_window: i32,
) -> Result<(), MetalKernelError> {
    const NUM_THREADS: usize = PAGED_ATTENTION_THREADS;
    const PARTITION_SIZE: usize = 512;
    const NUM_SIMD_LANES: usize = PAGED_ATTENTION_SIMD_LANES;

    // Initial paged attention kernel
    {
        let pipeline = AttentionPipelineKey {
            ty,
            cache_ty,
            head_size: head_size as usize,
            block_size: block_size as usize,
            partitioned: true,
            alibi: alibi_storage_and_offset.is_some(),
            scales: k_v_scale.is_some(),
            sinks: sinks.is_some(),
            hnd,
        }
        .key()
        .load(kernels, device)?;

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

        encoder.set_bytes_raw(
            19,
            core::mem::size_of_val(&sliding_window),
            &sliding_window as *const _ as *const c_void,
        );
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
    hnd: bool,
) -> Result<(), MetalKernelError> {
    let pipeline =
        gather_pipeline_key(out_ty, cache_ty, k_v_scale.is_some(), hnd).load(kernels, device)?;
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
    let pipeline = scale_pipeline_key(ty).load(kernels, device)?;
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
    const NUM_THREADS: usize = PAGED_ATTENTION_THREADS;
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
    let pipeline = reduce_pipeline_key(ty, head_size as usize, partition_size, sinks.is_some())
        .load(kernels, device)?;

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
const PAGED_DECODE_SHORT_THREADS: usize = 1024;
const PAGED_DECODE_SHORT_MAX_HEADS: usize = 32;
pub const PAGED_DECODE_LARGE_HEAD_SIZE: usize = 512;
// Larger grouped heads need this cap to fit 32 KiB of threadgroup memory.
pub const PAGED_DECODE_LARGE_HEAD_MAX_GQA: usize = 4;
const PAGED_DECODE_HND_SHORT_CONTEXT: usize = 256;
const PAGED_DECODE_SPLIT_PARTITIONS: usize = 32;
const PAGED_DECODE_SPLIT_THREADS: usize = 256;
pub(crate) const PAGED_DECODE_SPLIT_MIN_CONTEXT: usize = 768;
const PAGED_DECODE_SPLIT_MAX_CONTEXT: usize = 4096;
// Head-parallel decode trades KV reuse for fewer launches; keep it below the measured crossover.
const PAGED_DECODE_SHORT_CONTEXTS: [(usize, usize, usize); 2] = [(128, 2, 288), (256, 3, 320)];

struct ShortDecodeProfile {
    head_size: usize,
    query_heads: usize,
    kv_heads: usize,
    max_batch: usize,
    max_context: usize,
}

// Longer single-pass launches are limited to profiled head configurations.
const PAGED_DECODE_HND_SHORT_PROFILES: &[ShortDecodeProfile] = &[
    ShortDecodeProfile {
        head_size: 128,
        query_heads: 16,
        kv_heads: 8,
        max_batch: 1,
        max_context: 512,
    },
    ShortDecodeProfile {
        head_size: 256,
        query_heads: 8,
        kv_heads: 2,
        max_batch: 4,
        max_context: 1024,
    },
    ShortDecodeProfile {
        head_size: 512,
        query_heads: 8,
        kv_heads: 2,
        max_batch: 4,
        max_context: 1024,
    },
    ShortDecodeProfile {
        head_size: 256,
        query_heads: 8,
        kv_heads: 2,
        max_batch: 8,
        max_context: 512,
    },
    ShortDecodeProfile {
        head_size: 512,
        query_heads: 8,
        kv_heads: 2,
        max_batch: 8,
        max_context: 512,
    },
];
// Reduce kernels are instantiated for exactly these partition sizes.
const PAGED_DECODE_LONG_PARTITION: usize = 512;
pub(crate) const PAGED_DECODE_SHORT_PARTITION: usize = 128;
// Below this many threadgroups a 40-core GPU idles, so shorter partitions trade reduce work for occupancy.
const PAGED_DECODE_MIN_THREADGROUPS: usize = 64;

/// Inputs for either the HND or packed-cache decode family.
pub struct PagedDecodeArgs<'a> {
    pub ty: PagedAttentionDType,
    pub hnd: bool,
    pub sliding_window: i32,
    pub q: (&'a Buffer, usize),
    pub k_cache: (&'a Buffer, usize),
    pub v_cache: (&'a Buffer, usize),
    pub block_tables: (&'a Buffer, usize),
    pub context_lens: (&'a Buffer, usize),
    pub output: &'a Buffer,
    /// Required when the selected variant uses more than one partition.
    pub partials: Option<PagedDecodePartials<'a>>,
    pub scale: f32,
    pub variant: PagedDecodeVariant,
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

#[derive(Clone, Copy)]
pub enum PagedDecodeVariant {
    Short,
    Split,
    Grouped { partition_size: usize },
}

impl PagedDecodeVariant {
    pub(crate) fn pipeline_key(
        self,
        ty: PagedAttentionDType,
        head_size: usize,
        gqa: usize,
        hnd: bool,
    ) -> PipelineKey {
        let layout = if hnd { "hnd_" } else { "" };
        let name = match self {
            Self::Short => format!("paged_decode_{layout}short_{}_hs{head_size}", ty.to_repr()),
            Self::Split => format!("paged_decode_hnd_split_{}_hs{head_size}", ty.to_repr()),
            Self::Grouped { .. } => format!(
                "paged_decode_{layout}gqa_{}_hs{head_size}_g{gqa}",
                ty.to_repr()
            ),
        };
        PipelineKey {
            name,
            constants: None,
        }
    }

    pub fn num_partitions(self, max_context_len: usize) -> usize {
        match self {
            Self::Short => 1,
            Self::Split => PAGED_DECODE_SPLIT_PARTITIONS,
            Self::Grouped { partition_size } => max_context_len.div_ceil(partition_size),
        }
    }
}

pub(crate) fn split_reduce_pipeline_key(ty: PagedAttentionDType, head_size: usize) -> PipelineKey {
    PipelineKey {
        name: format!(
            "paged_decode_hnd_split_reduce_{}_hs{head_size}",
            ty.to_repr()
        ),
        constants: None,
    }
}

pub fn paged_decode_variant(
    max_context_len: usize,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_size: usize,
) -> PagedDecodeVariant {
    let short_context = PAGED_DECODE_SHORT_CONTEXTS
        .iter()
        .any(|&(head, min_gqa, limit)| {
            head_size == head && num_heads / num_kv_heads >= min_gqa && max_context_len <= limit
        });
    if num_seqs == 1 && num_heads <= PAGED_DECODE_SHORT_MAX_HEADS && short_context {
        return PagedDecodeVariant::Short;
    }
    paged_decode_grouped_variant(max_context_len, num_seqs, num_kv_heads)
}

pub fn paged_decode_hnd_variant(
    max_context_len: usize,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    sliding_window: Option<usize>,
    head_size: usize,
) -> PagedDecodeVariant {
    if num_seqs == 1
        && num_heads == 8
        && num_kv_heads == 2
        && head_size == PAGED_DECODE_LARGE_HEAD_SIZE
        && sliding_window.is_none()
        && (PAGED_DECODE_SPLIT_MIN_CONTEXT..=PAGED_DECODE_SPLIT_MAX_CONTEXT)
            .contains(&max_context_len)
    {
        return PagedDecodeVariant::Split;
    }
    let visible_context = max_context_len.min(sliding_window.unwrap_or(max_context_len));
    let profiled_short = PAGED_DECODE_HND_SHORT_PROFILES.iter().any(|profile| {
        head_size == profile.head_size
            && num_heads == profile.query_heads
            && num_kv_heads == profile.kv_heads
            && num_seqs <= profile.max_batch
            && visible_context <= profile.max_context
    });
    if profiled_short
        || (num_seqs == 1
            && num_heads <= PAGED_DECODE_SHORT_MAX_HEADS
            && visible_context <= PAGED_DECODE_HND_SHORT_CONTEXT)
    {
        return PagedDecodeVariant::Short;
    }
    paged_decode_variant(
        max_context_len,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_size,
    )
}

pub(crate) fn paged_decode_grouped_variant(
    max_context_len: usize,
    num_seqs: usize,
    num_kv_heads: usize,
) -> PagedDecodeVariant {
    let long_threadgroups =
        max_context_len.div_ceil(PAGED_DECODE_LONG_PARTITION) * num_seqs * num_kv_heads;
    let partition_size = if long_threadgroups < PAGED_DECODE_MIN_THREADGROUPS {
        PAGED_DECODE_SHORT_PARTITION
    } else {
        PAGED_DECODE_LONG_PARTITION
    };
    PagedDecodeVariant::Grouped { partition_size }
}

pub(crate) fn paged_decode_warmup_shapes(
    max_batch: usize,
    max_context: usize,
) -> Vec<(usize, usize)> {
    let mut contexts = vec![
        1,
        max_context,
        PAGED_DECODE_HND_SHORT_CONTEXT,
        PAGED_DECODE_SPLIT_MIN_CONTEXT,
        PAGED_DECODE_SPLIT_MAX_CONTEXT,
    ];
    contexts.extend(
        PAGED_DECODE_SHORT_CONTEXTS
            .iter()
            .map(|(_, _, limit)| *limit),
    );
    contexts.extend(
        PAGED_DECODE_HND_SHORT_PROFILES
            .iter()
            .map(|profile| profile.max_context),
    );
    contexts.extend(
        contexts
            .clone()
            .into_iter()
            .map(|context| context.saturating_add(1)),
    );
    contexts.retain(|context| *context <= max_context);
    contexts.sort_unstable();
    contexts.dedup();
    let mut batches = vec![1, max_batch];
    batches.extend(
        PAGED_DECODE_HND_SHORT_PROFILES
            .iter()
            .flat_map(|profile| [profile.max_batch, profile.max_batch + 1]),
    );
    batches.retain(|batch| *batch <= max_batch);
    batches.sort_unstable();
    batches.dedup();
    batches
        .into_iter()
        .flat_map(|batch| contexts.iter().map(move |context| (batch, *context)))
        .collect()
}

/// Short decode uses one threadgroup per query head; grouped decode shares KV reads within each partition.
pub fn call_paged_decode_gqa(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: PagedDecodeArgs<'_>,
) -> Result<(), MetalKernelError> {
    let max_num_partitions = args.variant.num_partitions(args.max_context_len as usize);
    let gqa = args.num_heads / args.num_kv_heads;
    let (grid_heads, threads, kernel_param) = match args.variant {
        PagedDecodeVariant::Short => (args.num_heads, PAGED_DECODE_SHORT_THREADS, gqa),
        PagedDecodeVariant::Split => (args.num_heads, PAGED_DECODE_SPLIT_THREADS, gqa),
        PagedDecodeVariant::Grouped { partition_size } => (
            args.num_kv_heads,
            PAGED_DECODE_THREADS,
            partition_size as i32,
        ),
    };
    let pipeline = args
        .variant
        .pipeline_key(args.ty, args.head_size as usize, gqa as usize, args.hnd)
        .load(kernels, device)?;

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
        for (index, value) in [
            (9, args.max_num_blocks_per_seq),
            (10, args.q_stride),
            (11, args.kv_block_stride),
            (12, args.kv_head_stride),
            (14, max_parts),
            (15, kernel_param),
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
        if args.hnd {
            encoder.set_bytes_raw(
                16,
                core::mem::size_of_val(&args.sliding_window),
                &args.sliding_window as *const _ as *const c_void,
            );
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: max_num_partitions,
                height: grid_heads as usize,
                depth: args.num_seqs as usize,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }

    if let Some(partials) = &args.partials {
        if matches!(args.variant, PagedDecodeVariant::Split) {
            let pipeline = split_reduce_pipeline_key(args.ty, args.head_size as usize)
                .load(kernels, device)?;
            let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_output_buffer(0, Some(args.output), 0);
            encoder.set_input_buffer(1, Some(partials.exp_sums), 0);
            encoder.set_input_buffer(2, Some(partials.max_logits), 0);
            encoder.set_input_buffer(3, Some(partials.tmp_out), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: args.num_heads as usize,
                    height: args.num_seqs as usize,
                    depth: 1,
                },
                MTLSize {
                    width: PAGED_DECODE_SHORT_THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            return Ok(());
        }
        let PagedDecodeVariant::Grouped { partition_size } = args.variant else {
            unreachable!("the short decode kernel does not use partials")
        };
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
                partition_size,
                max_context_len: args.max_context_len,
                num_seqs: args.num_seqs,
                num_heads: args.num_heads,
                head_size: args.head_size,
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    #[test]
    fn warmup_shapes_cover_decode_dispatch_families() {
        for (head_size, heads, kv_heads) in [(256, 24, 4), (512, 8, 2), (128, 16, 8)] {
            for window in [None, Some(1024)] {
                let key = |context, batch| {
                    paged_decode_hnd_variant(context, batch, heads, kv_heads, window, head_size)
                        .pipeline_key(PagedAttentionDType::BF16, head_size, heads / kv_heads, true)
                        .name
                };
                let prepared = paged_decode_warmup_shapes(32, 32768)
                    .into_iter()
                    .map(|(batch, context)| key(context, batch))
                    .collect::<std::collections::HashSet<_>>();
                for batch in 1..=32 {
                    for context in [
                        1, 128, 256, 257, 320, 512, 513, 768, 1024, 1025, 4096, 4097, 8192, 32768,
                    ] {
                        assert!(
                            prepared.contains(&key(context, batch)),
                            "missing head {head_size}, batch {batch}, context {context}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn hnd_short_profiles_bound_context_and_batch() {
        for profile in PAGED_DECODE_HND_SHORT_PROFILES {
            let select = |context, batch| {
                paged_decode_hnd_variant(
                    context,
                    batch,
                    profile.query_heads,
                    profile.kv_heads,
                    None,
                    profile.head_size,
                )
            };
            for batch in [1, profile.max_batch] {
                let variant = select(profile.max_context, batch);
                if profile.head_size == PAGED_DECODE_LARGE_HEAD_SIZE
                    && batch == 1
                    && profile.max_context >= PAGED_DECODE_SPLIT_MIN_CONTEXT
                {
                    assert!(matches!(variant, PagedDecodeVariant::Split));
                } else {
                    assert!(matches!(variant, PagedDecodeVariant::Short));
                    assert_eq!(variant.num_partitions(profile.max_context), 1);
                }
            }
            assert!(matches!(
                select(profile.max_context + 1, profile.max_batch),
                PagedDecodeVariant::Grouped { .. }
            ));
            assert!(matches!(
                select(profile.max_context, profile.max_batch + 1),
                PagedDecodeVariant::Grouped { .. }
            ));
        }
    }

    #[test]
    fn hnd_short_profiles_use_the_visible_window_and_keep_unprofiled_shapes() {
        for head_size in [256, 512] {
            let windowed = paged_decode_hnd_variant(8192, 4, 8, 2, Some(1024), head_size);
            assert!(matches!(windowed, PagedDecodeVariant::Short));
            assert_eq!(windowed.num_partitions(8192), 1);
            assert!(matches!(
                paged_decode_hnd_variant(8192, 4, 8, 2, Some(1025), head_size),
                PagedDecodeVariant::Grouped { .. }
            ));
        }
        for (batch, heads, kv_heads, head_size, context) in [
            (4, 16, 4, 256, 512),
            (1, 16, 4, 512, 512),
            (1, 8, 8, 128, 512),
            (8, 8, 2, 256, 513),
            (8, 8, 2, 512, 513),
        ] {
            assert!(matches!(
                paged_decode_hnd_variant(context, batch, heads, kv_heads, None, head_size),
                PagedDecodeVariant::Grouped { .. }
            ));
        }
    }

    #[test]
    fn hnd_split_is_limited_to_profiled_global_decode() {
        for context in [
            PAGED_DECODE_SPLIT_MIN_CONTEXT,
            1024,
            2048,
            PAGED_DECODE_SPLIT_MAX_CONTEXT,
        ] {
            let variant = paged_decode_hnd_variant(context, 1, 8, 2, None, 512);
            assert!(matches!(variant, PagedDecodeVariant::Split));
            assert_eq!(
                variant.num_partitions(context),
                PAGED_DECODE_SPLIT_PARTITIONS
            );
        }
        for (context, batch, heads, kv_heads, head_size, window) in [
            (PAGED_DECODE_SPLIT_MIN_CONTEXT - 1, 1, 8, 2, 512, None),
            (PAGED_DECODE_SPLIT_MAX_CONTEXT + 1, 1, 8, 2, 512, None),
            (1024, 2, 8, 2, 512, None),
            (1024, 1, 16, 4, 512, None),
            (1024, 1, 8, 4, 512, None),
            (1024, 1, 8, 2, 256, None),
            (1024, 1, 8, 2, 512, Some(1024)),
        ] {
            assert!(!matches!(
                paged_decode_hnd_variant(context, batch, heads, kv_heads, window, head_size),
                PagedDecodeVariant::Split
            ));
        }
        for head_size in [256, 512] {
            assert!(matches!(
                paged_decode_hnd_variant(543, 8, 8, 2, Some(512), head_size),
                PagedDecodeVariant::Short
            ));
            assert!(matches!(
                paged_decode_hnd_variant(543, 8, 8, 2, Some(513), head_size),
                PagedDecodeVariant::Grouped { .. }
            ));
        }
    }

    #[test]
    fn short_decode_has_no_partials() {
        for (head, min_gqa, limit) in PAGED_DECODE_SHORT_CONTEXTS {
            for context in [1, limit] {
                let variant = paged_decode_variant(context, 1, min_gqa * 8, 8, head);
                assert!(matches!(variant, PagedDecodeVariant::Short));
                assert_eq!(variant.num_partitions(context), 1);
            }
            assert!(matches!(
                paged_decode_variant(limit + 1, 1, min_gqa * 8, 8, head),
                PagedDecodeVariant::Grouped { .. }
            ));
        }
        for (batch, heads, head_size) in [
            (4, 16, 128),
            (1, 8, 128),
            (1, 16, 192),
            (1, 64, 128),
            (1, 16, 256),
        ] {
            let context = 256;
            let variant = paged_decode_variant(context, batch, heads, 8, head_size);
            assert!(matches!(variant, PagedDecodeVariant::Grouped { .. }));
            assert!(variant.num_partitions(context) > 1);
        }
    }
}
