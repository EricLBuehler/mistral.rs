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

const COPY_BLOCKS: &str = include_str!("copy_blocks.metal");
const RESHAPE_AND_CACHE: &str = include_str!("reshape_and_cache.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    CopyBlocks,
    ReshapeAndCache,
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
    #[error("reshape and cache dtype mismatch, got {got:?}, expected {expected:?}")]
    ReshapeAndCacheDTypeMismatch {
        expected: Vec<ReshapeAndCacheDType>,
        got: ReshapeAndCacheDType,
    },
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
            Source::CopyBlocks => COPY_BLOCKS,
            Source::ReshapeAndCache => RESHAPE_AND_CACHE,
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
    let pipeline = kernels.load_pipeline(device, Source::CopyBlocks, name)?;

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
pub enum ReshapeAndCacheDType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
}

#[allow(clippy::too_many_arguments)]
pub fn call_reshape_and_cache(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: ReshapeAndCacheDType,
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
        ReshapeAndCacheDType::F32 => "reshape_and_cache_float",
        ReshapeAndCacheDType::BF16 => "reshape_and_cache_bfloat16_t",
        ReshapeAndCacheDType::F16 => "reshape_and_cache_half",
    };
    let pipeline = kernels.load_pipeline(device, Source::CopyBlocks, name)?;

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
    let thread_group_size = MTLSize {
        width: (num_heads * head_size).min(1024) as u64,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}
