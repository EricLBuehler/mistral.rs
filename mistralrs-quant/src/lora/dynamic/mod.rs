#[cfg(any(feature = "cuda", test))]
mod cuda;
#[cfg(feature = "cuda")]
mod cuda_ffi;
mod execution;
mod expert;
#[cfg(feature = "cuda")]
mod expert_cuda;
mod linear;
mod loader;
mod moe_cuda;
#[cfg(feature = "cuda")]
mod moe_cuda_ffi;
mod raw;
mod reference;
mod registry;

pub use execution::{
    with_lora_execution, LoraAdapterWeights, LoraExecution, LoraExecutionArena,
    LoraExecutionArenaStats, LoraSlotId, LoraWeights,
};
pub use expert::{
    add_expert_delta_reference, DynamicLoraWeights, LoraExpertDelta, LoraExpertExecution,
    LoraExpertInputMode, LoraExpertProjection, LoraExpertProjectionNames,
    LoraExpertProjectionWeights, LoraExpertSiteHandle, LoraExpertSiteSpec, LoraExpertWeights,
    LoraGateUpOrder,
};
pub use linear::maybe_wrap_dynamic_lora;
pub(crate) use linear::maybe_wrap_dynamic_lora_with_key;
pub use loader::{load_dynamic_lora_weights, plan_dynamic_lora_weights, DynamicLoraLoadPlan};
#[cfg(feature = "cuda")]
pub use moe_cuda::{
    launch_routed_lora_direct, launch_routed_lora_grouped, RoutedLoraCudaMetadata,
    RoutedLoraCudaWeightTable, RoutedLoraDirectLaunch, RoutedLoraGroupedLaunch,
};
pub use moe_cuda::{
    RoutedLoraAdapterWeight, RoutedLoraInputMode, RoutedLoraMetadataLayout,
    RoutedLoraProjectionLayout, ROUTED_LORA_BASE_SLOT, ROUTED_LORA_BLOCK_SIZE,
    ROUTED_LORA_MAX_RANK, ROUTED_LORA_WMMA_RANK_CAP,
};
pub use raw::{apply_dynamic_lora_delta, register_dynamic_lora_site};
pub(crate) use registry::LoraParallelism;
pub use registry::{
    LoraLayerRegistry, LoraLinearSpec, LoraRuntimeId, LoraSiteHandle, LoraSiteKey, LoraSiteSlice,
};

pub(crate) use execution::current_lora_execution;
pub(crate) use reference::add_delta;
