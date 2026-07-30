use std::{
    cell::RefCell,
    collections::HashMap,
    fmt,
    sync::{Arc, Weak},
};

use candle_core::{
    cuda::{
        cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr},
        CudaDType,
    },
    CpuStorage, CudaDevice, CudaStorage, DType, DeviceLocation, InplaceOp1, Layout, Result,
    Storage, Tensor, WithDType,
};
use half::{bf16, f16};

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

use super::execution::PreparedExpertAdapters;
use super::moe_cuda::prefer_small_batch_direct;

use super::{
    launch_routed_lora_direct, launch_routed_lora_grouped, LoraExecution, LoraExpertDelta,
    LoraExpertInputMode, LoraExpertProjection, LoraExpertProjectionWeights, LoraExpertSiteHandle,
    LoraExpertWeights, LoraGateUpOrder, RoutedLoraAdapterWeight, RoutedLoraCudaMetadata,
    RoutedLoraCudaWeightTable, RoutedLoraDirectLaunch, RoutedLoraGroupedLaunch,
    RoutedLoraInputMode, RoutedLoraMetadataLayout, RoutedLoraProjectionLayout,
    ROUTED_LORA_MAX_RANK, ROUTED_LORA_WMMA_RANK_CAP,
};

const NAIVE_DIRECT_SPARSITY_FACTOR: usize = 8;
const NATIVE_DIRECT_SHORT_ROUTE_LIMIT: usize = 128;
const NATIVE_DIRECT_CONTRACTION_ROUTE_LIMIT: usize = 512;
const NATIVE_DIRECT_MIN_CONTRACTION_FACTOR: usize = 2;
#[cfg(feature = "cutile")]
const CUTILE_NATIVE_DIRECT_MAX_RANK: usize = 16;
#[cfg(feature = "cutile")]
const CUTILE_NATIVE_DIRECT_MIN_ROUTES: usize = 256;
#[cfg(feature = "cutile")]
const CUTILE_NATIVE_DIRECT_COMPUTE_MAJOR: i32 = 12;
const EXPERT_CUDA_CACHE_CAPACITY: usize = 32;
const EXPERT_CUDA_WEIGHT_CACHE_CAPACITY: usize = 256;

fn supported_direct_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

fn prefer_unsorted_direct(layout: RoutedLoraMetadataLayout, dtype: DType, max_rank: usize) -> bool {
    supported_direct_dtype(dtype)
        && (prefer_small_batch_direct(layout, max_rank)
            || (max_rank <= ROUTED_LORA_WMMA_RANK_CAP
                && layout
                    .num_routes()
                    .checked_mul(NAIVE_DIRECT_SPARSITY_FACTOR)
                    .zip(layout.num_experts().checked_mul(layout.num_adapter_slots()))
                    .is_some_and(|(route_work, pairs)| route_work <= pairs)))
}

#[derive(Clone, Copy)]
struct NativeDirectShape {
    max_rank: usize,
    strongly_contracts_features: bool,
}

fn prefer_native_direct(
    layout: RoutedLoraMetadataLayout,
    dtype: DType,
    shape: NativeDirectShape,
) -> bool {
    prefer_unsorted_direct(layout, dtype, shape.max_rank)
        || (supported_direct_dtype(dtype)
            && shape.max_rank <= ROUTED_LORA_WMMA_RANK_CAP
            && (layout.num_routes() <= NATIVE_DIRECT_SHORT_ROUTE_LIMIT
                || (shape.strongly_contracts_features
                    && layout.num_routes() <= NATIVE_DIRECT_CONTRACTION_ROUTE_LIMIT)))
}

fn strongly_contracts_features(input_features: usize, output_features: usize) -> bool {
    input_features >= output_features.saturating_mul(NATIVE_DIRECT_MIN_CONTRACTION_FACTOR)
}

#[cfg(feature = "cutile")]
fn prefer_native_over_cutile(
    compute_major: i32,
    layout: RoutedLoraMetadataLayout,
    dtype: DType,
    shape: NativeDirectShape,
) -> bool {
    compute_major == CUTILE_NATIVE_DIRECT_COMPUTE_MAJOR
        && dtype == DType::BF16
        && shape.max_rank <= CUTILE_NATIVE_DIRECT_MAX_RANK
        && shape.strongly_contracts_features
        && layout.num_routes() >= CUTILE_NATIVE_DIRECT_MIN_ROUTES
        && layout.num_routes() <= NATIVE_DIRECT_CONTRACTION_ROUTE_LIMIT
}

fn supported_cuda_rank(rank: usize) -> bool {
    rank <= ROUTED_LORA_MAX_RANK
}

fn needs_grouped_metadata(
    layout: RoutedLoraMetadataLayout,
    dtype: DType,
    shape: NativeDirectShape,
    cutile_compute_major: Option<i32>,
) -> bool {
    #[cfg(feature = "cutile")]
    {
        if dtype == DType::BF16
            && !prefer_native_over_cutile(
                cutile_compute_major.expect("cuTile compute capability"),
                layout,
                dtype,
                shape,
            )
        {
            !prefer_unsorted_direct(layout, dtype, shape.max_rank)
        } else {
            !prefer_native_direct(layout, dtype, shape)
        }
    }
    #[cfg(not(feature = "cutile"))]
    {
        let _ = cutile_compute_major;
        !prefer_native_direct(layout, dtype, shape)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ResourceKey {
    device: DeviceLocation,
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    num_adapter_slots: usize,
    needs_metadata: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct WeightTableKey {
    device: DeviceLocation,
    site_id: u32,
    adapter_ids: Vec<usize>,
}

struct CachedWeightTable {
    table: Arc<RoutedLoraCudaWeightTable>,
    gate_up_rank: usize,
    down_rank: usize,
    adapters: Vec<Weak<LoraExpertWeights>>,
}

fn weak_identity_matches<T>(cached: &[Weak<T>], current: &[&Arc<T>]) -> bool {
    cached.len() == current.len()
        && cached.iter().zip(current).all(|(cached, current)| {
            cached
                .upgrade()
                .is_some_and(|adapter| Arc::ptr_eq(&adapter, current))
        })
}

struct WeightResources {
    table: Option<Arc<RoutedLoraCudaWeightTable>>,
    gate_up_active: bool,
    down_active: bool,
    gate_up_rank: usize,
    down_rank: usize,
    gate_up_scratch: Option<CudaSlice<f32>>,
    down_scratch: Option<CudaSlice<f32>>,
}

impl WeightResources {
    fn new() -> Self {
        Self {
            table: None,
            gate_up_active: false,
            down_active: false,
            gate_up_rank: 0,
            down_rank: 0,
            gate_up_scratch: None,
            down_scratch: None,
        }
    }

    fn set_table(&mut self, weights: &Arc<CachedWeightTable>) {
        self.gate_up_active = weights.gate_up_rank != 0;
        self.down_active = weights.down_rank != 0;
        self.gate_up_rank = weights.gate_up_rank;
        self.down_rank = weights.down_rank;
        self.table = Some(weights.table.clone());
    }
}

struct ConfiguredSite {
    execution_id: u64,
    site_id: u32,
    source_topk_ptr: u64,
    topk_ids: Tensor,
    adapter_ids: Vec<usize>,
    token_slots: Vec<u32>,
    adapters: Vec<Weak<LoraExpertWeights>>,
}

struct PersistentDirectResource {
    device: CudaDevice,
    layout: RoutedLoraMetadataLayout,
    configured: ConfiguredSite,
    table: Arc<RoutedLoraCudaWeightTable>,
    adapter: Arc<LoraExpertWeights>,
}

struct ExpertCudaResources {
    device: CudaDevice,
    layout: RoutedLoraMetadataLayout,
    token_adapter_slots: CudaSlice<u32>,
    host_token_adapter_slots: Vec<u32>,
    metadata: Option<RoutedLoraCudaMetadata>,
    weights: WeightResources,
    configured: Option<ConfiguredSite>,
    token_slot_uploads: usize,
    metadata_builds: usize,
}

#[derive(Default)]
pub(super) struct ExpertCudaCache {
    resources: HashMap<ResourceKey, ExpertCudaResources>,
    weight_tables: HashMap<WeightTableKey, Arc<CachedWeightTable>>,
    weight_table_uploads: usize,
}

impl ExpertCudaCache {
    pub(super) fn len(&self) -> usize {
        self.resources.len()
    }

    pub(super) fn stats(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.resources.len(),
            self.weight_tables.len(),
            self.weight_table_uploads,
            self.resources
                .values()
                .map(|resource| resource.token_slot_uploads)
                .sum(),
            self.resources
                .values()
                .map(|resource| resource.metadata_builds)
                .sum(),
        )
    }
}

impl fmt::Debug for ExpertCudaCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExpertCudaCache")
            .field("resources", &self.resources.len())
            .finish()
    }
}

fn tensor_ptr(
    tensor: &Tensor,
    stream: &candle_core::cuda::cudarc::driver::CudaStream,
) -> Result<Option<u64>> {
    if !tensor.is_contiguous() {
        return Ok(None);
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        return Ok(None);
    };
    let offset = layout.start_offset();
    let pointer = match tensor.dtype() {
        DType::F32 => slice_ptr_on_stream(storage.as_cuda_slice::<f32>()?, offset, stream).0,
        DType::F16 => slice_ptr_on_stream(storage.as_cuda_slice::<f16>()?, offset, stream).0,
        DType::BF16 => slice_ptr_on_stream(storage.as_cuda_slice::<bf16>()?, offset, stream).0,
        DType::U32 => slice_ptr_on_stream(storage.as_cuda_slice::<u32>()?, offset, stream).0,
        _ => return Ok(None),
    };
    Ok(Some(pointer))
}

fn descriptor(
    weights: &LoraExpertProjectionWeights,
    dtype: DType,
    stream: &candle_core::cuda::cudarc::driver::CudaStream,
) -> Result<Option<RoutedLoraAdapterWeight>> {
    if !supported_cuda_rank(weights.rank())
        || weights.a().dtype() != dtype
        || weights.b().dtype() != dtype
        || weights.scales().dtype() != DType::F32
        || !weights.a().is_contiguous()
        || !weights.b().is_contiguous()
        || !weights.scales().is_contiguous()
    {
        return Ok(None);
    }
    let Some(a) = tensor_ptr(weights.a(), stream)? else {
        return Ok(None);
    };
    let Some(b) = tensor_ptr(weights.b(), stream)? else {
        return Ok(None);
    };
    let Some(scales) = tensor_ptr(weights.scales(), stream)? else {
        return Ok(None);
    };
    let rank = u32::try_from(weights.rank()).map_err(candle_core::Error::wrap)?;
    Ok(Some(RoutedLoraAdapterWeight {
        a,
        b,
        scales,
        rank,
        rank_stride: rank,
        scale: 1.0,
        flags: 0,
    }))
}

fn projection_descriptors(
    adapters: &PreparedExpertAdapters,
    projection: LoraExpertProjection,
    dtype: DType,
    stream: &candle_core::cuda::cudarc::driver::CudaStream,
) -> Result<Option<Vec<RoutedLoraAdapterWeight>>> {
    let mut descriptors = Vec::with_capacity(adapters.slots.len());
    for (_, adapter) in &adapters.slots {
        let Some(weights) = adapter.projection(projection) else {
            descriptors.push(RoutedLoraAdapterWeight::empty());
            continue;
        };
        let Some(descriptor) = descriptor(weights, dtype, stream)? else {
            return Ok(None);
        };
        descriptors.push(descriptor);
    }
    Ok(Some(descriptors))
}

fn persistent_single_adapter_table(
    adapter: &Arc<LoraExpertWeights>,
    dtype: DType,
    device: &CudaDevice,
) -> Result<Option<(Arc<RoutedLoraCudaWeightTable>, bool)>> {
    let mut cache = adapter.cuda_table();
    if let Some(table) = cache.as_ref() {
        return Ok(Some((table.clone(), false)));
    }
    let stream = device.cuda_stream();
    let mut descriptors = Vec::with_capacity(3);
    for projection in [
        LoraExpertProjection::Gate,
        LoraExpertProjection::Up,
        LoraExpertProjection::Down,
    ] {
        let Some(weights) = adapter.projection(projection) else {
            descriptors.push(RoutedLoraAdapterWeight::empty());
            continue;
        };
        let Some(descriptor) = descriptor(weights, dtype, &stream)? else {
            return Ok(None);
        };
        descriptors.push(descriptor);
    }
    let table = Arc::new(RoutedLoraCudaWeightTable::new(device, &descriptors, 3, 1)?);
    *cache = Some(table.clone());
    Ok(Some((table, true)))
}

fn cached_weight_table(
    cache: &mut ExpertCudaCache,
    adapters: &PreparedExpertAdapters,
    site: &LoraExpertSiteHandle,
    dtype: DType,
    device: &CudaDevice,
) -> Result<Option<Arc<CachedWeightTable>>> {
    let gate_up_rank = adapters
        .slots
        .iter()
        .flat_map(|(_, adapter)| [adapter.gate(), adapter.up()].into_iter().flatten())
        .map(LoraExpertProjectionWeights::rank)
        .max()
        .unwrap_or(0);
    let down_rank = adapters
        .slots
        .iter()
        .filter_map(|(_, adapter)| adapter.down())
        .map(LoraExpertProjectionWeights::rank)
        .max()
        .unwrap_or(0);
    if adapters.slots.len() == 1 {
        let adapter = &adapters.slots[0].1;
        let Some((table, uploaded)) = persistent_single_adapter_table(adapter, dtype, device)?
        else {
            return Ok(None);
        };
        cache.weight_table_uploads += usize::from(uploaded);
        return Ok(Some(Arc::new(CachedWeightTable {
            table,
            gate_up_rank,
            down_rank,
            adapters: vec![Arc::downgrade(adapter)],
        })));
    }
    let adapter_ids = adapters
        .slots
        .iter()
        .map(|(_, adapter)| Arc::as_ptr(adapter) as usize)
        .collect::<Vec<_>>();
    let key = WeightTableKey {
        device: site.device().location(),
        site_id: site.id()?,
        adapter_ids,
    };
    let current_adapters = adapters
        .slots
        .iter()
        .map(|(_, adapter)| adapter)
        .collect::<Vec<_>>();
    if let Some(weights) = cache.weight_tables.get(&key) {
        if weak_identity_matches(&weights.adapters, &current_adapters) {
            return Ok(Some(weights.clone()));
        }
        cache.weight_tables.remove(&key);
    }
    let stream = device.cuda_stream();
    let Some(gate) = projection_descriptors(adapters, LoraExpertProjection::Gate, dtype, &stream)?
    else {
        return Ok(None);
    };
    let Some(up) = projection_descriptors(adapters, LoraExpertProjection::Up, dtype, &stream)?
    else {
        return Ok(None);
    };
    let Some(down) = projection_descriptors(adapters, LoraExpertProjection::Down, dtype, &stream)?
    else {
        return Ok(None);
    };
    let descriptors = gate.into_iter().chain(up).chain(down).collect::<Vec<_>>();
    let table = Arc::new(RoutedLoraCudaWeightTable::new(
        device,
        &descriptors,
        3,
        adapters.slots.len(),
    )?);
    let weights = Arc::new(CachedWeightTable {
        table,
        gate_up_rank,
        down_rank,
        adapters: adapters
            .slots
            .iter()
            .map(|(_, adapter)| Arc::downgrade(adapter))
            .collect(),
    });
    if cache.weight_tables.len() >= EXPERT_CUDA_WEIGHT_CACHE_CAPACITY {
        if let Some(evicted) = cache.weight_tables.keys().next().cloned() {
            cache.weight_tables.remove(&evicted);
        }
    }
    cache.weight_tables.insert(key, weights.clone());
    cache.weight_table_uploads += 1;
    Ok(Some(weights))
}

fn normalize_topk(topk_ids: &Tensor) -> Result<Option<Tensor>> {
    if !topk_ids.device().is_cuda() {
        return Ok(None);
    }
    let topk_ids = if topk_ids.dtype() == DType::U32 {
        topk_ids.clone()
    } else {
        topk_ids.to_dtype(DType::U32)?
    };
    Ok(Some(topk_ids.contiguous()?))
}

fn persistent_direct_resource(
    execution: &LoraExecution,
    site: &LoraExpertSiteHandle,
    topk_ids: &Tensor,
    dtype: DType,
) -> Result<Option<PersistentDirectResource>> {
    let Some((&slot, _)) = execution.rows_by_slot().first_key_value() else {
        return Ok(None);
    };
    if execution.rows_by_slot().len() != 1
        || execution
            .row_slots()
            .iter()
            .any(|row_slot| *row_slot != Some(slot))
    {
        return Ok(None);
    }
    let Some(adapter) = execution.expert_weights(site, slot)?.cloned() else {
        return Ok(None);
    };
    let Some(topk_ids) = normalize_topk(topk_ids)? else {
        return Ok(None);
    };
    let (num_tokens, top_k) = topk_ids.dims2()?;
    let layout = RoutedLoraMetadataLayout::new(num_tokens, top_k, site.spec().num_experts(), 1)?;
    let device = topk_ids.device().as_cuda_device()?.clone();
    let Some((table, _)) = persistent_single_adapter_table(&adapter, dtype, &device)? else {
        return Ok(None);
    };
    let source_topk_ptr = tensor_ptr(&topk_ids, &device.cuda_stream())?.unwrap_or(0);
    Ok(Some(PersistentDirectResource {
        device,
        layout,
        configured: ConfiguredSite {
            execution_id: execution.execution_id(),
            site_id: site.id()?,
            source_topk_ptr,
            topk_ids,
            adapter_ids: vec![Arc::as_ptr(&adapter) as usize],
            token_slots: vec![0; execution.row_slots().len()],
            adapters: vec![Arc::downgrade(&adapter)],
        },
        table,
        adapter,
    }))
}

fn configure_resource(
    resource: &mut ExpertCudaResources,
    execution_id: u64,
    site: &LoraExpertSiteHandle,
    adapters: &PreparedExpertAdapters,
    weights: &Arc<CachedWeightTable>,
    topk_ids: &Tensor,
    source_topk_ptr: u64,
) -> Result<Option<()>> {
    let site_id = site.id()?;
    let adapter_ids = adapters
        .slots
        .iter()
        .map(|(_, adapter)| Arc::as_ptr(adapter) as usize)
        .collect::<Vec<_>>();
    resource.weights.set_table(weights);
    if resource.configured.as_ref().is_some_and(|configured| {
        configured.execution_id == execution_id
            && configured.site_id == site_id
            && source_topk_ptr != 0
            && configured.source_topk_ptr == source_topk_ptr
            && configured.adapter_ids == adapter_ids
            && configured.token_slots == adapters.token_slots
    }) {
        return Ok(Some(()));
    }

    let stream = resource.device.cuda_stream();
    if resource.host_token_adapter_slots != adapters.token_slots {
        resource
            .device
            .memcpy_htod(&adapters.token_slots, &mut resource.token_adapter_slots)?;
        resource
            .host_token_adapter_slots
            .clone_from(&adapters.token_slots);
        resource.token_slot_uploads += 1;
    }

    if let Some(metadata) = &mut resource.metadata {
        let (token_slots_ptr, _token_slots_guard) =
            resource.token_adapter_slots.device_ptr(&stream);
        let (topk_storage, topk_layout) = topk_ids.storage_and_layout();
        let Storage::Cuda(topk_storage) = &*topk_storage else {
            return Ok(None);
        };
        let topk_slice = topk_storage.as_cuda_slice::<u32>()?;
        let (topk_ptr, _topk_guard) =
            slice_ptr_on_stream(topk_slice, topk_layout.start_offset(), &stream);
        unsafe {
            metadata.build(token_slots_ptr, topk_ptr)?;
        }
        resource.metadata_builds += 1;
    }
    resource.configured = Some(ConfiguredSite {
        execution_id,
        site_id,
        source_topk_ptr,
        topk_ids: topk_ids.clone(),
        adapter_ids,
        token_slots: adapters.token_slots.clone(),
        adapters: adapters
            .slots
            .iter()
            .map(|(_, adapter)| Arc::downgrade(adapter))
            .collect(),
    });
    Ok(Some(()))
}

fn with_prepared_resource<T>(
    execution: &LoraExecution,
    site: &LoraExpertSiteHandle,
    topk_ids: &Tensor,
    dtype: DType,
    f: impl FnOnce(&mut ExpertCudaResources) -> Result<Option<T>>,
) -> Result<Option<T>> {
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
        || !topk_ids.device().is_cuda()
        || site.device().location() != topk_ids.device().location()
        || execution.row_slots().len() != topk_ids.dim(0)?
    {
        return Ok(None);
    }
    let adapters = execution.prepared_expert_adapters(site)?;
    if adapters.slots.is_empty() {
        return Ok(None);
    }
    let Some(topk_ids) = normalize_topk(topk_ids)? else {
        return Ok(None);
    };
    let (num_tokens, top_k) = topk_ids.dims2()?;
    let layout = RoutedLoraMetadataLayout::new(
        num_tokens,
        top_k,
        site.spec().num_experts(),
        adapters.slots.len(),
    )?;
    let device = topk_ids.device().as_cuda_device()?;
    let mut cache = execution.expert_cuda_cache();
    let Some(weights) = cached_weight_table(&mut cache, &adapters, site, dtype, device)? else {
        return Ok(None);
    };
    let hidden = site.spec().hidden_size();
    let intermediate = site.spec().local_intermediate_size();
    #[cfg(feature = "cutile")]
    let cutile_compute_major = Some(crate::cutile::device_compute_capability(device).0);
    #[cfg(not(feature = "cutile"))]
    let cutile_compute_major = None;
    let needs_metadata = [
        NativeDirectShape {
            max_rank: weights.gate_up_rank,
            strongly_contracts_features: strongly_contracts_features(hidden, intermediate),
        },
        NativeDirectShape {
            max_rank: weights.down_rank,
            strongly_contracts_features: strongly_contracts_features(intermediate, hidden),
        },
    ]
    .into_iter()
    .filter(|shape| shape.max_rank != 0)
    .any(|shape| needs_grouped_metadata(layout, dtype, shape, cutile_compute_major));
    let key = ResourceKey {
        device: topk_ids.device().location(),
        num_tokens,
        top_k,
        num_experts: site.spec().num_experts(),
        num_adapter_slots: adapters.slots.len(),
        needs_metadata,
    };
    let source_topk_ptr = if topk_ids.dtype() == DType::U32 && topk_ids.is_contiguous() {
        tensor_ptr(&topk_ids, &device.cuda_stream())?.unwrap_or(0)
    } else {
        0
    };
    if !cache.resources.contains_key(&key) && cache.resources.len() >= EXPERT_CUDA_CACHE_CAPACITY {
        if let Some(evicted) = cache.resources.keys().next().copied() {
            cache.resources.remove(&evicted);
        }
    }
    let resource = match cache.resources.entry(key) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let token_adapter_slots = unsafe { device.alloc::<u32>(num_tokens)? };
            let metadata = if needs_metadata {
                Some(RoutedLoraCudaMetadata::new(device, layout)?)
            } else {
                None
            };
            entry.insert(ExpertCudaResources {
                device: device.clone(),
                layout,
                token_adapter_slots,
                host_token_adapter_slots: Vec::new(),
                metadata,
                weights: WeightResources::new(),
                configured: None,
                token_slot_uploads: 0,
                metadata_builds: 0,
            })
        }
    };
    if configure_resource(
        resource,
        execution.execution_id(),
        site,
        &adapters,
        &weights,
        &topk_ids,
        source_topk_ptr,
    )?
    .is_none()
    {
        return Ok(None);
    }
    f(resource)
}

fn validate_launch(
    site: &LoraExpertSiteHandle,
    input: &Tensor,
    base_output: &Tensor,
    topk_ids: &Tensor,
    projection: LoraExpertProjection,
    input_mode: LoraExpertInputMode,
    routed_weights: Option<&Tensor>,
) -> Result<bool> {
    let (num_tokens, top_k) = topk_ids.dims2()?;
    let (input_features, output_features) = site.spec().projection_shape(projection);
    let input_shape_ok = match input_mode {
        LoraExpertInputMode::TokenRows => input
            .dims2()
            .is_ok_and(|dims| dims == (num_tokens, input_features)),
        LoraExpertInputMode::RoutedRows => input
            .dims3()
            .is_ok_and(|dims| dims == (num_tokens, top_k, input_features)),
    };
    let weights_ok = routed_weights.is_none_or(|weights| {
        weights.dtype() == DType::F32
            && weights.is_contiguous()
            && weights
                .dims2()
                .is_ok_and(|dims| dims == (num_tokens, top_k))
            && weights.device().location() == input.device().location()
    });
    Ok(input_shape_ok
        && base_output
            .dims3()
            .is_ok_and(|dims| dims == (num_tokens, top_k, output_features))
        && input.is_contiguous()
        && base_output.is_contiguous()
        && weights_ok
        && input.dtype() == base_output.dtype()
        && input.dtype() == site.activation_dtype()
        && input.device().is_cuda()
        && input.device().location() == base_output.device().location()
        && input.device().location() == topk_ids.device().location())
}

fn routed_input_mode(input_mode: LoraExpertInputMode) -> RoutedLoraInputMode {
    match input_mode {
        LoraExpertInputMode::TokenRows => RoutedLoraInputMode::TokenRows,
        LoraExpertInputMode::RoutedRows => RoutedLoraInputMode::RoutedRows,
    }
}

trait RoutedLoraElement: CudaDType + DeviceRepr + WithDType {}

impl RoutedLoraElement for f32 {}
impl RoutedLoraElement for f16 {}
impl RoutedLoraElement for bf16 {}

struct ProjectionRun<'a> {
    input: &'a Tensor,
    base_output: &'a Tensor,
    routed_weights: Option<&'a Tensor>,
    input_mode: RoutedLoraInputMode,
    input_features: usize,
    output_features: usize,
    output_row_stride: usize,
    output_slice_stride: usize,
    num_slices: usize,
    max_rank: usize,
    weight_slice_offset: usize,
    in_place: bool,
}

struct ProjectionCudaContext<'a> {
    device: &'a CudaDevice,
    metadata_layout: RoutedLoraMetadataLayout,
    token_adapter_slots: Option<&'a CudaSlice<u32>>,
    metadata: Option<&'a RoutedLoraCudaMetadata>,
    configured: &'a ConfiguredSite,
    table: &'a RoutedLoraCudaWeightTable,
    scratch: &'a mut Option<CudaSlice<f32>>,
    #[cfg(feature = "cutile")]
    use_direct: bool,
}

fn launch_projection<T: RoutedLoraElement>(
    context: &mut ProjectionCudaContext<'_>,
    run: &ProjectionRun<'_>,
    output_storage: &mut CudaStorage,
    output_layout: &Layout,
) -> Result<()> {
    let projection_layout = RoutedLoraProjectionLayout::new(
        run.input_features,
        run.output_features,
        run.output_row_stride,
        run.output_slice_stride,
        run.num_slices,
        run.max_rank,
        run.input_mode,
    )?;
    let (input_storage, input_layout) = run.input.storage_and_layout();
    let Storage::Cuda(input_storage) = &*input_storage else {
        candle_core::bail!("routed LoRA input storage is not CUDA");
    };
    let input_slice = input_storage.as_cuda_slice::<T>()?;
    if !output_layout.is_contiguous() {
        candle_core::bail!("routed LoRA output storage is not contiguous");
    }
    let output = output_storage.as_cuda_slice_mut::<T>()?;

    let stream = context.device.cuda_stream();
    let (input_ptr, _input_guard) =
        slice_ptr_on_stream(input_slice, input_layout.start_offset(), &stream);
    let (output_ptr, output_guard) =
        slice_ptr_mut_on_stream(output, output_layout.start_offset(), &stream);
    let (token_slots_ptr, _token_slots_guard) = match context.token_adapter_slots {
        Some(token_adapter_slots) => {
            let (pointer, guard) = token_adapter_slots.device_ptr(&stream);
            (pointer, Some(guard))
        }
        None => (0, None),
    };
    let (topk_storage, topk_layout) = context.configured.topk_ids.storage_and_layout();
    let Storage::Cuda(topk_storage) = &*topk_storage else {
        candle_core::bail!("normalized routed LoRA topk IDs are not CUDA");
    };
    let topk_slice = topk_storage.as_cuda_slice::<u32>()?;
    let (topk_ptr, _topk_guard) =
        slice_ptr_on_stream(topk_slice, topk_layout.start_offset(), &stream);
    let routed_storage = run.routed_weights.map(Tensor::storage_and_layout);
    let (routed_weights_ptr, _routed_weights_guard) = match &routed_storage {
        Some((storage, layout)) => {
            let Storage::Cuda(storage) = &**storage else {
                candle_core::bail!("validated routed LoRA weights are not CUDA");
            };
            let (pointer, guard) = slice_ptr_on_stream(
                storage.as_cuda_slice::<f32>()?,
                layout.start_offset(),
                &stream,
            );
            (pointer, Some(guard))
        }
        None => (0, None),
    };
    let adapters = context
        .configured
        .adapters
        .iter()
        .map(Weak::upgrade)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| candle_core::Error::msg("routed LoRA adapter expired before launch"))?;
    let mut weight_storage = Vec::new();
    for adapter in &adapters {
        for slice in run.weight_slice_offset..run.weight_slice_offset + run.num_slices {
            let projection = match slice {
                0 => LoraExpertProjection::Gate,
                1 => LoraExpertProjection::Up,
                2 => LoraExpertProjection::Down,
                _ => candle_core::bail!("invalid routed LoRA descriptor slice"),
            };
            if let Some(weights) = adapter.projection(projection) {
                weight_storage.push((weights.a().storage_and_layout(), weights.a().dtype()));
                weight_storage.push((weights.b().storage_and_layout(), weights.b().dtype()));
                weight_storage.push((
                    weights.scales().storage_and_layout(),
                    weights.scales().dtype(),
                ));
            }
        }
    }
    let mut _weight_guards = Vec::with_capacity(weight_storage.len());
    for ((storage, layout), dtype) in &weight_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("routed LoRA adapter weights are not CUDA");
        };
        let (_, guard) = match dtype {
            DType::F32 => slice_ptr_on_stream(
                storage.as_cuda_slice::<f32>()?,
                layout.start_offset(),
                &stream,
            ),
            DType::F16 => slice_ptr_on_stream(
                storage.as_cuda_slice::<f16>()?,
                layout.start_offset(),
                &stream,
            ),
            DType::BF16 => slice_ptr_on_stream(
                storage.as_cuda_slice::<bf16>()?,
                layout.start_offset(),
                &stream,
            ),
            dtype => candle_core::bail!("invalid routed LoRA weight dtype {dtype:?}"),
        };
        _weight_guards.push(guard);
    }

    let native_shape = NativeDirectShape {
        max_rank: run.max_rank,
        strongly_contracts_features: strongly_contracts_features(
            projection_layout.input_features(),
            projection_layout.output_features(),
        ),
    };
    #[cfg(feature = "cutile")]
    let native_over_cutile = prefer_native_over_cutile(
        crate::cutile::device_compute_capability(context.device).0,
        context.metadata_layout,
        run.input.dtype(),
        native_shape,
    );

    unsafe {
        #[cfg(feature = "cutile")]
        let cutile_launched = if run.input.dtype() != DType::BF16 || native_over_cutile {
            false
        } else if context.use_direct
            && !prefer_small_batch_direct(context.metadata_layout, run.max_rank)
        {
            matches!(
                crate::cutile::try_cutile_routed_lora_no_sort(
                    context.device,
                    context.metadata_layout,
                    context.token_adapter_slots,
                    topk_slice,
                    topk_layout.start_offset(),
                    context.table,
                    crate::cutile::CutileRoutedLoraLaunch {
                        input: input_ptr,
                        output: output_ptr,
                        route_input_rows: 0,
                        route_output_rows: 0,
                        route_output_scales: routed_weights_ptr,
                        dtype: run.input.dtype(),
                        projection: projection_layout,
                        weight_slice_offset: run.weight_slice_offset,
                    },
                )?,
                crate::cutile::CutileRoutedLoraStatus::Launched
            )
        } else if !context.use_direct {
            matches!(
                crate::cutile::try_cutile_routed_lora(
                    context.device,
                    context.metadata.expect("grouped routed LoRA metadata"),
                    context.table,
                    crate::cutile::CutileRoutedLoraLaunch {
                        input: input_ptr,
                        output: output_ptr,
                        route_input_rows: 0,
                        route_output_rows: 0,
                        route_output_scales: routed_weights_ptr,
                        dtype: run.input.dtype(),
                        projection: projection_layout,
                        weight_slice_offset: run.weight_slice_offset,
                    },
                )?,
                crate::cutile::CutileRoutedLoraStatus::Launched
            )
        } else {
            false
        };
        #[cfg(not(feature = "cutile"))]
        let cutile_launched = false;

        if !cutile_launched {
            if prefer_native_direct(context.metadata_layout, run.input.dtype(), native_shape) {
                let launch = RoutedLoraDirectLaunch {
                    input: input_ptr,
                    output: output_ptr,
                    token_adapter_slots: token_slots_ptr,
                    topk_expert_ids: topk_ptr,
                    route_input_rows: 0,
                    route_output_rows: 0,
                    route_output_scales: routed_weights_ptr,
                    dtype: run.input.dtype(),
                    metadata: context.metadata_layout,
                    projection: projection_layout,
                    weight_slice_offset: run.weight_slice_offset,
                    output_splits: None,
                };
                launch_routed_lora_direct(context.table, launch)?;
            } else {
                let scratch_elements = context
                    .metadata_layout
                    .hidden_elements(run.num_slices, run.max_rank)?;
                if context
                    .scratch
                    .as_ref()
                    .is_none_or(|scratch| scratch.len() < scratch_elements)
                {
                    *context.scratch = Some(context.device.alloc::<f32>(scratch_elements)?);
                }
                let (hidden_ptr, hidden_guard) = context
                    .scratch
                    .as_mut()
                    .map(|scratch| slice_ptr_mut_on_stream(scratch, 0, &stream))
                    .map(|(pointer, guard)| (pointer, Some(guard)))
                    .unwrap_or((0, None));
                launch_routed_lora_grouped(
                    context.metadata.expect("grouped routed LoRA metadata"),
                    context.table,
                    RoutedLoraGroupedLaunch {
                        input: input_ptr,
                        hidden: hidden_ptr,
                        output: output_ptr,
                        route_input_rows: 0,
                        route_output_rows: 0,
                        route_output_scales: routed_weights_ptr,
                        dtype: run.input.dtype(),
                        projection: projection_layout,
                        weight_slice_offset: run.weight_slice_offset,
                    },
                )?;
                drop(hidden_guard);
            }
        }
    }
    drop(output_guard);
    Ok(())
}

struct RoutedLoraInplace<'a> {
    context: RefCell<ProjectionCudaContext<'a>>,
    run: &'a ProjectionRun<'a>,
}

impl InplaceOp1 for RoutedLoraInplace<'_> {
    fn name(&self) -> &'static str {
        "routed-lora-inplace"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        candle_core::bail!("routed LoRA in-place accumulation requires CUDA storage")
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let mut context = self.context.borrow_mut();
        match self.run.input.dtype() {
            DType::F32 => launch_projection::<f32>(&mut context, self.run, storage, layout),
            DType::F16 => launch_projection::<f16>(&mut context, self.run, storage, layout),
            DType::BF16 => launch_projection::<bf16>(&mut context, self.run, storage, layout),
            dtype => candle_core::bail!("routed LoRA CUDA does not support {dtype:?}"),
        }
    }
}

fn run_projection(context: ProjectionCudaContext<'_>, run: ProjectionRun<'_>) -> Result<Tensor> {
    let output = if run.in_place {
        run.base_output.clone()
    } else {
        run.base_output.copy()?
    };
    output.inplace_op1(&RoutedLoraInplace {
        context: RefCell::new(context),
        run: &run,
    })?;
    Ok(output)
}

fn run_persistent_direct(
    resource: &PersistentDirectResource,
    run: ProjectionRun<'_>,
) -> Result<Tensor> {
    let mut scratch = None;
    let context = ProjectionCudaContext {
        device: &resource.device,
        metadata_layout: resource.layout,
        token_adapter_slots: None,
        metadata: None,
        configured: &resource.configured,
        table: &resource.table,
        scratch: &mut scratch,
        #[cfg(feature = "cutile")]
        use_direct: true,
    };
    run_projection(context, run)
}

fn run_for_dtype(
    resource: &mut ExpertCudaResources,
    projection: LoraExpertProjection,
    run: ProjectionRun<'_>,
) -> Result<Tensor> {
    let ExpertCudaResources {
        device,
        layout,
        token_adapter_slots,
        metadata,
        weights,
        configured,
        ..
    } = resource;
    let table = weights
        .table
        .as_ref()
        .expect("active routed LoRA projection table");
    let scratch = match projection {
        LoraExpertProjection::Gate | LoraExpertProjection::Up => &mut weights.gate_up_scratch,
        LoraExpertProjection::Down => &mut weights.down_scratch,
    };
    let configured = configured
        .as_ref()
        .expect("routed LoRA resource is configured");
    #[cfg(feature = "cutile")]
    let use_direct = prefer_unsorted_direct(*layout, run.input.dtype(), run.max_rank);
    run_projection(
        ProjectionCudaContext {
            device,
            metadata_layout: *layout,
            token_adapter_slots: Some(token_adapter_slots),
            metadata: metadata.as_ref(),
            configured,
            table,
            scratch,
            #[cfg(feature = "cutile")]
            use_direct,
        },
        run,
    )
}

pub(super) fn try_add_delta(
    execution: &LoraExecution,
    site: &LoraExpertSiteHandle,
    delta: &LoraExpertDelta<'_>,
    in_place: bool,
) -> Result<Option<Tensor>> {
    let LoraExpertDelta {
        projection,
        input,
        base_output,
        topk_ids,
        routed_weights,
        input_mode,
    } = delta;
    if *projection != LoraExpertProjection::Down
        || !validate_launch(
            site,
            input,
            base_output,
            topk_ids,
            *projection,
            *input_mode,
            *routed_weights,
        )?
    {
        return Ok(None);
    }
    let (_, output_features) = site.spec().projection_shape(*projection);
    if let Some(resource) = persistent_direct_resource(execution, site, topk_ids, input.dtype())? {
        let Some(weights) = resource.adapter.down() else {
            return Ok(Some(base_output.clone()));
        };
        if prefer_unsorted_direct(resource.layout, input.dtype(), weights.rank()) {
            return run_persistent_direct(
                &resource,
                ProjectionRun {
                    input,
                    base_output,
                    routed_weights: *routed_weights,
                    input_mode: routed_input_mode(*input_mode),
                    input_features: input.dim(candle_core::D::Minus1)?,
                    output_features,
                    output_row_stride: output_features,
                    output_slice_stride: 0,
                    num_slices: 1,
                    max_rank: weights.rank(),
                    weight_slice_offset: 2,
                    in_place,
                },
            )
            .map(Some);
        }
    }
    with_prepared_resource(execution, site, topk_ids, input.dtype(), |resource| {
        if !resource.weights.down_active {
            return Ok(Some(base_output.clone()));
        }
        let max_rank = resource.weights.down_rank;
        run_for_dtype(
            resource,
            *projection,
            ProjectionRun {
                input,
                base_output,
                routed_weights: *routed_weights,
                input_mode: routed_input_mode(*input_mode),
                input_features: input.dim(candle_core::D::Minus1)?,
                output_features,
                output_row_stride: output_features,
                output_slice_stride: 0,
                num_slices: 1,
                max_rank,
                weight_slice_offset: 2,
                in_place,
            },
        )
        .map(Some)
    })
}

fn gate_up_for_kernel(
    site: &LoraExpertSiteHandle,
    base_gate_up: &Tensor,
    topk_ids: &Tensor,
) -> Result<Tensor> {
    let (num_tokens, top_k) = topk_ids.dims2()?;
    let intermediate = site.spec().local_intermediate_size();
    let gate_up = base_gate_up.reshape((num_tokens, top_k, intermediate * 2))?;
    match site.spec().gate_up_order() {
        LoraGateUpOrder::Concatenated => Ok(gate_up),
        LoraGateUpOrder::Interleaved => {
            let gate_up = gate_up.reshape((num_tokens, top_k, intermediate, 2))?;
            let gate = gate_up
                .narrow(candle_core::D::Minus1, 0, 1)?
                .squeeze(candle_core::D::Minus1)?
                .contiguous()?;
            let up = gate_up
                .narrow(candle_core::D::Minus1, 1, 1)?
                .squeeze(candle_core::D::Minus1)?
                .contiguous()?;
            Tensor::cat(&[&gate, &up], candle_core::D::Minus1)
        }
    }
}

fn split_kernel_gate_up(
    site: &LoraExpertSiteHandle,
    gate_up: Tensor,
    topk_ids: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (num_tokens, top_k) = topk_ids.dims2()?;
    let intermediate = site.spec().local_intermediate_size();
    let gate_up = gate_up.reshape((num_tokens, top_k, intermediate * 2))?;
    Ok((
        gate_up
            .narrow(candle_core::D::Minus1, 0, intermediate)?
            .contiguous()?,
        gate_up
            .narrow(candle_core::D::Minus1, intermediate, intermediate)?
            .contiguous()?,
    ))
}

pub(super) fn try_add_gate_up_delta_combined(
    execution: &LoraExecution,
    site: &LoraExpertSiteHandle,
    input: &Tensor,
    base_gate_up: &Tensor,
    topk_ids: &Tensor,
    in_place: bool,
) -> Result<Option<Tensor>> {
    let intermediate = site.spec().local_intermediate_size();
    let expected = topk_ids.elem_count() * intermediate * 2;
    if input.dims2()? != (topk_ids.dim(0)?, site.spec().hidden_size())
        || base_gate_up.elem_count() != expected
        || !input.is_contiguous()
        || !base_gate_up.is_contiguous()
        || input.dtype() != base_gate_up.dtype()
        || input.dtype() != site.activation_dtype()
        || !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16)
        || !input.device().is_cuda()
        || input.device().location() != site.device().location()
        || input.device().location() != base_gate_up.device().location()
        || input.device().location() != topk_ids.device().location()
    {
        return Ok(None);
    }
    let base_gate_up = gate_up_for_kernel(site, base_gate_up, topk_ids)?;
    let persistent = persistent_direct_resource(execution, site, topk_ids, input.dtype())?;
    let persistent_rank = persistent
        .as_ref()
        .map(|resource| {
            [resource.adapter.gate(), resource.adapter.up()]
                .into_iter()
                .flatten()
                .map(LoraExpertProjectionWeights::rank)
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    if persistent.is_some() && persistent_rank == 0 {
        return Ok(Some(base_gate_up));
    }
    let output = if persistent.as_ref().is_some_and(|resource| {
        prefer_unsorted_direct(resource.layout, input.dtype(), persistent_rank)
    }) {
        let resource = persistent.as_ref().expect("checked persistent resource");
        Some(run_persistent_direct(
            resource,
            ProjectionRun {
                input,
                base_output: &base_gate_up,
                routed_weights: None,
                input_mode: RoutedLoraInputMode::TokenRows,
                input_features: site.spec().hidden_size(),
                output_features: intermediate,
                output_row_stride: intermediate * 2,
                output_slice_stride: intermediate,
                num_slices: 2,
                max_rank: persistent_rank,
                weight_slice_offset: 0,
                in_place,
            },
        )?)
    } else {
        with_prepared_resource(execution, site, topk_ids, input.dtype(), |resource| {
            if !resource.weights.gate_up_active {
                return Ok(Some(base_gate_up.clone()));
            }
            let max_rank = resource.weights.gate_up_rank;
            run_for_dtype(
                resource,
                LoraExpertProjection::Gate,
                ProjectionRun {
                    input,
                    base_output: &base_gate_up,
                    routed_weights: None,
                    input_mode: RoutedLoraInputMode::TokenRows,
                    input_features: site.spec().hidden_size(),
                    output_features: intermediate,
                    output_row_stride: intermediate * 2,
                    output_slice_stride: intermediate,
                    num_slices: 2,
                    max_rank,
                    weight_slice_offset: 0,
                    in_place,
                },
            )
            .map(Some)
        })?
    };
    Ok(output)
}

pub(super) fn try_add_gate_up_delta(
    execution: &LoraExecution,
    site: &LoraExpertSiteHandle,
    input: &Tensor,
    base_gate_up: &Tensor,
    topk_ids: &Tensor,
    in_place: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    try_add_gate_up_delta_combined(execution, site, input, base_gate_up, topk_ids, in_place)?
        .map(|output| split_kernel_gate_up(site, output, topk_ids))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_dispatch_matches_work_and_projection_boundaries() -> Result<()> {
        let at_limit = RoutedLoraMetadataLayout::new(8, 2, 8, 1)?;
        let over_work_limit = RoutedLoraMetadataLayout::new(17, 1, 8, 1)?;
        let many_routes = RoutedLoraMetadataLayout::new(32, 1, 8, 1)?;
        let low_rank_many_routes = RoutedLoraMetadataLayout::new(64, 2, 8, 1)?;
        let low_rank_over_limit = RoutedLoraMetadataLayout::new(65, 2, 8, 1)?;
        let persistent_arena = RoutedLoraMetadataLayout::new(24, 2, 4, 2)?;
        assert!(prefer_unsorted_direct(at_limit, DType::BF16, 64));
        assert!(!prefer_unsorted_direct(over_work_limit, DType::BF16, 64));
        assert!(!prefer_unsorted_direct(many_routes, DType::F16, 64));
        assert!(prefer_unsorted_direct(low_rank_many_routes, DType::F16, 8));
        assert!(!prefer_unsorted_direct(low_rank_over_limit, DType::F16, 8));
        assert!(prefer_unsorted_direct(at_limit, DType::F32, 17));

        let sparse_rank128 = RoutedLoraMetadataLayout::new(1, 8, 128, 1)?;
        let dense_rank128 = RoutedLoraMetadataLayout::new(1, 8, 8, 1)?;
        assert!(!prefer_small_batch_direct(sparse_rank128, 128));
        assert!(prefer_unsorted_direct(sparse_rank128, DType::BF16, 128));
        assert!(!prefer_unsorted_direct(dense_rank128, DType::BF16, 128));
        assert!(!prefer_unsorted_direct(
            persistent_arena,
            DType::F16,
            ROUTED_LORA_WMMA_RANK_CAP + 1,
        ));

        let native_short = RoutedLoraMetadataLayout::new(16, 8, 128, 4)?;
        let native_contraction_limit = RoutedLoraMetadataLayout::new(64, 8, 128, 4)?;
        let native_over_limit = RoutedLoraMetadataLayout::new(65, 8, 128, 4)?;
        let contraction_shape = NativeDirectShape {
            max_rank: 128,
            strongly_contracts_features: true,
        };
        let expansion_shape = NativeDirectShape {
            max_rank: 128,
            strongly_contracts_features: false,
        };
        let rank129_contraction = NativeDirectShape {
            max_rank: 129,
            ..contraction_shape
        };
        assert!(!prefer_unsorted_direct(
            native_contraction_limit,
            DType::BF16,
            128
        ));
        assert!(prefer_native_direct(
            native_short,
            DType::BF16,
            expansion_shape
        ));
        assert!(prefer_native_direct(
            native_contraction_limit,
            DType::BF16,
            contraction_shape
        ));
        assert!(!prefer_native_direct(
            native_contraction_limit,
            DType::BF16,
            expansion_shape
        ));
        assert!(!prefer_native_direct(
            native_over_limit,
            DType::BF16,
            contraction_shape
        ));
        assert!(!prefer_native_direct(
            native_contraction_limit,
            DType::BF16,
            rank129_contraction
        ));
        assert!(strongly_contracts_features(4096, 1536));
        assert!(!strongly_contracts_features(4096, 4096));
        assert!(!strongly_contracts_features(4096, 14336));
        assert!(supported_cuda_rank(ROUTED_LORA_MAX_RANK));
        assert!(!supported_cuda_rank(ROUTED_LORA_MAX_RANK + 1));
        #[cfg(feature = "cutile")]
        assert!(needs_grouped_metadata(
            native_contraction_limit,
            DType::BF16,
            contraction_shape,
            Some(8)
        ));
        #[cfg(feature = "cutile")]
        assert!(!needs_grouped_metadata(
            native_contraction_limit,
            DType::F16,
            contraction_shape,
            Some(8)
        ));
        #[cfg(not(feature = "cutile"))]
        assert!(!needs_grouped_metadata(
            native_contraction_limit,
            DType::BF16,
            contraction_shape,
            None
        ));
        assert!(needs_grouped_metadata(
            native_contraction_limit,
            DType::BF16,
            expansion_shape,
            {
                #[cfg(feature = "cutile")]
                {
                    Some(8)
                }
                #[cfg(not(feature = "cutile"))]
                {
                    None
                }
            }
        ));
        #[cfg(feature = "cutile")]
        {
            let rank16_contraction = NativeDirectShape {
                max_rank: 16,
                ..contraction_shape
            };
            let below_cutile_native_limit = RoutedLoraMetadataLayout::new(31, 8, 128, 4)?;
            let at_cutile_native_limit = RoutedLoraMetadataLayout::new(32, 8, 128, 4)?;
            assert!(prefer_native_over_cutile(
                CUTILE_NATIVE_DIRECT_COMPUTE_MAJOR,
                native_contraction_limit,
                DType::BF16,
                rank16_contraction
            ));
            assert!(prefer_native_over_cutile(
                CUTILE_NATIVE_DIRECT_COMPUTE_MAJOR,
                at_cutile_native_limit,
                DType::BF16,
                rank16_contraction
            ));
            assert!(!prefer_native_over_cutile(
                CUTILE_NATIVE_DIRECT_COMPUTE_MAJOR,
                below_cutile_native_limit,
                DType::BF16,
                rank16_contraction
            ));
            assert!(!prefer_native_over_cutile(
                8,
                native_contraction_limit,
                DType::BF16,
                rank16_contraction
            ));
            assert!(!needs_grouped_metadata(
                native_contraction_limit,
                DType::BF16,
                rank16_contraction,
                Some(CUTILE_NATIVE_DIRECT_COMPUTE_MAJOR)
            ));
        }
        Ok(())
    }

    #[test]
    fn weak_identity_rejects_expired_and_reordered_entries() {
        let first = Arc::new(1u8);
        let second = Arc::new(2u8);
        let cached = vec![Arc::downgrade(&first), Arc::downgrade(&second)];
        assert!(weak_identity_matches(&cached, &[&first, &second]));
        assert!(!weak_identity_matches(&cached, &[&second, &first]));
        drop(first);
        assert!(!weak_identity_matches(&cached, &[&second, &second]));
    }
}
