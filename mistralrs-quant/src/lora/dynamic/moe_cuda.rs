#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::Result;

pub const ROUTED_LORA_BASE_SLOT: u32 = u32::MAX;
pub const ROUTED_LORA_BLOCK_SIZE: usize = 16;
pub const ROUTED_LORA_MAX_RANK: usize = 512;
pub const ROUTED_LORA_WMMA_RANK_CAP: usize = 128;
#[cfg(any(feature = "cuda", test))]
pub(super) const ROUTED_LORA_SMALL_BATCH_RANK_CAP: usize = 64;
#[cfg(any(feature = "cuda", test))]
pub(super) const ROUTED_LORA_SMALL_BATCH_ROUTE_RANK_LIMIT: usize = 1024;
#[cfg(any(feature = "cuda", test))]
const ROUTED_LORA_TARGET_CTAS_PER_SM: usize = 2;
#[cfg(any(feature = "cuda", test))]
const ROUTED_LORA_RECOMPUTE_BUDGET_NUMERATOR: usize = 3;
#[cfg(any(feature = "cuda", test))]
const ROUTED_LORA_RECOMPUTE_BUDGET_DENOMINATOR: usize = 2;
#[cfg(any(feature = "cuda", test))]
const ROUTED_LORA_DIRECT_OUTPUT_CHUNK: usize = 128;
#[cfg(any(feature = "cuda", test))]
const ROUTED_LORA_WMMA_RANK64_SPLIT_CAP: usize = 8;
#[cfg(any(feature = "cuda", test))]
const ROUTED_LORA_WMMA_RANK128_SPLIT_CAP: usize = 4;

#[cfg(any(feature = "cuda", test))]
fn validate_selected_descriptor_rank(
    max_rank_by_slice: &[usize],
    weight_slice_offset: usize,
    num_slices: usize,
    projection_max_rank: usize,
) -> Result<()> {
    let end = weight_slice_offset
        .checked_add(num_slices)
        .ok_or_else(|| candle_core::Error::msg("routed LoRA descriptor slice range overflow"))?;
    let selected_max_rank = max_rank_by_slice
        .get(weight_slice_offset..end)
        .ok_or_else(|| candle_core::Error::msg("routed LoRA descriptor slice range is invalid"))?
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if selected_max_rank > projection_max_rank {
        candle_core::bail!(
            "routed LoRA projection max rank {projection_max_rank} is smaller than selected descriptor rank {selected_max_rank}"
        );
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn output_splits(
    sm_count: usize,
    base_ctas: usize,
    output_tiles: usize,
    split_cap: usize,
) -> usize {
    let target_ctas = sm_count.max(1) * ROUTED_LORA_TARGET_CTAS_PER_SM;
    target_ctas
        .div_ceil(base_ctas.max(1))
        .min(output_tiles.max(1))
        .min(split_cap)
        .max(1)
}

#[cfg(any(feature = "cuda", test))]
fn output_recompute_cap(projection: RoutedLoraProjectionLayout) -> usize {
    let total_features = projection
        .input_features()
        .saturating_add(projection.output_features());
    total_features
        .saturating_mul(ROUTED_LORA_RECOMPUTE_BUDGET_NUMERATOR)
        .checked_div(
            projection
                .input_features()
                .saturating_mul(ROUTED_LORA_RECOMPUTE_BUDGET_DENOMINATOR),
        )
        .unwrap_or(usize::MAX)
        .saturating_add(1)
        .max(1)
}

#[cfg(any(feature = "cuda", test))]
fn direct_small_batch_output_splits(
    sm_count: usize,
    metadata: RoutedLoraMetadataLayout,
    projection: RoutedLoraProjectionLayout,
) -> usize {
    let base_ctas = metadata.num_routes() * projection.num_slices();
    let output_tiles = projection
        .output_features()
        .div_ceil(ROUTED_LORA_DIRECT_OUTPUT_CHUNK);
    let target_ctas = sm_count.max(1) * ROUTED_LORA_TARGET_CTAS_PER_SM;
    let total_tile_ctas = base_ctas.saturating_mul(output_tiles);
    let tiles_per_cta = if total_tile_ctas <= target_ctas {
        1
    } else {
        total_tile_ctas.div_ceil(target_ctas)
    };
    output_tiles.div_ceil(tiles_per_cta).max(1)
}

#[cfg(any(feature = "cuda", test))]
fn direct_recompute_output_splits(
    sm_count: usize,
    metadata: RoutedLoraMetadataLayout,
    projection: RoutedLoraProjectionLayout,
) -> usize {
    output_splits(
        sm_count,
        metadata.num_routes() * projection.num_slices(),
        projection
            .output_features()
            .div_ceil(ROUTED_LORA_DIRECT_OUTPUT_CHUNK),
        output_recompute_cap(projection),
    )
}

#[cfg(any(feature = "cuda", test))]
fn grouped_wmma_output_splits(
    sm_count: usize,
    metadata: RoutedLoraMetadataLayout,
    projection: RoutedLoraProjectionLayout,
) -> usize {
    let split_cap = if projection.max_rank() <= 64 {
        ROUTED_LORA_WMMA_RANK64_SPLIT_CAP
    } else {
        ROUTED_LORA_WMMA_RANK128_SPLIT_CAP
    };
    output_splits(
        sm_count,
        metadata
            .max_blocks()
            .saturating_mul(projection.num_slices()),
        projection
            .output_features()
            .div_ceil(ROUTED_LORA_BLOCK_SIZE * 8),
        split_cap.min(output_recompute_cap(projection)),
    )
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn prefer_small_batch_direct(
    metadata: RoutedLoraMetadataLayout,
    max_rank: usize,
) -> bool {
    max_rank <= ROUTED_LORA_SMALL_BATCH_RANK_CAP
        && metadata
            .num_routes()
            .checked_mul(max_rank)
            .is_some_and(|work| work <= ROUTED_LORA_SMALL_BATCH_ROUTE_RANK_LIMIT)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RoutedLoraInputMode {
    RoutedRows = 0,
    TokenRows = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedLoraMetadataLayout {
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    num_adapter_slots: usize,
    num_routes: usize,
    num_pairs: usize,
    block_size: usize,
    max_padded_routes: usize,
    max_blocks: usize,
}

impl RoutedLoraMetadataLayout {
    pub fn new(
        num_tokens: usize,
        top_k: usize,
        num_experts: usize,
        num_adapter_slots: usize,
    ) -> Result<Self> {
        if num_tokens == 0 || top_k == 0 || num_experts == 0 || num_adapter_slots == 0 {
            candle_core::bail!(
                "routed LoRA metadata dimensions and adapter capacity must be nonzero"
            );
        }
        let num_routes = num_tokens
            .checked_mul(top_k)
            .ok_or_else(|| candle_core::Error::msg("routed LoRA route count overflow"))?;
        let num_pairs = num_experts
            .checked_mul(num_adapter_slots)
            .ok_or_else(|| candle_core::Error::msg("routed LoRA pair count overflow"))?;
        let possible_pairs = num_routes.min(num_pairs);
        let max_blocks = possible_pairs
            .checked_add((num_routes - possible_pairs) / ROUTED_LORA_BLOCK_SIZE)
            .ok_or_else(|| candle_core::Error::msg("routed LoRA block count overflow"))?;
        let max_padded_routes = max_blocks
            .checked_mul(ROUTED_LORA_BLOCK_SIZE)
            .ok_or_else(|| candle_core::Error::msg("routed LoRA padding overflow"))?;
        for (name, value) in [
            ("num_tokens", num_tokens),
            ("top_k", top_k),
            ("num_experts", num_experts),
            ("num_adapter_slots", num_adapter_slots),
            ("num_routes", num_routes),
            ("num_pairs", num_pairs),
            ("max_padded_routes", max_padded_routes),
            ("max_blocks", max_blocks),
        ] {
            if value > i32::MAX as usize {
                candle_core::bail!("routed LoRA {name} exceeds the CUDA ABI limit");
            }
        }
        Ok(Self {
            num_tokens,
            top_k,
            num_experts,
            num_adapter_slots,
            num_routes,
            num_pairs,
            block_size: ROUTED_LORA_BLOCK_SIZE,
            max_padded_routes,
            max_blocks,
        })
    }

    pub fn num_tokens(self) -> usize {
        self.num_tokens
    }

    pub fn top_k(self) -> usize {
        self.top_k
    }

    pub fn num_experts(self) -> usize {
        self.num_experts
    }

    pub fn num_adapter_slots(self) -> usize {
        self.num_adapter_slots
    }

    pub fn num_routes(self) -> usize {
        self.num_routes
    }

    pub fn num_pairs(self) -> usize {
        self.num_pairs
    }

    pub fn block_size(self) -> usize {
        self.block_size
    }

    pub fn max_padded_routes(self) -> usize {
        self.max_padded_routes
    }

    pub fn max_blocks(self) -> usize {
        self.max_blocks
    }

    pub fn hidden_elements(self, num_slices: usize, max_rank: usize) -> Result<usize> {
        self.num_routes
            .checked_mul(num_slices)
            .and_then(|elements| elements.checked_mul(max_rank))
            .ok_or_else(|| candle_core::Error::msg("routed LoRA scratch size overflow"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedLoraProjectionLayout {
    input_features: usize,
    output_features: usize,
    output_row_stride: usize,
    output_slice_stride: usize,
    num_slices: usize,
    max_rank: usize,
    input_mode: RoutedLoraInputMode,
}

impl RoutedLoraProjectionLayout {
    pub fn new(
        input_features: usize,
        output_features: usize,
        output_row_stride: usize,
        output_slice_stride: usize,
        num_slices: usize,
        max_rank: usize,
        input_mode: RoutedLoraInputMode,
    ) -> Result<Self> {
        if input_features == 0
            || output_features == 0
            || num_slices == 0
            || max_rank == 0
            || max_rank > ROUTED_LORA_MAX_RANK
        {
            candle_core::bail!("invalid routed LoRA projection dimensions");
        }
        let required_row = output_slice_stride
            .checked_mul(num_slices - 1)
            .and_then(|offset| offset.checked_add(output_features))
            .ok_or_else(|| candle_core::Error::msg("routed LoRA output stride overflow"))?;
        if output_row_stride < required_row {
            candle_core::bail!(
                "routed LoRA output row stride {output_row_stride} is smaller than {required_row}"
            );
        }
        for (name, value) in [
            ("input_features", input_features),
            ("output_features", output_features),
            ("output_row_stride", output_row_stride),
            ("output_slice_stride", output_slice_stride),
            ("num_slices", num_slices),
            ("max_rank", max_rank),
        ] {
            if value > i32::MAX as usize {
                candle_core::bail!("routed LoRA {name} exceeds the CUDA ABI limit");
            }
        }
        Ok(Self {
            input_features,
            output_features,
            output_row_stride,
            output_slice_stride,
            num_slices,
            max_rank,
            input_mode,
        })
    }

    pub fn input_features(self) -> usize {
        self.input_features
    }

    pub fn output_features(self) -> usize {
        self.output_features
    }

    pub fn output_row_stride(self) -> usize {
        self.output_row_stride
    }

    pub fn output_slice_stride(self) -> usize {
        self.output_slice_stride
    }

    pub fn num_slices(self) -> usize {
        self.num_slices
    }

    pub fn max_rank(self) -> usize {
        self.max_rank
    }

    pub fn input_mode(self) -> RoutedLoraInputMode {
        self.input_mode
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct RoutedLoraAdapterWeight {
    pub a: u64,
    pub b: u64,
    pub scales: u64,
    pub rank: u32,
    pub rank_stride: u32,
    pub scale: f32,
    pub flags: u32,
}

impl RoutedLoraAdapterWeight {
    pub const fn empty() -> Self {
        Self {
            a: 0,
            b: 0,
            scales: 0,
            rank: 0,
            rank_stride: 0,
            scale: 0.0,
            flags: 0,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.rank == 0 {
            if self.a == 0 && self.b == 0 {
                return Ok(());
            }
            candle_core::bail!("disabled routed LoRA descriptors must have null A and B pointers");
        }
        if self.a == 0 || self.b == 0 {
            candle_core::bail!("active routed LoRA descriptors require A and B pointers");
        }
        if self.rank > self.rank_stride || self.rank as usize > ROUTED_LORA_MAX_RANK {
            candle_core::bail!("invalid routed LoRA rank or rank stride");
        }
        if self.scales == 0 && !self.scale.is_finite() {
            candle_core::bail!("routed LoRA scalar scale must be finite");
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
unsafe impl candle_core::cuda::cudarc::driver::DeviceRepr for RoutedLoraAdapterWeight {}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::{
        cuda::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut},
        CudaDevice,
    };

    use super::super::moe_cuda_ffi;

    pub struct RoutedLoraCudaWeightTable {
        device: CudaDevice,
        descriptors: CudaSlice<RoutedLoraAdapterWeight>,
        num_slices: usize,
        num_adapter_slots: usize,
        max_rank: usize,
        max_rank_by_slice: Vec<usize>,
        max_rank_stride_by_slice: Vec<usize>,
        sm80_or_newer: bool,
        sm_count: usize,
    }

    impl RoutedLoraCudaWeightTable {
        pub fn new(
            device: &CudaDevice,
            descriptors: &[RoutedLoraAdapterWeight],
            num_slices: usize,
            num_adapter_slots: usize,
        ) -> Result<Self> {
            if num_slices == 0
                || num_adapter_slots == 0
                || descriptors.len() != num_slices * num_adapter_slots
            {
                candle_core::bail!("routed LoRA descriptor table shape mismatch");
            }
            for descriptor in descriptors {
                descriptor.validate()?;
            }
            let max_rank = descriptors
                .iter()
                .map(|descriptor| descriptor.rank as usize)
                .max()
                .unwrap_or(0);
            let max_rank_by_slice = descriptors
                .chunks_exact(num_adapter_slots)
                .map(|slice| {
                    slice
                        .iter()
                        .map(|descriptor| descriptor.rank as usize)
                        .max()
                        .unwrap_or(0)
                })
                .collect();
            let max_rank_stride_by_slice = descriptors
                .chunks_exact(num_adapter_slots)
                .map(|slice| {
                    slice
                        .iter()
                        .map(|descriptor| descriptor.rank_stride as usize)
                        .max()
                        .unwrap_or(0)
                })
                .collect();
            use candle_core::cuda::cudarc::driver::{result, sys};
            let cu_device = device.cuda_stream().context().cu_device();
            let compute_major = unsafe {
                result::device::get_attribute(
                    cu_device,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
            }
            .unwrap_or(0);
            let sm_count = unsafe {
                result::device::get_attribute(
                    cu_device,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                )
            }
            .unwrap_or(1)
            .max(1) as usize;
            let mut table = unsafe { device.alloc::<RoutedLoraAdapterWeight>(descriptors.len())? };
            device.memcpy_htod(descriptors, &mut table)?;
            Ok(Self {
                device: device.clone(),
                descriptors: table,
                num_slices,
                num_adapter_slots,
                max_rank,
                max_rank_by_slice,
                max_rank_stride_by_slice,
                sm80_or_newer: compute_major >= 8,
                sm_count,
            })
        }

        pub fn update(&mut self, descriptors: &[RoutedLoraAdapterWeight]) -> Result<()> {
            if descriptors.len() != self.num_slices * self.num_adapter_slots {
                candle_core::bail!("routed LoRA descriptor table shape mismatch");
            }
            for descriptor in descriptors {
                descriptor.validate()?;
            }
            self.device
                .memcpy_htod(descriptors, &mut self.descriptors)?;
            self.max_rank = descriptors
                .iter()
                .map(|descriptor| descriptor.rank as usize)
                .max()
                .unwrap_or(0);
            self.max_rank_by_slice = descriptors
                .chunks_exact(self.num_adapter_slots)
                .map(|slice| {
                    slice
                        .iter()
                        .map(|descriptor| descriptor.rank as usize)
                        .max()
                        .unwrap_or(0)
                })
                .collect();
            self.max_rank_stride_by_slice = descriptors
                .chunks_exact(self.num_adapter_slots)
                .map(|slice| {
                    slice
                        .iter()
                        .map(|descriptor| descriptor.rank_stride as usize)
                        .max()
                        .unwrap_or(0)
                })
                .collect();
            Ok(())
        }

        pub fn num_slices(&self) -> usize {
            self.num_slices
        }

        pub fn num_adapter_slots(&self) -> usize {
            self.num_adapter_slots
        }

        pub fn descriptors(&self) -> &CudaSlice<RoutedLoraAdapterWeight> {
            &self.descriptors
        }

        pub fn max_rank(&self) -> usize {
            self.max_rank
        }

        pub(crate) fn validate_projection_rank(
            &self,
            weight_slice_offset: usize,
            num_slices: usize,
            projection_max_rank: usize,
        ) -> Result<()> {
            validate_selected_descriptor_rank(
                &self.max_rank_by_slice,
                weight_slice_offset,
                num_slices,
                projection_max_rank,
            )
        }

        pub fn max_rank_stride_in_range(
            &self,
            weight_slice_offset: usize,
            num_slices: usize,
        ) -> Option<usize> {
            let end = weight_slice_offset.checked_add(num_slices)?;
            self.max_rank_stride_by_slice
                .get(weight_slice_offset..end)?
                .iter()
                .copied()
                .max()
        }

        pub fn supports_wmma(&self, projection: RoutedLoraProjectionLayout) -> bool {
            self.sm80_or_newer && projection.max_rank() <= ROUTED_LORA_WMMA_RANK_CAP
        }

        fn direct_output_splits(
            &self,
            metadata: RoutedLoraMetadataLayout,
            projection: RoutedLoraProjectionLayout,
        ) -> usize {
            if prefer_small_batch_direct(metadata, projection.max_rank()) {
                return direct_small_batch_output_splits(self.sm_count, metadata, projection);
            }
            direct_recompute_output_splits(self.sm_count, metadata, projection)
        }

        fn wmma_output_splits(
            &self,
            metadata: RoutedLoraMetadataLayout,
            projection: RoutedLoraProjectionLayout,
        ) -> usize {
            grouped_wmma_output_splits(self.sm_count, metadata, projection)
        }
    }

    pub struct RoutedLoraCudaMetadata {
        device: CudaDevice,
        layout: RoutedLoraMetadataLayout,
        route_pair_ids: CudaSlice<u32>,
        pair_counts: CudaSlice<u32>,
        pair_offsets: CudaSlice<u32>,
        pair_cursors: CudaSlice<u32>,
        sorted_route_ids: CudaSlice<u32>,
        block_pair_ids: CudaSlice<u32>,
        num_active_routes: CudaSlice<u32>,
        num_padded_routes: CudaSlice<u32>,
        scan_workspace: CudaSlice<u8>,
    }

    impl RoutedLoraCudaMetadata {
        pub fn new(device: &CudaDevice, layout: RoutedLoraMetadataLayout) -> Result<Self> {
            let scan_workspace_bytes = unsafe {
                moe_cuda_ffi::routed_lora_metadata_workspace_size(
                    layout.num_experts() as i32,
                    layout.num_adapter_slots() as i32,
                )
            };
            if scan_workspace_bytes == 0 {
                candle_core::bail!("failed to size routed LoRA metadata scan workspace");
            }
            Ok(Self {
                device: device.clone(),
                layout,
                route_pair_ids: unsafe { device.alloc::<u32>(layout.num_routes())? },
                pair_counts: unsafe { device.alloc::<u32>(layout.num_pairs())? },
                pair_offsets: unsafe { device.alloc::<u32>(layout.num_pairs() + 1)? },
                pair_cursors: unsafe { device.alloc::<u32>(layout.num_pairs())? },
                sorted_route_ids: unsafe { device.alloc::<u32>(layout.max_padded_routes())? },
                block_pair_ids: unsafe { device.alloc::<u32>(layout.max_blocks())? },
                num_active_routes: unsafe { device.alloc::<u32>(1)? },
                num_padded_routes: unsafe { device.alloc::<u32>(1)? },
                scan_workspace: unsafe { device.alloc::<u8>(scan_workspace_bytes)? },
            })
        }

        pub fn layout(&self) -> RoutedLoraMetadataLayout {
            self.layout
        }

        pub fn sorted_route_ids(&self) -> &CudaSlice<u32> {
            &self.sorted_route_ids
        }

        pub fn block_pair_ids(&self) -> &CudaSlice<u32> {
            &self.block_pair_ids
        }

        pub fn route_pair_ids(&self) -> &CudaSlice<u32> {
            &self.route_pair_ids
        }

        pub fn pair_counts(&self) -> &CudaSlice<u32> {
            &self.pair_counts
        }

        pub fn pair_offsets(&self) -> &CudaSlice<u32> {
            &self.pair_offsets
        }

        pub fn num_active_routes(&self) -> &CudaSlice<u32> {
            &self.num_active_routes
        }

        pub fn num_padded_routes(&self) -> &CudaSlice<u32> {
            &self.num_padded_routes
        }

        /// Builds pair-sorted metadata on the current CUDA stream.
        ///
        /// # Safety
        /// `token_adapter_slots` and `topk_expert_ids` must be valid device pointers for this
        /// metadata shape and remain alive until the launch completes.
        pub unsafe fn build(
            &mut self,
            token_adapter_slots: u64,
            topk_expert_ids: u64,
        ) -> Result<()> {
            let stream = self.device.cuda_stream();
            let (route_pair_ids, _route_pair_ids_guard) =
                self.route_pair_ids.device_ptr_mut(&stream);
            let (pair_counts, _pair_counts_guard) = self.pair_counts.device_ptr_mut(&stream);
            let (pair_offsets, _pair_offsets_guard) = self.pair_offsets.device_ptr_mut(&stream);
            let (pair_cursors, _pair_cursors_guard) = self.pair_cursors.device_ptr_mut(&stream);
            let (sorted_route_ids, _sorted_route_ids_guard) =
                self.sorted_route_ids.device_ptr_mut(&stream);
            let (block_pair_ids, _block_pair_ids_guard) =
                self.block_pair_ids.device_ptr_mut(&stream);
            let (num_active_routes, _num_active_routes_guard) =
                self.num_active_routes.device_ptr_mut(&stream);
            let (num_padded_routes, _num_padded_routes_guard) =
                self.num_padded_routes.device_ptr_mut(&stream);
            let scan_workspace_bytes = self.scan_workspace.len();
            let (scan_workspace, _scan_workspace_guard) =
                self.scan_workspace.device_ptr_mut(&stream);
            let status = moe_cuda_ffi::launch_routed_lora_build_metadata(
                token_adapter_slots as *const u32,
                topk_expert_ids as *const u32,
                route_pair_ids as *mut u32,
                pair_counts as *mut u32,
                pair_offsets as *mut u32,
                pair_cursors as *mut u32,
                sorted_route_ids as *mut u32,
                block_pair_ids as *mut u32,
                num_active_routes as *mut u32,
                num_padded_routes as *mut u32,
                self.layout.num_tokens() as i32,
                self.layout.top_k() as i32,
                self.layout.num_experts() as i32,
                self.layout.num_adapter_slots() as i32,
                self.layout.block_size() as i32,
                self.layout.max_padded_routes() as i32,
                self.layout.max_blocks() as i32,
                scan_workspace as *mut core::ffi::c_void,
                scan_workspace_bytes,
                stream.cu_stream(),
            );
            check_status(status, "metadata build")
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct RoutedLoraDirectLaunch {
        pub input: u64,
        pub output: u64,
        pub token_adapter_slots: u64,
        pub topk_expert_ids: u64,
        pub route_input_rows: u64,
        pub route_output_rows: u64,
        pub route_output_scales: u64,
        pub dtype: DType,
        pub metadata: RoutedLoraMetadataLayout,
        pub projection: RoutedLoraProjectionLayout,
        pub weight_slice_offset: usize,
        pub output_splits: Option<usize>,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct RoutedLoraGroupedLaunch {
        pub input: u64,
        pub hidden: u64,
        pub output: u64,
        pub route_input_rows: u64,
        pub route_output_rows: u64,
        pub route_output_scales: u64,
        pub dtype: DType,
        pub projection: RoutedLoraProjectionLayout,
        pub weight_slice_offset: usize,
    }

    fn check_launch(
        metadata: RoutedLoraMetadataLayout,
        projection: RoutedLoraProjectionLayout,
        weights: &RoutedLoraCudaWeightTable,
        weight_slice_offset: usize,
    ) -> Result<()> {
        if weights.num_adapter_slots != metadata.num_adapter_slots()
            || weight_slice_offset
                .checked_add(projection.num_slices())
                .is_none_or(|end| end > weights.num_slices)
        {
            candle_core::bail!("routed LoRA launch and weight table shape mismatch");
        }
        weights.validate_projection_rank(
            weight_slice_offset,
            projection.num_slices(),
            projection.max_rank(),
        )?;
        Ok(())
    }

    fn check_status(status: i32, operation: &str) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            candle_core::bail!("routed LoRA CUDA {operation} failed with CUDA error {status}")
        }
    }

    fn optional_u32(pointer: u64) -> *const u32 {
        pointer as *const u32
    }

    fn optional_f32(pointer: u64) -> *const f32 {
        pointer as *const f32
    }

    /// Launches the fused small-route kernel and accumulates into `output`.
    ///
    /// # Safety
    /// All raw pointers must reference allocations on `weights`' device, have the shapes described
    /// by the layouts, and remain alive until work on the current stream completes. Input and output
    /// regions must not overlap. Explicit output rows must be injective, and every output row and
    /// slice region must be disjoint. The pointed-to adapter weights must outlive the descriptor
    /// table and every launch that consumes it.
    pub unsafe fn launch_routed_lora_direct(
        weights: &RoutedLoraCudaWeightTable,
        launch: RoutedLoraDirectLaunch,
    ) -> Result<()> {
        check_launch(
            launch.metadata,
            launch.projection,
            weights,
            launch.weight_slice_offset,
        )?;
        if launch.token_adapter_slots == 0 && launch.metadata.num_adapter_slots() != 1 {
            candle_core::bail!(
                "null routed LoRA token slots require a single adapter descriptor slot"
            );
        }
        let output_splits = launch
            .output_splits
            .unwrap_or_else(|| weights.direct_output_splits(launch.metadata, launch.projection));
        if output_splits == 0 || output_splits > u16::MAX as usize {
            candle_core::bail!("invalid routed LoRA direct output split count");
        }
        let stream = weights.device.cuda_stream();
        let (weight_ptr, _weight_guard) = weights.descriptors.device_ptr(&stream);
        let descriptor_offset = launch
            .weight_slice_offset
            .checked_mul(weights.num_adapter_slots)
            .and_then(|offset| offset.checked_mul(std::mem::size_of::<RoutedLoraAdapterWeight>()))
            .ok_or_else(|| candle_core::Error::msg("routed LoRA descriptor offset overflow"))?;
        let weight_ptr = weight_ptr + descriptor_offset as u64;
        let args = (
            launch.input,
            launch.output,
            weight_ptr,
            launch.token_adapter_slots,
            launch.topk_expert_ids,
            launch.route_input_rows,
            launch.route_output_rows,
            launch.route_output_scales,
            launch.metadata.num_tokens() as i32,
            launch.metadata.top_k() as i32,
            launch.metadata.num_experts() as i32,
            launch.metadata.num_adapter_slots() as i32,
            launch.projection.num_slices() as i32,
            launch.projection.input_features() as i32,
            launch.projection.output_features() as i32,
            launch.projection.output_row_stride() as i32,
            launch.projection.output_slice_stride() as i32,
            launch.projection.max_rank() as i32,
            launch.projection.input_mode() as i32,
            output_splits as i32,
            stream.cu_stream(),
        );
        let status = match launch.dtype {
            DType::F32 => moe_cuda_ffi::launch_routed_lora_direct_f32(
                args.0 as *const f32,
                args.1 as *mut f32,
                args.2 as *const RoutedLoraAdapterWeight,
                args.3 as *const u32,
                args.4 as *const u32,
                optional_u32(args.5),
                optional_u32(args.6),
                optional_f32(args.7),
                args.8,
                args.9,
                args.10,
                args.11,
                args.12,
                args.13,
                args.14,
                args.15,
                args.16,
                args.17,
                args.18,
                args.19,
                args.20,
            ),
            DType::F16 => moe_cuda_ffi::launch_routed_lora_direct_f16(
                args.0 as *const half::f16,
                args.1 as *mut half::f16,
                args.2 as *const RoutedLoraAdapterWeight,
                args.3 as *const u32,
                args.4 as *const u32,
                optional_u32(args.5),
                optional_u32(args.6),
                optional_f32(args.7),
                args.8,
                args.9,
                args.10,
                args.11,
                args.12,
                args.13,
                args.14,
                args.15,
                args.16,
                args.17,
                args.18,
                args.19,
                args.20,
            ),
            DType::BF16 => moe_cuda_ffi::launch_routed_lora_direct_bf16(
                args.0 as *const half::bf16,
                args.1 as *mut half::bf16,
                args.2 as *const RoutedLoraAdapterWeight,
                args.3 as *const u32,
                args.4 as *const u32,
                optional_u32(args.5),
                optional_u32(args.6),
                optional_f32(args.7),
                args.8,
                args.9,
                args.10,
                args.11,
                args.12,
                args.13,
                args.14,
                args.15,
                args.16,
                args.17,
                args.18,
                args.19,
                args.20,
            ),
            dtype => candle_core::bail!("routed LoRA CUDA does not support {dtype:?}"),
        };
        check_status(status, "direct launch")
    }

    /// Launches one grouped shrink and one grouped expand across every active adapter/expert pair.
    ///
    /// # Safety
    /// The raw pointers follow the same requirements as [`launch_routed_lora_direct`]. `hidden`
    /// must contain at least `metadata.hidden_elements(num_slices, max_rank)` elements.
    pub unsafe fn launch_routed_lora_grouped(
        metadata: &RoutedLoraCudaMetadata,
        weights: &RoutedLoraCudaWeightTable,
        launch: RoutedLoraGroupedLaunch,
    ) -> Result<()> {
        check_launch(
            metadata.layout,
            launch.projection,
            weights,
            launch.weight_slice_offset,
        )?;
        let stream = metadata.device.cuda_stream();
        let (weight_ptr, _weight_guard) = weights.descriptors.device_ptr(&stream);
        let descriptor_offset = launch
            .weight_slice_offset
            .checked_mul(weights.num_adapter_slots)
            .and_then(|offset| offset.checked_mul(std::mem::size_of::<RoutedLoraAdapterWeight>()))
            .ok_or_else(|| candle_core::Error::msg("routed LoRA descriptor offset overflow"))?;
        let weight_ptr = weight_ptr + descriptor_offset as u64;
        let (sorted_route_ids, _sorted_route_ids_guard) =
            metadata.sorted_route_ids.device_ptr(&stream);
        let (block_pair_ids, _block_pair_ids_guard) = metadata.block_pair_ids.device_ptr(&stream);
        if weights.supports_wmma(launch.projection)
            && matches!(launch.dtype, DType::F16 | DType::BF16)
        {
            let output_splits = weights.wmma_output_splits(metadata.layout, launch.projection);
            let status = match launch.dtype {
                DType::F16 => moe_cuda_ffi::launch_routed_lora_grouped_wmma_f16(
                    launch.input as *const half::f16,
                    launch.output as *mut half::f16,
                    weight_ptr as *const RoutedLoraAdapterWeight,
                    sorted_route_ids as *const u32,
                    block_pair_ids as *const u32,
                    optional_u32(launch.route_input_rows),
                    optional_u32(launch.route_output_rows),
                    optional_f32(launch.route_output_scales),
                    metadata.layout.num_routes() as i32,
                    metadata.layout.max_blocks() as i32,
                    metadata.layout.top_k() as i32,
                    metadata.layout.num_experts() as i32,
                    metadata.layout.num_adapter_slots() as i32,
                    launch.projection.num_slices() as i32,
                    launch.projection.input_features() as i32,
                    launch.projection.output_features() as i32,
                    launch.projection.output_row_stride() as i32,
                    launch.projection.output_slice_stride() as i32,
                    launch.projection.input_mode() as i32,
                    output_splits as i32,
                    stream.cu_stream(),
                ),
                DType::BF16 => moe_cuda_ffi::launch_routed_lora_grouped_wmma_bf16(
                    launch.input as *const half::bf16,
                    launch.output as *mut half::bf16,
                    weight_ptr as *const RoutedLoraAdapterWeight,
                    sorted_route_ids as *const u32,
                    block_pair_ids as *const u32,
                    optional_u32(launch.route_input_rows),
                    optional_u32(launch.route_output_rows),
                    optional_f32(launch.route_output_scales),
                    metadata.layout.num_routes() as i32,
                    metadata.layout.max_blocks() as i32,
                    metadata.layout.top_k() as i32,
                    metadata.layout.num_experts() as i32,
                    metadata.layout.num_adapter_slots() as i32,
                    launch.projection.num_slices() as i32,
                    launch.projection.input_features() as i32,
                    launch.projection.output_features() as i32,
                    launch.projection.output_row_stride() as i32,
                    launch.projection.output_slice_stride() as i32,
                    launch.projection.input_mode() as i32,
                    output_splits as i32,
                    stream.cu_stream(),
                ),
                _ => unreachable!(),
            };
            return check_status(status, "grouped WMMA");
        }
        let shrink_status = match launch.dtype {
            DType::F32 => moe_cuda_ffi::launch_routed_lora_grouped_shrink_f32(
                launch.input as *const f32,
                weight_ptr as *const RoutedLoraAdapterWeight,
                sorted_route_ids as *const u32,
                block_pair_ids as *const u32,
                optional_u32(launch.route_input_rows),
                launch.hidden as *mut f32,
                metadata.layout.num_routes() as i32,
                metadata.layout.max_blocks() as i32,
                metadata.layout.block_size() as i32,
                metadata.layout.top_k() as i32,
                metadata.layout.num_experts() as i32,
                metadata.layout.num_adapter_slots() as i32,
                launch.projection.num_slices() as i32,
                launch.projection.input_features() as i32,
                launch.projection.max_rank() as i32,
                launch.projection.input_mode() as i32,
                stream.cu_stream(),
            ),
            DType::F16 => moe_cuda_ffi::launch_routed_lora_grouped_shrink_f16(
                launch.input as *const half::f16,
                weight_ptr as *const RoutedLoraAdapterWeight,
                sorted_route_ids as *const u32,
                block_pair_ids as *const u32,
                optional_u32(launch.route_input_rows),
                launch.hidden as *mut f32,
                metadata.layout.num_routes() as i32,
                metadata.layout.max_blocks() as i32,
                metadata.layout.block_size() as i32,
                metadata.layout.top_k() as i32,
                metadata.layout.num_experts() as i32,
                metadata.layout.num_adapter_slots() as i32,
                launch.projection.num_slices() as i32,
                launch.projection.input_features() as i32,
                launch.projection.max_rank() as i32,
                launch.projection.input_mode() as i32,
                stream.cu_stream(),
            ),
            DType::BF16 => moe_cuda_ffi::launch_routed_lora_grouped_shrink_bf16(
                launch.input as *const half::bf16,
                weight_ptr as *const RoutedLoraAdapterWeight,
                sorted_route_ids as *const u32,
                block_pair_ids as *const u32,
                optional_u32(launch.route_input_rows),
                launch.hidden as *mut f32,
                metadata.layout.num_routes() as i32,
                metadata.layout.max_blocks() as i32,
                metadata.layout.block_size() as i32,
                metadata.layout.top_k() as i32,
                metadata.layout.num_experts() as i32,
                metadata.layout.num_adapter_slots() as i32,
                launch.projection.num_slices() as i32,
                launch.projection.input_features() as i32,
                launch.projection.max_rank() as i32,
                launch.projection.input_mode() as i32,
                stream.cu_stream(),
            ),
            dtype => candle_core::bail!("routed LoRA CUDA does not support {dtype:?}"),
        };
        check_status(shrink_status, "grouped shrink")?;

        let expand_status = match launch.dtype {
            DType::F32 => moe_cuda_ffi::launch_routed_lora_grouped_expand_f32(
                launch.hidden as *const f32,
                launch.output as *mut f32,
                weight_ptr as *const RoutedLoraAdapterWeight,
                sorted_route_ids as *const u32,
                block_pair_ids as *const u32,
                optional_u32(launch.route_output_rows),
                optional_f32(launch.route_output_scales),
                metadata.layout.num_routes() as i32,
                metadata.layout.max_blocks() as i32,
                metadata.layout.block_size() as i32,
                metadata.layout.num_experts() as i32,
                metadata.layout.num_adapter_slots() as i32,
                launch.projection.num_slices() as i32,
                launch.projection.output_features() as i32,
                launch.projection.output_row_stride() as i32,
                launch.projection.output_slice_stride() as i32,
                launch.projection.max_rank() as i32,
                stream.cu_stream(),
            ),
            DType::F16 => moe_cuda_ffi::launch_routed_lora_grouped_expand_f16(
                launch.hidden as *const f32,
                launch.output as *mut half::f16,
                weight_ptr as *const RoutedLoraAdapterWeight,
                sorted_route_ids as *const u32,
                block_pair_ids as *const u32,
                optional_u32(launch.route_output_rows),
                optional_f32(launch.route_output_scales),
                metadata.layout.num_routes() as i32,
                metadata.layout.max_blocks() as i32,
                metadata.layout.block_size() as i32,
                metadata.layout.num_experts() as i32,
                metadata.layout.num_adapter_slots() as i32,
                launch.projection.num_slices() as i32,
                launch.projection.output_features() as i32,
                launch.projection.output_row_stride() as i32,
                launch.projection.output_slice_stride() as i32,
                launch.projection.max_rank() as i32,
                stream.cu_stream(),
            ),
            DType::BF16 => moe_cuda_ffi::launch_routed_lora_grouped_expand_bf16(
                launch.hidden as *const f32,
                launch.output as *mut half::bf16,
                weight_ptr as *const RoutedLoraAdapterWeight,
                sorted_route_ids as *const u32,
                block_pair_ids as *const u32,
                optional_u32(launch.route_output_rows),
                optional_f32(launch.route_output_scales),
                metadata.layout.num_routes() as i32,
                metadata.layout.max_blocks() as i32,
                metadata.layout.block_size() as i32,
                metadata.layout.num_experts() as i32,
                metadata.layout.num_adapter_slots() as i32,
                launch.projection.num_slices() as i32,
                launch.projection.output_features() as i32,
                launch.projection.output_row_stride() as i32,
                launch.projection.output_slice_stride() as i32,
                launch.projection.max_rank() as i32,
                stream.cu_stream(),
            ),
            dtype => candle_core::bail!("routed LoRA CUDA does not support {dtype:?}"),
        };
        check_status(expand_status, "grouped expand")
    }

    pub use self::RoutedLoraCudaMetadata as Metadata;
    pub use self::RoutedLoraCudaWeightTable as WeightTable;
    pub use self::RoutedLoraDirectLaunch as DirectLaunch;
    pub use self::RoutedLoraGroupedLaunch as GroupedLaunch;
}

#[cfg(feature = "cuda")]
pub use cuda::{
    launch_routed_lora_direct, launch_routed_lora_grouped, DirectLaunch as RoutedLoraDirectLaunch,
    GroupedLaunch as RoutedLoraGroupedLaunch, Metadata as RoutedLoraCudaMetadata,
    WeightTable as RoutedLoraCudaWeightTable,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_abi_matches_cuda() {
        assert_eq!(std::mem::size_of::<RoutedLoraAdapterWeight>(), 40);
        assert_eq!(std::mem::align_of::<RoutedLoraAdapterWeight>(), 8);
        assert_eq!(std::mem::offset_of!(RoutedLoraAdapterWeight, rank), 24);
        assert_eq!(std::mem::offset_of!(RoutedLoraAdapterWeight, scale), 32);
    }

    #[test]
    fn metadata_capacity_covers_worst_case_pair_fragmentation() -> Result<()> {
        let layout = RoutedLoraMetadataLayout::new(64, 8, 128, 8)?;
        assert_eq!(layout.num_routes(), 512);
        assert_eq!(layout.num_pairs(), 1024);
        assert_eq!(layout.max_blocks(), 512);
        assert_eq!(layout.max_padded_routes(), 8192);
        let one_pair = RoutedLoraMetadataLayout::new(64, 8, 1, 1)?;
        assert_eq!(one_pair.max_blocks(), 32);
        assert_eq!(one_pair.max_padded_routes(), 512);
        Ok(())
    }

    #[test]
    fn projection_layout_supports_packed_gate_up_and_routed_down() -> Result<()> {
        let gate_up = RoutedLoraProjectionLayout::new(
            4096,
            14336,
            28672,
            14336,
            2,
            128,
            RoutedLoraInputMode::TokenRows,
        )?;
        assert_eq!(gate_up.num_slices(), 2);
        assert_eq!(gate_up.output_row_stride(), 28672);

        let down = RoutedLoraProjectionLayout::new(
            14336,
            4096,
            4096,
            0,
            1,
            128,
            RoutedLoraInputMode::RoutedRows,
        )?;
        assert_eq!(down.input_mode(), RoutedLoraInputMode::RoutedRows);
        Ok(())
    }

    #[test]
    fn descriptors_reject_invalid_rank_and_partial_pointers() {
        let partial = RoutedLoraAdapterWeight {
            a: 1,
            rank: 8,
            rank_stride: 8,
            ..RoutedLoraAdapterWeight::empty()
        };
        assert!(partial.validate().is_err());

        let bad_rank = RoutedLoraAdapterWeight {
            a: 1,
            b: 2,
            rank: 16,
            rank_stride: 8,
            scale: 1.0,
            ..RoutedLoraAdapterWeight::empty()
        };
        assert!(bad_rank.validate().is_err());
        assert!(RoutedLoraAdapterWeight::empty().validate().is_ok());
    }

    #[test]
    fn selected_descriptor_rank_validation_uses_only_requested_slices() -> Result<()> {
        let ranks = [64, 128, 32];
        validate_selected_descriptor_rank(&ranks, 0, 1, 64)?;
        validate_selected_descriptor_rank(&ranks, 2, 1, 32)?;
        assert!(validate_selected_descriptor_rank(&ranks, 0, 2, 64).is_err());
        assert!(validate_selected_descriptor_rank(&ranks, 1, 1, 127).is_err());
        Ok(())
    }

    #[test]
    fn selected_descriptor_rank_validation_rejects_invalid_ranges() {
        let ranks = [8, 16, 32];
        assert!(validate_selected_descriptor_rank(&ranks, 3, 1, 32).is_err());
        assert!(validate_selected_descriptor_rank(&ranks, usize::MAX, 2, 32).is_err());
    }

    #[test]
    fn output_split_heuristic_fills_small_grids_with_bounded_duplication() {
        assert_eq!(output_splits(120, 16, 56, 16), 15);
        assert_eq!(output_splits(120, 4, 56, 8), 8);
        assert_eq!(output_splits(120, 256, 56, 16), 1);
        assert_eq!(output_splits(120, 8, 2, 16), 2);
    }

    #[test]
    fn direct_small_batch_split_plan_matches_vllm_chunking() -> Result<()> {
        let topk_eight = RoutedLoraMetadataLayout::new(1, 8, 128, 1)?;
        let topk_two = RoutedLoraMetadataLayout::new(1, 2, 8, 1)?;
        for rank in [8, 16, 64] {
            let gate_up = RoutedLoraProjectionLayout::new(
                4096,
                14336,
                28672,
                14336,
                2,
                rank,
                RoutedLoraInputMode::TokenRows,
            )?;
            let down = RoutedLoraProjectionLayout::new(
                14336,
                4096,
                4096,
                0,
                1,
                rank,
                RoutedLoraInputMode::RoutedRows,
            )?;
            assert_eq!(
                direct_small_batch_output_splits(132, topk_eight, gate_up),
                16
            );
            assert_eq!(direct_small_batch_output_splits(132, topk_eight, down), 32);
            assert_eq!(
                direct_small_batch_output_splits(120, topk_eight, gate_up),
                14
            );
            assert_eq!(direct_small_batch_output_splits(120, topk_eight, down), 16);
            assert_eq!(direct_small_batch_output_splits(132, topk_two, gate_up), 56);
            assert_eq!(direct_small_batch_output_splits(132, topk_two, down), 32);
        }
        Ok(())
    }

    #[test]
    fn direct_rank128_uses_one_shot_recompute_budget() -> Result<()> {
        let metadata = RoutedLoraMetadataLayout::new(1, 8, 128, 1)?;
        let gate_up = RoutedLoraProjectionLayout::new(
            4096,
            14336,
            28672,
            14336,
            2,
            128,
            RoutedLoraInputMode::TokenRows,
        )?;
        let down = RoutedLoraProjectionLayout::new(
            14336,
            4096,
            4096,
            0,
            1,
            128,
            RoutedLoraInputMode::RoutedRows,
        )?;
        assert!(!prefer_small_batch_direct(metadata, 128));
        assert_eq!(direct_recompute_output_splits(132, metadata, gate_up), 7);
        assert_eq!(direct_recompute_output_splits(132, metadata, down), 2);
        Ok(())
    }

    #[test]
    fn grouped_wmma_split_plan_uses_the_launched_grid() -> Result<()> {
        let metadata = RoutedLoraMetadataLayout::new(32, 8, 128, 1)?;
        let gate_up = RoutedLoraProjectionLayout::new(
            4096,
            1536,
            3072,
            1536,
            2,
            64,
            RoutedLoraInputMode::TokenRows,
        )?;
        let down = RoutedLoraProjectionLayout::new(
            1536,
            4096,
            4096,
            0,
            1,
            64,
            RoutedLoraInputMode::RoutedRows,
        )?;
        assert_eq!(metadata.max_blocks(), 136);
        assert_eq!(grouped_wmma_output_splits(120, metadata, gate_up), 1);
        assert_eq!(grouped_wmma_output_splits(120, metadata, down), 2);
        Ok(())
    }

    #[test]
    fn output_recompute_cap_balances_occupancy_and_redundant_shrink() -> Result<()> {
        let gate_up = RoutedLoraProjectionLayout::new(
            4096,
            14336,
            28672,
            14336,
            2,
            64,
            RoutedLoraInputMode::TokenRows,
        )?;
        let down = RoutedLoraProjectionLayout::new(
            14336,
            4096,
            4096,
            0,
            1,
            64,
            RoutedLoraInputMode::RoutedRows,
        )?;
        assert_eq!(output_recompute_cap(gate_up), 7);
        assert_eq!(output_recompute_cap(down), 2);
        Ok(())
    }
}
