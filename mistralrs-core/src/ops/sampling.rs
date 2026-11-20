use super::*;

#[cfg(feature = "cuda")]
pub(super) const CUDA_TOPK_CHUNK_SIZE: usize = 2048;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOPK_MAX_EXACT_PACKED_VOCAB: usize = (1 << 24) + 1;
#[cfg(feature = "cuda")]
pub(super) const CUDA_TOPK_MAX_GRID_Y: usize = 65_535;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOPK_MAX_K: usize = 128;
#[cfg(feature = "cuda")]
pub(super) const CUDA_TOPK_MAX_STAGE2_CANDIDATES: usize = 47 * 1024;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_CATEGORICAL_PACKED_WIDTH: usize = 2;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOP1_PACKED_WIDTH: usize = 2;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOP1_INVALID_TOKEN: u32 = u32::MAX;
#[cfg(feature = "cuda")]
pub(super) const CUDA_ASYNC_TOKEN_RING_SLOTS: usize = 2;
#[cfg(feature = "cuda")]
pub(super) const CUDA_TOPK_SAMPLING_PARAM_WIDTH: usize = 5;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_DFLASH_SELECTOR_MAX_K: usize = 128;
#[cfg(all(feature = "cuda", test))]
pub(super) const CUDA_DFLASH_SELECTOR_INVALID_TOKEN: u32 = u32::MAX;
#[cfg(feature = "cuda")]
pub(super) const CUDA_DFLASH_SELECTOR_F32: i32 = 0;
#[cfg(feature = "cuda")]
pub(super) const CUDA_DFLASH_SELECTOR_BF16: i32 = 1;

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_ranked_packed_max_k(vocab: usize) -> Option<usize> {
    if vocab == 0 || vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        return None;
    }
    let chunks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let workspace_bound = CUDA_TOPK_MAX_STAGE2_CANDIDATES.checked_div(chunks)?;
    let max_k = vocab.min(CUDA_TOPK_MAX_K).min(workspace_bound);
    (max_k > 0).then_some(max_k)
}

// ============================================================================
// Optimized parallel topk for CUDA
// Uses a dedicated kernel that's much faster than full sort for small k
// Single kernel call writes both values and indices - no post-processing needed
// ============================================================================

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
pub(super) fn cuda_topk(input: &Tensor, k: usize) -> Result<TopKOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    let input = final_logits_row(input)?;
    let dims = input.dims();
    let ncols = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".to_string()))?;
    let nrows = (input.elem_count() / ncols) as i32;
    let ncols_i32 = ncols as i32;
    let k_i32 = k as i32;

    // Output shapes
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = k;
    let out_elem_count = nrows as usize * k;

    let (storage, _layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::BF16(inp) => inp.device_ptr(&stream),
        CudaStorageSlice::F16(inp) => inp.device_ptr(&stream),
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("cuda_topk only supports BF16/F16/F32"),
    };
    let src_ptr = src_ptr as *const c_void;

    // Allocate both output buffers
    let mut indices_dst = unsafe { dev.alloc::<u32>(out_elem_count) }?;
    let (indices_ptr, indices_guard) = indices_dst.device_ptr_mut(&stream);

    let (values_tensor, indices_tensor) = match input.dtype() {
        DType::BF16 => {
            let mut values_dst = unsafe { dev.alloc::<half::bf16>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_bf16(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::BF16(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        DType::F16 => {
            let mut values_dst = unsafe { dev.alloc::<half::f16>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_f16(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F16(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        DType::F32 => {
            let mut values_dst = unsafe { dev.alloc::<f32>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_f32(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        dt => candle_core::bail!("cuda_topk unsupported dtype: {:?}", dt),
    };

    Ok(TopKOutput {
        values: values_tensor,
        indices: indices_tensor,
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_topk_logits_f32_packed(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("cuda_topk_logits_f32_packed requires a positive finite temperature");
    }

    let input = input.contiguous()?;
    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_topk_logits_f32_packed requires F32 logits");
    }

    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("cuda_topk_logits_f32_packed got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!(
            "cuda_topk_logits_f32_packed k={} must be in [1, {}]",
            k,
            CUDA_TOPK_MAX_K
        );
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let stage2_candidates = nblocks * k;
    if stage2_candidates > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!(
            "cuda_topk_logits_f32_packed workspace too large: {} candidates",
            stage2_candidates
        );
    }

    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk_logits_f32_packed requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let (src_ptr, src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("cuda_topk_logits_f32_packed only supports F32"),
    };
    let src_ptr = unsafe { (src_ptr as *const f32).add(layout.start_offset()) };

    let workspace_elems = nblocks * k;
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut block_maxes = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(2 * k + 2) }?;

    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (block_maxes_ptr, block_maxes_guard) = block_maxes.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);

    unsafe {
        ffi::topk_large_f32_packed(
            src_ptr,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            block_maxes_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            packed_ptr as *mut f32,
            ncols as i32,
            k as i32,
            CUDA_TOPK_CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            stream_raw,
        );
    }

    drop(src_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(packed_guard);

    let packed_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };
    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_maxes),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
    ];

    Ok(TopKLogitsPackedOutput {
        packed: Tensor::from((
            candle_core::Storage::Cuda(packed_storage),
            Shape::from_dims(&[2 * k + 2]),
        )),
        k,
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTopKLogitsPackedWorkspace {
    pub(super) location: candle_core::DeviceLocation,
    pub(super) stream: Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    pub(super) capacity_rows: usize,
    pub(super) vocab: usize,
    pub(super) capacity_k: usize,
    pub(super) nblocks: usize,
    #[cfg(test)]
    pub(super) id: u64,
    pub(super) block_values: Tensor,
    pub(super) block_indices: Tensor,
    pub(super) block_maxes: Tensor,
    pub(super) block_sums: Tensor,
    pub(super) packed: Tensor,
}

#[cfg(all(feature = "cuda", test))]
pub(super) fn cuda_topk_logits_packed_workspace_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
impl CudaTopKLogitsPackedWorkspace {
    fn new(
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> Result<Self> {
        use candle_core::backend::BackendDevice;

        let capacity_rows = rows
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k row capacity overflow"))?;
        let capacity_k = k
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k width capacity overflow"))?;
        let workspace_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k workspace overflow"))?;
        let block_elems = capacity_rows
            .checked_mul(nblocks)
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k block workspace overflow"))?;
        let packed_width = capacity_k
            .checked_mul(2)
            .and_then(|width| width.checked_add(2))
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k packed width overflow"))?;
        let packed_elems = capacity_rows
            .checked_mul(packed_width)
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k packed workspace overflow"))?;
        let device = candle_core::Device::Cuda(dev.clone());
        Ok(Self {
            location: dev.location(),
            stream: dev.cuda_stream(),
            capacity_rows,
            vocab,
            capacity_k,
            nblocks,
            #[cfg(test)]
            id: cuda_topk_logits_packed_workspace_id(),
            block_values: Tensor::zeros(workspace_elems, DType::F32, &device)?,
            block_indices: Tensor::zeros(workspace_elems, DType::U32, &device)?,
            block_maxes: Tensor::zeros(block_elems, DType::F32, &device)?,
            block_sums: Tensor::zeros(block_elems, DType::F32, &device)?,
            packed: Tensor::zeros(packed_elems, DType::F32, &device)?,
        })
    }

    fn can_hold(
        &self,
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> bool {
        use candle_core::backend::BackendDevice;

        let stream = dev.cuda_stream();
        self.location == dev.location()
            && Arc::ptr_eq(self.stream.context(), stream.context())
            && self.stream.cu_stream() == stream.cu_stream()
            && self.capacity_rows >= rows
            && self.vocab == vocab
            && self.capacity_k >= k
            && self.nblocks == nblocks
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaRankedTopKPackedWorkspace {
    location: candle_core::DeviceLocation,
    stream: Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    capacity_rows: usize,
    vocab: usize,
    capacity_k: usize,
    nblocks: usize,
    block_values: Tensor,
    block_indices: Tensor,
    radix_state: Tensor,
    packed: Tensor,
}

#[cfg(feature = "cuda")]
impl CudaRankedTopKPackedWorkspace {
    fn new(
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> Result<Self> {
        use candle_core::backend::BackendDevice;

        let capacity_rows = rows
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k row capacity overflow"))?;
        let capacity_k = k
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k width capacity overflow"))?;
        let candidate_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k workspace overflow"))?;
        let radix_state_elems = capacity_rows
            .checked_mul(unsafe { ffi::topk_large_ranked_state_words_per_row() })
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k radix overflow"))?;
        let index_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k index overflow"))?;
        let packed_elems = capacity_rows
            .checked_mul(capacity_k)
            .and_then(|elems| elems.checked_mul(2))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k packed overflow"))?;
        let device = candle_core::Device::Cuda(dev.clone());
        Ok(Self {
            location: dev.location(),
            stream: dev.cuda_stream(),
            capacity_rows,
            vocab,
            capacity_k,
            nblocks,
            block_values: Tensor::zeros(candidate_elems, DType::F32, &device)?,
            block_indices: Tensor::zeros(index_elems, DType::U32, &device)?,
            radix_state: Tensor::zeros(radix_state_elems, DType::U32, &device)?,
            packed: Tensor::zeros(packed_elems, DType::F32, &device)?,
        })
    }

    fn can_hold(
        &self,
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> bool {
        use candle_core::backend::BackendDevice;

        let stream = dev.cuda_stream();
        self.location == dev.location()
            && Arc::ptr_eq(self.stream.context(), stream.context())
            && self.stream.cu_stream() == stream.cu_stream()
            && self.capacity_rows >= rows
            && self.vocab == vocab
            && self.capacity_k >= k
            && self.nblocks == nblocks
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_packed_batched(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
) -> Result<TopKLogitsPackedOutput> {
    let mut workspace = None;
    cuda_topk_logits_packed_batched_with_workspace(input, k, inverse_temperatures, &mut workspace)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_packed_batched_with_workspace(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
    cache: &mut Option<CudaTopKLogitsPackedWorkspace>,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_topk_logits_packed_batched";

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{OP} requires BF16, F16, or F32 logits");
    }
    if inverse_temperatures.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 inverse temperatures");
    }
    if !input.is_contiguous() || !inverse_temperatures.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    if !input.device().same_device(inverse_temperatures.device()) {
        candle_core::bail!("{OP} tensors must be on the same CUDA device");
    }

    let vocab =
        input.dims().last().copied().ok_or_else(|| {
            candle_core::Error::Msg(format!("{OP} requires logits with rank >= 1"))
        })?;
    if vocab == 0 {
        candle_core::bail!("{OP} got an empty vocabulary");
    }
    let batch = input.elem_count() / vocab;
    if batch == 0 {
        candle_core::bail!("{OP} got an empty batch");
    }
    if inverse_temperatures.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected inverse temperatures with shape [{batch}], got {:?}",
            inverse_temperatures.dims()
        );
    }
    let k = k.min(vocab);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {}]", CUDA_TOPK_MAX_K.min(vocab));
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if vocab > i32::MAX as usize {
        candle_core::bail!("{OP} vocabulary is too large: {vocab}");
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let candidates_per_row = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate count overflow")))?;
    if candidates_per_row > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!("{OP} workspace too large: {candidates_per_row} candidates per row");
    }
    let workspace_elems = batch
        .checked_mul(candidates_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate workspace overflow")))?;
    let block_elems = batch
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} block workspace overflow")))?;
    let packed_width = k
        .checked_mul(2)
        .and_then(|width| width.checked_add(2))
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    let packed_elems = batch
        .checked_mul(packed_width)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed output overflow")))?;

    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA inverse temperatures"),
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} only supports F32 inverse temperatures");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let needs_alloc = cache
        .as_ref()
        .is_none_or(|workspace| !workspace.can_hold(dev, batch, vocab, k, nblocks));
    if needs_alloc {
        *cache = Some(CudaTopKLogitsPackedWorkspace::new(
            dev, batch, vocab, k, nblocks,
        )?);
    }
    let workspace = cache
        .as_ref()
        .expect("CUDA top-k workspace was allocated above");
    let block_values = workspace.block_values.narrow(0, 0, workspace_elems)?;
    let block_indices = workspace.block_indices.narrow(0, 0, workspace_elems)?;
    let block_maxes = workspace.block_maxes.narrow(0, 0, block_elems)?;
    let block_sums = workspace.block_sums.narrow(0, 0, block_elems)?;
    let packed_dst = workspace.packed.narrow(0, 0, packed_elems)?;

    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr =
                unsafe { (ptr as *const $ty).add(input_layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &input_storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => candle_core::bail!("{OP} logits dtype mismatch"),
    };
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let (block_values_storage_guard, block_values_layout) = block_values.storage_and_layout();
    let candle_core::Storage::Cuda(block_values_storage) = &*block_values_storage_guard else {
        unreachable!("CUDA top-k workspace values are CUDA")
    };
    let CudaStorageSlice::F32(block_values_slice) = &block_values_storage.slice else {
        unreachable!("CUDA top-k workspace values are F32")
    };
    let (block_values_ptr, block_values_guard) = block_values_slice.device_ptr(&stream);
    let block_values_ptr =
        unsafe { (block_values_ptr as *mut f32).add(block_values_layout.start_offset()) };
    let (block_indices_storage_guard, block_indices_layout) = block_indices.storage_and_layout();
    let candle_core::Storage::Cuda(block_indices_storage) = &*block_indices_storage_guard else {
        unreachable!("CUDA top-k workspace indices are CUDA")
    };
    let CudaStorageSlice::U32(block_indices_slice) = &block_indices_storage.slice else {
        unreachable!("CUDA top-k workspace indices are U32")
    };
    let (block_indices_ptr, block_indices_guard) = block_indices_slice.device_ptr(&stream);
    let block_indices_ptr =
        unsafe { (block_indices_ptr as *mut u32).add(block_indices_layout.start_offset()) };
    let (block_maxes_storage, block_maxes_layout) = block_maxes.storage_and_layout();
    let candle_core::Storage::Cuda(block_maxes_storage) = &*block_maxes_storage else {
        unreachable!("CUDA top-k workspace maxima are CUDA")
    };
    let CudaStorageSlice::F32(block_maxes_slice) = &block_maxes_storage.slice else {
        unreachable!("CUDA top-k workspace maxima are F32")
    };
    let (block_maxes_ptr, block_maxes_guard) = block_maxes_slice.device_ptr(&stream);
    let block_maxes_ptr =
        unsafe { (block_maxes_ptr as *mut f32).add(block_maxes_layout.start_offset()) };
    let (block_sums_storage, block_sums_layout) = block_sums.storage_and_layout();
    let candle_core::Storage::Cuda(block_sums_storage) = &*block_sums_storage else {
        unreachable!("CUDA top-k workspace sums are CUDA")
    };
    let CudaStorageSlice::F32(block_sums_slice) = &block_sums_storage.slice else {
        unreachable!("CUDA top-k workspace sums are F32")
    };
    let (block_sums_ptr, block_sums_guard) = block_sums_slice.device_ptr(&stream);
    let block_sums_ptr =
        unsafe { (block_sums_ptr as *mut f32).add(block_sums_layout.start_offset()) };
    let (packed_storage_guard, packed_layout) = packed_dst.storage_and_layout();
    let candle_core::Storage::Cuda(packed_storage) = &*packed_storage_guard else {
        unreachable!("CUDA top-k packed workspace is CUDA")
    };
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        unreachable!("CUDA top-k packed workspace is F32")
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *mut f32).add(packed_layout.start_offset()) };
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };

    macro_rules! launch {
        ($kernel:path, $input:expr) => {{
            unsafe {
                $kernel(
                    $input,
                    temperature_ptr,
                    block_values_ptr,
                    block_indices_ptr,
                    block_maxes_ptr,
                    block_sums_ptr,
                    packed_ptr,
                    nrows_i32,
                    ncols_i32,
                    k_i32,
                    chunk_size_i32,
                    nblocks_i32,
                    stream.cu_stream() as i64,
                );
            }
        }};
    }
    match input.dtype() {
        DType::F32 => launch!(ffi::topk_large_f32_packed_batched, input_ptr.cast::<f32>()),
        DType::BF16 => launch!(ffi::topk_large_bf16_packed_batched, input_ptr),
        DType::F16 => launch!(ffi::topk_large_f16_packed_batched, input_ptr),
        _ => unreachable!(),
    }

    drop(input_guard);
    drop(temperature_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(packed_guard);
    Ok(TopKLogitsPackedOutput {
        packed: packed_dst.reshape((batch, packed_width))?,
        k,
        _workspace: vec![
            block_values.clone(),
            block_indices.clone(),
            block_maxes.clone(),
            block_sums.clone(),
        ],
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_ranked_packed_batched(
    input: &Tensor,
    k: usize,
) -> Result<RankedTopKPackedOutput> {
    let mut workspace = None;
    cuda_topk_ranked_packed_batched_with_workspace(input, k, &mut workspace)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_ranked_packed_batched_with_workspace(
    input: &Tensor,
    k: usize,
    cache: &mut Option<CudaRankedTopKPackedWorkspace>,
) -> Result<RankedTopKPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_topk_ranked_packed_batched";

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{OP} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    let vocab =
        input.dims().last().copied().ok_or_else(|| {
            candle_core::Error::Msg(format!("{OP} requires logits with rank >= 1"))
        })?;
    if vocab == 0 {
        candle_core::bail!("{OP} got an empty vocabulary");
    }
    let batch = input.elem_count() / vocab;
    if batch == 0 {
        candle_core::bail!("{OP} got an empty batch");
    }
    let k = k.min(vocab);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {}]", CUDA_TOPK_MAX_K.min(vocab));
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if vocab > i32::MAX as usize {
        candle_core::bail!("{OP} vocabulary is too large: {vocab}");
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let candidates_per_row = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate count overflow")))?;
    if candidates_per_row > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!("{OP} workspace too large: {candidates_per_row} candidates per row");
    }
    let workspace_elems = batch
        .checked_mul(candidates_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate workspace overflow")))?;
    let radix_state_words_per_row = unsafe { ffi::topk_large_ranked_state_words_per_row() };
    let radix_state_elems = batch
        .checked_mul(radix_state_words_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} radix workspace overflow")))?;
    let packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    let packed_elems = batch
        .checked_mul(packed_width)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed output overflow")))?;

    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let needs_alloc = cache
        .as_ref()
        .is_none_or(|workspace| !workspace.can_hold(dev, batch, vocab, k, nblocks));
    if needs_alloc {
        *cache = Some(CudaRankedTopKPackedWorkspace::new(
            dev, batch, vocab, k, nblocks,
        )?);
    }
    let workspace = cache
        .as_ref()
        .expect("CUDA ranked top-k workspace was allocated above");
    let block_values = workspace.block_values.narrow(0, 0, workspace_elems)?;
    let block_indices = workspace.block_indices.narrow(0, 0, workspace_elems)?;
    let radix_state = workspace.radix_state.narrow(0, 0, radix_state_elems)?;
    let packed_dst = workspace.packed.narrow(0, 0, packed_elems)?;

    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr =
                unsafe { (ptr as *const $ty).add(input_layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &input_storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => candle_core::bail!("{OP} logits dtype mismatch"),
    };
    let (block_values_storage_guard, block_values_layout) = block_values.storage_and_layout();
    let candle_core::Storage::Cuda(block_values_storage) = &*block_values_storage_guard else {
        unreachable!("CUDA ranked top-k workspace values are CUDA")
    };
    let CudaStorageSlice::F32(block_values_slice) = &block_values_storage.slice else {
        unreachable!("CUDA ranked top-k workspace values are F32")
    };
    let (block_values_ptr, block_values_guard) = block_values_slice.device_ptr(&stream);
    let block_values_ptr =
        unsafe { (block_values_ptr as *mut f32).add(block_values_layout.start_offset()) };
    let (block_indices_storage_guard, block_indices_layout) = block_indices.storage_and_layout();
    let candle_core::Storage::Cuda(block_indices_storage) = &*block_indices_storage_guard else {
        unreachable!("CUDA ranked top-k workspace indices are CUDA")
    };
    let CudaStorageSlice::U32(block_indices_slice) = &block_indices_storage.slice else {
        unreachable!("CUDA ranked top-k workspace indices are U32")
    };
    let (block_indices_ptr, block_indices_guard) = block_indices_slice.device_ptr(&stream);
    let block_indices_ptr =
        unsafe { (block_indices_ptr as *mut u32).add(block_indices_layout.start_offset()) };
    let (packed_storage_guard, packed_layout) = packed_dst.storage_and_layout();
    let candle_core::Storage::Cuda(packed_storage) = &*packed_storage_guard else {
        unreachable!("CUDA ranked top-k packed workspace is CUDA")
    };
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        unreachable!("CUDA ranked top-k packed workspace is F32")
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *mut f32).add(packed_layout.start_offset()) };
    let (radix_storage_guard, radix_layout) = radix_state.storage_and_layout();
    let candle_core::Storage::Cuda(radix_storage) = &*radix_storage_guard else {
        unreachable!("CUDA ranked top-k radix workspace is CUDA")
    };
    let CudaStorageSlice::U32(radix_slice) = &radix_storage.slice else {
        unreachable!("CUDA ranked top-k radix workspace is U32")
    };
    let (radix_ptr, radix_guard) = radix_slice.device_ptr(&stream);
    let radix_ptr = unsafe { (radix_ptr as *mut u32).add(radix_layout.start_offset()) };

    macro_rules! launch {
        ($kernel:path, $input:expr) => {{
            unsafe {
                $kernel(
                    $input,
                    block_values_ptr,
                    block_indices_ptr,
                    packed_ptr,
                    radix_ptr.cast::<c_void>(),
                    nrows_i32,
                    ncols_i32,
                    k_i32,
                    chunk_size_i32,
                    nblocks_i32,
                    stream.cu_stream() as i64,
                )
            }
        }};
    }
    let status = match input.dtype() {
        DType::F32 => launch!(
            ffi::topk_large_ranked_f32_packed_batched,
            input_ptr.cast::<f32>()
        ),
        DType::BF16 => launch!(ffi::topk_large_ranked_bf16_packed_batched, input_ptr),
        DType::F16 => launch!(ffi::topk_large_ranked_f16_packed_batched, input_ptr),
        _ => unreachable!(),
    };

    drop(input_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(packed_guard);
    drop(radix_guard);
    drop(block_values_storage_guard);
    drop(block_indices_storage_guard);
    drop(packed_storage_guard);
    drop(radix_storage_guard);
    if status != 0 {
        candle_core::bail!("{OP} CUDA launch failed with status {status}");
    }

    Ok(RankedTopKPackedOutput {
        packed: packed_dst.reshape((batch, packed_width))?,
        k,
        _workspace: vec![block_values, block_indices, radix_state],
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_f32_packed_batched(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
) -> Result<TopKLogitsPackedOutput> {
    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_topk_logits_f32_packed_batched requires F32 logits");
    }
    cuda_topk_logits_packed_batched(input, k, inverse_temperatures)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_dflash_greedy_select(
    topk: &RankedTopKPackedOutput,
    projected_hidden: &Tensor,
    predecessor_codebook: &Tensor,
    successor_codebook: &Tensor,
    anchors: &Tensor,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_dflash_greedy_select";
    let packed_topk = &topk.packed;
    let k = topk.k;

    let [rows, packed_width] = packed_topk.dims() else {
        candle_core::bail!("{OP} expected packed top-k with shape [batch * positions, 2 * k]");
    };
    let [hidden_rows, rank] = projected_hidden.dims() else {
        candle_core::bail!(
            "{OP} expected projected hidden states with shape [batch * positions, rank]"
        );
    };
    let [predecessor_vocab, predecessor_rank] = predecessor_codebook.dims() else {
        candle_core::bail!("{OP} expected predecessor codebook with shape [vocab, rank]");
    };
    let [successor_vocab, successor_rank] = successor_codebook.dims() else {
        candle_core::bail!("{OP} expected successor codebook with shape [vocab, rank]");
    };
    let [batch] = anchors.dims() else {
        candle_core::bail!("{OP} expected anchors with shape [batch]");
    };
    let (rows, packed_width, hidden_rows, rank) = (*rows, *packed_width, *hidden_rows, *rank);
    let (predecessor_vocab, predecessor_rank) = (*predecessor_vocab, *predecessor_rank);
    let (successor_vocab, successor_rank, batch) = (*successor_vocab, *successor_rank, *batch);

    if rows == 0 || batch == 0 || rank == 0 || predecessor_vocab == 0 {
        candle_core::bail!("{OP} does not support empty inputs");
    }
    if rows % batch != 0 {
        candle_core::bail!("{OP} row count {rows} is not divisible by batch size {batch}");
    }
    if hidden_rows != rows {
        candle_core::bail!("{OP} expected {rows} projected hidden rows, got {hidden_rows}");
    }
    if predecessor_vocab != successor_vocab || predecessor_rank != rank || successor_rank != rank {
        candle_core::bail!(
            "{OP} codebook shapes {:?} and {:?} do not match hidden rank {rank}",
            predecessor_codebook.dims(),
            successor_codebook.dims()
        );
    }
    let expected_packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    if packed_width != expected_packed_width {
        candle_core::bail!(
            "{OP} expected rank-only packed top-k width {expected_packed_width}, got {packed_width}"
        );
    }
    if k == 0 || k > CUDA_DFLASH_SELECTOR_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {CUDA_DFLASH_SELECTOR_MAX_K}]");
    }
    if predecessor_vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {predecessor_vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if packed_topk.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 packed top-k values");
    }
    if anchors.dtype() != DType::U32 {
        candle_core::bail!("{OP} requires U32 anchors");
    }
    for (name, tensor) in [
        ("projected hidden states", projected_hidden),
        ("predecessor codebook", predecessor_codebook),
        ("successor codebook", successor_codebook),
    ] {
        if !matches!(tensor.dtype(), DType::BF16 | DType::F32) {
            candle_core::bail!("{OP} requires BF16 or F32 {name}");
        }
    }
    for tensor in [
        packed_topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
    ] {
        if !tensor.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: OP });
        }
        if !packed_topk.device().same_device(tensor.device()) {
            candle_core::bail!("{OP} tensors must be on the same CUDA device");
        }
    }

    let positions = rows / batch;
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let positions_i32 = i32::try_from(positions).map_err(candle_core::Error::wrap)?;
    let rank_i32 = i32::try_from(rank).map_err(candle_core::Error::wrap)?;
    let vocab_i32 = i32::try_from(predecessor_vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let packed_width_i32 = i32::try_from(packed_width).map_err(candle_core::Error::wrap)?;

    let (packed_storage, packed_layout) = packed_topk.storage_and_layout();
    let packed_storage = match &*packed_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (hidden_storage, hidden_layout) = projected_hidden.storage_and_layout();
    let hidden_storage = match &*hidden_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (predecessor_storage, predecessor_layout) = predecessor_codebook.storage_and_layout();
    let predecessor_storage = match &*predecessor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (successor_storage, successor_layout) = successor_codebook.storage_and_layout();
    let successor_storage = match &*successor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (anchor_storage, anchor_layout) = anchors.storage_and_layout();
    let anchor_storage = match &*anchor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };

    let dev = packed_storage.device();
    let stream = dev.cuda_stream();
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        candle_core::bail!("{OP} packed top-k dtype mismatch");
    };
    let CudaStorageSlice::U32(anchor_slice) = &anchor_storage.slice else {
        candle_core::bail!("{OP} anchor dtype mismatch");
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
    let (anchor_ptr, anchor_guard) = anchor_slice.device_ptr(&stream);
    let anchor_ptr = unsafe { (anchor_ptr as *const u32).add(anchor_layout.start_offset()) };

    macro_rules! data_ptr {
        ($storage:expr, $layout:expr, $name:expr) => {{
            match &$storage.slice {
                CudaStorageSlice::F32(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr =
                        unsafe { (ptr as *const f32).add($layout.start_offset()) as *const c_void };
                    (ptr, CUDA_DFLASH_SELECTOR_F32, guard)
                }
                CudaStorageSlice::BF16(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr = unsafe {
                        (ptr as *const half::bf16).add($layout.start_offset()) as *const c_void
                    };
                    (ptr, CUDA_DFLASH_SELECTOR_BF16, guard)
                }
                _ => candle_core::bail!("{OP} {} dtype mismatch", $name),
            }
        }};
    }

    let (hidden_ptr, hidden_dtype, hidden_guard) =
        data_ptr!(hidden_storage, hidden_layout, "projected hidden states");
    let (predecessor_ptr, predecessor_dtype, predecessor_guard) = data_ptr!(
        predecessor_storage,
        predecessor_layout,
        "predecessor codebook"
    );
    let (successor_ptr, successor_dtype, successor_guard) =
        data_ptr!(successor_storage, successor_layout, "successor codebook");

    let mut selected = unsafe { dev.alloc::<u32>(rows) }?;
    let (selected_ptr, selected_guard) = selected.device_ptr_mut(&stream);
    unsafe {
        ffi::dflash_greedy_select(
            packed_ptr,
            hidden_ptr,
            predecessor_ptr,
            successor_ptr,
            anchor_ptr,
            selected_ptr as *mut u32,
            batch_i32,
            positions_i32,
            rank_i32,
            vocab_i32,
            k_i32,
            packed_width_i32,
            hidden_dtype,
            predecessor_dtype,
            successor_dtype,
            stream.cu_stream() as i64,
        );
    }

    drop(packed_guard);
    drop(hidden_guard);
    drop(predecessor_guard);
    drop(successor_guard);
    drop(anchor_guard);
    drop(selected_guard);

    Ok(Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(selected),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions]),
    )))
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_dflash_sample_select(
    input: DFlashSelectorSampleInput<'_>,
) -> Result<DFlashSelectorSampleOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_dflash_sample_select";
    let DFlashSelectorSampleInput {
        topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
        inverse_temperatures,
        uniforms,
    } = input;
    let packed_topk = &topk.packed;
    let k = topk.k;

    let [rows, packed_width] = packed_topk.dims() else {
        candle_core::bail!("{OP} expected packed top-k with shape [batch * positions, 2 * k]");
    };
    let [hidden_rows, rank] = projected_hidden.dims() else {
        candle_core::bail!(
            "{OP} expected projected hidden states with shape [batch * positions, rank]"
        );
    };
    let [predecessor_vocab, predecessor_rank] = predecessor_codebook.dims() else {
        candle_core::bail!("{OP} expected predecessor codebook with shape [vocab, rank]");
    };
    let [successor_vocab, successor_rank] = successor_codebook.dims() else {
        candle_core::bail!("{OP} expected successor codebook with shape [vocab, rank]");
    };
    let [batch] = anchors.dims() else {
        candle_core::bail!("{OP} expected anchors with shape [batch]");
    };
    let (rows, packed_width, hidden_rows, rank) = (*rows, *packed_width, *hidden_rows, *rank);
    let (predecessor_vocab, predecessor_rank) = (*predecessor_vocab, *predecessor_rank);
    let (successor_vocab, successor_rank, batch) = (*successor_vocab, *successor_rank, *batch);

    if rows == 0 || batch == 0 || rank == 0 || predecessor_vocab == 0 {
        candle_core::bail!("{OP} does not support empty inputs");
    }
    if rows % batch != 0 {
        candle_core::bail!("{OP} row count {rows} is not divisible by batch size {batch}");
    }
    if hidden_rows != rows {
        candle_core::bail!("{OP} expected {rows} projected hidden rows, got {hidden_rows}");
    }
    if predecessor_vocab != successor_vocab || predecessor_rank != rank || successor_rank != rank {
        candle_core::bail!(
            "{OP} codebook shapes {:?} and {:?} do not match hidden rank {rank}",
            predecessor_codebook.dims(),
            successor_codebook.dims()
        );
    }
    let expected_packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    if packed_width != expected_packed_width {
        candle_core::bail!(
            "{OP} expected rank-only packed top-k width {expected_packed_width}, got {packed_width}"
        );
    }
    if k == 0 || k > CUDA_DFLASH_SELECTOR_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {CUDA_DFLASH_SELECTOR_MAX_K}]");
    }
    if predecessor_vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {predecessor_vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    let positions = rows / batch;
    if inverse_temperatures.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected inverse temperatures with shape [{batch}], got {:?}",
            inverse_temperatures.dims()
        );
    }
    if uniforms.dims() != [batch, positions] {
        candle_core::bail!(
            "{OP} expected uniforms with shape [{batch}, {positions}], got {:?}",
            uniforms.dims()
        );
    }
    if packed_topk.dtype() != DType::F32
        || inverse_temperatures.dtype() != DType::F32
        || uniforms.dtype() != DType::F32
    {
        candle_core::bail!("{OP} requires F32 packed top-k, inverse temperatures, and uniforms");
    }
    if anchors.dtype() != DType::U32 {
        candle_core::bail!("{OP} requires U32 anchors");
    }
    for (name, tensor) in [
        ("projected hidden states", projected_hidden),
        ("predecessor codebook", predecessor_codebook),
        ("successor codebook", successor_codebook),
    ] {
        if !matches!(tensor.dtype(), DType::BF16 | DType::F32) {
            candle_core::bail!("{OP} requires BF16 or F32 {name}");
        }
    }
    for tensor in [
        packed_topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
        inverse_temperatures,
        uniforms,
    ] {
        if !tensor.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: OP });
        }
        if !packed_topk.device().same_device(tensor.device()) {
            candle_core::bail!("{OP} tensors must be on the same CUDA device");
        }
    }

    let sparse_elems = rows
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} output overflow")))?;
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let positions_i32 = i32::try_from(positions).map_err(candle_core::Error::wrap)?;
    let rank_i32 = i32::try_from(rank).map_err(candle_core::Error::wrap)?;
    let vocab_i32 = i32::try_from(predecessor_vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let packed_width_i32 = i32::try_from(packed_width).map_err(candle_core::Error::wrap)?;

    let (packed_storage, packed_layout) = packed_topk.storage_and_layout();
    let packed_storage = match &*packed_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (hidden_storage, hidden_layout) = projected_hidden.storage_and_layout();
    let hidden_storage = match &*hidden_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (predecessor_storage, predecessor_layout) = predecessor_codebook.storage_and_layout();
    let predecessor_storage = match &*predecessor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (successor_storage, successor_layout) = successor_codebook.storage_and_layout();
    let successor_storage = match &*successor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (anchor_storage, anchor_layout) = anchors.storage_and_layout();
    let anchor_storage = match &*anchor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (uniform_storage, uniform_layout) = uniforms.storage_and_layout();
    let uniform_storage = match &*uniform_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };

    let dev = packed_storage.device();
    let stream = dev.cuda_stream();
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        candle_core::bail!("{OP} packed top-k dtype mismatch");
    };
    let CudaStorageSlice::U32(anchor_slice) = &anchor_storage.slice else {
        candle_core::bail!("{OP} anchor dtype mismatch");
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} inverse temperature dtype mismatch");
    };
    let CudaStorageSlice::F32(uniform_slice) = &uniform_storage.slice else {
        candle_core::bail!("{OP} uniform dtype mismatch");
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
    let (anchor_ptr, anchor_guard) = anchor_slice.device_ptr(&stream);
    let anchor_ptr = unsafe { (anchor_ptr as *const u32).add(anchor_layout.start_offset()) };
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };
    let (uniform_ptr, uniform_guard) = uniform_slice.device_ptr(&stream);
    let uniform_ptr = unsafe { (uniform_ptr as *const f32).add(uniform_layout.start_offset()) };

    macro_rules! data_ptr {
        ($storage:expr, $layout:expr, $name:expr) => {{
            match &$storage.slice {
                CudaStorageSlice::F32(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr =
                        unsafe { (ptr as *const f32).add($layout.start_offset()) as *const c_void };
                    (ptr, CUDA_DFLASH_SELECTOR_F32, guard)
                }
                CudaStorageSlice::BF16(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr = unsafe {
                        (ptr as *const half::bf16).add($layout.start_offset()) as *const c_void
                    };
                    (ptr, CUDA_DFLASH_SELECTOR_BF16, guard)
                }
                _ => candle_core::bail!("{OP} {} dtype mismatch", $name),
            }
        }};
    }

    let (hidden_ptr, hidden_dtype, hidden_guard) =
        data_ptr!(hidden_storage, hidden_layout, "projected hidden states");
    let (predecessor_ptr, predecessor_dtype, predecessor_guard) = data_ptr!(
        predecessor_storage,
        predecessor_layout,
        "predecessor codebook"
    );
    let (successor_ptr, successor_dtype, successor_guard) =
        data_ptr!(successor_storage, successor_layout, "successor codebook");

    let mut selected = unsafe { dev.alloc::<u32>(rows) }?;
    let mut candidate_ids = unsafe { dev.alloc::<u32>(sparse_elems) }?;
    let mut candidate_probs = unsafe { dev.alloc::<f32>(sparse_elems) }?;
    let (selected_ptr, selected_guard) = selected.device_ptr_mut(&stream);
    let (candidate_ids_ptr, candidate_ids_guard) = candidate_ids.device_ptr_mut(&stream);
    let (candidate_probs_ptr, candidate_probs_guard) = candidate_probs.device_ptr_mut(&stream);
    crate::cuda::check_launch("dflash_sample_select", unsafe {
        ffi::dflash_sample_select(&crate::cuda::abi::DFlashSampleSelectParams {
            packed_topk: packed_ptr,
            projected_hidden: hidden_ptr,
            predecessor_codebook: predecessor_ptr,
            successor_codebook: successor_ptr,
            anchors: anchor_ptr,
            inverse_temperatures: temperature_ptr,
            uniforms: uniform_ptr,
            selected: selected_ptr as *mut u32,
            candidate_ids: candidate_ids_ptr as *mut u32,
            candidate_probs: candidate_probs_ptr as *mut f32,
            batch: batch_i32,
            positions: positions_i32,
            rank: rank_i32,
            vocab: vocab_i32,
            k: k_i32,
            packed_width: packed_width_i32,
            hidden_dtype,
            predecessor_dtype,
            successor_dtype,
            stream: stream.cu_stream() as i64,
        })
    })?;

    drop(packed_guard);
    drop(hidden_guard);
    drop(predecessor_guard);
    drop(successor_guard);
    drop(anchor_guard);
    drop(temperature_guard);
    drop(uniform_guard);
    drop(selected_guard);
    drop(candidate_ids_guard);
    drop(candidate_probs_guard);

    let tokens = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(selected),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions]),
    ));
    let candidate_ids = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(candidate_ids),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions, k]),
    ));
    let candidate_probs = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::F32(candidate_probs),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions, k]),
    ));

    Ok(DFlashSelectorSampleOutput {
        tokens,
        candidate_ids,
        candidate_probs,
    })
}

#[cfg(all(feature = "cuda", test))]
pub(crate) fn cuda_top1_logits_f32_packed_batched(
    input: &Tensor,
) -> Result<Top1LogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    const OP: &str = "cuda_top1_logits_f32_packed_batched";
    if input.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }

    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    let (batch, vocab) = (*batch, *vocab);
    if batch == 0 || vocab == 0 {
        candle_core::bail!("{OP} requires a non-empty batch and vocabulary");
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let workspace_elems = batch
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} workspace overflow")))?;
    let packed_elems = batch
        .checked_mul(CUDA_TOP1_PACKED_WIDTH)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} output overflow")))?;
    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let CudaStorageSlice::F32(input_slice) = &input_storage.slice else {
        candle_core::bail!("{OP} only supports F32 logits");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(packed_elems) }?;

    let (input_ptr, input_guard) = input_slice.device_ptr(&stream);
    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);
    let input_ptr = unsafe { (input_ptr as *const f32).add(input_layout.start_offset()) };

    unsafe {
        ffi::top1_large_f32_packed_batched(
            input_ptr,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            packed_ptr as *mut f32,
            std::ptr::null_mut(),
            nrows_i32,
            ncols_i32,
            chunk_size_i32,
            nblocks_i32,
            stream.cu_stream() as i64,
        );
    }

    drop(input_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(packed_guard);

    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
    ];
    let packed_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };

    Ok(Top1LogitsPackedOutput {
        packed: Tensor::from((
            candle_core::Storage::Cuda(packed_storage),
            Shape::from_dims(&[batch, CUDA_TOP1_PACKED_WIDTH]),
        )),
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_categorical_logits_f32_packed_batched(
    input: &Tensor,
    inverse_temperatures: &Tensor,
    uniforms: &Tensor,
) -> Result<CategoricalLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    const OP: &str = "cuda_categorical_logits_f32_packed_batched";
    if input.dtype() != DType::F32
        || inverse_temperatures.dtype() != DType::F32
        || uniforms.dtype() != DType::F32
    {
        candle_core::bail!("{OP} requires F32 tensors");
    }
    if !input.is_contiguous() || !inverse_temperatures.is_contiguous() || !uniforms.is_contiguous()
    {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    if !input.device().same_device(inverse_temperatures.device())
        || !input.device().same_device(uniforms.device())
    {
        candle_core::bail!("{OP} tensors must be on the same CUDA device");
    }

    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    let (batch, vocab) = (*batch, *vocab);
    if batch == 0 || vocab == 0 {
        candle_core::bail!("{OP} requires a non-empty batch and vocabulary");
    }
    if inverse_temperatures.dims() != [batch] || uniforms.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected sampling tensors with shape [{batch}], got {:?} and {:?}",
            inverse_temperatures.dims(),
            uniforms.dims()
        );
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let workspace_elems = batch
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} workspace overflow")))?;
    let packed_elems = batch
        .checked_mul(CUDA_CATEGORICAL_PACKED_WIDTH)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} output overflow")))?;
    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let CudaStorageSlice::F32(input_slice) = &input_storage.slice else {
        candle_core::bail!("{OP} only supports F32 logits");
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA inverse temperatures"),
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} only supports F32 inverse temperatures");
    };
    let (uniform_storage, uniform_layout) = uniforms.storage_and_layout();
    let uniform_storage = match &*uniform_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA uniforms"),
    };
    let CudaStorageSlice::F32(uniform_slice) = &uniform_storage.slice else {
        candle_core::bail!("{OP} only supports F32 uniforms");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(packed_elems) }?;

    let (input_ptr, input_guard) = input_slice.device_ptr(&stream);
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let (uniform_ptr, uniform_guard) = uniform_slice.device_ptr(&stream);
    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);
    let input_ptr = unsafe { (input_ptr as *const f32).add(input_layout.start_offset()) };
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };
    let uniform_ptr = unsafe { (uniform_ptr as *const f32).add(uniform_layout.start_offset()) };

    unsafe {
        ffi::categorical_large_f32_packed_batched(
            input_ptr,
            temperature_ptr,
            uniform_ptr,
            block_values_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            packed_ptr as *mut f32,
            nrows_i32,
            ncols_i32,
            chunk_size_i32,
            nblocks_i32,
            stream.cu_stream() as i64,
        );
    }

    drop(input_guard);
    drop(temperature_guard);
    drop(uniform_guard);
    drop(block_values_guard);
    drop(block_sums_guard);
    drop(packed_guard);

    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
    ];
    let packed_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };

    Ok(CategoricalLogitsPackedOutput {
        packed: Tensor::from((
            candle_core::Storage::Cuda(packed_storage),
            Shape::from_dims(&[batch, CUDA_CATEGORICAL_PACKED_WIDTH]),
        )),
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
pub struct CudaTop1LogitsWorkspace {
    pub(super) capacity_rows: usize,
    pub(super) ncols: usize,
    pub(super) nblocks: usize,
    pub(super) location: candle_core::DeviceLocation,
    pub(super) token_ring: CudaAsyncTokenRing,
    pub(super) slots: Vec<CudaTop1LogitsSlot>,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaTop1LogitsSlot {
    block_values: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    block_indices: candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
    packed: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    packed_host: candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<f32>,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaAsyncTokenRing {
    pub(super) id: u64,
    pub(super) next_slot: usize,
    pub(super) next_generation: u64,
    pub(super) slots: Vec<CudaAsyncTokenSlot>,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaAsyncTokenSlot {
    pub(super) owned_token_ids: Tensor,
    pub(super) token_ids_host: candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<u32>,
    pub(super) device_ready: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
    pub(super) host_complete: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
    pub(super) consumer_complete: candle_core::cuda_backend::cudarc::driver::CudaEvent,
    pub(super) reuse_ready: candle_core::cuda_backend::cudarc::driver::CudaEvent,
    pub(super) reuse_pending: bool,
    pub(super) pending: Option<CudaAsyncTokenPending>,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaAsyncTokenPending {
    pub(super) generation: u64,
    pub(super) nrows: usize,
    pub(super) producer_stream:
        std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    pub(super) consumer_stream:
        Option<std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>>,
    pub(super) token_ptr: u64,
    pub(super) token_end_ptr: u64,
    pub(super) token_released: bool,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaAsyncTokenReservation {
    pub(super) workspace_id: u64,
    pub(super) slot: usize,
    pub(super) generation: u64,
    pub(super) nrows: usize,
    pub(super) device_tokens: Tensor,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaAsyncTokenSubmission {
    pub(super) reservation: CudaAsyncTokenReservation,
    pub(super) device_ready: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
    pub(super) host_complete: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaPinnedHostPrefix<'a, T> {
    inner: &'a mut candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<T>,
    len: usize,
}

#[cfg(feature = "cuda")]
impl<T> candle_core::cuda_backend::cudarc::driver::HostSlice<T> for CudaPinnedHostPrefix<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> (
        &'a [T],
        candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    ) {
        let (slice, guard) = unsafe { self.inner.stream_synced_slice(stream) };
        (&slice[..self.len], guard)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> (
        &'a mut [T],
        candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    ) {
        let (slice, guard) = unsafe { self.inner.stream_synced_mut_slice(stream) };
        (&mut slice[..self.len], guard)
    }
}

#[cfg(feature = "cuda")]
impl CudaAsyncTokenSubmission {
    fn batch_size(&self) -> usize {
        self.reservation.nrows
    }

    fn wait(&self) -> Result<()> {
        self.host_complete
            .synchronize()
            .map_err(candle_core::Error::wrap)
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTop1Submission {
    pub(super) token: CudaAsyncTokenSubmission,
    pub(super) copy_packed: bool,
}

#[cfg(feature = "cuda")]
impl CudaTop1Submission {
    #[cfg(test)]
    pub(crate) fn device_tokens(&self) -> &Tensor {
        &self.token.reservation.device_tokens
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.token.batch_size()
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.token.wait()
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTop1Completion<'a> {
    token_ids: &'a [u32],
    packed: Option<&'a [f32]>,
}

#[cfg(feature = "cuda")]
impl<'a> CudaTop1Completion<'a> {
    pub(crate) fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }

    pub(crate) fn packed(&self) -> Option<&'a [f32]> {
        self.packed
    }
}

#[cfg(feature = "cuda")]
pub(super) struct CudaTop1SubmitOptions<'a> {
    nrows: usize,
    ncols: usize,
    token_ids_dst: Option<&'a Tensor>,
    copy_packed: bool,
    op: &'static str,
}

#[cfg(feature = "cuda")]
pub(super) fn final_logits_row(input: &Tensor) -> Result<Tensor> {
    let dims = input.dims();
    if dims.len() <= 1 {
        return input.contiguous();
    }
    let vocab = *dims.last().expect("rank checked above");
    if vocab == 0 {
        candle_core::bail!("logits last dimension is empty");
    }
    let rows = input.elem_count() / vocab;
    if rows == 0 {
        candle_core::bail!("logits tensor is empty");
    }
    input
        .reshape((rows, vocab))?
        .narrow(0, rows - 1, 1)?
        .reshape(vocab)?
        .contiguous()
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_async_token_ring_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
pub(super) fn same_cuda_stream(
    left: &candle_core::cuda_backend::cudarc::driver::CudaStream,
    right: &candle_core::cuda_backend::cudarc::driver::CudaStream,
) -> bool {
    std::sync::Arc::ptr_eq(left.context(), right.context()) && left.cu_stream() == right.cu_stream()
}

#[cfg(feature = "cuda")]
pub(super) fn new_cuda_async_token_slot(
    dev: &candle_core::CudaDevice,
    capacity_rows: usize,
) -> Result<CudaAsyncTokenSlot> {
    use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

    let stream = dev.cuda_stream();
    let context = stream.context();
    let mut token_ids = unsafe { dev.alloc::<u32>(capacity_rows) }?;
    let (_, token_ids_guard) = token_ids.device_ptr_mut(&stream);
    drop(token_ids_guard);
    let owned_token_ids = Tensor::from((
        candle_core::Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::U32(token_ids),
            device: dev.clone(),
        }),
        Shape::from_dims(&[capacity_rows, 1]),
    ));
    let event_flags = Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC);
    Ok(CudaAsyncTokenSlot {
        owned_token_ids,
        token_ids_host: unsafe { context.alloc_pinned::<u32>(capacity_rows) }
            .map_err(candle_core::Error::wrap)?,
        device_ready: std::sync::Arc::new(
            context
                .new_event(event_flags)
                .map_err(candle_core::Error::wrap)?,
        ),
        host_complete: std::sync::Arc::new(
            context
                .new_event(event_flags)
                .map_err(candle_core::Error::wrap)?,
        ),
        consumer_complete: context
            .new_event(event_flags)
            .map_err(candle_core::Error::wrap)?,
        reuse_ready: context
            .new_event(event_flags)
            .map_err(candle_core::Error::wrap)?,
        reuse_pending: false,
        pending: None,
    })
}

#[cfg(feature = "cuda")]
pub(super) fn new_cuda_async_token_ring(
    dev: &candle_core::CudaDevice,
    capacity_rows: usize,
) -> Result<CudaAsyncTokenRing> {
    let mut slots = Vec::with_capacity(CUDA_ASYNC_TOKEN_RING_SLOTS);
    for _ in 0..CUDA_ASYNC_TOKEN_RING_SLOTS {
        slots.push(new_cuda_async_token_slot(dev, capacity_rows)?);
    }
    Ok(CudaAsyncTokenRing {
        id: cuda_async_token_ring_id(),
        next_slot: 0,
        next_generation: 1,
        slots,
    })
}

#[cfg(feature = "cuda")]
impl CudaAsyncTokenRing {
    fn has_pending(&self) -> bool {
        self.slots.iter().any(|slot| slot.pending.is_some())
    }

    fn validate(&self, submission: &CudaAsyncTokenSubmission, op: &'static str) -> Result<()> {
        let reservation = &submission.reservation;
        if reservation.workspace_id != self.id {
            candle_core::bail!("{op} received a submission from a different workspace");
        }
        let slot = self.slots.get(reservation.slot).ok_or_else(|| {
            candle_core::Error::msg(format!("{op} received an invalid ring slot"))
        })?;
        let Some(pending) = &slot.pending else {
            candle_core::bail!("{op} received an inactive submission");
        };
        if pending.generation != reservation.generation {
            candle_core::bail!("{op} received a stale submission");
        }
        Ok(())
    }

    fn reserve(
        &mut self,
        input: &Tensor,
        token_ids_dst: Option<&Tensor>,
        nrows: usize,
        op: &'static str,
    ) -> Result<CudaAsyncTokenReservation> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;

        let stream = input.device().as_cuda_device()?.cuda_stream();
        let slot_index = (0..CUDA_ASYNC_TOKEN_RING_SLOTS)
            .map(|offset| (self.next_slot + offset) % CUDA_ASYNC_TOKEN_RING_SLOTS)
            .find(|&index| self.slots[index].pending.is_none())
            .ok_or_else(|| candle_core::Error::msg(format!("{op} submission ring is full")))?;
        self.next_slot = (slot_index + 1) % CUDA_ASYNC_TOKEN_RING_SLOTS;
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        let device_tokens = token_ids_dst
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| self.slots[slot_index].owned_token_ids.narrow(0, 0, nrows))?;
        let destination_capacity = match device_tokens.dims() {
            [capacity, 1] => *capacity,
            _ => 0,
        };
        if device_tokens.dtype() != DType::U32
            || destination_capacity < nrows
            || !device_tokens.is_contiguous()
            || !device_tokens.device().same_device(input.device())
        {
            candle_core::bail!(
                "{op} token destination must be contiguous CUDA U32 [capacity, 1] with capacity >= {nrows}"
            );
        }

        let (token_ptr, token_end_ptr) = {
            let (token_storage, token_layout) = device_tokens.storage_and_layout();
            let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
                unreachable!("token destination device was checked above")
            };
            let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
                unreachable!("token destination dtype was checked above")
            };
            let (token_ptr, token_guard) = token_slice.device_ptr(&stream);
            let token_ptr = unsafe { (token_ptr as *mut u32).add(token_layout.start_offset()) };
            let token_end_ptr = token_ptr as u64 + (nrows * std::mem::size_of::<u32>()) as u64;
            drop(token_guard);
            (token_ptr, token_end_ptr)
        };
        for other in &self.slots {
            let Some(pending) = &other.pending else {
                continue;
            };
            if pending.token_ptr >= token_end_ptr || token_ptr as u64 >= pending.token_end_ptr {
                continue;
            }
            if !pending.token_released {
                candle_core::bail!(
                    "{op} token destination is already leased by another submission"
                );
            }
            if other.reuse_pending {
                stream
                    .wait(&other.reuse_ready)
                    .map_err(candle_core::Error::wrap)?;
            }
        }
        let slot = &mut self.slots[slot_index];
        if slot.reuse_pending {
            stream
                .wait(&slot.reuse_ready)
                .map_err(candle_core::Error::wrap)?;
            slot.reuse_pending = false;
        }
        slot.pending = Some(CudaAsyncTokenPending {
            generation,
            nrows,
            producer_stream: stream,
            consumer_stream: None,
            token_ptr: token_ptr as u64,
            token_end_ptr,
            token_released: false,
        });
        Ok(CudaAsyncTokenReservation {
            workspace_id: self.id,
            slot: slot_index,
            generation,
            nrows,
            device_tokens,
        })
    }

    fn abort(&mut self, reservation: &CudaAsyncTokenReservation) {
        if reservation.workspace_id == self.id {
            if let Some(slot) = self.slots.get_mut(reservation.slot) {
                if slot
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.generation == reservation.generation)
                {
                    slot.pending = None;
                }
            }
        }
    }

    fn finish_launch(
        &mut self,
        reservation: CudaAsyncTokenReservation,
    ) -> Result<CudaAsyncTokenSubmission> {
        use candle_core::cuda_backend::CudaStorageSlice;

        let launch = (|| {
            let slot = &mut self.slots[reservation.slot];
            let pending = slot
                .pending
                .as_ref()
                .filter(|pending| pending.generation == reservation.generation)
                .expect("token reservation remains active through its producer launch");
            let stream = pending.producer_stream.clone();
            let dev = reservation.device_tokens.device().as_cuda_device()?;
            slot.device_ready
                .record(&stream)
                .map_err(candle_core::Error::wrap)?;
            let (token_storage, token_layout) = reservation.device_tokens.storage_and_layout();
            let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
                unreachable!("reserved token destination is CUDA")
            };
            let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
                unreachable!("reserved token destination is U32")
            };
            let token_copy = token_slice.slice(
                token_layout.start_offset()
                    ..token_layout
                        .start_offset()
                        .saturating_add(reservation.nrows),
            );
            let mut host_prefix = CudaPinnedHostPrefix {
                inner: &mut slot.token_ids_host,
                len: reservation.nrows,
            };
            dev.memcpy_dtoh(&token_copy, &mut host_prefix)?;
            slot.host_complete
                .record(&stream)
                .map_err(candle_core::Error::wrap)?;
            Result::<()>::Ok(())
        })();
        if let Err(error) = launch {
            self.abort(&reservation);
            return Err(error);
        }
        let slot = &self.slots[reservation.slot];
        Ok(CudaAsyncTokenSubmission {
            reservation,
            device_ready: slot.device_ready.clone(),
            host_complete: slot.host_complete.clone(),
        })
    }

    fn wait_on(
        &mut self,
        submission: &CudaAsyncTokenSubmission,
        consumer_stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
        op: &'static str,
    ) -> Result<()> {
        self.validate(submission, op)?;
        let slot = &mut self.slots[submission.reservation.slot];
        let pending = slot.pending.as_mut().expect("submission validated above");
        if same_cuda_stream(&pending.producer_stream, consumer_stream) {
            slot.reuse_ready
                .record(&pending.producer_stream)
                .map_err(candle_core::Error::wrap)?;
            slot.reuse_pending = true;
            pending.token_released = true;
            return Ok(());
        }
        if let Some(current) = &pending.consumer_stream {
            if same_cuda_stream(current, consumer_stream) {
                return Ok(());
            }
            candle_core::bail!("{op} only supports one cross-stream consumer per submission");
        }
        consumer_stream
            .wait(&submission.device_ready)
            .map_err(candle_core::Error::wrap)?;
        pending.consumer_stream = Some(consumer_stream.clone());
        Ok(())
    }

    fn release_after(
        &mut self,
        submission: &CudaAsyncTokenSubmission,
        consumer_stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
        op: &'static str,
    ) -> Result<()> {
        self.validate(submission, op)?;
        let slot = &mut self.slots[submission.reservation.slot];
        let pending = slot.pending.as_mut().expect("submission validated above");
        if same_cuda_stream(&pending.producer_stream, consumer_stream) {
            return Ok(());
        }
        let Some(current) = &pending.consumer_stream else {
            candle_core::bail!("{op} requires wait_on before release_after");
        };
        if !same_cuda_stream(current, consumer_stream) {
            candle_core::bail!("{op} consumer stream does not match wait_on");
        }
        slot.consumer_complete
            .record(consumer_stream)
            .map_err(candle_core::Error::wrap)?;
        pending
            .producer_stream
            .wait(&slot.consumer_complete)
            .map_err(candle_core::Error::wrap)?;
        slot.reuse_ready
            .record(&pending.producer_stream)
            .map_err(candle_core::Error::wrap)?;
        slot.reuse_pending = true;
        pending.consumer_stream = None;
        pending.token_released = true;
        Ok(())
    }

    fn complete<'a>(
        &'a mut self,
        submission: &CudaAsyncTokenSubmission,
        op: &'static str,
    ) -> Result<&'a [u32]> {
        self.validate(submission, op)?;
        let slot = &mut self.slots[submission.reservation.slot];
        let pending = slot.pending.as_ref().expect("submission validated above");
        if pending.consumer_stream.is_some() {
            candle_core::bail!("{op} requires release_after for the cross-stream consumer");
        }
        slot.host_complete
            .synchronize()
            .map_err(candle_core::Error::wrap)?;
        let nrows = pending.nrows;
        slot.pending = None;
        Ok(&slot
            .token_ids_host
            .as_slice()
            .map_err(candle_core::Error::wrap)?[..nrows])
    }

    fn cancel(&mut self, submission: &CudaAsyncTokenSubmission, op: &'static str) -> Result<()> {
        self.validate(submission, op)?;
        let slot = &mut self.slots[submission.reservation.slot];
        let pending = slot.pending.as_mut().expect("submission validated above");
        if let Some(consumer_stream) = pending.consumer_stream.take() {
            slot.consumer_complete
                .record(&consumer_stream)
                .map_err(candle_core::Error::wrap)?;
            pending
                .producer_stream
                .wait(&slot.consumer_complete)
                .map_err(candle_core::Error::wrap)?;
            slot.reuse_ready
                .record(&pending.producer_stream)
                .map_err(candle_core::Error::wrap)?;
            slot.reuse_pending = true;
        }
        slot.host_complete
            .synchronize()
            .map_err(candle_core::Error::wrap)?;
        slot.pending = None;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub(super) fn new_cuda_top1_slot(
    dev: &candle_core::CudaDevice,
    workspace_elems: usize,
    packed_elems: usize,
) -> Result<CudaTop1LogitsSlot> {
    let stream = dev.cuda_stream();
    let context = stream.context();

    Ok(CudaTop1LogitsSlot {
        block_values: unsafe { dev.alloc::<f32>(workspace_elems) }?,
        block_indices: unsafe { dev.alloc::<u32>(workspace_elems) }?,
        packed: unsafe { dev.alloc::<f32>(packed_elems) }?,
        packed_host: unsafe { context.alloc_pinned::<f32>(packed_elems) }
            .map_err(candle_core::Error::wrap)?,
    })
}

#[cfg(feature = "cuda")]
pub(super) fn new_cuda_top1_workspace(
    dev: &candle_core::CudaDevice,
    nrows: usize,
    ncols: usize,
    nblocks: usize,
) -> Result<CudaTop1LogitsWorkspace> {
    use candle_core::backend::BackendDevice;

    let workspace_elems = nrows
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg("CUDA top-1 workspace overflow".to_string()))?;
    let packed_elems = nrows
        .checked_mul(CUDA_TOP1_PACKED_WIDTH)
        .ok_or_else(|| candle_core::Error::Msg("CUDA top-1 output overflow".to_string()))?;
    let mut slots = Vec::with_capacity(CUDA_ASYNC_TOKEN_RING_SLOTS);
    for _ in 0..CUDA_ASYNC_TOKEN_RING_SLOTS {
        slots.push(new_cuda_top1_slot(dev, workspace_elems, packed_elems)?);
    }
    Ok(CudaTop1LogitsWorkspace {
        capacity_rows: nrows,
        ncols,
        nblocks,
        location: dev.location(),
        token_ring: new_cuda_async_token_ring(dev, nrows)?,
        slots,
    })
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_top1_submission<'a>(
    workspace: &'a CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    op: &'static str,
) -> Result<&'a CudaTop1LogitsSlot> {
    workspace.token_ring.validate(&submission.token, op)?;
    let slot_index = submission.token.reservation.slot;
    let slot = workspace
        .slots
        .get(slot_index)
        .ok_or_else(|| candle_core::Error::Msg(format!("{op} received an invalid ring slot")))?;
    Ok(slot)
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_top1_logits_submit_inner(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
    options: CudaTop1SubmitOptions<'_>,
) -> Result<CudaTop1Submission> {
    use candle_core::backend::{BackendDevice, BackendStorage};
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    let CudaTop1SubmitOptions {
        nrows,
        ncols,
        token_ids_dst,
        copy_packed,
        op,
    } = options;

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{op} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op });
    }
    if nrows == 0 || ncols == 0 {
        candle_core::bail!("{op} requires non-empty logits");
    }
    if ncols > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{op} vocabulary size {ncols} cannot be represented exactly by packed F32 indices"
        );
    }
    if nrows > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{op} batch is too large for a 2D CUDA launch: {nrows}");
    }
    let expected_elems = nrows
        .checked_mul(ncols)
        .ok_or_else(|| candle_core::Error::Msg(format!("{op} input size overflow")))?;
    if input.elem_count() != expected_elems {
        candle_core::bail!(
            "{op} expected {nrows} rows of {ncols} logits, got {} elements",
            input.elem_count()
        );
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("{op} requires CUDA logits"),
    };
    let dev = storage.device();
    let location = dev.location();
    let needs_alloc = cache.as_ref().is_none_or(|workspace| {
        workspace.capacity_rows < nrows
            || workspace.ncols != ncols
            || workspace.nblocks != nblocks
            || workspace.location != location
    });
    if needs_alloc {
        if cache
            .as_ref()
            .is_some_and(|workspace| workspace.token_ring.has_pending())
        {
            candle_core::bail!("{op} cannot resize while submissions are pending");
        }
        *cache = Some(new_cuda_top1_workspace(dev, nrows, ncols, nblocks)?);
    }

    let stream = dev.cuda_stream();
    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr = unsafe { (ptr as *const $ty).add(layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => unreachable!("logits dtype was validated above"),
    };
    let workspace = cache
        .as_mut()
        .expect("CUDA top-1 workspace was allocated above");
    let reservation = workspace
        .token_ring
        .reserve(input, token_ids_dst, nrows, op)?;
    let slot_index = reservation.slot;
    let device_tokens_storage = reservation.device_tokens.clone();
    let (token_storage, token_layout) = device_tokens_storage.storage_and_layout();
    let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
        unreachable!("reserved token destination is CUDA")
    };
    let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
        unreachable!("reserved token destination is U32")
    };
    let (token_ptr, token_guard) = token_slice.device_ptr(&stream);
    let token_ptr = unsafe { (token_ptr as *mut u32).add(token_layout.start_offset()) };
    let slot = &mut workspace.slots[slot_index];

    let result = (|| {
        let (block_values_ptr, block_values_guard) = slot.block_values.device_ptr_mut(&stream);
        let (block_indices_ptr, block_indices_guard) = slot.block_indices.device_ptr_mut(&stream);
        let (packed_ptr, packed_guard) = if copy_packed {
            let (packed_ptr, packed_guard) = slot.packed.device_ptr_mut(&stream);
            (packed_ptr as *mut f32, Some(packed_guard))
        } else {
            (std::ptr::null_mut(), None)
        };

        let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
        let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;
        let chunk_size_i32 =
            i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
        let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;
        macro_rules! launch {
            ($single:path, $batched:path, $ptr:expr) => {{
                unsafe {
                    if nrows == 1 {
                        $single(
                            $ptr,
                            block_values_ptr as *mut f32,
                            block_indices_ptr as *mut u32,
                            packed_ptr,
                            token_ptr,
                            ncols_i32,
                            chunk_size_i32,
                            nblocks_i32,
                            stream.cu_stream() as i64,
                        );
                    } else {
                        $batched(
                            $ptr,
                            block_values_ptr as *mut f32,
                            block_indices_ptr as *mut u32,
                            packed_ptr,
                            token_ptr,
                            nrows_i32,
                            ncols_i32,
                            chunk_size_i32,
                            nblocks_i32,
                            stream.cu_stream() as i64,
                        );
                    }
                }
            }};
        }
        match input.dtype() {
            DType::F32 => launch!(
                ffi::top1_large_f32_packed,
                ffi::top1_large_f32_packed_batched,
                input_ptr.cast::<f32>()
            ),
            DType::BF16 => launch!(
                ffi::top1_large_bf16_packed,
                ffi::top1_large_bf16_packed_batched,
                input_ptr
            ),
            DType::F16 => launch!(
                ffi::top1_large_f16_packed,
                ffi::top1_large_f16_packed_batched,
                input_ptr
            ),
            _ => unreachable!("logits dtype was validated above"),
        }

        drop(input_guard);
        drop(block_values_guard);
        drop(block_indices_guard);
        drop(packed_guard);
        drop(token_guard);
        if copy_packed {
            dev.memcpy_dtoh(&slot.packed, &mut slot.packed_host)?;
        }
        Result::<()>::Ok(())
    })();
    if let Err(error) = result {
        let _ = stream.synchronize();
        workspace.token_ring.abort(&reservation);
        return Err(error);
    }
    let token = match workspace.token_ring.finish_launch(reservation) {
        Ok(token) => token,
        Err(error) => {
            let _ = stream.synchronize();
            return Err(error);
        }
    };
    Ok(CudaTop1Submission { token, copy_packed })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_logits_submit_batched(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: None,
            copy_packed: false,
            op: OP,
        },
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_logits_submit_batched_packed(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched_packed";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: None,
            copy_packed: true,
            op: OP,
        },
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_logits_submit_batched_into(
    input: &Tensor,
    token_ids_dst: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched_into";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: Some(token_ids_dst),
            copy_packed: false,
            op: OP,
        },
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_device_tokens_wait_on(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    consumer_stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    const OP: &str = "cuda_top1_device_tokens_wait_on";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    workspace
        .token_ring
        .wait_on(&submission.token, consumer_stream, OP)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_device_tokens_release_after(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    consumer_stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    const OP: &str = "cuda_top1_device_tokens_release_after";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    workspace
        .token_ring
        .release_after(&submission.token, consumer_stream, OP)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_submission_complete<'a>(
    workspace: &'a mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
) -> Result<CudaTop1Completion<'a>> {
    const OP: &str = "cuda_top1_submission_complete";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    let slot_index = submission.token.reservation.slot;
    let nrows = submission.token.reservation.nrows;
    let token_ids = workspace.token_ring.complete(&submission.token, OP)?;
    let packed = if submission.copy_packed {
        Some(
            &workspace.slots[slot_index]
                .packed_host
                .as_slice()
                .map_err(candle_core::Error::wrap)?[..nrows * CUDA_TOP1_PACKED_WIDTH],
        )
    } else {
        None
    };
    Ok(CudaTop1Completion { token_ids, packed })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_submission_cancel(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
) -> Result<()> {
    const OP: &str = "cuda_top1_submission_cancel";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    workspace.token_ring.cancel(&submission.token, OP)
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_top1_logits_f32_packed_cached_inner<'a>(
    input: &Tensor,
    nrows: usize,
    ncols: usize,
    cache: &'a mut Option<CudaTop1LogitsWorkspace>,
    op: &'static str,
) -> Result<&'a [f32]> {
    let submission = cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows,
            ncols,
            token_ids_dst: None,
            copy_packed: true,
            op,
        },
    )?;
    let completion = cuda_top1_submission_complete(
        cache
            .as_mut()
            .expect("CUDA top-1 workspace was allocated during submission"),
        &submission,
    )?;
    Ok(completion
        .packed()
        .expect("packed output was requested during submission"))
}

#[cfg(feature = "cuda")]
pub fn cuda_top1_logits_f32_cached(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<[f32; CUDA_TOP1_PACKED_WIDTH]> {
    const OP: &str = "cuda_top1_logits_f32_cached";
    let input = final_logits_row(input)?;
    let ncols = input.elem_count();
    let packed = cuda_top1_logits_f32_packed_cached_inner(&input, 1, ncols, cache, OP)?;
    Ok([packed[0], packed[1]])
}

#[cfg(feature = "cuda")]
#[cfg(test)]
pub(crate) fn cuda_top1_logits_f32_packed_batched_cached(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<Vec<[f32; CUDA_TOP1_PACKED_WIDTH]>> {
    const OP: &str = "cuda_top1_logits_f32_packed_batched_cached";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    let (batch, vocab) = (*batch, *vocab);
    let packed = cuda_top1_logits_f32_packed_cached_inner(input, batch, vocab, cache, OP)?;
    Ok(packed
        .as_chunks::<CUDA_TOP1_PACKED_WIDTH>()
        .0
        .iter()
        .map(|row| [row[0], row[1]])
        .collect())
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(crate) struct CudaTopKSamplingParams {
    pub(crate) inverse_temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) min_p: f32,
    pub(crate) uniform: f32,
}

#[cfg(feature = "cuda")]
pub struct CudaTopKSamplingWorkspace {
    capacity_rows: usize,
    capacity_k: usize,
    vocab: usize,
    location: candle_core::DeviceLocation,
    ranked: Option<CudaRankedTopKPackedWorkspace>,
    token_ring: CudaAsyncTokenRing,
    slots: Vec<CudaTopKSamplingSlot>,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaTopKSamplingSlot {
    params: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    params_host: candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<f32>,
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTopKSamplingSubmission {
    pub(super) token: CudaAsyncTokenSubmission,
}

#[cfg(feature = "cuda")]
impl CudaTopKSamplingSubmission {
    pub(crate) fn batch_size(&self) -> usize {
        self.token.batch_size()
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.token.wait()
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTopKSamplingCompletion<'a> {
    token_ids: &'a [u32],
}

#[cfg(feature = "cuda")]
impl<'a> CudaTopKSamplingCompletion<'a> {
    pub(crate) fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }
}

#[cfg(feature = "cuda")]
pub(super) fn new_cuda_topk_sampling_slot(
    dev: &candle_core::CudaDevice,
    capacity_rows: usize,
) -> Result<CudaTopKSamplingSlot> {
    let stream = dev.cuda_stream();
    let context = stream.context();
    let param_elems = capacity_rows
        .checked_mul(CUDA_TOPK_SAMPLING_PARAM_WIDTH)
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling parameter overflow"))?;
    let mut params_host =
        unsafe { context.alloc_pinned::<f32>(param_elems) }.map_err(candle_core::Error::wrap)?;
    params_host
        .as_mut_slice()
        .map_err(candle_core::Error::wrap)?
        .fill(0.0);

    Ok(CudaTopKSamplingSlot {
        params: unsafe { dev.alloc::<f32>(param_elems) }?,
        params_host,
    })
}

#[cfg(feature = "cuda")]
pub(super) fn new_cuda_topk_sampling_workspace(
    dev: &candle_core::CudaDevice,
    rows: usize,
    vocab: usize,
    k: usize,
) -> Result<CudaTopKSamplingWorkspace> {
    use candle_core::backend::BackendDevice;

    let capacity_rows = rows
        .checked_next_power_of_two()
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling row capacity overflow"))?;
    let capacity_k = k
        .checked_next_power_of_two()
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling width capacity overflow"))?;
    let mut slots = Vec::with_capacity(CUDA_ASYNC_TOKEN_RING_SLOTS);
    for _ in 0..CUDA_ASYNC_TOKEN_RING_SLOTS {
        slots.push(new_cuda_topk_sampling_slot(dev, capacity_rows)?);
    }
    Ok(CudaTopKSamplingWorkspace {
        capacity_rows,
        capacity_k,
        vocab,
        location: dev.location(),
        ranked: None,
        token_ring: new_cuda_async_token_ring(dev, capacity_rows)?,
        slots,
    })
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_topk_sampling_params(
    params: &[CudaTopKSamplingParams],
    rows: usize,
    vocab: usize,
    op: &'static str,
) -> Result<usize> {
    if params.len() != rows {
        candle_core::bail!("{op} expected {rows} sampling parameter rows");
    }
    let mut max_k = 0usize;
    for params in params {
        if !params.inverse_temperature.is_finite() || params.inverse_temperature <= 0.0 {
            candle_core::bail!("{op} requires positive finite inverse temperatures");
        }
        if params.top_k == 0 || params.top_k > CUDA_TOPK_MAX_K {
            candle_core::bail!("{op} top-k must be in [1, {CUDA_TOPK_MAX_K}]");
        }
        if !params.top_p.is_finite() || !params.min_p.is_finite() {
            candle_core::bail!("{op} requires finite top-p and min-p values");
        }
        if !(0.0..1.0).contains(&params.uniform) {
            candle_core::bail!("{op} requires uniforms in [0, 1)");
        }
        max_k = max_k.max(params.top_k.min(vocab));
    }
    Ok(max_k)
}

#[cfg(feature = "cuda")]
pub(super) fn copy_cuda_topk_sampling_params(
    dev: &candle_core::CudaDevice,
    slot: &mut CudaTopKSamplingSlot,
    params: &[CudaTopKSamplingParams],
) -> Result<()> {
    let host = slot
        .params_host
        .as_mut_slice()
        .map_err(candle_core::Error::wrap)?;
    for (row, params) in params.iter().enumerate() {
        let start = row * CUDA_TOPK_SAMPLING_PARAM_WIDTH;
        let top_k = u16::try_from(params.top_k).map_err(candle_core::Error::wrap)?;
        host[start] = params.inverse_temperature;
        host[start + 1] = f32::from(top_k);
        host[start + 2] = params.top_p;
        host[start + 3] = params.min_p;
        host[start + 4] = params.uniform;
    }
    dev.memcpy_htod(&slot.params_host, &mut slot.params)?;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_topk_sampling_submit_inner(
    input: &Tensor,
    token_ids_dst: Option<&Tensor>,
    params: &[CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
    op: &'static str,
) -> Result<CudaTopKSamplingSubmission> {
    use candle_core::backend::{BackendDevice, BackendStorage};
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{op} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op });
    }
    let [rows, vocab] = input.dims() else {
        candle_core::bail!("{op} requires logits with shape [batch, vocab]");
    };
    if *rows == 0 || *vocab == 0 {
        candle_core::bail!("{op} requires non-empty logits");
    }
    let max_k = validate_cuda_topk_sampling_params(params, *rows, *vocab, op)?;
    let (storage, _) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{op} requires CUDA logits"),
    };
    let dev = storage.device();
    let needs_alloc = cache.as_ref().is_none_or(|workspace| {
        workspace.capacity_rows < *rows
            || workspace.capacity_k < max_k
            || workspace.vocab != *vocab
            || workspace.location != dev.location()
    });
    if needs_alloc {
        if cache
            .as_ref()
            .is_some_and(|workspace| workspace.token_ring.has_pending())
        {
            candle_core::bail!("{op} cannot resize while submissions are pending");
        }
        *cache = Some(new_cuda_topk_sampling_workspace(dev, *rows, *vocab, max_k)?);
    }

    let stream = dev.cuda_stream();
    let workspace = cache
        .as_mut()
        .expect("CUDA top-k sampling workspace was allocated above");
    let reservation = workspace
        .token_ring
        .reserve(input, token_ids_dst, *rows, op)?;
    let slot_index = reservation.slot;
    let result = (|| {
        let ranked =
            cuda_topk_ranked_packed_batched_with_workspace(input, max_k, &mut workspace.ranked)?;
        let slot = &mut workspace.slots[slot_index];
        copy_cuda_topk_sampling_params(dev, slot, params)?;

        let (packed_storage, packed_layout) = ranked.packed.storage_and_layout();
        let candle_core::Storage::Cuda(packed_storage) = &*packed_storage else {
            unreachable!("ranked top-k output is CUDA")
        };
        let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
            unreachable!("ranked top-k output is F32")
        };
        let (token_storage, token_layout) = reservation.device_tokens.storage_and_layout();
        let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
            unreachable!("reserved token destination is CUDA")
        };
        let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
            unreachable!("reserved token destination is U32")
        };
        let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
        let (params_ptr, params_guard) = slot.params.device_ptr(&stream);
        let (tokens_ptr, tokens_guard) = token_slice.device_ptr(&stream);
        let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
        let params_ptr = params_ptr as *const f32;
        let tokens_ptr = unsafe { (tokens_ptr as *mut u32).add(token_layout.start_offset()) };
        unsafe {
            ffi::sample_ranked_topk(
                packed_ptr,
                params_ptr,
                tokens_ptr,
                i32::try_from(*rows).map_err(candle_core::Error::wrap)?,
                i32::try_from(ranked.k).map_err(candle_core::Error::wrap)?,
                stream.cu_stream() as i64,
            );
        }
        drop(packed_guard);
        drop(params_guard);
        drop(tokens_guard);
        Result::<()>::Ok(())
    })();
    if let Err(error) = result {
        let _ = stream.synchronize();
        workspace.token_ring.abort(&reservation);
        return Err(error);
    }
    let token = match workspace.token_ring.finish_launch(reservation) {
        Ok(token) => token,
        Err(error) => {
            let _ = stream.synchronize();
            return Err(error);
        }
    };
    Ok(CudaTopKSamplingSubmission { token })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_sampling_submit_batched(
    input: &Tensor,
    params: &[CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
) -> Result<CudaTopKSamplingSubmission> {
    cuda_topk_sampling_submit_inner(
        input,
        None,
        params,
        cache,
        "cuda_topk_sampling_submit_batched",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_sampling_submit_batched_into(
    input: &Tensor,
    token_ids_dst: &Tensor,
    params: &[CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
) -> Result<CudaTopKSamplingSubmission> {
    cuda_topk_sampling_submit_inner(
        input,
        Some(token_ids_dst),
        params,
        cache,
        "cuda_topk_sampling_submit_batched_into",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_sampling_device_tokens_wait_on(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
    consumer_stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    workspace.token_ring.wait_on(
        &submission.token,
        consumer_stream,
        "cuda_topk_sampling_device_tokens_wait_on",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_sampling_device_tokens_release_after(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
    consumer_stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    workspace.token_ring.release_after(
        &submission.token,
        consumer_stream,
        "cuda_topk_sampling_device_tokens_release_after",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_sampling_submission_complete<'a>(
    workspace: &'a mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
) -> Result<CudaTopKSamplingCompletion<'a>> {
    let token_ids = workspace
        .token_ring
        .complete(&submission.token, "cuda_topk_sampling_submission_complete")?;
    if token_ids.contains(&CUDA_TOP1_INVALID_TOKEN) {
        candle_core::bail!("invalid CUDA top-k sampling output");
    }
    Ok(CudaTopKSamplingCompletion { token_ids })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_sampling_submission_cancel(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
) -> Result<()> {
    workspace
        .token_ring
        .cancel(&submission.token, "cuda_topk_sampling_submission_cancel")
}

#[allow(dead_code)]
pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

#[allow(dead_code)]
pub struct TopKLogitsPackedOutput {
    /// Each row is packed as `[values; indices_as_f32; softmax_denominator; global_max]`.
    pub packed: Tensor,
    pub k: usize,
    _workspace: Vec<Tensor>,
}

#[cfg(feature = "cuda")]
pub(crate) struct RankedTopKPackedOutput {
    /// Each row is packed as `[values; indices_as_f32]`.
    pub(crate) packed: Tensor,
    pub(crate) k: usize,
    pub(super) _workspace: Vec<Tensor>,
}

#[cfg(feature = "cuda")]
pub(crate) struct CategoricalLogitsPackedOutput {
    /// Each row is packed as `[token_index_as_f32, full_softmax_logprob]`.
    pub(crate) packed: Tensor,
    _workspace: Vec<Tensor>,
}

#[cfg(all(feature = "cuda", test))]
pub(crate) struct Top1LogitsPackedOutput {
    pub(crate) packed: Tensor,
    _workspace: Vec<Tensor>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DFlashSelectorSampleInput<'a> {
    pub(crate) topk: &'a RankedTopKPackedOutput,
    pub(crate) projected_hidden: &'a Tensor,
    pub(crate) predecessor_codebook: &'a Tensor,
    pub(crate) successor_codebook: &'a Tensor,
    pub(crate) anchors: &'a Tensor,
    pub(crate) inverse_temperatures: &'a Tensor,
    pub(crate) uniforms: &'a Tensor,
}

#[cfg(feature = "cuda")]
pub(crate) struct DFlashSelectorSampleOutput {
    pub(crate) tokens: Tensor,
    pub(crate) candidate_ids: Tensor,
    pub(crate) candidate_probs: Tensor,
}

#[cfg(feature = "cuda")]
pub fn cuda_apply_sparse_penalties_f32(
    input: &Tensor,
    token_ids: &Tensor,
    counts: &Tensor,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires F32 logits");
    }
    if token_ids.dtype() != DType::U32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires U32 token ids");
    }
    if counts.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires F32 counts");
    }
    if token_ids.elem_count() != counts.elem_count() {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 token ids/counts length mismatch: {} vs {}",
            token_ids.elem_count(),
            counts.elem_count()
        );
    }
    if !token_ids.device().same_device(input.device())
        || !counts.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 tensors must be on the same CUDA device"
        );
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let counts = counts.contiguous()?;

    let elem_count = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 got empty logits");
    }
    if elem_count > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 input is too large: {elem_count} elements"
        );
    }
    if n_tokens > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 token list is too large: {n_tokens} elements"
        );
    }
    let elem_count_i32 = i32::try_from(elem_count).map_err(candle_core::Error::wrap)?;
    let n_tokens_i32 = i32::try_from(n_tokens).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA logits"),
    };
    let CudaStorageSlice::F32(src) = &input_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports F32 logits");
    };

    let (token_storage, token_layout) = token_ids.storage_and_layout();
    let token_storage = match &*token_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA token ids"),
    };
    let CudaStorageSlice::U32(token_src) = &token_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports U32 token ids");
    };

    let (count_storage, count_layout) = counts.storage_and_layout();
    let count_storage = match &*count_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA counts"),
    };
    let CudaStorageSlice::F32(count_src) = &count_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports F32 counts");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut out = unsafe { dev.alloc::<f32>(elem_count) }?;

    let (src_ptr, src_guard) = src.device_ptr(&stream);
    let (token_ptr, token_guard) = token_src.device_ptr(&stream);
    let (count_ptr, count_guard) = count_src.device_ptr(&stream);
    let (out_ptr, out_guard) = out.device_ptr_mut(&stream);

    let src_ptr = unsafe { (src_ptr as *const f32).add(input_layout.start_offset()) };
    let token_ptr = unsafe { (token_ptr as *const u32).add(token_layout.start_offset()) };
    let count_ptr = unsafe { (count_ptr as *const f32).add(count_layout.start_offset()) };

    unsafe {
        ffi::apply_sparse_penalties_f32(
            src_ptr as *const c_void,
            out_ptr as *mut c_void,
            token_ptr,
            count_ptr,
            elem_count_i32,
            n_tokens_i32,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            stream.cu_stream() as i64,
        );
    }

    drop(src_guard);
    drop(token_guard);
    drop(count_guard);
    drop(out_guard);

    let out_storage = CudaStorage {
        slice: CudaStorageSlice::F32(out),
        device: dev.clone(),
    };
    Ok(Tensor::from((
        candle_core::Storage::Cuda(out_storage),
        input.shape().clone(),
    )))
}

#[cfg(feature = "cuda")]
pub fn cuda_apply_sparse_logits_bias_f32(
    input: &Tensor,
    token_ids: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires F32 logits");
    }
    if token_ids.dtype() != DType::U32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires U32 token ids");
    }
    if biases.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires F32 biases");
    }
    if token_ids.elem_count() != biases.elem_count() {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 token ids/biases length mismatch: {} vs {}",
            token_ids.elem_count(),
            biases.elem_count()
        );
    }
    if !token_ids.device().same_device(input.device())
        || !biases.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 tensors must be on the same CUDA device"
        );
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let biases = biases.contiguous()?;

    let elem_count = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 got empty logits");
    }
    if elem_count > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 input is too large: {elem_count} elements"
        );
    }
    if n_tokens > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 token list is too large: {n_tokens} elements"
        );
    }
    let elem_count_i32 = i32::try_from(elem_count).map_err(candle_core::Error::wrap)?;
    let n_tokens_i32 = i32::try_from(n_tokens).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA logits"),
    };
    let CudaStorageSlice::F32(src) = &input_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports F32 logits");
    };

    let (token_storage, token_layout) = token_ids.storage_and_layout();
    let token_storage = match &*token_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA token ids"),
    };
    let CudaStorageSlice::U32(token_src) = &token_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports U32 token ids");
    };

    let (bias_storage, bias_layout) = biases.storage_and_layout();
    let bias_storage = match &*bias_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA biases"),
    };
    let CudaStorageSlice::F32(bias_src) = &bias_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports F32 biases");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut out = unsafe { dev.alloc::<f32>(elem_count) }?;

    let (src_ptr, src_guard) = src.device_ptr(&stream);
    let (token_ptr, token_guard) = token_src.device_ptr(&stream);
    let (bias_ptr, bias_guard) = bias_src.device_ptr(&stream);
    let (out_ptr, out_guard) = out.device_ptr_mut(&stream);

    let src_ptr = unsafe { (src_ptr as *const f32).add(input_layout.start_offset()) };
    let token_ptr = unsafe { (token_ptr as *const u32).add(token_layout.start_offset()) };
    let bias_ptr = unsafe { (bias_ptr as *const f32).add(bias_layout.start_offset()) };

    unsafe {
        ffi::apply_sparse_logits_bias_f32(
            src_ptr as *const c_void,
            out_ptr as *mut c_void,
            token_ptr,
            bias_ptr,
            elem_count_i32,
            n_tokens_i32,
            stream.cu_stream() as i64,
        );
    }

    drop(src_guard);
    drop(token_guard);
    drop(bias_guard);
    drop(out_guard);

    let out_storage = CudaStorage {
        slice: CudaStorageSlice::F32(out),
        device: dev.clone(),
    };
    Ok(Tensor::from((
        candle_core::Storage::Cuda(out_storage),
        input.shape().clone(),
    )))
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_apply_causal_mask_f32(
    scores: &Tensor,
    q_offset: usize,
    prefix_len: usize,
) -> Result<()> {
    struct CausalMaskF32 {
        q_offset: usize,
        prefix_len: usize,
    }

    impl candle_core::InplaceOp1 for CausalMaskF32 {
        fn name(&self) -> &'static str {
            "causal-mask-f32"
        }

        fn cpu_fwd(
            &self,
            _storage: &mut candle_core::CpuStorage,
            _layout: &candle_core::Layout,
        ) -> Result<()> {
            candle_core::bail!("causal-mask-f32 requires CUDA storage")
        }

        fn cuda_fwd(
            &self,
            storage: &mut candle_core::CudaStorage,
            layout: &candle_core::Layout,
        ) -> Result<()> {
            use candle_core::backend::BackendStorage;
            use candle_core::cuda_backend::cudarc::driver::DevicePtrMut;
            use candle_core::cuda_backend::CudaStorageSlice;
            use std::ffi::c_void;

            let (batch_heads, q_len, kv_len) = layout.shape().dims3()?;
            let batch_heads = i32::try_from(batch_heads).map_err(candle_core::Error::wrap)?;
            let q_len = i32::try_from(q_len).map_err(candle_core::Error::wrap)?;
            let kv_len = i32::try_from(kv_len).map_err(candle_core::Error::wrap)?;
            let q_offset = i32::try_from(self.q_offset).map_err(candle_core::Error::wrap)?;
            let prefix_len = i32::try_from(self.prefix_len).map_err(candle_core::Error::wrap)?;
            if !layout.is_contiguous() {
                candle_core::bail!("causal-mask-f32 requires contiguous scores")
            }
            let dev = storage.device();
            let stream = dev.cuda_stream();
            let CudaStorageSlice::F32(scores) = &mut storage.slice else {
                candle_core::bail!("causal-mask-f32 requires F32 scores")
            };
            let (scores_ptr, scores_guard) = scores.device_ptr_mut(&stream);
            let scores_ptr =
                unsafe { (scores_ptr as *mut f32).add(layout.start_offset()) as *mut c_void };
            unsafe {
                ffi::apply_causal_mask_f32(
                    scores_ptr,
                    batch_heads,
                    q_len,
                    kv_len,
                    q_offset,
                    prefix_len,
                    stream.cu_stream() as i64,
                );
            }
            drop(scores_guard);
            Ok(())
        }
    }

    scores.inplace_op1(&CausalMaskF32 {
        q_offset,
        prefix_len,
    })
}

#[cfg(feature = "metal")]
pub fn metal_apply_sparse_penalties(
    input: &Tensor,
    token_ids: &Tensor,
    counts: &Tensor,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> Result<Tensor> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("metal_apply_sparse_penalties requires F32/F16/BF16 logits");
    }
    if token_ids.dtype() != DType::U32 || counts.dtype() != DType::F32 {
        candle_core::bail!("metal_apply_sparse_penalties token_ids must be u32, counts f32");
    }
    let dtype = input.dtype();
    let n = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if counts.elem_count() != n_tokens {
        candle_core::bail!("token_ids and counts length mismatch");
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let counts = counts.contiguous()?;

    let (input_s, input_l) = input.storage_and_layout();
    let (tok_s, tok_l) = token_ids.storage_and_layout();
    let (cnt_s, cnt_l) = counts.storage_and_layout();
    let (Storage::Metal(input_s), Storage::Metal(tok_s), Storage::Metal(cnt_s)) =
        (&*input_s, &*tok_s, &*cnt_s)
    else {
        candle_core::bail!("metal_apply_sparse_penalties requires Metal tensors");
    };
    let device = input_s.device().clone();

    let out_buf = device.new_buffer(n, dtype, "penalties-out")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("penalties-copy");
    {
        use mistralrs_quant::metal_kernels::Kernels;
        mistralrs_quant::metal_kernels::call_copy_logits(
            device.device(),
            &encoder,
            &Kernels::new(),
            dtype,
            input_s.buffer(),
            input_l.start_offset() * input.dtype().size_in_bytes(),
            &out_buf,
            n,
        )
        .map_err(|e| candle_core::Error::Msg(format!("metal copy: {e}")))?;
    }
    encoder.set_label("penalties-apply");
    mistralrs_quant::metal_kernels::call_apply_sparse_penalties(
        device.device(),
        &encoder,
        &mistralrs_quant::metal_kernels::Kernels::new(),
        dtype,
        &out_buf,
        tok_s.buffer(),
        cnt_s.buffer(),
        n,
        n_tokens,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
    )
    .map_err(|e| candle_core::Error::Msg(format!("metal penalties: {e}")))?;
    let _ = (tok_l, cnt_l);
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(out_buf, device.clone(), n, dtype)),
        Shape::from(input.dims()),
    )))
}

#[cfg(feature = "metal")]
#[allow(clippy::cast_possible_truncation)]
pub fn metal_topk_logits_packed(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    const MAX_K: usize = 128;
    const CHUNK_SIZE: usize = 2048;

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("metal_topk_logits_packed requires a positive finite temperature");
    }
    let input = input.contiguous()?;
    if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("metal_topk_logits_packed requires F32/F16/BF16 logits");
    }
    let dtype = input.dtype();
    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("metal_topk_logits_packed got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > MAX_K {
        candle_core::bail!("metal_topk_logits_packed k={k} must be in [1, {MAX_K}]");
    }
    let nblocks = ncols.div_ceil(CHUNK_SIZE);

    let (input_s, input_l) = input.storage_and_layout();
    let Storage::Metal(input_s) = &*input_s else {
        candle_core::bail!("metal_topk_logits_packed requires Metal tensor");
    };
    let device = input_s.device().clone();

    let block_values_buf = device.new_buffer(nblocks * k, DType::F32, "topk-block-values")?;
    let block_indices_buf = device.new_buffer(nblocks * k, DType::U32, "topk-block-indices")?;
    let block_maxes_buf = device.new_buffer(nblocks, DType::F32, "topk-block-maxes")?;
    let block_sums_buf = device.new_buffer(nblocks, DType::F32, "topk-block-sums")?;
    let packed_buf = device.new_buffer(2 * k + 2, DType::F32, "topk-packed")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("topk-logits-packed");

    let inv_temp = (1.0_f64 / temperature) as f32;
    let input_offset = input_l.start_offset() * input.dtype().size_in_bytes();

    mistralrs_quant::metal_kernels::call_topk_logits_packed(
        device.device(),
        &encoder,
        &mistralrs_quant::metal_kernels::Kernels::new(),
        dtype,
        input_s.buffer(),
        &block_values_buf,
        &block_indices_buf,
        &block_maxes_buf,
        &block_sums_buf,
        &packed_buf,
        ncols,
        k,
        CHUNK_SIZE,
        inv_temp,
    )
    .map_err(|e| candle_core::Error::Msg(format!("metal_topk_logits_packed kernel error: {e}")))?;
    let _ = (
        input_offset,
        &block_values_buf,
        &block_indices_buf,
        &block_maxes_buf,
        &block_sums_buf,
    );

    let packed = Tensor::from((
        Storage::Metal(MetalStorage::new(
            packed_buf,
            device.clone(),
            2 * k + 2,
            DType::F32,
        )),
        Shape::from(vec![2 * k + 2]),
    ));
    Ok(TopKLogitsPackedOutput {
        packed,
        k,
        _workspace: vec![],
    })
}
