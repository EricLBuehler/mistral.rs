use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CudaGraphMemoryUse {
    Capture,
    Retain,
    Eager,
}

pub(crate) struct CudaGraphCaptureMemory<'a> {
    pub plan: Option<crate::memory_plan::RuntimeMemoryPlan>,
    pub reservation: Option<&'a crate::resource_plan::RuntimeMemoryReservation>,
    pub devices: &'a [Device],
    pub primary: &'a Device,
    pub activation_dtype: DType,
    pub block_size: usize,
    pub step: &'a CudaGraphDecodeStep,
    pub kv_cache: &'a [(Tensor, Tensor)],
    pub model: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
}

impl CudaGraphCaptureMemory<'_> {
    pub(super) fn shape(
        &self,
        device: &Device,
    ) -> candle_core::Result<super::super::cuda_graph_budget::CaptureMemoryShape> {
        let query_tokens = self.step.input_ids.elem_count();
        let mut keys = Vec::new();
        for (name, map) in [
            ("slot_mappings", Some(&self.step.metadata.slot_mappings)),
            ("block_tables", self.step.metadata.block_tables.as_ref()),
            ("context_lens", self.step.metadata.context_lens.as_ref()),
            (
                "full_block_tables",
                self.step.metadata.full_block_tables.as_ref(),
            ),
            (
                "full_context_lens",
                self.step.metadata.full_context_lens.as_ref(),
            ),
        ] {
            push_graph_tensor_keys(name, map, &mut keys);
        }
        push_flashinfer_graph_tensor_keys(&self.step.metadata, &mut keys);
        let mut input_bytes = query_tokens
            .checked_mul(self.step.input_ids.dtype().size_in_bytes() + std::mem::size_of::<i64>())
            .ok_or_else(|| candle_core::Error::msg("CUDA capture input size overflow"))?;
        for key in keys.iter().filter(|key| key.location == device.location()) {
            let bytes = key
                .shape
                .iter()
                .try_fold(key.dtype.size_in_bytes(), |bytes, dim| {
                    bytes.checked_mul(*dim)
                })
                .ok_or_else(|| candle_core::Error::msg("CUDA capture metadata size overflow"))?;
            input_bytes = input_bytes
                .checked_add(bytes)
                .ok_or_else(|| candle_core::Error::msg("CUDA capture input size overflow"))?;
        }
        let mut attention_bytes = 0;
        let mut attention_is_complete = cfg!(target_family = "unix")
            && self.model.is_some()
            && self.plan.is_none_or(|plan| !plan.custom_prefill_mask);
        if let Some(model) = self.model {
            #[cfg(target_family = "unix")]
            let mut fa3_specs = std::collections::HashSet::new();
            #[allow(unused_mut)]
            let mut fa3_bytes = 0usize;
            let mut transient_attention_bytes = 0usize;
            let mut flashinfer_heads = 0usize;
            let mut flashinfer_dim = 0usize;
            let context = self
                .step
                .metadata
                .max_context_len
                .unwrap_or_default()
                .max(self.step.metadata.full_max_context_len.unwrap_or_default());
            let context = if context == 0 {
                model.max_seq_len()
            } else {
                context
            };
            let partitions = context.div_ceil(PAGED_ATTENTION_PARTITION_SIZE).max(1);
            for (layer, (key_cache, _value_cache)) in
                self.kv_cache.iter().enumerate().take(model.num_layers())
            {
                if !key_cache.device().same_device(device) || !model.layer_has_paged_kv_cache(layer)
                {
                    continue;
                }
                let flashinfer = model.attention_backend_kind_for_layer(layer)
                    == AttentionBackendKind::FlashInfer;
                let features = model.prefix_prefill_attention_features(layer);
                attention_is_complete &= features.is_some_and(|features| {
                    !DecodePlan::requires_host_context_lengths(
                        model.attention_backend_kind_for_layer(layer),
                        model.k_head_dim_for_layer(layer),
                        model.sliding_window_for_layer(layer).is_some()
                            || features.has_sliding_window,
                    )
                });
                if flashinfer {
                    flashinfer_heads = flashinfer_heads.max(model.num_attn_heads_for_layer(layer));
                    flashinfer_dim = flashinfer_dim.max(model.v_head_dim_for_layer(layer));
                }
                #[cfg(target_family = "unix")]
                if mistralrs_paged_attn::USE_FA3_FP8_PAGED
                    && self.activation_dtype == DType::BF16
                    && flashinfer
                    && key_cache.dtype() == DType::F8E4M3
                    && _value_cache.dtype() == DType::F8E4M3
                    && crate::flashinfer::fa3_device_num_sm(device).is_some()
                {
                    let (num_pages, kv_heads, page_size, head_dim) = key_cache.dims4()?;
                    let batch = self.step.input_ids.dim(0)?;
                    let query_len = self.step.input_ids.dim(1)?;
                    let key = (crate::flashinfer::Fa3PagedScheduleShape {
                        device: device.location(),
                        view: crate::flashinfer::Fa3DecodeView::Logical,
                        batch,
                        query_len,
                        causal: query_len > 1,
                        q_heads: model.num_attn_heads_for_layer(layer),
                        kv_heads,
                        head_dim,
                        page_size,
                    })
                    .decode_schedule_key();
                    if let Some(key) = key.filter(|_| {
                        num_pages > 0
                            && _value_cache.dims() == key_cache.dims()
                            && _value_cache.device().same_device(device)
                    }) {
                        if let Some(indices) =
                            self.step.metadata.flashinfer.as_ref().and_then(|metadata| {
                                metadata
                                    .views
                                    .logical
                                    .paged_kv
                                    .indices
                                    .get(&device.location())
                            })
                        {
                            if fa3_specs.insert(key) {
                                let bytes =
                                    super::super::cuda_graph_budget::fa3_capture_pool_bytes(
                                        super::super::cuda_graph_budget::Fa3CaptureMemoryShape {
                                            batch,
                                            query_len,
                                            heads: key.q_heads,
                                            head_dim,
                                            splits: key.num_splits,
                                            pages_per_sequence: indices
                                                .elem_count()
                                                .div_ceil(query_tokens),
                                        },
                                    )
                                    .map_err(candle_core::Error::wrap)?;
                                fa3_bytes = fa3_bytes.checked_add(bytes).ok_or_else(|| {
                                    candle_core::Error::msg("FA3 capture workspace size overflow")
                                })?;
                            }
                        }
                    }
                }
                let bytes = super::super::cuda_graph_budget::partitioned_attention_bytes(
                    query_tokens,
                    model.num_attn_heads_for_layer(layer),
                    model.v_head_dim_for_layer(layer),
                    if flashinfer {
                        CAPTURE_ATTENTION_OUTPUT_BUFFERS
                    } else {
                        partitions
                            .checked_add(CAPTURE_ATTENTION_OUTPUT_BUFFERS)
                            .ok_or_else(|| {
                                candle_core::Error::msg("CUDA capture partition size overflow")
                            })?
                    },
                    self.activation_dtype.size_in_bytes(),
                )
                .map_err(candle_core::Error::wrap)?;
                transient_attention_bytes = transient_attention_bytes.max(bytes);
            }
            let split_rows =
                flashinfer_split_rows(&self.step.metadata, self.step.input_ids.dim(0)?)?;
            let flashinfer_bytes = if let Some(&rows) = split_rows.get(&device.location()) {
                super::super::cuda_graph_budget::partitioned_attention_bytes(
                    rows,
                    flashinfer_heads,
                    flashinfer_dim,
                    1,
                    self.activation_dtype.size_in_bytes(),
                )
                .map_err(candle_core::Error::wrap)?
            } else {
                0
            };
            attention_bytes = super::super::cuda_graph_budget::capture_attention_bytes(
                fa3_bytes,
                flashinfer_bytes,
                transient_attention_bytes,
            )
            .map_err(candle_core::Error::wrap)?;
            let page_tables = model
                .max_seq_len()
                .div_ceil(self.block_size)
                .checked_mul(self.step.input_ids.dim(0)?)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<i32>()))
                .ok_or_else(|| candle_core::Error::msg("CUDA capture page table size overflow"))?;
            input_bytes = input_bytes
                .checked_add(page_tables)
                .ok_or_else(|| candle_core::Error::msg("CUDA capture input size overflow"))?;
        }
        Ok(super::super::cuda_graph_budget::CaptureMemoryShape {
            query_tokens,
            input_bytes,
            attention_bytes,
            attention_is_complete,
        })
    }

    pub(super) fn fits(
        &self,
        usage: CudaGraphMemoryUse,
        protect_other_models: bool,
    ) -> candle_core::Result<bool> {
        let Some(plan) = self.plan else {
            return Ok(true);
        };
        let revision = self
            .reservation
            .filter(|_| usage != CudaGraphMemoryUse::Eager)
            .map(|reservation| (reservation, reservation.revision()));
        for device in self.devices.iter().filter(|device| device.is_cuda()) {
            let workspace = if usage == CudaGraphMemoryUse::Retain {
                0
            } else {
                super::super::cuda_graph_budget::capture_workspace_bytes(
                    plan,
                    device.same_device(self.primary),
                    &self.shape(device)?,
                )
                .map_err(candle_core::Error::wrap)?
            };
            let mut pending =
                mistralrs_quant::gguf_affine_reserved_bytes(device, self.activation_dtype);
            if usage != CudaGraphMemoryUse::Eager {
                let prefix = self
                    .reservation
                    .map(|reservation| reservation.pending_prefix_bytes(device.location()))
                    .transpose()
                    .map_err(candle_core::Error::wrap)?
                    .unwrap_or_default();
                pending = pending.checked_add(prefix).ok_or_else(|| {
                    candle_core::Error::msg("CUDA graph pending prefix reservation overflow")
                })?;
            }
            let other_workspace = self
                .reservation
                .filter(|_| protect_other_models)
                .map(|reservation| reservation.other_workspace_bytes(device.location()))
                .transpose()
                .map_err(candle_core::Error::wrap)?
                .unwrap_or_default();
            let snapshot = crate::MemoryUsage.query(device)?;
            let memory_available = if usage == CudaGraphMemoryUse::Eager {
                super::super::cuda_graph_budget::eager_memory_available
            } else {
                super::super::cuda_graph_budget::capture_memory_available
            };
            if !memory_available(plan, snapshot, workspace, pending, other_workspace)
                .map_err(candle_core::Error::wrap)?
            {
                return Ok(false);
            }
        }
        Ok(revision.is_none_or(|(reservation, before)| reservation.revision() == before))
    }
}

pub(crate) fn prepare_cuda_capture_memory(
    state: &mut CudaDecodeGraphState,
    memory: &CudaGraphCaptureMemory<'_>,
) -> candle_core::Result<bool> {
    admit_cuda_graph_memory(state, memory, CudaGraphMemoryUse::Capture)
}

pub(crate) fn admit_cuda_eager_after_capture(
    state: &mut CudaDecodeGraphState,
    memory: &CudaGraphCaptureMemory<'_>,
) -> candle_core::Result<()> {
    admit_cuda_graph_memory(state, memory, CudaGraphMemoryUse::Eager).map(|_| ())
}

pub(super) fn admit_cuda_graph_memory(
    state: &mut CudaDecodeGraphState,
    memory: &CudaGraphCaptureMemory<'_>,
    usage: CudaGraphMemoryUse,
) -> candle_core::Result<bool> {
    let protect_other_models = usage != CudaGraphMemoryUse::Eager;
    let result = (|| {
        if memory.fits(usage, protect_other_models)? {
            return Ok(true);
        }
        if protect_other_models && memory.fits(usage, false)? {
            return Ok(false);
        }
        for device in memory.devices.iter().filter(|device| device.is_cuda()) {
            crate::MemoryUsage.synchronize_cuda_context(device)?;
            crate::MemoryUsage.trim_cuda_memory_pool(device, 0)?;
            trim_cuda_graph_memory(&device.as_cuda_device()?.cuda_stream())?;
        }
        loop {
            if memory.fits(usage, protect_other_models)? {
                return Ok(true);
            }
            if protect_other_models && memory.fits(usage, false)? {
                return Ok(false);
            }
            if state.evict_lru_for_memory_pressure(CAPTURE_RECLAIM_BATCH_SIZE) == 0 {
                if usage == CudaGraphMemoryUse::Retain
                    || (usage == CudaGraphMemoryUse::Capture
                        && memory.fits(CudaGraphMemoryUse::Eager, false)?)
                {
                    return Ok(false);
                }
                candle_core::bail!("CUDA decode workspace cannot fit within the runtime memory budget after reclaiming cached graphs");
            }
        }
    })();
    if result.is_err() {
        state.block_eager_retry();
    }
    result
}

pub(crate) fn finish_cuda_capture_memory(
    state: &mut CudaDecodeGraphState,
    memory: &CudaGraphCaptureMemory<'_>,
    entry: CudaDecodeGraphEntry,
) -> candle_core::Result<Option<CudaDecodeGraphEntry>> {
    match admit_cuda_graph_memory(state, memory, CudaGraphMemoryUse::Retain) {
        Ok(true) => Ok(Some(entry)),
        Ok(false) => {
            release_cuda_graph_entries(vec![entry]);
            Ok(None)
        }
        Err(err) => {
            release_cuda_graph_entries(vec![entry]);
            Err(err)
        }
    }
}

pub(super) struct MemoryPoolScopeState {
    pub(super) graph_guards: usize,
    pub(super) engine_retainers: usize,
    pub(super) release_threshold: u64,
}

#[must_use]
pub(crate) struct CudaGraphMemoryPoolGuard {
    pub(super) stream: Arc<CudaStream>,
    pub(super) pool: Option<usize>,
}

impl Drop for CudaGraphMemoryPoolGuard {
    fn drop(&mut self) {
        let Some(pool) = self.pool else { return };
        if let Err(err) = self.stream.context().bind_to_thread() {
            tracing::warn!("Failed to bind CUDA context while restoring graph memory pool: {err}");
        }
        let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
        let mut scopes = scopes
            .lock()
            .expect("CUDA graph memory pool scopes poisoned");
        let Some(scope) = scopes.get_mut(&pool) else {
            tracing::warn!("CUDA graph memory pool scope disappeared before restoration");
            return;
        };
        scope.graph_guards = scope
            .graph_guards
            .checked_sub(1)
            .expect("CUDA graph memory pool guard underflow");
        if scope.graph_guards != 0 {
            return;
        }
        let restore_threshold = scope.engine_retainers == 0;
        if restore_threshold {
            let pool_ptr = pool as sys::CUmemoryPool;
            if let Err(err) = set_memory_pool_release_threshold(pool_ptr, scope.release_threshold) {
                tracing::warn!("Failed to restore CUDA graph memory pool threshold: {err:?}");
            }
        }
        let _ = self.stream.synchronize();
        if let Err(err) = trim_cuda_graph_memory_bound(&self.stream) {
            tracing::warn!("Failed to trim CUDA graph memory after capture: {err:?}");
        }
        if restore_threshold {
            scopes.remove(&pool);
        }
    }
}

#[must_use]
pub(crate) struct CudaMemoryPoolRetention {
    pub(super) stream: Arc<CudaStream>,
    pub(super) pool: Option<usize>,
}

impl Drop for CudaMemoryPoolRetention {
    fn drop(&mut self) {
        let Some(pool) = self.pool else { return };
        if let Err(err) = self.stream.context().bind_to_thread() {
            tracing::warn!(
                "Failed to bind CUDA context while releasing memory pool retention: {err}"
            );
        }
        let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
        let mut scopes = scopes
            .lock()
            .expect("CUDA graph memory pool scopes poisoned");
        let Some(scope) = scopes.get_mut(&pool) else {
            tracing::warn!("CUDA memory pool retention disappeared before restoration");
            return;
        };
        scope.engine_retainers = scope
            .engine_retainers
            .checked_sub(1)
            .expect("CUDA memory pool retention underflow");
        if scope.engine_retainers != 0 || scope.graph_guards != 0 {
            return;
        }
        if let Err(err) =
            set_memory_pool_release_threshold(pool as sys::CUmemoryPool, scope.release_threshold)
        {
            tracing::warn!("Failed to restore CUDA memory pool retention threshold: {err:?}");
        }
        scopes.remove(&pool);
    }
}

pub(super) fn memory_pool_scope(
    scopes: &mut HashMap<usize, MemoryPoolScopeState>,
    pool: sys::CUmemoryPool,
) -> candle_core::Result<&mut MemoryPoolScopeState> {
    use std::collections::hash_map::Entry;

    Ok(match scopes.entry(pool as usize) {
        Entry::Occupied(scope) => scope.into_mut(),
        Entry::Vacant(entry) => {
            let release_threshold = memory_pool_release_threshold(pool)?;
            set_memory_pool_release_threshold(pool, u64::MAX)?;
            entry.insert(MemoryPoolScopeState {
                graph_guards: 0,
                engine_retainers: 0,
                release_threshold,
            })
        }
    })
}

pub(crate) fn retain_cuda_memory_pool(
    stream: &Arc<CudaStream>,
) -> candle_core::Result<CudaMemoryPoolRetention> {
    let pool = if stream.context().has_async_alloc() {
        let pool = cuda_memory_pool(stream)?;
        let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
        let mut scopes = scopes
            .lock()
            .expect("CUDA graph memory pool scopes poisoned");
        let scope = memory_pool_scope(&mut scopes, pool)?;
        scope.engine_retainers = scope
            .engine_retainers
            .checked_add(1)
            .expect("CUDA memory pool retention overflow");
        Some(pool as usize)
    } else {
        None
    };
    Ok(CudaMemoryPoolRetention {
        stream: stream.clone(),
        pool,
    })
}

pub(super) fn memory_pool_release_threshold(pool: sys::CUmemoryPool) -> candle_core::Result<u64> {
    let mut value = 0u64;
    let result = unsafe {
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut value as *mut u64).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool release threshold lookup failed"));
    }
    Ok(value)
}

pub(super) fn set_memory_pool_release_threshold(
    pool: sys::CUmemoryPool,
    mut value: u64,
) -> candle_core::Result<()> {
    let result = unsafe {
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut value as *mut u64).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool release threshold setup failed"));
    }
    Ok(())
}

pub(super) fn cuda_memory_pool(stream: &Arc<CudaStream>) -> candle_core::Result<sys::CUmemoryPool> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let mut pool = std::ptr::null_mut();
    let result = unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool lookup failed"));
    }
    Ok(pool)
}

pub(crate) fn cuda_graph_memory_pool_scope_active(device: &Device) -> candle_core::Result<bool> {
    let Device::Cuda(device) = device else {
        return Ok(false);
    };
    let stream = device.cuda_stream();
    if !stream.context().has_async_alloc() {
        return Ok(false);
    }
    let pool = cuda_memory_pool(&stream)? as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    Ok(scopes
        .get(&pool)
        .is_some_and(|scope| scope.graph_guards != 0))
}

pub(super) fn trim_cuda_graph_memory_bound(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
    let result = unsafe { sys::cuDeviceGraphMemTrim(stream.context().cu_device()) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(
            candle_core::Error::msg(format!("{result:?}")).context("CUDA graph memory trim failed")
        );
    }
    Ok(())
}

pub(super) fn cuda_graph_memory_attribute(
    stream: &Arc<CudaStream>,
    attribute: sys::CUgraphMem_attribute,
) -> candle_core::Result<usize> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let mut value = 0usize;
    let result = unsafe {
        sys::cuDeviceGetGraphMemAttribute(
            stream.context().cu_device(),
            attribute,
            (&mut value as *mut usize).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph memory attribute lookup failed"));
    }
    Ok(value)
}

pub(crate) fn trim_cuda_graph_memory(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("CUDA graph memory trim context bind failed"))?;
    let pool = cuda_memory_pool(stream)? as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    if scopes
        .get(&pool)
        .is_some_and(|scope| scope.graph_guards != 0)
    {
        return Ok(());
    }
    stream
        .synchronize()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("CUDA graph memory trim synchronization failed"))?;
    trim_cuda_graph_memory_bound(stream)
}

pub(super) static CUDA_GRAPH_CAPTURES: AtomicU64 = AtomicU64::new(0);
// Unified-memory graphs retain allocations, so new shapes run eagerly below this memory floor.
pub(super) const UNIFIED_CAPTURE_MIN_AVAILABLE_BYTES: usize = 2 << 30;
pub(super) static CAPTURE_SKIP_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether a new graph may be captured without eating the memory the system needs on a unified device.
pub(crate) fn cuda_graph_capture_headroom(device: &Device) -> candle_core::Result<bool> {
    if !crate::utils::normal::is_integrated_gpu(device) {
        return Ok(true);
    }
    let available = crate::MemoryUsage.query(device)?.available();
    let ok = available >= UNIFIED_CAPTURE_MIN_AVAILABLE_BYTES;
    if !ok && !CAPTURE_SKIP_LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "Unified memory is low ({} MiB available); new CUDA graph shapes will run eagerly.",
            available >> 20
        );
    }
    Ok(ok)
}

/// Logs the memory the captured graphs hold and what is left for the runtime.
pub(crate) fn log_cuda_graph_memory(device: &Device) -> candle_core::Result<()> {
    let Device::Cuda(cuda) = device else {
        return Ok(());
    };
    let reserved = cuda_graph_memory_attribute(
        &cuda.cuda_stream(),
        sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
    )?;
    let available = crate::MemoryUsage.query(device)?.available();
    tracing::info!(
        "CUDA graphs reserve {} MiB; {} MiB of memory available after startup.",
        reserved >> 20,
        available >> 20
    );
    Ok(())
}

/// Graph captures started so far; timing that spans a change includes a one-off capture.
pub(crate) fn cuda_graph_capture_count() -> u64 {
    CUDA_GRAPH_CAPTURES.load(Ordering::Relaxed)
}

pub(crate) fn prepare_cuda_graph_memory_pool(
    stream: &Arc<CudaStream>,
) -> candle_core::Result<CudaGraphMemoryPoolGuard> {
    CUDA_GRAPH_CAPTURES.fetch_add(1, Ordering::Relaxed);
    if !stream.context().has_async_alloc() {
        return Ok(CudaGraphMemoryPoolGuard {
            stream: stream.clone(),
            pool: None,
        });
    }

    let pool = cuda_memory_pool(stream)?;
    let pool_key = pool as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let mut scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    let scope = memory_pool_scope(&mut scopes, pool)?;
    scope.graph_guards = scope
        .graph_guards
        .checked_add(1)
        .expect("CUDA graph memory pool guard overflow");
    drop(scopes);

    let guard = CudaGraphMemoryPoolGuard {
        stream: stream.clone(),
        pool: Some(pool_key),
    };
    for attr in [
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
    ] {
        let mut enabled = 1i32;
        let result =
            unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut enabled as *mut i32).cast()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph mempool reuse setup failed"));
        }
    }

    Ok(guard)
}
