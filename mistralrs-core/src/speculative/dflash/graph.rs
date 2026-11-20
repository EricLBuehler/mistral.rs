use super::*;

#[cfg(any(cuda_flash_attn, test))]
pub(super) struct DFlashGraphHostRows {
    pub(super) token_ids: Vec<u32>,
    pub(super) rope_indices: Vec<u32>,
    pub(super) anchors: Vec<u32>,
    pub(super) selector_inverse_temperatures: Option<Vec<f32>>,
    pub(super) selector_uniforms: Option<Vec<f32>>,
}

#[cfg(any(cuda_flash_attn, test))]
#[derive(Clone, Copy)]
pub(super) struct DFlashGraphHostInput<'a> {
    pub(super) anchors: &'a [u32],
    pub(super) start_positions: &'a [usize],
    pub(super) mask_token_id: u32,
    pub(super) block: usize,
    pub(super) batch_bucket: usize,
    pub(super) sampling: Option<DFlashSamplingInputs<'a>>,
}

#[cfg(any(cuda_flash_attn, test))]
impl DFlashGraphHostRows {
    pub(super) fn update(&mut self, input: DFlashGraphHostInput<'_>) -> Result<()> {
        let DFlashGraphHostInput {
            anchors,
            start_positions,
            mask_token_id,
            block,
            batch_bucket,
            sampling,
        } = input;
        if anchors.is_empty() || anchors.len() != start_positions.len() {
            candle_core::bail!("DFlash graph inputs must contain matching non-empty rows");
        }
        if block == 0 || batch_bucket < anchors.len() {
            candle_core::bail!("DFlash graph input shape is invalid");
        }
        if let Some(sampling) = sampling {
            let expected_uniforms = anchors
                .len()
                .checked_mul(block - 1)
                .ok_or_else(|| candle_core::Error::msg("DFlash sampling input size overflow"))?;
            if sampling.inverse_temperatures.len() != anchors.len()
                || sampling.uniforms.len() != expected_uniforms
            {
                candle_core::bail!("DFlash sampling inputs do not match the graph rows");
            }
        }

        self.token_ids.clear();
        self.rope_indices.clear();
        self.anchors.clear();
        self.token_ids.reserve(batch_bucket * block);
        self.rope_indices.reserve(batch_bucket * block);
        self.anchors.reserve(batch_bucket);
        for row in 0..batch_bucket {
            let source = row.min(anchors.len() - 1);
            let anchor = anchors[source];
            let start = start_positions[source];
            self.anchors.push(anchor);
            self.token_ids.push(anchor);
            self.token_ids
                .extend(std::iter::repeat_n(mask_token_id, block - 1));
            for offset in 0..block {
                let position = start
                    .checked_add(offset)
                    .ok_or_else(|| candle_core::Error::msg("DFlash graph position overflow"))?;
                self.rope_indices
                    .push(u32::try_from(position).map_err(candle_core::Error::wrap)?);
            }
        }

        match sampling {
            Some(sampling) => {
                let inverse_temperatures = self
                    .selector_inverse_temperatures
                    .get_or_insert_with(Vec::new);
                inverse_temperatures.clear();
                inverse_temperatures.resize(batch_bucket, 0.0);
                inverse_temperatures[..anchors.len()]
                    .copy_from_slice(sampling.inverse_temperatures);
                let uniforms = self.selector_uniforms.get_or_insert_with(Vec::new);
                uniforms.clear();
                uniforms.resize(batch_bucket * (block - 1), 0.0);
                uniforms[..sampling.uniforms.len()].copy_from_slice(sampling.uniforms);
            }
            None => {
                self.selector_inverse_temperatures = None;
                self.selector_uniforms = None;
            }
        }
        Ok(())
    }
}

#[cfg(any(cuda_flash_attn, test))]
pub(super) fn dflash_graph_host_rows(
    input: DFlashGraphHostInput<'_>,
) -> Result<DFlashGraphHostRows> {
    let DFlashGraphHostInput {
        block,
        batch_bucket,
        sampling,
        ..
    } = input;
    let mut rows = DFlashGraphHostRows {
        token_ids: Vec::with_capacity(batch_bucket * block),
        rope_indices: Vec::with_capacity(batch_bucket * block),
        anchors: Vec::with_capacity(batch_bucket),
        selector_inverse_temperatures: sampling.map(|_| Vec::with_capacity(batch_bucket)),
        selector_uniforms: sampling.map(|_| Vec::with_capacity(batch_bucket * (block - 1))),
    };
    rows.update(input)?;
    Ok(rows)
}

#[cfg(cuda_flash_attn)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct DFlashCudaGraphKey {
    pub(super) batch_bucket: usize,
    pub(super) block: usize,
    pub(super) selector_mode: DFlashSelectorMode,
}

#[cfg(cuda_flash_attn)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum DFlashSelectorMode {
    Disabled,
    Greedy,
    Sampling,
}

#[cfg(cuda_flash_attn)]
pub(super) struct DFlashCudaGraphBuffers {
    pub(super) token_ids: Var,
    pub(super) rope_indices: Var,
    pub(super) anchors: Var,
    pub(super) block_tables: Var,
    pub(super) slot_mapping: Var,
    pub(super) cumulative_kv_lens: Var,
    pub(super) cumulative_query_lens: Tensor,
    pub(super) output_tokens: Var,
    pub(super) selector_inverse_temperatures: Option<Var>,
    pub(super) selector_uniforms: Option<Var>,
    pub(super) output_candidate_ids: Option<Var>,
    pub(super) output_candidate_probs: Option<Var>,
}

#[cfg(cuda_flash_attn)]
pub(super) struct DFlashCudaGraphEntry {
    pub(super) key: DFlashCudaGraphKey,
    pub(super) staging: crate::pipeline::cuda_graph::CudaGraphHostStaging,
    pub(super) buffers: DFlashCudaGraphBuffers,
    pub(super) token_embedding: Arc<dyn QuantMethod>,
    pub(super) lm_head: Arc<dyn QuantMethod>,
    pub(super) mask_token_id: u32,
    pub(super) host_rows: DFlashGraphHostRows,
    pub(super) graph: crate::pipeline::cuda_graph::CudaGraphHandle,
}

#[cfg(cuda_flash_attn)]
pub(super) struct DFlashCudaGraphState {
    pub(super) entries: Vec<DFlashCudaGraphEntry>,
    pub(super) warmed: HashSet<DFlashCudaGraphKey>,
    pub(super) failed: HashSet<DFlashCudaGraphKey>,
    pub(super) capacity: usize,
}

#[cfg(cuda_flash_attn)]
impl Default for DFlashCudaGraphState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            warmed: HashSet::new(),
            failed: HashSet::new(),
            capacity: DFLASH_CUDA_GRAPH_DEFAULT_CACHE_CAPACITY,
        }
    }
}

#[cfg(cuda_flash_attn)]
pub(super) struct DFlashCudaGraphRun<'a> {
    pub(super) model: &'a DFlashDraftModel,
    pub(super) key: DFlashCudaGraphKey,
    pub(super) anchors: &'a [u32],
    pub(super) start_positions: &'a [usize],
    pub(super) sampling: Option<DFlashSamplingInputs<'a>>,
    pub(super) attention_batch: &'a WindowedKvBatch,
    pub(super) token_embedding: &'a Arc<dyn QuantMethod>,
    pub(super) lm_head: &'a Arc<dyn QuantMethod>,
    pub(super) real_batch: usize,
}

#[cfg(cuda_flash_attn)]
pub(super) struct DFlashGraphTemporarySequences<'a> {
    pub(super) pool: &'a Mutex<WindowedKvPool>,
    pub(super) seq_ids: Vec<usize>,
}

#[cfg(cuda_flash_attn)]
impl<'a> DFlashGraphTemporarySequences<'a> {
    pub(super) fn acquire(pool: &'a Mutex<WindowedKvPool>, count: usize) -> Result<Self> {
        let mut locked = pool.lock().expect("dflash windowed pool poisoned");
        let mut seq_ids = Vec::with_capacity(count);
        let mut candidate = usize::MAX;
        while seq_ids.len() < count {
            if locked.sequence(candidate).is_none() {
                if let Err(err) = locked.acquire_at(candidate, 0) {
                    for seq_id in &seq_ids {
                        locked.release(*seq_id);
                    }
                    return Err(err);
                }
                seq_ids.push(candidate);
            }
            candidate = candidate.checked_sub(1).ok_or_else(|| {
                candle_core::Error::msg("DFlash graph temporary sequence id space exhausted")
            })?;
        }
        Ok(Self { pool, seq_ids })
    }

    pub(super) fn attention_batch(&self, batch: usize, block: usize) -> Result<WindowedKvBatch> {
        let queries = self.seq_ids[..batch]
            .iter()
            .map(|seq_id| WindowedKvQuery {
                seq_id: *seq_id,
                query_len: block,
            })
            .collect::<Vec<_>>();
        self.pool
            .lock()
            .expect("dflash windowed pool poisoned")
            .scratch_graph_batch(&queries, batch)
    }
}

#[cfg(cuda_flash_attn)]
impl Drop for DFlashGraphTemporarySequences<'_> {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().expect("dflash windowed pool poisoned");
        for seq_id in &self.seq_ids {
            pool.release(*seq_id);
        }
    }
}

#[cfg(cuda_flash_attn)]
pub(super) struct DFlashCudaGraphOutput {
    pub(super) tokens: Tensor,
    pub(super) candidate_ids: Option<Tensor>,
    pub(super) candidate_probs: Option<Tensor>,
}

#[cfg(any(cuda_flash_attn, test))]
pub(super) fn copy_dflash_graph_output_rows(output: &Tensor, real_batch: usize) -> Result<Tensor> {
    output.narrow(0, 0, real_batch)?.copy()
}

#[cfg(cuda_flash_attn)]
impl DFlashCudaGraphOutput {
    pub(super) fn finish(self, real_batch: usize) -> Result<DFlashProposalBatch> {
        let tokens = copy_dflash_graph_output_rows(&self.tokens, real_batch)?;
        match (self.candidate_ids, self.candidate_probs) {
            (Some(candidate_ids), Some(candidate_probs)) => Ok(DFlashProposalBatch::DeviceSparse {
                tokens,
                candidate_ids: copy_dflash_graph_output_rows(&candidate_ids, real_batch)?,
                candidate_probs: copy_dflash_graph_output_rows(&candidate_probs, real_batch)?,
            }),
            (None, None) => Ok(DFlashProposalBatch::DeviceTokens(tokens)),
            _ => candle_core::bail!("DFlash selector returned incomplete sparse probabilities"),
        }
    }
}

#[cfg(cuda_flash_attn)]
impl DFlashCudaGraphBuffers {
    pub(super) fn new(
        key: DFlashCudaGraphKey,
        rows: &DFlashGraphHostRows,
        batch: &WindowedKvBatch,
        model: &DFlashDraftModel,
    ) -> Result<Self> {
        let device = &model.device;
        let token_ids = Var::from_tensor(&Tensor::from_vec(
            rows.token_ids.clone(),
            (key.batch_bucket, key.block),
            device,
        )?)?;
        let rope_indices = Var::from_tensor(&Tensor::from_vec(
            rows.rope_indices.clone(),
            (key.batch_bucket * key.block,),
            device,
        )?)?;
        let anchors = Var::from_tensor(&Tensor::from_vec(
            rows.anchors.clone(),
            (key.batch_bucket,),
            device,
        )?)?;
        let block_tables = Var::from_tensor(&Tensor::from_vec(
            batch.block_tables_for_graph().to_vec(),
            (key.batch_bucket, batch.block_table_width_for_graph()),
            device,
        )?)?;
        let slot_mapping = Var::from_tensor(&Tensor::from_vec(
            batch.slot_mapping_for_graph().to_vec(),
            (key.batch_bucket * key.block,),
            device,
        )?)?;
        let cumulative_kv_lens = Var::from_tensor(&Tensor::from_vec(
            batch.cumulative_kv_lens_for_graph().to_vec(),
            (key.batch_bucket + 1,),
            device,
        )?)?;
        let cumulative_query_lens = Tensor::from_vec(
            batch.cumulative_query_lens_for_graph().to_vec(),
            (key.batch_bucket + 1,),
            device,
        )?;
        let output_tokens = Var::zeros((key.batch_bucket, key.block - 1), DType::U32, device)?;
        let (
            selector_inverse_temperatures,
            selector_uniforms,
            output_candidate_ids,
            output_candidate_probs,
        ) = if key.selector_mode == DFlashSelectorMode::Sampling {
            let selector = model
                .selector
                .as_ref()
                .expect("sampling graph key requires a selector");
            let inverse_temperatures = rows
                .selector_inverse_temperatures
                .as_ref()
                .expect("sampling graph rows require inverse temperatures");
            let uniforms = rows
                .selector_uniforms
                .as_ref()
                .expect("sampling graph rows require uniforms");
            (
                Some(Var::from_tensor(&Tensor::from_vec(
                    inverse_temperatures.clone(),
                    (key.batch_bucket,),
                    device,
                )?)?),
                Some(Var::from_tensor(&Tensor::from_vec(
                    uniforms.clone(),
                    (key.batch_bucket, key.block - 1),
                    device,
                )?)?),
                Some(Var::zeros(
                    (key.batch_bucket, key.block - 1, selector.top_k),
                    DType::U32,
                    device,
                )?),
                Some(Var::zeros(
                    (key.batch_bucket, key.block - 1, selector.top_k),
                    DType::F32,
                    device,
                )?),
            )
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            token_ids,
            rope_indices,
            anchors,
            block_tables,
            slot_mapping,
            cumulative_kv_lens,
            cumulative_query_lens,
            output_tokens,
            selector_inverse_temperatures,
            selector_uniforms,
            output_candidate_ids,
            output_candidate_probs,
        })
    }

    pub(super) fn metadata(&self) -> WindowedKvBatchTensors {
        WindowedKvBatchTensors {
            block_tables: self.block_tables.as_detached_tensor(),
            slot_mapping: self.slot_mapping.as_detached_tensor(),
            cumulative_query_lens: self.cumulative_query_lens.clone(),
            cumulative_kv_lens: self.cumulative_kv_lens.as_detached_tensor(),
        }
    }

    pub(super) fn update(
        &self,
        rows: &DFlashGraphHostRows,
        batch: &WindowedKvBatch,
        staging: &mut crate::pipeline::cuda_graph::CudaGraphHostStaging,
    ) -> Result<()> {
        let location = self.token_ids.device().location();
        staging.update(|staging| {
            staging.copy_from_u32_slice(
                "dflash_token_ids",
                location,
                &rows.token_ids,
                &self.token_ids,
            )?;
            staging.copy_from_u32_slice(
                "dflash_rope_indices",
                location,
                &rows.rope_indices,
                &self.rope_indices,
            )?;
            staging.copy_from_u32_slice(
                "dflash_anchors",
                location,
                &rows.anchors,
                &self.anchors,
            )?;
            staging.copy_from_u32_slice(
                "dflash_block_tables",
                location,
                batch.block_tables_for_graph(),
                &self.block_tables,
            )?;
            staging.copy_from_i64_slice(
                "dflash_slot_mapping",
                location,
                batch.slot_mapping_for_graph(),
                &self.slot_mapping,
            )?;
            staging.copy_from_u32_slice(
                "dflash_cumulative_kv_lens",
                location,
                batch.cumulative_kv_lens_for_graph(),
                &self.cumulative_kv_lens,
            )?;
            match (
                &rows.selector_inverse_temperatures,
                &rows.selector_uniforms,
                &self.selector_inverse_temperatures,
                &self.selector_uniforms,
            ) {
                (
                    Some(inverse_temperatures),
                    Some(uniforms),
                    Some(dst_temperatures),
                    Some(dst_uniforms),
                ) => {
                    staging.copy_from_f32_slice(
                        "dflash_selector_inverse_temperatures",
                        location,
                        inverse_temperatures,
                        dst_temperatures,
                    )?;
                    staging.copy_from_f32_slice(
                        "dflash_selector_uniforms",
                        location,
                        uniforms,
                        dst_uniforms,
                    )
                }
                (None, None, None, None) => Ok(()),
                _ => candle_core::bail!("DFlash graph selector sampling state changed"),
            }
        })
    }
}

#[cfg(cuda_flash_attn)]
impl DFlashCudaGraphEntry {
    pub(super) fn matches_dependencies(
        &self,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> bool {
        Arc::ptr_eq(&self.token_embedding, token_embedding) && Arc::ptr_eq(&self.lm_head, lm_head)
    }

    pub(super) fn launch(&mut self, real_batch: usize) -> Result<DFlashProposalBatch> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::Replay);
        let phase_timer = CudaPhaseTimer::start(self.graph.stream())?;
        self.graph.launch()?;
        self.staging.record_graph_complete()?;
        let output = DFlashCudaGraphOutput {
            tokens: self.buffers.output_tokens.as_detached_tensor(),
            candidate_ids: self
                .buffers
                .output_candidate_ids
                .as_ref()
                .map(Var::as_detached_tensor),
            candidate_probs: self
                .buffers
                .output_candidate_probs
                .as_ref()
                .map(Var::as_detached_tensor),
        }
        .finish(real_batch)?;
        if let Some(timer) = phase_timer {
            timer.finish("dflash_graph", real_batch, self.key.block)?;
        }
        graph_event.success();
        record_cuda_graph_dispatch(
            CudaGraphComponent::DFlash,
            CudaGraphDispatchMode::Replay,
            CudaGraphDispatchReason::CacheHit,
        );
        Ok(output)
    }

    pub(super) fn replay(
        &mut self,
        anchors: &[u32],
        start_positions: &[usize],
        sampling: Option<DFlashSamplingInputs<'_>>,
        batch: &WindowedKvBatch,
        real_batch: usize,
    ) -> Result<DFlashProposalBatch> {
        self.host_rows.update(DFlashGraphHostInput {
            anchors,
            start_positions,
            mask_token_id: self.mask_token_id,
            block: self.key.block,
            batch_bucket: self.key.batch_bucket,
            sampling,
        })?;
        self.buffers
            .update(&self.host_rows, batch, &mut self.staging)?;
        self.staging.order_before_graph()?;
        self.launch(real_batch)
    }

    pub(super) fn release(self) -> (Arc<CudaStream>, Result<()>) {
        let Self {
            key: _,
            staging,
            buffers,
            token_embedding,
            lm_head,
            mask_token_id: _,
            host_rows,
            graph,
        } = self;
        let release = release_dflash_cuda_graph_resources(graph, (staging, buffers, host_rows));
        drop((token_embedding, lm_head));
        release
    }
}

#[cfg(cuda_flash_attn)]
pub(super) fn release_dflash_cuda_graph_resources<T>(
    graph: crate::pipeline::cuda_graph::CudaGraphHandle,
    resources: T,
) -> (Arc<CudaStream>, Result<()>) {
    let stream = graph.stream().clone();
    let mut release_result = stream
        .synchronize()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("DFlash CUDA graph entry release wait failed"));
    drop(resources);
    if let Err(err) = stream.context().check_err() {
        if release_result.is_ok() {
            release_result = Err(candle_core::Error::wrap(err)
                .context("DFlash CUDA graph entry storage release failed"));
        }
    }
    let storage_result = stream
        .synchronize()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("DFlash CUDA graph entry storage release wait failed"));
    if release_result.is_ok() {
        release_result = storage_result;
    }
    drop(graph);
    (stream, release_result)
}

#[cfg(cuda_flash_attn)]
pub(super) fn release_dflash_cuda_graphs(entries: Vec<DFlashCudaGraphEntry>) {
    let mut streams = Vec::new();
    for entry in entries {
        let (stream, release_result) = entry.release();
        if let Err(err) = release_result {
            tracing::warn!("Failed to release DFlash CUDA graph entry storage: {err:?}");
        }
        if !streams.iter().any(|known: &Arc<CudaStream>| {
            known.context().cu_device() == stream.context().cu_device()
        }) {
            streams.push(stream);
        }
    }
    for stream in streams {
        if let Err(err) = crate::pipeline::cuda_graph::trim_cuda_graph_memory(&stream) {
            tracing::warn!("Failed to trim released DFlash CUDA graph memory: {err:?}");
        }
    }
}

#[cfg(cuda_flash_attn)]
pub(super) fn release_dflash_cuda_graph(entry: DFlashCudaGraphEntry) {
    release_dflash_cuda_graphs(vec![entry]);
}

#[cfg(cuda_flash_attn)]
impl Drop for DFlashCudaGraphState {
    fn drop(&mut self) {
        let entries = std::mem::take(&mut self.entries);
        record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, 0);
        release_dflash_cuda_graphs(entries);
    }
}

#[cfg(cuda_flash_attn)]
impl DFlashCudaGraphState {
    pub(super) fn evict_lru_for_memory_pressure(&mut self, max_entries: usize) -> usize {
        let entries = drain_dflash_lru_entries(&mut self.entries, max_entries);
        let evicted = entries.len();
        if evicted == 0 {
            return 0;
        }
        record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, self.entries.len());
        release_dflash_cuda_graphs(entries);
        record_cuda_graph_evictions(
            CudaGraphComponent::DFlash,
            CudaGraphEvictionReason::MemoryPressure,
            evicted,
        );
        evicted
    }

    pub(super) fn store(&mut self, entry: DFlashCudaGraphEntry) {
        if let Some(evicted) = take_cuda_graph_capacity_eviction(&mut self.entries, self.capacity) {
            record_cuda_graph_evictions(
                CudaGraphComponent::DFlash,
                CudaGraphEvictionReason::Capacity,
                1,
            );
            release_dflash_cuda_graph(evicted);
        }
        self.warmed.remove(&entry.key);
        self.failed.remove(&entry.key);
        self.entries.push(entry);
        record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, self.entries.len());
    }

    pub(super) fn retire_failed_entry(
        &mut self,
        entry: DFlashCudaGraphEntry,
        operation: &str,
        failure: candle_core::Error,
    ) -> Result<()> {
        let key = entry.key;
        let synchronize_result = entry
            .graph
            .stream()
            .synchronize()
            .map_err(candle_core::Error::wrap);
        self.failed.insert(key);
        self.warmed.remove(&key);
        record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, self.entries.len());
        release_dflash_cuda_graph(entry);
        match synchronize_result {
            Ok(()) => {
                tracing::warn!(
                    batch_bucket = key.batch_bucket,
                    block = key.block,
                    "DFlash CUDA graph {operation} failed; disabled this shape and retrying eagerly: {failure:?}"
                );
                Ok(())
            }
            Err(synchronize_err) => Err(candle_core::Error::msg(format!(
                "DFlash CUDA graph {operation} failed: {failure}; recovery synchronization failed: {synchronize_err}"
            ))),
        }
    }

    pub(super) fn eager(
        model: &DFlashDraftModel,
        key: DFlashCudaGraphKey,
        rows: &DFlashGraphHostRows,
        attention_batch: &WindowedKvBatch,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
        real_batch: usize,
    ) -> Result<DFlashProposalBatch> {
        let buffers = DFlashCudaGraphBuffers::new(key, rows, attention_batch, model)?;
        let Device::Cuda(cuda_device) = &model.device else {
            candle_core::bail!("DFlash CUDA graph expected a CUDA device");
        };
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        model
            .cuda_graph_output(key, &buffers, attention_batch, token_embedding, lm_head)?
            .finish(real_batch)
    }

    pub(super) fn capture(
        model: &DFlashDraftModel,
        key: DFlashCudaGraphKey,
        host_rows: DFlashGraphHostRows,
        attention_batch: &WindowedKvBatch,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<DFlashCudaGraphEntry> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::Capture);
        let buffers = DFlashCudaGraphBuffers::new(key, &host_rows, attention_batch, model)?;
        model.device.synchronize()?;
        let Device::Cuda(cuda_device) = &model.device else {
            candle_core::bail!("DFlash CUDA graph expected a CUDA device");
        };
        let stream = cuda_device.cuda_stream();
        let _memory_pool_guard =
            crate::pipeline::cuda_graph::prepare_cuda_graph_memory_pool(&stream)?;
        let restore_event_tracking =
            crate::pipeline::cuda_graph::disable_event_tracking_for_capture(&stream);
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        if let Err(err) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                &stream,
                restore_event_tracking,
            );
            return Err(candle_core::Error::msg(err.to_string())
                .context("DFlash CUDA graph begin capture failed"));
        }

        let capture_result =
            model.cuda_graph_output(key, &buffers, attention_batch, token_embedding, lm_head);
        match capture_result {
            Ok(output) => {
                let copy_result = (|| {
                    crate::cuda::graph::copy_tensor(
                        &output.tokens,
                        &buffers.output_tokens.as_detached_tensor(),
                    )?;
                    match (
                        output.candidate_ids,
                        output.candidate_probs,
                        &buffers.output_candidate_ids,
                        &buffers.output_candidate_probs,
                    ) {
                        (Some(ids), Some(probs), Some(dst_ids), Some(dst_probs)) => {
                            crate::cuda::graph::copy_tensor(&ids, &dst_ids.as_detached_tensor())?;
                            crate::cuda::graph::copy_tensor(&probs, &dst_probs.as_detached_tensor())
                        }
                        (None, None, None, None) => Ok(()),
                        _ => candle_core::bail!("DFlash graph selector output shape changed"),
                    }
                })();
                if let Err(err) = copy_result {
                    crate::pipeline::cuda_graph::end_cuda_capture_discard(&stream);
                    crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                        &stream,
                        restore_event_tracking,
                    );
                    return Err(err.context("DFlash CUDA graph output copy capture failed"));
                }
            }
            Err(err) => {
                crate::pipeline::cuda_graph::end_cuda_capture_discard(&stream);
                crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                    &stream,
                    restore_event_tracking,
                );
                return Err(err.context("DFlash CUDA graph forward capture failed"));
            }
        }

        let graph = match crate::pipeline::cuda_graph::CudaGraphHandle::end_capture(&stream) {
            Ok(Some(graph)) => graph,
            Ok(None) => {
                crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                    &stream,
                    restore_event_tracking,
                );
                return Err(candle_core::Error::msg(
                    "DFlash CUDA graph capture returned no graph",
                ));
            }
            Err(err) => {
                crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                    &stream,
                    restore_event_tracking,
                );
                return Err(err);
            }
        };
        crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
            &stream,
            restore_event_tracking,
        );
        graph.upload()?;
        let staging = crate::pipeline::cuda_graph::CudaGraphHostStaging::new(stream)?;
        let entry = DFlashCudaGraphEntry {
            key,
            staging,
            buffers,
            token_embedding: token_embedding.clone(),
            lm_head: lm_head.clone(),
            mask_token_id: model.mask_token_id,
            host_rows,
            graph,
        };
        graph_event.success();
        Ok(entry)
    }

    pub(super) fn run(
        &mut self,
        run: DFlashCudaGraphRun<'_>,
    ) -> Result<Option<DFlashProposalBatch>> {
        let DFlashCudaGraphRun {
            model,
            key,
            anchors,
            start_positions,
            sampling,
            attention_batch,
            token_embedding,
            lm_head,
            real_batch,
        } = run;
        if let Some(position) = self.entries.iter().position(|entry| {
            entry.key == key && entry.matches_dependencies(token_embedding, lm_head)
        }) {
            let mut entry = self.entries.remove(position);
            match entry.replay(
                anchors,
                start_positions,
                sampling,
                attention_batch,
                real_batch,
            ) {
                Ok(tokens) => {
                    self.entries.push(entry);
                    return Ok(Some(tokens));
                }
                Err(err) => {
                    self.retire_failed_entry(entry, "replay", err)?;
                    let graph_event = CudaGraphEventGuard::new(
                        CudaGraphComponent::DFlash,
                        CudaGraphEvent::EagerFallback,
                    );
                    let result = Self::eager(
                        model,
                        key,
                        &dflash_graph_host_rows(DFlashGraphHostInput {
                            anchors,
                            start_positions,
                            mask_token_id: model.mask_token_id,
                            block: key.block,
                            batch_bucket: key.batch_bucket,
                            sampling,
                        })?,
                        attention_batch,
                        token_embedding,
                        lm_head,
                        real_batch,
                    )
                    .map(Some)
                    .map_err(|err| err.context("DFlash eager fallback after graph replay failed"));
                    if result.is_ok() {
                        graph_event.success();
                    }
                    return result;
                }
            }
        }
        let mismatched = self
            .entries
            .iter()
            .position(|entry| entry.key == key)
            .map(|position| self.entries.remove(position));
        if let Some(entry) = mismatched {
            record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, self.entries.len());
            release_dflash_cuda_graph(entry);
        }
        if self.failed.contains(&key) {
            return Ok(None);
        }
        if self.warmed.insert(key) {
            let graph_event =
                CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::EagerFallback);
            let rows = dflash_graph_host_rows(DFlashGraphHostInput {
                anchors,
                start_positions,
                mask_token_id: model.mask_token_id,
                block: key.block,
                batch_bucket: key.batch_bucket,
                sampling,
            })?;
            let result = Self::eager(
                model,
                key,
                &rows,
                attention_batch,
                token_embedding,
                lm_head,
                real_batch,
            )
            .map(Some);
            if result.is_ok() {
                graph_event.success();
            }
            return result;
        }

        let rows = dflash_graph_host_rows(DFlashGraphHostInput {
            anchors,
            start_positions,
            mask_token_id: model.mask_token_id,
            block: key.block,
            batch_bucket: key.batch_bucket,
            sampling,
        })?;
        let mut entry =
            match Self::capture(model, key, rows, attention_batch, token_embedding, lm_head) {
                Ok(entry) => entry,
                Err(err) => {
                    self.failed.insert(key);
                    tracing::warn!(
                        batch_bucket = key.batch_bucket,
                        block = key.block,
                        "DFlash CUDA graph capture disabled for this shape: {err:?}"
                    );
                    return Ok(None);
                }
            };
        let tokens = match entry.launch(real_batch) {
            Ok(tokens) => tokens,
            Err(err) => {
                self.retire_failed_entry(entry, "first launch", err)?;
                let graph_event = CudaGraphEventGuard::new(
                    CudaGraphComponent::DFlash,
                    CudaGraphEvent::EagerFallback,
                );
                let result = Self::eager(
                    model,
                    key,
                    &dflash_graph_host_rows(DFlashGraphHostInput {
                        anchors,
                        start_positions,
                        mask_token_id: model.mask_token_id,
                        block: key.block,
                        batch_bucket: key.batch_bucket,
                        sampling,
                    })?,
                    attention_batch,
                    token_embedding,
                    lm_head,
                    real_batch,
                )
                .map(Some)
                .map_err(|err| {
                    err.context("DFlash eager fallback after first graph launch failed")
                });
                if result.is_ok() {
                    graph_event.success();
                }
                return result;
            }
        };
        tracing::debug!(
            batch_bucket = key.batch_bucket,
            block = key.block,
            "Captured DFlash CUDA graph"
        );
        self.store(entry);
        Ok(Some(tokens))
    }
}
