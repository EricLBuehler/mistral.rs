use super::*;

pub(super) struct CudaDecodeGraphMetadataInput<'a> {
    pub(super) metadata: &'a PagedAttentionInputMetadata,
    pub(super) seqlen_offsets: &'a [usize],
    pub(super) position_ids: &'a [usize],
    pub(super) seq_len: usize,
    pub(super) block_size: usize,
    pub(super) kv_cache: &'a [(Tensor, Tensor)],
    pub(super) model_metadata: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
    pub(super) activation_dtype: DType,
}

pub(crate) struct CudaDecodeGraphMetadataBuffers {
    pub(super) requirements: PagedDecodeMetadataRequirements,
    pub(super) flashinfer_views_alias: bool,
    pub(super) slot_mappings: CudaGraphVarMap,
    pub(super) block_tables: Option<CudaGraphVarMap>,
    pub(super) context_lens: Option<CudaGraphVarMap>,
    pub(super) full_block_tables: Option<CudaGraphVarMap>,
    pub(super) full_context_lens: Option<CudaGraphVarMap>,
    pub(super) paged_kv_indptr: Option<CudaGraphVarMap>,
    pub(super) paged_kv_indices: Option<CudaGraphVarMap>,
    pub(super) paged_kv_last_page_len: Option<CudaGraphVarMap>,
    pub(super) full_paged_kv_indptr: Option<CudaGraphVarMap>,
    pub(super) full_paged_kv_indices: Option<CudaGraphVarMap>,
    pub(super) full_paged_kv_last_page_len: Option<CudaGraphVarMap>,
    pub(super) paged_kv_request_indices: Option<CudaGraphVarMap>,
    pub(super) paged_kv_tile_indices: Option<CudaGraphVarMap>,
    pub(super) paged_kv_o_indptr: Option<CudaGraphVarMap>,
    pub(super) paged_kv_chunk_size: Option<CudaGraphVarMap>,
    pub(super) paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    pub(super) full_paged_kv_request_indices: Option<CudaGraphVarMap>,
    pub(super) full_paged_kv_tile_indices: Option<CudaGraphVarMap>,
    pub(super) full_paged_kv_o_indptr: Option<CudaGraphVarMap>,
    pub(super) full_paged_kv_chunk_size: Option<CudaGraphVarMap>,
    pub(super) full_paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    pub(super) flashinfer_decode_tmp_v: Option<HashMap<DeviceLocation, Tensor>>,
    pub(super) flashinfer_decode_tmp_s: Option<HashMap<DeviceLocation, Tensor>>,
    pub(super) fa3_decode: Option<Fa3DecodeState>,
    pub(super) rope_positions: CudaGraphVarMap,
}

impl CudaDecodeGraphMetadataBuffers {
    pub(super) fn new(
        input: CudaDecodeGraphMetadataInput<'_>,
    ) -> candle_core::Result<(Self, PagedAttentionInputMetadata)> {
        let CudaDecodeGraphMetadataInput {
            metadata,
            seqlen_offsets,
            position_ids,
            seq_len,
            block_size,
            kv_cache,
            model_metadata,
            activation_dtype,
        } = input;
        let slot_mappings = var_map_from_tensor_map(&metadata.slot_mappings)?;
        if seqlen_offsets.len() != position_ids.len() {
            candle_core::bail!(
                "CUDA graph decode has {} KV offsets but {} position ends",
                seqlen_offsets.len(),
                position_ids.len()
            );
        }
        let rope_positions =
            rope_positions_var_map(&metadata.slot_mappings, position_ids, seq_len)?;
        let (flashinfer_decode_tmp_v, flashinfer_decode_tmp_s) = flashinfer_decode_scratch_maps(
            metadata,
            seqlen_offsets.len(),
            kv_cache,
            model_metadata,
            activation_dtype,
        )?;
        let fa3_decode = metadata
            .flashinfer
            .as_ref()
            .map(|flashinfer| {
                make_fa3_decode_state(
                    flashinfer,
                    seqlen_offsets.len(),
                    seq_len,
                    kv_cache,
                    model_metadata,
                    activation_dtype,
                )
            })
            .transpose()?
            .flatten();
        let flashinfer_views_alias = flashinfer_views_alias(metadata);
        let requirements = PagedDecodeMetadataRequirements::graph(
            metadata.block_tables.is_some(),
            metadata.context_lens.is_some(),
            metadata.flashinfer.is_some(),
            metadata.flashinfer.is_some(),
        );
        let mut buffers = Self {
            requirements,
            flashinfer_views_alias,
            slot_mappings,
            block_tables: option_var_map_from_tensor_map(metadata.block_tables.as_ref())?,
            context_lens: option_var_map_from_tensor_map(metadata.context_lens.as_ref())?,
            full_block_tables: option_var_map_from_tensor_map_if_distinct(
                metadata.full_block_tables.as_ref(),
                flashinfer_views_alias,
            )?,
            full_context_lens: option_var_map_from_tensor_map_if_distinct(
                metadata.full_context_lens.as_ref(),
                flashinfer_views_alias,
            )?,
            paged_kv_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
            )?,
            paged_kv_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
            )?,
            paged_kv_last_page_len: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
            )?,
            full_paged_kv_indptr: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                flashinfer_views_alias,
            )?,
            full_paged_kv_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_last_page_len: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
                flashinfer_views_alias,
            )?,
            paged_kv_request_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
            )?,
            paged_kv_tile_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
            )?,
            paged_kv_o_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.o_indptr),
            )?,
            paged_kv_chunk_size: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
            )?,
            paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
            )?,
            full_paged_kv_request_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_tile_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_o_indptr: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.o_indptr),
                flashinfer_views_alias,
            )?,
            full_paged_kv_chunk_size: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
                flashinfer_views_alias,
            )?,
            full_paged_kv_block_valid_mask: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
                flashinfer_views_alias,
            )?,
            flashinfer_decode_tmp_v,
            flashinfer_decode_tmp_s,
            fa3_decode,
            rope_positions,
        };
        if flashinfer_views_alias {
            buffers.full_block_tables = buffers.block_tables.clone();
            buffers.full_context_lens = buffers.context_lens.clone();
            buffers.full_paged_kv_indptr = buffers.paged_kv_indptr.clone();
            buffers.full_paged_kv_indices = buffers.paged_kv_indices.clone();
            buffers.full_paged_kv_last_page_len = buffers.paged_kv_last_page_len.clone();
            buffers.full_paged_kv_request_indices = buffers.paged_kv_request_indices.clone();
            buffers.full_paged_kv_tile_indices = buffers.paged_kv_tile_indices.clone();
            buffers.full_paged_kv_o_indptr = buffers.paged_kv_o_indptr.clone();
            buffers.full_paged_kv_chunk_size = buffers.paged_kv_chunk_size.clone();
            buffers.full_paged_kv_block_valid_mask = buffers.paged_kv_block_valid_mask.clone();
        }
        let metadata = buffers.metadata_from(metadata, block_size);
        Ok((buffers, metadata))
    }

    pub(super) fn finish_capture(&mut self, metadata: &PagedAttentionInputMetadata) {
        let tile_plan_used = metadata
            .flashinfer
            .as_ref()
            .is_some_and(FlashInferMetadata::decode_tile_plan_was_used);
        self.requirements = PagedDecodeMetadataRequirements::graph(
            self.block_tables.is_some(),
            self.context_lens.is_some(),
            self.fa3_decode.is_some() || tile_plan_used,
            tile_plan_used,
        );
    }

    pub(super) fn copy_from(
        &mut self,
        metadata: &PagedAttentionInputMetadata,
        position_ids: &[usize],
        seq_len: usize,
        host_staging: &mut CudaGraphHostStaging,
    ) -> candle_core::Result<()> {
        let graph_update = if metadata.has_host_staged_decode_tensors() {
            Some(
                metadata
                    .decode_rows
                    .as_ref()
                    .expect("host-staged decode metadata requires source rows")
                    .build_graph_update(self.requirements)
                    .map_err(candle_core::Error::msg)?,
            )
        } else {
            None
        };
        let metadata = graph_update.as_ref().unwrap_or(metadata);
        copy_var_map(
            &self.slot_mappings,
            &metadata.slot_mappings,
            "slot_mappings",
            host_staging,
        )?;
        if self.requirements.context_lens {
            copy_option_var_map(
                &self.context_lens,
                metadata.context_lens.as_ref(),
                "context_lens",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_context_lens,
                    metadata.full_context_lens.as_ref(),
                    "full_context_lens",
                    host_staging,
                )?;
            }
        }
        if self.requirements.block_tables {
            copy_option_var_map(
                &self.block_tables,
                metadata.block_tables.as_ref(),
                "block_tables",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_block_tables,
                    metadata.full_block_tables.as_ref(),
                    "full_block_tables",
                    host_staging,
                )?;
            }
        }
        if self.requirements.flashinfer_paged_kv {
            copy_option_var_map(
                &self.paged_kv_last_page_len,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
                "paged_kv_last_page_len",
                host_staging,
            )?;
            copy_option_var_map(
                &self.paged_kv_indptr,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
                "paged_kv_indptr",
                host_staging,
            )?;
            copy_option_var_map(
                &self.paged_kv_indices,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
                "paged_kv_indices",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_paged_kv_last_page_len,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
                    "full_paged_kv_last_page_len",
                    host_staging,
                )?;
                copy_option_var_map(
                    &self.full_paged_kv_indptr,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                    "full_paged_kv_indptr",
                    host_staging,
                )?;
                copy_option_var_map(
                    &self.full_paged_kv_indices,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                    "full_paged_kv_indices",
                    host_staging,
                )?;
            }
        }
        if self.requirements.flashinfer_tile_plan {
            copy_flashinfer_tile_plan(
                metadata,
                false,
                FlashInferTilePlanVars {
                    request_indices: &self.paged_kv_request_indices,
                    kv_tile_indices: &self.paged_kv_tile_indices,
                    o_indptr: &self.paged_kv_o_indptr,
                    kv_chunk_size: &self.paged_kv_chunk_size,
                    block_valid_mask: &self.paged_kv_block_valid_mask,
                },
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_flashinfer_tile_plan(
                    metadata,
                    true,
                    FlashInferTilePlanVars {
                        request_indices: &self.full_paged_kv_request_indices,
                        kv_tile_indices: &self.full_paged_kv_tile_indices,
                        o_indptr: &self.full_paged_kv_o_indptr,
                        kv_chunk_size: &self.full_paged_kv_chunk_size,
                        block_valid_mask: &self.full_paged_kv_block_valid_mask,
                    },
                    host_staging,
                )?;
            }
        }
        copy_rope_positions(&self.rope_positions, position_ids, seq_len, host_staging)?;
        Ok(())
    }

    pub(super) fn flashinfer_metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> Option<FlashInferMetadata> {
        let original = metadata.flashinfer.as_ref()?;
        let logical = FlashInferPagedAttentionView {
            block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            max_context_len: original
                .views
                .logical
                .max_context_len
                .or_else(|| bucket_context_len_from_vars(&self.full_block_tables, block_size)),
            paged_kv: flashinfer_paged_kv_from_vars(
                &self.full_paged_kv_indptr,
                &self.full_paged_kv_indices,
                &self.full_paged_kv_last_page_len,
            )?,
            tile_plan: flashinfer_tile_plan_from_vars(
                &self.full_paged_kv_request_indices,
                &self.full_paged_kv_tile_indices,
                &self.full_paged_kv_o_indptr,
                &self.full_paged_kv_chunk_size,
                &self.full_paged_kv_block_valid_mask,
            )?,
        };
        let sliding = if let Some(view) = original.views.sliding.as_ref() {
            Some(FlashInferPagedAttentionView {
                block_tables: option_tensor_map_from_var_map(&self.block_tables),
                context_lens: option_tensor_map_from_var_map(&self.context_lens),
                max_context_len: view
                    .max_context_len
                    .or_else(|| bucket_context_len_from_vars(&self.block_tables, block_size)),
                paged_kv: flashinfer_paged_kv_from_vars(
                    &self.paged_kv_indptr,
                    &self.paged_kv_indices,
                    &self.paged_kv_last_page_len,
                )?,
                tile_plan: flashinfer_tile_plan_from_vars(
                    &self.paged_kv_request_indices,
                    &self.paged_kv_tile_indices,
                    &self.paged_kv_o_indptr,
                    &self.paged_kv_chunk_size,
                    &self.paged_kv_block_valid_mask,
                )?,
            })
        } else {
            None
        };

        Some(
            FlashInferMetadata {
                views: FlashInferPagedAttentionViews { logical, sliding },
                decode_tmp_v: self.flashinfer_decode_tmp_v.clone(),
                decode_tmp_s: self.flashinfer_decode_tmp_s.clone(),
                fa3_decode: self.fa3_decode.clone(),
                decode_tile_plan_used: None,
            }
            .track_decode_tile_plan(),
        )
    }

    pub(super) fn metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> PagedAttentionInputMetadata {
        PagedAttentionInputMetadata {
            block_tables: option_tensor_map_from_var_map(&self.block_tables),
            context_lens: option_tensor_map_from_var_map(&self.context_lens),
            block_size: metadata.block_size,
            paged_context_lens_cpu: metadata.paged_context_lens_cpu.clone(),
            full_paged_context_lens_cpu: metadata.full_paged_context_lens_cpu.clone(),
            slot_mappings: tensor_map_from_var_map(&self.slot_mappings),
            max_context_len: graph_context_len(
                metadata.max_context_len,
                bucket_context_len_from_vars(&self.block_tables, block_size),
            ),
            full_block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            full_context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            full_max_context_len: graph_context_len(
                metadata.full_max_context_len,
                bucket_context_len_from_vars(&self.full_block_tables, block_size),
            ),
            is_first_prompt_chunk: metadata.is_first_prompt_chunk,
            is_final_prompt_chunk: metadata.is_final_prompt_chunk,
            needs_logits: metadata.needs_logits,
            prompt_chunk_attention_policy: metadata.prompt_chunk_attention_policy,
            has_noncausal_mm_context: metadata.has_noncausal_mm_context,
            prefix_gather_workspace_limit: metadata.prefix_gather_workspace_limit,
            attention_memory: metadata.attention_memory.clone(),
            prefix_masks: metadata.prefix_masks.clone(),
            mm_prefix_ranges: metadata.mm_prefix_ranges.clone(),
            full_mm_prefix_ranges: metadata.full_mm_prefix_ranges.clone(),
            prefill_attention_heads: metadata.prefill_attention_heads,
            prefill_key_value_heads: metadata.prefill_key_value_heads,
            prefill_head_dim: metadata.prefill_head_dim,
            flashinfer: self.flashinfer_metadata_from(metadata, block_size),
            rope_positions: Some(tensor_map_from_var_map(&self.rope_positions)),
            num_cached_tokens: metadata.num_cached_tokens.clone(),
            query_lens: metadata.query_lens.clone(),
            cu_seqlens_q: metadata.cu_seqlens_q.clone(),
            cu_seqlens_kv: metadata.cu_seqlens_kv.clone(),
            decode_rows: metadata.decode_rows.clone(),
        }
    }
}

pub(super) fn flashinfer_decode_scratch_maps(
    metadata: &PagedAttentionInputMetadata,
    batch: usize,
    kv_cache: &[(Tensor, Tensor)],
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
    activation_dtype: DType,
) -> candle_core::Result<FlashInferDecodeScratchMaps> {
    let Some(model_metadata) = model_metadata else {
        return Ok((None, None));
    };
    let split_rows = flashinfer_split_rows(metadata, batch)?;
    if split_rows.is_empty() {
        return Ok((None, None));
    }

    let mut specs: HashMap<DeviceLocation, (Device, DType, usize, usize)> = HashMap::new();
    let layer_count = model_metadata.num_layers().min(kv_cache.len());
    for (layer_idx, (key_cache, value_cache)) in kv_cache.iter().enumerate().take(layer_count) {
        if model_metadata.attention_backend_kind_for_layer(layer_idx)
            != AttentionBackendKind::FlashInfer
        {
            continue;
        }
        let location = key_cache.device().location();
        if !split_rows.contains_key(&location) {
            continue;
        }
        if key_cache.dtype() != value_cache.dtype() {
            candle_core::bail!("FlashInfer graph scratch expects matching KV cache dtypes");
        }
        let (_, _, _, head_dim) = key_cache.dims4()?;
        let num_qo_heads = model_metadata.num_attn_heads_for_layer(layer_idx);
        let entry = specs.entry(location).or_insert((
            key_cache.device().clone(),
            activation_dtype,
            num_qo_heads,
            head_dim,
        ));
        if entry.1 != activation_dtype {
            candle_core::bail!("FlashInfer graph scratch expects one activation dtype per device");
        }
        entry.2 = entry.2.max(num_qo_heads);
        entry.3 = entry.3.max(head_dim);
    }

    let mut tmp_v = HashMap::new();
    let mut tmp_s = HashMap::new();
    for (location, rows) in split_rows {
        let Some((device, dtype, num_qo_heads, head_dim)) = specs.get(&location) else {
            continue;
        };
        tmp_v.insert(location, unsafe {
            Tensor::empty((rows, *num_qo_heads, *head_dim), *dtype, device)?
        });
        tmp_s.insert(location, unsafe {
            Tensor::empty((rows, *num_qo_heads), DType::F32, device)?
        });
    }

    if tmp_v.is_empty() {
        Ok((None, None))
    } else {
        Ok((Some(tmp_v), Some(tmp_s)))
    }
}

pub(super) fn flashinfer_split_rows(
    metadata: &PagedAttentionInputMetadata,
    batch: usize,
) -> candle_core::Result<HashMap<DeviceLocation, usize>> {
    let mut rows = HashMap::new();
    collect_flashinfer_split_rows(
        flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
        batch,
        &mut rows,
    )?;
    collect_flashinfer_split_rows(
        flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
        batch,
        &mut rows,
    )?;
    Ok(rows)
}

pub(super) fn device_location_sort_key(location: &DeviceLocation) -> (u8, usize) {
    match location {
        DeviceLocation::Cpu => (0, 0),
        DeviceLocation::Cuda { gpu_id } => (1, *gpu_id),
        DeviceLocation::Metal { gpu_id } => (2, *gpu_id),
    }
}

pub(super) fn push_graph_tensor_keys(
    name: &'static str,
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    keys: &mut Vec<CudaGraphTensorKey>,
) {
    if let Some(map) = map {
        keys.extend(map.iter().map(|(location, tensor)| CudaGraphTensorKey {
            name,
            location: *location,
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
        }));
    }
}

pub(super) fn push_flashinfer_graph_tensor_keys(
    metadata: &PagedAttentionInputMetadata,
    keys: &mut Vec<CudaGraphTensorKey>,
) {
    let paged = flashinfer_paged_view(metadata);
    let full = flashinfer_full_view(metadata);
    push_graph_tensor_keys(
        "paged_kv_indptr",
        paged.map(|view| &view.paged_kv.indptr),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_indices",
        paged.map(|view| &view.paged_kv.indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_last_page_len",
        paged.map(|view| &view.paged_kv.last_page_len),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_indptr",
        full.map(|view| &view.paged_kv.indptr),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_indices",
        full.map(|view| &view.paged_kv.indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_last_page_len",
        full.map(|view| &view.paged_kv.last_page_len),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_request_indices",
        paged.map(|view| &view.tile_plan.request_indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_tile_indices",
        paged.map(|view| &view.tile_plan.kv_tile_indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_o_indptr",
        paged.map(|view| &view.tile_plan.o_indptr),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_chunk_size",
        paged.map(|view| &view.tile_plan.kv_chunk_size),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_block_valid_mask",
        paged.map(|view| &view.tile_plan.block_valid_mask),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_request_indices",
        full.map(|view| &view.tile_plan.request_indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_tile_indices",
        full.map(|view| &view.tile_plan.kv_tile_indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_o_indptr",
        full.map(|view| &view.tile_plan.o_indptr),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_chunk_size",
        full.map(|view| &view.tile_plan.kv_chunk_size),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_block_valid_mask",
        full.map(|view| &view.tile_plan.block_valid_mask),
        keys,
    );
}

pub(super) fn flashinfer_paged_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    let views = &metadata.flashinfer.as_ref()?.views;
    Some(views.sliding.as_ref().unwrap_or(&views.logical))
}

pub(super) fn flashinfer_full_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    Some(&metadata.flashinfer.as_ref()?.views.logical)
}

pub(super) fn flashinfer_views_alias(metadata: &PagedAttentionInputMetadata) -> bool {
    metadata
        .flashinfer
        .as_ref()
        .is_some_and(|flashinfer| flashinfer.views.sliding.is_none())
}

pub(super) fn flashinfer_paged_kv_from_vars(
    indptr: &Option<CudaGraphVarMap>,
    indices: &Option<CudaGraphVarMap>,
    last_page_len: &Option<CudaGraphVarMap>,
) -> Option<FlashInferPagedKv> {
    Some(FlashInferPagedKv {
        indptr: option_tensor_map_from_var_map(indptr)?,
        indices: option_tensor_map_from_var_map(indices)?,
        last_page_len: option_tensor_map_from_var_map(last_page_len)?,
    })
}

pub(super) fn flashinfer_tile_plan_from_vars(
    request_indices: &Option<CudaGraphVarMap>,
    kv_tile_indices: &Option<CudaGraphVarMap>,
    o_indptr: &Option<CudaGraphVarMap>,
    kv_chunk_size: &Option<CudaGraphVarMap>,
    block_valid_mask: &Option<CudaGraphVarMap>,
) -> Option<FlashInferTilePlan> {
    Some(FlashInferTilePlan {
        request_indices: option_tensor_map_from_var_map(request_indices)?,
        kv_tile_indices: option_tensor_map_from_var_map(kv_tile_indices)?,
        o_indptr: option_tensor_map_from_var_map(o_indptr)?,
        kv_chunk_size: option_tensor_map_from_var_map(kv_chunk_size)?,
        block_valid_mask: option_tensor_map_from_var_map(block_valid_mask)?,
    })
}

pub(super) fn collect_flashinfer_split_rows(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    batch: usize,
    split_rows: &mut HashMap<DeviceLocation, usize>,
) -> candle_core::Result<()> {
    let Some(map) = map else {
        return Ok(());
    };
    for (location, tensor) in map {
        let rows = tensor.dims1()?;
        if rows > batch {
            split_rows
                .entry(*location)
                .and_modify(|current| *current = (*current).max(rows))
                .or_insert(rows);
        }
    }
    Ok(())
}

pub(super) fn bucket_context_len_from_vars(
    map: &Option<CudaGraphVarMap>,
    block_size: usize,
) -> Option<usize> {
    map.as_ref()
        .and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

pub(super) fn bucket_context_len(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    block_size: usize,
) -> Option<usize> {
    map.and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

pub(super) fn paged_attention_partition_context_len(actual: usize) -> usize {
    actual
        .div_ceil(PAGED_ATTENTION_PARTITION_SIZE)
        .max(1)
        .saturating_mul(PAGED_ATTENTION_PARTITION_SIZE)
}

pub(super) fn graph_context_len(actual: Option<usize>, capacity: Option<usize>) -> Option<usize> {
    match (actual, capacity) {
        (Some(actual), Some(capacity)) => {
            Some(paged_attention_partition_context_len(actual).min(capacity))
        }
        (Some(actual), None) => Some(actual),
        (None, capacity) => capacity,
    }
}

pub(super) fn var_map_from_tensor_map(
    map: &HashMap<DeviceLocation, Tensor>,
) -> candle_core::Result<CudaGraphVarMap> {
    map.iter()
        .map(|(location, tensor)| Ok((*location, Var::from_tensor(tensor)?)))
        .collect()
}

pub(super) fn option_var_map_from_tensor_map(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
) -> candle_core::Result<Option<CudaGraphVarMap>> {
    map.map(var_map_from_tensor_map).transpose()
}

pub(super) fn option_var_map_from_tensor_map_if_distinct(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    aliases_existing: bool,
) -> candle_core::Result<Option<CudaGraphVarMap>> {
    if aliases_existing {
        Ok(None)
    } else {
        option_var_map_from_tensor_map(map)
    }
}

pub(super) fn tensor_map_from_var_map(map: &CudaGraphVarMap) -> HashMap<DeviceLocation, Tensor> {
    map.iter()
        .map(|(location, var)| (*location, var.as_detached_tensor()))
        .collect()
}

pub(super) fn option_tensor_map_from_var_map(
    map: &Option<CudaGraphVarMap>,
) -> Option<HashMap<DeviceLocation, Tensor>> {
    map.as_ref().map(tensor_map_from_var_map)
}

pub(super) fn copy_var_map(
    dst: &CudaGraphVarMap,
    src: &HashMap<DeviceLocation, Tensor>,
    name: &'static str,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    if dst.len() != src.len() {
        candle_core::bail!("{name} device count changed during CUDA graph replay");
    }
    for (location, dst) in dst {
        let src = src
            .get(location)
            .ok_or_else(|| candle_core::Error::msg(format!("{name} missing {location:?}")))?;
        if src.device().is_cpu() && dst.device().is_cuda() {
            host_staging.copy_from(name, *location, src, dst)?;
        } else {
            dst.set(src)?;
        }
    }
    Ok(())
}

pub(super) fn copy_option_var_map(
    dst: &Option<CudaGraphVarMap>,
    src: Option<&HashMap<DeviceLocation, Tensor>>,
    name: &'static str,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    match (dst, src) {
        (Some(dst), Some(src)) => copy_var_map(dst, src, name, host_staging),
        (None, None) => Ok(()),
        _ => candle_core::bail!("{name} changed optional state during CUDA graph replay"),
    }
}

pub(super) struct FlashInferTilePlanVars<'a> {
    pub(super) request_indices: &'a Option<CudaGraphVarMap>,
    pub(super) kv_tile_indices: &'a Option<CudaGraphVarMap>,
    pub(super) o_indptr: &'a Option<CudaGraphVarMap>,
    pub(super) kv_chunk_size: &'a Option<CudaGraphVarMap>,
    pub(super) block_valid_mask: &'a Option<CudaGraphVarMap>,
}

pub(super) fn copy_flashinfer_tile_plan(
    metadata: &PagedAttentionInputMetadata,
    full: bool,
    vars: FlashInferTilePlanVars<'_>,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    let view = if full {
        flashinfer_full_view(metadata)
    } else {
        flashinfer_paged_view(metadata)
    };
    copy_option_var_map(
        vars.request_indices,
        view.map(|view| &view.tile_plan.request_indices),
        if full {
            "full_paged_kv_request_indices"
        } else {
            "paged_kv_request_indices"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.kv_tile_indices,
        view.map(|view| &view.tile_plan.kv_tile_indices),
        if full {
            "full_paged_kv_tile_indices"
        } else {
            "paged_kv_tile_indices"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.o_indptr,
        view.map(|view| &view.tile_plan.o_indptr),
        if full {
            "full_paged_kv_o_indptr"
        } else {
            "paged_kv_o_indptr"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.kv_chunk_size,
        view.map(|view| &view.tile_plan.kv_chunk_size),
        if full {
            "full_paged_kv_chunk_size"
        } else {
            "paged_kv_chunk_size"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.block_valid_mask,
        view.map(|view| &view.tile_plan.block_valid_mask),
        if full {
            "full_paged_kv_block_valid_mask"
        } else {
            "paged_kv_block_valid_mask"
        },
        host_staging,
    )
}

pub(super) fn rope_positions_var_map(
    slot_mappings: &HashMap<DeviceLocation, Tensor>,
    position_ids: &[usize],
    seq_len: usize,
) -> candle_core::Result<CudaGraphVarMap> {
    slot_mappings
        .iter()
        .map(|(location, tensor)| {
            let positions = decode_positions_tensor(position_ids, seq_len, tensor.device())?;
            Ok((*location, Var::from_tensor(&positions)?))
        })
        .collect()
}

pub(super) fn copy_rope_positions(
    dst: &CudaGraphVarMap,
    position_ids: &[usize],
    seq_len: usize,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    let positions = decode_positions_tensor(position_ids, seq_len, &Device::Cpu)?;
    for (location, dst) in dst {
        if dst.device().is_cuda() {
            host_staging.copy_from("rope_positions", *location, &positions, dst)?;
        } else {
            dst.set(&positions)?;
        }
    }
    Ok(())
}
