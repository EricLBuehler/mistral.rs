#![allow(clippy::cast_possible_truncation)]

use std::{any::Any, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use text_models_inputs_processor::PagedAttentionMeta;
use tokenizers::Tokenizer;

use crate::{device_map::DeviceMapper, sequence::Sequence};

#[derive(PartialEq)]
pub enum InputsProcessorType {
    Text,
    Vision,
    Embedding,
}

pub struct InputProcessorOutput {
    pub inputs: Box<dyn Any>,
    pub seq_indices: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct InputsProcessorValidationError(pub(crate) String);

pub(crate) fn is_inputs_processor_validation_error(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        source
            .downcast_ref::<InputsProcessorValidationError>()
            .is_some()
    })
}

#[cfg(test)]
mod validation_error_tests {
    use super::*;

    #[test]
    fn detects_validation_errors_through_context() {
        let error = anyhow::Error::new(InputsProcessorValidationError("bad input".to_string()))
            .context("planning failed");

        assert!(is_inputs_processor_validation_error(&error));
        assert!(!is_inputs_processor_validation_error(&anyhow::anyhow!(
            "internal failure"
        )));
    }
}

/// Processor: Prepare inputs for the model (potentially preparing the images if applicable)
pub trait InputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        _tokenizer: Option<Arc<Tokenizer>>,
        _input_seqs: &mut [&mut Sequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        Ok(())
    }

    /// This should also enable matmul via f16 if prompt and the sequence length is greater than 32.
    /// Otherwise, matmul via f16 is disabled.
    ///
    /// This should return a type which can be downcasted to the proper type as used in `forward_inputs`
    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput>;

    fn get_type(&self) -> InputsProcessorType;
}

// ========================= Test models input processor

pub mod text_models_inputs_processor {
    use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

    use anyhow::Result;
    use candle_core::{DType, Device, DeviceLocation, Tensor, WithDType};
    use tokenizers::Tokenizer;

    use crate::{
        device_map::DeviceMapper,
        flashinfer::{
            decode_split_capacity_pages as flashinfer_decode_split_capacity_pages,
            decode_split_pages as flashinfer_decode_split_pages, flashinfer_metadata,
            flashinfer_paged_kv, flashinfer_tile_plan, flashinfer_view,
            make_paged_kv_decode_tensors, make_paged_kv_decode_tensors_from_lens,
            make_paged_kv_tensors, FlashInferMetadata, FlashInferPagedAttentionView,
            FlashInferPagedAttentionViews,
        },
        get_mut_arcmutex,
        paged_attention::{
            block_aligned_sliding_window_start,
            block_hash::{noncausal_mm_ranges, MultimodalAttentionPolicy},
            block_table_rows::{BlockTableRanges, BlockTableRows, BlockTableSnapshot},
            AttentionBackendKind, KVCacheManager, _PAD_SLOT_ID,
        },
        pipeline::{recurrent_batch_kind_for_input, RecurrentBatchKind},
        sequence::Sequence,
        AdapterLease,
    };

    use super::{InputProcessorOutput, InputsProcessor, InputsProcessorType};

    const CUDA_GRAPH_CONTEXT_BUCKET_MIN_TOKENS: usize = 512;

    fn cuda_graph_context_bucket_tokens(
        required_tokens: usize,
        max_context_len: Option<usize>,
    ) -> usize {
        let required_tokens = required_tokens.max(1);
        let bucket = required_tokens
            .max(CUDA_GRAPH_CONTEXT_BUCKET_MIN_TOKENS)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX);
        max_context_len
            .map(|limit| bucket.min(limit.max(required_tokens)))
            .unwrap_or(bucket)
    }

    fn cuda_graph_block_table_len_with_cap(
        blocks: usize,
        block_size: usize,
        enable_cuda_graph_padding: bool,
        live_context_len: usize,
        max_context_len: Option<usize>,
    ) -> usize {
        if !enable_cuda_graph_padding || !crate::perf_flags::cuda_graphs_enabled() {
            return blocks;
        }
        let required_tokens = live_context_len.max(blocks.saturating_mul(block_size));
        cuda_graph_context_bucket_tokens(required_tokens, max_context_len)
            .div_ceil(block_size)
            .max(blocks)
            .max(1)
    }

    fn _make_tensor_with_pad<D: WithDType>(
        x: Vec<Vec<D>>,
        max_len: usize,
        pad: D,
        device: &Device,
    ) -> Result<Tensor> {
        let mut padded_x = Vec::new();
        for mut x_i in x {
            assert!(x_i.len() <= max_len);
            x_i.extend([pad].repeat(max_len - x_i.len()));
            let shape = (x_i.len(),);
            padded_x.push(Tensor::from_vec(x_i, shape, device)?);
        }
        Tensor::cat(&padded_x[..], 0).map_err(anyhow::Error::msg)
    }

    fn make_block_table_tensor<T: BlockTableRows + ?Sized>(
        rows: &T,
        max_len: usize,
    ) -> Result<Tensor> {
        let mut values = Vec::with_capacity(rows.len() * max_len);
        for row in 0..rows.len() {
            let table = rows.row(row);
            assert!(table.len() <= max_len);
            values.extend(table.iter().map(|&block| block as u32));
            values.extend(std::iter::repeat_n(0, max_len - table.len()));
        }
        Ok(Tensor::from_vec(
            values,
            (rows.len(), max_len),
            &Device::Cpu,
        )?)
    }

    fn decode_metadata_tensor(
        tensor: &Tensor,
        device: &Device,
        stage_on_host: bool,
    ) -> candle_core::Result<Tensor> {
        if stage_on_host {
            Ok(tensor.clone())
        } else {
            tensor.to_device(device)
        }
    }

    #[derive(Clone)]
    pub struct PagedAttentionMeta {
        pub sliding_window: Option<usize>,
        pub block_size: usize,
        pub max_paged_context_len: usize,
        pub attention_backend: AttentionBackendKind,
        pub has_flashinfer_decode_layers: bool,
        pub prefill_attention_heads: usize,
        pub prefill_key_value_heads: usize,
        pub prefill_head_dim: usize,
        pub kv_cache_manager: Arc<tokio::sync::Mutex<KVCacheManager>>,
        pub prompt_chunk_size: Option<usize>,
        pub(crate) scheduled_prompt_chunks:
            Option<Vec<crate::pipeline::prompt_chunks::PromptChunkPlan>>,
        pub prompt_chunk_attention_policy: MultimodalAttentionPolicy,
        pub has_noncausal_mm_context: bool,
        pub prefix_gather_workspace_limit: Option<usize>,
        pub mm_prefix_ranges_by_seq_id: HashMap<usize, Vec<(usize, usize)>>,
        pub full_mm_prefix_ranges_by_seq_id: HashMap<usize, Vec<(usize, usize)>>,
        pub(crate) enable_packed_prefill: bool,
        /// False only for non-final chunks of a chunked prompt; block-diffusion models skip
        /// canvas generation until the prompt is fully encoded.
        pub is_final_prompt_chunk: bool,
    }

    impl PagedAttentionMeta {
        pub(crate) fn set_noncausal_mm_context(&mut self, input_seqs: &[&mut Sequence]) {
            self.set_noncausal_mm_context_views(input_seqs, true);
        }

        pub(crate) fn set_noncausal_mm_context_views(
            &mut self,
            input_seqs: &[&mut Sequence],
            include_full_attention: bool,
        ) {
            self.mm_prefix_ranges_by_seq_id.clear();
            self.full_mm_prefix_ranges_by_seq_id.clear();
            for seq in input_seqs {
                let full_ranges = noncausal_mm_ranges(seq.mm_features(), None);
                if !full_ranges.is_empty() {
                    if include_full_attention {
                        self.full_mm_prefix_ranges_by_seq_id
                            .insert(*seq.id(), full_ranges.clone());
                    }
                    self.mm_prefix_ranges_by_seq_id
                        .insert(*seq.id(), full_ranges);
                }
            }
            self.has_noncausal_mm_context = !self.mm_prefix_ranges_by_seq_id.is_empty()
                || !self.full_mm_prefix_ranges_by_seq_id.is_empty();
        }
    }

    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    pub struct PagedAttentionInputMetadata {
        /// Block tables, windowed when a global sliding_window is set.
        pub block_tables: Option<HashMap<DeviceLocation, Tensor>>,
        /// Context lens, capped by sliding_window when set.
        pub context_lens: Option<HashMap<DeviceLocation, Tensor>>,
        pub block_size: Option<usize>,
        pub paged_context_lens_cpu: Option<Vec<usize>>,
        pub full_paged_context_lens_cpu: Option<Vec<usize>>,
        pub slot_mappings: HashMap<DeviceLocation, Tensor>,
        pub max_context_len: Option<usize>,
        /// Full (unwindowed) block tables, always covering the entire context.
        /// For models with per-layer sliding windows (GPT-OSS, Gemma2), layers
        /// without a sliding window should use these instead of `block_tables`.
        pub full_block_tables: Option<HashMap<DeviceLocation, Tensor>>,
        /// Full context lens (not capped by sliding_window).
        pub full_context_lens: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_max_context_len: Option<usize>,
        pub is_first_prompt_chunk: bool,
        pub is_final_prompt_chunk: bool,
        pub prompt_chunk_attention_policy: MultimodalAttentionPolicy,
        pub has_noncausal_mm_context: bool,
        pub prefix_gather_workspace_limit: Option<usize>,
        pub mm_prefix_ranges: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_mm_prefix_ranges: Option<HashMap<DeviceLocation, Tensor>>,
        pub prefill_attention_heads: usize,
        pub prefill_key_value_heads: usize,
        pub prefill_head_dim: usize,
        pub flashinfer: Option<FlashInferMetadata>,
        pub rope_positions: Option<HashMap<DeviceLocation, Tensor>>,
        /// Number of cached tokens per sequence (from prefix cache hits).
        /// When present and > 0, gather_kv_cache + Sdpa is used during prefill
        /// instead of flash attention. The Q/K/V tensors should only contain
        /// the NEW (non-cached) tokens.
        pub num_cached_tokens: Option<Vec<usize>>,
        /// Number of new tokens per sequence (query lengths).
        pub query_lens: Option<Vec<usize>>,
        /// Cumulative query lengths [batch+1], u32, for Sdpa varlen flash path.
        /// Precomputed to avoid Tensor::new in the forward hot path.
        pub cu_seqlens_q: Option<HashMap<DeviceLocation, Tensor>>,
        /// Cumulative KV lengths [batch+1], u32, for gather_kv_cache and flash_attn_varlen.
        /// Each entry is sum of (cached + new) tokens.
        pub cu_seqlens_kv: Option<HashMap<DeviceLocation, Tensor>>,
        /// Host rows this decode metadata was built from (decode steps only).
        pub decode_rows: Option<Arc<DecodePagedRows>>,
    }

    impl PagedAttentionInputMetadata {
        pub(crate) fn has_host_staged_decode_tensors(&self) -> bool {
            self.decode_rows.is_some()
                && self
                    .slot_mappings
                    .iter()
                    .any(|(location, tensor)| *location != tensor.device().location())
        }

        pub(crate) fn materialize_decode_tensors(&self) -> Result<Self> {
            if !self.has_host_staged_decode_tensors() {
                return Ok(self.clone());
            }
            self.decode_rows
                .as_ref()
                .expect("host-staged decode metadata requires source rows")
                .build_materialized()
        }

        /// Create a dummy input metadata, assuming that this will NOT be used for decoding.
        /// This is used for the case of imatrix generation.
        pub fn dummy(dev: &Device) -> candle_core::Result<Self> {
            Ok(PagedAttentionInputMetadata {
                block_tables: None,
                context_lens: None,
                block_size: None,
                paged_context_lens_cpu: None,
                full_paged_context_lens_cpu: None,
                max_context_len: None,
                full_block_tables: None,
                full_context_lens: None,
                full_max_context_len: None,
                slot_mappings: HashMap::from([(dev.location(), Tensor::new(&[0f32], dev)?)]),
                is_first_prompt_chunk: true,
                is_final_prompt_chunk: true,
                prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
                has_noncausal_mm_context: false,
                prefix_gather_workspace_limit: None,
                mm_prefix_ranges: None,
                full_mm_prefix_ranges: None,
                prefill_attention_heads: 1,
                prefill_key_value_heads: 1,
                prefill_head_dim: 1,
                flashinfer: None,
                rope_positions: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
                decode_rows: None,
            })
        }

        /// Build metadata for a prefill whose query tensor has been reduced to
        /// selected logits positions while K/V still live in the original paged
        /// cache. This is used by KV-sharing models that can skip hidden-state
        /// work for prompt tokens that will not produce logits.
        pub(crate) fn for_reduced_prefill_queries(
            &self,
            devices: &[Device],
            num_cached_tokens: &[usize],
            query_lens: &[usize],
        ) -> Result<Self> {
            if num_cached_tokens.len() != query_lens.len() {
                anyhow::bail!(
                    "reduced prefill metadata length mismatch: cached={} query={}",
                    num_cached_tokens.len(),
                    query_lens.len()
                );
            }
            if query_lens.is_empty() || query_lens.contains(&0) {
                anyhow::bail!("reduced prefill metadata requires at least one query token");
            }

            let batch_size = query_lens.len();
            let max_query_len = query_lens.iter().copied().max().unwrap_or(0);
            let slot_mappings_cpu = _make_tensor_with_pad(
                query_lens.iter().map(|len| vec![0i64; *len]).collect(),
                max_query_len,
                _PAD_SLOT_ID,
                &Device::Cpu,
            )?
            .reshape((batch_size, max_query_len))?;

            let context_lens = num_cached_tokens
                .iter()
                .zip(query_lens.iter())
                .map(|(cached, query)| cached + query)
                .collect::<Vec<_>>();
            let context_lens_cpu = Tensor::from_vec(
                context_lens
                    .iter()
                    .map(|len| *len as u32)
                    .collect::<Vec<_>>(),
                (batch_size,),
                &Device::Cpu,
            )?;
            let mut rope_positions = Vec::with_capacity(batch_size * max_query_len);
            for (&cached, &query_len) in num_cached_tokens.iter().zip(query_lens.iter()) {
                for seq_idx in 0..max_query_len {
                    let seq_idx = seq_idx.min(query_len - 1);
                    rope_positions.push((cached + seq_idx) as u32);
                }
            }
            let rope_positions_cpu =
                Tensor::from_vec(rope_positions, (batch_size * max_query_len,), &Device::Cpu)?;

            let mut cu_q = Vec::with_capacity(batch_size + 1);
            cu_q.push(0u32);
            for &query_len in query_lens {
                cu_q.push(cu_q.last().copied().unwrap_or(0) + query_len as u32);
            }
            let cu_q_cpu = Tensor::from_vec(cu_q, (batch_size + 1,), &Device::Cpu)?;

            let mut cu_kv = Vec::with_capacity(batch_size + 1);
            cu_kv.push(0u32);
            for (&cached, &query_len) in num_cached_tokens.iter().zip(query_lens.iter()) {
                cu_kv.push(cu_kv.last().copied().unwrap_or(0) + (cached + query_len) as u32);
            }
            let cu_kv_cpu = Tensor::from_vec(cu_kv, (batch_size + 1,), &Device::Cpu)?;
            let block_size = self
                .block_size
                .ok_or_else(|| anyhow::anyhow!("missing paged attention block size"))?;
            let paged_decode_context_lens = self
                .paged_context_lens_cpu
                .as_deref()
                .unwrap_or(&context_lens);
            let full_decode_context_lens = self
                .full_paged_context_lens_cpu
                .as_deref()
                .unwrap_or(paged_decode_context_lens);
            let (
                decode_request_indices_cpu,
                decode_kv_tile_indices_cpu,
                decode_o_indptr_cpu,
                decode_kv_chunk_size_cpu,
                decode_block_valid_mask_cpu,
            ) = make_paged_kv_decode_tensors_from_lens(
                paged_decode_context_lens,
                block_size,
                Some(flashinfer_decode_split_pages(
                    block_size,
                    batch_size,
                    self.prefill_key_value_heads,
                    paged_decode_context_lens.iter().copied().max().unwrap_or(0),
                )),
            )?;
            let (
                full_decode_request_indices_cpu,
                full_decode_kv_tile_indices_cpu,
                full_decode_o_indptr_cpu,
                full_decode_kv_chunk_size_cpu,
                full_decode_block_valid_mask_cpu,
            ) = make_paged_kv_decode_tensors_from_lens(
                full_decode_context_lens,
                block_size,
                Some(flashinfer_decode_split_pages(
                    block_size,
                    batch_size,
                    self.prefill_key_value_heads,
                    full_decode_context_lens.iter().copied().max().unwrap_or(0),
                )),
            )?;

            let mut slot_mappings = HashMap::new();
            let mut context_lens_map = HashMap::new();
            let mut rope_positions = HashMap::new();
            let mut cu_q_map = HashMap::new();
            let mut cu_kv_map = HashMap::new();
            let mut decode_request_indices_map = HashMap::new();
            let mut decode_kv_tile_indices_map = HashMap::new();
            let mut decode_o_indptr_map = HashMap::new();
            let mut decode_kv_chunk_size_map = HashMap::new();
            let mut decode_block_valid_mask_map = HashMap::new();
            let mut full_decode_request_indices_map = HashMap::new();
            let mut full_decode_kv_tile_indices_map = HashMap::new();
            let mut full_decode_o_indptr_map = HashMap::new();
            let mut full_decode_kv_chunk_size_map = HashMap::new();
            let mut full_decode_block_valid_mask_map = HashMap::new();
            for device in devices {
                slot_mappings.insert(device.location(), slot_mappings_cpu.to_device(device)?);
                context_lens_map.insert(device.location(), context_lens_cpu.to_device(device)?);
                rope_positions.insert(device.location(), rope_positions_cpu.to_device(device)?);
                cu_q_map.insert(device.location(), cu_q_cpu.to_device(device)?);
                cu_kv_map.insert(device.location(), cu_kv_cpu.to_device(device)?);
                decode_request_indices_map.insert(
                    device.location(),
                    decode_request_indices_cpu.to_device(device)?,
                );
                decode_kv_tile_indices_map.insert(
                    device.location(),
                    decode_kv_tile_indices_cpu.to_device(device)?,
                );
                decode_o_indptr_map
                    .insert(device.location(), decode_o_indptr_cpu.to_device(device)?);
                decode_kv_chunk_size_map.insert(
                    device.location(),
                    decode_kv_chunk_size_cpu.to_device(device)?,
                );
                decode_block_valid_mask_map.insert(
                    device.location(),
                    decode_block_valid_mask_cpu.to_device(device)?,
                );
                full_decode_request_indices_map.insert(
                    device.location(),
                    full_decode_request_indices_cpu.to_device(device)?,
                );
                full_decode_kv_tile_indices_map.insert(
                    device.location(),
                    full_decode_kv_tile_indices_cpu.to_device(device)?,
                );
                full_decode_o_indptr_map.insert(
                    device.location(),
                    full_decode_o_indptr_cpu.to_device(device)?,
                );
                full_decode_kv_chunk_size_map.insert(
                    device.location(),
                    full_decode_kv_chunk_size_cpu.to_device(device)?,
                );
                full_decode_block_valid_mask_map.insert(
                    device.location(),
                    full_decode_block_valid_mask_cpu.to_device(device)?,
                );
            }
            let full_context_lens = self
                .full_block_tables
                .as_ref()
                .map(|_| context_lens_map.clone());
            let full_max_context_len = self
                .full_block_tables
                .as_ref()
                .and_then(|_| context_lens.iter().copied().max());
            let flashinfer =
                self.flashinfer.as_ref().map(|flashinfer| {
                    let decode_tile_plan = flashinfer_tile_plan(
                        decode_request_indices_map.clone(),
                        decode_kv_tile_indices_map.clone(),
                        decode_o_indptr_map.clone(),
                        decode_kv_chunk_size_map.clone(),
                        decode_block_valid_mask_map.clone(),
                    );
                    let full_decode_tile_plan = flashinfer_tile_plan(
                        full_decode_request_indices_map.clone(),
                        full_decode_kv_tile_indices_map.clone(),
                        full_decode_o_indptr_map.clone(),
                        full_decode_kv_chunk_size_map.clone(),
                        full_decode_block_valid_mask_map.clone(),
                    );
                    let logical_tile_plan = if flashinfer.views.sliding.is_some() {
                        full_decode_tile_plan
                    } else {
                        decode_tile_plan.clone()
                    };
                    let logical = FlashInferPagedAttentionView {
                        tile_plan: logical_tile_plan,
                        ..flashinfer.views.logical.clone()
                    };
                    let sliding = flashinfer.views.sliding.as_ref().map(|view| {
                        FlashInferPagedAttentionView {
                            tile_plan: decode_tile_plan,
                            ..view.clone()
                        }
                    });
                    FlashInferMetadata {
                        views: FlashInferPagedAttentionViews { logical, sliding },
                        decode_tmp_v: None,
                        decode_tmp_s: None,
                        fa3_decode: None,
                        #[cfg(feature = "cuda")]
                        decode_tile_plan_used: None,
                    }
                });

            Ok(PagedAttentionInputMetadata {
                block_tables: self.block_tables.clone(),
                context_lens: Some(context_lens_map),
                block_size: self.block_size,
                paged_context_lens_cpu: Some(paged_decode_context_lens.to_vec()),
                full_paged_context_lens_cpu: Some(full_decode_context_lens.to_vec()),
                slot_mappings,
                max_context_len: context_lens.iter().copied().max(),
                full_block_tables: self.full_block_tables.clone(),
                full_context_lens,
                full_max_context_len,
                is_first_prompt_chunk: false,
                is_final_prompt_chunk: self.is_final_prompt_chunk,
                prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
                has_noncausal_mm_context: self.has_noncausal_mm_context,
                prefix_gather_workspace_limit: self.prefix_gather_workspace_limit,
                mm_prefix_ranges: self.mm_prefix_ranges.clone(),
                full_mm_prefix_ranges: self.full_mm_prefix_ranges.clone(),
                prefill_attention_heads: self.prefill_attention_heads,
                prefill_key_value_heads: self.prefill_key_value_heads,
                prefill_head_dim: self.prefill_head_dim,
                flashinfer,
                rope_positions: Some(rope_positions),
                num_cached_tokens: Some(num_cached_tokens.to_vec()),
                query_lens: Some(query_lens.to_vec()),
                cu_seqlens_q: Some(cu_q_map),
                cu_seqlens_kv: Some(cu_kv_map),
                decode_rows: None,
            })
        }
    }

    /// Flash attention sequence length metadata.
    ///
    /// `cumulative_seqlens_q/k` describe the physical Q/K layout. They use padded
    /// lengths for normal batches and logical lengths when `packed` is true.
    ///
    /// `logical_k` describes full logical K lengths. `sliding_k`, when present,
    /// describes the physical K lengths returned by a rotating/sliding KV cache.
    ///
    /// For the **prefix cache path**, K/V are gathered from the paged cache into a
    /// packed (non-padded) layout via `gather_kv_cache`. The packed K/V lengths are
    /// given by `PagedAttentionInputMetadata::cu_seqlens_kv`, NOT by the normal
    /// `logical_k/sliding_k` metadata here. The prefix cache attention call must
    /// build a local `FlashParams` matching the gathered KV layout.
    #[derive(Clone, Debug)]
    pub struct FlashKMeta {
        pub max: u32,
        pub cumulative_seqlens: HashMap<DeviceLocation, Tensor>,
    }

    impl FlashKMeta {
        pub fn empty() -> Self {
            Self {
                max: 0,
                cumulative_seqlens: HashMap::new(),
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct FlashParams {
        pub max_q: u32,
        pub cumulative_seqlens_q: HashMap<DeviceLocation, Tensor>,
        pub logical_k: FlashKMeta,
        pub sliding_k: Option<FlashKMeta>,
        pub causal: bool,
        pub(crate) packed: bool,
        #[cfg_attr(
            not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")),
            allow(dead_code)
        )]
        pub(crate) varlen_segment_lens: Option<Vec<usize>>,
    }

    impl FlashParams {
        pub fn empty(causal: bool) -> Self {
            Self {
                max_q: 0,
                cumulative_seqlens_q: HashMap::new(),
                logical_k: FlashKMeta::empty(),
                sliding_k: None,
                causal,
                packed: false,
                varlen_segment_lens: None,
            }
        }

        pub fn k_meta(&self, sliding_window: Option<usize>) -> &FlashKMeta {
            if sliding_window.is_some() {
                self.sliding_k.as_ref().unwrap_or(&self.logical_k)
            } else {
                &self.logical_k
            }
        }
    }

    pub struct InputMetadata {
        pub input: Tensor,
        pub positions: Vec<usize>,
        pub context_lens: Vec<(usize, usize)>, // (start index, len)
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>, // For paged attention
        pub flash_meta: FlashParams,
    }

    pub struct InnerInputProcessorOutput {
        pub inputs: InputMetadata,
        pub seq_indices: Vec<usize>,
    }

    fn flash_param_devices(device: &Device, mapper: Option<&dyn DeviceMapper>) -> Vec<Device> {
        mapper
            .map(|mapper| mapper.get_unique_devices())
            .unwrap_or_else(|| vec![device.clone()])
    }

    fn cumulative_seqlens_map(
        lengths: &[u32],
        devices: &[Device],
    ) -> Result<(u32, HashMap<DeviceLocation, Tensor>)> {
        let max = *lengths.iter().max().unwrap_or(&0);
        if devices.is_empty() {
            return Ok((max, HashMap::new()));
        }

        // Create tensors on CPU first to avoid CUDA context issues when copying
        // between different GPU devices. Each GPU has its own CUDA context, and
        // candle/cudarc doesn't properly switch contexts when doing GPU-to-GPU
        // transfers (which go through CPU). By creating on CPU first, we avoid
        // the cross-context memory access that causes CUDA_ERROR_INVALID_VALUE.
        let cumulative_seqlens = Tensor::new(lengths, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .cumsum(0)?
            .to_dtype(DType::U32)?;

        let mut cumulative_seqlens_map = HashMap::new();
        for device in devices {
            cumulative_seqlens_map.insert(device.location(), cumulative_seqlens.to_device(device)?);
        }

        Ok((max, cumulative_seqlens_map))
    }

    fn packed_rope_positions(seqlen_offsets: &[usize], query_lens: &[usize]) -> Result<Vec<u32>> {
        if seqlen_offsets.len() != query_lens.len() {
            anyhow::bail!(
                "packed RoPE position length mismatch: {} offsets for {} queries",
                seqlen_offsets.len(),
                query_lens.len()
            );
        }
        let mut positions = Vec::with_capacity(query_lens.iter().sum());
        for (&offset, &query_len) in seqlen_offsets.iter().zip(query_lens) {
            for position in offset..offset + query_len {
                positions.push(u32::try_from(position)?);
            }
        }
        Ok(positions)
    }

    fn sliding_k_lengths(
        seqlens_q: &[u32],
        seqlens_k: &[u32],
        sliding_window: usize,
    ) -> Result<Vec<u32>> {
        if seqlens_q.len() != seqlens_k.len() {
            anyhow::bail!(
                "sliding FlashAttention metadata length mismatch: q={} k={}",
                seqlens_q.len(),
                seqlens_k.len()
            );
        }
        let window = u32::try_from(sliding_window)?;
        seqlens_q
            .iter()
            .zip(seqlens_k)
            .map(|(&query_len, &logical_k_len)| {
                let past_len = logical_k_len.checked_sub(query_len).ok_or_else(|| {
                    anyhow::anyhow!(
                        "sliding FlashAttention query length {query_len} exceeds K length {logical_k_len}"
                    )
                })?;
                if query_len > 1 {
                    past_len
                        .min(window)
                        .checked_add(query_len)
                        .ok_or_else(|| anyhow::anyhow!("sliding FlashAttention K length overflow"))
                } else {
                    Ok(logical_k_len.min(window))
                }
            })
            .collect()
    }

    pub(crate) fn make_flash_params(
        device: &Device,
        mapper: Option<&dyn DeviceMapper>,
        seqlens_q: &[u32],
        seqlens_k: &[u32],
        sliding_window: Option<usize>,
        causal: bool,
        packed: bool,
    ) -> Result<FlashParams> {
        let devices = flash_param_devices(device, mapper);
        let (max_q, cumulative_seqlens_q) = cumulative_seqlens_map(seqlens_q, &devices)?;
        let (logical_max_k, logical_cumulative_seqlens_k) =
            cumulative_seqlens_map(seqlens_k, &devices)?;
        let logical_k = FlashKMeta {
            max: logical_max_k,
            cumulative_seqlens: logical_cumulative_seqlens_k,
        };
        let sliding_k = sliding_window
            .map(|window| -> Result<FlashKMeta> {
                let sliding_seqlens_k = sliding_k_lengths(seqlens_q, seqlens_k, window)?;
                let (sliding_max_k, sliding_cumulative_seqlens_k) =
                    cumulative_seqlens_map(&sliding_seqlens_k, &devices)?;
                Ok(FlashKMeta {
                    max: sliding_max_k,
                    cumulative_seqlens: sliding_cumulative_seqlens_k,
                })
            })
            .transpose()?;

        Ok(FlashParams {
            max_q,
            cumulative_seqlens_q,
            logical_k,
            sliding_k,
            causal,
            packed,
            varlen_segment_lens: None,
        })
    }

    // chunk_offset_toks is the number of tokens by which the tokens are offset,
    // chunk_offset_toks / prompt_chunksize = number of batches
    //
    // prefix_cache_lens: when provided, indicates how many tokens per sequence are already
    // cached in the paged KV cache. Only new (non-cached) tokens will be included in the
    // input tensor, and slot_mappings will only cover new token slots. Block tables still
    // cover the entire context so that context_attention_fwd can read cached blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn make_prompt_chunk<T: WithDType + Debug>(
        chunk_offset_toks: usize,
        toks: Vec<&[T]>,
        seq_ids: &[usize],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        prefix_cache_lens: Option<&[usize]>,
        sliding_window: Option<usize>,
        allow_packed_prefill: bool,
    ) -> Result<InputMetadata> {
        // Determine effective tokens per sequence after prefix cache trimming
        let effective_lens: Vec<usize> = toks
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let cached = prefix_cache_lens.map_or(0, |lens| lens[i]);
                seq.len().saturating_sub(cached)
            })
            .collect();
        let max_len = *effective_lens.iter().max().expect("No sequences");
        let padding_tok = T::zero();
        let has_any_cache_hit = prefix_cache_lens.is_some_and(|lens| lens.iter().any(|&l| l > 0));
        let prompt_chunk_causal = paged_attn_metadata.as_ref().is_none_or(|metadata| {
            metadata.prompt_chunk_attention_policy == MultimodalAttentionPolicy::Causal
        });
        let all_model_devices_cuda = mapper.is_none_or(|mapper| {
            mapper
                .get_unique_devices()
                .iter()
                .all(|device| device.is_cuda())
        });
        let packed_prefill = allow_packed_prefill
            && effective_lens.len() > 1
            && effective_lens.iter().any(|len| *len != max_len)
            && device.is_cuda()
            && all_model_devices_cuda
            && crate::using_flash_attn()
            && !return_raw_logits
            && last_n_context_len.is_none()
            && chunk_offset_toks == 0
            && !has_any_cache_hit
            && paged_attn_metadata.as_ref().is_some_and(|metadata| {
                metadata.enable_packed_prefill
                    && metadata.is_final_prompt_chunk
                    && metadata.prompt_chunk_attention_policy == MultimodalAttentionPolicy::Causal
            });
        if packed_prefill {
            tracing::debug!(
                sequences = effective_lens.len(),
                tokens = effective_lens.iter().sum::<usize>(),
                padded_tokens = effective_lens.len() * max_len,
                "Using packed prompt prefill"
            );
        }
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();
        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut full_block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut full_paged_attn_context_lens = Vec::new();
        let flash_attn = crate::using_flash_attn();
        let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
        let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
        let mut num_cached_tokens_vec: Vec<usize> = Vec::new();
        let mut query_lens_vec: Vec<usize> = Vec::new();
        for (seq_idx, (seq_id, ctxt)) in seq_ids.iter().zip(&toks).enumerate() {
            let cached = prefix_cache_lens.map_or(0, |lens| lens[seq_idx]);
            let full_prompt_len = ctxt.len();
            // The new (non-cached) tokens to process
            let new_toks = &ctxt[cached..];
            let new_len = new_toks.len();

            let offset = last_n_context_len.unwrap_or_default();
            // seqlen_offset includes cached prefix so position IDs are correct
            seqlen_offsets.push(offset.1 + chunk_offset_toks + cached);

            position_ids.push(new_len + chunk_offset_toks + cached);
            let mut input_toks = new_toks.to_vec();
            if !packed_prefill {
                input_toks.extend(std::iter::repeat_n(
                    padding_tok,
                    max_len.saturating_sub(input_toks.len()),
                ));
            }
            // If we are returning raw logits, we want to not trim the logits at all.
            if return_raw_logits {
                if last_n_context_len.is_some() {
                    anyhow::bail!("`return_raw_logits` is incompatible with `last_n_context_len`");
                }

                context_lens.push((0, input_toks.len()));
            } else {
                context_lens.push((
                    new_len.saturating_sub(last_n_context_len.map(|(a, _)| a).unwrap_or(1)),
                    last_n_context_len.map(|(a, _)| a).unwrap_or(1),
                ));
            }

            if flash_attn {
                seqlens_q.push(input_toks.len() as u32);
                seqlens_k.push((input_toks.len() + chunk_offset_toks + cached) as u32);
            }

            seqs_tensors.push(Tensor::new(input_toks, device)?.unsqueeze(0)?);

            if has_any_cache_hit {
                num_cached_tokens_vec.push(cached);
                query_lens_vec.push(new_len);
            }

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let kv_mgr = get_mut_arcmutex!(paged_attn_metadata.kv_cache_manager);
                let block_ids = kv_mgr.get_block_ids(*seq_id);

                if block_ids.is_none() {
                    // Will be None during profiling.
                    slot_mappings.push([_PAD_SLOT_ID].repeat(new_len));
                    continue;
                }
                let table: Vec<usize> = block_ids.unwrap().to_vec();
                drop(kv_mgr);

                // Slot mappings only for new tokens (cached tokens are already in cache)
                let slot_start = cached + chunk_offset_toks;
                let slot_end = full_prompt_len + chunk_offset_toks;
                let mut slot_mapping = Vec::new();
                let mut ctxt_len = Vec::new();
                for i in slot_start..slot_end {
                    ctxt_len.push(i);

                    let block_number = if i / paged_attn_metadata.block_size >= table.len() {
                        panic!(
                            "Block table is too small (prompt)! i={} block_size={} table_len={}",
                            i,
                            paged_attn_metadata.block_size,
                            table.len()
                        );
                    } else {
                        table.get(i / paged_attn_metadata.block_size).unwrap()
                    };
                    let block_offset = i % paged_attn_metadata.block_size;
                    // Use checked arithmetic to prevent overflow
                    let slot = block_number
                        .checked_mul(paged_attn_metadata.block_size)
                        .and_then(|v| v.checked_add(block_offset))
                        .expect("Slot calculation overflowed");
                    slot_mapping.push(
                        slot.try_into()
                            .expect("Slot value too large for target integer type"),
                    );
                }
                slot_mappings.push(slot_mapping);
                let full_context_len = chunk_offset_toks + cached + new_len;
                full_block_tables.push(table.clone());
                full_paged_attn_context_lens.push(full_context_len);

                if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                    let mut block_aligned_start = block_aligned_sliding_window_start(
                        full_context_len,
                        new_len,
                        sliding_window,
                        paged_attn_metadata.block_size,
                    );
                    if let Some(mm_start) = paged_attn_metadata
                        .mm_prefix_ranges_by_seq_id
                        .get(seq_id)
                        .into_iter()
                        .flatten()
                        .filter(|&&(start, end)| start < slot_end && slot_start < end)
                        .map(|&(start, _)| start)
                        .min()
                    {
                        block_aligned_start = block_aligned_start.min(
                            mm_start / paged_attn_metadata.block_size
                                * paged_attn_metadata.block_size,
                        );
                    }
                    let paged_context_len = full_context_len - block_aligned_start;
                    let slide_idx = block_aligned_start / paged_attn_metadata.block_size;
                    let needed_blocks = paged_context_len.div_ceil(paged_attn_metadata.block_size);
                    let slide_end = (slide_idx + needed_blocks).min(table.len());
                    block_tables.push(table.get(slide_idx..slide_end).unwrap_or(&[]).to_vec());
                    paged_attn_context_lens.push((0..paged_context_len).collect());
                } else {
                    block_tables.push(table.clone());
                    paged_attn_context_lens.push(ctxt_len);
                }
            }
        }

        let flash_meta = if flash_attn {
            make_flash_params(
                device,
                mapper,
                &seqlens_q,
                &seqlens_k,
                sliding_window,
                prompt_chunk_causal,
                packed_prefill,
            )?
        } else {
            FlashParams::empty(prompt_chunk_causal)
        };

        let input_concat_dim = if packed_prefill { 1 } else { 0 };
        let input = Tensor::cat(&seqs_tensors, input_concat_dim).unwrap();

        let paged_attn_meta = if let Some(paged_attn_metadata) = &paged_attn_metadata {
            // Create paged attention tensors on CPU first (see comment above about CUDA contexts)
            let prefill_query_lens = slot_mappings.iter().map(Vec::len).collect::<Vec<_>>();
            let slot_mappings = if packed_prefill {
                let slots = slot_mappings.into_iter().flatten().collect::<Vec<_>>();
                let slot_count = slots.len();
                Tensor::from_vec(slots, (slot_count,), &Device::Cpu)?
            } else {
                let max_slot_mapping_len = slot_mappings.iter().map(Vec::len).max().unwrap();
                _make_tensor_with_pad(
                    slot_mappings,
                    max_slot_mapping_len,
                    _PAD_SLOT_ID,
                    &Device::Cpu,
                )?
            };

            let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
            let block_size = paged_attn_metadata.block_size;
            let full_context_lens_for_fi = if has_any_cache_hit {
                num_cached_tokens_vec
                    .iter()
                    .zip(query_lens_vec.iter())
                    .map(|(cached, query_len)| cached + query_len)
                    .collect::<Vec<_>>()
            } else {
                prefill_query_lens.clone()
            };
            let paged_context_lens_for_fi = if sliding_window.is_some() {
                paged_attn_context_lens
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<_>>()
            } else {
                full_context_lens_for_fi.clone()
            };
            let (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len) =
                make_paged_kv_tensors(
                    &block_tables,
                    &paged_context_lens_for_fi,
                    block_size,
                    block_tables.len() * max_block_table_len,
                )?;
            let decode_split_pages = flashinfer_decode_split_pages(
                block_size,
                block_tables.len(),
                paged_attn_metadata.prefill_key_value_heads,
                paged_context_lens_for_fi.iter().copied().max().unwrap_or(0),
            );
            let tiles_per_row = max_block_table_len
                .max(1)
                .div_ceil(flashinfer_decode_split_capacity_pages(block_size));
            let (request_indices, kv_tile_indices, o_indptr, kv_chunk_size, block_valid_mask) =
                make_paged_kv_decode_tensors(
                    &block_tables,
                    &paged_context_lens_for_fi,
                    block_size,
                    Some(decode_split_pages),
                    block_tables.len() * tiles_per_row,
                )?;
            let block_tables = _make_tensor_with_pad(
                block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_block_table_len,
                0,
                &Device::Cpu,
            )?;
            let block_tables = block_tables.reshape(((), max_block_table_len))?;

            let max_context_len = paged_attn_context_lens
                .iter()
                .map(|x| x.len())
                .max()
                .unwrap();

            let context_lens = _make_tensor_with_pad(
                paged_attn_context_lens
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_context_len,
                0,
                &Device::Cpu,
            )?
            .reshape(((),))?;
            let packed_rope_positions = if packed_prefill {
                let positions = packed_rope_positions(&seqlen_offsets, &prefill_query_lens)?;
                let position_count = positions.len();
                Some(Tensor::from_vec(
                    positions,
                    (position_count,),
                    &Device::Cpu,
                )?)
            } else {
                None
            };

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut rope_positions_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();
            let mut mm_prefix_ranges_map = HashMap::new();
            let mut full_mm_prefix_ranges_map = HashMap::new();
            let mut full_block_tables_map = HashMap::new();
            let mut full_context_lens_map = HashMap::new();
            let mut paged_kv_indptr_map = HashMap::new();
            let mut paged_kv_indices_map = HashMap::new();
            let mut paged_kv_last_page_len_map = HashMap::new();
            let mut request_indices_map = HashMap::new();
            let mut kv_tile_indices_map = HashMap::new();
            let mut o_indptr_map = HashMap::new();
            let mut kv_chunk_size_map = HashMap::new();
            let mut block_valid_mask_map = HashMap::new();
            let mut full_paged_kv_indptr_map = HashMap::new();
            let mut full_paged_kv_indices_map = HashMap::new();
            let mut full_paged_kv_last_page_len_map = HashMap::new();
            let mut full_request_indices_map = HashMap::new();
            let mut full_kv_tile_indices_map = HashMap::new();
            let mut full_o_indptr_map = HashMap::new();
            let mut full_kv_chunk_size_map = HashMap::new();
            let mut full_block_valid_mask_map = HashMap::new();

            let (
                full_block_tables_tensor,
                full_context_lens_tensor,
                full_max_context_len,
                full_paged_kv_tensors,
                full_decode_tensors,
            ) = if sliding_window.is_some() {
                let full_max_block_table_len =
                    full_block_tables.iter().map(|x| x.len()).max().unwrap_or(1);
                let full_paged_kv_tensors = Some(make_paged_kv_tensors(
                    &full_block_tables,
                    &full_paged_attn_context_lens,
                    block_size,
                    full_block_tables.len() * full_max_block_table_len,
                )?);
                let full_decode_split_pages = flashinfer_decode_split_pages(
                    block_size,
                    full_block_tables.len(),
                    paged_attn_metadata.prefill_key_value_heads,
                    full_context_lens_for_fi.iter().copied().max().unwrap_or(0),
                );
                let full_tiles_per_row = full_max_block_table_len
                    .max(1)
                    .div_ceil(flashinfer_decode_split_capacity_pages(block_size));
                let full_decode_tensors = Some(make_paged_kv_decode_tensors(
                    &full_block_tables,
                    &full_paged_attn_context_lens,
                    block_size,
                    Some(full_decode_split_pages),
                    full_block_tables.len() * full_tiles_per_row,
                )?);
                let full_block_tables_tensor = _make_tensor_with_pad(
                    full_block_tables
                        .iter()
                        .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                    full_max_block_table_len,
                    0,
                    &Device::Cpu,
                )?
                .reshape(((), full_max_block_table_len))?;
                let full_context_lens_tensor = Tensor::from_vec(
                    full_paged_attn_context_lens
                        .iter()
                        .map(|x| *x as u32)
                        .collect::<Vec<_>>(),
                    (full_paged_attn_context_lens.len(),),
                    &Device::Cpu,
                )?;
                let full_max_context_len = full_paged_attn_context_lens.iter().copied().max();
                (
                    Some(full_block_tables_tensor),
                    Some(full_context_lens_tensor),
                    full_max_context_len,
                    full_paged_kv_tensors,
                    full_decode_tensors,
                )
            } else {
                (None, None, None, None, None)
            };
            let kv_window_starts = full_paged_attn_context_lens
                .iter()
                .zip(paged_context_lens_for_fi.iter())
                .map(|(full_len, paged_len)| full_len.saturating_sub(*paged_len))
                .collect::<Vec<_>>();
            let mm_prefix_ranges_tensor = crate::paged_attention::mm_prefix::make_ranges_tensor(
                seq_ids,
                &paged_attn_metadata.mm_prefix_ranges_by_seq_id,
                &kv_window_starts,
                &paged_context_lens_for_fi,
                &prefill_query_lens,
            )?;
            let full_kv_window_starts = vec![0; seq_ids.len()];
            let full_mm_prefix_ranges_tensor = if sliding_window.is_some() {
                crate::paged_attention::mm_prefix::make_ranges_tensor(
                    seq_ids,
                    &paged_attn_metadata.full_mm_prefix_ranges_by_seq_id,
                    &full_kv_window_starts,
                    &full_paged_attn_context_lens,
                    &prefill_query_lens,
                )?
            } else {
                None
            };

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                if let Some(positions) = &packed_rope_positions {
                    rope_positions_map
                        .insert(device.location(), positions.clone().to_device(&device)?);
                }
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
                if let Some(mm_prefix_ranges_tensor) = &mm_prefix_ranges_tensor {
                    mm_prefix_ranges_map.insert(
                        device.location(),
                        mm_prefix_ranges_tensor.clone().to_device(&device)?,
                    );
                }
                if let Some(full_mm_prefix_ranges_tensor) = &full_mm_prefix_ranges_tensor {
                    full_mm_prefix_ranges_map.insert(
                        device.location(),
                        full_mm_prefix_ranges_tensor.clone().to_device(&device)?,
                    );
                }
                paged_kv_indptr_map.insert(
                    device.location(),
                    paged_kv_indptr.clone().to_device(&device)?,
                );
                paged_kv_indices_map.insert(
                    device.location(),
                    paged_kv_indices.clone().to_device(&device)?,
                );
                paged_kv_last_page_len_map.insert(
                    device.location(),
                    paged_kv_last_page_len.clone().to_device(&device)?,
                );
                request_indices_map.insert(
                    device.location(),
                    request_indices.clone().to_device(&device)?,
                );
                kv_tile_indices_map.insert(
                    device.location(),
                    kv_tile_indices.clone().to_device(&device)?,
                );
                o_indptr_map.insert(device.location(), o_indptr.clone().to_device(&device)?);
                kv_chunk_size_map
                    .insert(device.location(), kv_chunk_size.clone().to_device(&device)?);
                block_valid_mask_map.insert(
                    device.location(),
                    block_valid_mask.clone().to_device(&device)?,
                );
                if let Some(full_block_tables_tensor) = &full_block_tables_tensor {
                    full_block_tables_map.insert(
                        device.location(),
                        full_block_tables_tensor.clone().to_device(&device)?,
                    );
                }
                if let Some(full_context_lens_tensor) = &full_context_lens_tensor {
                    full_context_lens_map.insert(
                        device.location(),
                        full_context_lens_tensor.clone().to_device(&device)?,
                    );
                }
                if let Some((indptr, indices, last_page_len)) = &full_paged_kv_tensors {
                    full_paged_kv_indptr_map
                        .insert(device.location(), indptr.clone().to_device(&device)?);
                    full_paged_kv_indices_map
                        .insert(device.location(), indices.clone().to_device(&device)?);
                    full_paged_kv_last_page_len_map
                        .insert(device.location(), last_page_len.clone().to_device(&device)?);
                }
                if let Some((req, kv, o, chunk, valid)) = &full_decode_tensors {
                    full_request_indices_map
                        .insert(device.location(), req.clone().to_device(&device)?);
                    full_kv_tile_indices_map
                        .insert(device.location(), kv.clone().to_device(&device)?);
                    full_o_indptr_map.insert(device.location(), o.clone().to_device(&device)?);
                    full_kv_chunk_size_map
                        .insert(device.location(), chunk.clone().to_device(&device)?);
                    full_block_valid_mask_map
                        .insert(device.location(), valid.clone().to_device(&device)?);
                }
            }

            let prompt_chunk_attention_policy = paged_attn_metadata.prompt_chunk_attention_policy;
            let sliding_flashinfer_view = if sliding_window.is_some() {
                Some(flashinfer_view(
                    Some(block_tables_map.clone()),
                    Some(context_lens_map.clone()),
                    Some(max_context_len),
                    flashinfer_paged_kv(
                        paged_kv_indptr_map.clone(),
                        paged_kv_indices_map.clone(),
                        paged_kv_last_page_len_map.clone(),
                    ),
                    flashinfer_tile_plan(
                        request_indices_map.clone(),
                        kv_tile_indices_map.clone(),
                        o_indptr_map.clone(),
                        kv_chunk_size_map.clone(),
                        block_valid_mask_map.clone(),
                    ),
                ))
            } else {
                None
            };
            let logical_flashinfer_view = if sliding_window.is_some() {
                flashinfer_view(
                    Some(full_block_tables_map.clone()),
                    Some(full_context_lens_map.clone()),
                    full_max_context_len,
                    flashinfer_paged_kv(
                        full_paged_kv_indptr_map.clone(),
                        full_paged_kv_indices_map.clone(),
                        full_paged_kv_last_page_len_map.clone(),
                    ),
                    flashinfer_tile_plan(
                        full_request_indices_map.clone(),
                        full_kv_tile_indices_map.clone(),
                        full_o_indptr_map.clone(),
                        full_kv_chunk_size_map.clone(),
                        full_block_valid_mask_map.clone(),
                    ),
                )
            } else {
                flashinfer_view(
                    Some(block_tables_map.clone()),
                    Some(context_lens_map.clone()),
                    Some(max_context_len),
                    flashinfer_paged_kv(
                        paged_kv_indptr_map.clone(),
                        paged_kv_indices_map.clone(),
                        paged_kv_last_page_len_map.clone(),
                    ),
                    flashinfer_tile_plan(
                        request_indices_map.clone(),
                        kv_tile_indices_map.clone(),
                        o_indptr_map.clone(),
                        kv_chunk_size_map.clone(),
                        block_valid_mask_map.clone(),
                    ),
                )
            };
            let flashinfer = Some(flashinfer_metadata(
                logical_flashinfer_view,
                sliding_flashinfer_view,
            ));

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                block_size: Some(block_size),
                paged_context_lens_cpu: Some(paged_context_lens_for_fi.clone()),
                full_paged_context_lens_cpu: Some(full_paged_attn_context_lens.clone()),
                max_context_len: Some(max_context_len),
                full_block_tables: if full_block_tables_map.is_empty() {
                    None
                } else {
                    Some(full_block_tables_map)
                },
                full_context_lens: if full_context_lens_map.is_empty() {
                    None
                } else {
                    Some(full_context_lens_map)
                },
                full_max_context_len,
                is_first_prompt_chunk: chunk_offset_toks == 0 && !has_any_cache_hit,
                is_final_prompt_chunk: paged_attn_metadata.is_final_prompt_chunk,
                prompt_chunk_attention_policy,
                // Keep the slow path local to chunks whose query rows overlap a noncausal range.
                has_noncausal_mm_context: mm_prefix_ranges_tensor.is_some()
                    || full_mm_prefix_ranges_tensor.is_some(),
                prefix_gather_workspace_limit: paged_attn_metadata.prefix_gather_workspace_limit,
                mm_prefix_ranges: if mm_prefix_ranges_map.is_empty() {
                    None
                } else {
                    Some(mm_prefix_ranges_map)
                },
                full_mm_prefix_ranges: if full_mm_prefix_ranges_map.is_empty() {
                    None
                } else {
                    Some(full_mm_prefix_ranges_map)
                },
                prefill_attention_heads: paged_attn_metadata.prefill_attention_heads,
                prefill_key_value_heads: paged_attn_metadata.prefill_key_value_heads,
                prefill_head_dim: paged_attn_metadata.prefill_head_dim,
                flashinfer,
                rope_positions: if rope_positions_map.is_empty() {
                    None
                } else {
                    Some(rope_positions_map)
                },
                num_cached_tokens: if has_any_cache_hit {
                    Some(num_cached_tokens_vec.clone())
                } else {
                    None
                },
                // Always set: saves a per-layer GPU->CPU slot-mapping sync in forward_prefix.
                query_lens: Some(prefill_query_lens.clone()),
                cu_seqlens_q: if has_any_cache_hit {
                    // Cumulative query lengths for Sdpa varlen: [0, q0, q0+q1, ...]
                    let mut cu_q = vec![0u32];
                    for &ql in &query_lens_vec {
                        cu_q.push(cu_q.last().unwrap() + ql as u32);
                    }
                    let cu_q_t = Tensor::new(&cu_q[..], &Device::Cpu)?;
                    let devices = mapper.unwrap().get_unique_devices();
                    let mut map = HashMap::new();
                    for device in &devices {
                        map.insert(device.location(), cu_q_t.to_device(device)?);
                    }
                    Some(map)
                } else {
                    None
                },
                cu_seqlens_kv: if has_any_cache_hit {
                    // Cumulative KV lengths: [0, c0+q0, c0+q0+c1+q1, ...]
                    // U32 to match flash-attn varlen expectations
                    let mut cu_kv = vec![0u32];
                    for (&nc, &ql) in num_cached_tokens_vec.iter().zip(query_lens_vec.iter()) {
                        cu_kv.push(cu_kv.last().unwrap() + (nc + ql) as u32);
                    }
                    let cu_kv_t = Tensor::new(&cu_kv[..], &Device::Cpu)?;
                    let devices = mapper.unwrap().get_unique_devices();
                    let mut map = HashMap::new();
                    for device in &devices {
                        map.insert(device.location(), cu_kv_t.to_device(device)?);
                    }
                    Some(map)
                } else {
                    None
                },
                decode_rows: None,
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
        })
    }

    fn completion_input_tensor<T: WithDType>(
        host_tokens: Vec<T>,
        batch: usize,
        host_width: usize,
        staged_device_rows: &[Tensor],
        device: &Device,
    ) -> Result<Tensor> {
        let host = Tensor::from_vec(host_tokens, (batch, host_width), device)?;
        if staged_device_rows.is_empty() {
            return Ok(host);
        }
        let staged_device_rows = staged_device_rows
            .iter()
            .map(|tokens| tokens.to_device(device)?.to_dtype(T::DTYPE))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let staged = Tensor::stack(&staged_device_rows, 0)?;
        Ok(Tensor::cat(&[&host, &staged], 1)?)
    }

    fn make_completion_chunk<T: WithDType + From<u32> + Clone + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
        decode_window: usize,
    ) -> Result<InputMetadata> {
        // Pad each sequence by the padding token to the max len.
        let flash_attn = crate::using_flash_attn();
        let mut input_tokens = Vec::new();
        let mut input_width = None;
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();

        let mut slot_mappings = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut full_paged_attn_context_lens = Vec::new();
        let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
        let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
        // Staged speculative tokens are appended to the decode input only when
        // the whole batch has the same fixed proposal width. The generic
        // verifier keeps the target forward rectangular in this first batched
        // implementation; mixed staged/no-staged batches fall back to a normal
        // one-token decode and the driver clears the stale staged proposals.
        let use_staged_speculative =
            crate::speculative::staging::staged_batch_width(input_seqs).is_some();
        let use_device_staged = use_staged_speculative
            && input_seqs
                .iter()
                .any(|seq| seq.active_staged_speculative_tokens().as_device().is_some());
        let sequence_block_tables = paged_attn_metadata.as_ref().map(|paged_attn_metadata| {
            let kv_mgr = get_mut_arcmutex!(paged_attn_metadata.kv_cache_manager);
            input_seqs
                .iter()
                .map(|seq| {
                    Arc::<[usize]>::from(
                        kv_mgr
                            .get_block_ids(*seq.id())
                            .expect("Sequence must have allocated blocks for completion"),
                    )
                })
                .collect::<Vec<_>>()
        });
        let mut host_input_width = None;
        for (seq_idx, (seq, ctxt)) in input_seqs.iter().zip(toks).enumerate() {
            let staged_speculative = if use_staged_speculative && !use_device_staged {
                seq.active_staged_speculative_tokens()
                    .as_host()
                    .expect("host-backed speculative batch changed storage kind")
            } else {
                &[]
            };
            let start_pos = ctxt.len().saturating_sub(decode_window);
            let mut ctxt = ctxt[start_pos..].to_vec();
            ctxt.extend(staged_speculative.iter().copied().map(T::from));
            let host_width = ctxt.len();
            let query_len = host_width
                + if use_device_staged {
                    seq.active_staged_speculative_len()
                } else {
                    0
                };
            let effective_context_len = start_pos + query_len;
            seqlen_offsets.push(start_pos);
            context_lens.push((0, query_len));
            position_ids.push(effective_context_len);

            if flash_attn {
                seqlens_q.push(query_len as u32);
                seqlens_k.push(effective_context_len as u32);
            }

            match input_width {
                Some(width) if width != query_len => {
                    anyhow::bail!("completion input rows must have one query width")
                }
                None => input_width = Some(query_len),
                Some(_) => {}
            }
            match host_input_width {
                Some(width) if width != host_width => {
                    anyhow::bail!("completion input host rows must have one query width")
                }
                None => host_input_width = Some(host_width),
                Some(_) => {}
            }
            input_tokens.extend(ctxt);

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let table = &sequence_block_tables
                    .as_ref()
                    .expect("paged block tables were snapshotted")[seq_idx];

                let block_start = start_pos - seq.token_offset();
                let block_end = block_start + query_len;
                let mut slot_mapping = Vec::with_capacity(query_len);
                for block_pos in block_start..block_end {
                    let block_number = if block_pos / paged_attn_metadata.block_size >= table.len()
                    {
                        panic!("Block table is too small (completion)! block_pos={} block_size={} table_len={}", block_pos, paged_attn_metadata.block_size, table.len());
                    } else {
                        table
                            .get(block_pos / paged_attn_metadata.block_size)
                            .unwrap()
                    };
                    let block_offset = block_pos % paged_attn_metadata.block_size;
                    // Use checked arithmetic to prevent overflow
                    let slot = block_number
                        .checked_mul(paged_attn_metadata.block_size)
                        .and_then(|v| v.checked_add(block_offset))
                        .expect("Slot calculation overflowed");
                    let slot = slot
                        .try_into()
                        .expect("Slot value too large for target integer type");
                    slot_mapping.push(slot);
                }
                slot_mappings.push(slot_mapping);

                for row in 0..query_len {
                    let full_context_len = start_pos + row + 1;

                    full_paged_attn_context_lens.push(full_context_len);

                    let paged_attn_context_len = if let Some(sliding_window) =
                        paged_attn_metadata.sliding_window
                    {
                        let window_start = full_context_len.saturating_sub(sliding_window);
                        let block_aligned_start = (window_start / paged_attn_metadata.block_size)
                            * paged_attn_metadata.block_size;
                        full_context_len - block_aligned_start
                    } else {
                        full_context_len
                    };
                    paged_attn_context_lens.push(paged_attn_context_len);
                }
            }
        }

        let paged_single_token_decode = paged_attn_metadata.is_some()
            && context_lens.iter().all(|&(_, query_len)| query_len == 1);
        let flash_meta = if flash_attn && !paged_single_token_decode {
            make_flash_params(
                device,
                mapper,
                &seqlens_q,
                &seqlens_k,
                sliding_window,
                true,
                false,
            )?
        } else {
            FlashParams::empty(true)
        };

        let paged_attn_meta = if let Some(paged_attn_input) = &paged_attn_metadata {
            let query_len = context_lens.first().map_or(1, |(_, q)| *q);
            let block_tables = BlockTableSnapshot::from_sequence_tables(
                sequence_block_tables.expect("paged block tables were snapshotted"),
                query_len,
            );
            let rows = Arc::new(DecodePagedRows {
                slot_mappings,
                block_tables,
                context_lens: paged_attn_context_lens,
                full_context_lens: full_paged_attn_context_lens,
                query_len,
                block_size: paged_attn_input.block_size,
                use_standard_metadata: paged_attn_input.attention_backend
                    == AttentionBackendKind::Standard,
                max_paged_context_len: paged_attn_input.max_paged_context_len,
                sliding_window: paged_attn_input.sliding_window,
                decode_window,
                devices: mapper.unwrap().get_unique_devices(),
                num_kv_heads: paged_attn_input.prefill_key_value_heads,
            });
            Some(rows.build()?)
        } else {
            None
        };

        let staged_device_rows = if use_device_staged {
            input_seqs
                .iter()
                .map(|seq| match seq.active_staged_speculative_tokens() {
                    crate::speculative::SpeculativeTokens::Host(tokens) => {
                        Tensor::new(tokens.as_slice(), device)
                    }
                    crate::speculative::SpeculativeTokens::Device(tokens) => Ok(tokens.clone()),
                })
                .collect::<candle_core::Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let input = completion_input_tensor(
            input_tokens,
            input_seqs.len(),
            host_input_width.unwrap_or_default(),
            &staged_device_rows,
            device,
        )?;
        if input.dims() != [input_seqs.len(), input_width.unwrap_or_default()] {
            anyhow::bail!("completion input tensor shape changed while staging proposals");
        }

        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
        })
    }

    /// Host-side per-row decode inputs plus everything needed to materialize the paged-attention
    /// metadata from them. Kept on the metadata so the CUDA graph layer can rebuild a batch-padded
    /// twin through the same code path.
    #[derive(Clone, Debug)]
    pub struct DecodePagedRows {
        pub slot_mappings: Vec<Vec<i64>>,
        pub(crate) block_tables: BlockTableSnapshot,
        pub context_lens: Vec<usize>,
        pub full_context_lens: Vec<usize>,
        pub query_len: usize,
        pub block_size: usize,
        pub use_standard_metadata: bool,
        pub max_paged_context_len: usize,
        pub sliding_window: Option<usize>,
        pub decode_window: usize,
        pub devices: Vec<Device>,
        pub num_kv_heads: usize,
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct PagedDecodeMetadataRequirements {
        pub block_tables: bool,
        pub context_lens: bool,
        pub flashinfer_paged_kv: bool,
        pub flashinfer_tile_plan: bool,
    }

    impl PagedDecodeMetadataRequirements {
        fn conservative(rows: &DecodePagedRows) -> Self {
            Self {
                block_tables: rows.use_standard_metadata || rows.decode_window > 1,
                context_lens: rows.use_standard_metadata,
                flashinfer_paged_kv: true,
                flashinfer_tile_plan: true,
            }
        }

        #[cfg(any(feature = "cuda", test))]
        pub(crate) fn graph(
            block_tables: bool,
            context_lens: bool,
            flashinfer_paged_kv: bool,
            flashinfer_tile_plan: bool,
        ) -> Self {
            Self {
                block_tables: block_tables || context_lens,
                context_lens,
                flashinfer_paged_kv: flashinfer_paged_kv || flashinfer_tile_plan,
                flashinfer_tile_plan,
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct DecodePagedRowsGraphKey {
        batch_size: usize,
        query_len: usize,
        block_size: usize,
        use_standard_metadata: bool,
        sliding_window: Option<usize>,
        decode_window: usize,
        devices: Vec<DeviceLocation>,
        num_kv_heads: usize,
        paged_block_table_len: usize,
        full_block_table_len: usize,
    }

    struct DecodeViewHostTensors {
        block_tables: Option<Tensor>,
        context_lens: Option<Tensor>,
        paged_kv: Option<(Tensor, Tensor, Tensor)>,
        tile_plan: Option<(Tensor, Tensor, Tensor, Tensor, Tensor)>,
    }

    #[derive(Default)]
    struct DecodeViewDeviceMaps {
        block_tables: HashMap<DeviceLocation, Tensor>,
        context_lens: HashMap<DeviceLocation, Tensor>,
        paged_kv_indptr: HashMap<DeviceLocation, Tensor>,
        paged_kv_indices: HashMap<DeviceLocation, Tensor>,
        paged_kv_last_page_len: HashMap<DeviceLocation, Tensor>,
        request_indices: HashMap<DeviceLocation, Tensor>,
        kv_tile_indices: HashMap<DeviceLocation, Tensor>,
        o_indptr: HashMap<DeviceLocation, Tensor>,
        kv_chunk_size: HashMap<DeviceLocation, Tensor>,
        block_valid_mask: HashMap<DeviceLocation, Tensor>,
    }

    impl DecodeViewDeviceMaps {
        fn insert(
            &mut self,
            host: &DecodeViewHostTensors,
            device: &Device,
            stage_on_host: bool,
        ) -> candle_core::Result<()> {
            let location = device.location();
            if let Some(tensor) = host.block_tables.as_ref() {
                self.block_tables.insert(
                    location,
                    decode_metadata_tensor(tensor, device, stage_on_host)?,
                );
            }
            if let Some(tensor) = host.context_lens.as_ref() {
                self.context_lens.insert(
                    location,
                    decode_metadata_tensor(tensor, device, stage_on_host)?,
                );
            }
            if let Some((indptr, indices, last_page_len)) = host.paged_kv.as_ref() {
                self.paged_kv_indptr.insert(
                    location,
                    decode_metadata_tensor(indptr, device, stage_on_host)?,
                );
                self.paged_kv_indices.insert(
                    location,
                    decode_metadata_tensor(indices, device, stage_on_host)?,
                );
                self.paged_kv_last_page_len.insert(
                    location,
                    decode_metadata_tensor(last_page_len, device, stage_on_host)?,
                );
            }
            if let Some((request, tile, output, chunk, valid)) = host.tile_plan.as_ref() {
                self.request_indices.insert(
                    location,
                    decode_metadata_tensor(request, device, stage_on_host)?,
                );
                self.kv_tile_indices.insert(
                    location,
                    decode_metadata_tensor(tile, device, stage_on_host)?,
                );
                self.o_indptr.insert(
                    location,
                    decode_metadata_tensor(output, device, stage_on_host)?,
                );
                self.kv_chunk_size.insert(
                    location,
                    decode_metadata_tensor(chunk, device, stage_on_host)?,
                );
                self.block_valid_mask.insert(
                    location,
                    decode_metadata_tensor(valid, device, stage_on_host)?,
                );
            }
            Ok(())
        }
    }

    impl DecodePagedRows {
        pub fn batch_size(&self) -> usize {
            self.slot_mappings.len()
        }

        fn paged_block_tables(&self) -> BlockTableRanges<'_> {
            let ranges = self
                .context_lens
                .iter()
                .zip(&self.full_context_lens)
                .enumerate()
                .map(|(row, (&context_len, &full_context_len))| {
                    let table = self.block_tables.row(row);
                    if self.sliding_window.is_none() {
                        return 0..table.len();
                    }
                    let block_start =
                        full_context_len.saturating_sub(context_len) / self.block_size;
                    let block_end = block_start
                        .saturating_add(context_len.div_ceil(self.block_size))
                        .min(table.len());
                    block_start.min(block_end)..block_end
                })
                .collect();
            BlockTableRanges::new(&self.block_tables, ranges)
        }

        #[cfg(feature = "cuda")]
        pub(crate) fn full_block_table(&self, row: usize) -> &[usize] {
            self.block_tables.row(row)
        }

        #[cfg(test)]
        pub(crate) fn materialized_block_tables(&self) -> Vec<Vec<usize>> {
            let tables = self.paged_block_tables();
            (0..tables.len())
                .map(|row| tables.row(row).to_vec())
                .collect()
        }

        /// Pad to `batch_size` rows. Pad rows alias row 0 for every read (same block table and context)
        /// and carry `_PAD_SLOT_ID` slot mappings, so the cache kernels skip their KV writes.
        pub fn padded(&self, batch_size: usize) -> Self {
            let mut rows = self.clone();
            let q = self.query_len;
            while rows.slot_mappings.len() < batch_size {
                rows.slot_mappings
                    .push(vec![_PAD_SLOT_ID; self.slot_mappings[0].len()]);
                rows.block_tables
                    .push_rows_for_table(self.block_tables.row_table_index(0), q);
                rows.context_lens.extend_from_slice(&self.context_lens[..q]);
                rows.full_context_lens
                    .extend_from_slice(&self.full_context_lens[..q]);
            }
            rows
        }

        #[cfg(feature = "cuda")]
        pub(crate) fn graph_key(&self) -> DecodePagedRowsGraphKey {
            let batch_size = self.batch_size();
            assert!(self
                .slot_mappings
                .iter()
                .all(|slots| slots.len() == self.query_len));
            assert_eq!(self.block_tables.len(), batch_size * self.query_len);
            assert_eq!(self.context_lens.len(), batch_size * self.query_len);
            assert_eq!(self.full_context_lens.len(), batch_size * self.query_len);
            let paged_block_tables = self.paged_block_tables();
            let max_context_len = self.context_lens.iter().copied().max().unwrap_or(0);
            let full_max_context_len = self.full_context_lens.iter().copied().max().unwrap_or(0);
            let graph_capacity =
                (!self.use_standard_metadata).then_some(self.max_paged_context_len);
            let paged_graph_capacity = self
                .sliding_window
                .map(|window| {
                    window
                        .saturating_add(self.block_size.saturating_sub(1))
                        .min(self.max_paged_context_len)
                })
                .or(graph_capacity);
            let paged_block_table_len = cuda_graph_block_table_len_with_cap(
                (0..paged_block_tables.len())
                    .map(|row| paged_block_tables.row(row).len())
                    .max()
                    .unwrap_or(1),
                self.block_size,
                true,
                max_context_len,
                paged_graph_capacity,
            );
            let full_block_table_len = cuda_graph_block_table_len_with_cap(
                (0..self.block_tables.len())
                    .map(|row| self.block_tables.row(row).len())
                    .max()
                    .unwrap_or(1),
                self.block_size,
                true,
                full_max_context_len,
                graph_capacity,
            );
            DecodePagedRowsGraphKey {
                batch_size,
                query_len: self.query_len,
                block_size: self.block_size,
                use_standard_metadata: self.use_standard_metadata,
                sliding_window: self.sliding_window,
                decode_window: self.decode_window,
                devices: self.devices.iter().map(Device::location).collect(),
                num_kv_heads: self.num_kv_heads,
                paged_block_table_len,
                full_block_table_len,
            }
        }

        pub fn build(self: &Arc<Self>) -> Result<PagedAttentionInputMetadata> {
            let stage_on_host = crate::perf_flags::cuda_graphs_enabled()
                && self.devices.iter().all(Device::is_cuda);
            if stage_on_host {
                self.build_graph_staged()
            } else {
                self.build_inner(false, PagedDecodeMetadataRequirements::conservative(self))
            }
        }

        pub(crate) fn build_materialized(self: &Arc<Self>) -> Result<PagedAttentionInputMetadata> {
            self.build_inner(false, PagedDecodeMetadataRequirements::conservative(self))
        }

        #[cfg(any(feature = "cuda", test))]
        pub(crate) fn build_graph_update(
            self: &Arc<Self>,
            requirements: PagedDecodeMetadataRequirements,
        ) -> Result<PagedAttentionInputMetadata> {
            self.build_inner(true, requirements)
        }

        pub(crate) fn build_graph_staged(self: &Arc<Self>) -> Result<PagedAttentionInputMetadata> {
            let max_slot_mapping_len = self.slot_mappings.iter().map(Vec::len).max().unwrap_or(1);
            let slot_mappings = _make_tensor_with_pad(
                self.slot_mappings.clone(),
                max_slot_mapping_len,
                _PAD_SLOT_ID,
                &Device::Cpu,
            )?;
            let slot_mappings = self
                .devices
                .iter()
                .map(|device| (device.location(), slot_mappings.clone()))
                .collect();
            let max_context_len = self.context_lens.iter().copied().max().unwrap_or(0);
            let full_max_context_len = self.full_context_lens.iter().copied().max().unwrap_or(0);
            Ok(PagedAttentionInputMetadata {
                block_tables: None,
                context_lens: None,
                block_size: Some(self.block_size),
                paged_context_lens_cpu: Some(self.context_lens.clone()),
                full_paged_context_lens_cpu: Some(self.full_context_lens.clone()),
                slot_mappings,
                max_context_len: self.use_standard_metadata.then_some(max_context_len),
                full_block_tables: None,
                full_context_lens: None,
                full_max_context_len: self.use_standard_metadata.then_some(full_max_context_len),
                is_first_prompt_chunk: false,
                is_final_prompt_chunk: true,
                prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
                has_noncausal_mm_context: false,
                prefix_gather_workspace_limit: None,
                mm_prefix_ranges: None,
                full_mm_prefix_ranges: None,
                prefill_attention_heads: 1,
                prefill_key_value_heads: 1,
                prefill_head_dim: 1,
                flashinfer: None,
                rope_positions: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
                decode_rows: Some(self.clone()),
            })
        }

        fn build_inner(
            self: &Arc<Self>,
            stage_on_host: bool,
            requirements: PagedDecodeMetadataRequirements,
        ) -> Result<PagedAttentionInputMetadata> {
            // Create paged attention tensors on CPU first (see make_prompt_chunk for explanation)
            let max_slot_mapping_len = self.slot_mappings.iter().map(Vec::len).max().unwrap_or(1);
            let slot_mappings = _make_tensor_with_pad(
                self.slot_mappings.clone(),
                max_slot_mapping_len,
                _PAD_SLOT_ID,
                &Device::Cpu,
            )?;

            let block_tables = self.paged_block_tables();
            let paged_attn_context_lens = &self.context_lens;
            let full_block_tables = &self.block_tables;
            let full_paged_attn_context_lens = &self.full_context_lens;
            let block_size = self.block_size;
            let use_standard_metadata = self.use_standard_metadata;
            let max_block_table_len = (0..block_tables.len())
                .map(|row| block_tables.row(row).len())
                .max()
                .expect("block_tables should not be empty when paged attention is enabled");
            let full_max_block_table_len = (0..full_block_tables.len())
                .map(|row| full_block_tables.row(row).len())
                .max()
                .unwrap_or(0)
                .max(1);
            let max_context_len = paged_attn_context_lens.iter().copied().max().unwrap_or(0);
            let full_max_context_len = full_paged_attn_context_lens
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            let graph_capacity = (!use_standard_metadata).then_some(self.max_paged_context_len);
            let paged_graph_capacity = self
                .sliding_window
                .map(|window| {
                    window
                        .saturating_add(block_size.saturating_sub(1))
                        .min(self.max_paged_context_len)
                })
                .or(graph_capacity);
            let max_block_table_len = cuda_graph_block_table_len_with_cap(
                max_block_table_len,
                block_size,
                true,
                max_context_len,
                paged_graph_capacity,
            );

            let batch_size = block_tables.len();
            let paged_kv = requirements
                .flashinfer_paged_kv
                .then(|| {
                    make_paged_kv_tensors(
                        &block_tables,
                        paged_attn_context_lens,
                        block_size,
                        batch_size * max_block_table_len,
                    )
                })
                .transpose()?;
            let tile_plan = requirements
                .flashinfer_tile_plan
                .then(|| {
                    let decode_split_pages = flashinfer_decode_split_pages(
                        block_size,
                        batch_size,
                        self.num_kv_heads,
                        max_context_len,
                    );
                    let tiles_per_row = max_block_table_len
                        .max(1)
                        .div_ceil(flashinfer_decode_split_capacity_pages(block_size));
                    make_paged_kv_decode_tensors(
                        &block_tables,
                        paged_attn_context_lens,
                        block_size,
                        Some(decode_split_pages),
                        batch_size * tiles_per_row,
                    )
                })
                .transpose()?;
            let block_tables_tensor = if requirements.block_tables {
                Some(
                    make_block_table_tensor(&block_tables, max_block_table_len)?
                        .reshape(((), max_block_table_len))?,
                )
            } else {
                None
            };
            let context_lens_tensor = requirements
                .context_lens
                .then(|| {
                    Tensor::from_vec(
                        paged_attn_context_lens
                            .iter()
                            .map(|x| *x as u32)
                            .collect::<Vec<_>>(),
                        (paged_attn_context_lens.len(),),
                        &Device::Cpu,
                    )
                })
                .transpose()?;
            let paged_tensors = DecodeViewHostTensors {
                block_tables: block_tables_tensor,
                context_lens: context_lens_tensor,
                paged_kv,
                tile_plan,
            };
            let full_matches_paged = self.sliding_window.is_none();
            let full_tensors = if full_matches_paged {
                None
            } else {
                let full_max_block_table_len = cuda_graph_block_table_len_with_cap(
                    full_max_block_table_len,
                    block_size,
                    true,
                    full_max_context_len,
                    graph_capacity,
                );
                let block_tables_tensor = if requirements.block_tables {
                    Some(
                        make_block_table_tensor(full_block_tables, full_max_block_table_len)?
                            .reshape(((), full_max_block_table_len))?,
                    )
                } else {
                    None
                };
                let context_lens_tensor = requirements
                    .context_lens
                    .then(|| {
                        Tensor::from_vec(
                            full_paged_attn_context_lens
                                .iter()
                                .map(|x| *x as u32)
                                .collect::<Vec<_>>(),
                            (full_paged_attn_context_lens.len(),),
                            &Device::Cpu,
                        )
                    })
                    .transpose()?;
                let paged_kv = requirements
                    .flashinfer_paged_kv
                    .then(|| {
                        make_paged_kv_tensors(
                            full_block_tables,
                            full_paged_attn_context_lens,
                            block_size,
                            full_block_tables.len() * full_max_block_table_len,
                        )
                    })
                    .transpose()?;
                let tile_plan = requirements
                    .flashinfer_tile_plan
                    .then(|| {
                        let split_pages = flashinfer_decode_split_pages(
                            block_size,
                            batch_size,
                            self.num_kv_heads,
                            full_max_context_len,
                        );
                        let tiles_per_row = full_max_block_table_len
                            .max(1)
                            .div_ceil(flashinfer_decode_split_capacity_pages(block_size));
                        make_paged_kv_decode_tensors(
                            full_block_tables,
                            full_paged_attn_context_lens,
                            block_size,
                            Some(split_pages),
                            full_block_tables.len() * tiles_per_row,
                        )
                    })
                    .transpose()?;
                Some(DecodeViewHostTensors {
                    block_tables: block_tables_tensor,
                    context_lens: context_lens_tensor,
                    paged_kv,
                    tile_plan,
                })
            };

            let mut slot_mappings_map = HashMap::new();
            let mut paged_maps = DecodeViewDeviceMaps::default();
            let mut full_maps = DecodeViewDeviceMaps::default();
            for device in &self.devices {
                slot_mappings_map.insert(
                    device.location(),
                    decode_metadata_tensor(&slot_mappings, device, stage_on_host)?,
                );
                paged_maps.insert(&paged_tensors, device, stage_on_host)?;
                if let Some(full_tensors) = full_tensors.as_ref() {
                    full_maps.insert(full_tensors, device, stage_on_host)?;
                }
            }
            if full_matches_paged {
                full_maps = DecodeViewDeviceMaps {
                    block_tables: paged_maps.block_tables.clone(),
                    context_lens: paged_maps.context_lens.clone(),
                    paged_kv_indptr: paged_maps.paged_kv_indptr.clone(),
                    paged_kv_indices: paged_maps.paged_kv_indices.clone(),
                    paged_kv_last_page_len: paged_maps.paged_kv_last_page_len.clone(),
                    request_indices: paged_maps.request_indices.clone(),
                    kv_tile_indices: paged_maps.kv_tile_indices.clone(),
                    o_indptr: paged_maps.o_indptr.clone(),
                    kv_chunk_size: paged_maps.kv_chunk_size.clone(),
                    block_valid_mask: paged_maps.block_valid_mask.clone(),
                };
            }

            let flashinfer = requirements.flashinfer_paged_kv.then(|| {
                let sliding = (!full_matches_paged).then(|| {
                    flashinfer_view(
                        requirements
                            .context_lens
                            .then_some(paged_maps.block_tables.clone()),
                        requirements
                            .context_lens
                            .then_some(paged_maps.context_lens.clone()),
                        requirements.context_lens.then_some(max_context_len),
                        flashinfer_paged_kv(
                            paged_maps.paged_kv_indptr.clone(),
                            paged_maps.paged_kv_indices.clone(),
                            paged_maps.paged_kv_last_page_len.clone(),
                        ),
                        flashinfer_tile_plan(
                            paged_maps.request_indices.clone(),
                            paged_maps.kv_tile_indices.clone(),
                            paged_maps.o_indptr.clone(),
                            paged_maps.kv_chunk_size.clone(),
                            paged_maps.block_valid_mask.clone(),
                        ),
                    )
                });
                let logical = flashinfer_view(
                    requirements
                        .context_lens
                        .then_some(full_maps.block_tables.clone()),
                    requirements
                        .context_lens
                        .then_some(full_maps.context_lens.clone()),
                    requirements.context_lens.then_some(full_max_context_len),
                    flashinfer_paged_kv(
                        full_maps.paged_kv_indptr.clone(),
                        full_maps.paged_kv_indices.clone(),
                        full_maps.paged_kv_last_page_len.clone(),
                    ),
                    flashinfer_tile_plan(
                        full_maps.request_indices.clone(),
                        full_maps.kv_tile_indices.clone(),
                        full_maps.o_indptr.clone(),
                        full_maps.kv_chunk_size.clone(),
                        full_maps.block_valid_mask.clone(),
                    ),
                );
                flashinfer_metadata(logical, sliding)
            });

            Ok(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: requirements.block_tables.then_some(paged_maps.block_tables),
                context_lens: requirements.context_lens.then_some(paged_maps.context_lens),
                block_size: Some(block_size),
                paged_context_lens_cpu: Some(paged_attn_context_lens.clone()),
                full_paged_context_lens_cpu: Some(full_paged_attn_context_lens.clone()),
                max_context_len: requirements.context_lens.then_some(max_context_len),
                full_block_tables: requirements.block_tables.then_some(full_maps.block_tables),
                full_context_lens: requirements.context_lens.then_some(full_maps.context_lens),
                full_max_context_len: requirements.context_lens.then_some(full_max_context_len),
                is_first_prompt_chunk: false,
                is_final_prompt_chunk: true,
                prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
                has_noncausal_mm_context: false,
                prefix_gather_workspace_limit: None,
                mm_prefix_ranges: None,
                full_mm_prefix_ranges: None,
                prefill_attention_heads: 1,
                prefill_key_value_heads: 1,
                prefill_head_dim: 1,
                flashinfer,
                rope_positions: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
                decode_rows: Some(self.clone()),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_completion_prefill_chunk<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
        decode_window: usize,
    ) -> Result<InputMetadata> {
        let prefix_cache_lens = toks
            .iter()
            .map(|ctxt| ctxt.len().saturating_sub(decode_window))
            .collect::<Vec<_>>();
        make_prompt_chunk(
            0,
            toks,
            &input_seqs.iter().map(|seq| *seq.id()).collect::<Vec<_>>(),
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata,
            mapper,
            Some(&prefix_cache_lens),
            sliding_window,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_prompt_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
    ) -> Result<InnerInputProcessorOutput> {
        let offset = input_seqs[0].token_offset();
        // Collect prefix cache lens when paged attention is in use
        let prefix_cache_lens: Vec<usize> =
            input_seqs.iter().map(|s| s.prefix_cache_len()).collect();
        let has_paged_attn = paged_attn_metadata.is_some();
        make_prompt_chunk(
            offset,
            toks,
            &input_seqs.iter().map(|s| *s.id()).collect::<Vec<_>>(),
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata,
            mapper,
            if has_paged_attn {
                Some(&prefix_cache_lens)
            } else {
                None
            },
            sliding_window,
            true,
        )
        .map(|inputs| InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_completion_input<T: WithDType + std::fmt::Debug + From<u32> + Clone>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
    ) -> Result<InnerInputProcessorOutput> {
        if no_kv_cache {
            return get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
                None,
            );
        }

        make_completion_chunk(
            toks,
            input_seqs,
            device,
            paged_attn_metadata,
            mapper,
            sliding_window,
            1,
        )
        .map(|inputs| InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    /// `get_completion_input` for models that consume more than one new token per decode step
    /// (e.g. block diffusion, where each step feeds the last committed canvas to the encoder).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_completion_input_windowed<
        T: WithDType + std::fmt::Debug + From<u32> + Clone,
    >(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
        decode_window: usize,
    ) -> Result<InnerInputProcessorOutput> {
        if no_kv_cache {
            return get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
                None,
            );
        }

        let inputs = if paged_attn_metadata.is_some() {
            make_completion_prefill_chunk(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
                sliding_window,
                decode_window,
            )
        } else {
            make_completion_chunk(
                toks,
                input_seqs,
                device,
                paged_attn_metadata,
                mapper,
                sliding_window,
                decode_window,
            )
        }?;
        Ok(InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    #[derive(Clone)]
    pub struct ModelInputs {
        pub input_ids: Tensor,
        pub input_ids_full: Option<Tensor>,
        pub seqlen_offsets: Vec<usize>,
        pub seqlen_offsets_full: Option<Vec<usize>>,
        pub context_lens: Vec<(usize, usize)>,
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
        pub flash_meta: FlashParams,
        pub flash_meta_full: Option<FlashParams>,
        pub recurrent_batch_kind: RecurrentBatchKind,
        pub adapter_leases: Arc<[Option<AdapterLease>]>,
    }

    fn adapter_leases(
        input_seqs: &[&mut Sequence],
        seq_indices: &[usize],
    ) -> Arc<[Option<AdapterLease>]> {
        seq_indices
            .iter()
            .map(|&index| input_seqs[index].adapter_lease().cloned())
            .collect::<Vec<_>>()
            .into()
    }

    pub struct TextInputsProcessor;

    impl InputsProcessor for TextInputsProcessor {
        fn process_inputs(
            &self,
            _: Option<Arc<Tokenizer>>,
            input_seqs: &mut [&mut Sequence],
            is_prompt: bool,
            is_xlora: bool,
            device: &Device,
            no_kv_cache: bool,
            last_n_context_len: Option<(usize, usize)>,
            return_raw_logits: bool,
            sliding_window: Option<usize>,
            _: Option<Arc<dyn Any>>,
            mut paged_attn_metadata: Option<PagedAttentionMeta>,
            mapper: Option<&dyn DeviceMapper>,
        ) -> Result<InputProcessorOutput> {
            let flash_sliding_window = if no_kv_cache { None } else { sliding_window };
            if is_xlora && !is_prompt {
                let prompt = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let completion = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids_full,
                            positions: seqlen_offsets_full,
                            context_lens: _,
                            position_ids,
                            paged_attn_meta: _,
                            flash_meta: flash_meta_full,
                        },
                    seq_indices,
                } = prompt;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids: _,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices: _,
                } = completion;
                let adapter_leases = adapter_leases(input_seqs, &seq_indices);
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: Some(input_ids_full),
                    seqlen_offsets,
                    seqlen_offsets_full: Some(seqlen_offsets_full),
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: Some(flash_meta_full),
                    recurrent_batch_kind: RecurrentBatchKind::Decode,
                    adapter_leases,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else if is_xlora && is_prompt {
                let metadata = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let adapter_leases = adapter_leases(input_seqs, &seq_indices);
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids: input_ids.clone(),
                    input_ids_full: Some(input_ids),
                    seqlen_offsets: seqlen_offsets.clone(),
                    seqlen_offsets_full: Some(seqlen_offsets),
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta: flash_meta.clone(),
                    flash_meta_full: Some(flash_meta),
                    recurrent_batch_kind: RecurrentBatchKind::Prefill,
                    adapter_leases,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else if is_prompt {
                let metadata = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let adapter_leases = adapter_leases(input_seqs, &seq_indices);
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: None,
                    recurrent_batch_kind: RecurrentBatchKind::Prefill,
                    adapter_leases,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else {
                let recurrent_batch_kind = recurrent_batch_kind_for_input(
                    false,
                    crate::speculative::staging::staged_batch_width(input_seqs).is_some(),
                );
                let metadata = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let adapter_leases = adapter_leases(input_seqs, &seq_indices);
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: None,
                    recurrent_batch_kind,
                    adapter_leases,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            }
        }

        fn get_type(&self) -> InputsProcessorType {
            InputsProcessorType::Text
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn assert_zero_padded_rows(rows: &[Vec<u32>], prefixes: &[&[u32]]) {
            assert_eq!(rows.len(), prefixes.len());
            for (row, prefix) in rows.iter().zip(prefixes) {
                assert_eq!(&row[..prefix.len()], *prefix);
                assert!(row[prefix.len()..].iter().all(|&value| value == 0));
            }
        }

        #[test]
        fn completion_input_keeps_staged_rows_device_backed() {
            let staged = vec![
                Tensor::from_vec(vec![10u32, 11], 2, &Device::Cpu).unwrap(),
                Tensor::from_vec(vec![20u32, 21], 2, &Device::Cpu).unwrap(),
            ];
            let input =
                completion_input_tensor(vec![7u32, 8], 2, 1, &staged, &Device::Cpu).unwrap();
            assert_eq!(
                input.to_vec2::<u32>().unwrap(),
                vec![vec![7, 10, 11], vec![8, 20, 21]]
            );
        }

        #[test]
        fn cuda_graph_context_buckets_track_live_rows() {
            const CACHE_CAPACITY: usize = 1_604_288;
            assert_eq!(
                cuda_graph_block_table_len_with_cap(4, 32, true, 128, Some(CACHE_CAPACITY)),
                16
            );
            assert_eq!(
                cuda_graph_block_table_len_with_cap(16, 32, true, 512, Some(CACHE_CAPACITY)),
                16
            );
            assert_eq!(
                cuda_graph_block_table_len_with_cap(17, 32, true, 513, Some(CACHE_CAPACITY)),
                32
            );
            assert_eq!(
                cuda_graph_block_table_len_with_cap(33, 32, true, 1025, Some(CACHE_CAPACITY)),
                64
            );
        }

        #[test]
        fn flashinfer_decode_metadata_ignores_aggregate_cache_capacity() {
            const CACHE_CAPACITY: usize = 1_604_288;
            let table = vec![1, 2, 3, 4];
            let metadata = Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![127]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
                context_lens: vec![128],
                full_context_lens: vec![128],
                query_len: 1,
                block_size: 32,
                use_standard_metadata: false,
                max_paged_context_len: CACHE_CAPACITY,
                sliding_window: None,
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 4,
            })
            .build()
            .unwrap();
            let view = &metadata.flashinfer.unwrap().views.logical;
            assert_eq!(view.paged_kv.indices[&Device::Cpu.location()].dims(), &[16]);
            assert_eq!(
                view.tile_plan.request_indices[&Device::Cpu.location()].dims(),
                &[2]
            );
        }

        #[test]
        fn fa3_graph_update_builds_csr_without_fallback_metadata() {
            let table = vec![1, 2, 3, 4];
            let rows = Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![127]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
                context_lens: vec![128],
                full_context_lens: vec![128],
                query_len: 1,
                block_size: 32,
                use_standard_metadata: false,
                max_paged_context_len: 1_604_288,
                sliding_window: None,
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 4,
            });
            let metadata = rows
                .build_graph_update(PagedDecodeMetadataRequirements::graph(
                    false, false, true, false,
                ))
                .unwrap();
            assert!(metadata.block_tables.is_none());
            assert!(metadata.context_lens.is_none());
            let view = &metadata.flashinfer.unwrap().views.logical;
            assert!(!view.paged_kv.indices.is_empty());
            assert!(view.tile_plan.request_indices.is_empty());
            assert!(view.tile_plan.kv_tile_indices.is_empty());
            assert!(view.tile_plan.o_indptr.is_empty());
            assert!(view.tile_plan.kv_chunk_size.is_empty());
            assert!(view.tile_plan.block_valid_mask.is_empty());
        }

        #[test]
        fn padded_decode_rows_alias_row_zero_without_kv_writes() {
            let rows = DecodePagedRows {
                slot_mappings: vec![vec![40, 41], vec![72, 73]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                    vec![vec![1, 2], vec![3]],
                    2,
                ),
                context_lens: vec![40, 41, 8, 9],
                full_context_lens: vec![40, 41, 8, 9],
                query_len: 2,
                block_size: 32,
                use_standard_metadata: true,
                max_paged_context_len: 1024,
                sliding_window: None,
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 4,
            };
            let padded = rows.padded(4);
            assert_eq!(padded.batch_size(), 4);
            assert_eq!(padded.slot_mappings[2], vec![_PAD_SLOT_ID, _PAD_SLOT_ID]);
            assert_eq!(padded.slot_mappings[3], vec![_PAD_SLOT_ID, _PAD_SLOT_ID]);
            assert_eq!(padded.block_tables.len(), 8);
            assert_eq!(padded.block_tables.unique_table_count(), 2);
            assert_eq!(
                &padded.materialized_block_tables()[4..],
                &[vec![1, 2], vec![1, 2], vec![1, 2], vec![1, 2]]
            );
            assert_eq!(&padded.context_lens[4..], &[40, 41, 40, 41]);
            assert_eq!(&padded.full_context_lens[4..], &[40, 41, 40, 41]);
            let metadata = Arc::new(padded).build().unwrap();
            assert_eq!(metadata.slot_mappings[&Device::Cpu.location()].dims(), &[8]);
            assert_eq!(
                metadata.paged_context_lens_cpu.as_deref(),
                Some(&[40, 41, 8, 9, 40, 41, 40, 41][..])
            );
            assert!(metadata.decode_rows.is_some());
        }

        #[test]
        fn unsliding_decode_rows_share_canonical_tables_and_metadata() {
            let rows = Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![39, 40], vec![71, 72]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                    vec![vec![10, 11, 12], vec![20, 21, 22]],
                    2,
                ),
                context_lens: vec![9, 10, 7, 8],
                full_context_lens: vec![9, 10, 7, 8],
                query_len: 2,
                block_size: 4,
                use_standard_metadata: true,
                max_paged_context_len: 128,
                sliding_window: None,
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 1,
            });

            assert_eq!(rows.block_tables.unique_table_count(), 2);
            assert_eq!(
                rows.materialized_block_tables(),
                vec![
                    vec![10, 11, 12],
                    vec![10, 11, 12],
                    vec![20, 21, 22],
                    vec![20, 21, 22],
                ]
            );

            let metadata = rows.build_materialized().unwrap();
            let location = Device::Cpu.location();
            assert_eq!(
                metadata.block_tables.as_ref().unwrap()[&location].id(),
                metadata.full_block_tables.as_ref().unwrap()[&location].id()
            );
            let flashinfer = metadata.flashinfer.unwrap();
            assert!(flashinfer.views.sliding.is_none());
            let block_tables = flashinfer.views.logical.block_tables.unwrap()[&location]
                .to_vec2::<u32>()
                .unwrap();
            assert_zero_padded_rows(
                &block_tables,
                &[&[10, 11, 12], &[10, 11, 12], &[20, 21, 22], &[20, 21, 22]],
            );
        }

        #[test]
        fn sliding_decode_rows_materialize_exact_logical_and_windowed_tables() {
            let rows = Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![47, 48], vec![91, 92]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                    vec![vec![10, 11, 12, 13], vec![20, 21, 22]],
                    2,
                ),
                context_lens: vec![5, 6, 4, 5],
                full_context_lens: vec![9, 10, 4, 5],
                query_len: 2,
                block_size: 4,
                use_standard_metadata: true,
                max_paged_context_len: 128,
                sliding_window: Some(4),
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 1,
            });

            assert_eq!(
                rows.materialized_block_tables(),
                vec![vec![11, 12], vec![11, 12], vec![20], vec![20, 21]]
            );
            let metadata = rows.build_materialized().unwrap();
            let location = Device::Cpu.location();
            assert_eq!(
                metadata.block_tables.unwrap()[&location]
                    .to_vec2::<u32>()
                    .unwrap(),
                vec![vec![11, 12], vec![11, 12], vec![20, 0], vec![20, 21]]
            );
            let full_block_tables = metadata.full_block_tables.unwrap()[&location]
                .to_vec2::<u32>()
                .unwrap();
            assert_zero_padded_rows(
                &full_block_tables,
                &[
                    &[10, 11, 12, 13],
                    &[10, 11, 12, 13],
                    &[20, 21, 22],
                    &[20, 21, 22],
                ],
            );
        }

        #[test]
        fn ragged_prompt_selects_each_last_real_token() {
            let short = [1u32, 2];
            let long = [3u32, 4, 5, 6];
            let input = make_prompt_chunk(
                0,
                vec![short.as_slice(), long.as_slice()],
                &[0, 1],
                &Device::Cpu,
                None,
                false,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

            assert_eq!(input.input.dims(), &[2, 4]);
            assert_eq!(input.context_lens, vec![(1, 1), (3, 1)]);
        }

        #[test]
        fn packed_rope_positions_preserve_logical_offsets() {
            let positions = packed_rope_positions(&[0, 16, 32], &[3, 1, 4]).unwrap();

            assert_eq!(positions, vec![0, 1, 2, 16, 32, 33, 34, 35]);
        }

        #[test]
        fn packed_rope_positions_reject_mismatched_metadata() {
            let error = packed_rope_positions(&[0, 16], &[3]).unwrap_err();

            assert!(error.to_string().contains("2 offsets for 1 queries"));
        }

        #[test]
        fn packed_flash_params_preserve_ragged_boundaries() {
            let params = make_flash_params(
                &Device::Cpu,
                None,
                &[0, 3, 1, 4],
                &[0, 3, 1, 4],
                None,
                true,
                true,
            )
            .unwrap();
            let cumulative = params.cumulative_seqlens_q[&Device::Cpu.location()]
                .to_vec1::<u32>()
                .unwrap();

            assert_eq!(params.max_q, 4);
            assert_eq!(cumulative, vec![0, 3, 4, 8]);
        }

        #[test]
        fn sliding_single_token_uses_the_retained_window() {
            assert_eq!(
                sliding_k_lengths(&[0, 1], &[0, 101], 4).unwrap(),
                vec![0, 4]
            );
        }

        #[test]
        fn sliding_query_longer_than_the_window_keeps_the_full_query() {
            assert_eq!(sliding_k_lengths(&[0, 8], &[0, 8], 4).unwrap(), vec![0, 8]);
        }

        #[test]
        fn sliding_cached_multi_token_append_keeps_retained_and_new_tokens() {
            assert_eq!(
                sliding_k_lengths(&[0, 3], &[0, 103], 4).unwrap(),
                vec![0, 7]
            );
        }

        #[test]
        fn fresh_packed_sliding_metadata_preserves_logical_boundaries() {
            let params = make_flash_params(
                &Device::Cpu,
                None,
                &[0, 3, 1, 6],
                &[0, 3, 1, 6],
                Some(4),
                true,
                true,
            )
            .unwrap();
            let sliding = params.sliding_k.unwrap();

            assert_eq!(sliding.max, 6);
            assert_eq!(
                sliding.cumulative_seqlens[&Device::Cpu.location()]
                    .to_vec1::<u32>()
                    .unwrap(),
                vec![0, 3, 4, 10]
            );
        }

        #[test]
        fn sliding_metadata_rejects_inconsistent_lengths() {
            assert!(sliding_k_lengths(&[0, 2], &[0], 4).is_err());
            assert!(sliding_k_lengths(&[0, 3], &[0, 2], 4).is_err());
        }
    }
}
