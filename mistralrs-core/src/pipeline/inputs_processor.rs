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

/// Processor: Prepare inputs for the model (potentially preparing the images if applicable)
pub trait InputsProcessor {
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
        get_mut_arcmutex,
        paged_attention::{AttentionBackendKind, KVCacheManager, _PAD_SLOT_ID},
        sequence::Sequence,
    };

    use super::{InputProcessorOutput, InputsProcessor, InputsProcessorType};

    const CUDA_GRAPH_CONTEXT_BUCKET_TOKENS: usize = 256;
    const FLASHINFER_DECODE_SPLIT_PAGES: usize = 1;
    pub(crate) const FLASHINFER_PREFILL_TILE_Q: usize = 64;
    pub(crate) const FLASHINFER_PREFILL_MAX_GROUP_SIZE: usize = 8;
    const TABLE_SIGNATURE_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const TABLE_SIGNATURE_PRIME: u64 = 0x100000001b3;

    fn cuda_graph_block_table_len(blocks: usize, block_size: usize) -> usize {
        if crate::perf_flags::cuda_graphs_enabled() {
            let block_bucket = CUDA_GRAPH_CONTEXT_BUCKET_TOKENS.div_ceil(block_size).max(1);
            blocks.div_ceil(block_bucket).max(1) * block_bucket
        } else {
            blocks
        }
    }

    fn make_paged_kv_tensors(
        tables: &[Vec<usize>],
        context_lens: &[usize],
        block_size: usize,
        padded_indices_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let batch_size = tables.len();
        let mut paged_kv_indices = Vec::new();
        let mut paged_kv_indptr = Vec::with_capacity(batch_size + 1);
        let mut paged_kv_last_page_len = Vec::with_capacity(batch_size);
        paged_kv_indptr.push(0i32);
        let mut nnz_pages = 0i32;
        for (table, context_len) in tables.iter().zip(context_lens.iter()) {
            let num_blocks = context_len.div_ceil(block_size);
            if num_blocks > table.len() {
                anyhow::bail!(
                    "paged kv block table is too small: context_len={} block_size={} blocks={} table_len={}",
                    context_len,
                    block_size,
                    num_blocks,
                    table.len()
                );
            }
            nnz_pages += num_blocks as i32;
            paged_kv_indptr.push(nnz_pages);
            paged_kv_indices.extend(table.iter().take(num_blocks).map(|x| *x as i32));
            let last_page_len = if num_blocks == 0 {
                0usize
            } else {
                let consumed = (num_blocks - 1) * block_size;
                if *context_len < consumed {
                    anyhow::bail!(
                        "paged kv context len underflow: context_len={} consumed={}",
                        context_len,
                        consumed
                    );
                }
                *context_len - consumed
            };
            paged_kv_last_page_len.push(last_page_len as i32);
        }
        if paged_kv_indices.len() > padded_indices_len {
            anyhow::bail!(
                "paged kv indices exceed padded length: nnz={} padded={}",
                paged_kv_indices.len(),
                padded_indices_len
            );
        }
        paged_kv_indices.resize(padded_indices_len, 0);

        let paged_kv_indptr = Tensor::from_vec(paged_kv_indptr, (batch_size + 1,), &Device::Cpu)?;
        let paged_kv_indices =
            Tensor::from_vec(paged_kv_indices, (padded_indices_len,), &Device::Cpu)?;
        let paged_kv_last_page_len =
            Tensor::from_vec(paged_kv_last_page_len, (batch_size,), &Device::Cpu)?;
        Ok((paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len))
    }

    fn block_table_signature(
        tables: &[Vec<usize>],
        context_lens: &[usize],
        block_size: usize,
    ) -> Vec<u64> {
        tables
            .iter()
            .zip(context_lens.iter())
            .map(|(table, context_len)| {
                let blocks = context_len.div_ceil(block_size).min(table.len());
                let mut hash = TABLE_SIGNATURE_OFFSET_BASIS;
                hash = (hash ^ blocks as u64).wrapping_mul(TABLE_SIGNATURE_PRIME);
                for block in table.iter().take(blocks) {
                    hash = (hash ^ *block as u64).wrapping_mul(TABLE_SIGNATURE_PRIME);
                }
                hash
            })
            .collect()
    }

    fn make_paged_kv_decode_tensors(
        tables: &[Vec<usize>],
        context_lens: &[usize],
        block_size: usize,
        split_pages: Option<usize>,
        padded_tiles_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        if tables.len() != context_lens.len() {
            anyhow::bail!(
                "paged kv decode table/context length mismatch: tables={} context_lens={}",
                tables.len(),
                context_lens.len()
            );
        }
        let chunk_pages = split_pages.unwrap_or(usize::MAX).max(1);
        let mut request_indices = Vec::new();
        let mut kv_tile_indices = Vec::new();
        let mut o_indptr = Vec::with_capacity(tables.len() + 1);
        o_indptr.push(0i32);
        for (batch_idx, (table, context_len)) in tables.iter().zip(context_lens.iter()).enumerate()
        {
            let num_blocks = context_len.div_ceil(block_size);
            if num_blocks > table.len() {
                anyhow::bail!(
                    "paged kv decode block table is too small: context_len={} block_size={} blocks={} table_len={}",
                    context_len,
                    block_size,
                    num_blocks,
                    table.len()
                );
            }
            let num_chunks = num_blocks.max(1).div_ceil(chunk_pages);
            for kv_tile_idx in 0..num_chunks {
                request_indices.push(batch_idx as i32);
                kv_tile_indices.push(kv_tile_idx as i32);
            }
            o_indptr.push(request_indices.len() as i32);
        }
        if request_indices.len() > padded_tiles_len {
            anyhow::bail!(
                "paged kv decode tiles exceed padded length: tiles={} padded={}",
                request_indices.len(),
                padded_tiles_len
            );
        }
        let valid_tiles_len = request_indices.len();
        request_indices.resize(padded_tiles_len, 0);
        kv_tile_indices.resize(padded_tiles_len, 0);
        let mut block_valid_mask = vec![1u8; valid_tiles_len];
        block_valid_mask.resize(padded_tiles_len, 0);

        let request_indices = Tensor::from_vec(request_indices, (padded_tiles_len,), &Device::Cpu)?;
        let kv_tile_indices = Tensor::from_vec(kv_tile_indices, (padded_tiles_len,), &Device::Cpu)?;
        let o_indptr = Tensor::from_vec(o_indptr, (tables.len() + 1,), &Device::Cpu)?;
        let chunk_size = split_pages.unwrap_or(1) * block_size;
        let kv_chunk_size = Tensor::from_vec(vec![chunk_size as i32], (1,), &Device::Cpu)?;
        let block_valid_mask =
            Tensor::from_vec(block_valid_mask, (padded_tiles_len,), &Device::Cpu)?;
        Ok((
            request_indices,
            kv_tile_indices,
            o_indptr,
            kv_chunk_size,
            block_valid_mask,
        ))
    }

    fn make_paged_kv_decode_tensors_from_lens(
        context_lens: &[usize],
        block_size: usize,
        split_pages: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let chunk_pages = split_pages.unwrap_or(usize::MAX).max(1);
        let tables = context_lens
            .iter()
            .map(|len| vec![0; len.div_ceil(block_size)])
            .collect::<Vec<_>>();
        let padded_tiles_len = context_lens
            .iter()
            .map(|len| len.div_ceil(block_size).max(1).div_ceil(chunk_pages))
            .sum::<usize>()
            .max(1);
        make_paged_kv_decode_tensors(
            &tables,
            context_lens,
            block_size,
            split_pages,
            padded_tiles_len,
        )
    }

    fn make_paged_kv_prefill_tensors(
        query_lens: &[usize],
        block_size: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let mut q_indptr = Vec::with_capacity(query_lens.len() + 1);
        let mut request_indices = Vec::new();
        let mut qo_tile_indices = Vec::new();
        let mut kv_tile_indices = Vec::new();
        q_indptr.push(0i32);
        let mut total_q = 0i32;

        for (batch_idx, &query_len) in query_lens.iter().enumerate() {
            let packed_query_len = query_len * FLASHINFER_PREFILL_MAX_GROUP_SIZE;
            for qo_tile_idx in 0..packed_query_len.div_ceil(FLASHINFER_PREFILL_TILE_Q) {
                request_indices.push(batch_idx as i32);
                qo_tile_indices.push(qo_tile_idx as i32);
                kv_tile_indices.push(0i32);
            }
            total_q += query_len as i32;
            q_indptr.push(total_q);
        }

        let tile_count = request_indices.len();
        let block_valid_mask = vec![1u8; tile_count];
        let q_indptr_tensor =
            Tensor::from_vec(q_indptr.clone(), (query_lens.len() + 1,), &Device::Cpu)?;
        let request_indices = Tensor::from_vec(request_indices, (tile_count,), &Device::Cpu)?;
        let qo_tile_indices = Tensor::from_vec(qo_tile_indices, (tile_count,), &Device::Cpu)?;
        let kv_tile_indices = Tensor::from_vec(kv_tile_indices, (tile_count,), &Device::Cpu)?;
        let o_indptr = Tensor::from_vec(q_indptr, (query_lens.len() + 1,), &Device::Cpu)?;
        let kv_chunk_size = Tensor::from_vec(vec![block_size as i32], (1,), &Device::Cpu)?;
        let block_valid_mask = Tensor::from_vec(block_valid_mask, (tile_count,), &Device::Cpu)?;

        Ok((
            q_indptr_tensor,
            request_indices,
            qo_tile_indices,
            kv_tile_indices,
            o_indptr,
            kv_chunk_size,
            block_valid_mask,
        ))
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

    #[derive(Clone)]
    pub struct PagedAttentionMeta {
        pub sliding_window: Option<usize>,
        pub block_size: usize,
        pub attention_backend: AttentionBackendKind,
        pub kv_cache_manager: Arc<tokio::sync::Mutex<KVCacheManager>>,
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
        pub disable_cuda_graphs: bool,
        pub paged_kv_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_last_page_len: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_last_page_len: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_q_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_qo_tile_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_request_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_tile_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_o_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_chunk_size: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_block_valid_mask: Option<HashMap<DeviceLocation, Tensor>>,
        pub block_table_signature: Option<Vec<u64>>,
        pub full_paged_kv_q_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_qo_tile_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_request_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_tile_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_o_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_chunk_size: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_paged_kv_block_valid_mask: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_block_table_signature: Option<Vec<u64>>,
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
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub(crate) struct FlashInferDecodeMetadata<'a> {
        pub paged_kv_indptr: &'a Tensor,
        pub paged_kv_indices: &'a Tensor,
        pub paged_kv_last_page_len: &'a Tensor,
        pub request_indices: &'a Tensor,
        pub kv_tile_indices: &'a Tensor,
        pub o_indptr: &'a Tensor,
        pub kv_chunk_size: &'a Tensor,
        pub block_valid_mask: &'a Tensor,
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub(crate) struct FlashInferPrefillMetadata<'a> {
        pub paged_kv_indptr: &'a Tensor,
        pub paged_kv_indices: &'a Tensor,
        pub paged_kv_last_page_len: &'a Tensor,
        pub q_indptr: &'a Tensor,
        pub request_indices: &'a Tensor,
        pub qo_tile_indices: &'a Tensor,
        pub kv_tile_indices: &'a Tensor,
        pub o_indptr: &'a Tensor,
        pub kv_chunk_size: &'a Tensor,
        pub block_valid_mask: &'a Tensor,
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn get_metadata_tensor<'a>(
        map: Option<&'a HashMap<DeviceLocation, Tensor>>,
        device: &DeviceLocation,
        missing_msg: &'static str,
    ) -> candle_core::Result<&'a Tensor> {
        map.and_then(|tensors| tensors.get(device))
            .ok_or_else(|| candle_core::Error::msg(missing_msg))
    }

    impl PagedAttentionInputMetadata {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        pub(crate) fn flashinfer_decode_metadata(
            &self,
            device: &DeviceLocation,
            use_full: bool,
            _use_tensor_cores: bool,
        ) -> candle_core::Result<FlashInferDecodeMetadata<'_>> {
            let paged_kv_indptr = if use_full {
                self.full_paged_kv_indptr.as_ref()
            } else {
                self.paged_kv_indptr.as_ref()
            };
            let paged_kv_indices = if use_full {
                self.full_paged_kv_indices.as_ref()
            } else {
                self.paged_kv_indices.as_ref()
            };
            let paged_kv_last_page_len = if use_full {
                self.full_paged_kv_last_page_len.as_ref()
            } else {
                self.paged_kv_last_page_len.as_ref()
            };
            let request_indices = if use_full {
                self.full_paged_kv_request_indices
                    .as_ref()
                    .or(self.paged_kv_request_indices.as_ref())
            } else {
                self.paged_kv_request_indices.as_ref()
            };
            let kv_tile_indices = if use_full {
                self.full_paged_kv_tile_indices
                    .as_ref()
                    .or(self.paged_kv_tile_indices.as_ref())
            } else {
                self.paged_kv_tile_indices.as_ref()
            };
            let o_indptr = if use_full {
                self.full_paged_kv_o_indptr
                    .as_ref()
                    .or(self.paged_kv_o_indptr.as_ref())
            } else {
                self.paged_kv_o_indptr.as_ref()
            };
            let kv_chunk_size = if use_full {
                self.full_paged_kv_chunk_size
                    .as_ref()
                    .or(self.paged_kv_chunk_size.as_ref())
            } else {
                self.paged_kv_chunk_size.as_ref()
            };
            let block_valid_mask = if use_full {
                self.full_paged_kv_block_valid_mask
                    .as_ref()
                    .or(self.paged_kv_block_valid_mask.as_ref())
            } else {
                self.paged_kv_block_valid_mask.as_ref()
            };

            Ok(FlashInferDecodeMetadata {
                paged_kv_indptr: get_metadata_tensor(
                    paged_kv_indptr,
                    device,
                    "paged_kv_indptr missing",
                )?,
                paged_kv_indices: get_metadata_tensor(
                    paged_kv_indices,
                    device,
                    "paged_kv_indices missing",
                )?,
                paged_kv_last_page_len: get_metadata_tensor(
                    paged_kv_last_page_len,
                    device,
                    "paged_kv_last_page_len missing",
                )?,
                request_indices: get_metadata_tensor(
                    request_indices,
                    device,
                    "paged_kv_request_indices missing",
                )?,
                kv_tile_indices: get_metadata_tensor(
                    kv_tile_indices,
                    device,
                    "paged_kv_tile_indices missing",
                )?,
                o_indptr: get_metadata_tensor(o_indptr, device, "paged_kv_o_indptr missing")?,
                kv_chunk_size: get_metadata_tensor(
                    kv_chunk_size,
                    device,
                    "paged_kv_chunk_size missing",
                )?,
                block_valid_mask: get_metadata_tensor(
                    block_valid_mask,
                    device,
                    "paged_kv_block_valid_mask missing",
                )?,
            })
        }

        #[cfg(all(feature = "cuda", target_family = "unix"))]
        pub(crate) fn flashinfer_prefill_metadata(
            &self,
            device: &DeviceLocation,
            use_full: bool,
        ) -> candle_core::Result<FlashInferPrefillMetadata<'_>> {
            let paged_kv_indptr = if use_full {
                self.full_paged_kv_indptr.as_ref()
            } else {
                self.paged_kv_indptr.as_ref()
            };
            let paged_kv_indices = if use_full {
                self.full_paged_kv_indices.as_ref()
            } else {
                self.paged_kv_indices.as_ref()
            };
            let paged_kv_last_page_len = if use_full {
                self.full_paged_kv_last_page_len.as_ref()
            } else {
                self.paged_kv_last_page_len.as_ref()
            };
            let q_indptr = if use_full {
                self.full_paged_kv_q_indptr.as_ref()
            } else {
                self.paged_kv_q_indptr.as_ref()
            };
            let request_indices = if use_full {
                self.full_paged_kv_request_indices.as_ref()
            } else {
                self.paged_kv_request_indices.as_ref()
            };
            let qo_tile_indices = if use_full {
                self.full_paged_kv_qo_tile_indices.as_ref()
            } else {
                self.paged_kv_qo_tile_indices.as_ref()
            };
            let kv_tile_indices = if use_full {
                self.full_paged_kv_tile_indices.as_ref()
            } else {
                self.paged_kv_tile_indices.as_ref()
            };
            let o_indptr = if use_full {
                self.full_paged_kv_o_indptr.as_ref()
            } else {
                self.paged_kv_o_indptr.as_ref()
            };
            let kv_chunk_size = if use_full {
                self.full_paged_kv_chunk_size.as_ref()
            } else {
                self.paged_kv_chunk_size.as_ref()
            };
            let block_valid_mask = if use_full {
                self.full_paged_kv_block_valid_mask.as_ref()
            } else {
                self.paged_kv_block_valid_mask.as_ref()
            };

            Ok(FlashInferPrefillMetadata {
                paged_kv_indptr: get_metadata_tensor(
                    paged_kv_indptr,
                    device,
                    "paged_kv_indptr missing",
                )?,
                paged_kv_indices: get_metadata_tensor(
                    paged_kv_indices,
                    device,
                    "paged_kv_indices missing",
                )?,
                paged_kv_last_page_len: get_metadata_tensor(
                    paged_kv_last_page_len,
                    device,
                    "paged_kv_last_page_len missing",
                )?,
                q_indptr: get_metadata_tensor(q_indptr, device, "paged_kv_q_indptr missing")?,
                request_indices: get_metadata_tensor(
                    request_indices,
                    device,
                    "paged_kv_request_indices missing",
                )?,
                qo_tile_indices: get_metadata_tensor(
                    qo_tile_indices,
                    device,
                    "paged_kv_qo_tile_indices missing",
                )?,
                kv_tile_indices: get_metadata_tensor(
                    kv_tile_indices,
                    device,
                    "paged_kv_tile_indices missing",
                )?,
                o_indptr: get_metadata_tensor(o_indptr, device, "paged_kv_o_indptr missing")?,
                kv_chunk_size: get_metadata_tensor(
                    kv_chunk_size,
                    device,
                    "paged_kv_chunk_size missing",
                )?,
                block_valid_mask: get_metadata_tensor(
                    block_valid_mask,
                    device,
                    "paged_kv_block_valid_mask missing",
                )?,
            })
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
                disable_cuda_graphs: false,
                paged_kv_indptr: None,
                paged_kv_indices: None,
                paged_kv_last_page_len: None,
                full_paged_kv_indptr: None,
                full_paged_kv_indices: None,
                full_paged_kv_last_page_len: None,
                paged_kv_q_indptr: None,
                paged_kv_qo_tile_indices: None,
                paged_kv_request_indices: None,
                paged_kv_tile_indices: None,
                paged_kv_o_indptr: None,
                paged_kv_chunk_size: None,
                paged_kv_block_valid_mask: None,
                block_table_signature: None,
                full_paged_kv_q_indptr: None,
                full_paged_kv_qo_tile_indices: None,
                full_paged_kv_request_indices: None,
                full_paged_kv_tile_indices: None,
                full_paged_kv_o_indptr: None,
                full_paged_kv_chunk_size: None,
                full_paged_kv_block_valid_mask: None,
                full_block_table_signature: None,
                rope_positions: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
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
            let rope_positions_cpu = Tensor::from_vec(
                num_cached_tokens
                    .iter()
                    .map(|len| *len as u32)
                    .collect::<Vec<_>>(),
                (batch_size,),
                &Device::Cpu,
            )?;

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
            let (
                q_indptr_cpu,
                _prefill_request_indices_cpu,
                qo_tile_indices_cpu,
                _prefill_kv_tile_indices_cpu,
                _prefill_o_indptr_cpu,
                _prefill_kv_chunk_size_cpu,
                _prefill_block_valid_mask_cpu,
            ) = make_paged_kv_prefill_tensors(query_lens, 1)?;
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
                Some(FLASHINFER_DECODE_SPLIT_PAGES),
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
                Some(FLASHINFER_DECODE_SPLIT_PAGES),
            )?;

            let mut slot_mappings = HashMap::new();
            let mut context_lens_map = HashMap::new();
            let mut rope_positions = HashMap::new();
            let mut cu_q_map = HashMap::new();
            let mut cu_kv_map = HashMap::new();
            let mut q_indptr_map = HashMap::new();
            let mut qo_tile_indices_map = HashMap::new();
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
                q_indptr_map.insert(device.location(), q_indptr_cpu.to_device(device)?);
                qo_tile_indices_map
                    .insert(device.location(), qo_tile_indices_cpu.to_device(device)?);
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
                disable_cuda_graphs: self.disable_cuda_graphs,
                paged_kv_indptr: self.paged_kv_indptr.clone(),
                paged_kv_indices: self.paged_kv_indices.clone(),
                paged_kv_last_page_len: self.paged_kv_last_page_len.clone(),
                full_paged_kv_indptr: self.full_paged_kv_indptr.clone(),
                full_paged_kv_indices: self.full_paged_kv_indices.clone(),
                full_paged_kv_last_page_len: self.full_paged_kv_last_page_len.clone(),
                paged_kv_q_indptr: Some(q_indptr_map.clone()),
                paged_kv_qo_tile_indices: Some(qo_tile_indices_map.clone()),
                paged_kv_request_indices: Some(decode_request_indices_map.clone()),
                paged_kv_tile_indices: Some(decode_kv_tile_indices_map.clone()),
                paged_kv_o_indptr: Some(decode_o_indptr_map.clone()),
                paged_kv_chunk_size: Some(decode_kv_chunk_size_map.clone()),
                paged_kv_block_valid_mask: Some(decode_block_valid_mask_map.clone()),
                block_table_signature: self.block_table_signature.clone(),
                full_paged_kv_q_indptr: self
                    .full_paged_kv_indptr
                    .as_ref()
                    .map(|_| q_indptr_map.clone()),
                full_paged_kv_qo_tile_indices: self
                    .full_paged_kv_indptr
                    .as_ref()
                    .map(|_| qo_tile_indices_map.clone()),
                full_paged_kv_request_indices: self
                    .full_paged_kv_indptr
                    .as_ref()
                    .map(|_| full_decode_request_indices_map.clone()),
                full_paged_kv_tile_indices: self
                    .full_paged_kv_indptr
                    .as_ref()
                    .map(|_| full_decode_kv_tile_indices_map.clone()),
                full_paged_kv_o_indptr: self
                    .full_paged_kv_indptr
                    .as_ref()
                    .map(|_| full_decode_o_indptr_map.clone()),
                full_paged_kv_chunk_size: self
                    .full_paged_kv_indptr
                    .as_ref()
                    .map(|_| full_decode_kv_chunk_size_map.clone()),
                full_paged_kv_block_valid_mask: self
                    .full_paged_kv_indptr
                    .as_ref()
                    .map(|_| full_decode_block_valid_mask_map.clone()),
                full_block_table_signature: self.full_block_table_signature.clone(),
                rope_positions: Some(rope_positions),
                num_cached_tokens: Some(num_cached_tokens.to_vec()),
                query_lens: Some(query_lens.to_vec()),
                cu_seqlens_q: Some(cu_q_map),
                cu_seqlens_kv: Some(cu_kv_map),
            })
        }
    }

    /// Flash attention sequence length metadata.
    ///
    /// `cumulative_seqlens_q/k` use **padded** lengths (each sequence is padded to
    /// `max_len` in the batch). This matches the padded Q tensor in the normal
    /// prefill and decode paths.
    ///
    /// `logical_k` describes full logical K lengths. `sliding_k`, when present,
    /// describes retained K lengths after a rotating/sliding KV cache has already
    /// truncated K/V to the configured window.
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
    }

    impl FlashParams {
        pub fn empty(causal: bool) -> Self {
            Self {
                max_q: 0,
                cumulative_seqlens_q: HashMap::new(),
                logical_k: FlashKMeta::empty(),
                sliding_k: None,
                causal,
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

    pub(crate) fn make_flash_params(
        device: &Device,
        mapper: Option<&dyn DeviceMapper>,
        seqlens_q: &[u32],
        seqlens_k: &[u32],
        sliding_window: Option<usize>,
        causal: bool,
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
                let window = window as u32;
                let sliding_seqlens_k = seqlens_k
                    .iter()
                    .map(|len| (*len).min(window))
                    .collect::<Vec<_>>();
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
        // Pad each sequence by the padding token to the max len.
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
        let has_any_cache_hit = prefix_cache_lens.is_some_and(|lens| lens.iter().any(|&l| l > 0));
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
            let mut padded = new_toks.to_vec();
            padded.extend(std::iter::repeat_n(
                padding_tok,
                max_len.saturating_sub(padded.len()),
            ));
            // If we are returning raw logits, we want to not trim the logits at all.
            if return_raw_logits {
                if last_n_context_len.is_some() {
                    anyhow::bail!("`return_raw_logits` is incompatible with `last_n_context_len`");
                }

                context_lens.push((0, padded.len()));
            } else {
                context_lens.push((
                    padded
                        .len()
                        .saturating_sub(last_n_context_len.map(|(a, _)| a).unwrap_or(1)),
                    last_n_context_len.map(|(a, _)| a).unwrap_or(1),
                ));
            }

            if flash_attn {
                // Padded lengths, see FlashParams doc comment for prefix cache nuance.
                seqlens_q.push(padded.len() as u32);
                seqlens_k.push((padded.len() + chunk_offset_toks + cached) as u32);
            }

            seqs_tensors.push(Tensor::new(padded, device).unwrap().unsqueeze(0).unwrap());

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
                    let window_start = full_context_len.saturating_sub(sliding_window);
                    let block_aligned_start = (window_start / paged_attn_metadata.block_size)
                        * paged_attn_metadata.block_size;
                    if block_aligned_start <= slot_start {
                        let paged_context_len = full_context_len - block_aligned_start;
                        let slide_idx = block_aligned_start / paged_attn_metadata.block_size;
                        let needed_blocks =
                            paged_context_len.div_ceil(paged_attn_metadata.block_size);
                        let slide_end = (slide_idx + needed_blocks).min(table.len());
                        block_tables.push(table.get(slide_idx..slide_end).unwrap_or(&[]).to_vec());
                        paged_attn_context_lens.push((0..paged_context_len).collect());
                    } else {
                        block_tables.push(table.clone());
                        paged_attn_context_lens.push(ctxt_len);
                    }
                } else {
                    block_tables.push(table.clone());
                    paged_attn_context_lens.push(ctxt_len);
                }
            }
        }

        let flash_meta = if flash_attn {
            make_flash_params(device, mapper, &seqlens_q, &seqlens_k, sliding_window, true)?
        } else {
            FlashParams::empty(true)
        };

        let input = Tensor::cat(&seqs_tensors, 0).unwrap();

        let paged_attn_meta = if let Some(paged_attn_metadata) = &paged_attn_metadata {
            // Create paged attention tensors on CPU first (see comment above about CUDA contexts)
            let max_slot_mapping_len = slot_mappings.iter().map(|x| x.len()).max().unwrap();
            let prefill_query_lens = slot_mappings.iter().map(Vec::len).collect::<Vec<_>>();
            let slot_mappings = _make_tensor_with_pad(
                slot_mappings,
                max_slot_mapping_len,
                _PAD_SLOT_ID,
                &Device::Cpu,
            )?;

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
                full_context_lens_for_fi
            };
            let (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len) =
                make_paged_kv_tensors(
                    &block_tables,
                    &paged_context_lens_for_fi,
                    block_size,
                    block_tables.len() * max_block_table_len,
                )?;
            let (
                q_indptr,
                request_indices,
                qo_tile_indices,
                kv_tile_indices,
                o_indptr,
                kv_chunk_size,
                block_valid_mask,
            ) = make_paged_kv_prefill_tensors(&prefill_query_lens, block_size)?;
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

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();
            let mut full_block_tables_map = HashMap::new();
            let mut full_context_lens_map = HashMap::new();
            let mut paged_kv_indptr_map = HashMap::new();
            let mut paged_kv_indices_map = HashMap::new();
            let mut paged_kv_last_page_len_map = HashMap::new();
            let mut q_indptr_map = HashMap::new();
            let mut request_indices_map = HashMap::new();
            let mut qo_tile_indices_map = HashMap::new();
            let mut kv_tile_indices_map = HashMap::new();
            let mut o_indptr_map = HashMap::new();
            let mut kv_chunk_size_map = HashMap::new();
            let mut block_valid_mask_map = HashMap::new();
            let mut full_paged_kv_indptr_map = HashMap::new();
            let mut full_paged_kv_indices_map = HashMap::new();
            let mut full_paged_kv_last_page_len_map = HashMap::new();
            let mut full_q_indptr_map = HashMap::new();
            let mut full_request_indices_map = HashMap::new();
            let mut full_qo_tile_indices_map = HashMap::new();
            let mut full_kv_tile_indices_map = HashMap::new();
            let mut full_o_indptr_map = HashMap::new();
            let mut full_kv_chunk_size_map = HashMap::new();
            let mut full_block_valid_mask_map = HashMap::new();

            let (
                full_block_tables_tensor,
                full_context_lens_tensor,
                full_max_context_len,
                full_paged_kv_tensors,
                full_prefill_tensors,
            ) = if sliding_window.is_some() {
                let full_max_block_table_len =
                    full_block_tables.iter().map(|x| x.len()).max().unwrap_or(1);
                let full_paged_kv_tensors = Some(make_paged_kv_tensors(
                    &full_block_tables,
                    &full_paged_attn_context_lens,
                    block_size,
                    full_block_tables.len() * full_max_block_table_len,
                )?);
                let full_prefill_tensors = Some(make_paged_kv_prefill_tensors(
                    &prefill_query_lens,
                    block_size,
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
                    full_prefill_tensors,
                )
            } else {
                (None, None, None, None, None)
            };

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
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
                q_indptr_map.insert(device.location(), q_indptr.clone().to_device(&device)?);
                request_indices_map.insert(
                    device.location(),
                    request_indices.clone().to_device(&device)?,
                );
                qo_tile_indices_map.insert(
                    device.location(),
                    qo_tile_indices.clone().to_device(&device)?,
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
                if let Some((q, req, qo, kv, o, chunk, valid)) = &full_prefill_tensors {
                    full_q_indptr_map.insert(device.location(), q.clone().to_device(&device)?);
                    full_request_indices_map
                        .insert(device.location(), req.clone().to_device(&device)?);
                    full_qo_tile_indices_map
                        .insert(device.location(), qo.clone().to_device(&device)?);
                    full_kv_tile_indices_map
                        .insert(device.location(), kv.clone().to_device(&device)?);
                    full_o_indptr_map.insert(device.location(), o.clone().to_device(&device)?);
                    full_kv_chunk_size_map
                        .insert(device.location(), chunk.clone().to_device(&device)?);
                    full_block_valid_mask_map
                        .insert(device.location(), valid.clone().to_device(&device)?);
                }
            }

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
                disable_cuda_graphs: has_any_cache_hit,
                paged_kv_indptr: Some(paged_kv_indptr_map),
                paged_kv_indices: Some(paged_kv_indices_map),
                paged_kv_last_page_len: Some(paged_kv_last_page_len_map),
                full_paged_kv_indptr: if full_paged_kv_indptr_map.is_empty() {
                    None
                } else {
                    Some(full_paged_kv_indptr_map)
                },
                full_paged_kv_indices: if full_paged_kv_indices_map.is_empty() {
                    None
                } else {
                    Some(full_paged_kv_indices_map)
                },
                full_paged_kv_last_page_len: if full_paged_kv_last_page_len_map.is_empty() {
                    None
                } else {
                    Some(full_paged_kv_last_page_len_map)
                },
                paged_kv_q_indptr: Some(q_indptr_map),
                paged_kv_qo_tile_indices: Some(qo_tile_indices_map),
                paged_kv_request_indices: Some(request_indices_map),
                paged_kv_tile_indices: Some(kv_tile_indices_map),
                paged_kv_o_indptr: Some(o_indptr_map),
                paged_kv_chunk_size: Some(kv_chunk_size_map),
                paged_kv_block_valid_mask: Some(block_valid_mask_map),
                block_table_signature: None,
                full_paged_kv_q_indptr: if full_q_indptr_map.is_empty() {
                    None
                } else {
                    Some(full_q_indptr_map)
                },
                full_paged_kv_qo_tile_indices: if full_qo_tile_indices_map.is_empty() {
                    None
                } else {
                    Some(full_qo_tile_indices_map)
                },
                full_paged_kv_request_indices: if full_request_indices_map.is_empty() {
                    None
                } else {
                    Some(full_request_indices_map)
                },
                full_paged_kv_tile_indices: if full_kv_tile_indices_map.is_empty() {
                    None
                } else {
                    Some(full_kv_tile_indices_map)
                },
                full_paged_kv_o_indptr: if full_o_indptr_map.is_empty() {
                    None
                } else {
                    Some(full_o_indptr_map)
                },
                full_paged_kv_chunk_size: if full_kv_chunk_size_map.is_empty() {
                    None
                } else {
                    Some(full_kv_chunk_size_map)
                },
                full_paged_kv_block_valid_mask: if full_block_valid_mask_map.is_empty() {
                    None
                } else {
                    Some(full_block_valid_mask_map)
                },
                full_block_table_signature: None,
                rope_positions: None,
                num_cached_tokens: if has_any_cache_hit {
                    Some(num_cached_tokens_vec.clone())
                } else {
                    None
                },
                query_lens: if has_any_cache_hit {
                    Some(query_lens_vec.clone())
                } else {
                    None
                },
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

    fn make_completion_chunk<T: WithDType + From<u32> + Clone + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
    ) -> Result<InputMetadata> {
        // Pad each sequence by the padding token to the max len.
        let flash_attn = crate::using_flash_attn();
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();

        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut full_block_tables = Vec::new();
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
        for (seq, ctxt) in input_seqs.iter().zip(toks) {
            let staged_speculative = if use_staged_speculative {
                seq.active_staged_speculative_tokens()
            } else {
                &[]
            };
            let start_pos = ctxt.len().saturating_sub(1);
            let mut ctxt = ctxt[start_pos..].to_vec();
            ctxt.extend(staged_speculative.iter().copied().map(T::from));
            let query_len = ctxt.len();
            let effective_context_len = start_pos + query_len;
            seqlen_offsets.push(start_pos);
            context_lens.push((0, query_len));
            position_ids.push(effective_context_len);

            if flash_attn {
                seqlens_q.push(query_len as u32);
                seqlens_k.push(effective_context_len as u32);
            }

            seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let kv_mgr = get_mut_arcmutex!(paged_attn_metadata.kv_cache_manager);
                let table: Vec<usize> = kv_mgr
                    .get_block_ids(*seq.id())
                    .expect("Sequence must have allocated blocks for completion")
                    .to_vec();
                drop(kv_mgr);

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

                    // Always collect the full (unwindowed) block tables.
                    full_block_tables.push(table.clone());
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
                    if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                        let window_start = full_context_len.saturating_sub(sliding_window);
                        let slide_idx = window_start / paged_attn_metadata.block_size;
                        let needed_blocks =
                            paged_attn_context_len.div_ceil(paged_attn_metadata.block_size);
                        let slide_end = (slide_idx + needed_blocks).min(table.len());
                        block_tables.push(table.get(slide_idx..slide_end).unwrap_or(&[]).to_vec());
                    } else {
                        block_tables.push(table.clone());
                    }
                    paged_attn_context_lens.push(paged_attn_context_len);
                }
            }
        }

        let flash_meta = if flash_attn {
            make_flash_params(device, mapper, &seqlens_q, &seqlens_k, sliding_window, true)?
        } else {
            FlashParams::empty(true)
        };

        let paged_attn_meta = if let Some(paged_attn_input) = &paged_attn_metadata {
            // Create paged attention tensors on CPU first (see make_prompt_chunk for explanation)
            let max_slot_mapping_len = slot_mappings.iter().map(Vec::len).max().unwrap_or(1);
            let use_standard_metadata =
                paged_attn_input.attention_backend == AttentionBackendKind::Standard;
            let slot_mappings = _make_tensor_with_pad(
                slot_mappings,
                max_slot_mapping_len,
                _PAD_SLOT_ID,
                &Device::Cpu,
            )?;

            let max_block_table_len = block_tables
                .iter()
                .map(|x| x.len())
                .max()
                .expect("block_tables should not be empty when paged attention is enabled");
            let max_block_table_len =
                cuda_graph_block_table_len(max_block_table_len, paged_attn_input.block_size);

            let batch_size = block_tables.len();
            let block_size = paged_attn_input.block_size;
            let (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len) =
                make_paged_kv_tensors(
                    &block_tables,
                    &paged_attn_context_lens,
                    block_size,
                    batch_size * max_block_table_len,
                )?;

            let split_pages = Some(FLASHINFER_DECODE_SPLIT_PAGES);
            let tiles_per_row = max_block_table_len
                .max(1)
                .div_ceil(FLASHINFER_DECODE_SPLIT_PAGES);
            let (request_indices, kv_tile_indices, o_indptr, kv_chunk_size, block_valid_mask) =
                make_paged_kv_decode_tensors(
                    &block_tables,
                    &paged_attn_context_lens,
                    block_size,
                    split_pages,
                    batch_size * tiles_per_row,
                )?;
            let paged_block_table_signature =
                block_table_signature(&block_tables, &paged_attn_context_lens, block_size);
            let full_matches_paged = paged_attn_input.sliding_window.is_none();

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

            let max_context_len = paged_attn_context_lens.iter().max().unwrap();

            let context_lens = Tensor::from_vec(
                paged_attn_context_lens
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>(),
                (paged_attn_context_lens.len(),),
                &Device::Cpu,
            )?;
            // Build full (unwindowed) block tables and context lens.
            let full_max_block_table_len =
                full_block_tables.iter().map(|x| x.len()).max().unwrap_or(0);
            let full_max_block_table_len = cuda_graph_block_table_len(
                full_max_block_table_len.max(1),
                paged_attn_input.block_size,
            );

            let full_block_tables_tensor = _make_tensor_with_pad(
                full_block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                full_max_block_table_len,
                0,
                &Device::Cpu,
            )?;
            let full_block_tables_tensor =
                full_block_tables_tensor.reshape(((), full_max_block_table_len))?;

            let full_max_context_len = full_paged_attn_context_lens
                .iter()
                .max()
                .copied()
                .unwrap_or(0);

            let (full_paged_kv_indptr, full_paged_kv_indices, full_paged_kv_last_page_len) =
                make_paged_kv_tensors(
                    &full_block_tables,
                    &full_paged_attn_context_lens,
                    block_size,
                    full_block_tables.len() * full_max_block_table_len,
                )?;
            let full_split_pages = Some(FLASHINFER_DECODE_SPLIT_PAGES);
            let full_tiles_per_row = full_max_block_table_len
                .max(1)
                .div_ceil(FLASHINFER_DECODE_SPLIT_PAGES);
            let (
                full_paged_kv_request_indices,
                full_paged_kv_tile_indices,
                full_paged_kv_o_indptr,
                full_paged_kv_chunk_size,
                full_paged_kv_block_valid_mask,
            ) = make_paged_kv_decode_tensors(
                &full_block_tables,
                &full_paged_attn_context_lens,
                block_size,
                full_split_pages,
                full_block_tables.len() * full_tiles_per_row,
            )?;
            let full_block_table_signature = block_table_signature(
                &full_block_tables,
                &full_paged_attn_context_lens,
                block_size,
            );

            let full_context_lens_tensor = Tensor::from_vec(
                full_paged_attn_context_lens
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>(),
                (full_paged_attn_context_lens.len(),),
                &Device::Cpu,
            )?;

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();
            let mut full_block_tables_map = HashMap::new();
            let mut full_context_lens_map = HashMap::new();
            let mut paged_kv_indptr_map = HashMap::new();
            let mut paged_kv_indices_map = HashMap::new();
            let mut paged_kv_last_page_len_map = HashMap::new();
            let mut full_paged_kv_indptr_map = HashMap::new();
            let mut full_paged_kv_indices_map = HashMap::new();
            let mut full_paged_kv_last_page_len_map = HashMap::new();
            let mut paged_kv_request_indices_map = HashMap::new();
            let mut paged_kv_tile_indices_map = HashMap::new();
            let mut paged_kv_o_indptr_map = HashMap::new();
            let mut paged_kv_chunk_size_map = HashMap::new();
            let mut paged_kv_block_valid_mask_map = HashMap::new();
            let mut full_paged_kv_request_indices_map = HashMap::new();
            let mut full_paged_kv_tile_indices_map = HashMap::new();
            let mut full_paged_kv_o_indptr_map = HashMap::new();
            let mut full_paged_kv_chunk_size_map = HashMap::new();
            let mut full_paged_kv_block_valid_mask_map = HashMap::new();

            for device in devices {
                let location = device.location();
                let slot_mappings_device = slot_mappings.clone().to_device(&device)?;
                let paged_kv_indptr_device = paged_kv_indptr.clone().to_device(&device)?;
                let paged_kv_indices_device = paged_kv_indices.clone().to_device(&device)?;
                let paged_kv_last_page_len_device =
                    paged_kv_last_page_len.clone().to_device(&device)?;
                let request_indices_device = request_indices.clone().to_device(&device)?;
                let kv_tile_indices_device = kv_tile_indices.clone().to_device(&device)?;
                let o_indptr_device = o_indptr.clone().to_device(&device)?;
                let kv_chunk_size_device = kv_chunk_size.clone().to_device(&device)?;
                let block_valid_mask_device = block_valid_mask.clone().to_device(&device)?;

                slot_mappings_map.insert(location, slot_mappings_device);
                paged_kv_indptr_map.insert(location, paged_kv_indptr_device.clone());
                paged_kv_indices_map.insert(location, paged_kv_indices_device.clone());
                paged_kv_last_page_len_map.insert(location, paged_kv_last_page_len_device.clone());
                paged_kv_request_indices_map.insert(location, request_indices_device.clone());
                paged_kv_tile_indices_map.insert(location, kv_tile_indices_device.clone());
                paged_kv_o_indptr_map.insert(location, o_indptr_device.clone());
                paged_kv_chunk_size_map.insert(location, kv_chunk_size_device.clone());
                paged_kv_block_valid_mask_map.insert(location, block_valid_mask_device.clone());

                if use_standard_metadata {
                    let block_tables_device = block_tables.clone().to_device(&device)?;
                    let context_lens_device = context_lens.clone().to_device(&device)?;
                    block_tables_map.insert(location, block_tables_device.clone());
                    context_lens_map.insert(location, context_lens_device.clone());
                    if full_matches_paged {
                        full_block_tables_map.insert(location, block_tables_device);
                        full_context_lens_map.insert(location, context_lens_device);
                    } else {
                        full_block_tables_map.insert(
                            location,
                            full_block_tables_tensor.clone().to_device(&device)?,
                        );
                        full_context_lens_map.insert(
                            location,
                            full_context_lens_tensor.clone().to_device(&device)?,
                        );
                    }
                }

                if full_matches_paged {
                    full_paged_kv_indptr_map.insert(location, paged_kv_indptr_device);
                    full_paged_kv_indices_map.insert(location, paged_kv_indices_device);
                    full_paged_kv_last_page_len_map.insert(location, paged_kv_last_page_len_device);
                    full_paged_kv_request_indices_map.insert(location, request_indices_device);
                    full_paged_kv_tile_indices_map.insert(location, kv_tile_indices_device);
                    full_paged_kv_o_indptr_map.insert(location, o_indptr_device);
                    full_paged_kv_chunk_size_map.insert(location, kv_chunk_size_device);
                    full_paged_kv_block_valid_mask_map.insert(location, block_valid_mask_device);
                } else {
                    full_paged_kv_indptr_map
                        .insert(location, full_paged_kv_indptr.clone().to_device(&device)?);
                    full_paged_kv_indices_map
                        .insert(location, full_paged_kv_indices.clone().to_device(&device)?);
                    full_paged_kv_last_page_len_map.insert(
                        location,
                        full_paged_kv_last_page_len.clone().to_device(&device)?,
                    );
                    full_paged_kv_request_indices_map.insert(
                        location,
                        full_paged_kv_request_indices.clone().to_device(&device)?,
                    );
                    full_paged_kv_tile_indices_map.insert(
                        location,
                        full_paged_kv_tile_indices.clone().to_device(&device)?,
                    );
                    full_paged_kv_o_indptr_map
                        .insert(location, full_paged_kv_o_indptr.clone().to_device(&device)?);
                    full_paged_kv_chunk_size_map.insert(
                        location,
                        full_paged_kv_chunk_size.clone().to_device(&device)?,
                    );
                    full_paged_kv_block_valid_mask_map.insert(
                        location,
                        full_paged_kv_block_valid_mask.clone().to_device(&device)?,
                    );
                }
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: use_standard_metadata.then_some(block_tables_map),
                context_lens: use_standard_metadata.then_some(context_lens_map),
                block_size: Some(block_size),
                paged_context_lens_cpu: Some(paged_attn_context_lens.clone()),
                full_paged_context_lens_cpu: Some(full_paged_attn_context_lens.clone()),
                max_context_len: use_standard_metadata.then_some(*max_context_len),
                full_block_tables: use_standard_metadata.then_some(full_block_tables_map),
                full_context_lens: use_standard_metadata.then_some(full_context_lens_map),
                full_max_context_len: use_standard_metadata.then_some(full_max_context_len),
                is_first_prompt_chunk: false,
                disable_cuda_graphs: input_seqs.iter().any(|seq| seq.tools.is_some()),
                paged_kv_indptr: Some(paged_kv_indptr_map),
                paged_kv_indices: Some(paged_kv_indices_map),
                paged_kv_last_page_len: Some(paged_kv_last_page_len_map),
                full_paged_kv_indptr: Some(full_paged_kv_indptr_map),
                full_paged_kv_indices: Some(full_paged_kv_indices_map),
                full_paged_kv_last_page_len: Some(full_paged_kv_last_page_len_map),
                paged_kv_q_indptr: None,
                paged_kv_qo_tile_indices: None,
                paged_kv_request_indices: Some(paged_kv_request_indices_map),
                paged_kv_tile_indices: Some(paged_kv_tile_indices_map),
                paged_kv_o_indptr: Some(paged_kv_o_indptr_map),
                paged_kv_chunk_size: Some(paged_kv_chunk_size_map),
                paged_kv_block_valid_mask: Some(paged_kv_block_valid_mask_map),
                block_table_signature: Some(paged_block_table_signature),
                full_paged_kv_q_indptr: None,
                full_paged_kv_qo_tile_indices: None,
                full_paged_kv_request_indices: Some(full_paged_kv_request_indices_map),
                full_paged_kv_tile_indices: Some(full_paged_kv_tile_indices_map),
                full_paged_kv_o_indptr: Some(full_paged_kv_o_indptr_map),
                full_paged_kv_chunk_size: Some(full_paged_kv_chunk_size_map),
                full_paged_kv_block_valid_mask: Some(full_paged_kv_block_valid_mask_map),
                full_block_table_signature: Some(full_block_table_signature),
                rope_positions: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input: Tensor::cat(&seqs_tensors, 0).unwrap(),
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
        })
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
        )
        .map(|inputs| InnerInputProcessorOutput {
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
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else {
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
}
